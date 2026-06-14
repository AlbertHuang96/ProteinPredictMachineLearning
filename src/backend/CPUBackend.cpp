

#include "rfaa/Backend.h"

namespace rfaa {

// ===== 构造/析构 =====
CPUBackend::CPUBackend(int n_threads) : n_threads_(n_threads) {
    threadpool_ = new ThreadPool();
    threadpool_->init(n_threads_, &CPUBackend::compute_thread);
}

CPUBackend::~CPUBackend() {
    if (threadpool_) {
        threadpool_->free();
        delete threadpool_;
    }
    delete[] work_data_;
}

std::unique_ptr<CPUBackend> CPUBackend::create(int n_threads) {
    return std::make_unique<CPUBackend>(n_threads);
}

// ===== graph_plan (公有，可外部调用预估算) =====
ComputePlan CPUBackend::graph_plan(ComputeGraph * cgraph) const {
    ComputePlan plan;
    plan.n_threads  = n_threads_;
    plan.threadpool = threadpool_;
    plan.work_size  = 0;

    int max_tasks = 1;
    for (int i = 0; i < cgraph->n_nodes(); i++) {
        Tensor * node = cgraph->node(i);
        int n_tasks   = get_n_tasks(node, n_threads_);
        max_tasks     = std::max(max_tasks, n_tasks);

        size_t cur = estimate_work_size(node, n_threads_);
        plan.work_size = std::max(plan.work_size, cur);
    }
    plan.n_threads = std::min(max_tasks, n_threads_);
    if (plan.work_size > 0)
        plan.work_size += CACHE_LINE_SIZE * n_threads_;

    return plan;
}

// ===== graph_compute (Backend 接口) =====
Status CPUBackend::graph_compute(ComputeGraph * cgraph) {
    ComputePlan plan = graph_plan(cgraph);

    // 确保工作缓冲区够大
    if (work_size_ < plan.work_size) {
        delete[] work_data_;
        work_data_ = new uint8_t[plan.work_size];
        if (!work_data_) { work_size_ = 0; return Status::ALLOC_FAILED; }
        work_size_ = plan.work_size;
    }
    plan.work_data = work_data_;

    // 提交任务 + 主线程也参与计算
    threadpool_->submit(cgraph, &plan);

    // Thread 0 = 当前线程（不创建 std::thread）
    ThreadState main_state;
    main_state.id   = 0;
    main_state.pool = threadpool_;
    compute_thread(&main_state);

    threadpool_->barrier_wait();

    return threadpool_->ec;
}

// ===== compute_thread (静态，线程池回调) =====
void CPUBackend::compute_thread(ThreadState * state) {
    ThreadPool  * tp     = state->pool;
    ComputeGraph * cgraph = tp->cgraph;
    ComputePlan  * cplan  = tp->cplan;
    int ith               = state->id;

    ComputeParams params;
    params.ith        = ith;
    params.nth        = tp->n_threads_cur.load();
    params.wsize      = cplan->work_size;
    params.wdata      = cplan->work_data;
    params.threadpool = tp;

    for (int node_n = 0;
         node_n < cgraph->n_nodes() &&
         tp->abort.load(std::memory_order_relaxed) != node_n;
         node_n++) {

        Tensor * node = cgraph->node(node_n);

        dispatch_node(node, &params);

        if (ith == 0 && cplan->abort_callback &&
            cplan->abort_callback(cplan->abort_callback_data)) {
            tp->abort.store(node_n + 1, std::memory_order_relaxed);
        }

        if (node_n + 1 < cgraph->n_nodes()) {
            tp->barrier_wait();
        }
    }

    tp->barrier_wait();
}

int CPUBackend::get_n_tasks(Tensor * node, int n_threads) {
    switch (node->op) {
        case OP_MUL_MAT: {
            // 矩阵乘法：按行并行
            // 总行数 = dims[1]，每线程至少处理 32 行
            int64_t rows = node->dims()[1];  // M
            int max_tasks = static_cast<int>(rows / 32);
            return std::max(1, std::min(max_tasks, n_threads));
        }
        case OP_SOFT_MAX: {
            // ggml ne[1] * ne[2] * ne[3]
            // softmax：沿最后一维，每个"行"是一个任务
            int64_t outer = node->numel() / node->dims()[0];
            int max_tasks = static_cast<int>(outer);
            return std::clamp(max_tasks, 1, n_threads);
        }
        case OP_RMS_NORM: {
            int64_t rows = node->numel() / node->dims()[0];
            return std::min(static_cast<int>(rows / 16), n_threads);
        }
        case OP_FLASH_ATTN_BACK:
        case OP_FLASH_ATTN_EXT: {
            // 注意力：按 batch × head 并行
            return n_threads;
        }
        case OP_NONE:
        case OP_VIEW:
        case OP_RESHAPE:
            return 1;
        default:
            // 逐元素操作：数据量够大才用多线程
            if (node->numel() > 1024 * 1024) {
                return std::min(4, n_threads);  // 最多 4 线程
            }
            return 1;
    }

    return -1;
}
size_t CPUBackend::estimate_work_size(Tensor * node, int n_threads, int n_tasks) {
    size_t cur = 0;

    switch (node->op) {
        // ===== 需要反量化的操作 =====
        case OP_ADD:
        case OP_ADD1:
        case OP_MUL:
            // RFAA 目前只支持 F32，暂不要反量化缓冲
            // 如果将来支持 F16/I8，这里需要：
            // if (is_quantized(node->src[0]->type))
            //     cur = sizeof(float) * node->src[0]->dims()[0] * n_tasks;
            break;

        // ===== 拷贝/类型转换 =====
        case OP_CPY:
        case OP_DUP:
            // 如果是跨类型拷贝，需要 F32 中间缓冲
            break;

        // ===== 注意力 =====
        case OP_FLASH_ATTN_EXT: {
            // S = QK^T 的中间结果
            // shape: (B, H, L, L) 或类似
            //int n_tasks = get_n_tasks(node, n_threads);
            // 保守估计：attn weights 的中间存储
            cur = sizeof(float) * node->dims()[2] * node->dims()[3] * n_tasks;
        } break;
        case OP_FLASH_ATTN_BACK: {
            // 反向还需要存储 dS
            
        } break;
        // ===== 交叉熵 =====
        case OP_CROSS_ENTROPY_LOSS: {
            //int n_tasks = get_n_tasks(node, n_threads);
            cur = sizeof(float) * (n_tasks + node->dims()[0] * n_tasks);
        } break;

        // ===== 归一化 =====
        case OP_RMS_NORM:
        case OP_NORM: {
            // 每线程需要均值和方差的暂存空间
            //int n_tasks = get_n_tasks(node, n_threads);
            int64_t rows = node->numel() / node->dims()[0];
            int rows_per_task = static_cast<int>(rows / n_tasks + 1);
            cur = sizeof(float) * rows_per_task;
        } break;

        // ===== 拼接 =====
        case OP_CONCAT: {
            // 可能需要拷贝暂存区
            cur = node->numel() * sizeof(float);
        } break;

        // ===== 默认：无额外缓冲区 =====
        case OP_NONE:
        case OP_VIEW:
        case OP_RESHAPE:
        case OP_PERMUTE:
        case OP_TRANSPOSE:
        case OP_SQR:
        case OP_SQRT:
        case OP_SCALE:
        default:
            cur = 0;
            break;
    }

    return cur;
}

} // namespace rfaa




#include "ppml/Backend.h"
#include <cstdlib>
#include <cstdio>

namespace ppml {

// ===== CPUBufferType 实现 =====
CPUBufferType* CPUBufferType::instance() {
    static CPUBufferType s_instance;
    return &s_instance;
}

void* CPUBufferType::alloc(size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, 32);
#else
    return std::aligned_alloc(32, ((size + 31) / 32) * 32);
#endif
}

void CPUBufferType::free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// ===== CPUBackend::buffer_type / supports_buffer_type =====
const BufferType* CPUBackend::buffer_type() const {
    return CPUBufferType::instance();
}

bool CPUBackend::supports_buffer_type(const BufferType* buft) const {
    // CPU backend 支持所有 host 端 buffer 类型
    if (!buft) return false;
    return buft->is_host();
}

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

bool CPUBackend::supports_op(TensorF32* node) const {
    const TensorF32* src0 = node->src[0];
    const TensorF32* src1 = node->src[1];
    
    if (node->op == OP_NONE   || node->op == OP_RESHAPE ||
        node->op == OP_VIEW   || node->op == OP_PERMUTE ||
        node->op == OP_TRANSPOSE) {
        return true;
    }

    switch (node->op) {
        case OP_CPY:
        case OP_SET_ROWS:
        case OP_EDGE_GATHER_ROWS:
        case OP_PER_EDGE_MATMUL:
        case OP_SCATTER_ADD:
        case OP_PER_EDGE_MATMUL_BACK_KERNEL:
        case OP_PER_EDGE_MATMUL_BACK_GATHERED:
        case OP_CONCAT_BACK:
        case OP_OUTER_PROD_MEAN:
        case OP_OUTER_PROD_MEAN_BACK:
        case OP_OUTER_PROD:
        case OP_OUTER_PROD_BACK:
        // ggml IQ quantize
        // now only have F32
            return true;
        
        case OP_MUL_MAT:
        // ggml src1->type == vec_dot_type(src0->type)
            if (!src1) return true;
            return src1->type == TENSOR_TYPE_F32;

        case OP_SOFT_MAX_BACK: {
            if (!src0 || !src1) return false;
            if (src0->type != TENSOR_TYPE_F32 ||
                src1->type != TENSOR_TYPE_F32) {
                    return false;
                }
            // ggml op_params max_bias == 0.0f
            return true;
        }

        case OP_GET_ROWS_BACK:
            if (!src0) return true;
            return src0->type == TENSOR_TYPE_F32 ||
                   src0->type == TENSOR_TYPE_F16;
        case OP_OUT_PROD:
            if (!src0 || !src1) return false;
            // if quantize ne[2]/ne[3] match
            return src0->type == TENSOR_TYPE_F32 &&
                   src1->type == TENSOR_TYPE_F32 && 
                   node->type == TENSOR_TYPE_F32;
        default:
            return true;
    }
}

// ===== graph_plan (公有，可外部调用预估算) =====
ComputePlan CPUBackend::graph_plan(ComputeGraph * cgraph) const {
    ComputePlan plan;
    plan.n_threads  = n_threads_;
    plan.threadpool = threadpool_;
    plan.work_size  = 0;

    int max_tasks = 1;
    for (int i = 0; i < cgraph->n_nodes(); i++) {
        TensorF32 * node = cgraph->graph_node(i);
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
    // ---- no_alloc 延迟分配：对 data_==nullptr 的中间节点分配 backend buffer ----
    // 混训时 scheduler 已通过 reserve_graph_memory 预分配了全部张量，置 skip_alloc_=true，
    // 这里跳过本端自己的 gallocr，避免两套 gallocr 反复 re-bind 张量导致跨 split 读到错位 buffer。
    if (!skip_alloc_) {
        // 释放上一图分配的 buffer（上一图消费方已在上次 graph_compute 返回后读取完 data()）。
        gallocr_.release();
        bool need_alloc = false;
        for (int i = 0; i < cgraph->n_nodes(); ++i) {
            if (cgraph->graph_node(i)->data() == nullptr) { need_alloc = true; break; }
        }
        if (!need_alloc) {
            for (int i = 0; i < cgraph->n_leafs(); ++i) {
                if (cgraph->graph_leaf(i)->data() == nullptr) { need_alloc = true; break; }
            }
        }
        if (need_alloc) {
            gallocr_.set_n_backends(1);
            gallocr_.backends()[0].buft = CPUBufferType::instance();
            auto backend_id_of = [](TensorF32*) -> int { return 0; };
            if (!gallocr_.reserve(cgraph, backend_id_of, 1) ||
                !gallocr_.alloc(cgraph, backend_id_of, 1)) {
                if (getenv("GRAPH_DEBUG_GALLOCR")) {
                    fprintf(stderr, "[gallocr] ALLOC_FAILED: n_nodes=%d n_leafs=%d peak=%zu\n",
                            cgraph->n_nodes(), cgraph->n_leafs(), gallocr_.backend_peak(0));
                }
                return Status::ALLOC_FAILED;
            }
            if (getenv("GRAPH_DEBUG_GALLOCR")) {
                fprintf(stderr, "[gallocr] ok: n_nodes=%d n_leafs=%d peak=%zu bytes (%.2f GB)\n",
                        cgraph->n_nodes(), cgraph->n_leafs(),
                        gallocr_.backend_peak(0),
                        gallocr_.backend_peak(0) / (1024.0 * 1024.0 * 1024.0));
            }
        }
    }

    ComputePlan plan = graph_plan(cgraph);

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

        TensorF32 * node = cgraph->graph_node(node_n);

        dispatch_node(node, &params);

        if (ith == 0 && cplan->abort_callback &&
            cplan->abort_callback(cplan->abort_callback_data)) {
            tp->abort.store(node_n + 1, std::memory_order_relaxed);
        }

        if (node_n + 1 < cgraph->n_nodes()) {
            tp->barrier_wait();
        }

        // ---- 开发诊断：找第一个输出含 NaN 的节点（GRAPH_DEBUG_NAN=1）----
        if (ith == 0 && getenv("GRAPH_DEBUG_NAN") && node->data() && node->op != OP_NONE) {
            const int64_t ne_ = node->numel();
            const float* nd = node->data();
            for (int64_t q = 0; q < ne_; ++q) {
                if (!std::isfinite(nd[q])) {   // NaN 或 Inf 都报
                    fprintf(stderr,
                            "[nan-node] FIRST_NAN node_n=%d op=%d src0op=%d dims=[%lld,%lld,%lld,%lld] "
                            "numel=%lld idx=%lld src0data=%p\n",
                            node_n, (int)node->op,
                            (node->src[0] ? (int)node->src[0]->op : -1),
                            (long long)(node->shape().ndim()>0?node->shape().dims[0]:-1),
                            (long long)(node->shape().ndim()>1?node->shape().dims[1]:-1),
                            (long long)(node->shape().ndim()>2?node->shape().dims[2]:-1),
                            (long long)(node->shape().ndim()>3?node->shape().dims[3]:-1),
                            (long long)ne_, (long long)q,
                            node->src[0] ? (void*)node->src[0]->data() : nullptr);
                    // 对 MUL/MUL_MAT/DIV：dump src0/src1 前 8 个值，判断哪个输入含 NaN/巨大
                    if ((node->op == OP_MUL_MAT || node->op == OP_MUL || node->op == OP_DIV) &&
                        node->src[0] && node->src[1] &&
                        node->src[0]->data() && node->src[1]->data()) {
                        fprintf(stderr, "  [nan-node] src0 op=%d head8: ", (int)node->src[0]->op);
                        const float* s0 = node->src[0]->data();
                        for (int h = 0; h < 8 && h < node->src[0]->numel(); ++h) fprintf(stderr, "%g,", s0[h]);
                        fprintf(stderr, "\n  [nan-node] src1 op=%d head8: ", (int)node->src[1]->op);
                        const float* s1 = node->src[1]->data();
                        for (int h = 0; h < 8 && h < node->src[1]->numel(); ++h) fprintf(stderr, "%g,", s1[h]);
                        fprintf(stderr, "\n");
                    }
                    break;  // 只报每个节点的首个 NaN
                }
            }
        }
    }

    tp->barrier_wait();
}

int CPUBackend::get_n_tasks(TensorF32 * node, int n_threads) {
    switch (node->op) {
        case OP_MUL_MAT: {
            // 矩阵乘法：按行并行
            int64_t rows = node->shape().dims[1];  // M
            int max_tasks = static_cast<int>(rows / 32);
            return std::max(1, std::min(max_tasks, n_threads));
        }
        case OP_SOFT_MAX: {
            // softmax：沿最后一维，每个"行"是一个任务
            int64_t outer = node->numel() / node->shape().dims[0];
            int max_tasks = static_cast<int>(outer);
            return std::max(1, std::min(max_tasks, n_threads));
        }
        case OP_RMS_NORM: {
            int64_t rows = node->numel() / node->shape().dims[0];
            return std::min(static_cast<int>(rows / 16), n_threads);
        }
        case OP_FLASH_ATTN_BACK:
        case OP_FLASH_ATTN_EXT: {
            // 注意力：按 batch × head 并行
            return n_threads;
        }
        case OP_CONCAT: {
            // 拼接：按输出元素数决定是否多线程
            if (node->numel() > 1024 * 1024) {
                return std::min(4, n_threads);
            }
            return 1;
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
size_t CPUBackend::estimate_work_size(TensorF32 * node, int n_threads, int n_tasks) {
    size_t cur = 0;

    // graph_plan 调用本函数时只传 2 参数，n_tasks 为默认 -1；
    // 若不修正，RMS_NORM/NORM/FLASH_ATTN/CROSS_ENTROPY 等分支会用 n_tasks=-1
    // 算出负值(在 size_t 下溢出为超大) → work_size 巨大 → new uint8_t[work_size] 抛 bad_alloc。
    if (n_tasks < 1) n_tasks = get_n_tasks(node, n_threads);
    if (n_tasks < 1) n_tasks = 1;

    switch (node->op) {
        // ===== 需要反量化的操作 =====
        case OP_ADD:
        case OP_ADD1:
        case OP_MUL:
            // PPML 目前只支持 F32，暂不要反量化缓冲
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
            cur = sizeof(float) * node->shape().dims[2] * node->shape().dims[3] * n_tasks;
            // TODO: further need to change to tiled version

        } break;
        case OP_FLASH_ATTN_BACK: {
            // 反向还需要存储 dS
            // D = head dim (如 64 or 128)
            // Q's head dim
            const int64_t D = node->src[0]->shape().dims[0];

            // Lkv = K 的序列长度，对齐到 UNROLL
            const int64_t ne11 = align_up(node->src[1]->shape().dims[1], SOFT_MAX_UNROLL);

            // mxDn: 取 max 是为了用较大的维度兜底，×2 因为 S + SM 两份
            const int64_t mxDn = std::max(D, ne11) * 2;

            // PPML 目前只支持 F32，直接计算
            cur  = sizeof(float) * mxDn * n_tasks;   // S: softmax 分数缓冲
            cur += sizeof(float) * mxDn * n_tasks;   // SM: max 缓冲 (高估 ×2)
            
        } break;
        // ===== 交叉熵 =====
        case OP_CROSS_ENTROPY_LOSS: {
            //int n_tasks = get_n_tasks(node, n_threads);
            cur = sizeof(float) * (n_tasks + node->shape().dims[0] * n_tasks);
        } break;

        // ===== 归一化 =====
        case OP_RMS_NORM:
        case OP_NORM: {
            // 每线程需要均值和方差的暂存空间
            //int n_tasks = get_n_tasks(node, n_threads);
            int64_t rows = node->numel() / node->shape().dims[0];
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
        case OP_TRI_MUL:
        case OP_SQR:
        case OP_SQRT:
        case OP_SCALE:
        default:
            cur = 0;
            break;
    }

    return cur;
}

} // namespace ppml

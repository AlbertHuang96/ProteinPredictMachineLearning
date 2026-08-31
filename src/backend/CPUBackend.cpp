


#include "ppml/Backend.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

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
    // ⚠️ n_threads 必须恒等于 n_threads_（实际参与线程数 = 主线程 + 全部 worker）。
    // 线程池 submit 设 n_threads_cur = plan.n_threads，barrier 期待该数量的线程参与；
    // 但 worker_loop 会在【全部】n_threads_-1 个 worker 上跑 compute_thread，主线程也参与，
    // 共 n_threads_ 个线程调 barrier。若 plan.n_threads < n_threads_（例如 max_tasks 较小），
    // barrier 期待 n_threads 个线程但实际有 n_threads_ 个线程调用 → 计数错 → 线程失步 →
    // 后续 kernel_elemwise 读被改 buffer 崩（单线程不崩、多线程崩）。故必须恒为 n_threads_。
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
        // ---- needs_realloc 接入（2026-08-31，单后端 CPU）：布局未变则复用 buffer ----
        // 复用前提：can_reuse（已分配过 && has_snapshot && !needs_realloc）→ 不 release、
        // 不重建，直接 alloc 走复用路径（只重绑 data）。解决跨 graph_compute buffer 悬垂。
        // 回退开关：PPML_NO_GALLOCR_REUSE=1 时走原逻辑（每轮 release + reserve + alloc）。
        // 注意：can_reuse 只对"图结构/张量大小未变"成立；compute_and_read 增量图 n_nodes
        //   每次变化 → needs_realloc 恒 true → 走重建（无回归）。
        bool gallocr_reuse = true;
        if (getenv("PPML_NO_GALLOCR_REUSE")) {
            gallocr_reuse = (std::strcmp(getenv("PPML_NO_GALLOCR_REUSE"), "1") != 0);
        }
        if (gallocr_reuse && gallocr_.can_reuse(cgraph)) {
            gallocr_.backends()[0].buft = CPUBufferType::instance();
            auto backend_id_of = [](TensorF32*) -> int { return 0; };
            if (!gallocr_.alloc(cgraph, backend_id_of, 1)) {
                return Status::ALLOC_FAILED;
            }
        } else {
            // 释放上一图分配的 buffer（上一图消费方已在上次 graph_compute 返回后读取完 data()）。
            gallocr_.release();
            // ⚠️ need_alloc 必须恒 true：release() 已释放全部 buffer 并复位快照内节点 data()，
            //    但跨图共享节点（SE3 的 compute_and_read 独立 cg 与主图共享 backend gallocr_）
            //    的 data() 可能仍非空且指向已释放 buffer —— 若依赖 data()==nullptr 判定跳过
            //    重新分配，主图 compute 就会读悬垂指针 → SIGSEGV（[ELEM-PTR] 显示 data 非空仍崩）。
            //    skip_alloc_=true（scheduler 预分配）时整个分支跳过，不受影响。
            bool need_alloc = true;
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
        }   // else（重建路径）闭合
    }

    // ---- 提交前单线程解析所有 view src ----
    // view（view_src 共享源数据）作为别的 op 的 src 时 data() 为 null（kernel_cpy 只在 view 自身
    // 被 dispatch 时解析）。若在 dispatch_node 里多线程解析，会写共享 src->data()/buffer_ 造成竞态
    // （崩溃位置漂移）。这里由主线程在提交线程池前一次性解析，消除竞态。
    for (int i = 0; i < cgraph->n_nodes(); ++i) {
        TensorF32* node = cgraph->graph_node(i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            TensorF32* src = node->src[s];
            if (!src || !src->view_src) continue;
            TensorF32* base = src;
            int guard = 0;
            while (base->view_src && guard++ < 64) base = base->view_src;
            if (base->data()) {
                src->bind_data(base->data());
                src->buffer_      = base->buffer_;
                src->buffer_offs_ = base->buffer_offs_;
            }
        }
    }

    // ---- GRAPH_DEBUG_ALLOC=1：提交前扫描未分配的中间节点（data==nullptr），
    // 定位 [ELEM-NULL]/NaN 源（哪些节点被 gallocr 漏分配，dispatch 时读 null/悬垂）。----
    if (getenv("GRAPH_DEBUG_ALLOC")) {
        for (int i = 0; i < cgraph->n_nodes(); ++i) {
            TensorF32* nd = cgraph->graph_node(i);
            if (!nd) continue;
            if (nd->data() != nullptr) continue;
            const bool viewish = (nd->op == OP_VIEW || nd->op == OP_RESHAPE ||
                                  nd->op == OP_PERMUTE || nd->op == OP_TRANSPOSE);
            if (viewish || nd->view_src) continue;
            fprintf(stderr,
                    "[NOALLOC] node=%d op=%d numel=%lld ndim=%d flag=0x%x buf=%p view_src=%p"
                    " src0={op=%d numel=%lld data=%p} src1={op=%d numel=%lld data=%p}\n",
                    i, (int)nd->op, (long long)nd->numel(), (int)nd->shape().ndim(),
                    (unsigned)nd->flag, (void*)nd->buffer_, (void*)nd->view_src,
                    (nd->src[0] ? (int)nd->src[0]->op : -1),
                    (nd->src[0] ? (long long)nd->src[0]->numel() : -1),
                    (nd->src[0] ? (void*)nd->src[0]->data() : nullptr),
                    (nd->src[1] ? (int)nd->src[1]->op : -1),
                    (nd->src[1] ? (long long)nd->src[1]->numel() : -1),
                    (nd->src[1] ? (void*)nd->src[1]->data() : nullptr));
        }
    }

    ComputePlan plan = graph_plan(cgraph);

    // ⚠️ 不再强制 skip_alloc_ 时 plan.n_threads=1：那会破坏"n_threads 必须等于实际参与线程数
    // n_threads_"的不变量（见 graph_plan）。graph_plan 现在恒返回 n_threads_（全部线程参与），
    // 多线程 barrier 计数一致，正确。

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

        // ---- 开发诊断：找第一个输出含巨大值(爆炸源)的节点（GRAPH_DEBUG_NAN=1）----
        // 在 NaN 之前抓真正的爆炸起点：max|v| > EXPLODE_THRESH 即报（不放过已是 NaN 的）。
        if (ith == 0 && getenv("GRAPH_DEBUG_NAN") && node->data() && node->op != OP_NONE) {
            const float EXPLODE_THRESH = 1e4f;
            const int64_t ne_e = node->numel();
            const float* nde = node->data();
            static long long explode_count = 0;
            for (int64_t q = 0; q < ne_e && explode_count < 8; ++q) {
                float v = nde[q];
                if (std::isfinite(v) && (v > EXPLODE_THRESH || v < -EXPLODE_THRESH)) {
                    auto opname = [](TensorF32* t)->int { return t ? (int)t->op : -1; };
                    TensorF32* s0 = node->src[0];
                    TensorF32* s1 = node->src[1];
                    fprintf(stderr,
                            "[explode-node] #%lld node_n=%d op=%d src0op=%d src1op=%d "
                            "dims=[%lld,%lld,%lld,%lld] numel=%lld idx=%lld val=%g | "
                            "s0:(op=%d dims=[%lld,%lld,%lld,%lld]) s0.s0:(op=%d) s1:(op=%d dims=[%lld,%lld,%lld,%lld]) s1.s0:(op=%d)\n",
                            explode_count, node_n, (int)node->op, opname(s0), opname(s1),
                            (long long)(node->shape().ndim()>0?node->shape().dims[0]:-1),
                            (long long)(node->shape().ndim()>1?node->shape().dims[1]:-1),
                            (long long)(node->shape().ndim()>2?node->shape().dims[2]:-1),
                            (long long)(node->shape().ndim()>3?node->shape().dims[3]:-1),
                            (long long)ne_e, (long long)q, (double)v,
                            opname(s0),
                            (long long)(s0&&s0->shape().ndim()>0?s0->shape().dims[0]:-1),
                            (long long)(s0&&s0->shape().ndim()>1?s0->shape().dims[1]:-1),
                            (long long)(s0&&s0->shape().ndim()>2?s0->shape().dims[2]:-1),
                            (long long)(s0&&s0->shape().ndim()>3?s0->shape().dims[3]:-1),
                            opname(s0?s0->src[0]:nullptr),
                            opname(s1),
                            (long long)(s1&&s1->shape().ndim()>0?s1->shape().dims[0]:-1),
                            (long long)(s1&&s1->shape().ndim()>1?s1->shape().dims[1]:-1),
                            (long long)(s1&&s1->shape().ndim()>2?s1->shape().dims[2]:-1),
                            (long long)(s1&&s1->shape().ndim()>3?s1->shape().dims[3]:-1),
                            opname(s1?s1->src[0]:nullptr));
                    // ⚠️ OP_DIV 专项（GNormBias 的 scale3=t/(norm+eps)）：打印 t(src0=relu) 与
                    //    denom(src1=norm+eps) 统计。若 t≈norm 但 norm 巨大而输入 x 正常 → 布局错位。
                    if ((int)node->op == 8 /*OP_DIV*/ && node->shape().ndim() >= 3) {
                        const float* td = s0 ? s0->data() : nullptr;
                        const float* dd_ = s1 ? s1->data() : nullptr;
                        if (td && s0->numel() > 0) {
                            float tmin = td[0], tmax = td[0];
                            for (int64_t kk = 1; kk < s0->numel(); ++kk) { float tv = td[kk]; if (tv<tmin)tmin=tv; if (tv>tmax)tmax=tv; }
                            fprintf(stderr, "  [div-gnorm] t(src0=relu) dims=[%lld,%lld,%lld] min=%g max=%g numel=%lld\n",
                                    (long long)(s0->shape().ndim()>0?s0->shape().dims[0]:-1),
                                    (long long)(s0->shape().ndim()>1?s0->shape().dims[1]:-1),
                                    (long long)(s0->shape().ndim()>2?s0->shape().dims[2]:-1),
                                    (double)tmin, (double)tmax, (long long)s0->numel());
                        }
                        if (dd_ && s1->numel() > 0) {
                            float dmin = dd_[0], dmax = dd_[0];
                            for (int64_t kk = 1; kk < s1->numel(); ++kk) { float dv = dd_[kk]; if (dv<dmin)dmin=dv; if (dv>dmax)dmax=dv; }
                            fprintf(stderr, "  [div-gnorm] denom(src1=norm+eps) dims=[%lld,%lld,%lld] min=%g max=%g numel=%lld\n",
                                    (long long)(s1->shape().ndim()>0?s1->shape().dims[0]:-1),
                                    (long long)(s1->shape().ndim()>1?s1->shape().dims[1]:-1),
                                    (long long)(s1->shape().ndim()>2?s1->shape().dims[2]:-1),
                                    (double)dmin, (double)dmax, (long long)s1->numel());
                        }
                    }
                    // ⚠️ PER_EDGE_MATMUL 专项：打印 kernel(src0) / gathered(src1) 的统计，
                    //    定位 SE3 核生成爆炸是 kernel 值异常还是 gathered(节点特征) 异常。
                    if ((int)node->op == 107 /*OP_PER_EDGE_MATMUL*/) {
                        const float* kd = s0 ? s0->data() : nullptr;
                        const float* gd = s1 ? s1->data() : nullptr;
                        if (kd && s0->numel() > 0) {
                            float kmin = kd[0], kmax = kd[0];
                            for (int64_t kk = 1; kk < s0->numel(); ++kk) {
                                float kv = kd[kk];
                                if (kv < kmin) kmin = kv;
                                if (kv > kmax) kmax = kv;
                            }
                            fprintf(stderr, "  [per-edge] KERNEL dims=[%lld,%lld,%lld] min=%g max=%g numel=%lld\n",
                                    (long long)(s0->shape().ndim()>0?s0->shape().dims[0]:-1),
                                    (long long)(s0->shape().ndim()>1?s0->shape().dims[1]:-1),
                                    (long long)(s0->shape().ndim()>2?s0->shape().dims[2]:-1),
                                    (double)kmin, (double)kmax, (long long)s0->numel());
                        } else {
                            fprintf(stderr, "  [per-edge] KERNEL data=null\n");
                        }
                        if (gd && s1->numel() > 0) {
                            float gmin = gd[0], gmax = gd[0];
                            for (int64_t kk = 1; kk < s1->numel(); ++kk) {
                                float gv = gd[kk];
                                if (gv < gmin) gmin = gv;
                                if (gv > gmax) gmax = gv;
                            }
                            fprintf(stderr, "  [per-edge] GATHERED dims=[%lld,%lld] min=%g max=%g numel=%lld\n",
                                    (long long)(s1->shape().ndim()>0?s1->shape().dims[0]:-1),
                                    (long long)(s1->shape().ndim()>1?s1->shape().dims[1]:-1),
                                    (double)gmin, (double)gmax, (long long)s1->numel());
                        } else {
                            fprintf(stderr, "  [per-edge] GATHERED data=null\n");
                        }
                    }
                    ++explode_count;
                    break;
                }
            }
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




#include "ppml/Backend.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <unordered_map>

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
        
        case OP_MUL_MAT: {
            // 前期 fp16 支持（2026-09-10）：A/B 同为 F16（数据 2 字节/元素）也可算，
            //   kernel 内转 fp32 累加，输出仍 F32。混合类型（一 F16 一 F32）暂不支持。
            if (!src1) return true;
            // int8 分块量化（2026-09-10 准备）：只有 kernel 就绪时才放开调度，
            //   否则量化输入会被送进 F32/F16 kernel 按 float 读块数据（错误/越界）。
            if (is_quantized_type(src1->type) || (src0 && is_quantized_type(src0->type))) {
                // 形状合法性：量化张量的最内维须为块元素数整数倍（量化块不能跨行）
                const bool shape_ok =
                    (src0 != nullptr) &&
                    (!is_quantized_type(src0->type) || src0->quant_shape_valid()) &&
                    (!is_quantized_type(src1->type) || src1->quant_shape_valid());
                return QUANT_MUL_MAT_KERNELS_READY && shape_ok &&
                       quant_mul_mat_combo_supported(src0->type, src1->type) &&
                       node->type == TENSOR_TYPE_F32;
            }
            const bool ok_f32 = (src0 && src0->type == TENSOR_TYPE_F32 &&
                                 src1->type == TENSOR_TYPE_F32);
            const bool ok_f16 = (src0 && src0->type == TENSOR_TYPE_F16 &&
                                 src1->type == TENSOR_TYPE_F16);
            return ok_f32 || ok_f16;
        }

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

    // ---- 产出/使用前 leaf 指纹（PPML_HASH_NODES=1）：参数/常量 leaf 若与激活互相覆盖，
    //      会表现为"两次运行（布局不同）在同一节点开始分歧"。leaf 不在 n_nodes 里，单独打。----
    if (getenv("PPML_HASH_NODES")) {
        for (int i = 0; i < cgraph->n_leafs(); ++i) {
            TensorF32* lf = cgraph->graph_leaf(i);
            if (!lf || !lf->data() || lf->nbytes() <= 0) continue;
            const uint8_t* q = reinterpret_cast<const uint8_t*>(lf->data());
            uint64_t h = 1469598103934665603ull;
            const size_t nb = (size_t)lf->nbytes();
            for (size_t k = 0; k < nb; ++k) { h ^= q[k]; h *= 1099511628211ull; }
            std::fprintf(stderr, "[NHL] leaf=%d op=%d flag=0x%x nb=%zu fnv=%016llx\n",
                         i, (int)lf->op, (unsigned)lf->flag, nb, (unsigned long long)h);
        }
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

    // 产出时刻指纹（见下方调用点注释）：只由 ith==0 打印，读值发生在 barrier 之后
    static const bool hash_nodes = (std::getenv("PPML_HASH_NODES") != nullptr);
    (void)0;
    static const int watch_op = [] {
        const char* w = std::getenv("PPML_HASH_WATCH");
        return w ? std::atoi(w) : -1;
    }();
    // 按"执行序号"watch 单个节点（0 基；序号 = 本 run 第几个被执行的节点，两次布局可对齐同一逻辑节点）
    static const long long watch_idx = [] {
        const char* w = std::getenv("PPML_HASH_WATCH_IDX");
        return w ? std::atoll(w) : -1;
    }();
    // 限定 watch 的张量字节数（避免 op 级 watch 打出上万个实例；0 = 不限）
    static const size_t watch_nb = [] {
        const char* w = std::getenv("PPML_HASH_WATCH_NB");
        return w ? (size_t)std::strtoull(w, nullptr, 0) : (size_t)0;
    }();
    static long long nh_seq = 0;   // 每个被执行节点的序号（只有 ith==0 递增 ⇒ 无竞争）

    // 结构身份 sid：hash(op, type, dims, nbytes, op_params[0..1], 各 src 的 sid)（递归、记忆化）。
    //   为什么需要：指纹只有 (op, nb, dims) 时，**无法区分"同形状但不同逻辑张量"**
    //   （例如两个 op=38 nb=6528 的 view）⇒ "读时内容 != 产出时内容" 的判定会有假阳性。
    //   sid 跨 run 可比（只依赖图结构）⇒ 同 sid 才允许比对数值。
    std::unordered_map<const TensorF32*, uint64_t> sid_map;
    std::function<uint64_t(TensorF32*)> sid_of = [&](TensorF32* t) -> uint64_t {
        if (!t) return 0;
        auto it = sid_map.find(t);
        if (it != sid_map.end()) return it->second;
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
        mix((uint64_t)(int)t->op);
        mix((uint64_t)(int)t->type);
        const int nd = t->shape().ndim();
        mix((uint64_t)nd);
        for (int i = 0; i < 4; ++i) mix((uint64_t)(i < nd ? t->shape().dims[i] : 1));
        mix((uint64_t)t->nbytes());
        mix((uint64_t)(uint32_t)t->op_params[0]);
        mix((uint64_t)(uint32_t)t->op_params[1]);
        if (t->buffer_ == nullptr && t->view_src == nullptr && t->data() != nullptr) {
            // 叶子：用 PARAM/CONST 标志参与（**不要**用 data 指针做盐——那会让所有下游 sid 跨 run 不可比，
            //   sid 的用途正是跨 run 对齐同一逻辑节点；同形状不同权重撞车由 fnv 值差异暴露）。
            mix((uint64_t)(t->flag & (TENSOR_FLAG_PARAM | TENSOR_FLAG_CONST)));
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (!t->src[s]) break;
            mix(sid_of(t->src[s]));
        }
        sid_map[t] = h;
        return h;
    };
    auto nh = [&](TensorF32* nd, long long seq) {
        if (!nd || nd->op == OP_NONE) return;
        const size_t nb = (size_t)nd->nbytes();
        if (nb == 0) return;
        const int ndim = nd->shape().ndim();
        const int64_t d0 = ndim > 0 ? nd->shape().dims[0] : 0;
        const int64_t d1 = ndim > 1 ? nd->shape().dims[1] : 1;
        const int64_t d2 = ndim > 2 ? nd->shape().dims[2] : 1;
        const int64_t d3 = ndim > 3 ? nd->shape().dims[3] : 1;
        TensorF32* real = nd;
        int g = 0;
        while (real && !real->data() && real->view_src && g++ < 64) real = real->view_src;
        if (!real || !real->data()) {
            std::fprintf(stderr, "[NH] op=%d nb=%zu dims=[%lld,%lld,%lld,%lld] NO_DATA\n",
                         (int)nd->op, nb, (long long)d0, (long long)d1, (long long)d2, (long long)d3);
            return;
        }
        const uint8_t* p = reinterpret_cast<const uint8_t*>(real->data());
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < nb; ++i) { h ^= p[i]; h *= 1099511628211ull; }
        std::fprintf(stderr, "[NH] seq=%lld sid=%016llx op=%d nb=%zu dims=[%lld,%lld,%lld,%lld] ptr=%p fnv=%016llx\n",
                     seq, (unsigned long long)sid_of(nd), (int)nd->op, nb,
                     (long long)d0, (long long)d1, (long long)d2, (long long)d3,
                     (const void*)p, (unsigned long long)h);
        const bool watch_now = ((watch_op >= 0 && (int)nd->op == watch_op) ||
                                (watch_idx >= 0 && seq == watch_idx + 1)) &&
                               (watch_nb == 0 || nb == watch_nb);

        // watch 模式（PPML_HASH_WATCH=<op>）：该 op 每次执行时，把它各 src **此刻** 的哈希打出来。
        // 用途：若某 src 的"此刻哈希" != 它在产出时刻记录的 [NH] 哈希 ⇒ 该张量的 buffer 被提前复用/覆盖了
        //   （= 跨张量别名 bug），这正是"输出因布局不同而不同"的机制。
        if (watch_now) {
            // 先打指针/重叠（关键判据：dst 与某 src 的地址区间重叠 ⇒ kernel 一边写一边读 ⇒ 结果依布局/线程划分）
            const uintptr_t self_p = (uintptr_t)real->data();
            const size_t self_nb = (size_t)nd->nbytes();
            {
                const float* df = reinterpret_cast<const float*>(real->data());
                const int64_t nfl = (int64_t)(self_nb / sizeof(float));
                std::fprintf(stderr, "[NHW] ptr self=%p nb=%zu head=[%g,%g,%g] tail=[%g,%g,%g]\n",
                             (const void*)self_p, self_nb,
                             nfl > 0 ? df[0] : 0.0f, nfl > 1 ? df[1] : 0.0f, nfl > 2 ? df[2] : 0.0f,
                             nfl > 2 ? df[nfl - 3] : 0.0f, nfl > 1 ? df[nfl - 2] : 0.0f,
                             nfl > 0 ? df[nfl - 1] : 0.0f);
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                TensorF32* sc = nd->src[s];
                if (!sc) continue;
                TensorF32* rsc = sc;
                int g2 = 0;
                while (rsc && !rsc->data() && rsc->view_src && g2++ < 64) rsc = rsc->view_src;
                uint64_t hs = 0;
                bool ok = false;
                if (rsc && rsc->data() && sc->nbytes() > 0) {
                    const uint8_t* q = reinterpret_cast<const uint8_t*>(rsc->data());
                    hs = 1469598103934665603ull;
                    for (size_t i = 0; i < (size_t)sc->nbytes(); ++i) { hs ^= q[i]; hs *= 1099511628211ull; }
                    ok = true;
                }
                const uintptr_t sp = rsc ? (uintptr_t)rsc->data() : 0;
                const size_t snb = sc ? (size_t)sc->nbytes() : 0;
                const bool ov = sp && self_p &&
                                (self_p < sp + snb) && (sp < self_p + self_nb);
                std::fprintf(stderr,
                             "[NHW] self_op=%d src%d: op=%d nb=%zu ptr=%p sid=%016llx OVERLAP=%d\n",
                             (int)nd->op, s, (int)sc->op, snb, (const void*)sp,
                             (unsigned long long)sid_of(sc), (int)ov);
                if (ok) {
                    std::fprintf(stderr, "[NHW]   self_op=%d src%d op=%d fnv=%016llx\n",
                                 (int)nd->op, s, (int)sc->op, (unsigned long long)hs);
                }
            }
        }
    };

    for (int node_n = 0;
         node_n < cgraph->n_nodes() &&
         tp->abort.load(std::memory_order_relaxed) != node_n;
         node_n++) {

        TensorF32 * node = cgraph->graph_node(node_n);

        // watch 模式的**执行前**读值：此刻上一节点的 barrier 已过、本节点还没开始写
        //   ⇒ src 的字节就是 kernel 将读到的字节（不受"之后被复用"污染）。
        //   与执行后（[NHW]）以及另一 run 的 [NHP] 三者对比，才能判定差异到底在哪一侧。
        if (ith == 0) ++nh_seq;   // 节点执行序号（1 基；两次布局可用同一序号对齐同一逻辑节点）
        if (ith == 0 &&
            ((watch_op >= 0 && (int)node->op == watch_op) ||
             (watch_idx >= 0 && nh_seq == watch_idx + 1)) &&
            (watch_nb == 0 || (size_t)node->nbytes() == watch_nb)) {
            {
                // 自身信息（含 view 链）：用于定位"这个节点是哪个张量、落在哪段 buffer"
                TensorF32* vs = node->view_src;
                int nsrc = 0;
                for (int s = 0; s < GGML_MAX_SRC; ++s) if (node->src[s]) ++nsrc;
                const int ndv = node->shape().ndim();
                std::fprintf(stderr,
                             "[NHI] seq=%lld op=%d nb=%zu ndim=%d dims=[%lld,%lld,%lld,%lld] "
                             "ptr=%p view_src=%p(vs_op=%d vs_nb=%zu vs_ptr=%p) n_src=%d\n",
                             nh_seq, (int)node->op, (size_t)node->nbytes(), ndv,
                             (long long)(ndv > 0 ? node->shape().dims[0] : 0),
                             (long long)(ndv > 1 ? node->shape().dims[1] : 1),
                             (long long)(ndv > 2 ? node->shape().dims[2] : 1),
                             (long long)(ndv > 3 ? node->shape().dims[3] : 1),
                             (const void*)node->data(), (const void*)vs,
                             vs ? (int)vs->op : -1, vs ? (size_t)vs->nbytes() : 0,
                             vs ? (const void*)vs->data() : nullptr, nsrc);
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                TensorF32* sc = node->src[s];
                if (!sc) continue;
                TensorF32* rsc = sc;
                int g3 = 0;
                while (rsc && !rsc->data() && rsc->view_src && g3++ < 64) rsc = rsc->view_src;
                if (!rsc || !rsc->data() || sc->nbytes() <= 0) {
                    std::fprintf(stderr, "[NHP] self_op=%d src%d op=%d nb=%zu NO_DATA\n",
                                 (int)node->op, s, (int)sc->op, (size_t)sc->nbytes());
                    continue;
                }
                const uint8_t* q = reinterpret_cast<const uint8_t*>(rsc->data());
                uint64_t hs = 1469598103934665603ull;
                for (size_t i = 0; i < (size_t)sc->nbytes(); ++i) { hs ^= q[i]; hs *= 1099511628211ull; }
                const float* qf = reinterpret_cast<const float*>(q);
                const int64_t nfl = (int64_t)(sc->nbytes() / sizeof(float));
                const int ndv = sc->shape().ndim();
                std::fprintf(stderr,
                             "[NHP] self_op=%d src%d sid=%016llx op=%d nb=%zu dims=[%lld,%lld,%lld,%lld] view=%d "
                             "ptr=%p fnv=%016llx head=[%g,%g,%g] tail=[%g,%g,%g] nth=%d\n",
                             (int)node->op, s, (unsigned long long)sid_of(sc), (int)sc->op, (size_t)sc->nbytes(),
                             (long long)(ndv > 0 ? sc->shape().dims[0] : 0),
                             (long long)(ndv > 1 ? sc->shape().dims[1] : 1),
                             (long long)(ndv > 2 ? sc->shape().dims[2] : 1),
                             (long long)(ndv > 3 ? sc->shape().dims[3] : 1),
                             (int)(sc->view_src != nullptr),
                             (const void*)q, (unsigned long long)hs,
                             nfl > 0 ? qf[0] : 0.0f, nfl > 1 ? qf[1] : 0.0f, nfl > 2 ? qf[2] : 0.0f,
                             nfl > 2 ? qf[nfl - 3] : 0.0f, nfl > 1 ? qf[nfl - 2] : 0.0f, nfl > 0 ? qf[nfl - 1] : 0.0f,
                             (int)params.nth);
            }
        }

        dispatch_node(node, &params);

        if (ith == 0 && cplan->abort_callback &&
            cplan->abort_callback(cplan->abort_callback_data)) {
            tp->abort.store(node_n + 1, std::memory_order_relaxed);
        }

        if (node_n + 1 < cgraph->n_nodes()) {
            tp->barrier_wait();
        }

        // ---- 产出时刻指纹（PPML_HASH_NODES=1，2026-09-14 加）----
        //   为什么必须在这里：任何"split 结束后再读"的指纹都可能读到已被 gallocr 复用的 buffer；
        //   此处紧跟 barrier（含最后一个节点见循环后同款代码）⇒ 读到的就是"刚算完、未被复用"的真值。
        //   用途：把两次运行（不同布局）的 [NH] 序列逐行 diff ⇒ 第一个 fnv 不同的节点 = 分歧起点。
        if (ith == 0 && hash_nodes && node_n + 1 < cgraph->n_nodes()) nh(node, nh_seq);

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
    // 最后一个节点没有"下一轮 barrier"，在这里补一次（此时全体线程已 barrier，读值是安全/新鲜的）
    if (ith == 0 && hash_nodes && cgraph->n_nodes() > 0) {
        nh(cgraph->graph_node(cgraph->n_nodes() - 1), nh_seq);
    }
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
            // 逐元素 op：F32/F16 都不需要额外缓冲（F16 由 kernel 内逐元素转 fp32）。
            // int8 量化（Q8_0/Q8_1）尚未支持；若将来支持，需要：
            //   if (is_quantized_type(node->src[0]->type))
            //       cur = sizeof(float) * node->src[0]->shape().dims[0] * n_tasks;
            break;

        // ===== 矩阵乘：量化权重/激活的反量化暂存（2026-09-10 int8 准备）=====
        case OP_MUL_MAT: {
            // F32/F16 路径不需要额外缓冲（cur 保持 0，行为与之前完全一致）；
            // 量化路径的 kernel（把块反量化成 f32/f16 再算）需要每线程的反量化缓冲：
            //   权重按行反量化 K 个元素 → sizeof(float) * K；激活同理（各自独立缓冲）。
            const TensorF32* s0 = node->src[0];
            const TensorF32* s1 = node->src[1];
            const int64_t K = s0 ? s0->shape().dims[0] : 0;
            if (K > 0) {
                if (s0 && is_quantized_type(s0->type))
                    cur += sizeof(float) * static_cast<size_t>(K) * static_cast<size_t>(n_tasks);
                if (s1 && is_quantized_type(s1->type))
                    cur += sizeof(float) * static_cast<size_t>(K) * static_cast<size_t>(n_tasks);
            }
        } break;

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
            // 2026-09-11：dim() 安全访问（缺失维=1）—— 原裸读 dims[2]/dims[3]，<4D 节点上
            //   越界读 std::vector 是 UB → work_size 可能爆炸成超大分配。
            cur = sizeof(float) * static_cast<size_t>(node->shape().dim(2))
                                * static_cast<size_t>(node->shape().dim(3))
                                * static_cast<size_t>(n_tasks);

        } break;
        case OP_FLASH_ATTN_BACK: {
            // 反向还需要存储 dS
            // D = head dim (如 64 or 128)
            // Q's head dim
            const int64_t D = node->src[0]->shape().dim(0);

            // Lkv = K 的序列长度，对齐到 UNROLL
            const int64_t ne11 = align_up(node->src[1]->shape().dim(1), SOFT_MAX_UNROLL);

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

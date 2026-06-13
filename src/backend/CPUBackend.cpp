

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

} // namespace rfaa

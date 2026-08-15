

#include "ppml/Backend.h"

namespace ppml {

// ===== worker_loop：线程常驻 + sleep/wake =====
void worker_loop(ThreadState * state) {
    ThreadPool * tp = state->pool;

    while (true) {
        {
            std::unique_lock<std::mutex> lock(tp->mtx);
            tp->cv.wait(lock, [tp]() {
                return tp->n_graph > 0 || tp->stop;
            });
        }
        if (tp->stop) break;

        // 调用 CPUBackend 注入的回调
        tp->worker_fn_(state);

        if (tp->n_graph.fetch_sub(1) == 1)
            tp->n_graph.store(0);
    }
}

// ===== init =====
void ThreadPool::init(int n_threads, void (*fn)(ThreadState *)) {
    n_threads_max = n_threads;
    worker_fn_    = fn;
    workers       = new ThreadState[n_threads]();

    for (int i = 0; i < n_threads; i++) {
        workers[i].id   = i;
        workers[i].pool = this;
        if (i > 0)
            workers[i].os_thread = std::thread(worker_loop, &workers[i]);
    }
}

void ThreadPool::free() {
    stop.store(true);
    cv.notify_all();
    for (int i = 1; i < n_threads_max; i++)
        workers[i].os_thread.join();
    delete[] workers;
}

void ThreadPool::submit(ComputeGraph * g, ComputePlan * p) {
    cgraph = g;
    cplan  = p;
    n_threads_cur.store(p->n_threads);
    abort.store(-1);
    n_graph.fetch_add(1);
    cv.notify_all();
}

void ThreadPool::barrier_wait() {
    int passed = n_barrier_passed.fetch_add(1, std::memory_order_acq_rel);
    int nth = n_threads_cur.load(std::memory_order_relaxed);
    if (passed == nth - 1) {
        n_barrier_passed.store(0, std::memory_order_release);
        n_barrier.fetch_add(1, std::memory_order_release);
    } else {
        int cur = n_barrier.load(std::memory_order_acquire);
        while (n_barrier.load(std::memory_order_acquire) == cur) {
            // CPU yield hint (cross-platform)
#if defined(_MSC_VER)
            _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#else
            // fallback for ARM etc.
#endif
        }
    }
}

} // namespace ppml

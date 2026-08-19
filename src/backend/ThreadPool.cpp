
#include "ppml/Backend.h"

namespace ppml {

// 每线程本地 sense（感翻转屏障）。thread_local 使每个 OS 线程独立翻转，
// 跨图持续（每个线程在每张图里调用屏障次数一致，sense 保持同步）。
static thread_local int g_tp_sense = 0;

// ===== worker_loop：线程常驻，按图代际（graph_seq）被唤醒 =====
// 每次 submit 递增 graph_seq 并 notify_all。worker 醒来后消费该代际（last_seen=graph_seq），
// 调用 worker_fn_（即 compute_thread，内含节点级感翻转屏障）与主线程同步计算整个图。
// 计算完回到 cv.wait 等待下一个图。顺序训练下 submit 都在上一图完成后发生，
// 因此每个 worker 此时必在 cv.wait 中，notify_all 必然唤醒全部 n-1 个 worker。
void worker_loop(ThreadState * state) {
    ThreadPool * tp = state->pool;

    while (true) {
        {
            std::unique_lock<std::mutex> lock(tp->mtx);
            tp->cv.wait(lock, [tp, state]() {
                return tp->stop.load() || tp->graph_seq.load() != state->last_seen;
            });
        }
        if (tp->stop.load()) break;

        // 消费当前图代际（只消费一次，防止重复计算）
        state->last_seen = tp->graph_seq.load();

        // 调用 CPUBackend 注入的回调（compute_thread：逐节点 + 屏障同步）
        tp->worker_fn_(state);
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
    workers = nullptr;
}

void ThreadPool::submit(ComputeGraph * g, ComputePlan * p) {
    cgraph = g;
    cplan  = p;
    n_threads_cur.store(p->n_threads);
    abort.store(-1);
    // 感翻转屏障每个参与代际初值 = 当前线程数（主线程 + worker）
    n_barrier_passed.store(p->n_threads, std::memory_order_release);
    // 唤醒所有 worker（epoch +1）
    graph_seq.fetch_add(1);
    cv.notify_all();
}

// ===== 感翻转屏障（sense-reversing barrier）=====
// 要求：每个参与线程（主线程 + n-1 worker）对同一屏障调用相同次数。
// 每线程本地 sense 翻转（thread_local）；fetch_sub 到 1 者（最后到达）重置计数器并
// 翻转全局 sense，其余线程自旋直到全局 sense 等于其本地 sense。
// nth<=1（单线程）无需同步。
// 注：kernel 内部子屏障（如 kernel_mul_mat 的 GPU/CPU 路径各 2 次）与本节点边界屏障
// 复用同一屏障，只要所有线程对每节点调用次数一致（dispatch 对全部线程跑同一 kernel）
// 即正确。单线程时 nth=1 直接返回，无任何开销。
void ThreadPool::barrier_wait() {
    int nth = n_threads_cur.load(std::memory_order_relaxed);
    if (nth <= 1) return;

    int s = (g_tp_sense ^= 1);
    if (n_barrier_passed.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        n_barrier_passed.store(nth, std::memory_order_release);
        n_barrier_sense.fetch_xor(1, std::memory_order_release);
    } else {
        while (n_barrier_sense.load(std::memory_order_acquire) != s) {
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

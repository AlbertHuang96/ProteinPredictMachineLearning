
#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include "ComputeGraph.h"

#define CACHE_LINE_SIZE 64

namespace rfaa {

// ==================== 前向声明 ====================
class Backend;
class CPUBackend;

// ==================== 状态码 ====================
enum class Status {
    SUCCESS = 0,
    ALLOC_FAILED,
    NOT_SUPPORTED,
    ABORTED
};

// ==================== 辅助类型 ====================

struct ThreadState {
    int          id;       // 0 ~ n_threads-1
    struct ThreadPool * pool;
    std::thread  os_thread;
};

struct ComputePlan {
    size_t       work_size  = 0;
    uint8_t    * work_data  = nullptr;
    int          n_threads  = 1;
    struct ThreadPool * threadpool = nullptr;

    bool (*abort_callback)(void *) = nullptr;
    void * abort_callback_data     = nullptr;
};

struct ComputeParams {
    int          ith;
    int          nth;
    size_t       wsize;
    uint8_t    * wdata;
    struct ThreadPool * threadpool;
};

// ==================== ThreadPool ====================

struct alignas(64) ThreadPool {
    // ===== 生命周期 =====
    std::mutex              mtx;
    std::condition_variable cv;
    std::atomic<bool>       stop  {false};

    // ===== 任务 =====
    std::atomic<int>        n_graph {0};
    ComputeGraph *          cgraph = nullptr;
    ComputePlan  *          cplan  = nullptr;

    // ===== 同步 =====
    alignas(64) std::atomic<int> n_barrier        {0};
    alignas(64) std::atomic<int> n_barrier_passed {0};
    alignas(64) std::atomic<int> current_chunk    {0};

    // ===== 控制 =====
    std::atomic<int> n_threads_cur {1};
    std::atomic<int> abort         {-1};

    // ===== 线程 =====
    ThreadState *  workers = nullptr;
    int            n_threads_max = 1;
    Status         ec = Status::SUCCESS;

    // ===== 方法 =====
    void init(int n_threads, void (*worker_fn)(ThreadState *));
    void free();
    void submit(ComputeGraph * g, ComputePlan * p);
    void barrier_wait();

private:
    void (*worker_fn_)(ThreadState *) = nullptr;   // 由 CPUBackend 注入
};

// ==================== Backend (抽象基类) ====================

class Backend {
public:
    virtual ~Backend() = default;

    virtual const char * get_name() const = 0;
    virtual Status graph_compute(ComputeGraph * cgraph) = 0;
    virtual void   synchronize() = 0;
};

// ==================== CPUBackend ====================

class CPUBackend : public Backend {
public:
    explicit CPUBackend(int n_threads = 1);
    ~CPUBackend() override;

    // ===== 工厂 =====
    static std::unique_ptr<CPUBackend> create(int n_threads = 1);

    // ===== Backend 接口 =====
    const char * get_name() const override { return "CPU"; }
    Status       graph_compute(ComputeGraph * cgraph) override;
    void         synchronize() override {}

    // ===== 公开（图规划，可以被外部调用预估算资源）=====
    ComputePlan  graph_plan(ComputeGraph * cgraph) const;

private:
    // ===== 图计算线程（静态，无 this 依赖）=====
    static void compute_thread(ThreadState * state);

    // ===== op 分发 =====
    static Status  dispatch_node(Tensor * node, ComputeParams * p);

    static int get_n_tasks(Tensor * node, int n_threads);
    static size_t estimate_work_size(Tensor * node, int n_threads, int n_tasks = -1);

    // ===== kernels (static: 无需 this, 仅操作张量数据) =====
    static void kernel_elemwise(Tensor * node, ComputeParams * p);
    static void kernel_mul_mat (Tensor * node, ComputeParams * p);
    static void kernel_softmax (Tensor * node, ComputeParams * p);
    static void kernel_rms_norm(Tensor * node, ComputeParams * p);
    static void kernel_silu    (Tensor * node);
    static void kernel_gelu    (Tensor * node);
    static void kernel_relu    (Tensor * node);
    static void kernel_dup     (Tensor * node);
    static void kernel_scale   (Tensor * node, ComputeParams * p);
    static void kernel_add1    (Tensor * node, ComputeParams * p);
    static void kernel_sum     (Tensor * node, ComputeParams * p);
    static void kernel_mean    (Tensor * node, ComputeParams * p);

    // ===== 数据成员 =====
    int          n_threads_;
    ThreadPool * threadpool_ = nullptr;
    uint8_t    * work_data_  = nullptr;
    size_t       work_size_  = 0;
};

} // namespace rfaa

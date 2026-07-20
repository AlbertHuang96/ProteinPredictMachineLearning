
#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include <vector>
#include <unordered_map>
#include "ComputeGraph.h"

static constexpr int SOFT_MAX_UNROLL = 32;  // SIMD unroll

#define CACHE_LINE_SIZE 64

namespace rfaa {

// ==================== 前向声明 ====================
class Backend;
class CPUBackend;
struct RFAAContext;
struct ThreadPool;
template<typename T> class Tensor;
using TensorF32 = Tensor<float>;

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

// Buffer
enum class BufferUsage {
    WEIGHTS = 0;
    COMPUTE = 1;
    STORAGE = 2;
};

// buffer type ggml_backend_buffer_type_t
class BufferType {
public:
    virtual const char* get_name() const = 0;
    virtual void* alloc(size_t size) = 0;
    virtual void free(void* ptr) = 0;
    virtual size_t get_alignment() const { return 32; }
    virtual size_t get_max_size() const { return SIZE_MAX; }
    // default size SIZE_MAX

    virtual size_t get_alloc_size(const TensorF32* tensor) const {
        return tensor->nbytes();
    }

    virtual bool is_host() const { return false; }
    virtual ~BufferType() = default;
}

// ggml_backend_buffer
class  Buffer {
public:
    Buffer(BufferType* buft, size_t size, BufferUsage usage = BufferUsage::COMPUTE)
    : buft_(buft), size_(size), usage_(usage) {}

    virtual ~Buffer() = default;

    const BufferType* type()  const { return buft_; }
    size_t            size()  const { return size_; }
    BufferUsage       usage() const { return usage_; }

    // ggml_backend_buffer_i.get_base
    virtual void* data() = 0;
    virtual void memset_tensor(TensorF32* tensor, uint8_t value, size_t offset, size_t size) {
        uint8_t ptr = static_cast<uint8_t*>(data());
        std::memset(ptr + offset, value, size);
    }

    virtual void set_tensor(TensorF32* tensor, const void* data, size_t offset, size_t size) {
        uint8_t* ptr = static_cast<uint8_t*>(data());
        std::memcpy(ptr + offset, data, size);
    }

    virtual void get_tensor(const TensorF32* tensor,
                            void* data, size_t offset, size_t size) {
        const uint8_t* ptr = static_cast<const uint8_t*>(data());
        std::memcpy(data, ptr + offset, size);
    }
    
    // cpy_tensor  not implement yet

    // clear the whole buffer with value
    virtual void clear(uint8_t value) {
        std::memset(data(), value, size_);
    }

protected:
    BufferType* buft_  = nullptr;
    size_t      size_  = 0;
    BufferUsage usage_ = BufferUsage::COMPUTE;

}


// ==================== Backend (抽象基类) ====================

class Backend {
public:
    virtual ~Backend() = default;

    virtual const char * get_name() const = 0;
    virtual Status graph_compute(ComputeGraph * cgraph) = 0;
    virtual void   synchronize() = 0;

    // ===== 调度器需要的能力查询 =====
    // 是否支持该 op
    virtual bool supports_op(TensorF32 * node) const { return true; }

    // 该后端的默认 buffer 类型
    virtual const BufferType* buffer_type() const = 0;

    // 是否支持给定的 buffer 类型（跨后端传输用）
    virtual bool supports_buffer_type(const BufferType* buft) const { return false; }
    // for CPU RAM host memory buffer it is always true
    //static bool ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {  
    //return ggml_backend_buft_is_host(buft) 

    // 优先级（越大越优先被调度）
    virtual int priority() const { return 0; }
};

class BackendScheduler {
public:
    BackendScheduler();
    ~BackendScheduler();

    // 注册后端（按 priority 自动排序）
    void add_backend(Backend * backend);

    // 核心：将一张图分裂成多段，每段分配给对应后端
    void split_graph(ComputeGraph * graph);

    bool alloc_splits();

    // 获取分裂后的第 i 段子图
    ComputeGraph * get_split(int i);
    int n_splits() const { return n_splits_; }

    // 获取指定 tensor 的"绑定后端"
    int tensor_backend_id(Tensor * t) const;
    int tensor_backend_id(Tensor * t, int default_id) const;

private:
    // ===== 三趟扫描 =====
    void pass_assign_leafs(ComputeGraph * graph);        // 第一趟：叶子节点分配
    void pass_expand_assignments(ComputeGraph * graph);   // 第二趟：扩展分配
    void pass_fill_unassigned(ComputeGraph * graph);      // 第三趟：填充未分配节点

    // helper
    bool is_view_op(int op) const;
    void set_backend_if_supported(Tensor * node, int backend_id);
    int  count_supported_inputs(Tensor * node, int backend_id) const;
    bool tensor_buffer_compatible(const Tensor * src, int backend_id) const;

    // alloc_splits invoke this function to allocate the memory
    bool reserve_graph_memory();

    static constexpr int MAX_SPLITS = 64;

    // ===== 数据成员 =====
    std::vector<Backend *> backends_;          // 按优先级排序的后端列表
    int n_backends_ = 0;

    // tensor → backend_id 映射
    using BackendMap = std::unordered_map<const Tensor *, int>;
    BackendMap backend_map_;

    ComputeGraph* current_graph_ = nullptr;
    
    std::vector<int> node_backend_id_;
    std::vector<int> prev_node_backend_id_;
    std::vector<int> leaf_backend_id_;
    std::vector<int> prev_leaf_backend_id_;
    
    std::vector<int> bufts_;

    bool graph_reserved_ = false;

    // 分裂结果
    int n_splits_ = 0;
    ComputeGraph * splits_[MAX_SPLITS];   // 每段子图
    int split_backend_[MAX_SPLITS];       // 每段对应的后端

    // 图输入收集
    int n_graph_inputs_ = 0;
    Tensor * graph_inputs_[256];

    // context
    RFAAContext * ctx_ = nullptr;
};

// ==================== CPUBackend ====================

class CPUBackend : public Backend {
public:
    explicit CPUBackend(int n_threads = 1);
    ~CPUBackend() override;

    // ===== 工厂 =====
    static std::unique_ptr<CPUBackend> create(int n_threads = 1);

    // ===== Backend 接口 =====
    virtual bool supports_op(TensorF32 * node) const override;
    
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
    static constexpr int64_t align_up(int64_t n, int64_t align) {
        return (n + align - 1) / align * align;
    }

    // ===== kernels (static: 无需 this, 仅操作张量数据) =====
    static void kernel_elemwise(Tensor * node, ComputeParams * p);
    static void kernel_mul_mat (Tensor * node, ComputeParams * p);
    static void kernel_softmax (Tensor * node, ComputeParams * p);
    static void kernel_softmax_back(Tensor * node, ComputeParams * p);
    static void kernel_rms_norm (Tensor * node, ComputeParams * p);
    static void kernel_norm     (Tensor * node, ComputeParams * p);
    static void kernel_norm_back(Tensor * node, ComputeParams * p);
    static void kernel_silu    (Tensor * node);
    static void kernel_gelu    (Tensor * node);
    static void kernel_relu    (Tensor * node);
    static void kernel_dup     (Tensor * node);
    static void kernel_scale   (Tensor * node, ComputeParams * p);
    static void kernel_add1    (Tensor * node, ComputeParams * p);
    static void kernel_sum     (Tensor * node, ComputeParams * p);
    static void kernel_mean    (Tensor * node, ComputeParams * p);

    static void kernel_sigmoid (Tensor * node, ComputeParams * p);

    // ===== 数据成员 =====
    int          n_threads_;
    ThreadPool * threadpool_ = nullptr;
    uint8_t    * work_data_  = nullptr;
    size_t       work_size_  = 0;
};

// ==================== CUDABackend ====================

class CUDABackend : public Backend {
public:
    CUDABackend(int device_id = 0);
    ~CUDABackend() override;

    const char * name() const override { return "CUDA"; }

    Status init() override;
    Status graph_compute(ComputeGraph * cgraph, ComputePlan * plan) override;
    bool   supports_op(enum tensor_op op) override;
    bool   supports_buftype(int /* buft */) override { return false; }

    // 设备端 buffer 分配
    BufferType buffer_type() const override { return BufferType::CUDA; }
    Tensor * buffer_new(size_t nbytes);

private:
    int device_id_ = 0;

    // CUDA dispatch
    static Status dispatch_node(Tensor * node, ComputeParams * p);

    // CUDA kernels
    // elemwise op
    static void kernel_elemwise_add_cuda(Tensor * node, ComputeParams * p);
    static void kernel_elemwise_sub_cuda(Tensor * node, ComputeParams * p);
    static void kernel_elemwise_mul_cuda(Tensor * node, ComputeParams * p);
    static void kernel_elemwise_div_cuda(Tensor * node, ComputeParams * p);

    static void kernel_mul_mat_cuda (Tensor * node, ComputeParams * p);
    static void kernel_softmax_cuda (Tensor * node, ComputeParams * p);
    static void kernel_softmax_back_cuda(Tensor * node, ComputeParams * p);
    static void kernel_norm_cuda     (Tensor * node, ComputeParams * p);
    static void kernel_norm_back_cuda(Tensor * node, ComputeParams * p);
    static void kernel_dup_cuda      (Tensor * node);
    static void kernel_scale_cuda    (Tensor * node, ComputeParams * p);
    static void kernel_add1_cuda     (Tensor * node, ComputeParams * p);
    static void kernel_sum_cuda      (Tensor * node, ComputeParams * p);
    static void kernel_mean_cuda     (Tensor * node, ComputeParams * p);

    // CUDA unary kernels
    static void kernel_relu_cuda   (Tensor * node);
    static void kernel_gelu_cuda   (Tensor * node);
    static void kernel_sigmoid_cuda(Tensor * node);
    static void kernel_silu_cuda   (Tensor * node);
    static void kernel_tanh_cuda   (Tensor * node);
    static void kernel_exp_cuda    (Tensor * node);
};

} // namespace rfaa

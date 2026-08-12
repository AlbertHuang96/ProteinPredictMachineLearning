

#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include "ComputeGraph.h"
#include "Context.h"

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

    friend void worker_loop(ThreadState * state);

private:
    void (*worker_fn_)(ThreadState *) = nullptr;   // 由 CPUBackend 注入
};

// Buffer
enum class BufferUsage {
    WEIGHTS = 0,
    COMPUTE = 1,
    STORAGE = 2,
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
};

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
        uint8_t* ptr = static_cast<uint8_t*>(data());
        std::memset(ptr + offset, value, size);
    }

    virtual void set_tensor(TensorF32* tensor, const void* src, size_t offset, size_t size) {
        uint8_t* ptr = static_cast<uint8_t*>(data());
        std::memcpy(ptr + offset, src, size);
    }

    virtual void get_tensor(const TensorF32* tensor,
                            void* dst, size_t offset, size_t size) {
        const uint8_t* ptr = static_cast<const uint8_t*>(data());
        std::memcpy(dst, ptr + offset, size);
    }

    // 对标 ggml_backend_buffer_is_host
    virtual bool is_host() const {
        return buft_ ? buft_->is_host() : false;
    }

    // 对标 ggml_backend_buffer_copy_tensor
    // 跨 buffer 直接拷贝（如 GPU→GPU cudaMemcpyDeviceToDevice）
    // 返回 true 表示成功，false 表示需要 fallback 到 staging buffer 方式
    virtual bool cpy_tensor(const TensorF32* src, TensorF32* dst,
                            size_t src_offset, size_t dst_offset, size_t size) {
        return false;  // 默认不支持
    }

    // clear the whole buffer with value
    virtual void clear(uint8_t value) {
        std::memset(data(), value, size_);
    }

    // 对标 ggml_backend_buffer_set_usage
    virtual void set_usage(BufferUsage usage) { usage_ = usage; }

protected:
    BufferType* buft_  = nullptr;
    size_t      size_  = 0;
    BufferUsage usage_ = BufferUsage::COMPUTE;
};

// ============================================================
// CUDABufferType — GPU 端 buffer 分配器
// ============================================================
class CUDABufferType : public BufferType {
public:
    explicit CUDABufferType(int device_id = 0) : device_id_(device_id) {}

    const char* get_name() const override { return "CUDA"; }
    void* alloc(size_t size) override;
    void free(void* ptr) override;
    size_t get_alignment() const override { return 128; }  // CUDA 对齐 128 bytes
    bool is_host() const override { return false; }

    int device_id() const { return device_id_; }

    static CUDABufferType* instance(int device_id = 0);

private:
    int device_id_ = 0;
};

// ============================================================
// DefaultBuffer — 通用 buffer 实现（对标 ggml_backend_buffer）
// ============================================================
class DefaultBuffer : public Buffer {
public:
    DefaultBuffer(BufferType* buft, size_t size, BufferUsage usage = BufferUsage::COMPUTE)
        : Buffer(buft, size, usage) {
        if (size > 0) {
            ptr_ = static_cast<uint8_t*>(buft->alloc(size));
            if (!ptr_) {
                throw RFAAError("DefaultBuffer: allocation failed");
            }
            own_ptr_ = true;
        }
    }

    // 外部已有内存的包装
    DefaultBuffer(BufferType* buft, size_t size, void* external_ptr,
                  BufferUsage usage = BufferUsage::COMPUTE)
        : Buffer(buft, size, usage), ptr_(static_cast<uint8_t*>(external_ptr)), own_ptr_(false) {}

    ~DefaultBuffer() override {
        if (own_ptr_ && ptr_ && buft_) {
            buft_->free(ptr_);
        }
    }

    void* data() override { return ptr_; }

    // ===== 覆盖 set_tensor / get_tensor 以支持 GPU =====
    void set_tensor(TensorF32* tensor, const void* data, size_t offset, size_t size) override {
        if (is_host()) {
            std::memcpy(ptr_ + offset, data, size);
        } else {
            // GPU buffer: cudaMemcpy HostToDevice
            const CUDABufferType* cuda_buft = dynamic_cast<const CUDABufferType*>(buft_);
            int device = cuda_buft ? cuda_buft->device_id() : 0;
            cudaSetDevice(device);
            cudaMemcpy(ptr_ + offset, data, size, cudaMemcpyHostToDevice);
        }
    }

    void get_tensor(const TensorF32* tensor, void* data, size_t offset, size_t size) override {
        if (is_host()) {
            std::memcpy(data, ptr_ + offset, size);
        } else {
            // GPU buffer: cudaMemcpy DeviceToHost
            const CUDABufferType* cuda_buft = dynamic_cast<const CUDABufferType*>(buft_);
            int device = cuda_buft ? cuda_buft->device_id() : 0;
            cudaSetDevice(device);
            cudaMemcpy(data, ptr_ + offset, size, cudaMemcpyDeviceToHost);
        }
    }

    // 对标 ggml_backend_buffer_copy_tensor: GPU→GPU 同 device 直接拷贝
    bool cpy_tensor(const TensorF32* src, TensorF32* dst,
                    size_t src_offset, size_t dst_offset, size_t size) override {
        // 只处理 GPU buffer
        if (is_host()) return false;

        const CUDABufferType* cuda_buft = dynamic_cast<const CUDABufferType*>(buft_);
        if (!cuda_buft) return false;

        // 检查 src 是否也在 GPU buffer 上
        if (!src->buffer_ || src->buffer_->is_host()) return false;

        const CUDABufferType* src_cuda = dynamic_cast<const CUDABufferType*>(src->buffer_->type());
        if (!src_cuda || src_cuda->device_id() != cuda_buft->device_id()) return false;

        // GPU→GPU 同 device: cudaMemcpyDeviceToDevice
        cudaSetDevice(cuda_buft->device_id());
        cudaMemcpy(ptr_ + dst_offset,
                   static_cast<const uint8_t*>(src->buffer_->data()) + src_offset,
                   size, cudaMemcpyDeviceToDevice);
        return true;
    }

private:
    uint8_t* ptr_     = nullptr;
    bool     own_ptr_ = true;
};

// ============================================================
// MultiBuffer — 将多个 buffer 包装为一个（对标 ggml_backend_multi_buffer）
// ============================================================
class MultiBuffer : public Buffer {
public:
    explicit MultiBuffer(std::vector<Buffer*>&& buffers)
        : Buffer(nullptr, 0, BufferUsage::COMPUTE) {
        buffers_ = std::move(buffers);
        for (auto* b : buffers_) {
            size_ += b->size();
        }
    }

    ~MultiBuffer() override {
        for (auto* b : buffers_) delete b;
    }

    void* data() override { return buffers_.empty() ? nullptr : buffers_[0]->data(); }

    // 对标 ggml_backend_multi_buffer_clear
    void clear(uint8_t value) override {
        for (auto* b : buffers_) {
            b->clear(value);
        }
    }

    // 对标 ggml_backend_multi_buffer_set_usage
    void set_usage(BufferUsage usage) {
        usage_ = usage;
        for (auto* b : buffers_) {
            b->set_usage(usage);
        }
    }

    Buffer* get_buffer(int i) { return (i >= 0 && i < (int)buffers_.size()) ? buffers_[i] : nullptr; }
    int     n_buffers() const { return (int)buffers_.size(); }

private:
    std::vector<Buffer*> buffers_;
};

// ============================================================
// TensorAllocator — 对标 ggml_tallocr
// 在单个 buffer 内按顺序切分空间给各个 tensor
// ============================================================
class TensorAllocator {
public:
    explicit TensorAllocator(Buffer* buffer)
        : buffer_(buffer)
        , base_(static_cast<uint8_t*>(buffer->data()))
        , offset_(0)
        , alignment_(buffer->type() ? buffer->type()->get_alignment() : 32) {}

    // 从 buffer 中为 tensor 分配空间，返回是否成功
    bool alloc(TensorF32* tensor) {
        size_t alloc_size = buffer_->type()
            ? buffer_->type()->get_alloc_size(tensor)
            : tensor->nbytes();

        // 对齐
        size_t aligned_size   = GGML_PAD(alloc_size, alignment_);
        size_t aligned_offset = GGML_PAD(offset_, alignment_);

        if (aligned_offset + aligned_size > buffer_->size()) {
            return false;  // buffer 空间不足
        }

        tensor->data_        = reinterpret_cast<float*>(base_ + aligned_offset);
        tensor->buffer_      = buffer_;
        tensor->buffer_offs_ = aligned_offset;
        tensor->own_data_    = false;  // buffer 管理生命周期

        offset_ = aligned_offset + aligned_size;
        return true;
    }

    size_t offset() const { return offset_; }

private:
    Buffer*  buffer_;
    uint8_t* base_;
    size_t   offset_;
    size_t   alignment_;
};

// ============================================================
// 全局分配函数声明
// ============================================================

// 对标 ggml_backend_buft_alloc_buffer
Buffer* alloc_buffer(BufferType* buft, size_t size, BufferUsage usage = BufferUsage::COMPUTE);

// 对标 ggml_backend_alloc_ctx_tensors_from_buft
// 将 graph 中所有 tensor 分配到 buft 类型的 buffer 中
// no_alloc = true  时只计算 nbytes_total，不实际分配（阶段一）
// no_alloc = false 时执行实际分配（阶段二）
// 返回分配好的 buffer（可能是 DefaultBuffer 或 MultiBuffer），调用方负责释放
Buffer* alloc_ctx_tensors_from_buft(
    ComputeGraph* graph,
    BufferType*   buft,
    size_t*       nbytes_total,
    bool          no_alloc);

// 对标 ggml_backend_multi_buffer_alloc_buffer
// 将已有的多个 buffer 包装为一个 MultiBuffer
Buffer* alloc_multi_buffer(std::vector<Buffer*>& buffers);

// 对标 ggml_backend_buffer_is_multi_buffer
bool is_multi_buffer(const Buffer* buffer);

// 对标 ggml_backend_tensor_copy
// 将 src tensor 的数据拷贝到 dst tensor（跨后端拷贝，自动处理 CPU↔GPU）
bool backend_tensor_copy(const TensorF32* src, TensorF32* dst);

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
    int tensor_backend_id(TensorF32* t) const;
    int tensor_backend_id(TensorF32* t, int default_id) const;

    // 执行所有 split（含跨后端拷贝）
    Status graph_compute();

private:
    // ===== 三趟扫描 =====
    void pass_assign_leafs(ComputeGraph * graph);        // 第一趟：叶子节点分配
    void pass_expand_assignments(ComputeGraph * graph);   // 第二趟：扩展分配
    void pass_fill_unassigned(ComputeGraph * graph);      // 第三趟：填充未分配节点

    // ===== Pass 5: 切分 + 构建子图 =====
    void build_splits(ComputeGraph * graph);

    // helper
    bool is_view_op(int op) const;
    void set_backend_if_supported(TensorF32* node, int backend_id);
    int  count_supported_inputs(TensorF32* node, int backend_id) const;
    bool tensor_buffer_compatible(const TensorF32* src, int backend_id) const;

    // alloc_splits invoke this function to allocate the memory
    bool reserve_graph_memory();

    static constexpr int MAX_SPLITS = 64;
    static constexpr int MAX_SPLIT_INPUTS = 32;

    // ===== SplitInfo =====
    struct SplitInfo {
        int        backend_id = 0;
        int        i_start    = 0;
        int        i_end      = 0;
        int        n_inputs   = 0;
        TensorF32* inputs[MAX_SPLIT_INPUTS];
    };

    // ===== 数据成员 =====
    std::vector<Backend *> backends_;          // 按优先级排序的后端列表
    int n_backends_ = 0;

    // tensor → backend_id 映射
    using BackendMap = std::unordered_map<const TensorF32*, int>;
    BackendMap backend_map_;

    ComputeGraph* current_graph_ = nullptr;
    
    std::vector<int> node_backend_id_;
    std::vector<int> prev_node_backend_id_;
    std::vector<int> leaf_backend_id_;
    std::vector<int> prev_leaf_backend_id_;
    
    std::vector<const BufferType*> bufts_;

    bool graph_reserved_ = false;

    // 持有 allocated buffer 的生命周期（避免 dangling pointer）
    std::vector<std::unique_ptr<Buffer>> reserved_buffers_;

    // 分裂结果
    int n_splits_ = 0;
    SplitInfo splits_[MAX_SPLITS];

    // 图输入收集
    int n_graph_inputs_ = 0;
    TensorF32* graph_inputs_[256];

    // tensor 拷贝映射：(src_tensor, target_backend_id) → copied_tensor
    using CopyKey = std::pair<const TensorF32*, int>;
    struct CopyKeyHash {
        size_t operator()(const CopyKey& k) const {
            return std::hash<const TensorF32*>()(k.first) ^ (std::hash<int>()(k.second) << 1);
        }
    };
    std::unordered_map<CopyKey, TensorF32*, CopyKeyHash> copy_tensor_map_;

    // context
    RFAAContext * ctx_ = nullptr;
};

// ============================================================
// CPUBufferType — CPU 端 buffer 分配器
// ============================================================
class CPUBufferType : public BufferType {
public:
    const char* get_name() const override { return "CPU"; }
    void* alloc(size_t size) override;
    void free(void* ptr) override;
    size_t get_alignment() const override { return 32; }
    bool is_host() const override { return true; }

    static CPUBufferType* instance();
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
    const BufferType* buffer_type() const override;
    bool supports_buffer_type(const BufferType* buft) const override;
    
    const char * get_name() const override { return "CPU"; }
    Status       graph_compute(ComputeGraph * cgraph) override;

    void         synchronize() override {} 
    // it is NULL for CPU backend, because CPU backend is synchronous

    // ===== 公开（图规划，可以被外部调用预估算资源）=====
    ComputePlan  graph_plan(ComputeGraph * cgraph) const;

private:
    // ===== 图计算线程（静态，无 this 依赖）=====
    static void compute_thread(ThreadState * state);

    // ===== op 分发 =====
    static Status  dispatch_node(TensorF32 * node, ComputeParams * p);

    static int get_n_tasks(TensorF32 * node, int n_threads);
    static size_t estimate_work_size(TensorF32 * node, int n_threads, int n_tasks = -1);
    static constexpr int64_t align_up(int64_t n, int64_t align) {
        return (n + align - 1) / align * align;
    }

    // ===== kernels (static: 无需 this, 仅操作张量数据) =====
    static void kernel_elemwise(TensorF32 * node, ComputeParams * p);
    static void kernel_mul_mat (TensorF32 * node, ComputeParams * p);
    static void kernel_out_prod(TensorF32 * node, ComputeParams * p);
    static void kernel_tri_mul (TensorF32 * node, ComputeParams * p);
    static void kernel_tri_mul_back(TensorF32 * node, ComputeParams * p);
    static void kernel_outer_prod_mean(TensorF32 * node, ComputeParams * p);
    static void kernel_outer_prod_mean_back(TensorF32 * node, ComputeParams * p);
    static void kernel_outer_prod(TensorF32 * node, ComputeParams * p);
    static void kernel_outer_prod_back(TensorF32 * node, ComputeParams * p);
    static void kernel_softmax (TensorF32 * node, ComputeParams * p);
    static void kernel_softmax_back(TensorF32 * node, ComputeParams * p);
    static void kernel_rms_norm (TensorF32 * node, ComputeParams * p);
    static void kernel_norm     (TensorF32 * node, ComputeParams * p);
    static void kernel_norm_back(TensorF32 * node, ComputeParams * p);
    static void kernel_silu    (TensorF32 * node);
    static void kernel_gelu    (TensorF32 * node);
    static void kernel_relu    (TensorF32 * node);
    static void kernel_dup     (TensorF32 * node);
    static void kernel_scale   (TensorF32 * node, ComputeParams * p);
    static void kernel_add1    (TensorF32 * node, ComputeParams * p);
    static void kernel_sum     (TensorF32 * node, ComputeParams * p);
    static void kernel_sum_rows(TensorF32 * node, ComputeParams * p);
    static void kernel_mean    (TensorF32 * node, ComputeParams * p);
    static void kernel_repeat  (TensorF32 * node, ComputeParams * p);
    static void kernel_repeat_back(TensorF32 * node, ComputeParams * p);
    static void kernel_sqr     (TensorF32 * node, ComputeParams * p);
    static void kernel_concat  (TensorF32 * node, ComputeParams * p);

    // ===== shape op（方案 B：Tensor 无 nb，均需实际拷贝）=====
    // kernel_cpy:    整块 memcpy src→dst（reshape/view/cont，元素顺序不变）
    static void kernel_cpy     (TensorF32 * node, ComputeParams * p);
    // kernel_permute: 按 op_params 的 dims 映射重排数据（permute/transpose）
    static void kernel_permute (TensorF32 * node, ComputeParams * p);

    // ===== set_rows (scatter 写入指定行) =====
    static void kernel_set_rows     (TensorF32 * node, ComputeParams * p);

    // ===== get_rows / get_rows_back (embedding 查表) =====
    static void kernel_get_rows     (TensorF32 * node, ComputeParams * p);
    static void kernel_get_rows_back(TensorF32 * node, ComputeParams * p);

    // ===== SE3 消息传递三件套（方案 B）=====
    static void kernel_edge_gather_rows(TensorF32 * node, ComputeParams * p);
    static void kernel_per_edge_matmul (TensorF32 * node, ComputeParams * p);
    static void kernel_scatter_add     (TensorF32 * node, ComputeParams * p);
    static void kernel_per_edge_matmul_back_kernel   (TensorF32 * node, ComputeParams * p);
    static void kernel_per_edge_matmul_back_gathered (TensorF32 * node, ComputeParams * p);

    static void kernel_sigmoid (TensorF32 * node, ComputeParams * p);

    // ===== 数据成员 =====
    int          n_threads_;
    ThreadPool * threadpool_ = nullptr;
    uint8_t    * work_data_  = nullptr;
    size_t       work_size_  = 0;
};

// ==================== CUDABackend ====================

// ============================================================
// CUDABufferType — GPU 端 buffer 分配器
// ============================================================

/* class CUDABufferType : public BufferType {
public:
    explicit CUDABufferType(int device_id = 0) : device_id_(device_id) {}

    const char* get_name() const override { return "CUDA"; }
    void* alloc(size_t size) override;
    void free(void* ptr) override;
    size_t get_alignment() const override { return 128; }  // CUDA 对齐 128 bytes
    bool is_host() const override { return false; }

    int device_id() const { return device_id_; }

    static CUDABufferType* instance(int device_id = 0);

private:
    int device_id_ = 0;
}; */

class CUDABackend : public Backend {
public:
    CUDABackend(int device_id = 0);
    ~CUDABackend() override;

    const char * get_name() const override { return "CUDA"; }
    Status       graph_compute(ComputeGraph * cgraph) override;
    void         synchronize() override;

    bool supports_op(TensorF32 * node) const override;
    const BufferType* buffer_type() const override;
    bool supports_buffer_type(const BufferType* buft) const override;

    // GPU 优先级高于 CPU
    // set it 0 when we test cpu
    int priority() const override { return 1; }

private:
    int device_id_ = 0;

    // CUDA dispatch
    static Status dispatch_node(TensorF32 * node, ComputeParams * p);

    // CUDA kernels
    static void kernel_elemwise_add_cuda(TensorF32 * node, ComputeParams * p);
    static void kernel_elemwise_sub_cuda(TensorF32 * node, ComputeParams * p);
    static void kernel_elemwise_mul_cuda(TensorF32 * node, ComputeParams * p);
    static void kernel_elemwise_div_cuda(TensorF32 * node, ComputeParams * p);

    static void kernel_mul_mat_cuda (TensorF32 * node, ComputeParams * p);
    static void kernel_out_prod_cuda(TensorF32 * node, ComputeParams * p);
    static void kernel_softmax_cuda (TensorF32 * node, ComputeParams * p);
    static void kernel_softmax_back_cuda(TensorF32 * node, ComputeParams * p);
    static void kernel_norm_cuda     (TensorF32 * node, ComputeParams * p);
    static void kernel_norm_back_cuda(TensorF32 * node, ComputeParams * p);
    static void kernel_dup_cuda      (TensorF32 * node);
    static void kernel_scale_cuda    (TensorF32 * node, ComputeParams * p);
    static void kernel_add1_cuda     (TensorF32 * node, ComputeParams * p);
    static void kernel_sum_cuda      (TensorF32 * node, ComputeParams * p);
    static void kernel_mean_cuda     (TensorF32 * node, ComputeParams * p);
    static void kernel_concat_cuda   (TensorF32 * node, ComputeParams * p);

    // SE3 消息传递三件套（方案 B）+ per_edge_matmul 反向核
    static void kernel_edge_gather_rows_cuda (TensorF32 * node, ComputeParams * p);
    static void kernel_per_edge_matmul_cuda  (TensorF32 * node, ComputeParams * p);
    static void kernel_scatter_add_cuda      (TensorF32 * node, ComputeParams * p);
    static void kernel_per_edge_matmul_back_kernel_cuda   (TensorF32 * node, ComputeParams * p);
    static void kernel_per_edge_matmul_back_gathered_cuda (TensorF32 * node, ComputeParams * p);

    // CUDA unary kernels
    static void kernel_relu_cuda   (TensorF32 * node);
    static void kernel_gelu_cuda   (TensorF32 * node);
    static void kernel_sigmoid_cuda(TensorF32 * node);
    static void kernel_silu_cuda   (TensorF32 * node);
    static void kernel_tanh_cuda   (TensorF32 * node);
    static void kernel_exp_cuda    (TensorF32 * node);
};

} // namespace rfaa

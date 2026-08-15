#pragma once

#include "Core.h"
#include <cstring>
#include <array>
#include <cuda_fp16.h>

#define GGML_MAX_OP_PARAMS      64
#define GGML_MAX_SRC            10

namespace ppml {

// 前向声明 — 避免循环依赖
class Buffer;
class TensorAllocator;

enum tensor_flag {
    TENSOR_FLAG_INPUT   = 1,
    TENSOR_FLAG_OUTPUT  = 2,
    TENSOR_FLAG_PARAM   = 4,
    TENSOR_FLAG_LOSS    = 8,
    TENSOR_FLAG_COMPUTE = 16,
    TENSOR_FLAG_NO_WEIGHT_DECAY = 32,   // 权重衰减豁免（bias / LayerNorm 的 gamma、beta）
};

enum tensor_type {
    TENSOR_TYPE_F32  = 0,
    TENSOR_TYPE_F16  = 1,
    TENSOR_TYPE_Q4_0 = 2,
    TENSOR_TYPE_Q4_1 = 3,
    // TENSOR_TYPE_Q4_2 = 4, support has been removed
    // TENSOR_TYPE_Q4_3 (5) support has been removed
    TENSOR_TYPE_Q5_0 = 6,
    TENSOR_TYPE_Q5_1 = 7,
    TENSOR_TYPE_Q8_0 = 8,
    TENSOR_TYPE_Q8_1 = 9,
    // k-quantizations
    TENSOR_TYPE_Q2_K = 10,
    TENSOR_TYPE_Q3_K = 11,
    TENSOR_TYPE_Q4_K = 12,
    TENSOR_TYPE_Q5_K = 13,
    TENSOR_TYPE_Q6_K = 14,
    TENSOR_TYPE_Q8_K = 15,
    TENSOR_TYPE_I8,
    TENSOR_TYPE_I16,
    TENSOR_TYPE_I32,
    TENSOR_TYPE_COUNT,
};


enum tensor_op {
    OP_NONE = 0,
    OP_DUP,
    OP_ADD,
    OP_ADD_ID,
    OP_ADD1,
    OP_ACC,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_SQR,
    OP_SQRT,
    OP_LOG,
    OP_SIN,
    OP_COS,
    OP_SUM,
    OP_SUM_ROWS,
    OP_CUMSUM,
    OP_MEAN,
    OP_ARGMAX,
    OP_COUNT_EQUAL,
    OP_REPEAT,
    OP_REPEAT_BACK,
    OP_CONCAT,
    OP_SILU_BACK,
    OP_NORM, // normalize
    OP_NORM_BACK,
    OP_RMS_NORM,
    OP_RMS_NORM_BACK,
    OP_GROUP_NORM,
    OP_L2_NORM,
    OP_MUL_MAT,
    OP_MUL_MAT_ID,
    OP_OUT_PROD,
    OP_SCALE,
    OP_SET,
    OP_CPY,
    OP_CONT,
    OP_RESHAPE,
    OP_VIEW,
    OP_PERMUTE,
    OP_TRANSPOSE,
    OP_GET_ROWS,
    OP_GET_ROWS_BACK,
    OP_SET_ROWS,
    OP_DIAG,
    OP_DIAG_MASK_INF,
    OP_DIAG_MASK_ZERO,
    OP_SOFT_MAX,
    OP_SOFT_MAX_BACK,
    OP_ROPE,
    OP_ROPE_BACK,
    OP_CLAMP,
    OP_CONV_TRANSPOSE_1D,
    OP_IM2COL,
    OP_IM2COL_BACK,
    OP_IM2COL_3D,
    OP_COL2IM_1D,
    OP_CONV_2D,
    OP_CONV_3D,
    OP_CONV_2D_DW,
    OP_CONV_TRANSPOSE_2D,
    OP_POOL_1D,
    OP_POOL_2D,
    OP_POOL_2D_BACK,
    OP_UPSCALE,
    OP_PAD,
    OP_PAD_REFLECT_1D,
    OP_ROLL,
    OP_ARANGE,
    OP_TIMESTEP_EMBEDDING,
    OP_ARGSORT,
    OP_TOP_K,
    OP_LEAKY_RELU,
    OP_TRI,
    OP_FILL,
    OP_FLASH_ATTN_EXT,
    OP_FLASH_ATTN_BACK,
    OP_SSM_CONV,
    OP_SSM_SCAN,
    OP_WIN_PART,
    OP_WIN_UNPART,
    OP_GET_REL_POS,
    OP_ADD_REL_POS,
    OP_RWKV_WKV6,
    OP_GATED_LINEAR_ATTN,
    OP_RWKV_WKV7,
    OP_SOLVE_TRI,
    OP_GATED_DELTA_NET,
    OP_UNARY,
    OP_MAP_CUSTOM1,
    OP_MAP_CUSTOM2,
    OP_MAP_CUSTOM3,
    OP_CUSTOM,
    OP_CROSS_ENTROPY_LOSS,
    OP_CROSS_ENTROPY_LOSS_BACK,
    OP_OPT_STEP_ADAMW,
    OP_OPT_STEP_SGD,
    OP_GLU,
    OP_FAPE,
    OP_FAPE_BACK,
    OP_TRI_MUL,
    OP_TRI_MUL_BACK,
    // msa2pair outer-product-mean: einsum('bikd,bjkd->bijd', left, right/N)
    //   left [D,L,N,B], right [D,L,N,B] → dst [D,L,L,B]，收缩 seq 维 N（dims[2]）
    //   反向由 compute_backward 组合现有图 op 完成（见 ComputeGraph.cpp）。
    OP_OUTER_PROD_MEAN,
    // outer_product_mean 的反向：dL/dleft = grad⊗rightᵀ / N, dL/dright = gradᵀ⊗left / N
    OP_OUTER_PROD_MEAN_BACK,
    // gate 的 outer product（纯外积，无收缩）：left [D,L,B] × right [D,L,B] → [D*D,L,L,B]
    //   gate[(d1*D+d2), i, j, b] = left[d1,i,b] * right[d2,j,b]（特征维笛卡尔积 D×D→D*D）
    // 反向由 compute_backward 组合现有图 op 完成（见 ComputeGraph.cpp）。
    OP_OUTER_PROD,
    // outer_product 的反向
    OP_OUTER_PROD_BACK,
    // SE3 消息传递三件套（方案 B）
    // 按 edge_index 从节点特征取源节点行（gather）
    OP_EDGE_GATHER_ROWS,
    // 逐边矩阵乘: kernel[E,M,K] @ gathered[E,K,1] → (E,M)（消息生成）
    OP_PER_EDGE_MATMUL,
    // 按 edge_index 把边消息散点累加到目标节点（scatter-add）
    OP_SCATTER_ADD,
    // per_edge_matmul 反向（梯度 wrt kernel 与 gathered）
    OP_PER_EDGE_MATMUL_BACK_KERNEL,     // grad(E,M)⊗gathered(E,K) → dkernel(E,M,K)
    OP_PER_EDGE_MATMUL_BACK_GATHERED,   // kernel(E,M,K)ᵀ@grad(E,M) → dgathered(E,K)
    OP_COUNT,
};


// 张量接口：CPU/CUDA 统一抽象
template<typename T = float>
class Tensor {
public:
    // 构造/析构
    Tensor() = default;
    Tensor(const Shape& shape, Device device = Device::CPU);
    Tensor(const Shape& shape, T* data, Device device = Device::CPU, bool own = false);
    ~Tensor();
    
    // 禁止拷贝，允许移动
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // 基础属性
    const Shape& shape() const { return shape_; }
    Device device() const { return device_; }
    DType dtype() const;
    // DType deprecate?

    int64_t numel() const { return shape_.numel(); }
    size_t nbytes() const { return numel() * sizeof(T); }
    
    // 数据访问
    T* data() { return data_; }
    const T* data() const { return data_; }
    T* data(Device target);  // 获取指定设备的数据（自动拷贝）
    
    // 设备转移
    Tensor to(Device device) const;
    Tensor cpu() const { return to(Device::CPU); }
    Tensor cuda() const { return to(Device::CUDA); }
    
    // 索引访问 (仅 CPU)
    T& operator()(std::initializer_list<int64_t> indices);
    T operator()(std::initializer_list<int64_t> indices) const;
    
    // 视图操作
    Tensor view(const Shape& new_shape) const;
    Tensor slice(int dim, int64_t start, int64_t end) const;
    Tensor select(int dim, int64_t index) const;
    
    // 维度操作
    Tensor unsqueeze(int dim) const;  // 在指定维度插入大小为1的维度
    Tensor permute(const std::vector<int>& dims) const; // 维度重排
    
    // 内存管理
    void zero_();
    void copy_from(const Tensor& other);
    
    // 打印调试用
    std::string to_string() const;

    bool is_contiguous() const {
        // for now the data is contiguous
        return own_data_;
    }

    bool is_scalar() const { 
        for (int i = 0; i < shape_.ndim(); i++) {
            if (shape_.dims[i] != 1) {
                return false;
            }
        }
        return true;
    }

    // compare shape
    bool same_shape(const Tensor& other) const {
        return shape_.dims == other.shape_.dims;
    }

    bool can_repeat(const Tensor& other) const {
        if (shape_.ndim() != other.shape_.ndim()) return false;
        // check any dim is zero of two tensors
        if (shape_.numel() == 0 || other.shape_.numel() == 0) return false;

        for (int i = 0; i < shape_.ndim(); i++) {
            // divisible check
            if (shape_.dims[i] % other.shape_.dims[i] != 0) {
                return false;
            }
        }
        return true;
    }

    // ==== 新增：从 Context 初始化（不自己分配内存）====
    void init_from_context(int n_dims, const int64_t* ne, void* data_ptr) {
        Shape s;
        // need to modify?
        // no need to change: ne is the number of elements
        for (int i = 0; i < n_dims; i++) s.dims.push_back(ne[i]);
        shape_ = s;
        data_ = static_cast<T*>(data_ptr);
        device_ = Device::CPU;
        own_data_ = false;  // Context 管理生命周期
        src.fill(nullptr);  // 显式清空 src，避免残留脏指针
    }

    //tensor flag
    int32_t flag;

    enum tensor_op op;

    enum tensor_type type;

    // op_params: 存储 op 特定参数 (如 UNARY 的 subtype, RMS_NORM 的 eps 等)
    int32_t op_params[GGML_MAX_OP_PARAMS] = {0};

    //src GGML_MAX_SRC (固定大小数组，语义与 ggml 一致：src[i]=a，i<GGML_MAX_SRC，空位为 nullptr)
    std::array<Tensor*, GGML_MAX_SRC> src{};  // 默认初始化为全 nullptr

    // ==== buffer 分配相关 ====
    Buffer*     buffer_      = nullptr;  // 所属的 backend buffer
    Tensor<T>*  view_src     = nullptr;  // view tensor 的源 tensor
    size_t      buffer_offs_ = 0;        // 在 buffer 中的偏移量

    
private:
    Shape shape_;
    T* data_ = nullptr;
    Device device_ = Device::CPU;
    bool own_data_ = true;
    
    void allocate();
    void deallocate();

    friend class TensorAllocator;  // 允许 TensorAllocator 直接设置 data_/buffer_/own_data_
    //friend class BackendManager;   // 允许 BackendManager 设置 view tensor 的 data_
};

// 特化常用类型
using TensorF32 = Tensor<float>;
using TensorF16 = Tensor<half>;

using TensorI64 = Tensor<int64_t>;

// 工厂函数
template<typename T = float>
Tensor<T> zeros(const Shape& shape, Device device = Device::CPU) {
    Tensor<T> t(shape, device);
    t.zero_();
    return t;
}

template<typename T = float>
Tensor<T> from_numpy(T* data, const Shape& shape) {
    return Tensor<T>(shape, data, Device::CPU, false);
}

} // namespace ppml

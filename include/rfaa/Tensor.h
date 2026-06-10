#pragma once

#include "Core.h"
#include <cstring>
#include <cuda_fp16.h>

namespace rfaa {

enum tensor_flag {
    TENSOR_FLAG_INPUT   = 1,
    TENSOR_FLAG_OUTPUT  = 2,
    TENSOR_FLAG_PARAM   = 4,
    TENSOR_FLAG_LOSS    = 8,
    TENSOR_FLAG_COMPUTE = 16,
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

    // ==== 新增：从 Context 初始化（不自己分配内存）====
    void init_from_context(int n_dims, const int64_t* ne, void* data_ptr) {
        Shape s;
        for (int i = 0; i < n_dims; i++) s.dims.push_back(ne[i]);
        shape_ = s;
        data_ = static_cast<T*>(data_ptr);
        device_ = Device::CPU;
        own_data_ = false;  // Context 管理生命周期
    }

    //tensor flag
    int32_t flag;

    enum tensor_op op;

    
private:
    Shape shape_;
    T* data_ = nullptr;
    Device device_ = Device::CPU;
    bool own_data_ = true;
    
    void allocate();
    void deallocate();
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

} // namespace rfaa

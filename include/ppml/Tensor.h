#pragma once

#include "Core.h"
#include <cstring>
#include <array>
#include <vector>
#include <cuda_fp16.h>
#include "ppml/QuantBlocks.h"   // int8 分块量化块结构（Q8_0/Q8_1）

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
    TENSOR_FLAG_CONST   = 64,           // 常量叶子（data 存于 const_data_，由 Gallocr 分配后填充）
    // 注：TENSOR_FLAG_CONST 置位 = 静态可复用常量（Gallocr 填充后保留 const_data_，图可复用）；
    //     不置位 = 动态一次性常量（如 dropout 随机掩码，Gallocr 填充后清空+shrink const_data_）。
    TENSOR_FLAG_SE3     = 128,          // SE3 等变模块参数：梯度尺度与主图不匹配（offset 直接连坐标，
    //     FAPE 梯度经 coords→offset→SE3 权重放大），AdamW 对其用分层小 lr（se3_lr_scale，默认 0.1）。
    TENSOR_FLAG_LORA    = 256,          // LoRA 低秩旁路参数（A/B）：冻结主权重，仅训旁路。
    //     AdamW 对其用放大 lr（lora_lr_scale，默认 10.0），微调通常比预训练 lr 高 1~2 个量级。
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

// ============================================================================
// int8 分块量化：类型谓词与块尺寸换算（2026-09-10 准备，非 kernel 支持代码）
//   量化张量的数据按「块」存放（Q8_0: 34B/32元素），因此：
//     - 必须用 Tensor::nbytes()（按块计量）/ nblocks()，**不要**用 type_size_bytes()（每元素）
//     - numel 必须是 quant_block_elems(type) 的整数倍
// ============================================================================
inline bool is_quantized_type(tensor_type t) {
    switch (t) {
        case TENSOR_TYPE_Q4_0: case TENSOR_TYPE_Q4_1:
        case TENSOR_TYPE_Q5_0: case TENSOR_TYPE_Q5_1:
        case TENSOR_TYPE_Q8_0: case TENSOR_TYPE_Q8_1:
        case TENSOR_TYPE_Q2_K: case TENSOR_TYPE_Q3_K: case TENSOR_TYPE_Q4_K:
        case TENSOR_TYPE_Q5_K: case TENSOR_TYPE_Q6_K: case TENSOR_TYPE_Q8_K:
            return true;
        default:
            return false;
    }
}

// 已实现支持的量化类型（准备阶段：仅 Q8_0/Q8_1；其余枚举存在但无块结构/kernel）
inline bool is_quantized_type_supported(tensor_type t) {
    return t == TENSOR_TYPE_Q8_0 || t == TENSOR_TYPE_Q8_1;
}

// 每块元素数（Q8 家族 = 32；不支持的量化类型返回 0）
inline int quant_block_elems(tensor_type t) {
    switch (t) {
        case TENSOR_TYPE_Q8_0: return QK8_0;
        case TENSOR_TYPE_Q8_1: return QK8_1;
        default:               return 0;
    }
}

// 每块字节数（含 scale/中间量；不支持的量化类型返回 0）
inline size_t quant_block_bytes(tensor_type t) {
    switch (t) {
        case TENSOR_TYPE_Q8_0: return sizeof(block_q8_0);
        case TENSOR_TYPE_Q8_1: return sizeof(block_q8_1);
        default:               return 0;
    }
}

// 每元素平均字节数（含块内元数据，用于内存估算：Q8_0 = 34/32 ≈ 1.0625 B/元素）
inline double quant_bytes_per_elem(tensor_type t) {
    const int qk = quant_block_elems(t);
    return qk > 0 ? (double)quant_block_bytes(t) / (double)qk : 0.0;
}

// 权重量化类型 → 对应的「激活/vec_dot」类型（ggml 约定：src1->type == vec_dot_type(src0->type)）
//   Q8_0 权重 → 激活量化成 Q8_1（带 sum，供后续带 zero-point 组合修正）
inline tensor_type quant_vec_dot_type(tensor_type w) {
    switch (w) {
        case TENSOR_TYPE_Q8_0: return TENSOR_TYPE_Q8_1;
        default:               return TENSOR_TYPE_F32;
    }
}

// 量化 mul_mat 组合是否受支持（权重 w × 激活 a）：
//   - 权重：Q8_0（后续可扩 Q4_0 等）
//   - 激活/右操作数：F32、F16 或 Q8_1/Q8_0（对称组合）
inline bool quant_mul_mat_combo_supported(tensor_type w, tensor_type a) {
    if (!is_quantized_type_supported(w)) return false;
    return a == TENSOR_TYPE_F32 || a == TENSOR_TYPE_F16 || is_quantized_type_supported(a);
}


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
    // concat 反向：把 grad 沿拼接维切回各 src（concat 的梯度回传）
    OP_CONCAT_BACK,
    // 全局最大归约：max_all(x) → 标量。用于 softmax 数值稳定（max 减稳，避免 exp 溢出）
    OP_MAX_ALL,
    // relu 反向：relu_back(grad, x) → grad * (x>0)（解决 RadialFunc 里 relu 无反向断链）
    OP_RELU_BACK,
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
    // 每元素字节数按 type 计算（前期 fp16 支持，2026-09-10）：
    //   图节点统一为 Tensor<float>*，但 type==TENSOR_TYPE_F16 时数据按 16 位 half 存放
    //   （2 字节/元素）；否则按模板 T 大小（F32=4）。若恒用 sizeof(T)，F16 节点会被
    //   Gallocr/cpy 按 4 字节分配与搬运 → 内存翻倍且跨后端拷贝错位。
    size_t type_size_bytes() const {
        return (type == TENSOR_TYPE_F16) ? 2 : sizeof(T);
    }
    // 是否量化（数据按块存放，须用 nbytes()/nblocks()，不能用 type_size_bytes()）
    bool is_quantized() const { return is_quantized_type(type); }
    // 量化形状合法性（2026-09-10）：
    //   ① 最内维（dims[0]）必须是块元素数的整数倍 —— 量化块**不允许跨行**；
    //   ② numel 也须是块元素数整数倍（① 成立时 ② 自动成立，保留作双保险）。
    //   ⚠️ 只查 numel 是不够的：如 {31,128} 的 numel=3968 是 32 的倍数，但 dims[0]=31
    //      会让行内块跨越行边界 → 块索引错位（所以 nbytes() 对非法形状返回 0）。
    bool quant_shape_valid() const {
        const int qk = quant_block_elems(type);
        if (qk <= 0) return false;
        if (shape_.ndim() < 1 || (shape_.dims[0] % qk) != 0) return false;
        return (numel() % qk) == 0;
    }
    // 块数（量化类型且形状合法：numel/块元素数；否则 0）
    int64_t nblocks() const {
        if (!is_quantized_type(type) || !quant_shape_valid()) return 0;
        return numel() / quant_block_elems(type);
    }
    // 量化块指针访问（调用方负责确认 type 匹配、numel 为块元素数整数倍）
    template<typename B> B*       blocks()       { return reinterpret_cast<B*>(data_); }
    template<typename B> const B* blocks() const { return reinterpret_cast<const B*>(data_); }

    // 字节数：
    //   - 量化类型（2026-09-10 int8 准备）：按块计量 = nblocks × 块字节数（Q8_0: 34B/32元素）。
    //     形状非法（numel 非块元素数整数倍 / 类型无块结构）时返回 0，由调用处校验后拒绝。
    //   - 非量化：numel × 每元素字节数（F16=2，其余 sizeof(T)）。
    size_t nbytes() const {
        if (is_quantized_type(type)) {
            if (!quant_shape_valid()) return 0;   // 形状非法（最内维非块元素数整数倍等）→ 0，由调用处拒绝
            return static_cast<size_t>(numel() / quant_block_elems(type)) * quant_block_bytes(type);
        }
        return numel() * type_size_bytes();
    }
    
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
        //    而 new_tensor 用 placement new 且 context 缓冲区可能是复用/非全零，
        //    导致 op 读到垃圾值（如 32653）→ dispatch_body 无 case → NOT_SUPPORTED。
        //    （本会话 SE3 测试 `node#92 op=32653` 即此根因：RadialFunc 的 BN 参数
        //     经 new TensorF32 后 op 未初始化。）
        op        = OP_NONE;
        flag      = 0;
        type      = TENSOR_TYPE_F32;
                                       //    否则 type 读到垃圾值（非 F32/F16）→ build_backward_expand 断言失败。
        for (int i = 0; i < GGML_MAX_OP_PARAMS; ++i) op_params[i] = 0;
    }

    // 重新绑定数据指针（no_alloc 空壳 → 构建期暂存区 / 后续 Gallocr backend buffer）
    void bind_data(void* p) { data_ = static_cast<T*>(p); }

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

    // ==== 常量叶子宿主数据 ====
    // 供常量叶子（constant_tensor 等）使用：数据存于此处，data() 保持 nullptr，
    // 使 Gallocr 判定其为 managed 并分配 backend buffer，分配后由 Gallocr 从
    // const_data_ 填充。TENSOR_FLAG_CONST 置位 → 保留（图可复用）；否则填充后清空释放。
    std::vector<float> const_data_;

    
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

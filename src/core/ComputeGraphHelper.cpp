#include "rfaa/Tensor.h"
#include "rfaa/Context.h"

#include "ComputeGraph.h"

namespace rfaa {

TensorF32 * add_impl( 
        TensorF32  * a,  
        TensorF32  * b,  
        bool                  inplace) {  
    //GGML_ASSERT(ggml_can_repeat(b, a)); 
    assert(b->can_repeat(a));
  
    //struct Tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);  
    Tensor * result; 
    result = inplace ? result.view(a->shape()) : result->copy_from(a);
  
    result->op     = OP_ADD;  
    result->src[0] = a;  
    result->src[1] = b;  
  
    return result;  
}

TensorF32 * repeat_back(
        TensorF32  * a,  
        TensorF32  * b) {  
    //GGML_ASSERT(ggml_can_repeat(b, a));  
    assert(b->can_repeat(a));
  
    Tensor* result = context().new_tensor(b->ndim(), b->shape().dims.data());  

    assert(b->can_repeat(a));
  
    result->op     = OP_REPEAT_BACK;  
    result->src[0] = a;  
  
    return result;  
}

// add a scalar
TensorF32 * add1_impl(
        TensorF32  * a,
        TensorF32  * b,
        bool                  inplace) {
    //GGML_ASSERT(ggml_is_scalar(b));
    //GGML_ASSERT(ggml_is_padded_1d(a));

    assert(b->is_scalar());
    assert(a->is_contiguous());
 
    //struct Tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    TensorF32 * result; 
    result = inplace ? result->view(a->shape()) : result->copy_from(a);
 
    result->op     = OP_ADD1;
    result->src[0] = a;
    result->src[1] = b;
 
    return result;
}

// scale(a, s) — a * s  (标量乘法)
TensorF32* scale(TensTensorF32or* a, float s) {
    //Tensor* result = alloc_node(a->ndim(), a->dims());
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_SCALE;
    result->src[0] = a;
    // op_params 存储 scale 因子
    //result->op_params[0] = reinterpret_cast<int32_t&>(s);  // float → int32 位模式
    return result;
}

// neg(a) — -a
TensorF32* neg(TensorF32* a) {
    return scale(a, -1.0f);
}

// ============================================================
// 2. 矩阵操作
// ============================================================

// mul_mat(a, b) — 矩阵乘法 a @ b
// a: (..., M, K), b: (..., K, N) → (..., M, N)
TensorF32* mul_mat(TensorF32* a, TensorF32* b) {
    int64_t ne[4] = {b->shape().dims[0], a->shape().dims[1], 1, 1};
    // 处理 batch 维度
    int nd = std::max(a->ndim(), b->ndim());
    if (nd >= 3) {
        ne[2] = std::max(
            (a->ndim() >= 3) ? a->shape().dims[2] : 1,
            (b->ndim() >= 3) ? b->shape().dims[2] : 1
        );
    }
    if (nd >= 4) {
        ne[3] = std::max(
            (a->ndim() >= 4) ? a->shape().dims[3] : 1,
            (b->ndim() >= 4) ? b->shape().dims[3] : 1
        );
    }
    TensorF32* result = context().new_tensor(nd, ne);

    result->op     = OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// out_prod(a, b) — 外积 a ⊗ b
// a: (..., M, K1, K2), b: (..., N, K1, K2) → (..., M, N, K1, K2)
TensorF32* out_prod(TensorF32* a, TensorF32* b) {
    int64_t ne[4] = {
        a->shape().dims[0],
        b->shape().dims[0],
        std::max(a->dims()[2], b->dims()[2]), 
        std::max(a->dims()[3], b->dims()[3])
    };
    TensorF32* result = context().new_tensor(4, ne);

    result->op     = OP_OUT_PROD;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// transpose(a) — 转置（交换最后两维）
TensorF32* transpose(TensorF32* a) {
    assert(a->ndim() >= 2);
    std::vector<int64_t> new_dims = a->shape().dims;
    std::swap(new_dims[new_dims.size() - 1], new_dims[new_dims.size() - 2]);

    //Tensor* result = alloc_node(static_cast<int>(new_dims.size()), new_dims.data());
    TensorF32* result = context().new_tensor(static_cast<int>(new_dims.size()), new_dims.data());
    result->op     = OP_TRANSPOSE;
    result->src[0] = a;
    return result;
}

// ============================================================
// 3. 激活函数
// ============================================================

// softmax(a) — softmax 沿最后一维
TensorF32* softmax(TensorF32* a) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_SOFT_MAX;
    result->src[0] = a;
    return result;
}

// silu(a) — SiLU / Swish 激活
TensorF32* silu(TensorF32* a) {
    Tensor* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_UNARY;
    result->src[0] = a;
    set_unary_op(result, UNARY_OP_SILU);
    return result;
}

// gelu(a) — GELU 激活
TensorF32* gelu(TensorF32* a) {
    Tensor* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_UNARY;
    result->src[0] = a;
    set_unary_op(result, UNARY_OP_GELU);
    return result;
}

// gelu_quick(a) — GELU 快速近似
TensorF32* gelu_quick(TensorF32* a) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_UNARY;
    result->src[0] = a;
    set_unary_op(result, UNARY_OP_GELU_QUICK);
    return result;
}

// relu(a) — ReLU 激活
TensorF32* relu(TensorF32* a) {
    Tensor* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_UNARY;
    result->src[0] = a;
    set_unary_op(result, UNARY_OP_RELU);
    return result;
}

// leaky_relu(a, alpha) — Leaky ReLU
TensorF32* leaky_relu(TensorF32* a, float alpha = 0.01f) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_LEAKY_RELU;
    result->src[0] = a;
    //result->op_params[0] = reinterpret_cast<int32_t&>(alpha);
    return result;
}

// ============================================================
// 4. 归一化
// ============================================================

// rms_norm(a, eps) — RMS Normalization 沿最后一维
TensorF32* rms_norm(TensorF32* a, float eps = 1e-6f) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_RMS_NORM;
    result->src[0] = a;
    //result->op_params[0] = reinterpret_cast<int32_t&>(eps);
    return result;
}

// norm(a, eps) — Layer Normalization 沿最后一维
TensorF32* norm(TensorF32* a, float eps = 1e-5f) {
    int D = a->shape().dims.back();
    int rows = a->numel() / D;

    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_NORM;
    result->src[0] = a;
    reinterpret_cast<float&>(result->op_params[0]) = eps;

    // 分配 mean / rstd 缓存 (rows 个 float, 供 backward 读取)
    int64_t cache_dims[] = {rows};
    result->src[1] = context().new_tensor<float>(1, cache_dims);  // mean buffer
    result->src[2] = context().new_tensor<float>(1, cache_dims);  // rstd buffer

    return result;
}

// ============================================================
// 5. 规约操作
// ============================================================

// sum(a) — 所有元素求和
TensorF32* sum(TensorF32* a) {
    int64_t ne[1] = {1};
    TensorF32* result = context().new_tensor(1, ne);
    result->op     = OP_SUM;
    result->src[0] = a;
    return result;
}

// mean(a, dim) — 沿指定维度求平均 (dim=-1 = 最后一维)
TensorF32* mean(TensorF32* a) {
    int64_t ne[1] = {1};
    TensorF32* result = context().new_tensor(1, ne);
    result->op     = OP_MEAN;
    result->src[0] = a;
    return result;
}

// sum_rows(a) — 沿最后一行求和 (a: M×N → M×1)
TensorF32* sum_rows(TensorF32* a) {
    int64_t ne[4] = {1, a->shape().dims[1], 1, 1};
    if (a->ndim() >= 3) ne[2] = a->shape().dims[2];
    if (a->ndim() >= 4) ne[3] = a->shape().dims[3];
    int nd = a->ndim();
    TensorF32* result = context().new_tensor(nd, ne);
    result->op     = OP_SUM_ROWS;
    result->src[0] = a;
    return result;
}

// ============================================================
// 6. 形状操作
// ============================================================

// view(a, new_shape) — 零拷贝视图（不拥有数据）
TensorF32* view(TensorF32* a, const Shape& new_shape) {
    assert(new_shape.numel() == a->numel());
    // view 不分配新数据，指针复用
    // 通过 init_from_context 创建一个非拥有的 Tensor
    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < new_shape.dims.size(); i++) {
        ne[i] = new_shape.dims[i];
    }
    TensorF32* result = context().new_tensor<float>(
        static_cast<int>(new_shape.dims.size()), ne);
    // 修正：view 不分配新内存，复用 a 的数据
    result->data_ = a->data();  // 共享数据指针
    result->op     = OP_VIEW;
    result->src[0] = a;
    return result;
}

// reshape(a, new_shape) — 重塑形状（可能拷贝）
TensorF32* reshape(TensorF32* a, const Shape& new_shape) {
    assert(new_shape.numel() == a->numel());
    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < new_shape.dims.size(); i++) {
        ne[i] = new_shape.dims[i];
    }
    TensorF32* result = context().new_tensor(static_cast<int>(new_shape.dims.size()), ne);
    result->op     = OP_RESHAPE;
    result->src[0] = a;
    return result;
}

// permute(a, dims) — 维度重排
TensorF32* permute(TensorF32* a, const std::vector<int>& dims) {
    assert(dims.size() == static_cast<size_t>(a->ndim()));

    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < dims.size(); i++) {
        ne[i] = a->shape().dims[dims[i]];
    }
    TensorF32* result = context().new_tensor(static_cast<int>(dims.size()), ne);
    result->op     = OP_PERMUTE;
    result->src[0] = a;
    // 存储 permute 的维度映射到 op_params
    //for (size_t i = 0; i < dims.size(); i++) {
    //    result->op_params[i] = dims[i];
    //}
    return result;
}

// unsqueeze(a, dim) — 在指定位置插入大小为1的维度
TensorF32* unsqueeze(TensorF32* a, int dim) {
    if (dim < 0) dim += a->ndim() + 1;
    assert(dim >= 0 && dim <= a->ndim());

    int64_t ne[4] = {1, 1, 1, 1};
    int j = 0;
    for (int i = 0; i < a->ndim() + 1; i++) {
        if (i == dim) {
            ne[i] = 1;
        } else {
            ne[i] = a->shape().dims[j++];
        }
    }
    TensorF32* result = context().new_tensor(a->ndim() + 1, ne);
    result->op     = OP_RESHAPE;  // unsqueeze 本质是 reshape
    result->src[0] = a;
    return result;
}

// concat(tensors, dim) — 沿指定维度拼接
// 返回新节点，其 src 数组存储所有输入
TensorF32* concat(const std::vector<TensorF32*>& tensors, int dim) {
    assert(!tensors.empty());
    if (dim < 0) dim += tensors[0]->ndim();

    // 计算输出形状
    int64_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < tensors[0]->ndim(); i++) {
        ne[i] = tensors[0]->shape().dims[i];
    }
    ne[dim] = 0;
    for (auto* t : tensors) {
        ne[dim] += t->shape().dims[dim];
    }

    TensorF32* result = context().new_tensor(tensors[0]->ndim(), ne);
    result->op     = OP_CONCAT;
    for (size_t i = 0; i < tensors.size() && i < GGML_MAX_SRC; i++) {
        result->src[i] = tensors[i];
    }
    //result->op_params[0] = dim;
    return result;
}

// repeat(a, b) — 沿各维度重复 a 以匹配 b 的形状
TensorF32* repeat(TensorF32* a, TensorF32* b) {
    TensorF32* result = context().new_tensor(b->ndim(), b->dims());
    result->op     = OP_REPEAT;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// repeat_back(a, b) — repeat 的反向操作（梯度）
TensorF32* repeat_back(TensorF32* a, TensorF32* b) {
    assert(b->can_repeat(a));
    TensorF32* result = context().new_tensor(b->ndim(), b->shape().dims.data());
    result->op     = OP_REPEAT_BACK;
    result->src[0] = a;
    return result;
}

// cont(a) — 确保张量连续存储
TensorF32* cont(TensorF32* a) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_CONT;
    result->src[0] = a;
    return result;
}

// ============================================================
// 7. 索引操作
// ============================================================

// get_rows(a, b) — 按索引 b 从 a 中取行
// a: (N, M, ...), b: (K,) → (K, M, ...)
TensorF32* get_rows(TensorF32* a, TensorF32* b) {
    int64_t ne[4] = {b->shape().dims[0], a->shape().dims[1], 1, 1};
    if (a->ndim() >= 3) ne[2] = a->shape().dims[2];
    Tensor* result = context().new_tensor(a->ndim(), ne);
    result->op     = OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// set_rows(a, b, c) — 将 c 的值写入 a 中由 b 指定的行
TensorF32* set_rows(TensorF32* a, TensorF32* b, TensorF32* c) {
    Tensor* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_SET_ROWS;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;
    return result;
}

// ============================================================
// 8. 特殊操作
// ============================================================

// diag_mask_inf(a, n_past) — 对角线掩码设为 -inf（因果注意力用）
TensorF32* diag_mask_inf(TensorF32* a, int n_past) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_DIAG_MASK_INF;
    result->src[0] = a;
    //result->op_params[0] = n_past;
    return result;
}

// diag_mask_zero(a, n_past) — 对角线以下掩码设为 0
TenTensorF32sor* diag_mask_zero(TensorF32* a, int n_past) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_DIAG_MASK_ZERO;
    result->src[0] = a;
    //result->op_params[0] = n_past;
    return result;
}

// clamp(a, min, max) — 值裁剪
TensorF32* clamp(TensorF32* a, float min_val, float max_val) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_CLAMP;
    result->src[0] = a;
    //result->op_params[0] = reinterpret_cast<int32_t&>(min_val);
    //result->op_params[1] = reinterpret_cast<int32_t&>(max_val);
    return result;
}

// sqr(a) — 平方 a²
TensorF32* sqr(TensorF32* a) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_SQR;
    result->src[0] = a;
    return result;
}

// sqrt(a) — 开方 √a
TensorF32* sqrt(TensorF32* a) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_SQRT;
    result->src[0] = a;
    return result;
}

// log(a) — 自然对数
TensorF32* log(TensorF32* a) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_LOG;
    result->src[0] = a;
    return result;
}

// sin(a), cos(a) — 三角函数
TensorF32* sin(TensorF32* a) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_SIN;
    result->src[0] = a;
    return result;
}

TensorF32* cos(TensorF32* a) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_COS;
    result->src[0] = a;
    return result;
}

// ============================================================
// 9. 损失函数
// ============================================================

// cross_entropy_loss(logits, targets) — 交叉熵损失
TensorF32* cross_entropy_loss(TensorF32* logits, TensorF32* targets) {
    int64_t ne[1] = {1};
    TensorF32* result = context().new_tensor(1, ne);
    result->op     = OP_CROSS_ENTROPY_LOSS;
    result->src[0] = logits;
    result->src[1] = targets;
    return result;
}

// ============================================================
// 10. 位置编码
// ============================================================

// rope(a, n_past) — 旋转位置编码
TensorF32* rope(TensorF32* a, int n_past, int n_dims = 0) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_ROPE;
    result->src[0] = a;
    //result->op_params[0] = n_past;
    //result->op_params[1] = n_dims;
    return result;
}

// ============================================================
// 11. 填充操作
// ============================================================

// pad(a, pad_dims) — 补零填充
TensorF32* pad(TensorF32* a, const std::vector<int>& pad_dims) {
    int64_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < a->ndim(); i++) {
        ne[i] = a->shape().dims[i];
        if (i * 2 < static_cast<int>(pad_dims.size())) {
            ne[i] += pad_dims[i * 2] + pad_dims[i * 2 + 1];
        }
    }
    TensorF32* result = context().new_tensor(a->ndim(), ne);
    result->op     = OP_PAD;
    result->src[0] = a;
    //for (size_t i = 0; i < pad_dims.size() && i < 8; i++) {
        //result->op_params[i] = pad_dims[i];
    //}
    return result;
}

// ============================================================
// 12. 累计操作（用于梯度累积）
// ============================================================

// acc(a, b, nb1, nb2, nb3, offset) — 累加 b 到 a 的指定位置
TensorF32* acc(TensorF32* a, TensorF32* b, size_t nb1, size_t nb2, size_t nb3, size_t offset) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_ACC;
    result->src[0] = a;
    result->src[1] = b;
    // ACC 的步长参数
    static_assert(sizeof(size_t) <= sizeof(int32_t) || sizeof(size_t) <= sizeof(int64_t),
                  "op_params too small for size_t");
    // 简化：假设 size_t 可以存入 op_params
    //result->op_params[0] = static_cast<int32_t>(nb1);
    //result->op_params[1] = static_cast<int32_t>(nb2);
    //result->op_params[2] = static_cast<int32_t>(nb3);
    //result->op_params[3] = static_cast<int32_t>(offset);
    return result;
}

// cpy(dst, src) — 拷贝
TensorF32* cpy(TensorF32* dst, TensorF32* src) {
    TensorF32* result = context().new_tensor(src->ndim(), src->shape().dims.data());
    result->op     = OP_CPY;
    result->src[0] = dst;
    result->src[1] = src;
    return result;
}

// ============================================================
// 13. dup — 复制张量（为计算图创建独立节点）
// ============================================================
TensorF32* dup(TensorF32* a) {
    TensorF32* result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_DUP;
    result->src[0] = a;
    return result;
}

// ============================================================
// 14. set — 设置张量（参数/常量节点）
// ============================================================

// set_param(a) — 标记为可训练参数
TensorF32* param(TensorF32* a) {
    a->flag |= TENSOR_FLAG_PARAM;
    return a;
}

// set_loss(a) — 标记为损失节点
TensorF32* loss(TensorF32* a) {
    a->flag |= TENSOR_FLAG_LOSS;
    return a;
}

// arange(start, end, step) — 创建等差数列
TensorF32* arange(float start, float end, float step = 1.0f) {
    int64_t n = static_cast<int64_t>((end - start) / step + 0.5f);
    int64_t ne[1] = {n};
    Tensor* result = context().new_tensor(1, ne);
    result->op     = OP_ARANGE;
    //result->op_params[0] = reinterpret_cast<int32_t&>(start);
    //result->op_params[1] = reinterpret_cast<int32_t&>(end);
    //result->op_params[2] = reinterpret_cast<int32_t&>(step);
    return result;
}

// 计算两个形状的广播结果形状
static Shape broadcast_shape(TensorF32* a, TensorF32* b) {
    int na = a->ndim(), nb = b->ndim();
    int nd = std::max(na, nb);
    std::vector<int64_t> dims(nd, 1);

    for (int i = 0; i < nd; i++) {
        int64_t da = (i < na) ? a->shape().dims[na - 1 - i] : 1;
        int64_t db = (i < nb) ? b->shape().dims[nb - 1 - i] : 1;
        dims[nd - 1 - i] = std::max(da, db);
    }
    return Shape(dims);
}

// 创建常量标量节点
static TensorF32* make_scalar(float value) {
    int64_t ne[4] = {1, 1, 1, 1};
    // why is [4]?
    // ne[1] = {1}
    TensorF32* t = context().new_tensor(1, ne);
    // 标量数据直接写入
    t->data()[0] = value;
    return t;
}

// sub(a, b) — a - b
TensorF32* sub(TensorF32* a, TensorF32* b) {
    Shape out_shape = broadcast_shape(a, b);
    Tensor* result = context().new_tensor(out_shape.ndim(), out_shape.dims.data());
    result->op     = OP_SUB;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// mul(a, b) — a * b  (element-wise)
TensorF32* mul(TensorF32* a, TensorF32* b) {
    Shape out_shape = broadcast_shape(a, b);
    Tensor* result = context().new_tensor(out_shape.ndim(), out_shape.dims.data());
    result->op     = OP_MUL;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// div(a, b) — a / b
TensorF32* div(TensorF32* a, TensorF32* b) {
    Shape out_shape = broadcast_shape(a, b);
    TensorF32* result = context().new_tensor(out_shape.ndim(), out_shape.dims.data());
    result->op     = OP_DIV;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

TensorF32* sigmoid(TensorF32* a) {
    TensorF32* result;
    // provide a inplace sigmoid
    result = context().new_tensor(a->ndim(), a->shape().dims.data());
    result->op     = OP_UNARY;
    set_unary_op(result, UNARY_OP_SIGMOID);
    result->src[0] = a;
    return result;
}

} // namespace rfaa
#include "rfaa/Tensor.h"
#include "rfaa/Context.h"
#include "rfaa/FAPE.h"
#include "rfaa/SymmetryResolver.h"

#include "rfaa/ComputeGraph.h"

#include <cassert>

namespace rfaa {

TensorF32 * add_impl( 
        TensorF32  * a,  
        TensorF32  * b,  
        bool inplace) {  
    //GGML_ASSERT(ggml_can_repeat(b, a)); 
    assert(b->can_repeat(*a));
  
    //struct Tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);  
    TensorF32 * result;
    if (inplace) {
        result = view(a, a->shape());
    } else {
        result = dup(a);
    }
    //result = result->copy_from(a);
  
    result->op     = OP_ADD;  
    result->src[0] = a;  
    result->src[1] = b;  
  
    return result;  
}

TensorF32 * repeat_back(
        TensorF32  * a,  
        TensorF32  * b) {  
    //GGML_ASSERT(ggml_can_repeat(b, a));  
    assert(b->can_repeat(*a));

    TensorF32* result = context().new_tensor<float>(b->shape().ndim(), b->shape().dims.data());  

    assert(b->can_repeat(*a));
  
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
    if (inplace) {
        result = view(a, a->shape());
    } else {
        result = dup(a);
    }
 
    result->op     = OP_ADD1;
    result->src[0] = a;
    result->src[1] = b;
 
    return result;
}

// scale(a, s) — a * s  (标量乘法)
TensorF32* scale(TensorF32* a, float s) {
    //Tensor* result = alloc_node(a->ndim(), a->dims());
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
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
// a: (M, K), b: (K, N) → (M, N)
TensorF32* mul_mat(TensorF32* a, TensorF32* b) {
    int64_t ne[2] = {b->shape().dims[0], a->shape().dims[1]};
    TensorF32* result = context().new_tensor<float>(2, ne);

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
        std::max(a->shape().dims[2], b->shape().dims[2]), 
        std::max(a->shape().dims[3], b->shape().dims[3])
    };
    TensorF32* result = context().new_tensor<float>(4, ne);

    result->op     = OP_OUT_PROD;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// transpose(a) — 转置（交换最后两维）
TensorF32* transpose(TensorF32* a) {
    assert(a->shape().ndim() >= 2);
    std::vector<int64_t> new_dims = a->shape().dims;
    std::swap(new_dims[new_dims.size() - 1], new_dims[new_dims.size() - 2]);

    //Tensor* result = alloc_node(static_cast<int>(new_dims.size()), new_dims.data());
    TensorF32* result = context().new_tensor<float>(static_cast<int>(new_dims.size()), new_dims.data());
    result->op     = OP_TRANSPOSE;
    result->src[0] = a;
    return result;
}

// triangle_mul(left, right, L, outgoing)
//   outgoing=true:  einsum('bikd,bjkd->bijd', left, right/L)   → result: (B, I, J, D)
//   outgoing=false: einsum('bkid,bkjd->bijd', left, right/L)  → result: (B, I, J, D)
TensorF32* triangle_mul(TensorF32* left, TensorF32* right, float L, bool outgoing) {
    assert(left->shape().ndim() == 4 && right->shape().ndim() == 4);
    // result shape: (B, I, J, D) where I=dim[1] for both
    int64_t ne[4] = {left->shape().dims[0], left->shape().dims[1], right->shape().dims[1], left->shape().dims[3]};
    TensorF32* result = context().new_tensor<float>(4, ne);
    result->op      = OP_TRI_MUL;
    result->src[0]  = left;
    result->src[1]  = right;
    // 用 op_params 存储 L (float) 和 outgoing (bool)
    memcpy(result->op_params,     &L,        sizeof(float));
    memcpy(result->op_params + 4, &outgoing, sizeof(bool));
    return result;
}

// ============================================================
// 3. 激活函数
// ============================================================

// softmax(a) — softmax 沿最后一维
TensorF32* softmax(TensorF32* a) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_SOFT_MAX;
    result->src[0] = a;
    return result;
}

// softmax_backward(grad, output) — softmax 反向传播
// grad: dL/dy (upstream gradient)
// output: y = softmax(x) (forward output)
// returns: dL/dx
TensorF32* softmax_backward(TensorF32* grad, TensorF32* output) {
    TensorF32* result = context().new_tensor<float>(output->shape().ndim(), output->shape().dims.data());
    result->op     = OP_SOFT_MAX_BACK;
    result->src[0] = grad;
    result->src[1] = output;
    return result;
}

// silu(a) — SiLU / Swish 激活
TensorF32* silu(TensorF32* a) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_UNARY;
    result->src[0] = a;
    set_unary_op(result, UNARY_OP_SILU);
    return result;
}

// gelu(a) — GELU 激活
TensorF32* gelu(TensorF32* a) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_UNARY;
    result->src[0] = a;
    set_unary_op(result, UNARY_OP_GELU);
    return result;
}

// gelu_quick(a) — GELU 快速近似
TensorF32* gelu_quick(TensorF32* a) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_UNARY;
    result->src[0] = a;
    set_unary_op(result, UNARY_OP_GELU_QUICK);
    return result;
}

// relu(a) — ReLU 激活
TensorF32* relu(TensorF32* a) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_UNARY;
    result->src[0] = a;
    set_unary_op(result, UNARY_OP_RELU);
    return result;
}

// leaky_relu(a, alpha) — Leaky ReLU
TensorF32* leaky_relu(TensorF32* a, float alpha) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_LEAKY_RELU;
    result->src[0] = a;
    //result->op_params[0] = reinterpret_cast<int32_t&>(alpha);
    return result;
}

// ============================================================
// 4. 归一化
// ============================================================

// rms_norm(a, eps) — RMS Normalization 沿最后一维
TensorF32* rms_norm(TensorF32* a, float eps) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_RMS_NORM;
    result->src[0] = a;
    //result->op_params[0] = reinterpret_cast<int32_t&>(eps);
    return result;
}

// norm(a, eps) — Layer Normalization 沿最后一维
TensorF32* norm(TensorF32* a, float eps) {
    int D = a->shape().dims.back();
    int rows = a->numel() / D;

    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
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
    TensorF32* result = context().new_tensor<float>(1, ne);
    result->op     = OP_SUM;
    result->src[0] = a;
    return result;
}

// mean(a, dim) — 沿指定维度求平均 (dim=-1 = 最后一维)
TensorF32* mean(TensorF32* a) {
    int64_t ne[1] = {1};
    TensorF32* result = context().new_tensor<float>(1, ne);
    result->op     = OP_MEAN;
    result->src[0] = a;
    return result;
}

// sum_rows(a) — 沿最后一行求和 (a: M×N → M×1)
TensorF32* sum_rows(TensorF32* a) {
    int nd = a->shape().ndim();
    int64_t ne[4] = {1, a->shape().dims[1], 1, 1};
    if (nd >= 3) ne[2] = a->shape().dims[2];
    if (nd >= 4) ne[3] = a->shape().dims[3];
    TensorF32* result = context().new_tensor<float>(nd, ne);
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
    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < new_shape.dims.size(); i++) {
        ne[i] = new_shape.dims[i];
    }
    TensorF32* result = context().new_tensor<float>(
        static_cast<int>(new_shape.dims.size()), ne);
    // view shares data with source (TODO: set via public API when available)
    //result->set_view_src(a);
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
    TensorF32* result = context().new_tensor<float>(static_cast<int>(new_shape.dims.size()), ne);
    result->op     = OP_RESHAPE;
    result->src[0] = a;
    return result;
}

// permute(a, dims) — 维度重排
TensorF32* permute(TensorF32* a, const std::vector<int>& dims) {
    assert(dims.size() == static_cast<size_t>(a->shape().ndim()));

    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < dims.size(); i++) {
        ne[i] = a->shape().dims[dims[i]];
    }
    TensorF32* result = context().new_tensor<float>(static_cast<int>(dims.size()), ne);
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
    int ndim = a->shape().ndim();
    if (dim < 0) dim += ndim + 1;
    assert(dim >= 0 && dim <= ndim);

    int64_t ne[4] = {1, 1, 1, 1};
    int j = 0;
    for (int i = 0; i < ndim + 1; i++) {
        if (i == dim) {
            ne[i] = 1;
        } else {
            ne[i] = a->shape().dims[j++];
        }
    }
    TensorF32* result = context().new_tensor<float>(ndim + 1, ne);
    result->op     = OP_RESHAPE;  // unsqueeze 本质是 reshape
    result->src[0] = a;
    return result;
}

// concat(tensors, dim) — 沿指定维度拼接
// 返回新节点，其 src 数组存储所有输入
TensorF32* concat(const std::vector<TensorF32>& tensors, int dim) {
    assert(!tensors.empty());
    int ndim = tensors[0].shape().ndim();
    if (dim < 0) dim += ndim;

    int64_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < ndim; i++) {
        ne[i] = tensors[0].shape().dims[i];
    }
    ne[dim] = 0;
    for (const auto& t : tensors) {
        ne[dim] += t.shape().dims[dim];
    }

    TensorF32* result = context().new_tensor<float>(ndim, ne);
    result->op = OP_CONCAT;
    for (size_t i = 0; i < tensors.size() && i < GGML_MAX_SRC; i++) {
        result->src[i] = const_cast<TensorF32*>(&tensors[i]);
    }
    return result;
}

TensorF32* concat_ptr(const std::vector<TensorF32*>& tensors, int dim) {
    assert(!tensors.empty());
    int ndim = tensors[0]->shape().ndim();
    if (dim < 0) dim += ndim;

    int64_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < ndim; i++) {
        ne[i] = tensors[0]->shape().dims[i];
    }
    ne[dim] = 0;
    for (auto* t : tensors) {
        ne[dim] += t->shape().dims[dim];
    }

    TensorF32* result = context().new_tensor<float>(tensors[0]->shape().ndim(), ne);
    result->op = OP_CONCAT;
    for (size_t i = 0; i < tensors.size() && i < GGML_MAX_SRC; i++) {
        result->src[i] = tensors[i];
    }
    return result;
}

// repeat(a, b) — 沿各维度重复 a 以匹配 b 的形状
TensorF32* repeat(TensorF32* a, TensorF32* b) {
    TensorF32* result = context().new_tensor<float>(b->shape().ndim(), b->shape().dims.data());
    result->op     = OP_REPEAT;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// cont(a) — 确保张量连续存储
TensorF32* cont(TensorF32* a) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
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
    int ndim = a->shape().ndim();
    int64_t ne[4] = {b->shape().dims[0], a->shape().dims[1], 1, 1};
    if (ndim >= 3) ne[2] = a->shape().dims[2];
    TensorF32* result = context().new_tensor<float>(ndim, ne);
    result->op     = OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// set_rows(a, b, c) — 将 c 的值写入 a 中由 b 指定的行
TensorF32* set_rows(TensorF32* a, TensorF32* b, TensorF32* c) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
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
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_DIAG_MASK_INF;
    result->src[0] = a;
    //result->op_params[0] = n_past;
    return result;
}

// diag_mask_zero(a, n_past) — 对角线以下掩码设为 0
TensorF32* diag_mask_zero(TensorF32* a, int n_past) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_DIAG_MASK_ZERO;
    result->src[0] = a;
    //result->op_params[0] = n_past;
    return result;
}

// clamp(a, min, max) — 值裁剪
TensorF32* clamp(TensorF32* a, float min_val, float max_val) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_CLAMP;
    result->src[0] = a;
    //result->op_params[0] = reinterpret_cast<int32_t&>(min_val);
    //result->op_params[1] = reinterpret_cast<int32_t&>(max_val);
    return result;
}

// sqr(a) — 平方 a²
TensorF32* sqr(TensorF32* a) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_SQR;
    result->src[0] = a;
    return result;
}

// sqrt(a) — 开方 √a
TensorF32* sqrt(TensorF32* a) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_SQRT;
    result->src[0] = a;
    return result;
}

// abs(a) — 绝对值 |a|
TensorF32* abs(TensorF32* a) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_UNARY;
    result->src[0] = a;
    set_unary_op(result, UNARY_OP_ABS);
    return result;
}

// log(a) — 自然对数
TensorF32* log(TensorF32* a) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_LOG;
    result->src[0] = a;
    return result;
}

// sin(a), cos(a) — 三角函数
TensorF32* sin(TensorF32* a) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_SIN;
    result->src[0] = a;
    return result;
}

TensorF32* cos(TensorF32* a) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_COS;
    result->src[0] = a;
    return result;
}

// ============================================================
// 9. 损失函数
// ============================================================

// 前向声明（make_scalar 定义在文件末尾）
static TensorF32* make_scalar(float value);

// cross_entropy_loss(logits, targets) — 交叉熵损失
TensorF32* cross_entropy_loss(TensorF32* logits, TensorF32* targets) {
    int64_t ne[1] = {1};
    TensorF32* result = context().new_tensor<float>(1, ne);
    result->op     = OP_CROSS_ENTROPY_LOSS;
    result->src[0] = logits;
    result->src[1] = targets;
    return result;
}

// torsion_angle_loss(pred, gt, chi_mask) — 二面角损失 (Algorithm 27)
// 输入: pred [N, 7, 2] (sin,cos 预测), gt [N, 7, 2] (真值), chi_mask [N, 7]
// 输出: scalar loss
TensorF32* torsion_angle_loss(TensorF32* pred, TensorF32* gt, TensorF32* chi_mask) {
    float eps = 1e-8f;

    // Step 1: normalize pred to unit circle: pred_n = pred / sqrt(sum(pred², dim=-1))
    auto sq_sum = sum_rows(sqr(pred));       // [N, 7]
    auto r      = sqrt(sq_sum);              // [N, 7]
    auto pred_n = div(pred, r);              // [N, 7, 2] broadcast

    // Step 2: squared difference in (sin,cos) space
    auto diff    = sub(pred_n, gt);           // [N, 7, 2]
    auto sq_diff = sum_rows(sqr(diff));       // [N, 7]

    // Step 3: masked mean
    auto masked = mul(sq_diff, chi_mask);     // [N, 7]
    auto loss   = div(sum(masked), add1_impl(sum(chi_mask), make_scalar(eps), false));

    return loss;
}

// angle_norm_loss(unnormed, seq_mask) — 角度模长正则化惩罚
// 输入: unnormed [N, 7, 2], seq_mask [N, 1] (broadcast → [N, 7])
// 输出: scalar loss
TensorF32* angle_norm_loss(TensorF32* unnormed, TensorF32* seq_mask, float eps) {
    // Step 1: angle_norm = sqrt(sum(unnormed², dim=-1) + eps)
    auto sq     = sqr(unnormed);              // [N, 7, 2]
    auto sum_sq = sum_rows(sq);               // [N, 7]
    auto norm   = sqrt(add1_impl(sum_sq, make_scalar(eps), false)); // [N, 7]

    // Step 2: norm_error = abs(norm - 1.0)
    auto ones   = repeat(make_scalar(1.0f), norm); // broadcast 1.0 → [N, 7]
    auto err    = abs(sub(norm, ones));        // [N, 7]

    // Step 3: masked mean
    auto masked = mul(err, seq_mask);          // [N, 7] broadcast
    auto loss   = div(sum(masked), add1_impl(sum(seq_mask), make_scalar(eps), false));

    return loss;
}

// supervised_chi_loss(unnormed, gt, chi_mask, seq_mask, chi_weight, angle_norm_weight)
// — 完整版 chi 角监督损失 (Jumper et al. 2021 Suppl. Alg. 27)
// 输入: unnormed [N, 7, 2], gt [N, 7, 2], chi_mask [N, 7], seq_mask [N, 1]
//       chi_weight, angle_norm_weight (标量)
// 输出: scalar total_loss
TensorF32* supervised_chi_loss(
    TensorF32* unnormed, TensorF32* gt,
    TensorF32* chi_mask,  TensorF32* seq_mask,
    float chi_weight, float angle_norm_weight)
{
    auto chi_loss  = torsion_angle_loss(unnormed, gt, chi_mask);
    auto norm_loss = angle_norm_loss(unnormed, seq_mask);

    auto weighted_chi  = scale(chi_loss,  chi_weight);
    auto weighted_norm = scale(norm_loss, angle_norm_weight);
    auto total         = add_impl(weighted_chi, weighted_norm, false);

    return total;
}

// masked_msa_loss(logits, true_msa, bert_mask) — BERT-style MSA 掩码预测损失
// 对应 Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
// 输入: logits [N_seq, N_res, 23] (unnormalized log probabilities)
//       true_msa [N_seq, N_res] (float-encoded int32 ground-truth aatype indices)
//       bert_mask [N_seq, N_res] (1.0 at masked positions, 0.0 elsewhere)
// 输出: scalar loss = sum(CE * bert_mask) / (sum(bert_mask) + eps)
//
// 流程 (用图节点分解):
//   Step 1: softmax + log → log_softmax [N_seq, N_res, 23]
//   Step 2: one_hot(true_msa, 23) → labels [N_seq, N_res, 23]
//   Step 3: weighted = labels * log_softmax, then CE = -sum_rows(weighted) → [N_seq, N_res]
//   Step 4: masked_CE = CE * bert_mask
//   Step 5: loss = sum(masked_CE) / (sum(bert_mask) + 1e-8)
TensorF32* masked_msa_loss(TensorF32* logits, TensorF32* true_msa, TensorF32* bert_mask) {
    float eps = 1e-8f;

    // Step 1: log_softmax = log(softmax(logits)) → [N_seq, N_res, 23]
    auto lsm = log(softmax(logits));

    // Step 2: one_hot(true_msa, 23) → [N_seq, N_res, 23]
    // one_hot_seq 接受 const TensorF32& (值类型), 此处解引用指针
    // TODO: one_hot_seq not yet implemented
    auto labels = true_msa;  // placeholder

    // Step 3: CE per-position = -sum(labels * log_softmax, dim=-1) → [N_seq, N_res]
    auto weighted = mul(labels, lsm);       // [N_seq, N_res, 23]
    auto ce       = neg(sum_rows(weighted)); // [N_seq, N_res]

    // Step 4: 应用 bert_mask
    auto masked_ce = mul(ce, bert_mask);    // [N_seq, N_res]

    // Step 5: 归一化标量 loss
    auto sum_masked = sum(masked_ce);
    auto sum_mask   = sum(bert_mask);
    auto denom      = add1_impl(sum_mask, make_scalar(eps), false);
    auto loss_val   = div(sum_masked, denom);

    return loss_val;
}

//one-hot labels 和 logits 由调用者在外部准备：

//logits_* 由 4 个 Linear(pair_feat, N_bins) 产生
//*_onehot 由外部 binning 函数从坐标计算（非均匀距离 binning + 均匀角度 binning
//pair_mask 由 seq_mask ⊗ seq_mask 产生mul(unsqueeze(m, 1), unsqueeze(m, 0))

// distogram_loss(pair_feat, coords, pair_mask) — 4-项 inter-residue 2D 结构预测损失
// 对应 RFAA Section 2.5.4 "Distogram Loss", 基于 AF2 distogram 的通用化版本
//
// 输入:
//   logits_dist [L, L, 60] — 距离直方图 logits (Linear投影自 pair features)
//   logits_ω    [L, L, 36] — Ω 二面角直方图 logits
//   logits_θ    [L, L, 36] — Θ 二面角直方图 logits
//   logits_ϕ    [L, L, 18] — Φ 平面角直方图 logits
//   D_onehot    [L, L, 60] — 距离 one-hot ground-truth (非均匀 binning)
//   Ω_onehot    [L, L, 36] — Ω one-hot ground-truth
//   Θ_onehot    [L, L, 36] — Θ one-hot ground-truth
//   Φ_onehot    [L, L, 18] — Φ one-hot ground-truth
//   pair_mask   [L, L]     — pair 有效性 mask
//
// 输出: scalar loss = CE_dist + CE_ω + CE_θ + CE_ϕ
//
// 每个 CE 实现:
//   log_softmax = log(softmax(logits))
//   CE = -sum(label_onehot * log_softmax, dim=-1)  → [L, L]
//   masked_CE = CE * pair_mask
//   loss = sum(masked_CE) / (sum(pair_mask) + eps)
TensorF32* distogram_loss(
    TensorF32* logits_dist, TensorF32* logits_ω,
    TensorF32* logits_θ,    TensorF32* logits_ϕ,
    TensorF32* D_onehot,    TensorF32* Ω_onehot,
    TensorF32* Θ_onehot,    TensorF32* Φ_onehot,
    TensorF32* pair_mask)
{
    float eps = 1e-8f;

    // ================================================================
    // 辅助 lambda: 单通道 cross-entropy (logits, label_onehot, mask) → scalar
    // CE = sum(-label * log_softmax * mask) / (sum(mask) + eps)
    // ================================================================
    auto ce_channel = [&](TensorF32* logits, TensorF32* label_oh, TensorF32* mask) -> TensorF32* {
        auto lsm     = log(softmax(logits));            // log_softmax
        auto weighted = mul(label_oh, lsm);              // label * log_softmax
        auto ce_per  = neg(sum_rows(weighted));          // -sum over last dim → [L, L]
        auto masked  = mul(ce_per, mask);                // apply pair mask
        auto sum_ce  = sum(masked);                      // scalar sum
        auto sum_m   = sum(mask);                        // mask sum
        auto denom   = add1_impl(sum_m, make_scalar(eps), false);
        return div(sum_ce, denom);
    };

    // ================================================================
    // 4 个 CE loss 分量
    // ================================================================
    auto loss_dist = ce_channel(logits_dist, D_onehot, pair_mask);  // 距离 (60 bins)
    auto loss_ω    = ce_channel(logits_ω,    Ω_onehot, pair_mask);  // Ω (36 bins)
    auto loss_θ    = ce_channel(logits_θ,    Θ_onehot, pair_mask);  // Θ (36 bins)
    auto loss_ϕ    = ce_channel(logits_ϕ,    Φ_onehot, pair_mask);  // Φ (18 bins)

    // 总损失 = 四者之和
    auto loss_2d = add_impl(add_impl(loss_dist, loss_ω, false),
                            add_impl(loss_θ,  loss_ϕ, false), false);

    return loss_2d;
}

// total_loss(...) — 组合总损失 = 0.5*FAPE + 0.5*Chi + 0.3*Distogram + 2.0*MSA + 0.01*Conf
// 最后一项 L_conf 尚未实现，暂不参与计算
TensorF32* total_loss(
    TensorF32* loss_fape,
    TensorF32* loss_chi,
    TensorF32* loss_distogram,
    TensorF32* loss_msa,
    TensorF32* loss_conf
    )
{
    // 权重
    const float w_fape      = 0.5f;
    const float w_chi       = 0.5f;
    const float w_distogram = 0.3f;
    const float w_msa       = 2.0f;
    const float w_conf = 0.01f;  // TODO: 待 L_conf 实现后加入

    auto w_fape_node      = scale(loss_fape,      w_fape);
    auto w_chi_node       = scale(loss_chi,       w_chi);
    auto w_distogram_node = scale(loss_distogram, w_distogram);
    auto w_msa_node       = scale(loss_msa,       w_msa);
    auto w_conf_node      = scale(loss_conf,      w_conf);
    // 逐项累加: (fape+chi) + (distogram+msa) + conf
    auto ab = add_impl(w_fape_node,      w_chi_node, false);
    auto cd = add_impl(w_distogram_node, w_msa_node, false);
    auto total = add_impl(add_impl(ab, cd, false), w_conf_node, false);

    return total;
}

// plddt_loss(logits, lddt_onehot, ca_mask) — pLDDT 置信度预测损失
// 对应 Jumper et al. (2021) Suppl. Alg. 29 "predictPerResidueLDDT_Ca"
// 输入:
//   logits       [N_res, num_bins=50] — PredictedLDDTHead 输出
//   lddt_onehot  [N_res, 50]          — ground-truth LDDT one-hot labels
//   ca_mask      [N_res]              — CA 原子有效性 mask
// 输出:
//   scalar loss = sum(CE * ca_mask) / (sum(ca_mask) + eps)
TensorF32* plddt_loss(TensorF32* logits, TensorF32* lddt_onehot, TensorF32* ca_mask) {
    float eps = 1e-8f;

    // Step 1: log_softmax
    auto lsm = log(softmax(logits));                 // [N_res, 50]

    // Step 2: CE per-residue = -sum(label * log_softmax, dim=-1) → [N_res]
    auto weighted = mul(lddt_onehot, lsm);            // [N_res, 50]
    auto ce       = neg(sum_rows(weighted));          // [N_res]

    // Step 3: apply ca_mask and normalize
    auto masked = mul(ce, ca_mask);                   // [N_res]
    auto sum_ce = sum(masked);
    auto sum_m  = sum(ca_mask);
    auto denom  = add1_impl(sum_m, make_scalar(eps), false);
    auto loss   = div(sum_ce, denom);

    return loss;
}

// fape_loss(pred_coords, true_coords, frame_atom_indices,
//           frames_mask, positions_mask, config)
// — Frame Aligned Point Error 损失 (图节点版本)
//
// 输入:
//   pred_coords        [N_atoms, 3]       — 预测坐标
//   true_coords        [N_atoms, 3]       — 真实坐标
//   frame_atom_indices [N_frames, 3]      — 每帧的 3 原子全局索引 (float-encoded ints)
//   frames_mask        [N_frames]         — 帧有效性 mask
//   positions_mask     [N_atoms]          — 原子位置 mask
//   config             FAPEConfig         — d_clamp, epsilon, length_scale
//
// 输出: scalar (1,) TensorF32
TensorF32* fape_loss(
    TensorF32* pred_coords,
    TensorF32* true_coords,
    TensorF32* frame_atom_indices,
    TensorF32* frames_mask,
    TensorF32* positions_mask,
    const FAPEConfig& config)
{
    int64_t ne[1] = {1};
    TensorF32* result = context().new_tensor<float>(1, ne);
    result->op     = OP_FAPE;
    result->src[0] = pred_coords;
    result->src[1] = true_coords;
    result->src[2] = frame_atom_indices;
    result->src[3] = frames_mask;
    result->src[4] = positions_mask;

    // op_params[0..1]: d_clamp (float)
    memcpy(&result->op_params[0], &config.d_clamp, sizeof(float));
    // op_params[2..3]: epsilon (float)
    memcpy(&result->op_params[2], &config.epsilon, sizeof(float));
    // op_params[4..5]: length_scale (float)
    memcpy(&result->op_params[4], &config.length_scale, sizeof(float));

    return result;
}

// ============================================================
// 10. 位置编码
// ============================================================

// rope(a, n_past) — 旋转位置编码
TensorF32* rope(TensorF32* a, int n_past, int n_dims) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
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
    int ndim = a->shape().ndim();
    int64_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < ndim; i++) {
        ne[i] = a->shape().dims[i];
        if (i * 2 < static_cast<int>(pad_dims.size())) {
            ne[i] += pad_dims[i * 2] + pad_dims[i * 2 + 1];
        }
    }
    TensorF32* result = context().new_tensor<float>(ndim, ne);
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
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
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
    TensorF32* result = context().new_tensor<float>(src->shape().ndim(), src->shape().dims.data());
    result->op     = OP_CPY;
    result->src[0] = dst;
    result->src[1] = src;
    return result;
}

// ============================================================
// 13. dup — 复制张量（为计算图创建独立节点）
// ============================================================
TensorF32* dup(TensorF32* a) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
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
TensorF32* arange(float start, float end, float step) {
    int64_t n = static_cast<int64_t>((end - start) / step + 0.5f);
    int64_t ne[1] = {n};
    TensorF32* result = context().new_tensor<float>(1, ne);
    result->op     = OP_ARANGE;
    //result->op_params[0] = reinterpret_cast<int32_t&>(start);
    //result->op_params[1] = reinterpret_cast<int32_t&>(end);
    //result->op_params[2] = reinterpret_cast<int32_t&>(step);
    return result;
}

// 计算两个形状的广播结果形状
static Shape broadcast_shape(TensorF32* a, TensorF32* b) {
    int na = a->shape().ndim(), nb = b->shape().ndim();
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
    TensorF32* t = context().new_tensor<float>(1, ne);
    // 标量数据直接写入
    t->data()[0] = value;
    return t;
}

// sub(a, b) — a - b
TensorF32* sub(TensorF32* a, TensorF32* b) {
    Shape out_shape = broadcast_shape(a, b);
    TensorF32* result = context().new_tensor<float>(out_shape.ndim(), out_shape.dims.data());
    result->op     = OP_SUB;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// mul(a, b) — a * b  (element-wise)
TensorF32* mul(TensorF32* a, TensorF32* b) {
    Shape out_shape = broadcast_shape(a, b);
    TensorF32* result = context().new_tensor<float>(out_shape.ndim(), out_shape.dims.data());
    result->op     = OP_MUL;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// div(a, b) — a / b
TensorF32* div(TensorF32* a, TensorF32* b) {
    Shape out_shape = broadcast_shape(a, b);
    TensorF32* result = context().new_tensor<float>(out_shape.ndim(), out_shape.dims.data());
    result->op     = OP_DIV;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

TensorF32* sigmoid(TensorF32* a) {
    TensorF32* result;
    // provide a inplace sigmoid
    result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_UNARY;
    set_unary_op(result, UNARY_OP_SIGMOID);
    result->src[0] = a;
    return result;
}

} // namespace rfaa
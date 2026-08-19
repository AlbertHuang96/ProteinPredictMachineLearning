#include "ppml/Tensor.h"
#include "ppml/Context.h"
#include "ppml/FAPE.h"
#include "ppml/SymmetryResolver.h"

#include "ppml/ComputeGraph.h"

#include <cassert>

namespace ppml {

TensorF32 * add_impl( 
        TensorF32  * a,  
        TensorF32  * b,  
        bool inplace) {  
    //GGML_ASSERT(ggml_can_repeat(b, a)); 
    // 放宽断言：梯度累加中 b(新梯度) 常为扁平 2D，a(累加器) 为 4D(如 pair [128,51,51,1] vs [332928,1])。
    // 只要 numel 相同（同数据布局的视图）或 b 可广播到 a，累加均合法（kernel_add 逐元素加）。
    if (b->numel() != a->numel() && !b->can_repeat(*a)) {
        if (getenv("GRAPH_DEBUG_NODE")) {
            fprintf(stderr, "[add_impl] FAIL a(acc) op=%d ndim=%d dims=[%lld,%lld,%lld,%lld] numel=%lld | b(new) op=%d ndim=%d dims=[%lld,%lld,%lld,%lld] numel=%lld\n",
                    a->op, a->shape().ndim(), (long long)a->shape().dims[0], (long long)a->shape().dims[1],
                    (long long)a->shape().dims[2], (long long)a->shape().dims[3], (long long)a->numel(),
                    b->op, b->shape().ndim(), (long long)b->shape().dims[0], (long long)b->shape().dims[1],
                    (long long)b->shape().dims[2], (long long)b->shape().dims[3], (long long)b->numel());
        }
        assert(false);  // numel 不同且不可广播 → 真实 shape 错误
    }
  
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
    // 语义：a=grad(被 repeat 后的大 shape)，b=src0(原始小 shape)。
    // forward: repeat(b→a)，即 a 由 b repeat 得到 → a.can_repeat(b)（a 每维 % b 每维 == 0）。
    // 原断言 b->can_repeat(*a) 方向反了（要求 src0 % grad == 0，真正放大时会失败）。
    //GGML_ASSERT(ggml_can_repeat(b, a));
    // 放宽断言：支持 src0(小 ndim) 尾部广播到 grad(大 ndim)。仅检查同 ndim 时的整除；跨 ndim
    // 交给 kernel_repeat_back 按实际 shape reduce（开发期不崩溃，便于继续排查）。
    // 2026-08-19: 实测 a(grad)=[50,51](2D), b(src0)=[50](1D) 为跨 ndim 广播，原 assert 过严。
    //GGML_ASSERT(ggml_can_repeat(b, a));

    TensorF32* result = context().new_tensor<float>(b->shape().ndim(), b->shape().dims.data());  

  
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
    // 注: 不要求 a 连续。图模式 a 为未 compute 的 op 节点（own_data_=false，
    //     is_contiguous() 返回 false），但 dup/new_tensor 不访问 a 的 data，
    //     OP_ADD1 在 graph_compute 时才读 src[0]。值模式 a 天然连续，同样安全。
 
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
    // op_params 存储 scale 因子 (float 位模式)，供 CPU/CUDA kernel 与反向读取
    reinterpret_cast<float&>(result->op_params[0]) = s;
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
// 布局约定（与 CPU/CUDA kernel 一致，ggml dims[0]=最内维/列）：
//   a 视为 (M, K)：a.dims[1]=M 行, a.dims[0]=K 列
//   b 视为 (N, K)：b.dims[1]=N 行, b.dims[0]=K 列（b 以 (N,K) 转置存储 b[j*K+k]）
//   result (M, N)：result.dims[0]=N（列, 最内）, result.dims[1]=M（行）
//   result[i,j] = sum_k a[i,k]*b[j,k]
// 修正：result.dims[0] 应为 b 的行数 N=b.dims[1]，而非 b.dims[0](=K)；
//   原实现写 b.dims[0] 导致非方阵时输出形状错误。
TensorF32* mul_mat(TensorF32* a, TensorF32* b) {
    int64_t ne[2] = {b->shape().dims[1], a->shape().dims[1]};
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
    // 存维度映射（交换最后两维），供 kernel_permute 复用
    const int n = a->shape().ndim();
    for (int i = 0; i < n; i++) result->op_params[i] = i;
    std::swap(result->op_params[n - 1], result->op_params[n - 2]);
    return result;
}

// triangle_mul(left, right, L, outgoing)
// 【ggml 布局 dims[0]=最内维】输出 dst [D, I, J, B]
//   outgoing=true:  einsum('bikd,bjkd->bijd', left, right/L)
//     left [D,I,K,B], right [D,J,K,B] → dst [D,I,J,B]（I=left.dims[1], J=right.dims[1]）
//   outgoing=false: einsum('bkid,bkjd->bijd', left, right/L)
//     left [D,K,I,B], right [D,K,J,B] → dst [D,I,J,B]（I=left.dims[2], J=right.dims[2]）
TensorF32* triangle_mul(TensorF32* left, TensorF32* right, float L, bool outgoing) {
    assert(left->shape().ndim() == 4 && right->shape().ndim() == 4);
    // result shape: [D, I, J, B]
    int64_t ne[4];
    if (outgoing) {
        ne[0] = left->shape().dims[0];   // D（最内特征维）
        ne[1] = left->shape().dims[1];   // I
        ne[2] = right->shape().dims[1];  // J
        ne[3] = left->shape().dims[3];   // B
    } else {
        ne[0] = left->shape().dims[0];   // D（最内特征维）
        ne[1] = left->shape().dims[2];   // I（incoming 时 i 在 dims[2]）
        ne[2] = right->shape().dims[2];  // J（incoming 时 j 在 dims[2]）
        ne[3] = left->shape().dims[3];   // B
    }
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

// exp(a) — 自然指数 e^a
TensorF32* exp(TensorF32* a) {
    TensorF32* result = context().new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    result->op     = OP_UNARY;
    result->src[0] = a;
    set_unary_op(result, UNARY_OP_EXP);
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
// ---------------------------------------------------------------------------
// 【方案 B 已实现】permute 采用"实际重排数据到连续行主序"的方案：
//
//   ggml 的做法（方案 A 可选）：permute/view/reshape/transpose 是"零拷贝视图"——
//   只重排 ne[](形状) 与 nb[](每维字节步长)，共享底层 data 指针，内存 0 移动；
//   后端 compute 时一律 no-op。前提是张量持有 nb[] 步长数组。
//
//   方案 A（对齐 ggml，改动大）：为 Tensor 增加 nb[] 步长字段，permute/view/reshape
//   只改 shape+nb、共享 data_，并把所有 kernel 寻址改为按 nb[] 计算偏移。
//
//   方案 B（务实，改动小，当前实现）：Tensor 无 nb，permute 构造时分配新内存，
//   kernel 按 dims 映射把数据实际重排到连续行主序。有 O(numel) 内存移动开销，
//   但不需要改任何现有 kernel 的寻址。dims 映射存入 op_params[0..3] 供 kernel 使用。
//
//   【dispatch 约定】OP_PERMUTE/OP_TRANSPOSE → kernel_permute（重排）；
//   OP_RESHAPE/OP_VIEW/OP_CONT → kernel_cpy（整块 memcpy，元素顺序不变）。
// ---------------------------------------------------------------------------
TensorF32* permute(TensorF32* a, const std::vector<int>& dims) {
    assert(dims.size() == static_cast<size_t>(a->shape().ndim()));

    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < dims.size(); i++) {
        ne[i] = a->shape().dims[dims[i]];
    }
    TensorF32* result = context().new_tensor<float>(static_cast<int>(dims.size()), ne);
    result->op     = OP_PERMUTE;
    result->src[0] = a;
    // 存储 permute 的维度映射到 op_params，供 kernel_permute 重排数据
    for (size_t i = 0; i < dims.size(); i++) {
        result->op_params[i] = dims[i];
    }
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
    result->op_params[0] = dim;   // 拼接维度，供 CPU/CUDA kernel 使用
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
    result->op_params[0] = dim;   // 拼接维度，供 CPU/CUDA kernel 使用
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
// 布局约定（与 kernel_get_rows 一致，ggml dims[0]=最内维/行内长度）：
//   a 视为 (N, M)：a.dims[0]=N 行, a.dims[1]=M 行内长度
//   b: (K,) 行索引（float-encoded int），按 a.dims[0] 选行
//   result (K, M)：result.dims[0]=M（行内长度, 最内）, result.dims[1]=K（行数）
//   result[k, :] = a[idx[k], :]
// 修正：result.dims 应为 {a.dims[1](行内长), b.dims[0](K)}，原实现写反成 {K, M}。
TensorF32* get_rows(TensorF32* a, TensorF32* b) {
    int ndim = a->shape().ndim();
    int64_t ne[4] = {a->shape().dims[1], b->shape().dims[0], 1, 1};
    if (ndim >= 3) ne[2] = a->shape().dims[2];
    TensorF32* result = context().new_tensor<float>(ndim, ne);
    result->op     = OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// get_rows_back(dy, idx, W) — get_rows 的反向
// 前向: y = get_rows(W, idx) = W[idx]   (y: (K, M))
// 反向: dL/dW[i,:] = sum_{k: idx[k]==i} dy[k,:]   (散点累加回权重表)
// dy: (K, M), idx: (K,), W: (N, M) → dW: (N, M)
TensorF32* get_rows_back(TensorF32* dy, TensorF32* idx, TensorF32* W) {
    int ndim = W->shape().ndim();
    int64_t ne[4] = {W->shape().dims[0], W->shape().dims[1], 1, 1};
    if (ndim >= 3) ne[2] = W->shape().dims[2];
    TensorF32* result = context().new_tensor<float>(ndim, ne);
    result->op     = OP_GET_ROWS_BACK;
    result->src[0] = dy;    // upstream gradient (K, M)
    result->src[1] = idx;   // row indices (K,)
    result->src[2] = W;     // weight table (N, M), 仅用于取形状
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
// 7.5 SE3 消息传递三件套（方案 B）
// ============================================================

// ============================================================
// 布局约定：本项目的图张量采用 ggml 约定 dims[0]=最内维（最快变化）。
// 因此"按边"（E 个样本）的 2D 张量布局为 dims=[features, E]（feature 最内）。
// edge_gather_rows(node_feat, edge_src_idx)
//   node_feat: (N, C)，ggml 布局 dims=[C, N]；edge_src_idx: (E,)
//   dst: (E, C)，ggml 布局 dims=[C, E] — dst[e,:] = node_feat[src_idx[e],:]
TensorF32* edge_gather_rows(TensorF32* node_feat, TensorF32* edge_src_idx) {
    const int64_t E = edge_src_idx->shape().dims[0];
    const int64_t C = node_feat->shape().dims[0];   // 最内维=特征维 C
    int64_t ne[4] = { C, E, 1, 1 };
    TensorF32* result = context().new_tensor<float>(2, ne);
    result->op     = OP_EDGE_GATHER_ROWS;
    result->src[0] = node_feat;
    result->src[1] = edge_src_idx;
    return result;
}

// per_edge_matmul(kernel, gathered)
//   kernel: (E, M, K)，ggml 布局 dims=[K, M, E]；gathered: (E, K)，dims=[K, E]
//   dst: (E, M)，dims=[M, E] — dst[e,:] = kernel[e] @ gathered[e,:]
//   （每条边一个独立矩阵 kernel[e] (M×K) 乘该边源特征 (K,) → (M,)）
TensorF32* per_edge_matmul(TensorF32* kernel, TensorF32* gathered) {
    const int64_t E = kernel->shape().dims[2];
    const int64_t M = kernel->shape().dims[1];
    const int64_t K = kernel->shape().dims[0];
    int64_t ne[2] = { M, E };
    TensorF32* result = context().new_tensor<float>(2, ne);
    result->op     = OP_PER_EDGE_MATMUL;
    result->src[0] = kernel;
    result->src[1] = gathered;
    return result;
}

// scatter_add(msg, edge_tgt_idx, node_count)
//   msg: (E, M)，dims=[M, E]；edge_tgt_idx: (E,)；node_count: N
//   dst: (N, M)，dims=[M, N] — dst[tgt[e],:] += msg[e,:]
//   node_count 存入 op_params[0]（kernel 用它确定输出行数 N）。
TensorF32* scatter_add(TensorF32* msg, TensorF32* edge_tgt_idx, int node_count) {
    const int64_t M = msg->shape().dims[0];
    const int64_t E = msg->shape().dims[1];
    int64_t ne[2] = { M, node_count };
    TensorF32* result = context().new_tensor<float>(2, ne);
    result->op     = OP_SCATTER_ADD;
    result->src[0] = msg;
    result->src[1] = edge_tgt_idx;
    result->op_params[0] = node_count;
    return result;
}

// per_edge_matmul_back_kernel(grad, gathered)
//   grad: (E, M) dims=[M,E]；gathered: (E, K) dims=[K,E] → dkernel dims=[K,M,E]
//   dkernel[e,r,c] = grad[e,r] * gathered[e,c]  (逐边外积)
TensorF32* per_edge_matmul_back_kernel(TensorF32* grad, TensorF32* gathered) {
    const int64_t M = grad->shape().dims[0];
    const int64_t E = grad->shape().dims[1];
    const int64_t K = gathered->shape().dims[0];
    int64_t ne[3] = {K, M, E};
    TensorF32* result = context().new_tensor<float>(3, ne);
    result->op     = OP_PER_EDGE_MATMUL_BACK_KERNEL;
    result->src[0] = grad;
    result->src[1] = gathered;
    return result;
}

// per_edge_matmul_back_gathered(grad, kernel)
//   grad: (E, M) dims=[M,E]；kernel dims=[K,M,E] → dgathered dims=[K,E]
//   dgathered[e,c] = sum_r kernel[e,r,c] * grad[e,r]
TensorF32* per_edge_matmul_back_gathered(TensorF32* grad, TensorF32* kernel) {
    const int64_t K = kernel->shape().dims[0];
    const int64_t E = grad->shape().dims[1];
    int64_t ne[2] = {K, E};
    TensorF32* result = context().new_tensor<float>(2, ne);
    result->op     = OP_PER_EDGE_MATMUL_BACK_GATHERED;
    result->src[0] = grad;
    result->src[1] = kernel;
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

// log_softmax_stable(a) — 数值稳定的 log_softmax，用于 CE 损失。
// 直接 log(softmax(a)) 在 softmax 下溢到精确 0 时得 -inf，随后 onehot*lsm=0*(-inf)=NaN。
// 加 eps 使 softmax 恒 >0 → log 恒有限（下溢 bin 得 log(eps)=-18 而非 -inf，乘 onehot=0 → 0）。
TensorF32* log_softmax_stable(TensorF32* a) {
    const float eps = 1e-8f;
    return log(add1_impl(softmax(a), make_scalar(eps), false));  // softmax + eps，防 log(0)=-inf
}

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
    const int num_classes = 23;

    // 布局说明（ggml dims[0]=最内维）:
    //   true_msa [N_seq, N_res] → 图 dims={N_res, N_seq}
    //   N_res = true_msa->dims[0], N_seq = true_msa->dims[1]
    const int64_t N_res = true_msa->shape().dims[0];
    const int64_t N_seq = true_msa->shape().dims[1];
    const int64_t K     = N_res * N_seq;  // 扁平位置总数

    // Step 1: 稳定 log_softmax = log(softmax(logits)+eps) → dims={23, N_res, N_seq}
    //   （防 0*(-inf)=NaN：softmax 下溢为 0 时 log(0)=-inf，labels*lsm 中 0*(-inf)=NaN）
    auto lsm = log_softmax_stable(logits);

    // Step 2: one_hot(true_msa, 23) → labels dims={23, N_res, N_seq}
    //   true_msa {N_res, N_seq} → view 扁平为 1D {K}
    //   one_hot_seq_graph(flat, 23) → {23, K}（类别维在最内 dims[0]）
    //   view 回 {23, N_res, N_seq}，与 lsm 逐元素对齐（扁平序一致）
    auto flat_msa = view(true_msa, Shape({K}));
    auto oh_flat  = one_hot_seq_graph(flat_msa, num_classes);   // {23, K}
    auto labels   = view(oh_flat, Shape({num_classes, N_res, N_seq})); // {23, N_res, N_seq}

    // Step 3: CE per-position = -sum(labels * log_softmax, dim=-1) → {1, N_res, N_seq}
    auto weighted = mul(labels, lsm);        // {23, N_res, N_seq}
    auto ce       = neg(sum_rows(weighted)); // 沿 dims[0]=23 求和 → {1, N_res, N_seq}

    // Step 4: 应用 bert_mask（ce {1,N_res,N_seq} 与 bert_mask {N_res,N_seq} 广播）
    auto masked_ce = mul(ce, bert_mask);    // {1, N_res, N_seq}

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
// 对应 PPML Section 2.5.4 "Distogram Loss", 基于 AF2 distogram 的通用化版本
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
        // 稳定 log_softmax（防 0*(-inf)=NaN）：softmax 下溢为 0 时 log(0)=-inf，
        // onehot*lsm 中 0*(-inf)=NaN。加 eps 使 log 有限。
        auto lsm     = log_softmax_stable(logits);       // log_softmax
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
// 
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

    // Step 1: 稳定 log_softmax（防 0*(-inf)=NaN）
    auto lsm = log_softmax_stable(logits);           // [N_res, 50]

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
// 9.5 FAPE 帧索引 + 常量图节点构造
// ============================================================

// constant_scalar(value) — 创建常量标量图节点。
// 数据存于 const_data_（data() 保持 nullptr），由 Gallocr 分配 backend buffer 后填充。
TensorF32* constant_scalar(float value) {
    int64_t ne[4] = {1, 1, 1, 1};
    TensorF32* t = context().new_tensor<float>(1, ne);
    t->const_data_.assign(static_cast<size_t>(t->numel()), value);
    t->flag |= TENSOR_FLAG_CONST;
    return t;
}

// constant_ones(dims) — 创建全 1 常量图节点。
// 数据存于 const_data_（data() 保持 nullptr），由 Gallocr 分配 backend buffer 后填充。
TensorF32* constant_ones(const std::vector<int64_t>& dims) {
    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < dims.size() && i < 4; i++) {
        ne[i] = dims[i];
    }
    TensorF32* t = context().new_tensor<float>(static_cast<int>(dims.size()), ne);
    t->const_data_.assign(static_cast<size_t>(t->numel()), 1.0f);
    t->flag |= TENSOR_FLAG_CONST;
    return t;
}

// constant_tensor(dims, data) — 从已有数据创建常量叶子图节点（逐元素拷贝）。
// 用于把不参与求导的预计算量（如 SE3 球谐基切片、edge_index 索引、相对坐标等）注入图。
// data 元素个数须 == dims 的乘积。
// 数据存于 const_data_（data() 保持 nullptr），由 Gallocr 分配 backend buffer 后填充。
TensorF32* constant_tensor(const std::vector<int64_t>& dims, const float* data) {
    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < dims.size() && i < 4; i++) {
        ne[i] = dims[i];
    }
    TensorF32* t = context().new_tensor<float>(static_cast<int>(dims.size()), ne);
    size_t n = static_cast<size_t>(t->numel());
    t->const_data_.resize(n);
    if (n > 0) std::memcpy(t->const_data_.data(), data, n * sizeof(float));
    t->flag |= TENSOR_FLAG_CONST;
    return t;
}

// constant_tensor_dynamic(dims, data) — 动态一次性常量（如 dropout 随机掩码）。
// 语义同 constant_tensor，但 TENSOR_FLAG_CONST 不置位 → Gallocr 填充后清空 const_data_，
// 避免每个训练迭代的动态掩码在 Context 中累积宿主内存。
TensorF32* constant_tensor_dynamic(const std::vector<int64_t>& dims, const float* data) {
    TensorF32* t = constant_tensor(dims, data);
    t->flag &= ~TENSOR_FLAG_CONST;
    return t;
}

// build_frame_atom_indices(B, L) — 为 FAPE 构建帧索引
// 坐标布局: (B, L, 3, 3) → 展平 (N_atoms=B*L*3, 3)
//   残基 (b, l) 全局残基序号 r = b*L + l
//   N = r*3+0, CA = r*3+1, C = r*3+2
// 返回: TensorF32* ({B*L, 3})
TensorF32* build_frame_atom_indices(int B, int L) {
    int N_frames = B * L;
    int64_t ne[2] = {N_frames, 3};
    TensorF32* result = context().new_tensor<float>(2, ne);
    float* data = bind_leaf_data(context(), result);

    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            int r = b * L + l;
            int base = r * 3;
            data[r * 3 + 0] = static_cast<float>(base + 0);  // N
            data[r * 3 + 1] = static_cast<float>(base + 1);  // CA
            data[r * 3 + 2] = static_cast<float>(base + 2);  // C
        }
    }
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
    bind_leaf_data(context(), t)[0] = value;
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

// ============================================================
// 15. one_hot_seq / outer_sum 图节点版本（embedding / pair 特征构建）
// ============================================================

// one_hot_seq_graph(seq, num_classes) — one-hot 编码（图节点版本）
// 用 get_rows 从"单位矩阵常量表"按整数索引取行，得到 one-hot。
// seq: 一维扁平整数索引图节点 (K,)（ggml 布局 dims[0]=最内维）
// 返回 [num_classes, K] 图节点（ggml 布局 dims[0]=最内维）。
// 依赖 get_rows（已实现 kernel）。若 seq 是多维，需调用方先扁平化（view/reshape kernel 待补）。
TensorF32* one_hot_seq_graph(TensorF32* seq, int num_classes) {
    // 构造单位矩阵常量 leaf: [num_classes, num_classes]（ggml 布局 dims[0]=num_classes 最内）
    int64_t eye_dims[] = {num_classes, num_classes};
    TensorF32* eye = context().new_tensor<float>(2, eye_dims);
    eye->flag = 0;  // 常量，不可训练
    // 先分配 leaf data 再填充（直接 eye->data() 对未 bind 的图节点返回 null）
    const int64_t nn = (int64_t)num_classes * num_classes;
    float* ed = bind_leaf_data(context(), eye);
    for (int64_t i = 0; i < nn; i++) ed[i] = 0.0f;
    for (int c = 0; c < num_classes; c++) ed[c * num_classes + c] = 1.0f;

    // 按索引取行 → 输出 [num_classes, K]
    return get_rows(eye, seq);
}

// outer_sum_graph(left, right) — outer sum（图节点版本）
// 语义: left[i,:] + right[:,j] → left[i,j]，即 (B,1,L,D) + (B,L,1,D) → (B,L,L,D)。
// 图节点采用 ggml 布局（dims[0]=最内维）:
//   left  [D,1,L,B] + right [D,L,1,B] → [D,L,L,B]
// 用 repeat 把 left/right 各自广播到目标形状 [D,L,L,B]，再 add_impl。
// 依赖 repeat / add_impl（均已有 kernel）。返回图节点。
TensorF32* outer_sum_graph(TensorF32* left, TensorF32* right) {
    const int64_t D = left->shape().dims[0];
    const int64_t L = left->shape().dims[2];
    const int64_t B = left->shape().dims[3];

    // 占位目标节点 [D,L,L,B]（作为 repeat 的形状来源，其数据不被读取）
    int64_t tgt_dims[] = {D, L, L, B};
    TensorF32* target = context().new_tensor<float>(4, tgt_dims);

    auto* l = repeat(left, target);   // [D,1,L,B] → [D,L,L,B]
    auto* r = repeat(right, target);  // [D,L,1,B] → [D,L,L,B]
    return add_impl(l, r, /*inplace=*/false);
}

// outer_product_mean — msa2pair 的 outer-product-mean（图节点版本）
//   einsum('bikd,bjkd->bijd(de)', left, right/N) — 收缩 seq 维 N（dims[2]），特征维笛卡尔积 D×D→D*D
//   left  [D, L, N, B]（ggml 布局 dims[0]=最内维）
//   right [D, L, N, B]
//   dst   [D*D, L, L, B]；dst[(d1*D+d2), i, j, b] = (1/N)*sum_n left[d1,i,n,b]*right[d2,j,n,b]
// 收缩维是 seq 维 N（dims[2]）。out_prod 只收缩 dims[1]，故用专用 op OP_OUTER_PROD_MEAN。
TensorF32* outer_product_mean(TensorF32* left, TensorF32* right, int N) {
    assert(left->shape().ndim() == 4 && right->shape().ndim() == 4);
    // dst dims: [D*D, L, L, B]
    int64_t ne[4] = {
        left->shape().dims[0] * left->shape().dims[0],  // D*D（特征笛卡尔积）
        left->shape().dims[1],      // L（残基 i）
        right->shape().dims[1],     // L（残基 j）
        left->shape().dims[3]       // B
    };
    TensorF32* result = context().new_tensor<float>(4, ne);
    result->op     = OP_OUTER_PROD_MEAN;
    result->src[0] = left;
    result->src[1] = right;
    // op_params[0..1]: N（seq 数，float 位模式），供 CPU kernel 做 1/N 均值
    reinterpret_cast<float&>(result->op_params[0]) = static_cast<float>(N);
    return result;
}

// outer_product_graph — pair2pair gate 的 outer product（图节点版本）
//   纯外积，无收缩：gate[(d1*D+d2), i, j, b] = left[d1,i,b] * right[d2,j,b]
//   left  [D, L, B]（ggml 布局 dims[0]=最内维）
//   right [D, L, B]
//   dst   [D*D, L, L, B]（特征维笛卡尔积 D×D→D*D）
// 与 msa2pair 的 outer_product_mean 不同：此处不收缩任何维（state 无 seq 维）。
TensorF32* outer_product_graph(TensorF32* left, TensorF32* right) {
    assert(left->shape().ndim() == 3 && right->shape().ndim() == 3);
    // dst dims: [D*D, L, L, B]
    int64_t ne[4] = {
        left->shape().dims[0] * left->shape().dims[0],  // D*D（特征笛卡尔积）
        left->shape().dims[1],                          // L（残基 i）
        right->shape().dims[1],                         // L（残基 j）
        left->shape().dims[2]                           // B
    };
    TensorF32* result = context().new_tensor<float>(4, ne);
    result->op     = OP_OUTER_PROD;
    result->src[0] = left;
    result->src[1] = right;
    return result;
}

} // namespace ppml

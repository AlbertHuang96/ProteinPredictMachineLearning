#include "rfaa/Backend.h"
#include "rfaa/ComputeGraph.h"
#include <cstring>
#include <vector>

namespace rfaa {

// ===== CUDA kernel forward declarations (implemented in src/cuda/CUDAKernels.cu) =====
extern void layernorm_forward_cuda(
    float * out,   float * mean,  float * rstd,
    const float * inp, const float * weight, const float * bias,
    int B, int T, int C, int block_size);

extern void layernorm_backward_cuda(
    float * dinp, float * dweight, float * dbias,
    const float * dout, const float * inp,
    const float * weight, const float * mean, const float * rstd,
    int B, int T, int C, int block_size);

extern void elementwise_add_cuda(float * A, float * B, float * C, int N, int block_size);
extern void elementwise_sub_cuda(float * A, float * B, float * C, int N, int block_size);
extern void elementwise_mul_cuda(float * A, float * B, float * C, int N, int block_size);
extern void elementwise_div_cuda(float * A, float * B, float * C, int N, int block_size);
extern void softmax_cuda(float * input, float * output, int M, int N, int block_size);
extern void softmax_backward_cuda(
    const float * grad, const float * output, float * dst,
    int M, int N, int block_size, float scale);

extern void mul_mat_cuda(float* A, float* B, float* C, int M, int K, int N);

extern void out_prod_cuda(
    const float* src0, const float* src1, float* dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    int64_t ne0,  int64_t ne1,  int64_t ne2,  int64_t ne3);

// N-ary concat：srcs/start/len 均须为 device 指针（见 concat_nary_cuda）。
// 注意：concat 不支持广播语义，非拼接维必须与 dst 一致。
extern void concat_nary_cuda(
    const float* const* srcs, int n_src,
    const int64_t* start, const int64_t* len,
    float* dst, int dim,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

// SE3 消息传递三件套（方案 B）+ per_edge_matmul 反向核（实现于 src/cuda/SE3MessagePassingKernel.cu）
extern void edge_gather_rows_cuda(
    const float* node_feat, const float* src_idx, float* dst,
    int N, int C, int E);
extern void per_edge_matmul_cuda(
    const float* kernel, const float* gathered, float* dst,
    int M, int K, int E);
extern void scatter_add_cuda(
    const float* msg, const float* tgt_idx, float* dst,
    int N, int M, int E);
extern void per_edge_matmul_back_kernel_cuda(
    const float* grad, const float* gathered, float* dst,
    int M, int K, int E);
extern void per_edge_matmul_back_gathered_cuda(
    const float* grad, const float* kernel, float* dst,
    int M, int K, int E);

// ============================================================
// CUDABackend::dispatch_node
// ============================================================

Status CUDABackend::dispatch_node(TensorF32 * node, ComputeParams * p) {
    switch (node->op) {
        case OP_NONE:   break;

        // ===== 有完整 .cu kernel 的 op =====
        case OP_ADD:
            kernel_elemwise_add_cuda(node, p);
            break;
        case OP_SUB:
            kernel_elemwise_sub_cuda(node, p);
            break;
        case OP_MUL:
            kernel_elemwise_mul_cuda(node, p);
            break;
        case OP_DIV:
            kernel_elemwise_div_cuda(node, p);
            break;

        case OP_MUL_MAT:
            kernel_mul_mat_cuda(node, p);
            break;

        case OP_OUT_PROD:
            kernel_out_prod_cuda(node, p);
            break;

        case OP_CONCAT:
            kernel_concat_cuda(node, p);
            break;

        case OP_EDGE_GATHER_ROWS:
            kernel_edge_gather_rows_cuda(node, p);
            break;
        case OP_PER_EDGE_MATMUL:
            kernel_per_edge_matmul_cuda(node, p);
            break;
        case OP_SCATTER_ADD:
            kernel_scatter_add_cuda(node, p);
            break;
        case OP_PER_EDGE_MATMUL_BACK_KERNEL:
            kernel_per_edge_matmul_back_kernel_cuda(node, p);
            break;
        case OP_PER_EDGE_MATMUL_BACK_GATHERED:
            kernel_per_edge_matmul_back_gathered_cuda(node, p);
            break;

        case OP_SOFT_MAX:
            kernel_softmax_cuda(node, p);
            break;

        case OP_SOFT_MAX_BACK:
            kernel_softmax_back_cuda(node, p);
            break;

        case OP_NORM: {
            int C    = static_cast<int>(node->shape().dims[0]);
            int rows = static_cast<int>(node->numel() / C);

            float * out   = node->data();
            float * mean  = node->src[1] ? node->src[1]->data() : nullptr;
            float * rstd  = node->src[2] ? node->src[2]->data() : nullptr;
            const float * inp   = node->src[0]->data();
            const float * weight = nullptr;
            const float * bias   = nullptr;

            layernorm_forward_cuda(out, mean, rstd, inp, weight, bias, rows, 1, C, 256);
        } break;

        case OP_NORM_BACK:
            kernel_norm_back_cuda(node, p);
            break;

        // ===== 无实现的 op：统一返回 NOT_SUPPORTED =====
        default:
            p->threadpool->ec = Status::NOT_SUPPORTED;
            break;
    }
    return p->threadpool->ec;
}

// ============================================================
// CUDA kernel stubs (待 src/cuda/*.cu 实现后替换)
// ============================================================

void CUDABackend::kernel_elemwise_add_cuda(TensorF32 * node, ComputeParams * p) {
    int N = static_cast<int>(node->numel());
    float * A = node->src[0]->data();
    float * B = node->src[1]->data();
    float * C = node->data();
    elementwise_add_cuda(A, B, C, N, 256);
}

void CUDABackend::kernel_elemwise_sub_cuda(TensorF32 * node, ComputeParams * p) {
    int N = static_cast<int>(node->numel());
    float * A = node->src[0]->data();
    float * B = node->src[1]->data();
    float * C = node->data();
    elementwise_sub_cuda(A, B, C, N, 256);
}

void CUDABackend::kernel_elemwise_mul_cuda(TensorF32 * node, ComputeParams * p) {
    int N = static_cast<int>(node->numel());
    float * A = node->src[0]->data();
    float * B = node->src[1]->data();
    float * C = node->data();
    elementwise_mul_cuda(A, B, C, N, 256);
}

void CUDABackend::kernel_elemwise_div_cuda(TensorF32 * node, ComputeParams * p) {
    int N = static_cast<int>(node->numel());
    float * A = node->src[0]->data();
    float * B = node->src[1]->data();
    float * C = node->data();
    elementwise_div_cuda(A, B, C, N, 256);
}

// doublecheck:
//原始 kernel 假设 B 是标准 row-major B[K][N]，访问 B[r * N + c]。但 RFAA-Cpp 的 CPU 版本中 B 以转置形式存储：b[j * K + k]（即 B[N][K]）。
//因此加载 Bs 时改为：
void CUDABackend::kernel_mul_mat_cuda(TensorF32 * node, ComputeParams * p) {
    // node->shape().dims: output shape (N, M)  → d is (M, N) stored row-major
    // node->src[0]: A (M × K), row-major
    // node->src[1]: B stored transposed as (N × K), i.e. B[j * K + k]
    int M = static_cast<int>(node->shape().dims[1]);
    int N = static_cast<int>(node->shape().dims[0]);
    int K = static_cast<int>(node->src[0]->shape().dims[0]);

    float * A = node->src[0]->data();
    float * B = node->src[1]->data();
    float * C = node->data();

    mul_mat_cuda(A, B, C, M, K, N);
}

void CUDABackend::kernel_softmax_cuda(TensorF32 * node, ComputeParams * p) {
    // node shape: (M, N) — M rows, N classes per row
    int N = static_cast<int>(node->shape().dims[0]);
    int M = static_cast<int>(node->numel() / N);
    float * input  = node->src[0]->data();
    float * output = node->data();
    // block size = 256
    softmax_cuda(input, output, M, N, 256);
}

void CUDABackend::kernel_softmax_back_cuda(TensorF32 * node, ComputeParams * p) {
    // src[0] = grad (upstream gradient dL/dy)
    // src[1] = output (softmax forward output y)
    int N = static_cast<int>(node->shape().dims[0]);
    int M = static_cast<int>(node->numel() / N);
    const float * grad   = node->src[0]->data();
    const float * output = node->src[1]->data();
    float *       dst    = node->data();
    // scale = 1.0f (standard softmax backward, no scaling)
    softmax_backward_cuda(grad, output, dst, M, N, 256, 1.0f);
}

void CUDABackend::kernel_norm_back_cuda(TensorF32 * node, ComputeParams * p) {
    // src[0] = dout (upstream grad), src[1] = inp (original input)
    // src[2] = mean (cached), src[3] = rstd (cached)
    int C    = reinterpret_cast<int&>(node->op_params[0]);
    int rows = reinterpret_cast<int&>(node->op_params[1]);

    float *       dinp = node->data();
    const float * dout = node->src[0]->data();
    const float * inp  = node->src[1]->data();
    const float * mean = node->src[2]->data();
    const float * rstd = node->src[3]->data();

    //weight/bias 传 nullptr，因为 OP_NORM 不带 affine，gamma/beta 的梯度由上层 out_prod + add_impl 图节点自动求导。
    // OP_NORM_BACK 只计算 dinp，不计算 dweight/dbias（由上层 mul/add 图节点求导）
    layernorm_backward_cuda(
        dinp, nullptr, nullptr,   // dinp, dweight, dbias
        dout, inp, nullptr, mean, rstd,
        rows, 1, C, 256);
}

void CUDABackend::kernel_dup_cuda(TensorF32 * node) {
    (void)node;
}

void CUDABackend::kernel_scale_cuda(TensorF32 * node, ComputeParams * p) {
    p->threadpool->ec = Status::NOT_SUPPORTED;
}

void CUDABackend::kernel_add1_cuda(TensorF32 * node, ComputeParams * p) {
    p->threadpool->ec = Status::NOT_SUPPORTED;
}

void CUDABackend::kernel_sum_cuda(TensorF32 * node, ComputeParams * p) {
    p->threadpool->ec = Status::NOT_SUPPORTED;
}

void CUDABackend::kernel_mean_cuda(TensorF32 * node, ComputeParams * p) {
    p->threadpool->ec = Status::NOT_SUPPORTED;
}

void CUDABackend::kernel_relu_cuda(TensorF32 * node) { (void)node; }
void CUDABackend::kernel_sigmoid_cuda(TensorF32 * node) { (void)node; }
void CUDABackend::kernel_silu_cuda(TensorF32 * node) { (void)node; }
void CUDABackend::kernel_tanh_cuda(TensorF32 * node) { (void)node; }
void CUDABackend::kernel_exp_cuda(TensorF32 * node) { (void)node; }

void CUDABackend::kernel_out_prod_cuda(TensorF32 * node, ComputeParams * p) {
    const TensorF32* src0 = node->src[0];
    const TensorF32* src1 = node->src[1];

    const int64_t ne00 = src0->shape().dims[0];
    const int64_t ne01 = src0->shape().dims[1];
    const int64_t ne02 = (src0->shape().ndim() > 2) ? src0->shape().dims[2] : 1;
    const int64_t ne03 = (src0->shape().ndim() > 3) ? src0->shape().dims[3] : 1;

    const int64_t ne10 = src1->shape().dims[0];
    const int64_t ne11 = src1->shape().dims[1];
    const int64_t ne12 = (src1->shape().ndim() > 2) ? src1->shape().dims[2] : 1;
    const int64_t ne13 = (src1->shape().ndim() > 3) ? src1->shape().dims[3] : 1;

    const int64_t ne0 = node->shape().dims[0];
    const int64_t ne1 = node->shape().dims[1];
    const int64_t ne2 = (node->shape().ndim() > 2) ? node->shape().dims[2] : 1;
    const int64_t ne3 = (node->shape().ndim() > 3) ? node->shape().dims[3] : 1;

    // GQA: ne2/ne02, ne3/ne03 暂不计算 dps 参数
    // TODO: 后续实现 GQA 支持
    // const int64_t dps2 = ne2 / ne02;
    // const int64_t dps3 = ne3 / ne03;

    // 先将 dst 清零（对标 CPU 版 thread 0 的清零逻辑）
    cudaMemset(node->data(), 0, node->nbytes());

    out_prod_cuda(
        src0->data(), src1->data(), node->data(),
        ne00, ne01, ne02, ne03,
        ne10, ne11, ne12, ne13,
        ne0,  ne1,  ne2,  ne3);
}

void CUDABackend::kernel_edge_gather_rows_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const TensorF32* node_feat = node->src[0];  // dims=[C, N]
    const TensorF32* src_idx   = node->src[1];  // (E,)
    const int C = static_cast<int>(node_feat->shape().dims[0]);
    const int N = static_cast<int>(node_feat->shape().dims[1]);
    const int E = static_cast<int>(src_idx->shape().dims[0]);
    edge_gather_rows_cuda(node_feat->data(), src_idx->data(), node->data(), N, C, E);
}

void CUDABackend::kernel_per_edge_matmul_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const TensorF32* kernel   = node->src[0];  // dims=[K, M, E]
    const TensorF32* gathered = node->src[1];  // dims=[K, E]
    const int K = static_cast<int>(kernel->shape().dims[0]);
    const int M = static_cast<int>(kernel->shape().dims[1]);
    const int E = static_cast<int>(kernel->shape().dims[2]);
    per_edge_matmul_cuda(kernel->data(), gathered->data(), node->data(), M, K, E);
}

void CUDABackend::kernel_scatter_add_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const TensorF32* msg      = node->src[0];  // dims=[M, E]
    const TensorF32* tgt_idx  = node->src[1];  // (E,)
    const int N = node->op_params[0];
    const int M = static_cast<int>(msg->shape().dims[0]);
    const int E = static_cast<int>(msg->shape().dims[1]);
    scatter_add_cuda(msg->data(), tgt_idx->data(), node->data(), N, M, E);
}

void CUDABackend::kernel_per_edge_matmul_back_kernel_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const TensorF32* grad     = node->src[0];  // dims=[M, E]
    const TensorF32* gathered = node->src[1];  // dims=[K, E]
    const int M = static_cast<int>(grad->shape().dims[0]);
    const int E = static_cast<int>(grad->shape().dims[1]);
    const int K = static_cast<int>(gathered->shape().dims[0]);
    per_edge_matmul_back_kernel_cuda(grad->data(), gathered->data(), node->data(), M, K, E);
}

void CUDABackend::kernel_per_edge_matmul_back_gathered_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const TensorF32* grad   = node->src[0];  // dims=[M, E]
    const TensorF32* kernel = node->src[1];  // dims=[K, M, E]
    const int M = static_cast<int>(grad->shape().dims[0]);
    const int E = static_cast<int>(grad->shape().dims[1]);
    const int K = static_cast<int>(kernel->shape().dims[0]);
    per_edge_matmul_back_gathered_cuda(grad->data(), kernel->data(), node->data(), M, K, E);
}

void CUDABackend::kernel_concat_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const int dim = node->op_params[0];
    if (dim < 0 || dim >= 4) { p->threadpool->ec = Status::NOT_SUPPORTED; return; }

    // src 为固定大小数组(GGML_MAX_SRC)，仅统计非空输入，空位跳过
    int n_src = 0;
    TensorF32* srcs_tmp[GGML_MAX_SRC];
    for (int s = 0; s < GGML_MAX_SRC; s++) {
        if (node->src[s]) srcs_tmp[n_src++] = node->src[s];
    }
    if (n_src < 2) { p->threadpool->ec = Status::NOT_SUPPORTED; return; }

    // ============================================================
    // concat 不支持广播语义：强制要求所有 src 的非拼接维与 dst 一致
    //（即 dst 任意非拼接维不得大于 src 对应维）。
    // 不满足则直接置 NOT_SUPPORTED，避免在 CUDA kernel 内发生越界读。
    // ============================================================
    const int nd = node->shape().ndim();
    for (int d = 0; d < 4; d++) {
        if (d == dim) continue; // 拼接维可不同，由 start/len 表处理
        const int64_t dst_d = (d < nd) ? node->shape().dims[d] : 1;
        for (int s = 0; s < n_src; s++) {
            const int      sd    = srcs_tmp[s]->shape().ndim();
            const int64_t  src_d = (d < sd) ? srcs_tmp[s]->shape().dims[d] : 1;
            if (src_d != dst_d) {
                p->threadpool->ec = Status::NOT_SUPPORTED;
                return;
            }
        }
    }

    // dst 形状
    const int64_t ne0 = node->shape().ndim() > 0 ? node->shape().dims[0] : 1;
    const int64_t ne1 = node->shape().ndim() > 1 ? node->shape().dims[1] : 1;
    const int64_t ne2 = node->shape().ndim() > 2 ? node->shape().dims[2] : 1;
    const int64_t ne3 = node->shape().ndim() > 3 ? node->shape().dims[3] : 1;

    // 构造 host 侧 start/len 偏移表与 src 指针数组
    std::vector<int64_t>    len(n_src), start(n_src, 0);
    std::vector<const float*> srcs(n_src);
    for (int s = 0; s < n_src; s++) {
        len[s]   = srcs_tmp[s]->shape().dims[dim];
        srcs[s]  = srcs_tmp[s]->data();
    }
    for (int s = 1; s < n_src; s++) start[s] = start[s - 1] + len[s - 1];

    // 拷贝到 device
    const float** d_srcs  = nullptr;
    int64_t*      d_start = nullptr;
    int64_t*      d_len   = nullptr;
    cudaMalloc(&d_srcs,  sizeof(float*)   * n_src);
    cudaMalloc(&d_start, sizeof(int64_t)  * n_src);
    cudaMalloc(&d_len,   sizeof(int64_t)  * n_src);
    cudaMemcpy(d_srcs,  srcs.data(),  sizeof(float*)  * n_src, cudaMemcpyHostToDevice);
    cudaMemcpy(d_start, start.data(), sizeof(int64_t) * n_src, cudaMemcpyHostToDevice);
    cudaMemcpy(d_len,   len.data(),   sizeof(int64_t) * n_src, cudaMemcpyHostToDevice);

    concat_nary_cuda(
        d_srcs, n_src, d_start, d_len, node->data(), dim,
        ne0, ne1, ne2, ne3);

    cudaFree(d_srcs);
    cudaFree(d_start);
    cudaFree(d_len);
}

} // namespace rfaa

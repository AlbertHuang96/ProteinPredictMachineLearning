#include "rfaa/Backend.h"
#include "rfaa/ComputeGraph.h"
#include <cstring>

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

} // namespace rfaa

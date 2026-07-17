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

// ============================================================
// CUDABackend::dispatch_node
// ============================================================

Status CUDABackend::dispatch_node(Tensor * node, ComputeParams * p) {
    switch (node->op) {
        case OP_NONE:   break;

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

        case OP_ADD1:
            kernel_add1_cuda(node, p);
            break;

        case OP_SCALE:
            kernel_scale_cuda(node, p);
            break;

        case OP_MUL_MAT:
            kernel_mul_mat_cuda(node, p);
            break;

        case OP_SOFT_MAX:
            kernel_softmax_cuda(node, p);
            break;

        case OP_NORM: {
            // 复用 layernorm_forward_cuda，需把 mean/rstd 写入 node->src[1]/src[2]
            int C    = static_cast<int>(node->dims()[0]);
            int rows = static_cast<int>(node->numel() / C);

            float * out   = node->data();
            float * mean  = node->src[1] ? node->src[1]->data() : nullptr;
            float * rstd  = node->src[2] ? node->src[2]->data() : nullptr;
            const float * inp   = node->src[0]->data();
            const float * weight = nullptr;  // OP_NORM 不带 affine，gamma/beta 由上层图节点处理
            const float * bias   = nullptr;

            layernorm_forward_cuda(out, mean, rstd, inp, weight, bias, rows, 1, C, 256);
        } break;

        case OP_NORM_BACK:
        // layer norm backward
            kernel_norm_back_cuda(node, p);
            break;

        case OP_RMS_NORM:
            // 当前使用 rms_norm 作为简化实现，可复用 layernorm 或单独实现
            // TODO: 实现 CUDA rms_norm kernel
            break;

        case OP_DUP:
            kernel_dup_cuda(node);
            break;

        case OP_SUM:
            kernel_sum_cuda(node, p);
            break;

        case OP_MEAN:
            kernel_mean_cuda(node, p);
            break;

        case OP_UNARY: {
            // 简化为 sigmoid/relu/gelu 等直接调用对应 kernel
            const unary_op uop = get_unary_op(node);
            switch (uop) {
                case UNARY_OP_RELU:    kernel_relu_cuda(node);    break;
                case UNARY_OP_GELU:    kernel_gelu_cuda(node);    break;
                case UNARY_OP_SIGMOID: kernel_sigmoid_cuda(node); break;
                case UNARY_OP_SILU:    kernel_silu_cuda(node);    break;
                case UNARY_OP_TANH:    kernel_tanh_cuda(node);    break;
                case UNARY_OP_EXP:     kernel_exp_cuda(node);     break;
                default:
                    p->threadpool->ec = Status::NOT_SUPPORTED;
                    break;
            }
        } break;

        default:
            p->threadpool->ec = Status::NOT_SUPPORTED;
            break;
    }
    return p->threadpool->ec;
}

// ============================================================
// CUDA kernel stubs (待 src/cuda/*.cu 实现后替换)
// ============================================================

void CUDABackend::kernel_elemwise_add_cuda(Tensor * node, ComputeParams * p) {
    int N = static_cast<int>(node->numel());
    float * A = node->src[0]->data();
    float * B = node->src[1]->data();
    float * C = node->data();
    elementwise_add_cuda(A, B, C, N, 256);
}

void CUDABackend::kernel_elemwise_sub_cuda(Tensor * node, ComputeParams * p) {
    int N = static_cast<int>(node->numel());
    float * A = node->src[0]->data();
    float * B = node->src[1]->data();
    float * C = node->data();
    elementwise_sub_cuda(A, B, C, N, 256);
}

void CUDABackend::kernel_elemwise_mul_cuda(Tensor * node, ComputeParams * p) {
    int N = static_cast<int>(node->numel());
    float * A = node->src[0]->data();
    float * B = node->src[1]->data();
    float * C = node->data();
    elementwise_mul_cuda(A, B, C, N, 256);
}

void CUDABackend::kernel_elemwise_div_cuda(Tensor * node, ComputeParams * p) {
    int N = static_cast<int>(node->numel());
    float * A = node->src[0]->data();
    float * B = node->src[1]->data();
    float * C = node->data();
    elementwise_div_cuda(A, B, C, N, 256);
}

void CUDABackend::kernel_mul_mat_cuda(Tensor * node, ComputeParams * p) {
    p->threadpool->ec = Status::NOT_SUPPORTED;
}

void CUDABackend::kernel_softmax_cuda(Tensor * node, ComputeParams * p) {
    // node shape: (M, N) — M rows, N classes per row
    int N = static_cast<int>(node->dims()[0]);
    int M = static_cast<int>(node->numel() / N);
    float * input  = node->src[0]->data();
    float * output = node->data();
    // block size = 256
    softmax_cuda(input, output, M, N, 256);
}

void CUDABackend::kernel_norm_back_cuda(Tensor * node, ComputeParams * p) {
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

void CUDABackend::kernel_dup_cuda(Tensor * node) {
    // 简单 memcpy 到 device
}

void CUDABackend::kernel_scale_cuda(Tensor * node, ComputeParams * p) {
    p->threadpool->ec = Status::NOT_SUPPORTED;
}

void CUDABackend::kernel_add1_cuda(Tensor * node, ComputeParams * p) {
    p->threadpool->ec = Status::NOT_SUPPORTED;
}

void CUDABackend::kernel_sum_cuda(Tensor * node, ComputeParams * p) {
    p->threadpool->ec = Status::NOT_SUPPORTED;
}

void CUDABackend::kernel_mean_cuda(Tensor * node, ComputeParams * p) {
    p->threadpool->ec = Status::NOT_SUPPORTED;
}

void CUDABackend::kernel_relu_cuda(Tensor * node) {}
void CUDABackend::kernel_gelu_cuda(Tensor * node) {}
void CUDABackend::kernel_sigmoid_cuda(Tensor * node) {}
void CUDABackend::kernel_silu_cuda(Tensor * node) {}
void CUDABackend::kernel_tanh_cuda(Tensor * node) {}
void CUDABackend::kernel_exp_cuda(Tensor * node) {}

} // namespace rfaa

#include <cuda_runtime.h>
#include <cstdint>

namespace ppml {

// ============================================================
// SE3 消息传递三件套 CUDA 核（方案 B）
//   edge_gather_rows / per_edge_matmul / scatter_add
//   + per_edge_matmul 的两个反向核
// ============================================================

static inline int se3_ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// ============================================================
// edge_gather_rows: dst[e,:] = node_feat[src_idx[e],:]
//   node_feat: (N, C)，src_idx: (E,)，dst: (E, C)
// ============================================================
__global__ void edge_gather_rows_kernel(
    const float* __restrict__ node_feat,
    const float* __restrict__ src_idx,
    float* __restrict__ dst,
    int N, int C, int E)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int i = (int)src_idx[e];
    if (i < 0) i = 0;
    if (i >= N) i = N - 1;

    const float* src = node_feat + (int64_t)i * C;
    float*       out = dst       + (int64_t)e * C;
    for (int c = 0; c < C; c++) {
        out[c] = src[c];
    }
}

void edge_gather_rows_cuda(
    const float* node_feat, const float* src_idx, float* dst,
    int N, int C, int E)
{
    constexpr int BLOCK = 256;
    int grid = E > 0 ? se3_ceil_div(E, BLOCK) : 0;
    if (grid <= 0) grid = 1;
    edge_gather_rows_kernel<<<grid, BLOCK>>>(node_feat, src_idx, dst, N, C, E);
}

// ============================================================
// per_edge_matmul: dst[e,:] = kernel[e] @ gathered[e,:]
//   kernel: (E, M, K) row-major，gathered: (E, K)，dst: (E, M)
// ============================================================
__global__ void per_edge_matmul_kernel(
    const float* __restrict__ kernel,
    const float* __restrict__ gathered,
    float* __restrict__ dst,
    int M, int K, int E)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    const float* K_e = kernel   + (int64_t)e * M * K;
    const float* g_e = gathered + (int64_t)e * K;
    float*       d_e = dst      + (int64_t)e * M;

    for (int r = 0; r < M; r++) {
        float val = 0.0f;
        for (int c = 0; c < K; c++) {
            val += K_e[(int64_t)r * K + c] * g_e[c];
        }
        d_e[r] = val;
    }
}

void per_edge_matmul_cuda(
    const float* kernel, const float* gathered, float* dst,
    int M, int K, int E)
{
    constexpr int BLOCK = 256;
    int grid = E > 0 ? se3_ceil_div(E, BLOCK) : 0;
    if (grid <= 0) grid = 1;
    per_edge_matmul_kernel<<<grid, BLOCK>>>(kernel, gathered, dst, M, K, E);
}

// ============================================================
// scatter_add: dst[tgt[e],:] += msg[e,:]
//   msg: (E, M)，tgt_idx: (E,)，dst: (N, M)
//   kernel 先清零 dst，再用 atomicAdd 累加（多边可指向同目标节点）
// ============================================================
__global__ void scatter_add_zero_kernel(float* dst, int64_t total) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) dst[idx] = 0.0f;
}

__global__ void scatter_add_kernel(
    const float* __restrict__ msg,
    const float* __restrict__ tgt_idx,
    float* __restrict__ dst,
    int N, int M, int E)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int i = (int)tgt_idx[e];
    if (i < 0 || i >= N) return;

    const float* m_e = msg + (int64_t)e * M;
    float*       d_i = dst + (int64_t)i * M;
    for (int c = 0; c < M; c++) {
        atomicAdd(&d_i[c], m_e[c]);
    }
}

void scatter_add_cuda(
    const float* msg, const float* tgt_idx, float* dst,
    int N, int M, int E)
{
    constexpr int BLOCK = 256;

    //    total=N*M==0 时得 grid=0 → cudaLaunchKernel invalid argument → zero_kernel
    //    未执行 → dst 残留上一轮数据(nan) → CONCAT 读到 nan → 混合 SE3 前向爆炸
    //    (loss 276万)。实测: 反向 scatter_add(grad, idx, N=0) 时 total=0 必现。
    //    grid=0 时 kernel 无线程执行, 语义等价空操作（后续 scatter_add_kernel ge 同理
    //    已由 kernel 内 `if (e >= E) return` 保护）。
    const int64_t total = (int64_t)N * M;
    int gz = total > 0 ? se3_ceil_div((int)total, BLOCK) : 0;
    if (gz <= 0) gz = 1;
    scatter_add_zero_kernel<<<gz, BLOCK>>>(dst, total);

    int ge = E > 0 ? se3_ceil_div(E, BLOCK) : 0;
    if (ge <= 0) ge = 1;
    scatter_add_kernel<<<ge, BLOCK>>>(msg, tgt_idx, dst, N, M, E);
}

// ============================================================
// per_edge_matmul_back_kernel: dkernel[e,r,c] = grad[e,r] * gathered[e,c]
//   grad: (E, M)，gathered: (E, K)，dst: (E, M, K)
// ============================================================
__global__ void per_edge_matmul_back_kernel_kernel(
    const float* __restrict__ grad,
    const float* __restrict__ gathered,
    float* __restrict__ dst,
    int M, int K, int E)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    const float* g_e = grad     + (int64_t)e * M;
    const float* f_e = gathered + (int64_t)e * K;
    float*       d_e = dst      + (int64_t)e * M * K;

    for (int r = 0; r < M; r++) {
        const float gr = g_e[r];
        for (int c = 0; c < K; c++) {
            d_e[(int64_t)r * K + c] = gr * f_e[c];
        }
    }
}

void per_edge_matmul_back_kernel_cuda(
    const float* grad, const float* gathered, float* dst,
    int M, int K, int E)
{
    constexpr int BLOCK = 256;
    const int grid = se3_ceil_div(E, BLOCK);
    per_edge_matmul_back_kernel_kernel<<<grid, BLOCK>>>(grad, gathered, dst, M, K, E);
}

// ============================================================
// per_edge_matmul_back_gathered: dgathered[e,c] = sum_r kernel[e,r,c]*grad[e,r]
//   grad: (E, M)，kernel: (E, M, K)，dst: (E, K)
// ============================================================
__global__ void per_edge_matmul_back_gathered_kernel(
    const float* __restrict__ grad,
    const float* __restrict__ kernel,
    float* __restrict__ dst,
    int M, int K, int E)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    const float* g_e = grad   + (int64_t)e * M;
    const float* k_e = kernel + (int64_t)e * M * K;
    float*       d_e = dst    + (int64_t)e * K;

    for (int c = 0; c < K; c++) {
        float val = 0.0f;
        for (int r = 0; r < M; r++) {
            val += k_e[(int64_t)r * K + c] * g_e[r];
        }
        d_e[c] = val;
    }
}

void per_edge_matmul_back_gathered_cuda(
    const float* grad, const float* kernel, float* dst,
    int M, int K, int E)
{
    constexpr int BLOCK = 256;
    const int grid = se3_ceil_div(E, BLOCK);
    per_edge_matmul_back_gathered_kernel<<<grid, BLOCK>>>(grad, kernel, dst, M, K, E);
}

} // namespace ppml

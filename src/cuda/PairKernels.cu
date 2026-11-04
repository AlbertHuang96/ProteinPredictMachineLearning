// ============================================================
// PairKernels.cu — RFAA pair/MSA 专用算子的 CUDA 实现（2026-10-24）
//
// 覆盖 OP-CHECK 里"CUDA 不支持"的 8 个算子：
//   OP_OUTER_PROD_MEAN(102) / _BACK(103)   —— msa2pair：特征笛卡尔积 + 收缩 seq
//   OP_OUTER_PROD(104) / _BACK(105)        —— pair2pair 门控：纯外积
//   OP_TRI_MUL(100) / _BACK(101)           —— 三角乘法（outgoing/incoming）
//   OP_FAPE_BACK(99)                       —— FAPE 结构损失反向
//   OP_REPEAT_BACK(21) 跨 ndim 广播         —— 前导维整体累加（同 ndim 已有 kernel）
//
// 约定（与 CPU kernel 逐位对齐，见 src/backend/CPUKernels.cpp）：
//   · 全部 ggml 布局：dims[0] = 最内维，索引 = (((i3*n2 + i2)*n1 + i1)*n0 + i0)
//   · 线程映射统一为"一维 grid-stride + 线程内按 CPU 的取模顺序解码"
//     ⇒ 每个输出元素都由**单线程**完成整段累加，累加顺序与 CPU 完全一致（无原子、无归约树差异）
//   · grid 只受 1D gridDim.x(2^31-1) 约束，kernel 内 grid-stride ⇒ 任意形状都不丢数据
//   · 唯一的例外是 FAPE_BACK：不同帧/episode 会写同一原子的梯度，必须 atomicAdd
// ============================================================
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace ppml {

static inline int64_t pk_cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// 1D 启动配置：grid 上限取 2^31-1（CUDA 对 gridDim.x 的实际限制），kernel 内 grid-stride 兜底
static inline void pk_launch_1d(int64_t total, int threads, int64_t* grid) {
    int64_t g = pk_cdiv(total, threads);
    if (g < 1) g = 1;
    if (g > 2147483647LL) g = 2147483647LL;
    *grid = g;
}

// 与 CPU 一致的非有限值护栏（三角乘是 NaN/Inf 高发点，clamp 到 ±1e4）
__device__ static inline float pk_finite_clamp(float v) {
    if (!(v == v)) return 0.0f;
    if (v >  1e4f) return  1e4f;
    if (v < -1e4f) return -1e4f;
    return v;
}

// ============================================================
// 1) OP_TRI_MUL 前向
//   outgoing: dst[d,i,j,b] = (1/L) * sum_k left[d,i,k,b] * right[d,j,k,b]   K = left.dims[2]
//   incoming: dst[d,i,j,b] = (1/L) * sum_k left[d,k,i,b] * right[d,k,j,b]   K = left.dims[1]
// ============================================================
__global__ void tri_mul_kernel(const float* __restrict__ left,
                               const float* __restrict__ right,
                               float* __restrict__ dst,
                               int64_t D, int64_t I, int64_t J, int64_t B,
                               int64_t K, int64_t total,
                               float inv_L, int outgoing)
{
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        int64_t tmp = idx;
        const int64_t b = tmp % B; tmp /= B;
        const int64_t j = tmp % J; tmp /= J;
        const int64_t i = tmp % I;
        const int64_t d = tmp / I;

        float sum = 0.0f;
        if (outgoing) {
            const int64_t lb = ((d * I + i) * K) * B + b;   // + k*B
            const int64_t rb = ((d * J + j) * K) * B + b;
            for (int64_t k = 0; k < K; ++k)
                sum += left[lb + k * B] * right[rb + k * B];
        } else {
            const int64_t lb = ((d * K) * I + i) * B + b;   // + k*I*B
            const int64_t rb = ((d * K) * J + j) * B + b;   // + k*J*B
            for (int64_t k = 0; k < K; ++k)
                sum += left[lb + k * I * B] * right[rb + k * J * B];
        }
        dst[idx] = pk_finite_clamp(sum * inv_L);
    }
}

// ============================================================
// 2) OP_TRI_MUL_BACK
//   gleft  [D,I,K,B] (outgoing) / [D,K,I,B] (incoming)
//   gright [D,J,K,B] (outgoing) / [D,K,J,B] (incoming)
// 4 个独立 kernel：各自的输出元素解码顺序与 CPU 对应循环一致
// ============================================================
__global__ void tri_mul_back_gleft_out_kernel(const float* __restrict__ grad,
                                              const float* __restrict__ right,
                                              float* __restrict__ gleft,
                                              int64_t D, int64_t I, int64_t J, int64_t K, int64_t B,
                                              int64_t total, float inv_L)
{
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t tmp = idx;
        const int64_t b = tmp % B; tmp /= B;
        const int64_t k = tmp % K; tmp /= K;
        const int64_t i = tmp % I;
        const int64_t d = tmp / I;
        float sum = 0.0f;
        const int64_t gb = ((d * I + i) * J) * B + b;
        const int64_t rb = ((d * J) * K + k) * B + b;       // + j*K*B
        for (int64_t j = 0; j < J; ++j)
            sum += grad[gb + j * B] * right[rb + j * K * B];
        gleft[idx] = sum * inv_L;
    }
}

__global__ void tri_mul_back_gright_out_kernel(const float* __restrict__ grad,
                                               const float* __restrict__ left,
                                               float* __restrict__ gright,
                                               int64_t D, int64_t I, int64_t J, int64_t K, int64_t B,
                                               int64_t total, float inv_L)
{
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t tmp = idx;
        const int64_t b = tmp % B; tmp /= B;
        const int64_t k = tmp % K; tmp /= K;
        const int64_t j = tmp % J;
        const int64_t d = tmp / J;
        float sum = 0.0f;
        const int64_t gb = ((d * I) * J + j) * B + b;       // + i*J*B
        const int64_t lb = ((d * I) * K + k) * B + b;       // + i*K*B
        for (int64_t i = 0; i < I; ++i)
            sum += grad[gb + i * J * B] * left[lb + i * K * B];
        gright[idx] = sum * inv_L;
    }
}

__global__ void tri_mul_back_gleft_in_kernel(const float* __restrict__ grad,
                                             const float* __restrict__ right,
                                             float* __restrict__ gleft,
                                             int64_t D, int64_t I, int64_t J, int64_t K, int64_t B,
                                             int64_t total, float inv_L)
{
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t tmp = idx;
        const int64_t b = tmp % B; tmp /= B;
        const int64_t i = tmp % I; tmp /= I;
        const int64_t k = tmp % K;
        const int64_t d = tmp / K;
        float sum = 0.0f;
        const int64_t gb = ((d * I + i) * J) * B + b;
        const int64_t rb = ((d * K + k) * J) * B + b;
        for (int64_t j = 0; j < J; ++j)
            sum += grad[gb + j * B] * right[rb + j * B];
        gleft[idx] = sum * inv_L;
    }
}

__global__ void tri_mul_back_gright_in_kernel(const float* __restrict__ grad,
                                              const float* __restrict__ left,
                                              float* __restrict__ gright,
                                              int64_t D, int64_t I, int64_t J, int64_t K, int64_t B,
                                              int64_t total, float inv_L)
{
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t tmp = idx;
        const int64_t b = tmp % B; tmp /= B;
        const int64_t j = tmp % J; tmp /= J;
        const int64_t k = tmp % K;
        const int64_t d = tmp / K;
        float sum = 0.0f;
        const int64_t gb = ((d * I) * J + j) * B + b;
        const int64_t lb = ((d * K + k) * I) * B + b;
        for (int64_t i = 0; i < I; ++i)
            sum += grad[gb + i * J * B] * left[lb + i * B];
        gright[idx] = sum * inv_L;
    }
}

// ============================================================
// 3) OP_OUTER_PROD_MEAN 前向（msa2pair）
//   dst[(((d1*D+d2)*L + i)*L + j)*B + b] = (1/N) sum_n left[d1,i,n,b] * right[d2,j,n,b]
// ============================================================
__global__ void outer_prod_mean_kernel(const float* __restrict__ left,
                                       const float* __restrict__ right,
                                       float* __restrict__ dst,
                                       int64_t D, int64_t L, int64_t N, int64_t B,
                                       int64_t total, float inv_N)
{
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t tmp = idx;
        const int64_t b  = tmp % B; tmp /= B;
        const int64_t j  = tmp % L; tmp /= L;
        const int64_t i  = tmp % L; tmp /= L;
        const int64_t d2 = tmp % D; tmp /= D;
        const int64_t d1 = tmp;                              // tmp 已 < D（见 CPU 侧同处注释）

        const int64_t lb = ((d1 * L + i) * N) * B + b;      // + n*B
        const int64_t rb = ((d2 * L + j) * N) * B + b;
        float sum = 0.0f;
        for (int64_t n = 0; n < N; ++n)
            sum += left[lb + n * B] * right[rb + n * B];
        dst[idx] = sum * inv_N;
    }
}

// ============================================================
// 4) OP_OUTER_PROD_MEAN_BACK
//   gleft [d1,i,n,b] = (1/N) sum_{d2,j} grad[(d1*D+d2),i,j,b] * right[d2,j,n,b]
//   gright[d2,j,n,b] = (1/N) sum_{d1,i} grad[(d1*D+d2),i,j,b] * left [d1,i,n,b]
// ============================================================
__global__ void outer_prod_mean_back_left_kernel(const float* __restrict__ grad,
                                                 const float* __restrict__ right,
                                                 float* __restrict__ gleft,
                                                 int64_t D, int64_t L, int64_t N, int64_t B,
                                                 int64_t total, float inv_N)
{
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t tmp = idx;
        const int64_t b  = tmp % B; tmp /= B;
        const int64_t n  = tmp % N; tmp /= N;
        const int64_t i  = tmp % L;
        const int64_t d1 = tmp / L;

        float sum = 0.0f;
        for (int64_t j = 0; j < L; ++j) {
            const int64_t gb = (((d1 * D) * L + i) * L + j) * B + b;   // + d2*L*L*B
            for (int64_t d2 = 0; d2 < D; ++d2) {
                const float gv = grad[gb + d2 * L * L * B];
                const float rv = right[((d2 * L + j) * N + n) * B + b];
                sum += gv * rv;
            }
        }
        gleft[idx] = sum * inv_N;
    }
}

__global__ void outer_prod_mean_back_right_kernel(const float* __restrict__ grad,
                                                  const float* __restrict__ left,
                                                  float* __restrict__ gright,
                                                  int64_t D, int64_t L, int64_t N, int64_t B,
                                                  int64_t total, float inv_N)
{
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t tmp = idx;
        const int64_t b  = tmp % B; tmp /= B;
        const int64_t n  = tmp % N; tmp /= N;
        const int64_t j  = tmp % L;
        const int64_t d2 = tmp / L;

        float sum = 0.0f;
        for (int64_t i = 0; i < L; ++i) {
            const int64_t gb = ((d2 * L + i) * L + j) * B + b;             // d1=0 基址，+ d1*D*L*L*B
            for (int64_t d1 = 0; d1 < D; ++d1) {
                const float gv = grad[gb + d1 * D * L * L * B];
                const float lv = left[((d1 * L + i) * N + n) * B + b];
                sum += gv * lv;
            }
        }
        gright[idx] = sum * inv_N;
    }
}

// ============================================================
// 5) OP_OUTER_PROD 前向（pair2pair gate）
//   dst[(((d1*D+d2)*L + i)*L + j)*B + b] = left[d1,i,b] * right[d2,j,b]
// ============================================================
__global__ void outer_prod_kernel(const float* __restrict__ left,
                                  const float* __restrict__ right,
                                  float* __restrict__ dst,
                                  int64_t D, int64_t L, int64_t B, int64_t total)
{
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t tmp = idx;
        const int64_t b  = tmp % B; tmp /= B;
        const int64_t j  = tmp % L; tmp /= L;
        const int64_t i  = tmp % L; tmp /= L;
        const int64_t d2 = tmp % D; tmp /= D;
        const int64_t d1 = tmp;                              // tmp 已 < D

        dst[idx] = left[((d1 * L + i) * B + b)] * right[((d2 * L + j) * B + b)];
    }
}

// ============================================================
// 6) OP_OUTER_PROD_BACK
//   gleft [d1,i,b] = sum_{d2,j} grad[(d1*D+d2),i,j,b] * right[d2,j,b]
//   gright[d2,j,b] = sum_{d1,i} grad[(d1*D+d2),i,j,b] * left [d1,i,b]
// ============================================================
__global__ void outer_prod_back_left_kernel(const float* __restrict__ grad,
                                            const float* __restrict__ right,
                                            float* __restrict__ gleft,
                                            int64_t D, int64_t L, int64_t B, int64_t total)
{
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t tmp = idx;
        const int64_t b  = tmp % B; tmp /= B;
        const int64_t i  = tmp % L;
        const int64_t d1 = tmp / L;

        float sum = 0.0f;
        for (int64_t j = 0; j < L; ++j) {
            const int64_t gb = (((d1 * D) * L + i) * L + j) * B + b;   // + d2*L*L*B
            for (int64_t d2 = 0; d2 < D; ++d2)
                sum += grad[gb + d2 * L * L * B] * right[((d2 * L + j) * B + b)];
        }
        gleft[idx] = sum;
    }
}

__global__ void outer_prod_back_right_kernel(const float* __restrict__ grad,
                                             const float* __restrict__ left,
                                             float* __restrict__ gright,
                                             int64_t D, int64_t L, int64_t B, int64_t total)
{
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t tmp = idx;
        const int64_t b  = tmp % B; tmp /= B;
        const int64_t j  = tmp % L;
        const int64_t d2 = tmp / L;

        float sum = 0.0f;
        for (int64_t i = 0; i < L; ++i) {
            const int64_t gb = ((d2 * L + i) * L + j) * B + b;        // d1=0 基址，+ d1*D*L*L*B
            for (int64_t d1 = 0; d1 < D; ++d1)
                sum += grad[gb + d1 * D * L * L * B] * left[((d1 * L + i) * B + b)];
        }
        gright[idx] = sum;
    }
}

// ============================================================
// 7) OP_REPEAT_BACK（通用 4 维版，覆盖跨 ndim 广播）
//   逐 dst 元素、线程内四重循环累加 —— 与 CPU kernel_repeat_back 完全同序
//   off/rd[] 由 host 侧按 CPU 规则算好传入（off=0 当 nd_src>=nd_dst）
// ============================================================
__global__ void repeat_back4_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                    int64_t s0, int64_t s1, int64_t s2, int64_t s3,
                                    int64_t d0, int64_t d1, int64_t d2, int64_t d3,
                                    int64_t r0, int64_t r1, int64_t r2, int64_t r3,
                                    int64_t total)
{
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t t = idx;
        const int64_t j0 = t % d0; t /= d0;
        const int64_t j1 = t % d1; t /= d1;
        const int64_t j2 = t % d2; t /= d2;
        const int64_t j3 = t;

        float sum = 0.0f;
        for (int64_t k0 = 0; k0 < r0; ++k0) {
            const int64_t i0 = j0 + k0 * d0;
            for (int64_t k1 = 0; k1 < r1; ++k1) {
                const int64_t i1 = j1 + k1 * d1;
                for (int64_t k2 = 0; k2 < r2; ++k2) {
                    const int64_t i2 = j2 + k2 * d2;
                    for (int64_t k3 = 0; k3 < r3; ++k3) {
                        const int64_t i3 = j3 + k3 * d3;
                        sum += src[((i3 * s2 + i2) * s1 + i1) * s0 + i0];
                    }
                }
            }
        }
        dst[idx] = sum;
    }
}

// ============================================================
// 8) OP_FAPE_BACK
//   与 CPU compute_forward_fape_back 同式（Phase 1：只回传到 Steps 3~6）
//   多帧/多原子会命中同一原子梯度 ⇒ 必须 atomicAdd
// ============================================================
// 每个 frame 构造 T_inv = [R^T | -R^T A]（12 float），fk=0 时退化为单位阵
__global__ void fape_back_tinv_kernel(const float* __restrict__ coords,
                                      const float* __restrict__ frame_idx,
                                      const float* __restrict__ frames_mask,
                                      float* __restrict__ tinv,
                                      int64_t N_frames, float epsilon)
{
    const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N_frames) return;

    float* T = tinv + n * 12;
    for (int k = 0; k < 12; ++k) T[k] = 0.0f;
    T[0] = 1.0f; T[4] = 1.0f; T[8] = 1.0f;

    if (frames_mask[n] == 0.0f) return;

    const int64_t iA = (int64_t)frame_idx[n * 3 + 0];
    const int64_t iB = (int64_t)frame_idx[n * 3 + 1];
    const int64_t iC = (int64_t)frame_idx[n * 3 + 2];

    const float ax = coords[iA * 3 + 0], ay = coords[iA * 3 + 1], az = coords[iA * 3 + 2];
    const float bx = coords[iB * 3 + 0], by = coords[iB * 3 + 1], bz = coords[iB * 3 + 2];
    const float cx = coords[iC * 3 + 0], cy = coords[iC * 3 + 1], cz = coords[iC * 3 + 2];

    const float v1x = bx - ax, v1y = by - ay, v1z = bz - az;
    const float v2x = cx - ax, v2y = cy - ay, v2z = cz - az;

    const float n1 = sqrtf(v1x * v1x + v1y * v1y + v1z * v1z + epsilon);
    const float e1x = v1x / n1, e1y = v1y / n1, e1z = v1z / n1;

    const float dot = v2x * e1x + v2y * e1y + v2z * e1z;
    const float u2x = v2x - dot * e1x, u2y = v2y - dot * e1y, u2z = v2z - dot * e1z;
    const float n2 = sqrtf(u2x * u2x + u2y * u2y + u2z * u2z + epsilon);
    const float e2x = u2x / n2, e2y = u2y / n2, e2z = u2z / n2;

    const float e3x = e1y * e2z - e1z * e2y;
    const float e3y = e1z * e2x - e1x * e2z;
    const float e3z = e1x * e2y - e1y * e2x;

    T[0] = e1x; T[1] = e1y; T[2] = e1z;
    T[3] = e2x; T[4] = e2y; T[5] = e2z;
    T[6] = e3x; T[7] = e3y; T[8] = e3z;
    T[9]  = -(e1x * ax + e1y * ay + e1z * az);
    T[10] = -(e2x * ax + e2y * ay + e2z * az);
    T[11] = -(e3x * ax + e3y * ay + e3z * az);
}

// 单 block 归约 sum(frames_mask) / sum(positions_mask)（CPU 只累加非零项，这里保持一致）
__global__ void fape_back_masksum_kernel(const float* __restrict__ frames_mask,
                                         const float* __restrict__ positions_mask,
                                         int64_t N_frames, int64_t N_atoms,
                                         float* __restrict__ out)
{
    __shared__ float sfm[256];
    __shared__ float spm[256];
    const int tid = threadIdx.x;

    float af = 0.0f, ap = 0.0f;
    for (int64_t n = tid; n < N_frames; n += blockDim.x)
        if (frames_mask[n] != 0.0f) af += frames_mask[n];
    for (int64_t j = tid; j < N_atoms; j += blockDim.x)
        if (positions_mask[j] != 0.0f) ap += positions_mask[j];

    sfm[tid] = af;
    spm[tid] = ap;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sfm[tid] += sfm[tid + s];
            spm[tid] += spm[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) { out[0] = sfm[0]; out[1] = spm[0]; }
}

// Steps 3~6 反向，逐 (frame, atom) 并行；d_pred 需预先清零
__global__ void fape_back_accum_kernel(const float* __restrict__ pred,
                                       const float* __restrict__ truth,
                                       const float* __restrict__ tinv_pred,
                                       const float* __restrict__ tinv_true,
                                       const float* __restrict__ frames_mask,
                                       const float* __restrict__ positions_mask,
                                       const float* __restrict__ upstream,
                                       const float* __restrict__ masksum,
                                       float* __restrict__ d_pred,
                                       int64_t N_atoms, int64_t N_frames,
                                       float epsilon, float d_clamp, float length_scale)
{
    const float up     = upstream ? upstream[0] : 1.0f;
    const float denom  = masksum[0] * masksum[1] + epsilon;
    const float d_masked = up / denom / length_scale;

    const int64_t total  = N_frames * N_atoms;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int64_t n = idx / N_atoms;
        const int64_t j = idx - n * N_atoms;

        const float fm = frames_mask[n];
        if (fm == 0.0f) continue;
        const float pm = positions_mask[j];
        if (pm == 0.0f) continue;

        const float* Tp = tinv_pred + n * 12;
        const float* Tt = tinv_true + n * 12;

        const float px = pred[j * 3 + 0], py = pred[j * 3 + 1], pz = pred[j * 3 + 2];
        const float lpx = Tp[0] * px + Tp[1] * py + Tp[2] * pz + Tp[9];
        const float lpy = Tp[3] * px + Tp[4] * py + Tp[5] * pz + Tp[10];
        const float lpz = Tp[6] * px + Tp[7] * py + Tp[8] * pz + Tp[11];

        const float tx = truth[j * 3 + 0], ty = truth[j * 3 + 1], tz = truth[j * 3 + 2];
        const float ltx = Tt[0] * tx + Tt[1] * ty + Tt[2] * tz + Tt[9];
        const float lty = Tt[3] * tx + Tt[4] * ty + Tt[5] * tz + Tt[10];
        const float ltz = Tt[6] * tx + Tt[7] * ty + Tt[8] * tz + Tt[11];

        const float dx = lpx - ltx, dy = lpy - lty, dz = lpz - ltz;
        const float dist = sqrtf(dx * dx + dy * dy + dz * dz + epsilon);
        if (dist >= d_clamp) continue;   // clamp 区内梯度为 0

        const float d_dist = d_masked * fm * pm;
        const float inv_dist = 1.0f / (dist + epsilon);
        const float d_lpx = d_dist * dx * inv_dist;
        const float d_lpy = d_dist * dy * inv_dist;
        const float d_lpz = d_dist * dz * inv_dist;

        // Step 3 反向：d(global) = R @ d(local)，R = (R^T)^T
        const float d_px = Tp[0] * d_lpx + Tp[3] * d_lpy + Tp[6] * d_lpz;
        const float d_py = Tp[1] * d_lpx + Tp[4] * d_lpy + Tp[7] * d_lpz;
        const float d_pz = Tp[2] * d_lpx + Tp[5] * d_lpy + Tp[8] * d_lpz;

        atomicAdd(&d_pred[j * 3 + 0], d_px);
        atomicAdd(&d_pred[j * 3 + 1], d_py);
        atomicAdd(&d_pred[j * 3 + 2], d_pz);
    }
}

// ============================================================
// Host wrappers
// ============================================================
void tri_mul_cuda(const float* left, const float* right, float* dst,
                  int64_t D, int64_t I, int64_t J, int64_t K, int64_t B,
                  float inv_L, int outgoing)
{
    const int64_t total = D * I * J * B;
    if (total <= 0) return;
    int64_t grid = 0;
    pk_launch_1d(total, 256, &grid);
    tri_mul_kernel<<<(int)grid, 256>>>(left, right, dst, D, I, J, B, K, total, inv_L, outgoing);
}

void tri_mul_back_cuda(const float* grad, const float* left, const float* right,
                       float* gleft, float* gright,
                       int64_t D, int64_t I, int64_t J, int64_t K, int64_t B,
                       float inv_L, int outgoing)
{
    int64_t grid = 0;
    if (gleft) {
        const int64_t total = outgoing ? (D * I * K * B) : (D * K * I * B);
        if (total > 0) {
            pk_launch_1d(total, 256, &grid);
            if (outgoing)
                tri_mul_back_gleft_out_kernel<<<(int)grid, 256>>>(grad, right, gleft, D, I, J, K, B, total, inv_L);
            else
                tri_mul_back_gleft_in_kernel <<<(int)grid, 256>>>(grad, right, gleft, D, I, J, K, B, total, inv_L);
        }
    }
    if (gright) {
        const int64_t total = outgoing ? (D * J * K * B) : (D * K * J * B);
        if (total > 0) {
            pk_launch_1d(total, 256, &grid);
            if (outgoing)
                tri_mul_back_gright_out_kernel<<<(int)grid, 256>>>(grad, left, gright, D, I, J, K, B, total, inv_L);
            else
                tri_mul_back_gright_in_kernel <<<(int)grid, 256>>>(grad, left, gright, D, I, J, K, B, total, inv_L);
        }
    }
}

void outer_prod_mean_cuda(const float* left, const float* right, float* dst,
                          int64_t D, int64_t L, int64_t N, int64_t B, float inv_N)
{
    const int64_t total = D * D * L * L * B;
    if (total <= 0) return;
    int64_t grid = 0;
    pk_launch_1d(total, 256, &grid);
    outer_prod_mean_kernel<<<(int)grid, 256>>>(left, right, dst, D, L, N, B, total, inv_N);
}

void outer_prod_mean_back_cuda(const float* grad, const float* left, const float* right,
                               float* gleft, float* gright,
                               int64_t D, int64_t L, int64_t N, int64_t B, float inv_N)
{
    const int64_t total = D * L * N * B;
    if (total <= 0) return;
    int64_t grid = 0;
    pk_launch_1d(total, 256, &grid);
    if (gleft)
        outer_prod_mean_back_left_kernel<<<(int)grid, 256>>>(grad, right, gleft, D, L, N, B, total, inv_N);
    if (gright)
        outer_prod_mean_back_right_kernel<<<(int)grid, 256>>>(grad, left, gright, D, L, N, B, total, inv_N);
}

void outer_prod_cuda(const float* left, const float* right, float* dst,
                     int64_t D, int64_t L, int64_t B)
{
    const int64_t total = D * D * L * L * B;
    if (total <= 0) return;
    int64_t grid = 0;
    pk_launch_1d(total, 256, &grid);
    outer_prod_kernel<<<(int)grid, 256>>>(left, right, dst, D, L, B, total);
}

void outer_prod_back_cuda(const float* grad, const float* left, const float* right,
                          float* gleft, float* gright,
                          int64_t D, int64_t L, int64_t B)
{
    const int64_t total = D * L * B;
    if (total <= 0) return;
    int64_t grid = 0;
    pk_launch_1d(total, 256, &grid);
    if (gleft)
        outer_prod_back_left_kernel <<<(int)grid, 256>>>(grad, right, gleft, D, L, B, total);
    if (gright)
        outer_prod_back_right_kernel<<<(int)grid, 256>>>(grad, left, gright, D, L, B, total);
}

void repeat_back4_cuda(const float* src, float* dst,
                       int64_t s0, int64_t s1, int64_t s2, int64_t s3,
                       int64_t d0, int64_t d1, int64_t d2, int64_t d3,
                       int64_t r0, int64_t r1, int64_t r2, int64_t r3)
{
    const int64_t total = d0 * d1 * d2 * d3;
    if (total <= 0) return;
    int64_t grid = 0;
    pk_launch_1d(total, 256, &grid);
    if (getenv("PPML_PAIR_DEBUG")) {
        fprintf(stderr, "[REPEAT-BACK4] total=%lld grid=%lld src=%p dst=%p sd=[%lld,%lld,%lld,%lld] "
                        "dd=[%lld,%lld,%lld,%lld] rd=[%lld,%lld,%lld,%lld]\n",
                (long long)total, (long long)grid, (const void*)src, (void*)dst,
                (long long)s0, (long long)s1, (long long)s2, (long long)s3,
                (long long)d0, (long long)d1, (long long)d2, (long long)d3,
                (long long)r0, (long long)r1, (long long)r2, (long long)r3);
    }
    repeat_back4_kernel<<<(int)grid, 256>>>(src, dst, s0, s1, s2, s3, d0, d1, d2, d3,
                                            r0, r1, r2, r3, total);
    if (getenv("PPML_PAIR_DEBUG")) {
        cudaError_t e = cudaGetLastError();
        fprintf(stderr, "[REPEAT-BACK4] launch err=%s\n", cudaGetErrorString(e));
    }
}

void fape_back_cuda(const float* upstream, const float* pred, const float* truth,
                    const float* frame_idx, const float* frames_mask, const float* positions_mask,
                    float* d_pred, int64_t N_atoms, int64_t N_frames,
                    float d_clamp, float epsilon, float length_scale)
{
    if (N_atoms <= 0 || N_frames <= 0 || !d_pred) return;

    const size_t tinv_bytes = (size_t)N_frames * 12 * sizeof(float);
    float* scratch = nullptr;
    if (cudaMalloc(&scratch, 2 * tinv_bytes + 2 * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "[FAPE-BACK-CUDA] cudaMalloc scratch 失败 (N_frames=%lld)\n",
                (long long)N_frames);
        return;
    }
    float* tinv_pred = scratch;
    float* tinv_true = scratch + (size_t)N_frames * 12;
    float* masksum   = scratch + (size_t)N_frames * 24;

    // T_inv（每帧一个线程：Gram-Schmidt 只有 12 个输出，线程级并行足够）
    {
        const int threads = 128;
        const int64_t grid = pk_cdiv(N_frames, threads);
        fape_back_tinv_kernel<<<(int)grid, threads>>>(pred,  frame_idx, frames_mask, tinv_pred,  N_frames, epsilon);
        fape_back_tinv_kernel<<<(int)grid, threads>>>(truth, frame_idx, frames_mask, tinv_true, N_frames, epsilon);
    }
    fape_back_masksum_kernel<<<1, 256>>>(frames_mask, positions_mask, N_frames, N_atoms, masksum);

    cudaMemsetAsync(d_pred, 0, (size_t)N_atoms * 3 * sizeof(float));

    {
        const int64_t total = N_frames * N_atoms;
        int64_t grid = 0;
        pk_launch_1d(total, 256, &grid);
        fape_back_accum_kernel<<<(int)grid, 256>>>(pred, truth, tinv_pred, tinv_true,
                                                   frames_mask, positions_mask,
                                                   upstream, masksum, d_pred,
                                                   N_atoms, N_frames,
                                                   epsilon, d_clamp, length_scale);
    }

    cudaFree(scratch);
}

} // namespace ppml

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ppml {

// ============================================================
// Helper: ceil_div
// ============================================================
static inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

static inline void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
    }
}

// ============================================================
// LayerNorm Forward CUDA Kernel
// ============================================================

__global__ void layernorm_forward_kernel(
    float * out,   float * mean,  float * rstd,
    const float * inp, const float * weight, const float * bias,
    int N, int C) {

    // 每个 block 处理一个样本 (row)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    // N = B * T
    // idx  = b * T + t
    // const float* inp_bt = inp + b * T * C + t * C;
    // b*T*C + t*C = (b*T + t)*C = idx*C
    const float * inp_bt   = inp   + idx * C;
    float *       out_bt   = out   + idx * C;

    // ---- 计算均值 ----
    float m = 0.0f;
    for (int i = 0; i < C; i++) {
        m += inp_bt[i];
    }
    m /= C;

    // ---- 计算方差 ----
    float v = 0.0f;
    for (int i = 0; i < C; i++) {
        float diff = inp_bt[i] - m;
        v += diff * diff;
    }
    v /= C;

    // ---- 归一化 + affine ----
    float s = 1.0f / sqrtf(v + 1e-5f);  // rstd
    for (int i = 0; i < C; i++) {
        float norm = (inp_bt[i] - m) * s;
        float wi   = weight ? weight[i] : 1.0f;
        float bi   = bias   ? bias[i]   : 0.0f;
        out_bt[i]  = norm * wi + bi;
    }

    // ---- 缓存 mean / rstd 供 backward 使用 ----
    mean[idx] = m;
    rstd[idx] = s;
}

void layernorm_forward_cuda(
    float * out,   float * mean,  float * rstd,
    const float * inp, const float * weight, const float * bias,
    int B, int T, int C,
    int block_size) {

    int N = B * T;
    int grid_size = ceil_div(N, block_size);
    layernorm_forward_kernel<<<grid_size, block_size>>>(
        out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// RMS Norm CUDA Kernel (OP_RMS_NORM)
// 语义对齐 CPU kernel_rms_norm (CPUKernels.cpp:1218):
//   dst[d] = src[d] * 1/sqrt(mean(src^2) + eps) ，沿 dims[0](最内维, 长 ncols) 归一化
// 布局: 每 block 处理一行 (nrows 行, 每行 ncols 列)
//   一个 warp(32 线程) 瓜分一行的 ncols 列 → 串行步长 WARP_SIZE 累加 x^2
//   → __shfl_xor_sync 蝴蝶全归约(所有 32 lane 都拿到 sum, 后续每个线程都要用 scale)
//   → 每个线程写回自己那列 dst = scale * x
// 前置条件: ncols % WARP_SIZE == 0 (与参考实现一致, 不满足时回落 CPU)
// ============================================================
#define WARP_SIZE 32

__global__ void rms_norm_f32_kernel(
    const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum: 每个线程累加 (ncols/32) 个 x^2
    for (int col = tid; col < ncols; col += WARP_SIZE) {
        const float xi = x[(int64_t)row * ncols + col];
        tmp += xi * xi;
    }

    // warp 内蝴蝶全归约: 5 轮后所有 32 个 lane 都持有整行的 x^2 之和
    // 用 __shfl_xor_sync (全归约) 而非 __shfl_down_sync:
    //   RMS 归约后每个线程都要用 scale 写自己那列, xor 一次归约即广播到全员;
    //   down 只把和归到 lane0, 还需额外一次 __shfl_sync(tmp,0) 广播, 多一跳。
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, WARP_SIZE);
    }

    const float mean  = tmp / ncols;          // mean(x^2)
    const float scale = rsqrtf(mean + eps);   // 1/sqrt(mean+eps)

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst[(int64_t)row * ncols + col] = scale * x[(int64_t)row * ncols + col];
    }
}

void rms_norm_cuda(const float * x, float * dst,
                   const int64_t ncols, const int64_t nrows,
                   const float eps, cudaStream_t stream) {
    if (ncols <= 0 || nrows <= 0) return;
    // 前置: ncols 必须 32 整除, 否则每个 warp 覆盖不完整行, 回落 CPU
    if (ncols % WARP_SIZE != 0) return;
    const dim3 block_dims(WARP_SIZE, 1, 1);
    rms_norm_f32_kernel<<<(unsigned)nrows, block_dims, 0, stream>>>(
        x, dst, (int)ncols, eps);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// OP_GET_ROWS (embedding 查表前向) CUDA kernel
// 简化 2D 版, 对齐 CPU kernel_get_rows (CPUKernels.cpp:1928):
//   W   (N, M) 行主序, dims[0]=N 行数, dims[1]=M 行内长
//   idx (K,)   float-encoded int 行索引
//   dst (K, M) 行主序: dst[k,:] = W[idx[k],:]
// 越界: 与 CPU 一致 —— idx<0 钳到 0, idx>=N 钳到 N-1
// 并行: blockIdx.x → 行 k (每行一个 block);
//       行内 M 维由 blockIdx.y*blockDim.x + threadIdx.x grid-stride 覆盖
// ============================================================
__global__ void get_rows_f32_kernel(
    const float * __restrict__ W,   // (N, M)
    const float * __restrict__ idx, // (K,)
    float       * __restrict__ dst, // (K, M)
    const int64_t N, const int64_t M, const int64_t K) {

    const int64_t k = blockIdx.x;                 // 一个 block 处理一行 dst[k,:]
    if (k >= K) return;

    // 读行索引并钳制 (对齐 CPU: i<0→0, i>=N→N-1)
    const float idxv = idx[k];
    int64_t i = (int64_t)idxv;
    if (i < 0)    i = 0;
    if (i >= N)   i = N - 1;

    // 沿 M 维 grid-stride 拷贝 (合并访存: W[i*M+m] 与 dst[k*M+m] 均连续)
    const float * __restrict__ src_row = W + i * M;
    float       * __restrict__ dst_row = dst + k * M;
    const int64_t tid = blockIdx.y * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.y * blockDim.x;
    for (int64_t m = tid; m < M; m += stride) {
        dst_row[m] = src_row[m];
    }
}

void get_rows_cuda(const float * W, const float * idx, float * dst,
                   int64_t N, int64_t M, int64_t K, cudaStream_t stream) {
    if (N <= 0 || M <= 0 || K <= 0) return;

    // block: 每 block 256 线程; grid.x = K 行, grid.y 覆盖 M 维 (gridDim.y 上限 65535)
    constexpr int BLOCK = 256;
    const int64_t need_y = (M + BLOCK - 1) / BLOCK;
    const int64_t gy = (need_y > 65535) ? 65535 : need_y;
    dim3 grid((unsigned)K, (unsigned)gy, 1);
    dim3 block(BLOCK, 1, 1);

    get_rows_f32_kernel<<<grid, block, 0, stream>>>(W, idx, dst, N, M, K);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// OP_GET_ROWS_BACK (embedding 查表反向) CUDA kernel — 方案 A
// 对齐 CPU kernel_get_rows_back (CPUKernels.cpp:1961):
//   dy (K,M) 上游梯度, idx(K,) float-encoded 行索引, W (N,M) 取形状
//   dst dW (N,M): 先清零, 再按 idx 散点累加 dW[i,:] += dy[k,:]
// 越界: i<0 || i>=N 丢弃 (对齐 CPU)
// 并行: 每线程处理一个 grad 行 k 的一列 col, atomicAdd 累加
//   → O(K*M) 复杂度 (对齐 CPU), 同一 dst 行被多 k 引用时 atomic 保证正确
// dst 清零由 host 侧 cudaMemset 完成 (对齐 CPU memset)
// ============================================================
__global__ void get_rows_back_scatter_kernel(
    const float * __restrict__ dy,   // (K, M)
    const float * __restrict__ idx,  // (K,)
    float       * __restrict__ dW,   // (N, M) 已清零
    const int64_t N, const int64_t M, const int64_t K) {

    // 线程网格: x → k, y → col (M 维分块)
    const int64_t k = blockIdx.x;
    if (k >= K) return;
    const int64_t i = (int64_t)idx[k];
    if (i < 0 || i >= N) return;   // 越界丢弃 (对齐 CPU)

    const int64_t col = blockIdx.y * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.y * blockDim.x;
    for (int64_t c = col; c < M; c += stride) {
        atomicAdd(&dW[i * M + c], dy[k * M + c]);
    }
}

void get_rows_back_cuda(const float * dy, const float * idx, float * dW,
                        int64_t N, int64_t M, int64_t K, cudaStream_t stream) {
    if (N <= 0 || M <= 0 || K <= 0) return;

    // 先清零 (对齐 CPU: memset 0 再散点累加)
    cudaMemset(dW, 0, (size_t)N * M * sizeof(float));

    // block: 每 block 256 线程; grid.x = K (grad 行), grid.y 覆盖 M 维
    constexpr int BLOCK = 256;
    const int64_t need_y = (M + BLOCK - 1) / BLOCK;
    const int64_t gy = (need_y > 65535) ? 65535 : need_y;
    dim3 grid((unsigned)K, (unsigned)gy, 1);
    dim3 block(BLOCK, 1, 1);

    get_rows_back_scatter_kernel<<<grid, block, 0, stream>>>(dy, idx, dW, N, M, K);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// LayerNorm Backward CUDA Kernel
// ============================================================

__global__ void layernorm_backward_kernel(
    float * dinp, float * dweight, float * dbias,
    const float * dout, const float * inp,
    const float * weight, const float * mean, const float * rstd,
    int B, int T, int C) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;
    int b = idx / T;
    int t = idx % T;

    const float * dout_bt = dout + b * T * C + t * C;
    const float * inp_bt  = inp  + b * T * C + t * C;
    float *       dinp_bt = dinp + b * T * C + t * C;
    const float   mean_bt = mean[b * T + t];
    const float   rstd_bt = rstd[b * T + t];

    // first: two reduce operations
    float dnorm_mean      = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i  = (weight ? weight[i] : 1.0f) * dout_bt[i];
        dnorm_mean      += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean      /= C;
    dnorm_norm_mean /= C;

    // now iterate again and accumulate all the gradients
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i  = (weight ? weight[i] : 1.0f) * dout_bt[i];
        // gradient contribution to bias
        if (dbias) atomicAdd(&dbias[i], dout_bt[i]);
        // gradient contribution to weight
        if (dweight) atomicAdd(&dweight[i], norm_bti * dout_bt[i]);
        // gradient contribution to input
        float dval = dnorm_i - dnorm_mean - norm_bti * dnorm_norm_mean;
        dval *= rstd_bt;
        dinp_bt[i] = dval;
    }
}

void layernorm_backward_cuda(
    float * dinp, float * dweight, float * dbias,
    const float * dout, const float * inp,
    const float * weight, const float * mean, const float * rstd,
    int B, int T, int C,
    int block_size) {

    int N = B * T;
    int grid_size = ceil_div(N, block_size);
    layernorm_backward_kernel<<<grid_size, block_size>>>(
        dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// Element-wise Ops CUDA Kernels (add / sub / mul / div)
// ============================================================

__global__ void elementwise_add_kernel(float * A, float * B, float * C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

__global__ void elementwise_sub_kernel(float * A, float * B, float * C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] - B[tid];
    }
}

__global__ void elementwise_mul_kernel(float * A, float * B, float * C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] * B[tid];
    }
}

__global__ void elementwise_div_kernel(float * A, float * B, float * C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] / B[tid];
    }
}

void elementwise_add_cuda(float * A, float * B, float * C, int N, int block_size) {
    int grid_size = ceil_div(N, block_size);
    elementwise_add_kernel<<<grid_size, block_size>>>(A, B, C, N);
    cudaCheck(cudaGetLastError());
}

void elementwise_sub_cuda(float * A, float * B, float * C, int N, int block_size) {
    int grid_size = ceil_div(N, block_size);
    elementwise_sub_kernel<<<grid_size, block_size>>>(A, B, C, N);
    cudaCheck(cudaGetLastError());
}

void elementwise_mul_cuda(float * A, float * B, float * C, int N, int block_size) {
    int grid_size = ceil_div(N, block_size);
    elementwise_mul_kernel<<<grid_size, block_size>>>(A, B, C, N);
    cudaCheck(cudaGetLastError());
}

void elementwise_div_cuda(float * A, float * B, float * C, int N, int block_size) {
    int grid_size = ceil_div(N, block_size);
    elementwise_div_kernel<<<grid_size, block_size>>>(A, B, C, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// 广播逐元素 op（模仿 ggml k_bin_bcast，按项目 CPU kernel_elemwise 语义简化）
//   dst[i] = op(a[i % an], b[i % bn])
//   - 项目广播语义：src 是 dst 去掉若干"前导最内维"后的尾部对齐后缀，
//     扁平序下 src 索引 = i % src_numel（对齐 CPU kernel_elemwise CPUKernels.cpp:447）。
//   - 与 k_bin_bcast 差异：丢弃 fast_div_modulo/fastmod、逐维取模、uint32 溢出防护
//     ——项目语义只需全局取模，int64 全程无溢出。
//   - an==n 时 i%an==i，故"全等/广播"两情形统一用取模，无需分支。
// 使用场景: OP_ADD/SUB/MUL/DIV 的 bias 广播、seq_emb 广播等（图里大量存在）。
// ============================================================
struct op_add_f { __device__ __forceinline__ float operator()(float a, float b) const { return a + b; } };
struct op_sub_f { __device__ __forceinline__ float operator()(float a, float b) const { return a - b; } };
struct op_mul_f { __device__ __forceinline__ float operator()(float a, float b) const { return a * b; } };
struct op_div_f { __device__ __forceinline__ float operator()(float a, float b) const { return a / b; } };

template <typename bin_op>
__global__ void bcast_elemwise_kernel(
    const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ dst,
    const int64_t n, const int64_t an, const int64_t bn, bin_op op) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += int64_t(gridDim.x) * blockDim.x) {
        dst[i] = op(a[i % an], b[i % bn]);
    }
}

// eop: 0=add 1=sub 2=mul 3=div
void bcast_elemwise_cuda(
    const float * a, const float * b, float * dst,
    int64_t n, int64_t an, int64_t bn, int eop) {
    if (n <= 0 || an <= 0 || bn <= 0) return;   // 防御：避免 i%0
    constexpr int BLOCK = 256;
    const int64_t grid = (n + BLOCK - 1) / BLOCK;
    switch (eop) {
        case 0: bcast_elemwise_kernel<<<(unsigned)grid, BLOCK>>>(a, b, dst, n, an, bn, op_add_f()); break;
        case 1: bcast_elemwise_kernel<<<(unsigned)grid, BLOCK>>>(a, b, dst, n, an, bn, op_sub_f()); break;
        case 2: bcast_elemwise_kernel<<<(unsigned)grid, BLOCK>>>(a, b, dst, n, an, bn, op_mul_f()); break;
        default: bcast_elemwise_kernel<<<(unsigned)grid, BLOCK>>>(a, b, dst, n, an, bn, op_div_f()); break;
    }
    cudaCheck(cudaGetLastError());
}

// ============================================================
// OP_ADD1 前向（加标量）：dst[i] = src[i] + b
// 对齐 CPU kernel_add1 (CPUKernels.cpp:1503)；b 为 src[1] 标量张量（D2H 读取）。
// 结构：一维 grid-stride。标量天然广播，无需逐维取模。
// ============================================================
__global__ void add1_f32_kernel(
    const float * __restrict__ src, float * __restrict__ dst,
    const float b, const int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += int64_t(gridDim.x) * blockDim.x) {
        dst[i] = src[i] + b;
    }
}

void add1_cuda(const float * src, float * dst, float b, int64_t n) {
    if (n <= 0) return;
    constexpr int BLOCK = 256;
    const int64_t grid = (n + BLOCK - 1) / BLOCK;
    add1_f32_kernel<<<(unsigned)grid, BLOCK>>>(src, dst, b, n);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// OP_DUP / OP_CPY / OP_CONT 整块拷贝（buffer-aware）
// 对齐 CPU kernel_cpy (CPUKernels.cpp:1385-1400) 的四分支语义：
//   src/dst 均 device → D2D；仅 src device → D2H；仅 dst device → H2D；均 host → memcpy。
// 与 ggml cpy_f32_q 的差异：本项目 Tensor 无 nb[]（连续行主序），无需逐维
//   nb 跨步解坐标；ggml 的 dequant 分支（cpy_blck_q_f32）本项目不涉及（全 F32）。
//   D2D 用 cudaMemcpy 同步（驱动最优），默认流顺序执行保证前后 kernel 依赖。
// ============================================================
// ============================================================
// OP_FAPE 前向：FAPE 结构损失（对齐 CPU compute_forward_fape CPUKernels.cpp:2395）
//   两阶段：
//     stage1 (fape_tinv_kernel)：每帧一个线程，由 3 个原子做 Gram-Schmidt 建
//       T_inv_pred/T_inv_true（各 12 floats：R^T 3 列 + 平移），存 tmp[n*24]。
//       fm==0 的帧保持单位阵（平移 0），对齐 CPU 默认初始化。
//     stage2 (fape_dist_kernel)：每 (frame,atom) 一个线程，把 pred/true 原子变换到
//       帧局部系求欧氏距离，clamp 到 d_clamp，×fm×pm，atomicAdd 归约 sum_loss/sum_fm/sum_pm。
//     stage3 (fape_finalize_kernel)：loss = sum_loss/(sum_fm*sum_pm+eps)/length_scale。
//   op_params: [0]=d_clamp [2]=epsilon [4]=length_scale（float 位模式，对齐 CPU）。
//   布局（ggml dims[0]=最内）：pred/true coords [3, N_atoms]；frame_indices [3, N_frames]；
//   frames_mask [1,N_frames]；positions_mask [1,N_atoms]。输出 dst 标量 [1]。
//   索引注意：CPU 用 src0->dims[1]=N_atoms 行数、src2->dims[1]=N_frames。
// ============================================================
__device__ __forceinline__ void fape_build_tinv(
    const float* coords, const int a, const int b, const int c, float* T,
    const float epsilon) {
    const float pAx = coords[a*3+0], pAy = coords[a*3+1], pAz = coords[a*3+2];
    const float pBx = coords[b*3+0], pBy = coords[b*3+1], pBz = coords[b*3+2];
    const float pCx = coords[c*3+0], pCy = coords[c*3+1], pCz = coords[c*3+2];
    float v1x = pBx-pAx, v1y = pBy-pAy, v1z = pBz-pAz;
    float v2x = pCx-pAx, v2y = pCy-pAy, v2z = pCz-pAz;
    float n1 = sqrtf(v1x*v1x + v1y*v1y + v1z*v1z + epsilon);
    float e1x = v1x/n1, e1y = v1y/n1, e1z = v1z/n1;
    float dot = v2x*e1x + v2y*e1y + v2z*e1z;
    float u2x = v2x - dot*e1x, u2y = v2y - dot*e1y, u2z = v2z - dot*e1z;
    float n2 = sqrtf(u2x*u2x + u2y*u2y + u2z*u2z + epsilon);
    float e2x = u2x/n2, e2y = u2y/n2, e2z = u2z/n2;
    float e3x = e1y*e2z - e1z*e2y;
    float e3y = e1z*e2x - e1x*e2z;
    float e3z = e1x*e2y - e1y*e2x;
    T[0]=e1x; T[1]=e1y; T[2]=e1z;
    T[3]=e2x; T[4]=e2y; T[5]=e2z;
    T[6]=e3x; T[7]=e3y; T[8]=e3z;
    T[9] =-(e1x*pAx + e1y*pAy + e1z*pAz);
    T[10]=-(e2x*pAx + e2y*pAy + e2z*pAz);
    T[11]=-(e3x*pAx + e3y*pAy + e3z*pAz);
}

__global__ void fape_tinv_kernel(
    const float * __restrict__ pred, const float * __restrict__ truth,
    const float * __restrict__ frame_idx, const float * __restrict__ fmask,
    float * __restrict__ tmp, const int64_t N_atoms, const int64_t N_frames,
    const float epsilon) {
    const int64_t n = blockIdx.x;
    if (n >= N_frames) return;
    float* Tp = tmp + n * 24;
    float* Tt = Tp + 12;
    // 默认单位阵 + 零平移（CPU 默认初始化）
    for (int k = 0; k < 12; k++) { Tp[k] = 0.0f; Tt[k] = 0.0f; }
    Tp[0] = Tp[4] = Tp[8] = 1.0f;
    Tt[0] = Tt[4] = Tt[8] = 1.0f;
    if (fmask[n] == 0.0f) return;

    const int ia = (int)frame_idx[n * 3 + 0];
    const int ib = (int)frame_idx[n * 3 + 1];
    const int ic = (int)frame_idx[n * 3 + 2];
    // 越界钳制（防御）
    const int a = (ia < 0 || ia >= N_atoms) ? (int)(N_atoms - 1) : ia;
    const int b = (ib < 0 || ib >= N_atoms) ? (int)(N_atoms - 1) : ib;
    const int c = (ic < 0 || ic >= N_atoms) ? (int)(N_atoms - 1) : ic;

    fape_build_tinv(pred, a, b, c, Tp, epsilon);
    fape_build_tinv(truth, a, b, c, Tt, epsilon);
}

__global__ void fape_dist_kernel(
    const float * __restrict__ pred, const float * __restrict__ truth,
    const float * __restrict__ fmask, const float * __restrict__ pmask,
    const float * __restrict__ tmp,
    float * __restrict__ sum_loss, float * __restrict__ sum_fm, float * __restrict__ sum_pm,
    const int64_t N_atoms, const int64_t N_frames,
    const float d_clamp, const float epsilon) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = N_frames * N_atoms;
    if (idx >= total) return;
    const int64_t n = idx / N_atoms;
    const int64_t j = idx % N_atoms;
    const float fm = fmask[n];
    if (fm == 0.0f) return;
    const float pm = pmask[j];
    if (pm == 0.0f) return;

    // ⚠️ 归约语义对齐 CPU：sum_pm 在每 (frame,atom) 有效对累加一次（CPU 内循环每帧累加），
    //    不能只对 n==0 帧累加（会差 N_valid_frames 倍，loss 错误放大）。
    if (j == 0) atomicAdd(sum_fm, fm);   // 每帧一次（由 j==0 线程累加，对齐 CPU 每帧累加）
    atomicAdd(sum_pm, pm);               // 每 (frame,atom) 有效对一次，对齐 CPU 内循环

    const float* Tp = tmp + n * 24;
    const float* Tt = Tp + 12;
    const float px = pred[j*3+0], py = pred[j*3+1], pz = pred[j*3+2];
    const float lpx = Tp[0]*px + Tp[1]*py + Tp[2]*pz + Tp[9];
    const float lpy = Tp[3]*px + Tp[4]*py + Tp[5]*pz + Tp[10];
    const float lpz = Tp[6]*px + Tp[7]*py + Tp[8]*pz + Tp[11];
    const float tx = truth[j*3+0], ty = truth[j*3+1], tz = truth[j*3+2];
    const float ltx = Tt[0]*tx + Tt[1]*ty + Tt[2]*tz + Tt[9];
    const float lty = Tt[3]*tx + Tt[4]*ty + Tt[5]*tz + Tt[10];
    const float ltz = Tt[6]*tx + Tt[7]*ty + Tt[8]*tz + Tt[11];
    const float dx = lpx-ltx, dy = lpy-lty, dz = lpz-ltz;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz + epsilon);
    if (dist > d_clamp) dist = d_clamp;
    atomicAdd(sum_loss, dist * fm * pm);
}

__global__ void fape_finalize_kernel(
    const float * __restrict__ sum_loss, const float * __restrict__ sum_fm,
    const float * __restrict__ sum_pm, float * __restrict__ dst,
    const float epsilon, const float length_scale) {
    const float denom = (*sum_fm) * (*sum_pm) + epsilon;
    dst[0] = (*sum_loss) / denom / length_scale;
}

// eop 约定（复用 op_params[0]=d_clamp, [2]=epsilon, [4]=length_scale）
void fape_cuda(const float* pred, const float* truth,
               const float* frame_idx, const float* fmask, const float* pmask,
               float* dst, int64_t N_atoms, int64_t N_frames,
               float d_clamp, float epsilon, float length_scale) {
    if (N_atoms <= 0 || N_frames <= 0) return;
    float *d_tmp = nullptr, *d_sl = nullptr, *d_sf = nullptr, *d_sp = nullptr;
    const size_t tmp_bytes = (size_t)N_frames * 24 * sizeof(float);
    if (cudaMalloc(&d_tmp, tmp_bytes) != cudaSuccess) return;
    if (cudaMalloc(&d_sl, sizeof(float)) != cudaSuccess) { cudaFree(d_tmp); return; }
    if (cudaMalloc(&d_sf, sizeof(float)) != cudaSuccess) { cudaFree(d_tmp); cudaFree(d_sl); return; }
    if (cudaMalloc(&d_sp, sizeof(float)) != cudaSuccess) { cudaFree(d_tmp); cudaFree(d_sl); cudaFree(d_sf); return; }
    cudaMemset(d_sl, 0, sizeof(float));
    cudaMemset(d_sf, 0, sizeof(float));
    cudaMemset(d_sp, 0, sizeof(float));

    const int64_t total = N_frames * N_atoms;
    constexpr int BLOCK = 256;
    const int64_t g1 = (N_frames + 1 - 1) / 1;
    const int64_t g2 = (total + BLOCK - 1) / BLOCK;
    fape_tinv_kernel<<<(unsigned)g1, 32>>>(pred, truth, frame_idx, fmask, d_tmp,
                                           N_atoms, N_frames, epsilon);
    fape_dist_kernel<<<(unsigned)g2, BLOCK>>>(pred, truth, fmask, pmask, d_tmp,
                                              d_sl, d_sf, d_sp, N_atoms, N_frames,
                                              d_clamp, epsilon);
    fape_finalize_kernel<<<1, 1>>>(d_sl, d_sf, d_sp, dst, epsilon, length_scale);
    cudaFree(d_tmp); cudaFree(d_sl); cudaFree(d_sf); cudaFree(d_sp);
    cudaCheck(cudaGetLastError());
}

void copy_tensor_cuda(const float * src, float * dst, int64_t n,
                      bool src_dev, bool dst_dev) {
    if (n <= 0) return;
    const size_t bytes = (size_t)n * sizeof(float);
    if (src_dev && dst_dev) {
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    } else if (src_dev) {
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    } else if (dst_dev) {
        cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    } else {
        std::memcpy(dst, src, bytes);
    }
    cudaCheck(cudaGetLastError());
}

// ============================================================
// OP_PERMUTE / OP_TRANSPOSE 前向：通用维度重排（对齐 CPU kernel_permute）
//   dst 的第 p 维来自 src 的第 dims[p] 维（op_params 存 int32 映射，行主序 dims[0]=最内）。
//   对 dst 每个连续元素 idx，反解 dst 坐标(j0..j3) → src 坐标 i_{dims[p]}=j_p
//   → src 线性偏移（行主序）写入。每线程一个 dst 元素，grid-stride。
//   参数经值传递（dims/sd/dd 各 4 个标量），避免额外 device 分配。
//   参考 ggml transposeSharedSwizzling（tile XOR swizzle 优化 2D）；本项目通用 4D
//   反解映射 + 直写（无 tile），以正确性优先（transpose 是 dims=[1,0,2,3] 特例）。
// ============================================================
__global__ void permute_f32_kernel(
    const float * __restrict__ src, float * __restrict__ dst,
    const int64_t total,
    const int ndim,
    const int32_t d0, const int32_t d1, const int32_t d2, const int32_t d3,
    const int64_t s0, const int64_t s1, const int64_t s2, const int64_t s3,
    const int64_t dd0, const int64_t dd1, const int64_t dd2, const int64_t dd3) {
    for (int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += int64_t(gridDim.x) * blockDim.x) {
        // 反解 dst 坐标 (j0..j3)，dims[0]=最内
        int64_t t = idx;
        const int64_t jv0 = t % dd0; t /= dd0;
        const int64_t jv1 = t % dd1; t /= dd1;
        const int64_t jv2 = t % dd2; t /= dd2;
        const int64_t jv3 = t;
        // src 坐标: i_{dims[p]} = j_p
        int64_t i0 = 0, i1 = 0, i2 = 0, i3 = 0;
        for (int p = 0; p < ndim && p < 4; p++) {
            const int64_t jv = (p == 0) ? jv0 : (p == 1) ? jv1 : (p == 2) ? jv2 : jv3;
            switch (p == 0 ? d0 : p == 1 ? d1 : p == 2 ? d2 : d3) {
                case 0: i0 = jv; break;
                case 1: i1 = jv; break;
                case 2: i2 = jv; break;
                case 3: i3 = jv; break;
                default: break;
            }
        }
        const int64_t off = ((i3 * s2 + i2) * s1 + i1) * s0 + i0;
        dst[idx] = src[off];
    }
}

// ============================================================
// OP_TRANSPOSE 2D 专用 kernel — tile + XOR swizzle（参考用户 transposeSharedSwizzling）
//   A: M×N（M=行数=src.dims[1]，N=行长度=src.dims[0]，dims[0]=最内）
//   B: N×M 转置输出。
//   XOR swizzle：shared tile[y][x^y]，读写两阶段均避免 bank conflict。
//   线程循环覆盖 tile（Bm×Bn），支持任意 block 尺寸/越界。
//   3D/4D transpose（交换倒数两维）由 kernel_transpose_cuda 回落通用 permute。
// ============================================================
template<int Bm, int Bn>
__global__ void transpose_tile_swizzle_kernel(
    const float * __restrict__ A, float * __restrict__ B,
    const int64_t M, const int64_t N) {
    __shared__ float tile[Bm][Bn];

    /* -------- 读取阶段 -------- */
    // (r0, c0) 表示 tile 内左上角元素在 matrixA 中的坐标
    const int64_t r0 = blockIdx.y * (int64_t)Bm;
    const int64_t c0 = blockIdx.x * (int64_t)Bn;

    // thread y 方向负责：矩阵 A 的行，shared memory 的行
    // thread x 方向负责：矩阵 A 的列，shared memory 的列
    // shared memory 中的元素 tile[y][x ^ y] = A[r0 + y, c0 + x]
#pragma unroll
    for (int y = threadIdx.y; y < Bm; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
        const int64_t r = r0 + y;
        if (r >= M) break;

#pragma unroll
        for (int x = threadIdx.x; x < Bn; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
            const int64_t c = c0 + x;
            if (c < N) {
                tile[y][x ^ y] = A[r * N + c];  // 将 A[r0 + y, c0 + x] 写入 tile[y][x ^ y]
            }
        }
    }

    __syncthreads();  // 同步线程块

    /* -------- 写入阶段 -------- */
    // (c0, r0) 表示 tile 内左上角元素在 matrixB 中的坐标
    // thread y 方向负责：矩阵 B 的行，shared memory 的列
    // thread x 方向负责：矩阵 B 的列，shared memory 的行
    // shared memory 中的元素 tile[x][x ^ y] = B[c0 + y, r0 + x]
#pragma unroll
    for (int y = threadIdx.y; y < Bn; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
        const int64_t c = c0 + y;
        if (c >= N) break;

#pragma unroll
        for (int x = threadIdx.x; x < Bm; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
            const int64_t r = r0 + x;
            if (r < M) { B[c * M + r] = tile[x][x ^ y]; }  // 将 tile[x][x ^ y] 写入 B[c0 + y, r0 + x]
        }
    }
}

void transpose_cuda(const float * src, float * dst, int64_t M, int64_t N) {
    if (M <= 0 || N <= 0) return;
    constexpr int Bm = 32, Bn = 32;
    dim3 block(16, 16);
    dim3 grid((unsigned)((N + Bn - 1) / Bn), (unsigned)((M + Bm - 1) / Bm));
    transpose_tile_swizzle_kernel<Bm, Bn><<<grid, block>>>(src, dst, M, N);
    cudaCheck(cudaGetLastError());
}

void permute_cuda(const float * src, float * dst, int64_t total, int ndim,
                  int32_t d0, int32_t d1, int32_t d2, int32_t d3,
                  int64_t s0, int64_t s1, int64_t s2, int64_t s3,
                  int64_t dd0, int64_t dd1, int64_t dd2, int64_t dd3) {
    if (total <= 0) return;
    constexpr int BLOCK = 256;
    const int64_t grid = (total + BLOCK - 1) / BLOCK;
    permute_f32_kernel<<<(unsigned)grid, BLOCK>>>(
        src, dst, total, ndim,
        d0, d1, d2, d3,
        s0, s1, s2, s3,
        dd0, dd1, dd2, dd3);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// Softmax CUDA Kernel (一个 Block 处理一行)
// ============================================================

__global__ void softmax_v1_kernel(float * input, float * output, int M, int N) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float * x = input  + row * N;
    float * y = output + row * N;

    // Pass 1: 并行求最大值
    float max_val = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, x[i]);
    }
    smem[tid] = max_val;
    __syncthreads();

    // Shared Memory 规约求全局最大值
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    // Pass 2: 并行求指数和
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += expf(x[i] - max_val);
    }
    smem[tid] = sum;
    __syncthreads();

    // Shared Memory 规约求总和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    sum = smem[0];
    __syncthreads();

    // Pass 3: 归一化写出
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}

void softmax_cuda(float * input, float * output, int M, int N, int block_size) {
    int smem_size = block_size * sizeof(float);
    softmax_v1_kernel<<<M, block_size, smem_size>>>(input, output, M, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// Softmax V2 CUDA Kernel (Warp Shuffle 两级规约)
// ============================================================

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float blockReduceMaxShuffle(float val, float * smem) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    // 第一步：Warp 内规约
    val = warpReduceMax(val);

    // 第二步：各 warp 的 lane 0 写入共享内存
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    // 第三步：Warp 0 从共享内存读取所有 warp 结果，再做一次 warp 内规约
    int num_warps = blockDim.x / 32;
    val = (lane < num_warps) ? smem[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);

    return val;
}

__device__ float blockReduceSumShuffle(float val, float * smem) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    // 第一步：Warp 内规约
    val = warpReduceSum(val);

    // 第二步：各 warp 的 lane 0 写入共享内存
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    // 第三步：Warp 0 从共享内存读取所有 warp 结果，再做一次 warp 内规约
    int num_warps = blockDim.x / 32;
    val = (lane < num_warps) ? smem[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}

__global__ void softmax_v2_kernel(float * input, float * output, int M, int N) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float * x = input  + row * N;
    float * y = output + row * N;

    // Pass 1: 并行求行最大值
    float max_val = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, x[i]);
    }
    max_val = blockReduceMaxShuffle(max_val, smem);

    // Pass 2: 并行求指数和
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += expf(x[i] - max_val);
    }
    sum = blockReduceSumShuffle(sum, smem);

    // Pass 3: 归一化写出
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}

void softmax_v2_cuda(float * input, float * output, int M, int N, int block_size) {
    // smem 只需存储 num_warps 个 float（每 warp 一个）
    int num_warps = block_size / 32;
    int smem_size = num_warps * sizeof(float);
    softmax_v2_kernel<<<M, block_size, smem_size>>>(input, output, M, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// Softmax Backward CUDA Kernel (Warp Shuffle 两级规约)
// ============================================================

__global__ void softmax_backward_kernel(
    const float * grad, const float * output, float * dst, int M, int N, float scale) {

    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    grad   += int64_t(row) * N;
    output += int64_t(row) * N;
    dst    += int64_t(row) * N;

    // Step 1: 计算 sum(y_j * dL/dy_j) — 梯度在概率上的加权和
    float dgf_dot = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        dgf_dot += output[col] * grad[col];
    }
    dgf_dot = blockReduceSumShuffle(dgf_dot, smem);

    // Step 2: dL/dx_i = scale * (grad[i] - dgf_dot) * output[i]
    for (int col = tid; col < N; col += blockDim.x) {
        dst[col] = scale * (grad[col] - dgf_dot) * output[col];
    }
}

void softmax_backward_cuda(
    const float * grad, const float * output, float * dst, int M, int N, int block_size, float scale) {

    int num_warps = block_size / 32;
    int smem_size = num_warps * sizeof(float);
    softmax_backward_kernel<<<M, block_size, smem_size>>>(grad, output, dst, M, N, scale);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// Block-Tiled GEMM CUDA Kernel (OP_MUL_MAT)
// D[M×N] = A[M×K] × B[K×N]  (B is stored transposed: B[N][K])
// ============================================================

template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
         int B_BLOCK_X = 32, int C_BLOCK_X = 16>
__global__ void blockTileGEMM(float* A, float* B, float* C, const int M, const int K, const int N) {
  __shared__ float As[Bm][Bk];  // tileA
  __shared__ float Bs[Bk][Bn];  // tileB

  // tileC 左上角
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  int tid = threadIdx.x;

  /*------ tileA ------*/
  constexpr int A_BLOCK_Y = blockSize / A_BLOCK_X;
  int A_THREAD_Y = tid / A_BLOCK_X;
  int A_THREAD_X = tid % A_BLOCK_X;

  /*------ tileB ------*/
  constexpr int B_BLOCK_Y = blockSize / B_BLOCK_X;
  int B_THREAD_Y = tid / B_BLOCK_X;
  int B_THREAD_X = tid % B_BLOCK_X;

  /*------ tileC ------*/
  constexpr int C_BLOCK_Y = blockSize / C_BLOCK_X;
  int C_THREAD_Y = tid / C_BLOCK_X;
  int C_THREAD_X = tid % C_BLOCK_X;

  constexpr int Tm = Bm / C_BLOCK_Y;
  constexpr int Tn = Bn / C_BLOCK_X;
  float Ct[Tm][Tn] = {0.0};

  // K- Loop
  for (int k = 0; k < K; k += Bk) {
    /* ------ 加载 As ------ */
#pragma unroll
    for (int i = A_THREAD_Y; i < Bm; i += A_BLOCK_Y) {
      int r = r0 + i;
#pragma unroll
      for (int j = A_THREAD_X; j < Bk; j += A_BLOCK_X) {
        int c = k + j;
        As[i][j] = (r < M && c < K) ? A[r * K + c] : 0.f;
      }
    }

    /* ------ 加载 Bs: B is stored as B[N][K] (transposed) ------ */
#pragma unroll
    for (int i = B_THREAD_Y; i < Bk; i += B_BLOCK_Y) {
      int r = k + i;  // K dimension index
#pragma unroll
      for (int j = B_THREAD_X; j < Bn; j += B_BLOCK_X) {
        int c = c0 + j;  // N dimension index
        // B is stored transposed: B[c][r] = B[c * K + r]
        Bs[i][j] = (r < K && c < N) ? B[c * K + r] : 0.f;
      }
    }

    __syncthreads();

    /* ------ 计算 tileA * tileB ------ */
#pragma unroll
    for (int p = 0; p < Bk; ++p) {
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
        int r = C_THREAD_Y + i * C_BLOCK_Y;
#pragma unroll
        for (int j = 0; j < Tn; ++j) {
          int c = C_THREAD_X + j * C_BLOCK_X;
          Ct[i][j] += As[r][p] * Bs[p][c];
        }
      }
    }

    __syncthreads();
  }

  /* ------ 写回 C ------ */
#pragma unroll
  for (int i = 0; i < Tm; ++i) {
    int r = r0 + C_THREAD_Y + i * C_BLOCK_Y;
#pragma unroll
    for (int j = 0; j < Tn; ++j) {
      int c = c0 + C_THREAD_X + j * C_BLOCK_X;
      if (r < M && c < N) { C[r * N + c] = Ct[i][j]; }
    }
  }
}

void mul_mat_cuda(float* A, float* B, float* C, int M, int K, int N) {
    // 使用 blockTileGEMM kernel，blockSize=256, shared memory ~8KB
    constexpr int Bm = 128, Bn = 128;
    dim3 block(256);
    dim3 grid((N + Bn - 1) / Bn, (M + Bm - 1) / Bm);

    blockTileGEMM<Bm, Bn><<<grid, block>>>(A, B, C, M, K, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// CONCAT CUDA Kernel (OP_CONCAT) — N-ary 版本
// 与 CPU kernel_concat 对齐：支持任意数量 src，沿 op_params[0]=dim 拼接。
// 数据连续，故将多维索引退化为 1D 线性索引反解。
// 通过 device 侧「src 指针数组 + start/len 偏移表」定位每个输出元素所属的 src。
//
// 重要前提：concat 不支持广播语义。调用方必须保证所有 src 的
// 非拼接维与 dst 完全一致（即 dst 任意非拼接维不得大于 src 对应维）。
// 本 kernel 复用 dst 的 ne0..ne3 作为 src 的 strides 基数（标准 concat 语义）。
// 若 dst 某非拼接维 > src 对应维，则会产生越界/错位读，属于非法输入。
// ============================================================

// 反解 idx 得到 (i0,i1,i2,i3)，i0 为最内维
__device__ static inline void concat_decompose_index(
    int64_t idx,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    int64_t& i0, int64_t& i1, int64_t& i2, int64_t& i3)
{
    int64_t t = idx;
    i0 = t % ne0; t /= ne0;
    i1 = t % ne1; t /= ne1;
    i2 = t % ne2; t /= ne2;
    i3 = t;
}

// 用 start/len 偏移表，将拼接维全局索引 gd 定位到所属 src 下标
__device__ static inline int concat_pick_src(
    int64_t gd, int n_src,
    const int64_t* __restrict__ start,
    const int64_t* __restrict__ len)
{
    for (int s = 0; s < n_src; s++) {
        if (gd < start[s] + len[s]) return s;
    }
    return n_src - 1; // 兜底（gd 落在最后一个 src 区间）
}

__global__ void concat_nary_kernel_f32(
    const float* const* __restrict__ srcs, // device 指针数组 [n_src]
    const int64_t*     __restrict__ start, // device 偏移表 [n_src]
    const int64_t*     __restrict__ len,   // device 长度表 [n_src]
    float* __restrict__ dst,
    int dim, int n_src,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)
{
    const int64_t total = ne0 * ne1 * ne2 * ne3;
    const int64_t idx   = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t i0, i1, i2, i3;
    concat_decompose_index(idx, ne0, ne1, ne2, ne3, i0, i1, i2, i3);

    // 取出拼接维上的全局索引 gd
    int64_t gd;
    switch (dim) {
        case 0:  gd = i0; break;
        case 1:  gd = i1; break;
        case 2:  gd = i2; break;
        default: gd = i3; break;
    }

    const int  s     = concat_pick_src(gd, n_src, start, len);
    const int64_t local = gd - start[s];
    const float* src   = srcs[s];

    // 拼接维用局部坐标 local，其余维沿用输出坐标。
    // 前提（无广播语义）：非拼接维与 dst 一致，故直接用 dst 的 ne 作 strides。
    int64_t a0 = (dim == 0) ? local : i0;
    int64_t a1 = (dim == 1) ? local : i1;
    int64_t a2 = (dim == 2) ? local : i2;
    int64_t a3 = (dim == 3) ? local : i3;

    dst[idx] = src[((a3 * ne2 + a2) * ne1 + a1) * ne0 + a0];
}

// host 包装：srcs/start/len 均须为已拷贝到 device 的指针
void concat_nary_cuda(
    const float* const* srcs, int n_src,
    const int64_t* start, const int64_t* len,
    float* dst, int dim,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)
{
    const int64_t total = ne0 * ne1 * ne2 * ne3;
    constexpr int BLOCK = 256;
    const int grid = ceil_div(static_cast<int>(total), BLOCK);

    concat_nary_kernel_f32<<<grid, BLOCK>>>(
        srcs, start, len, dst, dim, n_src, ne0, ne1, ne2, ne3);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// Unary Ops CUDA Kernel（RELU/SQRT/EXP 等常用激活，2026-08-23 提速）
// uop 与 unary_op 枚举对齐（ComputeGraph.h）：0=ABS,4=RELU,10=SQRT,13=EXP,5=GELU,...
// ============================================================

__global__ void unary_kernel(
    const float * __restrict__ src, float * __restrict__ dst, int N, int uop) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float x = src[tid];
    float y;
    //    0=ABS, 4=RELU, 5=GELU, 7=SILU, 8=TANH, 10=SIGMOID, 12=HARDSWISH,
    //    13=EXP, 14=LOG, 15=SQRT。之前把 10 当 SQRT（实为 SIGMOID）→ sqrt(负)→NaN。
    switch (uop) {
        case 0:  y = fabsf(x);        break;   // ABS
        case 4:  y = fmaxf(x, 0.0f);  break;   // RELU
        case 5:  y = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); break;  // GELU
        case 7:  y = x / (1.0f + expf(-x)); break;  // SILU
        case 8:  y = tanhf(x);        break;   // TANH
        case 10: y = 1.0f / (1.0f + expf(-x)); break;  // SIGMOID
        case 12: y = x * fmaxf(0.0f, fminf(1.0f, x / 6.0f + 0.5f)); break;  // HARDSWISH
        case 13: y = expf(x);         break;   // EXP
        case 14: y = logf(x);         break;   // LOG
        case 15: y = sqrtf(x);        break;   // SQRT
        default: y = x;               break;
    }
    dst[tid] = y;
}

void unary_cuda(const float * src, float * dst, int N, int uop, int block_size) {
    int grid_size = ceil_div(N, block_size);
    unary_kernel<<<grid_size, block_size>>>(src, dst, N, uop);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// OP_SUM / OP_MEAN: 全元素归约 → 标量 [1]
// 两级规约：Warp shuffle（用户参考实现）+ grid-stride + block 间 atomicAdd。
// 二者共用同一 reduce kernel，区别仅在最终标量乘 scale：
//   sum  → scale=1.0f；mean → scale=1/N（数学上 Σ(val_b/N) = Σ(val_b)/N）。
// ============================================================

__device__ float sum_warp_reduce(float val) {
    // 每次将右半边的值加到左半边（全 warp 同步）
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;  // lane 0 持有最终结果
}

// 每个 block 处理一段元素：grid-stride 累积到 val，warp 内规约，shared 归并，block 结果 atomicAdd 到 dst
__global__ void sum_reduce_kernel(const float * __restrict__ src, float * __restrict__ dst,
                                  long long n, int block_size, float scale) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    float val = 0.0f;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        val += src[i];
    }

    const int lane = threadIdx.x % 32;
    const int wid  = threadIdx.x / 32;
    // 第一级：Warp 内规约
    val = sum_warp_reduce(val);

    __shared__ float warp_results[32];  // block_size<=1024 → 最多 32 warp
    if (lane == 0) warp_results[wid] = val;
    __syncthreads();

    // 第二级：Warp 0 归并各 warp 结果
    const int num_warps = block_size / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        val = sum_warp_reduce(val);
        if (lane == 0) atomicAdd(dst, val * scale);
    }
}

// 内部实现：dst 须已清零（由 kernel_*_cuda 负责 memset device 0）
static void sum_impl(const float * src, float * dst, long long n, float scale) {
    if (n <= 0) return;
    constexpr int BLOCK = 256;
    // 每线程至少处理 1 个元素，block 数上限 ~4096 防 launch 超限
    long long want_blocks = (n + BLOCK - 1) / BLOCK;
    if (want_blocks > 4096) want_blocks = 4096;
    const int grid = (int)want_blocks;
    sum_reduce_kernel<<<grid, BLOCK>>>(src, dst, n, BLOCK, scale);
    cudaCheck(cudaGetLastError());
}

void sum_cuda(const float * src, float * dst, long long n) {
    sum_impl(src, dst, n, 1.0f);
}

void mean_cuda(const float * src, float * dst, long long n) {
    // 空数组保护：mean=0（对齐 CPU kernel_mean 语义）
    const float inv_n = (n > 0) ? (1.0f / (float)n) : 0.0f;
    sum_impl(src, dst, n, inv_n);
}

// ============================================================
// OP_MAX_ALL: 全元素归约 max → 标量 [1]（参考 sum/mean 的 reduce 结构）
// 两级规约：grid-stride + warp shuffle（fmaxf）+ shared 归并 + block 间 float atomicMax。
// NaN 语义：CPU kernel_max_all 跳 NaN（m 初值 -INF，!isnan(v)&&v>m 才更新）。
//   CUDA fmaxf 规则 fmaxf(x, NaN)=x → 用 fmaxf(val, src[i]) 累积时 NaN 自动被忽略，
//   与 CPU 一致；全 NaN → 保持 -INF（对齐 CPU 的 n>0 全 NaN → -INF）。空输入 → 0.0。
// 使用场景: SE3 edge_softmax 的 max 减稳（避免 exp 溢出）。
// ============================================================

// float 无原子 max，用 CAS 循环（block 值已 fmaxf 归约，传入 value 非 NaN）
__device__ float atomic_max_f(float* addr, float value) {
    int* a = reinterpret_cast<int*>(addr);
    int old = *a, assumed;
    do {
        assumed = old;
        if (value <= __int_as_float(assumed)) break;   // value 不更大（含相等）则无需更新
        old = atomicCAS(a, assumed, __float_as_int(value));
    } while (old != assumed);
    return __int_as_float(old);
}

__device__ float max_warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
    }
    return val;  // lane 0 持有最终结果
}

// 每个 block 处理一段元素：grid-stride 累积 max，warp 内规约，shared 归并，block 结果 atomic_max 到 dst
__global__ void max_reduce_kernel(const float * __restrict__ src, float * __restrict__ dst,
                                  long long n, int block_size) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    float val = -INFINITY;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        val = fmaxf(val, src[i]);   // fmaxf(x,NaN)=x → NaN 被忽略，对齐 CPU 跳 NaN
    }

    const int lane = threadIdx.x % 32;
    const int wid  = threadIdx.x / 32;
    // 第一级：Warp 内规约
    val = max_warp_reduce(val);

    __shared__ float warp_results[32];  // block_size<=1024 → 最多 32 warp
    if (lane == 0) warp_results[wid] = val;
    __syncthreads();

    // 第二级：Warp 0 归并各 warp 结果
    const int num_warps = block_size / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : -INFINITY;
        val = max_warp_reduce(val);
        if (lane == 0) atomic_max_f(dst, val);
    }
}

void max_all_cuda(const float * src, float * dst, long long n) {
    if (n <= 0) {
        const float zero = 0.0f;   // 对齐 CPU: 空输入 → 0.0
        cudaMemcpy(dst, &zero, sizeof(float), cudaMemcpyHostToDevice);
        return;
    }
    constexpr int BLOCK = 256;
    // 每线程至少处理 1 个元素，block 数上限 ~4096 防 launch 超限（同 sum_impl）
    long long want_blocks = (n + BLOCK - 1) / BLOCK;
    if (want_blocks > 4096) want_blocks = 4096;
    // 预置 -INF（不能用 cudaMemset 0，float max 的初始值须是 -INF）
    const float neg_inf = -INFINITY;
    cudaMemcpy(dst, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    max_reduce_kernel<<<(int)want_blocks, BLOCK>>>(src, dst, n, BLOCK);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// OP_SUM_ROWS: 沿最内维 dims[0] 归约，保留其余维（输出 {1, dims[1..3]}）
// 对齐 CPU kernel_sum_rows (CPUKernels.cpp:1536)：每个输出元素 = 一行连续 ne0 个元素的和。
// 参考 ggml reduce_rows_f32（ncols=ne[0] 最内维、每行一个 block），简化掉：
//   8 路 unroll（性能优化非必需）、PDL、block 大小启发式（固定 256）。
// block-per-row：grid = nrows（= ne1*ne2*ne3），block 内线程沿 ncols grid-stride
// 累加 → warp 归约（sum_warp_reduce 复用 OP_SUM 段）→ shared 归并 → lane0 写 dst[row]。
// norm=false（sum）；mean 由上层 scale(1/N) 实现。
// 反向为 repeat(grad, src0) 广播回原形状（ComputeGraph.cpp:471）。
// ============================================================
__global__ void sum_rows_kernel(
    const float * __restrict__ src, float * __restrict__ dst,
    const int64_t ncols, const int64_t nrows) {
    const int64_t row = blockIdx.x;   // 每 block 一行（grid = nrows）
    if (row >= nrows) return;
    const float * r = src + row * ncols;
    float val = 0.0f;
    for (int64_t i = threadIdx.x; i < ncols; i += blockDim.x) {
        val += r[i];
    }

    const int lane = threadIdx.x % 32;
    const int wid  = threadIdx.x / 32;
    val = sum_warp_reduce(val);

    __shared__ float warp_results[32];  // block_size<=1024 → 最多 32 warp
    if (lane == 0) warp_results[wid] = val;
    __syncthreads();

    const int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        val = sum_warp_reduce(val);
        if (lane == 0) dst[row] = val;
    }
}

void sum_rows_cuda(const float * src, float * dst, int64_t ncols, int64_t nrows) {
    if (ncols <= 0 || nrows <= 0) return;
    constexpr int BLOCK = 256;
    // gridDim.x 上限 2^31-1；防御性 cap（正常模型不可能达到）
    const int64_t grid = (nrows > 0x7fffffff) ? 0x7fffffff : nrows;
    sum_rows_kernel<<<(unsigned)grid, BLOCK>>>(src, dst, ncols, nrows);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// OP_RELU_BACK 反向（relu 的梯度）：dst[i] = (x[i] > 0.0f) ? grad[i] : 0.0f
// 对齐 CPU kernel_relu_back (CPUKernels.cpp:1616)：
//   d(relu(x))/dx = step(x)，链式法则 → grad * 1[x>0]。
//   x<=0（含 x=0 边界）梯度置 0（PyTorch 同约定）。
// ggml 无专门 relu_back op，用 grad * step(x) 组合；本项目用专用 op。
// 结构：一维 grid-stride，每线程一个元素。src[0]=grad, src[1]=x。
// ============================================================
__global__ void relu_back_f32_kernel(
    const float * __restrict__ grad, const float * __restrict__ x,
    float * __restrict__ dst, const int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += int64_t(gridDim.x) * blockDim.x) {
        dst[i] = (x[i] > 0.0f) ? grad[i] : 0.0f;
    }
}

void relu_back_cuda(const float * grad, const float * x, float * dst, int64_t n) {
    if (n <= 0) return;
    constexpr int BLOCK = 256;
    const int64_t grid = (n + BLOCK - 1) / BLOCK;
    relu_back_f32_kernel<<<(unsigned)grid, BLOCK>>>(grad, x, dst, n);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// OP_REPEAT_BACK: 归约（repeat 的逆操作，梯度和）
// 语义（对齐 CPU kernel_repeat_back，仅支持同 ndim，即 src 与 dst 尾部对齐且
//       src 各维 = dst 各维 × 整数重复因子）:
//   dst[j] = Σ_{k} src[j + k*dd] ，其中对每个维度 d:
//     - dd[d] = dst 该维大小，ne0d = src 该维大小，重复因子 rd[d] = ne0d / dd[d]
//     - k 遍历 [0, rd[d])。
// 结构: 每线程一个 dst 元素，单线程串行累加所有重复拷贝（原结构，无 warp reduce）。
// 越界防护: 原版只检查 tid0>=ne0 就 return，tid1/tid2/tid3 越界线程会越界写 dst；
//          本版改为 3D grid-stride 全覆盖：tid23 在 kernel 内按 gridDim.z*blockDim.z
//          步进遍历，tid0/tid1 越界 return，tid2/tid3 由 tid23<ne2*ne3 保证合法。
// 使用场景: OP_ADD/OP_MUL/OP_REPEAT 反向中 repeat_back(grad, src)。
// ============================================================
template <typename T>
__global__ void repeat_back_kernel(
    const T * __restrict__ src, T * __restrict__ dst,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t ne0,  const int64_t ne1,  const int64_t ne2,  const int64_t ne3) {
    const int64_t tid0  = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t tid1  = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
    // tid23 覆盖 [0, ne2*ne3)，grid-stride 遍历防 gridDim.z 超限（原版只算一次）
    const int64_t tid23_base   = int64_t(blockIdx.z) * blockDim.z + threadIdx.z;
    const int64_t tid23_stride = int64_t(gridDim.z) * blockDim.z;
    // 越界防护：tid0/tid1 越界直接退出（原版只查 tid0）
    if (tid0 >= ne0 || tid1 >= ne1) return;
    for (int64_t tid23 = tid23_base; tid23 < ne2 * ne3; tid23 += tid23_stride) {
        const int64_t tid2 = tid23 % ne2;
        const int64_t tid3 = tid23 / ne2;
        T sum = 0;
        for (int64_t i3 = tid3; i3 < ne03; i3 += ne3) {
            for (int64_t i2 = tid2; i2 < ne02; i2 += ne2) {
                for (int64_t i1 = tid1; i1 < ne01; i1 += ne1) {
                    for (int64_t i0 = tid0; i0 < ne00; i0 += ne0) {
                        sum += src[((i3 * ne02 + i2) * ne01 + i1) * ne00 + i0];
                    }
                }
            }
        }
        dst[(tid3 * ne2 + tid2) * ne1 * ne0 + tid1 * ne0 + tid0] = sum;
    }
}

void repeat_back_cuda(
    const float * src, float * dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne0,  int64_t ne1,  int64_t ne2,  int64_t ne3) {
    if (ne00 <= 0 || ne0 <= 0) return;
    if (ne01 <= 0) ne01 = 1;
    if (ne02 <= 0) ne02 = 1;
    if (ne03 <= 0) ne03 = 1;
    if (ne1  <= 0) ne1  = 1;
    if (ne2  <= 0) ne2  = 1;
    if (ne3  <= 0) ne3  = 1;

    // block 总线程数须 ≤ 1024 (32*8*4 = 1024)。曾用 BLOCK_Z=32 使 32*8*32=8192 超限 → launch 静默失败。
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 8;
    constexpr int BLOCK_Z = 4;

    const int64_t grid_x = (ne0 + BLOCK_X - 1) / BLOCK_X;
    const int64_t grid_y = (ne1 + BLOCK_Y - 1) / BLOCK_Y;
    const int64_t grid_z = (ne2 * ne3 + BLOCK_Z - 1) / BLOCK_Z;

    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    // gridDim.z 硬件上限 65535，超出部分由 kernel 内 tid23 grid-stride 兜底
    const int64_t gz = (grid_z > 65535) ? 65535 : grid_z;
    dim3 grid((unsigned)grid_x, (unsigned)grid_y, (unsigned)gz);
    repeat_back_kernel<float><<<grid, block>>>(
        src, dst,
        ne00, ne01, ne02, ne03,
        ne0,  ne1,  ne2,  ne3);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// OP_REPEAT 前向（简化版）：对齐 CPU kernel_repeat (CPUKernels.cpp:1626)
//   dst[j] = src[s]，其中对每个维 d: s_d = j_d % ne0_d
//   （src 缺维/维=1 时取模得 0，天然处理跨 ndim 广播）
// 结构：一维 grid-stride，每线程一个 dst 元素。
// 与 ggml k_bin_bcast 的差异：本项目 repeat 的 dst 各维必为 src 各维整数倍，
// 无需 bcast 分支/标量特判/fastdiv 优化，直接逐元素映射即可（int64 无溢出）。
// 使用场景: 掩码广播 [1,1,N,1]→[D,L,N,B]、seq_emb 广播、BN gamma/beta 广播等。
// ============================================================
template <typename T>
__global__ void repeat_f32_kernel(
    const T * __restrict__ src, T * __restrict__ dst,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t ne0,  const int64_t ne1,  const int64_t ne2,  const int64_t ne3) {
    const int64_t total = ne0 * ne1 * ne2 * ne3;
    for (int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += int64_t(gridDim.x) * blockDim.x) {
        int64_t t = idx;
        const int64_t i3 = t % ne3; t /= ne3;
        const int64_t i2 = t % ne2; t /= ne2;
        const int64_t i1 = t % ne1; t /= ne1;
        const int64_t i0 = t;
        const int64_t s0 = i0 % ne00;
        const int64_t s1 = i1 % ne01;
        const int64_t s2 = i2 % ne02;
        const int64_t s3 = i3 % ne03;
        dst[idx] = src[((s3 * ne02 + s2) * ne01 + s1) * ne00 + s0];
    }
}

void repeat_cuda(
    const float * src, float * dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne0,  int64_t ne1,  int64_t ne2,  int64_t ne3) {
    if (ne00 <= 0 || ne01 <= 0 || ne02 <= 0 || ne03 <= 0) return;
    if (ne0  <= 0 || ne1  <= 0 || ne2  <= 0 || ne3  <= 0) return;
    const int64_t total = ne0 * ne1 * ne2 * ne3;
    constexpr int BLOCK = 256;
    const int64_t grid = (total + BLOCK - 1) / BLOCK;
    // gridDim.x 上限 2^31-1；防御性 cap（正常模型不可能达到）
    const int64_t g = (grid > 0x7fffffff) ? 0x7fffffff : grid;
    repeat_f32_kernel<float><<<(unsigned)g, BLOCK>>>(
        src, dst,
        ne00, ne01, ne02, ne03,
        ne0,  ne1,  ne2,  ne3);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// OP_SET_ROWS 前向（散点覆写，简化版）：对齐 CPU kernel_set_rows (CPUKernels.cpp:1873)
// 语义: set_rows(a, b, c) → dst = a 全量拷贝（保留未覆盖行），再把 c[k] 覆写到
//       b[k] 指定行（越界钳制）。a/c/dst 严格 2D(N,M)，b 为 (K,) float 编码索引。
// 与 ggml k_set_rows 的差异：ggml 是"行重排/收集"（src0 与 dst 同形状 + rows 张量
// 外层维取模广播 + fast_div_modulo + PDL 异步就绪），本项目为"散点覆写"，只需：
//   1) 拷贝 kernel: dst[i] = a[i]                        （N*M 全量）
//   2) 散点 kernel: dst[clamp(b[k])*M + m] = c[k*M + m]   （k,m 网格）
// 重复索引语义：CPU 版预扫 b 检测重复 → 有重复则跳过散点（仅保留 a 拷贝）。
// 重复检测放宿主侧（K 小，D2H 代价可忽略），保证与 CPU 完全一致。
// ============================================================
__global__ void set_rows_copy_kernel(
    const float * __restrict__ a, float * __restrict__ dst,
    const int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += int64_t(gridDim.x) * blockDim.x) {
        dst[i] = a[i];
    }
}

__global__ void set_rows_scatter_kernel(
    const float * __restrict__ c, const float * __restrict__ b,
    float * __restrict__ dst,
    const int64_t K, const int64_t M, const int64_t N) {
    const int64_t total = K * M;
    for (int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += int64_t(gridDim.x) * blockDim.x) {
        const int64_t k = idx / M;
        const int64_t m = idx % M;
        int64_t row = (int64_t)b[k];
        if (row < 0) row = 0;
        if (row >= N) row = N - 1;
        dst[row * M + m] = c[idx];
    }
}

// ============================================================
// OP_SCALE 前向（标量乘）：dst[i] = scale * x[i]
// 对齐 CPU kernel_scale (CPUKernels.cpp:1482)；s 存 op_params[0] float 位模式。
// 参考 ggml scale_f32（含 bias 参数 + PDL）；本项目 scale() helper 无 bias
// （仅标量乘），故省略 bias，PDL 由默认流顺序执行保证（无需）。
// 结构：一维 grid-stride，每线程多个元素。
// ============================================================
__global__ void scale_f32_kernel(
    const float * __restrict__ x, float * __restrict__ dst,
    const float scale, const int64_t nelements) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < nelements; i += int64_t(gridDim.x) * blockDim.x) {
        dst[i] = scale * x[i];
    }
}

void scale_cuda(const float * x, float * dst, float scale, int64_t nelements) {
    if (nelements <= 0) return;
    constexpr int BLOCK = 256;
    const int64_t grid = (nelements + BLOCK - 1) / BLOCK;
    // gridDim.x 上限 2^31-1；防御性 cap
    const int64_t g = (grid > 0x7fffffff) ? 0x7fffffff : grid;
    scale_f32_kernel<<<(unsigned)g, BLOCK>>>(x, dst, scale, nelements);
    cudaCheck(cudaGetLastError());
}

void set_rows_cuda(
    const float * a, const float * b, const float * c, float * dst,
    int64_t N, int64_t M, int64_t K) {
    if (N <= 0 || M <= 0) return;
    const int64_t n_total = N * M;
    constexpr int BLOCK = 256;
    const int64_t g1 = (n_total + BLOCK - 1) / BLOCK;
    set_rows_copy_kernel<<<(unsigned)g1, BLOCK>>>(a, dst, n_total);

    if (K <= 0) { cudaCheck(cudaGetLastError()); return; }  // 无索引：仅保留 a 拷贝

    // 宿主预扫重复索引（对齐 CPU：越界索引不参与查重）
    std::vector<float> b_host((size_t)K);
    cudaMemcpy(b_host.data(), b, K * sizeof(float), cudaMemcpyDeviceToHost);  // 同步
    std::vector<uint8_t> seen((size_t)N, 0);
    bool dup = false;
    for (int64_t k = 0; k < K && !dup; ++k) {
        const int64_t i1 = (int64_t)b_host[(size_t)k];
        if (i1 < 0 || i1 >= N) continue;
        if (seen[(size_t)i1]) dup = true;
        seen[(size_t)i1] = 1;
    }

    if (!dup) {
        const int64_t s_total = K * M;
        const int64_t g2 = (s_total + BLOCK - 1) / BLOCK;
        set_rows_scatter_kernel<<<(unsigned)g2, BLOCK>>>(c, b, dst, K, M, N);
    }
    cudaCheck(cudaGetLastError());
}

} // namespace ppml

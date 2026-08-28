#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace ppml {

// ============================================================
// Helper: ceil_div
// ============================================================
static inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

static inline void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        // TODO: proper error handling
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
    // ⚠️ uop 必须与 unary_op 枚举精确对齐（ComputeGraph.h）：
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

} // namespace ppml

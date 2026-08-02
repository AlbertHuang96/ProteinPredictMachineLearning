#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace rfaa {

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

} // namespace rfaa

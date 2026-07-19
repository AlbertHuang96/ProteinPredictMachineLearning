#include <cuda_runtime.h>
#include <cmath>

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

} // namespace rfaa

#include <cuda_runtime.h>
#include <cuda_fp16.h>   // __half / __half2float（mul_mat fp16 输入路径）
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
// Tensor Core / MMA 硬件能力检测
//   - compute capability >= 8.0 (Ampere+) : mma.sync m16n8k8 支持 TF32 与 F16
//   - compute capability >= 7.0 (Volta+)   : mma.sync fp16 (sm_70/75)
//   - CUTLASS tensor-core 路径需要 sm_80+（TF32）或 sm_75+（fp16）
//   RTX 2050 = sm_86 → TF32 MMA 可用（支持 CUTLASS SM80_16x8x8_*）。
// ============================================================
bool cutlass_hw_supported(int* out_major, int* out_minor) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
    if (out_major) *out_major = prop.major;
    if (out_minor) *out_minor = prop.minor;
    return prop.major >= 8;   // sm_80+：TF32/F16 tensor core MMA
}

// ============================================================
// TF32 tensor-core GEMM kernel（自包含，无 CUTLASS 依赖）
//   C[M,N] = A[M,K] * B[N,K]ᵀ（B 按 N×K 行主序传入，即 B[N][K]）
//   对应 CUTLASS 模板 SM80_16x8x8_F32TF32TF32F32_TN（fp32 精度 tensor core）。
//   用户模板的 F16F16F16F16 是 fp16 变体（A/B/C/D 全 fp16，精度低）；
//   TF32 是「fp32 精度」GEMM 的 tensor core 路径（A/B 输入 fp32 硬件截断
//   到 19bit tf32，累加 fp32）。RTX 2050 = sm_86 支持。
//   每 warp 独立算一个 16×8 C 子块 + 沿 K 循环 step 8；grid 覆盖 M×N。
//   fragment 布局（PTX mma.m16n8k8 tf32 官方表，row.col）：
//     A(16×8): a0=(g,t) a1=(g,t+4) a2=(g+8,t) a3=(g+8,t+4)，g=lane>>2, t=lane&3
//     B(8×8) : b0=(t,g) b1=(t+4,g)
//     C(16×8): c0=(g,2t) c1=(g,2t+1) c2=(g+8,2t) c3=(g+8,2t+1)
//   tf32 截断：输入 fp32 位掩低 13 bit mantissa（round-toward-zero 近似 tf32）。
//   C 列 c0..c3 与 A 列错开 → 写回 C 按 c 布局精确落位。
// ============================================================
__device__ __forceinline__ float tf32_from_f32(float x) {
    // 截断到 tf32（19 bit：1 sign + 8 exp + 10 mantissa）→ 保留高 19 bit
    unsigned u = __float_as_uint(x);
    u &= 0xffffe000u;               // 清低 13 bit mantissa
    return __uint_as_float(u);
}

__global__ void tf32_mma_gemm_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    const int M, const int K, const int N) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    // ---- 编译期守卫（2026-09-10）：仅 sm_80+ 编译 tf32/m16n8k8 MMA 指令 ----
    //  否则低架构目标（如 sm_61/sm_75）ptxas 报 "Feature '.tf32' requires .target sm_80
    //  or higher"、"'mma' requires sm_70"、"'.m16n8k8' requires sm_80"。低架构走下方
    //  #else 空实现（host pass 与低 arch pass 均安全），运行时由 cutlass_hw_supported()
    //  拦截（本 kernel 仅 bench 验证用，未接入 mul_mat）。
    // 每 warp 一个 16×8 C 子块；block = 8 warps(256)，每 warp 负责一行 C 子块组
    const int warp_id = threadIdx.x >> 5;    // 0..7
    const int lane    = threadIdx.x & 31;
    const int g = lane >> 2;                 // groupID 0..7
    const int t = lane & 3;                  // tid 0..3

    // 每个 warp 的 C 子块：M 方向 block 覆盖 8 warps × 16 = 128 行？——不，
    // 用 blockIdx.y 覆盖 M，warp_id 沿 N 方向扩展（每 warp 一个独立 16×8）。
    // 简化：每 block 覆盖 16 行 × (8 warps × 8 列 = 64 列)。
    const int m_base = blockIdx.y * 16;
    const int n_base = blockIdx.x * 64 + warp_id * 8;
    if (m_base + 16 > M || n_base + 8 > N) return;   // 完整 tile 才计算（本验证版不做边界 padding）

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    for (int kk = 0; kk < K; kk += 8) {
        // A fragment (16×8 at kk): A[row][kk+col]
        const float a0 = tf32_from_f32(A[(m_base + g)     * K + (kk + t)]);
        const float a1 = tf32_from_f32(A[(m_base + g)     * K + (kk + t + 4)]);
        const float a2 = tf32_from_f32(A[(m_base + g + 8) * K + (kk + t)]);
        const float a3 = tf32_from_f32(A[(m_base + g + 8) * K + (kk + t + 4)]);
        // B fragment：mma 中 B 是 K×N col-major（row=k, col=n），每线程 2 值同列(g)不同
        // k 行（t 与 t+4）。内存存 B_row[n][k]（行主序 [N][K]）→ b0/b1 行都 = n_base+g，
        // 内存列 = kk+t 与 kk+t+4。⚠️ 之前误把 t 当行索引（行列对调）→ 结果错位。
        const float b0 = tf32_from_f32(B[(n_base + g) * K + (kk + t)]);
        const float b1 = tf32_from_f32(B[(n_base + g) * K + (kk + t + 4)]);
        // mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(__float_as_uint(a0)), "r"(__float_as_uint(a1)),
              "r"(__float_as_uint(a2)), "r"(__float_as_uint(a3)),
              "r"(__float_as_uint(b0)), "r"(__float_as_uint(b1)));
    }
    // 写回 C（按 C fragment 布局）
    C[(m_base + g)     * N + (n_base + 2 * t)]     = c0;
    C[(m_base + g)     * N + (n_base + 2 * t + 1)] = c1;
    C[(m_base + g + 8) * N + (n_base + 2 * t)]     = c2;
    C[(m_base + g + 8) * N + (n_base + 2 * t + 1)] = c3;
#else
    // 低于 sm_80：PTX 不支持 mma.tf32 / m16n8k8 —— 编译期留空实现，避免 ptxas 报错。
    // 运行时 cutlass_hw_supported() 返回 false → tf32_gemm_bench_cuda 直接返回 1，不会启动本 kernel。
    (void)A; (void)B; (void)C; (void)M; (void)K; (void)N;
#endif
}

// 性能测试包装：M×N×K fp32 GEMM（C=A*Bᵀ, B[N][K]），返回毫秒（ms_out），0=成功
int tf32_gemm_bench_cuda(const float* A, const float* B, float* C,
                         int M, int K, int N, float* ms_out) {
    if (!cutlass_hw_supported(nullptr, nullptr)) return 1;   // 非 sm_80+ 不支持
    const size_t a_sz = (size_t)M * K, b_sz = (size_t)N * K, c_sz = (size_t)M * N;
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    if (cudaMalloc(&dA, a_sz * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&dB, b_sz * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&dC, c_sz * sizeof(float)) != cudaSuccess) return 1;
    cudaMemcpy(dA, A, a_sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, b_sz * sizeof(float), cudaMemcpyHostToDevice);

    // grid: x = N/64（每 block 64 列），y = M/16
    dim3 block(256);
    dim3 grid((N + 63) / 64, (M + 15) / 16);
    auto launch = [&]() { tf32_mma_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, K, N); };
    launch();
    cudaError_t err = cudaDeviceSynchronize();

    if (ms_out && err == cudaSuccess) {
        cudaEvent_t e_s, e_e;
        cudaEventCreate(&e_s); cudaEventCreate(&e_e);
        // warmup
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        cudaEventRecord(e_s);
        constexpr int R = 50;
        for (int i = 0; i < R; i++) launch();
        cudaEventRecord(e_e);
        cudaEventSynchronize(e_e);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e_s, e_e);
        *ms_out = ms / R;
        cudaEventDestroy(e_s); cudaEventDestroy(e_e);
    }
    cudaMemcpy(C, dC, c_sz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return (err == cudaSuccess) ? 0 : 1;
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
//   TA/TB（2026-09-10 前期 fp16 支持）：输入元素可 float 或 __half；
//   读入 global 时经 to_float() 转 fp32 存入 smem，累加与输出始终 fp32。
// ============================================================

__device__ __forceinline__ float to_float(float x)  { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }

template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
         int B_BLOCK_X = 32, int C_BLOCK_X = 16, typename TA, typename TB>
__global__ void blockTileGEMM(const TA* A, const TB* B, float* C, const int M, const int K, const int N) {
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
        As[i][j] = (r < M && c < K) ? to_float(A[r * K + c]) : 0.f;
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
        Bs[i][j] = (r < K && c < N) ? to_float(B[c * K + r]) : 0.f;
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

    blockTileGEMM<Bm, Bn, 8, 256, 8, 32, 16, float, float><<<grid, block>>>(A, B, C, M, K, N);
    cudaCheck(cudaGetLastError());
}

// fp16 输入版本（前期支持，2026-09-10）：A/B 按 16 位 __half 存储（2 字节/元素），输出 fp32。
// 语义与 mul_mat_cuda 完全一致：A=(M,K) 行主序；B 以 (N,K) 转置存储 B[n*K+k]。
void mul_mat_cuda_f16(const void* A, const void* B, float* C, int M, int K, int N) {
    constexpr int Bm = 128, Bn = 128;
    dim3 block(256);
    dim3 grid((N + Bn - 1) / Bn, (M + Bm - 1) / Bm);
    blockTileGEMM<Bm, Bn, 8, 256, 8, 32, 16, __half, __half><<<grid, block>>>(
        reinterpret_cast<const __half*>(A), reinterpret_cast<const __half*>(B), C, M, K, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================
// fp16 Tensor-Core GEMM（mma.m16n8k16，**fp32 累加**）—— 2026-09-10
//   C[M,N] = A[M,K] × B[N,K]ᵀ（B 以 [N][K] 行主序存储，语义与 SIMT 版完全一致）
//
//   Fragment 布局（PTX m16n8k16 row.col，g = lane>>2, t = lane&3）：
//     A(16×16, row):  a0={A[g][2t],A[g][2t+1]}       a1={A[g+8][2t],A[g+8][2t+1]}
//                     a2={A[g][2t+8],A[g][2t+9]}     a3={A[g+8][2t+8],A[g+8][2t+9]}
//     B(16×8,  col):  b0={B_l[2t][g],B_l[2t+1][g]}   b1={B_l[2t+8][g],B_l[2t+9][g]}
//                     （B_l[k][n] = 输入 B 的 B_in[n][k]）
//     C(16×8,  f32):  c0=C[g][2t]  c1=C[g][2t+1]  c2=C[g+8][2t]  c3=C[g+8][2t+1]
//   本实现为「简化版」：fragment 直接从 global 按行取 2 个连续 half（不用 ldmatrix/smem）。
//   可优化（未做）：smem 暂存 + ldmatrix 批量加载、cp.async 双缓冲、每 warp 多 tile、
//     stmatrix 写回 —— 见函数上方注释的 pipeline 说明。
//   仅 sm_80+ 编译（低架构空实现）；运行时由 cutlass_hw_supported() 拦截。
// ============================================================

// 取 (r,c) 与 (r,c+1) 打包为一个 .b32（低 16 位 = 第一个元素）；越界元素置 0
__device__ __forceinline__ unsigned mma_ld_half2(const __half* __restrict__ base,
                                                 int r, int c, int rows, int cols) {
    const __half zero = __float2half(0.0f);
    const __half h0 = (r < rows && c     >= 0 && c     < cols) ? base[(size_t)r * cols + c]     : zero;
    const __half h1 = (r < rows && c + 1 >= 0 && c + 1 < cols) ? base[(size_t)r * cols + c + 1] : zero;
    const __half2 h2 = __halves2half2(h0, h1);   // low = h0（PTX 寄存器低半 = 第一个 half）
    unsigned u;
    memcpy(&u, &h2, sizeof(u));
    return u;
}

template<int WARPS_PER_BLOCK, int N_TILES>
__global__ void mul_mat_mma_f16_kernel(const __half* __restrict__ A,
                                       const __half* __restrict__ B,
                                       float* __restrict__ C,
                                       const int M, const int K, const int N) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const int lane = (int)(threadIdx.x & 31u);
    const int warp = (int)(threadIdx.x >> 5);
    const int g = lane >> 2;     // groupID 0..7
    const int t = lane & 3;      // 0..3

    const int m0 = blockIdx.y * 16;                                       // M 方向 16 行
    const int cols_per_warp = 8 * N_TILES;
    const int n_base = blockIdx.x * (WARPS_PER_BLOCK * cols_per_warp) + warp * cols_per_warp;
    if (m0 >= M || n_base >= N) return;

    // 每 warp N_TILES 个 16×8 tile 的 fp32 累加器
    float acc[N_TILES][4];
#pragma unroll
    for (int j = 0; j < N_TILES; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[j][q] = 0.f;

    for (int k0 = 0; k0 < K; k0 += 16) {
        // A fragment 每 warp 只载入一次，供 N_TILES 次 mma 复用（提升 mma/访存比）
        const unsigned a0 = mma_ld_half2(A, m0 + g,     k0 + 2 * t,     M, K);
        const unsigned a1 = mma_ld_half2(A, m0 + g + 8, k0 + 2 * t,     M, K);
        const unsigned a2 = mma_ld_half2(A, m0 + g,     k0 + 2 * t + 8, M, K);
        const unsigned a3 = mma_ld_half2(A, m0 + g + 8, k0 + 2 * t + 8, M, K);
#pragma unroll
        for (int j = 0; j < N_TILES; ++j) {
            const int nj = n_base + j * 8;
            const unsigned b0 = mma_ld_half2(B, nj + g, k0 + 2 * t,     N, K);
            const unsigned b1 = mma_ld_half2(B, nj + g, k0 + 2 * t + 8, N, K);
            // D = A(16×16) × B(16×8) + D，A/B 为 f16、C/D 为 f32（fp32 累加）
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }

    // 写回（C fragment 布局；边界外跳过）
    const int r0 = m0 + g, r1 = m0 + g + 8;
#pragma unroll
    for (int j = 0; j < N_TILES; ++j) {
        const int ca = n_base + j * 8 + 2 * t;
        const int cb = ca + 1;
        if (r0 < M) { if (ca < N) C[(size_t)r0 * N + ca] = acc[j][0];
                      if (cb < N) C[(size_t)r0 * N + cb] = acc[j][1]; }
        if (r1 < M) { if (ca < N) C[(size_t)r1 * N + ca] = acc[j][2];
                      if (cb < N) C[(size_t)r1 * N + cb] = acc[j][3]; }
    }
#else
    // 低于 sm_80：m16n8k16 mma 不可用 —— 编译期留空（运行时由 cutlass_hw_supported 拦截）
    (void)A; (void)B; (void)C; (void)M; (void)K; (void)N;
#endif
}

// fp16 tensor-core GEMM 入口（fp32 累加）。
// 返回 0 = 已执行；1 = 硬件不支持（< sm_80）或尺寸非法 → 调用方回落 SIMT 版 mul_mat_cuda_f16。
int mul_mat_mma_f16(const void* A, const void* B, float* C, int M, int K, int N) {
    if (!cutlass_hw_supported(nullptr, nullptr)) return 1;   // 运行时判 CC >= 8.0
    if (M <= 0 || N <= 0 || K <= 0) return 1;
    constexpr int WARPS   = 4;     // 每 block 4 warps
    constexpr int N_TILES = 4;     // 每 warp 4 个 16×8 tile（A fragment 复用，提升 mma/访存比）
    dim3 block(WARPS * 32);
    dim3 grid((unsigned)((N + WARPS * 8 * N_TILES - 1) / (WARPS * 8 * N_TILES)),
              (unsigned)((M + 15) / 16));
    mul_mat_mma_f16_kernel<WARPS, N_TILES><<<grid, block>>>(
        reinterpret_cast<const __half*>(A), reinterpret_cast<const __half*>(B), C, M, K, N);
    cudaCheck(cudaGetLastError());
    return 0;
}

// ============================================================
// fp16 Tensor-Core GEMM v2：smem + ldmatrix + cp.async 双缓冲流水（2026-09-10）
//   C[M,N] = A[M,K] × B[N,K]ᵀ（A[M][K]、B[N][K] 行主序 half；输出 fp32）
//   分块：BM=64, BN=64, BK=32；block=128 线程(4 warps)；每 warp 32(M)×32(N)
//   smem：双缓冲 sA[2][64][32] + sB[2][64][32]（16 KB，16B 对齐）
//   流水：cp.async 预取下一块 → commit → wait_group<1> → __syncthreads → mma → __syncthreads
//   加载：A → ldmatrix.x4（row-major fragment）；B → ldmatrix.x2（非转置：
//         smem 行=n、列=k 时，ldmatrix 给出的 {M[g][2t],M[g][2t+1]} 恰为 b0/b1 所需布局）
//   边界：cp.async 的 src-size 参数做零填充（行/列越界时 src_size=0，不读 global）；
//         **不做 block 级 early return**（否则 __syncthreads 死锁）→ 越界 fragment 全 0，
//         epilogue 按 M/N 边界跳过写回。
//   守卫：仅 sm_80+ 编译（低架构空实现），运行时由 cutlass_hw_supported() 拦截。
// ============================================================
#define PPML_G2_BM 64
#define PPML_G2_BN 64
#define PPML_G2_BK 32
#define PPML_G2_STAGES 2
#define PPML_G2_THREADS 128

// cp.async：从 global 拷贝 16B 到 smem；src_bytes<16 时余下字节填 0（0 则全零且不读 global）
__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src, unsigned src_bytes) {
    const unsigned sdst = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(sdst), "l"(gmem_src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;\n"); }
template<int NWAIT>
__device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;\n" :: "n"(NWAIT)); }

// ldmatrix：x4（4 个 8×8 b16 矩阵）/ x2（2 个）
__device__ __forceinline__ void ldsm_x4_b16(unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3,
                                            const void* smem_lane_addr) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(smem_lane_addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2_b16(unsigned& d0, unsigned& d1, const void* smem_lane_addr) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(smem_lane_addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(d0), "=r"(d1) : "r"(a));
}
// m16n8k16：D = A×B + D（A/B f16，C/D f32 累加）
__device__ __forceinline__ void mma_f16_f32_acc(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void mul_mat_mma_f16_smem_kernel(const __half* __restrict__ A,
                                            const __half* __restrict__ B,
                                            float* __restrict__ C,
                                            const int M, const int K, const int N) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int BM = PPML_G2_BM, BN = PPML_G2_BN, BK = PPML_G2_BK;
    constexpr int STAGES = PPML_G2_STAGES;
    constexpr int NTH = PPML_G2_THREADS;
    constexpr int SEG = BK / 8;                 // 每行 16B 片段数（8 half/片段）

    __shared__ __align__(16) __half sA[STAGES][BM][BK];
    __shared__ __align__(16) __half sB[STAGES][BN][BK];

    const int tid      = (int)threadIdx.x;
    const int m_block  = blockIdx.y * BM;
    const int n_block  = blockIdx.x * BN;
    const int k_blocks = (K + BK - 1) / BK;

    // ---- 异步加载一块 [k0, k0+BK) 到 stage s（A: BM×BK，B: BN×BK）----
    // cp.async 的两个硬约束（踩坑记录）：
    //   ① cp-size=16 时 **src-size 只允许 4/8/16**；尾部 rem*2（如 12 字节）是未定义行为；
    //   ② **global 源地址必须按 cp-size(16B) 对齐**。当 K 不是 8 的倍数时，行偏移
    //      row*K*2 字节不再是 16B 的倍数 → 只有 row ≡ 0 (mod 8/gcd) 的行才对齐。
    // 故：仅当"整片段 + 源 16B 对齐"时才发 cp.async；否则退化为同步手工拷贝 + 补零
    //      （正确性优先；K 为 8 倍数时（真实模型的常见情形）全部走 cp.async，性能不受影响）。
    auto load_tile = [&](int k0, int s) {
        for (int idx = tid; idx < BM * SEG; idx += NTH) {          // A
            const int row = idx / SEG, seg = idx % SEG;
            const int m = m_block + row, k = k0 + seg * 8;
            __half* dst = &sA[s][row][seg * 8];
            const int rem = (m < M) ? (K - k) : 0;                 // 有效 half 数
            const __half* src = (rem > 0) ? (A + (size_t)m * K + k) : A;
            if (rem >= 8 && ((reinterpret_cast<uintptr_t>(src) & 15u) == 0u)) {
                cp_async_16B(dst, src, 16u);
            } else {
                for (int i = 0; i < 8; ++i)
                    dst[i] = (i < rem) ? src[i] : __float2half(0.0f);
            }
        }
        for (int idx = tid; idx < BN * SEG; idx += NTH) {          // B
            const int row = idx / SEG, seg = idx % SEG;
            const int n = n_block + row, k = k0 + seg * 8;
            __half* dst = &sB[s][row][seg * 8];
            const int rem = (n < N) ? (K - k) : 0;
            const __half* src = (rem > 0) ? (B + (size_t)n * K + k) : B;
            if (rem >= 8 && ((reinterpret_cast<uintptr_t>(src) & 15u) == 0u)) {
                cp_async_16B(dst, src, 16u);
            } else {
                for (int i = 0; i < 8; ++i)
                    dst[i] = (i < rem) ? src[i] : __float2half(0.0f);
            }
        }
    };

    load_tile(0, 0);
    cp_async_commit_group();

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int g      = lane >> 2;
    const int t      = lane & 3;
    const int warp_m = warp >> 1;                       // 0..1（M 方向 32 行）
    const int warp_n = warp & 1;                        // 0..1（N 方向 32 列）
    const int a_mat  = lane >> 3;                       // ldmatrix.x4：矩阵 0..3
    const int a_lr   = lane & 7;                        // 矩阵内行 0..7
    const int b_l16  = lane & 15;                       // ldmatrix.x2 仅 lane 0..15 有效

    for (int kb = 0; kb < k_blocks; ++kb) {
        if (kb + 1 < k_blocks) load_tile((kb + 1) * BK, (kb + 1) % STAGES);
        cp_async_commit_group();                        // 空组也合法 → 统一 wait 语义
        cp_async_wait_group<1>();                       // 至多 1 组在途（= 下一块）→ 当前块已就绪
        __syncthreads();

        const int s = kb % STAGES;
        const int a_row_base = warp_m * 32;
        const int b_col_base = warp_n * 32;

#pragma unroll
        for (int k16 = 0; k16 < BK; k16 += 16) {
            unsigned a[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {               // A：2 个 16×16 tile
                const int r  = a_row_base + i * 16 + ((a_mat & 1) ? 8 : 0) + a_lr;
                const int cc = k16 + ((a_mat & 2) ? 8 : 0);
                ldsm_x4_b16(a[i][0], a[i][1], a[i][2], a[i][3], &sA[s][r][cc]);
            }
            unsigned b[4][2];
#pragma unroll
            for (int j = 0; j < 4; ++j) {               // B：4 个 8×16 tile
                const int r  = b_col_base + j * 8 + (b_l16 & 7);
                const int cc = k16 + ((b_l16 >> 3) ? 8 : 0);
                ldsm_x2_b16(b[j][0], b[j][1], &sB[s][r][cc]);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    mma_f16_f32_acc(acc[i][j], a[i], b[j]);
        }
        __syncthreads();                                // 所有 warp 用完本 stage 后才能覆写
    }

    // ---- epilogue：写回 C（越界跳过）----
    const int m_base = m_block + warp_m * 32;
    const int n_base = n_block + warp_n * 32;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r0 = m_base + i * 16 + g;
        const int r1 = r0 + 8;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c0 = n_base + j * 8 + 2 * t;
            const int c1 = c0 + 1;
            if (r0 < M) { if (c0 < N) C[(size_t)r0 * N + c0] = acc[i][j][0];
                          if (c1 < N) C[(size_t)r0 * N + c1] = acc[i][j][1]; }
            if (r1 < M) { if (c0 < N) C[(size_t)r1 * N + c0] = acc[i][j][2];
                          if (c1 < N) C[(size_t)r1 * N + c1] = acc[i][j][3]; }
        }
    }
#else
    (void)A; (void)B; (void)C; (void)M; (void)K; (void)N;
#endif
}

// 流水版入口：返回 0 = 已执行；1 = 不支持（< sm_80 或尺寸非法）→ 调用方回落
int mul_mat_mma_f16_smem(const void* A, const void* B, float* C, int M, int K, int N) {
    if (!cutlass_hw_supported(nullptr, nullptr)) return 1;
    if (M <= 0 || N <= 0 || K <= 0) return 1;
    dim3 block(PPML_G2_THREADS);
    dim3 grid((unsigned)((N + PPML_G2_BN - 1) / PPML_G2_BN),
              (unsigned)((M + PPML_G2_BM - 1) / PPML_G2_BM));
    mul_mat_mma_f16_smem_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(A), reinterpret_cast<const __half*>(B), C, M, K, N);
    cudaCheck(cudaGetLastError());
    return 0;
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
    //    13=EXP, 14=LOG, 15=SQRT, 16=SQR。之前把 10 当 SQRT（实为 SIGMOID）→ sqrt(负)→NaN。
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
        case 16: y = x * x;           break;   // SQR（独立 op OP_SQR 用，非 unary）
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
// OP_CLAMP: 逐元素值裁剪 dst = clamp(src, lo, hi)
//   lo/hi 由宿主侧从 op_params[0]/[1]（float 位模式）读出后传入。
//   NaN 透传（与 CPU kernel_clamp 一致）；范围外取 lo/hi。
// ============================================================
__global__ void clamp_kernel(
    const float * __restrict__ src, float * __restrict__ dst, int N,
    const float lo, const float hi) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float x = src[tid];
    dst[tid] = (x <= lo) ? lo : ((x >= hi) ? hi : x);
}

void clamp_cuda(const float * src, float * dst, int N, float lo, float hi, int block_size) {
    if (N <= 0) return;
    int grid_size = ceil_div(N, block_size);
    clamp_kernel<<<grid_size, block_size>>>(src, dst, N, lo, hi);
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

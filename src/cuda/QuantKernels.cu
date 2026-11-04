// QuantKernels.cu —— int8 分块量化的设备端「量化 / 反量化」kernel（W8A8 Step 2，2026-10-17）
// ============================================================
// 约定（与 include/ppml/QuantBlocks.h 完全一致）：
//   · 块大小 QK8_0 = QK8_1 = 32；块结构 34B / 36B，与 ggml 二进制兼容；
//   · 激活量化器：d = max/(-128)（max = 取到 max|x| 的那个带符号值），
//     q = clamp(round(x/d), [-128, 127])，s = d * Σq 存在 ds.y；
//   · 量化块不跨行 ⇒ 元素数必须是 32 的整数倍（调用方保证 K % 32 == 0）。
// 布局：输入按 [K*N] 连续（graph 的 src1 为 [K,N]，K 为最内维）
//   ⇒ 每 32 个连续元素恰好构成一个完整量化块，块号 ib 直接对应
//      y[ib] = 第 (ib*32/ K) 列的第 ((ib*32) % K / 32) 块。
// 实现：
//   · quantize：1 warp 处理 1 块（lane 各持 1 个元素），shfl 归约 amax（同时携带带符号 max）
//     与整数 qsum，lane0 写 ds —— 与 host 参考实现逐位一致（tie 的选取由 `>` 保证首个胜出）。
//   · dequantize：1 thread = 1 元素，仅供测试对拍与 < sm_61 的 fallback。
// ============================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#include "ppml/QuantBlocks.h"

namespace ppml {

namespace {

constexpr unsigned kFullMask = 0xffffffffu;

__global__ void quantize_q8_1_kernel(const float* __restrict__ x,
                                     block_q8_1* __restrict__ y,
                                     int64_t nblocks) {
    const int warps_per_cta = blockDim.x >> 5;
    const int64_t ib = static_cast<int64_t>(blockIdx.x) * warps_per_cta + (threadIdx.x >> 5);
    if (ib >= nblocks) return;
    const int lane = threadIdx.x & 31;

    const float v = x[ib * QK8_1 + lane];

    // 归约 max|x|，同时携带「取到 max 的那个带符号值」（严格 `>` ⇒ 平局取胜规则与 host 一致）
    float amax = fabsf(v), vmax = v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float oa = __shfl_xor_sync(kFullMask, amax, off);
        const float ov = __shfl_xor_sync(kFullMask, vmax, off);
        if (oa > amax) { amax = oa; vmax = ov; }
    }

    // d = max / -128：把最大幅值钉到 -128（int8 唯一多出的码位），d 可带符号
    const float d   = (amax == 0.0f) ? 0.0f : vmax / -128.0f;
    const float inv = (d == 0.0f) ? 0.0f : 1.0f / d;

    float q = roundf(v * inv);
    q = q > 127.0f ? 127.0f : (q < -128.0f ? -128.0f : q);   // 反号极值需要 +128 ⇒ 饱和
    const int8_t qi = static_cast<int8_t>(q);
    y[ib].qs[lane] = qi;

    int qsum = static_cast<int>(qi);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) qsum += __shfl_xor_sync(kFullMask, qsum, off);

    if (lane == 0) y[ib].ds = __floats2half2_rn(d, d * static_cast<float>(qsum));
}

__global__ void dequantize_q8_0_kernel(const block_q8_0* __restrict__ x,
                                       float* __restrict__ y, int64_t ne) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= ne) return;
    const int64_t ib = i / QK8_0, iq = i - ib * QK8_0;
    y[i] = __half2float(x[ib].d) * static_cast<float>(x[ib].qs[iq]);
}

__global__ void dequantize_q8_1_kernel(const block_q8_1* __restrict__ x,
                                       float* __restrict__ y, int64_t ne) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= ne) return;
    const int64_t ib = i / QK8_1, iq = i - ib * QK8_1;
    y[i] = __half2float(__low2half(x[ib].ds)) * static_cast<float>(x[ib].qs[iq]);
}

} // namespace

// ------------------------------------------------------------------
// 主机侧 launcher（供 src/backend/CUDAKernels.cpp 与测试调用）
//   返回 0 = 成功；非 0 = 参数非法 / CUDA 错误。
// ------------------------------------------------------------------
int quantize_rows_q8_1_cuda(const float* x, block_q8_1* y, int64_t ne, cudaStream_t stream) {
    if (!x || !y || ne <= 0 || (ne % QK8_1) != 0) return 1;
    const int64_t nblocks = ne / QK8_1;
    constexpr int kWarps = 8;                                   // 256 线程 = 8 warp ⇒ 每 CTA 8 块
    const int grid = static_cast<int>((nblocks + kWarps - 1) / kWarps);
    quantize_q8_1_kernel<<<grid, kWarps * 32, 0, stream>>>(x, y, nblocks);
    return (cudaGetLastError() == cudaSuccess) ? 0 : 1;
}

int dequantize_q8_0_cuda(const block_q8_0* x, float* y, int64_t ne, cudaStream_t stream) {
    if (!x || !y || ne <= 0) return 1;
    constexpr int kThreads = 256;
    const int grid = static_cast<int>((ne + kThreads - 1) / kThreads);
    dequantize_q8_0_kernel<<<grid, kThreads, 0, stream>>>(x, y, ne);
    return (cudaGetLastError() == cudaSuccess) ? 0 : 1;
}

int dequantize_q8_1_cuda(const block_q8_1* x, float* y, int64_t ne, cudaStream_t stream) {
    if (!x || !y || ne <= 0) return 1;
    constexpr int kThreads = 256;
    const int grid = static_cast<int>((ne + kThreads - 1) / kThreads);
    dequantize_q8_1_kernel<<<grid, kThreads, 0, stream>>>(x, y, ne);
    return (cudaGetLastError() == cudaSuccess) ? 0 : 1;
}

} // namespace ppml

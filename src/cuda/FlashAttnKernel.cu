// ============================================================
// FlashAttnKernel.cu — Flash Attention v1（1-pass）CUDA 前向（Step ②，2026-09-22）
//
//   flash_attn_forward_cuda()：与 flash_attn_forward_cpu() **同签名、同数学** ⇒ 1:1 对拍 ✓
//
//   布局（同 include/ppml/FlashAttn.h）：
//     Q/K/V/O = [d, L, h, B]（dims[0]=d 最内 ✓）；LSE = [L, h, B]；bias = [Lk, Lq, h, B]（可空 ✓）
//     元素偏移：Q/K/V/O (b,hh,i,dd) = ((b*h+hh)*L + i)*d + dd；bias (b,hh,i,j) = ((b*h+hh)*L + i)*L + j ✓
//
//   v1 线程分工：**一个线程负责一行 query**（简单、无需 warp 归约 ✓）
//     · block = FA_BR(128) 线程；query 行 i = blockIdx.x*FA_BR + threadIdx.x ✓
//     · 该行的 Q 常驻寄存器（qreg[d] ✓）；O 累加器 acc[d] 也在寄存器 ✓
//     · 沿 K 按 FA_BC(64) 分块；K/V tile 协作加载进 smem（行主序）
//       ⇒ 计算时同一指令所有线程读**同一地址**（广播 ✓ 无 bank conflict ✓，无需 padding ✓）
//     · 在线 softmax 全在线程私有寄存器：**只在行最大值更新时**做一次 alpha 缩放 ✓
//       （避免每个 j 都做 d 次乘；同时避免 exp(-inf - -inf) = NaN ✗ 的两处守卫 ✓）
//   causal：块级只遍历到 `min(L, row0+FA_BR)`（省 ~50% ✓），对角块再逐元素 `j > i ⇒ -inf` ✓
//   尾部：L 非 FA_BR/FA_BC 倍数安全（加载/计算都用边界判定 ✓）；越界线程不提前退出（否则
//         __syncthreads 死锁 ✗），只用 `active` 跳过自己的行 ✓
//
//    v1 已知效率点（留给后续优化，非本步目标）：
//     ① Q 行由各线程直读 global ⇒ 跨线程 stride=d，**非合并访问** ✗（数据量小、每行只读一次 ⇒ 可接受）；
//     ② 用精确 expf 而非 __expf（先保证数值；__expf 在 |x| 大时有 ~1e-4 相对误差 ✗）；
//     ③ 无 cp.async / 双缓冲 / ldmatrix ✗；一个线程一行 ⇒ 每 S 元素只有 d 次 MAC，无 warp 级并行 ✗；
//     ④ Bc 固定 64（cfg.Bc 暂不生效 ✓）。
// ============================================================
#include "ppml/FlashAttn.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>   // getenv（PPML_FLASH_ATTN_TC ✓）
#include <cstdio>    // std::fprintf（融合内核的调试打印 ✓）

namespace ppml {

namespace {

constexpr int FA_D_MAX = 64;    // v1：head_dim ≤ 64（项目实际 d=32 ✓）；超过则返回不支持 ⇒ 调用方回落 CPU ✓
constexpr int FA_BC    = 64;    // K/V tile 宽度
constexpr int FA_BR    = 128;   // = blockDim.x（一线程一行）

__global__ void flash_attn_fwd_kernel(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      const float* __restrict__ Bias,
                                      float* __restrict__ O,
                                      float* __restrict__ LSE,
                                      const int L, const int d,
                                      const float scale, const int causal, const float eps) {
    // ============================================================
    // 【②-v2 第二步】warp 内 **d 维切分给 lane**（真 flash 布局 ✓）
    //   v1（一线程一行）的问题（实测 MSA 形状 74–114 ms，旧三段只要 33.7 ms ✗✗）：
    //     ① 每 (i,j) 的 d 维 MAC 是一条 32 长的 FMA 依赖链 ✗ ⇒ latency-bound（SM 利用率 ~0.3% ✗）；
    //     ② 每 lane 要完整持有整行 Q/O（d 个寄存器 ✗）⇒ 在飞 warp 数受限 ✗。
    //   v2 布局：block = 128 线程 = 4 warps ⇒ 每 block **32 行**（每 warp 8 行 ✓）
    //     lane = row_in_warp(0..7)*4 + g(0..3) ⇒ **同一行的 4 个 lane 连续** ✓
    //     ⇒ 行归约只要 shfl_xor(1) + shfl_xor(2) 两步 ✓
    //   关键巧思：s 已全归约 ⇒ **online softmax 的 m/l 在同一行 4 个 lane 上天然一致** ✓
    //     ⇒ 零额外通信（否则还得再归约一次 ✗）；每 lane 只持 d/4 个元素（d ≤ 64 ⇒ ≤16 ✓）。
    //   开销：每 (i,j) 每 lane 8 次 FMA（dg=8，4 累加器 ⇒ 依赖链 2 层 ✓）+ 2 次 shuffle（≈25% ✗ 可接受 ✓）。
    //   smem：**动态分配**，行跨距 = d + PAD(4) ⇒ 16B 对齐 ✓ 且 8 行不同 bank（无广播冲突 ✗→✓）。
    //   ⚠️ d 必须是 4 的倍数（行首/分组首 16B 对齐 ✓）——host 已把关 ✓；数学与 v1 一致（求和顺序变 ✓）。
    // ============================================================
    constexpr int PAD = 4;                    // smem 行跨距补 4（16B 对齐 + 消 bank 冲突 ✓）
    const int STRIDE = d + PAD;
    extern __shared__ float fsmem[];
    float* sK = fsmem;                        // [FA_BC][STRIDE]
    float* sV = fsmem + FA_BC * STRIDE;       // [FA_BC][STRIDE]

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row_in_warp = lane >> 2;        // 0..7
    const int g            = lane & 3;        // d 分组 0..3
    const int bh   = (int)blockIdx.y;                     // b*h + hh
    const int row0 = (int)blockIdx.x * (8 * 4);           // 每 block 32 行 ✓
    const int i    = row0 + warp * 8 + row_in_warp;
    const bool active = (i < L);
    const size_t base = (size_t)bh * (size_t)L * (size_t)d;
    const int dg = d / 4;                                 // 本 lane 负责的元素数（≤ 16 ✓）
    const int c0 = g * dg;                                // 本 lane 的起始维 ✓

    // 本 block 需要遍历的最大键数（causal ⇒ 到本 block 最后一行 ✓）
    int jmax_block = L;
    if (causal) {
        const int last_row = (row0 + 31 < L) ? (row0 + 31) : (L - 1);
        jmax_block = last_row + 1;
    }
    // ⚠️ tile 循环上界必须**整块统一**（= jmax_block ✓）：循环里有 __syncthreads() ⇒
    //    warp 之间迭代次数不同 = 屏障不匹配 = UB ✗（最初按 warp 取上界 ⇒ causal 用例直接算错 ✗）。
    //    causal 的省算改到 j 循环内用 **warp 统一**的早退 ✓（无屏障 ⇒ 安全 ✓）+ 逐元素掩码兜底 ✓。
    const int iwarp_max = ((row0 + warp * 8 + 7) < L) ? (row0 + warp * 8 + 7) : (L - 1);

    // ---- 本 lane 的 Q 切片 + 累加器（dg ≤ 16 ⇒ 寄存器大减 ✓）----
    float qr[16];
    float acc[16];
    for (int t = 0; t < dg; ++t) {
        qr[t]  = active ? Q[base + (size_t)i * d + c0 + t] : 0.f;
        acc[t] = 0.f;
    }
    float m = -INFINITY;   // running row max（同行 4 lane 一致 ✓）
    float l = 0.f;         // running Σ exp（未归一化 ✓）

    for (int j0 = 0; j0 < jmax_block; j0 += FA_BC) {
        const int nb = ((j0 + FA_BC) < jmax_block) ? FA_BC : (jmax_block - j0);

        // ---- K/V tile 协作加载（写进带 PAD 的行跨距 ✓）----
        for (int t = tid; t < nb * d; t += 128) {
            const int r = t / d, c = t % d;
            const size_t off = base + (size_t)(j0 + r) * d + c;
            sK[r * STRIDE + c] = K[off];
            sV[r * STRIDE + c] = V[off];
        }
        __syncthreads();

        const int j1 = j0 + nb;
        for (int j = j0; j < j1; ++j) {
            if (causal && j > iwarp_max) break;      // warp 统一 ⇒ 无屏障风险 ✓（本 warp 8 行的 j 都够了 ✓）
            // ---- 本 lane 那 dg 个元素的局部点积（4 累加器 ⇒ 依赖链 dg/4 ≈ 2 层 ✓）----
            const float* krow = &sK[(size_t)(j - j0) * STRIDE + c0];
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            int t = 0;
            for (; t + 4 <= dg; t += 4) {
                a0 += qr[t + 0] * krow[t + 0];
                a1 += qr[t + 1] * krow[t + 1];
                a2 += qr[t + 2] * krow[t + 2];
                a3 += qr[t + 3] * krow[t + 3];
            }
            for (; t < dg; ++t) a0 += qr[t] * krow[t];
            float s = (a0 + a1) + (a2 + a3);

            // ---- 行内 4 lane 归约（lane 连续 ⇒ xor 1/2 两步 ✓；全 warp 必须都参与 ✓）----
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);

            s *= scale;
            if (Bias && active) s += Bias[((size_t)bh * (size_t)L + (size_t)i) * (size_t)L + (size_t)j];
            if (causal && j > i) s = -INFINITY;      // 对角块逐元素掩码 ✓
            if (!active)         s = -INFINITY;      // 越界行：不读 bias/不写回（m/l 无害 ✓）

            // ---- 在线 softmax：同行 4 lane 天然一致（s 已归约 ✓）⇒ 零通信 ✓ ----
            if (s > m) {
                const float alpha = (m == -INFINITY) ? 0.f : expf(m - s);
                l *= alpha;
                for (int q = 0; q < dg; ++q) acc[q] *= alpha;
                m = s;
            }
            const float beta = (s == -INFINITY) ? 0.f : expf(s - m);
            l += beta;
            const float* vrow = &sV[(size_t)(j - j0) * STRIDE + c0];
            for (int q = 0; q < dg; ++q) acc[q] += beta * vrow[q];
        }
        __syncthreads();     // 所有线程用完本 tile 才能覆写 ✓
    }

    // ---- 收尾：每 lane 写自己那 dg 个元素 ✓；LSE 由 g==0 写 ✓（l==0 ⇒ O=0、LSE=-inf ✓）----
    if (active) {
        float* o = O + base + (size_t)i * d + c0;
        if (l > 0.f) {
            const float inv = 1.0f / (l + eps);
            for (int q = 0; q < dg; ++q) o[q] = acc[q] * inv;
            if (g == 0 && LSE) LSE[(size_t)bh * (size_t)L + (size_t)i] = m + logf(l + eps);
        } else {
            for (int q = 0; q < dg; ++q) o[q] = 0.f;
            if (g == 0 && LSE) LSE[(size_t)bh * (size_t)L + (size_t)i] = -INFINITY;
        }
    }
}

// ============================================================
// 【②-v2 第一步】ILP=8 的点积：拆 FMA 依赖链 ✓（反向 kernel 的 S / dP 用它 ✓）
//   为什么：v1 的 `for (dd) s += a[dd]*b[dd]` 是一条 n 长依赖链 ✗（FMA 延迟 ~4 cycle ⇒ n=32 时
//   ~128 cycle 才出一个结果 ✗）；8 个独立累加器 ⇒ 链长降到 n/8 ✓、ILP=8 ⇒ 接近吞吐上限 ✓。
// ⚠️ 求和顺序与 v1 不同 ⇒ 与 CPU 对拍仍按 ~2e-3 容差 ✓（dQ 路径不变 ⇒ 仍 1e-5 ✓）
// ============================================================
__device__ __forceinline__ float dot_ilp8(const float* __restrict__ a, const float* __restrict__ b, const int n) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f, s4 = 0.f, s5 = 0.f, s6 = 0.f, s7 = 0.f;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 += a[i + 0] * b[i + 0]; s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
        s4 += a[i + 4] * b[i + 4]; s5 += a[i + 5] * b[i + 5];
        s6 += a[i + 6] * b[i + 6]; s7 += a[i + 7] * b[i + 7];
    }
    for (; i < n; ++i) s0 += a[i] * b[i];
    return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

}  // namespace

// 【②-TC】常量与 TC kernel 的前置声明（**定义在文件后半**（反向之前 ✓）⇒ host 里要先声明 ✓）
constexpr int FA_TC_BR = 64;       // 每 block 行数（4 warps × 16 ✓）
constexpr int FA_TC_BN = 64;       // K/V tile 的键数（8 个 n8 步 ✓）
__global__ void flash_attn_fwd_tc_kernel(const float* __restrict__ Q,
                                         const float* __restrict__ K,
                                         const float* __restrict__ V,
                                         const float* __restrict__ Bias,
                                         float* __restrict__ O,
                                         float* __restrict__ LSE,
                                         const int L, const int d,
                                         const float scale, const int causal, const float eps);

// 【②-TC】运行时能力判定（实现见 src/cuda/CUDAKernels.cu，namespace ppml ✓）—— 复用已有函数 ✓
bool cutlass_hw_supported(int* out_major, int* out_minor);

int flash_attn_forward_cuda(const FlashAttnConfig& cfg,
                            const float* Q, const float* K, const float* V,
                            const float* bias,
                            float* O, float* LSE) {
    if (!Q || !K || !V || !O) return 1;
    if (cfg.B <= 0 || cfg.h <= 0 || cfg.L <= 0 || cfg.d <= 0) return 1;
    if (cfg.d > FA_D_MAX) return 1;                    // 上限 64 ⇒ 调用方回落 CPU ✓
    if ((cfg.d % 4) != 0) return 1;                    // 【②-v2】lane 切分要求 d 是 4 的倍数（16B 对齐 ✓）

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) return 1;

    const float scale = (cfg.scale > 0.f) ? cfg.scale : 1.0f / std::sqrt((float)cfg.d);

    // ---- 【②-TC】可选：QKᵀ 走 TF32 tensor core（PPML_FLASH_ATTN_TC=1 ✓；默认 0 ⇒ 走 SIMT v2 路径 ✓）----
    //   ⚠️ 这里**每次调用都读 env**（不做 static 缓存 ✓）⇒ 测试可在同一进程内 setenv 做 A/B ✓（成本 ns 级 ✓）
    //   ⚠️ 需要 sm_80+（cutlass_hw_supported ✓）且 d 是 8 的倍数（每 k8 块 4 个值 ✓）；不满足一律回落 SIMT ✓
    const char* tc_env = getenv("PPML_FLASH_ATTN_TC");
    if (tc_env && tc_env[0] && tc_env[0] != '0' && (cfg.d % 8) == 0) {
        int maj = 0, minr = 0;
        if (cutlass_hw_supported(&maj, &minr)) {
            constexpr int tc_pad = 4;
            const size_t smem_tc = (size_t)2 * FA_TC_BN * (cfg.d + tc_pad) * sizeof(float);
            const dim3 grid_tc((unsigned)((cfg.L + FA_TC_BR - 1) / FA_TC_BR), (unsigned)(cfg.B * cfg.h));
            flash_attn_fwd_tc_kernel<<<grid_tc, dim3(128), smem_tc>>>(Q, K, V, bias, O, LSE,
                                                                     cfg.L, cfg.d, scale,
                                                                     cfg.causal ? 1 : 0, cfg.softmax_eps);
            if (cudaGetLastError() != cudaSuccess) return 1;
            return 0;
        }
    }

    // 【②-v2】block = 128 线程 = 4 warps × 8 行 ⇒ 每 block 32 行 ✓
    //   smem = 2 张 [FA_BC][d + PAD] 表（动态分配 ✓；d=32 时 18.4 KB ✓、d=64 时 34.8 KB ✓ 都在 48 KB 内 ✓）
    constexpr int rows_per_block = 8 * 4;
    constexpr int fwd_pad = 4;
    const size_t smem_bytes = (size_t)2 * FA_BC * (cfg.d + fwd_pad) * sizeof(float);
    const dim3 grid((unsigned)((cfg.L + rows_per_block - 1) / rows_per_block), (unsigned)(cfg.B * cfg.h));
    const dim3 block(128);

    flash_attn_fwd_kernel<<<grid, block, smem_bytes>>>(Q, K, V, bias, O, LSE,
                                                       cfg.L, cfg.d, scale,
                                                       cfg.causal ? 1 : 0, cfg.softmax_eps);
    if (cudaGetLastError() != cudaSuccess) return 1;
    return 0;
}

// ============================================================
// 【②-TC】前向的 QKᵀ 走 TF32 tensor core（**PV 与在线 softmax 仍 SIMT** ✓ 先换一半便于量收益）
//   为什么不能直接调 `mul_mat_mma_tf32_smem`（我们已实现的 TF32 流水 GEMM ✗）：
//     它是**整块 GEMM 算子**——A/B 从 global 读、结果写 global ✗；而 flash 前向的 scores 必须
//     **留在寄存器**直接喂在线 softmax（一旦落地 O(L²) 就把 flash 的意义抹掉 ✗）⇒ 只能复用"零件"：
//       ① 就地舍入 `cvt.rna.tf32.f32`（tf32_bits ✓，与 src/cuda/CUDAKernels.cu 的 tf32_rna(:1505) 同款 ✓）
//       ② mma m16n8k8 inline PTX（mma_t32 ✓，同 :1516 mma_tf32_f32_acc ✓）
//       ③ **fragment 取数索引**（同 mul_mat_mma_tf32_smem_kernel ✓，当年靠穷举探针定死 ✓）
//     ⚠️ 三块都是 `__device__ __forceinline__`（TU 内可见 ✗）⇒ 本文件**复制**一份（名字加 _t32 ✓）。
//
//   布局（比 v2 的 8 行/warp 扩到 **16 行/warp**，因为 mma 的 m=16 ✓）：
//     block = 128 线程 = 4 warps ⇒ 每 block **64 行**；grid.x = ceil(L/64) ✓
//     四元组（g = lane>>2 ∈ 0..7，t = lane&3 ∈ 0..3）按 mma 约定：
//       A(Q) : a0=(row g, col t) a1=(row g+8, col t) a2=(row g, col t+4) a3=(row g+8, col t+4)
//       C(S) : c0=(row g, col 2t) c1=(row g, col 2t+1) c2=(row g+8, col 2t) c3=(row g+8, col 2t+1)
//       B(K) : b0=(key n=g, dim t) b1=(key n=g, dim t+4)   ← B 以 [key][dim] 行主序存 ✓
//     ⇒ 每 lane 负责 **2 行**（row0+g / row0+g+8 ✓），每行 8 个 score 由四元组 4 lane 分摊（列 2t/2t+1 ✓）
//     ⇒ **行 max 用四元组内 `__shfl_xor(1)+(2)`** ✓（与 v1/v2 的 shuffle 归约同源 ✓）
//     ⇒ **Σ exp 用 lane 局部和**（每 lane 只管自己那 2 列 ✓）⇒ 只在**最后**归约一次 ✓
//     ⇒ PV（仍 SIMT ✓）：每 lane 对它 2 行的 d/4 分量累加（accO[2][8] ✓）：
//        `accO[r][q] += beta_c * V[key=col_c][t*dg + q]`（c = 它的 2 列 ✓；V 从带 PAD 的 smem 读 ✓）
//   ⚠️ 精度：tf32 10-bit 尾数 ⇒ 单元素相对误差 ~4.9e-4 ⇒ **对拍容差放宽到 ~1e-3** ✓（v2 是 1e-4 ✓）
//   ⚠️ 独立开关 `PPML_FLASH_ATTN_TC=1`（默认 0 ⇒ 走已验证的 SIMT v2 路径 ✓ 旧测试全不受影响 ✓）
//   ⚠️ 屏障安全：tile 循环上界用**块统一**的 jmax_block ✓（循环内有 __syncthreads ✓）
//   ⚠️ tail 安全：K/V tile 的**未用行零填充**（否则 stale smem 的 inf/NaN × 0 = NaN 会污染 ✗）
// ============================================================
// （FA_TC_BR / FA_TC_BN 已在文件前部声明 ✓ —— host 也要用，故提到前面 ✓）

// tf32 就地舍入（与 src/cuda/CUDAKernels.cu 的 tf32_rna 同款 ✓；低架构留空实现 ✓）
__device__ __forceinline__ unsigned tf32_bits(float x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    unsigned r;
    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
#else
    (void)x;
    return 0u;
#endif
}

// m16n8k8 tf32 mma（与 src/cuda/CUDAKernels.cu 的 mma_tf32_f32_acc 同款 ✓）
__device__ __forceinline__ void mma_t32(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void)c; (void)a; (void)b;
#endif
}

__global__ void flash_attn_fwd_tc_kernel(const float* __restrict__ Q,
                                         const float* __restrict__ K,
                                         const float* __restrict__ V,
                                         const float* __restrict__ Bias,
                                         float* __restrict__ O,
                                         float* __restrict__ LSE,
                                         const int L, const int d,
                                         const float scale, const int causal, const float eps) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int PAD = 4;
    const int STRIDE = d + PAD;
    extern __shared__ float fsmem[];
    float* sK = fsmem;                        // [FA_TC_BN][STRIDE]（键 × 维 ✓）
    float* sV = fsmem + FA_TC_BN * STRIDE;    // [FA_TC_BN][STRIDE] ✓

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g    = lane >> 2;                            // 0..7（mma 行组 / B 的键索引 ✓）
    const int t    = lane & 3;                             // 0..3（列对 / d 分段 ✓）
    const int bh   = (int)blockIdx.y;
    const size_t base = (size_t)bh * (size_t)L * (size_t)d;

    const int row0 = (int)blockIdx.x * FA_TC_BR + warp * 16;   // 本 warp 首行（16 行/warp ✓）
    const int r0   = row0 + g;                                 // 本 lane 第 1 行 ✓
    const int r1   = row0 + g + 8;                             // 本 lane 第 2 行 ✓
    // ⚠️ 旧的 per-lane "d/4 分段"（dg / ds0）已随 PV mma 退役 ✗→✓：
    //   O 现在直接以 **PV mma 的 C 碎片** 布局持有（每 lane 每 n8 瓦片 2 列 ✓），不再需要手写分段 ✓

    // 块统一的 j 上界（causal ⇒ 到本 block 最后一行 ✓；循环内有 __syncthreads ⇒ 必须块统一 ✓）
    int jmax_block = L;
    if (causal) {
        const int last = (int)blockIdx.x * FA_TC_BR + FA_TC_BR - 1;
        jmax_block = ((last < L) ? last : (L - 1)) + 1;
    }

    // ---- A(Q) 碎片：本 lane 的 2 行 × d（每 k8 块 4 值 ✓），只载一次 ✓ ----
    const int nkb = d / 8;                          // d=32 ⇒ 4 个 k8 块 ✓（d ≤ 64 ✓）
    unsigned aq[8][4];
    for (int kb = 0; kb < nkb; ++kb) {
        const int k8 = kb * 8;
        const float q0a = (r0 < L) ? Q[base + (size_t)r0 * d + k8 + t]     : 0.f;
        const float q0b = (r0 < L) ? Q[base + (size_t)r0 * d + k8 + t + 4] : 0.f;
        const float q1a = (r1 < L) ? Q[base + (size_t)r1 * d + k8 + t]     : 0.f;
        const float q1b = (r1 < L) ? Q[base + (size_t)r1 * d + k8 + t + 4] : 0.f;
        aq[kb][0] = tf32_bits(q0a);
        aq[kb][1] = tf32_bits(q1a);
        aq[kb][2] = tf32_bits(q0b);
        aq[kb][3] = tf32_bits(q1b);
    }

    // ---- 在线 softmax 的每行状态（本 lane 的 2 行 ✓；行内 4 lane 一致 ✓）----
    float mr0 = -INFINITY, mr1 = -INFINITY;         // running max
    float lr0 = 0.f, lr1 = 0.f;                     // 局部 Σ exp（最后一次性归约 ✓）
    // 【②-TC/PV】O 累加器 = **PV mma 的 C 碎片**（每个 n8(d) 瓦片 4 值 ✓）：
    //   accO[v][0]=(r0, v*8+2t) [1]=(r0, v*8+2t+1) [2]=(r1, v*8+2t) [3]=(r1, v*8+2t+1) ✓
    //   d ≤ 64（host 已保证 ✓）⇒ 瓦片数 ≤ 8 ✓
    float accO[8][4];
#pragma unroll
    for (int v = 0; v < 8; ++v) { accO[v][0] = 0.f; accO[v][1] = 0.f; accO[v][2] = 0.f; accO[v][3] = 0.f; }

    for (int j0 = 0; j0 < jmax_block; j0 += FA_TC_BN) {
        const int nb = ((j0 + FA_TC_BN) < jmax_block) ? FA_TC_BN : (jmax_block - j0);
        // K/V tile：**整块**（未用行写 0 ✓ —— 防 stale smem 的 inf/NaN 污染 ✗）
        for (int tt = tid; tt < FA_TC_BN * d; tt += 128) {
            const int r = tt / d, c = tt % d;
            const bool ok = (r < nb);
            const size_t off = base + (size_t)(j0 + r) * d + c;
            sK[r * STRIDE + c] = ok ? K[off] : 0.f;
            sV[r * STRIDE + c] = ok ? V[off] : 0.f;
        }
        __syncthreads();

        const int nsteps = (nb + 7) / 8;
        for (int n8 = 0; n8 < nsteps; ++n8) {
            const int kbase = j0 + n8 * 8;                  // 本 n8 步的首个 key（全局 ✓）
            // ---- S 瓦片 = Q(16×32)·Kᵀ(32×8)：4 个 k8 块各 1 次 mma ✓ ----
            float cf[4] = {0.f, 0.f, 0.f, 0.f};
            const int bn_row = n8 * 8 + g;                  // B 的行 = tile 内 key 号 ✓
            for (int kb = 0; kb < nkb; ++kb) {
                const int k8 = kb * 8;
                unsigned bf[2];
                bf[0] = tf32_bits(sK[bn_row * STRIDE + k8 + t]);
                bf[1] = tf32_bits(sK[bn_row * STRIDE + k8 + t + 4]);
                mma_t32(cf, aq[kb], bf);
            }

            // ---- scale → bias → 掩码（顺序同 v2：先 scale 再 bias ✓）----
            cf[0] *= scale; cf[1] *= scale; cf[2] *= scale; cf[3] *= scale;
            const int cj0 = kbase + 2 * t;                  // 本 lane 持有的两列（key ✓）
            const int cj1 = cj0 + 1;
            if (Bias) {
                cf[0] += Bias[((size_t)bh * (size_t)L + (size_t)r0) * (size_t)L + (size_t)cj0];
                cf[1] += Bias[((size_t)bh * (size_t)L + (size_t)r0) * (size_t)L + (size_t)cj1];
                cf[2] += Bias[((size_t)bh * (size_t)L + (size_t)r1) * (size_t)L + (size_t)cj0];
                cf[3] += Bias[((size_t)bh * (size_t)L + (size_t)r1) * (size_t)L + (size_t)cj1];
            }
            if (causal) {                                   // 对角块逐元素掩码 ✓
                if (cj0 > r0) cf[0] = -INFINITY;
                if (cj1 > r0) cf[1] = -INFINITY;
                if (cj0 > r1) cf[2] = -INFINITY;
                if (cj1 > r1) cf[3] = -INFINITY;
            }
            const bool k0ok = (n8 * 8 + 2 * t)     < nb;    // tile 尾部越界列 ✓
            const bool k1ok = (n8 * 8 + 2 * t + 1) < nb;
            if (!k0ok) { cf[0] = -INFINITY; cf[2] = -INFINITY; }
            if (!k1ok) { cf[1] = -INFINITY; cf[3] = -INFINITY; }
            if (r0 >= L) { cf[0] = -INFINITY; cf[1] = -INFINITY; }
            if (r1 >= L) { cf[2] = -INFINITY; cf[3] = -INFINITY; }

            // ---- 行 max：四元组内 2 步 shuffle ✓ ----
            float mx0 = fmaxf(cf[0], cf[1]);
            float mx1 = fmaxf(cf[2], cf[3]);
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));

            // ---- 在线 softmax：本 lane 只管自己那 2 列（Σ 最后再归约 ✓）----
            const float mn0 = fmaxf(mr0, mx0);
            const float al0 = (mr0 == -INFINITY) ? 0.f : expf(mr0 - mn0);
            const float be0 = (cf[0] == -INFINITY) ? 0.f : expf(cf[0] - mn0);
            const float be1 = (cf[1] == -INFINITY) ? 0.f : expf(cf[1] - mn0);
            lr0 = lr0 * al0 + be0 + be1;
            mr0 = mn0;

            const float mn1 = fmaxf(mr1, mx1);
            const float al1 = (mr1 == -INFINITY) ? 0.f : expf(mr1 - mn1);
            const float bf0 = (cf[2] == -INFINITY) ? 0.f : expf(cf[2] - mn1);
            const float bf1 = (cf[3] == -INFINITY) ? 0.f : expf(cf[3] - mn1);
            lr1 = lr1 * al1 + bf0 + bf1;
            mr1 = mn1;
            // O 的 C 碎片按行缩放（两个 alpha 都就绪后统一做 ✓；行内 4 lane 的 al 一致 ✓）
            for (int v = 0; v < nkb; ++v) {
                accO[v][0] *= al0; accO[v][1] *= al0;
                accO[v][2] *= al1; accO[v][3] *= al1;
            }

            // ---- 【②-TC/PV】PV **走 mma** ✓（v2：把 P 从 C 碎片 repack 成 A 碎片，B = V 的 col-major 碎片 ✓）----
            //   m16n8k8：A(m×k) 行主序、B(n×k) col-major ⇒ 与 QKᵀ 的 B 用法同构 ✓（B 碎片 = {B[n=g][k=t], B[n=g][k=t+4]} ✓）
            //     · A(P)：{a0=(g,t) a1=(g+8,t) a2=(g,t+4) a3=(g+8,t+4)}，其中 t = **key 号** ✓
            //     · B(V)：n = 头维（v*8+g ✓）、k = key（t / t+4 ✓）⇒ 直接读 sV[key][dim] ✓
            //   P 的 repack：C 碎片里每 lane 只有 key 列 2t/2t+1 ⇒ **2 次 shuffle**（取 be0/be1 各一次）
            //   再按 t 的奇偶本地选 ✓（⚠️ 不能把"选 be0 还是 be1"放进被 shuffle 的表达式 ✗ ——
            //     那样用的是**源 lane** 的奇偶，等于错列 ✗）
            const int srcA = (lane & ~3) | (t >> 1);         // 持有 key = t   的 lane ✓
            const int srcB = (lane & ~3) | ((t >> 1) + 2);   // 持有 key = t+4 的 lane ✓
            const bool tpar = (t & 1) != 0;
            const float e0A = __shfl_sync(0xffffffffu, be0, srcA);
            const float e1A = __shfl_sync(0xffffffffu, be1, srcA);
            const float e0B = __shfl_sync(0xffffffffu, be0, srcB);
            const float e1B = __shfl_sync(0xffffffffu, be1, srcB);
            const float f0A = __shfl_sync(0xffffffffu, bf0, srcA);
            const float f1A = __shfl_sync(0xffffffffu, bf1, srcA);
            const float f0B = __shfl_sync(0xffffffffu, bf0, srcB);
            const float f1B = __shfl_sync(0xffffffffu, bf1, srcB);
            unsigned aP[4];
            aP[0] = tf32_bits(tpar ? e1A : e0A);   // P(r0, key=t)    ✓
            aP[1] = tf32_bits(tpar ? f1A : f0A);   // P(r1, key=t)    ✓
            aP[2] = tf32_bits(tpar ? e1B : e0B);   // P(r0, key=t+4)  ✓
            aP[3] = tf32_bits(tpar ? f1B : f0B);   // P(r1, key=t+4)  ✓
#pragma unroll
            for (int v = 0; v < nkb; ++v) {
                unsigned bv[2];
                bv[0] = tf32_bits(sV[(size_t)(n8 * 8 + t)     * STRIDE + (v * 8 + g)]);
                bv[1] = tf32_bits(sV[(size_t)(n8 * 8 + t + 4) * STRIDE + (v * 8 + g)]);
                mma_t32(accO[v], aP, bv);
            }
        }
        __syncthreads();     // 覆写 tile 前全员读完 ✓
    }

    // ---- 收尾：Σ 一次性四元组归约 ✓；写 O（**C 碎片**逐 n8(d) 瓦片 ✓）与 LSE（t==0 ✓）----
    lr0 += __shfl_xor_sync(0xffffffffu, lr0, 1);
    lr0 += __shfl_xor_sync(0xffffffffu, lr0, 2);
    lr1 += __shfl_xor_sync(0xffffffffu, lr1, 1);
    lr1 += __shfl_xor_sync(0xffffffffu, lr1, 2);

    const float inv0 = (lr0 > 0.f) ? (1.0f / (lr0 + eps)) : 0.f;   // 全屏蔽行 ⇒ inv=0 ⇒ O 全 0 ✓
    const float inv1 = (lr1 > 0.f) ? (1.0f / (lr1 + eps)) : 0.f;
#pragma unroll
    for (int v = 0; v < nkb; ++v) {
        if (r0 < L) {
            float* o = O + base + (size_t)r0 * d + v * 8 + 2 * t;
            o[0] = accO[v][0] * inv0;
            o[1] = accO[v][1] * inv0;
        }
        if (r1 < L) {
            float* o = O + base + (size_t)r1 * d + v * 8 + 2 * t;
            o[0] = accO[v][2] * inv1;
            o[1] = accO[v][3] * inv1;
        }
    }
    if (t == 0 && LSE) {
        if (r0 < L) LSE[(size_t)bh * (size_t)L + (size_t)r0] = (lr0 > 0.f) ? (mr0 + logf(lr0 + eps)) : -INFINITY;
        if (r1 < L) LSE[(size_t)bh * (size_t)L + (size_t)r1] = (lr1 > 0.f) ? (mr1 + logf(lr1 + eps)) : -INFINITY;
    }
#else
    (void)Q; (void)K; (void)V; (void)Bias; (void)O; (void)LSE;
    (void)L; (void)d; (void)scale; (void)causal; (void)eps;
#endif
}

// ============================================================
// 反向（Step ③）：flash_attn_backward_cuda —— 与 CPU 版同数学 ✓
//   P_ij = exp(S_ij − LSE_i)（与 forward 的 (l+eps) 归一化严格一致 ✓）
//   D_i  = Σ_d dO·O；dS = P·(dP − D)；dQ/dK 带 scale ✓、dV 不带 ✓、dbias = dS ✓
//
//   v1 线程分工（同前向：**一线程一行 query** ✓）：
//     · qreg / doreg 常驻寄存器；dQ 累加器 accQ 也在寄存器 ⇒ **dQ 完全无原子 ✓**
//     · K/V tile 进 smem（同上：同指令全线程读同一地址 ⇒ 广播、无 bank conflict ✓）
//     · dK/dV 用 **atomicAdd** 直连 global（j 轴跨线程共享 ⇒ 只能原子 ✓）
//     · dbias 每个 (i,j) 恰被一个线程写 ⇒ **直接赋值** ✓（越界/掩码位由 host memset 保证 0 ✓）
//
//   ★★ ②-v3 实验结论：**smem 分级归约（sDK/sDV + 每 tile flush）实测更慢 ✗ ⇒ 已回退 ★★**
//     动机：v1 的逻辑原子数 = 2·d·L²·h·B（真 MSA 形状 ≈ 3.5e9 ✗）看起来是瓶颈 ✗。
//     实测（本地 RTX 2050，B=1 h=8 L=103 d=32 causal）：
//       v1（直连 global atomicAdd）：**1.42 ms**
//       v3（smem 原子累加 + 每 tile 一次 block 级 flush，全局原子 ÷128）：**11.74 ms** ✗✗（慢 8.3×）
//     原因：① **smem 原子同 warp 同地址无法被硬件聚合**（32 路串行 ✗），而 **global 原子在同 warp
//             同地址时会被硬件聚合成 1 次 L2 原子**（sm_70+ ✓）⇒ v1 的"逻辑 3.5e9"实际 ≈ 3.5e9/32
//             ≈ 1.1e8 次硬件原子 ✓，本来就不贵 ✓；
//           ② smem 从 16 KB 涨到 32 KB ⇒ **占用率减半** ✗；③ 每 tile 多一次 __syncthreads ✓。
//     ⇒ **结论：本项目反向的瓶颈不是原子** ✗（是"一线程一行"的计算布局/占用率 ✗，见 ②-v2 待办 ✓）；
//        留此注释防止后人重做这个"优化" ✗。
//    head_dim ≤ 32：寄存器 qreg+doreg+accQ ≈ 3d 个 float ⇒ d>32 会溢出 ✗（返回 1 回落 CPU ✓）
//    LSE == -inf 的行（全屏蔽）：P≡0 ⇒ 该行梯度 0 ⇒ 跳过 ✓
//    dK/dV 原子 ⇒ 求和顺序非确定 ⇒ 与 CPU 对拍容差 ~1e-3 ✓
//    smem 用量 = 2 张 [FA_BC × 32] 表 = 16 KB ✓（32 KB 会掉占用率 ✗）
// ============================================================
constexpr int FA_DB_D_MAX = 32;

__global__ void flash_attn_bwd_kernel(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      const float* __restrict__ Bias,
                                      const float* __restrict__ O,
                                      const float* __restrict__ dO,
                                      const float* __restrict__ LSE,
                                      float* __restrict__ dQ,
                                      float* __restrict__ dK,
                                      float* __restrict__ dV,
                                      float* __restrict__ dBias,
                                      float* __restrict__ Dbuf,      // 【②-B2】输出 D_i = Σ_d dO·O（[L,h,B] ✓）
                                      const int L, const int d,
                                      const float scale, const int causal) {
    // ============================================================
    // 【②-v2 第二步（反向）】warp 内 d 维切分（与前向同一布局 ✓）
    //   动机（实测 ✓）：第一步（ILP=8）对反向**几乎无效**（1.41 → 1.45 ms ✗）；
    //     根因是**指令发射量**：一线程一行时每 (i,j) 每 lane 要发 ~5d = 160 条
    //     （2 个点积 + accQ + dK 原子 + dV 原子，各 d 条 ✗）；切分后每 lane 只发 ~5·(d/4) = 40 条 ✓。
    //   布局（同前向 ✓）：block = 128 线程 = 4 warps ⇒ 每 block **32 行**（每 warp 8 行 ✓）
    //     lane = row_in_warp(0..7)*4 + g(0..3)；每 lane 负责 d/4 个维（起始 c0 = g*dg ✓）
    //   归约：s 与 dp 都是"行内 d 维点积" ⇒ 各 2 次 __shfl_xor（1、2 步 ✓）⇒ 同行 4 lane 结果一致
    //     ⇒ p / ds / D 在同行 4 lane 上天然一致 ✓（dbias 只让 g==0 写 ✓）
    //   ⚠️ 语义与 v1 一致：causal 掩码 j>i ⇒ p=0 ✓；lse=-inf 的行（全屏蔽）⇒ p=0 ⇒ 梯度 0 ✓
    //   ⚠️ 屏障安全：tile 循环上界用**块统一**的 jmax_block ✓；causal 省算只用 **warp 统一**的 break ✓
    //      （前向踩过：tile 上界按 warp 取 ⇒ 屏障不匹配 = UB ⇒ 结果错 ✗）
    //   ⚠️ dK/dV 仍是 global atomicAdd（v3 的 smem 分级归约实测慢 8.3× ✗ 别改 ✓）；
    //      同行 4 lane 写**不同 c 段** ✓、同址的 8 个行 lane 会被硬件聚合 ✓
    // ============================================================
    constexpr int PAD = 4;                    // smem 行跨距补 4（16B 对齐 + 消 bank 冲突 ✓）
    const int STRIDE = d + PAD;
    extern __shared__ float fsmem[];
    float* sK = fsmem;                        // [FA_BC][STRIDE]
    float* sV = fsmem + FA_BC * STRIDE;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row_in_warp = lane >> 2;        // 0..7
    const int g            = lane & 3;        // d 分组 0..3
    const int bh   = (int)blockIdx.y;
    const int row0 = (int)blockIdx.x * (8 * 4);           // 每 block 32 行 ✓
    const int i    = row0 + warp * 8 + row_in_warp;
    const bool active = (i < L);
    const size_t base = (size_t)bh * (size_t)L * (size_t)d;
    const int dg = d / 4;                                 // 本 lane 的维数（≤ 16 ✓）
    const int c0 = g * dg;

    int jmax_block = L;
    if (causal) {
        const int last_row = (row0 + 31 < L) ? (row0 + 31) : (L - 1);
        jmax_block = last_row + 1;
    }
    const int iwarp_max = ((row0 + warp * 8 + 7) < L) ? (row0 + warp * 8 + 7) : (L - 1);

    const float lse  = (active && LSE) ? LSE[(size_t)bh * (size_t)L + (size_t)i] : -INFINITY;
    const bool  live = active && (lse > -INFINITY);       // 全屏蔽行 ⇒ 整行梯度 0 ⇒ 跳过 ✓

    // ---- 本 lane 的行切片 + 累加器（dg ≤ 16 ⇒ 寄存器大减 ✓）----
    float qr[16], dOr[16], accQ[16];
    float dpart = 0.f;                                    // D_i = Σ_d dO·O 的局部和（再 2 次 shuffle ✓）
    for (int t = 0; t < dg; ++t) {
        const int c = c0 + t;
        qr[t]   = live ? Q [base + (size_t)i * d + c] : 0.f;
        dOr[t]  = live ? dO[base + (size_t)i * d + c] : 0.f;
        accQ[t] = 0.f;
        if (live) dpart += dOr[t] * O[base + (size_t)i * d + c];
    }
    dpart += __shfl_xor_sync(0xffffffffu, dpart, 1);
    dpart += __shfl_xor_sync(0xffffffffu, dpart, 2);
    const float D = dpart;                                // 同行 4 lane 一致 ✓
    // 【②-B2】把 D_i 导出（B2 的 dK/dV kernel 需要它；否则每个 j 分组都要重算 8 次 ✗）
    if (active && g == 0 && Dbuf) Dbuf[(size_t)bh * (size_t)L + (size_t)i] = D;

    for (int j0 = 0; j0 < jmax_block; j0 += FA_BC) {
        const int nb = ((j0 + FA_BC) < jmax_block) ? FA_BC : (jmax_block - j0);
        for (int t = tid; t < nb * d; t += 128) {
            const int r = t / d, c = t % d;
            const size_t off = base + (size_t)(j0 + r) * d + c;
            sK[r * STRIDE + c] = K[off];
            sV[r * STRIDE + c] = V[off];
        }
        __syncthreads();

        const int j1 = j0 + nb;
        for (int j = j0; j < j1; ++j) {
            if (causal && j > iwarp_max) break;           // warp 统一 ⇒ 无屏障风险 ✓

            // ---- 本 lane 的局部 S（4 累加器 ✓）⇒ 2 次 shuffle 成全行 s ✓ ----
            const float* krow = &sK[(size_t)(j - j0) * STRIDE + c0];
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            int t = 0;
            for (; t + 4 <= dg; t += 4) {
                a0 += qr[t + 0] * krow[t + 0];
                a1 += qr[t + 1] * krow[t + 1];
                a2 += qr[t + 2] * krow[t + 2];
                a3 += qr[t + 3] * krow[t + 3];
            }
            for (; t < dg; ++t) a0 += qr[t] * krow[t];
            float s = (a0 + a1) + (a2 + a3);
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            s *= scale;
            if (Bias && live) s += Bias[((size_t)bh * (size_t)L + (size_t)i) * (size_t)L + (size_t)j];
            if (causal && j > i) s = -INFINITY;           // 对角块逐元素掩码 ✓

            // ---- P（与 forward 同一 LSE 口径 ✓）；非 live 行 ⇒ p=0（避免 -inf−(-inf)=NaN ✗）----
            const float p = live ? expf(s - lse) : 0.f;

            // ---- 本 lane 的局部 dP ⇒ 2 次 shuffle ✓ ----
            const float* vrow = &sV[(size_t)(j - j0) * STRIDE + c0];
            float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
            t = 0;
            for (; t + 4 <= dg; t += 4) {
                b0 += dOr[t + 0] * vrow[t + 0];
                b1 += dOr[t + 1] * vrow[t + 1];
                b2 += dOr[t + 2] * vrow[t + 2];
                b3 += dOr[t + 3] * vrow[t + 3];
            }
            for (; t < dg; ++t) b0 += dOr[t] * vrow[t];
            float dp = (b0 + b1) + (b2 + b3);
            dp += __shfl_xor_sync(0xffffffffu, dp, 1);
            dp += __shfl_xor_sync(0xffffffffu, dp, 2);

            const float ds = p * (dp - D);                // 同行 4 lane 一致 ✓

            // ---- 梯度累加（dK/dV 直连 global 原子 ✓ —— 同 warp 同地址硬件聚合 ✓）----
            if (live) {
                if (g == 0 && dBias) {
                    dBias[((size_t)bh * (size_t)L + (size_t)i) * (size_t)L + (size_t)j] = ds;
                }
                for (int q = 0; q < dg; ++q) accQ[q] += scale * ds * krow[q];
                if (dK || dV) {
                    const size_t off_j = base + (size_t)j * d + c0;
                    for (int q = 0; q < dg; ++q) {
                        if (dK) atomicAdd(&dK[off_j + q], scale * ds * qr[q]);
                        if (dV) atomicAdd(&dV[off_j + q], p * dOr[q]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---- dQ 写回（每 lane 写自己那 dg 个元素 ✓ 无原子 ✓；非 live 行由 host memset 保 0 ✓）----
    if (live && dQ) {
        float* dq = dQ + base + (size_t)i * d + c0;
        for (int q = 0; q < dg; ++q) dq[q] = accQ[q];
    }
}

// ============================================================
// 【②-B2】dK/dV 的**无原子** GEMM 型 kernel（2026-09-23）
//   背景（实测 ✓）：原先 dK/dV 融进 B1 用 global atomicAdd ⇒ **占反向 83% 时间** ✗
//     （真 MSA 形状拆解：dQ-only 112 ms / +dK +285 ms / +dV +254 ms ✓）；
//     而反向的 lane 切分把"每 lane 指令数" ÷4 ✓，却被"硬件原子聚合倍数 32→8"的反向效应抵消 ✗（净 1.3× ✓）。
//   ⇒ 正解 = **消灭原子**：把 dK/dV 做成独立 kernel —— 每个 block 独占一块 (j, dd) ⇒ 沿 i 归约
//     ⇒ 每个 (j,dd) 元素**恰好被写一次** ✓ 零原子 ✓（代价 = 重算一遍 s/p/dp ✓，与 B1 同量级 ✓）。
//   布局：block = 128 线程 = 4 warps ⇒ 每 block **32 个 key**（每 warp 8 key ✓）
//     lane = j_sub(0..7)*4 + g(0..3) ⇒ 同一 key 的 4 lane 连续 ⇒ s / dp 各 2 次 shuffle ✓
//     每 lane 在寄存器里累 dK/dV 的 d/4 个分量 ✓
//   沿 i 按 BI=32 行分块：Q/dO 协作进 smem（带 PAD ✓）；K/V 只载一次并常驻**寄存器** ✓
//   依赖 B1 输出的 `Dbuf`（D_i = Σ_d dO·O ✓）：否则每个 j 分组都要重复算 D（8× 冗余 ✗）
//   ⚠️ 屏障安全：i 循环范围只依赖 blockIdx / L ✓（causal 的起点 j_min 也是块统一 ✓）
//   ⚠️ d 需为 4 的倍数（lane 切片 16B 对齐 ✓）——host/`supports_op` 已把关 ✓
// ============================================================
constexpr int FA_DKV_BNJ = 32;   // 每 block 的 key 数
constexpr int FA_DKV_BI  = 32;   // i 方向分块行数

__global__ void flash_attn_bwd_dkv_kernel(const float* __restrict__ Q,
                                          const float* __restrict__ K,
                                          const float* __restrict__ V,
                                          const float* __restrict__ Bias,
                                          const float* __restrict__ dO,
                                          const float* __restrict__ LSE,
                                          const float* __restrict__ Dbuf,
                                          float* __restrict__ dK,
                                          float* __restrict__ dV,
                                          const int L, const int d,
                                          const float scale, const int causal) {
    constexpr int PAD = 4;
    const int STRIDE = d + PAD;
    extern __shared__ float fsmem[];
    float* sK  = fsmem;                                  // [FA_DKV_BNJ][STRIDE]
    float* sV  = sK + FA_DKV_BNJ * STRIDE;               // [FA_DKV_BNJ][STRIDE]
    float* sQ  = sV + FA_DKV_BNJ * STRIDE;               // [FA_DKV_BI][STRIDE]
    float* sDO = sQ + FA_DKV_BI * STRIDE;                // [FA_DKV_BI][STRIDE]

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int jsub = lane >> 2;                          // 0..7
    const int g    = lane & 3;                           // d 分组
    const int bh   = (int)blockIdx.y;
    const size_t base = (size_t)bh * (size_t)L * (size_t)d;
    const int dg = d / 4;
    const int c0 = g * dg;
    const int j  = (int)blockIdx.x * FA_DKV_BNJ + warp * 8 + jsub;   // 本 lane 负责的 key ✓
    const bool jok = (j < L);

    // ---- K/V：本 block 的 32 行，载一次（同时写 smem 与寄存器切片 ✓）----
    for (int t = tid; t < FA_DKV_BNJ * d; t += 128) {
        const int r = t / d, c = t % d;
        const int jj = (int)blockIdx.x * FA_DKV_BNJ + r;
        sK[r * STRIDE + c] = (jj < L) ? K[base + (size_t)jj * d + c] : 0.f;
        sV[r * STRIDE + c] = (jj < L) ? V[base + (size_t)jj * d + c] : 0.f;
    }
    __syncthreads();                                     // smem 就绪才能取自己的切片 ✓

    const float* krow0 = &sK[(warp * 8 + jsub) * STRIDE + c0];
    const float* vrow0 = &sV[(warp * 8 + jsub) * STRIDE + c0];
    float kr[16], vr[16];
    for (int q = 0; q < dg; ++q) { kr[q] = krow0[q]; vr[q] = vrow0[q]; }   // 常驻寄存器，沿 i 复用 ✓

    float aK[16], aV[16];
    for (int q = 0; q < dg; ++q) { aK[q] = 0.f; aV[q] = 0.f; }

    // causal ⇒ 只有 i ≥ j 才贡献 ⇒ i 从本 block 的首个 key 起（块统一起点 ✓）
    int i_start = 0;
    if (causal) {
        i_start = ((int)blockIdx.x * FA_DKV_BNJ / FA_DKV_BI) * FA_DKV_BI;
        if (i_start > L) i_start = L;
    }

    for (int i0 = i_start; i0 < L; i0 += FA_DKV_BI) {
        const int nib = ((i0 + FA_DKV_BI) < L) ? FA_DKV_BI : (L - i0);
        for (int t = tid; t < nib * d; t += 128) {
            const int r = t / d, c = t % d;
            const size_t off = base + (size_t)(i0 + r) * d + c;
            sQ [r * STRIDE + c] = Q [off];
            sDO[r * STRIDE + c] = dO[off];
        }
        __syncthreads();

        for (int ii = 0; ii < nib; ++ii) {
            const int i = i0 + ii;
            const float* qrow = &sQ [(size_t)ii * STRIDE + c0];
            const float* drow = &sDO[(size_t)ii * STRIDE + c0];

            // ---- s_ij = scale·(q_i·k_j)：局部和 + 2 次 shuffle ✓ ----
            float s = 0.f;
            for (int q = 0; q < dg; ++q) s += qrow[q] * kr[q];
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            s *= scale;
            if (Bias && jok) s += Bias[((size_t)bh * (size_t)L + (size_t)i) * (size_t)L + (size_t)j];
            if (causal && j > i) s = -INFINITY;

            const float lse = LSE[(size_t)bh * (size_t)L + (size_t)i];
            const float p = (jok && lse > -INFINITY) ? expf(s - lse) : 0.f;

            // ---- dP_ij = dO_i·v_j：局部和 + 2 次 shuffle ✓ ----
            float dp = 0.f;
            for (int q = 0; q < dg; ++q) dp += drow[q] * vr[q];
            dp += __shfl_xor_sync(0xffffffffu, dp, 1);
            dp += __shfl_xor_sync(0xffffffffu, dp, 2);

            const float D  = Dbuf[(size_t)bh * (size_t)L + (size_t)i];   // B1 已算好 ✓
            const float ds = p * (dp - D);

            if (jok) {
                if (dK) for (int q = 0; q < dg; ++q) aK[q] += scale * ds * qrow[q];
                if (dV) for (int q = 0; q < dg; ++q) aV[q] += p * drow[q];
            }
        }
        __syncthreads();                                 // 覆盖 sQ/sDO 前全员用完 ✓
    }

    // ---- 写回：每个 (j,dd) 恰好一个 lane 写 ⇒ **无原子** ✓ ----
    if (jok) {
        const size_t off = base + (size_t)j * d + c0;
        for (int q = 0; q < dg; ++q) {
            if (dK) dK[off + q] = aK[q];
            if (dV) dV[off + q] = aV[q];
        }
    }
}

// ============================================================
// 【②-TC/B2】dK/dV 的 **tensor-core 版**（2026-09-24）—— 与 `flash_attn_bwd_dkv_kernel` 同数学 ✓ 零原子 ✓
//   ★ 关键写法：**转置计算** `Sᵀ = K·Qᵀ`（M = keys ✓、N = queries ✓、K = d ✓）
//     ⇒ 输出 (keys × queries) 直接落在 C 碎片里 ✓ ⇒ 再用**与前向 PV 完全相同的 repack**
//       （C 碎片 → A 碎片：每个值 2 次 shuffle + 本地按 query 奇偶选 ✓）得到 dK/dV mma 的 A 碎片 ✓
//     ⇒ **不需要 movmatrix、也不需要额外的转置流水** ✓✓（tf32 没有 movmatrix ✗）
//   ★ B 操作数按下标**随机取**：Sᵀ 用 Q（smem ✓）、dPᵀ 用 dO（smem ✓）、dK 用 Q ✓、dV 用 dO ✓ ⇒ 无 shuffle ✓
//   布局：block = 128 线程 = 4 warps ⇒ **每 warp 16 个 key**（每 block 64 key ✓）
//     lane：g = lane>>2（0..7 = M 行组/key ✓）、t = lane&3（列对 = query 的 2t/2t+1 ✓ / d 的 k8 段 ✓）
//   ⚠️ 仅支持 d ≤ 32（`FA_DB_D_MAX` ✓）且 d % 8 == 0（TC 要求 ✓）⇒ nkb = d/8 ≤ 4 ⇒ 寄存器可控 ✓
//   ⚠️ 尾部 query 行**必须零填充 smem** ＋ 逐元素把 query ≥ nib 的 p 置 0 ✓（否则 stale 值污染其它 key ✗）
//   ⚠️ 屏障安全：i 循环上界只依赖 blockIdx/L ✓；含 __syncthreads 的循环上界块统一 ✓
// ============================================================
constexpr int FA_DKV_TC_BNJ = 64;   // 每 block 的 key 数（4 warps × 16 ✓）
constexpr int FA_DKV_TC_BI  = 32;   // i（query）方向分块行数

__global__ void flash_attn_bwd_dkv_tc_kernel(const float* __restrict__ Q,
                                             const float* __restrict__ K,
                                             const float* __restrict__ V,
                                             const float* __restrict__ Bias,
                                             const float* __restrict__ dO,
                                             const float* __restrict__ LSE,
                                             const float* __restrict__ Dbuf,
                                             float* __restrict__ dK,
                                             float* __restrict__ dV,
                                             const int L, const int d,
                                             const float scale, const int causal) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int PAD = 4;
    const int STRIDE = d + PAD;
    extern __shared__ float fsmem[];
    float* sQ  = fsmem;                              // [FA_DKV_TC_BI][STRIDE]
    float* sDO = sQ + FA_DKV_TC_BI * STRIDE;         // [FA_DKV_TC_BI][STRIDE]

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int bh   = (int)blockIdx.y;
    const size_t base = (size_t)bh * (size_t)L * (size_t)d;
    const int nkb  = d / 8;                          // ≤ 4 ✓

    const int j0 = (int)blockIdx.x * FA_DKV_TC_BNJ + warp * 16;
    const int jA = j0 + g;                           // C 碎片的前 8 行 ✓
    const int jB = j0 + g + 8;                       // 后 8 行 ✓
    const bool okA = (jA < L), okB = (jB < L);

    // ---- A(K)/A(V)：16 key × d 的 k8 块碎片（载一次，沿 i 复用 ✓）----
    unsigned aK[4][4], aV[4][4];
    for (int kb = 0; kb < nkb; ++kb) {
        const int k8 = kb * 8;
        const int oA = k8 + t, oB = k8 + t + 4;
        const size_t rA = base + (size_t)jA * d, rB = base + (size_t)jB * d;
        aK[kb][0] = tf32_bits(okA ? K[rA + oA] : 0.f);
        aK[kb][1] = tf32_bits(okB ? K[rB + oA] : 0.f);
        aK[kb][2] = tf32_bits(okA ? K[rA + oB] : 0.f);
        aK[kb][3] = tf32_bits(okB ? K[rB + oB] : 0.f);
        aV[kb][0] = tf32_bits(okA ? V[rA + oA] : 0.f);
        aV[kb][1] = tf32_bits(okB ? V[rB + oA] : 0.f);
        aV[kb][2] = tf32_bits(okA ? V[rA + oB] : 0.f);
        aV[kb][3] = tf32_bits(okB ? V[rB + oB] : 0.f);
    }

    // ---- dK/dV 累加器（PV/dK/dV mma 的 C 碎片：**(key, d 列)** ✓）----
    float accK[4][4], accV[4][4];
#pragma unroll
    for (int v = 0; v < 4; ++v)
#pragma unroll
        for (int q = 0; q < 4; ++q) { accK[v][q] = 0.f; accV[v][q] = 0.f; }

    int i_start = 0;
    if (causal) {
        i_start = (int)blockIdx.x * FA_DKV_TC_BNJ;    // 本 block 的首个 key ✓（块统一 ✓）
        if (i_start > L) i_start = L;
    }

    const int  srcA = (lane & ~3) | (t >> 1);         // 持有 query = t   的四元组 lane ✓
    const int  srcB = (lane & ~3) | ((t >> 1) + 2);   // 持有 query = t+4 的 ✓
    const bool tpar = (t & 1) != 0;

    for (int i0 = i_start; i0 < L; i0 += FA_DKV_TC_BI) {
        const int nib = ((i0 + FA_DKV_TC_BI) < L) ? FA_DKV_TC_BI : (L - i0);
        // ⚠️ 尾部行零填充（否则 stale smem 会被 mma 当真值 ⇒ 污染其它 key 的累加 ✗）
        for (int tt = tid; tt < FA_DKV_TC_BI * d; tt += 128) {
            const int r = tt / d, c = tt % d;
            const bool ok = (r < nib);
            const size_t off = base + (size_t)(i0 + r) * d + c;
            sQ [r * STRIDE + c] = ok ? Q [off] : 0.f;
            sDO[r * STRIDE + c] = ok ? dO[off] : 0.f;
        }
        __syncthreads();

        for (int ib = 0; ib < nib; ib += 8) {
            // ---- Sᵀ = K·Qᵀ（M=16 key ✓, N=8 query ✓, K=d ✓）与 dPᵀ = V·dOᵀ（同构 ✓）----
            float cf[4] = {0.f, 0.f, 0.f, 0.f};      // Sᵀ[key][query]
            float cg[4] = {0.f, 0.f, 0.f, 0.f};      // dPᵀ[key][query]
            const int qrow = ib + g;                 // B 的 n 索引 = query（tile 内 ✓）
            for (int kb = 0; kb < nkb; ++kb) {
                const int k8 = kb * 8;
                unsigned bq[2], bo[2];
                bq[0] = tf32_bits(sQ [(size_t)qrow * STRIDE + k8 + t]);
                bq[1] = tf32_bits(sQ [(size_t)qrow * STRIDE + k8 + t + 4]);
                bo[0] = tf32_bits(sDO[(size_t)qrow * STRIDE + k8 + t]);
                bo[1] = tf32_bits(sDO[(size_t)qrow * STRIDE + k8 + t + 4]);
                mma_t32(cf, aK[kb], bq);
                mma_t32(cg, aV[kb], bo);
            }

            // ---- 逐元素（口径严格照 B2 ✓）：s = scale·(k·q) + bias ⇒ 掩码 ⇒ p = exp(s − LSE) ----
            const int  i_a = i0 + ib + 2 * t, i_b = i_a + 1;            // 本 lane 的两个 query ✓（**含 ib** ✗→✓）
            const bool va = (i_a < i0 + nib) && (i_a < L);
            const bool vb = (i_b < i0 + nib) && (i_b < L);
            float s0 = cf[0] * scale, s1 = cf[1] * scale, s2 = cf[2] * scale, s3 = cf[3] * scale;
            if (Bias) {
                if (va && okA) s0 += Bias[((size_t)bh * (size_t)L + (size_t)i_a) * (size_t)L + (size_t)jA];
                if (vb && okA) s1 += Bias[((size_t)bh * (size_t)L + (size_t)i_b) * (size_t)L + (size_t)jA];
                if (va && okB) s2 += Bias[((size_t)bh * (size_t)L + (size_t)i_a) * (size_t)L + (size_t)jB];
                if (vb && okB) s3 += Bias[((size_t)bh * (size_t)L + (size_t)i_b) * (size_t)L + (size_t)jB];
            }
            float lsea = -INFINITY, lseb = -INFINITY, Da = 0.f, Db = 0.f;
            if (va) { lsea = LSE [(size_t)bh * (size_t)L + (size_t)i_a]; Da = Dbuf[(size_t)bh * (size_t)L + (size_t)i_a]; }
            if (vb) { lseb = LSE [(size_t)bh * (size_t)L + (size_t)i_b]; Db = Dbuf[(size_t)bh * (size_t)L + (size_t)i_b]; }
            if (!okA || !va || (causal && jA > i_a)) s0 = -INFINITY;
            if (!okA || !vb || (causal && jA > i_b)) s1 = -INFINITY;
            if (!okB || !va || (causal && jB > i_a)) s2 = -INFINITY;
            if (!okB || !vb || (causal && jB > i_b)) s3 = -INFINITY;
            // ⚠️ LSE 必须**逐元素对号** ✗→✓：C 碎片的四元组是
            //   c0=(key jA, query i_a) c1=(jA, i_b) c2=(jB, i_a) c3=(jB, i_b) ✓
            //   ⇒ p1 用 **lseb**、p2 用 **lsea**（写反时：非因果只差 ~1% ✓，一旦有 causal 掩码则崩 10~150% ✗）
            const float p0 = (s0 == -INFINITY || lsea == -INFINITY) ? 0.f : expf(s0 - lsea);
            const float p1 = (s1 == -INFINITY || lseb == -INFINITY) ? 0.f : expf(s1 - lseb);
            const float p2 = (s2 == -INFINITY || lsea == -INFINITY) ? 0.f : expf(s2 - lsea);
            const float p3 = (s3 == -INFINITY || lseb == -INFINITY) ? 0.f : expf(s3 - lseb);
            const float ds0 = p0 * (cg[0] - Da) * scale;    // dS = P·(dP − D)·scale（scale 折进 dK ✓）
            const float ds1 = p1 * (cg[1] - Db) * scale;
            const float ds2 = p2 * (cg[2] - Da) * scale;
            const float ds3 = p3 * (cg[3] - Db) * scale;

            // ---- repack：C 碎片 (key, query=2t/2t+1) → A 碎片 (key, query=t/t+4) ✓ ----
            //   ⚠️ 不能把"选哪个成员"放进被 shuffle 的表达式 ✗（那会用**源 lane** 的奇偶 ⇒ 取错列 ✗）
            const float p0A = __shfl_sync(0xffffffffu, p0, srcA), p1A = __shfl_sync(0xffffffffu, p1, srcA);
            const float p0B = __shfl_sync(0xffffffffu, p0, srcB), p1B = __shfl_sync(0xffffffffu, p1, srcB);
            const float p2A = __shfl_sync(0xffffffffu, p2, srcA), p3A = __shfl_sync(0xffffffffu, p3, srcA);
            const float p2B = __shfl_sync(0xffffffffu, p2, srcB), p3B = __shfl_sync(0xffffffffu, p3, srcB);
            const float q0A = __shfl_sync(0xffffffffu, ds0, srcA), q1A = __shfl_sync(0xffffffffu, ds1, srcA);
            const float q0B = __shfl_sync(0xffffffffu, ds0, srcB), q1B = __shfl_sync(0xffffffffu, ds1, srcB);
            const float q2A = __shfl_sync(0xffffffffu, ds2, srcA), q3A = __shfl_sync(0xffffffffu, ds3, srcA);
            const float q2B = __shfl_sync(0xffffffffu, ds2, srcB), q3B = __shfl_sync(0xffffffffu, ds3, srcB);
            unsigned aP[4], aD[4];
            aP[0] = tf32_bits(tpar ? p1A : p0A);   // P (jA, i0+ib+t)    ✓
            aP[1] = tf32_bits(tpar ? p3A : p2A);   // P (jB, i0+ib+t)    ✓
            aP[2] = tf32_bits(tpar ? p1B : p0B);   // P (jA, i0+ib+t+4)  ✓
            aP[3] = tf32_bits(tpar ? p3B : p2B);   // P (jB, i0+ib+t+4)  ✓
            aD[0] = tf32_bits(tpar ? q1A : q0A);   // dS(jA, i0+ib+t)    ✓
            aD[1] = tf32_bits(tpar ? q3A : q2A);   // dS(jB, i0+ib+t)    ✓
            aD[2] = tf32_bits(tpar ? q1B : q0B);   // dS(jA, i0+ib+t+4)  ✓
            aD[3] = tf32_bits(tpar ? q3B : q2B);   // dS(jB, i0+ib+t+4)  ✓

            // ---- dV = Pᵀ·dO ✓、dK = scale·dSᵀ·Q ✓（B 的 n = d 维 ✓、k = query ✓）----
#pragma unroll
            for (int v = 0; v < nkb; ++v) {
                const int dim = v * 8 + g;
                unsigned bp[2], bd[2];
                bp[0] = tf32_bits(sDO[(size_t)(ib + t)     * STRIDE + dim]);
                bp[1] = tf32_bits(sDO[(size_t)(ib + t + 4) * STRIDE + dim]);
                bd[0] = tf32_bits(sQ [(size_t)(ib + t)     * STRIDE + dim]);
                bd[1] = tf32_bits(sQ [(size_t)(ib + t + 4) * STRIDE + dim]);
                mma_t32(accV[v], aP, bp);
                mma_t32(accK[v], aD, bd);
            }
        }
        __syncthreads();                                 // 覆写 sQ/sDO 前全员用完 ✓
    }

    // ---- 写回：每个 (key, dim) 恰好一个 lane 写 ⇒ **零原子** ✓ ----
#pragma unroll
    for (int v = 0; v < nkb; ++v) {
        const int dim = v * 8 + 2 * t;
        if (okA) {
            if (dK) { float* p = dK + base + (size_t)jA * d + dim; p[0] = accK[v][0]; p[1] = accK[v][1]; }
            if (dV) { float* p = dV + base + (size_t)jA * d + dim; p[0] = accV[v][0]; p[1] = accV[v][1]; }
        }
        if (okB) {
            if (dK) { float* p = dK + base + (size_t)jB * d + dim; p[0] = accK[v][2]; p[1] = accK[v][3]; }
            if (dV) { float* p = dV + base + (size_t)jB * d + dim; p[0] = accV[v][2]; p[1] = accV[v][3]; }
        }
    }
#else
    (void)Q; (void)K; (void)V; (void)Bias; (void)dO; (void)LSE; (void)Dbuf; (void)dK; (void)dV;
    (void)L; (void)d; (void)scale; (void)causal;
#endif
}

// ============================================================
// 【②-TC/B1】dQ(+dbias+D 导出) 的 **tensor-core 版**（2026-09-24）—— 与 `flash_attn_bwd_kernel` 同数学 ✓
//   结构与**前向 TC kernel 同构**（M = queries ✓）：
//     · S  = Q·Kᵀ （A = Q 寄存器 ✓、B = K smem ✓）⇒ C = (query × key) ✓
//     · dP = dO·Vᵀ（A = dO 寄存器 ✓、B = V smem ✓）
//     · 逐元素：p = exp(scale·s + bias − LSE_i) ✓、ds = p·(dP − D_i)·scale ✓、**dbias = ds** ✓
//     · **dQ = Σ_j ds·K** ⇒ A = dS（由 C 碎片 repack ✓ —— 与前向 PV / ②-B2 用的是**同一个 repack** ✓）、
//       B = K（n = d 维 ✓、k = key ✓，按碎片下标随机取 smem ✓）
//   ⚠️⚠️ 轴与 ②-B2 相反 ✗→✓：这里的 C 碎片是 (row = **query** ✓, col = **key** ✓)
//        ⇒ **LSE / D 必须按"行"（r0/r1）取** ✓（B2 那边是按"列" ✓）—— 写反的症状同样是"非因果小错、因果大错" ✗
//   ⚠️ 仅支持 d ≤ 32 且 d%8==0（host 把关 ✓）；dK/dV 必须为 nullptr（它们由 ②-B2 负责 ✓）
//   布局：block = 128 线程 = 4 warps ⇒ **每 warp 16 行 query**、每 block 64 行 ✓；key 方向按 64 一块进 smem ✓
//   ⚠️ 尾部 key **必须零填充 smem** ✓ ＋ 逐元素把无效 key 的 p 置 0 ✓（否则污染 dQ ✗）
// ============================================================
constexpr int FA_DQ_TC_BR = 64;    // 每 block 的 query 行数（4 warps × 16 ✓）
constexpr int FA_DQ_TC_BN = 64;    // 每 tile 的 key 数（smem K/V ✓）

__global__ void flash_attn_bwd_dq_tc_kernel(const float* __restrict__ Q,
                                            const float* __restrict__ K,
                                            const float* __restrict__ V,
                                            const float* __restrict__ Bias,
                                            const float* __restrict__ O,
                                            const float* __restrict__ dO,
                                            const float* __restrict__ LSE,
                                            float* __restrict__ dQ,
                                            float* __restrict__ dBias,
                                            float* __restrict__ Dbuf,
                                            const int L, const int d,
                                            const float scale, const int causal) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int PAD = 4;
    const int STRIDE = d + PAD;
    extern __shared__ float fsmem[];
    float* sK = fsmem;                                 // [FA_DQ_TC_BN][STRIDE]
    float* sV = sK + FA_DQ_TC_BN * STRIDE;             // [FA_DQ_TC_BN][STRIDE]

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int bh   = (int)blockIdx.y;
    const size_t base = (size_t)bh * (size_t)L * (size_t)d;
    const int nkb  = d / 8;

    const int row0 = (int)blockIdx.x * FA_DQ_TC_BR + warp * 16;
    const int r0 = row0 + g, r1 = row0 + g + 8;
    const bool ok0 = (r0 < L), ok1 = (r1 < L);

    // ---- A(Q)/A(dO) 碎片（载一次 ✓）----
    unsigned aQ[4][4], aD[4][4];
    for (int kb = 0; kb < nkb; ++kb) {
        const int k8 = kb * 8;
        const size_t o0 = base + (size_t)r0 * d + k8, o1 = base + (size_t)r1 * d + k8;
        aQ[kb][0] = tf32_bits(ok0 ? Q [o0 + t]     : 0.f);
        aQ[kb][1] = tf32_bits(ok1 ? Q [o1 + t]     : 0.f);
        aQ[kb][2] = tf32_bits(ok0 ? Q [o0 + t + 4] : 0.f);
        aQ[kb][3] = tf32_bits(ok1 ? Q [o1 + t + 4] : 0.f);
        aD[kb][0] = tf32_bits(ok0 ? dO[o0 + t]     : 0.f);
        aD[kb][1] = tf32_bits(ok1 ? dO[o1 + t]     : 0.f);
        aD[kb][2] = tf32_bits(ok0 ? dO[o0 + t + 4] : 0.f);
        aD[kb][3] = tf32_bits(ok1 ? dO[o1 + t + 4] : 0.f);
    }

    // ---- D_i = Σ_d dO·O（本 lane 覆盖列 {k8+t, k8+t+4} ✓；四元组归约 ⇒ 行内 4 lane 一致 ✓）----
    float pD0 = 0.f, pD1 = 0.f;
    for (int kb = 0; kb < nkb; ++kb) {
        const int k8 = kb * 8;
        if (ok0) pD0 += dO[base + (size_t)r0*d + k8 + t]     * O[base + (size_t)r0*d + k8 + t]
                      +  dO[base + (size_t)r0*d + k8 + t + 4] * O[base + (size_t)r0*d + k8 + t + 4];
        if (ok1) pD1 += dO[base + (size_t)r1*d + k8 + t]     * O[base + (size_t)r1*d + k8 + t]
                      +  dO[base + (size_t)r1*d + k8 + t + 4] * O[base + (size_t)r1*d + k8 + t + 4];
    }
    pD0 += __shfl_xor_sync(0xffffffffu, pD0, 1);
    pD0 += __shfl_xor_sync(0xffffffffu, pD0, 2);
    pD1 += __shfl_xor_sync(0xffffffffu, pD1, 1);
    pD1 += __shfl_xor_sync(0xffffffffu, pD1, 2);
    if (t == 0 && Dbuf) {                        // 行内一致 ⇒ 每行只写一次 ✓
        if (ok0) Dbuf[(size_t)bh * (size_t)L + (size_t)r0] = pD0;
        if (ok1) Dbuf[(size_t)bh * (size_t)L + (size_t)r1] = pD1;
    }
    const float lse0 = ok0 ? LSE[(size_t)bh * (size_t)L + (size_t)r0] : -INFINITY;
    const float lse1 = ok1 ? LSE[(size_t)bh * (size_t)L + (size_t)r1] : -INFINITY;

    // ---- dQ 累加器（C 碎片：**(query, d 列)** ✓）----
    float accQ[4][4];
#pragma unroll
    for (int v = 0; v < 4; ++v)
#pragma unroll
        for (int q = 0; q < 4; ++q) accQ[v][q] = 0.f;

    // causal ⇒ key 只需到本 block 最大的 query ✓（块统一 ⇒ 屏障安全 ✓）
    int jmax = L;
    if (causal) {
        const int last = (int)blockIdx.x * FA_DQ_TC_BR + FA_DQ_TC_BR - 1;
        jmax = ((last < L) ? last : (L - 1)) + 1;
    }

    const int  srcA = (lane & ~3) | (t >> 1);         // 持有 key = t   的四元组 lane ✓
    const int  srcB = (lane & ~3) | ((t >> 1) + 2);   // 持有 key = t+4 的 ✓
    const bool tpar = (t & 1) != 0;

    for (int j0 = 0; j0 < jmax; j0 += FA_DQ_TC_BN) {
        const int nb = ((j0 + FA_DQ_TC_BN) < jmax) ? FA_DQ_TC_BN : (jmax - j0);
        // ⚠️ 尾部 key 零填充（否则 stale smem 被当真值 ⇒ 污染 dQ ✗）
        for (int tt = tid; tt < FA_DQ_TC_BN * d; tt += 128) {
            const int r = tt / d, c = tt % d;
            const bool ok = (r < nb);
            const size_t off = base + (size_t)(j0 + r) * d + c;
            sK[r * STRIDE + c] = ok ? K[off] : 0.f;
            sV[r * STRIDE + c] = ok ? V[off] : 0.f;
        }
        __syncthreads();

        for (int jb = 0; jb < nb; jb += 8) {
            // ---- S = Q·Kᵀ（C = (query × key) ✓）与 dP = dO·Vᵀ ----
            float cf[4] = {0.f, 0.f, 0.f, 0.f};
            float cg[4] = {0.f, 0.f, 0.f, 0.f};
            const int krow = jb + g;                      // B 的 n 索引 = key（tile 内 ✓）
            for (int kb = 0; kb < nkb; ++kb) {
                const int k8 = kb * 8;
                unsigned bK[2], bV[2];
                bK[0] = tf32_bits(sK[(size_t)krow * STRIDE + k8 + t]);
                bK[1] = tf32_bits(sK[(size_t)krow * STRIDE + k8 + t + 4]);
                bV[0] = tf32_bits(sV[(size_t)krow * STRIDE + k8 + t]);
                bV[1] = tf32_bits(sV[(size_t)krow * STRIDE + k8 + t + 4]);
                mma_t32(cf, aQ[kb], bK);
                mma_t32(cg, aD[kb], bV);
            }

            // ---- 逐元素（口径严格照 B1 ✓）：C 是 (row=r0/r1 ✓, col=key=2t/2t+1 ✓) => LSE/D **按行** ✓ ----
            const int  j_a = j0 + jb + 2 * t, j_b = j_a + 1;      // 本 lane 的两个 key（**含 j0+jb** ✓）
            const bool ka = (j_a < j0 + nb) && (j_a < L);
            const bool kb2 = (j_b < j0 + nb) && (j_b < L);
            float s0 = cf[0] * scale, s1 = cf[1] * scale, s2 = cf[2] * scale, s3 = cf[3] * scale;
            if (Bias) {
                if (ok0 && ka)  s0 += Bias[((size_t)bh * (size_t)L + (size_t)r0) * (size_t)L + (size_t)j_a];
                if (ok0 && kb2) s1 += Bias[((size_t)bh * (size_t)L + (size_t)r0) * (size_t)L + (size_t)j_b];
                if (ok1 && ka)  s2 += Bias[((size_t)bh * (size_t)L + (size_t)r1) * (size_t)L + (size_t)j_a];
                if (ok1 && kb2) s3 += Bias[((size_t)bh * (size_t)L + (size_t)r1) * (size_t)L + (size_t)j_b];
            }
            if (!ok0 || !ka  || (causal && j_a > r0)) s0 = -INFINITY;
            if (!ok0 || !kb2 || (causal && j_b > r0)) s1 = -INFINITY;
            if (!ok1 || !ka  || (causal && j_a > r1)) s2 = -INFINITY;
            if (!ok1 || !kb2 || (causal && j_b > r1)) s3 = -INFINITY;
            // ⚠️ cf[0]/cf[1] 的行都是 **r0**（列才是 key ✓）⇒ 用 lse0/pD0 ✓；cf[2]/cf[3] 用 lse1/pD1 ✓
            const float p0 = (s0 == -INFINITY || lse0 == -INFINITY) ? 0.f : expf(s0 - lse0);
            const float p1 = (s1 == -INFINITY || lse0 == -INFINITY) ? 0.f : expf(s1 - lse0);
            const float p2 = (s2 == -INFINITY || lse1 == -INFINITY) ? 0.f : expf(s2 - lse1);
            const float p3 = (s3 == -INFINITY || lse1 == -INFINITY) ? 0.f : expf(s3 - lse1);
            // ⚠️ ds **不带 scale** ✗→✓：dbias = ds（B1 口径 ✓ 未缩放）；scale 只属于 dQ ✓
            //   （第一版把 scale 折进 ds ⇒ dbias 整体偏小 1/√d ✗：d=32 时比值恰为 0.1768 ✓ 很好认 ✓）
            const float ds0 = p0 * (cg[0] - pD0);
            const float ds1 = p1 * (cg[1] - pD0);
            const float ds2 = p2 * (cg[2] - pD1);
            const float ds3 = p3 * (cg[3] - pD1);
            if (dBias) {                                      // dbias = ds ✓（(row,key) 唯一 ✓；掩码位靠 host memset 0 ✓）
                if (ok0 && ka)  dBias[((size_t)bh * (size_t)L + (size_t)r0) * (size_t)L + (size_t)j_a] = ds0;
                if (ok0 && kb2) dBias[((size_t)bh * (size_t)L + (size_t)r0) * (size_t)L + (size_t)j_b] = ds1;
                if (ok1 && ka)  dBias[((size_t)bh * (size_t)L + (size_t)r1) * (size_t)L + (size_t)j_a] = ds2;
                if (ok1 && kb2) dBias[((size_t)bh * (size_t)L + (size_t)r1) * (size_t)L + (size_t)j_b] = ds3;
            }

            // ---- repack：C 碎片 (row, key=2t/2t+1) → A 碎片 **(row, key=t/t+4)** ✓（与前向 PV 同一个 repack ✓）----
            const float e0A = __shfl_sync(0xffffffffu, ds0, srcA), e1A = __shfl_sync(0xffffffffu, ds1, srcA);
            const float e0B = __shfl_sync(0xffffffffu, ds0, srcB), e1B = __shfl_sync(0xffffffffu, ds1, srcB);
            const float f0A = __shfl_sync(0xffffffffu, ds2, srcA), f1A = __shfl_sync(0xffffffffu, ds3, srcA);
            const float f0B = __shfl_sync(0xffffffffu, ds2, srcB), f1B = __shfl_sync(0xffffffffu, ds3, srcB);
            unsigned aDs[4];
            aDs[0] = tf32_bits(scale * (tpar ? e1A : e0A));   // dS·scale (r0, key = t)    ✓
            aDs[1] = tf32_bits(scale * (tpar ? f1A : f0A));   // dS·scale (r1, key = t)    ✓
            aDs[2] = tf32_bits(scale * (tpar ? e1B : e0B));   // dS·scale (r0, key = t+4)  ✓
            aDs[3] = tf32_bits(scale * (tpar ? f1B : f0B));   // dS·scale (r1, key = t+4)  ✓

            // ---- dQ = Σ_j ds·K ⇒ B = K (n = d 维 ✓, k = key ✓) ----
#pragma unroll
            for (int v = 0; v < nkb; ++v) {
                const int dim = v * 8 + g;
                unsigned bd[2];
                bd[0] = tf32_bits(sK[(size_t)(jb + t)     * STRIDE + dim]);
                bd[1] = tf32_bits(sK[(size_t)(jb + t + 4) * STRIDE + dim]);
                mma_t32(accQ[v], aDs, bd);
            }
        }
        __syncthreads();                                  // 覆写 sK/sV 前全员用完 ✓
    }

    // ---- 写回 dQ（C 碎片：row = r0/r1 ✓、dim = v*8+2t/+1 ✓；无需归一化 ✓）----
#pragma unroll
    for (int v = 0; v < nkb; ++v) {
        const int dim = v * 8 + 2 * t;
        if (ok0) { float* p = dQ + base + (size_t)r0 * d + dim; p[0] = accQ[v][0]; p[1] = accQ[v][1]; }
        if (ok1) { float* p = dQ + base + (size_t)r1 * d + dim; p[0] = accQ[v][2]; p[1] = accQ[v][3]; }
    }
#else
    (void)Q; (void)K; (void)V; (void)Bias; (void)O; (void)dO; (void)LSE;
    (void)dQ; (void)dBias; (void)Dbuf; (void)L; (void)d; (void)scale; (void)causal;
#endif
}

// ============================================================
// 【②-TC/融合】**单趟融合反向**（2026-09-24，方案 B）—— dQ + dK + dV + dbias 一个 kernel ✓
//   ★ 动机：②-B1（dQ）与 ②-B2（dK/dV）各自**独立重算** S = Q·Kᵀ 与 dP = dO·Vᵀ ✗ ⇒ 融合省掉其中一遍 ✓
//     （拆解实测：B1 的 S/dP ≈ 其一大部分 mma ✓；融合后每瓦片 mma 28 → ~20 ✓）
//   ★ 基础：**照抄已验证的 `flash_attn_bwd_dkv_tc_kernel`** ✓（Sᵀ=K·Qᵀ、dPᵀ=V·dOᵀ、逐元素、repack、dK/dV ✓），
//     只加三件事 ✓：
//       (a) **自己算 D_i = Σ_d dO·O**（不再依赖 ②-B1 的 Dbuf ⇒ 可以整趟跳过 B1 ✓）：每 query tile 用
//           `tid < nib` 的一行线程**串行**算一行点积（≤32 次乘加 ✓ 极便宜 ✓）写进 `sD[]` ✓；
//       (b) **dbias = ds（未缩放 ✗→✓）**：从 C 碎片直接写（每个 (query,key) 唯一 ✓ 无冲突 ✓）；
//       (c) **dQ = Σ_j ds·K**：C 碎片是 (key, query) 而 dQ 需要 (query, dim) ✗ ⇒
//           **先把 ds 落到 per-warp smem staging（16 query × 16 key ✓）**，再按 A 碎片下标**随机读** ✓
//           ⇒ 不需要跨 quad 的 transpose shuffle ✓（与 B2 里 B 操作数"随机取 smem"同一招 ✓）；
//           块只拥有 64 个 key ⇒ dQ 只是**部分和** ⇒ 结尾 **atomicAdd** ✓（每元素每 block 一次 ✓ ≈ nblk·Lq·d·h·B ✓）。
//   ⚠️ dQ 的求和顺序**非确定** ✗（原子 ✓）——已知取舍 ✓（要确定性就得上分片 + 第二趟 ✓）。
//   ⚠️ 逐元素口径与 ②-B1/②-B2 **逐字一致** ✗→✓：LSE/D **按列**（本 C 是 (key, query) ✓）、
//      `p = exp(scale·s + bias − LSE)`、`dbias = ds = p·(dP − D)`（**不带 scale** ✓）、dK/dV 用 `ds·scale` ✓。
//   ⚠️ 仅 d ≤ 32 且 d%8==0 ✓（host 把关）；smem = 2·BI·(d+4) + BI + 4·256 个 float ✓
// ============================================================
__global__ void flash_attn_bwd_fused_tc_kernel(const float* __restrict__ Q,
                                               const float* __restrict__ K,
                                               const float* __restrict__ V,
                                               const float* __restrict__ Bias,
                                               const float* __restrict__ O,
                                               const float* __restrict__ dO,
                                               const float* __restrict__ LSE,
                                               float* __restrict__ dQ,
                                               float* __restrict__ dQpart,   // split-K 工作区（nullptr ⇒ 原子版 ✓）
                                               float* __restrict__ dK,
                                               float* __restrict__ dV,
                                               float* __restrict__ dBias,
                                               const int L, const int d,
                                               const float scale, const int causal) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    constexpr int PAD = 4;
    const int STRIDE = d + PAD;
    extern __shared__ float fsmem[];
    float* sQ   = fsmem;                                   // [FA_DKV_TC_BI][STRIDE]
    float* sDO  = sQ  + FA_DKV_TC_BI * STRIDE;             // [FA_DKV_TC_BI][STRIDE]
    float* sD   = sDO + FA_DKV_TC_BI * STRIDE;             // [FA_DKV_TC_BI]  D_i ✓
    float* sDQw = sD  + FA_DKV_TC_BI;                      // 每 warp 256 = 16 query × 16 key 的 ds staging ✓
    float* sPq  = sDQw + 4 * 256;                          // ★ 块内归约：[warp][16 query][d] 的部分和 ✓

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int bh   = (int)blockIdx.y;
    const size_t base = (size_t)bh * (size_t)L * (size_t)d;
    const int nkb  = d / 8;                                // ≤ 4 ✓

    float* sDQ = sDQw + warp * 256;                        // 本 warp 私有（只需 __syncwarp ✓）

    const int j0 = (int)blockIdx.x * FA_DKV_TC_BNJ + warp * 16;
    const int jA = j0 + g;                                 // C 碎片的前 8 行 ✓
    const int jB = j0 + g + 8;                             // 后 8 行 ✓
    const bool okA = (jA < L), okB = (jB < L);

    // ---- dQ 的冲刷目标（三种模式 ✓）----
    //   · split-K（dQpart ≠ nullptr ✓）：写**本 block 的切片** ⇒ 纯写、**零原子**、**可复现** ✓；
    //     ⚠️ 切片布局 [bh][blk][Lq][d] ✓；非 causal 时每 block 都写满全 q ✓，causal 时只写 q ≥ i_start
    //       ⇒ 归约端按 `blk ≤ q/FA_DKV_TC_BNJ` 截断即可，**无需 memset** ✓✓
    //   · 单 key-block（gridDim.x == 1 ✓，L ≤ 64）：本 block 的 dQ 就是完整值 ⇒ 也纯写 ✓
    //   · 其余（原子版 ✓）：atomicAdd（顺序非确定 ✗，已知取舍 ✓）
    const int  blk       = (int)blockIdx.x;
    const bool exclusive = (dQpart != nullptr) || (gridDim.x == 1);
    float* dQdst = nullptr;
    if (dQpart) dQdst = dQpart + ((size_t)bh * gridDim.x + blk) * (size_t)L * (size_t)d;
    else if (exclusive && dQ) dQdst = dQ + base;            // ⚠️ 必须带 bh 偏移 ✗→✓（否则所有 bh 都写到 bh=0 ✗）

    // ---- A(K)/A(V)：16 key × d 的 k8 块碎片（载一次，沿 i 复用 ✓）----
    unsigned aK[4][4], aV[4][4];
    for (int kb = 0; kb < nkb; ++kb) {
        const int k8 = kb * 8;
        const int oA = k8 + t, oB = k8 + t + 4;
        const size_t rA = base + (size_t)jA * d, rB = base + (size_t)jB * d;
        aK[kb][0] = tf32_bits(okA ? K[rA + oA] : 0.f);
        aK[kb][1] = tf32_bits(okB ? K[rB + oA] : 0.f);
        aK[kb][2] = tf32_bits(okA ? K[rA + oB] : 0.f);
        aK[kb][3] = tf32_bits(okB ? K[rB + oB] : 0.f);
        aV[kb][0] = tf32_bits(okA ? V[rA + oA] : 0.f);
        aV[kb][1] = tf32_bits(okB ? V[rB + oA] : 0.f);
        aV[kb][2] = tf32_bits(okA ? V[rA + oB] : 0.f);
        aV[kb][3] = tf32_bits(okB ? V[rB + oB] : 0.f);
    }

    // ---- dK/dV 累加器（C 碎片：**(key, d 列)** ✓）与 dQ 累加器（(query, d 列) ✓，每组冲刷 ✓）----
    float accK[4][4], accV[4][4], accQ[4][4];
#pragma unroll
    for (int v = 0; v < 4; ++v)
#pragma unroll
        for (int q = 0; q < 4; ++q) { accK[v][q] = 0.f; accV[v][q] = 0.f; accQ[v][q] = 0.f; }

    int i_start = 0;
    if (causal) {
        i_start = (int)blockIdx.x * FA_DKV_TC_BNJ;         // 本 block 的首个 key ✓（块统一 ✓）
        if (i_start > L) i_start = L;
    }

    const int  srcA = (lane & ~3) | (t >> 1);              // 持有 query = t   的四元组 lane ✓
    const int  srcB = (lane & ~3) | ((t >> 1) + 2);        // 持有 query = t+4 的 ✓
    const bool tpar = (t & 1) != 0;

    for (int i0 = i_start; i0 < L; i0 += FA_DKV_TC_BI) {
        const int nib = ((i0 + FA_DKV_TC_BI) < L) ? FA_DKV_TC_BI : (L - i0);
        // ⚠️ 尾部行零填充（否则 stale smem 会被 mma 当真值 ⇒ 污染其它 key 的累加 ✗）
        for (int tt = tid; tt < FA_DKV_TC_BI * d; tt += 128) {
            const int r = tt / d, c = tt % d;
            const bool ok = (r < nib);
            const size_t off = base + (size_t)(i0 + r) * d + c;
            sQ [r * STRIDE + c] = ok ? Q [off] : 0.f;
            sDO[r * STRIDE + c] = ok ? dO[off] : 0.f;
        }
        // (a) D_i = Σ_d dO·O —— 一行线程算一行（≤32 次乘加 ✓ 极便宜 ✓）
        if (tid < nib) {
            const size_t off = base + (size_t)(i0 + tid) * d;
            float acc = 0.f;
            for (int c = 0; c < d; ++c) acc += dO[off + c] * O[off + c];
            sD[tid] = acc;
        }
        __syncthreads();

        for (int ib = 0; ib < nib; ib += 16) {
            // ---- 两个 8-query 子步：Sᵀ/dPᵀ + 逐元素 + dbias + repack + dK/dV（与 ②-B2 逐字一致 ✓）----
            for (int h = 0; h < 2; ++h) {
                const int ibh = ib + h * 8;
                const bool hok = (ibh < nib);              // 块统一 ✓（ib/nib 都块统一 ✓）
                float ds0 = 0.f, ds1 = 0.f, ds2 = 0.f, ds3 = 0.f;   // 供 dQ staging（无效子步写 0 ✓）
                if (hok) {
                    float cf[4] = {0.f, 0.f, 0.f, 0.f};    // Sᵀ[key][query]
                    float cg[4] = {0.f, 0.f, 0.f, 0.f};    // dPᵀ[key][query]
                    const int qrow = ibh + g;              // B 的 n 索引 = query（tile 内 ✓）
                    for (int kb = 0; kb < nkb; ++kb) {
                        const int k8 = kb * 8;
                        unsigned bq[2], bo[2];
                        bq[0] = tf32_bits(sQ [(size_t)qrow * STRIDE + k8 + t]);
                        bq[1] = tf32_bits(sQ [(size_t)qrow * STRIDE + k8 + t + 4]);
                        bo[0] = tf32_bits(sDO[(size_t)qrow * STRIDE + k8 + t]);
                        bo[1] = tf32_bits(sDO[(size_t)qrow * STRIDE + k8 + t + 4]);
                        mma_t32(cf, aK[kb], bq);
                        mma_t32(cg, aV[kb], bo);
                    }
                    // 逐元素（口径照 ②-B2 ✓；D 现在来自 sD ✓）
                    const int  i_a = i0 + ibh + 2 * t, i_b = i_a + 1;
                    const bool va = (i_a < i0 + nib) && (i_a < L);
                    const bool vb = (i_b < i0 + nib) && (i_b < L);
                    float s0 = cf[0] * scale, s1 = cf[1] * scale, s2 = cf[2] * scale, s3 = cf[3] * scale;
                    if (Bias) {
                        if (va && okA) s0 += Bias[((size_t)bh * (size_t)L + (size_t)i_a) * (size_t)L + (size_t)jA];
                        if (vb && okA) s1 += Bias[((size_t)bh * (size_t)L + (size_t)i_b) * (size_t)L + (size_t)jA];
                        if (va && okB) s2 += Bias[((size_t)bh * (size_t)L + (size_t)i_a) * (size_t)L + (size_t)jB];
                        if (vb && okB) s3 += Bias[((size_t)bh * (size_t)L + (size_t)i_b) * (size_t)L + (size_t)jB];
                    }
                    float lsea = -INFINITY, lseb = -INFINITY, Da = 0.f, Db = 0.f;
                    if (va) { lsea = LSE[(size_t)bh * (size_t)L + (size_t)i_a]; Da = sD[i_a - i0]; }
                    if (vb) { lseb = LSE[(size_t)bh * (size_t)L + (size_t)i_b]; Db = sD[i_b - i0]; }
                    if (!okA || !va || (causal && jA > i_a)) s0 = -INFINITY;
                    if (!okA || !vb || (causal && jA > i_b)) s1 = -INFINITY;
                    if (!okB || !va || (causal && jB > i_a)) s2 = -INFINITY;
                    if (!okB || !vb || (causal && jB > i_b)) s3 = -INFINITY;
                    // ⚠️ LSE **按列（query）** 对号 ✓（C 是 (key, query) ✓）
                    const float p0 = (s0 == -INFINITY || lsea == -INFINITY) ? 0.f : expf(s0 - lsea);
                    const float p1 = (s1 == -INFINITY || lseb == -INFINITY) ? 0.f : expf(s1 - lseb);
                    const float p2 = (s2 == -INFINITY || lsea == -INFINITY) ? 0.f : expf(s2 - lsea);
                    const float p3 = (s3 == -INFINITY || lseb == -INFINITY) ? 0.f : expf(s3 - lseb);
                    const float du0 = p0 * (cg[0] - Da);          // **未缩放** ds ⇒ dbias ✓
                    const float du1 = p1 * (cg[1] - Db);
                    const float du2 = p2 * (cg[2] - Da);
                    const float du3 = p3 * (cg[3] - Db);
                    ds0 = du0 * scale;                            // 缩放版 ⇒ dK ✓
                    ds1 = du1 * scale;
                    ds2 = du2 * scale;
                    ds3 = du3 * scale;
                    // (b) dbias = ds（未缩放 ✓）—— 每个 (query,key) 唯一 ✓ 直接写 ✓
                    if (dBias) {
                        if (okA && va) dBias[((size_t)bh * (size_t)L + (size_t)i_a) * (size_t)L + (size_t)jA] = du0;
                        if (okA && vb) dBias[((size_t)bh * (size_t)L + (size_t)i_b) * (size_t)L + (size_t)jA] = du1;
                        if (okB && va) dBias[((size_t)bh * (size_t)L + (size_t)i_a) * (size_t)L + (size_t)jB] = du2;
                        if (okB && vb) dBias[((size_t)bh * (size_t)L + (size_t)i_b) * (size_t)L + (size_t)jB] = du3;
                    }
                    // repack：C 碎片 (key, query=2t/2t+1) → A 碎片 (key, query=t/t+4) ✓（同 ②-B2 ✓）
                    const float p0A = __shfl_sync(0xffffffffu, p0, srcA), p1A = __shfl_sync(0xffffffffu, p1, srcA);
                    const float p0B = __shfl_sync(0xffffffffu, p0, srcB), p1B = __shfl_sync(0xffffffffu, p1, srcB);
                    const float p2A = __shfl_sync(0xffffffffu, p2, srcA), p3A = __shfl_sync(0xffffffffu, p3, srcA);
                    const float p2B = __shfl_sync(0xffffffffu, p2, srcB), p3B = __shfl_sync(0xffffffffu, p3, srcB);
                    const float q0A = __shfl_sync(0xffffffffu, ds0, srcA), q1A = __shfl_sync(0xffffffffu, ds1, srcA);
                    const float q0B = __shfl_sync(0xffffffffu, ds0, srcB), q1B = __shfl_sync(0xffffffffu, ds1, srcB);
                    const float q2A = __shfl_sync(0xffffffffu, ds2, srcA), q3A = __shfl_sync(0xffffffffu, ds3, srcA);
                    const float q2B = __shfl_sync(0xffffffffu, ds2, srcB), q3B = __shfl_sync(0xffffffffu, ds3, srcB);
                    unsigned aP[4], aD[4];
                    aP[0] = tf32_bits(tpar ? p1A : p0A);
                    aP[1] = tf32_bits(tpar ? p3A : p2A);
                    aP[2] = tf32_bits(tpar ? p1B : p0B);
                    aP[3] = tf32_bits(tpar ? p3B : p2B);
                    aD[0] = tf32_bits(tpar ? q1A : q0A);
                    aD[1] = tf32_bits(tpar ? q3A : q2A);
                    aD[2] = tf32_bits(tpar ? q1B : q0B);
                    aD[3] = tf32_bits(tpar ? q3B : q2B);
#pragma unroll
                    for (int v = 0; v < nkb; ++v) {
                        const int dim = v * 8 + g;
                        unsigned bp[2], bd[2];
                        bp[0] = tf32_bits(sDO[(size_t)(ibh + t)     * STRIDE + dim]);
                        bp[1] = tf32_bits(sDO[(size_t)(ibh + t + 4) * STRIDE + dim]);
                        bd[0] = tf32_bits(sQ [(size_t)(ibh + t)     * STRIDE + dim]);
                        bd[1] = tf32_bits(sQ [(size_t)(ibh + t + 4) * STRIDE + dim]);
                        mma_t32(accV[v], aP, bp);
                        mma_t32(accK[v], aD, bd);
                    }
                }
                // (c) staging：ds（缩放版 ✓）落到 (query local, key local) ⇒ 供 dQ 的 A 碎片随机读 ✓
                //   C 四元组：c0=(key jA, query i_a) c1=(jA, i_b) c2=(jB, i_a) c3=(jB, i_b) ✓
                //   ⇒ C 的 (key local, query local) = (g | g+8, h*8+2t | +1) ✓
                sDQ[(h * 8 + 2 * t)     * 16 + (g)    ] = ds0;
                sDQ[(h * 8 + 2 * t + 1) * 16 + (g)    ] = ds1;
                sDQ[(h * 8 + 2 * t)     * 16 + (g + 8)] = ds2;
                sDQ[(h * 8 + 2 * t + 1) * 16 + (g + 8)] = ds3;
            }
            __syncwarp();                                       // staging 只在本 warp 内共享 ✓

            // ---- (c) dQ = Σ_j ds·K：A = (query, key) 从 staging 读 ✓、B = K（n=dim, k=key ✓）----
#pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
                const int key0 = j0 + kh * 8;                  // 本 key-half 的首个全局 key ✓
                unsigned aDs[4];
                aDs[0] = tf32_bits(sDQ[(g)     * 16 + kh * 8 + t]);        // A[q=g    ][key=t]    ✓
                aDs[1] = tf32_bits(sDQ[(g + 8) * 16 + kh * 8 + t]);        // A[q=g+8  ][key=t]    ✓
                aDs[2] = tf32_bits(sDQ[(g)     * 16 + kh * 8 + t + 4]);    // A[q=g    ][key=t+4]  ✓
                aDs[3] = tf32_bits(sDQ[(g + 8) * 16 + kh * 8 + t + 4]);    // A[q=g+8  ][key=t+4]  ✓
#pragma unroll
                for (int v = 0; v < nkb; ++v) {
                    const int dim = v * 8 + g;
                    unsigned bd[2];
                    bd[0] = tf32_bits(((key0 + t)     < L) ? K[base + (size_t)(key0 + t)     * d + dim] : 0.f);
                    bd[1] = tf32_bits(((key0 + t + 4) < L) ? K[base + (size_t)(key0 + t + 4) * d + dim] : 0.f);
                    mma_t32(accQ[v], aDs, bd);
                }
            }
            // ---- 冲刷 dQ：⚠️ **4 个 warp 各自持有同一 (q,dim) 的部分和** ✗（各管 16 个 key ✓）
            //   原子版靠 atomicAdd 求和 ✓；纯写版若不先归约就是"**最后写的赢**" ✗（丢 3/4 贡献 ✗ + 非确定 ✗）
            //   ⇒ 先写 per-warp 私有 smem 片 → 屏障 → **固定顺序**求和（可复现 ✓）→ 写切片/原子 ✓
#pragma unroll
            for (int v = 0; v < nkb; ++v) {
                const int c0 = v * 8 + 2 * t, c1 = c0 + 1;
                float* w = sPq + (size_t)warp * 16 * (size_t)d;
                w[(size_t)g       * d + c0] = accQ[v][0];
                w[(size_t)g       * d + c1] = accQ[v][1];
                w[(size_t)(g + 8) * d + c0] = accQ[v][2];
                w[(size_t)(g + 8) * d + c1] = accQ[v][3];
                accQ[v][0] = 0.f; accQ[v][1] = 0.f; accQ[v][2] = 0.f; accQ[v][3] = 0.f;
            }
            __syncthreads();
            for (int e = tid; e < 16 * d; e += 128) {
                const int ql = e / d, c = e % d;
                const int q  = i0 + ib + ql;
                if (q >= L || q >= i0 + nib) continue;
                const float sum = ((sPq[ql * d + c] + sPq[16 * d + ql * d + c])
                                 + (sPq[32 * d + ql * d + c] + sPq[48 * d + ql * d + c]));
                if (exclusive) { if (dQdst) dQdst[(size_t)q * d + c] = sum; }
                else if (dQ)   { atomicAdd(dQ + base + (size_t)q * d + c, sum); }
            }
            __syncthreads();                                    // 下一组的写片必须等本轮读完 ✓
        }
        __syncthreads();                                        // 覆写 sQ/sDO/sD 前全员用完 ✓
    }

    // ---- 写回 dK/dV：每个 (key, dim) 恰好一个 lane 写 ⇒ **零原子** ✓ ----
#pragma unroll
    for (int v = 0; v < nkb; ++v) {
        const int dim = v * 8 + 2 * t;
        if (okA) {
            if (dK) { float* p = dK + base + (size_t)jA * d + dim; p[0] = accK[v][0]; p[1] = accK[v][1]; }
            if (dV) { float* p = dV + base + (size_t)jA * d + dim; p[0] = accV[v][0]; p[1] = accV[v][1]; }
        }
        if (okB) {
            if (dK) { float* p = dK + base + (size_t)jB * d + dim; p[0] = accK[v][2]; p[1] = accK[v][3]; }
            if (dV) { float* p = dV + base + (size_t)jB * d + dim; p[0] = accV[v][2]; p[1] = accV[v][3]; }
        }
    }
#else
    (void)Q; (void)K; (void)V; (void)Bias; (void)O; (void)dO; (void)LSE;
    (void)dQ; (void)dQpart; (void)dK; (void)dV; (void)dBias; (void)L; (void)d; (void)scale; (void)causal;
#endif
}

// ============================================================
// 【②-TC/split-K 归约】把各 key-block 的 dQ 部分和相加 ⇒ **零原子 + 可复现** ✓（2026-09-24）
//   布局：`part[(bh * nblk + blk) * (L·d) + q·d + c]` ✓（由融合 kernel 写 ✓），归约到 `dQ[bh][q][c]` ✓
//   ⚠️ causal 时只需累加 `blk ≤ q / FA_DKV_TC_BNJ` 的切片 ✓（更大的 blk 压根没写 ✓ 读它 = 读脏数据 ✗）
//      ⇒ **无需事先 memset 工作区** ✓✓
//   成本：读 nblk·L·d·h·B + 写 L·d·h·B 个 float ⇒ 纯带宽 ✓（MSA 形状 ≈135 MB ⇒ ~1 ms 级 ✓）
// ============================================================
__global__ void flash_attn_bwd_dq_reduce_kernel(const float* __restrict__ part,
                                                float* __restrict__ dQ,
                                                const int L, const int d, const int nblk,
                                                const int causal) {
    const int    bh  = (int)blockIdx.y;
    const size_t n   = (size_t)L * (size_t)d;
    const float* p   = part + (size_t)bh * (size_t)nblk * n;
    float*       out = dQ   + (size_t)bh * n;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const int q = (int)(i / (size_t)d);
        int blk_max = nblk - 1;
        if (causal) {
            const int bq = q / FA_DKV_TC_BNJ;
            if (bq < blk_max) blk_max = bq;
        }
        float s = 0.f;
        for (int blk = 0; blk <= blk_max; ++blk) s += p[(size_t)blk * n + i];
        out[i] = s;
    }
}

int flash_attn_backward_cuda(const FlashAttnConfig& cfg,
                             const float* Q, const float* K, const float* V,
                             const float* bias, const float* O, const float* dO, const float* LSE,
                             float* dQ, float* dK, float* dV, float* dbias) {
    if (!Q || !K || !V || !O || !dO || !LSE) return 1;
    if (cfg.B <= 0 || cfg.h <= 0 || cfg.L <= 0 || cfg.d <= 0) return 1;
    if (cfg.d > FA_DB_D_MAX) return 1;                     // 寄存器上限（d ≤ 32）⇒ 调用方回落 CPU ✓
    if ((cfg.d % 4) != 0) return 1;                        // 【②-v2 反向】lane 切分要求 d 是 4 的倍数 ✓
    if (!dQ && !dK && !dV && !dbias) return 1;

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) return 1;

    const float scale = (cfg.scale > 0.f) ? cfg.scale : 1.0f / std::sqrt((float)cfg.d);
    const size_t nq = (size_t)cfg.B * cfg.h * cfg.L * cfg.d;
    const size_t nb_ = (size_t)cfg.B * cfg.h * cfg.L * cfg.L;
    const size_t nl  = (size_t)cfg.B * cfg.h * cfg.L;

    // ★ 输出约定：dQ 全量写但仍清零（稳 ✓）、dbias 靠 memset 保证掩码位为 0 ✓；
    //   dK/dV **不再 memset**（②-B2 是"每元素恰好写一次"的无原子 kernel ✓ ⇒ memset 纯浪费 ✗）
    if (dQ    && cudaMemset(dQ,    0, nq * sizeof(float)) != cudaSuccess) return 1;
    if (dbias && cudaMemset(dbias, 0, nb_ * sizeof(float)) != cudaSuccess) return 1;

    // 【②-B2】D_i 临时缓冲：B1 产出 → B2 消费（只跨两次 launch ✓，量 = L·h·B 个 float ✓）
    float* dDbuf = nullptr;
    if ((dK || dV) && cudaMalloc(&dDbuf, nl * sizeof(float)) != cudaSuccess) return 1;

    // 【②-TC/融合】可选：**单趟融合反向**（dQ+dK+dV+dbias 全在一个 kernel ✓ —— 省掉 ②-B1 的重复 S/dP ✓）
    //   开关：`PPML_FLASH_ATTN_FUSED`（**且** `PPML_FLASH_ATTN_TC=1` ✓；默认都关 ⇒ 现有路径逐位不变 ✓）
    //     · `=1` ⇒ **原子版**：dQ 走 atomicAdd（顺序非确定 ✗；无额外显存 ✓）
    //     · `=2` ⇒ **split-K 版**：dQ 写工作区 `[bh][blk][Lq][d]` + 归约 kernel ⇒ **零原子 + 可复现** ✓
    //              ⚠️ 工作区 = nblk·L·d·h·B 个 float（MSA 形状 4×824×256×32×4 B ≈ **108 MB** ✗）
    //              ⇒ 申请失败**自动回落原子版** ✓（不报错、不改变正确性 ✓）
    //   ⚠️ 需 sm_80+、d ≤ 32 且 d%8==0；不满足一律回落现有两趟路径 ✓
    {
        const char* fu = getenv("PPML_FLASH_ATTN_FUSED");
        const char* te = getenv("PPML_FLASH_ATTN_TC");
        if (fu && fu[0] && fu[0] != '0' && te && te[0] && te[0] != '0' && (cfg.d % 8) == 0) {
            int maj = 0, minr = 0;
            if (cutlass_hw_supported(&maj, &minr)) {
                const int    nblk_k    = (cfg.L + FA_DKV_TC_BNJ - 1) / FA_DKV_TC_BNJ;   // key 方向的分块数 ✓
                const bool   want_split = (fu[0] >= '2');                              // "2" ⇒ split-K ✓
                const size_t smem_f = (size_t)2 * FA_DKV_TC_BI * (cfg.d + 4) * sizeof(float)   // Q/dO（pad=4 ✓）
                                    + (size_t)FA_DKV_TC_BI * sizeof(float)                      // D_i ✓
                                    + (size_t)4 * 16 * 16 * sizeof(float)                       // ds staging：每 warp 256 ✓
                                    + (size_t)4 * 16 * cfg.d * sizeof(float);                   // ★ 块内归约片 [4 warp][16][d] ✓
                const dim3 grid_f((unsigned)nblk_k, (unsigned)(cfg.B * cfg.h));

                float* dQpart = nullptr;
                if (want_split && nblk_k > 1) {
                    const size_t part_n = (size_t)cfg.B * (size_t)cfg.h * (size_t)nblk_k
                                        * (size_t)cfg.L * (size_t)cfg.d;
                    if (cudaMalloc(&dQpart, part_n * sizeof(float)) != cudaSuccess) {
                        dQpart = nullptr;          // 申请不到 ⇒ 回落原子版 ✓（正确性不变 ✓）
                        (void)cudaGetLastError();  // 清掉 OOM 标志 ✓
                    }
                }
                flash_attn_bwd_fused_tc_kernel<<<grid_f, 128, smem_f>>>(Q, K, V, bias, O, dO, LSE,
                                                                        dQ, dQpart, dK, dV, dbias,
                                                                        cfg.L, cfg.d, scale,
                                                                        cfg.causal ? 1 : 0);
                if (cudaGetLastError() != cudaSuccess) { if (dQpart) cudaFree(dQpart); return 1; }
                if (dQpart) {
                    if (dQ) {
                        const size_t n   = (size_t)cfg.L * (size_t)cfg.d;
                        const unsigned gx = (unsigned)((n + 255) / 256);
                        const dim3 grid_r(gx > 64u ? 64u : gx, (unsigned)(cfg.B * cfg.h));
                        flash_attn_bwd_dq_reduce_kernel<<<grid_r, 256>>>(dQpart, dQ, cfg.L, cfg.d,
                                                                         nblk_k, cfg.causal ? 1 : 0);
                        if (cudaGetLastError() != cudaSuccess) { cudaFree(dQpart); return 1; }
                    }
                    cudaFree(dQpart);
                    if (getenv("GRAPH_DEBUG_LOSS"))
                        std::fprintf(stderr, "[FLASH-FUSED] split-K 生效：nblk=%d 工作区 %.1f MB ⇒ dQ 零原子、可复现\n",
                                     nblk_k, (double)((size_t)cfg.B * cfg.h * nblk_k * cfg.L * cfg.d * sizeof(float)) / 1048576.0);
                } else if (getenv("GRAPH_DEBUG_LOSS")) {
                    std::fprintf(stderr, "[FLASH-FUSED] 原子版生效（dQ 顺序非确定 ✗）\n");
                }
                if (dDbuf) cudaFree(dDbuf);
                return 0;                       // 已算全（dQ/dK/dV/dbias）⇒ 跳过 B1/B2 ✓
            }
        }
    }

    // ---- B1：dQ + dbias + 导出 D；**不再**算 dK/dV（那条路是原子 ✗ 占反向 83% 时间 ✗）----
    //   原 dK/dV 的代码保留在 kernel 里（传 nullptr 即跳过 ✓），便于将来 A/B 对照 ✓
    constexpr int rows_per_block = 8 * 4;
    constexpr int bwd_pad = 4;
    const size_t smem_b1 = (size_t)2 * FA_BC * (cfg.d + bwd_pad) * sizeof(float);
    const dim3 grid1((unsigned)((cfg.L + rows_per_block - 1) / rows_per_block), (unsigned)(cfg.B * cfg.h));
    // 【②-TC/B1】可选：dQ(+dbias+D) 走 **TF32 tensor core**（复用同一个 `PPML_FLASH_ATTN_TC` 开关 ✓；
    //   默认 0 ⇒ 走已验证的 SIMT B1 ✓）。⚠️ 它只算 dQ/dbias/D（dK/dV 交给 ②-B2 ✓）⇒ 不满足即回落 SIMT ✓
    bool dq_done = false;
    {
        const char* tc_env1 = getenv("PPML_FLASH_ATTN_TC");
        if (tc_env1 && tc_env1[0] && tc_env1[0] != '0' && (cfg.d % 8) == 0) {
            int maj = 0, minr = 0;
            if (cutlass_hw_supported(&maj, &minr)) {
                const size_t smem_tc1 = (size_t)2 * FA_DQ_TC_BN * (cfg.d + bwd_pad) * sizeof(float);  // K/V
                const dim3 grid_tc1((unsigned)((cfg.L + FA_DQ_TC_BR - 1) / FA_DQ_TC_BR),
                                    (unsigned)(cfg.B * cfg.h));
                flash_attn_bwd_dq_tc_kernel<<<grid_tc1, 128, smem_tc1>>>(Q, K, V, bias, O, dO, LSE,
                                                                         dQ, dbias, dDbuf,
                                                                         cfg.L, cfg.d, scale,
                                                                         cfg.causal ? 1 : 0);
                if (cudaGetLastError() != cudaSuccess) return 1;
                dq_done = true;
            }
        }
    }
    if (!dq_done)
        flash_attn_bwd_kernel<<<grid1, 128, smem_b1>>>(Q, K, V, bias, O, dO, LSE, dQ,
                                                       nullptr, nullptr, dbias, dDbuf,
                                                       cfg.L, cfg.d, scale, cfg.causal ? 1 : 0);

    // ---- B2：dK/dV 的无原子 GEMM 型 kernel ✓（每 block 独占 32 个 key × 全 d ⇒ 沿 i 归约 ✓）----
    if (dK || dV) {
        // 【②-TC/B2】可选：dK/dV 走 **TF32 tensor core**（复用同一个 `PPML_FLASH_ATTN_TC` 开关 ✓，
        //   默认 0 ⇒ 走已验证的 SIMT B2 ✓）；⚠️ 每次调用读 env（不做 static 缓存 ⇒ 测试可同进程 A/B ✓）
        //   ⚠️ 需 sm_80+ 且 d % 8 == 0（d ≤ 32 已由上面的 FA_DB_D_MAX 把关 ✓）；不满足一律回落 SIMT ✓
        const char* tc_env = getenv("PPML_FLASH_ATTN_TC");
        bool dkv_done = false;
        if (tc_env && tc_env[0] && tc_env[0] != '0' && (cfg.d % 8) == 0) {
            int maj = 0, minr = 0;
            if (cutlass_hw_supported(&maj, &minr)) {
                const size_t smem_tcb2 = (size_t)2 * FA_DKV_TC_BI * (cfg.d + bwd_pad) * sizeof(float); // Q/dO
                const dim3 grid_tc2((unsigned)((cfg.L + FA_DKV_TC_BNJ - 1) / FA_DKV_TC_BNJ),
                                    (unsigned)(cfg.B * cfg.h));
                flash_attn_bwd_dkv_tc_kernel<<<grid_tc2, 128, smem_tcb2>>>(Q, K, V, bias, dO, LSE, dDbuf,
                                                                           dK, dV, cfg.L, cfg.d, scale,
                                                                           cfg.causal ? 1 : 0);
                if (cudaGetLastError() != cudaSuccess) return 1;
                dkv_done = true;
            }
        }
        if (!dkv_done) {
            const size_t smem_b2 = (size_t)2 * FA_DKV_BNJ * (cfg.d + bwd_pad) * sizeof(float)   // K/V
                                 + (size_t)2 * FA_DKV_BI  * (cfg.d + bwd_pad) * sizeof(float);  // Q/dO
            const dim3 grid2((unsigned)((cfg.L + FA_DKV_BNJ - 1) / FA_DKV_BNJ), (unsigned)(cfg.B * cfg.h));
            flash_attn_bwd_dkv_kernel<<<grid2, 128, smem_b2>>>(Q, K, V, bias, dO, LSE, dDbuf,
                                                               dK, dV, cfg.L, cfg.d, scale, cfg.causal ? 1 : 0);
        }
    }
    if (dDbuf) cudaFree(dDbuf);
    if (cudaGetLastError() != cudaSuccess) return 1;
    return 0;
}

}  // namespace ppml

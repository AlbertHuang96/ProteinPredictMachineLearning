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

    // ---- B1：dQ + dbias + 导出 D；**不再**算 dK/dV（那条路是原子 ✗ 占反向 83% 时间 ✗）----
    //   原 dK/dV 的代码保留在 kernel 里（传 nullptr 即跳过 ✓），便于将来 A/B 对照 ✓
    constexpr int rows_per_block = 8 * 4;
    constexpr int bwd_pad = 4;
    const size_t smem_b1 = (size_t)2 * FA_BC * (cfg.d + bwd_pad) * sizeof(float);
    const dim3 grid1((unsigned)((cfg.L + rows_per_block - 1) / rows_per_block), (unsigned)(cfg.B * cfg.h));
    flash_attn_bwd_kernel<<<grid1, 128, smem_b1>>>(Q, K, V, bias, O, dO, LSE, dQ,
                                                   nullptr, nullptr, dbias, dDbuf,
                                                   cfg.L, cfg.d, scale, cfg.causal ? 1 : 0);

    // ---- B2：dK/dV 的无原子 GEMM 型 kernel ✓（每 block 独占 32 个 key × 全 d ⇒ 沿 i 归约 ✓）----
    if (dK || dV) {
        const size_t smem_b2 = (size_t)2 * FA_DKV_BNJ * (cfg.d + bwd_pad) * sizeof(float)   // K/V
                             + (size_t)2 * FA_DKV_BI  * (cfg.d + bwd_pad) * sizeof(float);  // Q/dO
        const dim3 grid2((unsigned)((cfg.L + FA_DKV_BNJ - 1) / FA_DKV_BNJ), (unsigned)(cfg.B * cfg.h));
        flash_attn_bwd_dkv_kernel<<<grid2, 128, smem_b2>>>(Q, K, V, bias, dO, LSE, dDbuf,
                                                           dK, dV, cfg.L, cfg.d, scale, cfg.causal ? 1 : 0);
    }
    if (dDbuf) cudaFree(dDbuf);
    if (cudaGetLastError() != cudaSuccess) return 1;
    return 0;
}

}  // namespace ppml

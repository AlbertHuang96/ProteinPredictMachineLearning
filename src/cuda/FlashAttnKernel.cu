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
    __shared__ float sK[FA_BC * FA_D_MAX];
    __shared__ float sV[FA_BC * FA_D_MAX];

    const int tid  = (int)threadIdx.x;
    const int bh   = (int)blockIdx.y;                     // b*h + hh
    const int row0 = (int)blockIdx.x * FA_BR;             // 本 block 的首个 query 行
    const int i    = row0 + tid;                          // 本线程负责的 query 行
    const bool active = (i < L);
    const size_t base = (size_t)bh * (size_t)L * (size_t)d;

    // 本 block 需要遍历的最大 K 行数（causal ⇒ 到本 block 最后一行 ✓）
    int jmax_block = L;
    if (causal) {
        const int last_row = (row0 + FA_BR - 1 < L) ? (row0 + FA_BR - 1) : (L - 1);
        jmax_block = last_row + 1;
    }

    // ---- 本线程的 Q 行 + 累加器（全部寄存器 ✓）----
    float qreg[FA_D_MAX];
    float acc[FA_D_MAX];
    for (int dd = 0; dd < d; ++dd) {
        qreg[dd] = (active && Q) ? Q[base + (size_t)i * d + dd] : 0.f;
        acc[dd]  = 0.f;
    }
    float m = -INFINITY;   // running row max
    float l = 0.f;         // running Σ exp（未归一化）

    for (int j0 = 0; j0 < jmax_block; j0 += FA_BC) {
        const int nb = ((j0 + FA_BC) < jmax_block) ? FA_BC : (jmax_block - j0);

        // ---- K/V tile 协作加载（行主序：t = r*d + c ✓；越界行不加载 ⇒ 下面按 j1 限制读取 ✓）----
        for (int t = tid; t < nb * d; t += FA_BR) {
            const int r = t / d, c = t % d;
            const size_t off = base + (size_t)(j0 + r) * d + c;
            sK[t] = K[off];
            sV[t] = V[off];
        }
        __syncthreads();

        if (active) {
            const int j1 = ((j0 + FA_BC) < (causal ? (i + 1) : L)) ? (j0 + FA_BC) : (causal ? (i + 1) : L);
            for (int j = j0; j < j1; ++j) {
                // ---- S = scale·(q·k_j) + bias ----
                const float* k = &sK[(size_t)(j - j0) * d];
                float s = 0.f;
                for (int dd = 0; dd < d; ++dd) s += qreg[dd] * k[dd];
                s *= scale;
                if (Bias) {
                    s += Bias[((size_t)bh * (size_t)L + (size_t)i) * (size_t)L + (size_t)j];
                }
                if (causal && j > i) s = -INFINITY;      // 对角块逐元素掩码 ✓

                // ---- 在线 softmax：仅在行最大值更新时缩放（两处 -inf 守卫避免 NaN ✗）----
                if (s > m) {
                    const float alpha = (m == -INFINITY) ? 0.f : expf(m - s);
                    l *= alpha;
                    for (int dd = 0; dd < d; ++dd) acc[dd] *= alpha;
                    m = s;
                }
                const float beta = (s == -INFINITY) ? 0.f : expf(s - m);
                l += beta;
                const float* v = &sV[(size_t)(j - j0) * d];
                for (int dd = 0; dd < d; ++dd) acc[dd] += beta * v[dd];
            }
        }
        __syncthreads();     // 所有线程用完本 tile 才能覆写 ✓
    }

    // ---- 收尾：O = acc/(l+eps)、LSE = m + log(l+eps)；l==0（全屏蔽行）⇒ O=0、LSE=-inf ✓ ----
    if (active) {
        float* o = O + base + (size_t)i * d;
        if (l > 0.f) {
            const float inv = 1.0f / (l + eps);
            for (int dd = 0; dd < d; ++dd) o[dd] = acc[dd] * inv;
            if (LSE) LSE[(size_t)bh * (size_t)L + (size_t)i] = m + logf(l + eps);
        } else {
            for (int dd = 0; dd < d; ++dd) o[dd] = 0.f;
            if (LSE) LSE[(size_t)bh * (size_t)L + (size_t)i] = -INFINITY;
        }
    }
}

}  // namespace

int flash_attn_forward_cuda(const FlashAttnConfig& cfg,
                            const float* Q, const float* K, const float* V,
                            const float* bias,
                            float* O, float* LSE) {
    if (!Q || !K || !V || !O) return 1;
    if (cfg.B <= 0 || cfg.h <= 0 || cfg.L <= 0 || cfg.d <= 0) return 1;
    if (cfg.d > FA_D_MAX) return 1;                    // v1 限制 ⇒ 调用方回落 CPU ✓

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) return 1;

    const float scale = (cfg.scale > 0.f) ? cfg.scale : 1.0f / std::sqrt((float)cfg.d);
    const dim3 grid((unsigned)((cfg.L + FA_BR - 1) / FA_BR), (unsigned)(cfg.B * cfg.h));
    const dim3 block((unsigned)FA_BR);

    flash_attn_fwd_kernel<<<grid, block>>>(Q, K, V, bias, O, LSE,
                                           cfg.L, cfg.d, scale,
                                           cfg.causal ? 1 : 0, cfg.softmax_eps);
    if (cudaGetLastError() != cudaSuccess) return 1;
    return 0;
}

// ============================================================
// 反向（Step ③）：flash_attn_backward_cuda —— 与 CPU 版同数学 ✓
//   P_ij = exp(S_ij − LSE_i)（与 forward 的 (l+eps) 归一化严格一致 ✓）
//   D_i  = Σ_d dO·O；dS = P·(dP − D)；dQ/dK 带 scale ✓、dV 不带 ✓、dbias = dS ✓
//
//   v1 线程分工（同前向：**一线程一行 query** ✓）：
//     · qreg / doreg 常驻寄存器；dQ 累加器 accQ 也在寄存器 ⇒ **dQ 完全无原子 ✓**
//     · K/V tile 进 smem（同上：同指令全线程读同一地址 ⇒ 广播、无 bank conflict ✓）
//     · dK/dV 用 **atomicAdd**（j 轴跨线程共享 ⇒ v1 只能原子 ✗；v2 再改 smem 分层归约 ✓）
//     · dbias 每个 (i,j) 恰被一个线程写 ⇒ **直接赋值** ✓（越界/掩码位由 host memset 保证 0 ✓）
//    head_dim ≤ 32：寄存器 qreg+doreg+accQ ≈ 3d 个 float ⇒ d>32 会溢出 ✗（返回 1 回落 CPU ✓）
//    LSE == -inf 的行（全屏蔽）：P≡0 ⇒ 该行梯度 0 ⇒ 跳过 ✓
//    dK/dV 原子 ⇒ 求和顺序非确定 ⇒ 与 CPU 对拍容差 ~1e-3 ✓
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
                                      const int L, const int d,
                                      const float scale, const int causal) {
    __shared__ float sK[FA_BC * FA_DB_D_MAX];
    __shared__ float sV[FA_BC * FA_DB_D_MAX];

    const int tid  = (int)threadIdx.x;
    const int bh   = (int)blockIdx.y;
    const int row0 = (int)blockIdx.x * FA_BR;
    const int i    = row0 + tid;
    const bool active = (i < L);
    const size_t base = (size_t)bh * (size_t)L * (size_t)d;

    int jmax_block = L;
    if (causal) {
        const int last_row = (row0 + FA_BR - 1 < L) ? (row0 + FA_BR - 1) : (L - 1);
        jmax_block = last_row + 1;
    }

    const float lse    = (active && LSE) ? LSE[(size_t)bh * (size_t)L + (size_t)i] : -INFINITY;
    const bool  live   = active && (lse > -INFINITY);     // 全屏蔽行 ⇒ 整行梯度 0 ⇒ 跳过 ✓

    // ---- 本线程的行数据（全寄存器 ✓）----
    float qreg[FA_DB_D_MAX], doreg[FA_DB_D_MAX], accQ[FA_DB_D_MAX];
    float D = 0.f;                                        // D_i = Σ_d dO·O ✓
    for (int dd = 0; dd < d; ++dd) {
        qreg[dd]  = live ? Q[base + (size_t)i * d + dd] : 0.f;
        doreg[dd] = live ? dO[base + (size_t)i * d + dd] : 0.f;
        accQ[dd]  = 0.f;
        if (live) D += doreg[dd] * O[base + (size_t)i * d + dd];
    }

    for (int j0 = 0; j0 < jmax_block; j0 += FA_BC) {
        const int nb = ((j0 + FA_BC) < jmax_block) ? FA_BC : (jmax_block - j0);
        for (int t = tid; t < nb * d; t += FA_BR) {
            const int r = t / d, c = t % d;
            const size_t off = base + (size_t)(j0 + r) * d + c;
            sK[t] = K[off];
            sV[t] = V[off];
        }
        __syncthreads();

        if (live) {
            int j1 = j0 + FA_BC;
            if (j1 > (causal ? (i + 1) : L)) j1 = causal ? (i + 1) : L;
            for (int j = j0; j < j1; ++j) {
                // ---- S ⇒ P（与 forward 同一 LSE 口径 ✓）----
                const float* k = &sK[(size_t)(j - j0) * d];
                float s = 0.f;
                for (int dd = 0; dd < d; ++dd) s += qreg[dd] * k[dd];
                s *= scale;
                if (Bias) s += Bias[((size_t)bh * (size_t)L + (size_t)i) * (size_t)L + (size_t)j];
                const float p = expf(s - lse);

                const float* v = &sV[(size_t)(j - j0) * d];
                float dp = 0.f;
                for (int dd = 0; dd < d; ++dd) dp += doreg[dd] * v[dd];
                const float ds = p * (dp - D);

                // ---- 梯度累加 ----
                if (dBias) dBias[((size_t)bh * (size_t)L + (size_t)i) * (size_t)L + (size_t)j] = ds;
                for (int dd = 0; dd < d; ++dd) accQ[dd] += scale * ds * k[dd];
                if (dK || dV) {
                    const size_t off_j = base + (size_t)j * d;
                    for (int dd = 0; dd < d; ++dd) {
                        if (dK) atomicAdd(&dK[off_j + dd], scale * ds * qreg[dd]);
                        if (dV) atomicAdd(&dV[off_j + dd], p * doreg[dd]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (active && dQ) {                                    // dQ 无原子（每元素只被本线程写 ✓）
        float* dq = dQ + base + (size_t)i * d;
        for (int dd = 0; dd < d; ++dd) dq[dd] = accQ[dd];
    }
}

int flash_attn_backward_cuda(const FlashAttnConfig& cfg,
                             const float* Q, const float* K, const float* V,
                             const float* bias, const float* O, const float* dO, const float* LSE,
                             float* dQ, float* dK, float* dV, float* dbias) {
    if (!Q || !K || !V || !O || !dO || !LSE) return 1;
    if (cfg.B <= 0 || cfg.h <= 0 || cfg.L <= 0 || cfg.d <= 0) return 1;
    if (cfg.d > FA_DB_D_MAX) return 1;                     // 寄存器上限 ⇒ 调用方回落 CPU ✓
    if (!dQ && !dK && !dV && !dbias) return 1;

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) return 1;

    const float scale = (cfg.scale > 0.f) ? cfg.scale : 1.0f / std::sqrt((float)cfg.d);
    const size_t nq = (size_t)cfg.B * cfg.h * cfg.L * cfg.d;
    const size_t nb_ = (size_t)cfg.B * cfg.h * cfg.L * cfg.L;

    // ★ 输出约定：先清零（dK/dV 靠原子累加、dbias 靠 memset 保证掩码位为 0 ✓）
    if (dQ    && cudaMemset(dQ,    0, nq  * sizeof(float)) != cudaSuccess) return 1;
    if (dK    && cudaMemset(dK,    0, nq  * sizeof(float)) != cudaSuccess) return 1;
    if (dV    && cudaMemset(dV,    0, nq  * sizeof(float)) != cudaSuccess) return 1;
    if (dbias && cudaMemset(dbias, 0, nb_ * sizeof(float)) != cudaSuccess) return 1;

    const dim3 grid((unsigned)((cfg.L + FA_BR - 1) / FA_BR), (unsigned)(cfg.B * cfg.h));
    const dim3 block((unsigned)FA_BR);
    flash_attn_bwd_kernel<<<grid, block>>>(Q, K, V, bias, O, dO, LSE, dQ, dK, dV, dbias,
                                           cfg.L, cfg.d, scale, cfg.causal ? 1 : 0);
    if (cudaGetLastError() != cudaSuccess) return 1;
    return 0;
}

}  // namespace ppml

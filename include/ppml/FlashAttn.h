// ============================================================
// FlashAttn.h — Flash Attention v1（1-pass）接口 —— Step ① CPU 参考实现（2026-09-22）
//
// 目的：用**一个算子**替换现有的「out_prod(scores) → soft_max → out_prod(PV)」三段
//   （现状见 src/cuda/AttentionKernel.cu：把 L×L 的 scores 落到 HBM ✗，显存与 HBM 流量都是 O(L²) ✗）。
//
// 本项目布局约定（ggml：dims[0] 最内）：
//   · Q/K/V/O : [d, L, h, B]（= torch (B, h, L, d)）—— 元素 (b,hh,i,dd) 偏移 = ((b*h+hh)*L + i)*d + dd ✓
//   · LSE     : [L, h, B]（dims[0]=L ✓ 与 O 的行一一对应 ✓）—— 偏移 = (b*h+hh)*L + i
//   · bias    : [Lk, Lq, h, B]（可选；dims[0]=**key** 轴 ✓ 与 softmax 沿 dims[0] 的项目口径一致 ✓）
//               —— 元素 (b,hh,i,j) 偏移 = ((b*h+hh)*L + i)*L + j（i=query, j=key）✓
//
// 数学：O = softmax(scale·QKᵀ + bias) · V，scale = 1/√d；causal ⇒ j ≤ i（默认开 ✓）
//   前向 1-pass：不落地 P/S，只写 O 与 LSE = m + log(l)（反向用 P = exp(S − LSE) 重算 ✓）
//
// ⚠️ softmax 分母口径：项目图 softmax 用 1/(Σ+1e-9)（见 2026-09-11 统一），而旧 AttentionKernel.cu
//   是 exp(x−max)/Σ（无 eps）✗ ⇒ 这里 `softmax_eps` 可配（默认 1e-9 对齐项目口径 ✓；
//   与旧 kernel 对拍时置 0 ⇒ 数学等价 ✓，差异仅 ~1e-9 相对量级 ✓）。
// ============================================================
#pragma once

#include <cstddef>

namespace ppml {

struct FlashAttnConfig {
    int   B      = 1;        // batch
    int   h      = 1;        // heads
    int   L      = 0;        // sequence length（Q 与 K/V 相同长度；cross-attention 将来另加 Lk）
    int   d      = 0;        // head dim
    bool  causal = true;     // 因果掩码（j ≤ i）✓
    float scale  = 0.f;      // 0 ⇒ 用 1/sqrt(d)
    float softmax_eps = 1e-9f;   // 分母 eps（0 ⇒ 与旧 kernel 完全一致 ✓）
    int   Br     = 64;       // Q 行块（Step ① 只影响循环结构；CUDA 版用它定线程/寄存器 ✓）
    int   Bc     = 64;       // K 列块（在线 softmax 的分块宽度 ✓）
};

// 前向（blocked online-softmax，等价于朴素 softmax 注意力 ✓）
//   Q/K/V/O 均连续 fp32；LSE 可为 nullptr（不需要时 ✓）；bias 可为 nullptr ✓
//   返回 false = 参数非法（空指针/非正维度）
bool flash_attn_forward_cpu(const FlashAttnConfig& cfg,
                            const float* Q, const float* K, const float* V,
                            const float* bias,
                            float* O, float* LSE);

// 朴素三遍参考（scores → softmax → PV），数学严格对齐旧 AttentionKernel.cu：
//   scale = 1/sqrt(d)、softmax = exp(x − max)/Σ（**不含 eps**），用于「与旧路径对拍」✓
//   （仅供测试/校验；性能不是目标 ✓）
bool flash_attn_forward_ref_cpu(const FlashAttnConfig& cfg,
                                const float* Q, const float* K, const float* V,
                                const float* bias,
                                float* O, float* LSE);

// ============================================================
// CUDA 前向（Step ②）—— 与 flash_attn_forward_cpu **同签名、同数学** ⇒ 可 1:1 对拍 ✓
//   返回 0 = 已执行；1 = 不支持（无 CUDA 设备 / head_dim > 64 / 参数非法）⇒ 调用方回落 CPU ✓
//   ⚠️ 调用方负责：Q/K/V/O/LSE（及 bias）都是**设备指针**且已分配；本函数不分配显存 ✓
//   （kernel 见 src/cuda/FlashAttnKernel.cu；v1 = 一线程一行、无 warp 归约、无 cp.async ✓）
// ============================================================
int flash_attn_forward_cuda(const FlashAttnConfig& cfg,
                            const float* Q, const float* K, const float* V,
                            const float* bias,
                            float* O, float* LSE);

// ============================================================
// 反向（Step ③ 前向=无/③=dQ + ④=dK,dV）—— 与 forward 同布局、同 eps 口径 ✓
//   记 LSE 已含 eps（forward：LSE = m + log(l+eps)），则
//     P_ij = exp(S_ij − LSE_i)  **与 forward 的归一化严格一致** ✓（= exp(s−m)/(l+eps) ✓）
//     D_i  = Σ_d dO_id·O_id
//     dS_ij = P_ij·(Σ_d dO_id·V_jd − D_i)
//     dQ_id = scale·Σ_j dS_ij·K_jd     dK_jd = scale·Σ_i dS_ij·Q_id     dV_jd = Σ_i P_ij·dO_id
//     dbias_ij = dS_ij（★ 掩码位（causal j>i）与全屏蔽行 ⇒ 0 ✓）
//   输出约定：本函数**先清零** dQ/dK/dV/dbias 再累加 ✓；传 nullptr 的项不计算（也不清零 ✓）
//   LSE_i == -inf（全屏蔽行）⇒ 该行所有梯度为 0 ✓（原先前向输出 O_i = 0 ✓）
// ============================================================
bool flash_attn_backward_cpu(const FlashAttnConfig& cfg,
                             const float* Q, const float* K, const float* V,
                             const float* bias, const float* O, const float* dO, const float* LSE,
                             float* dQ, float* dK, float* dV, float* dbias);

// CUDA 反向（Step ③）—— 与 CPU 版**同签名同数学** ⇒ 1:1 对拍 ✓
//   返回 0 = 已执行；1 = 不支持（无设备 / head_dim > 32 / 缺 O,dO,LSE / 参数非法）⇒ 回落 CPU ✓
//   ⚠️ dQ/dK/dV/dbias 为**设备指针**（可 nullptr 跳过 ✓）；本函数内部先 cudaMemset 清零 ✓
//   ⚠️ dK/dV 走 atomicAdd（求和顺序非确定 ⇒ 对拍容差放宽到 ~1e-3 ✓）；v1 = 一线程一行 ✓
int flash_attn_backward_cuda(const FlashAttnConfig& cfg,
                             const float* Q, const float* K, const float* V,
                             const float* bias, const float* O, const float* dO, const float* LSE,
                             float* dQ, float* dK, float* dV, float* dbias);

// 便捷尺寸计算（测试与将来图节点用 ✓）
inline size_t flash_attn_attn_numel(int B, int h, int L, int d) { return (size_t)B * h * L * d; }
inline size_t flash_attn_lse_numel (int B, int h, int L)        { return (size_t)B * h * L; }
inline size_t flash_attn_bias_numel(int B, int h, int L)        { return (size_t)B * h * L * L; }

}  // namespace ppml

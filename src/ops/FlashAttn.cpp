// ============================================================
// FlashAttn.cpp — Flash Attention v1（1-pass）CPU 参考实现（Step ①，2026-09-22）
//
//   · flash_attn_forward_cpu     ：blocked online-softmax（与将来 CUDA 版同一算法骨架 ✓）
//   · flash_attn_forward_ref_cpu ：朴素三遍参考（= 旧 AttentionKernel.cu 数学 ✓，用于对拍）
//
//   ⚠️ Step ① 目标 = **正确性基线**：单线程、标量、无向量化（AVX2/ThreadPool 放 v2 ✓）。
//      分块只是为了让 CPU 版与 CUDA 版共用同一「沿 K 分块 + 在线 softmax」结构，便于逐块对拍 ✓。
//
//   布局与口径见 include/ppml/FlashAttn.h 顶部注释（dims[0] 最内；LSE [L,h,B]；bias [Lk,Lq,h,B]）✓
// ============================================================
#include "ppml/FlashAttn.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace ppml {

namespace {

// 元素偏移（布局见头文件）
inline size_t off_qkv(int b, int hh, int i, int d, int L, int h) {
    return ((size_t)b * h + hh) * (size_t)L * (size_t)d + (size_t)i * (size_t)d;
}
inline size_t off_lse(int b, int hh, int i, int L, int h) {
    return ((size_t)b * h + hh) * (size_t)L + (size_t)i;
}
inline size_t off_bias(int b, int hh, int i, int j, int L, int h) {
    return (((size_t)b * h + hh) * (size_t)L + (size_t)i) * (size_t)L + (size_t)j;
}

// 参数合法性
bool cfg_ok(const FlashAttnConfig& cfg, const float* Q, const float* K, const float* V, float* O) {
    if (!Q || !K || !V || !O) return false;
    if (cfg.B <= 0 || cfg.h <= 0 || cfg.L <= 0 || cfg.d <= 0) return false;
    return true;
}

}  // namespace

// ------------------------------------------------------------
// 1) blocked online-softmax 前向（1-pass：只写 O 与 LSE ✓）
//    对每个 query 行 i：
//      m,l,acc 从 -inf/0/0 起，沿 K 按 Bc 分块流式推进（causal ⇒ 只到 j = i ✓，省 ~50% ✓）
//      m_new = max(m, tile_max)；alpha = exp(m - m_new)（m=-inf 时取 0，避免 exp(-inf - -inf)=NaN ✗）
//      acc = alpha*acc + P·V_tile；l = alpha*l + Σ P
//    收尾：O = acc/(l+eps)；LSE = m + log(l+eps)；l==0（全屏蔽行）⇒ O=0、LSE=-inf ✓
// ------------------------------------------------------------
bool flash_attn_forward_cpu(const FlashAttnConfig& cfg,
                            const float* Q, const float* K, const float* V,
                            const float* bias,
                            float* O, float* LSE) {
    if (!cfg_ok(cfg, Q, K, V, O)) return false;

    const int   B  = cfg.B, H = cfg.h, L = cfg.L, d = cfg.d;
    const int   Bc = (cfg.Bc > 0) ? cfg.Bc : 64;
    const float sc = (cfg.scale > 0.f) ? cfg.scale : 1.0f / std::sqrt((float)d);
    const float eps = cfg.softmax_eps;

    std::vector<float> acc((size_t)d, 0.f);
    std::vector<float> s_tile;                 // 本 tile 的 S（复用为 P ✓）

    for (int b = 0; b < B; ++b) {
        for (int hh = 0; hh < H; ++hh) {
            const float* Qb = Q + off_qkv(b, hh, 0, d, L, H);
            const float* Kb = K + off_qkv(b, hh, 0, d, L, H);
            const float* Vb = V + off_qkv(b, hh, 0, d, L, H);
            float*       Ob = O + off_qkv(b, hh, 0, d, L, H);
            float*       Lb = LSE ? (LSE + off_lse(b, hh, 0, L, H)) : nullptr;
            const float* Bb = bias;   // off_bias(...) 在行内用 ✓

            for (int i = 0; i < L; ++i) {
                const float* q = Qb + (size_t)i * d;
                const int jmax = cfg.causal ? (i + 1) : L;   // causal ⇒ 只到 i ✓

                float m = -INFINITY;   // running max
                float l = 0.f;         // running Σ exp（未归一化）
                std::fill(acc.begin(), acc.end(), 0.f);

                for (int j0 = 0; j0 < jmax; j0 += Bc) {
                    const int j1 = std::min(j0 + Bc, jmax);
                    const int nb = j1 - j0;
                    s_tile.resize((size_t)nb);

                    // --- (a) S_j = scale·(q·k_j) + bias[j][i]，同时取 tile 内行最大 ---
                    float m_tile = -INFINITY;
                    for (int j = j0; j < j1; ++j) {
                        const float* k = Kb + (size_t)j * d;
                        float s = 0.f;
                        for (int dd = 0; dd < d; ++dd) s += q[dd] * k[dd];
                        s *= sc;
                        if (Bb) s += Bb[off_bias(b, hh, i, j, L, H)];   // bias 基址 = 全局基址 ✓
                        s_tile[(size_t)(j - j0)] = s;
                        if (s > m_tile) m_tile = s;
                    }

                    // --- (b) 在线 softmax 合并：alpha = exp(m - m_new) ---
                    const float m_new = (m > m_tile) ? m : m_tile;
                    const float alpha = (m == -INFINITY) ? 0.f : std::exp(m - m_new);

                    float l_tile = 0.f;
                    for (int j = 0; j < nb; ++j) {
                        const float p = std::exp(s_tile[(size_t)j] - m_new);
                        s_tile[(size_t)j] = p;      // 复用为 P ✓
                        l_tile += p;
                    }

                    // --- (c) acc = alpha·acc + P·V_tile ---
                    for (int dd = 0; dd < d; ++dd) {
                        float pv = 0.f;
                        for (int j = 0; j < nb; ++j) pv += s_tile[(size_t)j] * Vb[(size_t)(j0 + j) * d + dd];
                        acc[(size_t)dd] = alpha * acc[(size_t)dd] + pv;
                    }
                    l = alpha * l + l_tile;
                    m = m_new;
                }

                // --- 收尾：写 O 与 LSE（eps 口径与项目 softmax 一致 ✓） ---
                float* o = Ob + (size_t)i * d;
                if (l > 0.f) {
                    const float inv = 1.0f / (l + eps);
                    for (int dd = 0; dd < d; ++dd) o[dd] = acc[(size_t)dd] * inv;
                    if (Lb) Lb[i] = m + std::log(l + eps);
                } else {
                    // 全屏蔽行（如 bias 全为 -inf）：O=0、LSE=-inf ✓（反向据此跳过该行 ✓）
                    for (int dd = 0; dd < d; ++dd) o[dd] = 0.f;
                    if (Lb) Lb[i] = -INFINITY;
                }
            }
        }
    }
    return true;
}

// ------------------------------------------------------------
// 3) 反向（Step ③）：朴素逐行实现（先清零输出，再按 i 累加 ✓）
//    P = exp(S − LSE)（与前面 forward 的 (l+eps) 归一化严格一致 ✓）
//    D_i = Σ_d dO·O；dS = P·(dP − D)；dQ/dK 带 scale ✓、dV 不带 ✓、dbias = dS ✓
//    causal ⇒ 只访问 j ≤ i ⇒ dbias 上三角保持 0 ✓（先整块清零 ✓）
//    LSE == -inf 的行 ⇒ 整行梯度 0 ⇒ 直接跳过 ✓
// ------------------------------------------------------------
bool flash_attn_backward_cpu(const FlashAttnConfig& cfg,
                             const float* Q, const float* K, const float* V,
                             const float* bias, const float* O, const float* dO, const float* LSE,
                             float* dQ, float* dK, float* dV, float* dbias) {
    if (!Q || !K || !V || !O || !dO || !LSE) return false;
    if (cfg.B <= 0 || cfg.h <= 0 || cfg.L <= 0 || cfg.d <= 0) return false;
    if (!dQ && !dK && !dV && !dbias) return false;

    const int   B = cfg.B, H = cfg.h, L = cfg.L, d = cfg.d;
    const float sc = (cfg.scale > 0.f) ? cfg.scale : 1.0f / std::sqrt((float)d);
    const size_t nq = (size_t)B * H * L * d;
    const size_t nb = (size_t)B * H * L * L;

    if (dQ)    std::fill(dQ,    dQ    + nq, 0.f);
    if (dK)    std::fill(dK,    dK    + nq, 0.f);
    if (dV)    std::fill(dV,    dV    + nq, 0.f);
    if (dbias) std::fill(dbias, dbias + nb, 0.f);      // ★ 掩码位必须是 0 ✓

    for (int b = 0; b < B; ++b) {
        for (int hh = 0; hh < H; ++hh) {
            const float* Qb = Q + off_qkv(b, hh, 0, d, L, H);
            const float* Kb = K + off_qkv(b, hh, 0, d, L, H);
            const float* Vb = V + off_qkv(b, hh, 0, d, L, H);
            const float* Ob = O + off_qkv(b, hh, 0, d, L, H);
            const float* dOb = dO + off_qkv(b, hh, 0, d, L, H);
            const float* Lb = LSE + off_lse(b, hh, 0, L, H);

            for (int i = 0; i < L; ++i) {
                const float lse = Lb[i];
                if (!(lse > -INFINITY)) continue;                    // 全屏蔽行 ⇒ 梯度全 0 ✓

                const float* q     = Qb + (size_t)i * d;
                const float* o_row = Ob + (size_t)i * d;
                const float* do_row = dOb + (size_t)i * d;
                float*       dq_row = dQ ? (dQ + off_qkv(b, hh, i, d, L, H)) : nullptr;

                float D = 0.f;                                       // D_i = Σ_d dO·O ✓
                for (int dd = 0; dd < d; ++dd) D += do_row[dd] * o_row[dd];

                const int jmax = cfg.causal ? (i + 1) : L;
                for (int j = 0; j < jmax; ++j) {
                    const float* k = Kb + (size_t)j * d;
                    float s = 0.f;
                    for (int dd = 0; dd < d; ++dd) s += q[dd] * k[dd];
                    s *= sc;
                    if (bias) s += bias[off_bias(b, hh, i, j, L, H)];
                    const float p = std::exp(s - lse);

                    float dp = 0.f;                                  // dP_ij = Σ_d dO·V ✓
                    for (int dd = 0; dd < d; ++dd) dp += do_row[dd] * Vb[(size_t)j * d + dd];
                    const float ds = p * (dp - D);

                    if (dbias) dbias[off_bias(b, hh, i, j, L, H)] = ds;
                    if (dq_row) for (int dd = 0; dd < d; ++dd) dq_row[dd] += sc * ds * k[dd];
                    if (dK) {
                        float* dk = dK + off_qkv(b, hh, j, d, L, H);
                        for (int dd = 0; dd < d; ++dd) dk[dd] += sc * ds * q[dd];
                    }
                    if (dV) {
                        float* dv = dV + off_qkv(b, hh, j, d, L, H);
                        for (int dd = 0; dd < d; ++dd) dv[dd] += p * do_row[dd];
                    }
                }
            }
        }
    }
    return true;
}

// ------------------------------------------------------------
// 2) 朴素三遍参考（与旧 AttentionKernel.cu 数学一致，**无 eps** ✓）
//    scores[i][j] = scale·(q_i·k_j) + bias[j][i]
//    softmax 行内：x - max ⇒ exp ⇒ /Σ；O_i = Σ_j p_ij v_j；LSE_i = max + log Σ ✓
// ------------------------------------------------------------
bool flash_attn_forward_ref_cpu(const FlashAttnConfig& cfg,
                                const float* Q, const float* K, const float* V,
                                const float* bias,
                                float* O, float* LSE) {
    if (!cfg_ok(cfg, Q, K, V, O)) return false;

    const int   B = cfg.B, H = cfg.h, L = cfg.L, d = cfg.d;
    const float sc = (cfg.scale > 0.f) ? cfg.scale : 1.0f / std::sqrt((float)d);

    std::vector<float> p((size_t)L, 0.f);

    for (int b = 0; b < B; ++b) {
        for (int hh = 0; hh < H; ++hh) {
            const float* Qb = Q + off_qkv(b, hh, 0, d, L, H);
            const float* Kb = K + off_qkv(b, hh, 0, d, L, H);
            const float* Vb = V + off_qkv(b, hh, 0, d, L, H);
            float*       Ob = O + off_qkv(b, hh, 0, d, L, H);
            float*       Lb = LSE ? (LSE + off_lse(b, hh, 0, L, H)) : nullptr;

            for (int i = 0; i < L; ++i) {
                const float* q = Qb + (size_t)i * d;
                const int jmax = cfg.causal ? (i + 1) : L;

                float mx = -INFINITY;
                for (int j = 0; j < jmax; ++j) {
                    const float* k = Kb + (size_t)j * d;
                    float s = 0.f;
                    for (int dd = 0; dd < d; ++dd) s += q[dd] * k[dd];
                    s *= sc;
                    if (bias) s += bias[off_bias(b, hh, i, j, L, H)];
                    p[(size_t)j] = s;
                    if (s > mx) mx = s;
                }
                float sum = 0.f;
                for (int j = 0; j < jmax; ++j) { p[(size_t)j] = std::exp(p[(size_t)j] - mx); sum += p[(size_t)j]; }

                float* o = Ob + (size_t)i * d;
                for (int dd = 0; dd < d; ++dd) {
                    float pv = 0.f;
                    for (int j = 0; j < jmax; ++j) pv += p[(size_t)j] * Vb[(size_t)j * d + dd];
                    o[dd] = pv / sum;                        // 旧 kernel：无 eps ✓
                }
                if (Lb) Lb[i] = (sum > 0.f) ? (mx + std::log(sum)) : -INFINITY;
            }
        }
    }
    return true;
}

}  // namespace ppml

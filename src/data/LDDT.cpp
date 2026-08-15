#include "ppml/LDDT.h"
#include <algorithm>
#include <vector>
#include <cmath>

namespace ppml {

// ============================================================================
// LDDT 距离阈值 (Mariani et al. 2013)
// 4 个阈值: 0.5Å, 1Å, 2Å, 4Å
// 对每个残基对，检查 |d_pred - d_true| < 阈值 是否成立
// LDDT_i = 在所有阈值上正确的接触对数 / 有效接触总数
// ============================================================================
static constexpr float LDDT_THRESHOLDS[] = {0.5f, 1.0f, 2.0f, 4.0f};
static constexpr int   N_THRESHOLDS = 4;

// ============================================================================
// compute_lddt_ca
// ============================================================================

void compute_lddt_ca(
    const float* pred_coords,
    const float* true_coords,
    const float* mask,
    int N,
    float cutoff,
    float* lddt_out)
{
    // 初始化输出
    std::memset(lddt_out, 0, N * sizeof(float));

    if (N == 0) return;

    // ================================================================
    // Step 1: 计算所有残基对的距离
    // ================================================================

    // 预计算 CA 距离矩阵 (N, N)
    std::vector<float> pred_dist(N * N, 0.0f);
    std::vector<float> true_dist(N * N, 0.0f);

    for (int i = 0; i < N; ++i) {
        if (mask[i] < 0.5f) continue;

        float pix = pred_coords[i * 3 + 0];
        float piy = pred_coords[i * 3 + 1];
        float piz = pred_coords[i * 3 + 2];

        float tix = true_coords[i * 3 + 0];
        float tiy = true_coords[i * 3 + 1];
        float tiz = true_coords[i * 3 + 2];

        for (int j = i + 1; j < N; ++j) {
            if (mask[j] < 0.5f) continue;

            // 预测距离
            float dx_p = pix - pred_coords[j * 3 + 0];
            float dy_p = piy - pred_coords[j * 3 + 1];
            float dz_p = piz - pred_coords[j * 3 + 2];
            float dp = std::sqrt(dx_p * dx_p + dy_p * dy_p + dz_p * dz_p);

            // 真实距离
            float dx_t = tix - true_coords[j * 3 + 0];
            float dy_t = tiy - true_coords[j * 3 + 1];
            float dz_t = tiz - true_coords[j * 3 + 2];
            float dt = std::sqrt(dx_t * dx_t + dy_t * dy_t + dz_t * dz_t);

            pred_dist[i * N + j] = dp;
            pred_dist[j * N + i] = dp;
            true_dist[i * N + j] = dt;
            true_dist[j * N + i] = dt;
        }
    }

    // ================================================================
    // Step 2: per-residue LDDT
    // ================================================================

    for (int i = 0; i < N; ++i) {
        if (mask[i] < 0.5f) {
            lddt_out[i] = 0.0f;
            continue;
        }

        // 统计: 对残基 i，哪些残基 j 在 cutoff 内 (基于真实距离)
        int total_pairs = 0;           // 总有效接触数
        int preserved[N_THRESHOLDS] = {0};  // 每个阈值下正确数

        for (int j = 0; j < N; ++j) {
            if (i == j || mask[j] < 0.5f) continue;

            float dt = true_dist[i * N + j];
            if (dt >= cutoff) continue;  // 只考虑真实距离 < cutoff 的接触

            total_pairs++;
            float dp = pred_dist[i * N + j];
            float abs_diff = std::abs(dp - dt);

            for (int t = 0; t < N_THRESHOLDS; ++t) {
                if (abs_diff < LDDT_THRESHOLDS[t]) {
                    preserved[t]++;
                }
            }
        }

        // LDDT_i = (1/N_THRESHOLDS) * sum_t(preserved[t] / total_pairs)
        if (total_pairs > 0) {
            float lddt_sum = 0.0f;
            for (int t = 0; t < N_THRESHOLDS; ++t) {
                lddt_sum += static_cast<float>(preserved[t]) / total_pairs;
            }
            lddt_out[i] = lddt_sum / N_THRESHOLDS;
        } else {
            lddt_out[i] = 0.0f;
        }
    }
}

// ============================================================================
// lddt_to_onehot
// ============================================================================

void lddt_to_onehot(
    const float* lddt,
    int N,
    int num_bins,
    float* onehot_out)
{
    std::memset(onehot_out, 0, N * num_bins * sizeof(float));

    for (int i = 0; i < N; ++i) {
        // bin_index = floor(lddt * num_bins)
        int bin = static_cast<int>(lddt[i] * num_bins);
        // clamp to [0, num_bins-1]
        bin = std::max(0, std::min(num_bins - 1, bin));
        onehot_out[i * num_bins + bin] = 1.0f;
    }
}

// ============================================================================
// logits_to_plddt — 推理用: logits → pLDDT scores
// ============================================================================

void logits_to_plddt(
    const float* logits,
    int N,
    int num_bins,
    float* plddt_out)
{
    float bin_width = 1.0f / num_bins;

    for (int i = 0; i < N; ++i) {
        const float* row = logits + i * num_bins;

        // Step 1: softmax
        float max_val = row[0];
        for (int b = 1; b < num_bins; ++b) {
            max_val = std::max(max_val, row[b]);
        }

        float sum_exp = 0.0f;
        float probs[64];  // num_bins ≤ 64, 栈上分配
        for (int b = 0; b < num_bins; ++b) {
            probs[b] = std::exp(row[b] - max_val);
            sum_exp += probs[b];
        }
        for (int b = 0; b < num_bins; ++b) {
            probs[b] /= sum_exp;
        }

        // Step 2: 期望值 = sum(prob_b * center_b)
        float expected = 0.0f;
        for (int b = 0; b < num_bins; ++b) {
            float center = (b + 0.5f) * bin_width;
            expected += probs[b] * center;
        }

        plddt_out[i] = expected * 100.0f;
    }
}

} // namespace ppml

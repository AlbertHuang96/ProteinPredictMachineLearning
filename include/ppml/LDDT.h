#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>

namespace ppml {

// ============================================================================
// LDDT (Local Distance Difference Test) 计算
// ============================================================================

/**
 * @brief 计算 per-residue LDDT-Cα
 * 
 * 对应 AlphaFold lddt.lddt(per_residue=True):
 *   对每个残基，检查其 Cα 与所有其他残基 Cα 的距离，
 *   若预测距离与真实距离的差异 < 阈值，则该接触"正确"。
 *   LDDT_i = (正确接触数) / (有效接触总数)
 * 
 * 参考: Mariani et al. (2013) "lDDT: a local superposition-free score
 *       for comparing protein structures"
 * 
 * @param pred_coords  预测 CA 坐标 (N, 3) 扁平数组
 * @param true_coords  真实 CA 坐标 (N, 3) 扁平数组
 * @param mask         残基有效性 (N,) 1=有效 0=忽略
 * @param N            残基数
 * @param cutoff       距离截断 (Å), 仅考虑 < cutoff 的残基对, 默认 15.0
 * @param lddt_out     (out) per-residue LDDT (N floats), ∈ [0, 1]
 */
void compute_lddt_ca(
    const float* pred_coords,
    const float* true_coords,
    const float* mask,
    int N,
    float cutoff,
    float* lddt_out);

/**
 * @brief 将 LDDT 值 binning 为 one-hot (用于 CE loss)
 * 
 * bin_index = floor(lddt * num_bins), clamp to [0, num_bins-1]
 * 
 * @param lddt        per-residue LDDT (N,) ∈ [0, 1]
 * @param N           残基数
 * @param num_bins    bin 数 (通常 50)
 * @param onehot_out  (out) one-hot labels (N * num_bins floats)
 */
void lddt_to_onehot(
    const float* lddt,
    int N,
    int num_bins,
    float* onehot_out);

/**
 * @brief 从 logits 计算 pLDDT 分数 (推理用)
 * 
 * pLDDT = 100 * sum(softmax(logits) * bin_centers)
 * bin_centers[b] = (b + 0.5) / num_bins
 * 
 * @param logits     logits (N, num_bins) 扁平数组
 * @param N          残基数
 * @param num_bins   bin 数
 * @param plddt_out  (out) pLDDT scores (N floats), ∈ [0, 100]
 */
void logits_to_plddt(
    const float* logits,
    int N,
    int num_bins,
    float* plddt_out);

} // namespace ppml

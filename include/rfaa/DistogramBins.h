#pragma once

#include "Tensor.h"
#include <vector>
#include <cstdint>
#include <cmath>

namespace rfaa {

// ============================================================================
// Distogram Binning 常量 (RFAA Section 2.5.4)
// ============================================================================

// 距离 binning 参数
constexpr float DIST_DMIN    = 1.2f;   // 距离下限 (Å)
constexpr float DIST_DMID    = 4.0f;   // 中段分界 (Å)
constexpr float DIST_DMAX    = 20.0f;  // 距离上限 (Å)
constexpr float DIST_CLAMP   = 18.5f;  // Cβ-Cβ 距离裁剪上限
constexpr int   DIST_N_FINE  = 30;     // 细粒度 bin 数 (1.2-4Å)
constexpr int   DIST_N_COARSE = 30;    // 粗粒度 bin 数 (4-20Å)
constexpr int   DIST_N_BINS  = 60;     // 总 bin 数 (不含溢出)

// 角度 binning 参数
constexpr int   OMEGA_N_BINS = 36;     // Ω (omega) 二面角: [-π, π) → 36 bins
constexpr int   THETA_N_BINS = 36;     // Θ (theta) 二面角: [-π, π) → 36 bins
constexpr int   PHI_N_BINS   = 18;     // Φ (phi)   平面角: [0, π)  → 18 bins

// ============================================================================
// Pseudo-Cβ 坐标计算
// ============================================================================

/**
 * @brief 从 backbone 坐标 (N, CA, C) 构造 pseudo-Cβ 坐标
 * 
 * 使用 Rosetta 默认参数 (RFAA Section 2.5.4):
 *   x = b - a
 *   y = c - b
 *   z = x × y
 *   d = -0.57910144·z + 0.5689693·x - 0.5441217·y + b
 * 
 * 对于核酸和任意原子帧，虽然 d 无物理意义，
 * 但它是局部坐标系下的几何一致量，网络可以学会预测。
 * 
 * @param ax,ay,az  N 原子坐标
 * @param bx,by,bz  CA 原子坐标
 * @param cx,cy,cz  C 原子坐标
 * @param dx,dy,dz  (out) pseudo-Cβ 坐标
 */
inline void pseudo_cb(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float cx, float cy, float cz,
    float& dx, float& dy, float& dz)
{
    // x = B - A  (CA → N)
    float xx = bx - ax, xy = by - ay, xz = bz - az;
    // y = C - B  (C → CA)
    float yx = cx - bx, yy = cy - by, yz = cz - bz;
    // z = x × y
    float zx = xy * yz - xz * yy;
    float zy = xz * yx - xx * yz;
    float zz = xx * yy - xy * yx;

    // d = -0.57910144*z + 0.5689693*x - 0.5441217*y + b
    dx = -0.57910144f * zx + 0.5689693f * xx - 0.5441217f * yx + bx;
    dy = -0.57910144f * zy + 0.5689693f * xy - 0.5441217f * yy + by;
    dz = -0.57910144f * zz + 0.5689693f * xz - 0.5441217f * yz + bz;
}

// ============================================================================
// 几何量计算
// ============================================================================

/**
 * @brief 计算两个向量的点积
 */
inline float vec_dot(float ax, float ay, float az, float bx, float by, float bz) {
    return ax * bx + ay * by + az * bz;
}

/**
 * @brief 计算向量的模长
 */
inline float vec_norm(float x, float y, float z) {
    return std::sqrt(x * x + y * y + z * z);
}

/**
 * @brief 计算二面角 Dihedral(a, b, c, d)
 * 
 * 数学公式 (RFAA Section 2.5.4):
 *   v1 = b - a,  v2 = c - b,  v3 = d - c
 *   n1 = v1 × v2,  n2 = v2 × v3
 *   dihedral = atan2(|v2| * (n1 × n2) · (v2/|v2|), |v2| * n1 · n2)
 * 
 * @return 二面角 ∈ [-π, π)
 */
inline float dihedral_angle(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float cx, float cy, float cz,
    float dx, float dy, float dz)
{
    // v1 = B - A
    float v1x = bx - ax, v1y = by - ay, v1z = bz - az;
    // v2 = C - B
    float v2x = cx - bx, v2y = cy - by, v2z = cz - bz;
    // v3 = D - C
    float v3x = dx - cx, v3y = dy - cy, v3z = dz - cz;

    // n1 = v1 × v2
    float n1x = v1y * v2z - v1z * v2y;
    float n1y = v1z * v2x - v1x * v2z;
    float n1z = v1x * v2y - v1y * v2x;

    // n2 = v2 × v3
    float n2x = v2y * v3z - v2z * v3y;
    float n2y = v2z * v3x - v2x * v3z;
    float n2z = v2x * v3y - v2y * v3x;

    // n1 × n2
    float cx_n = n1y * n2z - n1z * n2y;
    float cy_n = n1z * n2x - n1x * n2z;
    float cz_n = n1x * n2y - n1y * n2x;

    float v2_norm = vec_norm(v2x, v2y, v2z);

    // 分子 = |v2| * (n1 × n2) · v2 / |v2| = (n1 × n2) · v2
    float numer = cx_n * v2x + cy_n * v2y + cz_n * v2z;

    // 分母 = |v2| * n1 · n2
    float denom = v2_norm * (n1x * n2x + n1y * n2y + n1z * n2z);

    return std::atan2(numer, denom);
}

/**
 * @brief 计算平面角 Planar(a, b, c) — 以 B 为顶点的 ∠ABC
 * 
 * 公式: arccos((A-B)·(C-B) / (|A-B|·|C-B|))
 * 
 * @return 平面角 ∈ [0, π]
 */
inline float planar_angle(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float cx, float cy, float cz)
{
    // v1 = A - B
    float v1x = ax - bx, v1y = ay - by, v1z = az - bz;
    // v2 = C - B
    float v2x = cx - bx, v2y = cy - by, v2z = cz - bz;

    float dot   = vec_dot(v1x, v1y, v1z, v2x, v2y, v2z);
    float norm1 = vec_norm(v1x, v1y, v1z);
    float norm2 = vec_norm(v2x, v2y, v2z);

    float cos_val = dot / (norm1 * norm2 + 1e-8f);
    // clamp to [-1, 1] for numerical safety
    cos_val = std::max(-1.0f, std::min(1.0f, cos_val));
    return std::acos(cos_val);
}

// ============================================================================
// Binning 函数 — 连续值 → one-hot bin index
// ============================================================================

/**
 * @brief 非均匀距离 binning (RFAA Section 2.5.4)
 * 
 * bin 0:    [0,       1.2)           ← 超短距离
 * bin 1-29: [1.2 + (4-1.2)*(i-1)/29,  1.2 + (4-1.2)*i/29)  细粒度 0.097Å/bin
 * bin 30-59:[4 + (20-4)*(i-30)/30,    4 + (20-4)*i/30)      粗粒度 0.533Å/bin
 * bin 60:   [20,       ∞)            ← 溢出
 * 
 * @param dist  Cβ-Cβ 距离 (Å), 已 clamp 到 18.5
 * @return bin index ∈ [0, 60]
 */
inline int distance_to_bin(float dist) {
    if (dist < DIST_DMIN)  return 0;
    if (dist < DIST_DMID) {
        // bin 1-29: fine-grained
        float t = (dist - DIST_DMIN) / (DIST_DMID - DIST_DMIN);
        return 1 + static_cast<int>(t * (DIST_N_FINE - 1));
    }
    if (dist < DIST_DMAX) {
        // bin 30-59: coarse-grained
        float t = (dist - DIST_DMID) / (DIST_DMAX - DIST_DMID);
        return 30 + static_cast<int>(t * DIST_N_COARSE);
    }
    return 60; // overflow
}

/**
 * @brief 二面角 binning (均匀 36 bins over [-π, π))
 * 
 * bin_i = [-π + 2π·i/36, -π + 2π·(i+1)/36)
 * 
 * @param angle  二面角 ∈ [-π, π)
 * @return bin index ∈ [0, 35]
 */
inline int dihedral_to_bin(float angle) {
    // normalize to [0, 2π)
    float norm = angle + static_cast<float>(M_PI);          // [0, 2π)
    int bin = static_cast<int>(norm * OMEGA_N_BINS / (2.0f * static_cast<float>(M_PI)));
    return std::max(0, std::min(OMEGA_N_BINS - 1, bin));
}

/**
 * @brief 平面角 binning (均匀 18 bins over [0, π))
 * 
 * bin_i = [π·i/18, π·(i+1)/18)
 * 
 * @param angle  平面角 ∈ [0, π]
 * @return bin index ∈ [0, 17]
 */
inline int planar_to_bin(float angle) {
    int bin = static_cast<int>(angle * PHI_N_BINS / static_cast<float>(M_PI));
    return std::max(0, std::min(PHI_N_BINS - 1, bin));
}

// ============================================================================
// 批量 Binning — 从坐标张量生成 one-hot label 张量
// ============================================================================

/**
 * @brief 从坐标计算距离 one-hot labels
 * 
 * 输入: coords (N, 3, 3) — N/CA/C 坐标
 * 输出: D_onehot (N, N, 60) — 每对残基的距离直方图 one-hot
 * 
 * @param coords   残基坐标 (N, 3, 3), 最后一维是 xyz
 * @param seq_mask 残基有效性 mask (N,)
 * @param D_onehot (out) 距离 one-hot labels (N*N*60 floats, 调用者分配)
 */
void compute_distance_onehot(
    const TensorF32& coords,
    const float* seq_mask,
    float* D_onehot);

/**
 * @brief 从坐标计算 Ω (omega) 二面角 one-hot labels
 * 
 * Ω[l][l'] = Dihedral(CA_l, CB_l, CA_l', CB_l')
 * 
 * @param coords   残基坐标 (N, 3, 3)
 * @param seq_mask 残基有效性 mask (N,)
 * @param Ω_onehot (out) Ω one-hot labels (N*N*36 floats)
 */
void compute_omega_onehot(
    const TensorF32& coords,
    const float* seq_mask,
    float* Ω_onehot);

/**
 * @brief 从坐标计算 Θ (theta) 二面角 one-hot labels
 * 
 * Θ[l][l'] = Dihedral(N_l, CA_l, CB_l, CB_l')
 * 
 * @param coords   残基坐标 (N, 3, 3)
 * @param seq_mask 残基有效性 mask (N,)
 * @param Θ_onehot (out) Θ one-hot labels (N*N*36 floats)
 */
void compute_theta_onehot(
    const TensorF32& coords,
    const float* seq_mask,
    float* Θ_onehot);

/**
 * @brief 从坐标计算 Φ (phi) 平面角 one-hot labels
 * 
 * Φ[l][l'] = Planar(CA_l, CB_l, CB_l')
 * 
 * @param coords   残基坐标 (N, 3, 3)
 * @param seq_mask 残基有效性 mask (N,)
 * @param Φ_onehot (out) Φ one-hot labels (N*N*18 floats)
 */
void compute_phi_onehot(
    const TensorF32& coords,
    const float* seq_mask,
    float* Φ_onehot);

/**
 * @brief 一次性计算全部 4 个 one-hot label 张量
 * 
 * 用于 training data preparation, 在模型前向之前一次性完成。
 * 
 * @param coords   残基坐标 (N, 3, 3) — N/CA/C 格式
 * @param seq_mask 残基有效性 mask (N,) 1=有效 0=忽略
 * @param D_onehot  (out) 距离 one-hot (N*N*60)
 * @param Ω_onehot  (out) Ω one-hot (N*N*36)
 * @param Θ_onehot  (out) Θ one-hot (N*N*36)
 * @param Φ_onehot  (out) Φ one-hot (N*N*18)
 */
void compute_all_distogram_onehots(
    const TensorF32& coords,
    const float* seq_mask,
    float* D_onehot,
    float* Ω_onehot,
    float* Θ_onehot,
    float* Φ_onehot);

} // namespace rfaa

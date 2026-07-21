#pragma once

#include "Tensor.h"
#include "AtomFrame.h"
#include <vector>
#include <cstdint>
#include <cmath>

namespace rfaa {

// ============================================================================
// 4×4 齐次刚体变换矩阵 (SE(3)) — 使用项目 Tensor 作为底层存储
// ============================================================================

/**
 * @brief 4×4 齐次刚体变换矩阵 (SE(3))
 * 
 * 按列主序存储: m[col][row] = data[col*4 + row]
 *   col 0 (R_{xx}, R_{yx}, R_{zx}, 0)    ← 旋转矩阵第一列 + 0
 *   col 1 (R_{xy}, R_{yy}, R_{zy}, 0)    ← 旋转矩阵第二列 + 0
 *   col 2 (R_{xz}, R_{yz}, R_{zz}, 0)    ← 旋转矩阵第三列 + 0
 *   col 3 (t_x,  t_y,  t_z,  1)         ← 平移向量 + 1
 * 
 * 等价于 TensorF32 形状 (4, 4)，内存布局为:
 *   data[0]=Rxx, data[4]=Rxy, data[8]=Rxz, data[12]=tx
 *   data[1]=Ryx, data[5]=Ryy, data[9]=Ryz, data[13]=ty
 *   data[2]=Rzx, data[6]=Rzy, data[10]=Rzz, data[14]=tz
 *   data[3]=0,   data[7]=0,   data[11]=0,   data[15]=1
 */
struct Rigid4x4 {
    float m[16];  // 扁平列主序: m[col*4 + row]

    Rigid4x4() { std::memset(m, 0, sizeof(m)); m[15] = 1.0f; }

    // 索引: (row, col) → m[col*4 + row]
    float  operator()(int row, int col) const { return m[col * 4 + row]; }
    float& operator()(int row, int col)       { return m[col * 4 + row]; }

    // 构造单位矩阵
    static Rigid4x4 identity() {
        Rigid4x4 T;
        std::memset(T.m, 0, sizeof(T.m));
        T(0, 0) = 1.0f; T(1, 1) = 1.0f; T(2, 2) = 1.0f; T(3, 3) = 1.0f;
        return T;
    }

    /**
     * @brief 从 3 个原子坐标通过 Gram-Schmidt 构建局部坐标系 (逆变换 T^{-1})
     * 
     * 输入: 3 个原子位置 (xyz float triplet)
     * 输出: 逆刚体变换 T^{-1} (将全局坐标变换到以 A 为原点的局部坐标系)
     * 
     * 数学:
     *   v1 = B - A,  v2 = C - A
     *   e1 = normalize(v1), e2 = normalize(v2 - (v2·e1)e1), e3 = e1 × e2
     *   T^{-1} = [R^T | -R^T*A ; 0 | 1]
     */
    static Rigid4x4 from_three_points(
        float ax, float ay, float az,   // A (如 N)
        float bx, float by, float bz,   // B (如 CA)
        float cx, float cy, float cz)   // C (如 C)
    {
        // v1 = B - A, v2 = C - A
        float v1x = bx - ax, v1y = by - ay, v1z = bz - az;
        float v2x = cx - ax, v2y = cy - ay, v2z = cz - az;

        // e1 = normalize(v1)
        float n1 = std::sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
        float e1x = v1x / n1, e1y = v1y / n1, e1z = v1z / n1;

        // e2 = normalize(v2 - (v2·e1)e1)
        float dot = v2x * e1x + v2y * e1y + v2z * e1z;
        float u2x = v2x - dot * e1x, u2y = v2y - dot * e1y, u2z = v2z - dot * e1z;
        float n2 = std::sqrt(u2x * u2x + u2y * u2y + u2z * u2z);
        float e2x = u2x / n2, e2y = u2y / n2, e2z = u2z / n2;

        // e3 = e1 × e2
        float e3x = e1y * e2z - e1z * e2y;
        float e3y = e1z * e2x - e1x * e2z;
        float e3z = e1x * e2y - e1y * e2x;

        // R^T 的第 i 列 = e_{i+1} (作为列向量)
        // 平移 = -R^T * A
        float tx = -(e1x * ax + e1y * ay + e1z * az);
        float ty = -(e2x * ax + e2y * ay + e2z * az);
        float tz = -(e3x * ax + e3y * ay + e3z * az);

        Rigid4x4 T;
        // col 0 = R^T column 0 = (e1x, e1y, e1z, 0)
        T(0, 0) = e1x; T(1, 0) = e1y; T(2, 0) = e1z; T(3, 0) = 0.0f;
        // col 1 = R^T column 1 = (e2x, e2y, e2z, 0)
        T(0, 1) = e2x; T(1, 1) = e2y; T(2, 1) = e2z; T(3, 1) = 0.0f;
        // col 2 = R^T column 2 = (e3x, e3y, e3z, 0)
        T(0, 2) = e3x; T(1, 2) = e3y; T(2, 2) = e3z; T(3, 2) = 0.0f;
        // col 3 = translation = (tx, ty, tz, 1)
        T(0, 3) = tx;  T(1, 3) = ty;  T(2, 3) = tz;  T(3, 3) = 1.0f;

        return T;
    }

    /**
     * @brief 将刚体变换作用在点上: p' = R*p + t
     */
    void transform_point(float px, float py, float pz,
                         float& ox, float& oy, float& oz) const
    {
        ox = m[0] * px + m[4] * py + m[8]  * pz + m[12];
        oy = m[1] * px + m[5] * py + m[9]  * pz + m[13];
        oz = m[2] * px + m[6] * py + m[10] * pz + m[14];
    }
};

// ============================================================================
// FAPE 损失函数参数
// ============================================================================
struct FAPEConfig {
    float length_scale = 1.0f;
    float d_clamp     = 10.0f;
    float epsilon     = 1e-4f;

    FAPEConfig() = default;
    FAPEConfig(float ls, float dc, float eps = 1e-4f)
        : length_scale(ls), d_clamp(dc), epsilon(eps) {}
};

// ============================================================================
// compute_fape — 核心 FAPE 损失计算 (Tensor 接口)
// ============================================================================

/**
 * @brief 计算 Frame Aligned Point Error (FAPE) 损失
 * 
 * 对应 RFAA compute_general_fape() 的 6 步流水线:
 * 
 * Step 1: gather_frame_atoms()
 *   从全局坐标张量中按帧偏移量采集每个 frame 的 3 个原子坐标
 * 
 * Step 2: compute_frame_transforms()
 *   Gram-Schmidt 正交化 → 4×4 逆刚体变换矩阵
 * 
 * Step 3: transform_all_atoms()
 *   用每帧的逆变换将所有原子变换到局部坐标系
 * 
 * Step 4: compute_pairwise_distances()
 *   计算 (N_frames, N_atoms) 距离矩阵
 * 
 * Step 5: apply_clamp_and_masks()
 *   clamp(e, 0, d_clamp) × frames_mask × positions_mask
 * 
 * Step 6: reduce_loss()
 *   L = Σ clamped / (Σ frames_mask · Σ positions_mask + ε) / length_scale
 * 
 * @param pred_coords     预测坐标 (N_atoms, 3)  TensorF32
 * @param true_coords     目标坐标 (N_atoms, 3)  TensorF32
 * @param frames          每原子帧偏移 (N_atoms × 3 offsets)
 * @param residue_ids     每原子残基 ID  (N_atoms,)
 * @param residue_starts  残基起始索引
 * @param frames_mask     帧有效性 mask (N_atoms,)
 * @param positions_mask  原子位置 mask (N_atoms,)
 * @param config          FAPE 配置
 * @return float          FAPE 损失标量
 */
float compute_fape(
    const TensorF32& pred_coords,
    const TensorF32& true_coords,
    const std::vector<std::vector<AtomFrameOffset>>& frames,
    const std::vector<int>& residue_ids,
    const std::vector<int>& residue_starts,
    const TensorF32& frames_mask,
    const TensorF32& positions_mask,
    const FAPEConfig& config = FAPEConfig{}
);

/**
 * @brief 简化版 compute_fape — 不区分残基 (所有原子视为同一线性序列)
 * 
 * 自动构造虚拟残基信息 (每原子一个残基)
 */
float compute_fape_simple(
    const TensorF32& pred_coords,
    const TensorF32& true_coords,
    const std::vector<std::vector<AtomFrameOffset>>& frames,
    const TensorF32& frames_mask,
    const TensorF32& positions_mask,
    const FAPEConfig& config = FAPEConfig{}
);

} // namespace rfaa

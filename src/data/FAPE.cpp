#include "ppml/FAPE.h"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <cmath>
#include <cstring>
#include <vector>

namespace ppml {

// ============================================================================
// 辅助宏: 坐标张量索引 (N_atoms, 3) — row-major: coords[i*3 + c]
// ============================================================================
#define COORD(data, i, c) ((data)[(i) * 3 + (c)])

// ============================================================================
// Step 1: gather_frame_atoms — 按帧偏移量采集 3 个原子坐标
// ============================================================================

/**
 * @brief 从 (N, 3) 坐标张量中按帧偏移量采集每帧的 3 个原子坐标
 * 
 * 输出: frame_coords[n*3+k] 存放帧 n 的第 k 个原子的 (x,y,z)
 *       frames_mask 中退化帧位置为 0
 * 
 * @param coords_data  全局坐标 raw pointer (N_atoms*3 floats)
 * @param N            原子总数
 * @param frames       每原子的帧偏移量 (N_atoms × 3 offsets)
 * @param residue_ids  每原子残基 ID
 * @param res_starts   残基起始索引列表 (残基 ID → 原子起始索引)
 * @param fmask        帧有效性 mask (in/out, 退化帧置 0)
 * @return             frame_coords (N*9 floats, 扁平存储: [n*9 + k*3 + xyz])
 */
static std::vector<float> gather_frame_atoms(
    const float* coords_data,
    int N,
    const std::vector<std::vector<AtomFrameOffset>>& frames,
    const std::vector<int>& residue_ids,
    const std::vector<int>& res_starts,
    float* fmask)
{
    std::vector<float> frame_coords(N * 9, 0.0f);  // N × 3 atoms × 3 xyz

    // 构建残基 ID → 起始索引的反向映射
    std::unordered_map<int, int> res_to_start;
    for (size_t i = 0; i < res_starts.size(); ++i) {
        res_to_start[res_starts[i]] = static_cast<int>(i);
    }

    for (int n = 0; n < N; ++n) {
        const auto& frame = frames[n];

        // 检查退化帧: 三个偏移量完全相同
        if (frame[0] == frame[1] && frame[1] == frame[2]) {
            fmask[n] = 0.0f;
            continue;
        }

        int res_n = residue_ids[n];
        auto it = res_to_start.find(res_n);
        if (it == res_to_start.end()) {
            fmask[n] = 0.0f;
            continue;
        }
        int res_start = it->second;

        bool frame_valid = true;
        for (int k = 0; k < 3; ++k) {
            int target_res = res_n + frame[k].residue_offset;
            auto tit = res_to_start.find(target_res);
            if (tit == res_to_start.end()) {
                frame_valid = false;
                break;
            }
            int target_res_start = tit->second;
            int global_idx = target_res_start + frame[k].atom_offset;

            if (global_idx < 0 || global_idx >= N || residue_ids[global_idx] != target_res) {
                frame_valid = false;
                break;
            }

            // 拷贝坐标: coords_data[global_idx*3 + 0..2]
            int dst_base = n * 9 + k * 3;
            int src_base = global_idx * 3;
            frame_coords[dst_base + 0] = coords_data[src_base + 0];
            frame_coords[dst_base + 1] = coords_data[src_base + 1];
            frame_coords[dst_base + 2] = coords_data[src_base + 2];
        }

        if (!frame_valid) {
            fmask[n] = 0.0f;
        }
    }

    return frame_coords;
}

// ============================================================================
// Step 2: compute_frame_transforms — Gram-Schmidt → 4×4 逆变换
// ============================================================================

/**
 * @brief 对每帧的 3 原子做 Gram-Schmidt → Rigid4x4 逆变换矩阵
 * 
 * frame_coords 扁平布局: [n*9 + k*3 + xyz]
 * 退化帧 (fmask=0) → 单位矩阵
 */
static std::vector<Rigid4x4> compute_frame_transforms(
    const std::vector<float>& frame_coords,
    const float* fmask,
    int N)
{
    std::vector<Rigid4x4> transforms(N);

    for (int n = 0; n < N; ++n) {
        if (fmask[n] < 0.5f) {
            transforms[n] = Rigid4x4::identity();
            continue;
        }

        int base = n * 9;
        transforms[n] = Rigid4x4::from_three_points(
            frame_coords[base + 0], frame_coords[base + 1], frame_coords[base + 2],  // A
            frame_coords[base + 3], frame_coords[base + 4], frame_coords[base + 5],  // B
            frame_coords[base + 6], frame_coords[base + 7], frame_coords[base + 8]   // C
        );
    }

    return transforms;
}

// ============================================================================
// Step 3: transform_all_atoms — 批量刚体变换
// ============================================================================

/**
 * @brief 用每帧的逆变换将所有原子变换到局部坐标系
 * 
 * 输出: local_pred (N_frames × N_atoms × 3) 扁平数组
 * 
 * 退化帧填充 0 (后续被 mask 消除)
 */
static std::vector<float> transform_all_atoms(
    const float* coords_data,
    int N_atoms,
    const std::vector<Rigid4x4>& transforms,
    const float* fmask,
    int N_frames)
{
    std::vector<float> local(N_frames * N_atoms * 3, 0.0f);

    for (int i = 0; i < N_frames; ++i) {
        if (fmask[i] < 0.5f) continue;

        const Rigid4x4& T = transforms[i];
        float* local_i = local.data() + i * N_atoms * 3;

        for (int j = 0; j < N_atoms; ++j) {
            int src_j = j * 3;
            int dst_j = j * 3;
            T.transform_point(
                coords_data[src_j + 0], coords_data[src_j + 1], coords_data[src_j + 2],
                local_i[dst_j + 0], local_i[dst_j + 1], local_i[dst_j + 2]
            );
        }
    }

    return local;
}

// ============================================================================
// Step 4: compute_pairwise_distances — 局部坐标距离矩阵
// ============================================================================

/**
 * @brief e[i][j] = sqrt(||local_pred[i][j] - local_true[i][j]||^2 + epsilon)
 * 
 * 输入: local_pred, local_true — 扁平 (N_frames × N_atoms × 3)
 * 输出: errors — 扁平 (N_frames × N_atoms)
 */
static std::vector<float> compute_pairwise_distances(
    const std::vector<float>& local_pred,
    const std::vector<float>& local_true,
    const float* fmask,
    int N_frames,
    int N_atoms,
    float epsilon)
{
    std::vector<float> errors(N_frames * N_atoms, 0.0f);

    for (int i = 0; i < N_frames; ++i) {
        if (fmask[i] < 0.5f) continue;

        int base_i = i * N_atoms * 3;
        int err_i  = i * N_atoms;

        for (int j = 0; j < N_atoms; ++j) {
            int off = base_i + j * 3;
            float dx = local_pred[off + 0] - local_true[off + 0];
            float dy = local_pred[off + 1] - local_true[off + 1];
            float dz = local_pred[off + 2] - local_true[off + 2];
            errors[err_i + j] = std::sqrt(dx * dx + dy * dy + dz * dz + epsilon);
        }
    }

    return errors;
}

// ============================================================================
// Step 5: apply_clamp_and_masks — clamp + 双重 mask
// ============================================================================

/**
 * @brief masked[i][j] = clamp(e[i][j], 0, d_clamp) × fmask[i] × pmask[j]
 * 
 * 输出: masked_errors 扁平 (N_frames × N_atoms)
 */
static std::vector<float> apply_clamp_and_masks(
    const std::vector<float>& errors,
    const float* fmask,
    const float* pmask,
    int N_frames,
    int N_atoms,
    float d_clamp)
{
    std::vector<float> masked(N_frames * N_atoms, 0.0f);

    for (int i = 0; i < N_frames; ++i) {
        float fi = fmask[i];
        if (fi < 0.5f) continue;

        int base = i * N_atoms;
        for (int j = 0; j < N_atoms; ++j) {
            float pj = pmask[j];
            if (pj < 0.5f) continue;

            float clamped = std::min(errors[base + j], d_clamp);
            masked[base + j] = clamped * fi * pj;
        }
    }

    return masked;
}

// ============================================================================
// Step 6: reduce_loss — 归约求和 + 归一化
// ============================================================================

/**
 * @brief L = Σ masked / (Σ fmask · Σ pmask + ε) / length_scale
 */
static float reduce_loss(
    const std::vector<float>& masked_errors,
    const float* fmask,
    const float* pmask,
    int N_frames,
    int N_atoms,
    float length_scale,
    float epsilon)
{
    // 分子: 所有有效 pair 误差和
    float sum_errors = 0.0f;
    for (int i = 0; i < N_frames; ++i) {
        int base = i * N_atoms;
        for (int j = 0; j < N_atoms; ++j) {
            sum_errors += masked_errors[base + j];
        }
    }

    // 分母: 有效 frame 数 × 有效 atom 数
    float sum_fm = 0.0f;
    for (int i = 0; i < N_frames; ++i) sum_fm += fmask[i];

    float sum_pm = 0.0f;
    for (int j = 0; j < N_atoms; ++j) sum_pm += pmask[j];

    float normalization = sum_fm * sum_pm + epsilon;
    return sum_errors / normalization / length_scale;
}

// ============================================================================
// compute_fape — 主入口 (6 步流水线, Tensor 接口)
// ============================================================================

float compute_fape(
    const TensorF32& pred_coords,
    const TensorF32& true_coords,
    const std::vector<std::vector<AtomFrameOffset>>& frames,
    const std::vector<int>& residue_ids,
    const std::vector<int>& residue_starts,
    const TensorF32& frames_mask,
    const TensorF32& positions_mask,
    const FAPEConfig& config)
{
    // ================================================================
    // 输入验证
    // ================================================================
    const auto& pred_shape = pred_coords.shape();
    const auto& true_shape = true_coords.shape();

    if (pred_shape.ndim() != 2 || pred_shape.dims[1] != 3) {
        throw std::invalid_argument("compute_fape: pred_coords must be (N, 3)");
    }
    if (!true_coords.same_shape(pred_coords)) {
        throw std::invalid_argument("compute_fape: pred_coords and true_coords shape mismatch");
    }

    const int N_atoms  = static_cast<int>(pred_shape.dims[0]);
    const int N_frames = N_atoms;  // 每个原子一个帧

    if (N_atoms == 0) return 0.0f;

    if (static_cast<int>(frames.size()) != N_atoms) {
        throw std::invalid_argument("compute_fape: frames.size() != N_atoms");
    }
    if (static_cast<int>(residue_ids.size()) != N_atoms) {
        throw std::invalid_argument("compute_fape: residue_ids.size() != N_atoms");
    }
    if (frames_mask.shape().numel() != N_atoms) {
        throw std::invalid_argument("compute_fape: frames_mask size != N_atoms");
    }
    if (positions_mask.shape().numel() != N_atoms) {
        throw std::invalid_argument("compute_fape: positions_mask size != N_atoms");
    }

    // ================================================================
    // 获取 raw pointers (假设 CPU)
    // ================================================================
    const float* pred_data = pred_coords.data();
    const float* true_data = true_coords.data();
    const float* fmask_in  = frames_mask.data();
    const float* pmask     = positions_mask.data();

    // 可修改的帧 mask 副本
    std::vector<float> fmask(N_atoms);
    std::memcpy(fmask.data(), fmask_in, N_atoms * sizeof(float));

    // ================================================================
    // Step 1: gather — 采集帧原子坐标
    // ================================================================
    auto pred_frame_coords = gather_frame_atoms(pred_data, N_atoms, frames,
                                                 residue_ids, residue_starts, fmask.data());
    auto true_frame_coords = gather_frame_atoms(true_data, N_atoms, frames,
                                                 residue_ids, residue_starts, fmask.data());

    // ================================================================
    // Step 2: Gram-Schmidt — 构建 4×4 逆变换矩阵
    // ================================================================
    auto T_inv_pred = compute_frame_transforms(pred_frame_coords, fmask.data(), N_frames);
    auto T_inv_true = compute_frame_transforms(true_frame_coords, fmask.data(), N_frames);

    // ================================================================
    // Step 3: Transform — 批量刚体变换到局部坐标系
    // ================================================================
    auto local_pred = transform_all_atoms(pred_data, N_atoms, T_inv_pred, fmask.data(), N_frames);
    auto local_true = transform_all_atoms(true_data, N_atoms, T_inv_true, fmask.data(), N_frames);

    // ================================================================
    // Step 4: Distance — 计算 (N_frames, N_atoms) 距离矩阵
    // ================================================================
    auto errors = compute_pairwise_distances(local_pred, local_true,
                                              fmask.data(), N_frames, N_atoms, config.epsilon);

    // ================================================================
    // Step 5: Clamp + Mask — 截断 + 双重 mask
    // ================================================================
    auto masked_errors = apply_clamp_and_masks(errors, fmask.data(), pmask,
                                                N_frames, N_atoms, config.d_clamp);

    // ================================================================
    // Step 6: Reduce — 加权平均得标量损失
    // ================================================================
    return reduce_loss(masked_errors, fmask.data(), pmask,
                       N_frames, N_atoms, config.length_scale, config.epsilon);
}

// ============================================================================
// compute_fape_simple — 不区分残基的简化版
// ============================================================================

float compute_fape_simple(
    const TensorF32& pred_coords,
    const TensorF32& true_coords,
    const std::vector<std::vector<AtomFrameOffset>>& frames,
    const TensorF32& frames_mask,
    const TensorF32& positions_mask,
    const FAPEConfig& config)
{
    const int N = static_cast<int>(pred_coords.shape().dims[0]);
    if (N == 0) return 0.0f;

    // 构造虚拟残基: 每原子一个独立残基
    std::vector<int> residue_ids(N);
    std::vector<int> residue_starts(N);
    for (int i = 0; i < N; ++i) {
        residue_ids[i]   = i;
        residue_starts[i] = i;
    }

    return compute_fape(pred_coords, true_coords, frames,
                        residue_ids, residue_starts,
                        frames_mask, positions_mask, config);
}

} // namespace ppml

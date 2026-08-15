#pragma once

#include "Tensor.h"
#include "ComputeGraph.h"
#include "Backend.h"

#include <vector>
#include <string>
#include <limits>

namespace ppml {

// ============================================================
// PerLossClipConfig — 逐项损失 + 全局的梯度裁剪配置
// ============================================================

/// @brief 各 loss 项的独立 max_norm 及全局裁剪配置
///
/// 各 per-loss max_norm 默认为 INFINITY 即不裁剪；
/// 仅 global_max_norm 默认开启（AF2 推荐 0.1）。
struct PerLossClipConfig {
    // ---- per-loss 独立裁剪阈值 ----
    // 设为 INFINITY 则跳过该项裁剪
    float fape_max_norm      = std::numeric_limits<float>::infinity();
    float chi_max_norm       = std::numeric_limits<float>::infinity();
    float distogram_max_norm = std::numeric_limits<float>::infinity();
    float msa_max_norm       = std::numeric_limits<float>::infinity();
    float conf_max_norm      = std::numeric_limits<float>::infinity();

    // ---- 累加后的全局裁剪阈值 ----
    // AF2 标准推荐 0.1
    float global_max_norm    = 0.1f;
};

// ============================================================
// LossGradientInfo — 单个 loss 项的梯度诊断信息
// ============================================================

/// @brief 逐项 loss 梯度信息，用于调参/监控
struct LossGradientInfo {
    const TensorF32* loss_node   = nullptr;  // loss 节点指针
    std::string      loss_name;              // 可选的描述名（如 "FAPE", "Chi" 等）
    float            raw_norm     = 0.0f;    // 裁剪前的 per-loss 梯度 L2 范数
    float            clipped_norm = 0.0f;    // 裁剪后的 per-loss 梯度 L2 范数
    bool             was_clipped  = false;   // 是否触发了 per-loss 裁剪
};

// ============================================================
// 核心 API
// ============================================================

/// @brief 全局梯度裁剪（单次 backward 完成后调用）
///
/// 遍历 cgraph 中所有标记为 TENSOR_FLAG_PARAM 的 leaf 节点，
/// 收集其梯度，计算全局 L2 范数。若超过 max_norm，则等比例缩放所有参数梯度。
///
/// 这是一个后处理函数，应在 backward compute 完成后、optimizer.step() 前调用。
///
/// @param cgraph   计算图（已完成 backward compute）
/// @param max_norm 梯度范数上界（推荐值: 0.1 for AF2-style）
/// @return         截断前的全局梯度范数（用于日志/monitoring）
float clip_grad_norm(ComputeGraph* cgraph, float max_norm);

/// @brief 逐项 backward + per-loss 裁剪 + 累加 + 全局裁剪
///
/// 对每个 (loss_node, per_loss_max_norm) 对，依次：
///   1. 激活当前 loss（设置 upstream grad = 1.0），抑制其他 loss（grad = 0.0）
///   2. 执行 graph_compute 完成 backward
///   3. 计算所有参数梯度的 L2 范数，若超过 per_loss_max_norm 则等比例缩放
///   4. 将裁剪后的梯度累加到持久化 accumulator
///
/// 全部 loss 项处理完成后，对 accumulator 执行 global_max_norm 裁剪。
///
/// 注意：
///   - cgraph 必须已调用 build_backward_expand()
///   - 各 loss_node 必须已标记 TENSOR_FLAG_LOSS
///   - 此函数会多次调用 graph_compute，调用方应确保数据已就绪
///
/// @param cgraph      计算图（已 build_backward_expand）
/// @param backend     后端（CPU/CUDA），用于执行 graph_compute
/// @param loss_nodes  所有 loss 节点（按顺序对应 clip_norms）
/// @param clip_norms  与 loss_nodes 一一对应的 per-loss max_norm
/// @param config      裁剪配置（含 global_max_norm）
/// @param out_info    [out, optional] 各 loss 的诊断信息
/// @return            全局裁剪后的总梯度范数
float accumulate_per_loss_gradients(
    ComputeGraph*                     cgraph,
    Backend*                          backend,
    const std::vector<TensorF32*>&    loss_nodes,
    const std::vector<float>&         clip_norms,
    const PerLossClipConfig&          config,
    std::vector<LossGradientInfo>*    out_info = nullptr);

/// @brief 便捷包装：从 PerLossClipConfig + loss 节点 自动组装参数
///
/// 接受按语义命名的 5 个 loss 节点（nullptr 可跳过），
/// 自动构建 loss_nodes 和 clip_norms 向量，调用 accumulate_per_loss_gradients。
///
/// @param cgraph         计算图
/// @param backend        后端
/// @param loss_fape      FAPE loss 节点 (可为 nullptr)
/// @param loss_chi       Chi loss 节点 (可为 nullptr)
/// @param loss_distogram Distogram loss 节点 (可为 nullptr)
/// @param loss_msa       MSA loss 节点 (可为 nullptr)
/// @param loss_conf      Confidence loss 节点 (可为 nullptr)
/// @param config         裁剪配置
/// @param out_info       [out, optional] 诊断信息
/// @return               全局裁剪后的总梯度范数
float apply_per_loss_clip(
    ComputeGraph*           cgraph,
    Backend*                backend,
    TensorF32*              loss_fape,
    TensorF32*              loss_chi,
    TensorF32*              loss_distogram,
    TensorF32*              loss_msa,
    TensorF32*              loss_conf,
    const PerLossClipConfig& config,
    std::vector<LossGradientInfo>* out_info = nullptr);

} // namespace ppml

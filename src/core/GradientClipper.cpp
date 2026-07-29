#include "rfaa/GradientClipper.h"
#include "rfaa/Tensor.h"

#include <cmath>
#include <cstring>
#include <cassert>

namespace rfaa {

// ============================================================
// 内部辅助函数
// ============================================================

/// @brief 激活单个 loss 节点：将其 upstream grad 设为 1.0，其他 loss 设为 0.0
static void activate_single_loss(
    ComputeGraph* cgraph,
    TensorF32* active_loss,
    const std::vector<TensorF32*>& all_losses)
{
    for (auto* loss_node : all_losses) {
        if (!loss_node) continue;
        TensorF32* grad = cgraph->graph_get_grad(loss_node);
        if (!grad) continue;
        float val = (loss_node == active_loss) ? 1.0f : 0.0f;
        int64_t n = grad->numel();
        // 对 GPU tensor 也需写入；此处假设 graph_compute 已在正确设备上下文
        float* gdata = grad->data();
        for (int64_t j = 0; j < n; j++) {
            gdata[j] = val;
        }
    }
}

/// @brief 清空所有参数的梯度缓冲区（grad_accs）
static void clear_param_grads(ComputeGraph* cgraph) {
    for (int i = 0; i < cgraph->n_leafs(); i++) {
        TensorF32* param = cgraph->leafs()[i];
        if (!(param->flag & TENSOR_FLAG_PARAM)) continue;
        TensorF32* grad = cgraph->graph_get_grad(param);
        if (grad) {
            int64_t n = grad->numel();
            std::fill(grad->data(), grad->data() + n, 0.0f);
        }
    }
}

/// @brief 计算所有参数梯度的 L2 范数平方
/// @param buffers 参数缓冲区列表
/// @return 总 L2 范数
static float compute_total_grad_norm(const std::vector<TensorF32*>& param_list, ComputeGraph* cgraph) {
    double norm_sq = 0.0;  // double 防溢出
    for (auto* param : param_list) {
        TensorF32* grad = cgraph->graph_get_grad(param);
        if (!grad) continue;
        float* g = grad->data();
        int64_t n = grad->numel();
        for (int64_t j = 0; j < n; j++) {
            norm_sq += static_cast<double>(g[j]) * static_cast<double>(g[j]);
        }
    }
    return static_cast<float>(std::sqrt(norm_sq));
}

/// @brief 等比例缩放所有参数梯度
static void scale_param_grads(const std::vector<TensorF32*>& param_list, ComputeGraph* cgraph, float scale) {
    for (auto* param : param_list) {
        TensorF32* grad = cgraph->graph_get_grad(param);
        if (!grad) continue;
        float* g = grad->data();
        int64_t n = grad->numel();
        for (int64_t j = 0; j < n; j++) {
            g[j] *= scale;
        }
    }
}

/// @brief 收集所有带梯度的参数节点
static std::vector<TensorF32*> collect_params(ComputeGraph* cgraph) {
    std::vector<TensorF32*> params;
    for (int i = 0; i < cgraph->n_leafs(); i++) {
        TensorF32* leaf = cgraph->leafs()[i];
        if (leaf->flag & TENSOR_FLAG_PARAM) {
            TensorF32* grad = cgraph->graph_get_grad(leaf);
            if (grad) {
                params.push_back(leaf);
            }
        }
    }
    return params;
}

// ============================================================
// 公共 API
// ============================================================

float clip_grad_norm(ComputeGraph* cgraph, float max_norm) {
    if (max_norm <= 0.0f) return 0.0f;

    auto params = collect_params(cgraph);
    if (params.empty()) return 0.0f;

    float total_norm = compute_total_grad_norm(params, cgraph);

    if (total_norm > max_norm && total_norm > 1e-12f) {
        float scale = max_norm / total_norm;
        scale_param_grads(params, cgraph, scale);
    }

    return total_norm;
}

float accumulate_per_loss_gradients(
    ComputeGraph*                     cgraph,
    Backend*                          backend,
    const std::vector<TensorF32*>&    loss_nodes,
    const std::vector<float>&         clip_norms,
    const PerLossClipConfig&          config,
    std::vector<LossGradientInfo>*    out_info)
{
    assert(loss_nodes.size() == clip_norms.size());
    assert(cgraph && backend);

    // ---- Step 0: 收集所有可训练参数及其梯度缓冲区 ----
    auto param_list = collect_params(cgraph);
    if (param_list.empty()) return 0.0f;

    // 为每个参数分配一个 CPU 端持久化累加器
    struct ParamAccum {
        TensorF32* param;
        std::vector<float> accum;  // CPU-side persistent accumulator
    };
    std::vector<ParamAccum> accumulators;
    accumulators.reserve(param_list.size());
    for (auto* p : param_list) {
        TensorF32* grad = cgraph->graph_get_grad(p);
        int64_t n = grad ? grad->numel() : 0;
        accumulators.push_back({p, std::vector<float>(static_cast<size_t>(n), 0.0f)});
    }

    // ---- Step 1: 逐 loss 执行 backward + per-loss clip + accumulate ----
    std::vector<LossGradientInfo> info_vec;

    for (size_t k = 0; k < loss_nodes.size(); k++) {
        TensorF32* loss_node = loss_nodes[k];
        if (!loss_node) continue;

        float per_loss_max = clip_norms[k];

        // 1a. 激活当前 loss，抑制其他 loss
        activate_single_loss(cgraph, loss_node, loss_nodes);

        // 1b. 清空参数梯度（上一轮残留）
        clear_param_grads(cgraph);

        // 1c. 执行 backward compute
        backend->graph_compute(cgraph);

        // 1d. 计算 per-loss norm
        float loss_norm = compute_total_grad_norm(param_list, cgraph);
        bool was_clipped = false;

        // 1e. per-loss clip
        if (loss_norm > per_loss_max && per_loss_max > 0.0f && loss_norm > 1e-12f) {
            float s = per_loss_max / loss_norm;
            scale_param_grads(param_list, cgraph, s);
            was_clipped = true;
        }

        // 1f. 累加到 persistent accumulator（CPU 端）
        for (size_t a = 0; a < accumulators.size(); a++) {
            TensorF32* grad = cgraph->graph_get_grad(accumulators[a].param);
            if (!grad) continue;
            float* gdata = grad->data();
            int64_t n = grad->numel();
            for (int64_t j = 0; j < n; j++) {
                accumulators[a].accum[static_cast<size_t>(j)] += gdata[j];
            }
        }

        // 1g. 记录诊断信息
        LossGradientInfo info;
        info.loss_node    = loss_node;
        info.raw_norm     = loss_norm;
        info.clipped_norm = was_clipped ? per_loss_max : loss_norm;
        info.was_clipped  = was_clipped;
        info_vec.push_back(info);
    }

    // ---- Step 2: 将累加结果写回 cgraph 的参数梯度 ----
    for (size_t a = 0; a < accumulators.size(); a++) {
        TensorF32* grad = cgraph->graph_get_grad(accumulators[a].param);
        if (!grad) continue;
        int64_t n = grad->numel();
        std::memcpy(grad->data(), accumulators[a].accum.data(),
                    static_cast<size_t>(n) * sizeof(float));
    }

    // ---- Step 3: 全局裁剪 ----
    float total_norm = clip_grad_norm(cgraph, config.global_max_norm);

    // ---- Step 4: 输出诊断 ----
    if (out_info) {
        *out_info = std::move(info_vec);
    }

    return total_norm;
}

float apply_per_loss_clip(
    ComputeGraph*           cgraph,
    Backend*                backend,
    TensorF32*              loss_fape,
    TensorF32*              loss_chi,
    TensorF32*              loss_distogram,
    TensorF32*              loss_msa,
    TensorF32*              loss_conf,
    const PerLossClipConfig& config,
    std::vector<LossGradientInfo>* out_info)
{
    // 构建 loss_nodes 和 clip_norms 向量
    // 顺序: FAPE, Chi, Distogram, MSA, Conf
    struct LossEntry {
        TensorF32* node;
        float      max_norm;
        const char* name;
    };

    LossEntry entries[] = {
        {loss_fape,      config.fape_max_norm,      "FAPE"},
        {loss_chi,       config.chi_max_norm,       "Chi"},
        {loss_distogram, config.distogram_max_norm, "Distogram"},
        {loss_msa,       config.msa_max_norm,       "MSA"},
        {loss_conf,      config.conf_max_norm,      "Conf"},
    };

    std::vector<TensorF32*> loss_nodes;
    std::vector<float>      clip_norms;
    loss_nodes.reserve(5);
    clip_norms.reserve(5);

    for (const auto& entry : entries) {
        if (entry.node) {
            loss_nodes.push_back(entry.node);
            clip_norms.push_back(entry.max_norm);
        }
    }

    std::vector<LossGradientInfo> info;
    float result = accumulate_per_loss_gradients(
        cgraph, backend, loss_nodes, clip_norms, config, &info);

    // 注入 loss name
    for (auto& inf : info) {
        for (const auto& entry : entries) {
            if (inf.loss_node == entry.node) {
                inf.loss_name = entry.name;
                break;
            }
        }
    }

    if (out_info) {
        *out_info = std::move(info);
    }

    return result;
}

} // namespace rfaa

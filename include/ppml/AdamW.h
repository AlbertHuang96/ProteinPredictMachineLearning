#pragma once

#include "Tensor.h"
#include "ComputeGraph.h"

#include <vector>
#include <map>
#include <cstdint>

namespace ppml {

// ============================================================
// AdamW — 解耦权重衰减（decoupled weight decay）优化器
// ============================================================
//
// 采用 PyTorch 标准的 decoupled AdamW 语义：
//   g = grad(W)
//   m = beta1*m + (1-beta1)*g            （一阶矩）
//   v = beta2*v + (1-beta2)*g^2          （二阶矩）
//   m_hat = m / (1 - beta1^t)
//   v_hat = v / (1 - beta2^t)
//   W -= lr * m_hat / (sqrt(v_hat) + eps) + lr * weight_decay * W   ← 权重衰减不进 m/v
//
// 注意：区别于 coupled（L2 正则进入梯度 m/v），decoupled 把 weight_decay
//       作为独立减项，且仅对带 TENSOR_FLAG_NO_WEIGHT_DECAY 之外的参数生效。
//       bias 与 LayerNorm 的 gamma/beta 被标记为豁免（见 TENSOR_FLAG_NO_WEIGHT_DECAY）。
class AdamW {
public:
    AdamW(float lr, float weight_decay, float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, bool bias_correction = true);

    // 收集 cgraph 中所有带梯度的 PARAM 参数，初始化一阶/二阶矩缓冲。
    // 同时记录每个参数的 no_weight_decay 标志。模型结构不变时只需调用一次。
    void init_from_graph(ComputeGraph* cgraph);

    // 单步更新：须在 build_backward_expand + graph_compute(+clip_grad_norm) 之后调用。
    void step(ComputeGraph* cgraph);

    // 清空所有参数的梯度缓冲区（可选，单次 backward 全链回传时通常无需手动清零）。
    void zero_grad(ComputeGraph* cgraph);

    // ---- 断点续训：序列化 m/v 动量 (与 GGUF extra 张量 "opt.m.<name>"/"opt.v.<name>" 对应) ----
    // 导出：按 (params, param_names) 同序输出；未参与本图 (无状态) 的参数给空向量。
    void export_momentum(const std::vector<TensorF32*>& params,
                         const std::vector<std::string>& param_names,
                         std::vector<std::vector<float>>& ms,
                         std::vector<std::vector<float>>& vs) const;
    // 恢复：从名字映射读回 m/v。须在 init_from_graph 之后调用（states_ 已按 param 建好）。
    // 尺寸不匹配或缺失 → 保持原状（0 初始化），不会破坏训练。
    void import_momentum(const std::vector<TensorF32*>& params,
                         const std::vector<std::string>& param_names,
                         const std::map<std::string, std::vector<float>>& raw_tensors);

    float lr() const            { return lr_; }
    void  set_lr(float lr)      { lr_ = lr; }
    size_t param_count() const  { return states_.size(); }
    int   step_count() const    { return step_count_; }
    void  set_step_count(int c) { step_count_ = c; }

private:
    struct ParamState {
        TensorF32*         param;
        bool               no_weight_decay;
        float              se3_lr_scale = 1.0f;  // SE3 参数分层 lr 缩放（TENSOR_FLAG_SE3 时 0.1）
        std::vector<float> m;   // 一阶矩
        std::vector<float> v;   // 二阶矩
    };
    std::vector<ParamState> states_;

    float lr_;
    float weight_decay_;
    float beta1_;
    float beta2_;
    float eps_;
    bool  bias_correction_;
    int   step_count_ = 0;
};

} // namespace ppml

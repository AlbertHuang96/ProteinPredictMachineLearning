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
          float eps = 1e-8f, bool bias_correction = true,
          float lora_lr_scale = 10.0f);

    // 收集 cgraph 中所有带梯度的 PARAM 参数，初始化一阶/二阶矩缓冲。
    // 同时记录每个参数的 no_weight_decay 标志。模型结构不变时只需调用一次。
    void init_from_graph(ComputeGraph* cgraph);

    // LoRA 参数 lr 放大倍率（TENSOR_FLAG_LORA 标记，默认 10.0）
    float lora_lr_scale() const { return lora_lr_scale_; }
    void  set_lora_lr_scale(float s) { lora_lr_scale_ = s; }

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

    // ---- 数据并行参数分片（阶段 A，ZeRO-1 式）----
    //   只为本 rank 拥有的参数建 m/v 状态 ⇒ 优化器状态内存按 world 分摊。
    //   owner 规则（**两端必须完全一致**）：按参数在图中出现顺序的全局序号 idx，owner = idx % world。
    //   使用要求：必须真的连上对端（world>1）才启用；否则会漏更新一半参数（train.cpp 里做门闩）。
    void set_shard(int my_rank, int world) { my_rank_ = my_rank; world_ = world; }
    bool shard_enabled() const { return world_ > 1; }
    int  my_rank() const       { return my_rank_; }
    int  world() const         { return world_; }
    int  n_owned() const       { return n_owned_; }

    float lr() const            { return lr_; }
    void  set_lr(float lr)      { lr_ = lr; }
    size_t param_count() const  { return states_.size(); }
    int   step_count() const    { return step_count_; }
    void  set_step_count(int c) { step_count_ = c; }

private:
    struct ParamState {
        TensorF32*         param;
        int                owner_rank     = 0;    // 该参数归属的 rank（分片时用；未分片恒 0）
        bool               no_weight_decay;
        float              se3_lr_scale  = 1.0f;  // SE3 参数分层 lr 缩放（TENSOR_FLAG_SE3 时 0.01）
        float              lora_lr_scale = 1.0f;  // LoRA 参数分层 lr 缩放（TENSOR_FLAG_LORA 时放大）
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
    float lora_lr_scale_ = 10.0f;   // LoRA 参数 lr 放大倍率（默认 10.0）
    int   step_count_ = 0;
    int   my_rank_    = 0;          // 参数分片：本 rank
    int   world_      = 1;          // 参数分片：world（<=1 ⇒ 不分片）
    int   n_owned_    = 0;          // 本 rank 拥有的参数个数（init_from_graph 统计）
};

} // namespace ppml

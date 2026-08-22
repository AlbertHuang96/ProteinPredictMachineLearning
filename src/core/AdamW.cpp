#include "ppml/AdamW.h"
#include "ppml/Backend.h"

#include <cmath>
#include <algorithm>
#include <cstring>

namespace ppml {

AdamW::AdamW(float lr, float weight_decay, float beta1, float beta2,
             float eps, bool bias_correction)
    : lr_(lr),
      weight_decay_(weight_decay),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      bias_correction_(bias_correction)
{}

void AdamW::init_from_graph(ComputeGraph* cgraph) {
    states_.clear();
    // 参数 (TENSOR_FLAG_PARAM) 在 build_forward_impl 中被归类为 node 而非 leaf
    // (ComputeGraph.cpp:137: OP_NONE && !PARAM 才作 leaf)，故遍历 n_nodes()。
    for (int i = 0; i < cgraph->n_nodes(); i++) {
        TensorF32* node = cgraph->graph_node(i);
        if (!(node->flag & TENSOR_FLAG_PARAM)) continue;          // 非参数，跳过
        TensorF32* grad = cgraph->graph_get_grad(node);
        if (!grad) continue;                                      // 无梯度（未参与 loss），跳过
        const int64_t n = grad->numel();
        ParamState s;
        s.param            = node;
        s.no_weight_decay  = (node->flag & TENSOR_FLAG_NO_WEIGHT_DECAY) != 0;
        // SE3 等变参数：梯度尺度与主图不匹配（offset→坐标→FAPE 链放大），用分层小 lr。
        s.se3_lr_scale     = (node->flag & TENSOR_FLAG_SE3) ? 0.1f : 1.0f;
        s.m.assign(static_cast<size_t>(n), 0.0f);
        s.v.assign(static_cast<size_t>(n), 0.0f);
        states_.push_back(std::move(s));
    }
}

// ---- 内部工具：把张量数据读到 CPU 向量（兼容 backend buffer / 裸 CPU / CUDA 裸张量）----
namespace {
std::vector<float> read_tensor_values(const TensorF32* t) {
    std::vector<float> buf(static_cast<size_t>(t->numel()));
    const size_t bytes = static_cast<size_t>(t->numel()) * sizeof(float);
    if (t->buffer_) {
        // 从 backend buffer 读取（CPU buffer 直接 memcpy；CUDA buffer 内部自动 H2D）
        t->buffer_->get_tensor(t, buf.data(), t->buffer_offs_, bytes);
    } else if (t->device() == Device::CPU) {
        std::memcpy(buf.data(), t->data(), bytes);
    } else {
        // CUDA 且无 buffer：拷贝一份 CPU 张量读取
        TensorF32 tcpu = t->cpu();
        std::memcpy(buf.data(), tcpu.data(), bytes);
    }
    return buf;
}

void write_tensor_values(TensorF32* t, const std::vector<float>& vals) {
    const size_t bytes = static_cast<size_t>(t->numel()) * sizeof(float);
    if (t->buffer_) {
        t->buffer_->set_tensor(t, vals.data(), t->buffer_offs_, bytes);
    } else {
        std::memcpy(t->data(), vals.data(), bytes);
    }
}
} // namespace

void AdamW::step(ComputeGraph* cgraph) {
    if (states_.empty()) return;

    step_count_++;
    const int   t = step_count_;
    const float b1 = beta1_, b2 = beta2_;
    const float lr = lr_, eps = eps_, wd = weight_decay_;
    const bool  bc = bias_correction_;

    // 偏差校正分母（避免逐参数重复求幂）
    const float bc1 = bc ? (1.0f - std::pow(b1, static_cast<float>(t))) : 1.0f;
    const float bc2 = bc ? (1.0f - std::pow(b2, static_cast<float>(t))) : 1.0f;

    for (auto& s : states_) {
        TensorF32* grad = cgraph->graph_get_grad(s.param);
        if (!grad) continue;

        // 参数与梯度尺寸应一致；否则跳过（防御）
        const int64_t n = grad->numel();
        if (n != s.param->numel()) continue;

        // 读取参数与梯度的 CPU 副本
        std::vector<float> w = read_tensor_values(s.param);
        std::vector<float> g = read_tensor_values(grad);

        float* m = s.m.data();
        float* v = s.v.data();
        const bool nod = s.no_weight_decay;
        const float s3 = s.se3_lr_scale;   // SE3 参数分层 lr 缩放（默认 1.0，SE3 0.1）

        for (int64_t i = 0; i < n; ++i) {
            const float gi = g[static_cast<size_t>(i)];

            // m = beta1*m + (1-beta1)*g
            m[i] = b1 * m[i] + (1.0f - b1) * gi;
            // v = beta2*v + (1-beta2)*g^2
            v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;

            // decoupled AdamW 更新：
            //   p -= lr * m_hat / (sqrt(v_hat) + eps) + lr * wd * p
            //   weight_decay 不进 m/v，且 no_weight_decay 参数跳过 wd 项
            const float m_hat = m[i] / bc1;
            const float v_hat = v[i] / bc2;
            const float update = lr * s3 * m_hat / (std::sqrt(v_hat) + eps);
            w[static_cast<size_t>(i)] -= update;
            if (wd != 0.0f && !nod) {
                w[static_cast<size_t>(i)] -= lr * s3 * wd * w[static_cast<size_t>(i)];
            }
        }

        // 写回参数
        write_tensor_values(s.param, w);
    }
}

void AdamW::zero_grad(ComputeGraph* cgraph) {
    for (int i = 0; i < cgraph->n_nodes(); i++) {
        TensorF32* node = cgraph->graph_node(i);
        if (!(node->flag & TENSOR_FLAG_PARAM)) continue;
        TensorF32* grad = cgraph->graph_get_grad(node);
        if (!grad) continue;
        std::vector<float> z(static_cast<size_t>(grad->numel()), 0.0f);
        write_tensor_values(grad, z);
    }
}

} // namespace ppml

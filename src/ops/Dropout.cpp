#include "ppml/Dropout.h"
#include "ppml/ComputeGraph.h"
#include <algorithm>

namespace ppml {

Dropout::Dropout(int broadcast_dim, float p_drop)
    : broadcast_dim_(broadcast_dim)
    , p_drop_(p_drop)
    , training_(true)
    , rng_(std::random_device{}())
    , dist_(1.0 - p_drop)  // Bernoulli with success probability (1 - p_drop)
{
    if (p_drop_ < 0.0f || p_drop_ >= 1.0f) {
        throw PPMLError("p_drop must be in [0, 1)");
    }
}

void Dropout::set_training(bool training) {
    training_ = training;
}

bool Dropout::is_training() const {
    return training_;
}

void Dropout::generate_mask(const Shape& shape, std::vector<float>& out_mask) {
    int64_t total_elements = shape.numel();
    out_mask.assign(static_cast<size_t>(total_elements), 0.0f);
    float scale = 1.0f / (1.0f - p_drop_);  // 缩放因子烘焙进掩码

    if (broadcast_dim_ >= 0 && broadcast_dim_ < shape.ndim()) {
        // Broadcast mode: generate mask where broadcast_dim dimension shares the same value
        int64_t outer_dims = 1;
        int64_t broadcast_size = shape.dims[broadcast_dim_];
        int64_t inner_dims = 1;
        for (int i = 0; i < broadcast_dim_; ++i) outer_dims *= shape.dims[i];
        for (int i = broadcast_dim_ + 1; i < shape.ndim(); ++i) inner_dims *= shape.dims[i];

        for (int64_t outer = 0; outer < outer_dims; ++outer) {
            for (int64_t inner = 0; inner < inner_dims; ++inner) {
                float mask_val = dist_(rng_) ? scale : 0.0f;  // Shared value
                for (int64_t b = 0; b < broadcast_size; ++b) {
                    int64_t idx = outer * broadcast_size * inner_dims + b * inner_dims + inner;
                    out_mask[idx] = mask_val;
                }
            }
        }
    } else {
        // No broadcast: generate independent random values for each element
        for (int64_t i = 0; i < total_elements; ++i) {
            out_mask[i] = dist_(rng_) ? scale : 0.0f;
        }
    }
}

TensorF32 Dropout::forward(const TensorF32& x) {
    // If not in training mode, return input directly (no dropout during evaluation)
    if (!training_) {
        TensorF32 output(x.shape(), x.device());
        output.copy_from(x);
        return output;
    }

    // Generate mask (already scaled by 1/(1-p))
    std::vector<float> mask;
    generate_mask(x.shape(), mask);

    // Apply mask: output = mask * x (mask already includes scale)
    TensorF32 output(x.shape(), x.device());
    float* out_data = output.data();
    const float* x_data = x.data();
    int64_t total_elements = x.numel();
    for (int64_t i = 0; i < total_elements; ++i) {
        out_data[i] = mask[static_cast<size_t>(i)] * x_data[i];
    }
    return output;
}

// ===== Dropout::forward_graph (图模式) =====
// 用现有图 op 实现: 生成随机掩码常量叶子 (语义同 forward, 已含缩放), 再 mul(x, mask)。
// mask 是叶子 (不参与求导) → mul 反向只把 grad_out*mask 回传给 x, 与值版 dropout 语义一致。
TensorF32* Dropout::forward_graph(TensorF32* x) {
    // If not in training mode, return input directly (no dropout during evaluation)
    if (!training_) {
        return x;
    }
    std::vector<float> mask;
    generate_mask(x->shape(), mask);
    std::vector<int64_t> dims(x->shape().dims.begin(), x->shape().dims.end());
    TensorF32* mask_node = constant_tensor(dims, mask.data());
    return mul(x, mask_node);
}

} // namespace ppml

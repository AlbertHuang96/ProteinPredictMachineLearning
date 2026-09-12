#include "ppml/Dropout.h"
#include "ppml/ComputeGraph.h"
#include <algorithm>
#include <map>
#include <string>

namespace ppml {

Dropout::Dropout(int broadcast_dim, float p_drop)
    : broadcast_dim_(broadcast_dim)
    , p_drop_(p_drop)
    , training_(true)
    , rng_(ppml_rng_seed(100u + static_cast<uint32_t>(p_drop * 100.0f) * 10u + static_cast<uint32_t>(broadcast_dim)))
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
// 每次 forward 现场生成随机掩码常量叶子（与 x 同形, 已含缩放 1/(1-p)），再 mul(x, mask)。
// 掩码用 constant_tensor_dynamic（TENSOR_FLAG_CONST 不置位）：数据存于 const_data_，
// 由 Gallocr 分配 backend buffer 后填充并立即清空 const_data_，宿主内存不累积。
// 每次调用新建掩码 leaf → 每次前向都随机化（与值版语义一致）。
TensorF32* Dropout::forward_graph(TensorF32* x) {
    // If not in training mode, return input directly (no dropout during evaluation)
    if (!training_) {
        return x;
    }

    // 生成掩码（含缩放 1/(1-p)）
    std::vector<float> mask;
    generate_mask(x->shape(), mask);

    // 动态一次性常量叶子：dims 取 x 的图布局 dims（dims[0]=最内维）
    std::vector<int64_t> dims;
    for (int i = 0; i < x->shape().ndim(); i++) dims.push_back(x->shape().dims[i]);
    TensorF32* mask_node = constant_tensor_dynamic(dims, mask.data());

    return mul(x, mask_node);
}

} // namespace ppml

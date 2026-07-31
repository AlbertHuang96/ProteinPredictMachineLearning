#include "rfaa/Dropout.h"
#include <algorithm>

namespace rfaa {

Dropout::Dropout(int broadcast_dim, float p_drop)
    : broadcast_dim_(broadcast_dim)
    , p_drop_(p_drop)
    , training_(true)
    , rng_(std::random_device{}())
    , dist_(1.0 - p_drop)  // Bernoulli with success probability (1 - p_drop)
{
    if (p_drop_ < 0.0f || p_drop_ >= 1.0f) {
        throw RFAAError("p_drop must be in [0, 1)");
    }
}

void Dropout::set_training(bool training) {
    training_ = training;
}

bool Dropout::is_training() const {
    return training_;
}

TensorF32 Dropout::forward(const TensorF32& x) {
    // If not in training mode, return input directly (no dropout during evaluation)
    if (!training_) {
        TensorF32 output(x.shape(), x.device());
        output.copy_from(x);
        return output;
    }
    
    const float* x_data = x.data();
    int64_t total_elements = x.numel();
    
    // Generate dropout mask
    TensorF32 mask(x.shape(), x.device());
    float* mask_data = mask.data();
    
    if (broadcast_dim_ >= 0 && broadcast_dim_ < x.shape().ndim()) {
        // Broadcast mode: generate mask where broadcast_dim dimension shares the same value
        
        // Calculate dimensions for broadcasting
        int64_t outer_dims = 1;
        int64_t broadcast_size = x.shape().dims[broadcast_dim_];
        int64_t inner_dims = 1;
        
        for (int i = 0; i < broadcast_dim_; ++i) {
            outer_dims *= x.shape().dims[i];
        }
        for (int i = broadcast_dim_ + 1; i < x.shape().ndim(); ++i) {
            inner_dims *= x.shape().dims[i];
        }
        
        // Generate mask: for each (outer, inner) position, generate 1 random value
        // and broadcast to broadcast_size
        for (int64_t outer = 0; outer < outer_dims; ++outer) {
            for (int64_t inner = 0; inner < inner_dims; ++inner) {
                float mask_val = dist_(rng_) ? 1.0f : 0.0f;  // Shared value
                for (int64_t b = 0; b < broadcast_size; ++b) {
                    int64_t idx = outer * broadcast_size * inner_dims + b * inner_dims + inner;
                    mask_data[idx] = mask_val;
                }
            }
        }
    } else {
        // No broadcast: generate independent random values for each element
        for (int64_t i = 0; i < total_elements; ++i) {
            mask_data[i] = dist_(rng_) ? 1.0f : 0.0f;
        }
    }
    
    // Apply mask and scaling: output = mask * x / (1 - p_drop)
    TensorF32 output(x.shape(), x.device());
    float* out_data = output.data();
    float scale = 1.0f / (1.0f - p_drop_);
    
    for (int64_t i = 0; i < total_elements; ++i) {
        out_data[i] = mask_data[i] * x_data[i] * scale;
    }
    
    return output;
}

} // namespace rfaa

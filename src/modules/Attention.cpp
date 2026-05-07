#include "rfaa/Attention.h"

namespace rfaa {

// Attention 模块的桩实现
// 实际应调用 CUDA kernel 或 CPU 实现

SelfAttention::SelfAttention(const AttnConfig& config) : config_(config) {}
SelfAttention::~SelfAttention() = default;

TensorF32 SelfAttention::forward(const TensorF32& x, const TensorF32* bias) {
    // 简化：返回输入 (占位)
    TensorF32 output;
    output.copy_from(x);
    return output;
}

MSARowAttention::MSARowAttention(const AttnConfig& config) : config_(config) {}

TensorF32 MSARowAttention::forward(const TensorF32& msa, const TensorF32& pair_bias) {
    TensorF32 row_attention_forward;
    // tmp spaceholder
    row_attention_forward.copy_from(msa);
    return row_attention_forward;
}

MSAColAttention::MSAColAttention(const AttnConfig& config) : config_(config) {}

TensorF32 MSAColAttention::forward(const TensorF32& msa) {
    TensorF32 col_attention_forward;
    // tmp spaceholder
    col_attention_forward.copy_from(msa);
    return col_attention_forward;
}

CrossAttention::CrossAttention(int q_dim, int kv_dim, int n_head)
    : q_dim_(q_dim), kv_dim_(kv_dim), n_head_(n_head) {}

TensorF32 CrossAttention::forward(const TensorF32& query, const TensorF32& kv) {
    TensorF32 cross_attention_forward;
    // tmp spaceholder
    cross_attention_forward.copy_from(query);
    return cross_attention_forward;
}

TriangleMultiplication::TriangleMultiplication(int dim, Direction dir)
    : dim_(dim), dir_(dir) {}

TensorF32 TriangleMultiplication::forward(const TensorF32& pair) {
    TensorF32 tri_mul_forward;
    // tmp spaceholder
    tri_mul_forward.copy_from(pair);
    return tri_mul_forward;
}

FeedForward::FeedForward(int dim, int hidden_dim, float dropout)
    : dim_(dim), hidden_dim_(hidden_dim) {}

TensorF32 FeedForward::forward(const TensorF32& x) {
    TensorF32 feed_forward;
    // tmp spaceholder
    feed_forward.copy_from(x);
    return feed_forward;
}

} // namespace rfaa

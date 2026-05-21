#include "rfaa/Attention.h"

namespace rfaa {

// Attention 模块的桩实现
// 实际应调用 CUDA kernel 或 CPU 实现

SelfAttention::SelfAttention(const AttnConfig& config) : config_(config) {}
SelfAttention::~SelfAttention() = default;

TensorF32 SelfAttention::forward(const TensorF32& Q, const TensorF32& K, const TensorF32& V, const TensorF32* bias) {
    // 简化：返回输入 (占位)
    TensorF32 output;
    output.copy_from(Q);
    return output;
}

MSARowAttention::MSARowAttention(const AttnConfig& config) : config_(config) {
    // 

}

TensorF32 MSARowAttention::forward(const TensorF32& msa, const TensorF32& pair_biased) {
    
    LayerNorm msa_layernorm(D_MSA);
    TensorF32 msa_norm = msa_layernorm.forward(msa);
    LayerNorm pair_layernorm(D_PAIR);
    TensorF32 pair_norm = pair_layernorm.forward(pair_biased);
    
    TensorF32 Q = Wq.forward(msa_norm);
    TensorF32 K = Wk.forward(msa_norm);
    TensorF32 V = Wv.forward(msa_norm);

    TensorF32 bias = to_b(pair_norm);
    TensorF32 gate = sigmoid(to_g(msa_norm));
    // call the multi_head_attention function
    //TensorF32 attn_out = multi_head_attention(Q, K, V, bias);
    TensorF32 attn_out = self_attn_.forward(Q, K, V, bias);
    // gated attention
    attn_out = gate * attn_out;
    TensorF32 attn_out_proj = to_out(attn_out);

    return attn_out_proj;
}

MSAColAttention::MSAColAttention(const AttnConfig& config) : config_(config) {}

TensorF32 MSAColAttention::forward(const TensorF32& msa) {
    TensorF32 col_attention_forward;
    // tmp spaceholder
    col_attention_forward.copy_from(msa);
    return col_attention_forward;
}

PairRowAttention::PairRowAttention(const AttnConfig& config) : config_(config) {
    // init all the linear layers
    to_out.zeros_weight();
    to_g.zeros_weight();
    to_g.ones_bias();
}

TensorF32 PairRowAttention::forward(const TensorF32& pair, const TensorF32& str_bias) {

    // row attention
    TensorF32 pair_row = pair.permute({0, 2, 1, 3});
    TensorF32 str_bias_row = str_bias.permute({0, 2, 1, 3});

    LayerNorm pair_layernorm(D_PAIR);
    TensorF32 pair_norm = pair_layernorm.forward(pair_row);
    LayerNorm bias_layernorm(D_PAIR);
    TensorF32 bias_norm = bias_layernorm.forward(str_bias_row);
    
    TensorF32 Q = Wq.forward(pair_norm);
    TensorF32 K = Wk.forward(pair_norm);
    TensorF32 V = Wv.forward(pair_norm);

    TensorF32 bias = to_b(bias_norm);
    TensorF32 gate = sigmoid(to_g(pair_norm));
    // call the multi_head_attention function
    //TensorF32 attn_out = multi_head_attention(Q, K, V, bias);
    TensorF32 attn_out = self_attn_.forward(Q, K, V, bias);
    // gated attention
    attn_out = gate * attn_out;
    TensorF32 attn_out_proj = to_out(attn_out);
    attn_out_proj = attn_out_proj.permute({0, 2, 1, 3});
    return attn_out_proj;
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
    : dim_(dim), hidden_dim_(hidden_dim) {
        
    }

TensorF32 FeedForward::forward(const TensorF32& x) {
    TensorF32 x_norm = x_layernorm.forward(x);
    TensorF32 x_hidden = linear1_.forward(x_norm);
    x_hidden = relu(x_hidden);
    x_hidden = dropout_.forward(x_hidden);
    TensorF32 x_out = linear2_.forward(x_hidden);
    return x_out;
    
}

} // namespace rfaa

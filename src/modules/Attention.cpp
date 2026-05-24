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

MSAColAttention::MSAColAttention(const AttnConfig& config) : config_(config) {

}

// difference between msa row and col?
// no attention bias
TensorF32 MSAColAttention::forward(const TensorF32& msa) {
    LayerNorm msa_layernorm(D_MSA);
    TensorF32 msa_norm = msa_layernorm.forward(msa);
    
    TensorF32 Q = Wq.forward(msa_norm);
    TensorF32 K = Wk.forward(msa_norm);
    TensorF32 V = Wv.forward(msa_norm);

    TensorF32 gate = sigmoid(to_g(msa_norm));
    // call the multi_head_attention function
    //TensorF32 attn_out = multi_head_attention(Q, K, V, bias);
    TensorF32 attn_out = self_attn_.forward(Q, K, V);
    // gated attention
    attn_out = gate * attn_out;
    TensorF32 attn_out_proj = to_out(attn_out);

    return attn_out_proj;
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

PairColAttention::PairColAttention(const AttnConfig& config) : config_(config) {
    // init all the linear layers
    to_out.zeros_weight();
    to_g.zeros_weight();
    to_g.ones_bias();
}

// pair column attention will use str_bias as well and add to the attention score
TensorF32 PairColAttention::forward(const TensorF32& pair, const TensorF32& str_bias) {
    // col attention
    LayerNorm pair_layernorm(D_PAIR);
    TensorF32 pair_norm = pair_layernorm.forward(pair);
    LayerNorm bias_layernorm(D_PAIR);
    TensorF32 bias_norm = bias_layernorm.forward(str_bias);
    
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

TriangleMultiplication::TriangleMultiplication(int dim)
    : dim_(dim) {
        left_gate_.zeros_weight();
        right_gate_.zeros_weight();
        gate_.zeros_weight();

        output_proj_.zeros_weight();
    }

TensorF32 TriangleMultiplication::forward(const TensorF32& pair, bool bOutgoing = true) {
    TensorF32 pair_norm = layernorm_.forward(pair);
    TensorF32 left = left_proj_.forward(pair_norm); //(B, L, L,D_HIDDEN_TRIMUL)
    TensorF32 right = right_proj_.forward(pair_norm);
    TensorF32 left_gate = sigmoid(left_gate_.forward(pair_norm));
    TensorF32 right_gate = sigmoid(right_gate_.forward(pair_norm));
    left = left * left_gate;
    right = right * right_gate;
    // outer product for outgoing and incoming
    TensorF32 tri_mul_forward;
    if (bOutgoing) {
        // need the unit test for the triangle mult function
        tri_mul_forward = triangle_mult(left, right, float(pair.shape().dims[1]), true);  
        // (B, L, L, D_HIDDEN_TRIMUL*D_HIDDEN_TRIMUL)
        //tri_mul_forward = outer_product(left, right);  // (B, L, L, D_HIDDEN_TRIMUL*D_HIDDEN_TRIMUL)
    } else {
        tri_mul_forward = triangle_mult(left, right, float(pair.shape().dims[1]), false);  
        //tri_mul_forward = outer_product(right, left);  // (B, L, L, D_HIDDEN_TRIMUL*D_HIDDEN_TRIMUL)
    }
    tri_mul_forward = output_layernorm_.forward(tri_mul_forward);
    tri_mul_forward = output_proj_.forward(tri_mul_forward);

    //(B, L, L, D_PAIR))
    TensorF32 gate = sigmoid(gate_.forward(pair_norm));
    tri_mul_forward = gate * tri_mul_forward;

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

TemplatePairStack::TemplatePairStack() {
    // init all the layers
    
    gate_proj_.zeros_weight();
    gate_proj_.ones_bias();
}

TensorF32 TemplatePairStack::forward(const TensorF32& pair, const TensorF32& rbf_feature, const TensorF32& state) {
    
    TensorF32 rbf_proj = rbf_proj_.forward(rbf_feature);  // (B,L,L,128)
    
    TensorF32 state_normed = state_norm_.forward(state);
    
            // different weights for left and right?
    TensorF32 left = left_proj_.forward(state_normed);   // (B,L,16)
    TensorF32 right = right_proj_.forward(state_normed); // (B,L,16)
    TensorF32 gate = outer_product(left, right);  // (B,L,L,256)
    gate = gate_proj_.forward(gate);  // (B,L,L,128)
    gate = sigmoid(gate);
    rbf_feature = rbf_feature * gate;
    
    TensorF32 pair_tmp;
    pair_tmp.copy_from(pair);
    pair_tmp = pair_tmp + drop_row.forward(tri_mul_out_->forward(pair_tmp, true));
    pair_tmp = pair_tmp + drop_row.forward(tri_mul_in_->forward(pair_tmp, false));
    pair_tmp = pair_tmp + drop_row_.forward(pair_row_attn_.forward(pair_tmp, rbf_proj));
    pair_tmp = pair_tmp + drop_col_.forward(pair_col_attn_.forward(pair_tmp, rbf_proj));
    pair_tmp = pair_tmp + pair_ff_.forward(pair_tmp);
    return pair_tmp;
}

} // namespace rfaa

#include "rfaa/Attention.h"

namespace rfaa {

// Attention 模块的桩实现
// 实际应调用 CUDA kernel 或 CPU 实现

SelfAttention::SelfAttention(const AttnConfig& config) : config_(config) {}
SelfAttention::~SelfAttention() = default;

TensorF32 SelfAttention::forward(const TensorF32& Q, const TensorF32& K, const TensorF32& V, const TensorF32* bias) {
    // ===== 1. K^T: 交换最后两维 =====
    // permute op
    Tensor* K_T = permute(K, {0, 1, 3, 2});  // (B, H, D_head, L)

    // tensor multiply out_prod
    // ===== 2. scores = Q @ K^T =====
    Tensor* scores = out_prod(Q, K_T);         // (B, H, L, L)

    // sqrt op
    // ===== 3. scale = 1/sqrt(d_head) =====
    //float scale_val = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
    // test for Q.shape()[3] before make a tensor
    Tensor* dim_head = make_scalar(Q.shape()[3]);
    float scale_val = 1.0f / sqrt(dim_head);
    Tensor* scaled = scale(scores, scale_val);

    // elementwise tensor add
    // ===== 4. add bias =====
    if (bias != nullptr) {
        scaled = add_impl(scaled, bias);           // broadcast (B,1,L,L) → (B,H,L,L)
    }

    // softmax op
    // ===== 5. softmax =====
    Tensor* attn = softmax(scaled);

    // ===== 6. out = attn @ V =====
    // mul_mat
    Tensor* output = out_prod(attn, V);        // (B, H, L, D_head)

    return output;
}

MSARowAttention::MSARowAttention(const AttnConfig& config) : config_(config) {
    // 

}

TensorF32 MSARowAttention::forward(const TensorF32& msa, const TensorF32& pair_biased) {
    
    //LayerNorm msa_row_layernorm(D_MSA);
    TensorF32 msa_norm = msa_row_layernorm_.forward(msa);
    //LayerNorm pair_row_layernorm(D_PAIR);
    TensorF32 pair_norm = pair_row_layernorm_.forward(pair_biased);
    
    TensorF32 Q = msa_row_Wq_.forward(msa_norm);
    TensorF32 K = msa_row_Wk_.forward(msa_norm);
    TensorF32 V = msa_row_Wv_.forward(msa_norm);

    TensorF32 bias = msa_row_to_b_.forward(pair_norm);
    TensorF32 gate = sigmoid(msa_row_to_g_.forward(msa_norm));
    // call the multi_head_attention function
    //TensorF32 attn_out = multi_head_attention(Q, K, V, bias);
    TensorF32 attn_out = self_attn_.forward(Q, K, V, bias);
    // gated attention
    //TensorF32 gated_attn_out = gate * attn_out;
    TensorF32 gated_attn_out = out_prod(gate, attn_out);
    TensorF32 attn_out_proj = msa_row_to_out_.forward(gated_attn_out);

    return attn_out_proj;
}

MSAColAttention::MSAColAttention(const AttnConfig& config) : config_(config) {

}

// difference between msa row and col?
// no attention bias
TensorF32 MSAColAttention::forward(const TensorF32& msa) {
    //LayerNorm msa_col_layernorm(D_MSA);
    TensorF32 msa_norm = msa_col_layernorm_.forward(msa);
    
    TensorF32 Q = msa_col_Wq_.forward(msa_norm);
    TensorF32 K = msa_col_Wk_.forward(msa_norm);
    TensorF32 V = msa_col_Wv_.forward(msa_norm);
    TensorF32 gate = sigmoid(msa_col_to_g_.forward(msa_norm));
    // call the multi_head_attention function
    //TensorF32 attn_out = multi_head_attention(Q, K, V, bias);
    TensorF32 attn_out = self_attn_.forward(Q, K, V);
    // gated attention
    TensorF32 gated_attn_out = out_prod(gate, attn_out);
    TensorF32 attn_out_proj = msa_col_to_out_.forward(gated_attn_out);

    return attn_out_proj;
}

MSAGlobalColAttention::MSAGlobalColAttention(const AttnConfig& config) : MSAColAttention(config) {
    // 可以在这里覆盖父类的成员变量初始化
    Wk(D_MSA, D_MSA);
    Wv(D_MSA, D_MSA);
    // not multi-headed attention

    to_out.zeros_weight();
    to_g.zeros_weight();
    to_g.ones_bias();
}

TensorF32 MSAGlobalColAttention::forward(const TensorF32& msa) {
    //LayerNorm msa_global_col_layernorm(D_MSA);
    TensorF32 msa_norm = msa_global_col_layernorm_.forward(msa);
    
    TensorF32 Q = msa_global_col_Wq_.forward(msa_norm);
    // mean graph node?
    Q = mean(Q, 1);
    // Q = Q.mean(dim=1);
    // mean Q_mean[b, l, d] = (Q[b, 0, l, d] + Q[b, 1, l, d] + ... + Q[b, 7, l, d]) / 8

    // (B, L, h, d_head)
    TensorF32 K = msa_global_col_Wk_.forward(msa_norm);
    TensorF32 V = msa_global_col_Wv_.forward(msa_norm);

    TensorF32 gate = sigmoid(msa_global_col_to_g_.forward(msa_norm));
    // call the multi_head_attention function
    //TensorF32 attn_out = multi_head_attention(Q, K, V, bias);
    TensorF32 attn_out = self_attn_.forward(Q, K, V);
    // concat on the head dimension
    //attn = rearrange(attn, 'b l h n -> b 1 l (h n)') # (B, 1, L, d_msa)
    // attn rearrange (B, 1, L, d_msa)
    
    // gated attention
    TensorF32 gated_attn_out = out_prod(gate, attn_out);
    TensorF32 attn_out_proj = msa_global_col_to_out_.forward(gated_attn_out);

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
    //TensorF32 pair_row = pair.permute({0, 2, 1, 3});
    //TensorF32 str_bias_row = str_bias.permute({0, 2, 1, 3});
    TensorF32 pair_row = permute(pair, {0, 2, 1, 3});
    TensorF32 str_bias_row = permute(str_bias, {0, 2, 1, 3});

    //LayerNorm pair_layernorm(D_PAIR);
    TensorF32 pair_norm = pair_row_layernorm_.forward(pair_row);
    //LayerNorm bias_layernorm(D_PAIR);
    TensorF32 bias_norm = bias_row_layernorm_.forward(str_bias_row);
    
    TensorF32 Q = pair_row_Wq_.forward(pair_norm);
    TensorF32 K = pair_row_Wk_.forward(pair_norm);
    TensorF32 V = pair_row_Wv_.forward(pair_norm);

    TensorF32 bias = pair_row_to_b_.forward(bias_norm);
    TensorF32 gate = sigmoid(pair_row_to_g_.forward(pair_norm));
    // call the multi_head_attention function
    //TensorF32 attn_out = multi_head_attention(Q, K, V, bias);
    TensorF32 attn_out = self_attn_.forward(Q, K, V, bias);
    // gated attention
    // elementwise multiply:?
    //TensorF32 gated_attn_out = gate * attn_out;
    TensorF32 gated_attn_out = out_prod(gate, attn_out);
    TensorF32 attn_out_proj = pair_row_to_out_.forward(gated_attn_out);
    //attn_out_proj = attn_out_proj.permute({0, 2, 1, 3});
    TensorF32 result = permute(attn_out_proj, {0, 2, 1, 3});
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
    //LayerNorm pair_layernorm(D_PAIR);
    TensorF32 pair_norm = pair_col_layernorm_.forward(pair);
    //LayerNorm bias_layernorm(D_PAIR);
    TensorF32 bias_norm = bias_col_layernorm_.forward(str_bias);
    
    TensorF32 Q = pair_col_Wq_.forward(pair_norm);
    TensorF32 K = pair_col_Wk_.forward(pair_norm);
    TensorF32 V = pair_col_Wv_.forward(pair_norm);

    //TensorF32 bias = to_b(bias_norm);
    TensorF32 bias = pair_col_to_b_.forward(bias_norm);
    TensorF32 gate = sigmoid(pair_col_to_g_.forward(pair_norm));
    //TensorF32 gate = sigmoid(to_g(pair_norm));
    
    // call the multi_head_attention function
    //TensorF32 attn_out = multi_head_attention(Q, K, V, bias);
    TensorF32 attn_out = self_attn_.forward(Q, K, V, bias);
    // gated attention
    //TensorF32 gated_attn_out = gate * attn_out;
    TensorF32 gated_attn_out = out_prod(gate, attn_out);
    TensorF32 attn_out_proj = pair_col_to_out_.forward(gated_attn_out);
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
    //left = left * left_gate;
    //right = right * right_gate;
    TensorF32 left_gated = out_prod(left, left_gate);
    TensorF32 right_gated = out_prod(right, right_gate);

    // outer product for outgoing and incoming
    TensorF32 tri_mul_forward;
    if (bOutgoing) {
        // need the unit test for the triangle mult function
        tri_mul_forward = triangle_mult(left_gated, right_gated, float(pair.shape().dims[1]), true);  
        // (B, L, L, D_HIDDEN_TRIMUL*D_HIDDEN_TRIMUL)
        //tri_mul_forward = outer_product(left, right);  // (B, L, L, D_HIDDEN_TRIMUL*D_HIDDEN_TRIMUL)
    } else {
        tri_mul_forward = triangle_mult(left_gated, right_gated, float(pair.shape().dims[1]), false);  
        //tri_mul_forward = outer_product(right, left);  // (B, L, L, D_HIDDEN_TRIMUL*D_HIDDEN_TRIMUL)
    }
    TensorF32 tri_mul_forward_norm = output_layernorm_.forward(tri_mul_forward);
    TensorF32 tri_mul_forward_proj = output_proj_.forward(tri_mul_forward_norm);

    //(B, L, L, D_PAIR))
    TensorF32 gate = sigmoid(gate_.forward(pair_norm));
    //tri_mul_forward = gate * tri_mul_forward;
    TensorF32 tri_mul_forward_gated = out_prod(gate, tri_mul_forward_proj);

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

TensorF32 TemplatePairStack::forward(const TensorF32& pair, TensorF32& rbf_feature, const TensorF32& state) {
    
    TensorF32 rbf_proj = rbf_proj_.forward(rbf_feature);  // (B,L,L,128)
    
    TensorF32 state_normed = state_norm_.forward(state);
    
            // different weights for left and right?
    TensorF32 left = left_proj_.forward(state_normed);   // (B,L,16)
    TensorF32 right = right_proj_.forward(state_normed); // (B,L,16)
    TensorF32 gate = out_prod(left, right);  // (B,L,L,256)
    TensorF32 gate_proj = gate_proj_.forward(gate);  // (B,L,L,128)
    TensorF32 gate_sig = sigmoid(gate_proj);
    //rbf_feature = rbf_feature * gate;
    //rbf_feature = out_prod(rbf_feature, gate_sig);
    Tensor* out_rbf_feature = out_prod(rbf_feature, gate_sig);
    rbf_feature = &(*out_rbf_feature);

    TensorF32 pair_tmp;
    pair_tmp.copy_from(pair);
    // dup op?
    /* pair_tmp = pair_tmp + drop_row_.forward(tri_mul_out_->forward(pair_tmp, true));
    pair_tmp = pair_tmp + drop_row_.forward(tri_mul_in_->forward(pair_tmp, false));
    pair_tmp = pair_tmp + drop_row_.forward(pair_row_attn_.forward(pair_tmp, rbf_proj));
    pair_tmp = pair_tmp + drop_col_.forward(pair_col_attn_.forward(pair_tmp, rbf_proj));
    pair_tmp = pair_tmp + pair_ff_.forward(pair_tmp); */

    Tensor* tri_out = tri_mul_out_->forward(pair_tmp, true);
    Tensor* tri_out_drop_row = drop_row_.forward(tri_out);
    Tensor* pair_tri_out = add_impl(pair_tmp, tri_out_drop_row);

    Tensor* tri_in = tri_mul_in_->forward(pair_tri_out, true);
    Tensor* tri_in_drop_row = drop_row_.forward(tri_in);
    Tensor* pair_tri_in = add_impl(pair_tri_out, tri_in_drop_row);

    Tensor* pair_row_attn = pair_row_attn_.forward(pair_tri_in, rbf_proj);
    Tensor* pair_row_attn_drop_row = drop_row_.forward(pair_row_attn);
    Tensor* pair_row_attn = add_impl(pair_tri_in, pair_row_attn_drop_row);

    Tensor* pair_col_attn = pair_col_attn_.forward(pair_row_attn, rbf_proj);
    Tensor* pair_col_attn_drop_col = drop_col_.forward(pair_row_attn);
    Tensor* pair_ff = pair_ff_.forward(pair_col_attn_drop_col);
    Tensor* pair_final = add_impl(pair_row_attn, pair_ff);

    return pair_final;
}

} // namespace rfaa

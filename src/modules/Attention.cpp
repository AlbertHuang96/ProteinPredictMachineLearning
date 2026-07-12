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

// ===== MSARowAttention (non-owning pointer 版本) =====
// 旧构造函数 (保留注释):
// MSARowAttention::MSARowAttention(const AttnConfig& config) : config_(config) {}
void MSARowAttention::set_params(const AttnConfig& config,
                                 LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                                 LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv) {
    config_ = config;
    to_b_   = to_b;   to_g_   = to_g;   to_out_ = to_out;
    Wq_     = Wq;     Wk_     = Wk;     Wv_     = Wv;
}

TensorF32 MSARowAttention::forward(const TensorF32& msa, const TensorF32& pair_biased) {
    // 输入已由调用方做 layernorm
    TensorF32 Q = Wq_->forward(msa);
    TensorF32 K = Wk_->forward(msa);
    TensorF32 V = Wv_->forward(msa);
    TensorF32 bias = to_b_->forward(pair_biased);
    TensorF32 gate = sigmoid(to_g_->forward(msa));
    TensorF32 attn_out = self_attn_.forward(Q, K, V, bias);
    TensorF32 gated_attn_out = out_prod(gate, attn_out);
    return to_out_->forward(gated_attn_out);
}

// ===== MSAColAttention (non-owning pointer 版本) =====
// 旧构造函数 (保留注释):
// MSAColAttention::MSAColAttention(const AttnConfig& config) : config_(config) {}
void MSAColAttention::set_params(const AttnConfig& config,
                                 LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                                 LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv) {
    config_ = config;
    to_b_   = to_b;   to_g_   = to_g;   to_out_ = to_out;
    Wq_     = Wq;     Wk_     = Wk;     Wv_     = Wv;
}

TensorF32 MSAColAttention::forward(const TensorF32& msa) {
    // 输入已由调用方做 layernorm
    TensorF32 Q = Wq_->forward(msa);
    TensorF32 K = Wk_->forward(msa);
    TensorF32 V = Wv_->forward(msa);
    TensorF32 gate = sigmoid(to_g_->forward(msa));
    TensorF32 attn_out = self_attn_.forward(Q, K, V);
    TensorF32 gated_attn_out = out_prod(gate, attn_out);
    return to_out_->forward(gated_attn_out);
}

// ===== MSAGlobalColAttention (non-owning pointer 版本) =====
// 旧构造函数 (保留注释):
// MSAGlobalColAttention::MSAGlobalColAttention(const AttnConfig& config) : MSAColAttention(config) { ... }
// 父类 MSAColAttention::set_params() 注入参数, 子类无额外成员

TensorF32 MSAGlobalColAttention::forward(const TensorF32& msa) {
    // 输入已由调用方做 layernorm
    TensorF32 Q = Wq_->forward(msa);
    Q = mean(Q, 1);
    TensorF32 K = Wk_->forward(msa);
    TensorF32 V = Wv_->forward(msa);
    TensorF32 gate = sigmoid(to_g_->forward(msa));
    TensorF32 attn_out = self_attn_.forward(Q, K, V);
    TensorF32 gated_attn_out = out_prod(gate, attn_out);
    return to_out_->forward(gated_attn_out);
}


// ===== PairRowAttention (non-owning pointer 版本) =====
// 旧构造函数 (保留注释):
// PairRowAttention::PairRowAttention(const AttnConfig& config) : config_(config) { ... }
void PairRowAttention::set_params(const AttnConfig& config,
                                  LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                                  LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv) {
    config_ = config;
    to_b_   = to_b;   to_g_   = to_g;   to_out_ = to_out;
    Wq_     = Wq;     Wk_     = Wk;     Wv_     = Wv;
}

TensorF32 PairRowAttention::forward(const TensorF32& pair, const TensorF32& str_bias) {
    // 输入已由调用方做 layernorm + permute
    TensorF32 Q = Wq_->forward(pair);
    TensorF32 K = Wk_->forward(pair);
    TensorF32 V = Wv_->forward(pair);
    TensorF32 bias = to_b_->forward(str_bias);
    TensorF32 gate = sigmoid(to_g_->forward(pair));
    TensorF32 attn_out = self_attn_.forward(Q, K, V, bias);
    TensorF32 gated_attn_out = out_prod(gate, attn_out);
    TensorF32 attn_out_proj = to_out_->forward(gated_attn_out);
    return attn_out_proj;
}

// ===== PairColAttention (non-owning pointer 版本) =====
// 旧构造函数 (保留注释):
// PairColAttention::PairColAttention(const AttnConfig& config) : config_(config) { ... }
void PairColAttention::set_params(const AttnConfig& config,
                                  LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                                  LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv) {
    config_ = config;
    to_b_   = to_b;   to_g_   = to_g;   to_out_ = to_out;
    Wq_     = Wq;     Wk_     = Wk;     Wv_     = Wv;
}

TensorF32 PairColAttention::forward(const TensorF32& pair, const TensorF32& str_bias) {
    // 输入已由调用方做 layernorm
    TensorF32 Q = Wq_->forward(pair);
    TensorF32 K = Wk_->forward(pair);
    TensorF32 V = Wv_->forward(pair);
    TensorF32 bias = to_b_->forward(str_bias);
    TensorF32 gate = sigmoid(to_g_->forward(pair));
    TensorF32 attn_out = self_attn_.forward(Q, K, V, bias);
    TensorF32 gated_attn_out = out_prod(gate, attn_out);
    return to_out_->forward(gated_attn_out);
}

CrossAttention::CrossAttention(int q_dim, int kv_dim, int n_head)
    : q_dim_(q_dim), kv_dim_(kv_dim), n_head_(n_head) {}

TensorF32 CrossAttention::forward(const TensorF32& query, const TensorF32& kv) {
    TensorF32 cross_attention_forward;
    // tmp spaceholder
    cross_attention_forward.copy_from(query);
    return cross_attention_forward;
}

// ===== TriangleMultiplication (non-owning pointer 版本) =====
// 旧构造函数 (保留注释):
// TriangleMultiplication::TriangleMultiplication(int dim) : dim_(dim) { ... }
void TriangleMultiplication::set_params(int dim,
                                        LayerNorm*   layernorm,     LinearLayer* left_proj,
                                        LinearLayer* right_proj,    LinearLayer* left_gate,
                                        LinearLayer* right_gate,    LinearLayer* gate,
                                        LayerNorm*   output_layernorm, LinearLayer* out_proj) {
    dim_             = dim;
    layernorm_        = layernorm;        // D_PAIR (128)
    left_proj_        = left_proj;        // D_PAIR (128) → D_HIDDEN_TRIMUL (128)
    right_proj_       = right_proj;       // D_PAIR (128) → D_HIDDEN_TRIMUL (128)
    left_gate_        = left_gate;        // D_PAIR (128) → D_HIDDEN_TRIMUL (128)
    right_gate_       = right_gate;       // D_PAIR (128) → D_HIDDEN_TRIMUL (128)
    gate_             = gate;             // D_PAIR (128) → D_PAIR (128)
    output_layernorm_ = output_layernorm; // D_HIDDEN_TRIMUL (128)
    out_proj_         = out_proj;         // D_HIDDEN_TRIMUL (128) → D_PAIR (128)
}

TensorF32 TriangleMultiplication::forward(const TensorF32& pair, bool bOutgoing) {
    TensorF32 pair_norm = layernorm_->forward(pair);
    TensorF32 left = left_proj_->forward(pair_norm); //(B, L, L,D_HIDDEN_TRIMUL)
    TensorF32 right = right_proj_->forward(pair_norm);
    TensorF32 left_gate = sigmoid(left_gate_->forward(pair_norm));
    TensorF32 right_gate = sigmoid(right_gate_->forward(pair_norm));
    TensorF32 left_gated = out_prod(left, left_gate);
    TensorF32 right_gated = out_prod(right, right_gate);

    // outer product for outgoing and incoming
    TensorF32 tri_mul_forward;
    if (bOutgoing) {
        tri_mul_forward = triangle_mult(left_gated, right_gated, float(pair.shape().dims[1]), true);  
    } else {
        tri_mul_forward = triangle_mult(left_gated, right_gated, float(pair.shape().dims[1]), false);  
    }
    TensorF32 tri_mul_forward_norm = output_layernorm_->forward(tri_mul_forward);
    TensorF32 tri_mul_forward_proj = out_proj_->forward(tri_mul_forward_norm);

    TensorF32 gate = sigmoid(gate_->forward(pair_norm));
    TensorF32 tri_mul_forward_gated = out_prod(gate, tri_mul_forward_proj);

    return tri_mul_forward;
}
}

// ===== FeedForward (non-owning pointer 版本) =====
// 旧构造函数 (保留注释):
// FeedForward::FeedForward(int dim, int hidden_dim, float dropout)
//     : dim_(dim), hidden_dim_(hidden_dim) {}
void FeedForward::set_params(int dim, int hidden_dim, float dropout,
                             LayerNorm* layernorm, LinearLayer* linear1, LinearLayer* linear2) {
    dim_          = dim;
    hidden_dim_   = hidden_dim;
    dropout_rate_ = dropout;
    layernorm_    = layernorm;   // dim_
    linear1_      = linear1;     // dim_ → dim_*hidden_dim_
    linear2_      = linear2;     // dim_*hidden_dim_ → dim_
}

TensorF32 FeedForward::forward(const TensorF32& x) {
    // 旧: x_layernorm.forward(x); → layernorm_->forward(x)
    TensorF32 x_norm = layernorm_->forward(x);
    TensorF32 x_hidden = linear1_->forward(x_norm);
    x_hidden = relu(x_hidden);
    x_hidden = dropout_.forward(x_hidden);
    TensorF32 x_out = linear2_->forward(x_hidden);
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

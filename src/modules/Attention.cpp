#include "rfaa/Attention.h"
#include "rfaa/Embedding.h"

namespace rfaa {

// Attention 模块的桩实现
// 实际应调用 CUDA kernel 或 CPU 实现

SelfAttention::SelfAttention(const AttnConfig& config) : config_(config) {}
SelfAttention::~SelfAttention() = default;

TensorF32 SelfAttention::forward(const TensorF32& Q, const TensorF32& K, const TensorF32& V, const TensorF32* bias) {
    // ===== 1. K^T: 交换最后两维 =====
    auto K_T = permute(const_cast<TensorF32*>(&K), {0, 1, 3, 2});  // (B, H, D_head, L)

    // ===== 2. scores = Q @ K^T =====
    auto scores = out_prod(const_cast<TensorF32*>(&Q), K_T);  // (B, H, L, L)

    // ===== 3. scale = 1/sqrt(d_head) =====
    float scale_val = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
    auto scaled = scale(scores, scale_val);

    // ===== 4. add bias =====
    if (bias != nullptr) {
        scaled = add_impl(scaled, const_cast<TensorF32*>(bias), /*inplace=*/false);
    }

    // ===== 5. softmax =====
    auto attn = softmax(scaled);

    // ===== 6. out = attn @ V =====
    auto output = out_prod(attn, const_cast<TensorF32*>(&V));

    TensorF32 result;
    result.copy_from(*output);
    return result;
}

// ===== MSARowAttention (non-owning pointer 版本) =====
// 旧构造函数 (保留注释):
// MSARowAttention::MSARowAttention(const AttnConfig& config) : config_(config) {}
void MSARowAttention::set_params(const AttnConfig& config,
                                 LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                                 LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv) {
    config_ = config;
    self_attn_ = std::make_unique<SelfAttention>(config);
    to_b_   = to_b;   to_g_   = to_g;   to_out_ = to_out;
    Wq_     = Wq;     Wk_     = Wk;     Wv_     = Wv;
}

TensorF32 MSARowAttention::forward(const TensorF32& msa, const TensorF32& pair_biased) {
    auto Q    = Wq_->forward(msa);
    auto K    = Wk_->forward(msa);
    auto V    = Wv_->forward(msa);
    auto bias = to_b_->forward(pair_biased);
    auto gv   = to_g_->forward(msa);
    auto gate = sigmoid(&gv);
    auto attn_out = self_attn_->forward(Q, K, V, &bias);
    auto gated_attn_out = out_prod(gate, &attn_out);
    return to_out_->forward(*gated_attn_out);
}

// ===== MSAColAttention (non-owning pointer 版本) =====
void MSAColAttention::set_params(const AttnConfig& config,
                                 LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                                 LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv) {
    config_ = config;
    self_attn_ = std::make_unique<SelfAttention>(config);
    to_b_   = to_b;   to_g_   = to_g;   to_out_ = to_out;
    Wq_     = Wq;     Wk_     = Wk;     Wv_     = Wv;
}

TensorF32 MSAColAttention::forward(const TensorF32& msa) {
    auto Q    = Wq_->forward(msa);
    auto K    = Wk_->forward(msa);
    auto V    = Wv_->forward(msa);
    auto gv   = to_g_->forward(msa);
    auto gate = sigmoid(&gv);
    auto attn_out = self_attn_->forward(Q, K, V);
    auto gated_attn_out = out_prod(gate, &attn_out);
    return to_out_->forward(*gated_attn_out);
}

// ===== MSAGlobalColAttention =====
TensorF32 MSAGlobalColAttention::forward(const TensorF32& msa) {
    auto Q = Wq_->forward(msa);
    Q = mean(Q, 1);
    auto K    = Wk_->forward(msa);
    auto V    = Wv_->forward(msa);
    auto gv   = to_g_->forward(msa);
    auto gate = sigmoid(&gv);
    auto attn_out = self_attn_->forward(Q, K, V);
    auto gated_attn_out = out_prod(gate, &attn_out);
    return to_out_->forward(*gated_attn_out);
}

// ===== PairRowAttention =====
void PairRowAttention::set_params(const AttnConfig& config,
                                  LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                                  LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv) {
    config_ = config;
    self_attn_ = std::make_unique<SelfAttention>(config);
    to_b_   = to_b;   to_g_   = to_g;   to_out_ = to_out;
    Wq_     = Wq;     Wk_     = Wk;     Wv_     = Wv;
}

TensorF32 PairRowAttention::forward(const TensorF32& pair, const TensorF32& str_bias) {
    auto Q    = Wq_->forward(pair);
    auto K    = Wk_->forward(pair);
    auto V    = Wv_->forward(pair);
    auto bias = to_b_->forward(str_bias);
    auto gv   = to_g_->forward(pair);
    auto gate = sigmoid(&gv);
    auto attn_out = self_attn_->forward(Q, K, V, &bias);
    auto gated_attn_out = out_prod(gate, &attn_out);
    return to_out_->forward(*gated_attn_out);
}

// ===== PairColAttention =====
void PairColAttention::set_params(const AttnConfig& config,
                                  LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                                  LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv) {
    config_ = config;
    self_attn_ = std::make_unique<SelfAttention>(config);
    to_b_   = to_b;   to_g_   = to_g;   to_out_ = to_out;
    Wq_     = Wq;     Wk_     = Wk;     Wv_     = Wv;
}

TensorF32 PairColAttention::forward(const TensorF32& pair, const TensorF32& str_bias) {
    auto Q    = Wq_->forward(pair);
    auto K    = Wk_->forward(pair);
    auto V    = Wv_->forward(pair);
    auto bias = to_b_->forward(str_bias);
    auto gv   = to_g_->forward(pair);
    auto gate = sigmoid(&gv);
    auto attn_out = self_attn_->forward(Q, K, V, &bias);
    auto gated_attn_out = out_prod(gate, &attn_out);
    return to_out_->forward(*gated_attn_out);
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
    auto pair_norm = layernorm_->forward(const_cast<TensorF32*>(&pair));  // TensorF32*
    auto left  = left_proj_->forward(*pair_norm);   // 解引用传引用
    auto right = right_proj_->forward(*pair_norm);
    auto lgv   = left_gate_->forward(*pair_norm);
    auto rgv   = right_gate_->forward(*pair_norm);
    auto left_gate  = sigmoid(&lgv);
    auto right_gate = sigmoid(&rgv);
    auto left_gated  = out_prod(&left, left_gate);
    auto right_gated = out_prod(&right, right_gate);

    auto tri_mul_forward = triangle_mul(left_gated, right_gated, float(pair.shape().dims[1]), bOutgoing);

    auto tri_mul_forward_norm  = output_layernorm_->forward(tri_mul_forward);
    auto tri_mul_forward_proj  = out_proj_->forward(*tri_mul_forward_norm);

    auto gv = gate_->forward(*pair_norm);
    auto gate_out = sigmoid(&gv);
    auto tri_mul_forward_gated = out_prod(gate_out, &tri_mul_forward_proj);

    return std::move(*tri_mul_forward_gated);
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
    auto x_norm   = layernorm_->forward(const_cast<TensorF32*>(&x));  // 返回 TensorF32*
    auto x_hidden = linear1_->forward(*x_norm);                      // 解引用后传引用
    auto* x_relu  = relu(&x_hidden);                                   // relu 接受指针，返回指针
    auto x_dropped = dropout_.forward(*x_relu);                        // dropout 接受值引用
    auto x_out = linear2_->forward(x_dropped);                        // 传引用
    return x_out;
}

void TemplatePairStack::set_params(
    LinearLayer* rbf_proj,
    LayerNorm*   state_norm,
    LinearLayer* left_proj,   LinearLayer* right_proj,  LinearLayer* gate_proj,
    TriangleMultiplication* tri_mul_out, TriangleMultiplication* tri_mul_in,
    PairRowAttention* pair_row_attn, PairColAttention* pair_col_attn,
    FeedForward* pair_ff)
{
    rbf_proj_   = rbf_proj;
    state_norm_ = state_norm;
    left_proj_  = left_proj;
    right_proj_ = right_proj;
    gate_proj_  = gate_proj;
    tri_mul_out_ = tri_mul_out;
    tri_mul_in_  = tri_mul_in;
    pair_row_attn_ = pair_row_attn;
    pair_col_attn_ = pair_col_attn;
    pair_ff_       = pair_ff;

    // 初始化: gate_proj 权重零初始化, bias 置 1
    if (gate_proj_) {
        gate_proj_->zeros_weight();
        gate_proj_->ones_bias();
    }
}

TensorF32 TemplatePairStack::forward(const TensorF32& pair, TensorF32& rbf_feature, const TensorF32& state) {
    
    TensorF32 rbf_proj = rbf_proj_->forward(rbf_feature);  // (B,L,L,128)
    
    auto& state_normed = *state_norm_->forward(const_cast<TensorF32*>(&state));
    
    TensorF32 left  = left_proj_->forward(state_normed);   // (B,L,16)
    TensorF32 right = right_proj_->forward(state_normed);  // (B,L,16)
    auto gate  = out_prod(&left, &right);                   // (B,L,L,256)
    TensorF32 gate_proj = gate_proj_->forward(*gate);       // (B,L,L,128)
    auto gate_sig  = sigmoid(&gate_proj);
    auto out_rbf_feature = out_prod(&rbf_feature, gate_sig);
    rbf_feature.copy_from(*out_rbf_feature);

    TensorF32 pair_tmp;
    pair_tmp.copy_from(pair);

    auto tri_out = tri_mul_out_->forward(pair_tmp, true);
    auto tri_out_drop = drop_row_.forward(tri_out);
    auto pair_tri_out = add_impl(&pair_tmp, &tri_out_drop, /*inplace=*/false);

    auto tri_in = tri_mul_in_->forward(*pair_tri_out, false);
    auto tri_in_drop = drop_row_.forward(tri_in);
    auto pair_tri_in = add_impl(pair_tri_out, &tri_in_drop, /*inplace=*/false);

    auto pair_row_attn = pair_row_attn_->forward(*pair_tri_in, rbf_proj);
    auto row_drop = drop_row_.forward(pair_row_attn);
    auto pair_after_row = add_impl(pair_tri_in, &row_drop, /*inplace=*/false);

    auto pair_col_attn = pair_col_attn_->forward(*pair_after_row, rbf_proj);
    auto col_drop = drop_col_.forward(pair_col_attn);
    auto pair_after_col = add_impl(pair_after_row, &col_drop, /*inplace=*/false);

    auto pair_ff_out = pair_ff_->forward(*pair_after_col);
    auto pair_final = add_impl(pair_after_col, &pair_ff_out, /*inplace=*/false);

    TensorF32 result;
    result.copy_from(*pair_final);
    return result;
}

} // namespace rfaa

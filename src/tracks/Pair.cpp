#include "rfaa/Track.h"

namespace rfaa {

PairTrack::PairTrack(int seq_len, int dim, Device device)
    : seq_len_(seq_len), dim_(dim) {
    device_ = device;
    repr_ = zeros<float>({1, seq_len, seq_len, dim}, device);
}

void PairTrack::init_from_embedding(const TensorF32& left, const TensorF32& right, const TensorF32& bond_feats, const TensorF32& dist_matrix) {
    // left: (B, L, D), right: (B, L, D)
    // pair = outer_product(left, right) -> Linear
    EmbeddingLayer emb_left(NAATOKENS, D_PAIR);
    EmbeddingLayer emb_right(NAATOKENS, D_PAIR);
    TensorF32 left_emb = emb_left.forward(left).unsqueeze(1);   // (B, 1, L, D_PAIR)
    TensorF32 right_emb = emb_right.forward(right).unsqueeze(2); // (B, L, 1, D_PAIR)

    // outer sum
    repr_ = outer_sum(left_emb, right_emb);  // (B, L, L, D_PAIR)
    PositionalEncoding pos_enc(-32, 32, 8, D_PAIR);
    repr_ = repr_ + pos_enc.forward(repr_, index, bond_feats, dist_matrix, TensorF32()/*same_chain*/);  // 加上位置编码
}

TensorF32 PairTrack::templ_stack(const TensorF32& in_templ, const TensorF32& rbf_feat, const TensorF32& t1d) {
    TensorF32 templ = in_templ;  // (B, T, L, L, d_templ)
    int L = templ.shape().dims[2];
    int T = templ.shape().dims[1];
    int B = templ.shape().dims[0];
    templ.reshape({B*T, L, L, 64});  // (B*L*L, 1, d_templ)
    TensorF32 t1d_reshaped = t1d;
    t1d_reshaped.reshape({B*T, L, D_T1D});
    LinearLayer t1d_proj(D_T1D, D_STATE);
    TensorF32 state_proj = t1d_proj.forward(t1d_reshaped);
    TensorF32 out;
    for (int i = 0; i < 2; i++) {
        TensorF32 input = templ;
        TemplatePairStack block;
        templ = block.forward(input, rbf_feat, state_proj);
    }
    
    // d_templ = 64
    LayerNorm layernorm(64);
    out = layernorm.forward(templ);
    out.reshape({B, T, L, L, 64});
    return out;
}

void PairTrack::inject_template(const TensorF32& in_templ) {
    TensorF32 pair;
    pair.copy_from(pair_track_->representation());
    pair = pair.reshape({B*L*L, 1, D_PAIR});
    //D_TEMPL = 64
    // (B, T, L, L, d_templ) = (B, T, L, L, 64)
    int T = in_templ.shape().dims[1];
    TensorF32 templ = in_templ.permute({0, 2, 3, 1, 4}).reshape({B*L*L, 1, 64});
    SelfAttention self_attention();
    TensorF32 out = self_attention.forward(pair, templ, templ);  // (B*L*L, 1, D_PAIR)
    out = out.reshape({B, L, L, D_PAIR});
    pair = pair.reshape({B, L, L, D_PAIR});
    pair = pair + out;  // 注入模板信息
    pair_track_->representation() = pair;  // 更新 pair track 的表示
}

void PairTrack::update_from_msa(const TensorF32& msa) {
    // Step 2: msa2pair
    // Outer Product Mean
}

void PairTrack::update_self(const TensorF32& state_gate, const TensorF32& rbf_feat) {
    // Step 3: pair2pair
    // Triangle Multiplication
    // Biased Axial Attention
}

TensorF32 PairTrack::get_attention_bias(int n_head) const {
    // pair -> (B, L, L, n_head)

    TensorF32 bias;
    return bias;  // 占位 // representation()
}

} // namespace rfaa

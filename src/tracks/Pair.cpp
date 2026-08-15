#include "ppml/Track.h"

namespace ppml {

PairTrack::PairTrack(int seq_len, int dim, Device device)
    : seq_len_(seq_len), dim_(dim) {
    device_ = device;
    repr_ = zeros<float>({1, seq_len, seq_len, dim}, device);
}

void PairTrack::set_embeddings(EmbeddingLayer* left_emb, EmbeddingLayer* right_emb,
                                PositionalEncoding* pos_enc) {
    left_emb_  = left_emb;
    right_emb_ = right_emb;
    pos_enc_   = pos_enc;
}

void PairTrack::init_from_embedding(const TensorF32& left, const TensorF32& right,
                                     const TensorF32& bond_feats, const TensorF32& dist_matrix,
                                     const TensorF32& index) {
    // left: (B, L, D), right: (B, L, D)
    TensorF32 left_emb = left_emb_->forward_exec(left).unsqueeze(1);   // (B, 1, L, D_PAIR)
    TensorF32 right_emb = right_emb_->forward_exec(right).unsqueeze(2); // (B, L, 1, D_PAIR)

    // outer sum
    repr_ = outer_sum(left_emb, right_emb);  // (B, L, L, D_PAIR)
    // PositionalEncoding
    auto pos_out = pos_enc_->forward(repr_, index, bond_feats, dist_matrix, TensorF32()/*same_chain*/);
    repr_.copy_from(*add_impl(&repr_, &pos_out, /*inplace=*/false));
}

// deprecated 
TensorF32 PairTrack::templ_stack(const TensorF32& in_templ, const TensorF32& rbf_feat, const TensorF32& t1d) {
    /* int L = in_templ.shape().dims[2];
    int T = in_templ.shape().dims[1];
    int B = in_templ.shape().dims[0];
    TensorF32 templ = in_templ.view({B*T, L, L, 64});  // (B*T, L, L, 64)

    // tensor copy constructor was deleted i.e. = t1d;
    TensorF32 t1d_reshaped = t1d.view({B*T, L, D_T1D});

    // TODO: LinearLayer 值类型构造已删除，需用 create()
    // LinearLayer* t1d_proj = LinearLayer::create(D_T1D, D_STATE);
    // TensorF32 state_proj = t1d_proj->forward(t1d_reshaped);
    TensorF32 out;
    // TODO: TemplatePairStack 指针化完成前，此函数暂不可用
    // for (int i = 0; i < 2; i++) {
    //     templ = tps_.forward(templ, rbf_feat, state_proj);
    // }
    
    // d_templ = 64
    // TODO: LayerNorm 值类型构造已删除，需用 create()
    // LayerNorm* layernorm = LayerNorm::create(64);
    // out = layernorm->forward(templ);
    out = templ;  // placeholder
    out = out.view({B, T, L, L, 64});
    return out; */
}

void PairTrack::inject_template(const TensorF32& in_templ) {
    // 直接 view 原始 repr_，梯度可反向传播（copy_from 会切断梯度链）
    int B = repr_.shape().dims[0], L = repr_.shape().dims[2];
    auto pair  = repr_.view(Shape{B*L*L, 1, D_PAIR});
    //D_TEMPL = 64
    // (B, T, L, L, d_templ) = (B, T, L, L, 64)
    int T = in_templ.shape().dims[1];
    auto templ = in_templ.permute({0, 2, 3, 1, 4}).view(Shape{B*L*L, 1, 64});
    CrossAttention cross_attn(D_PAIR, 64, 8);
    auto out = cross_attn.forward(pair, templ);  // (B*L*L, 1, D_PAIR)
    out = out.view(Shape{B, L, L, D_PAIR});
    pair = pair.view(Shape{B, L, L, D_PAIR});
    repr_.copy_from(*add_impl(&pair, &out, /*inplace=*/false));  // 注入模板信息，直接更新 repr_
}

/* void PairTrack::update_from_msa(const TensorF32& msa) {
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
} */

} // namespace ppml

#include "ppml/Track.h"

#include "ppml/Embedding.h"

namespace ppml {

StateTrack::StateTrack(int seq_len, int dim, Device device)
    : seq_len_(seq_len), dim_(dim) {
    device_ = device;
    repr_ = zeros<float>({1, seq_len, dim}, device);
}

// deprecated
void StateTrack::init_from_embedding(const TensorF32& seq_tokens) {
    // seq_tokens: (B, L) -> Embedding -> (B, L, D_STATE)
    //int D_STATE = 32;             // State 隐层维度
    // 80 -> 32
    // 80 was the all-atom token types

    //EmbeddingLayer embedding(NAATOKENS, D_STATE);
    //repr_ = embedding.forward(seq_tokens);
}

// deprecated
void StateTrack::inject_template(const TensorF32& t1d, const TensorF32& tor_feat) {
    // 此函数已内联到 PPMLModel::forward() 中
    /*
    LinearLayer emb_t1d(D_T1D + D_TOR, 64);
    LinearLayer proj_t1d(64, 64);
    TensorF32 t1d_tor = concat(t1d, tor_feat, -1);
    TensorF32 t1d_emb = emb_t1d.forward(t1d_tor);
    t1d_emb = proj_t1d.forward(relu(t1d_emb));
    int B = t1d_emb.shape().dims[0], L = t1d_emb.shape().dims[2];
    int T = t1d_emb.shape().dims[1];
    repr_ = repr_.view(Shape{B * L, 1, D_STATE});
    TensorF32 t1d_permuted = t1d_emb.permute({0, 2, 1, 3});
    TensorF32 t1d_reshaped = t1d_permuted.view(Shape{B * L, T, 64});
    CrossAttention cross_attn(D_STATE, 64, 8);
    TensorF32 out = cross_attn.forward(repr_, t1d_reshaped);
    out = out.view(Shape{B, L, 64});
    repr_ = *add_impl(&repr_, &out, false);
    */
}

void StateTrack::rebuild_from_se3(const TensorF32& msa_query, 
                                   const TensorF32& pair,
                                   const TensorF32& coords) {
    // Step 4: str2str
    // SE3 Transformer
    // 完全替换 state
}

TensorF32 StateTrack::get_gate(int gate_dim) const {
    // state outer product -> gate
    TensorF32 gate;
    return gate; 
    //return repr_;  // 占位
}

} // namespace ppml

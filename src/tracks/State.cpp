#include "rfaa/Track.h"

#include "rfaa/Embedding.h"

namespace rfaa {

StateTrack::StateTrack(int seq_len, int dim, Device device)
    : seq_len_(seq_len), dim_(dim) {
    device_ = device;
    repr_ = zeros<float>({1, seq_len, dim}, device);
}

void StateTrack::init_from_embedding(const TensorF32& seq_tokens) {
    // seq_tokens: (B, L) -> Embedding -> (B, L, D_STATE)
    //int D_STATE = 32;             // State 隐层维度
    // 80 -> 32
    // 80 was the all-atom token types

    EmbeddingLayer embedding(NAATOKENS, D_STATE);
    repr_ = embedding.forward(seq_tokens);
}

void StateTrack::inject_template(const TensorF32& t1d, const TensorF32& tor_feat) {
    // Cross Attention
    // state as query, t1d as key/value
    LinearLayer emb_t1d(D_T1D + D_TOR, 64);
    LinearLayer proj_t1d(64, 64);
    // according to the src code: dtempl = 64
    TensorF32 t1d_tor = concat(t1d, tor_feat, /*dim=*/-1);  // (B, T, L, 110)
    TensorF32 t1d_emb = emb_t1d.forward(t1d_tor);  // (B, T, L, 64)
    t1d_emb = proj_t1d.forward(relu(t1d_emb));       
    // (B, T, L, 64)
    // d_q = d_state
    // d_k = d_v = 64 
    // implement a reshape method?
    repr_ = repr_.view({B * L, 1, D_STATE});
    TensorF32 t1d_permuted = t1d_emb.permute({0, 2, 1, 3});
    TensorF32 t1d_reshaped = t1d_permuted.view({B * L, T, 64});
    //CrossAttention cross_attention(D_STATE, 64, 8);
    SelfAttention cross_attention(D_STATE, 64, 8);
    TensorF32 out = cross_attention.forward(repr_, t1d_reshaped, t1d_reshaped);
    // (B, L, 64)
    out = out.view({B, L, 64});
    repr_ = repr_ + out;  // 注入模板信息
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

} // namespace rfaa

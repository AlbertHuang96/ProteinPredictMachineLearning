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

void StateTrack::inject_template(const TensorF32& t1d) {
    // Cross Attention
    // state as query, t1d as key/value
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

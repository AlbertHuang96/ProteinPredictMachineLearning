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
    Embedding emb_left(NAATOKENS, D_PAIR);
    Embedding emb_right(NAATOKENS, D_PAIR);
    TensorF32 left_emb = emb_left.forward(left).unsqueeze(1);   // (B, 1, L, D_PAIR)
    TensorF32 right_emb = emb_right.forward(right).unsqueeze(2); // (B, L, 1, D_PAIR)

    // outer sum
    repr_ = outer_sum(left_emb, right_emb);  // (B, L, L, D_PAIR)
    PositionalEncoding pos_enc(-32, 32, 8, D_PAIR);
    repr_ = repr_ + pos_enc.forward(repr_, index, bond_feats, dist_matrix, TensorF32()/*same_chain*/);  // 加上位置编码
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

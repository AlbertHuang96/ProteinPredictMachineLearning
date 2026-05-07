#include "rfaa/Track.h"

namespace rfaa {

PairTrack::PairTrack(int seq_len, int dim, Device device)
    : seq_len_(seq_len), dim_(dim) {
    device_ = device;
    repr_ = zeros<float>({1, seq_len, seq_len, dim}, device);
}

void PairTrack::init_from_embedding(const TensorF32& left, const TensorF32& right) {
    // left: (B, L, D), right: (B, L, D)
    // pair = outer_product(left, right) -> Linear
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

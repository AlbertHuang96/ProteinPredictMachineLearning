#include "rfaa/Track.h"

namespace rfaa {

MSATrack::MSATrack(int n_seq, int seq_len, int dim, Device device)
    : n_seq_(n_seq), seq_len_(seq_len), dim_(dim) {
    device_ = device;
    repr_ = zeros<float>({1, n_seq, seq_len, dim}, device);
}

void MSATrack::init_from_features(const TensorF32& features) {
    // features: (B, N, L, 164) 或 (B, N, L, 83)
    // 通过 Linear 投影到 dim
    int feat_dim = features.shape().dims.back();
    
    // Linear(feat_dim -> dim)
    // 这里简化，实际应调用 LinearLayer
    //repr_ = features;  // 占位
    repr_.copy_from(features);  // 实际应进行线性变换
}

void MSATrack::update_self(const TensorF32& pair_bias, const TensorF32& state) {
    // Step 1: msa2msa
    // state -> msa[:, 0]
    // Row Attention with pair_bias
    // Col Attention
    // FeedForward
}

TensorF32 MSATrack::query_row() const {
    return repr_.select(1, 0);  // msa[:, 0]
}

} // namespace rfaa

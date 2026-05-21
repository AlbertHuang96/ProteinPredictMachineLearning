#include "rfaa/Track.h"

#include "rfaa/Embedding.h"

namespace rfaa {

MSATrack::MSATrack(int n_seq, int seq_len, int dim, Device device)
    : n_seq_(n_seq), seq_len_(seq_len), dim_(dim) {
    device_ = device;
    repr_ = zeros<float>({1, n_seq, seq_len, dim}, device);
}

// input is msa latent tensor 
void MSATrack::init_from_features(const TensorF32& features) {
    // features: (B, N, L, 164) 或 (B, N, L, 83)
    // 通过 Linear 投影到 dim
    int feat_dim = features.shape().dims.back();
    
    // 164 -> 256 hidden dim
    LinearLayer linear(feat_dim, dim_);
    repr_ = linear.forward(features);

    // Linear(feat_dim -> dim)
    // 这里简化，实际应调用 LinearLayer
    //repr_ = features;  // 占位
    //repr_.copy_from(features);  // 实际应进行线性变换

    // here we only make a linear operation 
    // but notice that
    // the supplemental note of the paper was as followed:
    // seq = linear(seq)
    // msa += seq
    // the source code may be the wrong or another version
    // but the source code was not the same:
    // in the Embeddings.py
    // a learned 2-entry nn.Embedding(2, d_model) is element-wise added to 
    // distinguish the query row (index 0) from all other MSA rows (index 1)
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

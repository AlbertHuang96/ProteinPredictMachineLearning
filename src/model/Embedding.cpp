#include "rfaa/Model.h"
#include "rfaa/Embedding.h"
#include "rfaa/MathUtils.h"
#include <random>

namespace rfaa {

// Embedding 层实现

class EmbeddingLayer {
public:
// num_embeddings: 词汇表大小 (NAATOKENS)
// embedding_dim: 嵌入向量维度 (D_STATE)
    EmbeddingLayer(int num_embeddings, int embedding_dim)
        : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
        // Xavier 初始化
        weights_ = zeros<float>({num_embeddings, embedding_dim}, Device::CPU);
        std::random_device rd;
        std::mt19937 gen(rd());
        float scale = sqrtf(2.0f / (num_embeddings + embedding_dim));
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (int i = 0; i < num_embeddings * embedding_dim; ++i) {
            weights_.data()[i] = dist(gen);
        }
    }
    
    // forward
    // backward
    TensorF32 forward(const TensorF32& indices) {
        // indices: (B, L) 整数索引
        // output: (B, L, D)
        int B = indices.shape().dims[0];
        int L = indices.shape().dims[1];
        
        // B for batch
        // L for sequence length
        // D = embedding_dim_ 嵌入向量的维度
        // 区分 nunm_embeddings_ = vocab size 词汇表大小
        TensorF32 output({B, L, embedding_dim_}, indices.device());
        
        // CPU 实现
        if (indices.device() == Device::CPU) {
            for (int b = 0; b < B; ++b) {
                for (int l = 0; l < L; ++l) {
                    int idx = static_cast<int>(indices.data()[b * L + l]);
                    // idx range [0, V - 1]
                    idx = std::max(0, std::min(idx, num_embeddings_ - 1));
                    // [(b * L + l) * D] indexing into a D-dim vector
                    // problem: output = wte + wpe
                    // wpe = position embedding, wte = token embedding
                    // implement the position emb outside of this function
                    std::memcpy(
                        output.data() + (b * L + l) * embedding_dim_,
                        weights_.data() + idx * embedding_dim_,
                        embedding_dim_ * sizeof(float)
                    );
                }
            }
        }
        // CUDA 实现应调用 kernel
        
        return output;
    }
    
private:
    int num_embeddings_;
    int embedding_dim_;
    TensorF32 weights_;
};

class LinearLayer {
public:
    LinearLayer(int in_features, int out_features, bool bias = true)
        : in_features_(in_features), out_features_(out_features), has_bias_(bias) {
        weight_ = zeros<float>({out_features, in_features}, Device::CPU);
        if (bias) {
            bias_ = zeros<float>({out_features}, Device::CPU);
        }
        
        // 初始化
        std::random_device rd;
        std::mt19937 gen(rd());
        float scale = sqrtf(2.0f / in_features);
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (int i = 0; i < out_features * in_features; ++i) {
            weight_.data()[i] = dist(gen);
        }
    }

    void zeros_weight() {
        std::memset(weight_.data(), 0, weight_.shape().numel() * sizeof(float));
    }

    void ones_bias() {
        std::memset(bias_.data(), 1, bias_.shape().numel() * sizeof(float));
    }
    
    
    TensorF32 forward(const TensorF32& x) {
        // x: (..., in_features)
        // output: (..., out_features)
        
        int batch = x.shape().numel() / in_features_;
        TensorF32 output(Shape({batch, out_features_}), x.device());
        
        // 简化的矩阵乘法 (实际应使用 cuBLAS)
        for (int b = 0; b < batch; ++b) {
            for (int o = 0; o < out_features_; ++o) {
                float sum = has_bias_ ? bias_.data()[o] : 0.0f;
                // dim of weights: (out_features, in_features)
                for (int i = 0; i < in_features_; ++i) {
                    sum += x.data()[b * in_features_ + i] * weight_.data()[o * in_features_ + i];
                }
                // dim of output: (batch, out_features)
                output.data()[b * out_features_ + o] = sum;
            }
        }
        
        // 恢复原始形状 (最后维变为 out_features)
        auto out_shape = x.shape();
        out_shape.dims.back() = out_features_;
        return output.view(out_shape);
    }
    
private:
    int in_features_, out_features_;
    bool has_bias_;
    TensorF32 weight_;
    TensorF32 bias_;
};

// LayerNorm
class LayerNorm {
public:
    LayerNorm(int normalized_shape, float eps = 1e-5)
        : normalized_shape_(normalized_shape), eps_(eps) {
        gamma_ = zeros<float>({normalized_shape}, Device::CPU);
        beta_ = zeros<float>({normalized_shape}, Device::CPU);
        
        // 初始化为 gamma=1, beta=0
        for (int i = 0; i < normalized_shape; ++i) {
            gamma_.data()[i] = 1.0f;
        }
    }
    
    TensorF32 forward(const TensorF32& x) {
        // 在最后一个维度上做 LayerNorm
        int batch = x.shape().numel() / normalized_shape_;
        TensorF32 output(x.shape(), x.device());
        
        for (int b = 0; b < batch; ++b) {
            // 计算均值
            float mean = 0.0f;
            for (int i = 0; i < normalized_shape_; ++i) {
                mean += x.data()[b * normalized_shape_ + i];
            }
            mean /= normalized_shape_;
            
            // 计算方差
            float var = 0.0f;
            for (int i = 0; i < normalized_shape_; ++i) {
                float diff = x.data()[b * normalized_shape_ + i] - mean;
                var += diff * diff;
            }
            var /= normalized_shape_;
            
            // 归一化
            float inv_std = 1.0f / sqrtf(var + eps_);
            for (int i = 0; i < normalized_shape_; ++i) {
                float normalized = (x.data()[b * normalized_shape_ + i] - mean) * inv_std;
                output.data()[b * normalized_shape_ + i] = 
                    normalized * gamma_.data()[i] + beta_.data()[i];
            }
        }
        
        return output;
    }
    
private:
    int normalized_shape_;
    float eps_;
    TensorF32 gamma_;
    TensorF32 beta_;
};

class BondEmbedding {
private:
    LinearLayer emb_;
    
    int d_init_;
    int d_pair_;

    const int NBYTES = 8;

public:
    BondEmbedding(int d_init, int d_pair)
        : d_init_(d_init), d_pair_(d_pair) {
            if (d_init_ == 0) {
                d_init_ = NBYTES;
            }
            emb_ = LinearLayer(d_init_, d_pair_);
            
        }

        //ChemData().NBTYPES represents the number of categorical bond types the model recognizes, 
        //and its value is 8
        TensorF32 forward(const TensorF32& bond_feats) {
            // bond_feats: (B, L, L, d_init)
            // output: (B, L, L, d_pair)
            int B = bond_feats.shape().dims[0];
            int L = bond_feats.shape().dims[1];
            
            TensorF32 output;
            TensorF32 one_hot = one_hot(bond_feats, NBYTES); // (B, L, L, d_init)
            output = emb_.forward(one_hot); // (B, L, L, d_pair)
            
            return output;
        }
};

class FullEmbedding {
private:
    LinearLayer emb_;
    EmbeddingLayer emb_q_;
    int d_init_;
    int d_msa_;
public:
    FullEmbedding(int d_init, int d_msa) : d_init_(d_init), d_msa_(d_msa) {
        if (d_init_ == 0) {
            d_init_ = NAATOKENS - 1 + 4;
        }
        emb_ = LinearLayer(d_init_, d_msa_);
        emb_q_ = EmbeddingLayer(NAATOKENS, d_msa_);
    }
    
    TensorF32 forward(const TensorF32& msa, const TensorF32& seq, const TensorF32& idx) {
        // msa: (B, N, L, d_init)
        // seq: (B, L)
        // idx: (B, L)
        
        int B = msa.shape().dims[0];
        int N = msa.shape().dims[1];
        int L = msa.shape().dims[2];
        
        // MSA embedding
        TensorF32 msa_emb = emb_.forward(msa);
        
        // Query embedding
        TensorF32 seq_emb = emb_q_.forward(seq).unsqueeze(1); // (B, 1, L, d_msa_)
        // unsequeeze
        //unsqueeze(1) → 在维度1插入大小为1的维度
        // expand:
        //seq.expand(-1, N, -1, -1)：将 seq 在第1维扩展到 N
        //seq 形状：(B, 1, L, d_model)
        //-1 表示"保持该维度不变"
        //扩展后：(B, N, L, d_model) — 把同一个序列嵌入复制 N 份
        
        // Add query embedding to MSA embedding
        TensorF32 output({B, N, L, d_msa_}, msa.device());
        for (int b = 0; b < B; ++b) {
            for (int n = 0; n < N; ++n) {
                for (int l = 0; l < L; ++l) {
                    for (int d = 0; d < d_msa_; ++d) {
                        output.data()[((b * N + n) * L + l) * d_msa_ + d] =
                            msa_emb.data()[((b * N + n) * L + l) * d_msa_ + d] +
                            seq_emb.data()[(b * L + l) * d_msa_ + d];
                    }
                }
            }
        }
        
        return output;
    }
/* if d_init==0:
            d_init=ChemData().NAATOKENS-1+4
        self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
        self.emb_q = nn.Embedding(ChemData().NAATOKENS, d_msa) # embedding for query sequence */
/* def forward(self, msa, seq, idx):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L)
        #   - idx: Residue index
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        N = msa.shape[1] # number of sequenes in MSA
        msa = self.emb(msa) # (B, N, L, d_model) # MSA embedding
        seq = self.emb_q(seq).unsqueeze(1) # (B, 1, L, d_model) -- query embedding
        msa = msa + seq.expand(-1, N, -1, -1) # adding query embedding to MSA
        #return self.drop(msa)
        return (msa) */
};

} // namespace rfaa

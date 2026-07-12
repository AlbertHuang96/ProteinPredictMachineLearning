//#include "rfaa/Tensor.h"

#pragma once
#include "Tensor.h"


#include "rfaa/Model.h"

#include "rfaa/MathUtils.h"
#include <random>

#include "rfaa/ComputeGraph.h"
#include "rfaa/Context.h"

namespace rfaa {

// Embedding 层实现

class EmbeddingLayer {
public:
// num_embeddings: 词汇表大小 (NAATOKENS)
// embedding_dim: 嵌入向量维度 (D_STATE)
    /* EmbeddingLayer(int num_embeddings, int embedding_dim)
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
    } */

    EmbeddingLayer() = default;

    static EmbeddingLayer* create(int num_embeddings, int embedding_dim);
    // ===== 图模式：get_rows =====
    TensorF32* forward_graph(TensorF32* indices);
        // embedding 本质是 get_rows(weight, indices)
        //return get_rows(weights_, indices);
    
    // TODO
    // all the forward_exec need to be removed
    // forward
    // backward
    TensorF32 forward_exec(const TensorF32& indices);
        
        // B for batch
        // L for sequence length
        // D = embedding_dim_ 嵌入向量的维度
        // 区分 nunm_embeddings_ = vocab size 词汇表大小
    
    TensorF32* weight() { return weights_; }
    
private:
    int num_embeddings_;
    int embedding_dim_;
    TensorF32* weights_ = nullptr;
    //TensorF32 weights_;
};

class LinearLayer {
public:
    /* LinearLayer(int in_features, int out_features, bool bias = true)
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
    } */

    LinearLayer() = default;

    // 工厂函数：从 context 分配权重
    static LinearLayer* create(int in_features, int out_features, bool bias = true);

    void zeros_weight() {
        std::memset(weight_->data(), 0, weight_->shape().numel() * sizeof(float));
    }

    void ones_bias() {
        std::memset(bias_->data(), 1, bias_->shape().numel() * sizeof(float));
    }
    
    // ===== 图模式前向（训练用）=====
    // x: 输入 Tensor* (图节点), 返回输出 Tensor* (图节点)
    TensorF32* forward_graph(TensorF32* x);
    
    TensorF32 forward(const TensorF32& x);

    // 获取权重指针（加载/保存用）
    TensorF32* weight() { return weight_; }
    TensorF32* bias()   { return bias_; }
    
private:
    void init_weights();
        
    
    int in_features_, out_features_;
    bool has_bias_;
    TensorF32* weight_ = nullptr;  // ← 改为指针, Context 管理
    TensorF32* bias_   = nullptr;
    //TensorF32 weight_;
    //TensorF32 bias_;
};

// LayerNorm
class LayerNorm {
public:
    /* LayerNorm(int normalized_shape, float eps = 1e-5)
        : normalized_shape_(normalized_shape), eps_(eps) {
        gamma_ = zeros<float>({normalized_shape}, Device::CPU);
        beta_ = zeros<float>({normalized_shape}, Device::CPU);
        
        // 初始化为 gamma=1, beta=0
        for (int i = 0; i < normalized_shape; ++i) {
            gamma_.data()[i] = 1.0f;
        }
    } */

    LayerNorm() = default;

    static LayerNorm* create(int normalized_shape, float eps = 1e-5);

    TensorF32* forward(TensorF32* x);
    
    TensorF32 forward_exec(const TensorF32& x);
            
    
private:
    int normalized_shape_;
    float eps_;
    TensorF32* gamma_ = nullptr;
    TensorF32* beta_  = nullptr;
    //TensorF32 gamma_;
    //TensorF32 beta_;
};

class BondEmbedding {
public:
    // 旧值类型构造函数 (保留注释):
    // BondEmbedding(int d_init, int d_pair);
    BondEmbedding() = default;

    // 由 RFAAModel 注入已创建的参数指针
    void set_params(LinearLayer* emb, int d_pair);
        
    // ChemData().NBTYPES represents the number of categorical bond types the model recognizes, 
    // and its value is 8
    TensorF32 forward(const TensorF32& bond_feats);

private:
    // 旧值类型 (保留注释):
    // LinearLayer emb_;
    LinearLayer* emb_ = nullptr;   // NBYTES (8) → d_pair_

    int d_pair_ = 0;
    static constexpr int NBYTES = 8;
};

class FullEmbedding {
public:
    // 旧值类型构造函数 (保留注释):
    // FullEmbedding(int d_init, int d_msa);
    FullEmbedding() = default;

    // 由 RFAAModel 注入已创建的参数指针
    void set_params(LinearLayer* emb, EmbeddingLayer* emb_q, int d_msa);
    
    TensorF32 forward(const TensorF32& msa, const TensorF32& seq, const TensorF32& idx);
        
    // Query embedd
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

private:
    // 旧值类型 (保留注释):
    // LinearLayer emb_;
    // EmbeddingLayer emb_q_;
    LinearLayer*    emb_   = nullptr; // (NAATOKENS-1+4=83) → d_msa_
    EmbeddingLayer* emb_q_ = nullptr; // (NAATOKENS, d_msa_)

    int d_msa_ = 0;
};

} // namespace rfaa


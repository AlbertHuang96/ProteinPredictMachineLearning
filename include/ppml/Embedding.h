//#include "ppml/Tensor.h"

#pragma once
#include "Tensor.h"


//#include "ppml/Model.h"

#include "ppml/MathUtils.h"
#include <random>

#include "ppml/ComputeGraph.h"
#include "ppml/Context.h"

namespace ppml {

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
    // se3=true：权重/bias 打 TENSOR_FLAG_SE3（AdamW 分层小 lr，缓解 SE3 梯度尺度不匹配）。
    static LinearLayer* create(int in_features, int out_features, bool bias = true, bool se3 = false);

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

    // ===== LoRA 低秩微调 =====
    // 冻结主权重（去 PARAM 标，断梯度；权值保留参与前向）
    void freeze() {
        if (weight_) weight_->flag &= ~TENSOR_FLAG_PARAM;
        if (bias_)   bias_->flag   &= ~TENSOR_FLAG_PARAM;
    }
    // 启用 LoRA：分配旁路 A/B（A 存 [in,rank]，B 存 [rank,out]，与 weight_ 同布局），
    // A~N(0,0.02)、B=0（初始旁路输出=0，不破坏预训练权重）。默认同时冻结主权重。
    void enable_lora(int rank, float alpha);
    // LoRA 旁路访问（save/load/调试用）
    TensorF32* lora_A()  { return lora_A_; }
    TensorF32* lora_B()  { return lora_B_; }
    int   lora_rank()  const { return lora_rank_; }
    float lora_alpha() const { return lora_alpha_; }

private:
    void init_weights();
    void lora_init();   // A~N(0,σ), B=0

    int in_features_, out_features_;
    bool has_bias_;
    TensorF32* weight_ = nullptr;  // ← 改为指针, Context 管理
    TensorF32* bias_   = nullptr;
    // LoRA 旁路：主权重冻结，仅训 A/B
    TensorF32* lora_A_  = nullptr;   // dims=[in, rank]  （mul_mat 的 b 作转置，K=in 最内）
    TensorF32* lora_B_  = nullptr;   // dims=[rank, out] （mul_mat 的 b 作转置，K=rank 最内）
    int   lora_rank_  = 0;
    float lora_alpha_ = 0.f;
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
    
    // 获取参数指针（加载/保存/迁移用）
    TensorF32* gamma() { return gamma_; }
    TensorF32* beta()  { return beta_; }
    
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

    // 由 PPMLModel 注入已创建的参数指针
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

    // 由 PPMLModel 注入已创建的参数指针
    void set_params(LinearLayer* emb, EmbeddingLayer* emb_q, int d_msa);
    
    TensorF32 forward(const TensorF32& msa, const TensorF32& seq, const TensorF32& idx);

    // ===== 图模式前向（训练用）=====
    // 输入输出均为图节点指针：msa_emb + broadcast(query_seq_emb)
    // 返回 msa embedding (B,N,L,d_msa)
    TensorF32* forward_graph(TensorF32* msa, TensorF32* seq, TensorF32* idx);
        
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

} // namespace ppml


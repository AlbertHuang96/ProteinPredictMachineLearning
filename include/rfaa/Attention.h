#pragma once

#include "Tensor.h"

namespace rfaa {

// Attention 配置
struct AttnConfig {
    int dim;           // 输入维度
    int n_head;        // head 数
    int head_dim;      // 每个 head 维度 (dim / n_head)
    float dropout;     // dropout 率
    bool use_bias;     // 是否使用 bias
    
    AttnConfig(int d, int h, float drop = 0.0f, bool bias = true)
        : dim(d), n_head(h), head_dim(d / h), dropout(drop), use_bias(bias) {
        if (dim % n_head != 0) {
            throw RFAAError("dim must be divisible by n_head");
        }
    }
};

// 基础 Self-Attention
class SelfAttention {
public:
    explicit SelfAttention(const AttnConfig& config);
    ~SelfAttention();
    
    // Q, K, V 来自同一输入
    TensorF32 forward(const TensorF32& x, const TensorF32* bias = nullptr);
    
    // 加载权重
    void load_weights(const std::string& prefix);
    
private:
    AttnConfig config_;
    //struct Impl;
    //std::unique_ptr<Impl> impl_;
};

// 带 Pair Bias 的 MSA Row Attention
class MSARowAttention {
public:
    explicit MSARowAttention(const AttnConfig& config);
    
    // msa: (B, N, L, D), pair: (B, L, L, n_head)
    TensorF32 forward(const TensorF32& msa, const TensorF32& pair);
    
private:
    LinearLayer to_b(D_PAIR, N_HEAD);  // 将 pair 转换为 attention bias
    LinearLayer to_g(D_MSA, N_HEAD * D_MSA);
    LinearLayer to_out(N_HEAD * D_MSA, D_MSA);
    
    LinearLayer Wq(D_MSA, N_HEAD * D_MSA); 
    // n_head 个 head，每个 head D_MSA/n_head 维
    LinearLayer Wk(D_MSA, N_HEAD * D_MSA);
    LinearLayer Wv(D_MSA, N_HEAD * D_MSA);

    AttnConfig config_;
    //struct Impl;
    //std::unique_ptr<Impl> impl_;
};

// MSA Column Attention (在 N 维度)
class MSAColAttention {
public:
    explicit MSAColAttention(const AttnConfig& config);
    
    // msa: (B, N, L, D)
    TensorF32 forward(const TensorF32& msa);
    
private:
    AttnConfig config_;
    //struct Impl;
    //std::unique_ptr<Impl> impl_;
};

// Cross Attention (State ←→ Template)
class CrossAttention {
public:
    CrossAttention(int q_dim, int kv_dim, int n_head);
    
    // query: (B, L, q_dim), kv: (B, T, L, kv_dim)
    TensorF32 forward(const TensorF32& query, const TensorF32& kv);
    
private:
    int q_dim_, kv_dim_, n_head_;
    //struct Impl;
    //std::unique_ptr<Impl> impl_;
};

// Triangle Multiplication (Outgoing / Incoming)
class TriangleMultiplication {
public:
    enum class Direction { Outgoing, Incoming };
    
    TriangleMultiplication(int dim, Direction dir);
    
    // pair: (B, L, L, D)
    TensorF32 forward(const TensorF32& pair);
    
private:
    int dim_;
    Direction dir_;
    //struct Impl;
    //std::unique_ptr<Impl> impl_;
};

// FeedForward
class FeedForward {
public:
    FeedForward(int dim, int hidden_dim, float dropout = 0.0f);
    
    TensorF32 forward(const TensorF32& x);
    
private:
    int dim_, hidden_dim_;
    //struct Impl;
    //std::unique_ptr<Impl> impl_;
};

} // namespace rfaa

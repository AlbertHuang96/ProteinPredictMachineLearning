#pragma once

#include "Tensor.h"
#include "ComputeGraph.h"

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
    TensorF32 forward(const TensorF32& Q, const TensorF32& K, const TensorF32& V, const TensorF32* bias = nullptr);
    
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
    TensorF32 forward(const TensorF32& msa, const TensorF32& pair_biased);
    
private:
    SelfAttention self_attn_;
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
    SelfAttention self_attn_;
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

class MSAGlobalColAttention : public MSAColAttention {
public:    explicit MSAGlobalColAttention(const AttnConfig& config) : MSAColAttention(config);

    TensorF32 forward(const TensorF32& msa);
};

class PairRowAttention {
public:
    explicit PairRowAttention(const AttnConfig& config);
    
    //
    TensorF32 forward(const TensorF32& pair, const TensorF32& str_bias);
    
private:
    SelfAttention self_attn_;
    LinearLayer to_b(D_PAIR, N_HEAD);  // 将 str_bias 转换为 attention bias
    LinearLayer to_g(D_PAIR, N_HEAD * D_PAIR_HIDDEN);
    LinearLayer to_out(N_HEAD * D_PAIR_HIDDEN, D_PAIR);
    
    LinearLayer Wq(D_PAIR, N_HEAD * D_PAIR_HIDDEN); 
    // n_head 个 head，每个 head D_MSA/n_head 维
    LinearLayer Wk(D_PAIR, N_HEAD * D_PAIR_HIDDEN);
    LinearLayer Wv(D_PAIR, N_HEAD * D_PAIR_HIDDEN);

    AttnConfig config_;
    //struct Impl;
    //std::unique_ptr<Impl> impl_;
};


class PairColAttention {
public:
    explicit PairColAttention(const AttnConfig& config);
    
    //
    TensorF32 forward(const TensorF32& pair, const TensorF32& str_bias);
    
private:
    SelfAttention self_attn_;
    LinearLayer to_b(D_PAIR, N_HEAD);  // 将 str_bias 转换为 attention bias
    LinearLayer to_g(D_PAIR, N_HEAD * D_PAIR_HIDDEN);
    LinearLayer to_out(N_HEAD * D_PAIR_HIDDEN, D_PAIR);
    
    LinearLayer Wq(D_PAIR, N_HEAD * D_PAIR_HIDDEN); 
    // n_head 个 head，每个 head D_MSA/n_head 维
    LinearLayer Wk(D_PAIR, N_HEAD * D_PAIR_HIDDEN);
    LinearLayer Wv(D_PAIR, N_HEAD * D_PAIR_HIDDEN);

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
    //enum class Direction { Outgoing, Incoming };
    
    TriangleMultiplication(int dim);
    
    // pair: (B, L, L, D)
    TensorF32 forward(const TensorF32& pair, bool bOutgoing = true);
    
private:
    static constexpr int D_HIDDEN_TRIMUL = 128;
    LayerNorm layernorm_(D_PAIR);
    LinearLayer left_proj_(D_PAIR, D_HIDDEN_TRIMUL);
    LinearLayer right_proj_(D_PAIR, D_HIDDEN_TRIMUL);
    LinearLayer left_gate_(D_PAIR, D_HIDDEN_TRIMUL);
    LinearLayer right_gate_(D_PAIR, D_HIDDEN_TRIMUL);
    LinearLayer gate_(D_PAIR, D_PAIR);
    LayerNorm output_layernorm_(D_HIDDEN_TRIMUL);
    LinearLayer out_proj_(D_HIDDEN_TRIMUL, D_PAIR);
     //struct Impl;
     //std::unique_ptr<Impl> impl_;
    int dim_;
    //Direction dir_;
    //struct Impl;
    //std::unique_ptr<Impl> impl_;
};

// FeedForward
class FeedForward {
public:
    FeedForward(int dim, int hidden_dim, float dropout = 0.1f);
    
    TensorF32 forward(const TensorF32& x);
    
private:
    int dim_, hidden_dim_;
    LayerNorm layernorm_(dim_);
    LinearLayer linear1_(dim_, dim_ * hidden_dim_); 
    // linear1 kaiming normal initialization
    LinearLayer linear2_(dim_ * hidden_dim_, dim_); 
    // linear2_  zero initialization
    Dropout dropout_(dropout_);
    //struct Impl;
    //std::unique_ptr<Impl> impl_;
};

class TemplatePairStack {
public:
    TemplatePairStack();
    
    TensorF32 forward(const TensorF32& pair, TensorF32& rbf_feature, const TensorF32& state);
    
private:
    LinearLayer rbf_proj_(D_RBF, D_PAIR);

    LayerNorm state_norm_(D_STATE);
    
    LinearLayer left_proj_(D_STATE, 16);
    LinearLayer right_proj_(D_STATE, 16);
    LinearLayer gate_proj_(16 * 16, D_PAIR);

    TriangleMultiplication tri_mul_out_;
    TriangleMultiplication tri_mul_in_;

    Dropout drop_row_(1, 0.15);
    Dropout drop_col_(2, 0.15);
    PairRowAttention pair_row_attn_;
    PairColAttention pair_col_attn_;
            // FeedForward
    FeedForward pair_ff_(D_PAIR, 2);

}

} // namespace rfaa

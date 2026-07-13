#pragma once

#include "Core.h"
#include "Tensor.h"
#include "ComputeGraph.h"
#include "Embedding.h"

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
    MSARowAttention() = default;
    void set_params(const AttnConfig& config,
                    LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                    LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv);
    
    TensorF32 forward(const TensorF32& msa, const TensorF32& pair_biased);
    
private:
    SelfAttention self_attn_;

    LinearLayer* to_b_  = nullptr; // D_PAIR (128) → N_HEAD (8)
    LinearLayer* to_g_  = nullptr; // D_MSA (256)  → N_HEAD*D_MSA (2048)
    LinearLayer* to_out_ = nullptr; // N_HEAD*D_MSA (2048) → D_MSA (256)
    LinearLayer* Wq_    = nullptr; // D_MSA (256)  → N_HEAD*D_MSA (2048)
    LinearLayer* Wk_    = nullptr; // D_MSA (256)  → N_HEAD*D_MSA (2048)
    LinearLayer* Wv_    = nullptr; // D_MSA (256)  → N_HEAD*D_MSA (2048)
    AttnConfig config_;
};

// MSA Column Attention
class MSAColAttention {
public:
    MSAColAttention() = default;
    void set_params(const AttnConfig& config,
                    LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                    LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv);
    
    // msa 应为已过 layernorm 的输入
    TensorF32 forward(const TensorF32& msa);
    
protected:
    SelfAttention self_attn_;
    LinearLayer* to_b_  = nullptr; // D_PAIR (128) → N_HEAD (8)
    LinearLayer* to_g_  = nullptr; // D_MSA (256)  → N_HEAD*D_MSA (2048)
    LinearLayer* to_out_ = nullptr; // N_HEAD*D_MSA (2048) → D_MSA (256)
    LinearLayer* Wq_    = nullptr; // D_MSA (256)  → N_HEAD*D_MSA (2048)
    LinearLayer* Wk_    = nullptr; // D_MSA (256)  → N_HEAD*D_MSA (2048)
    LinearLayer* Wv_    = nullptr; // D_MSA (256)  → N_HEAD*D_MSA (2048)
    AttnConfig config_;
};

class MSAGlobalColAttention : public MSAColAttention {
public:
    MSAGlobalColAttention() = default;
    // 继承 MSAColAttention::set_params()

    // msa 应为已过 layernorm 的输入
    TensorF32 forward(const TensorF32& msa);
};

class PairRowAttention {
public:
    PairRowAttention() = default;
    void set_params(const AttnConfig& config,
                    LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                    LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv);
    
    // pair/str_bias 应为已过 layernorm 的输入
    TensorF32 forward(const TensorF32& pair, const TensorF32& str_bias);
    
private:
    SelfAttention self_attn_;
    LinearLayer* to_b_  = nullptr; // D_PAIR (128) → N_HEAD (8)
    LinearLayer* to_g_  = nullptr; // D_PAIR (128) → N_HEAD*D_PAIR_HIDDEN (256)
    LinearLayer* to_out_ = nullptr; // N_HEAD*D_PAIR_HIDDEN (256) → D_PAIR (128)
    LinearLayer* Wq_    = nullptr; // D_PAIR (128) → N_HEAD*D_PAIR_HIDDEN (256)
    LinearLayer* Wk_    = nullptr; // D_PAIR (128) → N_HEAD*D_PAIR_HIDDEN (256)
    LinearLayer* Wv_    = nullptr; // D_PAIR (128) → N_HEAD*D_PAIR_HIDDEN (256)
    AttnConfig config_;
};


class PairColAttention {
public:
    PairColAttention() = default;
    void set_params(const AttnConfig& config,
                    LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                    LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv);
    
    // pair/str_bias 应为已过 layernorm 的输入
    TensorF32 forward(const TensorF32& pair, const TensorF32& str_bias);
    
private:
    SelfAttention self_attn_;
    LinearLayer* to_b_  = nullptr; // D_PAIR (128) → N_HEAD (8)
    LinearLayer* to_g_  = nullptr; // D_PAIR (128) → N_HEAD*D_PAIR_HIDDEN (256)
    LinearLayer* to_out_ = nullptr; // N_HEAD*D_PAIR_HIDDEN (256) → D_PAIR (128)
    LinearLayer* Wq_    = nullptr; // D_PAIR (128) → N_HEAD*D_PAIR_HIDDEN (256)
    LinearLayer* Wk_    = nullptr; // D_PAIR (128) → N_HEAD*D_PAIR_HIDDEN (256)
    LinearLayer* Wv_    = nullptr; // D_PAIR (128) → N_HEAD*D_PAIR_HIDDEN (256)
    AttnConfig config_;
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
    
    // 旧值类型构造函数 (保留注释):
    // TriangleMultiplication(int dim);
    TriangleMultiplication() = default;
    void set_params(int dim,
                    LayerNorm*   layernorm,     LinearLayer* left_proj,
                    LinearLayer* right_proj,    LinearLayer* left_gate,
                    LinearLayer* right_gate,    LinearLayer* gate,
                    LayerNorm*   output_layernorm, LinearLayer* out_proj);
    
    // pair: (B, L, L, D)
    TensorF32 forward(const TensorF32& pair, bool bOutgoing = true);
    
private:
    static constexpr int D_HIDDEN_TRIMUL = 128;

    // 旧值类型 (保留注释):
    // LayerNorm layernorm_(D_PAIR);
    // LinearLayer left_proj_(D_PAIR, D_HIDDEN_TRIMUL);
    // LinearLayer right_proj_(D_PAIR, D_HIDDEN_TRIMUL);
    // LinearLayer left_gate_(D_PAIR, D_HIDDEN_TRIMUL);
    // LinearLayer right_gate_(D_PAIR, D_HIDDEN_TRIMUL);
    // LinearLayer gate_(D_PAIR, D_PAIR);
    // LayerNorm output_layernorm_(D_HIDDEN_TRIMUL);
    // LinearLayer out_proj_(D_HIDDEN_TRIMUL, D_PAIR);

    LayerNorm*   layernorm_        = nullptr; // D_PAIR (128)
    LinearLayer* left_proj_        = nullptr; // D_PAIR (128) → D_HIDDEN_TRIMUL (128)
    LinearLayer* right_proj_       = nullptr; // D_PAIR (128) → D_HIDDEN_TRIMUL (128)
    LinearLayer* left_gate_        = nullptr; // D_PAIR (128) → D_HIDDEN_TRIMUL (128)
    LinearLayer* right_gate_       = nullptr; // D_PAIR (128) → D_HIDDEN_TRIMUL (128)
    LinearLayer* gate_             = nullptr; // D_PAIR (128) → D_PAIR (128)
    LayerNorm*   output_layernorm_ = nullptr; // D_HIDDEN_TRIMUL (128)
    LinearLayer* out_proj_         = nullptr; // D_HIDDEN_TRIMUL (128) → D_PAIR (128)

    int dim_ = 0;
    //Direction dir_;
    //struct Impl;
    //std::unique_ptr<Impl> impl_;
};

// FeedForward
class FeedForward {
public:
    // 旧值类型构造函数 (保留注释):
    // FeedForward(int dim, int hidden_dim, float dropout = 0.1f);
    FeedForward() = default;
    void set_params(int dim, int hidden_dim, float dropout,
                    LayerNorm* layernorm, LinearLayer* linear1, LinearLayer* linear2);
    
    TensorF32 forward(const TensorF32& x);
    
private:
    int dim_ = 0, hidden_dim_ = 0;
    float dropout_rate_ = 0.1f;

    // 旧值类型 (保留注释):
    // LayerNorm layernorm_(dim_);
    // LinearLayer linear1_(dim_, dim_ * hidden_dim_);  // kaiming normal init
    // LinearLayer linear2_(dim_ * hidden_dim_, dim_);  // zero init

    LayerNorm*   layernorm_ = nullptr; // dim_
    LinearLayer* linear1_   = nullptr; // dim_ → dim_*hidden_dim_
    LinearLayer* linear2_   = nullptr; // dim_*hidden_dim_ → dim_

    Dropout dropout_;
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

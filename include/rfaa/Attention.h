#pragma once

#include "Core.h"
#include "Tensor.h"
#include "ComputeGraph.h"
#include "Dropout.h"

namespace rfaa {
class LinearLayer;
class LayerNorm;
}

namespace rfaa {

// Attention 配置
struct AttnConfig {
    int dim;           // 输入维度
    int n_head;        // head 数
    int head_dim;      // 每个 head 维度 (dim / n_head)
    float dropout;     // dropout 率
    bool use_bias;     // 是否使用 bias
    
    AttnConfig() = default;
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

    // ===== 图模式前向（训练用）=====
    // 输入/输出均为图节点指针，无 copy_from 值拷贝，图节点沿前向一路传播
    TensorF32* forward_graph(TensorF32* Q, TensorF32* K, TensorF32* V, TensorF32* bias = nullptr);

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

    // ===== 图模式前向（训练用）=====
    // 输入/输出均为图节点指针 (ggml 布局 dims[0]=最内维)
    // msa: 值 (B,N,L,D_MSA) = 图 [D_MSA, L, N, B]
    // pair_biased: 值 (B,L,L,D_PAIR) = 图 [D_PAIR, L, L, B]
    // 返回: 值 (B,N,L,D_MSA) = 图 [D_MSA, L, N, B]
    TensorF32* forward_graph(TensorF32* msa, TensorF32* pair_biased);

private:
    std::unique_ptr<SelfAttention> self_attn_;

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

    // ===== 图模式前向（训练用）=====
    // 输入/输出均为图节点指针 (ggml 布局 dims[0]=最内维)
    // msa: 值 (B,N,L,D_MSA) = 图 [D_MSA, L, N, B]
    // 返回: 值 (B,N,L,D_MSA) = 图 [D_MSA, L, N, B]
    TensorF32* forward_graph(TensorF32* msa);

protected:
    std::unique_ptr<SelfAttention> self_attn_;
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

    // ===== 图模式前向（训练用）=====
    // 输入/输出均为图节点指针 (ggml 布局 dims[0]=最内维)
    // msa: 值 (B,N,L,D_MSA) = 图 [D_MSA, L, N, B]
    // 返回: 值 (B,N,L,D_MSA) = 图 [D_MSA, L, N, B]
    TensorF32* forward_graph(TensorF32* msa);
};

class PairRowAttention {
public:
    PairRowAttention() = default;
    void set_params(const AttnConfig& config,
                    LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                    LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv);
    
    // pair/str_bias 应为已过 layernorm 的输入
    TensorF32 forward(const TensorF32& pair, const TensorF32& str_bias);

    // ===== 图模式前向（训练用）=====
    // 输入/输出均为图节点指针 (ggml 布局 dims[0]=最内维)
    // pair: 值 (B,Lr,Lc,D_PAIR) = 图 [D_PAIR, Lc, Lr, B]
    // str_bias: 值 (B,Lr,Lc,D_PAIR) = 图 [D_PAIR, Lc, Lr, B]
    // 返回: 值 (B,Lr,Lc,D_PAIR) = 图 [D_PAIR, Lc, Lr, B]
    TensorF32* forward_graph(TensorF32* pair, TensorF32* str_bias);

private:
    std::unique_ptr<SelfAttention> self_attn_;
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

    // ===== 图模式前向（训练用）=====
    // 输入/输出均为图节点指针 (ggml 布局 dims[0]=最内维)
    // pair: 值 (B,Lr,Lc,D_PAIR) = 图 [D_PAIR, Lc, Lr, B]
    // str_bias: 值 (B,Lr,Lc,D_PAIR) = 图 [D_PAIR, Lc, Lr, B]
    // 返回: 值 (B,Lr,Lc,D_PAIR) = 图 [D_PAIR, Lc, Lr, B]
    TensorF32* forward_graph(TensorF32* pair, TensorF32* str_bias);

private:
    std::unique_ptr<SelfAttention> self_attn_;
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
    
    // query: (B*L, 1, q_dim), kv: (B*L, T, kv_dim)
    // 输出: (B*L, 1, q_dim)
    TensorF32 forward(const TensorF32& query, const TensorF32& kv);

    // ===== 图模式前向（训练用）=====
    // 输入/输出均为图节点指针 (ggml 布局 dims[0]=最内维)
    // query: 值 (B*L,1,q_dim) = 图 [q_dim, 1, 1, B*L]
    // kv:    值 (B*L,T,kv_dim) = 图 [kv_dim, T, 1, B*L]
    // 返回: 值 (B*L,1,q_dim) = 图 [q_dim, 1, 1, B*L]
    TensorF32* forward_graph(TensorF32* query, TensorF32* kv);

private:
    int q_dim_, kv_dim_, n_head_;
    int head_dim_;   // = proj_dim / n_head, where proj_dim = max(q_dim, kv_dim) aligned to n_head
    int proj_dim_;   // 公共投影维度 = n_head * head_dim

    // 投影层: 把 Q 和 KV 投影到相同的 proj_dim
    std::unique_ptr<LinearLayer> Wq_;   // q_dim → proj_dim
    std::unique_ptr<LinearLayer> Wk_;   // kv_dim → proj_dim
    std::unique_ptr<LinearLayer> Wv_;   // kv_dim → proj_dim
    std::unique_ptr<LinearLayer> Wo_;   // proj_dim → q_dim (输出投影)
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

    // ===== 图模式前向（训练用）=====
    // 输入/输出均为图节点指针
    TensorF32* forward_graph(TensorF32* pair, bool bOutgoing = true);

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

    // ===== 图模式前向（训练用）=====
    // 输入/输出均为图节点指针。训练时 dropout 先以 identity 处理（图 drop 后续补）
    TensorF32* forward_graph(TensorF32* x);

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
    // 旧值类型构造函数 (保留注释):
    // TemplatePairStack();
    TemplatePairStack() = default;

    // 由 RFAAModel 注入已创建的参数指针
    void set_params(
        LinearLayer* rbf_proj,
        LayerNorm*   state_norm,
        LinearLayer* left_proj,   LinearLayer* right_proj,  LinearLayer* gate_proj,
        TriangleMultiplication* tri_mul_out, TriangleMultiplication* tri_mul_in,
        PairRowAttention* pair_row_attn, PairColAttention* pair_col_attn,
        FeedForward* pair_ff);
    
    TensorF32 forward(const TensorF32& pair, TensorF32& rbf_feature, const TensorF32& state);
    
private:
    // 旧值类型 (保留注释):
    // LinearLayer rbf_proj_(D_RBF, D_PAIR);
    // LayerNorm state_norm_(D_STATE);
    // LinearLayer left_proj_(D_STATE, 16);
    // LinearLayer right_proj_(D_STATE, 16);
    // LinearLayer gate_proj_(16 * 16, D_PAIR);
    // TriangleMultiplication tri_mul_out_;
    // TriangleMultiplication tri_mul_in_;
    // PairRowAttention pair_row_attn_;
    // PairColAttention pair_col_attn_;
    // FeedForward pair_ff_(D_PAIR, 2);

    LinearLayer* rbf_proj_   = nullptr; // D_RBF (64) → D_PAIR (128)
    LayerNorm*   state_norm_ = nullptr; // D_STATE (32)
    LinearLayer* left_proj_  = nullptr; // D_STATE (32) → 16
    LinearLayer* right_proj_ = nullptr; // D_STATE (32) → 16
    LinearLayer* gate_proj_  = nullptr; // 16*16 (256) → D_PAIR (128)

    TriangleMultiplication* tri_mul_out_ = nullptr;
    TriangleMultiplication* tri_mul_in_  = nullptr;

    Dropout drop_row_{1, 0.15f};
    Dropout drop_col_{2, 0.15f};

    PairRowAttention* pair_row_attn_ = nullptr;
    PairColAttention* pair_col_attn_ = nullptr;
    FeedForward*      pair_ff_       = nullptr;
};

} // namespace rfaa

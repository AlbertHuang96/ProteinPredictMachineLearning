#include "rfaa/Attention.h"
#include "rfaa/Embedding.h"

namespace rfaa {

// Attention 模块的桩实现
// 实际应调用 CUDA kernel 或 CPU 实现

SelfAttention::SelfAttention(const AttnConfig& config) : config_(config) {}
SelfAttention::~SelfAttention() = default;

TensorF32 SelfAttention::forward(const TensorF32& Q, const TensorF32& K, const TensorF32& V, const TensorF32* bias) {
    // 输入约定: Q, K, V 均为 (batch, n_head, D_head, L)
    // GGML dims = [L, D_head, n_head, batch]
    //
    // out_prod 语义: dst[i0,i1] = sum_k src0[i0,k] * src1[i1,k]
    // 收缩维是 dims[1], 即 D_head

    // ===== 1. scores = Q @ K^T =====
    // out_prod 隐含 B^T, 无需 permute K
    // Q: [L_q, D_head, H, B]  K: [L_k, D_head, H, B]
    // 收缩 D_head → scores: [L_q, L_k, H, B] = (B, H, L_q, L_k)
    auto scores = out_prod(const_cast<TensorF32*>(&Q), const_cast<TensorF32*>(&K));

    // ===== 2. scale = 1/sqrt(d_head) =====
    float scale_val = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
    auto scaled = scale(scores, scale_val);

    // ===== 3. add bias =====
    if (bias != nullptr) {
        scaled = add_impl(scaled, const_cast<TensorF32*>(bias), /*inplace=*/false);
    }

    // ===== 4. softmax =====
    auto attn = softmax(scaled);  // (B, H, L_q, L_k), dims = [L_k, L_q, H, B]

    // ===== 5. attn @ V =====
    // attn: [L_k, L_q, H, B], ne01 = L_q
    // V:    [L_v, D_head, H, B], ne11 = D_head
    // L_q ≠ D_head, 不能直接 out_prod
    //
    // 需要 permute 让收缩维对齐:
    // attn 需要在 L_k 上收缩 → permute 把 L_k 放到 dims[1]
    // attn: (B, H, L_q, L_k) → permute({0,1,3,2}) → (B, H, L_k, L_q)
    //       dims = [L_q, L_k, H, B], ne01 = L_k
    // V 需要在 L_k 上收缩 → permute 把 L_k 放到 dims[1]
    // V:    (B, H, D_head, L_k) → permute({0,1,3,2}) → (B, H, L_k, D_head)
    //       dims = [D_head, L_k, H, B], ne11 = L_k
    auto attn_t = permute(attn, {0, 1, 3, 2});
    auto V_t    = permute(const_cast<TensorF32*>(&V), {0, 1, 3, 2});
    auto output = out_prod(attn_t, V_t);  // (B, H, D_head, L_q)

    TensorF32 result;
    result.copy_from(*output);
    return result;
}

// ===== MSARowAttention (non-owning pointer 版本) =====
// 旧构造函数 (保留注释):
// MSARowAttention::MSARowAttention(const AttnConfig& config) : config_(config) {}
void MSARowAttention::set_params(const AttnConfig& config,
                                 LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                                 LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv) {
    config_ = config;
    self_attn_ = std::make_unique<SelfAttention>(config);
    to_b_   = to_b;   to_g_   = to_g;   to_out_ = to_out;
    Wq_     = Wq;     Wk_     = Wk;     Wv_     = Wv;
}

TensorF32 MSARowAttention::forward(const TensorF32& msa, const TensorF32& pair_biased) {
    auto Q    = Wq_->forward(msa);
    int B = static_cast<int>(Q.shape().dims[0]);
    int N = static_cast<int>(Q.shape().dims[1]);
    int L = static_cast<int>(Q.shape().dims[2]);
    //int D = static_cast<int>(Q.shape().dims[3]);
    int H = config_.n_head;
    int D = D_MSA;
    Q = Q.view({B * N, L, H, D});
    Q = Q.permute({0, 2, 3, 1});
    // (B, H, D, L)
    auto K    = Wk_->forward(msa);
    auto V    = Wv_->forward(msa);
    K = K.view(Shape({B * N, L, H, D}));
    K = K.permute({0, 2, 3, 1});

    V = V.view(Shape({B * N, L, H, D}));
    V = V.permute({0, 2, 3, 1});
    auto bias = to_b_->forward(pair_biased);
    auto gv   = to_g_->forward(msa);
    auto gate = sigmoid(&gv);
    auto attn_out = self_attn_->forward(Q, K, V, &bias);
    // (B*N, H, D, L)
    attn_out = attn_out.permute({0, 3, 1, 2});
    attn_out = attn_out.view(Shape({B, N, L, H * D}));
    
    auto gated_attn_out = out_prod(gate, &attn_out);
    return to_out_->forward(*gated_attn_out);
}

// ===== MSAColAttention (non-owning pointer 版本) =====
void MSAColAttention::set_params(const AttnConfig& config,
                                 LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                                 LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv) {
    config_ = config;
    self_attn_ = std::make_unique<SelfAttention>(config);
    to_b_   = to_b;   to_g_   = to_g;   to_out_ = to_out;
    Wq_     = Wq;     Wk_     = Wk;     Wv_     = Wv;
}

TensorF32 MSAColAttention::forward(const TensorF32& msa) {
    // Col attention: 沿 N_seq 维做 attention, L 合并进 batch
    // 输入 msa: (B, N, L, 256)
    auto Q = Wq_->forward(msa);  // (B, N, L, 2048)
    int B = static_cast<int>(Q.shape().dims[0]);
    int N = static_cast<int>(Q.shape().dims[1]);
    int L = static_cast<int>(Q.shape().dims[2]);
    int H = config_.n_head;      // 8
    int D = D_MSA;               // 256

    // Split heads: (B, N, L, 2048) → (B*L, N, 8, 256) → (B*L, 8, 256, N)
    Q = Q.view(Shape({B * L, N, H, D}));
    Q = Q.permute({0, 2, 3, 1});  // (B*L, 8, 256, N) = (batch, n_head, D_head, L_seq)

    auto K = Wk_->forward(msa);
    K = K.view(Shape({B * L, N, H, D}));
    K = K.permute({0, 2, 3, 1});

    auto V = Wv_->forward(msa);
    V = V.view(Shape({B * L, N, H, D}));
    V = V.permute({0, 2, 3, 1});

    auto gv   = to_g_->forward(msa);
    auto gate = sigmoid(&gv);

    auto attn_out = self_attn_->forward(Q, K, V);
    // attn_out: (B*L, 8, 256, N)

    // Merge heads: (B*L, 8, 256, N) → (B*L, N, 8*256) → (B, L, N, 2048) → (B, N, L, 2048)
    attn_out = attn_out.permute({0, 3, 1, 2});      // (B*L, N, 8, 256)
    attn_out = attn_out.view(Shape({B, L, N, H * D})); // (B, L, N, 2048)
    attn_out = attn_out.permute({0, 2, 1, 3});      // (B, N, L, 2048)  恢复原始 dim 顺序

    auto gated_attn_out = out_prod(gate, &attn_out);
    return to_out_->forward(*gated_attn_out);
}

// ===== MSAGlobalColAttention =====
TensorF32 MSAGlobalColAttention::forward(const TensorF32& msa) {
    // Global Col attention: Q 在 N_seq 维取 mean, 然后 1-to-many attention
    // 输入 msa: (B, N, L, 256)
    auto Q_raw = Wq_->forward(msa);  // (B, N, L, 2048)
    int B = static_cast<int>(Q_raw.shape().dims[0]);
    int N = static_cast<int>(Q_raw.shape().dims[1]);
    int L = static_cast<int>(Q_raw.shape().dims[2]);
    int H = config_.n_head;      // 8
    int D = D_MSA;               // 256

    // Q: 先 mean 再 split heads
    // mean 沿 dim=1(N_seq): (B, N, L, 2048) → (B, L, 2048)
    TensorF32 Q;
    {
        int feat_dim = L * H * D;  // L * 2048
        TensorF32 Q_mean(Shape({B, L, H * D}), Q_raw.device());  // (B, L, 2048)
        Q_mean.zero_();
        const float* src = Q_raw.data();
        float* dst = Q_mean.data();
        for (int b = 0; b < B; ++b) {
            for (int n = 0; n < N; ++n) {
                for (int f = 0; f < feat_dim; ++f) {
                    dst[b * feat_dim + f] += src[(b * N + n) * feat_dim + f];
                }
            }
        }
        // 除以 N
        float inv_n = 1.0f / static_cast<float>(N);
        for (int i = 0; i < B * feat_dim; ++i) {
            dst[i] *= inv_n;
        }
        Q = std::move(Q_mean);
    }
    // split heads: (B, L, 2048) → (B*L, 8, 256) → (B*L, 8, 256, 1)
    Q = Q.view(Shape({B * L, H, D}));      // (B*L, 8, 256)
    // 需要变成 4D: (B*L, 8, 256, 1) 即 (batch, n_head, D_head, L_seq=1)
    // 使用 view: (B*L, 8, 256) → (B*L, 8, 256, 1)
    Q = Q.view(Shape({B * L, H, D, 1}));

    // KV: split heads → (B*L, 8, 256, N)
    auto K = Wk_->forward(msa);
    K = K.view(Shape({B * L, N, H, D}));
    K = K.permute({0, 2, 3, 1});  // (B*L, 8, 256, N)

    auto V = Wv_->forward(msa);
    V = V.view(Shape({B * L, N, H, D}));
    V = V.permute({0, 2, 3, 1});  // (B*L, 8, 256, N)

    auto gv   = to_g_->forward(msa);
    auto gate = sigmoid(&gv);

    // Q: (B*L, 8, 256, 1), dims=[1, 256, 8, B*L], ne01=256
    // K: (B*L, 8, 256, N), dims=[N, 256, 8, B*L], ne11=256  ✓
    // scores = [1, N, 8, B*L] = (B*L, 8, 1, N) ✓
    auto attn_out = self_attn_->forward(Q, K, V);
    // attn_out: (B*L, 8, 256, 1)

    // Merge heads: (B*L, 8, 256, 1) → (B, L, 2048)
    attn_out = attn_out.view(Shape({B * L, H * D}));  // squeeze last dim → (B*L, 2048)
    attn_out = attn_out.view(Shape({B, L, H * D}));   // (B, L, 2048)
    // 扩展回 (B, N, L, 2048) 匹配 gate
    attn_out = attn_out.view(Shape({B, 1, L, H * D})); // (B, 1, L, 2048)
    // gate: (B, N, L, 2048), out_prod 会通过 repeat 广播 dim 1

    auto gated_attn_out = out_prod(gate, &attn_out);
    return to_out_->forward(*gated_attn_out);
}

// ===== PairRowAttention =====
void PairRowAttention::set_params(const AttnConfig& config,
                                  LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                                  LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv) {
    config_ = config;
    self_attn_ = std::make_unique<SelfAttention>(config);
    to_b_   = to_b;   to_g_   = to_g;   to_out_ = to_out;
    Wq_     = Wq;     Wk_     = Wk;     Wv_     = Wv;
}

TensorF32 PairRowAttention::forward(const TensorF32& pair, const TensorF32& str_bias) {
    // Row attention: 沿最后一个 L 维 (行) 做 attention
    // 输入 pair: (B, L_row, L_col, 128)
    auto Q = Wq_->forward(pair);  // (B, L_row, L_col, 256)
    int B   = static_cast<int>(Q.shape().dims[0]);
    int Lr  = static_cast<int>(Q.shape().dims[1]);  // L_row
    int Lc  = static_cast<int>(Q.shape().dims[2]);  // L_col
    int H   = config_.n_head;      // 8
    int D   = D_PAIR_HIDDEN;       // 32

    // Split heads: (B, L_row, L_col, 256) → (B*L_col, L_row, 8, 32) → (B*L_col, 8, 32, L_row)
    Q = Q.view(Shape({B * Lc, Lr, H, D}));
    Q = Q.permute({0, 2, 3, 1});  // (B*Lc, 8, 32, Lr)

    auto K = Wk_->forward(pair);
    K = K.view(Shape({B * Lc, Lr, H, D}));
    K = K.permute({0, 2, 3, 1});

    auto V = Wv_->forward(pair);
    V = V.view(Shape({B * Lc, Lr, H, D}));
    V = V.permute({0, 2, 3, 1});

    auto bias = to_b_->forward(str_bias);
    auto gv   = to_g_->forward(pair);
    auto gate = sigmoid(&gv);

    auto attn_out = self_attn_->forward(Q, K, V, &bias);
    // attn_out: (B*Lc, 8, 32, Lr)

    // Merge heads: (B*Lc, 8, 32, Lr) → (B*Lc, Lr, 256) → (B, Lr, Lc, 256)
    attn_out = attn_out.permute({0, 3, 1, 2});         // (B*Lc, Lr, 8, 32)
    attn_out = attn_out.view(Shape({B, Lc, Lr, H * D})); // (B, Lc, Lr, 256)
    attn_out = attn_out.permute({0, 2, 1, 3});         // (B, Lr, Lc, 256)

    auto gated_attn_out = out_prod(gate, &attn_out);
    return to_out_->forward(*gated_attn_out);
}

// ===== PairColAttention =====
void PairColAttention::set_params(const AttnConfig& config,
                                  LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                                  LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv) {
    config_ = config;
    self_attn_ = std::make_unique<SelfAttention>(config);
    to_b_   = to_b;   to_g_   = to_g;   to_out_ = to_out;
    Wq_     = Wq;     Wk_     = Wk;     Wv_     = Wv;
}

TensorF32 PairColAttention::forward(const TensorF32& pair, const TensorF32& str_bias) {
    // Col attention: 沿倒数第二个 L 维 (列) 做 attention
    // 输入 pair: (B, L_row, L_col, 128)
    auto Q = Wq_->forward(pair);  // (B, L_row, L_col, 256)
    int B   = static_cast<int>(Q.shape().dims[0]);
    int Lr  = static_cast<int>(Q.shape().dims[1]);  // L_row
    int Lc  = static_cast<int>(Q.shape().dims[2]);  // L_col
    int H   = config_.n_head;      // 8
    int D   = D_PAIR_HIDDEN;       // 32

    // Split heads: (B, L_row, L_col, 256) → (B*L_row, L_col, 8, 32) → (B*L_row, 8, 32, L_col)
    Q = Q.view(Shape({B * Lr, Lc, H, D}));
    Q = Q.permute({0, 2, 3, 1});  // (B*Lr, 8, 32, Lc)

    auto K = Wk_->forward(pair);
    K = K.view(Shape({B * Lr, Lc, H, D}));
    K = K.permute({0, 2, 3, 1});

    auto V = Wv_->forward(pair);
    V = V.view(Shape({B * Lr, Lc, H, D}));
    V = V.permute({0, 2, 3, 1});

    auto bias = to_b_->forward(str_bias);
    auto gv   = to_g_->forward(pair);
    auto gate = sigmoid(&gv);

    auto attn_out = self_attn_->forward(Q, K, V, &bias);
    // attn_out: (B*Lr, 8, 32, Lc)

    // Merge heads: (B*Lr, 8, 32, Lc) → (B*Lr, Lc, 256) → (B, Lr, Lc, 256)
    attn_out = attn_out.permute({0, 3, 1, 2});         // (B*Lr, Lc, 8, 32)
    attn_out = attn_out.view(Shape({B, Lr, Lc, H * D})); // (B, Lr, Lc, 256)

    auto gated_attn_out = out_prod(gate, &attn_out);
    return to_out_->forward(*gated_attn_out);
}

CrossAttention::CrossAttention(int q_dim, int kv_dim, int n_head)
    : q_dim_(q_dim), kv_dim_(kv_dim), n_head_(n_head)
{
    // 选择公共投影维度: 取 max(q_dim, kv_dim), 向上对齐到 n_head 的倍数
    int min_proj = std::max(q_dim, kv_dim);  // max(32, 64) = 64
    head_dim_ = (min_proj + n_head - 1) / n_head;  // ceil(64/8) = 8
    proj_dim_ = n_head * head_dim_;                // 8 * 8 = 64

    // 创建投影层
    Wq_ = std::unique_ptr<LinearLayer>(LinearLayer::create(q_dim_, proj_dim_, false));
    Wk_ = std::unique_ptr<LinearLayer>(LinearLayer::create(kv_dim_, proj_dim_, false));
    Wv_ = std::unique_ptr<LinearLayer>(LinearLayer::create(kv_dim_, proj_dim_, false));
    Wo_ = std::unique_ptr<LinearLayer>(LinearLayer::create(proj_dim_, q_dim_, false));
}

TensorF32 CrossAttention::forward(const TensorF32& query, const TensorF32& kv) {
    // 输入: query (B*L, 1, q_dim), kv (B*L, T, kv_dim)
    int BL  = static_cast<int>(query.shape().dims[0]);  // B*L
    int T   = static_cast<int>(kv.shape().dims[1]);      // 模板数
    int H   = n_head_;
    int D   = head_dim_;    // 8
    int PD  = proj_dim_;    // 64 = H * D

    // ===== 1. 投影到公共维度 =====
    auto Q = Wq_->forward(query);  // (B*L, 1, 64)
    auto K = Wk_->forward(kv);     // (B*L, T, 64)
    auto V = Wv_->forward(kv);     // (B*L, T, 64)

    // ===== 2. Split heads =====
    // Q: (B*L, 1, 64) → view → (B*L, 1, 8, 8) → permute → (B*L, 8, 8, 1)
    Q = Q.view(Shape({BL, 1, H, D}));
    Q = Q.permute({0, 2, 3, 1});  // (B*L, 8, 8, 1) = (batch, n_head, D_head, L_q=1)

    // K: (B*L, T, 64) → view → (B*L, T, 8, 8) → permute → (B*L, 8, 8, T)
    K = K.view(Shape({BL, T, H, D}));
    K = K.permute({0, 2, 3, 1});  // (B*L, 8, 8, T)

    // V: 同上
    V = V.view(Shape({BL, T, H, D}));
    V = V.permute({0, 2, 3, 1});  // (B*L, 8, 8, T)

    // ===== 3. Q @ K^T =====
    // Q: [1, 8, 8, BL], K: [T, 8, 8, BL]
    // out_prod 在 dims[1]=D_head=8 上收缩 → scores: [1, T, 8, BL] = (BL, 8, 1, T)
    auto scores = out_prod(&Q, &K);

    // ===== 4. Scale + softmax =====
    float scale_val = 1.0f / std::sqrt(static_cast<float>(D));
    auto scaled = scale(scores, scale_val);
    auto attn = softmax(scaled);  // (BL, 8, 1, T), dims = [T, 1, 8, BL]

    // ===== 5. attn @ V =====
    // attn: [T, 1, 8, BL], ne01 = 1
    // V:    [T, 8, 8, BL], ne11 = 8  → ne01 ≠ ne11
    // 需要 permute 让收缩维对齐:
    // attn: (BL, 8, 1, T) → permute({0,1,3,2}) → (BL, 8, T, 1)
    //       dims = [1, T, 8, BL], ne01 = T
    // V:    (BL, 8, 8, T) → permute({0,1,3,2}) → (BL, 8, T, 8)
    //       dims = [8, T, 8, BL], ne11 = T  ✓
    auto attn_t = permute(attn, {0, 1, 3, 2});
    auto V_t    = permute(&V, {0, 1, 3, 2});
    auto output = out_prod(attn_t, V_t);  // (BL, 8, 8, 1)

    // ===== 6. Merge heads =====
    // output: (BL, 8, 8, 1) → permute → (BL, 1, 8, 8) → view → (BL, 1, 64)
    auto merged = output->permute({0, 3, 1, 2});  // (BL, 1, 8, 8)
    merged = merged.view(Shape({BL, 1, PD}));     // (BL, 1, 64)

    // ===== 7. 输出投影: 64 → 32 =====
    auto result = Wo_->forward(merged);  // (B*L, 1, 32)
    return result;
}

// ===== TriangleMultiplication (non-owning pointer 版本) =====
// 旧构造函数 (保留注释):
// TriangleMultiplication::TriangleMultiplication(int dim) : dim_(dim) { ... }
void TriangleMultiplication::set_params(int dim,
                                        LayerNorm*   layernorm,     LinearLayer* left_proj,
                                        LinearLayer* right_proj,    LinearLayer* left_gate,
                                        LinearLayer* right_gate,    LinearLayer* gate,
                                        LayerNorm*   output_layernorm, LinearLayer* out_proj) {
    dim_             = dim;
    layernorm_        = layernorm;        // D_PAIR (128)
    left_proj_        = left_proj;        // D_PAIR (128) → D_HIDDEN_TRIMUL (128)
    right_proj_       = right_proj;       // D_PAIR (128) → D_HIDDEN_TRIMUL (128)
    left_gate_        = left_gate;        // D_PAIR (128) → D_HIDDEN_TRIMUL (128)
    right_gate_       = right_gate;       // D_PAIR (128) → D_HIDDEN_TRIMUL (128)
    gate_             = gate;             // D_PAIR (128) → D_PAIR (128)
    output_layernorm_ = output_layernorm; // D_HIDDEN_TRIMUL (128)
    out_proj_         = out_proj;         // D_HIDDEN_TRIMUL (128) → D_PAIR (128)
}

TensorF32 TriangleMultiplication::forward(const TensorF32& pair, bool bOutgoing) {
    auto pair_norm = layernorm_->forward(const_cast<TensorF32*>(&pair));  // TensorF32*
    auto left  = left_proj_->forward(*pair_norm);   // 解引用传引用
    auto right = right_proj_->forward(*pair_norm);
    auto lgv   = left_gate_->forward(*pair_norm);
    auto rgv   = right_gate_->forward(*pair_norm);
    auto left_gate  = sigmoid(&lgv);
    auto right_gate = sigmoid(&rgv);
    auto left_gated  = out_prod(&left, left_gate);
    auto right_gated = out_prod(&right, right_gate);

    auto tri_mul_forward = triangle_mul(left_gated, right_gated, float(pair.shape().dims[1]), bOutgoing);

    auto tri_mul_forward_norm  = output_layernorm_->forward(tri_mul_forward);
    auto tri_mul_forward_proj  = out_proj_->forward(*tri_mul_forward_norm);

    auto gv = gate_->forward(*pair_norm);
    auto gate_out = sigmoid(&gv);
    auto tri_mul_forward_gated = out_prod(gate_out, &tri_mul_forward_proj);

    return std::move(*tri_mul_forward_gated);
}


// ===== FeedForward (non-owning pointer 版本) =====
// 旧构造函数 (保留注释):
// FeedForward::FeedForward(int dim, int hidden_dim, float dropout)
//     : dim_(dim), hidden_dim_(hidden_dim) {}
void FeedForward::set_params(int dim, int hidden_dim, float dropout,
                             LayerNorm* layernorm, LinearLayer* linear1, LinearLayer* linear2) {
    dim_          = dim;
    hidden_dim_   = hidden_dim;
    dropout_rate_ = dropout;
    layernorm_    = layernorm;   // dim_
    linear1_      = linear1;     // dim_ → dim_*hidden_dim_
    linear2_      = linear2;     // dim_*hidden_dim_ → dim_
}

TensorF32 FeedForward::forward(const TensorF32& x) {
    auto x_norm   = layernorm_->forward(const_cast<TensorF32*>(&x));  // 返回 TensorF32*
    auto x_hidden = linear1_->forward(*x_norm);                      // 解引用后传引用
    auto* x_relu  = relu(&x_hidden);                                   // relu 接受指针，返回指针
    auto x_dropped = dropout_.forward(*x_relu);                        // dropout 接受值引用
    auto x_out = linear2_->forward(x_dropped);                        // 传引用
    return x_out;
}

void TemplatePairStack::set_params(
    LinearLayer* rbf_proj,
    LayerNorm*   state_norm,
    LinearLayer* left_proj,   LinearLayer* right_proj,  LinearLayer* gate_proj,
    TriangleMultiplication* tri_mul_out, TriangleMultiplication* tri_mul_in,
    PairRowAttention* pair_row_attn, PairColAttention* pair_col_attn,
    FeedForward* pair_ff)
{
    rbf_proj_   = rbf_proj;
    state_norm_ = state_norm;
    left_proj_  = left_proj;
    right_proj_ = right_proj;
    gate_proj_  = gate_proj;
    tri_mul_out_ = tri_mul_out;
    tri_mul_in_  = tri_mul_in;
    pair_row_attn_ = pair_row_attn;
    pair_col_attn_ = pair_col_attn;
    pair_ff_       = pair_ff;

    // 初始化: gate_proj 权重零初始化, bias 置 1
    if (gate_proj_) {
        gate_proj_->zeros_weight();
        gate_proj_->ones_bias();
    }
}

TensorF32 TemplatePairStack::forward(const TensorF32& pair, TensorF32& rbf_feature, const TensorF32& state) {
    
    TensorF32 rbf_proj = rbf_proj_->forward(rbf_feature);  // (B,L,L,128)
    
    auto& state_normed = *state_norm_->forward(const_cast<TensorF32*>(&state));
    
    TensorF32 left  = left_proj_->forward(state_normed);   // (B,L,16)
    TensorF32 right = right_proj_->forward(state_normed);  // (B,L,16)
    auto gate  = out_prod(&left, &right);                   // (B,L,L,256)
    TensorF32 gate_proj = gate_proj_->forward(*gate);       // (B,L,L,128)
    auto gate_sig  = sigmoid(&gate_proj);
    auto out_rbf_feature = out_prod(&rbf_feature, gate_sig);
    rbf_feature.copy_from(*out_rbf_feature);

    TensorF32 pair_tmp;
    pair_tmp.copy_from(pair);

    auto tri_out = tri_mul_out_->forward(pair_tmp, true);
    auto tri_out_drop = drop_row_.forward(tri_out);
    auto pair_tri_out = add_impl(&pair_tmp, &tri_out_drop, /*inplace=*/false);

    auto tri_in = tri_mul_in_->forward(*pair_tri_out, false);
    auto tri_in_drop = drop_row_.forward(tri_in);
    auto pair_tri_in = add_impl(pair_tri_out, &tri_in_drop, /*inplace=*/false);

    auto pair_row_attn = pair_row_attn_->forward(*pair_tri_in, rbf_proj);
    auto row_drop = drop_row_.forward(pair_row_attn);
    auto pair_after_row = add_impl(pair_tri_in, &row_drop, /*inplace=*/false);

    auto pair_col_attn = pair_col_attn_->forward(*pair_after_row, rbf_proj);
    auto col_drop = drop_col_.forward(pair_col_attn);
    auto pair_after_col = add_impl(pair_after_row, &col_drop, /*inplace=*/false);

    auto pair_ff_out = pair_ff_->forward(*pair_after_col);
    auto pair_final = add_impl(pair_after_col, &pair_ff_out, /*inplace=*/false);

    TensorF32 result;
    result.copy_from(*pair_final);
    return result;
}

} // namespace rfaa

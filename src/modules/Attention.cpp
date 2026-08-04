#include "rfaa/Attention.h"
#include "rfaa/Embedding.h"
#include "rfaa/Context.h"

namespace rfaa {

namespace {

// 把 to_b_ 投影出的 pair bias [H, Lq, Lk, B]（值 (B,Lk,Lq,H)）规整为
// self_attn scores 布局 [Lk, Lq, H, B*fold]（值 (B*fold, H, Lq, Lk)），
// 并将 batch 沿 fold 维广播（pair bias 与 fold 维无关）。
//
// 图 op 布局约定：值 (d0,d1,d2,d3) = 图 [d3,d2,d1,d0]。
//   scores = out_prod(K,Q) → [Lk,Lq,H,B] = 值 (B,H,Lq,Lk)
//   bias   = [H,Lq,Lk,B] = 值 (B,Lk,Lq,H)
// permute({1,2,0,3}) → [Lk,Lq,H,B] = 值 (B,H,Lq,Lk)，与 scores 逐元素一致。
TensorF32* prepare_pair_bias(TensorF32* bias,
                             int64_t Lq, int64_t Lk, int64_t B, int64_t fold) {
    bias = permute(bias, {1, 2, 0, 3});          // [H,Lq,Lk,B] → [Lk,Lq,H,B]
    if (fold != 1) {
        int64_t tgt[] = {Lk, Lq, bias->shape().dims[2], B * fold};
        TensorF32* target = context().new_tensor<float>(4, tgt);
        bias = repeat(bias, target);              // 广播 batch B → B*fold
    }
    return bias;
}

} // namespace

// Attention 模块的桩实现
// 实际应调用 CUDA kernel 或 CPU 实现

SelfAttention::SelfAttention(const AttnConfig& config) : config_(config) {}
SelfAttention::~SelfAttention() = default;

TensorF32 SelfAttention::forward(const TensorF32& Q, const TensorF32& K, const TensorF32& V, const TensorF32* bias) {
    // 输入约定: Q, K, V 均为 (batch, n_head, D_head, L)
    // GGML dims = [L, D_head, n_head, batch]
    //
    // out_prod 语义: dst[i0,i1] = sum_k src0[i0,k] * src1[i1,k]
    // 收缩维是 dims[1]

    // ===== 1. scores = K @ Q^T  (key 最内维 dims[0], 使 softmax 沿 key 轴归一) =====
    // out_prod(K, Q): src0=K [L_k, D_head, H, B], src1=Q [L_q, D_head, H, B]
    // 收缩 dims[1]=D_head → scores: [L_k, L_q, H, B] = 值 (B, H, L_q, L_k)
    auto scores = out_prod(const_cast<TensorF32*>(&K), const_cast<TensorF32*>(&Q));

    // ===== 2. scale = 1/sqrt(d_head) =====
    float scale_val = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
    auto scaled = scale(scores, scale_val);

    // ===== 3. add bias =====
    // 遗留：值版 bias 广播未处理（同 Bug3）。bias 需先 permute+repeat 展开成与
    // scaled 完全相同的形状再 add_impl（图模式已用 prepare_pair_bias 修复）。
    if (bias != nullptr) {
        scaled = add_impl(scaled, const_cast<TensorF32*>(bias), /*inplace=*/false);
    }

    // ===== 4. softmax 沿 dims[0]=L_k (key 轴) 归一 =====
    auto attn = softmax(scaled);  // [L_k, L_q, H, B] = 值 (B, H, L_q, L_k)

    // ===== 5. attn @ V =====
    // attn: [L_k, L_q, H, B] (key dim0, query dim1)
    // V:    [L_k, D_head, H, B] (key dim0, head_dim dim1)
    // 需把 query 放 dim0、key 放 dim1（attn），head_dim 放 dim0、key 放 dim1（V），
    // 使 out_prod 在 dims[1]=L_k 上收缩:
    auto attn_t = permute(attn, {1, 0, 2, 3});                              // [L_k,L_q,H,B] → [L_q,L_k,H,B]
    auto V_t    = permute(const_cast<TensorF32*>(&V), {1, 0, 2, 3});        // [L_k,D,H,B]   → [D,L_k,H,B]
    auto output = out_prod(attn_t, V_t);  // 收缩 dims[1]=L_k → [L_q, D_head, H, B] = 值 (B,H,D_head,L_q)

    TensorF32 result;
    result.copy_from(*output);
    return result;
}

// ===== SelfAttention::forward_graph (图模式) =====
// 与值版 forward 逻辑完全一致，但输入输出均为图节点指针，去掉 copy_from 值拷贝
TensorF32* SelfAttention::forward_graph(TensorF32* Q, TensorF32* K, TensorF32* V, TensorF32* bias) {
    // 输入约定: Q, K, V 均为 (batch, n_head, D_head, L)
    // GGML dims = [L, D_head, n_head, batch]

    // 1. scores = K @ Q^T (key 最内 dims[0], 使 softmax 沿 key 轴归一)
    // out_prod(K, Q): src0=K [L_k,D_head,H,B], src1=Q [L_q,D_head,H,B]
    // 收缩 dims[1]=D_head → [L_k, L_q, H, B]
    auto scores = out_prod(K, Q);
    // 2. scale = 1/sqrt(d_head)
    float scale_val = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
    auto scaled = scale(scores, scale_val);
    // 3. add bias
    if (bias != nullptr) {
        scaled = add_impl(scaled, bias, /*inplace=*/false);
    }
    // 4. softmax 沿 dims[0]=L_k (key 轴) 归一
    auto attn = softmax(scaled);  // [L_k, L_q, H, B]
    // 5. attn @ V: 把 query 放 dim0、key 放 dim1 (attn)，head_dim 放 dim0、key 放 dim1 (V)，
    //    使 out_prod 在 dims[1]=L_k 上收缩 → [L_q, D_head, H, B]
    auto attn_t = permute(attn, {1, 0, 2, 3});  // [L_k,L_q,H,B] → [L_q,L_k,H,B]
    auto V_t    = permute(V, {1, 0, 2, 3});     // [L_k,D,H,B]   → [D,L_k,H,B]
    auto output = out_prod(attn_t, V_t);        // 收缩 dims[1]=L_k → [L_q, D_head, H, B]

    return output;   // 返回图节点，不再 copy_from
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
    auto bias = to_b_->forward(pair_biased);  // 遗留：值版 bias 广播未处理（同 Bug3）
    auto gv   = to_g_->forward(msa);
    auto gate = sigmoid(&gv);
    auto attn_out = self_attn_->forward(Q, K, V, &bias);
    // (B*N, H, D, L)
    attn_out = attn_out.permute({0, 3, 1, 2});
    attn_out = attn_out.view(Shape({B, N, L, H * D}));
    
    auto gated_attn_out = out_prod(gate, &attn_out);
    return to_out_->forward(*gated_attn_out);
}

// ===== MSARowAttention::forward_graph (图模式) =====
// 与值版 forward 逻辑一致，但输入输出均为图节点指针（ggml 布局 dims[0]=最内维）
// 约定: 值 (d0,d1,d2,d3) = 图 [d3,d2,d1,d0]
//      值 .view(vshape)     → 图 view(node, reversed(vshape))
//      值 .permute(p) (4D)   → 图 permute(node, g), g[k] = 3 - p[3-k]
TensorF32* MSARowAttention::forward_graph(TensorF32* msa, TensorF32* pair_biased) {
    // 输入: msa 值 (B,N,L,D_MSA) = 图 [D_MSA, L, N, B]
    //       pair_biased 值 (B,L,L,D_PAIR) = 图 [D_PAIR, L, L, B]
    // Wq_->forward_graph(msa) → 值 (B,N,L,H*D) = 图 [H*D, L, N, B]
    auto* Q = Wq_->forward_graph(msa);
    int B = static_cast<int>(Q->shape().dims[3]);   // B (最外层)
    int N = static_cast<int>(Q->shape().dims[2]);   // N_seq
    int L = static_cast<int>(Q->shape().dims[1]);   // L (残基)
    int H = config_.n_head;                          // 8
    int D = D_MSA;                                   // 256

    // Split heads:
    // 值: (B,N,L,H*D) → view({B*N,L,H,D}) → (B*N,L,H,D) → permute({0,2,3,1}) → (B*N,H,D,L)
    // 图: [H*D,L,N,B] → view([D,H,L,B*N]) → permute({2,0,1,3}) → [L,D,H,B*N] (self_attn 契约)
    Q = view(Q, Shape{D, H, L, B * N});
    Q = permute(Q, {2, 0, 1, 3});
    auto* K = Wk_->forward_graph(msa);
    K = view(K, Shape{D, H, L, B * N});
    K = permute(K, {2, 0, 1, 3});
    auto* V = Wv_->forward_graph(msa);
    V = view(V, Shape{D, H, L, B * N});
    V = permute(V, {2, 0, 1, 3});

    // bias: to_b_ D_PAIR→N_HEAD, 值 (B,L,L,8) = 图 [8,L,L,B]
    // 规整为 scores 布局 [L,L,H,B*N]（值 (B*N,H,L,L)），并把 batch 沿 N_seq 广播。
    auto* bias = to_b_->forward_graph(pair_biased);
    bias = prepare_pair_bias(bias, /*Lq=*/L, /*Lk=*/L, /*B=*/B, /*fold=*/N);
    // gate: to_g_ D_MSA→H*D, 值 (B,N,L,H*D) = 图 [H*D,L,N,B]
    auto* gv   = to_g_->forward_graph(msa);
    auto* gate = sigmoid(gv);

    // self_attn: Q,K,V 均为 [L,D,H,B*N], 输出 [L,D,H,B*N] = 值 (B*N,H,D,L)
    auto* attn_out = self_attn_->forward_graph(Q, K, V, bias);

    // Merge heads:
    // 值: (B*N,H,D,L) → permute({0,3,1,2}) → (B*N,L,H,D) → view({B,N,L,H*D}) → (B,N,L,H*D)
    // 图: [L,D,H,B*N] → permute({1,2,0,3}) → [D,H,L,B*N] → view([H*D,L,N,B])
    auto* merged = permute(attn_out, {1, 2, 0, 3}); // [D,H,L,B*N] = 值 (B*N,L,H,D)
    merged = view(merged, Shape{H * D, L, N, B});   // [H*D,L,N,B] = 值 (B,N,L,H*D)

    // 门控: gate 与 merged 同为 [H*D,L,N,B]
    auto* gated = out_prod(gate, merged);
    // 输出投影: H*D → D_MSA, 返回 [D_MSA,L,N,B] = 值 (B,N,L,D_MSA)
    return to_out_->forward_graph(gated);
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

// ===== MSAColAttention::forward_graph (图模式) =====
// 与值版 forward 一致，输入输出均为图节点指针（ggml 布局 dims[0]=最内维）
TensorF32* MSAColAttention::forward_graph(TensorF32* msa) {
    // 输入: msa 值 (B,N,L,D_MSA) = 图 [D_MSA, L, N, B]
    // Wq_->forward_graph(msa) → 值 (B,N,L,H*D) = 图 [H*D, L, N, B]
    auto* Q = Wq_->forward_graph(msa);
    int B = static_cast<int>(Q->shape().dims[3]);   // B (最外层)
    int N = static_cast<int>(Q->shape().dims[2]);   // N_seq
    int L = static_cast<int>(Q->shape().dims[1]);   // L (残基)
    int H = config_.n_head;                          // 8
    int D = D_MSA;                                   // 256

    // Split heads (沿 N_seq 维做 attention, L 合并进 batch):
    // 值: (B,N,L,H*D) → view({B*L,N,H,D}) → (B*L,N,H,D) → permute({0,2,3,1}) → (B*L,H,D,N)
    // 图: [H*D,L,N,B] → view([D,H,N,B*L]) → permute({2,0,1,3}) → [N,D,H,B*L] (self_attn 契约)
    Q = view(Q, Shape{D, H, N, B * L});
    Q = permute(Q, {2, 0, 1, 3});
    auto* K = Wk_->forward_graph(msa);
    K = view(K, Shape{D, H, N, B * L});
    K = permute(K, {2, 0, 1, 3});
    auto* V = Wv_->forward_graph(msa);
    V = view(V, Shape{D, H, N, B * L});
    V = permute(V, {2, 0, 1, 3});

    // gate: to_g_ D_MSA→H*D, 值 (B,N,L,H*D) = 图 [H*D,L,N,B]
    auto* gv   = to_g_->forward_graph(msa);
    auto* gate = sigmoid(gv);

    // self_attn: Q,K,V 均为 [N,D,H,B*L], 输出 [N,D,H,B*L] = 值 (B*L,H,D,N)
    auto* attn_out = self_attn_->forward_graph(Q, K, V);

    // Merge heads:
    // 值: (B*L,H,D,N) → permute({0,3,1,2}) → (B*L,N,H,D) → view({B,L,N,H*D}) → (B,L,N,H*D)
    //     → permute({0,2,1,3}) → (B,N,L,H*D)
    // 图: [N,D,H,B*L] → permute({1,2,0,3}) → [D,H,N,B*L] → view([H*D,N,L,B]) → permute({0,2,1,3}) → [H*D,L,N,B]
    auto* merged = permute(attn_out, {1, 2, 0, 3}); // [D,H,N,B*L] = 值 (B*L,N,H,D)
    merged = view(merged, Shape{H * D, N, L, B});   // [H*D,N,L,B] = 值 (B,L,N,H*D)
    merged = permute(merged, {0, 2, 1, 3});          // [H*D,L,N,B] = 值 (B,N,L,H*D)

    // 门控: gate 与 merged 同为 [H*D,L,N,B]
    auto* gated = out_prod(gate, merged);
    // 输出投影: H*D → D_MSA, 返回 [D_MSA,L,N,B] = 值 (B,N,L,D_MSA)
    return to_out_->forward_graph(gated);
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

// ===== MSAGlobalColAttention::forward_graph (图模式) =====
// Global Col attention: Q 在 N_seq 维取 mean, 然后 1-to-many attention。
// 与值版 forward 逻辑一致，输入输出均为图节点指针（ggml 布局 dims[0]=最内维）。
// 关键点：现有 mean/sum 都是全局塌缩到标量，sum_rows 只沿最内维求和。
// 因此把待消去的 N_seq 先用 permute 挪到最内维 (graph dim0)，再 sum_rows，再 scale(1/N)。
// 图布局约定: 值 (d0,d1,d2,d3) = 图 [d3,d2,d1,d0]。
TensorF32* MSAGlobalColAttention::forward_graph(TensorF32* msa) {
    // 输入: msa 值 (B,N,L,D_MSA) = 图 [D_MSA, L, N, B]
    // Wq_->forward_graph(msa) → 值 (B,N,L,H*D) = 图 [H*D, L, N, B]
    auto* Q = Wq_->forward_graph(msa);
    int B = static_cast<int>(Q->shape().dims[3]);   // B (最外层)
    int N = static_cast<int>(Q->shape().dims[2]);   // N_seq
    int L = static_cast<int>(Q->shape().dims[1]);   // L (残基)
    int H = config_.n_head;                          // 8
    int D = D_MSA;                                   // 256

    // ===== 1. Q mean over N_seq (值 dim1) =====
    // 值: (B,N,L,H*D) → mean(dim=1) → (B,L,H*D)
    // 图: [H*D,L,N,B] → permute({2,1,0,3}) → [N,L,H*D,B] → sum_rows(沿最内维 N)
    //     → [1,L,H*D,B] → scale(1/N) → [1,L,H*D,B] → permute({0,2,1,3}) → [1,H*D,L,B]
    auto* qp = permute(Q, {2, 1, 0, 3});             // [N,L,H*D,B] = 值 (B,H*D,L,N)
    auto* qs = sum_rows(qp);                         // [1,L,H*D,B] = 值 (B,H*D,L,1)，Σ_n
    auto* qm = scale(qs, 1.0f / static_cast<float>(N)); // 均值
    auto* qm2 = permute(qm, {0, 2, 1, 3});           // [1,H*D,L,B] = 值 (B,L,H*D,1)

    // ===== 2. Split Q heads =====
    // 值: (B,L,H*D,1) → view({B*L,H,D,1}) → (B*L,H,D,1) (L_seq=1, self_attn 契约)
    // 图: [1,H*D,L,B] → view([1,D,H,B*L]) → [1,D,H,B*L]
    auto* Qh = view(qm2, Shape{1, D, H, B * L});

    // ===== 3. Split KV heads =====
    // 值: (B,N,L,H*D) → view({B*L,N,H,D}) → (B*L,N,H,D) → permute({0,2,3,1}) → (B*L,H,D,N)
    // 图: [H*D,L,N,B] → view([D,H,N,B*L]) → permute({2,0,1,3}) → [N,D,H,B*L]
    auto* K = Wk_->forward_graph(msa);
    K = view(K, Shape{D, H, N, B * L});
    K = permute(K, {2, 0, 1, 3});
    auto* V = Wv_->forward_graph(msa);
    V = view(V, Shape{D, H, N, B * L});
    V = permute(V, {2, 0, 1, 3});

    // ===== 4. gate =====
    // to_g_ D_MSA→H*D, 值 (B,N,L,H*D) = 图 [H*D,L,N,B]
    auto* gv   = to_g_->forward_graph(msa);
    auto* gate = sigmoid(gv);

    // ===== 5. self_attn: Q=[1,D,H,B*L], K/V=[N,D,H,B*L] → [1,D,H,B*L] = 值 (B*L,H,D,1) =====
    auto* attn_out = self_attn_->forward_graph(Qh, K, V);

    // ===== 6. Merge heads + 显式 repeat 广播 N_seq 维 =====
    // 值: (B*L,H,D,1) → view({B,L,H*D}) → (B,L,H*D) → view({B,1,L,H*D}) → (B,1,L,H*D)
    //     → repeat → (B,N,L,H*D)
    // 图: [1,D,H,B*L] → view([H*D,L,B]) → [H*D,L,B] → view([H*D,L,1,B]) → [H*D,L,1,B]
    //     → repeat 到 [H*D,L,N,B]（显式广播，避免依赖 out_prod 的 src1 广播）
    auto* merged = view(attn_out, Shape{H * D, L, B});   // [H*D,L,B] = 值 (B,L,H*D)
    merged = view(merged, Shape{H * D, L, 1, B});        // [H*D,L,1,B] = 值 (B,1,L,H*D)
    int64_t tgt[4] = {H * D, L, N, B};
    TensorF32* target = context().new_tensor<float>(4, tgt);
    merged = repeat(merged, target);                      // [H*D,L,N,B] = 值 (B,N,L,H*D)

    // ===== 7. 门控 + 输出投影 =====
    // gate 与 merged 同为 [H*D,L,N,B]
    auto* gated = out_prod(gate, merged);
    // 输出投影: H*D → D_MSA, 返回 [D_MSA,L,N,B] = 值 (B,N,L,D_MSA)
    return to_out_->forward_graph(gated);
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

    auto bias = to_b_->forward(str_bias);  // 遗留：值版 bias 广播未处理（同 Bug3）；前置不变量 pair 方阵(Lr==Lc==L)
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

// ===== PairRowAttention::forward_graph (图模式) =====
// 与值版 forward 一致，输入输出均为图节点指针（ggml 布局 dims[0]=最内维）
TensorF32* PairRowAttention::forward_graph(TensorF32* pair, TensorF32* str_bias) {
    // 输入: pair 值 (B,Lr,Lc,D_PAIR) = 图 [D_PAIR, Lc, Lr, B]
    //       str_bias 值 (B,Lr,Lc,D_PAIR) = 图 [D_PAIR, Lc, Lr, B]
    // Wq_->forward_graph(pair) → 值 (B,Lr,Lc,H*D) = 图 [H*D, Lc, Lr, B]
    auto* Q = Wq_->forward_graph(pair);
    int B  = static_cast<int>(Q->shape().dims[3]);   // B (最外层)
    int Lr = static_cast<int>(Q->shape().dims[2]);   // L_row
    int Lc = static_cast<int>(Q->shape().dims[1]);   // L_col
    int H  = config_.n_head;                         // 8
    int D  = D_PAIR_HIDDEN;                          // 32

    // Split heads (沿最后一个 L 维=行做 attention, L_col 合并进 batch):
    // 值: (B,Lr,Lc,H*D) → view({B*Lc,Lr,H,D}) → (B*Lc,Lr,H,D) → permute({0,2,3,1}) → (B*Lc,H,D,Lr)
    // 图: [H*D,Lc,Lr,B] → view([D,H,Lr,B*Lc]) → permute({2,0,1,3}) → [Lr,D,H,B*Lc] (self_attn 契约)
    Q = view(Q, Shape{D, H, Lr, B * Lc});
    Q = permute(Q, {2, 0, 1, 3});
    auto* K = Wk_->forward_graph(pair);
    K = view(K, Shape{D, H, Lr, B * Lc});
    K = permute(K, {2, 0, 1, 3});
    auto* V = Wv_->forward_graph(pair);
    V = view(V, Shape{D, H, Lr, B * Lc});
    V = permute(V, {2, 0, 1, 3});

    // bias: to_b_ D_PAIR→N_HEAD, 值 (B,Lr,Lc,8) = 图 [8,Lc,Lr,B]
    // 规整为 scores 布局 [Lr,Lr,H,B*Lc]（值 (B*Lc,H,Lr,Lr)），并把 batch 沿 Lc 广播。
    // 前置不变量：pair 为方阵（Lr==Lc==L），实际网络里 str_bias 来自 rbf_proj (B,L,L,D_PAIR)。
    auto* bias = to_b_->forward_graph(str_bias);
    bias = prepare_pair_bias(bias, /*Lq=*/Lr, /*Lk=*/Lr, /*B=*/B, /*fold=*/Lc);
    // gate: to_g_ D_PAIR→H*D, 值 (B,Lr,Lc,H*D) = 图 [H*D,Lc,Lr,B]
    auto* gv   = to_g_->forward_graph(pair);
    auto* gate = sigmoid(gv);

    // self_attn: Q,K,V 均为 [Lr,D,H,B*Lc], 输出 [Lr,D,H,B*Lc] = 值 (B*Lc,H,D,Lr)
    auto* attn_out = self_attn_->forward_graph(Q, K, V, bias);

    // Merge heads:
    // 值: (B*Lc,H,D,Lr) → permute({0,3,1,2}) → (B*Lc,Lr,H,D) → view({B,Lc,Lr,H*D}) → (B,Lc,Lr,H*D)
    //     → permute({0,2,1,3}) → (B,Lr,Lc,H*D)
    // 图: [Lr,D,H,B*Lc] → permute({1,2,0,3}) → [D,H,Lr,B*Lc] → view([H*D,Lr,Lc,B]) → permute({0,2,1,3}) → [H*D,Lc,Lr,B]
    auto* merged = permute(attn_out, {1, 2, 0, 3}); // [D,H,Lr,B*Lc] = 值 (B*Lc,Lr,H,D)
    merged = view(merged, Shape{H * D, Lr, Lc, B}); // [H*D,Lr,Lc,B] = 值 (B,Lc,Lr,H*D)
    merged = permute(merged, {0, 2, 1, 3});          // [H*D,Lc,Lr,B] = 值 (B,Lr,Lc,H*D)

    // 门控: gate 与 merged 同为 [H*D,Lc,Lr,B]
    auto* gated = out_prod(gate, merged);
    // 输出投影: H*D → D_PAIR, 返回 [D_PAIR,Lc,Lr,B] = 值 (B,Lr,Lc,D_PAIR)
    return to_out_->forward_graph(gated);
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

    auto bias = to_b_->forward(str_bias);  // 遗留：值版 bias 广播未处理（同 Bug3）；前置不变量 pair 方阵(Lr==Lc==L)
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

// ===== PairColAttention::forward_graph (图模式) =====
// 与值版 forward 一致，输入输出均为图节点指针（ggml 布局 dims[0]=最内维）
TensorF32* PairColAttention::forward_graph(TensorF32* pair, TensorF32* str_bias) {
    // 输入: pair 值 (B,Lr,Lc,D_PAIR) = 图 [D_PAIR, Lc, Lr, B]
    //       str_bias 值 (B,Lr,Lc,D_PAIR) = 图 [D_PAIR, Lc, Lr, B]
    // Wq_->forward_graph(pair) → 值 (B,Lr,Lc,H*D) = 图 [H*D, Lc, Lr, B]
    auto* Q = Wq_->forward_graph(pair);
    int B  = static_cast<int>(Q->shape().dims[3]);   // B (最外层)
    int Lr = static_cast<int>(Q->shape().dims[2]);   // L_row
    int Lc = static_cast<int>(Q->shape().dims[1]);   // L_col
    int H  = config_.n_head;                         // 8
    int D  = D_PAIR_HIDDEN;                          // 32

    // Split heads (沿倒数第二个 L 维=列做 attention, L_row 合并进 batch):
    // 值: (B,Lr,Lc,H*D) → view({B*Lr,Lc,H,D}) → (B*Lr,Lc,H,D) → permute({0,2,3,1}) → (B*Lr,H,D,Lc)
    // 图: [H*D,Lc,Lr,B] → view([D,H,Lc,B*Lr]) → permute({2,0,1,3}) → [Lc,D,H,B*Lr] (self_attn 契约)
    Q = view(Q, Shape{D, H, Lc, B * Lr});
    Q = permute(Q, {2, 0, 1, 3});
    auto* K = Wk_->forward_graph(pair);
    K = view(K, Shape{D, H, Lc, B * Lr});
    K = permute(K, {2, 0, 1, 3});
    auto* V = Wv_->forward_graph(pair);
    V = view(V, Shape{D, H, Lc, B * Lr});
    V = permute(V, {2, 0, 1, 3});

    // bias: to_b_ D_PAIR→N_HEAD, 值 (B,Lr,Lc,8) = 图 [8,Lc,Lr,B]
    // 规整为 scores 布局 [Lc,Lc,H,B*Lr]（值 (B*Lr,H,Lc,Lc)），并把 batch 沿 Lr 广播。
    // 前置不变量：pair 为方阵（Lr==Lc==L），实际网络里 str_bias 来自 rbf_proj (B,L,L,D_PAIR)。
    auto* bias = to_b_->forward_graph(str_bias);
    bias = prepare_pair_bias(bias, /*Lq=*/Lc, /*Lk=*/Lc, /*B=*/B, /*fold=*/Lr);
    // gate: to_g_ D_PAIR→H*D, 值 (B,Lr,Lc,H*D) = 图 [H*D,Lc,Lr,B]
    auto* gv   = to_g_->forward_graph(pair);
    auto* gate = sigmoid(gv);

    // self_attn: Q,K,V 均为 [Lc,D,H,B*Lr], 输出 [Lc,D,H,B*Lr] = 值 (B*Lr,H,D,Lc)
    auto* attn_out = self_attn_->forward_graph(Q, K, V, bias);

    // Merge heads:
    // 值: (B*Lr,H,D,Lc) → permute({0,3,1,2}) → (B*Lr,Lc,H,D) → view({B,Lr,Lc,H*D}) → (B,Lr,Lc,H*D)
    // 图: [Lc,D,H,B*Lr] → permute({1,2,0,3}) → [D,H,Lc,B*Lr] → view([H*D,Lc,Lr,B])
    auto* merged = permute(attn_out, {1, 2, 0, 3}); // [D,H,Lc,B*Lr] = 值 (B*Lr,Lc,H,D)
    merged = view(merged, Shape{H * D, Lc, Lr, B}); // [H*D,Lc,Lr,B] = 值 (B,Lr,Lc,H*D)

    // 门控: gate 与 merged 同为 [H*D,Lc,Lr,B]
    auto* gated = out_prod(gate, merged);
    // 输出投影: H*D → D_PAIR, 返回 [D_PAIR,Lc,Lr,B] = 值 (B,Lr,Lc,D_PAIR)
    return to_out_->forward_graph(gated);
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

// ===== CrossAttention::forward_graph (图模式) =====
// 与值版 forward 逻辑一致，但输入输出均为图节点指针（ggml 布局 dims[0]=最内维）
TensorF32* CrossAttention::forward_graph(TensorF32* query, TensorF32* kv) {
    // 输入: query 值 (B*L,1,q_dim) = 图 [q_dim, 1, 1, B*L]
    //       kv    值 (B*L,T,kv_dim) = 图 [kv_dim, T, 1, B*L]
    int BL = static_cast<int>(query->shape().dims[3]); // B*L
    int T  = static_cast<int>(kv->shape().dims[1]);    // 模板数
    int H  = n_head_;    // 8
    int D  = head_dim_;  // 8
    int PD = proj_dim_;  // 64 = H*D

    // ===== 1. 投影到公共维度 =====
    auto* Q = Wq_->forward_graph(query);  // 值 (BL,1,64) = 图 [64,1,1,BL]
    auto* K = Wk_->forward_graph(kv);     // 值 (BL,T,64) = 图 [64,T,1,BL]
    auto* V = Wv_->forward_graph(kv);     // 值 (BL,T,64) = 图 [64,T,1,BL]

    // ===== 2. Split heads =====
    // Q: 值 (BL,1,64) → view({BL,1,H,D}) → (BL,1,H,D) → permute({0,2,3,1}) → (BL,H,D,1)
    //    图 [64,1,1,BL] → view([D,H,1,BL]) → permute({2,0,1,3}) → [1,D,H,BL]
    Q = view(Q, Shape{D, H, 1, BL});
    Q = permute(Q, {2, 0, 1, 3});
    // K: 值 (BL,T,64) → view({BL,T,H,D}) → (BL,T,H,D) → permute({0,2,3,1}) → (BL,H,D,T)
    //    图 [64,T,1,BL] → view([D,H,T,BL]) → permute({2,0,1,3}) → [T,D,H,BL]
    K = view(K, Shape{D, H, T, BL});
    K = permute(K, {2, 0, 1, 3});
    // V: 同 K
    V = view(V, Shape{D, H, T, BL});
    V = permute(V, {2, 0, 1, 3});

    // ===== 3. Q @ K^T (收缩 dims[1]=D_head) =====
    // Q: [1,D,H,BL], K: [T,D,H,BL] → out_prod → [1,T,H,BL] = 值 (BL,H,1,T)
    auto* scores = out_prod(Q, K);

    // ===== 4. Scale + softmax =====
    float scale_val = 1.0f / std::sqrt(static_cast<float>(D));
    auto* scaled = scale(scores, scale_val);
    auto* attn = softmax(scaled);  // [1,T,H,BL]

    // ===== 5. attn @ V (permute 对齐收缩维) =====
    // 值版 attn/V 用 permute({0,1,3,2}); 图等价映射 g=[1,0,2,3]
    auto* attn_t = permute(attn, {1, 0, 2, 3});  // [1,T,H,BL] → [T,1,H,BL]
    auto* V_t    = permute(V, {1, 0, 2, 3});     // [T,D,H,BL] → [D,T,H,BL]
    auto* output = out_prod(attn_t, V_t);        // [T,D,H,BL] = 值 (BL,H,D,T)

    // ===== 6. Merge heads =====
    // 值: output → permute({0,3,1,2}) → (BL,T,H,D) → view({BL,1,PD}) → (BL,1,64)
    // 图: [T,D,H,BL] → permute({1,2,0,3}) → [D,H,T,BL] → view([PD,1,BL])
    auto* merged = permute(output, {1, 2, 0, 3});  // [D,H,T,BL] = 值 (BL,T,H,D)
    merged = view(merged, Shape{PD, 1, BL});        // [PD,1,BL] = 值 (BL,1,PD)

    // ===== 7. 输出投影: proj_dim → q_dim =====
    return Wo_->forward_graph(merged);  // 值 (B*L,1,q_dim) = 图 [q_dim,1,1,B*L]
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

// ===== TriangleMultiplication::forward_graph (图模式) =====
// 输入输出均为图节点指针，LayerNorm/Linear 走 forward_graph 指针接口
TensorF32* TriangleMultiplication::forward_graph(TensorF32* pair, bool bOutgoing) {
    auto pair_norm  = layernorm_->forward(pair);               // TensorF32*
    auto left       = left_proj_->forward_graph(pair_norm);    // TensorF32*
    auto right      = right_proj_->forward_graph(pair_norm);
    auto lgv        = left_gate_->forward_graph(pair_norm);
    auto rgv        = right_gate_->forward_graph(pair_norm);
    auto left_gate  = sigmoid(lgv);
    auto right_gate = sigmoid(rgv);
    auto left_gated  = out_prod(left, left_gate);
    auto right_gated = out_prod(right, right_gate);

    auto tri_mul_forward = triangle_mul(left_gated, right_gated, float(pair->shape().dims[1]), bOutgoing);

    auto tri_mul_forward_norm = output_layernorm_->forward(tri_mul_forward);
    auto tri_mul_forward_proj = out_proj_->forward_graph(tri_mul_forward_norm);

    auto gv = gate_->forward_graph(pair_norm);
    auto gate_out = sigmoid(gv);
    auto tri_mul_forward_gated = out_prod(gate_out, tri_mul_forward_proj);

    return tri_mul_forward_gated;   // 返回图节点
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

// ===== FeedForward::forward_graph (图模式) =====
// 输入输出均为图节点指针。训练时 dropout 以 identity 处理（图 drop 后续补充）
TensorF32* FeedForward::forward_graph(TensorF32* x) {
    auto x_norm    = layernorm_->forward(x);              // TensorF32*
    auto x_hidden  = linear1_->forward_graph(x_norm);     // TensorF32*
    auto x_relu    = relu(x_hidden);                       // relu 返回指针
    // dropout 图模式暂以 identity 处理（训练时图 drop 需后续专用 op）
    auto x_out     = linear2_->forward_graph(x_relu);
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

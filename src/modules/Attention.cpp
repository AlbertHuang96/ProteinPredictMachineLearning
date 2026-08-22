#include "ppml/Attention.h"
#include "ppml/Embedding.h"
#include "ppml/Context.h"
#include <cmath>
#include <algorithm>

namespace ppml {

namespace {

// ===== 值版基础操作（值 forward 用；图 helper out_prod/sigmoid/relu/add_impl/mul 等
//       返回图节点 data()=nullptr，值 copy_from 读 null 崩）=====
TensorF32 value_sigmoid(const TensorF32& x) {
    TensorF32 out(x.shape(), x.device());
    const float* xp = x.data();
    float* op = out.data();
    for (int64_t i = 0; i < x.numel(); ++i) op[i] = 1.0f / (1.0f + std::exp(-xp[i]));
    return out;
}
TensorF32 value_relu(const TensorF32& x) {
    TensorF32 out(x.shape(), x.device());
    const float* xp = x.data();
    float* op = out.data();
    for (int64_t i = 0; i < x.numel(); ++i) op[i] = xp[i] > 0.0f ? xp[i] : 0.0f;
    return out;
}
// 同形逐元素乘/加（要求 numel 一致）
TensorF32 value_mul_same(const TensorF32& a, const TensorF32& b) {
    TensorF32 out(a.shape(), a.device());
    const float* ap = a.data();
    const float* bp = b.data();
    float* op = out.data();
    const int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i) op[i] = ap[i] * bp[i];
    return out;
}
TensorF32 value_add_same(const TensorF32& a, const TensorF32& b) {
    TensorF32 out(a.shape(), a.device());
    const float* ap = a.data();
    const float* bp = b.data();
    float* op = out.data();
    const int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i) op[i] = ap[i] + bp[i];
    return out;
}
// 值版 reshape（返回拥有数据的拷贝）。⚠️ 值版禁止 `x = x.view(s)`：view 返回
// own_data_=false 的共享张量，move 赋值先 deallocate() 释放自身数据再接管悬垂指针 → use-after-free。
TensorF32 value_reshape(const TensorF32& x, const Shape& s) {
    if (s.numel() != x.shape().numel())
        throw PPMLError("value_reshape numel mismatch");
    TensorF32 out(s, x.device());
    out.copy_from(x);   // 行优先扁平拷贝，reshape 安全
    return out;
}
// 值版 batched attention：Q,K,V 值布局 (batch, H, D, Lq/Lk)。
// 与图版 out_prod(K,Q)+scale+softmax+out_prod(attn,V) 语义一致。
// bias 可选：值布局 (B, Lk, Lq, H)（batch 沿 fold=NQ/B 广播，对齐 prepare_pair_bias），
//   或已展开 (NQ, H, Lq, Lk)。
TensorF32 value_attention(const TensorF32& Q, const TensorF32& K, const TensorF32& V,
                          const TensorF32* bias) {
    const int64_t NQ = Q.shape().dims[0];
    const int64_t H  = Q.shape().dims[1];
    const int64_t D  = Q.shape().dims[2];
    const int64_t Lq = Q.shape().dims[3];
    const int64_t Lk = K.shape().dims[3];
    TensorF32 scores({NQ, H, Lq, Lk}, Q.device());
    const float* qd = Q.data();
    const float* kd = K.data();
    float* sd = scores.data();
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(D));
    for (int64_t b = 0; b < NQ; ++b)
        for (int64_t h = 0; h < H; ++h)
            for (int64_t i = 0; i < Lq; ++i)
                for (int64_t j = 0; j < Lk; ++j) {
                    float s = 0.0f;
                    const float* qp = qd + ((b * H + h) * D) * Lq + i;
                    const float* kp = kd + ((b * H + h) * D) * Lk + j;
                    for (int64_t d = 0; d < D; ++d) s += qp[d * Lq] * kp[d * Lk];
                    sd[((b * H + h) * Lq + i) * Lk + j] = s * inv_scale;
                }
    if (bias && bias->numel() > 0) {
        if (bias->shape().ndim() == 4 && bias->shape().dims[0] == NQ &&
            bias->shape().dims[1] == H && bias->shape().dims[2] == Lq &&
            bias->shape().dims[3] == Lk) {
            for (int64_t i = 0; i < scores.numel(); ++i) sd[i] += bias->data()[i];
        } else if (bias->shape().ndim() == 4 && bias->shape().dims[1] == Lk &&
                   bias->shape().dims[2] == Lq && bias->shape().dims[3] == H) {
            const int64_t B = bias->shape().dims[0];
            const int64_t fold = NQ / B;
            const float* bd = bias->data();
            for (int64_t b = 0; b < B; ++b)
                for (int64_t f = 0; f < fold; ++f)
                    for (int64_t h = 0; h < H; ++h)
                        for (int64_t i = 0; i < Lq; ++i)
                            for (int64_t j = 0; j < Lk; ++j) {
                                const float bv = bd[((b * Lk + j) * Lq + i) * H + h];  // (B,Lk,Lq,H)
                                sd[(((b * fold + f) * H + h) * Lq + i) * Lk + j] += bv;
                            }
        }
        // 其它形态忽略（原值版亦未处理 bias 广播）
    }
    // softmax 沿 j (Lk)
    for (int64_t b = 0; b < NQ; ++b)
        for (int64_t h = 0; h < H; ++h)
            for (int64_t i = 0; i < Lq; ++i) {
                const int64_t row = ((b * H + h) * Lq + i) * Lk;
                float mx = sd[row];
                for (int64_t j = 1; j < Lk; ++j) if (sd[row + j] > mx) mx = sd[row + j];
                float sum = 0.0f;
                for (int64_t j = 0; j < Lk; ++j) {
                    float e = std::exp(sd[row + j] - mx);
                    sd[row + j] = e; sum += e;
                }
                for (int64_t j = 0; j < Lk; ++j) sd[row + j] /= sum;
            }
    // output[b,h,d,i] = sum_j attn[b,h,i,j] * V[b,h,d,j]
    TensorF32 out({NQ, H, D, Lq}, Q.device());
    const float* vd = V.data();
    float* od = out.data();
    for (int64_t b = 0; b < NQ; ++b)
        for (int64_t h = 0; h < H; ++h)
            for (int64_t d = 0; d < D; ++d)
                for (int64_t i = 0; i < Lq; ++i) {
                    float s = 0.0f;
                    const float* sp = sd + ((b * H + h) * Lq + i) * Lk;
                    const float* vp = vd + ((b * H + h) * D + d) * Lk;
                    for (int64_t j = 0; j < Lk; ++j) s += sp[j] * vp[j];
                    od[((b * H + h) * D + d) * Lq + i] = s;
                }
    return out;
}
// 值版 triangle multiplication（对齐 kernel_tri_mul 语义；pair 方阵 I==J==K==L）：
//   outgoing: dst[b,i,j,d] = (1/L) * sum_k left[b,i,k,d] * right[b,j,k,d]
//   incoming: dst[b,i,j,d] = (1/L) * sum_k left[b,k,i,d] * right[b,k,j,d]
// 带 finite-clamp（与 kernel 一致，防 NaN 沿 pair 链污染）。
TensorF32 value_triangle_mul(const TensorF32& left, const TensorF32& right, float Lf, bool outgoing) {
    const int64_t B = left.shape().dims[0];
    const int64_t I = left.shape().dims[1];
    const int64_t J = left.shape().dims[2];
    const int64_t D = left.shape().dims[3];
    TensorF32 dst({B, I, J, D}, left.device());
    const float* l = left.data();
    const float* r = right.data();
    float* o = dst.data();
    const float inv_L = 1.0f / Lf;
    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < I; ++i)
            for (int64_t j = 0; j < J; ++j) {
                const int64_t dst_off = ((b * I + i) * J + j) * D;
                for (int64_t d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    if (outgoing) {
                        for (int64_t k = 0; k < J; ++k)
                            sum += l[((b * I + i) * J + k) * D + d] * r[((b * J + j) * J + k) * D + d];
                    } else {
                        for (int64_t k = 0; k < I; ++k)
                            sum += l[((b * I + k) * J + i) * D + d] * r[((b * J + k) * J + j) * D + d];
                    }
                    float v = sum * inv_L;
                    if (!(v == v) || v > 1e4f || v < -1e4f)
                        v = (v != v) ? 0.0f : (v > 1e4f ? 1e4f : -1e4f);
                    o[dst_off + d] = v;
                }
            }
    return dst;
}
// 值版外积（特征笛卡尔积）：left (B,..,D1), right (B,..,D2) → dst (B, I, J, D1*D2)
// dst[b,i,j,d1*D2+d2] = left[b,i,d1] * right[b,j,d2]（此处 left/right 均为 (B,L,D) 3D）
TensorF32 value_outer_cartesian(const TensorF32& left, const TensorF32& right) {
    const int64_t B = left.shape().dims[0];
    const int64_t L = left.shape().dims[1];
    const int64_t D1 = left.shape().dims[2];
    const int64_t D2 = right.shape().dims[2];
    TensorF32 dst({B, L, L, D1 * D2}, left.device());
    const float* lp = left.data();
    const float* rp = right.data();
    float* op = dst.data();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < L; ++i)
            for (int64_t j = 0; j < L; ++j)
                for (int64_t d1 = 0; d1 < D1; ++d1) {
                    const float lv = lp[(b * L + i) * D1 + d1];
                    for (int64_t d2 = 0; d2 < D2; ++d2)
                        op[((b * L + i) * L + j) * (D1 * D2) + d1 * D2 + d2] = lv * rp[(b * L + j) * D2 + d2];
                }
    return dst;
}

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
    // 输入约定: Q, K, V 均为 (batch, n_head, D_head, L)。纯值版实现
    // （图版 out_prod/scale/softmax 返回图节点 data()=nullptr，值 copy_from 会崩）。
    return value_attention(Q, K, V, bias);
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
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[MRA] in msa=(%lld,%lld,%lld,%lld) pair_biased=(%lld,%lld,%lld,%lld) Q=(%lld,%lld,%lld,%lld)\n",
        (long long)msa.shape().dims[0], (long long)msa.shape().dims[1], (long long)msa.shape().dims[2], (long long)msa.shape().dims[3],
        (long long)pair_biased.shape().dims[0], (long long)pair_biased.shape().dims[1], (long long)pair_biased.shape().dims[2], (long long)pair_biased.shape().dims[3],
        (long long)Q.shape().dims[0], (long long)Q.shape().dims[1], (long long)Q.shape().dims[2], (long long)Q.shape().dims[3]);
    int B = static_cast<int>(Q.shape().dims[0]);
    int N = static_cast<int>(Q.shape().dims[1]);
    int L = static_cast<int>(Q.shape().dims[2]);
    int H = config_.n_head;
    // D = 每头隐藏维 = 特征维/H（对齐图版；FullBlock 的 msa_full 64 维 → D=8）
    int D = (int)Q.shape().dims[3] / H;
    Q = value_reshape(Q, Shape({B * N, L, H, D}));
    Q = Q.permute({0, 2, 3, 1});
    // (B, H, D, L)
    auto K    = Wk_->forward(msa);
    auto V    = Wv_->forward(msa);
    K = value_reshape(K, Shape({B * N, L, H, D}));
    K = K.permute({0, 2, 3, 1});

    V = value_reshape(V, Shape({B * N, L, H, D}));
    V = V.permute({0, 2, 3, 1});
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[MRA] QKV OK (Q=%lld K=%lld V=%lld)\n",
        (long long)Q.numel(), (long long)K.numel(), (long long)V.numel());
    auto bias = to_b_->forward(pair_biased);  // 遗留：值版 bias 广播未处理（同 Bug3）
    auto gv   = to_g_->forward(msa);
    // 值版 sigmoid + 逐元素门控（图版 sigmoid/out_prod 返回图节点；out_prod 是外积非逐元素）
    TensorF32 gate(gv.shape(), gv.device());
    const float* gvp = gv.data();
    float* gt = gate.data();
    for (int64_t i = 0; i < gv.numel(); ++i) gt[i] = 1.0f / (1.0f + std::exp(-gvp[i]));
    auto attn_out = self_attn_->forward(Q, K, V, &bias);
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[MRA] self_attn OK (%lld,%lld,%lld,%lld)\n",
        (long long)attn_out.shape().dims[0], (long long)attn_out.shape().dims[1], (long long)attn_out.shape().dims[2], (long long)attn_out.shape().dims[3]);
    // (B*N, H, D, L)
    attn_out = attn_out.permute({0, 3, 1, 2});
    attn_out = value_reshape(attn_out, Shape({B, N, L, H * D}));

    TensorF32 gated(attn_out.shape(), attn_out.device());
    const float* ad = attn_out.data();
    float* gd2 = gated.data();
    for (int64_t i = 0; i < attn_out.numel(); ++i) gd2[i] = gt[i] * ad[i];
    return to_out_->forward(gated);
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
    // D = 每头隐藏维 = Q 特征维(H*D)/H。main 块 D=D_MSA=256；full 块(msa_full 64维) D=D_MSA_FULL/H=8。
    int D = (int)Q->shape().dims[0] / H;

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
    auto* gated = mul(gate, merged);   // 门控逐元素（原误用 out_prod 外积，会产生错误形状）
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
    auto Q = Wq_->forward(msa);  // (B, N, L, H*D)
    int B = static_cast<int>(Q.shape().dims[0]);
    int N = static_cast<int>(Q.shape().dims[1]);
    int L = static_cast<int>(Q.shape().dims[2]);
    int H = config_.n_head;      // 8
    // D = 每头隐藏维 = 特征维/H（对齐图版；FullBlock 的 msa_full 64 维 → D=8）
    int D = (int)Q.shape().dims[3] / H;

    // Split heads: (B, N, L, H*D) → (B*L, N, 8, D) → (B*L, 8, D, N)
    Q = value_reshape(Q, Shape({B * L, N, H, D}));
    Q = Q.permute({0, 2, 3, 1});  // (B*L, 8, 256, N) = (batch, n_head, D_head, L_seq)

    auto K = Wk_->forward(msa);
    K = value_reshape(K, Shape({B * L, N, H, D}));
    K = K.permute({0, 2, 3, 1});

    auto V = Wv_->forward(msa);
    V = value_reshape(V, Shape({B * L, N, H, D}));
    V = V.permute({0, 2, 3, 1});

    auto gv   = to_g_->forward(msa);
    // 值版 sigmoid + 逐元素门控（图版 sigmoid/out_prod 返回图节点；out_prod 是外积非逐元素）
    TensorF32 gate(gv.shape(), gv.device());
    const float* gvp = gv.data();
    float* gt = gate.data();
    for (int64_t i = 0; i < gv.numel(); ++i) gt[i] = 1.0f / (1.0f + std::exp(-gvp[i]));

    auto attn_out = self_attn_->forward(Q, K, V);
    // attn_out: (B*L, 8, D, N)

    // Merge heads: (B*L, 8, D, N) → (B*L, N, 8*D) → (B, L, N, H*D) → (B, N, L, H*D)
    attn_out = attn_out.permute({0, 3, 1, 2});      // (B*L, N, 8, D)
    attn_out = value_reshape(attn_out, Shape({B, L, N, H * D})); // (B, L, N, H*D)
    attn_out = attn_out.permute({0, 2, 1, 3});      // (B, N, L, H*D)  恢复原始 dim 顺序

    TensorF32 gated(attn_out.shape(), attn_out.device());
    const float* ad = attn_out.data();
    float* gd2 = gated.data();
    for (int64_t i = 0; i < attn_out.numel(); ++i) gd2[i] = gt[i] * ad[i];
    return to_out_->forward(gated);
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
    auto* gated = mul(gate, merged);   // 门控逐元素（原误用 out_prod 外积，会产生错误形状）
    // 输出投影: H*D → D_MSA, 返回 [D_MSA,L,N,B] = 值 (B,N,L,D_MSA)
    return to_out_->forward_graph(gated);
}

// ===== MSAGlobalColAttention =====
TensorF32 MSAGlobalColAttention::forward(const TensorF32& msa) {
    // Global Col attention: Q 在 N_seq 维取 mean, 然后 1-to-many attention
    // 输入 msa: (B, N, L, D_MSA)。main 块 D=256；full 块(msa_full 64维) D=8。
    auto Q_raw = Wq_->forward(msa);  // (B, N, L, H*D)
    int B = static_cast<int>(Q_raw.shape().dims[0]);
    int N = static_cast<int>(Q_raw.shape().dims[1]);
    int L = static_cast<int>(Q_raw.shape().dims[2]);
    int H = config_.n_head;      // 8
    // D = 每头隐藏维 = 特征维/H（对齐图版 forward_graph；不能用 D_MSA=256，FullBlock 是 64 维）
    int D = (int)Q_raw.shape().dims[3] / H;

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
    // split heads: (B, L, H*D) → (B*L, H, D) → (B*L, H, D, 1)
    Q = value_reshape(Q, Shape({B * L, H, D}));      // (B*L, H, D)
    // 需要变成 4D: (B*L, H, D, 1) 即 (batch, n_head, D_head, L_seq=1)
    Q = value_reshape(Q, Shape({B * L, H, D, 1}));

    // KV: split heads → (B*L, H, D, N)
    auto K = Wk_->forward(msa);
    K = value_reshape(K, Shape({B * L, N, H, D}));
    K = K.permute({0, 2, 3, 1});  // (B*L, H, D, N)

    auto V = Wv_->forward(msa);
    V = value_reshape(V, Shape({B * L, N, H, D}));
    V = V.permute({0, 2, 3, 1});  // (B*L, H, D, N)

    auto gv   = to_g_->forward(msa);
    // 值版 sigmoid（图版 sigmoid 返回图节点 data()=nullptr）
    TensorF32 gate(gv.shape(), gv.device());
    const float* gvp = gv.data();
    float* gt = gate.data();
    for (int64_t i = 0; i < gv.numel(); ++i) gt[i] = 1.0f / (1.0f + std::exp(-gvp[i]));

    // Q: (B*L, H, D, 1), K/V: (B*L, H, D, N) → value_attention 契约
    auto attn_out = self_attn_->forward(Q, K, V);
    // attn_out: (B*L, H, D, 1)

    // Merge heads: (B*L, H, D, 1) → (B, L, H*D) → (B, 1, L, H*D)
    attn_out = value_reshape(attn_out, Shape({B * L, H * D}));  // squeeze last dim → (B*L, H*D)
    attn_out = value_reshape(attn_out, Shape({B, L, H * D}));   // (B, L, H*D)
    // 扩展回 (B, 1, L, H*D) 匹配 gate
    attn_out = value_reshape(attn_out, Shape({B, 1, L, H * D})); // (B, 1, L, H*D)
    // gate: (B, N, L, H*D) × attn_out: (B, 1, L, H*D) → broadcast-mul 沿 dim1（值版 out_prod 语义）
    TensorF32 gated({B, N, L, H * D}, Q_raw.device());
    const float* gd = gate.data();
    const float* ad = attn_out.data();
    float* o = gated.data();
    const int64_t row_elems = (int64_t)L * H * D;
    for (int b = 0; b < B; ++b)
        for (int n = 0; n < N; ++n)
            for (int64_t r = 0; r < row_elems; ++r)
                o[((int64_t)(b * N + n) * row_elems) + r] =
                    gd[((int64_t)(b * N + n) * row_elems) + r] * ad[(int64_t)b * row_elems + r];
    return to_out_->forward(gated);
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
    // D = 每头隐藏维 = Q 特征维(H*D)/H。main 块 D=256；full 块(msa_full 64维) D=8。
    int D = (int)Q->shape().dims[0] / H;

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
    auto* gated = mul(gate, merged);   // 门控逐元素（原误用 out_prod 外积，会产生错误形状）
    // 输出投影: H*D → D_MSA, 返回 [D_MSA,L,N,B] = 值 (B,N,L,D_MSA)
    return to_out_->forward_graph(gated);
}

// ===== PairRowAttention =====
void PairRowAttention::set_params(const AttnConfig& config, LayerNorm* norm,
                                  LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                                  LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv) {
    config_ = config;
    self_attn_ = std::make_unique<SelfAttention>(config);
    norm_   = norm;
    to_b_   = to_b;   to_g_   = to_g;   to_out_ = to_out;
    Wq_     = Wq;     Wk_     = Wk;     Wv_     = Wv;
}

TensorF32 PairRowAttention::forward(const TensorF32& pair, const TensorF32& str_bias) {
    // Row attention: 沿最后一个 L 维 (行) 做 attention
    // 输入 pair: (B, L_row, L_col, 128)
    // AF2 PairAxialAttention: Q/K/V 用归一化 pair，gate(to_g) 用原始 pair。
    // 值版：norm 用 forward_exec，gate 用 value_sigmoid + 逐元素乘（图 helper 返回图节点）。
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[PRA] in pair=(%lld,%lld,%lld,%lld) str_bias=(%lld,%lld,%lld,%lld)\n",
        (long long)pair.shape().dims[0], (long long)pair.shape().dims[1], (long long)pair.shape().dims[2], (long long)pair.shape().dims[3],
        (long long)str_bias.shape().dims[0], (long long)str_bias.shape().dims[1], (long long)str_bias.shape().dims[2], (long long)str_bias.shape().dims[3]);
    TensorF32 pair_normed = norm_->forward_exec(pair);
    auto Q = Wq_->forward(pair_normed);  // (B, L_row, L_col, H*D)
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[PRA] norm+Wq OK Q=(%lld,%lld,%lld,%lld)\n",
        (long long)Q.shape().dims[0], (long long)Q.shape().dims[1], (long long)Q.shape().dims[2], (long long)Q.shape().dims[3]);
    int B   = static_cast<int>(Q.shape().dims[0]);
    int Lr  = static_cast<int>(Q.shape().dims[1]);  // L_row
    int Lc  = static_cast<int>(Q.shape().dims[2]);  // L_col
    int H   = config_.n_head;      // 8
    int D   = D_PAIR_HIDDEN;       // 32

    // Split heads: (B, L_row, L_col, H*D) → (B*L_col, L_row, 8, 32) → (B*L_col, 8, 32, L_row)
    Q = value_reshape(Q, Shape({B * Lc, Lr, H, D}));
    Q = Q.permute({0, 2, 3, 1});  // (B*Lc, 8, 32, Lr)

    auto K = Wk_->forward(pair_normed);
    K = value_reshape(K, Shape({B * Lc, Lr, H, D}));
    K = K.permute({0, 2, 3, 1});

    auto V = Wv_->forward(pair_normed);
    V = value_reshape(V, Shape({B * Lc, Lr, H, D}));
    V = V.permute({0, 2, 3, 1});

    auto bias = to_b_->forward(str_bias);  // (B, Lk, Lq, H) 方阵广播，value_attention 内处理
    auto gv   = to_g_->forward(pair);
    TensorF32 gate = value_sigmoid(gv);

    auto attn_out = self_attn_->forward(Q, K, V, &bias);
    // attn_out: (B*Lc, 8, 32, Lr)

    // Merge heads: (B*Lc, 8, 32, Lr) → (B*Lc, Lr, H*D) → (B, Lr, Lc, H*D)
    attn_out = attn_out.permute({0, 3, 1, 2});            // (B*Lc, Lr, 8, 32) 拥有数据
    attn_out = value_reshape(attn_out, Shape({B, Lc, Lr, H * D})); // (B, Lc, Lr, H*D) 拷贝
    attn_out = attn_out.permute({0, 2, 1, 3});            // (B, Lr, Lc, H*D) 拷贝（permute 是拷贝实现）

    TensorF32 gated = value_mul_same(gate, attn_out);
    return to_out_->forward(gated);
}

// ===== PairRowAttention::forward_graph (图模式) =====
// 与值版 forward 一致，输入输出均为图节点指针（ggml 布局 dims[0]=最内维）
TensorF32* PairRowAttention::forward_graph(TensorF32* pair, TensorF32* str_bias) {
    // 输入: pair 值 (B,Lr,Lc,D_PAIR) = 图 [D_PAIR, Lc, Lr, B]
    //       str_bias 值 (B,Lr,Lc,D_PAIR) = 图 [D_PAIR, Lc, Lr, B]
    // AF2 PairAxialAttention: Q/K/V 用归一化 pair，gate(to_g) 用原始 pair。
    // Wq_->forward_graph(pair_normed) → 值 (B,Lr,Lc,H*D) = 图 [H*D, Lc, Lr, B]
    auto* pair_normed = norm_->forward(pair);
    auto* Q = Wq_->forward_graph(pair_normed);
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
    auto* K = Wk_->forward_graph(pair_normed);
    K = view(K, Shape{D, H, Lr, B * Lc});
    K = permute(K, {2, 0, 1, 3});
    auto* V = Wv_->forward_graph(pair_normed);
    V = view(V, Shape{D, H, Lr, B * Lc});
    V = permute(V, {2, 0, 1, 3});

    // bias: to_b_ D_PAIR→N_HEAD, 值 (B,Lr,Lc,8) = 图 [8,Lc,Lr,B]
    // 规整为 scores 布局 [Lr,Lr,H,B*Lc]（值 (B*Lc,H,Lr,Lr)），并把 batch 沿 Lc 广播。
    // 前置不变量：pair 为方阵（Lr==Lc==L），实际网络里 str_bias 来自 rbf_proj (B,L,L,D_PAIR)。
    auto* bias = to_b_->forward_graph(str_bias);
    bias = prepare_pair_bias(bias, /*Lq=*/Lr, /*Lk=*/Lr, /*B=*/B, /*fold=*/Lc);
    // gate: to_g_ D_PAIR→H*D, 值 (B,Lr,Lc,H*D) = 图 [H*D,Lc,Lr,B]（gate 用原始 pair）
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
    auto* gated = mul(gate, merged);   // 门控逐元素（原误用 out_prod 外积，会产生错误形状）
    // 输出投影: H*D → D_PAIR, 返回 [D_PAIR,Lc,Lr,B] = 值 (B,Lr,Lc,D_PAIR)
    return to_out_->forward_graph(gated);
}

// ===== PairColAttention =====
void PairColAttention::set_params(const AttnConfig& config, LayerNorm* norm,
                                  LinearLayer* to_b,  LinearLayer* to_g,  LinearLayer* to_out,
                                  LinearLayer* Wq,    LinearLayer* Wk,    LinearLayer* Wv) {
    config_ = config;
    self_attn_ = std::make_unique<SelfAttention>(config);
    norm_   = norm;
    to_b_   = to_b;   to_g_   = to_g;   to_out_ = to_out;
    Wq_     = Wq;     Wk_     = Wk;     Wv_     = Wv;
}

TensorF32 PairColAttention::forward(const TensorF32& pair, const TensorF32& str_bias) {
    // Col attention: 沿倒数第二个 L 维 (列) 做 attention
    // 输入 pair: (B, L_row, L_col, 128)
    // AF2 PairAxialAttention: Q/K/V 用归一化 pair，gate(to_g) 用原始 pair。
    // 值版：norm 用 forward_exec，gate 用 value_sigmoid + 逐元素乘（图 helper 返回图节点）。
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[PCA] in pair=(%lld,%lld,%lld,%lld) str_bias=(%lld,%lld,%lld,%lld)\n",
        (long long)pair.shape().dims[0], (long long)pair.shape().dims[1], (long long)pair.shape().dims[2], (long long)pair.shape().dims[3],
        (long long)str_bias.shape().dims[0], (long long)str_bias.shape().dims[1], (long long)str_bias.shape().dims[2], (long long)str_bias.shape().dims[3]);
    TensorF32 pair_normed = norm_->forward_exec(pair);
    auto Q = Wq_->forward(pair_normed);  // (B, L_row, L_col, H*D)
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[PCA] norm+Wq OK Q=(%lld,%lld,%lld,%lld)\n",
        (long long)Q.shape().dims[0], (long long)Q.shape().dims[1], (long long)Q.shape().dims[2], (long long)Q.shape().dims[3]);
    int B   = static_cast<int>(Q.shape().dims[0]);
    int Lr  = static_cast<int>(Q.shape().dims[1]);  // L_row
    int Lc  = static_cast<int>(Q.shape().dims[2]);  // L_col
    int H   = config_.n_head;      // 8
    int D   = D_PAIR_HIDDEN;       // 32

    // Split heads: (B, L_row, L_col, H*D) → (B*L_row, L_col, 8, 32) → (B*L_row, 8, 32, L_col)
    Q = value_reshape(Q, Shape({B * Lr, Lc, H, D}));
    Q = Q.permute({0, 2, 3, 1});  // (B*Lr, 8, 32, Lc)

    auto K = Wk_->forward(pair_normed);
    K = value_reshape(K, Shape({B * Lr, Lc, H, D}));
    K = K.permute({0, 2, 3, 1});

    auto V = Wv_->forward(pair_normed);
    V = value_reshape(V, Shape({B * Lr, Lc, H, D}));
    V = V.permute({0, 2, 3, 1});

    auto bias = to_b_->forward(str_bias);  // (B, Lk, Lq, H) 方阵广播，value_attention 内处理
    auto gv   = to_g_->forward(pair);
    TensorF32 gate = value_sigmoid(gv);
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[PCA] QKV+bias+gate OK (Q=%lld K=%lld V=%lld gate=%lld bias=%lld)\n",
        (long long)Q.numel(), (long long)K.numel(), (long long)V.numel(),
        (long long)gate.numel(), (long long)bias.numel());

    auto attn_out = self_attn_->forward(Q, K, V, &bias);
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[PCA] self_attn OK (%lld,%lld,%lld,%lld)\n",
        (long long)attn_out.shape().dims[0], (long long)attn_out.shape().dims[1], (long long)attn_out.shape().dims[2], (long long)attn_out.shape().dims[3]);
    // attn_out: (B*Lr, 8, 32, Lc)

    // Merge heads: (B*Lr, 8, 32, Lc) → (B*Lr, Lc, H*D) → (B, Lr, Lc, H*D)
    // ⚠️ 不能用 attn_out = attn_out.view(...)：view 不拥有数据，move 赋值先释放自身再接管悬垂指针。
    //    view 须与底层拥有者分离声明，同作用域存活。
    TensorF32 merged = attn_out.permute({0, 3, 1, 2});          // (B*Lr, Lc, 8, 32) 拥有数据
    TensorF32 merged_view = merged.view(Shape({B, Lr, Lc, H * D})); // (B, Lr, Lc, H*D) 非拥有 view
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[PCA] merge OK gate=%lld mv=%lld\n",
        (long long)gate.numel(), (long long)merged_view.numel());

    TensorF32 gated = value_mul_same(gate, merged_view);
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[PCA] gated OK (%lld)\n", (long long)gated.numel());
    TensorF32 result = to_out_->forward(gated);
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[PCA] to_out OK\n");
    return result;
}

// ===== PairColAttention::forward_graph (图模式) =====
// 与值版 forward 一致，输入输出均为图节点指针（ggml 布局 dims[0]=最内维）
TensorF32* PairColAttention::forward_graph(TensorF32* pair, TensorF32* str_bias) {
    // 输入: pair 值 (B,Lr,Lc,D_PAIR) = 图 [D_PAIR, Lc, Lr, B]
    //       str_bias 值 (B,Lr,Lc,D_PAIR) = 图 [D_PAIR, Lc, Lr, B]
    // AF2 PairAxialAttention: Q/K/V 用归一化 pair，gate(to_g) 用原始 pair。
    // Wq_->forward_graph(pair_normed) → 值 (B,Lr,Lc,H*D) = 图 [H*D, Lc, Lr, B]
    auto* pair_normed = norm_->forward(pair);
    auto* Q = Wq_->forward_graph(pair_normed);
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
    auto* K = Wk_->forward_graph(pair_normed);
    K = view(K, Shape{D, H, Lc, B * Lr});
    K = permute(K, {2, 0, 1, 3});
    auto* V = Wv_->forward_graph(pair_normed);
    V = view(V, Shape{D, H, Lc, B * Lr});
    V = permute(V, {2, 0, 1, 3});

    // bias: to_b_ D_PAIR→N_HEAD, 值 (B,Lr,Lc,8) = 图 [8,Lc,Lr,B]
    // 规整为 scores 布局 [Lc,Lc,H,B*Lr]（值 (B*Lr,H,Lc,Lc)），并把 batch 沿 Lr 广播。
    // 前置不变量：pair 为方阵（Lr==Lc==L），实际网络里 str_bias 来自 rbf_proj (B,L,L,D_PAIR)。
    auto* bias = to_b_->forward_graph(str_bias);
    bias = prepare_pair_bias(bias, /*Lq=*/Lc, /*Lk=*/Lc, /*B=*/B, /*fold=*/Lr);
    // gate: to_g_ D_PAIR→H*D, 值 (B,Lr,Lc,H*D) = 图 [H*D,Lc,Lr,B]（gate 用原始 pair）
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
    auto* gated = mul(gate, merged);   // 门控逐元素（原误用 out_prod 外积，会产生错误形状）
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

    // 创建投影层 (无 bias, 与 set_params 注入的外部层同构)
    Wq_ = LinearLayer::create(q_dim_, proj_dim_, false);
    Wk_ = LinearLayer::create(kv_dim_, proj_dim_, false);
    Wv_ = LinearLayer::create(kv_dim_, proj_dim_, false);
    Wo_ = LinearLayer::create(proj_dim_, q_dim_, false);
}

void CrossAttention::set_params(LinearLayer* Wq, LinearLayer* Wk, LinearLayer* Wv, LinearLayer* Wo) {
    if (owns_params_) {
        delete Wq_;
        delete Wk_;
        delete Wv_;
        delete Wo_;
    }
    Wq_ = Wq; Wk_ = Wk; Wv_ = Wv; Wo_ = Wo;
    owns_params_ = false;
}

CrossAttention::~CrossAttention() {
    if (owns_params_) {
        delete Wq_;
        delete Wk_;
        delete Wv_;
        delete Wo_;
    }
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
    Q = value_reshape(Q, Shape({BL, 1, H, D}));
    Q = Q.permute({0, 2, 3, 1});  // (B*L, 8, 8, 1) = (batch, n_head, D_head, L_q=1)

    // K: (B*L, T, 64) → view → (B*L, T, 8, 8) → permute → (B*L, 8, 8, T)
    K = value_reshape(K, Shape({BL, T, H, D}));
    K = K.permute({0, 2, 3, 1});  // (B*L, 8, 8, T)

    // V: 同上
    V = value_reshape(V, Shape({BL, T, H, D}));
    V = V.permute({0, 2, 3, 1});  // (B*L, 8, 8, T)

    // ===== 3-5. Q@K^T / scale / softmax / attn@V（纯值版；图 out_prod/softmax 返回图节点）=====
    // Q (BL,H,D,1), K/V (BL,H,D,T) 与 value_attention 输入布局一致
    TensorF32 output = value_attention(Q, K, V, nullptr);  // (BL, H, D, 1)

    // ===== 6. Merge heads =====
    // output: (BL, H, D, 1) → permute({0,3,1,2}) → (BL, 1, H, D) → view → (BL, 1, PD)
    TensorF32 merged = output.permute({0, 3, 1, 2});  // (BL, 1, H, D) 拥有数据
    TensorF32 merged_v = merged.view(Shape({BL, 1, PD}));     // (BL, 1, PD) 非拥有 view

    // ===== 7. 输出投影: PD → q_dim =====
    auto result = Wo_->forward(merged_v);  // (B*L, 1, q_dim)
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
    // 值版：norm 用 forward_exec，gate 用 value_sigmoid，tri_mul 用 value_triangle_mul
    // （图版 layernorm_->forward/sigmoid/mul/triangle_mul 返回图节点 data()=nullptr）。
    TensorF32 pair_norm = layernorm_->forward_exec(pair);
    auto left  = left_proj_->forward(pair_norm);
    auto right = right_proj_->forward(pair_norm);
    auto lgv   = left_gate_->forward(pair_norm);
    auto rgv   = right_gate_->forward(pair_norm);
    TensorF32 left_gate  = value_sigmoid(lgv);
    TensorF32 right_gate = value_sigmoid(rgv);
    // gate 为逐元素缩放（非 out_prod 收缩），与 forward_graph 保持一致。
    TensorF32 left_gated  = value_mul_same(left, left_gate);
    TensorF32 right_gated = value_mul_same(right, right_gate);

    TensorF32 tri_mul_forward = value_triangle_mul(left_gated, right_gated,
                                                   float(pair.shape().dims[1]), bOutgoing);

    TensorF32 tri_mul_forward_norm = output_layernorm_->forward_exec(tri_mul_forward);
    TensorF32 tri_mul_forward_proj = out_proj_->forward(tri_mul_forward_norm);

    auto gv = gate_->forward(pair_norm);
    TensorF32 gate_out = value_sigmoid(gv);
    return value_mul_same(gate_out, tri_mul_forward_proj);
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
    // gate 应为逐元素缩放（AF2: left * left_gate），而非 out_prod 收缩（会把残基维缩掉，
    // 使 tri_mul 收到非标准 [16,16,L,B]，导致维度不一致）。→ 改用逐元素 mul。
    auto left_gated  = mul(left, left_gate);
    auto right_gated = mul(right, right_gate);

    auto tri_mul_forward = triangle_mul(left_gated, right_gated, float(pair->shape().dims[1]), bOutgoing);

    auto tri_mul_forward_norm = output_layernorm_->forward(tri_mul_forward);
    auto tri_mul_forward_proj = out_proj_->forward_graph(tri_mul_forward_norm);

    auto gv = gate_->forward_graph(pair_norm);
    auto gate_out = sigmoid(gv);
    auto tri_mul_forward_gated = mul(gate_out, tri_mul_forward_proj);

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
    // 值版：norm 用 forward_exec，relu 用 value_relu（图版 forward/relu 返回图节点）
    TensorF32 x_norm    = layernorm_->forward_exec(x);
    TensorF32 x_hidden  = linear1_->forward(x_norm);
    TensorF32 x_relu    = value_relu(x_hidden);
    TensorF32 x_dropped = dropout_.forward(x_relu);
    return linear2_->forward(x_dropped);
}

// ===== FeedForward::forward_graph (图模式) =====
// 输入输出均为图节点指针。dropout 用 Dropout::forward_graph (mask 常量叶子 + mul) 实现。
TensorF32* FeedForward::forward_graph(TensorF32* x) {
    auto x_norm    = layernorm_->forward(x);              // TensorF32*
    auto x_hidden  = linear1_->forward_graph(x_norm);     // TensorF32*
    auto x_relu    = relu(x_hidden);                       // relu 返回指针
    auto x_dropped = dropout_.forward_graph(x_relu);      // 图 dropout
    auto x_out     = linear2_->forward_graph(x_dropped);
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
    // 值版：norm 用 forward_exec，gate 用 value_outer_cartesian/value_sigmoid，
    //       残差用 value_add_same（图 helper out_prod/sigmoid/add_impl 返回图节点）。
    TensorF32 rbf_proj = rbf_proj_->forward(rbf_feature);  // (B,L,L,128)

    TensorF32 state_normed = state_norm_->forward_exec(state);

    TensorF32 left  = left_proj_->forward(state_normed);   // (B,L,16)
    TensorF32 right = right_proj_->forward(state_normed);  // (B,L,16)
    TensorF32 gate  = value_outer_cartesian(left, right);   // (B,L,L,256)
    TensorF32 gate_proj = gate_proj_->forward(gate);       // (B,L,L,128)
    TensorF32 gate_sig  = value_sigmoid(gate_proj);
    rbf_feature = value_mul_same(rbf_feature, gate_sig);   // 引用更新（对齐图版）

    TensorF32 pair_tmp;
    pair_tmp.copy_from(pair);

    TensorF32 tri_out = tri_mul_out_->forward(pair_tmp, true);
    TensorF32 tri_out_drop = drop_row_.forward(tri_out);
    pair_tmp = value_add_same(pair_tmp, tri_out_drop);

    TensorF32 tri_in = tri_mul_in_->forward(pair_tmp, false);
    TensorF32 tri_in_drop = drop_row_.forward(tri_in);
    pair_tmp = value_add_same(pair_tmp, tri_in_drop);

    TensorF32 row_attn = pair_row_attn_->forward(pair_tmp, rbf_proj);
    TensorF32 row_drop = drop_row_.forward(row_attn);
    pair_tmp = value_add_same(pair_tmp, row_drop);

    TensorF32 col_attn = pair_col_attn_->forward(pair_tmp, rbf_proj);
    TensorF32 col_drop = drop_col_.forward(col_attn);
    pair_tmp = value_add_same(pair_tmp, col_drop);

    TensorF32 pair_ff_out = pair_ff_->forward(pair_tmp);
    return value_add_same(pair_tmp, pair_ff_out);
}

// ===== TemplatePairStack::forward_graph (图模式) =====
// 与值版 forward 逻辑一致，输入输出均为图节点指针（ggml 布局 dims[0]=最内维）。
// rbf_feature 以引用传递并在内部 gate 后更新，供多次迭代（值版 tps_.forward 用引用回写）。
TensorF32* TemplatePairStack::forward_graph(TensorF32* pair, TensorF32*& rbf_feature, TensorF32* state) {
    // rbf_proj = rbf_proj_(rbf_feature)  [128,L,L,B*T]（用 gate 前的 rbf_feature）
    TensorF32* rbf_proj = rbf_proj_->forward_graph(rbf_feature);

    // gate = sigmoid(gate_proj(outer_product(state_normed)))
    TensorF32* state_normed = state_norm_->forward(state);          // [D_STATE,L,B*T]
    TensorF32* left  = left_proj_->forward_graph(state_normed);     // [16,L,B*T]
    TensorF32* right = right_proj_->forward_graph(state_normed);    // [16,L,B*T]
    TensorF32* gate  = outer_product_graph(left, right);            // [256,L,L,B*T]
    gate = gate_proj_->forward_graph(gate);                          // [128,L,L,B*T]
    gate = sigmoid(gate);                                            // [0,1]
    // 更新 rbf_feature（引用）: rbf_feature = rbf_feature * gate
    rbf_feature = mul(rbf_feature, gate);

    // 残差链: tri_mul_out → tri_mul_in → row_attn → col_attn → ff
    TensorF32* pair_tmp = pair;
    TensorF32* tri_out = tri_mul_out_->forward_graph(pair_tmp, /*bOutgoing=*/true);
    TensorF32* tri_out_drop = drop_row_.forward_graph(tri_out);
    pair_tmp = add_impl(pair_tmp, tri_out_drop, /*inplace=*/false);

    TensorF32* tri_in = tri_mul_in_->forward_graph(pair_tmp, /*bOutgoing=*/false);
    TensorF32* tri_in_drop = drop_row_.forward_graph(tri_in);
    pair_tmp = add_impl(pair_tmp, tri_in_drop, /*inplace=*/false);

    TensorF32* row_attn = pair_row_attn_->forward_graph(pair_tmp, rbf_proj);
    TensorF32* row_drop = drop_row_.forward_graph(row_attn);
    pair_tmp = add_impl(pair_tmp, row_drop, /*inplace=*/false);

    TensorF32* col_attn = pair_col_attn_->forward_graph(pair_tmp, rbf_proj);
    TensorF32* col_drop = drop_col_.forward_graph(col_attn);
    pair_tmp = add_impl(pair_tmp, col_drop, /*inplace=*/false);

    TensorF32* pair_ff_out = pair_ff_->forward_graph(pair_tmp);
    return add_impl(pair_tmp, pair_ff_out, /*inplace=*/false);
}

} // namespace ppml

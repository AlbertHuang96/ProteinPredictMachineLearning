#include "ppml/Model.h"
#include "ppml/Embedding.h"
#include "ppml/MathUtils.h"
#include <random>
#include <vector>

#include "ppml/ComputeGraph.h"
#include "ppml/Context.h"

namespace ppml {

// Embedding 层实现


    EmbeddingLayer* EmbeddingLayer::create(int num_embeddings, int embedding_dim) {
        auto* layer = new EmbeddingLayer();
        layer->num_embeddings_ = num_embeddings;
        layer->embedding_dim_  = embedding_dim;

        // 布局改为 (V, D) = (num_embeddings, embedding_dim)，每行是一个 token 的向量，
        // 以匹配 get_rows(W, idx) 的"按行取"语义（get_rows 沿 dim[0] 取行）
        int64_t dims[] = {num_embeddings, embedding_dim};  // (V, D)
        layer->weights_ = context().new_param_tensor<float>(2, dims);
        layer->weights_->flag = TENSOR_FLAG_PARAM;

        // Xavier init
        std::random_device rd;
        std::mt19937 gen(rd());
        float scale = sqrtf(2.0f / (num_embeddings + embedding_dim));
        std::normal_distribution<float> dist(0.0f, scale);
        for (int i = 0; i < num_embeddings * embedding_dim; ++i)
            layer->weights_->data()[i] = dist(gen);

        return layer;
    }

    // ===== 图模式：get_rows =====
    TensorF32* EmbeddingLayer::forward_graph(TensorF32* indices) {
        // embedding 本质是 get_rows(weight, indices)
        // weight: (V, D), indices: (K,) → output: (K, D)
        // 反向由 OP_GET_ROWS → get_rows_back 自动展开
        return get_rows(weights_, indices);
    }

    // TODO
    // all the forward_exec need to be removed
    // forward
    // backward
    TensorF32 EmbeddingLayer::forward_exec(const TensorF32& indices) {
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
                        weights_->data() + idx * embedding_dim_,
                        embedding_dim_ * sizeof(float)
                    );
                }
            }
        }
        // CUDA 实现应调用 kernel
        
        return output;
    }

    // weight() 已在 Embedding.h 中 inline 定义




    // 工厂函数：从 context 分配权重
    LinearLayer* LinearLayer::create(int in_features, int out_features, bool bias, bool se3) {
        LinearLayer* layer = new LinearLayer();
        layer->in_features_  = in_features;
        layer->out_features_ = out_features;
        layer->has_bias_     = bias;

        // ===== 从全局 context 分配权重 =====
        int64_t w_dims[] = {in_features, out_features};  // (in, out)
        layer->weight_ = context().new_param_tensor<float>(2, w_dims);
        layer->weight_->flag = TENSOR_FLAG_PARAM;  // ← 标记为可训练
        if (se3) layer->weight_->flag |= TENSOR_FLAG_SE3;  // SE3 分层 lr

        if (bias) {
            int64_t b_dims[] = {out_features};
            layer->bias_ = context().new_param_tensor<float>(1, b_dims);
            layer->bias_->flag = TENSOR_FLAG_PARAM | TENSOR_FLAG_NO_WEIGHT_DECAY;  // ← 可训练但不做 weight decay
            if (se3) layer->bias_->flag |= TENSOR_FLAG_SE3;
        }

        // ===== Xavier 初始化 =====
        layer->init_weights();
        return layer;
    }

    // zeros_weight / ones_bias 已在 Embedding.h 中 inline 定义，此处不重复

    // ===== 图模式前向（训练用）=====
    // x 可为任意维（图布局 dims[0]=最内特征维）：[D_in, d1, d2, d3]
    // 由于 mul_mat/kernel 只支持 2D，这里把 batch 维展平为 [D_in, M] 做矩阵乘，
    // 再把 bias 广播为 [D_out, M] 逐元素相加，最后 view 还原为 [D_out, d1, d2, d3]。
    TensorF32* LinearLayer::forward_graph(TensorF32* x) {
        const int x_ndim = x->shape().ndim();
        // 展平 batch 维（除最内特征维外）为 M
        int64_t M = 1;
        for (int i = 1; i < x_ndim; i++) M *= x->shape().dims[i];

        // 统一 view 成 2D [D_in, M]（1D [D_in] 视为 M=1），再 mul_mat → [D_out, M]
        TensorF32* x_flat = (x_ndim == 2) ? x : view(x, Shape{in_features_, M});
        TensorF32* y = mul_mat(x_flat, weight_);                  // [D_out, M] = W0 @ x

        // LoRA 低秩旁路: y += (alpha/rank) * B @ A @ x
        if (lora_A_ && lora_B_) {
            TensorF32* z  = mul_mat(x_flat, lora_A_);            // [rank, M]  = Aᵀx
            TensorF32* p  = mul_mat(z,     lora_B_);            // [out, M]  = Bᵀ(Aᵀx)
            const float lora_s = lora_alpha_ / lora_rank_;       // * α/r
            TensorF32* ps = scale(p, lora_s);
            y = add_impl(y, ps, /*inplace=*/false);            // 并联相加
        }

        if (has_bias_) {
            // bias [D_out] 广播到 [D_out, M]（kernel_repeat 尾部对齐，1D→2D）
            int64_t btgt[2] = {out_features_, M};
            TensorF32* btarget = context().new_tensor<float>(2, btgt);
            TensorF32* bias_br = repeat(bias_, btarget);
            y = add_impl(y, bias_br, /*inplace=*/false);  // 逐元素（形状相同）
        }

        // 还原多维形状 [D_out, d1, d2, d3]
        if (x_ndim > 2) {
            std::vector<int64_t> out_dims;
            out_dims.push_back(out_features_);
            for (int i = 1; i < x_ndim; i++) out_dims.push_back(x->shape().dims[i]);
            y = view(y, Shape(out_dims));
        }
        return y;
    }
    
    TensorF32 LinearLayer::forward(const TensorF32& x) {
        // x: (..., in_features)
        // output: (..., out_features)
        
        int batch = x.shape().numel() / in_features_;
        // ⚠️ 直接按 (..., out_features) 分配并返回**拥有数据**的张量。
        // 旧实现先分配 (batch,out) 再 return output.view(out_shape)（own_data_=false），
        // 局部 output 析构释放数据 → 返回悬垂指针（use-after-free，值版 forward 崩溃根因）。
        auto out_shape = x.shape();
        out_shape.dims.back() = out_features_;
        TensorF32 output(out_shape, x.device());
        
        // 简化的矩阵乘法 (实际应使用 cuBLAS)
        for (int b = 0; b < batch; ++b) {
            for (int o = 0; o < out_features_; ++o) {
                float sum = has_bias_ ? bias_->data()[o] : 0.0f;
                // dim of weights: (out_features, in_features)
                for (int i = 0; i < in_features_; ++i) {
                    sum += x.data()[b * in_features_ + i] * weight_->data()[o * in_features_ + i];
                }
                // LoRA 旁路: + (alpha/rank) * Σ_r B[o][r] * (Σ_i A[r][i] * x[b][i])
                //   A 存 [in,rank] → A[r][i] flat = i + r*in_features_
                //   B 存 [rank,out] → B[o][r] flat = r + o*lora_rank_
                if (lora_A_ && lora_B_) {
                    const float scale = lora_alpha_ / lora_rank_;
                    float z = 0.f;
                    for (int r = 0; r < lora_rank_; ++r) {
                        float acc = 0.f;
                        for (int i = 0; i < in_features_; ++i) {
                            acc += x.data()[b * in_features_ + i]
                                 * lora_A_->data()[i + r * in_features_];
                        }
                        z += lora_B_->data()[r + o * lora_rank_] * acc;
                    }
                    sum += scale * z;
                }
                // dim of output: (..., out_features)，行优先扁平索引与 (batch,out) 一致
                output.data()[b * out_features_ + o] = sum;
            }
        }
        
        return output;
    }

    // weight() / bias() 已在 Embedding.h 中 inline 定义
    

    void LinearLayer::init_weights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        float scale = sqrtf(2.0f / in_features_);
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (int i = 0; i < out_features_ * in_features_; ++i)
            weight_->data()[i] = dist(gen);

        if (has_bias_) {
            std::memset(bias_->data(), 0, out_features_ * sizeof(float));
        }
    }

    // ===== LoRA 低秩微调 =====
    // A 存 [in, rank]（dims[0]=in 最内，与 weight_ 同布局），B 存 [rank, out]。
    // mul_mat(a,b)=aᵀ@b，b 的 K=dims[0]、N=dims[1]：
    //   mul_mat(x[in,M], A[in,rank]) → [rank,M]   （A 的 K=in,N=rank）
    //   mul_mat(z[rank,M], B[rank,out]) → [out,M] （B 的 K=rank,N=out）
    void LinearLayer::enable_lora(int rank, float alpha) {
        if (rank <= 0 || lora_A_) return;   // 已启用则幂等
        lora_rank_  = rank;
        lora_alpha_ = alpha;
        int64_t a_dims[2] = {in_features_, rank};
        int64_t b_dims[2] = {rank, out_features_};
        lora_A_ = context().new_param_tensor<float>(2, a_dims);
        lora_B_ = context().new_param_tensor<float>(2, b_dims);
        lora_A_->flag = TENSOR_FLAG_PARAM | TENSOR_FLAG_LORA;
        lora_B_->flag = TENSOR_FLAG_PARAM | TENSOR_FLAG_LORA;
        lora_init();
        // 启用 LoRA 后默认冻结主权重（仅训旁路）
        freeze();
    }

    void LinearLayer::lora_init() {
        // 标准 LoRA 初始化：A ~ N(0, 0.02)，B = 0 → 初始 B@A=0，前向=原 W0，不破坏预训练
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (int64_t i = 0; i < lora_A_->numel(); ++i)
            lora_A_->data()[i] = dist(gen);
        std::memset(lora_B_->data(), 0, lora_B_->numel() * sizeof(float));
    }
   

// LayerNorm


    LayerNorm* LayerNorm::create(int normalized_shape, float eps) {
        auto* layer = new LayerNorm();
        layer->normalized_shape_ = normalized_shape;
        layer->eps_ = eps;

        int64_t dims[] = {normalized_shape};
        layer->gamma_ = context().new_param_tensor<float>(1, dims);
        layer->beta_  = context().new_param_tensor<float>(1, dims);
        layer->gamma_->flag = TENSOR_FLAG_PARAM | TENSOR_FLAG_NO_WEIGHT_DECAY;
        layer->beta_->flag  = TENSOR_FLAG_PARAM | TENSOR_FLAG_NO_WEIGHT_DECAY;

        // gamma = 1, beta = 0
        for (int i = 0; i < normalized_shape; i++)
            layer->gamma_->data()[i] = 1.0f;
        std::memset(layer->beta_->data(), 0, normalized_shape * sizeof(float));

        return layer;
    }

    TensorF32* LayerNorm::forward(TensorF32* x) {
        // y = norm(x) * gamma + beta
        // 图张量 ggml 布局 dims[0]=特征维；norm() 沿 dims[0] 归一化。
        // gamma/beta 为 [D]，需 view 成 [D,1,1] 再 repeat 到 x 的完整形状，
        // 然后逐元素 mul/add（不能用 out_prod：那是外积，会产生错误形状）。
        TensorF32* normed = norm(x, eps_);  // or norm() for layernorm
        int D = (int)normed->shape().dims[0];
        int nd = normed->shape().ndim();
        // gamma/beta [D] → view 成与 normed 同 ndim 的 [D,1,...,1]（尾部对齐 repeat 才正确）。
        // 原固定 [D,1,1]（3D）对 4D 输入（如 pair [D,L,L,B]）跨 ndim 尾部对齐会把 D 对齐到 L，
        // 导致 repeat 语义错乱（甚至巨大 shape）。
        TensorF32* gamma_v;
        TensorF32* beta_v;
        if (nd == 2)      { gamma_v = view(gamma_, Shape({D, 1}));       beta_v = view(beta_, Shape({D, 1})); }
        else if (nd == 3) { gamma_v = view(gamma_, Shape({D, 1, 1}));    beta_v = view(beta_, Shape({D, 1, 1})); }
        else              { gamma_v = view(gamma_, Shape({D, 1, 1, 1})); beta_v = view(beta_, Shape({D, 1, 1, 1})); }
        TensorF32* scaled  = mul(normed, repeat(gamma_v, normed));
        return add_impl(scaled, repeat(beta_v, scaled), /*inplace=*/false);
    }
    
    TensorF32 LayerNorm::forward_exec(const TensorF32& x) {
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
                    normalized * gamma_->data()[i] + beta_->data()[i];
            }
        }
        
        return output;
    }
    



    // ===== BondEmbedding (non-owning pointer 版本) =====
    // 旧构造函数 (保留注释):
    // BondEmbedding::BondEmbedding(int d_init, int d_pair)
    //     : d_init_(d_init), d_pair_(d_pair) {
    //     if (d_init_ == 0) d_init_ = NBYTES;
    //     emb_ = LinearLayer(d_init_, d_pair_);
    // }

    void BondEmbedding::set_params(LinearLayer* emb, int d_pair) {
        emb_ = emb;
        d_pair_ = d_pair;
    }

    //ChemData().NBTYPES represents the number of categorical bond types the model recognizes, 
    //and its value is 8
    TensorF32 BondEmbedding::forward(const TensorF32& bond_feats) {
        // bond_feats: (B, L, L)（每对残基一个 bond 类型整数）或 (B, L, L, d_init)
        // output: (B, L, L, d_pair)
        int B = bond_feats.shape().dims[0];
        int L = bond_feats.shape().dims[1];
        const int64_t n = bond_feats.numel();   // 总位置数（B*L*L 或 B*L*L*d_init）
        const int64_t rows = n / static_cast<int64_t>(L * L);  // 位置数（每行一个 bond 类型）
        
        // ⚠️ 值版修复（2026-08-22）：bond_feats 实际是 3D (B,L,L)，不能直接喂 one_hot_seq（期望 2D）。
        // 扁平化所有位置为 1D，对每个位置做 one-hot (NBYTES)，再 reshape 回 (B,L,L,NBYTES)。
        TensorF32 flat(Shape({rows * L * L}), bond_feats.device());  // 位置扁平
        flat.copy_from(bond_feats);   // 前 n 元素即各位置值（若 4D 输入取前 B*L*L 个）
        TensorF32 bond_onehot(Shape({B, L, L, NBYTES}), bond_feats.device());
        bond_onehot.zero_();
        const float* fd = flat.data();
        float* od = bond_onehot.data();
        for (int64_t r = 0; r < rows * L * L; ++r) {
            int idx = static_cast<int>(fd[r]);
            if (idx < 0 || idx >= NBYTES) idx = 0;  // 防御：越界 bond 类型置 0
            od[r * NBYTES + idx] = 1.0f;
        }
        TensorF32 output = emb_->forward(bond_onehot); // (B, L, L, d_pair)
        
        return output;
    }



    // ===== FullEmbedding (non-owning pointer 版本) =====
    // 旧构造函数 (保留注释):
    // FullEmbedding::FullEmbedding(int d_init, int d_msa) : d_init_(d_init), d_msa_(d_msa) {
    //     if (d_init_ == 0) d_init_ = NAATOKENS - 1 + 4;
    //     emb_ = LinearLayer(d_init_, d_msa_);
    //     emb_q_ = EmbeddingLayer(NAATOKENS, d_msa_);
    // }

    void FullEmbedding::set_params(LinearLayer* emb, EmbeddingLayer* emb_q, int d_msa) {
        emb_   = emb;
        emb_q_ = emb_q;
        d_msa_ = d_msa;
    }
    
    TensorF32 FullEmbedding::forward(const TensorF32& msa, const TensorF32& seq, const TensorF32& idx) {
        // msa: (B, N, L, d_init)
        // seq: (B, L)
        // idx: (B, L)
        
        int B = msa.shape().dims[0];
        int N = msa.shape().dims[1];
        int L = msa.shape().dims[2];
        
        // MSA embedding
        // 旧: TensorF32 msa_emb = emb_.forward(msa);
        TensorF32 msa_emb = emb_->forward(msa);
        
        // Query embedding
        // 旧: TensorF32 seq_emb = emb_q_.forward(seq).unsqueeze(1);
        TensorF32 seq_emb = emb_q_->forward_exec(seq).unsqueeze(1); // (B, 1, L, d_msa_)
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

    // ===== FullEmbedding::forward_graph (图模式) =====
    // 广播 query: msa_emb + repeat(seq_emb)  (B,N,L,d_msa)
    // 图节点采用 ggml 布局 (dims[0]=最内维):
    //   msa_emb : [d_msa, L, N, B]  (emb_->forward_graph 输出, mul_mat)
    //   seq_emb : [d_msa, B*L]      (emb_q_->forward_graph 输出, get_rows, 一维扁平索引)
    // 广播路径: seq_emb -> view [d_msa, L, 1, B] -> repeat 到 msa_emb 形状 -> add
    // ⚠️ 依赖: view / repeat / add_impl 图 op。其中 view 图节点当前缺 dispatch kernel，
    //    需先补 view/unsqueeze kernel 才能执行。布局以 PPMLModel 图化后的上游约定为准。
    TensorF32* FullEmbedding::forward_graph(TensorF32* msa, TensorF32* seq, TensorF32* idx) {
        // msa : (B, N, L, d_init) → emb 线性投影
        auto* msa_emb = emb_->forward_graph(msa);              // [d_msa, L, N, B]

        // seq : (B, L) 整数索引 → query 行嵌入 (get_rows 按行取, 输出 [d_msa, B*L])
        auto* seq_emb = emb_q_->forward_graph(seq);            // [d_msa, B*L]

        // 广播: seq_emb [d_msa, B*L] → view [d_msa, L, 1, B] → repeat → [d_msa, L, N, B]
        // 注: 依赖上游 msa_emb 的 ggml 布局 (dims[1]=L, dims[3]=B), 需与 PPMLModel 图化一致
        const int64_t L = msa_emb->shape().dims[1];
        const int64_t B = msa_emb->shape().dims[3];
        auto* seq_b     = view(seq_emb, Shape{seq_emb->shape().dims[0], L, 1, B});  // [d_msa, L, 1, B]
        auto* seq_expand = repeat(seq_b, msa_emb);             // 广播到 msa_emb 形状 [d_msa, L, N, B]

        // msa_emb + broadcast(query) (inplace 不改输入)
        auto* out = add_impl(msa_emb, seq_expand, /*inplace=*/false);
        (void)idx;
        return out;
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


} // namespace ppml

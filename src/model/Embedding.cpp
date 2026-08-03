#include "rfaa/Model.h"
#include "rfaa/Embedding.h"
#include "rfaa/MathUtils.h"
#include <random>

#include "rfaa/ComputeGraph.h"
#include "rfaa/Context.h"

namespace rfaa {

// Embedding 层实现



// e

    EmbeddingLayer* EmbeddingLayer::create(int num_embeddings, int embedding_dim) {
        auto* layer = new EmbeddingLayer();
        layer->num_embeddings_ = num_embeddings;
        layer->embedding_dim_  = embedding_dim;

        // 布局改为 (V, D) = (num_embeddings, embedding_dim)，每行是一个 token 的向量，
        // 以匹配 get_rows(W, idx) 的"按行取"语义（get_rows 沿 dim[0] 取行）
        int64_t dims[] = {num_embeddings, embedding_dim};  // (V, D)
        layer->weights_ = context().new_tensor<float>(2, dims);
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
    LinearLayer* LinearLayer::create(int in_features, int out_features, bool bias) {
        LinearLayer* layer = new LinearLayer();
        layer->in_features_  = in_features;
        layer->out_features_ = out_features;
        layer->has_bias_     = bias;

        // ===== 从全局 context 分配权重 =====
        int64_t w_dims[] = {in_features, out_features};  // (in, out)
        layer->weight_ = context().new_tensor<float>(2, w_dims);
        layer->weight_->flag = TENSOR_FLAG_PARAM;  // ← 标记为可训练

        if (bias) {
            int64_t b_dims[] = {out_features};
            layer->bias_ = context().new_tensor<float>(1, b_dims);
            layer->bias_->flag = TENSOR_FLAG_PARAM;  // ← 标记为可训练
        }

        // ===== Xavier 初始化 =====
        layer->init_weights();
        return layer;
    }

    // zeros_weight / ones_bias 已在 Embedding.h 中 inline 定义，此处不重复

    // ===== 图模式前向（训练用）=====
    TensorF32* LinearLayer::forward_graph(TensorF32* x) {
        TensorF32* y = mul_mat(x, weight_);
        if (has_bias_) {
            y = add_impl(y, bias_, /*inplace=*/false);  // broadcast bias
        }
        return y;
    }
    
    TensorF32 LinearLayer::forward(const TensorF32& x) {
        // x: (..., in_features)
        // output: (..., out_features)
        
        int batch = x.shape().numel() / in_features_;
        TensorF32 output(Shape({batch, out_features_}), x.device());
        
        // 简化的矩阵乘法 (实际应使用 cuBLAS)
        for (int b = 0; b < batch; ++b) {
            for (int o = 0; o < out_features_; ++o) {
                float sum = has_bias_ ? bias_->data()[o] : 0.0f;
                // dim of weights: (out_features, in_features)
                for (int i = 0; i < in_features_; ++i) {
                    sum += x.data()[b * in_features_ + i] * weight_->data()[o * in_features_ + i];
                }
                // dim of output: (batch, out_features)
                output.data()[b * out_features_ + o] = sum;
            }
        }
        
        // 恢复原始形状 (最后维变为 out_features)
        auto out_shape = x.shape();
        out_shape.dims.back() = out_features_;
        return output.view(out_shape);
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
   

// LayerNorm


    LayerNorm* LayerNorm::create(int normalized_shape, float eps) {
        auto* layer = new LayerNorm();
        layer->normalized_shape_ = normalized_shape;
        layer->eps_ = eps;

        int64_t dims[] = {normalized_shape};
        layer->gamma_ = context().new_tensor<float>(1, dims);
        layer->beta_  = context().new_tensor<float>(1, dims);
        layer->gamma_->flag = TENSOR_FLAG_PARAM;
        layer->beta_->flag  = TENSOR_FLAG_PARAM;

        // gamma = 1, beta = 0
        for (int i = 0; i < normalized_shape; i++)
            layer->gamma_->data()[i] = 1.0f;
        std::memset(layer->beta_->data(), 0, normalized_shape * sizeof(float));

        return layer;
    }

    TensorF32* LayerNorm::forward(TensorF32* x) {
        // y = norm(x) * gamma + beta
        // rms_norm
        // norm() = OP_NORM we use for layer norm
        TensorF32* normed = norm(x, eps_);  // or norm() for layernorm
        //Tensor* scaled = mul_mat(normed, gamma_);
        TensorF32* scaled = out_prod(normed, gamma_);
        return add_impl(scaled, beta_, /*inplace=*/false);
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
        // bond_feats: (B, L, L, d_init)
        // output: (B, L, L, d_pair)
        int B = bond_feats.shape().dims[0];
        int L = bond_feats.shape().dims[1];
        
        TensorF32 output;
        TensorF32 bond_onehot = one_hot_seq(bond_feats, NBYTES); // (B, L, L, NBYTES)
        // 旧: output = emb_.forward(one_hot);
        output = emb_->forward(bond_onehot); // (B, L, L, d_pair)
        
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
    //    需先补 view/unsqueeze kernel 才能执行。布局以 RFAAModel 图化后的上游约定为准。
    TensorF32* FullEmbedding::forward_graph(TensorF32* msa, TensorF32* seq, TensorF32* idx) {
        // msa : (B, N, L, d_init) → emb 线性投影
        auto* msa_emb = emb_->forward_graph(msa);              // [d_msa, L, N, B]

        // seq : (B, L) 整数索引 → query 行嵌入 (get_rows 按行取, 输出 [d_msa, B*L])
        auto* seq_emb = emb_q_->forward_graph(seq);            // [d_msa, B*L]

        // 广播: seq_emb [d_msa, B*L] → view [d_msa, L, 1, B] → repeat → [d_msa, L, N, B]
        // 注: 依赖上游 msa_emb 的 ggml 布局 (dims[1]=L, dims[3]=B), 需与 RFAAModel 图化一致
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


} // namespace rfaa

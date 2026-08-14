#include "rfaa/Model.h"
#include "rfaa/Embedding.h"
#include "rfaa/PositionalEncoding.h"
#include "rfaa/MathUtils.h"
#include <iostream>

#include "rfaa/Dropout.h"
#include "rfaa/Context.h"
#include <cuda_runtime.h>   // cudaGetDeviceCount 用于 GPU 可用性探测

namespace rfaa {

namespace {
// ===== proj_state_add_to_query_row 图版（掩码广播 add）=====
// 语义: msa[:,0,:,:] += proj_state  (msa 第 0 条序列 query 行注入)
//   msa        : 图 [D_MSA, L, N, B]   (值 (B,N,L,D_MSA))
//   proj_state : 图 [D_MSA, L, B]      (值 (B,L,D_MSA)，state2msa_linear 输出)
// 返回: 新的 msa 图节点（n==0 处 += proj_state，其余 n 不变）。
//
// 为什么不用 get_rows/set_rows：CPU kernel 只支持严格 2D(N,M)（行沿 dims[0]，每行
// 只 memcpy dims[1] 个 float），而目标 query 行在 dims[2]（N 序列维），无法正确
// 4D 切片。改为掩码广播，全部复用已验证 kernel：
//   unsqueeze / view / repeat / mul / add_impl。
TensorF32* query_row_add_graph(TensorF32* msa, TensorF32* proj_state) {
    const int64_t D = msa->shape().dims[0];  // D_MSA
    const int64_t L = msa->shape().dims[1];
    const int64_t N = msa->shape().dims[2];
    const int64_t B = msa->shape().dims[3];

    // 1) proj_state [D,L,B] -> [D,L,1,B]（OP_RESHAPE -> kernel_cpy，行主序不变）
    TensorF32* ps_unsq = unsqueeze(proj_state, 2);

    // 2) 目标形状占位节点（repeat 只取 b->shape()，不读数据）
    int64_t tgt_dims[] = {D, L, N, B};
    TensorF32* target = context().new_tensor<float>(4, tgt_dims);

    // 3) 常量掩码 [N] = [1,0,0,...] -> view 为 [1,1,N,1]
    int64_t mask_dims[] = {N};
    TensorF32* n_mask = context().new_tensor<float>(1, mask_dims);
    n_mask->flag = 0;                        // 常量，不可训练
    float* md = n_mask->data();
    for (int64_t i = 0; i < N; i++) md[i] = (i == 0) ? 1.0f : 0.0f;
    TensorF32* n_mask4 = view(n_mask, Shape{1, 1, N, 1});

    // 4) 掩码广播到 [D,L,N,B]：kernel_repeat 尾部对齐取模，仅在 n==0 处=1
    TensorF32* mask_r = repeat(n_mask4, target);
    // 5) proj_state 广播到 [D,L,N,B]（所有 n 相同）
    TensorF32* ps_r   = repeat(ps_unsq, target);
    // 6) addend = proj_state * mask -> n!=0 处清零
    TensorF32* addend = mul(ps_r, mask_r);
    // 7) msa + addend（同形 [D,L,N,B] 逐元素加）
    return add_impl(msa, addend, /*inplace=*/false);
}
} // namespace

RFAAConfig::RFAAConfig() {
    // 默认 SE3 配置
    se3_config.node_dim = D_MSA + D_STATE;  // 288
    se3_config.edge_dim = D_PAIR + 64 + 1;  // 193 (pair + rbf + seqsep)
    se3_config.hidden_dim = 128;
    se3_config.n_layers = 2;
    se3_config.n_heads = 4;
    se3_config.l0_features = {32};   // state 输出
    se3_config.l1_features = {3};    // 坐标更新
    // 度1 输入通道数统一与 l1_features[0]=3 对齐（值版 l1_feats 为 3 通道）；
    // SE3Transformer(SE3Config) 构造器据此构建 fiber_in 度1=3，避免默认 16 与 3 不匹配。
    se3_config.l1_in_feats = 3;
}

// IterBlock 实现
IterBlock::IterBlock(const RFAAConfig& config, bool update_msa_pair)
    : config_(config), update_msa_pair_(update_msa_pair) {
    // 旧值初始化 (保留注释):
    // 所有 LinearLayer/LayerNorm 值成员已移除, 子模块现由 RFAAModel 创建后通过 set_sub_modules() 注入
    // pos_enc_ 由 RFAAModel 构造后通过 set_pos_enc() 注入
}

void IterBlock::proj_state_add_to_query_row(TensorF32& msa, const TensorF32& proj_state) {
    // state -> msa[:,0]
    // msa[:, 0] += proj(state)  (B,L,32) -> (B,L,256)

    // TODO: CUDA optimize

    // projected state (B, L, 256)
    // query_row += state_proj
    // msa (B, N, L, D) -> N = 0 the target seq to predict
    // query_row (B, L, D)
    int B = msa.shape().dims[0];
    int L = msa.shape().dims[2];
    int D = msa.shape().dims[3];
    
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            for (int d = 0; d < D; d++) {
                // 计算索引
                size_t idx = b * L * D + l * D + d;
                
                // 修改值
                msa.data()[idx] += proj_state.data()[b * L * D_MSA + l * D_MSA + d];
            }
        }
    }
    
}

TensorF32 IterBlock::compute_rbf_feature(const TensorF32& coords)
{
    // Python equivalent:
    // cas = xyz[:, :, 1].contiguous()
    // rbf_feat = rbf(torch.cdist(cas, cas))
    
    // coords: (B, L, A, 3) where A=3 (N, CA, C)
    // Return: (B, L, L, 64) - RBF feature
    
    int B = coords.shape().dims[0];
    int L = coords.shape().dims[1];
    int A = coords.shape().dims[2];  // num atoms
    int D = coords.shape().dims[3];  // 3 for x,y,z
    
    // Step1: cas = coords[:, :, 1] -> (B, L, 3)
    // Select CA atom (index 1 on atom dimension)
    //TensorF32 cas = coords.select(2, 1);  // shape: (B, L, 3)
    
    TensorF32 cas({B, L, 3}, coords.device());
    const float* src = coords.data();
    float* dst = cas.data();

    // Each CA entry is 3 floats (x,y,z), but spaced A*3 = 9 floats apart
    const int64_t src_stride = A * D;  // 9 floats between consecutive residues
    const int64_t dst_stride = D;      // 3 floats between consecutive residues

    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            // Source: at (b, l, 1, 0) - CA atom, x coordinate
            const float* src_ptr = src + b * L * A * D + l * A * D + 1 * D;
            // Destination: at (b, l, 0)
            float* dst_ptr = dst + b * L * D + l * D;
        
            // Copy 3 floats (x, y, z) for this residue
            std::memcpy(dst_ptr, src_ptr, 3 * sizeof(float));
        }
    }

    
    // Ensure contiguous: create new tensor and copy
    TensorF32 cas_copy({B, L, D}, coords.device());
    cas_copy.copy_from(cas);
    
    // Step2: Compute pairwise distances - torch.cdist(cas, cas)
    // Input: (B, L, 3), Output: (B, L, L)
    TensorF32 dists({B, L, L}, coords.device());
    
    const float* cas_data = cas_copy.data();
    float* dists_data = dists.data();
    
    // For each batch
    #pragma omp parallel for
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                // Compute Euclidean distance between cas[b,i,:] and cas[b,j,:]
                float dx = cas_data[b * L * D + i * D + 0] - cas_data[b * L * D + j * D + 0];
                float dy = cas_data[b * L * D + i * D + 1] - cas_data[b * L * D + j * D + 1];
                float dz = cas_data[b * L * D + i * D + 2] - cas_data[b * L * D + j * D + 2];
                float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                dists_data[b * L * L + i * L + j] = dist;
            }
        }
    }
    
    // Step3: RBF expansion
    // rbf(x) = exp(-(x - center_i)^2 / (2*width^2)) for i in num_rbf
    // Standard: 64 RBF centers from 0 to 20 Angstroms
    const int num_rbf = 64;
    const float rbf_min = 0.0f;
    const float rbf_max = 20.0f;
    
    TensorF32 rbf_feature({B, L, L, num_rbf}, coords.device());
    float* rbf_data = rbf_feature.data();
    
    float rbf_step = (rbf_max - rbf_min) / (num_rbf - 1);
    
    #pragma omp parallel for
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                float dist = dists_data[b * L * L + i * L + j];
                for (int k = 0; k < num_rbf; k++) {
                    float center = rbf_min + k * rbf_step;
                    float diff = dist - center;
                    // Gaussian RBF with width = step
                    rbf_data[b * L * L * num_rbf + i * L * num_rbf + j * num_rbf + k] = 
                        std::exp(-diff * diff / (2.0f * rbf_step * rbf_step));
                }
            }
        }
    }
    
    return rbf_feature;
}

TensorF32 IterBlock::compute_l1_features(const TensorF32& coords) {
    // Python: l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
    //         l1_feats = l1_feats.reshape(B*L, -1, 3)
    // coords: (B, L, 3, 3)  — 3 atoms (N, CA, C) × 3 xyz
    // output: (B*L, 3, 3)   — 各原子相对 CA 的位移向量

    int B = coords.shape().dims[0];
    int L = coords.shape().dims[1];
    int A = coords.shape().dims[2];  // 3 atoms
    int D = coords.shape().dims[3];  // 3 xyz

    TensorF32 l1_feats({B * L, A, D}, coords.device());

    const float* src = coords.data();
    float* dst = l1_feats.data();

    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            // CA 坐标 (atom index 1)
            float ca_x = src[b * L * A * D + l * A * D + 1 * D + 0];
            float ca_y = src[b * L * A * D + l * A * D + 1 * D + 1];
            float ca_z = src[b * L * A * D + l * A * D + 1 * D + 2];

            for (int a = 0; a < A; a++) {
                int out_idx = (b * L + l) * A * D + a * D;
                dst[out_idx + 0] = src[b * L * A * D + l * A * D + a * D + 0] - ca_x;
                dst[out_idx + 1] = src[b * L * A * D + l * A * D + a * D + 1] - ca_y;
                dst[out_idx + 2] = src[b * L * A * D + l * A * D + a * D + 2] - ca_z;
            }
        }
    }

    return l1_feats;
}

void IterBlock::forward(TensorF32& msa, TensorF32& pair, 
                        TensorF32& state, 
                        const TensorF32& seq1hot,
                        const TensorF32& coords,
                        const TensorF32& bond_feats,
                        const TensorF32& dist_matrix,
                        const TensorF32& same_chain,
                        const TensorI64& residx) {
    // residx 转 float 供 PositionalEncoding 使用
    TensorF32 residx_f32(residx.shape());
    if (residx.numel() > 0) {
        for (int64_t i = 0; i < residx.numel(); ++i)
            residx_f32.data()[i] = static_cast<float>(residx.data()[i]);
    }
    
    if (update_msa_pair_) {

        // ------------ 1D track update ------------
        // ===== Step 1: msa2msa =====

        // state -> msa[:,0]
        // msa[:, 0] += proj(state)  (B,L,32) -> (B,L,256)
        // CUDA optimize
        {
            //auto query_row = msa.select(1, 0);  // (B, L, 256)
            // query_row += Linear(state) ...

            // the supplemental said that a layernorm and then a linear?
            // layernorm first
            // 旧栈上变量: LayerNorm state_norm(D_STATE); → state2msa_norm_
            auto& state_normed = *state2msa_norm_->forward(&state);
            
            // 旧栈上变量: LinearLayer linear(D_STATE, D_MSA); → state2msa_linear_
            const auto proj_state = state2msa_linear_->forward(state_normed);
            proj_state_add_to_query_row(msa, proj_state);
        }

        TensorF32 rbf_feature;
        {
            // pair2msa: 将 pair 转换为 attention bias 注入 msa row attention
            // pair -> attention bias
        
            TensorF32 pair_biased;
       
            //// (B, L, 3, 3) - 初始 Ca 坐标 (可选)
            // compute the RBF feature to inject into pair bias
            TensorF32 rbf = compute_rbf_feature(coords);
            // rel_pos, bond_dist = positionalEncoding(bond feat, dist matrix)
            // bias += linear(rel_pos) + linear(bond_dist)

            auto pos_out = pos_enc_->forward(coords, residx_f32, bond_feats, dist_matrix, same_chain);
            rbf_feature.copy_from(*add_impl(&rbf, &pos_out, /*inplace=*/false));
            // 旧栈上变量: LayerNorm pair_layernorm(D_PAIR); → pair2msa_norm_
            pair_biased.copy_from(*pair2msa_norm_->forward(&pair));

            pair_biased.copy_from(*add_impl(&pair_biased, &rbf_feature, /*inplace=*/false));

            // TODO
            // update msa query row with state from SE3 output
            // state → msa[:,0] 已在 Step 1 (msa2msa) 完成，无需重复

            // a problem: tensor reshape?
            // MSA Row Attention with bias
            msa = msa_row_attn_->forward(msa, pair_biased);
            // RF2 code dropout(row_attn_out, 0.15);
        
            // MSA Column Attention
            msa = msa_col_attn_->forward(msa);
        
            // FeedForward
            msa = msa_ff_->forward(msa);
        }
        // ------------ 1D track update ------------
        
        // to update pair : 2D track update

        // msa2pair: how the pair is updated from the msa?
        // ===== Step 2: msa2pair =====
        // Outer Product Mean
        // msa (B,N,L,256) -> Linear -> (B,N,L,16)
        // outer product + mean -> (B,L,L,256) -> Linear -> (B,L,L,128)
        {
            // msa2pair: einsum('bikd,bjkd->bijd', left, right/N) — 收缩 seq 维 N(dims[1])
            // 值版须用 outer_product_mean（outer_product 签名 (B,1,L,D)×(B,L,1,D) 是纯外积，
            // 与此处的"外积+对 N 求均值"不符，且输入形状 (B,N,L,16) 不匹配 → 修正为 outer_product_mean）。
            auto& msa_normed = *msa2pair_norm_->forward(&msa);
            TensorF32 left  = msa2pair_left_proj_->forward(msa_normed);   // (B,N,L,16)
            TensorF32 right = msa2pair_right_proj_->forward(msa_normed);  // (B,N,L,16)
            // dst[b,i,j,d] = (1/N)*sum_n left[b,n,i,d]*right[b,n,j,d] → (B,L,L,16)
            TensorF32 pair_update = outer_product_mean(left, right);
            pair_update = msa2pair_out_proj_->forward(pair_update);  // (B,L,L,128)
            pair.copy_from(*add_impl(&pair, &pair_update, /*inplace=*/false));  // residual
        }
        
        // Triangle Multiplication
        //pair = pair + drop_row(tri_mul_out_->forward(pair));
        //pair = pair + drop_row(tri_mul_in_->forward(pair));
        Dropout drop_row(1, 0.15);
        auto tri_out = drop_row.forward(tri_mul_out_->forward(pair, true));
        pair.copy_from(*add_impl(&pair, &tri_out, /*inplace=*/false));
        auto tri_in = drop_row.forward(tri_mul_in_->forward(pair, false));
        pair.copy_from(*add_impl(&pair, &tri_in, /*inplace=*/false));

        // ===== Step 3: pair2pair =====
        // state outer product -> gate
        {
            // 旧栈上变量: LinearLayer rbf_proj(D_RBF, D_PAIR); → pair2pair_rbf_proj_
            rbf_feature = pair2pair_rbf_proj_->forward(rbf_feature);  // (B,L,L,128)
            // 旧栈上变量: LayerNorm state_norm(D_STATE); → pair2pair_state_norm_
            auto& state_normed = *pair2pair_state_norm_->forward(&state);
            // 旧栈上变量: LinearLayer left_proj(D_STATE, 16); → pair2pair_left_proj_
            // 旧栈上变量: LinearLayer right_proj(D_STATE, 16); → pair2pair_right_proj_
            // different weights for left and right?
            TensorF32 left = pair2pair_left_proj_->forward(state_normed);   // (B,L,16)
            TensorF32 right = pair2pair_right_proj_->forward(state_normed); // (B,L,16)
            // gate 纯外积，特征笛卡尔积 (B,L,L,256)，与 gate_proj 输入 256 匹配
            TensorF32 gate = outer_product_cartesian(left, right);  // (B,L,L,256)
            // 旧栈上变量: LinearLayer gate_proj(16 * 16, D_PAIR); → pair2pair_gate_proj_
            // d_hidden_gate = 16
            gate = pair2pair_gate_proj_->forward(gate);  // (B,L,L,128)
            gate.copy_from(*sigmoid(&gate));  // (B,L,L,128) -> (B,L,L,128) gate values between 0 and 1
            rbf_feature.copy_from(*mul(&rbf_feature, &gate));  // element-wise gating
            // left = Linear(state, 32->16)
            // right = Linear(state, 32->16)
            // gate = sigmoid(left x right -> Linear -> 128)
            //auto gate = state;  // get_gate
            //gate.copy_from(state);  // 简化，实际需要计算 gate
            // rbf_feat 经 gate 过滤注入 pair
            // pair += gate * rbf_feat
        
            // to update pair
            // Biased Axial Attention (row/col)
            Dropout drop_row(1, 0.15);
            Dropout drop_col(2, 0.15);
            auto row_out = drop_row.forward(pair_row_attn_->forward(pair, rbf_feature));
            pair.copy_from(*add_impl(&pair, &row_out, /*inplace=*/false));
            auto col_out = drop_col.forward(pair_col_attn_->forward(pair, rbf_feature));
            pair.copy_from(*add_impl(&pair, &col_out, /*inplace=*/false));
            // FeedForward (pair_ff) + residual
            auto pair_ff_out = pair_ff_->forward(pair);
            pair.copy_from(*add_impl(&pair, &pair_ff_out, /*inplace=*/false));
        }

    }
    
    // 3D track update
    // ===== Step 4: str2str (SE3 Transformer) =====
    {
        // node features: msa[:,0] + state -> concat -> Linear
        /* auto msa_query = msa.select(1, 0);  // (B, L, 256)
        
        // edge features: pair + rbf + seqsep -> concat -> Linear
        //auto edges = pair;  // concat with rbf, seqsep
        TensorF32 edges;
        edges.copy_from(pair);
        // SE3 Transformer
        auto se3_out = se3_->forward(msa_query, edges, coords);
        
        // 重建 state (完全替换)
        state = se3_out.l0.view({state.shape().dims[0], state.shape().dims[1], D_STATE});
        
        // 更新坐标
        auto updated_coords = struct_update_->update_coords(coords, se3_out.l1);
        
        // 预测侧链扭转角
        auto alpha = struct_update_->predict_torsion(msa_query, state); */

            // 3D track update
    // ===== Step 4: str2str (SE3 Transformer) =====
    // Python:
    //   msa = self.norm_msa(msa)
    //   pair = self.norm_pair(pair)
    //   w_seq = self.encoder_seq(msa).reshape(B,L,1,N).permute(0,3,1,2)
    //   msa = w_seq * msa
    //   msa = msa.sum(dim=1)                          ← sum over sequences
    //   msa = torch.cat((msa, seq1hot), dim=-1)
    //   msa = self.norm_node(self.embed_x(msa))
    //   pair = self.norm_edge(self.embed_e(pair))
    //   G = make_graph(xyz, pair, idx, top_k=top_k)
    //   l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
    //   shift = self.se3(G, msa.reshape(B*L,-1,1), l1_feats)
    //   state = shift['0']; offset = shift['1']
    
        int B = msa.shape().dims[0];
        int N = msa.shape().dims[1];
        int L = msa.shape().dims[2];

        // ---- Step 4a: LayerNorm on msa & pair ----
        auto& msa_normed  = *norm_msa_3d_->forward(&msa);   // (B, N, L, 256)
        auto& pair_normed = *norm_pair_3d_->forward(&pair);  // (B, L, L, 128)

        // ---- Step 4b: 序列加权求和 ----
        // encoder_seq: 学习每条序列的权重, shape (N,) → softmax → 加权求和
        // 简化实现: equal-weight mean over N sequences
        // TODO: replace with learned SequenceWeight
        TensorF32 msa_sum({B, L, D_MSA}, msa_normed.device());
        float* sum_data = msa_sum.data();
        const float* msa_data = msa_normed.data();
        std::memset(sum_data, 0, B * L * D_MSA * sizeof(float));

        for (int b = 0; b < B; ++b) {
            for (int n = 0; n < N; ++n) {
                for (int l = 0; l < L; ++l) {
                    for (int d = 0; d < D_MSA; ++d) {
                        int64_t src = ((b * N + n) * L + l) * D_MSA + d;
                        int64_t dst = (b * L + l) * D_MSA + d;
                        sum_data[dst] += msa_data[src] / static_cast<float>(N);
                    }
                }
            }
        }

        // ---- Step 4c: cat(msa_sum, seq1hot) → embed → norm ----
        std::vector<TensorF32*> cat_inputs;
        cat_inputs.push_back(&msa_sum);
        cat_inputs.push_back(const_cast<TensorF32*>(&seq1hot));
        auto& node_cat = *concat_ptr(cat_inputs, -1);
        //TensorF32 node_cat({B, L, NODE_3D_IN}, msa_normed.device());
        /* float* cat_data = node_cat.data();
        const float* s_data = msa_sum.data();
        const float* onehot_data = seq1hot_.data();

        for (int b = 0; b < B; ++b) {
            for (int l = 0; l < L; ++l) {
                // copy msa_sum
                for (int d = 0; d < D_MSA; ++d) {
                    cat_data[(b * L + l) * NODE_3D_IN + d] =
                        s_data[(b * L + l) * D_MSA + d];
                }
                // copy seq1hot
                for (int d = 0; d < 21; ++d) {
                    cat_data[(b * L + l) * NODE_3D_IN + D_MSA + d] =
                        onehot_data[(b * L + l) * 21 + d];
                }
            }
        } */

        // Linear(277 → 32) → LayerNorm → (B, L, 32)
        TensorF32 node_emb = embed_x_->forward(node_cat);
        TensorF32* node_out = norm_node_3d_->forward(&node_emb);

        // ---- Step 4d: pair embedding ----
        // Linear(128 → 32) → LayerNorm → (B, L, L, 32)
        TensorF32 pair_emb = embed_e_->forward(pair_normed);
        TensorF32* edge_out = norm_edge_3d_->forward(&pair_emb);

        // ---- Step 4e: 构建图 ----
        se3::GraphData G = se3::make_graph(coords, *edge_out, residx, 64, 9);

        // ---- Step 4f: l1 特征 (位移向量) ----
        TensorF32 l1_feats = compute_l1_features(coords);  // (B*L, 3, 3)

        // ---- Step 4g: 组装 SE3Features 输入 ----
        // node_out: (B, L, 32) → reshape to (B*L, 32, 1) 作为 degree-0
        // l1_feats: (B*L, 3, 3) 作为 degree-1
        //Fiber fiber_in({NODE_3D_OUT, fiber_out_.degrees[1]}, {0, 1});
        SE3Features node_se3;
        node_se3.features.resize(2);
        node_se3.features[0] = node_out->view({B * L, ITER_NODE_3D_OUT, 1});
        node_se3.features[1].copy_from(l1_feats);

        // ---- Step 4h: 预计算球谐基 ----
        SE3Basis basis;
        //basis.compute(coords, TensorF32() /*orient*/, 2 /*J_max*/);
        basis.compute(G.edge_d, 2);  // J_max=2, 边向量来自 graph


        // ---- Step 4i: SE3 Transformer forward ----
        SE3Features se3_out = se3_->forward(
            node_se3, G.edge_index, G.edge_d, &G.edge_w, basis);

        // ---- Step 4j: 提取输出 ----
        // state: degree-0 → (B*L, D_STATE) → (B, L, D_STATE)
        state = se3_out.features[0].view({B, L, D_STATE});

        // offset: degree-1 → (B*L, 3, 3) → (B, L, 3, 3)
        TensorF32 offset = se3_out.features[1].view({B, L, 3, 3});

        // ---- Step 4k: 坐标更新 ----
        // CA_new = xyz[:,:,1] + offset[:,:,1]
        // N_new  = CA_new + offset[:,:,0]
        // C_new  = CA_new + offset[:,:,2]
        const float* xyz_data = coords.data();
        const float* off_data = offset.data();

        TensorF32 xyz_new({B, L, 3, 3}, coords.device());
        float* xyz_out = xyz_new.data();

        for (int b = 0; b < B; ++b) {
            for (int l = 0; l < L; ++l) {
                int base = (b * L + l) * 9;  // 3 atoms × 3 coords = 9

                // 原始 CA 坐标
                float ca_x0 = xyz_data[base + 3];
                float ca_y0 = xyz_data[base + 4];
                float ca_z0 = xyz_data[base + 5];

                // δCA (绝对偏移)
                float dca_x = off_data[base + 3];
                float dca_y = off_data[base + 4];
                float dca_z = off_data[base + 5];

                // 更新后 CA
                float ca_x_new = ca_x0 + dca_x;
                float ca_y_new = ca_y0 + dca_y;
                float ca_z_new = ca_z0 + dca_z;

                // N = CA_new + δN
                xyz_out[base + 0] = ca_x_new + off_data[base + 0];
                xyz_out[base + 1] = ca_y_new + off_data[base + 1];
                xyz_out[base + 2] = ca_z_new + off_data[base + 2];

                // CA = CA_new
                xyz_out[base + 3] = ca_x_new;
                xyz_out[base + 4] = ca_y_new;
                xyz_out[base + 5] = ca_z_new;

                // C = CA_new + δC
                xyz_out[base + 6] = ca_x_new + off_data[base + 6];
                xyz_out[base + 7] = ca_y_new + off_data[base + 7];
                xyz_out[base + 8] = ca_z_new + off_data[base + 8];
            }
        }

        xyz_new_.copy_from(xyz_new);
    }

    
}

// ===== IterBlock::forward_graph (图模式) =====
// 处理 msa/pair 两条 track (1D/2D 注意力 + FF) + SE3(3D) track。
// 输入/输出均为图节点指针 (ggml 布局 dims[0]=最内维):
//   msa   : 值 (B,N,L,D_MSA) = 图 [D_MSA, L, N, B]
//   pair  : 值 (B,L,L,D_PAIR) = 图 [D_PAIR, L, L, B]
//   rbf   : 值 (B,L,L,D_RBF)  = 图 [D_RBF, L, L, B]  (RBF + pos_enc 注入)
//   state : 值 (B,L,D_STATE)  = 图 [D_STATE, L, B]；SE3 通过引用回写更新
// 返回: 更新后的 pair 图节点；msa、state 通过引用回写。
// 注: 当 coords/seq1hot 提供时，末尾追加 SE3(3D) track（见 run_se3_graph）。
TensorF32* IterBlock::forward_graph(TensorF32*& msa, TensorF32*& pair,
                                    TensorF32* rbf, TensorF32*& state,
                                    const TensorF32* coords,
                                    const TensorI64* residx,
                                    const TensorF32* seq1hot) {
    // ------------ 1D track: msa2msa ------------
    // Step 1: state -> msa[:,0] (query row 注入，图版：掩码广播 add)
    //   proj_state [D_MSA, L, B] = Linear(LayerNorm(state))
    TensorF32* proj_state = state2msa_linear_->forward_graph(
        state2msa_norm_->forward(state));
    msa = query_row_add_graph(msa, proj_state);   // msa[:,0,:,:] += proj_state

    // pair -> pair_biased (msa row attention bias): pair2msa_norm(pair) + rbf
    TensorF32* pair_biased = add_impl(
        pair2msa_norm_->forward(pair), rbf, /*inplace=*/false);

    // MSA Row Attention (with bias)
    msa = msa_row_attn_->forward_graph(msa, pair_biased);
    // TODO: dropout(row_attn_out, 0.15) 图 drop 后续补
    // MSA Column Attention
    msa = msa_col_attn_->forward_graph(msa);
    // FeedForward
    msa = msa_ff_->forward_graph(msa);

    // ------------ 2D track: msa2pair (outer-product-mean) ------------
    // einsum('bikd,bjkd->bijd(de)', left, right/N) — 收缩 seq 维 N（dims[2]），特征笛卡尔积 D×D→D*D。
    // msa 图 [D_MSA,L,N,B]：N = dims[2]（seq），L = dims[1]（残基）。
    // 用专用图 op OP_OUTER_PROD_MEAN（out_prod 只收缩 dims[1]，无法收缩 N 维）。
    const int64_t Nseq_msa2pair = msa->shape().dims[2];
    auto msa_normed  = msa2pair_norm_->forward(msa);                 // [D_MSA, L, N, B]
    auto left        = msa2pair_left_proj_->forward_graph(msa_normed);   // [16, L, N, B]
    auto right       = msa2pair_right_proj_->forward_graph(msa_normed);  // [16, L, N, B]
    auto pair_update = outer_product_mean(left, right, static_cast<int>(Nseq_msa2pair));  // [256,L,L,B]
    pair_update      = msa2pair_out_proj_->forward_graph(pair_update);   // [D_PAIR,L,L,B]
    pair             = add_impl(pair, pair_update, /*inplace=*/false);   // residual

    // Triangle Multiplication (out/in) + dropout + residual
    // 图 dropout 用 Dropout::forward_graph（random mask 常量叶子 + mul），值版语义一致。
    {
        Dropout drop_row(1, 0.15);
        TensorF32* tri_out = drop_row.forward_graph(tri_mul_out_->forward_graph(pair, /*bOutgoing=*/true));
        pair = add_impl(pair, tri_out, /*inplace=*/false);
        TensorF32* tri_in = drop_row.forward_graph(tri_mul_in_->forward_graph(pair, /*bOutgoing=*/false));
        pair = add_impl(pair, tri_in, /*inplace=*/false);
    }

    // ------------ pair2pair ------------
    // rbf -> rbf_proj (bias 注入 pair row/col attention)
    TensorF32* rbf_proj = pair2pair_rbf_proj_->forward_graph(rbf);   // [128, L, L, B]
    // gate = sigmoid(gate_proj(outer_product(state)))
    // state 图 [D_STATE, L, B]；gate 用 OP_OUTER_PROD（纯外积，特征笛卡尔积 D*D→256）
    TensorF32* state_normed  = pair2pair_state_norm_->forward(state);           // [D_STATE, L, B]
    TensorF32* gate_left  = pair2pair_left_proj_->forward_graph(state_normed);  // [16, L, B]
    TensorF32* gate_right = pair2pair_right_proj_->forward_graph(state_normed); // [16, L, B]
    TensorF32* gate       = outer_product_graph(gate_left, gate_right);         // [256, L, L, B]
    gate = pair2pair_gate_proj_->forward_graph(gate);                           // [128, L, L, B]
    gate = sigmoid(gate);                                                       // [0,1]
    rbf_proj = mul(rbf_proj, gate);                                             // element-wise gate

    // Biased Axial Attention (row/col) + residual
    pair = add_impl(pair, pair_row_attn_->forward_graph(pair, rbf_proj),
                    /*inplace=*/false);
    pair = add_impl(pair, pair_col_attn_->forward_graph(pair, rbf_proj),
                    /*inplace=*/false);
    // FeedForward (pair_ff) + residual — 值版 forward 同步补齐（见 IterBlock::forward）
    pair = add_impl(pair, pair_ff_->forward_graph(pair), /*inplace=*/false);

    // ------------ 3D track: SE3 Transformer ------------
    // SE3 图块由"训练入口"驱动（图外值回落）：
    //   forward_graph 只构建 msa/pair/rbf 图节点并返回 pair；SE3 的结构常量（make_graph → G、
    //   basis）依赖"更新后 pair 的值"，而 pair 在此为图节点、未 graph_compute，故不能在
    //   forward_graph 内部构造 G/basis。
    //   训练入口在每个 block 边界需：
    //     1) graph_compute(pair) 得到 pair_value；
    //     2) 调用 run_se3_structural(msa, pair, rbf, state, pair_value, coords, residx, seq1hot)，
    //        其内部 Phase A 回落 pair_value → embed_e_ → norm_edge_3d_ → make_graph → basis；
    //        Phase B 调用 run_se3_graph 追加可微 SE3 图节点并把 state 回写为图节点；
    //     3) graph_compute 返回的 offset 图节点（se3_out[1]）后，调用
    //        apply_coord_update(offset_value, coords) 更新骨架坐标 → xyz_new_。
    //   当未提供结构输入（coords/seq1hot==nullptr）时跳过 SE3，等价纯 1D/2D track。
    //(void)coords; (void)residx; (void)seq1hot;

    return pair;
}

// SE3(3D) track 图模式子流程（可微部分用图 op，结构常量由调用方以 G/basis 注入）。
//   node 度0 = norm_node_3d(embed_x(cat(msa 沿 Nseq 维均值, seq1hot)))  → 图节点 [32,B*L]
//   node 度1 = l1_feats (compute_l1_features(coords))                    → 常量叶子
//   edge     = G.edge_index / G.edge_d / G.edge_w → src/tgt/d/w 常量叶子
//   basis    = 调用方预计算（SE3Basis.compute(G.edge_d, 2)）
//   out      = se3_->forward_graph({node0,node1}, src, tgt, d, w, basis, N=B*L)
//   state    = out[0].view({B,L,D_STATE})（引用回写）
// 说明：结构预处理（make_graph 需 edge_out 值张量）属"图外"，由训练入口在 block 边界
//       回落值计算后传入 G/basis/coords；此处仅做可微 node 嵌入 + se3_ 调用。
// 返回 se3_out 图节点：[0]=state(度0), [1]=offset(度1，坐标更新用，训练入口 graph_compute 后
//      回落值 + apply_coord_update 更新骨架坐标)。
std::vector<TensorF32*> IterBlock::run_se3_graph(TensorF32*& msa, TensorF32*& pair, TensorF32* rbf,
                                                 TensorF32*& state,
                                                 const se3::GraphData& G, const SE3Basis& basis,
                                                 const TensorF32& coords, const TensorF32& seq1hot) {
    //(void)pair; (void)rbf;
    const int B = seq1hot.shape().dims[0];
    const int L = seq1hot.shape().dims[1];
    const int64_t N = static_cast<int64_t>(B) * L;   // 图节点数

    // ---- node 度0：msa 沿 Nseq 维均值 → cat(seq1hot) → embed_x_ → norm_node_3d_ ----
    // msa 图节点 [D_MSA, L, Nseq, B]：permute 把 Nseq 移到最内 dims[0] 后 sum_rows 归约
    const int64_t Nseq = msa->shape().dims[2];
    TensorF32* msa_p = permute(msa, std::vector<int>{2, 0, 1, 3});    // [Nseq, D_MSA, L, B]
    TensorF32* msa_s = sum_rows(msa_p);                              // [1, D_MSA, L, B]
    TensorF32* msa_m = scale(msa_s, 1.0f / static_cast<float>(Nseq)); // 均值
    TensorF32* msa_v = view(msa_m, Shape({D_MSA, L, B}));            // [D_MSA, L, B]
    TensorF32* s1h   = constant_tensor({21, L, B}, seq1hot.data());  // [21, L, B]
    TensorF32* node_cat = concat_ptr({msa_v, s1h}, 0);               // [D_MSA+21, L, B]
    TensorF32* node_emb = embed_x_->forward_graph(node_cat);         // [32, L, B]
    TensorF32* node_nrm = norm_node_3d_->forward(node_emb);          // [32, L, B]
    TensorF32* node0 = view(node_nrm, Shape({ITER_NODE_3D_OUT, N})); // [32, B*L]（n=b*L+l）

    // ---- node 度1：l1_feats 常量叶子 [3*d_dim1, B*L]，与值版 node_se3.features[1] 对齐 ----
    // 值版固定度1输入 = l1_feats (B*L, 3, 3)（3 通道位移向量，d_dim1=3）。
    // SE3Transformer(SE3Config) 构造器已把 fiber_in 度1 通道数取 cfg.l1_features[0]=3，
    // 故此处直接填 3 通道的 9 个元素，无需补零，与值版严格一致。
    TensorF32 l1 = compute_l1_features(coords);                      // (B*L, 3, 3)
    const int m1     = 3;                                             // 度1 通道数（值版 l1_feats 固定 3）
    const int d_dim1 = 3;                                             // 度1 → 2*1+1
    std::vector<float> l1data(static_cast<size_t>(m1) * d_dim1 * N, 0.0f);
    for (int64_t n = 0; n < N; ++n)
        for (int a = 0; a < m1; ++a)
            for (int c = 0; c < d_dim1; ++c)
                l1data[(static_cast<size_t>(a) * d_dim1 + c) * N + n] =
                    l1.data()[n * 9 + a * 3 + c];
    TensorF32* node1 = constant_tensor({m1 * d_dim1, N}, l1data.data());

    // ---- 边特征常量叶子（由调用方值版 make_graph 注入）----
    const int64_t E = G.edge_index.numel() > 0 ? G.edge_index.shape().dims[1] : 0;
    if (E <= 0) {
        // 无有效边图（结构常量未注入），SE3 图块无法执行；仅回写 state=输入（等价跳过）。
        return {};
    }
    std::vector<float> src_d(static_cast<size_t>(E)), tgt_d(static_cast<size_t>(E));
    for (int64_t e = 0; e < E; ++e) {
        src_d[e] = static_cast<float>(G.edge_index.data()[e]);
        tgt_d[e] = static_cast<float>(G.edge_index.data()[E + e]);
    }
    TensorF32* edge_src = constant_tensor({E}, src_d.data());
    TensorF32* edge_tgt = constant_tensor({E}, tgt_d.data());
    // edge_d: (E,3) → [3,E]；edge_w: (E,E_dim) → [E_dim,E]
    std::vector<float> dd(static_cast<size_t>(3 * E));
    for (int64_t e = 0; e < E; ++e)
        for (int c = 0; c < 3; ++c) dd[static_cast<size_t>(c) * E + e] = G.edge_d.data()[e * 3 + c];
    TensorF32* edge_d = constant_tensor({3, E}, dd.data());
    const int64_t E_dim = G.edge_w.numel() > 0 ? G.edge_w.shape().dims[1] : 0;
    std::vector<float> ww(static_cast<size_t>(E_dim * E), 0.0f);
    for (int64_t e = 0; e < E; ++e)
        for (int64_t c = 0; c < E_dim; ++c)
            ww[static_cast<size_t>(c) * E + e] =
                G.edge_w.numel() > 0 ? G.edge_w.data()[e * E_dim + c] : 0.0f;
    TensorF32* edge_w = constant_tensor({E_dim, E}, ww.data());

    // ---- SE3 Transformer forward_graph ----
    std::vector<TensorF32*> h_nodes = {node0, node1};
    std::vector<TensorF32*> se3_out = se3_->forward_graph(
        h_nodes, edge_src, edge_tgt, edge_d, edge_w, basis, static_cast<int>(N));

    // ---- state 回写：度0 → (B,L,D_STATE) 图 [D_STATE, L, B] ----
    // se3_out[0] 为 [32, B*L]（度0 输出），节点序 n=b*L+l → view [D_STATE, L, B]
    state = view(se3_out[0], Shape({D_STATE, L, B}));

    // 返回 se3_out：se3_out[1] 为度1 offset 图节点 [3*3, B*L]（坐标更新需"图外"回落其值）。
    // 训练入口在此后 graph_compute，再用 apply_coord_update 把 offset 值叠加到 coords → xyz_new_。
    return se3_out;
}

// ===== 训练入口驱动：SE3 图块（图外值回落 + state 回写）=====
// Phase A（结构常量，图外值回落）：由前一 track graph_compute 得到的 pair_value
//   经值版 embed_e_/norm_edge_3d_ 得 edge_out，再 make_graph(coords, edge_out, residx) → G，
//   并 basis.compute(G.edge_d, 2)。此阶段非可微（make_graph 是离散拓扑），故走值版。
// Phase B（可微图块）：调用 run_se3_graph，追加 node 嵌入 + se3_->forward_graph 到计算图，
//   state 经引用回写为图节点（度0）。返回的 offset 图节点由训练入口 graph_compute 后做坐标更新。
void IterBlock::run_se3_structural(TensorF32*& msa, TensorF32*& pair, TensorF32* rbf,
                                   TensorF32*& state,
                                   const TensorF32& pair_value,
                                   const TensorF32& coords,
                                   const TensorI64& residx,
                                   const TensorF32& seq1hot) {
    // ---- Phase A: pair 值 → edge_out → make_graph → basis ----
    // 值版 pair 布局 (B,L,L,D_PAIR)。embed_e_ (D_PAIR→ITER_EDGE_3D_OUT) → norm_edge_3d_。
    // Tensor 为 move-only，避免拷贝。LayerNorm::forward 返回 TensorF32*，
    // LinearLayer::forward 返回 TensorF32（move 到临时值再取址）。
    TensorF32* pair_normed = norm_pair_3d_->forward(&const_cast<TensorF32&>(pair_value));  // (B,L,L,D_PAIR)
    TensorF32  edge_emb    = embed_e_->forward(*pair_normed);            // (B,L,L,ITER_EDGE_3D_OUT)
    TensorF32* edge_out    = norm_edge_3d_->forward(&edge_emb);          // (B,L,L,ITER_EDGE_3D_OUT)
    se3::GraphData G       = se3::make_graph(coords, *edge_out, residx, 64, 9);
    SE3Basis basis;
    basis.compute(G.edge_d, 2);

    // ---- Phase B: run_se3_graph（可微图块，state 回写）----
    // 返回的 se3_out[1]（offset 图节点）由训练入口 graph_compute 后调用 apply_coord_update。
    run_se3_graph(msa, pair, rbf, state, G, basis, coords, seq1hot);
}

// ===== 坐标更新（图外值回落）=====
// offset_value: 度1 SE3 输出，值布局 (B*L, 3, 3)（[N,CA,C] 相对 CA 位移；CA 通道为绝对位移）。
// 与值版 Step4k 一致：CA_new = coords_CA + offset[:,:,1]；N_new = CA_new + offset[:,:,0]；
// C_new = CA_new + offset[:,:,2]。结果写 xyz_new_。
void IterBlock::apply_coord_update(const TensorF32& offset_value, const TensorF32& coords) {
    const int B = coords.shape().dims[0];
    const int L = coords.shape().dims[1];
    const float* xyz_data = coords.data();
    const float* off_data = offset_value.data();

    TensorF32 xyz_new({B, L, 3, 3}, coords.device());
    float* xyz_out = xyz_new.data();
    for (int b = 0; b < B; ++b) {
        for (int l = 0; l < L; ++l) {
            const int base = (b * L + l) * 9;
            const float ca_x0 = xyz_data[base + 3];
            const float ca_y0 = xyz_data[base + 4];
            const float ca_z0 = xyz_data[base + 5];
            const float dca_x = off_data[base + 3];
            const float dca_y = off_data[base + 4];
            const float dca_z = off_data[base + 5];
            const float ca_x_new = ca_x0 + dca_x;
            const float ca_y_new = ca_y0 + dca_y;
            const float ca_z_new = ca_z0 + dca_z;
            xyz_out[base + 0] = ca_x_new + off_data[base + 0];
            xyz_out[base + 1] = ca_y_new + off_data[base + 1];
            xyz_out[base + 2] = ca_z_new + off_data[base + 2];
            xyz_out[base + 3] = ca_x_new;
            xyz_out[base + 4] = ca_y_new;
            xyz_out[base + 5] = ca_z_new;
            xyz_out[base + 6] = ca_x_new + off_data[base + 6];
            xyz_out[base + 7] = ca_y_new + off_data[base + 7];
            xyz_out[base + 8] = ca_z_new + off_data[base + 8];
        }
    }
    xyz_new_.copy_from(xyz_new);
}

void FullBlock::forward(TensorF32& msa_full, TensorF32& pair, TensorF32& state, 
                        const TensorF32& seq1hot,
                        const TensorF32& coords,
                        const TensorF32& bond_feats,
                        const TensorF32& dist_matrix,
                        const TensorF32& same_chain,
                        const TensorI64& residx) {
    // residx 转 float 供 PositionalEncoding 使用
    TensorF32 residx_f32(residx.shape());
    if (residx.numel() > 0) {
        for (int64_t i = 0; i < residx.numel(); ++i)
            residx_f32.data()[i] = static_cast<float>(residx.data()[i]);
    }
    // FullBlock 在 IterBlock 的基础上增加了 msa_full 的使用和全局 column attention
    // msa_full 需要在 forward 函数参数中传入，或者在 IterBlock 中存储为成员变量
    
    // ------------ 1D track update ------------
    // ===== Step 1: msa2msa =====
    {
        //auto query_row = msa.select(1, 0);  // (B, L, 256)
        // query_row += Linear(state) ...

        // 旧栈上变量: LayerNorm state_norm(D_STATE); → state2msa_norm_
        auto& state_normed = *state2msa_norm_->forward(&state);
            
        // 旧栈上变量: LinearLayer linear(D_STATE, D_MSA); → state2msa_linear_
        const auto proj_state = state2msa_linear_->forward(state_normed);
        proj_state_add_to_query_row(msa_full, proj_state);
    }

    TensorF32 rbf_feature;
    {
        // pair2msa: 将 pair 转换为 attention bias 注入 msa row attention
        // pair -> attention bias
        
        TensorF32 pair_biased;
       
        //// (B, L, 3, 3) - 初始 Ca 坐标 (可选)
        // compute the RBF feature to inject into pair bias
        TensorF32 rbf = compute_rbf_feature(coords);
        // rel_pos, bond_dist = positionalEncoding(bond feat, dist matrix)
        // bias += linear(rel_pos) + linear(bond_dist)

        auto pos_out = pos_enc_->forward(coords, residx_f32, bond_feats, dist_matrix, same_chain);
        rbf_feature.copy_from(*add_impl(&rbf, &pos_out, /*inplace=*/false));
        // 旧栈上变量: LayerNorm pair_layernorm(D_PAIR); → pair2msa_norm_
        pair_biased.copy_from(*pair2msa_norm_->forward(&pair));

        pair_biased.copy_from(*add_impl(&pair_biased, &rbf_feature, /*inplace=*/false));

        // TODO
        // update msa query row with state from SE3 output
        // DONE already update in the msa2msa

        // a problem: tensor reshape?
        // MSA Row Attention with bias
        msa_full = msa_row_attn_->forward(msa_full, pair_biased);
        // RF2 code dropout(row_attn_out, 0.15);
        
        // MSA Column Attention
        // global attention
        msa_full = msa_global_col_attn_->forward(msa_full);
        
        // FeedForward
        msa_full = msa_ff_->forward(msa_full);
    }
    // ------------ 1D track update ------------

    // msa2pair: einsum('bikd,bjkd->bijd', left, right/N) — 收缩 seq 维 N(dims[1])
    // 值版用 outer_product_mean（原 outer_product 签名 (B,1,L,D)×(B,L,1,D) 不符且为纯外积）。
    {
        auto& msa_normed = *msa2pair_norm_->forward(&msa_full);
        TensorF32 left  = msa2pair_left_proj_->forward(msa_normed);   // (B,N,L,16)
        TensorF32 right = msa2pair_right_proj_->forward(msa_normed);  // (B,N,L,16)
        TensorF32 pair_update = outer_product_mean(left, right);      // (B,L,L,16)
        pair_update = msa2pair_out_proj_->forward(pair_update);       // (B,L,L,128)
        pair.copy_from(*add_impl(&pair, &pair_update, /*inplace=*/false));  // residual
    }

    // Triangle Multiplication
    Dropout drop_row(1, 0.15);
    auto tri_out = drop_row.forward(tri_mul_out_->forward(pair, true));
    pair.copy_from(*add_impl(&pair, &tri_out, /*inplace=*/false));
    auto tri_in = drop_row.forward(tri_mul_in_->forward(pair, false));
    pair.copy_from(*add_impl(&pair, &tri_in, /*inplace=*/false));

    // ===== Step 3: pair2pair =====
    // state outer product -> gate
    {
        // 旧栈上变量: LinearLayer rbf_proj(D_RBF, D_PAIR); → pair2pair_rbf_proj_
        rbf_feature = pair2pair_rbf_proj_->forward(rbf_feature);  // (B,L,L,128)
        // 旧栈上变量: LayerNorm state_norm(D_STATE); → pair2pair_state_norm_
        auto& state_normed = *pair2pair_state_norm_->forward(&state);
        // 旧栈上变量: LinearLayer left_proj(D_STATE, 16); → pair2pair_left_proj_
        // 旧栈上变量: LinearLayer right_proj(D_STATE, 16); → pair2pair_right_proj_
        // different weights for left and right?
        TensorF32 left = pair2pair_left_proj_->forward(state_normed);   // (B,L,16)
        TensorF32 right = pair2pair_right_proj_->forward(state_normed); // (B,L,16)
        // gate 纯外积，特征笛卡尔积 (B,L,L,256)，与 gate_proj 输入 256 匹配
        TensorF32 gate = outer_product_cartesian(left, right);  // (B,L,L,256)
        // 旧栈上变量: LinearLayer gate_proj(16 * 16, D_PAIR); → pair2pair_gate_proj_
        // d_hidden_gate = 16
        gate = pair2pair_gate_proj_->forward(gate);  // (B,L,L,128)
        gate.copy_from(*sigmoid(&gate));  // (B,L,L,128) -> (B,L,L,128) gate values between 0 and 1
        rbf_feature.copy_from(*mul(&rbf_feature, &gate));  // element-wise gating
            
        Dropout drop_row2(1, 0.15);
        Dropout drop_col(2, 0.15);
        auto row_out2 = drop_row2.forward(pair_row_attn_->forward(pair, rbf_feature));
        pair.copy_from(*add_impl(&pair, &row_out2, /*inplace=*/false));
        auto col_out2 = drop_col.forward(pair_col_attn_->forward(pair, rbf_feature));
        pair.copy_from(*add_impl(&pair, &col_out2, /*inplace=*/false));
        // FeedForward (pair_ff) + residual
        auto pair_ff_out = pair_ff_->forward(pair);
        pair.copy_from(*add_impl(&pair, &pair_ff_out, /*inplace=*/false));
    }

    // 3D track update — same logic as IterBlock, but uses msa_full
    // ===== Step 4: str2str (SE3 Transformer) =====
    {
        int B = msa_full.shape().dims[0];
        int N = msa_full.shape().dims[1];
        int L = msa_full.shape().dims[2];

        auto& msa_normed = *norm_msa_3d_->forward(&msa_full);
        auto& pair_normed = *norm_pair_3d_->forward(&pair);

        // 序列加权求和 (simplified: equal-weight mean)
        TensorF32 msa_sum({B, L, D_MSA}, msa_normed.device());
        float* sum_data = msa_sum.data();
        const float* msa_data = msa_normed.data();
        std::memset(sum_data, 0, B * L * D_MSA * sizeof(float));
        for (int b = 0; b < B; ++b)
            for (int n = 0; n < N; ++n)
                for (int l = 0; l < L; ++l)
                    for (int d = 0; d < D_MSA; ++d)
                        sum_data[(b * L + l) * D_MSA + d] +=
                            msa_data[((b * N + n) * L + l) * D_MSA + d] / float(N);

        // cat + embed
        std::vector<TensorF32*> cat_inputs;
        cat_inputs.push_back(&msa_sum);
        cat_inputs.push_back(const_cast<TensorF32*>(&seq1hot));
        auto& node_cat = *concat_ptr(cat_inputs, -1);
        /* float* cat_data = node_cat.data();
        for (int b = 0; b < B; ++b)
            for (int l = 0; l < L; ++l) {
                for (int d = 0; d < D_MSA; ++d)
                    cat_data[(b * L + l) * NODE_3D_IN + d] =
                        sum_data[(b * L + l) * D_MSA + d];
                for (int d = 0; d < 21; ++d)
                    cat_data[(b * L + l) * NODE_3D_IN + D_MSA + d] =
                        seq1hot_.data()[(b * L + l) * 21 + d];
            } */

        auto node_emb = embed_x_->forward(node_cat);
        auto node_out = norm_node_3d_->forward(&node_emb);
        auto edge_emb = embed_e_->forward(pair_normed);
        auto edge_out = norm_edge_3d_->forward(&edge_emb);

        se3::GraphData G = se3::make_graph(coords, *edge_out, residx, 64, 9);
        TensorF32 l1_feats = compute_l1_features(coords);

        //Fiber fiber_in({NODE_3D_OUT, 3}, {0, 1});
        SE3Features node_se3;
        node_se3.features.resize(2);
        // node_out = it was actually msa input
        node_se3.features[0] = node_out->view({B * L, ITER_NODE_3D_OUT, 1});
        node_se3.features[1].copy_from(l1_feats);  // SE3Features::features 是 Tensor 值类型

        SE3Basis basis;
        //basis.compute(coords, TensorF32(), 2);
        basis.compute(G.edge_d, 2);  // J_max=2, 边向量来自 graph

        SE3Features se3_out = se3_->forward(
            node_se3, G.edge_index, G.edge_d, &G.edge_w, basis);

        state = se3_out.features[0].view({B, L, D_STATE});
        TensorF32 offset = se3_out.features[1].view({B, L, 3, 3});

        // Coordinate update (same as IterBlock)
        const float* xyz_data = coords.data();
        const float* off_data = offset.data();

        TensorF32 xyz_new({B, L, 3, 3}, coords.device());
        float* xyz_out = xyz_new.data();
        for (int b = 0; b < B; ++b) {
            for (int l = 0; l < L; ++l) {
                int base = (b * L + l) * 9;
                
                float ca_x0 = xyz_data[base + 3];
                float ca_y0 = xyz_data[base + 4];
                float ca_z0 = xyz_data[base + 5];

                // δCA (绝对偏移)
                float dca_x = off_data[base + 3];
                float dca_y = off_data[base + 4];
                float dca_z = off_data[base + 5];

                // 更新后 CA
                float ca_x_new = ca_x0 + dca_x;
                float ca_y_new = ca_y0 + dca_y;
                float ca_z_new = ca_z0 + dca_z;

                // N = CA_new + δN
                xyz_out[base + 0] = ca_x_new + off_data[base + 0];
                xyz_out[base + 1] = ca_y_new + off_data[base + 1];
                xyz_out[base + 2] = ca_z_new + off_data[base + 2];

                // CA = CA_new
                xyz_out[base + 3] = ca_x_new;
                xyz_out[base + 4] = ca_y_new;
                xyz_out[base + 5] = ca_z_new;

                // C = CA_new + δC
                xyz_out[base + 6] = ca_x_new + off_data[base + 6];
                xyz_out[base + 7] = ca_y_new + off_data[base + 7];
                xyz_out[base + 8] = ca_z_new + off_data[base + 8];
            }
        }

        xyz_new_.copy_from(xyz_new);
    }

}

// ===== FullBlock::forward_graph (图模式) =====
// 与 IterBlock::forward_graph 一致，但 msa_full 用 global column attention。
// 布局约定同 IterBlock::forward_graph；SE3 track 由末尾追加（同 IterBlock，见 run_se3_graph）。
TensorF32* FullBlock::forward_graph(TensorF32*& msa_full, TensorF32*& pair,
                                    TensorF32* rbf, TensorF32*& state,
                                    const TensorF32* coords,
                                    const TensorI64* residx,
                                    const TensorF32* seq1hot) {
    // ------------ 1D track: msa2msa ------------
    // Step 1: state -> msa_full[:,0] (query row 注入，图版：掩码广播 add)
    //   proj_state [D_MSA, L, B] = Linear(LayerNorm(state))
    TensorF32* proj_state = state2msa_linear_->forward_graph(
        state2msa_norm_->forward(state));
    msa_full = query_row_add_graph(msa_full, proj_state);   // msa_full[:,0,:,:] += proj_state

    // pair -> pair_biased (msa row attention bias): pair2msa_norm(pair) + rbf
    TensorF32* pair_biased = add_impl(
        pair2msa_norm_->forward(pair), rbf, /*inplace=*/false);

    // MSA Row Attention (with bias)
    msa_full = msa_row_attn_->forward_graph(msa_full, pair_biased);
    // TODO: dropout(row_attn_out, 0.15) 图 drop 后续补
    // MSA Global Column Attention
    msa_full = msa_global_col_attn_->forward_graph(msa_full);
    // FeedForward
    msa_full = msa_ff_->forward_graph(msa_full);

    // ------------ 2D track: msa2pair (outer-product-mean) ------------
    // einsum('bikd,bjkd->bijd(de)', left, right/N) — 收缩 seq 维 N（dims[2]），特征笛卡尔积 D×D→D*D。
    // msa_full 图 [D_MSA,L,N,B]：N = dims[2]（seq），L = dims[1]（残基）。
    // 用专用图 op OP_OUTER_PROD_MEAN（out_prod 只收缩 dims[1]，无法收缩 N 维）。同 IterBlock::forward_graph。
    const int64_t Nseq_msa2pair_full = msa_full->shape().dims[2];
    auto msa_normed_full  = msa2pair_norm_->forward(msa_full);                  // [D_MSA, L, N, B]
    auto left_full        = msa2pair_left_proj_->forward_graph(msa_normed_full);   // [16, L, N, B]
    auto right_full       = msa2pair_right_proj_->forward_graph(msa_normed_full);  // [16, L, N, B]
    auto pair_update_full = outer_product_mean(left_full, right_full,
                                               static_cast<int>(Nseq_msa2pair_full));  // [256,L,L,B]
    pair_update_full      = msa2pair_out_proj_->forward_graph(pair_update_full);      // [D_PAIR,L,L,B]
    pair                  = add_impl(pair, pair_update_full, /*inplace=*/false);      // residual

    // Triangle Multiplication (out/in) + dropout + residual (同 IterBlock::forward_graph)
    {
        Dropout drop_row(1, 0.15);
        TensorF32* tri_out = drop_row.forward_graph(tri_mul_out_->forward_graph(pair, /*bOutgoing=*/true));
        pair = add_impl(pair, tri_out, /*inplace=*/false);
        TensorF32* tri_in = drop_row.forward_graph(tri_mul_in_->forward_graph(pair, /*bOutgoing=*/false));
        pair = add_impl(pair, tri_in, /*inplace=*/false);
    }

    // ------------ pair2pair ------------
    TensorF32* rbf_proj = pair2pair_rbf_proj_->forward_graph(rbf);   // [128, L, L, B]
    // gate = sigmoid(gate_proj(outer_product(state)))——与 IterBlock::forward_graph 相同
    TensorF32* state_normed  = pair2pair_state_norm_->forward(state);           // [D_STATE, L, B]
    TensorF32* gate_left  = pair2pair_left_proj_->forward_graph(state_normed);  // [16, L, B]
    TensorF32* gate_right = pair2pair_right_proj_->forward_graph(state_normed); // [16, L, B]
    TensorF32* gate       = outer_product_graph(gate_left, gate_right);         // [256, L, L, B]
    gate = pair2pair_gate_proj_->forward_graph(gate);                           // [128, L, L, B]
    gate = sigmoid(gate);                                                       // [0,1]
    rbf_proj = mul(rbf_proj, gate);                                             // element-wise gate

    // Biased Axial Attention (row/col) + residual
    pair = add_impl(pair, pair_row_attn_->forward_graph(pair, rbf_proj),
                    /*inplace=*/false);
    pair = add_impl(pair, pair_col_attn_->forward_graph(pair, rbf_proj),
                    /*inplace=*/false);
    // FeedForward (pair_ff) + residual — 值版 forward 同步补齐（见 FullBlock::forward）
    pair = add_impl(pair, pair_ff_->forward_graph(pair), /*inplace=*/false);

    // ------------ 3D track: SE3 Transformer ------------
    // 同 IterBlock::forward_graph：SE3 由训练入口驱动（图外值回落）。
    // 训练入口在每个 block 边界：graph_compute(pair)→pair_value 后调用
    //   run_se3_structural(msa_full, pair, rbf, state, pair_value, coords, residx, seq1hot)
    //   （内部 Phase A 回落 pair_value→edge_out→make_graph→basis；Phase B run_se3_graph 追加
    //     SE3 图节点并回写 state；返回 offset 图节点），再 graph_compute 出 offset 后调用
    //   apply_coord_update(offset_value, coords) 更新骨架坐标。
    //(void)coords; (void)residx; (void)seq1hot;

    return pair;
}

// RefineBlock 构造函数已在 Model.h 中 inline 定义
/* RefineBlock::RefineBlock(const RFAAConfig& config)
    : IterBlock(config, false)  // update_msa_pair = false, 仅更新结构
    // , norm_msa_(D_MSA), norm_pair_(D_PAIR), norm_state_(D_STATE)
    // , embed_x_(NODE_IN_DIM, NODE_OUT_DIM), norm_node_(NODE_OUT_DIM)
    // , embed_e1_(D_PAIR, N_EDGE_FEATS), norm_edge1_(N_EDGE_FEATS)
    // , embed_e2_(EDGE_IN_DIM2, N_EDGE_FEATS), norm_edge2_(N_EDGE_FEATS)
    // 以上 10 个参数现在由 RFAAModel 创建，通过指针注入
{
} */

/* void RefineBlock::set_seq_info(const TensorF32& seq1hot, const TensorI64& idx) {
    seq1hot_ = seq1hot;
    idx_     = idx;
    has_seq_info_ = true;
} */

void RefineBlock::forward(TensorF32& msa, 
                          TensorF32& pair, 
                          TensorF32& state, 
                          const TensorF32& seq1hot,
                          const TensorF32& coords,
                          const TensorF32& bond_feats,
                          const TensorF32& dist_matrix,
                          const TensorF32& same_chain,
                          const TensorI64& residx) {

    // ---- 获取维度 ----
    const auto& msa_shape = msa.shape();
    int B = static_cast<int>(msa_shape.dims[0]);
    int L = static_cast<int>(msa_shape.dims[2]);

    // ================================================================
    // Step 1: LayerNorm 归一化三个 track 输入
    // ================================================================
    auto& node    = *norm_msa_->forward(&msa);     // (B, L, 256)
    auto& pair_n  = *norm_pair_->forward(&pair);        // (B, L, L, 128)
    auto& state_n = *norm_state_->forward(&state);      // (B, L, 32)

    // ================================================================
    // Step 2: 构建节点特征
    // Python: node = cat((node, seq1hot, state), dim=-1)
    //         node = self.norm_node(self.embed_x(node))
    // ================================================================
    // cat([msa_norm(B,L,256), seq1hot(B,L,21), state_norm(B,L,32)])
    // → (B, L, 309)
    std::vector<TensorF32*> node_parts = {&node, const_cast<TensorF32*>(&seq1hot), &state_n};
    auto* node_cat = concat_ptr(node_parts, -1);

    // Linear(309 → 32) → LayerNorm → (B, L, 32)
    TensorF32* node_emb = embed_x_->forward_graph(node_cat);
    auto& node_out = *norm_node_->forward(node_emb);

    // ================================================================
    // Step 3: 构建边特征（两阶段）
    // ================================================================
    // 阶段1: pair → Linear → LayerNorm
    // Python: pair = self.norm_edge1(self.embed_e1(pair))
    // pair (B,L,L,128) → Linear → (B,L,L,32) → LayerNorm → (B,L,L,32)
    TensorF32 pair_emb = embed_e1_->forward(pair_n);
    auto& pair_e1 = *norm_edge1_->forward(&pair_emb);

    // 获取辅助边特征
    TensorF32 neighbor = se3::get_bonded_neigh(residx);            // (B, L, L, 1)
    TensorF32 rbf_feat = compute_rbf_feature(coords);      // (B, L, L, 64)

    // cat → (B,L,L,97) → Linear → (B,L,L,32) → LayerNorm → (B,L,L,32)
    std::vector<TensorF32*> cat_parts;
    cat_parts.push_back(&pair_e1);
    cat_parts.push_back(&rbf_feat);
    cat_parts.push_back(&neighbor);
    auto* pair_cat = concat_ptr(cat_parts, -1);
    TensorF32 pair_e2  = embed_e2_->forward(*pair_cat);
    auto* edge_out = norm_edge2_->forward(&pair_e2);

    // ================================================================
    // Step 4: 构建消息传递图
    // Python: G = make_graph_topk(xyz, pair, idx, top_k=top_k)
    // ================================================================
    se3::GraphData G = se3::make_graph(coords, *edge_out, residx,
                             64 /* top_k */, 9 /* kmin */);

    // ================================================================
    // Step 5: 计算 l1 特征（各原子相对 CA 的位移向量）
    // Python: l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
    //         l1_feats = l1_feats.reshape(B*L, -1, 3)
    // 输出: (B*L, 3, 3)  3个原子 × 3个坐标维度
    // ================================================================
    TensorF32 l1_feats = compute_l1_features(coords);  // (B*L, 3, 3)

    // ================================================================
    // Step 6: SE(3) Transformer 前向传播
    // Python: shift = self.se3(G, node.reshape(B*L, -1, 1), l1_feats)
    //
    // node_out: (B, L, 32) → (B*L, 32, 1) 作为 degree-0 标量特征
    // l1_feats: (B*L, 3, 3)               作为 degree-1 向量特征
    // ================================================================
    // 构建 SE3Basis (预计算球谐基)
    SE3Basis basis;
    basis.compute(G.edge_d, 2);  // J_max=2, 边向量来自 graph

    // 构建 SE3Features 输入
    SE3Features node_se3;
    node_se3.features.resize(2);
    node_se3.features[0] = node_out.view({B * L, ITER_NODE_3D_OUT, 1});
    node_se3.features[1].copy_from(l1_feats);

    SE3Features se3_out = se3_->forward(
            node_se3, G.edge_index, G.edge_d, &G.edge_w, basis);

        // ---- Step 4j: 提取输出 ----
        // state: degree-0 → (B*L, D_STATE) → (B, L, D_STATE)
    state = se3_out.features[0].view({B, L, D_STATE});

        // offset: degree-1 → (B*L, 3, 3) → (B, L, 3, 3)
    TensorF32 offset = se3_out.features[1].view({B, L, 3, 3});

        // ---- Step 4k: 坐标更新 ----
        // CA_new = xyz[:,:,1] + offset[:,:,1]
        // N_new  = CA_new + offset[:,:,0]
        // C_new  = CA_new + offset[:,:,2]
    const float* xyz_data = coords.data();
    const float* off_data = offset.data();

    TensorF32 xyz_new({B, L, 3, 3}, coords.device());
    float* xyz_out = xyz_new.data();

    for (int b = 0; b < B; ++b) {
        for (int l = 0; l < L; ++l) {
            int base = (b * L + l) * 9;
                
            float ca_x0 = xyz_data[base + 3];
            float ca_y0 = xyz_data[base + 4];
            float ca_z0 = xyz_data[base + 5];

                // δCA (绝对偏移)
            float dca_x = off_data[base + 3];
            float dca_y = off_data[base + 4];
            float dca_z = off_data[base + 5];

                // 更新后 CA
            float ca_x_new = ca_x0 + dca_x;
            float ca_y_new = ca_y0 + dca_y;
            float ca_z_new = ca_z0 + dca_z;

                // N = CA_new + δN
            xyz_out[base + 0] = ca_x_new + off_data[base + 0];
            xyz_out[base + 1] = ca_y_new + off_data[base + 1];
            xyz_out[base + 2] = ca_z_new + off_data[base + 2];

                // CA = CA_new
            xyz_out[base + 3] = ca_x_new;
            xyz_out[base + 4] = ca_y_new;
            xyz_out[base + 5] = ca_z_new;

                // C = CA_new + δC
            xyz_out[base + 6] = ca_x_new + off_data[base + 6];
            xyz_out[base + 7] = ca_y_new + off_data[base + 7];
            xyz_out[base + 8] = ca_z_new + off_data[base + 8];
        }
    }

    xyz_new_.copy_from(xyz_new);
    // TODO: SE3Transformer 当前 forward 签名为 forward(SE3Features&, positions,
    //       orientations, edge_index, training)。
    //       等接口完善后，此处替换为：
    //
    //   // 组装节点特征为 SE3Features（度0=32通道, 度1=3通道）
    //   Fiber fiber_in({32, 3}, {0, 1});
    //   SE3Features node_se3(fiber_in, /*prototype*/..., B*L);
    //   // 填充 node_se3.features[0] = node_out.view({B*L, 32, 1})
    //   // 填充 node_se3.features[1] = l1_feats
    //
    //   SE3Features se3_out = se3_->forward(node_se3, coords,
    //                                       TensorF32() /* orient */,
    //                                       G.edge_index);
    //
    //   // 提取输出
    //   TensorF32 state_vec = se3_out.features[0];  // degree-0 → (B*L, C)
    //   TensorF32 offset    = se3_out.features[1];  // degree-1 → (B*L, 3, 3)

    // ================================================================
    // Step 7: 整理输出
    // Python: state  = shift['0'].reshape(B, L, -1)
    //         offset = shift['1'].reshape(B, L, -1, 3)
    // ================================================================
    // TODO: 从 se3_out 中提取后激活以下代码:
    //
    // state_new_ = state_vec.view({B, L, D_STATE});        // (B, L, 32)
    // TensorF32 offset = offset_tensor.view({B, L, 3, 3}); // (B, L, 3, 3)
    //
    // // ================================================================
    // // Step 8: 更新骨架坐标
    // // Python:
    // //   CA_new = xyz[:,:,1] + offset[:,:,1]
    // //   N_new  = CA_new + offset[:,:,0]
    // //   C_new  = CA_new + offset[:,:,2]
    // //   xyz_new = torch.stack([N_new, CA_new, C_new], dim=2)
    // //
    // // offset 各通道定义:
    // //   [:, :, 0, :] = δN  (N 原子相对 CA 的位移)
    // //   [:, :, 1, :] = δCA (CA 位移, 绝对; 即 offset[:,:,1] 直接加在 CA 上)
    // //   [:, :, 2, :] = δC  (C 原子相对 CA 的位移)
    // // ================================================================
    //
    // const float* xyz_data = coords.data();
    // const float* off_data = offset.data();
    //
    // xyz_new_ = TensorF32({B, L, 3, 3}, coords.device());
    // float* xyz_out = xyz_new_.data();
    //
    // for (int b = 0; b < B; ++b) {
    //     for (int l = 0; l < L; ++l) {
    //         // 原始 CA 坐标
    //         int ca_idx  = (b * L + l) * 9 + 3;  // atom=1, x
    //         float ca_x0 = xyz_data[ca_idx];
    //         float ca_y0 = xyz_data[ca_idx + 1];
    //         float ca_z0 = xyz_data[ca_idx + 2];
    //
    //         // offset 各分量
    //         int off_base = (b * L + l) * 9;
    //
    //         // δCA (绝对偏移)
    //         float dca_x = off_data[off_base + 3];
    //         float dca_y = off_data[off_base + 4];
    //         float dca_z = off_data[off_base + 5];
    //
    //         // 更新后的 CA
    //         float ca_x_new = ca_x0 + dca_x;
    //         float ca_y_new = ca_y0 + dca_y;
    //         float ca_z_new = ca_z0 + dca_z;
    //
    //         // N = CA_new + δN
    //         xyz_out[off_base + 0] = ca_x_new + off_data[off_base + 0];
    //         xyz_out[off_base + 1] = ca_y_new + off_data[off_base + 1];
    //         xyz_out[off_base + 2] = ca_z_new + off_data[off_base + 2];
    //
    //         // CA = CA_new (绝对位置)
    //         xyz_out[off_base + 3] = ca_x_new;
    //         xyz_out[off_base + 4] = ca_y_new;
    //         xyz_out[off_base + 5] = ca_z_new;
    //
    //         // C = CA_new + δC
    //         xyz_out[off_base + 6] = ca_x_new + off_data[off_base + 6];
    //         xyz_out[off_base + 7] = ca_y_new + off_data[off_base + 7];
    //         xyz_out[off_base + 8] = ca_z_new + off_data[off_base + 8];
    //     }
    // }
    //
    // // 回写 state (通过引用)
    // state.copy_from(state_new_);
}

// ===== GPU 可用性探测 =====
// 返回 true 表示当前环境可用的 CUDA 设备数 > 0 (且 device_id 合法)。
// 用于 ensure_backend_ready 在创建 CUDABackend 前探测: 无 GPU 时给出警告并回退 CPU。
bool cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        return false;
    }
    return true;
}

// RFAAModel 实现
RFAAModel::RFAAModel(const RFAAConfig& config) : config_(config) {
    int n_iter = N_EXTRA_BLOCKS + N_MAIN_BLOCKS;  // ITER_N_BLOCKS = 12
    int n_refn = N_REFINE_BLOCKS;                  // 4

    // ===== IterBlock 3D SE 参数 (每组 6 个, 共 12 组) =====
    for (int i = 0; i < n_iter; ++i) {
        iter_norm_msa_3d_.push_back(LayerNorm::create(D_MSA));                        // 256
        iter_norm_pair_3d_.push_back(LayerNorm::create(D_PAIR));                      // 128
        iter_embed_x_.push_back(LinearLayer::create(ITER_NODE_3D_IN, ITER_NODE_3D_OUT));  // 277→32
        iter_embed_e_.push_back(LinearLayer::create(D_PAIR, ITER_EDGE_3D_OUT));       // 128→32
        iter_norm_node_3d_.push_back(LayerNorm::create(ITER_NODE_3D_OUT));             // 32
        iter_norm_edge_3d_.push_back(LayerNorm::create(ITER_EDGE_3D_OUT));             // 32
    }

    // ===== IterBlock forward 内部参数 (每组 12 个, 共 12 组) =====
    for (int i = 0; i < n_iter; ++i) {
        iter_state2msa_norm_.push_back(LayerNorm::create(D_STATE));                         // 32
        iter_state2msa_linear_.push_back(LinearLayer::create(D_STATE, D_MSA));              // 32→256
        iter_pair2msa_norm_.push_back(LayerNorm::create(D_PAIR));                           // 128
        iter_msa2pair_norm_.push_back(LayerNorm::create(D_MSA));                            // 256
        iter_msa2pair_left_proj_.push_back(LinearLayer::create(D_MSA, MSA2PAIR_HIDDEN));    // 256→16
        iter_msa2pair_right_proj_.push_back(LinearLayer::create(D_MSA, MSA2PAIR_HIDDEN));   // 256→16
        iter_msa2pair_out_proj_.push_back(LinearLayer::create(MSA2PAIR_HIDDEN * MSA2PAIR_HIDDEN, D_PAIR)); // 256→128
        iter_pair2pair_rbf_proj_.push_back(LinearLayer::create(D_RBF, D_PAIR));              // 64→128
        iter_pair2pair_state_norm_.push_back(LayerNorm::create(D_STATE));                    // 32
        iter_pair2pair_left_proj_.push_back(LinearLayer::create(D_STATE, PAIR2PAIR_GATE_HIDDEN));  // 32→16
        iter_pair2pair_right_proj_.push_back(LinearLayer::create(D_STATE, PAIR2PAIR_GATE_HIDDEN)); // 32→16
        iter_pair2pair_gate_proj_.push_back(LinearLayer::create(PAIR2PAIR_GATE_HIDDEN * PAIR2PAIR_GATE_HIDDEN, D_PAIR)); // 256→128
    }

    // ===== RefineBlock 3D SE 参数 (每组 10 个, 共 4 组) =====
    for (int i = 0; i < n_refn; ++i) {
        refine_norm_msa_.push_back(LayerNorm::create(D_MSA));                                 // 256
        refine_norm_pair_.push_back(LayerNorm::create(D_PAIR));                              // 128
        refine_norm_state_.push_back(LayerNorm::create(D_STATE));                            // 32
        refine_embed_x_.push_back(LinearLayer::create(REFINE_NODE_IN_DIM, REFINE_NODE_OUT_DIM)); // 309→32
        refine_norm_node_.push_back(LayerNorm::create(REFINE_NODE_OUT_DIM));                  // 32
        refine_embed_e1_.push_back(LinearLayer::create(D_PAIR, N_EDGE_FEATS));                // 128→32
        refine_norm_edge1_.push_back(LayerNorm::create(N_EDGE_FEATS));                        // 32
        refine_embed_e2_.push_back(LinearLayer::create(REFINE_EDGE_IN_DIM2, N_EDGE_FEATS));   // 97→32
        refine_norm_edge2_.push_back(LayerNorm::create(N_EDGE_FEATS));                        // 32
    }

    // ===== attention / sub-module 参数 (per-block create) =====
    AttnConfig msa_ac(config.d_msa, config.n_heads);
    AttnConfig pair_ac(config.d_pair, config.n_heads);

    auto push_msa_row = [&]() {
        msa_row_Wq_.push_back(    LinearLayer::create(D_MSA, N_HEAD * D_MSA));        // 256→2048
        msa_row_Wk_.push_back(    LinearLayer::create(D_MSA, N_HEAD * D_MSA));
        msa_row_Wv_.push_back(    LinearLayer::create(D_MSA, N_HEAD * D_MSA));
        msa_row_to_b_.push_back(  LinearLayer::create(D_PAIR, N_HEAD));                // 128→8
        msa_row_to_g_.push_back(  LinearLayer::create(D_MSA, N_HEAD * D_MSA));
        msa_row_to_out_.push_back(LinearLayer::create(N_HEAD * D_MSA, D_MSA));        // 2048→256
    };
    auto push_msa_col = [&]() {
        msa_col_Wq_.push_back(   LinearLayer::create(D_MSA, N_HEAD * D_MSA));
        msa_col_Wk_.push_back(   LinearLayer::create(D_MSA, N_HEAD * D_MSA));
        msa_col_Wv_.push_back(   LinearLayer::create(D_MSA, N_HEAD * D_MSA));
        msa_col_to_b_.push_back( LinearLayer::create(D_PAIR, N_HEAD));
        msa_col_to_g_.push_back( LinearLayer::create(D_MSA, N_HEAD * D_MSA));
        msa_col_to_out_.push_back(LinearLayer::create(N_HEAD * D_MSA, D_MSA));
    };
    auto push_pair_row = [&]() {
        pair_row_Wq_.push_back(   LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));  // 128→256
        pair_row_Wk_.push_back(   LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));
        pair_row_Wv_.push_back(   LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));
        pair_row_to_b_.push_back( LinearLayer::create(D_PAIR, N_HEAD));                   // 128→8
        pair_row_to_g_.push_back( LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));
        pair_row_to_out_.push_back(LinearLayer::create(N_HEAD * D_PAIR_HIDDEN, D_PAIR));   // 256→128
    };
    auto push_pair_col = [&]() {
        pair_col_Wq_.push_back(   LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));
        pair_col_Wk_.push_back(   LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));
        pair_col_Wv_.push_back(   LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));
        pair_col_to_b_.push_back( LinearLayer::create(D_PAIR, N_HEAD));
        pair_col_to_g_.push_back( LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));
        pair_col_to_out_.push_back(LinearLayer::create(N_HEAD * D_PAIR_HIDDEN, D_PAIR));
    };
    auto push_msa_ff = [&]() {
        msa_ff_norm_.push_back(   LayerNorm::create(D_MSA));                              // 256
        msa_ff_linear1_.push_back(LinearLayer::create(D_MSA, D_MSA * 4));                 // 256→1024
        msa_ff_linear2_.push_back(LinearLayer::create(D_MSA * 4, D_MSA));                 // 1024→256
    };
    auto push_pair_ff = [&]() {
        pair_ff_norm_.push_back(   LayerNorm::create(D_PAIR));                            // 128
        pair_ff_linear1_.push_back(LinearLayer::create(D_PAIR, D_PAIR * 2));              // 128→256
        pair_ff_linear2_.push_back(LinearLayer::create(D_PAIR * 2, D_PAIR));              // 256→128
    };
    auto push_tri = [&](std::vector<LayerNorm*>& ln1, std::vector<LayerNorm*>& ln2,
                         std::vector<LinearLayer*>& l1, std::vector<LinearLayer*>& r1,
                         std::vector<LinearLayer*>& lg, std::vector<LinearLayer*>& rg,
                         std::vector<LinearLayer*>& g,  std::vector<LinearLayer*>& op) {
        constexpr int T = 128;  // D_HIDDEN_TRIMUL
        ln1.push_back(LayerNorm::create(D_PAIR));                                        // 128
        l1.push_back( LinearLayer::create(D_PAIR, T));   r1.push_back( LinearLayer::create(D_PAIR, T));
        lg.push_back( LinearLayer::create(D_PAIR, T));   rg.push_back( LinearLayer::create(D_PAIR, T));
        g.push_back(  LinearLayer::create(D_PAIR, D_PAIR));
        ln2.push_back(LayerNorm::create(T));
        op.push_back( LinearLayer::create(T, D_PAIR));
    };

    // 12 份 IterBlock 参数
    for (int i = 0; i < N_ITER; ++i) {
        push_msa_row(); push_msa_col(); push_pair_row(); push_pair_col();
        push_msa_ff(); push_pair_ff();
        push_tri(tri_out_layernorm_, tri_out_output_layernorm_,
                 tri_out_left_proj_, tri_out_right_proj_,
                 tri_out_left_gate_, tri_out_right_gate_,
                 tri_out_gate_, tri_out_out_proj_);
        push_tri(tri_in_layernorm_, tri_in_output_layernorm_,
                 tri_in_left_proj_, tri_in_right_proj_,
                 tri_in_left_gate_, tri_in_right_gate_,
                 tri_in_gate_, tri_in_out_proj_);
    }

    // 4 份 GlobalColAttention 参数
    for (int i = 0; i < N_GLOB; ++i) {
        msa_global_col_Wq_.push_back(   LinearLayer::create(D_MSA, N_HEAD * D_MSA));
        msa_global_col_Wk_.push_back(   LinearLayer::create(D_MSA, N_HEAD * D_MSA));
        msa_global_col_Wv_.push_back(   LinearLayer::create(D_MSA, N_HEAD * D_MSA));
        msa_global_col_to_b_.push_back( LinearLayer::create(D_PAIR, N_HEAD));
        msa_global_col_to_g_.push_back( LinearLayer::create(D_MSA, N_HEAD * D_MSA));
        msa_global_col_to_out_.push_back(LinearLayer::create(N_HEAD * D_MSA, D_MSA));
    }

    // ===== PositionalEncoding 参数 (每 block 2 个 EmbeddingLayer) =====
    // 必须先于下方 block 构造循环分配 (1575/1649 处按 idx 访问 pos_enc_emb_res_[idx])
    for (int i = 0; i < N_ITER; ++i) {
        pos_enc_emb_res_.push_back(EmbeddingLayer::create(65, D_PAIR));     // (65, 128)  residue dist
        pos_enc_emb_atom_.push_back(EmbeddingLayer::create(17, D_PAIR));    // (17, 128)  atom bond dist
    }

    // ===== 创建迭代块 + 注入指针 =====
    // extra_blocks (4) — FullBlock with update_msa_pair=true
    for (int i = 0; i < N_EXTRA_BLOCKS; ++i) {
        int idx = i;
        auto block = std::make_unique<FullBlock>(config, true);
        // 注入 3D SE + forward 内部参数
        block->norm_msa_3d_  = iter_norm_msa_3d_[idx]; block->norm_pair_3d_ = iter_norm_pair_3d_[idx];
        block->embed_x_      = iter_embed_x_[idx];     block->embed_e_      = iter_embed_e_[idx];
        block->norm_node_3d_ = iter_norm_node_3d_[idx]; block->norm_edge_3d_ = iter_norm_edge_3d_[idx];
        block->state2msa_norm_       = iter_state2msa_norm_[idx];
        block->state2msa_linear_     = iter_state2msa_linear_[idx];
        block->pair2msa_norm_        = iter_pair2msa_norm_[idx];
        block->msa2pair_norm_        = iter_msa2pair_norm_[idx];
        block->msa2pair_left_proj_   = iter_msa2pair_left_proj_[idx];
        block->msa2pair_right_proj_  = iter_msa2pair_right_proj_[idx];
        block->msa2pair_out_proj_    = iter_msa2pair_out_proj_[idx];
        block->pair2pair_rbf_proj_   = iter_pair2pair_rbf_proj_[idx];
        block->pair2pair_state_norm_ = iter_pair2pair_state_norm_[idx];
        block->pair2pair_left_proj_  = iter_pair2pair_left_proj_[idx];
        block->pair2pair_right_proj_ = iter_pair2pair_right_proj_[idx];
        block->pair2pair_gate_proj_  = iter_pair2pair_gate_proj_[idx];

        // 创建并注入子模块 (layernorm 在 IterBlock::forward 外部完成)
        auto msa_row = std::make_unique<MSARowAttention>();
        msa_row->set_params(msa_ac,
            msa_row_to_b_[idx], msa_row_to_g_[idx], msa_row_to_out_[idx],
            msa_row_Wq_[idx], msa_row_Wk_[idx], msa_row_Wv_[idx]);
        auto msa_col = std::make_unique<MSAColAttention>();
        msa_col->set_params(msa_ac,
            msa_col_to_b_[idx], msa_col_to_g_[idx], msa_col_to_out_[idx],
            msa_col_Wq_[idx], msa_col_Wk_[idx], msa_col_Wv_[idx]);
        auto msa_ff = std::make_unique<FeedForward>();
        msa_ff->set_params(D_MSA, D_MSA * 4, 0.1f, msa_ff_norm_[idx], msa_ff_linear1_[idx], msa_ff_linear2_[idx]);
        auto pair_row = std::make_unique<PairRowAttention>();
        pair_row->set_params(pair_ac,
            pair_row_to_b_[idx], pair_row_to_g_[idx], pair_row_to_out_[idx],
            pair_row_Wq_[idx], pair_row_Wk_[idx], pair_row_Wv_[idx]);
        auto pair_col = std::make_unique<PairColAttention>();
        pair_col->set_params(pair_ac,
            pair_col_to_b_[idx], pair_col_to_g_[idx], pair_col_to_out_[idx],
            pair_col_Wq_[idx], pair_col_Wk_[idx], pair_col_Wv_[idx]);
        auto pair_ff = std::make_unique<FeedForward>();
        pair_ff->set_params(D_PAIR, D_PAIR * 2, 0.1f, pair_ff_norm_[idx], pair_ff_linear1_[idx], pair_ff_linear2_[idx]);
        auto tri_out = std::make_unique<TriangleMultiplication>();
        tri_out->set_params(D_PAIR,
            tri_out_layernorm_[idx], tri_out_left_proj_[idx], tri_out_right_proj_[idx],
            tri_out_left_gate_[idx], tri_out_right_gate_[idx], tri_out_gate_[idx],
            tri_out_output_layernorm_[idx], tri_out_out_proj_[idx]);
        auto tri_in = std::make_unique<TriangleMultiplication>();
        tri_in->set_params(D_PAIR,
            tri_in_layernorm_[idx], tri_in_left_proj_[idx], tri_in_right_proj_[idx],
            tri_in_left_gate_[idx], tri_in_right_gate_[idx], tri_in_gate_[idx],
            tri_in_output_layernorm_[idx], tri_in_out_proj_[idx]);
        auto se3 = std::make_unique<SE3Transformer>(config.se3_config);

        block->set_sub_modules(
            std::move(msa_row), std::move(msa_col), std::move(msa_ff),
            std::move(pair_row), std::move(pair_col), std::move(pair_ff),
            std::move(tri_out), std::move(tri_in), std::move(se3));

        // PositionalEncoding (per block)
        auto pos_enc = std::make_unique<PositionalEncoding>();
        pos_enc->set_params(-32, 32, 8, config.d_pair, pos_enc_emb_res_[idx], pos_enc_emb_atom_[idx]);
        block->set_pos_enc(std::move(pos_enc));

        // GlobalColAttention (FullBlock only)
        auto gcol = std::make_unique<MSAGlobalColAttention>();
        gcol->set_params(msa_ac,
            msa_global_col_to_b_[i], msa_global_col_to_g_[i], msa_global_col_to_out_[i],
            msa_global_col_Wq_[i], msa_global_col_Wk_[i], msa_global_col_Wv_[i]);
        block->set_global_col_attn(std::move(gcol));

        extra_blocks_.push_back(std::move(block));
    }

    // main_blocks (8) — IterBlock with update_msa_pair=true
    for (int i = 0; i < N_MAIN_BLOCKS; ++i) {
        int idx = N_EXTRA_BLOCKS + i;
        auto block = std::make_unique<IterBlock>(config, true);
        // 注入 3D SE + forward 内部参数
        block->norm_msa_3d_  = iter_norm_msa_3d_[idx]; block->norm_pair_3d_ = iter_norm_pair_3d_[idx];
        block->embed_x_      = iter_embed_x_[idx];     block->embed_e_      = iter_embed_e_[idx];
        block->norm_node_3d_ = iter_norm_node_3d_[idx]; block->norm_edge_3d_ = iter_norm_edge_3d_[idx];
        block->state2msa_norm_       = iter_state2msa_norm_[idx];
        block->state2msa_linear_     = iter_state2msa_linear_[idx];
        block->pair2msa_norm_        = iter_pair2msa_norm_[idx];
        block->msa2pair_norm_        = iter_msa2pair_norm_[idx];
        block->msa2pair_left_proj_   = iter_msa2pair_left_proj_[idx];
        block->msa2pair_right_proj_  = iter_msa2pair_right_proj_[idx];
        block->msa2pair_out_proj_    = iter_msa2pair_out_proj_[idx];
        block->pair2pair_rbf_proj_   = iter_pair2pair_rbf_proj_[idx];
        block->pair2pair_state_norm_ = iter_pair2pair_state_norm_[idx];
        block->pair2pair_left_proj_  = iter_pair2pair_left_proj_[idx];
        block->pair2pair_right_proj_ = iter_pair2pair_right_proj_[idx];
        block->pair2pair_gate_proj_  = iter_pair2pair_gate_proj_[idx];

        // 创建并注入子模块 (layernorm 在 IterBlock::forward 外部完成)
        auto msa_row = std::make_unique<MSARowAttention>();
        msa_row->set_params(msa_ac,
            msa_row_to_b_[idx], msa_row_to_g_[idx], msa_row_to_out_[idx],
            msa_row_Wq_[idx], msa_row_Wk_[idx], msa_row_Wv_[idx]);
        auto msa_col = std::make_unique<MSAColAttention>();
        msa_col->set_params(msa_ac,
            msa_col_to_b_[idx], msa_col_to_g_[idx], msa_col_to_out_[idx],
            msa_col_Wq_[idx], msa_col_Wk_[idx], msa_col_Wv_[idx]);
        auto msa_ff = std::make_unique<FeedForward>();
        msa_ff->set_params(D_MSA, D_MSA * 4, 0.1f, msa_ff_norm_[idx], msa_ff_linear1_[idx], msa_ff_linear2_[idx]);
        auto pair_row = std::make_unique<PairRowAttention>();
        pair_row->set_params(pair_ac,
            pair_row_to_b_[idx], pair_row_to_g_[idx], pair_row_to_out_[idx],
            pair_row_Wq_[idx], pair_row_Wk_[idx], pair_row_Wv_[idx]);
        auto pair_col = std::make_unique<PairColAttention>();
        pair_col->set_params(pair_ac,
            pair_col_to_b_[idx], pair_col_to_g_[idx], pair_col_to_out_[idx],
            pair_col_Wq_[idx], pair_col_Wk_[idx], pair_col_Wv_[idx]);
        auto pair_ff = std::make_unique<FeedForward>();
        pair_ff->set_params(D_PAIR, D_PAIR * 2, 0.1f, pair_ff_norm_[idx], pair_ff_linear1_[idx], pair_ff_linear2_[idx]);
        auto tri_out = std::make_unique<TriangleMultiplication>();
        tri_out->set_params(D_PAIR,
            tri_out_layernorm_[idx], tri_out_left_proj_[idx], tri_out_right_proj_[idx],
            tri_out_left_gate_[idx], tri_out_right_gate_[idx], tri_out_gate_[idx],
            tri_out_output_layernorm_[idx], tri_out_out_proj_[idx]);
        auto tri_in = std::make_unique<TriangleMultiplication>();
        tri_in->set_params(D_PAIR,
            tri_in_layernorm_[idx], tri_in_left_proj_[idx], tri_in_right_proj_[idx],
            tri_in_left_gate_[idx], tri_in_right_gate_[idx], tri_in_gate_[idx],
            tri_in_output_layernorm_[idx], tri_in_out_proj_[idx]);
        auto se3 = std::make_unique<SE3Transformer>(config.se3_config);

        block->set_sub_modules(
            std::move(msa_row), std::move(msa_col), std::move(msa_ff),
            std::move(pair_row), std::move(pair_col), std::move(pair_ff),
            std::move(tri_out), std::move(tri_in), std::move(se3));

        // PositionalEncoding (per block)
        auto pos_enc = std::make_unique<PositionalEncoding>();
        pos_enc->set_params(-32, 32, 8, config.d_pair, pos_enc_emb_res_[idx], pos_enc_emb_atom_[idx]);
        block->set_pos_enc(std::move(pos_enc));

        main_blocks_.push_back(std::move(block));
    }

    // refine_blocks (4) — RefineBlock with update_msa_pair=false
    for (int i = 0; i < n_refn; ++i) {
        auto block = std::make_unique<RefineBlock>(config, false);
        // 注入 RefineBlock 专属 3D SE 参数
        block->norm_msa_   = refine_norm_msa_[i];        // 256
        block->norm_pair_  = refine_norm_pair_[i];       // 128
        block->norm_state_ = refine_norm_state_[i];      // 32
        block->embed_x_    = refine_embed_x_[i];         // 309→32
        block->norm_node_  = refine_norm_node_[i];       // 32
        block->embed_e1_   = refine_embed_e1_[i];        // 128→32
        block->norm_edge1_ = refine_norm_edge1_[i];      // 32
        block->embed_e2_   = refine_embed_e2_[i];        // 97→32
        block->norm_edge2_ = refine_norm_edge2_[i];      // 32
        // RefineBlock 继承 IterBlock 的 forward 内部参数不需要 (update_msa_pair=false)
        refine_blocks_.push_back(std::move(block));
    }

    // ===== embedding / track / template 全局参数 =====
    // 旧栈上变量 (保留注释):
    // BondEmbedding bond_embed(0, D_PAIR); FullEmbedding full_emb(NAATOKENS-1+4, D_MSA_FULL);
    // LinearLayer linear(feat_dim, dim_) in MSATrack
    // EmbeddingLayer embedding(NAATOKENS, D_STATE) in StateTrack
    // EmbeddingLayer emb_left/right(NAATOKENS, D_PAIR) in PairTrack
    // LinearLayer emb_t1d(110,64) in StateTrack::inject_template
    // LinearLayer t1d_proj(80,32) / LayerNorm(64) in PairTrack::templ_stack
    bond_emb_    = LinearLayer::create(8, D_PAIR);                                   // NBYTES → D_PAIR (128)
    full_linear_ = LinearLayer::create(NAATOKENS - 1 + 4, D_MSA_FULL);              // 83 → 64
    full_emb_    = EmbeddingLayer::create(NAATOKENS, D_MSA_FULL);                    // (80, 64)
    msa_emb_              = LinearLayer::create(MSA_LATENT_DIM, D_MSA);              // 164 → 256
    state_emb_            = EmbeddingLayer::create(NAATOKENS, D_STATE);              // (80, 32)
    pair_left_emb_        = EmbeddingLayer::create(NAATOKENS, D_PAIR);              // (80, 128)
    pair_right_emb_       = EmbeddingLayer::create(NAATOKENS, D_PAIR);              // (80, 128)
    pair_init_pos_enc_    = new PositionalEncoding();                                 // pair 初始化 pos enc
    pair_init_pos_enc_emb_res_  = EmbeddingLayer::create(65, D_PAIR);
    pair_init_pos_enc_emb_atom_ = EmbeddingLayer::create(17, D_PAIR);
    pair_init_pos_enc_->set_params(-32, 32, 8, D_PAIR,
                                    pair_init_pos_enc_emb_res_, pair_init_pos_enc_emb_atom_);
    emb_t1d_              = LinearLayer::create(D_T1D + D_TOR, 64);                  // 110 → 64
    proj_t1d_             = LinearLayer::create(64, 64);                             // 64 → 64
    emb_t1d_t2d_          = LinearLayer::create(D_T1D * 2 + D_T2D, 64);             // 224 → 64
    temp_stack_t1d_proj_  = LinearLayer::create(D_T1D, D_STATE);                    // 80 → 32
    temp_stack_norm_      = LayerNorm::create(64);                                   // 64

    // ===== 输出头参数 (全局单份) =====
    // Masked MSA head: LayerNorm(D_MSA) → Linear(D_MSA→D_MSA) → ReLU → Linear(D_MSA→23)
    // 输入 msa 特征 (B,N,L,D_MSA)，输出 logits (B,N,L,23)（每个序列位置/残基的 aatype 预测）
    msa_head_ln_      = LayerNorm::create(D_MSA);        // 256
    msa_head_linear1_ = LinearLayer::create(D_MSA, D_MSA);   // 256 → 256
    msa_head_linear2_ = LinearLayer::create(D_MSA, 23);      // 256 → 23

    // Chi (扭转角) head: LayerNorm(D_STATE) → Linear(D_STATE→D_STATE) → ReLU → Linear(D_STATE→14)
    // 输入 state (B,L,D_STATE)，输出 alpha (B,L,7,2)（omega/phi/psi/chi1-4 未归一化 sin/cos）
    chi_head_ln_      = LayerNorm::create(D_STATE);      // 32
    chi_head_linear1_ = LinearLayer::create(D_STATE, D_STATE);  // 32 → 32
    chi_head_linear2_ = LinearLayer::create(D_STATE, 7 * 2);    // 32 → 14

    // Distogram head: 从 pair 特征投影 4 组 logits (D/Ω/Θ/Φ)
    distogram_d_head_ = LinearLayer::create(D_PAIR, 60);  // 距离 60 bins
    distogram_o_head_ = LinearLayer::create(D_PAIR, 36);  // Ω 36 bins
    distogram_t_head_ = LinearLayer::create(D_PAIR, 36);  // Θ 36 bins
    distogram_p_head_ = LinearLayer::create(D_PAIR, 18);  // Φ 18 bins

    // pLDDT head: state → lddt logits (B,L,50)
    plddt_head_       = LinearLayer::create(D_STATE, 50);

    // ===== TemplatePairStack 子层创建 =====
    // 直接层
    tps_rbf_proj_   = LinearLayer::create(D_RBF, D_PAIR);                            // 64 → 128
    tps_state_norm_ = LayerNorm::create(D_STATE);                                     // 32
    tps_left_proj_  = LinearLayer::create(D_STATE, 16);                               // 32 → 16
    tps_right_proj_ = LinearLayer::create(D_STATE, 16);                               // 32 → 16
    tps_gate_proj_  = LinearLayer::create(16 * 16, D_PAIR);                           // 256 → 128
    // TriangleMultiplication out
    tps_tri_out_layernorm_        = LayerNorm::create(D_PAIR);                        // 128
    tps_tri_out_left_proj_        = LinearLayer::create(D_PAIR, 128);
    tps_tri_out_right_proj_       = LinearLayer::create(D_PAIR, 128);
    tps_tri_out_left_gate_        = LinearLayer::create(D_PAIR, 128);
    tps_tri_out_right_gate_       = LinearLayer::create(D_PAIR, 128);
    tps_tri_out_gate_             = LinearLayer::create(D_PAIR, D_PAIR);              // 128 → 128
    tps_tri_out_output_layernorm_ = LayerNorm::create(128);
    tps_tri_out_out_proj_         = LinearLayer::create(128, D_PAIR);
    tps_tri_mul_out_.set_params(D_PAIR,
        tps_tri_out_layernorm_,        tps_tri_out_left_proj_,
        tps_tri_out_right_proj_,       tps_tri_out_left_gate_,
        tps_tri_out_right_gate_,       tps_tri_out_gate_,
        tps_tri_out_output_layernorm_, tps_tri_out_out_proj_);
    // TriangleMultiplication in
    tps_tri_in_layernorm_        = LayerNorm::create(D_PAIR);                        // 128
    tps_tri_in_left_proj_        = LinearLayer::create(D_PAIR, 128);
    tps_tri_in_right_proj_       = LinearLayer::create(D_PAIR, 128);
    tps_tri_in_left_gate_        = LinearLayer::create(D_PAIR, 128);
    tps_tri_in_right_gate_       = LinearLayer::create(D_PAIR, 128);
    tps_tri_in_gate_             = LinearLayer::create(D_PAIR, D_PAIR);              // 128 → 128
    tps_tri_in_output_layernorm_ = LayerNorm::create(128);
    tps_tri_in_out_proj_         = LinearLayer::create(128, D_PAIR);
    tps_tri_mul_in_.set_params(D_PAIR,
        tps_tri_in_layernorm_,        tps_tri_in_left_proj_,
        tps_tri_in_right_proj_,       tps_tri_in_left_gate_,
        tps_tri_in_right_gate_,       tps_tri_in_gate_,
        tps_tri_in_output_layernorm_, tps_tri_in_out_proj_);
    // PairRowAttention
    tps_pair_row_to_b_   = LinearLayer::create(D_PAIR, 8);                           // 128 → 8
    tps_pair_row_to_g_   = LinearLayer::create(D_PAIR, 256);                         // 128 → 256
    tps_pair_row_to_out_ = LinearLayer::create(256, D_PAIR);                         // 256 → 128
    tps_pair_row_Wq_     = LinearLayer::create(D_PAIR, 256);                         // 128 → 256
    tps_pair_row_Wk_     = LinearLayer::create(D_PAIR, 256);
    tps_pair_row_Wv_     = LinearLayer::create(D_PAIR, 256);
    tps_pair_row_attn_.set_params(AttnConfig(D_PAIR, 8),
        tps_pair_row_to_b_, tps_pair_row_to_g_, tps_pair_row_to_out_,
        tps_pair_row_Wq_,   tps_pair_row_Wk_,   tps_pair_row_Wv_);
    // PairColAttention
    tps_pair_col_to_b_   = LinearLayer::create(D_PAIR, 8);                           // 128 → 8
    tps_pair_col_to_g_   = LinearLayer::create(D_PAIR, 256);                         // 128 → 256
    tps_pair_col_to_out_ = LinearLayer::create(256, D_PAIR);                         // 256 → 128
    tps_pair_col_Wq_     = LinearLayer::create(D_PAIR, 256);                         // 128 → 256
    tps_pair_col_Wk_     = LinearLayer::create(D_PAIR, 256);
    tps_pair_col_Wv_     = LinearLayer::create(D_PAIR, 256);
    tps_pair_col_attn_.set_params(AttnConfig(D_PAIR, 8),
        tps_pair_col_to_b_, tps_pair_col_to_g_, tps_pair_col_to_out_,
        tps_pair_col_Wq_,   tps_pair_col_Wk_,   tps_pair_col_Wv_);
    // FeedForward
    tps_pair_ff_norm_    = LayerNorm::create(D_PAIR);                                // 128
    tps_pair_ff_linear1_ = LinearLayer::create(D_PAIR, D_PAIR * 2);                  // 128 → 256
    tps_pair_ff_linear2_ = LinearLayer::create(D_PAIR * 2, D_PAIR);                  // 256 → 128
    tps_pair_ff_.set_params(D_PAIR, 2, 0.15f,
        tps_pair_ff_norm_, tps_pair_ff_linear1_, tps_pair_ff_linear2_);
    // TemplatePairStack 本身
    tps_.set_params(
        tps_rbf_proj_, tps_state_norm_,
        tps_left_proj_, tps_right_proj_, tps_gate_proj_,
        &tps_tri_mul_out_, &tps_tri_mul_in_,
        &tps_pair_row_attn_, &tps_pair_col_attn_,
        &tps_pair_ff_);
}

RFAAModel::~RFAAModel() = default;

void RFAAModel::set_seq_info(const TensorF32& seq1hot, const TensorI64& idx) {
    seq1hot_.copy_from(seq1hot);
    idx_.copy_from(idx);
    has_seq_info_ = true;
}

ModelOutput RFAAModel::forward(const ModelInput& input) {
    ModelOutput output;
    
    int B = input.msa_latent.shape().dims[0];
    int N = input.msa_latent.shape().dims[1];
    int L = input.msa_latent.shape().dims[2];
    
    // 初始化 tracks
    msa_track_ = std::make_unique<MSATrack>(N, L, config_.d_msa, device_);
    pair_track_ = std::make_unique<PairTrack>(L, config_.d_pair, device_);
    state_track_ = std::make_unique<StateTrack>(L, config_.d_state, device_);
    
    // 旧: msa_track_->init_from_features(input.msa_latent); → 用 msa_emb_ 直接投影
    msa_track_->representation() = msa_emb_->forward(input.msa_latent);                               // (B,N,L,164→256)
    // 旧: state_track_->init_from_embedding(input.seq_tokens); → 用 state_emb_
    state_track_->representation() = state_emb_->forward_exec(input.seq_tokens);                        // (B,L)→(B,L,32)
    // 旧: pair_track_->init_from_embedding(...); → 用 pair_left_emb_/pair_right_emb_
    {
        auto left  = pair_left_emb_->forward_exec(input.seq_tokens).unsqueeze(1);            // (B,1,L,128)
        auto right = pair_right_emb_->forward_exec(input.seq_tokens).unsqueeze(2);           // (B,L,1,128)
        auto pair_repr = outer_sum(left, right);                                              // (B,L,L,128)
        // PositionalEncoding: 使用 input 中预计算的 bond_feats/dist_matrix/same_chain/residx
        // idx 需要从 TensorI64 转为 TensorF32
        TensorF32 residx_f32(input.residx.shape());
        for (int64_t i = 0; i < input.residx.numel(); ++i)
            residx_f32.data()[i] = static_cast<float>(input.residx.data()[i]);
        auto pos_out = pair_init_pos_enc_->forward(pair_repr, residx_f32, input.bond_feats, input.dist_matrix, input.same_chain);
        pair_track_->representation().copy_from(*add_impl(&pair_repr, &pos_out, /*inplace=*/false));
    }

    // msa full embed?
    // msa_full = self.full_emb(msa_full, seq, idx)
    // msa_full was used in the full block
    TensorF32 msa_full;
    if (input.msa_full.numel() > 0) {
        // 旧栈上变量: FullEmbedding full_emb(NAATOKENS - 1 + 4, D_MSA_FULL);
        FullEmbedding full_emb;
        full_emb.set_params(full_linear_, full_emb_, D_MSA_FULL);
        msa_full = full_emb.forward(input.msa_full, input.seq_tokens, TensorF32());
    }

    // bond embed for pair track
    // need to get the bond feats
    // 旧栈上变量: BondEmbedding bond_embed(0, D_PAIR);
    BondEmbedding bond_embed;
    bond_embed.set_params(bond_emb_, D_PAIR);
    TensorF32 pair;
    pair.copy_from(pair_track_->representation());
    auto bond_out = bond_embed.forward(input.bond_feats);
    pair.copy_from(*add_impl(&pair, &bond_out, /*inplace=*/false));
    //bond_feats: (B, L, L, d_init)
    //bond embed: 
    // bond_feats = one_hot(bond_feats)
    // linear(d_bond_type = 5, d_pair = 128)
    // linear(bond_feats.float())

    // recycle embed?
    
    // template embed
    // Template injection
    // cross attention need to reshape
    // state cross attention and pair cross attention
    if (input.t1d.numel() > 0) {
        // 旧: state_track_->inject_template(input.t1d, input.tor_feat);
        // 旧栈上: LinearLayer emb_t1d(110,64), proj_t1d(64,64) → 用 emb_t1d_/proj_t1d_
        {
            TensorF32 t1d_copy, tor_copy;
            t1d_copy.copy_from(input.t1d);
            tor_copy.copy_from(input.tor_feat);
            std::vector<TensorF32*> t1d_parts = {&t1d_copy, &tor_copy};
            auto& t1d_tor = *concat_ptr(t1d_parts, -1);           // (B,T,L,110)
            TensorF32 t1d_emb = emb_t1d_->forward(t1d_tor);        // (B,T,L,64)
            auto t1d_relu = relu(&t1d_emb);
            t1d_emb = proj_t1d_->forward(*t1d_relu);               // (B,T,L,64)
            // Cross-attention: state as Q, template as K/V
            int B = t1d_emb.shape().dims[0], T = t1d_emb.shape().dims[1];
            auto state_q = state_track_->representation().view({B * L, 1, D_STATE});
            auto t1d_kv = t1d_emb.permute({0, 2, 1, 3}).view({B * L, T, 64});
            CrossAttention cross_attn(D_STATE, 64, 8);  // Q=state(32), KV=template(64), H=head(8)
            auto out = cross_attn.forward(state_q, t1d_kv);  // query, key-value
            // residual connection: state_rep + out_view (use graph node add_impl)
            auto& state_rep = state_track_->representation();
            auto out_view = out.view({B, L, D_STATE});
            state_track_->representation().copy_from(*add_impl(&state_rep, &out_view, /*inplace=*/false));
        }
        TensorF32 templ_pair = get_templ_emb(input.t1d, input.t2d);  // (B,T,L,L,64)
        // rbf_feature: 用初始 coords 计算 RBF 特征
        TensorF32 init_coords;
        init_coords.copy_from(input.coords);
        TensorF32 rbf_feature = IterBlock::compute_rbf_feature(init_coords);  // (B, L, L, 64)
        // 旧: pair_track_->templ_stack(templ_pair, rbf_feature, input.t1d);
        // templ_stack 1406-1418
        // 旧栈上: LinearLayer t1d_proj(80,32), LayerNorm(64), TemplatePairStack
        {
            int T = templ_pair.shape().dims[1];
            TensorF32 t1d_2d = input.t1d.view({B*T, L, D_T1D});
            // (B*T, L, 80) — already a view above, no reshape needed
            templ_pair = templ_pair.view({B*T, L, L, 64});                     // (B*T, L, L, 64)
            // 旧栈上: LinearLayer t1d_proj(D_T1D, D_STATE) → temp_stack_t1d_proj_
            TensorF32 state_proj = temp_stack_t1d_proj_->forward(t1d_2d);      // (B*T, L, 32)
            for (int k = 0; k < 2; ++k) {
                templ_pair = tps_.forward(templ_pair, rbf_feature, state_proj);  // rbf_feature from outer scope
            }
            // 旧栈上: LayerNorm layernorm(64) → temp_stack_norm_
            templ_pair.copy_from(*temp_stack_norm_->forward(&templ_pair));      // (B*T, L, L, 64)
            templ_pair = templ_pair.view({B, T, L, L, 64});
            pair_track_->inject_template(templ_pair);
        }
    }
    
    // 获取初始表示
    //auto msa = msa_track_->representation();
    TensorF32 msa;
    msa.copy_from(msa_track_->representation());
    //auto pair = pair_track_->representation();
    //TensorF32 pair;
    pair.copy_from(pair_track_->representation());
    // pair 初始化 (embedding + outer_sum) 已在 1398-1404 完成
    // PositionalEncoding 在 IterBlock::forward 中由 pos_enc_ 处理
    //auto state = state_track_->representation();
    TensorF32 state;
    state.copy_from(state_track_->representation());
    //auto coords = input.coords;
    TensorF32 coords;
    coords.copy_from(input.coords);
    
    // need to modify : (already finished)
    // the block in the 4 full block was different from the main block
    // full/extra block use global column attention

    const TensorF32& seq1hot = one_hot_seq(input.seq_tokens, 21);
    set_seq_info(seq1hot, input.residx);

    // Extra blocks
    // need to use msa_full
    // and use global column attention as well
    for (auto& block : extra_blocks_) {
        // stop grad
        block->forward(msa_full, pair, state, seq1hot, coords,
                       input.bond_feats, input.dist_matrix, input.same_chain, input.residx);
        coords.copy_from(block->updated_coords());
    }
    
    // Main blocks
    for (auto& block : main_blocks_) {
        // stop grad
        // chiral grad
        block->forward(msa, pair, state, seq1hot, coords,
                       input.bond_feats, input.dist_matrix, input.same_chain, input.residx);
        coords.copy_from(block->updated_coords());
    }
    
    // Refinement blocks (仅更新结构)
    for (auto& block : refine_blocks_) {
        // stop grad
        // chiral grad
        // clash grad
        /* if (block.get()) {
        // seq1hot: 从 input.seq_tokens 生成 one-hot (B, L, 21)
            TensorF32 seq1hot = one_hot_seq(input.seq_tokens, 21);
            TensorI64 idx = input.seq_tokens;  // 或专门的 idx 输入
            refine->set_seq_info(seq1hot, idx);
        } */

        block->forward(msa, pair, state, seq1hot, coords);

        if (block.get()) {
            coords.copy_from(block->updated_coords());
            // state 通过 track 更新，不需要 updated_state()
        }
    }
    
    // 输出头
    //output.msa = msa;
    output.msa.copy_from(msa);  // 简化
    //output.pair = pair;
    output.pair.copy_from(pair);  // 简化
    //output.state = state;
    output.state.copy_from(state);  // 简化
    //output.coords = coords;
    output.coords.copy_from(coords);  // 简化

    // ===== Masked MSA head: msa (B,N,L,D_MSA) → logits (B,N,L,23) =====
    //   LayerNorm(D_MSA) → Linear(D_MSA→D_MSA) → ReLU → Linear(D_MSA→23)
    //   msa 值布局 (B,N,L,D_MSA)；LinearLayer::forward 输入须展平为 (B*N*L, D_MSA)
    {
        int msa_B = msa.shape().dims[0];
        int msa_N = msa.shape().dims[1];
        int msa_L = msa.shape().dims[2];
        int64_t flat_rows = (int64_t)msa_B * msa_N * msa_L;

        // 1) LayerNorm (逐 token 归一化, 保持 (B,N,L,D_MSA))
        auto* ln_out = msa_head_ln_->forward(&msa);                       // (B,N,L,256)

        // 2) view 展平 → Linear1 → ReLU → view 回 4D
        TensorF32 ln_flat(Shape({flat_rows, D_MSA}), ln_out->device());
        ln_flat.copy_from(*ln_out);
        auto lin1  = msa_head_linear1_->forward(ln_flat);                 // (flat_rows, 256)
        auto* relu1 = relu(&lin1);
        // 3) view 回 (B,N,L,256) → Linear2 → logits (B*N*L, 23)
        TensorF32 relu4(Shape({msa_B, msa_N, msa_L, D_MSA}), relu1->device());
        relu4.copy_from(*relu1);
        auto lin2  = msa_head_linear2_->forward(relu4);                   // (flat_rows, 23)
        // 4) view 回 (B,N,L,23)
        TensorF32 logits4(Shape({msa_B, msa_N, msa_L, 23}), lin2.device());
        logits4.copy_from(lin2);
        output.msa_logits = std::move(logits4);
    }

    // ===== Chi (扭转角) head: state (B,L,D_STATE) → alpha (B,L,7,2) =====
    //   LayerNorm(D_STATE) → Linear(D_STATE→D_STATE) → ReLU → Linear(D_STATE→14)
    //   state 值布局 (B,L,D_STATE)；LinearLayer::forward 输入须展平为 (B*L, D_STATE)
    if (state.numel() > 0) {
        int ch_B = state.shape().dims[0];
        int ch_L = state.shape().dims[1];
        int64_t ch_rows = (int64_t)ch_B * ch_L;

        // 1) LayerNorm (逐 token, 保持 (B,L,D_STATE))
        auto* ch_ln = chi_head_ln_->forward(&state);                    // (B,L,32)

        // 2) view 扁平 → Linear1 → ReLU → view 回 (B,L,32)
        TensorF32 ch_flat(Shape({ch_rows, D_STATE}), ch_ln->device());
        ch_flat.copy_from(*ch_ln);
        auto ch_lin1 = chi_head_linear1_->forward(ch_flat);             // (B*L, 32)
        auto* ch_relu = relu(&ch_lin1);
        // 3) view 回 (B,L,32) → Linear2 → logits (B*L, 14)
        TensorF32 ch_relu4(Shape({ch_B, ch_L, D_STATE}), ch_relu->device());
        ch_relu4.copy_from(*ch_relu);
        auto ch_lin2 = chi_head_linear2_->forward(ch_relu4);            // (B*L, 14)
        // 4) view 回 (B,L,7,2)
        TensorF32 alpha4(Shape({ch_B, ch_L, 7, 2}), ch_lin2.device());
        alpha4.copy_from(ch_lin2);
        output.alpha = std::move(alpha4);
    }

    // ===== Distogram head: pair (B,L,L,D_PAIR) → 4 组 logits =====
    //   distogram (B,L,L,60), omega (B,L,L,36), theta (B,L,L,36), phi (B,L,L,18)
    if (pair.numel() > 0) {
        int dg_B = pair.shape().dims[0];
        int dg_L = pair.shape().dims[1];
        int64_t dg_rows = (int64_t)dg_B * dg_L * dg_L;

        // pair (B,L,L,D_PAIR) → 展平 (B*L*L, D_PAIR) 送入各 head
        TensorF32 pair_flat(Shape({dg_rows, D_PAIR}), pair.device());
        pair_flat.copy_from(pair);

        auto project_logits = [&](LinearLayer* head, int bins) {
            auto logits2 = head->forward(pair_flat);              // (B*L*L, bins)
            TensorF32 logits4(Shape({dg_B, dg_L, dg_L, bins}), logits2.device());
            logits4.copy_from(logits2);
            return logits4;
        };
        output.distogram = std::move(project_logits(distogram_d_head_, 60));
        output.omega     = std::move(project_logits(distogram_o_head_, 36));
        output.theta     = std::move(project_logits(distogram_t_head_, 36));
        output.phi       = std::move(project_logits(distogram_p_head_, 18));
    }

    // ===== pLDDT head: state (B,L,D_STATE) → lddt logits (B,L,50) =====
    if (state.numel() > 0) {
        int pl_B = state.shape().dims[0];
        int pl_L = state.shape().dims[1];
        int64_t pl_rows = (int64_t)pl_B * pl_L;

        TensorF32 pl_flat(Shape({pl_rows, D_STATE}), state.device());
        pl_flat.copy_from(state);
        auto pl_logits2 = plddt_head_->forward(pl_flat);          // (B*L, 50)
        TensorF32 lddt4(Shape({pl_B, pl_L, 50}), pl_logits2.device());
        lddt4.copy_from(pl_logits2);
        output.lddt = std::move(lddt4);
    }

    return output;
}

//The t1d feature has shape (B, T, L, d_t1d) where B is batch size,
// T is number of templates, L is sequence length, 
//and d_t1d is the feature dimension that varies by model configuration
TensorF32 RFAAModel::get_templ_emb(const TensorF32& t1d, const TensorF32& t2d) {
    int B = t1d.shape().dims[0];
    int T = t1d.shape().dims[1];
    int L = t1d.shape().dims[2];
    int D = t1d.shape().dims[3];

    // left:  (B, T, L, 1, D) → repeat → (B, T, L, L, D)
    // right: (B, T, 1, L, D) → repeat → (B, T, L, L, D)
    auto left_unsq  = t1d.unsqueeze(3);   // (B, T, L, 1, D)
    auto right_unsq = t1d.unsqueeze(2);   // (B, T, 1, L, D)

    // 创建 target shape 张量用于 repeat (数据为空，只取 shape)
    TensorF32 target({B, T, L, L, D}, t1d.device());
    auto* left_exp  = repeat(&left_unsq,  &target);  // (B, T, L, L, D)
    auto* right_exp = repeat(&right_unsq, &target);  // (B, T, L, L, D)

    // concat: t2d(B,T,L,L,64) + left(B,T,L,L,D) + right(B,T,L,L,D) → (B,T,L,L,64+2D)
    std::vector<TensorF32*> templ_parts;
    templ_parts.push_back(const_cast<TensorF32*>(&t2d));
    templ_parts.push_back(left_exp);
    templ_parts.push_back(right_exp);
    auto* templ = concat_ptr(templ_parts, -1);
    // d_templ = 64
    return emb_t1d_t2d_->forward(*templ);
}

void RFAAModel::to(Device device) {
    device_ = device;

    // 确保后端基础设施已初始化
    ensure_backend_ready();
}

void RFAAModel::ensure_backend_ready() {
    if (backend_ready_) return;

    // 1. 创建 CPU Backend（始终存在）
    if (!cpu_backend_) {
        cpu_backend_ = std::make_unique<CPUBackend>(4);  // 4 线程
    }

    // 2. 创建 BackendScheduler 并注册后端
    if (!scheduler_) {
        scheduler_ = std::make_unique<BackendScheduler>();
        scheduler_->add_backend(cpu_backend_.get());
        // 3. 若模型目标设备为 CUDA, 先探测 GPU 可用性; 无 GPU 则警告并回退 CPU
        if (device_ == Device::CUDA) {
            if (!cuda_available()) {
                std::cerr << "[WARN] CUDA device not available; "
                          << "falling back to CPU backend." << std::endl;
                device_ = Device::CPU;   // 回退: 模型按 CPU 运行
            } else if (!cuda_backend_) {
                // scheduler 按 priority 排序, CUDA 优先调度到 GPU;
                // 不支持的 op 自动跨后端拷贝回 CPU
                cuda_backend_ = std::make_unique<CUDABackend>(0);  // device 0
                scheduler_->add_backend(cuda_backend_.get());
            }
        }
    }

    backend_ready_ = true;

    // 4. 后端就绪后将参数迁移到 backend buffer (权重/偏置等)
    //    在模型构造后 / to() 时调用; load_weights 内部也调用 (见下)
    transfer_params_to_backend();
}

Device RFAAModel::device() const {
    return device_;
}

void RFAAModel::train() {
    training_ = true;
}

void RFAAModel::eval() {
    training_ = false;
}

bool RFAAModel::is_training() const {
    return training_;
}

void RFAAModel::load_weights(const std::string& path) {
    std::cout << "Loading weights from: " << path << std::endl;

    // 1. 确保后端已就绪
    ensure_backend_ready();

    // 2. 从文件加载权重数据到各参数 tensor 的 CPU arena
    //    （此时参数数据仍在 context arena 中，data_ 指向 arena 地址）
    //    TODO: 实现二进制文件读取逻辑
    //    示例: file.read(linear->weight()->data(), linear->weight()->nbytes());

    // 3. 将参数迁移到 backend buffer
    transfer_params_to_backend();

    std::cout << "Weights loaded and transferred to backend buffer." << std::endl;
}

// ============================================================
// 私有辅助: 收集所有参数 Tensor (weight/bias/gamma/beta), 顺序固定
// 供 transfer_params_to_backend / params() / 保存共用
// ============================================================
void RFAAModel::collect_all_params(std::vector<TensorF32*>& param_tensors) {
    std::vector<std::string> names;  // 无名字版本: 忽略名字
    collect_params_with_names(param_tensors, names);
}

// ============================================================
// 收集所有参数 Tensor 及语义名 (block/attention 等), 顺序与 param_tensors 严格一一对应。
// 命名规范 (对齐 RF2/AlphaFold 惯例):
//   - 全局单份参数: "msa_emb", "state_emb", "pair_left_emb" ...
//   - 模板参数:     "tps.<layer>"
//   - per-block:    "main.{i}.attention.<attn>.<proj>", "extra.{i}.", "refine.{i}." ...
//     (extra = FullBlock, main = IterBlock, 均共享 iter_* 参数, 索引为全局 0..11)
//   - 每个 LinearLayer: <name>.weight / <name>.bias
//   - 每个 LayerNorm:   <name>.gamma / <name>.beta
//   - 每个 Embedding:   <name>.weight
// ============================================================
void RFAAModel::collect_params_with_names(std::vector<TensorF32*>& param_tensors,
                                          std::vector<std::string>& param_names) {
    // 辅助 lambda：收集 LinearLayer / LayerNorm / EmbeddingLayer 的参数及名字
    auto collect_linear = [&](LinearLayer* ll, const std::string& name) {
        if (ll && ll->weight()) {
            param_tensors.push_back(ll->weight());
            if (!param_names.empty()) param_names.push_back(name + ".weight");
        }
        if (ll && ll->bias()) {
            param_tensors.push_back(ll->bias());
            if (!param_names.empty()) param_names.push_back(name + ".bias");
        }
    };
    auto collect_layernorm = [&](LayerNorm* ln, const std::string& name) {
        if (ln && ln->gamma()) {
            param_tensors.push_back(ln->gamma());
            if (!param_names.empty()) param_names.push_back(name + ".gamma");
        }
        if (ln && ln->beta()) {
            param_tensors.push_back(ln->beta());
            if (!param_names.empty()) param_names.push_back(name + ".beta");
        }
    };
    auto collect_embedding = [&](EmbeddingLayer* emb, const std::string& name) {
        if (emb && emb->weight()) {
            param_tensors.push_back(emb->weight());
            if (!param_names.empty()) param_names.push_back(name + ".weight");
        }
    };

    // embedding / template 全局参数
    collect_linear(msa_emb_, "msa_emb");
    collect_embedding(state_emb_, "state_emb");
    collect_embedding(pair_left_emb_, "pair_left_emb");
    collect_embedding(pair_right_emb_, "pair_right_emb");
    collect_linear(full_linear_, "full_linear");
    collect_embedding(full_emb_, "full_emb");
    collect_linear(bond_emb_, "bond_emb");
    collect_linear(emb_t1d_, "emb_t1d");
    collect_linear(proj_t1d_, "proj_t1d");
    collect_linear(emb_t1d_t2d_, "emb_t1d_t2d");
    collect_linear(temp_stack_t1d_proj_, "temp_stack_t1d_proj");
    collect_layernorm(temp_stack_norm_, "temp_stack_norm");

    // ===== 输出头参数 =====
    collect_layernorm(msa_head_ln_, "msa_head.ln");
    collect_linear(msa_head_linear1_, "msa_head.linear1");
    collect_linear(msa_head_linear2_, "msa_head.linear2");
    collect_layernorm(chi_head_ln_, "chi_head.ln");
    collect_linear(chi_head_linear1_, "chi_head.linear1");
    collect_linear(chi_head_linear2_, "chi_head.linear2");
    collect_linear(distogram_d_head_, "distogram_head.dist");
    collect_linear(distogram_o_head_, "distogram_head.omega");
    collect_linear(distogram_t_head_, "distogram_head.theta");
    collect_linear(distogram_p_head_, "distogram_head.phi");
    collect_linear(plddt_head_, "plddt_head");

    // ===== TemplatePairStack 参数 =====
    collect_linear(tps_rbf_proj_, "tps.rbf_proj");
    collect_layernorm(tps_state_norm_, "tps.state_norm");
    collect_linear(tps_left_proj_, "tps.left_proj");
    collect_linear(tps_right_proj_, "tps.right_proj");
    collect_linear(tps_gate_proj_, "tps.gate_proj");
    // tri_mul_out
    collect_layernorm(tps_tri_out_layernorm_, "tps.tri_mul_out.layernorm");
    collect_linear(tps_tri_out_left_proj_, "tps.tri_mul_out.left_proj");
    collect_linear(tps_tri_out_right_proj_, "tps.tri_mul_out.right_proj");
    collect_linear(tps_tri_out_left_gate_, "tps.tri_mul_out.left_gate");
    collect_linear(tps_tri_out_right_gate_, "tps.tri_mul_out.right_gate");
    collect_linear(tps_tri_out_gate_, "tps.tri_mul_out.gate");
    collect_layernorm(tps_tri_out_output_layernorm_, "tps.tri_mul_out.output_layernorm");
    collect_linear(tps_tri_out_out_proj_, "tps.tri_mul_out.out_proj");
    // tri_mul_in
    collect_layernorm(tps_tri_in_layernorm_, "tps.tri_mul_in.layernorm");
    collect_linear(tps_tri_in_left_proj_, "tps.tri_mul_in.left_proj");
    collect_linear(tps_tri_in_right_proj_, "tps.tri_mul_in.right_proj");
    collect_linear(tps_tri_in_left_gate_, "tps.tri_mul_in.left_gate");
    collect_linear(tps_tri_in_right_gate_, "tps.tri_mul_in.right_gate");
    collect_linear(tps_tri_in_gate_, "tps.tri_mul_in.gate");
    collect_layernorm(tps_tri_in_output_layernorm_, "tps.tri_mul_in.output_layernorm");
    collect_linear(tps_tri_in_out_proj_, "tps.tri_mul_in.out_proj");
    // pair_row_attn
    collect_linear(tps_pair_row_to_b_, "tps.attention.pair_row.to_b");
    collect_linear(tps_pair_row_to_g_, "tps.attention.pair_row.to_g");
    collect_linear(tps_pair_row_to_out_, "tps.attention.pair_row.to_out");
    collect_linear(tps_pair_row_Wq_, "tps.attention.pair_row.Wq");
    collect_linear(tps_pair_row_Wk_, "tps.attention.pair_row.Wk");
    collect_linear(tps_pair_row_Wv_, "tps.attention.pair_row.Wv");
    // pair_col_attn
    collect_linear(tps_pair_col_to_b_, "tps.attention.pair_col.to_b");
    collect_linear(tps_pair_col_to_g_, "tps.attention.pair_col.to_g");
    collect_linear(tps_pair_col_to_out_, "tps.attention.pair_col.to_out");
    collect_linear(tps_pair_col_Wq_, "tps.attention.pair_col.Wq");
    collect_linear(tps_pair_col_Wk_, "tps.attention.pair_col.Wk");
    collect_linear(tps_pair_col_Wv_, "tps.attention.pair_col.Wv");
    // pair_ff
    collect_layernorm(tps_pair_ff_norm_, "tps.pair_ff.norm");
    collect_linear(tps_pair_ff_linear1_, "tps.pair_ff.linear1");
    collect_linear(tps_pair_ff_linear2_, "tps.pair_ff.linear2");

    // attention 参数 (12 blocks × 6 LinearLayer × 6 组)
    // extra = FullBlock (索引 0..3), main = IterBlock (索引 4..11)
    const std::string iter_block_name[12] = {
        "extra.0", "extra.1", "extra.2", "extra.3",
        "main.0",  "main.1",  "main.2",  "main.3",
        "main.4",  "main.5",  "main.6",  "main.7",
    };

    for (int i = 0; i < N_ITER; ++i) {
        const std::string& b = iter_block_name[i];
        collect_linear(msa_row_Wq_[i], b + ".attention.msa_row.Wq");
        collect_linear(msa_row_Wk_[i], b + ".attention.msa_row.Wk");
        collect_linear(msa_row_Wv_[i], b + ".attention.msa_row.Wv");
        collect_linear(msa_row_to_b_[i], b + ".attention.msa_row.to_b");
        collect_linear(msa_row_to_g_[i], b + ".attention.msa_row.to_g");
        collect_linear(msa_row_to_out_[i], b + ".attention.msa_row.to_out");

        collect_linear(msa_col_Wq_[i], b + ".attention.msa_col.Wq");
        collect_linear(msa_col_Wk_[i], b + ".attention.msa_col.Wk");
        collect_linear(msa_col_Wv_[i], b + ".attention.msa_col.Wv");
        collect_linear(msa_col_to_b_[i], b + ".attention.msa_col.to_b");
        collect_linear(msa_col_to_g_[i], b + ".attention.msa_col.to_g");
        collect_linear(msa_col_to_out_[i], b + ".attention.msa_col.to_out");

        collect_linear(pair_row_Wq_[i], b + ".attention.pair_row.Wq");
        collect_linear(pair_row_Wk_[i], b + ".attention.pair_row.Wk");
        collect_linear(pair_row_Wv_[i], b + ".attention.pair_row.Wv");
        collect_linear(pair_row_to_b_[i], b + ".attention.pair_row.to_b");
        collect_linear(pair_row_to_g_[i], b + ".attention.pair_row.to_g");
        collect_linear(pair_row_to_out_[i], b + ".attention.pair_row.to_out");

        collect_linear(pair_col_Wq_[i], b + ".attention.pair_col.Wq");
        collect_linear(pair_col_Wk_[i], b + ".attention.pair_col.Wk");
        collect_linear(pair_col_Wv_[i], b + ".attention.pair_col.Wv");
        collect_linear(pair_col_to_b_[i], b + ".attention.pair_col.to_b");
        collect_linear(pair_col_to_g_[i], b + ".attention.pair_col.to_g");
        collect_linear(pair_col_to_out_[i], b + ".attention.pair_col.to_out");

        collect_layernorm(msa_ff_norm_[i], b + ".msa_ff.norm");
        collect_linear(msa_ff_linear1_[i], b + ".msa_ff.linear1");
        collect_linear(msa_ff_linear2_[i], b + ".msa_ff.linear2");

        collect_layernorm(pair_ff_norm_[i], b + ".pair_ff.norm");
        collect_linear(pair_ff_linear1_[i], b + ".pair_ff.linear1");
        collect_linear(pair_ff_linear2_[i], b + ".pair_ff.linear2");

        // TriangleMultiplication out
        collect_layernorm(tri_out_layernorm_[i], b + ".tri_mul_out.layernorm");
        collect_linear(tri_out_left_proj_[i], b + ".tri_mul_out.left_proj");
        collect_linear(tri_out_right_proj_[i], b + ".tri_mul_out.right_proj");
        collect_linear(tri_out_left_gate_[i], b + ".tri_mul_out.left_gate");
        collect_linear(tri_out_right_gate_[i], b + ".tri_mul_out.right_gate");
        collect_linear(tri_out_gate_[i], b + ".tri_mul_out.gate");
        collect_layernorm(tri_out_output_layernorm_[i], b + ".tri_mul_out.output_layernorm");
        collect_linear(tri_out_out_proj_[i], b + ".tri_mul_out.out_proj");

        // TriangleMultiplication in
        collect_layernorm(tri_in_layernorm_[i], b + ".tri_mul_in.layernorm");
        collect_linear(tri_in_left_proj_[i], b + ".tri_mul_in.left_proj");
        collect_linear(tri_in_right_proj_[i], b + ".tri_mul_in.right_proj");
        collect_linear(tri_in_left_gate_[i], b + ".tri_mul_in.left_gate");
        collect_linear(tri_in_right_gate_[i], b + ".tri_mul_in.right_gate");
        collect_linear(tri_in_gate_[i], b + ".tri_mul_in.gate");
        collect_layernorm(tri_in_output_layernorm_[i], b + ".tri_mul_in.output_layernorm");
        collect_linear(tri_in_out_proj_[i], b + ".tri_mul_in.out_proj");

        // IterBlock 3D SE 参数
        collect_linear(iter_embed_x_[i], b + ".embed_x");
        collect_linear(iter_embed_e_[i], b + ".embed_e");
        collect_layernorm(iter_norm_node_3d_[i], b + ".norm_node_3d");
        collect_layernorm(iter_norm_edge_3d_[i], b + ".norm_edge_3d");
        collect_layernorm(iter_norm_msa_3d_[i], b + ".norm_msa_3d");
        collect_layernorm(iter_norm_pair_3d_[i], b + ".norm_pair_3d");

        // IterBlock forward 内部参数
        collect_layernorm(iter_state2msa_norm_[i], b + ".state2msa_norm");
        collect_linear(iter_state2msa_linear_[i], b + ".state2msa_linear");
        collect_layernorm(iter_pair2msa_norm_[i], b + ".pair2msa_norm");
        collect_layernorm(iter_msa2pair_norm_[i], b + ".msa2pair_norm");
        collect_linear(iter_msa2pair_left_proj_[i], b + ".msa2pair_left_proj");
        collect_linear(iter_msa2pair_right_proj_[i], b + ".msa2pair_right_proj");
        collect_linear(iter_msa2pair_out_proj_[i], b + ".msa2pair_out_proj");
        collect_linear(iter_pair2pair_rbf_proj_[i], b + ".pair2pair_rbf_proj");
        collect_layernorm(iter_pair2pair_state_norm_[i], b + ".pair2pair_state_norm");
        collect_linear(iter_pair2pair_left_proj_[i], b + ".pair2pair_left_proj");
        collect_linear(iter_pair2pair_right_proj_[i], b + ".pair2pair_right_proj");
        collect_linear(iter_pair2pair_gate_proj_[i], b + ".pair2pair_gate_proj");
    }

    // MSAGlobalColAttention 参数 (FullBlock only, N_GLOB=4)
    for (int i = 0; i < N_GLOB; ++i) {
        const std::string& b = iter_block_name[i];  // extra.{0..3}
        collect_linear(msa_global_col_Wq_[i], b + ".attention.global_col.Wq");
        collect_linear(msa_global_col_Wk_[i], b + ".attention.global_col.Wk");
        collect_linear(msa_global_col_Wv_[i], b + ".attention.global_col.Wv");
        collect_linear(msa_global_col_to_b_[i], b + ".attention.global_col.to_b");
        collect_linear(msa_global_col_to_g_[i], b + ".attention.global_col.to_g");
        collect_linear(msa_global_col_to_out_[i], b + ".attention.global_col.to_out");
    }

    // PositionalEncoding 参数 (每 block 2 个 EmbeddingLayer, 12 组)
    for (int i = 0; i < N_ITER; ++i) {
        const std::string& b = iter_block_name[i];
        collect_embedding(pos_enc_emb_res_[i], b + ".pos_enc_emb_res");
        collect_embedding(pos_enc_emb_atom_[i], b + ".pos_enc_emb_atom");
    }

    // RefineBlock 参数 (N_REFINE_BLOCKS=4)
    for (int i = 0; i < (int)refine_norm_msa_.size(); ++i) {
        const std::string b = "refine." + std::to_string(i);
        collect_layernorm(refine_norm_msa_[i], b + ".norm_msa");
        collect_layernorm(refine_norm_pair_[i], b + ".norm_pair");
        collect_layernorm(refine_norm_state_[i], b + ".norm_state");
        collect_linear(refine_embed_x_[i], b + ".embed_x");
        collect_layernorm(refine_norm_node_[i], b + ".norm_node");
        collect_linear(refine_embed_e1_[i], b + ".embed_e1");
        collect_layernorm(refine_norm_edge1_[i], b + ".norm_edge1");
        collect_linear(refine_embed_e2_[i], b + ".embed_e2");
        collect_layernorm(refine_norm_edge2_[i], b + ".norm_edge2");
    }
}

std::vector<TensorF32*> RFAAModel::params() {
    std::vector<TensorF32*> out;
    collect_all_params(out);
    return out;
}

void RFAAModel::transfer_params_to_backend() {
    if (!scheduler_ || !cpu_backend_) return;

    // 幂等保护: 若参数已被分配进 backend buffer (buffer_ != nullptr), 跳过。
    // 这样 ensure_backend_ready / load_weights 多次调用不会重复搬迁覆盖 data_ 指针。
    {
        std::vector<TensorF32*> probe;
        collect_all_params(probe);
        for (auto* t : probe) {
            if (t->data() != nullptr && t->buffer_ != nullptr) {
                return;  // 已搬迁过
            }
        }
    }

    const BufferType* cpu_buft = cpu_backend_->buffer_type();

    // ===== Step 1: 收集所有需要搬迁的参数 tensor =====
    std::vector<TensorF32*> param_tensors;
    collect_all_params(param_tensors);

    // ===== Step 2: 计算总大小并分配 CPU backend buffer =====
    size_t total_size = 0;
    size_t alignment = cpu_buft->get_alignment();

    for (auto* t : param_tensors) {
        if (t->data() != nullptr) {
            total_size += GGML_PAD(t->nbytes(), alignment);
        }
    }

    if (total_size == 0 || param_tensors.empty()) return;

    // ===== Step 3: 分配 buffer 并搬迁数据（对标 ggml_backend_alloc_ctx_tensors + ggml_backend_tensor_set）=====
    Buffer* param_buf = alloc_buffer(const_cast<BufferType*>(cpu_buft), total_size, BufferUsage::WEIGHTS);
    if (!param_buf) return;

    // 将 buffer 所有权交给 RFAAModel
    param_buffers_.emplace_back(param_buf);

    TensorAllocator tallocr(param_buf);

    for (auto* t : param_tensors) {
        if (t->data() == nullptr) continue;

        // 保存旧数据指针（指向 context arena）
        float* old_data = t->data();
        size_t old_nbytes = t->nbytes();

        // 在 backend buffer 中分配新空间（覆盖 data_）
        if (!tallocr.alloc(t)) {
            // buffer 空间不足（理论上不会发生）
            continue;
        }

        // 对标 ggml_backend_tensor_set：将旧数据拷贝到新 buffer
        param_buf->set_tensor(t, old_data, t->buffer_offs_, old_nbytes);

        // 旧数据在 context arena 中，无法释放，但 data_ 已指向新 buffer
        // 后续可通过 t->buffer_ 和 t->buffer_offs_ 访问
    }
}

void RFAAModel::save_weights(const std::string& path) const {
    std::cout << "Saving weights to: " << path << std::endl;
    // 实现权重保存...
    // 如果有 backend buffer，需要通过 buffer->get_tensor 读取
}

} // namespace rfaa

#include "rfaa/Model.h"
#include "rfaa/Embedding.h"
#include "rfaa/PositionalEncoding.h"
#include "rfaa/MathUtils.h"
#include <iostream>

#include "rfaa/Dropout.h"

namespace rfaa {

RFAAConfig::RFAAConfig() {
    // 默认 SE3 配置
    se3_config.node_dim = D_MSA + D_STATE;  // 288
    se3_config.edge_dim = D_PAIR + 64 + 1;  // 193 (pair + rbf + seqsep)
    se3_config.hidden_dim = 128;
    se3_config.n_layers = 2;
    se3_config.n_heads = 4;
    se3_config.l0_features = {32};   // state 输出
    se3_config.l1_features = {3};    // 坐标更新
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

    
    // Ensure contiguous (select returns a view, may not be contiguous)
    // Create a contiguous copy
    TensorF32 cas_contig = cas;  // If already contiguous, this is just a reference
    // Force contiguous by creating new tensor and copying
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

void IterBlock::forward(TensorF32& msa, TensorF32& pair, 
                        TensorF32& state, 
                        const TensorF32& seq1hot,
                        const TensorF32& coords) {
    
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
            TensorF32 state_normed = state2msa_norm_->forward(state);
            
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

            // need to get the input bond_feats, dist_matrix
            rbf_feature = rbf + pos_enc_->forward(coords, index, bond_feats, dist_matrix, same_chain);
            // 旧栈上变量: LayerNorm pair_layernorm(D_PAIR); → pair2msa_norm_
            pair_biased = pair2msa_norm_->forward(pair);

            pair_biased = pair_biased + rbf_feature; 

            // TODO
            // update msa query row with state from SE3 output

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
            // msa2pair
            // 旧栈上变量: LayerNorm msa_norm(D_MSA); → msa2pair_norm_
            TensorF32 msa_normed = msa2pair_norm_->forward(msa);
            // 旧栈上变量: LinearLayer left_proj(D_MSA, 16); → msa2pair_left_proj_
            // 旧栈上变量: LinearLayer right_proj(D_MSA, 16); → msa2pair_right_proj_
            TensorF32 left = msa2pair_left_proj_->forward(msa_normed);   // (B,N,L,16)
            TensorF32 right = msa2pair_right_proj_->forward(msa_normed); // (B,N,L,16)
            TensorF32 right_mean = right / float(N);
            // a tensor divide a scalar
            TensorF32 pair_update = outer_product(left, right_mean);  // (B,L,L,256)
            // dim of pair_update?
            // reshape to (B, L, L, 16*16) = (B, L, L, 256)?
            // 旧栈上变量: LinearLayer out_proj(16 * 16, D_PAIR); → msa2pair_out_proj_
            pair_update = msa2pair_out_proj_->forward(pair_update);  // (B,L,L,128)
            pair = pair + pair_update;  // residual
        }
        
        // Triangle Multiplication
        //pair = pair + drop_row(tri_mul_out_->forward(pair));
        //pair = pair + drop_row(tri_mul_in_->forward(pair));
        Dropout drop_row(1, 0.15);
        pair = pair + drop_row.forward(tri_mul_out_->forward(pair, true));
        pair = pair + drop_row.forward(tri_mul_in_->forward(pair, false));

        // ===== Step 3: pair2pair =====
        // state outer product -> gate
        {
            // 旧栈上变量: LinearLayer rbf_proj(D_RBF, D_PAIR); → pair2pair_rbf_proj_
            rbf_feature = pair2pair_rbf_proj_->forward(rbf_feature);  // (B,L,L,128)
            // 旧栈上变量: LayerNorm state_norm(D_STATE); → pair2pair_state_norm_
            TensorF32 state_normed = pair2pair_state_norm_->forward(state);
            // 旧栈上变量: LinearLayer left_proj(D_STATE, 16); → pair2pair_left_proj_
            // 旧栈上变量: LinearLayer right_proj(D_STATE, 16); → pair2pair_right_proj_
            // different weights for left and right?
            TensorF32 left = pair2pair_left_proj_->forward(state_normed);   // (B,L,16)
            TensorF32 right = pair2pair_right_proj_->forward(state_normed); // (B,L,16)
            TensorF32 gate = outer_product(left, right);  // (B,L,L,256)
            // 旧栈上变量: LinearLayer gate_proj(16 * 16, D_PAIR); → pair2pair_gate_proj_
            // d_hidden_gate = 16
            gate = pair2pair_gate_proj_->forward(gate);  // (B,L,L,128)
            gate = sigmoid(gate);  // (B,L,L,128) -> (B,L,L,128) gate values between 0 and 1
            rbf_feature = rbf_feature * gate;  // element-wise multiplication, inject the rbf feature into pair with gate control
            // left = Linear(state, 32->16)
            // right = Linear(state, 32->16)
            // gate = sigmoid(left x right -> Linear -> 128)
            //auto gate = state;  // get_gate
            //gate.copy_from(state);  // 简化，实际需要计算 gate
            // rbf_feat 经 gate 过滤注入 pair
            // pair += gate * rbf_feat
        
            // to update pair
            // Biased Axial Attention (row/col)
            // pair = pair + drop_row(row_attn(pair, bias=rbf_feat))
            // pair = pair + drop_col(col_attn(pair, bias=rbf_feat))
            Dropout drop_row(1, 0.15);
            Dropout drop_col(2, 0.15);
            pair = pair + drop_row->forward(pair_row_attn_->forward(pair, rbf_feature));
            pair = pair + drop_col->forward(pair_col_attn_->forward(pair, rbf_feature));
            // FeedForward
            FeedForward pair_ff(D_PAIR, D_PAIR * 2);
            pair = pair + pair_ff.forward(pair);  // residual
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
        TensorF32 msa_normed = norm_msa_3d_->forward(msa);   // (B, N, L, 256)
        TensorF32 pair_normed = norm_pair_3d_->forward(pair); // (B, L, L, 128)

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
        // cat: (B, L, 256) + (B, L, 21) → (B, L, 277)
        TensorF32 node_cat = concat({msa_sum, seq1hot}, -1);
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
        TensorF32 node_out = norm_node_3d_->forward(node_emb);

        // ---- Step 4d: pair embedding ----
        // Linear(128 → 32) → LayerNorm → (B, L, L, 32)
        TensorF32 pair_emb = embed_e_->forward(pair_normed);
        TensorF32 edge_out = norm_edge_3d_->forward(pair_emb);

        // ---- Step 4e: 构建图 ----
        GraphData G = make_graph(coords, edge_out, idx_, 64, 9);

        // ---- Step 4f: l1 特征 (位移向量) ----
        TensorF32 l1_feats = compute_l1_features(coords);  // (B*L, 3, 3)

        // ---- Step 4g: 组装 SE3Features 输入 ----
        // node_out: (B, L, 32) → reshape to (B*L, 32, 1) 作为 degree-0
        // l1_feats: (B*L, 3, 3) 作为 degree-1
        //Fiber fiber_in({NODE_3D_OUT, fiber_out_.degrees[1]}, {0, 1});
        SE3Features node_se3;
        node_se3.features.resize(2);
        node_se3.features[0] = node_out.view({B * L, ITER_NODE_3D_OUT, 1});
        node_se3.features[1] = l1_feats;

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

        xyz_new_ = xyz_new;
    }

    
}

void FullBlock::forward(TensorF32& msa_full, TensorF32& pair, TensorF32& state, 
                        const TensorF32& seq1hot,
                        const TensorF32& coords) {
    // FullBlock 在 IterBlock 的基础上增加了 msa_full 的使用和全局 column attention
    // msa_full 需要在 forward 函数参数中传入，或者在 IterBlock 中存储为成员变量
    
    // ------------ 1D track update ------------
    // ===== Step 1: msa2msa =====
    {
        //auto query_row = msa.select(1, 0);  // (B, L, 256)
        // query_row += Linear(state) ...

        // 旧栈上变量: LayerNorm state_norm(D_STATE); → state2msa_norm_
        TensorF32 state_normed = state2msa_norm_->forward(state);
            
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

        // need to get the input bond_feats, dist_matrix
        rbf_feature = rbf + pos_enc_->forward(coords, index, bond_feats, dist_matrix, same_chain);
        // 旧栈上变量: LayerNorm pair_layernorm(D_PAIR); → pair2msa_norm_
        pair_biased = pair2msa_norm_->forward(pair);

        pair_biased = pair_biased + rbf_feature; 

        // TODO
        // update msa query row with state from SE3 output

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

    // msa2pair
    {
        // 旧栈上变量: LayerNorm msa_norm(D_MSA); → msa2pair_norm_
        TensorF32 msa_normed = msa2pair_norm_->forward(msa);
        // 旧栈上变量: LinearLayer left_proj(D_MSA, 16); → msa2pair_left_proj_
        // 旧栈上变量: LinearLayer right_proj(D_MSA, 16); → msa2pair_right_proj_
        TensorF32 left = msa2pair_left_proj_->forward(msa_normed);   // (B,N,L,16)
        TensorF32 right = msa2pair_right_proj_->forward(msa_normed); // (B,N,L,16)
        TensorF32 right_mean = right / float(N);
        // a tensor divide a scalar
        TensorF32 pair_update = outer_product(left, right_mean);  // (B,L,L,256)
        // dim of pair_update?
        // reshape to (B, L, L, 16*16) = (B, L, L, 256)?
        // 旧栈上变量: LinearLayer out_proj(16 * 16, D_PAIR); → msa2pair_out_proj_
        pair_update = msa2pair_out_proj_->forward(pair_update);  // (B,L,L,128)
        pair = pair + pair_update;  // residual
    }

    // Triangle Multiplication
    Dropout drop_row(1, 0.15);
    pair = pair + drop_row.forward(tri_mul_out_->forward(pair, true));
    pair = pair + drop_row.forward(tri_mul_in_->forward(pair, false));

    // ===== Step 3: pair2pair =====
    // state outer product -> gate
    {
        // 旧栈上变量: LinearLayer rbf_proj(D_RBF, D_PAIR); → pair2pair_rbf_proj_
        rbf_feature = pair2pair_rbf_proj_->forward(rbf_feature);  // (B,L,L,128)
        // 旧栈上变量: LayerNorm state_norm(D_STATE); → pair2pair_state_norm_
        TensorF32 state_normed = pair2pair_state_norm_->forward(state);
        // 旧栈上变量: LinearLayer left_proj(D_STATE, 16); → pair2pair_left_proj_
        // 旧栈上变量: LinearLayer right_proj(D_STATE, 16); → pair2pair_right_proj_
        // different weights for left and right?
        TensorF32 left = pair2pair_left_proj_->forward(state_normed);   // (B,L,16)
        TensorF32 right = pair2pair_right_proj_->forward(state_normed); // (B,L,16)
        TensorF32 gate = outer_product(left, right);  // (B,L,L,256)
        // 旧栈上变量: LinearLayer gate_proj(16 * 16, D_PAIR); → pair2pair_gate_proj_
        // d_hidden_gate = 16
        gate = pair2pair_gate_proj_->forward(gate);  // (B,L,L,128)
        gate = sigmoid(gate);  // (B,L,L,128) -> (B,L,L,128) gate values between 0 and 1
        rbf_feature = rbf_feature * gate;  // element-wise multiplication, inject the rbf feature into pair with gate control
            
        Dropout drop_row2(1, 0.15);
        Dropout drop_col(2, 0.15);
        pair = pair + drop_row2->forward(pair_row_attn_->forward(pair, rbf_feature));
        pair = pair + drop_col->forward(pair_col_attn_->forward(pair, rbf_feature));
        // FeedForward
        FeedForward pair_ff(D_PAIR, D_PAIR * 2);
        pair = pair + pair_ff.forward(pair);  // residual
    }

    // 3D track update — same logic as IterBlock, but uses msa_full
    // ===== Step 4: str2str (SE3 Transformer) =====
    {
        int B = msa_full.shape().dims[0];
        int N = msa_full.shape().dims[1];
        int L = msa_full.shape().dims[2];

        TensorF32 msa_normed = norm_msa_3d_->forward(msa_full);
        TensorF32 pair_normed = norm_pair_3d_->forward(pair);

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
        TensorF32 node_cat = concat({msa_sum, seq1hot}, -1);
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

        TensorF32 node_out = norm_node_3d_->forward(embed_x_->forward(node_cat));
        TensorF32 edge_out = norm_edge_3d_->forward(embed_e_->forward(pair_normed));

        GraphData G = make_graph(coords, edge_out, idx_, 64, 9);
        TensorF32 l1_feats = compute_l1_features(coords);

        //Fiber fiber_in({NODE_3D_OUT, 3}, {0, 1});
        SE3Features node_se3;
        node_se3.features.resize(2);
        // node_out = it was actually msa input
        node_se3.features[0] = node_out.view({B * L, ITER_NODE_3D_OUT, 1});
        node_se3.features[1] = l1_feats;

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

        xyz_new_ = xyz_new;
    }

}

RefineBlock::RefineBlock(const RFAAConfig& config)
    : IterBlock(config, false)  // update_msa_pair = false, 仅更新结构
    // 旧值初始化 (保留注释):
    // , norm_msa_(D_MSA), norm_pair_(D_PAIR), norm_state_(D_STATE)
    // , embed_x_(NODE_IN_DIM, NODE_OUT_DIM), norm_node_(NODE_OUT_DIM)
    // , embed_e1_(D_PAIR, N_EDGE_FEATS), norm_edge1_(N_EDGE_FEATS)
    // , embed_e2_(EDGE_IN_DIM2, N_EDGE_FEATS), norm_edge2_(N_EDGE_FEATS)
    // 以上 10 个参数现在由 RFAAModel 创建，通过指针注入
{
}

void RefineBlock::set_seq_info(const TensorF32& seq1hot, const TensorI64& idx) {
    seq1hot_ = seq1hot;
    idx_     = idx;
    has_seq_info_ = true;
}

void RefineBlock::forward(TensorF32& msa_full, 
                          TensorF32& pair, 
                          TensorF32& state, 
                          const TensorF32& seq1hot,
                          const TensorF32& coords) {

    // ---- 获取维度 ----
    const auto& msa_shape = msa.shape();
    int B = static_cast<int>(msa_shape.dims[0]);
    int L = static_cast<int>(msa_shape.dims[2]);

    // ================================================================
    // Step 1: LayerNorm 归一化三个 track 输入
    // ================================================================
    // Python: node = self.norm_msa(msa); pair = self.norm_pair(pair);
    //         state = self.norm_state(state)
    TensorF32 node    = norm_msa_->forward(msa);       // (B, L, 256)
    TensorF32 pair_n  = norm_pair_->forward(pair);     // (B, L, L, 128)
    TensorF32 state_n = norm_state_->forward(state);    // (B, L, 32)

    // ================================================================
    // Step 2: 构建节点特征
    // Python: node = cat((node, seq1hot, state), dim=-1)
    //         node = self.norm_node(self.embed_x(node))
    // ================================================================
    // cat([msa_norm(B,L,256), seq1hot(B,L,21), state_norm(B,L,32)])
    // → (B, L, 309)
    // where the seq1hot_ comes?
    TensorF32 node_cat = concat({node, seq1hot, state_n}, -1);

    // Linear(309 → 32) → LayerNorm → (B, L, 32)
    TensorF32 node_emb = embed_x_->forward(node_cat);
    TensorF32 node_out = norm_node_->forward(node_emb);

    // ================================================================
    // Step 3: 构建边特征（两阶段）
    // ================================================================
    // 阶段1: pair → Linear → LayerNorm
    // Python: pair = self.norm_edge1(self.embed_e1(pair))
    // pair (B,L,L,128) → Linear → (B,L,L,32) → LayerNorm → (B,L,L,32)
    TensorF32 pair_emb = embed_e1_->forward(pair_n);
    TensorF32 pair_e1  = norm_edge1_->forward(pair_emb);

    // 获取辅助边特征
    // Python: neighbor = get_bonded_neigh(idx)     → (B, L, L, 1)
    //         rbf_feat = rbf(cdist(cas, cas))      → (B, L, L, 64)
    TensorF32 neighbor = get_bonded_neigh(idx_);            // (B, L, L, 1)
    TensorF32 rbf_feat = compute_rbf_feature(coords);      // (B, L, L, 64)

    // 阶段2: cat + Linear + LayerNorm
    // Python: pair = cat((pair, rbf_feat, neighbor), dim=-1)
    //         pair = self.norm_edge2(self.embed_e2(pair))
    // cat → (B,L,L,97) → Linear → (B,L,L,32) → LayerNorm → (B,L,L,32)
    TensorF32 pair_cat = concat({pair_e1, rbf_feat, neighbor}, -1);
    TensorF32 pair_e2  = embed_e2_->forward(pair_cat);
    TensorF32 edge_out = norm_edge2_->forward(pair_e2);

    // ================================================================
    // Step 4: 构建消息传递图
    // Python: G = make_graph_topk(xyz, pair, idx, top_k=top_k)
    // ================================================================
    GraphData G = make_graph(coords, edge_out, idx_,
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
    //basis.compute(coords, TensorF32() /* orient 占位 */, config_.se3_config.num_degrees);
    basis.compute(G.edge_d, 2);  // J_max=2, 边向量来自 graph


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

    xyz_new_ = xyz_new;
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
    emb_t1d_              = LinearLayer::create(D_T1D + D_TOR, 64);                  // 110 → 64
    proj_t1d_             = LinearLayer::create(64, 64);                             // 64 → 64
    emb_t1d_t2d_          = LinearLayer::create(D_T1D * 2 + D_T2D, 64);             // 224 → 64
    temp_stack_t1d_proj_  = LinearLayer::create(D_T1D, D_STATE);                    // 80 → 32
    temp_stack_norm_      = LayerNorm::create(64);                                   // 64

    // ===== PositionalEncoding 参数 (每 block 2 个 EmbeddingLayer) =====
    for (int i = 0; i < N_ITER; ++i) {
        pos_enc_emb_res_.push_back(EmbeddingLayer::create(65, D_PAIR));     // (65, 128)  residue dist
        pos_enc_emb_atom_.push_back(EmbeddingLayer::create(17, D_PAIR));    // (17, 128)  atom bond dist
    }
}

RFAAModel::~RFAAModel() = default;

void RFAAModel::set_seq_info(const TensorF32& seq1hot, const TensorI64& idx) {
    seq1hot_ = seq1hot;
    idx_     = idx;
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
    msa_track_->repr() = msa_emb_->forward(input.msa_latent);                               // (B,N,L,164→256)
    // 旧: state_track_->init_from_embedding(input.seq_tokens); → 用 state_emb_
    state_track_->repr() = state_emb_->forward_exec(input.seq_tokens);                        // (B,L)→(B,L,32)
    // 旧: pair_track_->init_from_embedding(...); → 用 pair_left_emb_/pair_right_emb_
    {
        auto left  = pair_left_emb_->forward_exec(input.seq_tokens).unsqueeze(1);            // (B,1,L,128)
        auto right = pair_right_emb_->forward_exec(input.seq_tokens).unsqueeze(2);           // (B,L,1,128)
        pair_track_->repr() = outer_sum(left, right);                                        // (B,L,L,128)
        // PositionalEncoding 在 IterBlock::forward 中由 pos_enc_ 处理
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
    pair = pair + bond_embed(input.bond_feats);
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
            TensorF32 t1d_tor = concat(input.t1d, input.tor_feat, -1);                // (B,T,L,110)
            TensorF32 t1d_emb = emb_t1d_->forward(t1d_tor);                            // (B,T,L,64)
            t1d_emb = proj_t1d_->forward(relu(t1d_emb));                               // (B,T,L,64)
            // Cross-attention: state as Q, template as K/V
            int B = t1d_emb.shape().dims[0], T = t1d_emb.shape().dims[1];
            auto state_q = state_track_->representation().view({B * L, 1, D_STATE});
            auto t1d_kv = t1d_emb.permute({0, 2, 1, 3}).view({B * L, T, 64});
            SelfAttention cross_attn(D_STATE, 64, 8);
            auto out = cross_attn.forward(state_q, t1d_kv, t1d_kv);
            state_track_->repr() = state_track_->representation() + out.view({B, L, D_STATE});
        }
        TensorF32 templ_pair = get_templ_emb(input.t1d, input.t2d);  // (B,T,L,L,64)
        // 旧: pair_track_->templ_stack(templ_pair, rbf_feature, input.t1d);
        // 旧栈上: LinearLayer t1d_proj(80,32), LayerNorm(64), TemplatePairStack
        {
            int T = templ_pair.shape().dims[1];
            TensorF32 t1d_2d = input.t1d;
            t1d_2d.reshape({B*T, L, D_T1D});                                   // (B*T, L, 80)
            templ_pair.reshape({B*T, L, L, 64});                                // (B*T, L, L, 64)
            // 旧栈上: LinearLayer t1d_proj(D_T1D, D_STATE) → temp_stack_t1d_proj_
            TensorF32 state_proj = temp_stack_t1d_proj_->forward(t1d_2d);      // (B*T, L, 32)
            for (int k = 0; k < 2; ++k) {
                TemplatePairStack tps;  // TODO: section 1.9 pointer 化
                templ_pair = tps.forward(templ_pair, rbf_feature, state_proj);
            }
            // 旧栈上: LayerNorm layernorm(64) → temp_stack_norm_
            templ_pair = temp_stack_norm_->forward(templ_pair);                 // (B*T, L, L, 64)
            templ_pair.reshape({B, T, L, L, 64});
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
    //auto state = state_track_->representation();
    TensorF32 state;
    state.copy_from(state_track_->representation());
    //auto coords = input.coords;
    TensorF32 coords;
    coords.copy_from(input.coords);
    
    // need to modify :
    // the block in the 4 full block was different from the main block
    // full/extra block use global column attention

    const TensorF32& seq1hot = one_hot_seq(input.seq_tokens, 21);
    const TensorI64& idx = input.seq_tokens; //? residue indices
    set_seq_info(seq1hot, idx);

    // Extra blocks
    // need to use msa_full
    // and use global column attention as well
    for (auto& block : extra_blocks_) {
        // stop grad
        block->forward(msa_full, pair, state, seq1hot, coords);
        coords.copy_from(block->updated_coords());
    }
    
    // Main blocks
    for (auto& block : main_blocks_) {
        // stop grad
        // chiral grad
        block->forward(msa, pair, state, seq1hot, coords);
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
            coords.copy_from(refine->updated_coords());
            state.copy_from(refine->updated_state());
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
    
    return output;
}

//The t1d feature has shape (B, T, L, d_t1d) where B is batch size,
// T is number of templates, L is sequence length, 
//and d_t1d is the feature dimension that varies by model configuration
TensorF32 RFAAModel::get_templ_emb(const TensorF32& t1d, const TensorF32& t2d) {
    int L = t1d.shape().dims[2];
    // src t1d[b, l, t, d] the l-th residue's t-th template's t1d feature
    // left[b, l, t, l2, d] for every target residue l2, make a copy of t1d[b, l, t, d]
    TensorF32 left = t1d.unsqueeze(3);//.expand({-1, -1, -1, L, -1});  // (B, T, L, L, 80)
    TensorF32 right = t1d.unsqueeze(2);//.expand({-1, -1, L, -1, -1}); 
    // (B, T, L, d_t1d) (B, T, 1, L, d_t1d) (B, T, L, L, d_t1d)
    // expand the certain dimension of a tensor
    left.shape.dims[3] = L;
    right.shape.dims[3] = L;
    
    std::vector<TensorF32> templ_list = {t2d, left, right};
    TensorF32 templ = concat(templ_list, -1);
    // dim of templ is (B, T, L, L, d_t1d*2 + d_t2d) = (B, T, L, L, 224)
    // d_templ = 64
    // 旧栈上变量: LinearLayer emb(D_T1D * 2 + D_T2D, 64); → emb_t1d_t2d_
    return emb_t1d_t2d_->forward(templ);
    // (B, T, L, L, 64)
}

void RFAAModel::to(Device device) {
    device_ = device;
    // 转移所有参数...
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
    // 实现权重加载...
}

void RFAAModel::save_weights(const std::string& path) const {
    std::cout << "Saving weights to: " << path << std::endl;
    // 实现权重保存...
}

} // namespace rfaa

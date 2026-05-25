#include "rfaa/Model.h"
#include "rfaa/Embedding.h"
#include "rfaa/PositionalEncoding.h"
#include "rfaa/MathUtils.h"
#include <iostream>

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
    
    // 初始化子模块
    AttnConfig msa_attn_config(config.d_msa, config.n_heads);
    msa_row_attn_ = std::make_unique<MSARowAttention>(msa_attn_config);
    msa_col_attn_ = std::make_unique<MSAColAttention>(msa_attn_config);
    msa_ff_ = std::make_unique<FeedForward>(config.d_msa, config.d_msa * 4);
    
    tri_mul_out_ = std::make_unique<TriangleMultiplication>(
        config.d_pair, TriangleMultiplication::Direction::Outgoing);
    tri_mul_in_ = std::make_unique<TriangleMultiplication>(
        config.d_pair, TriangleMultiplication::Direction::Incoming);
    
    se3_ = std::make_unique<SE3Transformer>(config.se3_config);
    struct_update_ = std::make_unique<StructureUpdate>();
    
    // 初始化 PositionalEncoding
    pos_enc_ = std::make_unique<PositionalEncoding>(-32, 32, 8, config.d_pair);
}

void IterBlock::projStateAddToQueryRow(TensorF32& msa, const TensorF32& proj_state) {
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

TensorF32 IterBlock::computeRBFFeature(const TensorF32& coords)
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
                        TensorF32& state, const TensorF32& coords) {
    
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
            LayerNorm state_norm(D_STATE);
            TensorF32 state_normed = state_norm.forward(state);
            
            LinearLayer linear(D_STATE, D_MSA);
            const auto proj_state = linear.forward(state_normed);
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
            LayerNorm pair_layernorm(D_PAIR);
            pair_biased = pair_layernorm.forward(pair);

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
            LayerNorm msa_norm(D_MSA);
            TensorF32 msa_normed = msa_norm.forward(msa);
            LinearLayer left_proj(D_MSA, 16);
            LinearLayer right_proj(D_MSA, 16);
            TensorF32 left = left_proj.forward(msa_normed);   // (B,N,L,16)
            TensorF32 right = right_proj.forward(msa_normed); // (B,N,L,16)
            TensorF32 right_mean = right / float(N);
            // a tensor divide a scalar
            TensorF32 pair_update = outer_product(left, right_mean);  // (B,L,L,256)
            // dim of pair_update?
            // reshape to (B, L, L, 16*16) = (B, L, L, 256)?
            LinearLayer out_proj(16 * 16, D_PAIR);
            pair_update = out_proj.forward(pair_update);  // (B,L,L,128)
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
            LinearLayer rbf_proj(D_RBF, D_PAIR);
            rbf_feature = rbf_proj.forward(rbf_feature);  // (B,L,L,128)
            LayerNorm state_norm(D_STATE);
            TensorF32 state_normed = state_norm.forward(state);
            LinearLayer left_proj(D_STATE, 16);
            LinearLayer right_proj(D_STATE, 16);
            // different weights for left and right?
            TensorF32 left = left_proj.forward(state_normed);   // (B,L,16)
            TensorF32 right = right_proj.forward(state_normed); // (B,L,16)
            TensorF32 gate = outer_product(left, right);  // (B,L,L,256)
            LinearLayer gate_proj(16 * 16, D_PAIR);
            // d_hidden_gate = 16
            gate = gate_proj.forward(gate);  // (B,L,L,128)
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
        auto msa_query = msa.select(1, 0);  // (B, L, 256)
        
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
        auto alpha = struct_update_->predict_torsion(msa_query, state);
    }
}

void FullBlock::forward(TensorF32& msa_full, TensorF32& pair, TensorF32& state, 
                        const TensorF32& coords) {
    // FullBlock 在 IterBlock 的基础上增加了 msa_full 的使用和全局 column attention
    // msa_full 需要在 forward 函数参数中传入，或者在 IterBlock 中存储为成员变量
    
    // ------------ 1D track update ------------
    // ===== Step 1: msa2msa =====
    {
        //auto query_row = msa.select(1, 0);  // (B, L, 256)
        // query_row += Linear(state) ...

        LayerNorm state_norm(D_STATE);
        TensorF32 state_normed = state_norm.forward(state);
            
        LinearLayer linear(D_STATE, D_MSA);
        const auto proj_state = linear.forward(state_normed);
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
        LayerNorm pair_layernorm(D_PAIR);
        pair_biased = pair_layernorm.forward(pair);

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
        LayerNorm msa_norm(D_MSA);
        TensorF32 msa_normed = msa_norm.forward(msa);
        LinearLayer left_proj(D_MSA, 16);
        LinearLayer right_proj(D_MSA, 16);
        TensorF32 left = left_proj.forward(msa_normed);   // (B,N,L,16)
        TensorF32 right = right_proj.forward(msa_normed); // (B,N,L,16)
        TensorF32 right_mean = right / float(N);
        // a tensor divide a scalar
        TensorF32 pair_update = outer_product(left, right_mean);  // (B,L,L,256)
        // dim of pair_update?
        // reshape to (B, L, L, 16*16) = (B, L, L, 256)?
        LinearLayer out_proj(16 * 16, D_PAIR);
        pair_update = out_proj.forward(pair_update);  // (B,L,L,128)
        pair = pair + pair_update;  // residual
    }

    // Triangle Multiplication
    Dropout drop_row(1, 0.15);
    pair = pair + drop_row.forward(tri_mul_out_->forward(pair, true));
    pair = pair + drop_row.forward(tri_mul_in_->forward(pair, false));

    // ===== Step 3: pair2pair =====
    // state outer product -> gate
    {
        LinearLayer rbf_proj(D_RBF, D_PAIR);
        rbf_feature = rbf_proj.forward(rbf_feature);  // (B,L,L,128)
        LayerNorm state_norm(D_STATE);
        TensorF32 state_normed = state_norm.forward(state);
        LinearLayer left_proj(D_STATE, 16);
        LinearLayer right_proj(D_STATE, 16);
        // different weights for left and right?
        TensorF32 left = left_proj.forward(state_normed);   // (B,L,16)
        TensorF32 right = right_proj.forward(state_normed); // (B,L,16)
        TensorF32 gate = outer_product(left, right);  // (B,L,L,256)
        LinearLayer gate_proj(16 * 16, D_PAIR);
        // d_hidden_gate = 16
        gate = gate_proj.forward(gate);  // (B,L,L,128)
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

    // 3D track update
    // ===== Step 4: str2str (SE3 Transformer) =====

 }

// RFAAModel 实现
RFAAModel::RFAAModel(const RFAAConfig& config) : config_(config) {
    // 创建迭代块
    for (int i = 0; i < config.n_extra_blocks; ++i) {
        extra_blocks_.push_back(std::make_unique<FullBlock>(config, true));
    }
    for (int i = 0; i < config.n_main_blocks; ++i) {
        main_blocks_.push_back(std::make_unique<IterBlock>(config, true));
    }
    for (int i = 0; i < config.n_refine_blocks; ++i) {
        refine_blocks_.push_back(std::make_unique<IterBlock>(config, false));
    }
}

RFAAModel::~RFAAModel() = default;

ModelOutput RFAAModel::forward(const ModelInput& input) {
    ModelOutput output;
    
    int B = input.msa_latent.shape().dims[0];
    int N = input.msa_latent.shape().dims[1];
    int L = input.msa_latent.shape().dims[2];
    
    // 初始化 tracks
    msa_track_ = std::make_unique<MSATrack>(N, L, config_.d_msa, device_);
    pair_track_ = std::make_unique<PairTrack>(L, config_.d_pair, device_);
    state_track_ = std::make_unique<StateTrack>(L, config_.d_state, device_);
    
    // the msa cluster embed was contained in the three track init functions below
    // Embedding
    msa_track_->init_from_features(input.msa_latent);
    state_track_->init_from_embedding(input.seq_tokens);

    // pair track need to be initialized as well
    // pair track 从 seq_tokens 初始化 (left, right)
    // (B, L)
    pair_track_->init_from_embedding(input.seq_tokens, input.seq_tokens, input.bond_feats, input.dist_matrix);

    // msa full embed?
    // msa_full = self.full_emb(msa_full, seq, idx)
    // msa_full was used in the full block
    TensorF32 msa_full;
    if (input.msa_full.numel() > 0) {
        FullEmbedding full_emb(NAATOKENS - 1 + 4, D_MSA_FULL);
        msa_full = full_emb.forward(input.msa_full, input.seq_tokens, TensorF32());
    }

    // bond embed for pair track
    // need to get the bond feats
    BondEmbedding bond_embed(0, D_PAIR);
    TensorF32 pair;
    pair.copy_from(pair_track_->representation());
    pair = pair + bond_embed(input.bond_feats);
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
        state_track_->inject_template(input.t1d, input.tor_feat);
        //(B, T, L, L, 64)
        TensorF32 templ_pair = get_templ_emb(input.t1d, input.t2d);
        // template pair stack
        pair_track_->templ_stack(templ_pair, rbf_feature, input.t1d);
        pair_track_->inject_template(templ_pair);
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

    // Extra blocks
    // need to use msa_full
    // and use global column attention as well
    for (auto& block : extra_blocks_) {
        // stop grad
        block->forward(msa_full, pair, state, coords);
    }
    
    // Main blocks
    for (auto& block : main_blocks_) {
        // stop grad
        // chiral grad
        block->forward(msa, pair, state, coords);
    }
    
    // Refinement blocks (仅更新结构)
    for (auto& block : refine_blocks_) {
        // stop grad
        // chiral grad
        // clash grad
        block->forward(msa, pair, state, coords);
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
    LinearLayer emb(D_T1D * 2 + D_T2D, 64);
    return emb.forward(templ);
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

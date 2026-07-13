#pragma once

#include "Track.h"
#include "Attention.h"
#include "SE3Transformer.h"
#include "PositionalEncoding.h"
#include <string>

#include "rfaa/Embedding.h"

namespace rfaa {

// RFAA 模型配置
struct RFAAConfig {
    // 维度
    int d_msa = D_MSA;
    int d_pair = D_PAIR;
    int d_state = D_STATE;
    int d_t1d = D_T1D;
    
    // 结构
    int n_extra_blocks = N_EXTRA_BLOCKS;
    int n_main_blocks = N_MAIN_BLOCKS;
    int n_refine_blocks = N_REFINE_BLOCKS;
    int n_heads = N_HEAD;
    
    // SE3
    SE3Config se3_config;
    
    // 其他
    int max_seq_len = 2048;
    int max_n_seq = 512;
    int max_n_templ = 4;
    
    RFAAConfig();
};

// 模型输入
struct ModelInput {
    TensorF32 msa_latent;    // (B, N_clust, L, 164)
    TensorF32 msa_full;      // (B, N_extra, L, 83) - optional
    TensorF32 seq_tokens;    // (B, L) - 查询序列 token
    TensorF32 t1d;           // (B, T, L, 80) - 模板特征
    TensorF32 t2d;           // (B, T, L, L, ...) - 模板 2D 特征
    TensorF32 coords;        // (B, L, 3, 3) - 初始 Ca 坐标 (可选)
    TensorF32 tor_feat;
    TensorF32 bond_feats;    // (B, L, L, d_bond) - 键特征
    TensorF32 dist_matrix;   // (B, L, L) - 距离矩阵
};

// 模型输出
struct ModelOutput {
    TensorF32 msa;           // (B, N, L, D_MSA)
    TensorF32 pair;          // (B, L, L, D_PAIR)
    TensorF32 state;         // (B, L, D_STATE)
    TensorF32 coords;        // (B, L, 3, 3) - 更新后的坐标
    TensorF32 alpha;         // (B, L, NTOTALDOFS, 2) - 侧链扭转角
    
    // 辅助输出
    TensorF32 lddt;          // (B, L) - 每残基置信度
    TensorF32 distogram;     // (B, L, L, n_bins) - 距离分布
    TensorF32 pae;           // (B, L, L) - 预测对齐误差
};

// 迭代块 (IterBlock)
class IterBlock {
public:
    IterBlock(const RFAAConfig& config, bool update_msa_pair = true);
    
    // 由 RFAAModel 注入子模块 (non-owning pointers 已在之前注入, 这里是 unique_ptr 所有权转移)
    void set_sub_modules(
        std::unique_ptr<MSARowAttention>          msa_row,
        std::unique_ptr<MSAColAttention>          msa_col,
        std::unique_ptr<FeedForward>              msa_ff,
        std::unique_ptr<PairRowAttention>         pair_row,
        std::unique_ptr<PairColAttention>         pair_col,
        std::unique_ptr<FeedForward>              pair_ff,
        std::unique_ptr<TriangleMultiplication>   tri_out,
        std::unique_ptr<TriangleMultiplication>   tri_in,
        std::unique_ptr<SE3Transformer>           se3
    ) {
        msa_row_attn_  = std::move(msa_row);
        msa_col_attn_  = std::move(msa_col);
        msa_ff_        = std::move(msa_ff);
        pair_row_attn_ = std::move(pair_row);
        pair_col_attn_ = std::move(pair_col);
        pair_ff_       = std::move(pair_ff);
        tri_mul_out_   = std::move(tri_out);
        tri_mul_in_    = std::move(tri_in);
        se3_           = std::move(se3);
    }

    void set_pos_enc(std::unique_ptr<PositionalEncoding> p) { pos_enc_ = std::move(p); }
    
    // 执行一个迭代块
    // 输入/输出通过引用修改
    void forward(TensorF32& msa, TensorF32& pair, TensorF32& state, 
                 const TensorF32& seq1hot,
                 const TensorF32& coords);

    void proj_state_add_to_query_row(TensorF32& msa, const TensorF32& proj_state);

    TensorF32 compute_rbf_feature(const TensorF32& coords);
    
private:
    RFAAConfig config_;
    bool update_msa_pair_;
    
    // 子模块
    std::unique_ptr<MSARowAttention> msa_row_attn_;
    std::unique_ptr<MSAColAttention> msa_col_attn_;
    std::unique_ptr<PairRowAttention> pair_row_attn_;
    std::unique_ptr<PairColAttention> pair_col_attn_;
    std::unique_ptr<FeedForward> msa_ff_;
    std::unique_ptr<FeedForward> pair_ff_;
    std::unique_ptr<TriangleMultiplication> tri_mul_out_;
    std::unique_ptr<TriangleMultiplication> tri_mul_in_;
    std::unique_ptr<SE3Transformer> se3_;
    //std::unique_ptr<StructureUpdate> struct_update_;
    std::unique_ptr<PositionalEncoding> pos_enc_;

    // 3D track
    TensorF32 xyz_new_;              // (B, L, 3, 3)
    TensorF32 state_new_;            // (B, L, D_STATE) 可选，或直接用 state 引用

    // ---- 3D SE 参数 (non-owning pointer, 由 RFAAModel 创建) ----
    // 旧值类型 (保留注释):
    // LayerNorm norm_msa_3d_(D_MSA);
    // LayerNorm norm_pair_3d_(D_PAIR);
    // static constexpr int NODE_3D_IN  = D_MSA + 21;      // 256+21 = 277  → Core.h ITER_NODE_3D_IN
    // static constexpr int NODE_3D_OUT = N_L0_IN_FEATS;   // 32            → Core.h ITER_NODE_3D_OUT
    // static constexpr int EDGE_3D_OUT = N_EDGE_FEATS;    // 32            → Core.h ITER_EDGE_3D_OUT
    // LinearLayer embed_x_(NODE_3D_IN, NODE_3D_OUT);  
    // LinearLayer embed_e_(D_PAIR, EDGE_3D_OUT);  
    // LayerNorm  norm_node_3d_(NODE_3D_OUT);
    // LayerNorm  norm_edge_3d_(EDGE_3D_OUT);

    LayerNorm*  norm_msa_3d_  = nullptr; // D_MSA (256)
    LayerNorm*  norm_pair_3d_ = nullptr; // D_PAIR (128)
    LinearLayer* embed_x_     = nullptr; // ITER_NODE_3D_IN (277) → ITER_NODE_3D_OUT (32)
    LinearLayer* embed_e_     = nullptr; // D_PAIR (128) → ITER_EDGE_3D_OUT (32)
    LayerNorm*  norm_node_3d_ = nullptr; // ITER_NODE_3D_OUT (32)
    LayerNorm*  norm_edge_3d_ = nullptr; // ITER_EDGE_3D_OUT (32)

    // ---- forward 内部参数 (non-owning pointer, 由 RFAAModel 创建) ----
    // 旧值类型 (保留注释):
    // LayerNorm state_norm(D_STATE);     LinearLayer linear(D_STATE, D_MSA);
    // LayerNorm pair_layernorm(D_PAIR);
    // LayerNorm msa_norm(D_MSA);         LinearLayer left_proj(D_MSA, 16); LinearLayer right_proj(D_MSA, 16); LinearLayer out_proj(256, D_PAIR);
    // LinearLayer rbf_proj(D_RBF, D_PAIR); LayerNorm state_norm(D_STATE); LinearLayer left_proj(D_STATE, 16); LinearLayer right_proj(D_STATE, 16); LinearLayer gate_proj(256, D_PAIR);

    LayerNorm*   state2msa_norm_        = nullptr; // D_STATE (32)
    LinearLayer* state2msa_linear_      = nullptr; // D_STATE (32) → D_MSA (256)
    LayerNorm*   pair2msa_norm_         = nullptr; // D_PAIR (128)
    LayerNorm*   msa2pair_norm_         = nullptr; // D_MSA (256)
    LinearLayer* msa2pair_left_proj_    = nullptr; // D_MSA (256) → 16
    LinearLayer* msa2pair_right_proj_   = nullptr; // D_MSA (256) → 16
    LinearLayer* msa2pair_out_proj_     = nullptr; // 16*16 (256) → D_PAIR (128)
    LinearLayer* pair2pair_rbf_proj_    = nullptr; // D_RBF (64) → D_PAIR (128)
    LayerNorm*   pair2pair_state_norm_  = nullptr; // D_STATE (32)
    LinearLayer* pair2pair_left_proj_   = nullptr; // D_STATE (32) → 16
    LinearLayer* pair2pair_right_proj_  = nullptr; // D_STATE (32) → 16
    LinearLayer* pair2pair_gate_proj_   = nullptr; // 16*16 (256) → D_PAIR (128)

    // 额外输入 (由 caller 在 forward 前设置)
    //TensorF32 seq1hot_;              // (B, L, 21)
    //TensorI64 idx_;                  // (B, L)
    //bool has_seq_info_ = false;

public:
    const TensorF32& updated_coords() const { return xyz_new_; }
    //void set_seq_info(const TensorF32& seq1hot, const TensorI64& idx);
};

class FullBlock : public IterBlock {
public:
    explicit FullBlock(const RFAAConfig& config) : IterBlock(config, true) {}

    // a virtual dtor
    virtual ~FullBlock() = default;

    void set_global_col_attn(std::unique_ptr<MSAGlobalColAttention> p) {
        msa_global_col_attn_ = std::move(p);
    }

    void forward(TensorF32& msa_full, TensorF32& pair, TensorF32& state, 
                 const TensorF32& seq1hot,
                 const TensorF32& coords) override;
private:
    std::unique_ptr<MSAGlobalColAttention> msa_global_col_attn_;
};

class RefineBlock : public IterBlock {
public:
    explicit RefineBlock(const RFAAConfig& config) : RefineBlock(config, true) {}

    // a virtual dtor
    virtual ~RefineBlock() = default;

    void forward(TensorF32& msa, TensorF32& pair, TensorF32& state, 
                 const TensorF32& seq1hot,
                 const TensorF32& coords) override;

                 // 设置额外输入（在 forward 调用前设置）
    //void set_seq_info(const TensorF32& seq1hot, const TensorI64& idx);

    // 获取 SE3 更新后的坐标
    const TensorF32& updated_coords() const { return xyz_new_; }
    const TensorF32& updated_state()  const { return state_new_; }

private:
    // ---- LayerNorm 模块 (non-owning pointer) ----
    // 旧值类型 (保留注释):
    // LayerNorm norm_msa_(D_MSA);
    // LayerNorm norm_pair_(D_PAIR);
    // LayerNorm norm_state_(D_STATE);
    LayerNorm* norm_msa_   = nullptr; // D_MSA (256)
    LayerNorm* norm_pair_  = nullptr; // D_PAIR (128)
    LayerNorm* norm_state_ = nullptr; // D_STATE (32)

    // ---- Node 嵌入 (309 → 32) ----
    // 旧值类型 (保留注释):
    // static constexpr int NODE_IN_DIM  = D_MSA + 21 + D_STATE;  // → Core.h REFINE_NODE_IN_DIM
    // static constexpr int NODE_OUT_DIM = N_L0_IN_FEATS;          // → Core.h REFINE_NODE_OUT_DIM
    // LinearLayer embed_x_(NODE_IN_DIM, NODE_OUT_DIM);
    // LayerNorm  norm_node_(NODE_OUT_DIM);
    LinearLayer* embed_x_   = nullptr; // REFINE_NODE_IN_DIM (309) → REFINE_NODE_OUT_DIM (32)
    LayerNorm*   norm_node_ = nullptr; // REFINE_NODE_OUT_DIM (32)

    // ---- Edge 嵌入 第一阶段 (128 → 32) ----
    // 旧值类型 (保留注释):
    // LinearLayer embed_e1_(D_PAIR, N_EDGE_FEATS);
    // LayerNorm  norm_edge1_(N_EDGE_FEATS);
    LinearLayer* embed_e1_   = nullptr; // D_PAIR (128) → N_EDGE_FEATS (32)
    LayerNorm*   norm_edge1_ = nullptr; // N_EDGE_FEATS (32)

    // ---- Edge 嵌入 第二阶段 (32+64+1=97 → 32) ----
    // 旧值类型 (保留注释):
    // static constexpr int EDGE_IN_DIM2 = N_EDGE_FEATS + 64 + 1;  // → Core.h REFINE_EDGE_IN_DIM2
    // LinearLayer embed_e2_(EDGE_IN_DIM2, N_EDGE_FEATS);
    // LayerNorm  norm_edge2_(N_EDGE_FEATS);
    LinearLayer* embed_e2_   = nullptr; // REFINE_EDGE_IN_DIM2 (97) → N_EDGE_FEATS (32)
    LayerNorm*   norm_edge2_ = nullptr; // N_EDGE_FEATS (32)

    // ---- 额外输入（由外部设置）----
    //TensorF32 seq1hot_;       // (B, L, 21) 序列 one-hot
    //TensorI64 idx_;           // (B, L) 残基索引
    //bool      has_seq_info_ = false;

    // ---- 输出缓存 ----
    TensorF32 xyz_new_;       // (B, L, 3, 3) 更新后的坐标
    TensorF32 state_new_;     // (B, L, D_STATE) 更新后的 state
};

// RFAA 主模型
class RFAAModel {
public:
    explicit RFAAModel(const RFAAConfig& config = RFAAConfig{});
    ~RFAAModel();

    void set_seq_info(const TensorF32& seq1hot, const TensorI64& idx);
    
    // 前向传播
    ModelOutput forward(const ModelInput& input);

    TensorF32 get_templ_emb(const TensorF32& t1d, const TensorF32& t2d);
    
    // 加载/保存权重
    void load_weights(const std::string& path);
    void save_weights(const std::string& path) const;
    
    // 设备管理
    void to(Device device);
    Device device() const;
    
    // 训练/推理模式
    void train();
    void eval();
    bool is_training() const;
    
private:
    RFAAConfig config_;
    Device device_ = Device::CPU;
    bool training_ = false;

    // ===== embedding / template 参数 (全局单份) =====
    LinearLayer*     msa_emb_            = nullptr; // (164 → 256)
    EmbeddingLayer*  state_emb_          = nullptr; // (NAATOKENS, D_STATE)
    EmbeddingLayer*  pair_left_emb_      = nullptr; // (NAATOKENS, D_PAIR)
    EmbeddingLayer*  pair_right_emb_     = nullptr; // (NAATOKENS, D_PAIR)
    LinearLayer*     full_linear_        = nullptr; // (83 → D_MSA_FULL)
    EmbeddingLayer*  full_emb_           = nullptr; // (NAATOKENS, D_MSA_FULL)
    LinearLayer*     bond_emb_           = nullptr; // (8 → D_PAIR)
    LinearLayer*     emb_t1d_            = nullptr; // (110 → 64)
    LinearLayer*     proj_t1d_           = nullptr; // (64 → 64)
    LinearLayer*     emb_t1d_t2d_        = nullptr; // (224 → 64) get_templ_emb
    LinearLayer*     temp_stack_t1d_proj_ = nullptr; // (80 → 32) templ_stack
    LayerNorm*       temp_stack_norm_    = nullptr; // (64)

    // ===== attention / sub-module 参数 (per-block, vector) =====
    static constexpr int N_ITER = 12;   // extra(4) + main(8)
    static constexpr int N_GLOB = 4;    // extra_blocks (FullBlock)

    // --- MSARowAttention (6 LL ×12) ---
    // 旧: LinearLayer* msa_row_Wq_ etc.
    std::vector<LinearLayer*> msa_row_Wq_;     // D_MSA (256) → N_HEAD*D_MSA (2048)
    std::vector<LinearLayer*> msa_row_Wk_;     // D_MSA (256) → N_HEAD*D_MSA (2048)
    std::vector<LinearLayer*> msa_row_Wv_;     // D_MSA (256) → N_HEAD*D_MSA (2048)
    std::vector<LinearLayer*> msa_row_to_b_;   // D_PAIR (128) → N_HEAD (8)
    std::vector<LinearLayer*> msa_row_to_g_;   // D_MSA (256) → N_HEAD*D_MSA (2048)
    std::vector<LinearLayer*> msa_row_to_out_; // N_HEAD*D_MSA (2048) → D_MSA (256)

    // --- MSAColAttention (6 LL ×12) ---
    std::vector<LinearLayer*> msa_col_Wq_;     // D_MSA (256) → N_HEAD*D_MSA (2048)
    std::vector<LinearLayer*> msa_col_Wk_;     
    std::vector<LinearLayer*> msa_col_Wv_;     
    std::vector<LinearLayer*> msa_col_to_b_;   // D_PAIR (128) → N_HEAD (8)
    std::vector<LinearLayer*> msa_col_to_g_;   
    std::vector<LinearLayer*> msa_col_to_out_; 

    // --- MSAGlobalColAttention (6 LL ×4, FullBlock only) ---
    std::vector<LinearLayer*> msa_global_col_Wq_;    // D_MSA (256) → D_MSA (256)  single-head
    std::vector<LinearLayer*> msa_global_col_Wk_;    
    std::vector<LinearLayer*> msa_global_col_Wv_;    
    std::vector<LinearLayer*> msa_global_col_to_b_;  // D_PAIR (128) → N_HEAD (8)
    std::vector<LinearLayer*> msa_global_col_to_g_;  
    std::vector<LinearLayer*> msa_global_col_to_out_;

    // --- PairRowAttention (6 LL ×12) ---
    std::vector<LinearLayer*> pair_row_Wq_;     // D_PAIR (128) → N_HEAD*D_PAIR_HIDDEN (256)
    std::vector<LinearLayer*> pair_row_Wk_;     
    std::vector<LinearLayer*> pair_row_Wv_;     
    std::vector<LinearLayer*> pair_row_to_b_;   // D_PAIR (128) → N_HEAD (8)
    std::vector<LinearLayer*> pair_row_to_g_;   
    std::vector<LinearLayer*> pair_row_to_out_; 

    // --- PairColAttention (6 LL ×12) ---
    std::vector<LinearLayer*> pair_col_Wq_;     
    std::vector<LinearLayer*> pair_col_Wk_;     
    std::vector<LinearLayer*> pair_col_Wv_;     
    std::vector<LinearLayer*> pair_col_to_b_;   
    std::vector<LinearLayer*> pair_col_to_g_;   
    std::vector<LinearLayer*> pair_col_to_out_; 

    // --- FeedForward msa_ff_ (1 LN + 2 LL ×12) ---
    std::vector<LayerNorm*>   msa_ff_norm_;     // D_MSA (256)
    std::vector<LinearLayer*> msa_ff_linear1_;  // D_MSA (256) → D_MSA*4 (1024)
    std::vector<LinearLayer*> msa_ff_linear2_;  // D_MSA*4 (1024) → D_MSA (256)

    // --- FeedForward pair_ff_ (1 LN + 2 LL ×12) ---
    std::vector<LayerNorm*>   pair_ff_norm_;     // D_PAIR (128)
    std::vector<LinearLayer*> pair_ff_linear1_;  // D_PAIR (128) → D_PAIR*2 (256)
    std::vector<LinearLayer*> pair_ff_linear2_;  // D_PAIR*2 (256) → D_PAIR (128)

    // --- TriangleMultiplication out (2 LN + 6 LL ×12) ---
    std::vector<LayerNorm*>   tri_out_layernorm_;         // D_PAIR (128)
    std::vector<LinearLayer*> tri_out_left_proj_;         // D_PAIR (128) → D_HIDDEN_TRIMUL (128)
    std::vector<LinearLayer*> tri_out_right_proj_;        
    std::vector<LinearLayer*> tri_out_left_gate_;         
    std::vector<LinearLayer*> tri_out_right_gate_;        
    std::vector<LinearLayer*> tri_out_gate_;              // D_PAIR (128) → D_PAIR (128)
    std::vector<LayerNorm*>   tri_out_output_layernorm_;  // D_HIDDEN_TRIMUL (128)
    std::vector<LinearLayer*> tri_out_out_proj_;          // D_HIDDEN_TRIMUL (128) → D_PAIR (128)

    // --- TriangleMultiplication in (2 LN + 6 LL ×12) ---
    std::vector<LayerNorm*>   tri_in_layernorm_;
    std::vector<LinearLayer*> tri_in_left_proj_;
    std::vector<LinearLayer*> tri_in_right_proj_;
    std::vector<LinearLayer*> tri_in_left_gate_;
    std::vector<LinearLayer*> tri_in_right_gate_;
    std::vector<LinearLayer*> tri_in_gate_;
    std::vector<LayerNorm*>   tri_in_output_layernorm_;
    std::vector<LinearLayer*> tri_in_out_proj_;

    // --- PositionalEncoding (2 EmbeddingLayer ×12, per block) ---
    std::vector<EmbeddingLayer*> pos_enc_emb_res_;   // [0..11] (65, D_PAIR=128)  residue dist embedding
    std::vector<EmbeddingLayer*> pos_enc_emb_atom_;  // [0..11] (17, D_PAIR=128)  atom bond dist embedding

    // ===== 3D SE 参数 (per-block, 由 RFAAModel::create 统一创建后将指针注入 block) =====

    // --- IterBlock 3D SE (每组 6 个, ITER_N_BLOCKS=12 组) ---
    // 旧值类型 (保留注释):
    // LinearLayer embed_x_;  // ITER_NODE_3D_IN → ITER_NODE_3D_OUT
    // LinearLayer embed_e_;  // D_PAIR → ITER_EDGE_3D_OUT
    // LayerNorm  norm_node_3d_; LayerNorm norm_edge_3d_;
    // LayerNorm norm_msa_3d_; LayerNorm norm_pair_3d_;
    std::vector<LinearLayer*> iter_embed_x_;       // [0..11]  ITER_NODE_3D_IN (277) → ITER_NODE_3D_OUT (32)
    std::vector<LinearLayer*> iter_embed_e_;       // [0..11]  D_PAIR (128) → ITER_EDGE_3D_OUT (32)
    std::vector<LayerNorm*>   iter_norm_node_3d_;  // [0..11]  ITER_NODE_3D_OUT (32)
    std::vector<LayerNorm*>   iter_norm_edge_3d_;  // [0..11]  ITER_EDGE_3D_OUT (32)
    std::vector<LayerNorm*>   iter_norm_msa_3d_;   // [0..11]  D_MSA (256)
    std::vector<LayerNorm*>   iter_norm_pair_3d_;  // [0..11]  D_PAIR (128)

    // --- IterBlock forward 内部参数 (每组 12 个, 12 组) ---
    std::vector<LayerNorm*>   iter_state2msa_norm_;       // [0..11]  D_STATE (32)
    std::vector<LinearLayer*> iter_state2msa_linear_;     // [0..11]  D_STATE (32) → D_MSA (256)
    std::vector<LayerNorm*>   iter_pair2msa_norm_;        // [0..11]  D_PAIR (128)
    std::vector<LayerNorm*>   iter_msa2pair_norm_;        // [0..11]  D_MSA (256)
    std::vector<LinearLayer*> iter_msa2pair_left_proj_;   // [0..11]  D_MSA (256) → 16
    std::vector<LinearLayer*> iter_msa2pair_right_proj_;  // [0..11]  D_MSA (256) → 16
    std::vector<LinearLayer*> iter_msa2pair_out_proj_;    // [0..11]  256 → D_PAIR (128)
    std::vector<LinearLayer*> iter_pair2pair_rbf_proj_;   // [0..11]  D_RBF (64) → D_PAIR (128)
    std::vector<LayerNorm*>   iter_pair2pair_state_norm_; // [0..11]  D_STATE (32)
    std::vector<LinearLayer*> iter_pair2pair_left_proj_;  // [0..11]  D_STATE (32) → 16
    std::vector<LinearLayer*> iter_pair2pair_right_proj_; // [0..11]  D_STATE (32) → 16
    std::vector<LinearLayer*> iter_pair2pair_gate_proj_;  // [0..11]  256 → D_PAIR (128)

    // --- RefineBlock 3D SE (每组 10 个, N_REFINE_BLOCKS=4 组) ---
    // 旧值类型 (保留注释):
    // LayerNorm norm_msa_(D_MSA); LayerNorm norm_pair_(D_PAIR); LayerNorm norm_state_(D_STATE);
    // LinearLayer embed_x_(REFINE_NODE_IN_DIM, REFINE_NODE_OUT_DIM); LayerNorm norm_node_(REFINE_NODE_OUT_DIM);
    // LinearLayer embed_e1_(D_PAIR, N_EDGE_FEATS); LayerNorm norm_edge1_(N_EDGE_FEATS);
    // LinearLayer embed_e2_(REFINE_EDGE_IN_DIM2, N_EDGE_FEATS); LayerNorm norm_edge2_(N_EDGE_FEATS);
    std::vector<LayerNorm*>   refine_norm_msa_;     // [0..3]  D_MSA (256)
    std::vector<LayerNorm*>   refine_norm_pair_;    // [0..3]  D_PAIR (128)
    std::vector<LayerNorm*>   refine_norm_state_;   // [0..3]  D_STATE (32)
    std::vector<LinearLayer*> refine_embed_x_;      // [0..3]  REFINE_NODE_IN_DIM (309) → REFINE_NODE_OUT_DIM (32)
    std::vector<LayerNorm*>   refine_norm_node_;    // [0..3]  REFINE_NODE_OUT_DIM (32)
    std::vector<LinearLayer*> refine_embed_e1_;     // [0..3]  D_PAIR (128) → N_EDGE_FEATS (32)
    std::vector<LayerNorm*>   refine_norm_edge1_;   // [0..3]  N_EDGE_FEATS (32)
    std::vector<LinearLayer*> refine_embed_e2_;     // [0..3]  REFINE_EDGE_IN_DIM2 (97) → N_EDGE_FEATS (32)
    std::vector<LayerNorm*>   refine_norm_edge2_;   // [0..3]  N_EDGE_FEATS (32)


    // Tracks
    std::unique_ptr<MSATrack> msa_track_;
    std::unique_ptr<PairTrack> pair_track_;
    std::unique_ptr<StateTrack> state_track_;

    TensorF32 seq1hot_;       // (B, L, 21) 序列 one-hot
    TensorI64 idx_;           // (B, L) 残基索引
    bool      has_seq_info_ = false;
    
    // 迭代块
    std::vector<std::unique_ptr<IterBlock>> extra_blocks_;
    std::vector<std::unique_ptr<IterBlock>> main_blocks_;
    std::vector<std::unique_ptr<IterBlock>> refine_blocks_;
    
    // 输出头
    //struct OutputHeads;
    //std::unique_ptr<OutputHeads> heads_;
};

} // namespace rfaa

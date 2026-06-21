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
    
    // 执行一个迭代块
    // 输入/输出通过引用修改
    void forward(TensorF32& msa, TensorF32& pair, TensorF32& state, 
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
    std::unique_ptr<TriangleMultiplication> tri_mul_out_;
    std::unique_ptr<TriangleMultiplication> tri_mul_in_;
    std::unique_ptr<SE3Transformer> se3_;
    std::unique_ptr<StructureUpdate> struct_update_;
    std::unique_ptr<PositionalEncoding> pos_enc_;
};

class FullBlock : public IterBlock {
public:
    explicit FullBlock(const RFAAConfig& config) : IterBlock(config, true) {}

    // a virtual dtor
    virtual ~FullBlock() = default;

    void forward(TensorF32& msa, TensorF32& pair, TensorF32& state, 
                 const TensorF32& coords) override;
private:
    std::unique_ptr<MSAGlobalColAttention> msa_global_col_attn_;
};

// RFAA 主模型
class RFAAModel {
public:
    explicit RFAAModel(const RFAAConfig& config = RFAAConfig{});
    ~RFAAModel();
    
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

    // ===== 所有权重（从 context 分配，FLAG_PARAM）=====
    // embedding
    LinearLayer* msa_emb_           = nullptr;
    EmbeddingLayer* state_emb_      = nullptr;
    EmbeddingLayer* pair_left_emb_  = nullptr;
    EmbeddingLayer* pair_right_emb_ = nullptr;
    LinearLayer* full_linear_       = nullptr;
    EmbeddingLayer* full_emb_       = nullptr;
    LinearLayer* bond_emb_          = nullptr;
    LinearLayer* emb_t1d_           = nullptr;
    LinearLayer* proj_t1d_          = nullptr;
    LinearLayer* emb_t1d_t2d_       = nullptr; //get_templ_emb
    LinearLayer* temp_stack_t1d_proj_ = nullptr; //templ_stack
    LayerNorm* temp_stack_norm_     = nullptr;
    //msa2msa
    LayerNorm* msa2msa_norm_        = nullptr;
    LinearLayer* msa2msa_linear_    = nullptr;
    LayerNorm* pair2msa_norm_       = nullptr;
    //msa2pair
    LayerNorm* msa_norm_            = nullptr;
    LinearLayer* left_proj_         = nullptr;
    LinearLayer* right_proj_        = nullptr;
    LinearLayer* out_proj_          = nullptr;
    //pair2pair
    LinearLayer* rbf_proj_         = nullptr;
    LayerNorm* state_norm_         = nullptr;
    LinearLayer* left_proj_        = nullptr;
    LinearLayer* right_proj_       = nullptr;
    LinearLayer* gate_proj_        = nullptr;
    // sub attention block
    //msa_ff_
    LayerNorm* msa_ff_norm_        = nullptr;
    LinearLayer* msa_linear_1_     = nullptr;
    LinearLayer* msa_linear_2_     = nullptr;
    //msa_row_attn_;
    LayerNorm* msa_row_layernorm_   = nullptr;
    LayerNorm* pair_row_layernorm_  = nullptr;
    LinearLayer* msa_row_to_b_      = nullptr;
    LinearLayer* msa_row_to_g_      = nullptr;
    LinearLayer* msa_row_to_out_    = nullptr;
    LinearLayer* msa_row_Wq_        = nullptr;
    LinearLayer* msa_row_Wk_        = nullptr;
    LinearLayer* msa_row_Wv_        = nullptr;
    // msa_global_col_attn_;
    LayerNorm* msa_global_col_layernorm_   = nullptr;
    LinearLayer* msa_global_col_to_b_      = nullptr;
    LinearLayer* msa_global_col_to_g_      = nullptr;
    LinearLayer* msa_global_col_to_out_    = nullptr;
    LinearLayer* msa_global_col_Wq_        = nullptr;
    LinearLayer* msa_global_col_Wk_        = nullptr;
    LinearLayer* msa_global_col_Wv_        = nullptr;
    //msa_col_attn_;
    LayerNorm* msa_col_layernorm_   = nullptr;
    LinearLayer* msa_col_to_b_      = nullptr; 
    LinearLayer* msa_col_to_g_      = nullptr;
    LinearLayer* msa_col_to_out_    = nullptr;
    LinearLayer* msa_col_Wq_        = nullptr;
    LinearLayer* msa_col_Wk_        = nullptr;
    LinearLayer* msa_col_Wv_        = nullptr;
    //pair_row_attn_;
    LayerNorm* pair_row_layernorm_       = nullptr;
    LayerNorm* bias_row_layernorm_       = nullptr;
    LinearLayer* pair_row_to_b_      = nullptr;
    LinearLayer* pair_row_to_g_      = nullptr;
    LinearLayer* pair_row_to_out_    = nullptr;
    LinearLayer* pair_row_Wq_        = nullptr;
    LinearLayer* pair_row_Wk_        = nullptr;
    LinearLayer* pair_row_Wv_        = nullptr;
    //pair_col_attn_;
    LayerNorm* pair_col_layernorm_       = nullptr;
    LayerNorm* bias_col_layernorm_       = nullptr;
    LinearLayer* pair_col_to_b_      = nullptr;
    LinearLayer* pair_col_to_g_      = nullptr;
    LinearLayer* pair_col_to_out_    = nullptr;
    LinearLayer* pair_col_Wq_        = nullptr;
    LinearLayer* pair_col_Wk_        = nullptr;
    LinearLayer* pair_col_Wv_        = nullptr;
    //tri_mul_out_;
    LayerNorm*   tri_mul_out_layernorm_        = nullptr;
    LinearLayer* tri_mul_out_left_proj_        = nullptr;
    LinearLayer* tri_mul_out_right_proj_       = nullptr;
    LinearLayer* tri_mul_out_left_gate_        = nullptr;
    LinearLayer* tri_mul_out_right_gate_       = nullptr;
    LinearLayer* tri_mul_out_gate_             = nullptr;
    LayerNorm*   tri_mul_out_output_layernorm_ = nullptr;
    LinearLayer* tri_mul_out_out_proj_         = nullptr;
    //tri_mul_in_;
    LayerNorm*   tri_mul_in_layernorm_         = nullptr;
    LinearLayer* tri_mul_in_left_proj_         = nullptr;
    LinearLayer* tri_mul_in_right_proj_        = nullptr;
    LinearLayer* tri_mul_in_left_gate_         = nullptr;
    LinearLayer* tri_mul_in_right_gate_        = nullptr;
    LinearLayer* tri_mul_in_gate_              = nullptr;
    LayerNorm*   tri_mul_in_output_layernorm_  = nullptr;
    LinearLayer* tri_mul_in_out_proj_          = nullptr;

    //TODO: 3D SE


    // Tracks
    std::unique_ptr<MSATrack> msa_track_;
    std::unique_ptr<PairTrack> pair_track_;
    std::unique_ptr<StateTrack> state_track_;
    
    // 迭代块
    std::vector<std::unique_ptr<IterBlock>> extra_blocks_;
    std::vector<std::unique_ptr<IterBlock>> main_blocks_;
    std::vector<std::unique_ptr<IterBlock>> refine_blocks_;
    
    // 输出头
    //struct OutputHeads;
    //std::unique_ptr<OutputHeads> heads_;
};

} // namespace rfaa

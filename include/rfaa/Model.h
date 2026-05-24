#pragma once

#include "Track.h"
#include "Attention.h"
#include "SE3Transformer.h"
#include "PositionalEncoding.h"
#include <string>

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

    void projStateAddToQueryRow(TensorF32& msa, const TensorF32& proj_state);

    TensorF32 computeRBFFeature(const TensorF32& coords);
    
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

// RFAA 主模型
class RFAAModel {
public:
    explicit RFAAModel(const RFAAConfig& config = RFAAConfig{});
    ~RFAAModel();
    
    // 前向传播
    ModelOutput forward(const ModelInput& input);

    TensorF32 getTemplEmb(const TensorF32& t1d, const TensorF32& t2d);
    
    // 加载/保存权重
    void loadWeights(const std::string& path);
    void saveWeights(const std::string& path) const;
    
    // 设备管理
    void to(Device device);
    Device device() const;
    
    // 训练/推理模式
    void train();
    void eval();
    bool isTraining() const;
    
private:
    RFAAConfig config_;
    Device device_ = Device::CPU;
    bool training_ = false;
    
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

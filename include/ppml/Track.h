#pragma once

#include "Tensor.h"
#include "Embedding.h"
#include "Attention.h"
#include "PositionalEncoding.h"

namespace ppml {

// Track 基类：MSA / Pair / State 的公共接口
class Track {
public:
    virtual ~Track() = default;
    
    // 前向传播接口
    virtual void forward() = 0;
    
    // 获取当前表示
    virtual TensorF32& representation() = 0;
    virtual const TensorF32& representation() const = 0;
    
    // 维度信息
    virtual int feature_dim() const = 0;
    
    // 设备管理
    void to(Device device);
    Device device() const { return device_; }
    
protected:
    Device device_ = Device::CPU;
    TensorF32 repr_;  // 当前表示
};

// 1D MSA Track
class MSATrack : public Track {
public:
    MSATrack(int n_seq, int seq_len, int dim = D_MSA, Device device = Device::CPU);

    // impl need this dtor
    //~MSATrack() override = default;
    
    // 从输入特征初始化 (msa_latent: B,N,L,164 或 msa_full: B,N,L,83)
    void init_from_features(const TensorF32& features);
    
    // Step 1: msa2msa 自更新
    void update_self(const TensorF32& pair_bias, const TensorF32& state);
    
    // 获取 query row (msa[:,0])
    TensorF32 query_row() const;
    
    // 实现接口
    void forward() override {}
    TensorF32& representation() override { return repr_; }
    const TensorF32& representation() const override { return repr_; }
    int feature_dim() const override { return dim_; }
    
    int n_seq() const { return n_seq_; }
    int seq_len() const { return seq_len_; }
    
private:
    int n_seq_;
    int seq_len_;
    int dim_;
    
    // 子模块
    //struct Impl;
    // here is the problem: unique_ptr cannot be used in header if Impl is not defined?
    //std::unique_ptr<Impl> impl_;
};

// 2D Pair Track
class PairTrack : public Track {
public:
    PairTrack(int seq_len, int dim = D_PAIR, Device device = Device::CPU);
    
    // 设置外部注入的层 (由 PPMLModel 管理)
    void set_embeddings(EmbeddingLayer* left_emb, EmbeddingLayer* right_emb,
                        PositionalEncoding* pos_enc);
    
    // 从 embedding 初始化
    void init_from_embedding(const TensorF32& left, const TensorF32& right,
                             const TensorF32& bond_feats, const TensorF32& dist_matrix,
                             const TensorF32& index);
    
    void inject_template(const TensorF32& templ);
    TensorF32 templ_stack(const TensorF32& in_templ, const TensorF32& rbf_feat, const TensorF32& t1d);
    
    void forward() override {}
    TensorF32& representation() override { return repr_; }
    const TensorF32& representation() const override { return repr_; }
    int feature_dim() const override { return dim_; }
    
    int seq_len() const { return seq_len_; }
    
private:
    int seq_len_;
    int dim_;
    
    // 外部注入的层 (non-owning)
    EmbeddingLayer*    left_emb_   = nullptr;
    EmbeddingLayer*    right_emb_  = nullptr;
    PositionalEncoding* pos_enc_    = nullptr;
};

// 1D State Track
class StateTrack : public Track {
public:
    StateTrack(int seq_len, int dim = D_STATE, Device device = Device::CPU);
    
    // 从 embedding 初始化
    void init_from_embedding(const TensorF32& seq_tokens);
    
    // Template 注入 (cross-attention)
    void inject_template(const TensorF32& t1d, const TensorF32& tor_feat);
    
    // Step 4: str2str (SE3 Transformer 重建)
    void rebuild_from_se3(const TensorF32& msa_query, const TensorF32& pair, 
                          const TensorF32& coords);
    
    // 获取 state outer product gate
    TensorF32 get_gate(int gate_dim) const;
    
    // 实现接口
    void forward() override {}
    TensorF32& representation() override { return repr_; }
    const TensorF32& representation() const override { return repr_; }
    int feature_dim() const override { return dim_; }
    
    int seq_len() const { return seq_len_; }
    
private:
    int seq_len_;
    int dim_;
    
    //struct Impl;
    //std::unique_ptr<Impl> impl_;
};

} // namespace ppml

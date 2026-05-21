#pragma once

#include "Tensor.h"

namespace rfaa {

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
    
    // 从 embedding 初始化
    void init_from_embedding(const TensorF32& left, const TensorF32& right, const TensorF32& bond_feats, const TensorF32& dist_matrix);
    
    // Step 2: msa2pair (Outer Product Mean)
    void update_from_msa(const TensorF32& msa);
    
    // Step 3: pair2pair 自更新
    void update_self(const TensorF32& state_gate, const TensorF32& rbf_feat);
    
    // 获取 attention bias
    TensorF32 get_attention_bias(int n_head) const;
    
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

// 1D State Track
class StateTrack : public Track {
public:
    StateTrack(int seq_len, int dim = D_STATE, Device device = Device::CPU);
    
    // 从 embedding 初始化
    void init_from_embedding(const TensorF32& seq_tokens);
    
    // Template 注入 (cross-attention)
    void inject_template(const TensorF32& t1d);
    
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

} // namespace rfaa

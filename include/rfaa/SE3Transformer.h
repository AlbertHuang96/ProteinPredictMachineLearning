#pragma once

#include "Tensor.h"
#include <vector>

namespace rfaa {

// SE3 Transformer 配置
struct SE3Config {
    int node_dim;           // 节点特征维度 (l0)
    int edge_dim;           // 边特征维度
    int hidden_dim;         // 隐藏层维度
    int n_layers;           // 层数
    int n_heads;            // 注意力头数
    std::vector<int> l0_features;  // 各层 l0 输出维度
    std::vector<int> l1_features;  // 各层 l1 输出维度 (向量特征)
};

// SE3 Transformer: E(3) 等变图神经网络
class SE3Transformer {
public:
    explicit SE3Transformer(const SE3Config& config);
    ~SE3Transformer();
    
    // 前向传播
    // nodes: (B*L, node_dim) - 标量特征 (l0)
    // edges: (B*L, B*L, edge_dim) - 边特征
    // coords: (B*L, 3, 3) - l1 输入 (Ca-relative 向量)
    // 返回: {l0_out, l1_out}
    struct Output {
        TensorF32 l0;       // 标量输出 (B*L, l0_dim)
        TensorF32 l1;       // 向量输出 (B*L, 3, l1_dim)
    };
    
    Output forward(const TensorF32& nodes, const TensorF32& edges, 
                   const TensorF32& coords);
    
private:
    SE3Config config_;
    //struct Impl;
    //std::unique_ptr<Impl> impl_;
};

// 结构更新器：从 SE3 输出更新坐标
class StructureUpdate {
public:
    StructureUpdate();
    
    // 应用旋转+平移偏移更新坐标
    // coords: (B, L, 3, 3), l1_out: (B*L, 3, l1_dim)
    TensorF32 update_coords(const TensorF32& coords, const TensorF32& l1_out);
    
    // 预测侧链扭转角
    // msa_query: (B, L, D_MSA), state: (B, L, D_STATE)
    // 返回: alpha (B, L, NTOTALDOFS, 2) - cos/sin
    TensorF32 predict_torsion(const TensorF32& msa_query, const TensorF32& state);
    
private:
    //struct Impl;
    //std::unique_ptr<Impl> impl_;
};

} // namespace rfaa

#pragma once

#include "Tensor.h"
#include "Core.h"
#include <memory>
#include <vector>

namespace rfaa {

class EmbeddingLayer;  // 前向声明

/**
 * @brief Positional Encoding module for RFAA
 * 
 * This module computes positional encoding based on residue and atom distances.
 * It is used in the pair bias computation in IterBlock.
 * 
 * Python equivalent:
 *   class PositionalEncoding(nn.Module):
 *       def __init__(self, minpos_res=-32, maxpos_res=32, maxpos_atom=8):
 *       def forward(self, seq, idx, bond_feats, dist_matrix, same_chain=None):
 */
class PositionalEncoding {
public:
    // 旧值类型构造函数 (保留注释):
    // PositionalEncoding(int minpos_res = -32, int maxpos_res = 32, int maxpos_atom = 8, int d_pair = D_PAIR);
    PositionalEncoding() = default;
    void set_params(int minpos_res, int maxpos_res, int maxpos_atom, int d_pair,
                    EmbeddingLayer* emb_res, EmbeddingLayer* emb_atom);
    
    ~PositionalEncoding() = default;
    
    /**
     * @brief Forward pass (full version)
     * @param seq Sequence tokens (B, L)
     * @param idx Residue indices (B, L)
     * @param bond_feats Bond features (B, L, L)
     * @param dist_matrix Precomputed bond distances (B, L, L)
     * @param same_chain Same chain mask (B, L, L) - optional
     * @return Positional encoding tensor (B, L, L, d_pair)
     */
    TensorF32 forward(const TensorF32& seq,
                     const TensorF32& idx,
                     const TensorF32& bond_feats,
                     const TensorF32& dist_matrix,
                     const TensorF32& same_chain = TensorF32());
    
    /**
     * @brief Forward pass (simplified version using only coords)
     * 
     * This is a convenience method that computes positional encoding
     * using only coordinates. Used in RFAA.cpp line 196.
     * 
     * @param coords Coordinates (B, L, A, 3)
     * @return Positional encoding tensor (B, L, L, d_pair)
     */
    TensorF32 forward(const TensorF32& coords);
    
    /**
     * @brief Forward pass (graph version) — 常数注入方案
     * 
     * 与 forward 语义一致，但输入/输出均为图节点指针（ggml 布局 dims[0]=最内维）：
     *   seq        : 值 (B, L)         = 图 [L, B]（SM 原子 token ∈ [33,80)）
     *   idx        : 值 (B, L)         = 图 [L, B]
     *   bond_feats : 值 (B, L, L)      = 图 [L, L, B]
     *   dist_matrix: 值 (B, L, L)      = 图 [L, L, B]
     *   same_chain : 值 (B, L, L)      = 图 [L, L, B]（本实现未使用）
     * 返回: pair bias 图节点 [D_PAIR, L, L, B]。
     * 实现: 几何预处理（getResAtomDist + bucketize）走值版，结果桶索引以 constant_tensor
     *       注入图；emb_res_/emb_atom_ 用 forward_graph（get_rows）查表，add_impl 汇总。
     */
    TensorF32* forward_graph(TensorF32* seq, TensorF32* idx,
                             TensorF32* bond_feats, TensorF32* dist_matrix,
                             TensorF32* same_chain = nullptr);
    
private:
    /**
     * @brief Calculate residue and atom bond distances
     * 
     * Python equivalent: get_res_atom_dist()
     * 
     * @param idx Residue index (B, L)
     * @param bond_feats Bond features (B, L, L)
     * @param dist_matrix Precomputed bond distances (B, L, L)
     * @param sm_mask Small molecule mask (B, L)
     * @param minpos_res Minimum residue distance
     * @param maxpos_res Maximum residue distance
     * @param maxpos_atom Maximum atom bond distance
     * @param cyclize Cyclization mask (B, L) - optional
     * @return Pair of (res_dist, atom_dist), both (B, L, L)
     */
    std::pair<TensorF32, TensorF32> getResAtomDist(
        const TensorF32& idx,
        const TensorF32& bond_feats,
        const TensorF32& dist_matrix,
        const TensorF32& sm_mask,
        int minpos_res,
        int maxpos_res,
        int maxpos_atom,
        const TensorF32& cyclize = TensorF32());
    
    /**
     * @brief Bucketize distances (similar to torch.bucketize)
     * @param distances Distance tensor
     * @param bins Bucket boundaries (float values)
     * @return Bucket indices (int tensor)
     */
    TensorF32 bucketize(const TensorF32& distances, const std::vector<int>& bins);
    
private:
    int minpos_res_ = -32;
    int maxpos_res_ = 32;
    int maxpos_atom_ = 8;
    int d_pair_ = D_PAIR;
    
    // 旧值类型 (保留注释):
    // std::unique_ptr<EmbeddingLayer> emb_res_;
    // std::unique_ptr<EmbeddingLayer> emb_atom_;
    EmbeddingLayer* emb_res_  = nullptr; // (65, D_PAIR)
    EmbeddingLayer* emb_atom_ = nullptr; // (17, D_PAIR)
};

} // namespace rfaa

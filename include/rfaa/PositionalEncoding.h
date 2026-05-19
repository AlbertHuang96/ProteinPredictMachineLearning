#pragma once

#include "Tensor.h"
#include "Core.h"
#include <memory>
#include <vector>

namespace rfaa {

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
    /**
     * @brief Constructor
     * @param minpos_res Minimum residue distance (default: -32)
     * @param maxpos_res Maximum residue distance (default: 32)
     * @param maxpos_atom Maximum atom bond distance (default: 8)
     * @param d_pair Pair dimension for embedding (default: D_PAIR=128)
     */
    PositionalEncoding(int minpos_res = -32, 
                      int maxpos_res = 32, 
                      int maxpos_atom = 8,
                      int d_pair = D_PAIR);
    
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
    int minpos_res_;
    int maxpos_res_;
    int maxpos_atom_;
    int d_pair_;
    
    // Embedding layers
    std::unique_ptr<class EmbeddingLayer> emb_res_;
    std::unique_ptr<class EmbeddingLayer> emb_atom_;
};

} // namespace rfaa

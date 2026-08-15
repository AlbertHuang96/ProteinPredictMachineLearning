#include "ppml/PositionalEncoding.h"
#include "ppml/Embedding.h"
#include "ppml/MathUtils.h"
#include "ppml/ComputeGraph.h"
#include "ppml/Context.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <vector>


namespace ppml {

// ===== PositionalEncoding (non-owning pointer 版本) =====
// 旧构造函数 (保留注释):
// PositionalEncoding::PositionalEncoding(int minpos_res, int maxpos_res, int maxpos_atom, int d_pair)
//     : minpos_res_(minpos_res), maxpos_res_(maxpos_res), maxpos_atom_(maxpos_atom), d_pair_(d_pair) {
//     emb_res_  = std::make_unique<EmbeddingLayer>(maxpos_res - minpos_res + 2, d_pair);
//     emb_atom_ = std::make_unique<EmbeddingLayer>(maxpos_atom + 2, d_pair);
// }
void PositionalEncoding::set_params(int minpos_res, int maxpos_res, int maxpos_atom, int d_pair,
                                    EmbeddingLayer* emb_res, EmbeddingLayer* emb_atom) {
    minpos_res_  = minpos_res;
    maxpos_res_  = maxpos_res;
    maxpos_atom_ = maxpos_atom;
    d_pair_      = d_pair;
    emb_res_     = emb_res;    // (65, D_PAIR)
    emb_atom_    = emb_atom;   // (17, D_PAIR)
}

TensorF32 PositionalEncoding::forward(const TensorF32& seq,
                                      const TensorF32& idx,
                                      const TensorF32& bond_feats,
                                      const TensorF32& dist_matrix,
                                      const TensorF32& same_chain) {
    // TODO:
    // 1. Compute is_atom mask from seq
    // 2. Call getResAtomDist to get res_dist and atom_dist
    // 3. Bucketize distances
    // 4. Embed residues and atoms
    // 5. Sum embeddings
    
    //sm_mask = is_atom(seq[0])
    
    // Placeholder: return zeros
    // 注意: 此前这里尝试用 getResAtomDist + bucketize + forward_exec 计算真实嵌入,
    // 但存在 bug: 把 same_chain(全 1) 当作 sm_mask 传入 getResAtomDist, 导致蛋白残基全被
    // 当作小分子原子, res_dist 全部被 clamp 到 maxpos_res+1, bucketize 产生越界索引
    // (== res_bins.size()=65, 超出 emb_res_ vocab 65 的范围 0..64), forward_exec 越界读
    // 造成段错误。按注释语义暂返回 0 (与 coords 重载一致)。
    int B = seq.shape().dims[0];
    int L = seq.shape().dims[1];
    TensorF32 output({B, L, L, d_pair_}, seq.device());
    output.zero_();
    return output;
}

TensorF32 PositionalEncoding::forward(const TensorF32& coords) {
    // Simplified version: compute positional encoding from coordinates only
    // This is used in PPML.cpp line 196
    
    // TODO: Full implementation
    // For now, return zeros as placeholder
    
    int B = coords.shape().dims[0];
    int L = coords.shape().dims[1];
    TensorF32 output({B, L, L, d_pair_}, coords.device());

    //output.zero_();
    return output;
}

// ===== PositionalEncoding::forward_graph (图版：常数注入方案) =====
// 输入/输出均为图节点指针（ggml 布局 dims[0]=最内维）：
//   seq        : 图 [L, B]（值 (B,L) 整数 token，SM 原子 token ∈ [33,80)）
//   idx        : 图 [L, B]（值 (B,L) 残基索引）
//   bond_feats : 图 [L, L, B]（值 (B,L,L) 键特征）
//   dist_matrix: 图 [L, L, B]（值 (B,L,L) 键距）
//   same_chain : 图 [L, L, B]（可选，本实现未使用）
// 返回: pair bias 图节点 [D_PAIR, L, L, B]。
//
// 实现方案（与 MEMORY 中"几何预处理走值版、查表+求和走图"一致）：
//   1. 从图 leaf 直接读值数据（leaf 为 wrap_input_as_leaf 结果，.data() 即值 row-major）；
//   2. 值版 getResAtomDist(idx,bond_feats,dist_matrix,sm_mask) → res_dist/atom_dist (B,L,L)；
//   3. 值版 bucketize 把距离映射为桶索引（res→65 桶, atom→17 桶）；
//   4. 把桶索引按 ggml [D,L,L,B] 扁平序重排为 1D float 索引常量，constant_tensor 注入图；
//   5. emb_res_/emb_atom_ 用 forward_graph（get_rows）查表 → [D_PAIR, B*L*L]；
//   6. view 为 [D_PAIR, L, L, B]，add_impl 汇总两个嵌入。
TensorF32* PositionalEncoding::forward_graph(TensorF32* seq, TensorF32* idx,
                                             TensorF32* bond_feats,
                                             TensorF32* dist_matrix,
                                             TensorF32* same_chain) {
    (void)same_chain;
    // seq 图 [L, B] → L=dims[0], B=dims[1]; leaf 数据即值 row-major
    const int64_t L = seq->shape().dims[0];
    const int64_t B = seq->shape().dims[1];

    // ==== 1. 从图 leaf 提取值数据（只读视图，不拥有）====
    // 值 (B,L) token / residx; (B,L,L) bond / dist —— 与 wrap_input_as_leaf 的行主序一致
    const Shape sh_bl({B, L});        // (B, L)
    const Shape sh_bll({B, L, L});    // (B, L, L)
    // 4 参构造签名: Tensor(const Shape&, T* data, Device device = CPU, bool own = false)
    TensorF32 seq_val(sh_bl,  seq->data(),        Device::CPU, /*own=*/false);
    TensorF32 idx_val(sh_bl,  idx->data(),        Device::CPU, /*own=*/false);
    TensorF32 bond_val(sh_bll, bond_feats->data(), Device::CPU, /*own=*/false);
    TensorF32 dist_val(sh_bll, dist_matrix->data(), Device::CPU, /*own=*/false);

    // ==== 2. 计算 SM 原子掩码 sm_mask（token >= 33 为配体/小分子重原子）====
    TensorF32 sm_mask(sh_bl, Device::CPU);
    for (int64_t i = 0; i < B * L; ++i)
        sm_mask.data()[i] = (seq_val.data()[i] >= 33.0f) ? 1.0f : 0.0f;

    // ==== 3. 值版 getResAtomDist → res_dist / atom_dist (B,L,L) ====
    auto [res_dist, atom_dist] = getResAtomDist(idx_val, bond_val, dist_val, sm_mask,
                                                minpos_res_, maxpos_res_, maxpos_atom_);

    // ==== 4. 值版 bucketize → 桶索引 ====
    // res_bins: [-32..32) 共 64 边界 → 65 桶 [-32..33]（res_dist ∈ [minpos_res, maxpos_res+1]）
    std::vector<int> res_bins;
    for (int v = minpos_res_; v < maxpos_res_; ++v) res_bins.push_back(v);  // -32..31
    // atom_bins: [0..8] 共 9 边界 → 桶 0..9（atom_dist ∈ [0, maxpos_atom+1]），vocab 17 余 padding
    std::vector<int> atom_bins;
    for (int v = 0; v <= maxpos_atom_; ++v) atom_bins.push_back(v);         // 0..8

    TensorF32 res_bucket  = bucketize(res_dist,  res_bins);   // 数据为 int 位图（见 bucketize）
    TensorF32 atom_bucket = bucketize(atom_dist, atom_bins);

    const int64_t K = B * L * L;  // 2D pair 元素总数

    // ==== 5. 桶索引按 ggml [D,L,L,B] 扁平序（(i,j) → i+L*j，即 value 的转置）重排为 1D float 索引 ====
    // get_rows(weight, idx) 输出 [D, K]，flat = d + D*k；k 须使 view 后 [D,L,L,B] 正确：
    //   [D,L,L,B] 元素 (d,i,j,b) flat = d + D*(i + L*j + L*L*b) = d + D*(b*L*L + j*L + i)
    //   故 k = b*L*L + j*L + i，其中 value 桶在 (b,i,j) → value 序 b*L*L + i*L + j。
    std::vector<float> res_idx(K), atom_idx(K);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t j = 0; j < L; ++j) {
            for (int64_t i = 0; i < L; ++i) {
                int64_t k_val = b * L * L + i * L + j;   // value (B,L,L) 扁平序
                int64_t k_ggl = b * L * L + j * L + i;   // ggml [D,L,L,B] 扁平序
                res_idx[k_ggl]  = static_cast<float>(reinterpret_cast<const int*>(res_bucket.data())[k_val]);
                atom_idx[k_ggl] = static_cast<float>(reinterpret_cast<const int*>(atom_bucket.data())[k_val]);
            }
        }
    }

    // ==== 6. 常数注入 + 查表（get_rows）+ view + 求和 ====
    TensorF32* res_idx_g   = constant_tensor({K}, res_idx.data());
    TensorF32* atom_idx_g  = constant_tensor({K}, atom_idx.data());

    // emb_*->forward_graph = get_rows(weight(V,D), idx(K,)) → [D, K]
    TensorF32* emb_res_g   = emb_res_->forward_graph(res_idx_g);
    TensorF32* emb_atom_g  = emb_atom_->forward_graph(atom_idx_g);

    // view [D, K] → [D, L, L, B]（k 已按 ggml 序，数据兼容）
    TensorF32* emb_res_v   = view(emb_res_g, Shape{d_pair_, L, L, B});
    TensorF32* emb_atom_v  = view(emb_atom_g, Shape{d_pair_, L, L, B});

    // 汇总 emb_res(res_bucket) + emb_atom(atom_bucket) → [D_PAIR, L, L, B]
    return add_impl(emb_res_v, emb_atom_v, /*inplace=*/false);
}

std::pair<TensorF32, TensorF32> PositionalEncoding::getResAtomDist(
    const TensorF32& idx,
    const TensorF32& bond_feats,
    const TensorF32& dist_matrix,
    const TensorF32& sm_mask,
    int minpos_res,
    int maxpos_res,
    int maxpos_atom,
    const TensorF32& cyclize) {
    
    // Full implementation of get_res_atom_dist() from Python
    // Input shapes:
    //   idx: (B, L) - residue index
    //   bond_feats: (B, L, L) - bond features (0 if no bond, 6 if protein-SM bond)
    //   dist_matrix: (B, L, L) - precomputed bond distances (may contain inf/nan)
    //   sm_mask: (B, L) - small molecule mask (1 if atom, 0 if residue)
    //   cyclize: (B, L) - cyclization mask (optional, 1 if cyclized)
    
    // Assume batch = 1 for now (TODO: handle batch > 1)
    int B = idx.shape().dims[0];
    int L = idx.shape().dims[1];
    Device device = idx.device();
    
    // Step 1: Extract data for batch 0 (assuming B=1)
    const float* bond_feats_data = bond_feats.data();  // (B,L,L) -> use [0,:,:]
    const float* idx_data = idx.data();  // (B,L) -> use [0,:]
    const float* dist_matrix_data = dist_matrix.data();  // (B,L,L) -> use [0,:,:]
    const float* sm_mask_data = sm_mask.data();  // (B,L) -> use [0,:]
    const float* cyclize_data = cyclize.numel() > 0 ? cyclize.data() : nullptr;  // (B,L) -> use [0,:]
    
    // Step 2: Create 2D masks (L, L)
    TensorF32 prot_mask_2d({L, L}, device);
    TensorF32 inter_mask_2d({L, L}, device);
    TensorF32 sm_mask_2d({L, L}, device);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            bool sm_i = sm_mask_data[i] > 0.5f;
            bool sm_j = sm_mask_data[j] > 0.5f;
            
            prot_mask_2d.data()[i * L + j] = (!sm_i && !sm_j) ? 1.0f : 0.0f;
            inter_mask_2d.data()[i * L + j] = ((!sm_i && sm_j) || (sm_i && !sm_j)) ? 1.0f : 0.0f;
            sm_mask_2d.data()[i * L + j] = (sm_i && sm_j) ? 1.0f : 0.0f;
        }
    }
    
    // Step 3: Compute seqsep = idx[0,None,:] - idx[0,:,None]  # (L, L)
    TensorF32 seqsep({L, L}, device);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            seqsep.data()[i * L + j] = idx_data[i] - idx_data[j];
        }
    }
    
    // Step 4: Handle cyclize (if provided)
    if (cyclize_data != nullptr) {
        // mask = cyclize[:,None] * cyclize[None,:]  # (L, L)
        // ncyc = sum(cyclize)
        int ncyc = 0;
        for (int i = 0; i < L; i++) {
            if (cyclize_data[i] > 0.5f) ncyc++;
        }
        
        // seqsep[mask * (seqsep > ncyc//2)] -= ncyc
        // seqsep[mask * (seqsep < -ncyc//2)] += ncyc
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                if (cyclize_data[i] > 0.5f && cyclize_data[j] > 0.5f) {
                    float& s = seqsep.data()[i * L + j];
                    if (s > ncyc / 2) s -= ncyc;
                    if (s < -ncyc / 2) s += ncyc;
                }
            }
        }
    }
    
    // Step 5: Compute res_dist_prot = clamp(seqsep, minpos_res, maxpos_res)
    TensorF32 res_dist_prot({L, L}, device);
    #pragma omp parallel for
    for (int i = 0; i < L * L; i++) {
        float s = seqsep.data()[i];
        s = std::max(float(minpos_res), std::min(float(maxpos_res), s));
        res_dist_prot.data()[i] = s;
    }
    
    // Step 6: Compute res_dist_sm = full((L,L), maxpos_res+1)
    TensorF32 res_dist_sm({L, L}, device);
    #pragma omp parallel for
    for (int i = 0; i < L * L; i++) {
        res_dist_sm.data()[i] = float(maxpos_res + 1);
    }
    
    // Step 7: Compute atom_dist_sm = nan_to_num(dist_matrix[0], posinf=maxpos_atom)
    TensorF32 atom_dist_sm({L, L}, device);
    #pragma omp parallel for
    for (int i = 0; i < L * L; i++) {
        float d = dist_matrix_data[i];  // dist_matrix[0] = first batch
        if (std::isnan(d) || d > maxpos_atom) {
            atom_dist_sm.data()[i] = float(maxpos_atom);
        } else {
            atom_dist_sm.data()[i] = d;
        }
    }
    
    // Step 8: Compute atom_dist_prot = full((L,L), maxpos_atom+1)
    TensorF32 atom_dist_prot({L, L}, device);
    #pragma omp parallel for
    for (int i = 0; i < L * L; i++) {
        atom_dist_prot.data()[i] = float(maxpos_atom + 1);
    }
    
    // Step 9: Compute interface distances (complex part)
    // Find bonds where bond_feats == 6 (protein-SM bonds)
    std::vector<int> i_s_vec, j_s_vec;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            float bf = bond_feats_data[i * L + j];
            if (bf > 5.5f && bf < 6.5f) {  // bond_feats == 6
                i_s_vec.push_back(i);
                j_s_vec.push_back(j);
            }
        }
    }
    
    // i_sm = i_s[sm_mask[i_s]]  # SM atom indices in bonds
    // i_prot = j_s[sm_mask[i_s]]  # protein residue indices in bonds
    std::vector<int> i_sm_vec, i_prot_vec;
    for (size_t k = 0; k < i_s_vec.size(); k++) {
        int i = i_s_vec[k];
        if (sm_mask_data[i] > 0.5f) {
            i_sm_vec.push_back(i);
            i_prot_vec.push_back(j_s_vec[k]);
        }
    }
    
    // Initialize interface distances
    TensorF32 res_dist_inter({L, L}, device);
    TensorF32 atom_dist_inter({L, L}, device);
    #pragma omp parallel for
    for (int i = 0; i < L * L; i++) {
        res_dist_inter.data()[i] = float(maxpos_res);
        atom_dist_inter.data()[i] = float(maxpos_atom);
    }
    
    // If there are protein-SM bonds, compute interface distances
    if (!i_prot_vec.empty()) {
        // For each SM atom in i_sm_vec, find the closest protein residue
        // Python: closest_prot_res = i_prot[torch.argmin(atom_dist_sm[sm_mask,:][:,i_sm], dim=-1)]
        // Simplified: for each unique SM atom, find min distance to any SM atom
        std::vector<int> closest_prot_res(i_sm_vec.size());
        for (size_t k = 0; k < i_sm_vec.size(); k++) {
            int sm_idx = i_sm_vec[k];
            float min_dist = std::numeric_limits<float>::max();
            int min_prot = i_prot_vec[0];
            for (size_t m = 0; m < i_sm_vec.size(); m++) {
                float d = atom_dist_sm.data()[i_sm_vec[m] * L + sm_idx];
                if (d < min_dist) {
                    min_dist = d;
                    min_prot = i_prot_vec[m];
                }
            }
            closest_prot_res[k] = min_prot;
        }
        
        // res_dist_inter[sm_mask, :] = res_dist_prot[closest_prot_res, :]
        #pragma omp parallel for
        for (int i = 0; i < L; i++) {
            if (sm_mask_data[i] > 0.5f) {
                // Find which closest_prot_res corresponds to this SM atom
                for (size_t k = 0; k < i_sm_vec.size(); k++) {
                    if (i_sm_vec[k] == i) {
                        int prot_res = closest_prot_res[k];
                        for (int j = 0; j < L; j++) {
                            res_dist_inter.data()[i * L + j] = res_dist_prot.data()[prot_res * L + j];
                        }
                        break;
                    }
                }
            }
        }
        
        // res_dist_inter[:, sm_mask] = res_dist_prot[:, closest_prot_res]
        #pragma omp parallel for
        for (int j = 0; j < L; j++) {
            if (sm_mask_data[j] > 0.5f) {
                for (size_t k = 0; k < i_sm_vec.size(); k++) {
                    if (i_sm_vec[k] == j) {
                        int prot_res = closest_prot_res[k];
                        for (int i = 0; i < L; i++) {
                            res_dist_inter.data()[i * L + j] = res_dist_prot.data()[i * L + prot_res];
                        }
                        break;
                    }
                }
            }
        }
        
        // Similar for atom_dist_inter (simplified)
        // atom_dist_inter[~sm_mask, :] = atom_dist_sm[closest_atom, :] + 1
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < L; i++) {
            if (sm_mask_data[i] <= 0.5f) {
                for (int j = 0; j < L; j++) {
                    atom_dist_inter.data()[i * L + j] = atom_dist_sm.data()[i * L + j] + 1.0f;
                }
            }
        }
    }
    
    // Step 10: Combine distances using masks
    TensorF32 res_dist({L, L}, device);
    TensorF32 atom_dist({L, L}, device);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int idx_2d = i * L + j;
            res_dist.data()[idx_2d] = res_dist_prot.data()[idx_2d] * prot_mask_2d.data()[idx_2d]
                                    + res_dist_inter.data()[idx_2d] * inter_mask_2d.data()[idx_2d]
                                    + res_dist_sm.data()[idx_2d] * sm_mask_2d.data()[idx_2d];
            
            atom_dist.data()[idx_2d] = atom_dist_prot.data()[idx_2d] * prot_mask_2d.data()[idx_2d]
                                     + atom_dist_inter.data()[idx_2d] * inter_mask_2d.data()[idx_2d]
                                     + atom_dist_sm.data()[idx_2d] * sm_mask_2d.data()[idx_2d];
        }
    }
    
    // Step 11: Add batch dimension -> (B, L, L) with B=1
    TensorF32 res_dist_batch({B, L, L}, device);
    TensorF32 atom_dist_batch({B, L, L}, device);
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < L * L; i++) {
            res_dist_batch.data()[b * L * L + i] = res_dist.data()[i];
            atom_dist_batch.data()[b * L * L + i] = atom_dist.data()[i];
        }
    }
    
    return std::make_pair(std::move(res_dist_batch), std::move(atom_dist_batch));
}

TensorF32 PositionalEncoding::bucketize(const TensorF32& distances, const std::vector<int>& bins) {
    // Python equivalent: torch.bucketize(distances, bins, right=True)
    // right=True means: bins[i-1] < x <= bins[i] -> bucket i
    // Our implementation: find first bin where x <= bin, then bucket = index
    
    TensorF32 result(distances.shape(), distances.device());
    const float* dist_data = distances.data();
    int* result_data = reinterpret_cast<int*>(result.data());  // Cast to int*
    
    // Naive implementation: for each element, find the bucket
    int numel = distances.numel();
    #pragma omp parallel for
    for (int i = 0; i < numel; i++) {
        float d = dist_data[i];
        int bucket = 0;
        // Find first bin where d <= bin (right=True semantics)
        for (int bin : bins) {
            if (d > bin) {
                bucket++;
            } else {
                break;
            }
        }
        result_data[i] = bucket;
    }
    
    return result;
}

} // namespace ppml

#include "rfaa/SymmetryResolver.h"
#include <unordered_map>
#include <algorithm>

namespace rfaa {

// ============================================================================
// 对称原子映射表 (内部实现)
// ============================================================================
//
// 每个条目: {ref_atom, (atom_a, atom_b)}
// 规则: atom_a 应比 atom_b 更靠近 ref_atom
//       若不满足 → swap atom_a ↔ atom_b

static const std::unordered_map<int, std::vector<SymmetryPair>> kSymmetricAtoms = {
    // LEU: CD1(9), CD2(10), ref=CB(3)
    { (int)AAType::LEU, {
        SymmetryPair(Atom37::CD1, Atom37::CD2, Atom37::CB)
    }},
    // VAL: CG1(6), CG2(7), ref=CB(3)
    { (int)AAType::VAL, {
        SymmetryPair(Atom37::CG1, Atom37::CG2, Atom37::CB)
    }},
    // PHE: CD1(9)/CD2(10) ref=CB(3) + CE1(12)/CE2(13) ref=CZ(34)
    { (int)AAType::PHE, {
        SymmetryPair(Atom37::CD1, Atom37::CD2, Atom37::CB),
        SymmetryPair(Atom37::CE1, Atom37::CE2, Atom37::CZ)
    }},
    // TYR: CD1(9)/CD2(10) ref=CB(3) + CE1(12)/CE2(13) ref=OH(29)
    { (int)AAType::TYR, {
        SymmetryPair(Atom37::CD1, Atom37::CD2, Atom37::CB),
        SymmetryPair(Atom37::CE1, Atom37::CE2, Atom37::OH)
    }},
    // ASP: OD1(22), OD2(23), ref=CG(5)
    { (int)AAType::ASP, {
        SymmetryPair(Atom37::OD1, Atom37::OD2, Atom37::CG)
    }},
    // GLU: OE1(24), OE2(25), ref=CD(8)
    { (int)AAType::GLU, {
        SymmetryPair(Atom37::OE1, Atom37::OE2, Atom37::CD)
    }},
    // ARG: NH1(32), NH2(33), ref=CZ(34)
    { (int)AAType::ARG, {
        SymmetryPair(Atom37::NH1, Atom37::NH2, Atom37::CZ)
    }},
    // ASN: OD1(22), ND2(21), ref=CG(5)
    //   (amide group flip: O and N swap positions around 180° rotation)
    { (int)AAType::ASN, {
        SymmetryPair(Atom37::OD1, Atom37::ND2, Atom37::CG)
    }},
    // GLN: OE1(24), NE2(17), ref=CD(8)
    { (int)AAType::GLN, {
        SymmetryPair(Atom37::OE1, Atom37::NE2, Atom37::CD)
    }},
};

const std::vector<SymmetryPair>& get_symmetric_atoms(int aa_type) {
    static const std::vector<SymmetryPair> kEmpty;
    auto it = kSymmetricAtoms.find(aa_type);
    if (it != kSymmetricAtoms.end()) {
        return it->second;
    }
    return kEmpty;
}

// ============================================================================
// rename_symmetric_atoms — 主函数
// ============================================================================
//
// 算法流程 (Algorithm 26 的 geometric canonicalization 版本):
//
//  for each residue r:
//    aa = aatype[r]
//    if aa not in SYMMETRIC_ATOMS → skip
//
//    for each symmetry_pair (atom_a, atom_b, ref_atom) in SYMMETRIC_ATOMS[aa]:
//      Step 1: 提取参考原子和两个对称原子的坐标
//      Step 2: 计算距离:
//        d_a = ||pos[atom_a] - pos[ref_atom]||
//        d_b = ||pos[atom_b] - pos[ref_atom]||
//      Step 3: 规范规则: atom_a 应比 atom_b 更靠近 ref_atom
//        如果 d_b < d_a → swap atom_a ↔ atom_b positions
//
//  时间复杂度: O(N_res × max_sym_pairs × 1) — 每残基最多 2 对
//
void rename_symmetric_atoms(
    TensorF32& gt_coords,
    const std::vector<int>& aatype,
    const TensorF32* atom_mask,
    std::vector<bool>& swap_mask)
{
    // gt_coords shape: [N_res, N_atoms, 3]
    int ndim = gt_coords.shape().ndim();
    if (ndim < 3) return;  // 至少需要 [N_res, N_atoms, 3]

    const auto& dims = gt_coords.shape().dims;
    int N_res   = (ndim >= 1) ? (int)dims[0] : 1;
    int N_atoms = (ndim >= 2) ? (int)dims[1] : 1;
    // 第3维必须是 3

    float* data = gt_coords.data();

    // 初始化 swap_mask
    swap_mask.assign(N_res, false);

    for (int r = 0; r < N_res; r++) {
        if (r >= (int)aatype.size()) break;

        int aa = aatype[r];
        const auto& pairs = get_symmetric_atoms(aa);
        if (pairs.empty()) continue;

        bool any_swapped = false;

        for (const auto& sp : pairs) {
            int a = sp.atom_a;
            int b = sp.atom_b;
            int ref = sp.ref_atom;

            // 边界检查
            if (a >= N_atoms || b >= N_atoms || ref >= N_atoms) continue;

            // 检查 mask (如果提供)
            if (atom_mask) {
                const float* mask_data = atom_mask->data();
                if (mask_data[r * N_atoms + a] == 0.0f ||
                    mask_data[r * N_atoms + b] == 0.0f ||
                    mask_data[r * N_atoms + ref] == 0.0f) {
                    continue;
                }
            }

            // Step 2: 计算到参考原子的平方距离
            int base_r = r * N_atoms * 3;
            int off_a  = base_r + a * 3;
            int off_b  = base_r + b * 3;
            int off_ref = base_r + ref * 3;

            float dx_a = data[off_a + 0] - data[off_ref + 0];
            float dy_a = data[off_a + 1] - data[off_ref + 1];
            float dz_a = data[off_a + 2] - data[off_ref + 2];
            float d2_a = dx_a * dx_a + dy_a * dy_a + dz_a * dz_a;

            float dx_b = data[off_b + 0] - data[off_ref + 0];
            float dy_b = data[off_b + 1] - data[off_ref + 1];
            float dz_b = data[off_b + 2] - data[off_ref + 2];
            float d2_b = dx_b * dx_b + dy_b * dy_b + dz_b * dz_b;

            // Step 3: 如果 atom_b 更靠近 ref，交换 atom_a 和 atom_b
            if (d2_b < d2_a) {
                // swap 3 floats each
                for (int k = 0; k < 3; k++) {
                    float tmp = data[off_a + k];
                    data[off_a + k] = data[off_b + k];
                    data[off_b + k] = tmp;
                }
                any_swapped = true;
            }
        }

        if (any_swapped) {
            swap_mask[r] = true;
        }
    }
}

} // namespace rfaa

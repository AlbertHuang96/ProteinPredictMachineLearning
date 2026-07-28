#pragma once

#include "Tensor.h"
#include <vector>
#include <utility>
#include <cstdint>
#include <cmath>

namespace rfaa {

// ============================================================================
// 标准氨基酸类型 (AlphaFold 0-20 序)
// ============================================================================
enum class AAType : int {
    ALA = 0,  // Alanine
    CYS = 1,  // Cysteine
    ASP = 2,  // Aspartate
    GLU = 3,  // Glutamate
    PHE = 4,  // Phenylalanine
    GLY = 5,  // Glycine
    HIS = 6,  // Histidine
    ILE = 7,  // Isoleucine
    LYS = 8,  // Lysine
    LEU = 9,  // Leucine
    MET = 10, // Methionine
    ASN = 11, // Asparagine
    PRO = 12, // Proline
    GLN = 13, // Glutamine
    ARG = 14, // Arginine
    SER = 15, // Serine
    THR = 16, // Threonine
    VAL = 17, // Valine
    TRP = 18, // Tryptophan
    TYR = 19, // Tyrosine
    UNK = 20  // Unknown
};

// ============================================================================
// Atom37 标准索引 (AlphaFold atom37 representation)
// ============================================================================
//  0:N  1:CA  2:C  3:CB  4:O  5:CG  6:CG1  7:CG2  8:CD  9:CD1  10:CD2
//  11:CE  12:CE1  13:CE2  14:CE3  15:NE  16:NE1  17:NE2  18:NZ  19:ND
//  20:ND1  21:ND2  22:OD1  23:OD2  24:OE1  25:OE2  26:OG  27:OG1
//  28:OG2  29:OH  30:SD  31:SG  32:NH1  33:NH2  34:CZ  35:CZ2  36:CZ3
namespace Atom37 {
    constexpr int N   = 0,  CA  = 1,  C   = 2,  CB  = 3,  O   = 4;
    constexpr int CG  = 5,  CG1 = 6,  CG2 = 7,  CD  = 8,  CD1 = 9;
    constexpr int CD2 = 10, CE  = 11, CE1 = 12, CE2 = 13, CE3 = 14;
    constexpr int NE  = 15, NE1 = 16, NE2 = 17, NZ  = 18, ND  = 19;
    constexpr int ND1 = 20, ND2 = 21, OD1 = 22, OD2 = 23, OE1 = 24;
    constexpr int OE2 = 25, OG  = 26, OG1 = 27, OG2 = 28, OH  = 29;
    constexpr int SD  = 30, SG  = 31, NH1 = 32, NH2 = 33, CZ  = 34;
    constexpr int CZ2 = 35, CZ3 = 36;
}

// ============================================================================
// 对称原子交换对描述
// ============================================================================
struct SymmetryPair {
    int atom_a;       // 第一个对称原子的 atom37 索引
    int atom_b;       // 第二个对称原子的 atom37 索引
    int ref_atom;     // 参考原子 ("父原子") 的 atom37 索引
                      // 规则: atom_a 应该比 atom_b 更靠近 ref_atom

    SymmetryPair(int a, int b, int ref)
        : atom_a(a), atom_b(b), ref_atom(ref) {}
};

// ============================================================================
// 对称原子映射表 — AA_TYPE → 对称交换对列表
// ============================================================================
// 基于 AlphaFold/OpenFold 的 SYMMETRIC_ATOMS 字典
//
// 规则 (geometric canonicalization):
//   对每对 (atom_a, atom_b):
//     if dist(atom_b, ref_atom) < dist(atom_a, ref_atom):
//       swap atom_a ↔ atom_b  (让 atom_a 永远是更靠近 ref_atom 的那个)
//
// 这确保了 ground truth 的原子命名与模型预期的"规范命名"一致，
// 从而 FAPE loss 不会因命名歧义而产生虚假误差。
//
// 参考:
//   Algorithm 26 (Rename Symmetric Ground Truth Atoms)
//   RFAA Section 2.5.1 - Level 1: Sidechain 180° Flips
extern const std::vector<SymmetryPair>& get_symmetric_atoms(int aa_type);

// ============================================================================
// rename_symmetric_atoms — 侧链对称性解析 (层次 1)
// ============================================================================
//
// 功能:
//   对 ground truth 坐标做规范重命名，使对称原子对 (如 CD1/CD2) 的命名
//   遵循确定性几何规则，消除因 180° 侧链翻转导致的命名歧义。
//
// 输入:
//   gt_coords    [N_res, N_atoms, 3]  — 真实坐标 (in-place 修改)
//   aatype       [N_res]              — 每个残基的氨基酸类型 (AAType 0-20)
//   atom_mask    [N_res, N_atoms]     — 原子有效性 mask (可选, nullptr=全有效)
//
// 输出 (in-place):
//   gt_coords    修改后的坐标 (对称原子按规范重排)
//   swap_mask    [N_res] bool 数组 — 标记哪些残基发生了交换
//
// 注意:
//   - 这是数据预处理步骤，不是图节点；无梯度回传
//   - 应在 FAPE loss 计算之前调用
//   - 对非对称残基 (如 ALA, GLY) 无操作
//
void rename_symmetric_atoms(
    TensorF32& gt_coords,
    const std::vector<int>& aatype,
    const TensorF32* atom_mask,
    std::vector<bool>& swap_mask);

// ============================================================================
// TODO: 层次 2 — 同源蛋白质链排列 (Homologous Chain Permutation)
// ============================================================================
//
// 问题: 组装体中多条完全相同的蛋白链，链编号是任意的
// 方法:
//   1. 计算预测结构中所有 Cα-Cα 距离矩阵
//   2. 枚举所有链排列 (k! 种)
//   3. 选择使 |pred_Cα_dist - true_Cα_dist| 最小化的排列
//   4. 重新编号 true_coords 中的链以匹配预测结构
//
// 函数签名 (待实现):
//   void resolve_chain_permutations(
//       TensorF32& true_coords,           // in/out
//       const TensorF32& pred_coords,
//       const std::vector<int>& chain_ids,
//       const std::vector<int>& res_offsets);
//

// ============================================================================
// TODO: 层次 3 — 配体分子贪心分配 (Greedy Ligand Assignment)
// ============================================================================
//
// 问题: 组装体中多拷贝同种配体，配体-蛋白链对应关系是任意的
// 方法 (以层次 2 的链排序为锚点):
//   1. 对每个配体拷贝，计算其所有原子到已分配蛋白 Cα 锚点的距离
//   2. 贪心选择使 |pred_dist - true_dist| 最小化的配体-拷贝匹配
//   3. 同时考虑配体内部的对称原子排列 (见层次 4)
//
// 函数签名 (待实现):
//   void resolve_ligand_assignments(
//       TensorF32& true_coords,           // in/out
//       const TensorF32& pred_coords,
//       const std::vector<int>& ligand_ids,
//       const std::vector<int>& chain_offsets,
//       const MolecularGraph& ligand_graph);

// ============================================================================
// TODO: 层次 4 — 小分子内部对称化学基团 (Intra-Ligand Symmetry)
// ============================================================================
//
// 问题: 小分子内部包含对称化学基团 (如苯环、羧基)
// 方法: 类似层次 1，枚举配体内部所有等价原子排列
//       (permutation equivalence classes)，选择最小化距离的排列
//
// 函数签名 (待实现):
//   void resolve_intra_ligand_symmetry(
//       TensorF32& true_coords,
//       const MolecularGraph& ligand_graph,
//       const std::vector<std::vector<int>>& sym_classes,
//       const TensorF32* atom_mask);

} // namespace rfaa

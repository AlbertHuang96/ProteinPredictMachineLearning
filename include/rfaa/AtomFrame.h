#pragma once

#include "Tensor.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <utility>

namespace rfaa {

// ============================================================================
// 元素类型枚举 — 对应 RFAA Table S7 的 Frame Priorities
// ============================================================================
enum class ElementType : int32_t {
    // Metals / Cations (highest priority, most "frame-worthy")
    K  = 0,   // Potassium
    Li = 1,   // Lithium
    Ca = 2,   // Calcium
    Mg = 3,   // Magnesium
    Be = 4,   // Beryllium
    Y  = 5,   // Yttrium
    Tb = 6,   // Terbium
    U  = 7,   // Uranium
    V  = 8,   // Vanadium
    W  = 9,   // Tungsten
    Mo = 10,  // Molybdenum
    Cr = 11,  // Chromium
    Re = 12,  // Rhenium
    Mn = 13,  // Manganese
    Os = 14,  // Osmium
    Ru = 15,  // Ruthenium
    Fe = 16,  // Iron
    Pr = 17,  // Praseodymium
    Ir = 18,  // Iridium
    Rh = 19,  // Rhodium
    Co = 20,  // Cobalt
    Pt = 21,  // Platinum
    Pd = 22,  // Palladium
    Ni = 23,  // Nickel
    Au = 24,  // Gold
    Cu = 25,  // Copper
    Hg = 26,  // Mercury
    Zn = 27,  // Zinc
    Al = 28,  // Aluminium
    B  = 29,  // Boron
    Pb = 30,  // Lead
    Sn = 31,  // Tin
    Si = 32,  // Silicon
    C  = 33,  // Carbon
    Sb = 34,  // Antimony
    As = 35,  // Arsenic
    P  = 36,  // Phosphorus
    N  = 37,  // Nitrogen
    Te = 38,  // Tellurium
    Se = 39,  // Selenium
    S  = 40,  // Sulfur
    O  = 41,  // Oxygen
    I  = 42,  // Iodine
    Br = 43,  // Bromine
    Cl = 44,  // Chlorine
    F  = 45,  // Fluorine (lowest priority)
    UNKNOWN = 99
};

// ============================================================================
// ChemData — 化学元素元数据
// ============================================================================
class ChemData {
public:
    // 元素符号 → ElementType 映射
    static std::unordered_map<std::string, ElementType> element_map;

    // ElementType → 优先级 (Table S7)
    static std::unordered_map<ElementType, int> atom2frame_priority;

    // 初始化静态映射表
    static void init();
};

// ============================================================================
// Frame 定义
// ============================================================================

/**
 * @brief 单个帧的定义
 * 
 * 一个帧由 3 个原子组成 (A, B, C)，其中 B 是中心原子。
 * 帧的局部坐标系通过 Gram-Schmidt 正交化从这三个原子的坐标构造：
 *   - x_axis = normalize(C - A)
 *   - y_axis = normalize((B - A) × x_axis)
 *   - z_axis = x_axis × y_axis
 * 
 * 输出格式：(offset_pair, ...)，如 (-1,1), (0,1), (1,1)
 * - 第一个 int：原子在残基内的相对偏移（负数表示前一个残基）
 * - 第二个 int：残基的相对偏移（0=当前残基, 1=下一个残基）
 */
struct AtomFrameOffset {
    int atom_offset;    // 残基内原子偏移
    int residue_offset; // 残基偏移 (0, 1, -1 等)

    AtomFrameOffset() : atom_offset(0), residue_offset(0) {}
    AtomFrameOffset(int a, int r) : atom_offset(a), residue_offset(r) {}

    bool operator==(const AtomFrameOffset& other) const {
        return atom_offset == other.atom_offset && residue_offset == other.residue_offset;
    }

    bool operator!=(const AtomFrameOffset& other) const {
        return !(*this == other);
    }
};

/**
 * @brief 分子图的一条边
 */
struct GraphEdge {
    int src;
    int dst;
    int bond_type;  // 键类型 (single=1, double=2, triple=3, aromatic=4)

    GraphEdge() : src(0), dst(0), bond_type(1) {}
    GraphEdge(int s, int d, int bt = 1) : src(s), dst(d), bond_type(bt) {}
};

/**
 * @brief 分子图 (邻接表表示)
 */
class MolecularGraph {
public:
    MolecularGraph() : num_atoms_(0) {}

    explicit MolecularGraph(int n_atoms) : num_atoms_(n_atoms) {
        adj_.resize(n_atoms);
    }

    void add_edge(int u, int v, int bond_type = 1) {
        adj_[u].emplace_back(u, v, bond_type);
        adj_[v].emplace_back(v, u, bond_type);
    }

    int num_atoms() const { return num_atoms_; }
    const std::vector<GraphEdge>& neighbors(int node) const { return adj_[node]; }

    // 在图中查找所有长度为 n 的简单路径
    std::vector<std::vector<int>> find_all_paths_of_length_n(int n) const;

private:
    int num_atoms_;
    std::vector<std::vector<GraphEdge>> adj_;

    void dfs_paths(int u, int target_len,
                   std::vector<int>& current_path,
                   std::vector<bool>& visited,
                   std::vector<std::vector<int>>& results) const;
};

// ============================================================================
// construct_frames — 核心帧构造函数
// ============================================================================

/**
 * @brief 为分子图中的每个原子构造局部帧 (Frame)
 * 
 * 对应 Python RFAA 的 get_atom_frames() 逻辑：
 * 
 * 1. 在分子图中搜索所有长度为 2 的路径 (3个原子 A-B-C)
 * 2. 对每个原子 n：
 *    - Level 1: 如果 n 是某帧的中心原子 (n == B)，直接使用
 *    - Level 2: 如果 n 参与某帧但不是中心 (n == A 或 n == C)，调整后使用
 *    - Level 3: 孤立原子 → 输出退化帧 [(0,1), (0,1), (0,1)]
 * 3. 按优先级排序 (Table S7)，选择最优帧
 * 
 * @param graph          分子图 (邻接表)
 * @param element_types  每个原子的元素类型
 * @param residue_ids    每个原子所属的残基 ID (用于计算残基偏移)
 * @return std::vector<std::vector<AtomFrameOffset>>
 *         frames[n] = 原子 n 的帧 (3个偏移量: A, B, C)
 */
std::vector<std::vector<AtomFrameOffset>> construct_frames(
    const MolecularGraph& graph,
    const std::vector<ElementType>& element_types,
    const std::vector<int>& residue_ids
);

/**
 * @brief 从元素符号字符串构造帧 (便捷接口)
 * 
 * @param graph          分子图
 * @param element_symbols 每个原子的元素符号 (如 "C", "N", "O")
 * @param residue_ids    每个原子所属的残基 ID
 * @return 每个原子的帧偏移量
 */
std::vector<std::vector<AtomFrameOffset>> construct_frames(
    const MolecularGraph& graph,
    const std::vector<std::string>& element_symbols,
    const std::vector<int>& residue_ids
);

} // namespace rfaa

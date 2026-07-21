#include "rfaa/AtomFrame.h"
#include <algorithm>
#include <stdexcept>
#include <set>
#include <tuple>
#include <climits>

namespace rfaa {

// ============================================================================
// ChemData 静态成员初始化
// ============================================================================

std::unordered_map<std::string, ElementType> ChemData::element_map;
std::unordered_map<ElementType, int> ChemData::atom2frame_priority;

void ChemData::init() {
    // 只初始化一次
    if (!element_map.empty()) return;

    // --- 元素符号 → ElementType ---
    element_map["K"]  = ElementType::K;
    element_map["Li"] = ElementType::Li;
    element_map["Ca"] = ElementType::Ca;
    element_map["Mg"] = ElementType::Mg;
    element_map["Be"] = ElementType::Be;
    element_map["Y"]  = ElementType::Y;
    element_map["Tb"] = ElementType::Tb;
    element_map["U"]  = ElementType::U;
    element_map["V"]  = ElementType::V;
    element_map["W"]  = ElementType::W;
    element_map["Mo"] = ElementType::Mo;
    element_map["Cr"] = ElementType::Cr;
    element_map["Re"] = ElementType::Re;
    element_map["Mn"] = ElementType::Mn;
    element_map["Os"] = ElementType::Os;
    element_map["Ru"] = ElementType::Ru;
    element_map["Fe"] = ElementType::Fe;
    element_map["Pr"] = ElementType::Pr;
    element_map["Ir"] = ElementType::Ir;
    element_map["Rh"] = ElementType::Rh;
    element_map["Co"] = ElementType::Co;
    element_map["Pt"] = ElementType::Pt;
    element_map["Pd"] = ElementType::Pd;
    element_map["Ni"] = ElementType::Ni;
    element_map["Au"] = ElementType::Au;
    element_map["Cu"] = ElementType::Cu;
    element_map["Hg"] = ElementType::Hg;
    element_map["Zn"] = ElementType::Zn;
    element_map["Al"] = ElementType::Al;
    element_map["B"]  = ElementType::B;
    element_map["Pb"] = ElementType::Pb;
    element_map["Sn"] = ElementType::Sn;
    element_map["Si"] = ElementType::Si;
    element_map["C"]  = ElementType::C;
    element_map["Sb"] = ElementType::Sb;
    element_map["As"] = ElementType::As;
    element_map["P"]  = ElementType::P;
    element_map["N"]  = ElementType::N;
    element_map["Te"] = ElementType::Te;
    element_map["Se"] = ElementType::Se;
    element_map["S"]  = ElementType::S;
    element_map["O"]  = ElementType::O;
    element_map["I"]  = ElementType::I;
    element_map["Br"] = ElementType::Br;
    element_map["Cl"] = ElementType::Cl;
    element_map["F"]  = ElementType::F;

    // --- ElementType → Frame Priority (Table S7) ---
    // 数值越小 = 优先级越高 (越适合作为帧定义原子)
    atom2frame_priority[ElementType::K]  = 0;
    atom2frame_priority[ElementType::Li] = 1;
    atom2frame_priority[ElementType::Ca] = 2;
    atom2frame_priority[ElementType::Mg] = 3;
    atom2frame_priority[ElementType::Be] = 4;
    atom2frame_priority[ElementType::Y]  = 5;
    atom2frame_priority[ElementType::Tb] = 6;
    atom2frame_priority[ElementType::U]  = 7;
    atom2frame_priority[ElementType::V]  = 8;
    atom2frame_priority[ElementType::W]  = 9;
    atom2frame_priority[ElementType::Mo] = 10;
    atom2frame_priority[ElementType::Cr] = 11;
    atom2frame_priority[ElementType::Re] = 12;
    atom2frame_priority[ElementType::Mn] = 13;
    atom2frame_priority[ElementType::Os] = 14;
    atom2frame_priority[ElementType::Ru] = 15;
    atom2frame_priority[ElementType::Fe] = 16;
    atom2frame_priority[ElementType::Pr] = 17;
    atom2frame_priority[ElementType::Ir] = 18;
    atom2frame_priority[ElementType::Rh] = 19;
    atom2frame_priority[ElementType::Co] = 20;
    atom2frame_priority[ElementType::Pt] = 21;
    atom2frame_priority[ElementType::Pd] = 22;
    atom2frame_priority[ElementType::Ni] = 23;
    atom2frame_priority[ElementType::Au] = 24;
    atom2frame_priority[ElementType::Cu] = 25;
    atom2frame_priority[ElementType::Hg] = 26;
    atom2frame_priority[ElementType::Zn] = 27;
    atom2frame_priority[ElementType::Al] = 28;
    atom2frame_priority[ElementType::B]  = 29;
    atom2frame_priority[ElementType::Pb] = 30;
    atom2frame_priority[ElementType::Sn] = 31;
    atom2frame_priority[ElementType::Si] = 32;
    // 注意: Table S7 中 Si=32 和下一行 36 之间存在不一致，
    // 这里按照连续编号处理: C=33, Sb=34, As=35, P=36, N=37...
    atom2frame_priority[ElementType::C]  = 33;
    atom2frame_priority[ElementType::Sb] = 34;
    atom2frame_priority[ElementType::As] = 35;
    atom2frame_priority[ElementType::P]  = 36;
    atom2frame_priority[ElementType::N]  = 37;
    atom2frame_priority[ElementType::Te] = 38;
    atom2frame_priority[ElementType::Se] = 39;
    atom2frame_priority[ElementType::S]  = 40;
    atom2frame_priority[ElementType::O]  = 41;
    atom2frame_priority[ElementType::I]  = 42;
    atom2frame_priority[ElementType::Br] = 43;
    atom2frame_priority[ElementType::Cl] = 44;
    atom2frame_priority[ElementType::F]  = 45;

    // UNKNOWN 元素优先级最低
    atom2frame_priority[ElementType::UNKNOWN] = 999;
}

// ============================================================================
// MolecularGraph — 图路径搜索
// ============================================================================

void MolecularGraph::dfs_paths(
    int u, int target_len,
    std::vector<int>& current_path,
    std::vector<bool>& visited,
    std::vector<std::vector<int>>& results) const
{
    // 找到一条长度为 target_len 的路径 (包含 target_len+1 个节点)
    if (static_cast<int>(current_path.size()) == target_len + 1) {
        // 去重: 只保留 p[0] < p[-1] 的路径，避免 A-B-C 和 C-B-A 重复
        if (current_path.front() < current_path.back()) {
            results.push_back(current_path);
        }
        return;
    }

    for (const auto& edge : adj_[u]) {
        int v = edge.dst;
        if (!visited[v]) {
            visited[v] = true;
            current_path.push_back(v);
            dfs_paths(v, target_len, current_path, visited, results);
            current_path.pop_back();
            visited[v] = false;
        }
    }
}

std::vector<std::vector<int>> MolecularGraph::find_all_paths_of_length_n(int n) const {
    std::vector<std::vector<int>> results;
    std::vector<int> current_path;
    std::vector<bool> visited(num_atoms_, false);

    for (int start = 0; start < num_atoms_; ++start) {
        visited[start] = true;
        current_path.push_back(start);
        dfs_paths(start, n, current_path, visited, results);
        current_path.pop_back();
        visited[start] = false;
    }

    return results;
}

// ============================================================================
// 内部辅助函数
// ============================================================================

/**
 * @brief 计算一条路径的优先级
 * 
 * 优先级 = 路径中所有原子的 atom2frame_priority 之和，
 * 值越小优先级越高。
 */
static int compute_path_priority(
    const std::vector<int>& path,
    const std::vector<ElementType>& element_types)
{
    int total = 0;
    for (int atom_idx : path) {
        auto it = ChemData::atom2frame_priority.find(element_types[atom_idx]);
        total += (it != ChemData::atom2frame_priority.end())
                 ? it->second
                 : ChemData::atom2frame_priority[ElementType::UNKNOWN];
    }
    return total;
}

/**
 * @brief 计算两个原子之间的残基偏移
 * 
 * @param atom_a 原子 A 的全局索引
 * @param atom_b 原子 B 的全局索引
 * @param residue_ids 每个原子的残基 ID
 * @return 残基偏移量 (atom_b 的残基 - atom_a 的残基)
 */
static int compute_residue_offset(
    int atom_a, int atom_b,
    const std::vector<int>& residue_ids)
{
    return residue_ids[atom_b] - residue_ids[atom_a];
}

/**
 * @brief 计算两个原子之间的原子偏移
 * 
 * 原子偏移 = 同一残基内 atom_b 相对于 atom_a 的偏移
 * 如果不在同一残基内，则返回 0 (简化处理)
 */
static int compute_atom_offset(
    int atom_a, int atom_b,
    const std::vector<int>& residue_ids)
{
    if (residue_ids[atom_a] != residue_ids[atom_b]) {
        return 0; // 跨残基
    }
    // 同一残基内的相对偏移
    // 简单实现: 计算该残基内的起始位置
    int res_id = residue_ids[atom_a];
    int res_start = -1;
    for (size_t i = 0; i < residue_ids.size(); ++i) {
        if (residue_ids[i] == res_id) {
            res_start = static_cast<int>(i);
            break;
        }
    }
    return (atom_b - res_start) - (atom_a - res_start);
}

/**
 * @brief 将 3 原子路径 (A, B, C) 转换为帧偏移量
 * 
 * 帧格式: [A_offset, B_offset, C_offset]
 * 每个偏移量 = (atom_offset, residue_offset)
 */
static std::vector<AtomFrameOffset> path_to_frame_offsets(
    int a, int b, int c,
    const std::vector<int>& residue_ids)
{
    std::vector<AtomFrameOffset> frame(3);

    // B 是中心原子，残基偏移始终为 0
    int res_b = residue_ids[b];

    frame[0] = AtomFrameOffset(
        compute_atom_offset(b, a, residue_ids),
        compute_residue_offset(b, a, residue_ids)
    );

    frame[1] = AtomFrameOffset(0, 0); // B 自身

    frame[2] = AtomFrameOffset(
        compute_atom_offset(b, c, residue_ids),
        compute_residue_offset(b, c, residue_ids)
    );

    return frame;
}

// ============================================================================
// construct_frames — 主函数
// ============================================================================

std::vector<std::vector<AtomFrameOffset>> construct_frames(
    const MolecularGraph& graph,
    const std::vector<ElementType>& element_types,
    const std::vector<int>& residue_ids)
{
    // 确保 ChemData 映射表已初始化
    ChemData::init();

    const int N = graph.num_atoms();

    // 输入验证
    if (N == 0) {
        return {};
    }
    if (static_cast<int>(element_types.size()) != N) {
        throw std::invalid_argument(
            "construct_frames: element_types size (" +
            std::to_string(element_types.size()) +
            ") != graph.num_atoms (" + std::to_string(N) + ")"
        );
    }
    if (static_cast<int>(residue_ids.size()) != N) {
        throw std::invalid_argument(
            "construct_frames: residue_ids size (" +
            std::to_string(residue_ids.size()) +
            ") != graph.num_atoms (" + std::to_string(N) + ")"
        );
    }

    // ========================================================================
    // Step 1: 在分子图中搜索所有长度为 2 的路径 (3-原子路径: A-B-C)
    // ========================================================================
    std::vector<std::vector<int>> all_paths = graph.find_all_paths_of_length_n(2);

    // ========================================================================
    // Step 2: 按路径优先级排序 (Table S7 优先级之和，值越小越优先)
    // ========================================================================
    std::sort(all_paths.begin(), all_paths.end(),
        [&element_types](const std::vector<int>& p1, const std::vector<int>& p2) {
            return compute_path_priority(p1, element_types) <
                   compute_path_priority(p2, element_types);
        }
    );

    // ========================================================================
    // Step 3: 构建帧候选池
    // 
    // frame_candidates[n] = 原子 n 可作为中心原子的帧列表 (按优先级排序)
    // ========================================================================
    std::vector<std::vector<std::vector<AtomFrameOffset>>> frame_candidates(N);

    for (const auto& path : all_paths) {
        // path = [A, B, C], B 是中心原子
        int a = path[0];
        int b = path[1];
        int c = path[2];

        auto frame = path_to_frame_offsets(a, b, c, residue_ids);
        frame_candidates[b].push_back(frame);
    }

    // ========================================================================
    // Step 4: 为每个原子分配帧 (三级回退策略)
    // ========================================================================
    std::vector<std::vector<AtomFrameOffset>> result(N);

    for (int n = 0; n < N; ++n) {
        // --- Level 1: 原子 n 是某帧的中心原子 (n == B) ---
        if (!frame_candidates[n].empty()) {
            // 取优先级最高的帧 (已在 Step 2 排序)
            result[n] = frame_candidates[n][0];
            continue;
        }

        // --- Level 2: 原子 n 参与某帧但不是中心 (n == A 或 n == C) ---
        // 在所有帧候选池中搜索包含原子 n 的帧
        bool found_level2 = false;
        std::vector<AtomFrameOffset> best_level2_frame;
        int best_level2_priority = INT_MAX;

        for (int center = 0; center < N; ++center) {
            for (const auto& frame : frame_candidates[center]) {
                // frame = [A_offset, B_offset, C_offset]
                // 检查 n 是否为 A (offset 对应 center 的 -1 位) 或 C (+1 位)
                // 
                // 简化处理: 在所有 frame 候选中搜索包含 n 的帧
                // frame[0] 表示 A 相对 B 的偏移, frame[2] 表示 C 相对 B 的偏移
                
                // 如果 n 是帧中原子 A 的全局索引
                // 需要根据偏移量反算全局索引 (这里做简化处理)
                // 
                // 更稳健的方式: 直接遍历所有 path 找包含 n 的非中心路径
                (void)frame; // 在 Level 2 的简化实现中暂不使用
            }
        }

        // 更直接的 Level 2 实现: 遍历所有 path
        for (const auto& path : all_paths) {
            int a = path[0];
            int b = path[1];
            int c = path[2];

            if (a == n || c == n) {
                int priority = compute_path_priority(path, element_types);
                if (priority < best_level2_priority) {
                    best_level2_priority = priority;

                    // 将原子 n 转换为以它自身为中心的表示
                    // 如果 n == A，则原帧 B-C 可以用于定义 n 的局部坐标系
                    // 这里输出一个"借用"相邻中心帧的表示
                    if (a == n) {
                        // n 是 A, 借用 B 为中心 → 偏移量为 B 相对 n 的位置
                        best_level2_frame = {
                            AtomFrameOffset(0, 0),                                        // n 自身
                            AtomFrameOffset(
                                compute_atom_offset(n, b, residue_ids),
                                compute_residue_offset(n, b, residue_ids)
                            ),                                                            // B 相对 n
                            AtomFrameOffset(
                                compute_atom_offset(n, c, residue_ids),
                                compute_residue_offset(n, c, residue_ids)
                            )                                                             // C 相对 n
                        };
                    } else { // c == n
                        best_level2_frame = {
                            AtomFrameOffset(
                                compute_atom_offset(n, a, residue_ids),
                                compute_residue_offset(n, a, residue_ids)
                            ),                                                            // A 相对 n
                            AtomFrameOffset(
                                compute_atom_offset(n, b, residue_ids),
                                compute_residue_offset(n, b, residue_ids)
                            ),                                                            // B 相对 n
                            AtomFrameOffset(0, 0)                                         // n 自身
                        };
                    }
                    found_level2 = true;
                }
            }
        }

        if (found_level2) {
            result[n] = best_level2_frame;
            continue;
        }

        // --- Level 3: 完全孤立原子 → 退化帧 [(0,1),(0,1),(0,1)] ---
        // 退化为自身 + 两个虚拟邻居 (无有效化学键信息)
        result[n] = {
            AtomFrameOffset(0, 1),
            AtomFrameOffset(0, 1),
            AtomFrameOffset(0, 1)
        };
    }

    return result;
}

// ============================================================================
// 便捷接口: 从元素符号字符串构造帧
// ============================================================================

std::vector<std::vector<AtomFrameOffset>> construct_frames(
    const MolecularGraph& graph,
    const std::vector<std::string>& element_symbols,
    const std::vector<int>& residue_ids)
{
    ChemData::init();

    const int N = graph.num_atoms();
    if (static_cast<int>(element_symbols.size()) != N) {
        throw std::invalid_argument(
            "construct_frames: element_symbols size (" +
            std::to_string(element_symbols.size()) +
            ") != graph.num_atoms (" + std::to_string(N) + ")"
        );
    }

    std::vector<ElementType> element_types(N);
    for (int i = 0; i < N; ++i) {
        auto it = ChemData::element_map.find(element_symbols[i]);
        element_types[i] = (it != ChemData::element_map.end())
                           ? it->second
                           : ElementType::UNKNOWN;
    }

    return construct_frames(graph, element_types, residue_ids);
}

} // namespace rfaa

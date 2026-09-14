// ============================================================================
// 图版拓扑 pass（两遍管线的 Pass1「轻量版」）—— 2026-09-12
//
// 背景（为什么需要它）：
//   SE3 3D track 的 make_graph 拓扑依赖骨架坐标 coords（kNN），而 coords 又由 SE3 offset 更新，
//   因此「拓扑」与「前向」存在依赖关系。训练里现有三条路径：
//     - fixed（开关A）：所有 block 用**初始** coords 拓扑，只有一遍 → 拓扑最粗但最省；
//     - per_block（开关B）：逐 block 用当步 coords 重构图 + 每块一次子图 compute → 拓扑最准，
//       但每个 block 边界都会在新建空 cgraph 上从 block0 重算整条主干前缀（≈O(N²)）；
//     - pass1：Pass1（值版 forward）逐 block 更新 coords → Pass2 forward_graph 用其 coords 作
//       topo_coords 冻结构图。省掉了 O(N²)，但 Pass1 是**另一份值版实现**（340 行），
//       且值版/图版 dropout mask 各自随机 → 两链 msa/pair 有 ~0.15 相对差（见 PPML.cpp:3061）。
//
// 本文件 = pass1 的「图版 Pass1」：把 Pass1 从值版实现换成**图版 forward_graph 跑一遍**：
//   - fixed/pass1 两条路径在 forward_graph 内部都是「收集各 block 的 offset 图节点 → 统一
//     build_forward_expand + graph_compute 物化 → 读 offset 值 → apply_coord_update 链式更新
//     current_coords」（PPML.cpp:3398-3466），并且 go.coords 就是这份更新后的坐标值
//     （PPML.cpp:3598-3612）。⇒ 直接取 go.coords 即可，无需新造物化机制。
//   - 于是 Pass1 与 Pass2 共用同一套图版算子与权重：无重复实现、无 dropout 两链不一致。
//
// 与 per_block 的**已知差异**（不是 bug，是取舍）：本 pass 用冻结（初始）拓扑推进 coords，
//   不做 per-block 拓扑更新。实测逐块拓扑漂移（Experiment.md，2026-09-12）：
//     L=51（top_k=64 ≥ L-1 ⇒ 完全图）拓扑恒不变 ⇒ 无差异；
//     L=103（真 kNN）首块与最终结构约 30% 有向边不同、稳态约 2~3%/块 —— 故该近似的影响需实测。
//
// 用法（train.cpp 的 PPML_SE3_TOPO=pass1 分支，PPML_TOPO_PASS=graph 时调用）：
//   TensorF32 c = model.topo_pass(input);          // Pass1（图版）
//   GraphOutput g = model.forward_graph(input, true, &c);   // Pass2（冻结拓扑，单次 compute 反传）
// 环境变量：
//   PPML_TOPO_PASS_ITERS=n（默认 1）：Pass1 迭代轮数；第 1 轮 fixed，其后轮用上一轮 coords 作
//     topo_coords（拓扑逐步逼近收敛结构）。每轮 ≈ 1 遍图版构图 + 一次 SE3 统一 compute，慎用大 n。
// ============================================================================
#include "ppml/Model.h"
#include "ppml/Context.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace ppml {

TensorF32 PPMLModel::topo_pass(const ModelInput& input) {
    // ---- 保存/恢复 PPML_SE3_TOPO：本 pass 需要 fixed/pass1 分支（走统一 compute + coords 链）----
    const char* saved = std::getenv("PPML_SE3_TOPO");
    const std::string saved_s = saved ? saved : "";

    int iters = 1;
    if (const char* s = std::getenv("PPML_TOPO_PASS_ITERS")) {
        const int v = std::atoi(s);
        if (v > 0) iters = v;
    }

    TensorF32 coords_out;               // 上一轮 coords（第 1 轮为空 → forward_graph 用 input.coords）
    const TensorF32* topo_ptr = nullptr;
    const int64_t want = input.coords.numel();

    for (int it = 0; it < iters; ++it) {
        // 第 1 轮：fixed（无外部拓扑坐标）；后续轮：pass1 + 上一轮 coords（冻结拓扑再推进一轮）
        const char* mode = (it == 0) ? "fixed" : "pass1";
        setenv("PPML_SE3_TOPO", mode, 1);

        const auto t0 = std::chrono::high_resolution_clock::now();
        GraphOutput g = forward_graph(input, /*enable_se3=*/true, topo_ptr);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        const bool ok = (g.coords.numel() > 0 && g.coords.data() != nullptr);
        if (ok) {
            // 取值（own-memory 拷贝）：跨 graph_compute 存活，arena buffer 释放后仍有效
            coords_out.~TensorF32();
            new (&coords_out) TensorF32(g.coords.shape(), Device::CPU);
            coords_out.copy_from(g.coords);
            topo_ptr = &coords_out;
        }
        std::cout << "[TOPO-PASS] iter " << (it + 1) << "/" << iters << " mode=" << mode
                  << " " << ms << "ms coords numel=" << (ok ? coords_out.numel() : 0)
                  << (ok ? " OK" : " EMPTY") << std::endl;
        if (!ok) break;                 // 未驱动 SE3 / 拓扑空 → 返回空（调用方回落初始 coords）
        if (want > 0 && coords_out.numel() != want) {
            // 形状与输入坐标不一致（理论上不会发生）→ 放弃，避免把坏 coords 传下去
            std::fprintf(stderr, "[TOPO-PASS][WARN] coords numel=%lld != input numel=%lld, discard\n",
                         (long long)coords_out.numel(), (long long)want);
            coords_out.~TensorF32();
            new (&coords_out) TensorF32();
            topo_ptr = nullptr;
            break;
        }
    }

    // 恢复环境变量，避免污染后续 Pass2 的模式判定（Pass2 由 train.cpp 显式设为 pass1）
    if (saved_s.empty()) unsetenv("PPML_SE3_TOPO");
    else                 setenv("PPML_SE3_TOPO", saved_s.c_str(), 1);
    return coords_out;
}

} // namespace ppml

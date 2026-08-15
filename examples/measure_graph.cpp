// ============================================================================
// examples/measure_graph.cpp — 内存占用测量（只构建图 + 统计，不执行 compute）
//
// 目的: 在 no_alloc + work buffer 改造前，定量回答:
//   1. eager 全量: 若所有中间张量同时常驻（当前 eager 模式），共占多少字节
//      (= 为什么 1GB context 在第一个 Linear 就爆)。
//   2. no_alloc 峰值: 若用「引用计数生命周期复用」的 work buffer，单次
//      graph_compute 同时存活张量的峰值是多少（= 改造后 work buffer 应开多大）。
//   3. 常驻内存: 模型参数 + 输入数据（不参与 work buffer 复用）。
//
// 用法: 与 train.cpp 相同参数 (a3m fasta csv|dir [template_dir] [hhr])。
// 注意: 本程序只 build_forward_expand + build_backward_expand，不调 graph_compute，
//       故不会真正执行数值计算；构建期中间张量只分配虚拟页（bump allocator），
//       物理内存只承担被写页（输入/参数/常量），WSL 15GB 足够。
// ============================================================================
#include "ppml/Model.h"
#include "ppml/DataLoader.h"
#include "ppml/ComputeGraph.h"
#include "ppml/Context.h"
#include "ppml/PythonBridge.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

using namespace ppml;

// 将值张量包装为图节点 leaf（图模式 forward_graph 用）
TensorF32* wrap_value_as_leaf(const TensorF32& t, const std::vector<int64_t>& dims) {
    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < dims.size() && i < 4; i++) ne[i] = dims[i];
    TensorF32* leaf = context().new_tensor<float>(static_cast<int>(dims.size()), ne);
    if (t.device() == Device::CUDA) {
        TensorF32 tcpu = t.cpu();
        std::memcpy(leaf->data(), tcpu.data(), tcpu.numel() * sizeof(float));
    } else {
        std::memcpy(leaf->data(), t.data(), t.numel() * sizeof(float));
    }
    return leaf;
}

// 张量分配大小（view 张量不单独分配，返回 0）
static size_t alloc_size_of(const TensorF32* t) {
    if (!t) return 0;
    if (t->view_src != nullptr) return 0;          // view 复用源张量空间
    return t->nbytes();
}

// 判断是否常驻 leaf（op==NONE 的输入/参数/常量；不参与 work buffer 复用）
static bool is_persistent_leaf(const TensorF32* t) {
    return t && t->op == OP_NONE;
}

// ============================================================================
// 引用计数生命周期模拟: 得到 work buffer 的理论峰值（no_alloc 改造后）
//   graph.nodes 已按依赖拓扑序排列（前向在前，反向 grad 在后）。
//   遍历时: 一个张量被计算 = 它开始存活；其所有下游依赖都计算完 = 可释放。
// ============================================================================
static void simulate_peak(ComputeGraph* g,
                          size_t* work_peak, size_t* work_total,
                          size_t* persistent_total) {
    const int n_nodes = g->n_nodes();
    const int n_leafs = g->n_leafs();

    // 收集常驻 leaf 指针集（用于判断 src 是否为常驻）
    std::unordered_set<TensorF32*> leaf_set;
    leaf_set.reserve(n_leafs);
    size_t persistent = 0;
    for (int i = 0; i < n_leafs; ++i) {
        TensorF32* l = g->graph_leaf(i);
        leaf_set.insert(l);
        if (is_persistent_leaf(l) && alloc_size_of(l) > 0) {
            persistent += alloc_size_of(l);
        }
    }

    // refcount: 每个中间 node 被后续多少个 node 作为 src 引用
    std::unordered_map<TensorF32*, size_t> refcount;
    for (int i = 0; i < n_nodes; ++i) {
        TensorF32* node = g->graph_node(i);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            TensorF32* src = node->src[s];
            if (src && leaf_set.count(src) == 0) {  // 只统计中间张量（非常驻 leaf）
                refcount[src]++;
            }
        }
    }

    // 按拓扑序模拟
    size_t live = 0, peak = 0, total = 0;
    for (int i = 0; i < n_nodes; ++i) {
        TensorF32* node = g->graph_node(i);
        if (leaf_set.count(node)) continue;   // 常驻 leaf 不算 work buffer

        size_t sz = alloc_size_of(node);
        live += sz;          // 计算它 = 开始存活
        total += sz;         // 全量总和（eager 级别）
        peak = std::max(peak, live);

        // 释放不再被依赖的 src
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            TensorF32* src = node->src[s];
            if (!src || leaf_set.count(src)) continue;
            auto it = refcount.find(src);
            if (it == refcount.end()) continue;
            if (--it->second == 0) {
                live -= alloc_size_of(src);
            }
        }
    }

    *work_peak       = peak;
    *work_total      = total;
    *persistent_total = persistent;
}

int main(int argc, char* argv[]) {
    std::cerr << "[measure] step0: start\n";
    auto& py = PythonBridge::instance();
    if (!py.initialize()) {
        std::cerr << "Failed to initialize Python" << std::endl;
        return 1;
    }
    std::cerr << "[measure] step1: python initialized\n";

    PPMLConfig config;
    config.d_msa = 256;
    config.d_pair = 128;
    config.d_state = 32;
    config.n_extra_blocks = 4;
    config.n_main_blocks = 8;
    config.n_refine_blocks = 4;

    std::cerr << "[measure] step2: config built\n";
    std::cerr << "[measure] step2b: constructing PPMLModel...\n" << std::flush;
    PPMLModel model(config);
    std::cerr << "[measure] step2c: model constructed\n" << std::flush;
    model.to(Device::CUDA);    // 与 train.cpp 一致（backend 初始化路径相同）
    std::cerr << "[measure] step2d: model.to done\n" << std::flush;
    model.train();
    std::cerr << "[measure] step3: model built\n";

    std::string a3m_path     = (argc > 1) ? argv[1] : "query.a3m";
    std::string fasta_path   = (argc > 2) ? argv[2] : "data/P04637.fasta";
    std::string csv_path     = (argc > 3) ? argv[3] : "data/P04637_pdbs";
    std::string template_dir = (argc > 4) ? argv[4] : "data/P04637_template_coords";
    std::string hhr_path     = (argc > 5) ? argv[5] : "";

    std::string sequence;
    try {
        sequence = PPMLDataLoader::read_fasta_first_sequence(fasta_path);
        std::cout << "Read query sequence (L=" << sequence.length() << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "Failed to read FASTA: " << e.what() << std::endl;
        return 1;
    }

    PPMLDataLoader loader("", "", 512, 4, 2048);
    ModelInput input = loader.load_from_files(a3m_path, sequence, csv_path, template_dir, hhr_path);
    int L = static_cast<int>(sequence.length());

    if (input.coords.numel() == 0)       input.coords = zeros<float>({1, L, 3, 3}, Device::CPU);
    if (input.true_coords.numel() == 0)  input.true_coords = zeros<float>({1, L, 3, 3}, Device::CPU);
    std::cerr << "[measure] step4: data loaded, L=" << L << "\n";

    std::cout << "=== Building forward graph ...\n";
    auto go = model.forward_graph(input);
    std::cout << "forward_graph done.\n";
    std::cerr << "[measure] step5: forward_graph done\n";

    // ---- 组装 total_loss（与 train.cpp 相同，简化：只走 FAPE + MSA，其余可复现）----
    const int B = 1;
    const int N_atoms  = B * L * 3;
    const int N_frames = B * L;

    TensorF32 pred_flat = go.coords.view({N_atoms, 3});
    TensorF32 true_flat = input.true_coords.view({N_atoms, 3});
    TensorF32* pred_node = wrap_value_as_leaf(pred_flat, {N_atoms, 3});
    TensorF32* true_node = wrap_value_as_leaf(true_flat, {N_atoms, 3});

    TensorF32* frame_idx = build_frame_atom_indices(B, L);
    TensorF32* frames_mask    = constant_ones({N_frames});
    TensorF32* positions_mask = constant_ones({N_atoms});

    FAPEConfig fape_cfg;
    fape_cfg.length_scale = 10.0f;
    fape_cfg.d_clamp      = 10.0f;
    fape_cfg.epsilon      = 1e-4f;
    TensorF32* loss_fape_node =
        loss(fape_loss(pred_node, true_node, frame_idx, frames_mask, positions_mask, fape_cfg));

    TensorF32* loss_msa_node = loss(constant_scalar(0.0f));
    if (input.true_msa.numel() > 0 && go.msa_logits != nullptr) {
        int N_seq_msa = static_cast<int>(input.true_msa.shape().dims[1]);
        TensorF32 msa_true_2d = input.true_msa.view({N_seq_msa, L});
        TensorF32 msa_mask_2d = input.bert_mask.view({N_seq_msa, L});
        TensorF32* logits_node = view(go.msa_logits, Shape{23, L, N_seq_msa});
        TensorF32* true_n = wrap_value_as_leaf(msa_true_2d, {L, N_seq_msa});
        TensorF32* mask_n = wrap_value_as_leaf(msa_mask_2d, {L, N_seq_msa});
        loss_msa_node = loss(masked_msa_loss(logits_node, true_n, mask_n));
    }

    TensorF32* total_node = loss(total_loss(loss_fape_node, loss_msa_node,
                                            loss(constant_scalar(0.0f)),
                                            loss(constant_scalar(0.0f)),
                                            loss(constant_scalar(0.0f))));

    // ---- 构建完整前向 + 反向图（不执行）----
    PPMLContext* ctx = &context();
    ComputeGraph* cgraph = ComputeGraph::new_graph(ctx);
    std::cout << "=== build_forward_expand ...\n";
    cgraph->build_forward_expand(total_node);
    std::cout << "forward expand: nodes=" << cgraph->n_nodes() << " leafs=" << cgraph->n_leafs() << "\n";
    std::cout << "=== build_backward_expand ...\n";
    cgraph->build_backward_expand(ctx, nullptr);
    std::cout << "backward expand: nodes=" << cgraph->n_nodes() << " leafs=" << cgraph->n_leafs() << "\n";

    // ---- 统计 ----
    // (a) eager 全量总和: 所有 node + leaf 的 nbytes 之和
    size_t eager_all_nodes = 0;
    for (int i = 0; i < cgraph->n_nodes(); ++i) eager_all_nodes += cgraph->graph_node(i)->nbytes();
    size_t eager_all_leafs = 0;
    for (int i = 0; i < cgraph->n_leafs(); ++i) eager_all_leafs += cgraph->graph_leaf(i)->nbytes();

    // (b) no_alloc 峰值模拟
    size_t work_peak = 0, work_total = 0, persistent = 0;
    simulate_peak(cgraph, &work_peak, &work_total, &persistent);

    auto mb = [](size_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
    std::cout << "\n=========== 内存测量结果 ===========\n";
    std::cout << "L=" << L << "  config: n_extra=" << config.n_extra_blocks
              << " n_main=" << config.n_main_blocks << " n_refine=" << config.n_refine_blocks << "\n";
    std::cout << "graph nodes=" << cgraph->n_nodes() << "  leafs=" << cgraph->n_leafs() << "\n\n";

    std::cout << "[eager 全量] 所有 node 中间张量之和:  " << mb(eager_all_nodes) << " MB\n";
    std::cout << "[eager 全量] 所有 leaf(输入/参数/常量)之和: " << mb(eager_all_leafs) << " MB\n";
    std::cout << "[eager 全量] 总计 (当前模式内存上限):   " << mb(eager_all_nodes + eager_all_leafs) << " MB\n\n";

    std::cout << "[常驻]       参数+输入+常量 (不回收):   " << mb(persistent) << " MB\n";
    std::cout << "[no_alloc]   work buffer 峰值(模拟):    " << mb(work_peak) << " MB\n";
    std::cout << "[no_alloc]   work buffer 全量(理论):    " << mb(work_total) << " MB\n";
    std::cout << "\n[no_alloc 改造后总峰值 ≈ 常驻 + work 峰值] ≈ " << mb(persistent + work_peak) << " MB\n";

    py.finalize();
    return 0;
}

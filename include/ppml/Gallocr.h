#pragma once
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Tensor.h"
#include "DynTalloc.h"

namespace ppml {

class Buffer;
class BufferType;
class ComputeGraph;

// ============================================================
// Gallocr — 对标 ggml_gallocr：延迟内存分配 + 空间复用
//
// 为一张图做延迟分配：tensor 沿拓扑序"借/还"同一个 backend buffer，
// 使同时存活的数据仅占峰值（空间复用），替代 reserve_graph_memory 的全量常驻。
// 每个后端独立维护一个 DynTalloc（自由块 best-fit + 相邻合并）。
//
// 注意：当前 CPU/CUDA kernel 均写入 result->data()、读取 src[i]->data()，
//       不识别 data 别名，故不支持 ggml 的 inplace 复用（会读坏源）。
//       本实现仅做"生命周期空间复用"（等价 simulate_peak 的 refcount 模型）。
//
// 两阶段：
//   Phase 1 (reserve)：模拟分配，计算每个后端所需峰值大小（不分配 buffer）。
//   Phase 2 (alloc)  ：按峰值分配真实 buffer + DynTalloc，再模拟一遍绑定
//                     每个张量的 data_/buffer_/buffer_offs_。
// ============================================================
class Gallocr {
public:
    // 张量分配信息
    struct NodeInfo {
        TensorF32* tensor     = nullptr;
        int        n_children = 0;   // 被多少个计算节点作为 src 引用
        bool       allocated  = false;
        bool       managed    = false;  // true: 参与 DynTalloc 复用；false: 持久（已分配/参数）
        bool       is_output  = false;  // OUTPUT 张量永不复用（本实现暂不依赖）
        size_t     offset     = 0;
        size_t     alloc_size = 0;   // 对齐后的分配大小（GGML_PAD 后）
        Buffer*    buffer     = nullptr;
        int        backend_id = 0;
        // ---- needs_realloc 接入（2026-08-31，移植 ggml tensor_alloc 语义）----
        // 持久化"上次预留"记录：size_max=上次分配的 alloc_size（0=外部/view），
        // buffer_id=-1 表示该张量上次无需 galloc 分配（外部数据/预分配/view）。
        // 供 needs_realloc() 在跨 graph_compute 时判断布局是否仍有效。
        int64_t    size_max   = 0;
        int        buffer_id  = -1;
    };

    // 单后端分配器（一个 backend 对应一个 DynTalloc）
    struct LiveRange { size_t off; size_t size; };
    struct BackendAlloc {
        BufferType* buft   = nullptr;
        size_t      peak   = 0;                 // Phase1 计算的峰值（对齐后，= 最大同时存活字节数）
        size_t      high_watermark = 0;         // Phase1 的 max(offset+alloc_size)（buffer 需 ≥ 此值）
        DynTalloc*  talloc = nullptr;           // Phase1 用；Phase2 不用于 offset 决策
        std::vector<Buffer*> buffers;           // Phase2 实际分配的 buffer（本类持有所有权）
        std::vector<LiveRange> live;            // Phase2 当前存活区间（方向3 overlap 检查用）
        bool        use    = false;
    };

    Gallocr() = default;
    ~Gallocr();
    Gallocr(const Gallocr&) = delete;
    Gallocr& operator=(const Gallocr&) = delete;

    // Phase 1：计算各后端峰值（不分配 buffer）。返回 false 表示失败。
    //   graph:          待分配图
    //   backend_id_of:  张量 → 后端 id
    //   n_backends:     后端数量
    bool reserve(ComputeGraph* graph,
                 const std::function<int(TensorF32*)>& backend_id_of,
                 int n_backends);

    // Phase 2：按峰值分配 buffer 并绑定张量 data_/buffer_/buffer_offs_。
    //   调用前必须已 reserve()。返回 false 表示失败。
    //   2026-08-31（needs_realloc 接入）：内部先判 has_snapshot_ && !needs_realloc(graph)
    //   → 走复用路径 alloc_reuse()（不重建 buffer，只重绑 data，对齐 GGML alloc_graph）；
    //   否则走重建路径 alloc_rebuild()（原 alloc 实现，保留可回退）。
    bool alloc(ComputeGraph* graph,
               const std::function<int(TensorF32*)>& backend_id_of,
               int n_backends);

    // ---- needs_realloc 接入（2026-08-31，移植 ggml_gallocr_needs_realloc 语义）----
    // 上次 reserve 是否留有有效快照（reserve 成功置 true，release/reset_state 置 false）
    bool has_snapshot() const { return has_snapshot_; }
    // 判断当前 graph 是否能复用上次 reserve 的布局（buffer 无需重建）。
    // 对齐 GGML：n_nodes/n_leafs 变化、或任一节点 dst/src 预留记录失效
    // （buffer_id<0 但现需分配；或 size_max < 当前所需大小）→ 需要重分配。
    // 注意：调用时要求 buffer 仍存活（调用方须在 release 前判断）。
    bool needs_realloc(ComputeGraph* graph);
    // 统一复用判定：已实际分配过 buffer（allocated_）且布局未变（has_snapshot_ &&
    // !needs_realloc）。reserve 仅模拟不分配 buffer，首次 alloc 必须走重建路径。
    bool can_reuse(ComputeGraph* graph) {
        return allocated_ && has_snapshot_ && !needs_realloc(graph);
    }

    // 设置后端数量（分配 backends_ 槽位）。在设置 buft / 调用 reserve 前调用。
    void set_n_backends(int n) { backends_.resize(n); }

    // 注入"跨后端 producer"保活集合（2026-09-06）：build_splits 把消费者 node->src[j]
    //   rewired 到 OP_DUP cpy（cpy 不入图 nodes）→ gallocr 统计不到 producer 的跨后端
    //   消费 → refcount 被低估 → producer 算完即 free、空间被后续节点复用覆盖 →
    //   graph_compute Step1 D2H 读到垃圾 → 混合 loss 爆炸。注入后这些张量在
    //   compute_refcounts 中被标 is_output（不释放、不复用）。
    //   每次 reserve 前由调用方（BackendScheduler::reserve_graph_memory）重新设置。
    void set_pinned_srcs(const std::vector<const TensorF32*>& srcs);

    // 释放所有 backend buffer（与 reserve/alloc 配套）
    void release();

    // ============ 诊断（GRAPH_DEBUG_GALLOCR 时输出）============
    // 打印所有与 target 在 Phase2 中 offset 区间重叠的 managed 节点。
    // 用于排查 pred_coords 等叶子/输出被其它节点 buffer 复用覆盖的问题。
    // 返回重叠节点数。
    int diagnose_aliasing(TensorF32* target);

    // 后端峰值（reserve 后有效）
    size_t backend_peak(int b) const;
    size_t n_backends() const { return backends_.size(); }

    const std::vector<BackendAlloc>& backends() const { return backends_; }
    std::vector<BackendAlloc>&       backends()       { return backends_; }

private:
    void reset_state(int n_backends);
    void compute_refcounts(ComputeGraph* graph,
                           const std::function<int(TensorF32*)>& backend_id_of);

    // 分配一个张量（在其后端 talloc 上借空间）
    bool allocate_node(NodeInfo* ni);
    // 释放一个张量（还空间），供其后端 talloc 复用
    void free_node(NodeInfo* ni);
    // 把分配结果（offset/buffer）写回张量 data_/buffer_/buffer_offs_
    void bind_tensor(NodeInfo* ni);
    // 方向3：Phase2 分配时检查新区间是否与当前存活区间重叠（应恒不重叠）
    void check_live_overlap(BackendAlloc& ba, const NodeInfo* ni);
    void add_live(BackendAlloc& ba, size_t off, size_t size);
    void remove_live(BackendAlloc& ba, size_t off, size_t size);

    // ---- needs_realloc 接入（2026-08-31）----
    // 单张量预留记录（对标 ggml tensor_alloc）
    struct TensorAllocSnap {
        int    buffer_id = -1;   // -1: 外部/预分配/view（无需 galloc 分配）
        size_t size_max  = 0;    // 上次预留大小（0: 外部/view）
        size_t offset    = 0;    // 上次分配偏移（buffer 内）
    };
    struct NodeAllocSnap {
        TensorAllocSnap dst;
        TensorAllocSnap src[GGML_MAX_SRC];
    };
    struct LeafAllocSnap {
        TensorAllocSnap dst;
    };
    // 单张量预留记录是否仍有效（对标 ggml_gallocr_node_needs_realloc）
    bool node_alloc_valid(TensorF32* t, const TensorAllocSnap& a);
    // 复用路径：布局未变时用快照直接重绑 data（不重建 buffer，对齐 GGML alloc_graph）
    bool alloc_reuse(ComputeGraph* graph);
    // 重建路径：原 alloc 完整实现（2026-08-31 移出保留，可回退）
    bool alloc_rebuild(ComputeGraph* graph,
                       const std::function<int(TensorF32*)>& backend_id_of,
                       int n_backends);

    // 上次 reserve 成功后持久化的预留快照（独立于 nodes_/leaves_，
    // release()/reset_state() 不清除数组，仅 has_snapshot_=false 使其失效）
    std::vector<NodeAllocSnap> node_allocs_;
    std::vector<LeafAllocSnap> leaf_allocs_;
    int  n_nodes_snap_ = -1;
    int  n_leafs_snap_ = -1;
    bool has_snapshot_ = false;
    // 实际分配过 buffer（alloc_rebuild 成功置 true；release/reset_state 置 false）。
    // reserve 仅模拟（Phase1 不分配 buffer），故首次 alloc 前 buffer 为空，
    // 必须走重建路径分配；后续同布局 alloc 才允许走复用路径。
    bool allocated_     = false;

    // Phase1（reserve）记录的每个 managed 张量偏移。Phase2 直接复用该偏移，
    // 使两阶段布局完全一致（消除 best-fit 因 buffer 总大小不同导致的分叉）。
    std::unordered_map<const TensorF32*, size_t> phase1_offset_;
    bool recording_phase1_ = false;

    std::vector<BackendAlloc> backends_;
    std::unordered_map<const TensorF32*, NodeInfo*> node_map_;
    std::vector<NodeInfo> nodes_;    // 与 graph->nodes 对应
    std::vector<NodeInfo> leaves_;   // 与 graph->leafs 对应
    // 外部注入的跨后端 producer（见 set_pinned_srcs 注释），compute_refcounts 末尾应用
    std::unordered_set<const TensorF32*> pinned_srcs_;
};

} // namespace ppml

#pragma once
#include <cstddef>
#include <functional>
#include <unordered_map>
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
        Buffer*    buffer     = nullptr;
        int        backend_id = 0;
    };

    // 单后端分配器（一个 backend 对应一个 DynTalloc）
    struct BackendAlloc {
        BufferType* buft   = nullptr;
        size_t      peak   = 0;                 // Phase1 计算的峰值（对齐后）
        DynTalloc*  talloc = nullptr;           // Phase2
        std::vector<Buffer*> buffers;           // Phase2 实际分配的 buffer（本类持有所有权）
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
    bool alloc(ComputeGraph* graph,
               const std::function<int(TensorF32*)>& backend_id_of,
               int n_backends);

    // 设置后端数量（分配 backends_ 槽位）。在设置 buft / 调用 reserve 前调用。
    void set_n_backends(int n) { backends_.resize(n); }

    // 释放所有 backend buffer（与 reserve/alloc 配套）
    void release();

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

    std::vector<BackendAlloc> backends_;
    std::unordered_map<const TensorF32*, NodeInfo*> node_map_;
    std::vector<NodeInfo> nodes_;    // 与 graph->nodes 对应
    std::vector<NodeInfo> leaves_;   // 与 graph->leafs 对应
};

} // namespace ppml

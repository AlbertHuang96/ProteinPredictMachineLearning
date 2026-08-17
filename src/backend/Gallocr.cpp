#include "ppml/Gallocr.h"
#include "ppml/ComputeGraph.h"
#include "ppml/Backend.h"   // alloc_buffer, Buffer, BufferType

#include <algorithm>
#include <limits>

namespace ppml {

#define GGML_PAD(x, n) (((x) + ((n)-1)) & ~((n)-1))

Gallocr::~Gallocr() {
    release();
}

void Gallocr::reset_state(int n_backends) {
    // 保留调用方注入的 buft（set_n_backends 时已设置）
    std::vector<BufferType*> saved_buft(n_backends, nullptr);
    size_t old_n = backends_.size();
    for (size_t b = 0; b < old_n && b < (size_t)n_backends; b++) {
        saved_buft[b] = backends_[b].buft;
    }
    backends_.clear();
    backends_.resize(n_backends);
    for (int b = 0; b < n_backends; b++) {
        backends_[b].buft = saved_buft[b];
    }
    node_map_.clear();
    nodes_.clear();
    leaves_.clear();
}

size_t Gallocr::backend_peak(int b) const {
    if (b < 0 || b >= (int)backends_.size()) return 0;
    return backends_[b].peak;
}

// ============================================================
// compute_refcounts — 统计每个张量被多少个计算节点引用（作为 src）
//   只统计 managed（需复用分配）的张量，以便其最后一个消费者结束后释放。
// ============================================================
void Gallocr::compute_refcounts(
    ComputeGraph* graph,
    const std::function<int(TensorF32*)>& backend_id_of) {

    node_map_.clear();
    nodes_.resize(graph->n_nodes());
    leaves_.resize(graph->n_leafs());

    // 登记 nodes
    for (int i = 0; i < graph->n_nodes(); i++) {
        TensorF32* t = graph->graph_node(i);
        NodeInfo& ni = nodes_[i];
        ni.tensor     = t;
        ni.managed    = (t->data() == nullptr && t->view_src == nullptr);
        ni.backend_id = backend_id_of(t);
        ni.is_output  = (t->flag & TENSOR_FLAG_OUTPUT) != 0;
        node_map_[t]  = &ni;
    }
    // 登记 leafs
    for (int i = 0; i < graph->n_leafs(); i++) {
        TensorF32* t = graph->graph_leaf(i);
        NodeInfo& li = leaves_[i];
        li.tensor     = t;
        li.managed    = (t->data() == nullptr && t->view_src == nullptr);
        li.backend_id = backend_id_of(t);
        li.is_output  = (t->flag & TENSOR_FLAG_OUTPUT) != 0;
        node_map_[t]  = &li;
    }

    // 统计 refcount：遍历 nodes，对每个 src 若为 managed 则 +1
    for (int i = 0; i < graph->n_nodes(); i++) {
        TensorF32* node = graph->graph_node(i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            TensorF32* src = node->src[s];
            if (!src) continue;
            auto it = node_map_.find(src);
            if (it == node_map_.end() || !it->second->managed) continue;
            it->second->n_children++;
        }
    }
}

// ============================================================
// allocate_node — 在其后端 talloc 上分配
// ============================================================
bool Gallocr::allocate_node(NodeInfo* ni) {
    if (!ni->managed || ni->allocated) return true;
    if (ni->backend_id < 0 || ni->backend_id >= (int)backends_.size()) return false;

    BackendAlloc& ba = backends_[ni->backend_id];
    TensorF32*    t  = ni->tensor;

    size_t alloc_size = GGML_PAD(ba.buft->get_alloc_size(t), ba.buft->get_alignment());
    size_t offset = 0;
    if (!ba.talloc->alloc(alloc_size, offset)) {
        return false;  // 后端 buffer 空间不足（Phase2 峰值计算应已保证足够）
    }

    ni->offset    = offset;
    ni->buffer    = ba.buffers.empty() ? nullptr : ba.buffers[0];
    ni->allocated = true;

    // 峰值跟踪：Phase1 时 talloc 是"虚拟大空间"，used_bytes() 即当前存活字节数
    size_t live = ba.talloc->used_bytes();
    ba.peak = std::max(ba.peak, live);
    return true;
}

// ============================================================
// free_node — 释放（还空间），使后继张量可复用
// ============================================================
void Gallocr::free_node(NodeInfo* ni) {
    if (!ni->allocated) return;
    if (ni->is_output) return;  // OUTPUT 永不复用（本实现保守：也不释放）

    BackendAlloc& ba = backends_[ni->backend_id];
    TensorF32*    t  = ni->tensor;
    size_t alloc_size = GGML_PAD(ba.buft->get_alloc_size(t), ba.buft->get_alignment());
    ba.talloc->free_bytes(ni->offset, alloc_size);

    ni->allocated = false;
}

// ============================================================
// reserve — Phase 1：模拟分配，计算各后端峰值
//   用"虚拟大空间"的 DynTalloc 跑一遍借/还，跟踪 used_bytes 的峰值。
// ============================================================
bool Gallocr::reserve(
    ComputeGraph* graph,
    const std::function<int(TensorF32*)>& backend_id_of,
    int n_backends) {

    reset_state(n_backends);
    if (n_backends == 0) return false;

    // buft 由调用方在调用 reserve 前通过 backends()[b].buft 注入
    for (auto& ba : backends_) {
        if (!ba.buft) return false;
    }

    compute_refcounts(graph, backend_id_of);

    // Phase1 模拟：每个后端用虚拟大空间 DynTalloc（仅追踪偏移/大小，不占用真实内存）
    for (auto& ba : backends_) {
        ba.talloc = new DynTalloc(std::numeric_limits<size_t>::max() / 2, ba.buft->get_alignment());
    }

    // 3. 先分配 managed leaves（输入/常量，若需分配）
    for (auto& li : leaves_) {
        if (li.managed && !allocate_node(&li)) { release(); return false; }
    }

    // 4. 遍历 nodes（拓扑序）：分配当前节点，释放已无依赖的 src
    for (auto& ni : nodes_) {
        TensorF32* node = ni.tensor;
        // 先分配本节点
        if (ni.managed && !allocate_node(&ni)) { release(); return false; }

        // 释放 src：该节点被消费后，其依赖的 src 引用计数减一
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            TensorF32* src = node->src[s];
            if (!src) continue;
            auto it = node_map_.find(src);
            if (it == node_map_.end() || !it->second->managed) continue;
            NodeInfo* sni = it->second;
            if (--sni->n_children <= 0) {
                free_node(sni);
            }
        }
    }

    // 5. 保留每个后端的峰值，释放 Phase1 虚拟 talloc（Phase2 会重建真实 buffer）
    for (auto& ba : backends_) {
        ba.use = ba.peak > 0;
        delete ba.talloc;
        ba.talloc = nullptr;
    }
    return true;
}

// ============================================================
// alloc — Phase 2：按峰值分配真实 buffer，重新模拟并绑定 data_
// ============================================================
bool Gallocr::alloc(
    ComputeGraph* graph,
    const std::function<int(TensorF32*)>& backend_id_of,
    int n_backends) {

    if ((int)backends_.size() != n_backends) {
        // 未 reserve 或后端数变化 → 先 reserve
        if (!reserve(graph, backend_id_of, n_backends)) return false;
    }

    // 1. 为每个后端分配 peak 大小的 buffer，并创建真实 DynTalloc
    for (auto& ba : backends_) {
        if (!ba.use) continue;
        if (!ba.buft) continue;

        size_t size = ba.peak;
        if (size > 0) {
            Buffer* buf = alloc_buffer(ba.buft, size);
            if (!buf) { release(); return false; }
            ba.buffers.push_back(buf);
        }
        ba.talloc = new DynTalloc(size, ba.buft->get_alignment());
    }

    // 2. 重置 refcount（Phase1 已把 n_children 减到 0），重新统计
    compute_refcounts(graph, backend_id_of);

    // 3. 分配 managed leaves 并绑定
    for (auto& li : leaves_) {
        if (!li.managed) continue;
        if (!allocate_node(&li)) { release(); return false; }
        bind_tensor(&li);
    }

    // 4. 遍历 nodes，分配并绑定
    for (auto& ni : nodes_) {
        TensorF32* node = ni.tensor;
        if (ni.managed && !allocate_node(&ni)) { release(); return false; }
        if (ni.managed) bind_tensor(&ni);

        for (int s = 0; s < GGML_MAX_SRC; s++) {
            TensorF32* src = node->src[s];
            if (!src) continue;
            auto it = node_map_.find(src);
            if (it == node_map_.end() || !it->second->managed) continue;
            NodeInfo* sni = it->second;
            if (--sni->n_children <= 0) {
                free_node(sni);
            }
        }
    }

    return true;
}

// ============================================================
// bind_tensor — 把分配结果写回张量
// ============================================================
void Gallocr::bind_tensor(NodeInfo* ni) {
    if (!ni->allocated) return;
    TensorF32* t  = ni->tensor;
    Buffer*    b  = ni->buffer;
    if (!b) {
        // 从后端 buffers 中取第一个
        BackendAlloc& ba = backends_[ni->backend_id];
        if (ba.buffers.empty()) return;
        b = ba.buffers[0];
    }
    size_t off = ni->offset;
    void*  base = b->data();
    t->bind_data(static_cast<char*>(base) + off);
    t->buffer_      = b;
    t->buffer_offs_ = off;
}

// ============================================================
// release
// ============================================================
void Gallocr::release() {
    for (auto& ba : backends_) {
        for (Buffer* b : ba.buffers) delete b;
        ba.buffers.clear();
        delete ba.talloc;
        ba.talloc = nullptr;
        ba.peak = 0;
        ba.use = false;
    }
    node_map_.clear();
    nodes_.clear();
    leaves_.clear();
}

} // namespace ppml

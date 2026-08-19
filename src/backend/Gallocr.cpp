#include "ppml/Gallocr.h"
#include "ppml/ComputeGraph.h"
#include "ppml/Backend.h"   // alloc_buffer, Buffer, BufferType

#include <algorithm>
#include <cstdlib>
#include <cstdio>
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
    phase1_offset_.clear();
    for (auto& ba : backends_) {
        ba.high_watermark = 0;
        ba.live.clear();
    }
    recording_phase1_ = false;
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
        ni.is_output  = (t->flag & (TENSOR_FLAG_OUTPUT | TENSOR_FLAG_LOSS)) != 0;
        node_map_[t]  = &ni;
    }
    // 登记 leafs
    for (int i = 0; i < graph->n_leafs(); i++) {
        TensorF32* t = graph->graph_leaf(i);
        NodeInfo& li = leaves_[i];
        li.tensor     = t;
        li.managed    = (t->data() == nullptr && t->view_src == nullptr);
        li.backend_id = backend_id_of(t);
        li.is_output  = (t->flag & (TENSOR_FLAG_OUTPUT | TENSOR_FLAG_LOSS)) != 0;
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

    if (recording_phase1_) {
        // ---- Phase1：best-fit 模拟分配，记录偏移 + 峰值 ----
        if (!ba.talloc->alloc(alloc_size, offset)) {
            return false;  // 后端 buffer 空间不足（Phase1 为虚拟大空间，理论上不应发生）
        }
        phase1_offset_[t] = offset;
        ba.high_watermark = std::max(ba.high_watermark, offset + alloc_size);

        // 峰值 = 最大同时存活字节数（used_bytes 随借/还变化）
        size_t live = ba.talloc->used_bytes();
        ba.peak = std::max(ba.peak, live);
    } else {
        // ---- Phase2：直接复用 Phase1 记录的偏移（保证两阶段布局一致）----
        auto it = phase1_offset_.find(t);
        if (it == phase1_offset_.end()) return false;
        offset = it->second;

        // 方向3：分配前检查新区间与当前存活区间是否重叠（应恒不重叠，防御性断言）
        if (getenv("GRAPH_DEBUG_GALLOCR")) check_live_overlap(ba, ni);
        add_live(ba, offset, alloc_size);
    }

    ni->offset     = offset;
    ni->alloc_size = alloc_size;
    ni->buffer     = ba.buffers.empty() ? nullptr : ba.buffers[0];
    ni->allocated  = true;
    return true;
}

// ============================================================
// free_node — 释放（还空间），使后继张量可复用
// ============================================================
void Gallocr::free_node(NodeInfo* ni) {
    if (!ni->allocated) return;
    if (ni->is_output) return;  // OUTPUT 永不复用（本实现保守：也不释放）

    BackendAlloc& ba = backends_[ni->backend_id];
    if (recording_phase1_) {
        // Phase1：把空间还给虚拟 talloc（供后续张量复用）
        ba.talloc->free_bytes(ni->offset, ni->alloc_size);
    } else {
        // Phase2：从存活区间集合移除（方向3）
        remove_live(ba, ni->offset, ni->alloc_size);
    }
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
    recording_phase1_ = true;
    for (auto& ba : backends_) {
        ba.talloc = new DynTalloc(std::numeric_limits<size_t>::max() / 2, ba.buft->get_alignment());
    }

    // 3. 先分配 managed leaves（输入/常量，若需分配）
    for (auto& li : leaves_) {
        if (li.managed && !allocate_node(&li)) {
            if (getenv("GRAPH_DEBUG_GALLOCR")) {
                fprintf(stderr, "[gallocr] reserve FAIL at leaf idx=%ld n_bytes=%zu\n",
                        (long)(&li - &leaves_[0]), li.tensor ? li.tensor->nbytes() : 0);
            }
            release(); return false;
        }
    }

    // 4. 遍历 nodes（拓扑序）：分配当前节点，释放已无依赖的 src
    for (auto& ni : nodes_) {
        TensorF32* node = ni.tensor;
        // 先分配本节点
        if (ni.managed && !allocate_node(&ni)) {
            if (getenv("GRAPH_DEBUG_GALLOCR")) {
                fprintf(stderr, "[gallocr] reserve FAIL at node idx=%ld n_bytes=%zu\n",
                        (long)(&ni - &nodes_[0]), node ? node->nbytes() : 0);
            }
            release(); return false;
        }

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
    if (getenv("GRAPH_DEBUG_GALLOCR")) {
        for (auto& ba : backends_) {
            fprintf(stderr, "[gallocr] reserve done: backend=%d peak=%zu bytes (%.2f GB)\n",
                    (int)(&ba - &backends_[0]), ba.peak, ba.peak / (1024.0 * 1024.0 * 1024.0));
        }
    }
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

        // buffer 大小 = Phase1 的 max(offset+alloc_size)（high_watermark ≥ peak），
        // 直接复用 Phase1 记录的偏移，故不需要为 best-fit 碎片预留 —— 加少量 slack 兜底即可。
        size_t size = ba.high_watermark;
        if (size > 0) {
            // slack：12.5% + 1MB 兜底（防御 offset+alloc_size 的尾部边界）
            size_t slack = GGML_PAD(ba.high_watermark / 8 + (1 << 20), ba.buft->get_alignment());
            size += slack;
            Buffer* buf = alloc_buffer(ba.buft, size);
            if (!buf) {
                if (getenv("GRAPH_DEBUG_GALLOCR")) {
                    fprintf(stderr, "[gallocr] alloc FAIL: backend=%d size=%zu bytes (%.2f GB)\n",
                            (int)(&ba - &backends_[0]), size, size / (1024.0 * 1024.0 * 1024.0));
                }
                release(); return false;
            }
            ba.buffers.push_back(buf);
        }
        ba.talloc = new DynTalloc(size, ba.buft->get_alignment());
    }

    // 2. 重置 refcount（Phase1 已把 n_children 减到 0），重新统计
    //     Phase2 不再记录偏移（phase1_offset_ 已是 Phase1 的基线）
    recording_phase1_ = false;
    compute_refcounts(graph, backend_id_of);

    // 3. 分配 managed leaves 并绑定
    for (auto& li : leaves_) {
        if (!li.managed) continue;
        if (!allocate_node(&li)) {
            if (getenv("GRAPH_DEBUG_GALLOCR")) {
                fprintf(stderr, "[gallocr] alloc FAIL at leaf idx=%ld n_bytes=%zu peak=%zu\n",
                        (long)(&li - &leaves_[0]), li.tensor ? li.tensor->nbytes() : 0,
                        backends_[li.backend_id].peak);
            }
            release(); return false;
        }
        bind_tensor(&li);
    }

    // 4. 遍历 nodes，分配并绑定
    for (auto& ni : nodes_) {
        TensorF32* node = ni.tensor;
        if (ni.managed && !allocate_node(&ni)) {
            if (getenv("GRAPH_DEBUG_GALLOCR")) {
                BackendAlloc& ba = backends_[ni.backend_id];
                fprintf(stderr, "[gallocr] alloc FAIL at node idx=%ld n_bytes=%zu peak=%zu used=%zu free_blocks=%zu\n",
                        (long)(&ni - &nodes_[0]), node ? node->nbytes() : 0,
                        ba.peak,
                        ba.talloc ? ba.talloc->used_bytes() : 0,
                        ba.talloc ? ba.talloc->free_blocks().size() : 0);
            }
            release(); return false;
        }
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

    // 常量叶子：分配完成后从 const_data_ 填充数据。
    //   TENSOR_FLAG_CONST 置位 → 静态可复用（保留 const_data_，图可复用）；
    //   不置位 → 动态一次性（填充后清空+shrink，避免宿主内存累积）。
    if (!t->const_data_.empty() && t->data() != nullptr) {
        std::memcpy(t->data(), t->const_data_.data(),
                    t->const_data_.size() * sizeof(float));
        if (!(t->flag & TENSOR_FLAG_CONST)) {
            t->const_data_.clear();
            t->const_data_.shrink_to_fit();
        }
    }
}

// ============================================================
// 方向3：Phase2 存活区间跟踪 + overlap 检查
//   Phase2 复用 Phase1 偏移，布局本身一致；此处为防御性校验，
//   若未来有任何布局 bug 导致两个同时存活的张量区间重叠，立即暴露。
// ============================================================
void Gallocr::add_live(BackendAlloc& ba, size_t off, size_t size) {
    if (size == 0) return;
    ba.live.push_back({off, size});
}

void Gallocr::remove_live(BackendAlloc& ba, size_t off, size_t size) {
    if (size == 0) return;
    for (size_t i = 0; i < ba.live.size(); ++i) {
        if (ba.live[i].off == off && ba.live[i].size == size) {
            ba.live.erase(ba.live.begin() + i);
            return;
        }
    }
    // 未找到（如重复 free）—— 幂等，忽略。
}

void Gallocr::check_live_overlap(BackendAlloc& ba, const NodeInfo* ni) {
    const size_t off  = phase1_offset_[ni->tensor];
    const size_t size = GGML_PAD(ba.buft->get_alloc_size(ni->tensor), ba.buft->get_alignment());
    if (size == 0) return;
    for (const auto& r : ba.live) {
        if (off < r.off + r.size && r.off < off + size) {
            fprintf(stderr,
                    "[gallocr][LIVE-OVERLAP] tensor=%p n_bytes=%zu [off=%zu,+%zu) overlaps "
                    "live [off=%zu,+%zu) -- 两阶段布局错位/复用碰撞!\n",
                    (void*)ni->tensor, ni->tensor ? ni->tensor->nbytes() : 0,
                    off, size, r.off, r.size);
        }
    }
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
        ba.high_watermark = 0;
        ba.live.clear();
        ba.use = false;
    }
    node_map_.clear();
    nodes_.clear();
    leaves_.clear();
    phase1_offset_.clear();
    recording_phase1_ = false;
}

// ============================================================
// diagnose_aliasing — 查找与 target 在 Phase2 中 offset 区间重叠的 managed 节点。
// 用于排查 pred_coords 等被其它节点 buffer 复用覆盖的问题。
//  - 若 target 为 managed（在 gallocr buffer 中）→ 列出与其 [off, off+size)
//    区间重叠、且在目标生命周期内可能复用的节点。
//  - 若 target 非 managed（data() 或 view_src 非空，如 wrap_value_as_leaf 的叶子）→
//    其数据在独立 scratch/宿主内存，不受 gallocr buffer 复用影响，直接打印该结论。
// 返回重叠的 managed 节点数（仅对 managed target 有意义）。
// ============================================================
int Gallocr::diagnose_aliasing(TensorF32* target) {
    if (!target) return 0;
    int n_overlap = 0;

    // 目标在 node_map_ 中（nodes_/leaves_ 里的 NodeInfo）
    auto tinfo = node_map_.find(target);
    bool t_managed = false;
    size_t t_off = 0, t_size = 0;
    if (tinfo != node_map_.end() && tinfo->second->managed && tinfo->second->allocated) {
        t_managed = true;
        t_off  = tinfo->second->offset;
        t_size = GGML_PAD(tinfo->second->tensor->nbytes(),
                          backends_[tinfo->second->backend_id].buft->get_alignment());
    }

    if (!t_managed) {
        fprintf(stderr,
                "[gallocr][alias] target=%p n_bytes=%zu is NON-managed "
                "(data=%p view_src=%p) -> persistent scratch/leaf, "
                "NOT subject to gallocr buffer reuse.\n",
                (void*)target, target->nbytes(),
                (void*)target->data(), (void*)target->view_src);
        return 0;
    }

    // 目标 managed：遍历所有 node/leaf，找 offset 区间重叠的。
    auto overlaps = [&](const NodeInfo& oi) {
        if (!oi.managed || !oi.allocated || oi.tensor == target) return;
        size_t o_size = GGML_PAD(oi.tensor->nbytes(),
                                 backends_[oi.backend_id].buft->get_alignment());
        size_t o_off = oi.offset;
        if (o_off < t_off + t_size && t_off < o_off + o_size) {  // 区间相交
            fprintf(stderr,
                    "[gallocr][alias] OVERLAP target_off=%zu size=%zu <-> tensor=%p "
                    "off=%zu size=%zu (op=%d, managed=%d)\n",
                    t_off, t_size, (void*)oi.tensor, o_off, o_size,
                    (int)oi.tensor->op, (int)oi.managed);
            n_overlap++;
        }
    };
    for (auto& li : leaves_) overlaps(li);
    for (auto& ni : nodes_)  overlaps(ni);

    return n_overlap;
}

} // namespace ppml

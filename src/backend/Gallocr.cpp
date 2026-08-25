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

    // ===== 恢复参数 data()：参数(TENSOR_FLAG_PARAM)必须始终有有效数据 =====
    // 之前 Gallocr::release 曾把参数 bind_data(nullptr) 清空（旧 bug，已修复跳过 PARAM），
    // 但已清空过的参数 data() 仍为 null。若其 buffer_(param_buf) 非空，从这里恢复
    // data() = buffer_->data() + buffer_offs_，使参数重新指向 param_buf（非空）。
    // 这样参数不被 Gallocr 判定 managed → 不分配可复用中间 buffer → 不被覆盖污染。
    auto restore_param_data = [](TensorF32* t) {
        if (!(t->flag & TENSOR_FLAG_PARAM)) return;
        if (t->data() != nullptr) return;
        if (t->buffer_ != nullptr) {
            t->bind_data(static_cast<float*>(t->buffer_->data()) + t->buffer_offs_);
        }
    };
    for (int i = 0; i < graph->n_nodes(); i++) restore_param_data(graph->graph_node(i));
    for (int i = 0; i < graph->n_leafs(); i++) restore_param_data(graph->graph_leaf(i));

    // 登记 nodes
    for (int i = 0; i < graph->n_nodes(); i++) {
        TensorF32* t = graph->graph_node(i);
        NodeInfo& ni = nodes_[i];
        ni.tensor     = t;
        // 参数(TENSOR_FLAG_PARAM)需要 buffer，但该 buffer 必须存活到训练结束（不可被复用覆盖）。
        // 这里仍按 data()/view_src 判定 managed（参数 data() 通常非空则不分配）；
        // 若参数 data() 为 null 会被判定 managed 并分配——这是允许的（参数需要 buffer），
        // 但后续需保证其 buffer 不被复用（见 is_output / 存活标记）。
        ni.managed    = (t->data() == nullptr && t->view_src == nullptr);
        ni.backend_id = backend_id_of(t);
        ni.is_output  = (t->flag & (TENSOR_FLAG_OUTPUT | TENSOR_FLAG_LOSS)) != 0;
        // 诊断：恢复后仍 data()==nullptr（参数从未有独立数据）→ 真问题，需查参数创建/transfer。
        if ((t->flag & TENSOR_FLAG_PARAM) && ni.managed) {
            fprintf(stderr, "[gallocr] WARN param node data()==nullptr buffer_=%p nbytes=%zu view_src=%p const_data=%zu op=%d dims=[%lld,%lld,%lld,%lld]",
                    (const void*)t->buffer_, t->nbytes(), (const void*)t->view_src,
                    t->const_data_.size(), (int)t->op,
                    (long long)(t->shape().ndim()>0?t->shape().dims[0]:-1),
                    (long long)(t->shape().ndim()>1?t->shape().dims[1]:-1),
                    (long long)(t->shape().ndim()>2?t->shape().dims[2]:-1),
                    (long long)(t->shape().ndim()>3?t->shape().dims[3]:-1));
            // 追溯：该参数被哪个节点引用（定位它是哪层的权重）
            for (int j = 0; j < graph->n_nodes(); j++) {
                TensorF32* tj = graph->graph_node(j);
                for (int s = 0; s < GGML_MAX_SRC; s++) {
                    if (tj->src[s] == t) {
                        fprintf(stderr, " used_by(op=%d,src%d)", (int)tj->op, s);
                        break;
                    }
                }
            }
            fprintf(stderr, "\n");
        }
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
        // 诊断：恢复后仍 data()==nullptr → 真问题
        if ((t->flag & TENSOR_FLAG_PARAM) && li.managed) {
            fprintf(stderr, "[gallocr] WARN param leaf data()==nullptr buffer_=%p nbytes=%zu (param has NO data!)\n",
                    (const void*)t->buffer_, t->nbytes());
        }
        node_map_[t]  = &li;
    }

    // 统计 refcount：遍历 nodes，对每个 src 若为 managed 则 +1
    // view 特例：view 共享底层数据，view 本身非 managed（无独立 buffer）。
    //  - view 节点的 src（底层）不计数：view 不"消费/释放"底层，是共享。
    //  - 消费者引用 view 时：同时给 view 与其底层计数，保证底层在 view 的消费者间存活。
    // 跨后端消费保护（2026-08-23，方案A）：跨后端被消费的 managed 中间节点标记 is_output
    // （空间不复用），保证数据在跨后端拷贝（D2H/H2D）时有效。
    // 诊断开关：PPML_NO_CROSSBK_GUARD=1 禁用（排查保护是否引入其他问题）。
    const bool kGuard = !(getenv("PPML_NO_CROSSBK_GUARD") &&
                          std::strcmp(getenv("PPML_NO_CROSSBK_GUARD"), "1") == 0);
    //  2026-08-24 跨 split 保活（方案A 补强）：OP_DUP（build_splits 创建的跨后端 cpy 节点）
    //    的 src[0]（原始跨后端输入）必须保活。机制：build_splits 把 node->src[j] 替换为 cpy，
    //    gallocr 认为原始 src 只被 cpy 消费（n_children=1）→ alloc 时 src 空间被后续节点复用；
    //    但 graph_compute 的 Step 1 D2H 在 split 执行时才读原始 src → 读到被覆盖数据
    //    （GPU scatter 输入 msg 的 buffer 悬垂 → CUDA invalid argument / loss 爆炸）。
    //    标 is_output 后原始 src 空间不释放、不复用，保证 Step 1 D2H 数据有效。
    //    开关：PPML_ALIVE_CROSS_SPLIT=0 禁用（默认开）。
    const bool kAliveSplit = !(getenv("PPML_ALIVE_CROSS_SPLIT") &&
                               std::strcmp(getenv("PPML_ALIVE_CROSS_SPLIT"), "0") == 0);
    for (int i = 0; i < graph->n_nodes(); i++) {
        TensorF32* node = graph->graph_node(i);
        if (node->view_src) continue;  // view 节点：不把底层当作被消费计数
        int node_bk = backend_id_of(node);
        // OP_DUP（cpy）的 src[0] 是跨后端输入源，必须保活到 Step 1 D2H 之后。
        if (kAliveSplit && node->op == OP_DUP && node->src[0]) {
            auto dit = node_map_.find(node->src[0]);
            if (dit != node_map_.end() && dit->second->managed) {
                dit->second->is_output = true;
            }
        }
        //  2026-08-24 精准修复：SCATTER_ADD / EDGE_GATHER_ROWS 的 src[1]（边索引 leaf，
        //    如 SE3 的 edge_src/edge_tgt）必须独立 buffer——scatter_add 同时读 msg(src[0])
        //    和 tgt_idx(src[1])，gallocr 空间复用会让它们共享同一 buffer（实测 msg_buf==
        //    tgt_buf → CUDA scatter invalid argument / 数值错乱）。索引 leaf 标 is_output
        //    后 buffer 独立存活，不再与 msg 别名。只对这两个 op 的索引输入生效（避免
        //    误伤其他 CONST 常量，防止全量 is_output 导致 segfault）。
        if ((node->op == OP_SCATTER_ADD || node->op == OP_EDGE_GATHER_ROWS) && node->src[1]) {
            auto iit = node_map_.find(node->src[1]);
            if (iit != node_map_.end() && iit->second->managed) {
                iit->second->is_output = true;
            }
        }
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            TensorF32* src = node->src[s];
            if (!src) continue;
            auto it = node_map_.find(src);
            if (it == node_map_.end()) continue;
            NodeInfo* sni = it->second;
            if (kGuard && sni->backend_id >= 0 && node_bk >= 0 &&
                sni->backend_id != node_bk && sni->managed) {
                sni->is_output = true;
            }
            if (src->view_src) {
                // src 是 view：消费者依赖 view 及 view 的底层数据
                sni->n_children++;  // view 本身（非 managed，仅跟踪释放时序）
                TensorF32* under = src->src[0];
                if (under) {
                    auto uit = node_map_.find(under);
                    if (uit != node_map_.end() && uit->second->managed) {
                        uit->second->n_children++;  // 底层在 view 的消费者间存活
                        // 跨后端消费底层同样保护（同 node 循环条件）
                        if (uit->second->backend_id >= 0 && node_bk >= 0 &&
                            uit->second->backend_id != node_bk) {
                            uit->second->is_output = true;
                        }
                    }
                }
            } else if (sni->managed) {
                sni->n_children++;
            }
        }
    }
    // ⚠️ 关键修复（2026-08-22）：backward 的 grad 节点消费前向节点，但 grad 节点不在
    // graph->nodes() 里（由 build_backward_expand 通过 add_or_set 存入 grads[ihash]）。
    // 若不计数，前向节点（如 msa）在 forward 消费完后 refcount=0 被释放，backward 算梯度时
    // 读已释放/复用的 buffer → 梯度 0/垃圾。开 SE3 时节点多（8165）复用密集 → msa head 梯度
    // 实测 1e-9（被覆盖为 0）。修复：遍历 graph->grads，对非空 grad 节点的 src 也计数，
    // 保证前向节点存活到 backward 消费完。
    TensorF32** grads_arr = graph->graph_grads();
    if (grads_arr) {
        const int n_grad_slots = graph->graph_grad_slots();  // grads 数组大小 == hash 容量
        for (int gi = 0; gi < n_grad_slots; gi++) {
            TensorF32* gnode = grads_arr[gi];
            if (!gnode) continue;
            if (gnode->view_src) continue;
            int gnode_bk = backend_id_of(gnode);
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                TensorF32* src = gnode->src[s];
                if (!src) continue;
                auto it = node_map_.find(src);
                if (it == node_map_.end()) continue;
                NodeInfo* sni = it->second;
                // 跨后端保护（同 node 循环）：跨后端 managed 中间节点 is_output（kGuard 同 node 循环）
                if (kGuard && sni->backend_id >= 0 && gnode_bk >= 0 &&
                    sni->backend_id != gnode_bk && sni->managed) {
                    sni->is_output = true;
                }
                if (src->view_src) {
                    sni->n_children++;
                    TensorF32* under = src->src[0];
                    if (under) {
                        auto uit = node_map_.find(under);
                        if (uit != node_map_.end() && uit->second->managed) {
                            uit->second->n_children++;
                            if (kGuard && uit->second->backend_id >= 0 && gnode_bk >= 0 &&
                                uit->second->backend_id != gnode_bk) {
                                uit->second->is_output = true;
                            }
                        }
                    }
                } else if (sni->managed) {
                    sni->n_children++;
                }
            }
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

    // 复位上一轮（或上一图）绑定的张量指针元数据，使 compute_refcounts 的
    // managed 判定（依赖 data()==nullptr）重新成立。否则若上一轮 alloc 已把
    // tensor->data_/buffer_ 绑到 buffer、而本轮 reserve 不先复位，则
    // compute_refcounts 会误判 managed=false → 不重分配 → 悬垂指针崩溃。
    // 用【图本身】作为权威来源（而非 nodes_/leaves_ 快照）：遍历所有 graph 张量，
    // 凡 buffer_ 非空（曾被 gallocr 绑定）即复位。这样即便 free_node 把节点
    // allocated 置 false、或 nodes_ 快照与张量实际 buffer_ 不一致，也能可靠复位。
    // 叶子（buffer_==nullptr，数据在 context scratch）不受影响。
    auto reset_tensor = [](TensorF32* t) {
        // 跳过参数(TENSOR_FLAG_PARAM)：参数数据必须跨 graph_compute 存活到训练结束。
        // 若参数经 transfer_params_to_backend 迁到独立 param_buf（buffer_ 非空），这里 reset
        // 会 bind_data(nullptr) 清空参数 data_ 并清 buffer_，导致参数数据永久丢失 → 后续
        // kernel 读参数 data()==nullptr 段错误或 Gallocr 重新分配可复用 buffer 被覆盖 → NaN。
        // 参数 buffer 归 param_buffers_ 管理，绝不能被 Gallocr 当作可复用中间 buffer reset。
        if (t && (t->flag & TENSOR_FLAG_PARAM)) return;
        if (t && t->buffer_ != nullptr) {
            t->bind_data(nullptr);
            t->buffer_      = nullptr;
            t->buffer_offs_ = 0;
        }
    };
    for (int i = 0; i < graph->n_nodes(); ++i) reset_tensor(graph->graph_node(i));
    for (int i = 0; i < graph->n_leafs(); ++i) reset_tensor(graph->graph_leaf(i));

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
    //    诊断：跟踪 nbytes 最大的张量（定位异常超大 buffer，如 backward 节点 shape bug）
    long max_node_idx = -1; size_t max_node_bytes = 0; int max_node_op = -1;
    for (auto& ni : nodes_) {
        TensorF32* node = ni.tensor;
        if (node && node->nbytes() > max_node_bytes) {
            max_node_bytes = node->nbytes();
            max_node_idx   = &ni - &nodes_[0];
            max_node_op    = (int)node->op;
        }
        // 先分配本节点
        if (ni.managed && !allocate_node(&ni)) {
            if (getenv("GRAPH_DEBUG_GALLOCR")) {
                fprintf(stderr, "[gallocr] reserve FAIL at node idx=%ld n_bytes=%zu\n",
                        (long)(&ni - &nodes_[0]), node ? node->nbytes() : 0);
            }
            release(); return false;
        }

        // 释放 src：该节点被消费后，其依赖的 src 引用计数减一
        if (node->view_src) continue;  // view 节点不分配也不释放底层（共享，随消费者释放）
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            TensorF32* src = node->src[s];
            if (!src) continue;
            auto it = node_map_.find(src);
            if (it == node_map_.end()) continue;
            NodeInfo* sni = it->second;
            if (src->view_src) {
                // src 是 view：view 被消费，view 引用减一；到底层也减一（底层随 view 消费者释放）
                if (--sni->n_children <= 0) {
                    free_node(sni);  // view 非 managed，free_node 应跳过实际释放
                    TensorF32* under = src->src[0];
                    if (under) {
                        auto uit = node_map_.find(under);
                        if (uit != node_map_.end() && uit->second->managed &&
                            --uit->second->n_children <= 0) {
                            free_node(uit->second);
                        }
                    }
                }
            } else if (sni->managed && --sni->n_children <= 0) {
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
        // 诊断：打印 nbytes 最大的张量（定位异常超大 buffer 的来源 op）
        fprintf(stderr, "[gallocr] max tensor: node_idx=%ld n_bytes=%zu (%.2f GB) op=%d dims=[",
                max_node_idx, max_node_bytes, max_node_bytes / (1024.0 * 1024.0 * 1024.0), max_node_op);
        if (max_node_idx >= 0 && max_node_idx < (long)nodes_.size()) {
            TensorF32* mt = nodes_[max_node_idx].tensor;
            if (mt) {
                for (int d = 0; d < mt->shape().ndim(); ++d)
                    fprintf(stderr, "%s%lld", (d?",":""), (long long)mt->shape().dims[d]);
                fprintf(stderr, "]");
                auto pr = [](TensorF32* t) {
                    if (!t) { fprintf(stderr, "null"); return; }
                    fprintf(stderr, "[op=%d d=", (int)t->op);
                    for (int d = 0; d < t->shape().ndim(); ++d)
                        fprintf(stderr, "%s%lld", (d?",":""), (long long)t->shape().dims[d]);
                    fprintf(stderr, "]");
                };
                fprintf(stderr, " src0="); pr(mt->src[0]);
                fprintf(stderr, " src1="); pr(mt->src[1]);
                // 若 src0 是 ADD（op=2），打印其输入，定位 [332928,1] 的来源
                if (mt->src[0] && mt->src[0]->op == OP_ADD) {
                    fprintf(stderr, " src0_src0="); pr(mt->src[0]->src[0]);
                    fprintf(stderr, " src0_src1="); pr(mt->src[0]->src[1]);
                }
            }
        }
        fprintf(stderr, "\n");
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

        // 释放 src：该节点被消费后，其依赖的 src 引用计数减一（view 特例同 reserve）
        if (node->view_src) continue;  // view 节点不分配也不释放底层（共享，随消费者释放）
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            TensorF32* src = node->src[s];
            if (!src) continue;
            auto it = node_map_.find(src);
            if (it == node_map_.end()) continue;
            NodeInfo* sni = it->second;
            if (src->view_src) {
                if (--sni->n_children <= 0) {
                    free_node(sni);  // view 非 managed，free_node 应跳过实际释放
                    TensorF32* under = src->src[0];
                    if (under) {
                        auto uit = node_map_.find(under);
                        if (uit != node_map_.end() && uit->second->managed &&
                            --uit->second->n_children <= 0) {
                            free_node(uit->second);
                        }
                    }
                }
            } else if (sni->managed && --sni->n_children <= 0) {
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
    //   注意：若 buffer 是 device（CUDA），必须用 set_tensor（H2D），
    //   否则 std::memcpy 会把 device 指针当 host 源 → UB/崩溃。
    if (!t->const_data_.empty() && t->data() != nullptr) {
        if (b && !b->is_host()) {
            b->set_tensor(t, t->const_data_.data(), off,
                          t->const_data_.size() * sizeof(float));  // H2D
        } else {
            std::memcpy(t->data(), t->const_data_.data(),
                        t->const_data_.size() * sizeof(float));
        }
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
    // 复位所有曾绑定（buffer_!=nullptr）张量的指针元数据，否则这些张量仍指向
    // 即将被 cudaFree/delete 的 buffer，下一图 compute_refcounts 会因
    // data()!=nullptr 误判 managed=false 而不重分配 → 悬垂 device 指针崩溃。
    // 用 ni.buffer!=nullptr 判定（而非 ni.allocated）：free_node 在 Phase2 会把已
    // 释放节点 allocated 置 false，但 data_/buffer_ 仍指向即将删除的 buffer，
    // 若按 allocated 判定会漏掉 → 悬垂指针。
    auto reset_bound = [&](std::vector<NodeInfo>& infos) {
        for (auto& ni : infos) {
            // 跳过参数(TENSOR_FLAG_PARAM)：参数数据必须跨 graph_compute 存活到训练结束，
            // 不能被复位成 nullptr（否则 embedding/get_rows/linear 读参数 data()==nullptr 段错误）。
            // 参数若由 transfer_params_to_backend 迁到独立 param_buf，其 buffer_ 非空且不应被
            // Gallocr 当作可复用中间 buffer 清空；此处只清 Gallocr 自己分配的 managed 中间节点。
            if (ni.tensor->flag & TENSOR_FLAG_PARAM) continue;
            if (ni.buffer != nullptr) {
                ni.tensor->bind_data(nullptr);
                ni.tensor->buffer_      = nullptr;
                ni.tensor->buffer_offs_ = 0;
                ni.allocated = false;
            }
        }
    };
    reset_bound(nodes_);
    reset_bound(leaves_);

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

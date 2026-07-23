
#include "rfaa/Backend.h"
#include "rfaa/Context.h"
#include <algorithm>
#include <cstring>

namespace rfaa {

// ============================================================
// 构造/析构
// ============================================================

BackendScheduler::BackendScheduler() {
    std::memset(splits_, 0, sizeof(splits_));
    ctx_ = &context();  // 从全局 context 分配子图
}

BackendScheduler::~BackendScheduler() {
    // splits_ 由 context 管理，不需要手动释放
}

void BackendScheduler::add_backend(Backend * backend) {
    backends_.push_back(backend);
    // 按优先级降序排序（高优先级先被调度）
    std::sort(backends_.begin(), backends_.end(),
        [](Backend * a, Backend * b) { return a->priority() > b->priority(); });
    n_backends_ = static_cast<int>(backends_.size());
}

// ============================================================
// 辅助方法
// ============================================================

bool BackendScheduler::is_view_op(int op) const {
    return op == OP_VIEW || op == OP_RESHAPE || op == OP_PERMUTE || op == OP_TRANSPOSE;
}

void BackendScheduler::set_backend_if_supported(Tensor * node, int backend_id) {
    Backend * backend = backends_[backend_id];
    if (backend->supports_op(node)) {
        backend_map_[node] = backend_id;
    }
}

int BackendScheduler::count_supported_inputs(Tensor * node, int backend_id) const {
    int count = 0;
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        Tensor * src = node->src[j];
        if (!src) continue;

        // 输入已经分配了后端 且 buffer 兼容
        auto it = backend_map_.find(src);
        if (it != backend_map_.end() && tensor_buffer_compatible(src, backend_id)) {
            count++;
        }
    }
    return count;
}

bool BackendScheduler::tensor_buffer_compatible(const Tensor * src, int backend_id) const {
    auto it = backend_map_.find(src);
    if (it == backend_map_.end()) return false;

    int src_backend = it->second;
    Backend * target = backends_[backend_id];

    // 同一个后端 → 兼容
    if (src_backend == backend_id) return true;

    // 跨后端：检查 target 是否支持 src 的内存 buffer 类型
    Backend * source = backends_[src_backend];
    return target->supports_buffer_type(source->buffer_type());
}

int BackendScheduler::tensor_backend_id(Tensor * t) const {
    auto it = backend_map_.find(t);
    return (it != backend_map_.end()) ? it->second : -1;
}

int BackendScheduler::tensor_backend_id(Tensor * t, int default_id) const {
    int id = tensor_backend_id(t);
    return (id == -1) ? default_id : id;
}

// ============================================================
// alloc_splits — 检测 backend 变化并重新分配图内存
// 对标 ggml_backend_sched_alloc_splits
// ============================================================

bool BackendScheduler::alloc_splits() {
    // ===== Step 1: 检查 backend IDs 是否发生变化 =====
    bool backend_ids_changed = false;

    // 检查节点
    for (int i = 0; i < current_graph_->n_nodes(); i++) {
        Tensor* node = current_graph_->node(i);
        int cur_id = tensor_backend_id(node, 0);
        if (cur_id != prev_node_backend_ids_[i] &&
            bufts_[cur_id] != bufts_[prev_node_backend_ids_[i]]) {
            backend_ids_changed = true;
            break;
        }
    }

    // 检查叶子
    if (!backend_ids_changed) {
        for (int i = 0; i < current_graph_->n_leafs(); i++) {
            Tensor* leaf = current_graph_->leaf(i);
            int cur_id = tensor_backend_id(leaf, 0);
            if (cur_id != prev_leaf_backend_ids_[i] &&
                bufts_[cur_id] != bufts_[prev_leaf_backend_ids_[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // ===== Step 2: 如果需要，重新分配图内存 =====
    if (backend_ids_changed || !graph_reserved_) {
        // 同步所有后端（避免正在使用的 tensor 被移动）
        for (int i = 0; i < n_backends_; i++) {
            backends_[i]->synchronize();
        }

        // 根据当前 backend 分配，为每个后端预留内存
        if (!reserve_graph_memory()) {
            // GGML_LOG_ERROR
            return false;
        }

        // 保存当前分配作为"上一次"快照，供下次对比
        prev_node_backend_ids_ = node_backend_ids_;
        prev_leaf_backend_ids_ = leaf_backend_ids_;
        graph_reserved_ = true;
    }

    return true;
}


// ============================================================
// alloc_buffer — 对标 ggml_backend_buft_alloc_buffer
// ============================================================
Buffer* alloc_buffer(BufferType* buft, size_t size, BufferUsage usage) {
    if (!buft) return nullptr;

    // 零大小：返回空 buffer（对标 ggml 的 dummy buffer）
    if (size == 0) {
        return new DefaultBuffer(buft, 0, usage);
    }

    return new DefaultBuffer(buft, size, usage);
}

// ============================================================
// alloc_tensor_range — 对标 ggml 的 static alloc_tensor_range
// 将 [first_idx, last_idx) 区间的 tensor 分配到 buft 的 buffer 中
// ============================================================
static bool alloc_tensor_range(
    ComputeGraph*            graph,
    int                      first_idx,
    int                      last_idx,       // -1 表示到末尾
    BufferType*              buft,
    size_t                   buffer_size,
    std::vector<Buffer*>&    buffers)         // 输出：累积的 buffer 列表
{
    // 1. 分配 buffer
    Buffer* buffer = alloc_buffer(buft, buffer_size);
    if (!buffer) {
        return false;
    }

    buffers.push_back(buffer);

    // 2. 创建子分配器
    TensorAllocator tallocr(buffer);

    // 3. 先遍历 nodes
    int n_nodes = graph->n_nodes();
    for (int i = first_idx; i < n_nodes && (last_idx < 0 || i < last_idx); i++) {
        TensorF32* t = graph->node(i);

        // 跳过已分配或有 view_src 的 tensor
        if (t->data() != nullptr) {
            if (t->view_src == nullptr) {
                continue;  // 已独立分配
            } else if (t->buffer_ == nullptr) {
                // view of pre-allocated tensor → 让 view 指向源数据
                t->data_        = t->view_src->data_;
                t->buffer_      = t->view_src->buffer_;
                t->buffer_offs_ = t->view_src->buffer_offs_;
            }
            continue;
        }

        if (t->view_src != nullptr) {
            // view tensor：不需要新内存，指向源
            if (t->buffer_ == nullptr) {
                t->data_        = t->view_src->data_;
                t->buffer_      = t->view_src->buffer_;
                t->buffer_offs_ = t->view_src->buffer_offs_;
            }
            continue;
        }

        // 普通 tensor：从 buffer 中分配
        if (!tallocr.alloc(t)) {
            return false;
        }
    }

    // 4. 再遍历 leafs
    int n_leafs = graph->n_leafs();
    for (int i = 0; i < n_leafs; i++) {
        TensorF32* t = graph->leaf(i);

        // leaf 通常已经预分配（input/param），跳过
        if (t->data() != nullptr) continue;
        if (t->view_src != nullptr) {
            if (t->buffer_ == nullptr) {
                t->data_        = t->view_src->data_;
                t->buffer_      = t->view_src->buffer_;
                t->buffer_offs_ = t->view_src->buffer_offs_;
            }
            continue;
        }

        if (!tallocr.alloc(t)) {
            return false;
        }
    }

    return true;
}

// ============================================================
// alloc_ctx_tensors_from_buft — 对标 ggml_backend_alloc_ctx_tensors_from_buft_impl
// 将 graph 中所有需要分配的 tensor 分配到 buft 类型的 buffer
// ============================================================
Buffer* alloc_ctx_tensors_from_buft(
    ComputeGraph* graph,
    BufferType*   buft,
    size_t*       nbytes_total,
    bool          no_alloc)
{
    size_t alignment = buft->get_alignment();
    size_t max_size  = buft->get_max_size();

    std::vector<Buffer*> buffers;
    *nbytes_total = 0;

    size_t cur_buf_size = 0;
    int    first_idx    = 0;
    int    n_nodes      = graph->n_nodes();

    // ===== 遍历所有 node =====
    for (int i = 0; i < n_nodes; i++) {
        TensorF32* t = graph->node(i);

        // 计算此 tensor 需要的空间（对齐后）
        size_t this_size = 0;
        if (t->data() == nullptr && t->view_src == nullptr) {
            this_size = GGML_PAD(buft->get_alloc_size(t), alignment);
        }

        // 检查是否需要 flush 当前批次
        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            // 当前 batch 已满，分配掉 [first_idx, i)
            if (!no_alloc) {
                if (!alloc_tensor_range(graph, first_idx, i, buft, cur_buf_size, buffers)) {
                    // 清理已分配的 buffer
                    for (auto* b : buffers) delete b;
                    return nullptr;
                }
            }
            first_idx = i;
            *nbytes_total += cur_buf_size;
            cur_buf_size = this_size;
        } else {
            cur_buf_size += this_size;
        }
    }

    // ===== 遍历所有 leaf（leafs 通常已分配，只统计大小）=====
    for (int i = 0; i < graph->n_leafs(); i++) {
        TensorF32* t = graph->leaf(i);
        if (t->data() == nullptr && t->view_src == nullptr) {
            size_t this_size = GGML_PAD(buft->get_alloc_size(t), alignment);
            cur_buf_size += this_size;  // leaf 也加入最后一批
        }
    }

    // ===== 分配最后一批 =====
    if (cur_buf_size > 0) {
        *nbytes_total += cur_buf_size;
        if (!no_alloc) {
            // 最后一批：[first_idx, n_nodes) 加上所有需要分配的 leafs
            if (!alloc_tensor_range(graph, first_idx, -1, buft, cur_buf_size, buffers)) {
                for (auto* b : buffers) delete b;
                return nullptr;
            }
        }
    }

    // ===== 仅计算大小模式：返回 nullptr =====
    if (no_alloc) {
        return nullptr;
    }

    // ===== 无 tensor 需要分配 =====
    if (buffers.empty()) {
        return nullptr;
    }

    // ===== 返回结果 =====
    if (buffers.size() == 1) {
        return buffers[0];  // 单 buffer，直接返回
    }

    // 多 buffer → 包装为 MultiBuffer
    return new MultiBuffer(std::move(buffers));
}

// ============================================================
// alloc_multi_buffer — 对标 ggml_backend_multi_buffer_alloc_buffer
// 将已有的多个 buffer 包装为一个 MultiBuffer
// ============================================================
Buffer* alloc_multi_buffer(std::vector<Buffer*>& buffers) {
    if (buffers.empty()) return nullptr;
    if (buffers.size() == 1) return buffers[0];

    std::vector<Buffer*> copy;
    copy.reserve(buffers.size());
    for (auto* b : buffers) {
        copy.push_back(b);
    }
    return new MultiBuffer(std::move(copy));
}

// ============================================================
// is_multi_buffer — 对标 ggml_backend_buffer_is_multi_buffer
// ============================================================
bool is_multi_buffer(const Buffer* buffer) {
    if (!buffer) return false;
    return dynamic_cast<const MultiBuffer*>(buffer) != nullptr;
}

// ===== 辅助：根据 backend assignment 预留各后端内存 =====
bool BackendScheduler::reserve_graph_memory() {
    // 1. 更新 node_backend_ids / leaf_backend_ids
    node_backend_ids_.resize(current_graph_->n_nodes());
    for (int i = 0; i < current_graph_->n_nodes(); i++) {
        node_backend_ids_[i] = tensor_backend_id(current_graph_->node(i), 0);
    }

    leaf_backend_ids_.resize(current_graph_->n_leafs());
    for (int i = 0; i < current_graph_->n_leafs(); i++) {
        leaf_backend_ids_[i] = tensor_backend_id(current_graph_->leaf(i), 0);
    }

    // 2. 更新 bufts（缓存 buffer types）
    bufts_.resize(n_backends_);
    for (int i = 0; i < n_backends_; i++) {
        bufts_[i] = backends_[i]->buffer_type();
    }

    // 3. 为每个后端实际分配 buffer
    //    收集该后端的 tensor，各自分配
    for (int b = 0; b < n_backends_; b++) {
        size_t backend_size = 0;
        for (int i = 0; i < current_graph_->n_nodes(); i++) {
            if (node_backend_ids_[i] == b) {
                TensorF32* t = current_graph_->node(i);
                if (t->data() == nullptr && t->view_src == nullptr) {
                    backend_size += GGML_PAD(
                        bufts_[b]->get_alloc_size(t),
                        bufts_[b]->get_alignment());
                }
            }
        }

        // 也为 leafs 统计
        for (int i = 0; i < current_graph_->n_leafs(); i++) {
            if (leaf_backend_ids_[i] == b) {
                TensorF32* t = current_graph_->leaf(i);
                if (t->data() == nullptr && t->view_src == nullptr) {
                    backend_size += GGML_PAD(
                        bufts_[b]->get_alloc_size(t),
                        bufts_[b]->get_alignment());
                }
            }
        }

        if (backend_size > 0) {
            Buffer* buf = alloc_buffer(bufts_[b], backend_size);
            if (!buf) return false;

            // 子分配
            TensorAllocator tallocr(buf);
            for (int i = 0; i < current_graph_->n_nodes(); i++) {
                if (node_backend_ids_[i] == b) {
                    TensorF32* t = current_graph_->node(i);
                    if (t->data() == nullptr && t->view_src == nullptr) {
                        if (!tallocr.alloc(t)) return false;
                    }
                }
            }
            for (int i = 0; i < current_graph_->n_leafs(); i++) {
                if (leaf_backend_ids_[i] == b) {
                    TensorF32* t = current_graph_->leaf(i);
                    if (t->data() == nullptr && t->view_src == nullptr) {
                        if (!tallocr.alloc(t)) return false;
                    }
                }
            }

            // TODO: buf 生命周期由 scheduler 管理，后续需要存储到 splits 对应的 backend 中
            // 当前简化：暂存到 bufts_ 对应位置（后续可扩展为 buffer 列表）
        }
    }

    return true;
}


// ============================================================
// split_graph — 三趟扫描
// ============================================================

void BackendScheduler::split_graph(ComputeGraph * graph) {
    n_splits_ = 0;
    n_graph_inputs_ = 0;
    backend_map_.clear();

    // ============================
    // Pass 1: 叶子节点分配
    // 叶子节点（输入/参数）已经有数据在某个 buffer 上
    // 把它们分配给对应的后端
    // ============================
    pass_assign_leafs(graph);

    // ============================
    // Pass 2: 扩展分配（向下 + 向上两遍）
    // "向下"：高优先级后端的节点，其下游也分配同一后端
    // "向上"：高优先级后端的节点，其上游也分配同一后端
    // 跳过 CPU（最低优先级），确保 CPU 只在必要时被用
    // ============================
    pass_expand_assignments(graph);

    // ============================
    // Pass 3: 填充未分配节点
    // 仍有未分配的节点 → 找支持最多输入的后端
    // 已分配但 buffer 更优的 → 升级到更高优先级
    // ============================
    pass_fill_unassigned(graph);
}

void BackendScheduler::pass_assign_leafs(ComputeGraph * graph) {
    for (int i = 0; i < graph->n_leafs(); i++) {
        Tensor * leaf = graph->leaf(i);
        int leaf_id = tensor_backend_id(leaf);

        // 用户已经手动指定 → 不覆盖
        if (leaf_id != -1) continue;

        // 自动分配：找支持该 tensor buffer 的最近后端
        // 简化实现：优先 GPU（高 priority），其次 CPU
        for (int b = 0; b < n_backends_; b++) {
            if (backends_[b]->supports_buffer_type(leaf->dtype())) {
                backend_map_[leaf] = b;
                break;
            }
        }
    }
}

void BackendScheduler::pass_expand_assignments(ComputeGraph * graph) {
    // ===== 向下扩展 =====
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes(); i++) {
            Tensor * node = graph->node(i);
            if (is_view_op(node->op)) continue;

            int node_id = tensor_backend_id(node);

            if (node_id != -1) {
                // 已有分配
                if (node_id == n_backends_ - 1) {
                    // CPU（最低优先级）→ 阻断扩展
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = node_id;
                }
            } else if (cur_backend_id != -1) {
                // 当前在 GPU 段中，尝试把相邻节点也分配到同一后端
                set_backend_if_supported(node, cur_backend_id);
            }
        }
    }

    // ===== 向上扩展（从底向上遍历）=====
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes() - 1; i >= 0; i--) {
            Tensor * node = graph->node(i);
            if (is_view_op(node->op)) continue;

            int node_id = tensor_backend_id(node);

            if (node_id != -1) {
                if (node_id == n_backends_ - 1) {
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = node_id;
                }
            } else if (cur_backend_id != -1) {
                set_backend_if_supported(node, cur_backend_id);
            }
        }
    }
}

void BackendScheduler::pass_fill_unassigned(ComputeGraph * graph) {
    for (int i = 0; i < graph->n_nodes(); i++) {
        Tensor * node = graph->node(i);
        if (is_view_op(node->op)) continue;

        //int * node_id = &backend_map_[node];  // 创建条目如果不存在

        auto it = backend_map_.find(node);
        if (it == backend_map_.end()) {
        //if (*node_id == -1) {
            // 未分配：找支持最多输入的后端
            int best_supported = -1;
            int best_backend   = n_backends_ - 1;  // 默认 CPU

            for (int b = 0; b < n_backends_; b++) {
                if (backends_[b]->supports_op(node)) {
                    int n = count_supported_inputs(node, b);
                    if (n > best_supported) {
                        best_supported = n;
                        best_backend   = b;
                    }
                }
            }
            //*node_id = best_backend;
            backend_map_[node] = best_backend;
        }
        // else:
        // 已分配：可以考虑升级到更高优先级的后端
        // 简化：跳过升级逻辑
    }
}

// ============================================================
// 获取分裂后的子图
// ============================================================

ComputeGraph * BackendScheduler::get_split(int i) {
    if (i < 0 || i >= n_splits_) return nullptr;
    return splits_[i];
}

} // namespace rfaa
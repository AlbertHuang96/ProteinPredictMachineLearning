
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


// ===== 辅助：根据 backend assignment 预留各后端内存 =====
bool BackendScheduler::reserve_graph_memory() {
    // 1. 收集每个后端需要分配的 tensor 总大小
    std::vector<size_t> backend_mem_needed(n_backends_, 0);

    for (int i = 0; i < current_graph_->n_nodes(); i++) {
        Tensor* node = current_graph_->node(i);
        int backend_id = tensor_backend_id(node, 0);
        backend_mem_needed[backend_id] += node->nbytes();
    }

    for (int i = 0; i < current_graph_->n_leafs(); i++) {
        Tensor* leaf = current_graph_->leaf(i);
        int backend_id = tensor_backend_id(leaf, 0);
        backend_mem_needed[backend_id] += leaf->nbytes();
    }

    // 2. 更新 node_backend_ids / leaf_backend_ids
    node_backend_ids_.resize(current_graph_->n_nodes());
    for (int i = 0; i < current_graph_->n_nodes(); i++) {
        node_backend_ids_[i] = tensor_backend_id(current_graph_->node(i), 0);
    }

    leaf_backend_ids_.resize(current_graph_->n_leafs());
    for (int i = 0; i < current_graph_->n_leafs(); i++) {
        leaf_backend_ids_[i] = tensor_backend_id(current_graph_->leaf(i), 0);
    }

    // 3. 更新 bufts（缓存 buffer types）
    bufts_.resize(n_backends_);
    for (int i = 0; i < n_backends_; i++) {
        bufts_[i] = backends_[i]->buffer_type();
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
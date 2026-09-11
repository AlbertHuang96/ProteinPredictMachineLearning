
#include "ppml/Backend.h"
#include "ppml/Context.h"
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cuda_runtime.h>   // cudaMemGetInfo 用于 GPU 显存预算

namespace ppml {

// 查询当前 CUDA 设备剩余空闲显存（字节）；无 GPU 返回 0。
static size_t cuda_free_vram_bytes() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) return 0;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return free_b;
}

// ============================================================
// 构造/析构
// ============================================================

BackendScheduler::BackendScheduler() {
    splits_.clear();  // 动态 vector，无需 memset
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
    // OP_PERMUTE/OP_TRANSPOSE 是真数据重排（kernel 读写 dst），不是零拷贝视图：
    // 若当 view 跳过，split 范围仍会 dispatch 它们，CPU 会读写 GPU buffer → SIGSEGV。
    // 因此只保留 OP_VIEW/OP_RESHAPE（零拷贝共享）为 view op。
    return op == OP_VIEW || op == OP_RESHAPE;
}

// host producer 判定：数据实际在 host 的节点（参数 param / 常量 / 已绑定 host data / 共享 host 的
// view），绝不能分配到 GPU 后端：
//   - CUDABackend::supports_op 对 OP_NONE 返回 true，会被 pass_fill_unassigned 贪心放到 GPU；
//   - 但 CUDA dispatch 对 OP_NONE 直接 skip（不 dispatch、不做 H2D），其 data() 保持 host；
//   - 非 OP_NONE 但有 host data（如 view 共享的 host 源、已绑定 host 数据的节点）被分到 GPU 时，
//     build_splits 因与消费者同属 GPU 后端不建 cpy → GPU kernel 裸读 host 指针 → [CUDA-ERR]。
// 强制回落 CPU 后，GPU 消费者会经 build_splits 建 cpy（H2D），数值正确。
// 沿 view_src 链解析到底层：若底层是 OP_NONE 参数/常量，或持有 host data（buffer 为 null/host），
// 判定为 host 生产者。已显式挂 device buffer（非 host）的节点数据已在 device，不算。
bool BackendScheduler::node_is_host_producer(TensorF32* node) const {
    if (!node) return false;
    // 沿 view_src 链找底层数据载体（view 共享源数据，data 在源头）
    const TensorF32* base = node;
    int guard = 0;
    if (getenv("GRAPH_DEBUG_SCHED")) {
        fprintf(stderr, "[nhip] node=%p op=%d view_src=%p buffer_=%p data=%p CALLER=%p\n",
                (void*)node, (int)node->op, (void*)node->view_src,
                (void*)node->buffer_, (void*)node->data(),
                __builtin_return_address(0));
    }
    while (base->view_src && guard++ < 64) base = base->view_src;
    if (getenv("GRAPH_DEBUG_SCHED") && base != node) {
        fprintf(stderr, "[nhip]   base=%p op=%d buffer_=%p data=%p\n",
                (void*)base, (int)base->op, (void*)base->buffer_, (void*)base->data());
    }
    if (base->op == OP_NONE) return true;  // 参数/常量
    if (base->data() != nullptr) {
        if (base->buffer_ && !base->buffer_->is_host()) return false;  // device data
        return true;  // host data
    }
    return false;
}

void BackendScheduler::set_backend_if_supported(TensorF32* node, int backend_id) {
    // host 生产者（OP_NONE 参数/常量）不得放 GPU：数据在 host，dispatch 跳过它，
    // 也不会建 cpy（与消费者同后端），GPU kernel 会读 host 指针。
    if (node_is_host_producer(node)) return;

    Backend * backend = backends_[backend_id];
    if (!backend->supports_op(node)) return;

    // GPU 后端：受显存预算约束——预算不足时拒绝，留待回落 CPU。
    if (backend->priority() > 0) {
        if (gpu_assign_if_affordable(node, backend_id)) {
            backend_map_[node] = backend_id;
        }
        // 拒绝时不写入 backend_map_，后续 pass_fill_unassigned 会回落 CPU。
        return;
    }

    backend_map_[node] = backend_id;
}

bool BackendScheduler::gpu_assign_if_affordable(TensorF32* node, int gpu_backend_id) {
    // 获取该 GPU 后端的 buffer type（估算分配大小用；bufts_ 在 reserve 阶段才填充，
    // 因此这里直接从 backend 取）。
    const BufferType* bt = (gpu_backend_id < n_backends_)
                               ? backends_[gpu_backend_id]->buffer_type() : nullptr;

    // 无预算（0=不限/未启用）或该节点已有独立 device buffer（data()!=nullptr 非复用）→ 直接放行。
    if (gpu_vram_budget_ == 0 || node->data() != nullptr) {
        if (node->data() == nullptr && bt) {
            // 计入预算（node 尚未绑定 device buffer）
            gpu_reserved_bytes_ += GGML_PAD(bt->get_alloc_size(node), bt->get_alignment());
        }
        return true;
    }

    // 估算该节点在 GPU 上的分配大小
    size_t est = 0;
    if (bt) {
        est = GGML_PAD(bt->get_alloc_size(node), bt->get_alignment());
    }

    // 预算不足：拒绝放 GPU（回落 CPU）
    if (gpu_reserved_bytes_ + est > gpu_vram_budget_) {
        return false;
    }

    gpu_reserved_bytes_ += est;
    return true;
}

int BackendScheduler::count_supported_inputs(TensorF32* node, int backend_id) const {
    int count = 0;
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        TensorF32* src = node->src[j];
        if (!src) continue;

        // 输入已经分配了后端 且 buffer 兼容
        auto it = backend_map_.find(src);
        if (it != backend_map_.end() && tensor_buffer_compatible(src, backend_id)) {
            count++;
        }
    }
    return count;
}

bool BackendScheduler::tensor_buffer_compatible(const TensorF32* src, int backend_id) const {
    auto it = backend_map_.find(src);
    if (it == backend_map_.end()) return false;

    int src_backend = it->second;
    Backend * target = backends_[backend_id];

    // 同一个后端 → 兼容
    if (src_backend == backend_id) return true;

    // 关键：host 数据（buffer_==null 或 host buffer）的源，若目标是 GPU（非 host）后端，
    // 不能直接读——kernel 需要 device 指针，必须走 H2D 拷贝。
    bool src_host = (src->buffer_ == nullptr) || src->buffer_->is_host();
    bool target_host = target->buffer_type()->is_host();
    if (src_host && !target_host) return false;

    // 跨后端：检查 target 是否支持 src 的内存 buffer 类型
    Backend * source = backends_[src_backend];
    return target->supports_buffer_type(source->buffer_type());
}

int BackendScheduler::tensor_backend_id(TensorF32* t) const {
    auto it = backend_map_.find(t);
    return (it != backend_map_.end()) ? it->second : -1;
}

int BackendScheduler::tensor_backend_id(TensorF32* t, int default_id) const {
    int id = tensor_backend_id(t);
    return (id == -1) ? default_id : id;
}

// ============================================================
// alloc_splits — 检测 backend 变化并重新分配图内存
// 对标 ggml_backend_sched_alloc_splits
// ============================================================

bool BackendScheduler::alloc_splits() {
    if (!current_graph_) return false;

    // ===== Step 1: 检查 backend IDs 是否发生变化 =====
    // 设计意图：同一个 graph 多次执行时（如训练循环），
    // 如果 split_graph 后的 backend 分配与上次一致，则复用 buffer，避免重分配。
    bool backend_ids_changed = false;

    // 检查节点
    if (!prev_node_backend_id_.empty()) {
        for (int i = 0; i < current_graph_->n_nodes(); i++) {
            TensorF32* node = current_graph_->graph_node(i);
            // cur_id is the i-th node in the graph currently
            int cur_id = tensor_backend_id(node, 0);
            // prev_id is the i-th node recorded last time
            int prev_id = (i < (int)prev_node_backend_id_.size()) ? prev_node_backend_id_[i] : -1;
            if (cur_id != prev_id) {
                // check the backend id of the i-th node last time and this time
                backend_ids_changed = true;
                break;
            }
        }
    }

    // 检查叶子
    if (!backend_ids_changed && !prev_leaf_backend_id_.empty()) {
        for (int i = 0; i < current_graph_->n_leafs(); i++) {
            TensorF32* leaf = current_graph_->graph_leaf(i);
            int cur_id = tensor_backend_id(leaf, 0);
            int prev_id = (i < (int)prev_leaf_backend_id_.size()) ? prev_leaf_backend_id_[i] : -1;
            if (cur_id != prev_id) {
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
            return false;
        }

        // 保存当前分配作为"上一次"快照，供下次对比
        // prev_node_backend_id_ is the mapping of node -> backend id that recorded last time
        prev_node_backend_id_ = node_backend_id_;
        prev_leaf_backend_id_ = leaf_backend_id_;
        graph_reserved_ = true;
    }

    return true;
}


// ============================================================
// alloc_buffer — 对标 ggml_backend_buft_alloc_buffer
// ============================================================
Buffer* alloc_buffer(BufferType* buft, size_t size, BufferUsage usage) {
    if (!buft) return nullptr;

    // 诊断（2026-08-29）：打印每次 buffer 分配的大小与类型。
    // 触发环境变量: PPML_DEBUG_ALLOC=1（打印全部分配）;
    // 始终打印 >=1GB 的大分配（定位 OOM/预留失败）。
    const char* buf_name = buft->get_name();
    const size_t size_gb = size / (1024ull * 1024ull * 1024ull);
    if (getenv("PPML_DEBUG_ALLOC") != nullptr || size >= (1024ull*1024ull*1024ull)) {
        // 2026-09-11：附带 usage —— 区分"参数搬迁 buffer（WEIGHTS）"vs"gallocr 峰值/跨后端
        //   cpy（COMPUTE）"，否则日志里只有 type=CPU，无法判断 168.9GB 出自哪条路径。
        const char* usage_name = (usage == BufferUsage::WEIGHTS) ? "WEIGHTS" :
                                 (usage == BufferUsage::COMPUTE) ? "COMPUTE" : "STORAGE";
        fprintf(stderr, "[alloc-buffer] %-20s usage=%-8s size=%.2f GB (%zu bytes)\n",
                buf_name, usage_name, (double)size / (1024.0*1024.0*1024.0), size);
        if (size_gb > 40) {
            fprintf(stderr, "[alloc-buffer] ⚠️ 超大分配(>40GB) type=%s usage=%s size=%.2f GB\n",
                    buf_name, usage_name, (double)size / (1024.0*1024.0*1024.0));
        }
    }

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
        TensorF32* t = graph->graph_node(i);

        // 跳过已分配或有 view_src 的 tensor
        if (t->data() != nullptr) {
            if (t->view_src == nullptr) {
                continue;  // 已独立分配
            }
            // else if (t->buffer_ == nullptr) {
            //     t->data_        = t->view_src->data_;
            //     t->buffer_      = t->view_src->buffer_;
            //     t->buffer_offs_ = t->view_src->buffer_offs_;
            // }
            continue;
        }

        if (t->view_src != nullptr) {
            // view tensor：不需要新内存，指向源
            // if (t->buffer_ == nullptr) {
            //     t->data_        = t->view_src->data_;
            //     t->buffer_      = t->view_src->buffer_;
            //     t->buffer_offs_ = t->view_src->buffer_offs_;
            // }
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
        TensorF32* t = graph->graph_leaf(i);

        // leaf 通常已经预分配（input/param），跳过
        if (t->data() != nullptr) continue;
        if (t->view_src != nullptr) {
            /* if (t->buffer_ == nullptr) {
                t->data_        = t->view_src->data_;
                t->buffer_      = t->view_src->buffer_;
                t->buffer_offs_ = t->view_src->buffer_offs_;
            } */
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
        TensorF32* t = graph->graph_node(i);

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
        TensorF32* t = graph->graph_leaf(i);
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

// ============================================================
// is_device_pointer — 判定 data 指针是否指向 device（GPU）内存
// 有 buffer_ 时直接看 is_host()（可靠）；无 buffer_（裸 device 指针，
// 理论上不应出现在 scheduler 内）时用 4 字节 D2H 探测兜底：
//   cudaMemcpy(DeviceToHost) 对 device 指针成功、对 host 指针返回
//   cudaErrorInvalidDevicePointer。这是最可靠的判定（cudaPointerGetAttributes
//   对某些上下文/指针会失败而被误判为 host）。
// ============================================================
bool is_device_pointer(const TensorF32* src, const float* data) {
    if (!data) return false;
    if (src->buffer_) return !src->buffer_->is_host();
    float probe = 0.0f;
    cudaError_t err = cudaMemcpy(&probe, data, sizeof(float), cudaMemcpyDeviceToHost);
    //    说明 data 是 host）。但该失败会残留在 CUDA 错误状态，被后续任意 cudaGetLastError
    //    捕获 → 误报后续 kernel（如 OP_SCATTER_ADD op=108）invalid argument。
    //    这里必须清掉探测产生的错误，否则污染错误状态（混合训练 [CUDA-ERR] op=108 真凶）。
    if (err != cudaSuccess) {
        cudaGetLastError();  // 清掉探测失败的残留错误
    }
    return (err == cudaSuccess);
}

// ============================================================
// backend_tensor_copy — 对标 ggml_backend_tensor_copy
// 将 src tensor 的数据拷贝到 dst tensor（跨后端自动处理 CPU↔GPU）
// ============================================================
bool backend_tensor_copy(const TensorF32* src, TensorF32* dst) {
    if (src == dst) return true;

    // view 零拷贝共享源数据：src 本身 data() 为 null（数据在 view_src 链底层）。
    // 跨后端拷贝时必须从底层数据载体读，否则 set_tensor 从 null 拷贝 → 崩溃/垃圾。
    const TensorF32* real_src = src;
    {
        const TensorF32* cur = src;
        int guard = 0;
        while (cur && cur->view_src && guard++ < 64) cur = cur->view_src;
        real_src = cur;  // 底层数据载体
    }
    src = real_src;

    size_t nbytes = src->nbytes();

    // ===== Case 1: src 在 host buffer 上（或没有 buffer）=====
    if (!src->buffer_ || src->buffer_->is_host()) {
        // 防御性校验：src 无 buffer 但确实是 device 指针（如裸 cudaMalloc 张量）
        // 是非法状态——此处若直接当 host 做 memcpy/H2D 会 UB。配合 Gallocr::release
        // 已保证 scheduler 内 device 张量必有 buffer_，故正常路径不会命中。
        if (!src->buffer_ && src->data() &&
            is_device_pointer(src, static_cast<const float*>(src->data()))) {
            fprintf(stderr,
                    "[backend_tensor_copy][BUG] src tensor=%p has device data but no "
                    "buffer_ — 缺 buffer_ 元数据，无法安全拷贝\n", (void*)src);
            return false;
        }
        if (dst->buffer_) {
            dst->buffer_->set_tensor(const_cast<TensorF32*>(dst),
                                     src->data(), dst->buffer_offs_, nbytes);
            // 诊断（GRAPH_DEBUG_CROSSBK）：H2D 拷贝错误检查——scatter 前的 invalid argument
            //   真凶常在这里（set_tensor 的 cudaMemcpy 不查返回值，错误残留被后续捕获）。
            if (getenv("GRAPH_DEBUG_CROSSBK")) {
                cudaError_t he = cudaGetLastError();
                if (he != cudaSuccess) {
                    fprintf(stderr,
                            "[h2d-ERR] src=%p src_data=%p src_numel=%lld cpy_data=%p cpy_offs=%zu "
                            "cpy_buf=%p buf_size=%zu nbytes=%zu: %s\n",
                            (const void*)src, (const void*)src->data(), (long long)src->numel(),
                            (const void*)dst->data(), (size_t)dst->buffer_offs_,
                            (const void*)dst->buffer_, dst->buffer_ ? dst->buffer_->size() : 0,
                            nbytes, cudaGetErrorString(he));
                    cudaGetLastError();  // 清错误
                }
            }
        } else {
            std::memcpy(dst->data(), src->data(), nbytes);
        }
        return true;
    }

    // ===== Case 2: dst 在 host buffer 上（或没有 buffer）=====
    if (!dst->buffer_ || dst->buffer_->is_host()) {
        if (src->buffer_) {
            src->buffer_->get_tensor(src, dst->data(), src->buffer_offs_, nbytes);
        } else {
            std::memcpy(dst->data(), src->data(), nbytes);
        }
        return true;
    }

    // ===== Case 3: src 和 dst 都在 device buffer 上 =====
    // 尝试直接 cpy_tensor（GPU→GPU cudaMemcpyDeviceToDevice）
    if (dst->buffer_->cpy_tensor(src, const_cast<TensorF32*>(dst),
                                  src->buffer_offs_, dst->buffer_offs_, nbytes)) {
        return true;
    }

    // ===== Fallback: 通过 staging buffer（malloc → get → set → free）=====
    void* staging = std::malloc(nbytes);
    if (!staging) return false;

    src->buffer_->get_tensor(src, staging, src->buffer_offs_, nbytes);
    dst->buffer_->set_tensor(const_cast<TensorF32*>(dst), staging, dst->buffer_offs_, nbytes);
    std::free(staging);
    return true;
}

// ===== 辅助：根据 backend assignment 预留各后端内存 =====
// 改用 Gallocr 做延迟分配 + 空间复用：每个后端 buffer 只开峰值大小，
// 张量沿拓扑序"借/还"复用，替代原来的全量常驻 + bump 分配。
bool BackendScheduler::reserve_graph_memory() {
    // ---- needs_realloc 接入（2026-08-31，移植 ggml alloc_graph 懒重建语义）----
    // 上次 reserve 布局仍有效（图结构/张量大小未变）→ 不 release、不清 reserved_buffers_、
    // 不重建：直接复用现有 gallocr buffer（alloc 内部走 alloc_reuse 只重绑 data）。
    // 旧逻辑（无条件 release + reserve + alloc）保留在下方 else 分支。
    // 注意：compute_and_read 的增量 build_forward_expand 每次图在变（n_nodes 增长）
    //   → needs_realloc 恒 true → 走重建路径，行为与接入前完全一致（无回归）；
    //   纯训练（同图反复 compute）才走复用路径——正是要优化的跨调用 buffer 存活场景。
    if (gallocr_.can_reuse(current_graph_)) {
        // 复用路径：仍重设 buft 注入（防御 buft 指针变化），然后直接 alloc（走复用分支）。
        for (int b = 0; b < n_backends_; b++) {
            gallocr_.backends()[b].buft = const_cast<BufferType*>(bufts_[b]);
        }
        auto backend_id_of_reuse = [&](TensorF32* t) -> int {
            return tensor_backend_id(t, n_backends_ - 1);
        };
        if (!gallocr_.alloc(current_graph_, backend_id_of_reuse, n_backends_)) {
            return false;
        }
        // 跨后端拷贝节点（步骤7）的持久 buffer 在 reserved_buffers_ 中，布局未变 → 仍存活。
        return true;
    }

    // 释放上次分配的 buffer
    reserved_buffers_.clear();
    // 必须先 release 上一轮 gallocr buffer：Gallocr::alloc 每次 push 新 buffer 到
    //    ba.buffers（只有 release() 才 delete）。混合多样本下每个样本都走 reserve+alloc，
    //    若不先释放，旧 buffer 永不回收 → RSS 每样本 +5~6GB 单调增长直至 OOM。
    //    实测（2026-08-25）：纯 CPU 多样本 RSS 稳定 ~4.9GB；混合模式每样本 +5GB。
    gallocr_.release();

    // 1. 更新 node_backend_id_ / leaf_backend_id_
    //  2026-08-24 修复：默认值必须与 build_splits 一致（n_backends_-1=CPU），不能用 0(GPU)。
    //    backend_map_ 里没有的节点（view 类 op 在 build_splits 被 is_view_op 跳过、从不填
    //    backend_map_）若此处默认 GPU，gallocr 会在 GPU buffer 分配它，但 build_splits 默认
    //    CPU 归入 CPU split → CPU kernel 写无效 dst->data()（GPU buffer/未分配）→ SIGSEGV。
    //    崩溃实证：op=98(OP_TRANSPOSE) dst{data=nil buf=nil}，epoch3 跨后端混合模式下段错误。
    node_backend_id_.resize(current_graph_->n_nodes());
    for (int i = 0; i < current_graph_->n_nodes(); i++) {
        node_backend_id_[i] = tensor_backend_id(current_graph_->graph_node(i), n_backends_ - 1);
    }

    leaf_backend_id_.resize(current_graph_->n_leafs());
    for (int i = 0; i < current_graph_->n_leafs(); i++) {
        leaf_backend_id_[i] = tensor_backend_id(current_graph_->graph_leaf(i), n_backends_ - 1);
    }

    // 2. 更新 bufts（缓存 buffer types）
    bufts_.resize(n_backends_);
    for (int i = 0; i < n_backends_; i++) {
        bufts_[i] = backends_[i]->buffer_type();
    }

    // 3. 配置 Gallocr：注入每个后端的 buft
    gallocr_.set_n_backends(n_backends_);
    for (int b = 0; b < n_backends_; b++) {
        gallocr_.backends()[b].buft = const_cast<BufferType*>(bufts_[b]);
    }

    // 4. 张量 → 后端 id 映射
    //  2026-08-24 修复：默认 CPU（n_backends_-1），与 build_splits 的默认值一致。
    //    默认 GPU(0) 会让 backend_map_ 未覆盖的 view 节点在 gallocr 分配到 GPU buffer，
    //    而 build_splits 把它们归 CPU split → CPU kernel 写无效指针段错误。
    auto backend_id_of = [&](TensorF32* t) -> int {
        return tensor_backend_id(t, n_backends_ - 1);
    };

    // 2026-09-06 注入跨后端 producer 保活：build_splits 创建的 OP_DUP cpy 不入图 nodes，
    //   gallocr 统计不到 producer 的跨后端消费 → refcount 低估 → 算完即 free 被覆盖 →
    //   graph_compute Step1 D2H 读垃圾（混合 chi loss 3021 根因）。注入 copy_tensor_map_
    //   全部 src，compute_refcounts 中标 is_output（不释放/不复用）。
    if (!copy_tensor_map_.empty()) {
        std::vector<const TensorF32*> xbk_srcs;
        xbk_srcs.reserve(copy_tensor_map_.size());
        for (auto& kv : copy_tensor_map_) {
            const TensorF32* s = kv.first.first;
            if (s) xbk_srcs.push_back(s);
        }
        gallocr_.set_pinned_srcs(xbk_srcs);
    }

    // 5. Phase1：计算各后端峰值
    if (!gallocr_.reserve(current_graph_, backend_id_of, n_backends_)) {
        return false;
    }

    // 6. Phase2：分配 buffer 并绑定张量 data_/buffer_/buffer_offs_
    if (!gallocr_.alloc(current_graph_, backend_id_of, n_backends_)) {
        return false;
    }

    // 7. 跨后端拷贝节点（build_splits 创建的 dup(src)）单独持久分配，
    //    数量少、体积小，不参与复用（保持简单正确）。
    for (int b = 0; b < n_backends_; b++) {
        const BufferType* buft = bufts_[b];
        if (!buft) continue;

        size_t copy_size = 0;
        for (auto& kv : copy_tensor_map_) {
            TensorF32* cpy = kv.second;
            int cpy_backend_id = kv.first.second;  // target_backend_id
            if (cpy_backend_id == b &&
                cpy->data() == nullptr && cpy->view_src == nullptr) {
                copy_size += GGML_PAD(buft->get_alloc_size(cpy), buft->get_alignment());
            }
        }
        if (copy_size == 0) continue;

        Buffer* buf = alloc_buffer(const_cast<BufferType*>(buft), copy_size);
        if (!buf) return false;
        reserved_buffers_.emplace_back(buf);

        TensorAllocator tallocr(buf);
        for (auto& kv : copy_tensor_map_) {
            TensorF32* cpy = kv.second;
            int cpy_backend_id = kv.first.second;
            if (cpy_backend_id == b &&
                cpy->data() == nullptr && cpy->view_src == nullptr) {
                if (!tallocr.alloc(cpy)) return false;
            }
        }
    }

    return true;
}


// ============================================================
// 图备份/恢复（2026-09-01，回退路径）
//   backup_graph_nodes: split_graph 前保存每个 node/leaf 的 src 引用与 data/buffer/offs。
//   restore_graph_nodes: graph_compute 失败回退 CPU 前恢复原图，撤销 build_splits 污染
//     （node->src 被替换成未分配 cpy、节点被 bind 到 device buffer）。
//   备份的是指针快照（浅拷贝），node/leaf 对象本身不重建——它们属于 context 生命周期，
//   build_splits 替换的 src 指向新建的 cpy 节点（context arena），恢复时把 src 指回原对象。
// ============================================================
void BackendScheduler::backup_graph_nodes(ComputeGraph * graph) {
    graph_backup_.clear();
    const int n = graph->n_nodes() + graph->n_leafs();
    graph_backup_.reserve((size_t)n);
    for (int i = 0; i < graph->n_leafs(); i++) {
        TensorF32* t = graph->graph_leaf(i);
        GraphNodeBackup b;
        b.node = t;
        for (int s = 0; s < GGML_MAX_SRC; s++) b.src[s] = t->src[s];
        b.data   = t->data();
        b.buffer = t->buffer_;
        b.offs   = t->buffer_offs_;
        graph_backup_.push_back(b);
    }
    for (int i = 0; i < graph->n_nodes(); i++) {
        TensorF32* t = graph->graph_node(i);
        GraphNodeBackup b;
        b.node = t;
        for (int s = 0; s < GGML_MAX_SRC; s++) b.src[s] = t->src[s];
        b.data   = t->data();
        b.buffer = t->buffer_;
        b.offs   = t->buffer_offs_;
        graph_backup_.push_back(b);
    }
    graph_backup_valid_ = true;
}

void BackendScheduler::restore_graph_nodes() {
    if (!graph_backup_valid_) return;
    for (const GraphNodeBackup& b : graph_backup_) {
        if (!b.node) continue;
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (b.node->src[s] != b.src[s]) b.node->src[s] = b.src[s];
        }
        b.node->bind_data(b.data);
        b.node->buffer_      = b.buffer;
        b.node->buffer_offs_ = b.offs;
    }
    graph_backup_valid_ = false;
}

// ============================================================
// split_graph — 三趟扫描
// ============================================================

void BackendScheduler::split_graph(ComputeGraph * graph) {
    // 2026-09-01：切分会污染原图（node->src 替换成 cpy、bind device buffer），
    // 先备份以便 graph_compute 失败时 restore 后回退 CPU（避免 CPU 也算不出）。
    backup_graph_nodes(graph);

    current_graph_ = graph;  // 保存当前图引用，供 alloc_splits 使用
    n_splits_ = 0;
    splits_.clear();
    n_graph_inputs_ = 0;
    backend_map_.clear();
    copy_tensor_map_.clear();

    // 显存预算：GPU 空闲显存扣 20% 余量作预算；0 = 不限（无 GPU 或显存探测失败）。
    gpu_reserved_bytes_ = 0;
    if (n_backends_ > 1) {
        // 仅当存在非 CPU 后端（CUDA）时启用预算
        bool has_gpu = false;
        for (int b = 0; b < n_backends_; b++) {
            if (backends_[b]->priority() > 0) { has_gpu = true; break; }
        }
        // 可被环境变量覆盖（测试/调参用）：PPML_GPU_BUDGET_MB=0 表示不限。
        const char* budget_mb = std::getenv("PPML_GPU_BUDGET_MB");
        if (has_gpu && budget_mb && atoi(budget_mb) > 0) {
            gpu_vram_budget_ = (size_t)atoi(budget_mb) * 1024ULL * 1024ULL;
        } else {
            gpu_vram_budget_ = has_gpu ? (size_t)(cuda_free_vram_bytes() * 4ull / 5ull) : 0;
        }
    } else {
        gpu_vram_budget_ = 0;
    }

    // ============================
    // Pass 1: 叶子节点分配
    // ============================
    pass_assign_leafs(graph);

    // ============================
    // Pass 2: 扩展分配（向下 + 向上两遍）
    // ============================
    pass_expand_assignments(graph);

    // ============================
    // Pass 3: 填充未分配节点
    // ============================
    pass_fill_unassigned(graph);

    // ============================
    // Pass 5: 切分子图 + 创建跨后端拷贝节点
    // ============================
    build_splits(graph);
}

void BackendScheduler::pass_assign_leafs(ComputeGraph * graph) {
    for (int i = 0; i < graph->n_leafs(); i++) {
        TensorF32* leaf = graph->graph_leaf(i);

        // 用户已经手动指定 → 不覆盖
        if (tensor_backend_id(leaf) != -1) continue;

        // 自动分配：找支持该 tensor buffer 的后端
        // 优先 GPU（高 priority），其次 CPU
        // CPU backend 的 supports_buffer_type 对任何 host buffer 返回 true
        // 关键：host 数据叶子（buffer_==null 或 host buffer）必须放 CPU——
        // 否则 GPU 节点消费它时 src_backend==GPU 不会触发 H2D 拷贝，kernel 会把 host 指针当 device 读。
        for (int b = 0; b < n_backends_; b++) {
            if (!backends_[b]->supports_buffer_type(backends_[b]->buffer_type())) continue;
            if (backends_[b]->priority() > 0) {
                // GPU 后端：host 数据叶子不放 GPU；GPU 内存叶子（buffer_ 已设 device）可放。
                bool host_data = (leaf->buffer_ == nullptr) || leaf->buffer_->is_host();
                if (host_data) continue;
                if (!gpu_assign_if_affordable(leaf, b)) continue;
            }
            backend_map_[leaf] = b;
            break;
        }
    }
}

void BackendScheduler::pass_expand_assignments(ComputeGraph * graph) {
    // ===== 向下扩展 =====
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes(); i++) {
            TensorF32* node = graph->graph_node(i);
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
            TensorF32* node = graph->graph_node(i);
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
        TensorF32* node = graph->graph_node(i);
        if (getenv("GRAPH_DEBUG_SCHED")) {
            // 诊断：检测栈/堆地址范围的 node，定位悬垂指针
            uintptr_t a = (uintptr_t)node;
            int in_stack = (a >= 0x700000000000ULL && a <= 0x800000000000ULL);
            fprintf(stderr,
                "[sched] node[%d]=%p op=%d ndim=%d numel=%lld flag=0x%x %s\n",
                i, (void*)node, (int)node->op, (int)node->shape().ndim(),
                (long long)node->numel(), (unsigned)node->flag,
                in_stack ? "<STACK?!>" : "");
        }
        if (is_view_op(node->op)) continue;

        auto it = backend_map_.find(node);
        if (it == backend_map_.end()) {
            // 未分配：优先最高 priority 且支持该 op 的后端（GPU 优先）。
            // 不再按"输入兼容数"贪心——host 叶子在 CPU 时会把节点拽回 CPU，
            // 违背"尽量放 GPU"的意图；跨后端拷贝由 build_splits 处理。
            int best_backend = n_backends_ - 1;  // 默认 CPU
            // host 生产者（OP_NONE 参数/常量）：数据在 host，dispatch 跳过它也不会建 cpy，
            // 若放 GPU 消费者会裸读 host 指针 → [CUDA-ERR] HOST/INVALID → 崩溃。强制回落 CPU，
            // 由 build_splits 为 GPU 消费者建 H2D cpy。
            if (node_is_host_producer(node)) {
                backend_map_[node] = best_backend;  // CPU
                continue;
            }
            if (getenv("GRAPH_DEBUG_SCHED")) {
                fprintf(stderr,
                    "[sched] pass_fill node=%p op=%d n_bk=%d best0=%d bk0=%p v0=%p bk1=%p v1=%p\n",
                    (void*)node, (int)node->op, n_backends_, best_backend,
                    (void*)(n_backends_>0?backends_[0]:nullptr),
                    (n_backends_>0 && backends_[0])?*(void**)backends_[0]:nullptr,
                    (void*)(n_backends_>1?backends_[1]:nullptr),
                    (n_backends_>1 && backends_[1])?*(void**)backends_[1]:nullptr);
            }
            for (int b = 0; b < n_backends_; b++) {
                if (getenv("GRAPH_DEBUG_SCHED")) {
                    fprintf(stderr, "[sched]   try b=%d node=%p bk=%p v=%p\n",
                            b, (void*)node,
                            (void*)backends_[b],
                            backends_[b] ? *(void**)backends_[b] : nullptr);
                }
                if (backends_[b]->supports_op(node)) {
                    best_backend = b;  // backends_ 已按 priority 降序 → 第一个即最高
                    break;
                }
            }
            // 若选中的是 GPU 且显存预算不足 → 回落 CPU（最后后端）。
            if (backends_[best_backend]->priority() > 0 &&
                !gpu_assign_if_affordable(node, best_backend)) {
                best_backend = n_backends_ - 1;  // CPU
            }

            // （已回滚 2026-08-24：OUT_PROD/OUTER_PROD_BACK 强制 CPU 引入更多跨后端 H2D，loss 更不稳。
            //   跨后端需系统性修 Step 1/1b 的 D2H/H2D buffer 生命周期，见 memory。）

            // ===== 混合训练正确性护栏（避免段错误）=====
            // 仅广播 op 强制回落 CPU：CUDA elemwise kernel 用扁平 idx<n 索引，
            // 不支持广播（src numel != dst numel 越界）。host 源不再整节点禁用 GPU，
            // 改由 build_splits 插入 H2D 拷贝 + graph_compute 兜底处理。
            if (backends_[best_backend]->priority() > 0) {
                // (1) 广播检测：任意 op 只要存在 src numel != dst numel，
                //     CUDA kernel 假设扁平等长索引 -> 越界读/写 -> 段错误。强制 CPU。
                bool broadcast = false;
                {
                    long long dn = (long long)node->numel();
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        TensorF32* s = node->src[j];
                        if (s && (long long)s->numel() != dn) { broadcast = true; break; }
                    }
                }
                // (2) 广播检测：只有"真·逐元素"op 在 src numel != dst numel 时才构成
                //     广播（CUDA elemwise kernel 用扁平 idx<n 索引，会越界）。
                //     mul_mat / out_prod / norm / softmax / concat 等 op 的 src numel
                //     天然 ≠ dst numel（收缩/归约），属正常语义，不是广播。
                static const int kElemwiseBroadcastOps[] = {
                    OP_ADD, OP_ADD1, OP_ADD_ID, OP_SUB, OP_MUL, OP_DIV,
                    OP_SCALE, OP_SUM, OP_MEAN, OP_SQR, OP_SQRT, OP_LOG,
                    OP_SIN, OP_COS, OP_LEAKY_RELU, OP_UNARY, OP_CLAMP,
                    OP_FILL, OP_ARANGE, OP_GLU, OP_TRI_MUL, OP_OUTER_PROD_MEAN,
                    OP_OUTER_PROD,
                    -1
                };
                bool is_elemwise = false;
                for (int k = 0; kElemwiseBroadcastOps[k] >= 0; k++) {
                    if (node->op == (tensor_op)kElemwiseBroadcastOps[k]) { is_elemwise = true; break; }
                }
                bool real_broadcast = broadcast && is_elemwise;
                // (3) host 数据传递：
                //     历史护栏曾把"任一 src 沿视图链是 host"的节点整节点强制 CPU，
                //     导致叶子全 host 时所有节点都被拽回 CPU → GPU 0 算子。
                //     正确做法：host 源由 build_splits 插入 H2D 拷贝解决，不在此禁用 GPU。
                //     仅当 GPU 后端对该 op 显式不支持（supports_op 已过滤）时才回落 CPU。
                if (real_broadcast) {
                    if (getenv("GRAPH_DEBUG_SCHED")) {
                        fprintf(stderr,
                                "[sched] force CPU: op=%d (broadcast=1) — "
                                "CUDA elemwise kernel has no broadcast support (idx<n index)\n",
                                (int)node->op);
                    }
                    best_backend = n_backends_ - 1;  // CPU
                }
            }
            backend_map_[node] = best_backend;
            // 诊断（GRAPH_DEBUG_SCHED=1）：统计每 backend 分配的 op 数，检测 GPU 是否被分配到 op
            if (getenv("GRAPH_DEBUG_SCHED") && best_backend < 4) {
                if (sched_backend_cnt_ == nullptr) sched_backend_cnt_ = new int[4]();
                if (sched_backend_op_last_ == nullptr) sched_backend_op_last_ = new long[4]();
                sched_backend_cnt_[best_backend]++;
                sched_backend_op_last_[best_backend] = (int)node->op;
            }
        }
        // else:
        // 已分配：可以考虑升级到更高优先级的后端
        // 简化：跳过升级逻辑
    }

    // 诊断（GRAPH_DEBUG_SCHED=1）：打印每 backend 分配的 op 数（检测 GPU 是否被调用）
    if (getenv("GRAPH_DEBUG_SCHED") && sched_backend_cnt_ != nullptr) {
        int n_cpu = sched_backend_cnt_[0];
        long gpu_total = 0;
        for (int b = 1; b < n_backends_; b++) gpu_total += sched_backend_cnt_[b];
        fprintf(stderr, "[sched] split_graph: total_op_cpu=%d total_op_gpu=%lld (gpu_backends=%d)\n",
                n_cpu, (long long)gpu_total, n_backends_ - 1);
        for (int b = 1; b < n_backends_; b++) {
            fprintf(stderr, "[sched]   backend=%d op_count=%d last_op=%lld\n",
                    b, sched_backend_cnt_[b], sched_backend_op_last_[b]);
        }
    }
}

// ============================================================
// build_splits — Pass 5: 按 backend 边界切分子图 + 创建跨后端拷贝节点
// 对标 ggml_backend_sched_split_graph 的 pass 5
// ============================================================
void BackendScheduler::build_splits(ComputeGraph* graph) {
    int n_nodes = graph->n_nodes();

    // ===== Step 1: 跳过开头的 view op，确定第一个 split 的 backend =====
    int i = 0;
    for (; i < n_nodes; i++) {
        TensorF32* node = graph->graph_node(i);
        if (!is_view_op(node->op)) break;
    }
    if (i >= n_nodes) return;  // 全是 view op

    // ===== Step 2: 创建第一个 split =====
    splits_.resize(1);
    SplitInfo* split = &splits_[0];
    split->backend_id = tensor_backend_id(graph->graph_node(i), 0);
    split->i_start    = 0;
    split->n_inputs   = 0;
    int cur_backend_id = split->backend_id;

    // ===== Step 3: 遍历所有节点，切分 =====
    for (; i < n_nodes; i++) {
        TensorF32* node = graph->graph_node(i);
        if (is_view_op(node->op)) continue;

        //   原 default_id=0(GPU) 会把未分配的反向 op（如 OP_OUTER_PROD_MEAN_BACK，CUDA
        //   supports_op=false）当 GPU → 在 GPU split dispatch → CUDA 无 kernel 却走默认
        //   launch 路径 → buffer_offs 巨大 → invalid argument → 偶发 nan（msa 梯度缺失）。
        //   默认 CPU 安全：任何未分配节点都能被 CPU 后端执行。
        int node_backend_id = tensor_backend_id(node, n_backends_ - 1);

        // ---- 3a. 判断是否需要开新 split ----
        bool need_new_split = false;

        if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                TensorF32* src = node->src[j];
                if (!src) continue;

                // 条件 A: weight 在不可兼容的 backend 上
                if (src->buffer_ != nullptr) {
                    int src_backend_id = tensor_backend_id(src, -1);
                    if (src_backend_id != -1 &&
                        src_backend_id != cur_backend_id &&
                        !tensor_buffer_compatible(src, cur_backend_id)) {
                        need_new_split = true;
                        break;
                    }
                }

                // 条件 B: split 输入数达到上限
                if (split->n_inputs >= MAX_SPLIT_INPUTS) {
                    int src_backend_id = tensor_backend_id(src, -1);
                    CopyKey key = {src, cur_backend_id};
                    if (src_backend_id != cur_backend_id &&
                        copy_tensor_map_.find(key) == copy_tensor_map_.end() &&
                        !tensor_buffer_compatible(src, cur_backend_id)) {
                        need_new_split = true;
                        break;
                    }
                }
            }
        }

        // ---- 3b. backend 变了 或 需要新 split → 切分 ----
        if (node_backend_id != cur_backend_id || need_new_split) {
            split->i_end = i;
            n_splits_++;

            // 动态扩容（不再受固定 MAX_SPLITS 限制）
            splits_.resize(static_cast<size_t>(n_splits_) + 1);
            split = &splits_[n_splits_];
            split->backend_id = node_backend_id;
            split->i_start    = i;
            split->n_inputs   = 0;
            cur_backend_id    = node_backend_id;
        }

        // ---- 3c. 处理跨后端输入：创建拷贝节点 + 替换 node->src[j] ----
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            TensorF32* src = node->src[j];
            if (!src) continue;

            // 解析 src 的后端：view op 节点在 pass 中被跳过(backend_map_=-1)，
            // 需沿 src 链解析到底层非 view 节点的 backend。
            int src_backend_id = tensor_backend_id(src, -1);
            {
                TensorF32* cur = src;
                int guard = 0;
                while ((src_backend_id == -1) && cur && guard++ < 64) {
                    if (!is_view_op(cur->op)) break;
                    cur = cur->src[0];
                    src_backend_id = tensor_backend_id(cur, -1);
                }
            }
            // ⚠️ 2026-09-01 修复：host 数据被 GPU split 消费时必须建 H2D cpy。
            //   判据不依赖 backend_map_（host 常量 leaf 可能被 pass_fill 标到 GPU 导致
            //   src_backend_id==cur_backend_id 误判"同后端兼容"）→ 直接看数据实际位置：
            //   只要 src 数据在 host（buffer_==null 或 host buffer）而当前是 GPU split，
            //   一律视为 host 后端，走下方不兼容分支强制建 cpy。
            //   （混合训练 loss=0 真根因：GPU REPEAT 读 host mask → dispatch NOT_SUPPORTED
            //     → graph_compute 中断 → loss 链从未执行。）
            bool src_data_host = (src->buffer_ == nullptr) || src->buffer_->is_host();
            bool cur_is_gpu    = !backends_[cur_backend_id]->buffer_type()->is_host();
            if (src_data_host && cur_is_gpu) {
                src_backend_id = n_backends_ - 1;   // host 数据 == CPU backend
            } else if (src_backend_id == -1) {
                continue;
            }

            if (src_backend_id != cur_backend_id &&
                !tensor_buffer_compatible(src, cur_backend_id)) {

                CopyKey key = {src, cur_backend_id};
                auto it = copy_tensor_map_.find(key);
                if (it == copy_tensor_map_.end()) {
                    // 对标 ggml_dup_tensor_layout：创建同 shape 的图节点
                    // dup() 返回 OP_DUP 节点，data_=nullptr，由 allocator 后续分配
                    TensorF32* cpy = dup(src);
                    copy_tensor_map_[key] = cpy;

                    // 记录为 split 的输入（执行时需跨后端拷贝）
                    if (split->n_inputs < MAX_SPLIT_INPUTS) {
                        split->inputs[split->n_inputs++] = src;
                    }
                }

                // 替换 node 的 src 引用为拷贝后的 tensor
                node->src[j] = copy_tensor_map_[key];
            }
        }
    }

    // ===== Step 4: 最后一个 split =====
    split->i_end = n_nodes;
    n_splits_++;

    // ===== Step 5: 合并连续同 backend 的小 split（2026-08-23 提速）=====
    // 碎片化根因：op 级 backend 交替（GPU 只支持少量 op），连续同 backend 的 split 被
    // 切成极小段（1~4 节点），每个都做一次 backend->graph_compute（含 buffer/同步开销）。
    // 同 backend 的连续 split 合并后，Step 1 扫描范围 / Step 2 子图执行天然正确
    // （跨后端边界不变，仅同后端段变宽）。
    if (n_splits_ > 1 && !(getenv("PPML_NO_MERGE") && std::string(getenv("PPML_NO_MERGE")) == "1")) {
        const int kMergeMaxNodes = 64;   // 合并后 split 节点上限（防止 CPU 大 split 阻塞线程池）
        std::vector<SplitInfo> merged;
        merged.reserve(static_cast<size_t>(n_splits_));
        SplitInfo acc = splits_[0];
        for (int si = 1; si < n_splits_; si++) {
            SplitInfo& cur = splits_[si];
            if (cur.backend_id == acc.backend_id &&
                (cur.i_end - acc.i_start) <= kMergeMaxNodes) {
                acc.i_end = cur.i_end;   // 同 backend：直接扩展范围
                // n_inputs 仅 build_splits 记录用（Step 1 实际扫描节点范围），合并后取较大值即可
                if (cur.n_inputs > acc.n_inputs) {
                    for (int k = 0; k < cur.n_inputs; k++) acc.inputs[k] = cur.inputs[k];
                    acc.n_inputs = cur.n_inputs;
                }
            } else {
                merged.push_back(acc);
                acc = cur;
            }
        }
        merged.push_back(acc);
        // 写回
        splits_ = std::move(merged);
        n_splits_ = static_cast<int>(splits_.size());
    }
    if (getenv("GRAPH_DEBUG_SCHED")) {
        fprintf(stderr, "[sched] split_graph: after merge n_splits=%d\n", n_splits_);
    }
}

// ============================================================
// graph_compute — 遍历所有 split 执行（含跨后端拷贝）
// 对标 ggml_backend_sched_compute_splits
// ============================================================
Status BackendScheduler::graph_compute() {
    if (!current_graph_ || n_splits_ == 0) return Status::SUCCESS;
    if (getenv("GRAPH_DEBUG_SCHED")) {
        fprintf(stderr, "[sched] graph_compute: n_splits_=%d splits_size=%zu\n",
                n_splits_, splits_.size());
    }

    for (int si = 0; si < n_splits_; si++) {
        SplitInfo& sp = splits_[si];
        Backend* backend = backends_[sp.backend_id];

        // ---- Step 1: 跨后端拷贝（对标 ggml_backend_tensor_copy）----
        // 只依赖 sp.inputs 会漏拷贝：同一 src 喂多个 split/节点时，copy 只在首次创建的 split
        // 记录一次，其他消费该 (src,backend) 的 split 的 node 被 rewired 到同一个 cpy 却不执行拷贝，
        // 导致 CPU kernel 读到 device 输入 → 段错误。改为扫描本 split 节点范围，
        // 对每个 node 的 src（build_splits 已 rewired 到 cpy）逐一执行跨后端拷贝，保证全覆盖。
        for (int i = sp.i_start; i < sp.i_end; i++) {
            TensorF32* node = current_graph_->graph_node(i);
            if (!node) continue;
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                TensorF32* cpy = node->src[j];
                if (!cpy) continue;
                // cpy 若为跨后端拷贝节点（OP_DUP 且其 src[0] 属于其他后端），才需在此执行拷贝
                auto it = copy_tensor_map_.find(CopyKey{cpy->src[0], sp.backend_id});
                if (it == copy_tensor_map_.end()) continue;
                if (it->second != cpy) continue;   // 该 (src,backend) 的 cpy 正是此节点
                TensorF32* src = cpy->src[0];
                int src_bid = tensor_backend_id(src, -1);
                if (src_bid == sp.backend_id) continue;
                if (tensor_buffer_compatible(src, sp.backend_id)) continue;
                if (getenv("GRAPH_DEBUG_CROSSBK")) {
                    // 精确诊断：跨后端拷贝（尤其 get_rows 输出 → CPU transpose 的 D2H）
                    fprintf(stderr,
                        "[crossbk] S1 split=%d bk=%d src_op=%d src_ndim=%d src_dims=[%lld,%lld,%lld,%lld] "
                        "src_numel=%lld src_data=%p src_buf=%p src_buf_host=%d src_offs=%zu "
                        "cpy_numel=%lld -> 消费节点 op=%d\n",
                        si, sp.backend_id,
                        (int)src->op, (int)src->shape().ndim(),
                        (long long)(src->shape().ndim()>0?src->shape().dims[0]:-1),
                        (long long)(src->shape().ndim()>1?src->shape().dims[1]:-1),
                        (long long)(src->shape().ndim()>2?src->shape().dims[2]:-1),
                        (long long)(src->shape().ndim()>3?src->shape().dims[3]:-1),
                        (long long)src->numel(), (void*)src->data(),
                        (void*)src->buffer_, (src->buffer_ ? (int)src->buffer_->is_host() : -1),
                        (size_t)src->buffer_offs_, (long long)cpy->numel(),
                        (int)node->op);
                }
                backend_tensor_copy(src, cpy);
            }
        }

        // ---- Step 1b: CPU split 兜底 —— device 输入暂存为 host ----
        // 有些 CPU 节点的输入（device 张量）没被 build_splits 建立 cpy（如 view-op 链、
        // 未分配后端边），CPU kernel 会裸读 device 指针 → 段错误。
        // 这里在 thread0（split 分发前）把 device 输入 D2H 暂存并临时 rebind src->data_，
        // split 跑完后恢复。仅对 CPU 后端（host）split 需要。
        if (sp.backend_id == n_backends_ - 1) {   // CPU 是最后一个后端（最低 priority）
            for (int i = sp.i_start; i < sp.i_end; i++) {
                TensorF32* node = current_graph_->graph_node(i);
                if (!node) continue;
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    TensorF32* src = node->src[j];
                    if (!src || !src->data()) continue;
                    // 仅处理 device 输入：有 buffer_ 看 is_host()；无 buffer_ 时用探测兜底
                    bool is_device;
                    if (src->buffer_) is_device = !src->buffer_->is_host();
                    else is_device = is_device_pointer(src, static_cast<const float*>(src->data()));
                    if (!is_device) continue;  // 非 device
                    // 已是被拷到 host 的 cpy（上面已处理）则跳过：仅当 cpy 自身 data 已是 host。
                    // 注意：不能看 src->src[0]->is_host()（旧逻辑误把"cpy 源仍是 device"当跳过条件，
                    // 反而漏掉未完成拷贝的 device src → CPU kernel 段错误）。改为：cpy 自身已是 host 才跳。
                    if (src->op == OP_DUP && src->src[0] && !is_device_pointer(src, src->data())) continue;
                    if (getenv("GRAPH_DEBUG_CROSSBK")) {
                        // 精确诊断：CPU split 消费的 device src（get_rows 输出等）D2H 暂存
                        fprintf(stderr,
                            "[crossbk] S1b split=%d 消费节点 op=%d src_op=%d src_ndim=%d "
                            "src_dims=[%lld,%lld,%lld,%lld] src_numel=%lld src_data=%p src_buf=%p "
                            "src_buf_host=%d src_offs=%zu is_device=%d\n",
                            si, (int)node->op, (int)src->op, (int)src->shape().ndim(),
                            (long long)(src->shape().ndim()>0?src->shape().dims[0]:-1),
                            (long long)(src->shape().ndim()>1?src->shape().dims[1]:-1),
                            (long long)(src->shape().ndim()>2?src->shape().dims[2]:-1),
                            (long long)(src->shape().ndim()>3?src->shape().dims[3]:-1),
                            (long long)src->numel(), (void*)src->data(),
                            (void*)src->buffer_, (src->buffer_ ? (int)src->buffer_->is_host() : -1),
                            (size_t)src->buffer_offs_, (int)is_device);
                    }
                    // D2H 暂存（device 张量必有 buffer_，但无 buffer_ 时用裸 cudaMemcpy 兜底）
                    const int64_t n = src->numel();
                    host_stage_.emplace_back(src, src->data());
                    std::vector<float>& h = host_scratch_.emplace_back(static_cast<size_t>(n));
                    if (src->buffer_) {
                        src->buffer_->get_tensor(src, h.data(), src->buffer_offs_,
                                                static_cast<size_t>(n) * sizeof(float));
                    } else {
                        cudaMemcpy(h.data(), src->data(),
                                   static_cast<size_t>(n) * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                    }
                    src->bind_data(h.data());   // 临时让 CPU kernel 读 host
                }
            }
        }

        // ---- Step 2: 只执行该 split 范围的节点 ----
        // 构造一个临时子图，只包含 [i_start, i_end) 范围的节点
        // 对标 ggml_graph_view(graph, i_start, i_end)
        // 由于 ComputeGraph 是 placement new 的固定大小结构，
        // 这里直接修改 nodes 指针数组的起始位置来模拟子图
        if (getenv("GRAPH_DEBUG_SCHED")) {
            int n_scalar = 0;
            for (int k = sp.i_start; k < sp.i_end; k++) {
                TensorF32* nd = current_graph_->graph_node(k);
                if (nd && nd->numel() == 1) n_scalar++;
            }
            fprintf(stderr, "[sched] COMPUTE split=%d backend=%d i=[%d,%d) nodes=%d scalar(numel=1)=%d\n",
                    si, sp.backend_id, sp.i_start, sp.i_end, sp.i_end - sp.i_start, n_scalar);
        }
        int sub_n_nodes = sp.i_end - sp.i_start;
        TensorF32** saved_nodes = current_graph_->nodes;
        int saved_n_nodes = current_graph_->n_nodes_;

        // 临时替换为子图范围
        current_graph_->nodes = saved_nodes + sp.i_start;
        current_graph_->n_nodes_ = sub_n_nodes;

        // 混训：scheduler 已通过 reserve_graph_memory 预分配全部张量，
        // 让后端跳过自身 gallocr（避免两套 gallocr re-bind 冲突）。
        backend->set_skip_alloc(true);
        Status st = backend->graph_compute(current_graph_);
        backend->set_skip_alloc(false);

        //   若不等其完成就进入下一个 CPU split 的 Step 1b D2H（get_tensor/cudaMemcpy），
        //   可能读到未完成/垃圾数据 → 偶发 loss 巨大(nan)/segfault（每次运行结果不同）。
        //   在 GPU split 结束后强制同步（CPU split 同步是空操作，无开销）。
        if (sp.backend_id != n_backends_ - 1) {
            backend->synchronize();
        }

        // 恢复原始图状态
        current_graph_->nodes = saved_nodes;
        current_graph_->n_nodes_ = saved_n_nodes;

        // ---- Step 2b: 主动 D2H（生产者 split 后立即落地跨后端输出，2026-08-24）----
        // 本 split 算完的节点若被其他后端消费（copy_tensor_map_ 记录了 (src,backend)→cpy），
        // 立即 backend_tensor_copy 落地到 cpy（target 后端 buffer）。
        // 作用：避免"被动时机"（消费者 split 的 Step 1 才拷贝）读到被 gallocr 复用的 GPU buffer
        //   → 偶发巨大值（k_cat CONCAT 报 -1.98e6、新版 a_rows 跨后端 NaN）。
        //   生产后立即落地，src 数据刚算完，且 CPU 消费者用 cpy（host）不依赖 src GPU buffer 存活。
        // 主动落地（PPML_EAGER_D2H=1 启用）：生产者 split 后立即拷贝跨后端输出。
        //    invalid argument，且与方案A is_output 叠加后偶发 loss 巨大）。保留开关待进一步调试。
        if (getenv("PPML_EAGER_D2H") && std::strcmp(getenv("PPML_EAGER_D2H"), "1") == 0) {
            for (auto& kv : copy_tensor_map_) {
                const TensorF32* src = kv.first.first;
                int target_bk = kv.first.second;
                TensorF32* cpy = kv.second;
                if (!src || !cpy) continue;
                if (target_bk == sp.backend_id) continue;      // 同后端无需跨后端拷贝
                // 只处理本 split 计算产生的 src（src 在本 split 节点范围且已被计算）
                bool in_range = false;
                for (int k = sp.i_start; k < sp.i_end; k++) {
                    if (current_graph_->graph_node(k) == src) { in_range = true; break; }
                }
                if (!in_range) continue;
                if (!src->data()) continue;                    // 尚未计算
                if (tensor_buffer_compatible(src, target_bk)) continue;
                backend_tensor_copy(const_cast<TensorF32*>(src), cpy);
            }
        }

        // 恢复本 split 被临时 rebind 到 host 的 device 指针（Step 1b 暂存）
        for (auto& pr : host_stage_) {
            pr.first->bind_data(pr.second);
        }
        host_stage_.clear();
        host_scratch_.clear();

        if (st != Status::SUCCESS) {
            fprintf(stderr, "[sched] split=%d backend=%d i=[%d,%d) FAILED st=%d\n",
                    si, sp.backend_id, sp.i_start, sp.i_end, (int)st);
            // 2026-09-01：回退 CPU 前恢复原图（撤销 build_splits 的 src 替换/bind 污染），
            // 否则 train.cpp 的 CPU 回退全图会读被替换的未分配 cpy 节点 → loss=0。
            restore_graph_nodes();
            return st;
        }
    }

    return Status::SUCCESS;
}

// ============================================================
// 获取分裂后的第 i 段
// ============================================================

ComputeGraph * BackendScheduler::get_split(int i) {
    if (i < 0 || i >= n_splits_) return nullptr;
    // 返回原始图（子图通过 [i_start, i_end) 范围标识）
    return current_graph_;
}

} // namespace ppml
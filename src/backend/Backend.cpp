
#include "ppml/Backend.h"
#include "ppml/Context.h"
#include "ppml/AliasGuard.h"   // 假别名断言（PPML_ALIAS_GUARD=1）
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>            // exec_ab_check 的相对误差
#include <vector>           // exec_ab_check 的 host 影子缓冲
#include <unordered_map>    // exec_nan_check 的"缓冲最后写入者"登记表
#include <cuda_runtime.h>   // cudaMemGetInfo 用于 GPU 显存预算

namespace ppml {

// ============================================================
// 【2026-09-19 P1】pinned staging 池（定义在 src/cuda/StagingPool.cu ✓）
//   把跨后端边界的**阻塞** cudaMemcpy（pageable，驱动隐式同步 ✗）换成
//   "pinned arena + cudaMemcpyAsync + 事件排序"：
//     · D2H：默认流 record → copy 流 wait → async ⇒ 每个 split **只等一次**（staging_wait ✓）
//     · H2D：async → record → 默认流 wait ⇒ **无需 host 等待** ✓
//   开关 PPML_STAGING_ASYNC=0 可整体回落；任一失败一律回落原阻塞路径 ✓（不静默错值 ✓）
// ============================================================
extern int   staging_enabled();
extern int   staging_is_pinned(const void* p);
extern void* staging_alloc(int64_t bytes);
extern int   staging_d2h(void* dst_pinned, const void* src_dev, int64_t bytes);
extern int   staging_h2d(void* dst_dev, const void* src_pinned, int64_t bytes);
extern int   staging_wait();
extern void  staging_split_end();

// 真·CUDA device buffer 判定（排除 RemoteBuffer 等"非 host 但不是本地显存"的 buffer ✗）
static bool is_cuda_device_tensor(const TensorF32* t) {
    if (!t || !t->buffer_ || t->buffer_->is_host()) return false;
    const BufferType* bt = t->buffer_->type();
    return bt && !bt->is_host() && dynamic_cast<const CUDABufferType*>(bt) != nullptr;
}

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
// 【2026-09-20】把"**逐节点级**"调度诊断从 GRAPH_DEBUG_SCHED 里拆出来：
//   全量训练下 GRAPH_DEBUG_SCHED=1 每张图会打出上万行（[nhip] 每节点一次 + node dump + pass_fill + try b）✗，
//   既刷屏又拖慢 I/O（长训练日志可达 GB 级 ✗）⇒ 这些逐节点日志改由 **GRAPH_DEBUG_SCHED_VERBOSE=1** 控制 ✓；
//   **split 级**信息（`split_graph:` / PLAN / COMPUTE / `backend[…]=` / budget 逐出）仍由 GRAPH_DEBUG_SCHED=1 控制 ✓。
static bool sched_verbose() {
    static const bool on = (std::getenv("GRAPH_DEBUG_SCHED_VERBOSE") != nullptr);
    return on;
}

bool BackendScheduler::node_is_host_producer(TensorF32* node) const {
    if (!node) return false;
    // 沿 view_src 链找底层数据载体（view 共享源数据，data 在源头）
    const TensorF32* base = node;
    int guard = 0;
    if (sched_verbose()) {
        fprintf(stderr, "[nhip] node=%p op=%d view_src=%p buffer_=%p data=%p CALLER=%p\n",
                (void*)node, (int)node->op, (void*)node->view_src,
                (void*)node->buffer_, (void*)node->data(),
#if defined(_MSC_VER)
                nullptr);
#else
                __builtin_return_address(0));
#endif
    }
    while (base->view_src && guard++ < 64) base = base->view_src;
    if (sched_verbose() && base != node) {
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

// 【2026-09-22】构建期"数据居住地"判定（与运行期 buffer 状态解耦）★
//   背景：040350 与 043548 是**同一配置、同一张图**（split=0/2/4 行逐字一致 ✓），但同一个
//   CUDA split（i=[7191,7231)，含 3× op=0）一次崩（`[CUDA-ERR] illegal memory access` →
//   CPU 回退 → exit 134 ✗）一次正常 ✓。根因：build_splits 的 host 判据用**运行期** `buffer_`
//   状态（`src->buffer_ == nullptr || is_host()` ✗），而参数 transfer / staging rebind 的时机
//   在两次 run 里不同 ⇒ 判定漂移 ✗。
//   规则（构建期钉死 ✓）：
//     · op=0（参数/常量/输入）：首次观测记录居住地（1=host, 2=device）；随后
//       host→device 允许**单调升级** ✓（参数 transfer 完成后合法）；device→host 视为漂移，
//       **保守按 host 处理** ✓（宁可多建一次 H2D cpy，也不让 GPU 内核裸读 host 指针 ✗）。
//     · 其它 op：仍按当次构建的真实 buffer 状态判定（其生命周期由 gallocr 管理，稳定 ✓）。
bool BackendScheduler::sched_data_is_host(const TensorF32* t) {
    if (!t) return true;
    const bool cur_host = (t->buffer_ == nullptr) || t->buffer_->is_host();
    if (t->op != OP_NONE) return cur_host;

    unsigned char& rec = residence_fix_[t];   // 0 = 未记录
    if (rec == 0) {
        rec = cur_host ? 1u : 2u;
        if (getenv("GRAPH_DEBUG_SCHED")) {
            fprintf(stderr, "[sched] residence fix: op=0 %p numel=%lld bytes=%zu → %s（钉死 ✓）\n",
                    (const void*)t, (long long)t->numel(), t->nbytes(), cur_host ? "host" : "device");
        }
        return cur_host;
    }
    if (rec == 1u && !cur_host) {             // host→device：合法升级（参数被搬到显存 ✓）
        rec = 2u;
        if (getenv("GRAPH_DEBUG_SCHED")) {
            fprintf(stderr, "[sched] residence fix: op=0 %p host→device（升级记录 ✓）\n", (const void*)t);
        }
        return false;
    }
    if (rec == 2u && cur_host) {              // device→host：漂移 ⇒ 保守按 host（并在记录里降级，避免重复告警 ✓）
        static long drift_warn = 0;
        rec = 1u;
        if (getenv("GRAPH_DEBUG_SCHED") && drift_warn++ < 20) {
            fprintf(stderr, "[sched] residence fix: op=0 %p device→host 漂移 ⇒ 保守按 host（建 H2D cpy ✓）\n",
                    (const void*)t);
        }
        return true;
    }
    return rec == 1u;
}

void BackendScheduler::set_backend_if_supported(TensorF32* node, int backend_id, int idx) {
    // host 生产者（OP_NONE 参数/常量）不得放 GPU：数据在 host，dispatch 跳过它，
    // 也不会建 cpy（与消费者同后端），GPU kernel 会读 host 指针。
    if (node_is_host_producer(node)) return;

    Backend * backend = backends_[backend_id];
    backend->set_sched_index(idx);   // ★ 供后端做"按图节点区间"选择（远端 PPML_REMOTE_NODES ✓）
    if (!backend->supports_op(node)) return;

    // GPU 后端：受显存预算约束——预算不足时拒绝，留待回落 CPU。
    //   注：远端后端 vram_budgeted()==false ⇒ 不受此限（它不占显存 ✓）
    if (backend->vram_budgeted()) {
        if (gpu_assign_if_affordable(node, backend_id)) {
            backend_map_[node] = backend_id;
        }
        // 拒绝时不写入 backend_map_，后续 pass_fill_unassigned 会回落 CPU。
        return;
    }

    backend_map_[node] = backend_id;
}

// 【2026-09-19】显存预算逐出统计：原来**完全静默** ✗ ⇒ 无法回答
//   "那些夹在 CUDA 区中间的孤独 CPU op（如 op=30 MUL_MAT）到底是被预算逐出的、还是不被支持的" ✓
static long g_budget_refuse_cnt_ = 0;
static long g_budget_refuse_by_op_[160] = {0};   // 按下标=op 计数（>159 归到 0）

bool BackendScheduler::gpu_assign_if_affordable(TensorF32* node, int gpu_backend_id) {
    // 获取该 GPU 后端的 buffer type（估算分配大小用；bufts_ 在 reserve 阶段才填充，
    // 因此这里直接从 backend 取）。
    const BufferType* bt = (gpu_backend_id < n_backends_)
                               ? backends_[gpu_backend_id]->buffer_type() : nullptr;

    // 【2026-09-22】"host 源 ⇒ H2D cpy" 的**构建期**预算判定 ★
    //   本节点的输入里若有"host 居住"的源，而本节点要放 GPU ⇒ build_splits 必须为它建
    //   H2D cpy（持久 buffer，占显存）✓。这里**提前**把这份拷贝的字节计入预算：
    //     显存够 ⇒ 放行（构建期就定下"插 cpy"✓，与运行期 buffer 状态无关 ✓）；
    //     显存不够 ⇒ 拒绝该节点上 GPU（回落 CPU ✓），而不是构建期漏建 cpy 让内核裸读 host ✗。
    //   同一 src 只计一次（host_cpy_bytes_ ✓，每次构建由 assign_backends 清空 ✓）。
    auto h2d_bytes_for = [&](TensorF32* nd, std::vector<TensorF32*>* newly) -> size_t {
        if (gpu_vram_budget_ == 0) return 0;
        size_t sum = 0;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            TensorF32* s = nd->src[j];
            if (!s) continue;
            if (!sched_data_is_host(s)) continue;      // 已在 device / 会在 device 上算 ✓
            if (host_cpy_bytes_.count(s)) continue;    // 本次构建已计过 ✓
            sum += s->nbytes();
            if (newly) newly->push_back(s);
        }
        return sum;
    };
    auto h2d_account = [&](const std::vector<TensorF32*>& v) {
        for (TensorF32* s : v) {
            host_cpy_bytes_[s] = s->nbytes();
            host_cpy_reserved_ += s->nbytes();
        }
    };
    // 拒绝原因统计（与 op 无关的 H2D 预算拒绝 ✓）
    static long h2d_refuse_cnt = 0;

    // 无预算（0=不限/未启用）→ 直接放行（不做任何记账 ✓）
    if (gpu_vram_budget_ == 0) return true;

    // 该节点已有独立 device buffer（非复用分配）→ 仍要判定它的 host 输入的 H2D cpy 预算 ✓
    if (node->data() != nullptr) {
        std::vector<TensorF32*> newly;
        const size_t h2d = h2d_bytes_for(node, &newly);
        if (gpu_reserved_bytes_ + h2d > gpu_vram_budget_) {
            ++h2d_refuse_cnt;
            if (getenv("GRAPH_DEBUG_SCHED") && h2d_refuse_cnt <= 20) {
                fprintf(stderr,
                        "[sched] H2D-cpy refuse #%ld: op=%d need=%.2f MB used=%.1f/%.1f MB（回落 CPU ✓）\n",
                        h2d_refuse_cnt, (int)node->op, (double)h2d / 1048576.0,
                        (double)gpu_reserved_bytes_ / 1048576.0,
                        (double)gpu_vram_budget_ / 1048576.0);
            }
            return false;
        }
        gpu_reserved_bytes_ += h2d;
        h2d_account(newly);
        return true;
    }

    // 估算该节点在 GPU 上的分配大小
    size_t est = 0;
    if (bt) {
        est = GGML_PAD(bt->get_alloc_size(node), bt->get_alignment());
    }

    // 连同"host 输入的 H2D cpy"一起算预算 ✓
    std::vector<TensorF32*> newly;
    const size_t h2d = h2d_bytes_for(node, &newly);

    // 预算不足：拒绝放 GPU（回落 CPU）
    if (gpu_reserved_bytes_ + est + h2d > gpu_vram_budget_) {
        // 【2026-09-19】诊断：记录被逐出的 op（前 20 次打明细 ✓）
        ++g_budget_refuse_cnt_;
        const int op_id = (int)node->op;
        g_budget_refuse_by_op_[(op_id >= 0 && op_id < 160) ? op_id : 0]++;
        if (getenv("GRAPH_DEBUG_SCHED") && g_budget_refuse_cnt_ <= 20) {
            fprintf(stderr,
                    "[sched] budget refuse #%ld: op=%d est=%.2f MB h2d=%.2f MB used=%.1f/%.1f MB\n",
                    g_budget_refuse_cnt_, op_id, (double)est / 1048576.0, (double)h2d / 1048576.0,
                    (double)gpu_reserved_bytes_ / 1048576.0,
                    (double)gpu_vram_budget_ / 1048576.0);
        }
        return false;
    }

    gpu_reserved_bytes_ += est + h2d;
    h2d_account(newly);
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
// BufferType::new_buffer — 默认工厂（DefaultBuffer），远端 buffer 类型可覆盖
// 2026-09-13：把"建 buffer"收敛到单一钩子，alloc_buffer() 与 Gallocr arena 都走它，
//   新增后端（如 RemoteBufferType）无需改动 Gallocr/调度器。
// ============================================================
Buffer* BufferType::new_buffer(size_t size, BufferUsage usage) {
    return new DefaultBuffer(this, size, usage);
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
        return buft->new_buffer(0, usage);
    }

    return buft->new_buffer(size, usage);
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
// exec_nan_check — 执行期逐节点 NaN/爆炸检查（诊断，2026-10-20）
// ============================================================
//   动机：`[FWD-NAN]` 是"整图跑完后**按节点索引**扫描"，会把本次**没有执行**的
//   陈旧/未算节点也算进来（实测混合模式 epoch2 第一个命中是未算的反向节点），
//   真正的"第一个产生 NaN 的 kernel"被掩盖 ✗。
//   本函数在**每个节点 kernel 刚执行完**时调用，按**执行顺序**报告 ⇒ 第一次命中
//   就是源头算子 ✓。读值：host 直接读；device 分块 cudaMemcpy D2H（同流有序 ⇒
//   读到的是该 kernel 的结果）。
namespace {

// 扫描单张量：坏 = 含 NaN / ±Inf / max|v| > thr。
//   返回 false 表示无法扫描（data 为空 / D2H 失败）。
bool exec_scan_tensor(const TensorF32* t, float thr,
                      long long* n_nan, long long* n_inf, double* maxabs,
                      float* first_bad, long long* first_idx,
                      long long* n_zero = nullptr, double* minv = nullptr,
                      double* minpos = nullptr) {
    if (!t || !t->data() || t->numel() <= 0) return false;
    const int64_t n = t->numel();
    const float* dptr = static_cast<const float*>(t->data());
    const bool dev = is_device_pointer(t, dptr);

    static thread_local std::vector<float> chunk;
    constexpr int64_t kChunk = 1 << 20;   // 1M float = 4 MB / 块

    for (int64_t off = 0; off < n; off += kChunk) {
        const int64_t m = (off + kChunk <= n) ? kChunk : (n - off);
        const float* p = nullptr;
        if (dev) {
            chunk.resize(static_cast<size_t>(m));
            if (cudaMemcpy(chunk.data(), dptr + off, static_cast<size_t>(m) * sizeof(float),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                cudaGetLastError();   // 清掉残留错误，避免污染后续 kernel 判定
                return false;
            }
            p = chunk.data();
        } else {
            p = dptr + off;
        }
        for (int64_t k = 0; k < m; ++k) {
            const float v = p[k];
            if (std::isnan(v)) { ++(*n_nan); }
            else if (!std::isfinite(v)) { ++(*n_inf); }
            else {
                const double a = std::fabs(static_cast<double>(v));
                if (a > *maxabs) *maxabs = a;
                if (minv && static_cast<double>(v) < *minv) *minv = static_cast<double>(v);
                if (minpos && v != 0.0f && a < *minpos) *minpos = a;   // 最小非零幅值（看分母下限用）
                if (n_zero && v == 0.0f) ++(*n_zero);
                if (a > static_cast<double>(thr) && *first_idx < 0) {
                    *first_bad = v;
                    *first_idx = off + k;
                }
            }
            if (*first_idx < 0 && (std::isnan(v) || std::isinf(v))) {
                *first_bad = v;      // NaN/Inf 也算"第一个坏值"，便于看首次出现位置
                *first_idx = off + k;
            }
        }
    }
    return true;
}

// 生产记录（生产者校验和）：{ 张量对象 → 写入时的地址+统计 } 与 { 地址 → 最后写入者 }。
//   下游读到坏值时据此区分三种情形：
//     (a) REBIND —— 张量对象当前 data 地址 ≠ 写入时地址 ⇒ 指针被中途重绑；
//     (b) 内容被覆盖 —— 地址未变、生产者记录为干净、现在脏 ⇒ 别名/复用写坏；
//     (c) 上游本就脏 —— 生产者记录也脏（真正上游问题）。
// 【2026-10-20】从 exec_nan_check 的函数级 static 提到 TU 级：exec_view_probe 需要复用同一份
//   记录来判定"消费 view 的 kernel 读到的到底是不是生产者的内容"。
struct ExecWRec {
    const TensorF32* tensor = nullptr;
    uintptr_t addr = 0;
    int node_idx = -1;
    int op = -1;
    long long nan = 0;
    long long inf = 0;
    double maxabs = 0.0;
};
std::unordered_map<uintptr_t, ExecWRec>        g_exec_by_addr;
std::unordered_map<const TensorF32*, ExecWRec> g_exec_by_tensor;

}  // namespace

void exec_nan_check(const TensorF32* node, const char* backend_name, int node_idx) {
    static const bool  on = (std::getenv("PPML_EXEC_NAN_CHECK") != nullptr);
    if (!on || !node) return;
    static const float thr = [] {
        const char* e = std::getenv("PPML_EXEC_NAN");
        return e ? static_cast<float>(std::atof(e)) : 1e6f;
    }();
    static const int max_print = [] {
        const char* e = std::getenv("PPML_EXEC_NAN_MAX");
        return e ? std::atoi(e) : 8;
    }();
    // 追踪模式：把每个 split 前 N 个节点（无论好坏）都打印出来，用于判断
    //   "某节点执行时是干净的、被消费者读到却已经是 NaN" ⇒ 缓冲被复用抵消（aliasing）。
    static const int trace_max = [] {
        const char* e = std::getenv("PPML_EXEC_TRACE");
        return e ? std::atoi(e) : 0;
    }();
    // 起始下标（PPML_EXEC_TRACE_FROM，默认 0）：配合 TRACE 只看 [FROM, MAX) 区间——
    //   子图规模上千节点时，只看前 40 个抓不到真正的放大点，必须能滑窗。
    static const int trace_from = [] {
        const char* e = std::getenv("PPML_EXEC_TRACE_FROM");
        return e ? std::atoi(e) : 0;
    }();
    auto traced = [&](int idx) {
        return trace_max > 0 && idx < trace_max && idx >= trace_from;
    };
    static int printed = 0;
    if (node->op == OP_NONE || node->numel() <= 0 || !node->data()) return;
    if (printed >= max_print && !traced(node_idx)) return;

    long long nn = 0, ni = 0, fi = -1;
    double mx = 0.0; float fb = 0.0f;
    if (!exec_scan_tensor(node, thr, &nn, &ni, &mx, &fb, &fi)) return;

    if (traced(node_idx)) {
        fprintf(stderr,
                "[EXEC-TRACE] backend=%s node=%d op=%d self=%p data=%p numel=%lld nan=%lld inf=%lld maxabs=%.4e "
                "dims=[%lld,%lld,%lld,%lld]\n",
                backend_name ? backend_name : "?", node_idx, (int)node->op,
                (const void*)node, (const void*)node->data(), (long long)node->numel(), nn, ni, mx,
                (long long)(node->shape().ndim() > 0 ? node->shape().dims[0] : -1),
                (long long)(node->shape().ndim() > 1 ? node->shape().dims[1] : -1),
                (long long)(node->shape().ndim() > 2 ? node->shape().dims[2] : -1),
                (long long)(node->shape().ndim() > 3 ? node->shape().dims[3] : -1));
        fflush(stderr);
    }

    // 本节点 dst 刚写完 ⇒ 登记生产记录（对象级 + 地址级）
    {
        ExecWRec rec{node, reinterpret_cast<uintptr_t>(node->data()), node_idx, (int)node->op, nn, ni, mx};
        g_exec_by_addr[rec.addr]   = rec;
        g_exec_by_tensor[node]     = rec;
    }

    if (nn == 0 && ni == 0 && fi < 0) return;   // 干净
    if (printed >= max_print) return;
    ++printed;
    // 各 src 一并统计：区分"本算子引入"还是"上游已经坏"（多 src 时逐一看，
    //   例如 DIV 的坏值往往来自分母 src1，只看 src0 会误判为 ADD 的问题）
    char srcbuf[3][320];
    srcbuf[0][0] = srcbuf[1][0] = srcbuf[2][0] = '\0';
    for (int i = 0; i < 3 && node->src[i]; ++i) {
        long long snn = 0, sni = 0, sfi = -1, szero = 0;
        double smx = 0.0, smin = 1e30; float sfb = 0.0f;
        const TensorF32* s = node->src[i];
        if (!exec_scan_tensor(s, thr, &snn, &sni, &smx, &sfb, &sfi, &szero, &smin)) continue;

        // 生产者记录（对象级）：判 REBIND —— 写入时的地址与当前 data 是否一致
        ExecWRec pr{};
        bool has_prod = false, rebind = false;
        uintptr_t old_addr = 0;
        auto tit = g_exec_by_tensor.find(s);
        if (tit != g_exec_by_tensor.end()) {
            pr = tit->second;
            has_prod = true;
            if (s->data() && pr.addr != reinterpret_cast<uintptr_t>(s->data())) {
                rebind = true;
                old_addr = pr.addr;
            }
        }
        // 地址级：这块内存最后被谁写（可能是另一个张量 ⇒ 别名/复用）
        ExecWRec aw{};
        bool has_aw = false;
        if (s->data()) {
            auto ait = g_exec_by_addr.find(reinterpret_cast<uintptr_t>(s->data()));
            if (ait != g_exec_by_addr.end()) { aw = ait->second; has_aw = true; }
        }
        snprintf(srcbuf[i], sizeof(srcbuf[i]),
                 " src%d=(op=%d,data=%p,now[nan=%lld,inf=%lld,max=%.3e,min=%.3e,zero=%lld],"
                 "rebind=%d,was=%p,"
                 "prod=%s(op%d@n%d,nan=%lld,inf=%lld,max=%.3e),"
                 "addr_last_writer=%s(op%d@n%d))",
                 i, (int)s->op, (const void*)s->data(), snn, sni, smx, smin, szero,
                 rebind ? 1 : 0, (const void*)old_addr,
                 has_prod ? "" : "NONE", pr.op, pr.node_idx, pr.nan, pr.inf, pr.maxabs,
                 has_aw ? "" : "NONE", aw.op, aw.node_idx);
    }

    // DIV 分母溯源：src1 往往是 DUP/clamp/div 链的输出，沿 src[0] 上溯 6 层打印 op/numel/零值数，
    //   用于把"分母大面积 0"对回源码表达式（单线程跑一次即可定位，不必逐层加日志）。
    char divchain[640];
    divchain[0] = '\0';
    if ((int)node->op == 8 /*OP_DIV*/ && node->src[1]) {
        size_t used = 0;
        const TensorF32* c = node->src[1];
        for (int lv = 0; c && lv < 6 && used + 96 < sizeof(divchain); ++lv) {
            long long cn = 0, ci = 0, cz = 0, cf = -1;
            double cmx = 0.0, cmn = 1e30;
            float cfb = 0.f;
            const bool ok = exec_scan_tensor(c, thr, &cn, &ci, &cmx, &cfb, &cf, &cz, &cmn);
            used += static_cast<size_t>(snprintf(divchain + used, sizeof(divchain) - used,
                                                "%slv%d(op=%d,numel=%lld%s,nan=%lld,zero=%lld,max=%.3e)",
                                                lv ? " <- " : "", lv, (int)c->op,
                                                (long long)c->numel(), ok ? "" : "/no-data",
                                                cn, cz, cmx));
            c = c->src[0];
        }
    }

    // 缓冲重叠检查：dst 与某个 src 落在同一段内存（或部分重叠）⇒ 读写同一块内存，
    //   kernel 边写边读会自我污染（内存规划/别名 bug 的典型症状）。
    auto overlap = [](const TensorF32* a, const TensorF32* b) -> int {
        if (!a || !b || !a->data() || !b->data()) return 0;
        const uintptr_t a0 = (uintptr_t)a->data(), a1 = a0 + (uintptr_t)a->nbytes();
        const uintptr_t b0 = (uintptr_t)b->data(), b1 = b0 + (uintptr_t)b->nbytes();
        return (a0 < b1 && b0 < a1) ? 1 : 0;
    };
    const int al0 = overlap(node, node->src[0]);
    const int al1 = overlap(node, node->src[1]);

    fprintf(stderr,
            "[EXEC-NAN] #%d backend=%s node=%d op=%d self=%p data=%p src0=%p src1=%p numel=%lld nan=%lld inf=%lld "
            "maxabs=%.4e first_bad=%.6g @%lld dims=[%lld,%lld,%lld,%lld] alias0=%d alias1=%d%s%s%s%s\n",
            printed, backend_name ? backend_name : "?", node_idx, (int)node->op,
            (const void*)node, (const void*)node->data(),
            (const void*)node->src[0], (const void*)node->src[1],
            (long long)node->numel(), nn, ni, mx, (double)fb, fi,
            (long long)(node->shape().ndim() > 0 ? node->shape().dims[0] : -1),
            (long long)(node->shape().ndim() > 1 ? node->shape().dims[1] : -1),
            (long long)(node->shape().ndim() > 2 ? node->shape().dims[2] : -1),
            (long long)(node->shape().ndim() > 3 ? node->shape().dims[3] : -1),
            al0, al1, srcbuf[0], srcbuf[1], srcbuf[2],
            divchain[0] ? " | div-chain:↓" : "");
    if (divchain[0]) fprintf(stderr, "           div-chain:%s\n", divchain);
    fflush(stderr);
}

// ============================================================
// exec_view_probe — view 别名消费探针（诊断，默认关；2026-10-20）
// ============================================================
//   动机（实测证据，CUDA 混合调度、单 split 内相邻节点）：
//     node0 op=30 MUL_MAT dims=[32,51] max=1.8493e+02   ← GNormBias 的输入 x
//     node2 op=9  SQR      dims=[1,32,51] max=1.9303e-07 ← sqr(x) 本应 ~3.4e4
//   而 SQR 的输入是 `view(x)`（零拷贝别名）。后续 SUM_ROWS/SQRT 与该错误输入自洽
//   （sqrt(1.93e-7)=4.39e-4 ✓）⇒ 错在"消费 view 读到的内容"，不在算子本身。
//   本探针在**消费者 kernel 刚执行完**时，对每个 view 型 src 判定：
//     ① same_addr —— view 解析出的 data 是否等于 view 链根（真实载体）的 data；
//     ② 该地址当前内容 vs 根张量**写入时**的生产记录（maxabs/nan/inf）；
//   二者任一不符即打印 —— 直接区分"view 解析错(①)"与"根被提前释放/复用(②)"。
//   开关：PPML_EXEC_VIEW_PROBE=1；PPML_EXEC_VIEW_PROBE_MAX=12 限制打印条数。
void exec_view_probe(const TensorF32* node, const char* backend_name, int node_idx) {
    static const bool on = (std::getenv("PPML_EXEC_VIEW_PROBE") != nullptr);
    if (!on || !node) return;
    static const int max_print = [] {
        const char* e = std::getenv("PPML_EXEC_VIEW_PROBE_MAX");
        return e ? std::atoi(e) : 12;
    }();
    // 可选：只探测某些消费者算子（逗号分隔 op 编号），如 PPML_EXEC_VIEW_PROBE_OPS=9,15,10
    //   用途：全图前几十条命中往往被无关算子（如 gather）占满，需要聚焦数值敏感链。
    static const bool op_filter = (std::getenv("PPML_EXEC_VIEW_PROBE_OPS") != nullptr);
    static bool op_want[512] = {false};
    static const bool op_want_init = [] {
        const char* e = std::getenv("PPML_EXEC_VIEW_PROBE_OPS");
        if (!e) return true;
        const char* p = e;
        while (*p) {
            while (*p == ',' || *p == ' ') ++p;
            if (!*p) break;
            const int v = std::atoi(p);
            if (v >= 0 && v < 512) op_want[v] = true;
            while (*p && *p != ',') ++p;
        }
        return true;
    }();
    (void)op_want_init;
    static int printed = 0;

    // ---- (0) GET_ROWS 专项：确认 src0(W)/src1(idx) 的居住地 ----
    //   动机：CUDA kernel 直接读 `W->data()`/`idx->data()`（不做 H2D），若 idx 仍是 host
    //   指针而节点被派到 GPU，kernel 会读到无效内存（越界钳制后取值"看似正常但错"）。
    //   这也是 PPML_CUDA_DISABLE_OPS=41 能消除 1e16 梯度爆炸的最可能机理。
    if (node->op == OP_GET_ROWS && (!op_filter || op_want[(int)node->op]) && printed < max_print) {
        ++printed;
        auto resid = [](const TensorF32* t) -> const char* {
            if (!t) return "none";
            const TensorF32* b = t;
            int g = 0;
            while (b->view_src && g++ < 64) b = b->view_src;
            if (!b->buffer_) return "no-buffer";
            return b->buffer_->is_host() ? "HOST" : "DEVICE";
        };
        const TensorF32* W = node->src[0];
        const TensorF32* I = node->src[1];
        fprintf(stderr,
                "[VIEW-PROBE][GET_ROWS] backend=%s node=%d W(op=%d numel=%lld %s) idx(op=%d numel=%lld %s)",
                backend_name ? backend_name : "?", node_idx,
                W ? (int)W->op : -1, W ? (long long)W->numel() : -1, resid(W),
                I ? (int)I->op : -1, I ? (long long)I->numel() : -1, resid(I));
        // 若索引张量在 host，顺便打印前 3 个索引值（判断内容是否合理）
        if (I && I->data()) {
            const TensorF32* b = I;
            int g = 0;
            while (b->view_src && g++ < 64) b = b->view_src;
            if (!b->buffer_ || b->buffer_->is_host()) {
                const float* p = static_cast<const float*>(I->data());
                fprintf(stderr, " idx0..2=[%.1f %.1f %.1f]", p[0], p[1], p[2]);
            }
        }
        fprintf(stderr, "\n");
        fflush(stderr);
    }

    // ---- (1) 先查本节点的 view 型 src（必须在登记本节点之前，才能读到生产者的记录）----
    for (int s = 0; s < 3 && node->src[s]; ++s) {
        const TensorF32* v = node->src[s];
        if (!v->view_src) continue;                      // 非 view：不涉及别名
        const TensorF32* root = v;
        int guard = 0;
        while (root->view_src && guard++ < 64) root = root->view_src;

        const void* vd = v->data();
        const void* rd = root->data();
        // 当前内容（阈 0 ⇒ 只统计 nan/inf/maxabs）
        long long vn = 0, vi = 0, vf = -1;
        double vmx = 0.0; float vfb = 0.0f;
        const bool vok = exec_scan_tensor(v, 0.0f, &vn, &vi, &vmx, &vfb, &vf);
        // 根的写入记录
        auto it = g_exec_by_tensor.find(root);
        const bool has_rec = (it != g_exec_by_tensor.end());
        const ExecWRec rec = has_rec ? it->second : ExecWRec{};

        const bool addr_bad  = (vd != rd);                       // ① view 解析 ≠ 根载体
        const bool alias_bad = has_rec && rec.addr != (uintptr_t)vd;  // ② 读到的是别的地址
        const bool content_bad = has_rec && vok && rec.maxabs > 0.0 &&
                                 (vmx > rec.maxabs * 4.0 || vmx * 4.0 < rec.maxabs);
        // 聚焦算子（PPML_EXEC_VIEW_PROBE_OPS）⇒ 无条件打印（不做"坏"判定），便于看健康样本对照
        const bool focus = op_filter && op_want[(int)node->op];
        if (!focus && (!addr_bad && !alias_bad && !content_bad)) continue;
        if (printed >= max_print) return;
        ++printed;
        fprintf(stderr,
                "[VIEW-PROBE] #%d backend=%s node=%d op=%d src%d view(op=%d numel=%lld data=%p) "
                "root(op=%d numel=%lld data=%p) same_addr=%d "
                "rec=%s(op=%d@n%d addr=%p nan=%lld inf=%lld max=%.4e) now[nan=%lld inf=%lld max=%.4e]%s%s%s\n",
                printed, backend_name ? backend_name : "?", node_idx, (int)node->op, s,
                (int)v->op, (long long)v->numel(), vd,
                (int)root->op, (long long)root->numel(), rd, addr_bad ? 0 : 1,
                has_rec ? "" : "NONE", rec.op, rec.node_idx, (const void*)rec.addr,
                rec.nan, rec.inf, rec.maxabs, vn, vi, vmx,
                addr_bad ? " [BAD:view!=root]" : "",
                alias_bad ? " [BAD:addr-mismatch]" : "",
                content_bad ? " [BAD:content-changed]" : "");
        fflush(stderr);
    }

    // ---- (2) 登记本节点的写入记录（供后续消费者比对）----
    if (node->op == OP_NONE || node->numel() <= 0 || !node->data()) return;
    long long nn = 0, ni = 0, fi = -1;
    double mx = 0.0; float fb = 0.0f;
    if (!exec_scan_tensor(node, 0.0f, &nn, &ni, &mx, &fb, &fi)) return;
    ExecWRec rec{node, reinterpret_cast<uintptr_t>(node->data()), node_idx, (int)node->op, nn, ni, mx};
    g_exec_by_addr[rec.addr] = rec;
    g_exec_by_tensor[node]   = rec;
}

// ============================================================
// exec_ab_check — 单节点 CPU/GPU 数值 A/B（诊断，默认关；2026-10-20）
// ============================================================
//   见 Backend.h 注释。要点：GPU kernel 是异步的 ⇒ 先 cudaDeviceSynchronize；
//   然后把 node/src 的 data_ 临时指向 host 缓冲（buffer_ 置空），用 CPUBackend
//   走**同一条 dispatch 路径**重算，逐元素比对后恢复指针。
namespace {

struct ABAliasSave {
    TensorF32* t    = nullptr;
    void*      data = nullptr;
    Buffer*    buf  = nullptr;
    size_t     offs = 0;
};

inline void ab_copy_to_host(const TensorF32* t, void* dst) {
    if (!t || !t->data() || !dst) return;
    if (t->buffer_) t->buffer_->get_tensor(t, dst, t->buffer_offs_, t->nbytes());
    else            std::memcpy(dst, t->data(), t->nbytes());
}

// CPU kernel 不需要 work buffer、语义单纯的 op（其余可用 PPML_AB_OPS 显式放开）
inline bool ab_op_ok(int op) {
    switch (op) {
        case OP_ADD:  case OP_SUB:  case OP_MUL:   case OP_DIV:
        case OP_SQR:  case OP_SQRT: case OP_SUM_ROWS: case OP_REPEAT:
        case OP_ADD1: case OP_CLAMP:  case OP_UNARY:
        case OP_SCALE:  case OP_DUP:  case OP_CPY:
        case OP_MUL_MAT: case OP_SOFT_MAX: case OP_CONCAT:   // 依赖 work buffer（按节点精确分配）
        case OP_GET_ROWS: case OP_EDGE_GATHER_ROWS: case OP_PER_EDGE_MATMUL:
        case OP_SCATTER_ADD: case OP_VIEW: case OP_RESHAPE:
            return true;
        default:
            return false;
    }
}

}  // namespace

void exec_ab_check(const TensorF32* node_c, const char* backend_name, int node_idx) {
    static const bool on = (std::getenv("PPML_AB_CHECK") != nullptr);
    if (!on || !node_c) return;
    // 只对照 GPU 侧算出来的节点（CPU 节点本身没有分歧可言）
    if (backend_name && std::strcmp(backend_name, "CPU") == 0) return;

    TensorF32* node = const_cast<TensorF32*>(node_c);
    if (node->op == OP_NONE || node->numel() <= 0 || !node->data()) return;

    static const int    max_print = [] { const char* e = std::getenv("PPML_AB_MAX"); return e ? std::atoi(e) : 20; }();
    static const double tol       = [] { const char* e = std::getenv("PPML_AB_TOL"); return e ? std::atof(e) : 1e-3; }();
    static const bool   abort_on  = (std::getenv("PPML_AB_ABORT") != nullptr);
    static const long long ab_start = [] { const char* e = std::getenv("PPML_AB_START"); return e ? std::atoll(e) : -1; }();
    static const long long ab_stop  = [] { const char* e = std::getenv("PPML_AB_STOP");  return e ? std::atoll(e) : -1; }();
    static const bool   op_filter = (std::getenv("PPML_AB_OPS") != nullptr);
    static bool op_want[512] = {false};
    static const bool op_init = [] {
        const char* e = std::getenv("PPML_AB_OPS");
        if (!e) return true;
        const char* p = e;
        while (*p) {
            while (*p == ',' || *p == ' ') ++p;
            if (!*p) break;
            const int v = std::atoi(p);
            if (v >= 0 && v < 512) op_want[v] = true;
            while (*p && *p != ',') ++p;
        }
        return true;
    }();
    (void)op_init;
    static int printed = 0;

    if (ab_start >= 0 && node_idx < (int)ab_start) return;
    if (ab_stop  >= 0 && node_idx > (int)ab_stop)  return;
    if (!op_filter && !ab_op_ok((int)node->op)) return;
    if (op_filter && !op_want[(int)node->op]) return;
    if (printed >= max_print) return;

    // GPU kernel 可能仍在跑 ⇒ 必须同步后才能读结果
    cudaDeviceSynchronize();

    const int64_t n = node->numel();
    const size_t  bytes = static_cast<size_t>(n) * sizeof(float);

    // ① GPU 结果
    std::vector<float> gpu(static_cast<size_t>(n), 0.f);
    ab_copy_to_host(node, gpu.data());

    // ② src → host 影子
    ABAliasSave saved_node;
    saved_node = {node, node->data(), node->buffer_, node->buffer_offs_};

    std::vector<ABAliasSave> saved_srcs;
    std::vector<std::vector<float>> src_bufs;
    // 注意：CONCAT 等 n-ary op 会用满 GGML_MAX_SRC 个 src；只换前 3 个会让 CPU 侧读到
    //   device 指针（或干脆不写）⇒ 假分歧（实测 cpu_max=0）。
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        TensorF32* s = node->src[j];
        if (!s) continue;
        saved_srcs.push_back({s, s->data(), s->buffer_, s->buffer_offs_});
        src_bufs.emplace_back(static_cast<size_t>(s->numel()) + 4, 0.f);
        ab_copy_to_host(s, src_bufs.back().data());
        s->bind_data(src_bufs.back().data());
        s->buffer_      = nullptr;
        s->buffer_offs_ = 0;
    }

    // ③ CPU 重算（dst 用独立 host 缓冲，先清零）
    std::vector<float> cpu(static_cast<size_t>(n), 0.f);
    node->bind_data(cpu.data());
    node->buffer_      = nullptr;
    node->buffer_offs_ = 0;

    static CPUBackend* ab_cpu = new CPUBackend(1);   // 故意不析构（避免静态析构顺序问题）
    // work buffer 按节点实际需求分配（mul_mat / softmax / concat 等都依赖它；
    //   给固定 4MB 时 concat 会走"work 不足 ⇒ 不写 dst"的分支 ⇒ cpu_max=0 的假分歧）。
    static std::vector<uint8_t> ab_work;
    const size_t need = CPUBackend::ab_estimate_work_size(node);
    if (ab_work.size() < need + 64) ab_work.resize(need + 64, 0);
    ComputeParams p_ab;
    p_ab.ith        = 0;
    p_ab.nth        = 1;
    p_ab.wsize      = ab_work.size();
    p_ab.wdata      = ab_work.empty() ? nullptr : ab_work.data();
    p_ab.threadpool = ab_cpu->ab_threadpool();
    const Status st = CPUBackend::ab_dispatch_node(node, &p_ab);

    // ④ 恢复指针（无论比对结果如何都必须恢复）
    node->bind_data(saved_node.data);
    node->buffer_      = saved_node.buf;
    node->buffer_offs_ = saved_node.offs;
    for (auto& sv : saved_srcs) {
        sv.t->bind_data(sv.data);
        sv.t->buffer_      = sv.buf;
        sv.t->buffer_offs_ = sv.offs;
    }
    if (st != Status::SUCCESS) return;

    // ⑤ 比对
    long long n_diff = 0, n_nan = 0;
    double max_rel = 0.0, gpu_max = 0.0, cpu_max = 0.0;
    long long first_i = -1;
    float first_g = 0.f, first_c = 0.f;
    for (int64_t i = 0; i < n; ++i) {
        const float g = gpu[i], c = cpu[i];
        const double ag = std::fabs((double)g), ac = std::fabs((double)c);
        if (ag > gpu_max) gpu_max = ag;
        if (ac > cpu_max) cpu_max = ac;
        bool bad;
        double rel = 0.0;
        if (std::isnan(g) || std::isnan(c)) { bad = true; ++n_nan; }
        else if (std::isinf(g) || std::isinf(c)) { bad = (ag != ac); ++n_nan; }
        else {
            const double denom = std::max(std::max(ag, ac), 1e-12);
            rel = std::fabs((double)g - (double)c) / denom;
            bad = (rel > tol);
        }
        if (bad) {
            ++n_diff;
            if (rel > max_rel) max_rel = rel;
            if (first_i < 0) { first_i = i; first_g = g; first_c = c; }
        }
    }
    if (n_diff == 0) return;

    ++printed;
    std::fprintf(stderr,
                 "[AB] #%d backend=%s node=%d op=%d numel=%lld dims=[%lld,%lld,%lld,%lld] "
                 "diff=%lld nan=%lld max_rel=%.3e gpu_max=%.4e cpu_max=%.4e "
                 "first[i=%lld gpu=%.6e cpu=%.6e]\n",
                 printed, backend_name ? backend_name : "?", node_idx, (int)node->op,
                 (long long)n,
                 (long long)node->shape().dim(0), (long long)node->shape().dim(1),
                 (long long)node->shape().dim(2), (long long)node->shape().dim(3),
                 n_diff, n_nan, max_rel, gpu_max, cpu_max, first_i, first_g, first_c);
    std::fflush(stderr);
    if (abort_on) std::abort();
}

// ============================================================
// exec_stale_* — "生产者代际"断言（诊断，默认关；2026-10-20）
// ============================================================
namespace {

struct StaleState {
    bool      on        = false;
    bool      abort_on  = false;
    int       max_print = 20;
    int       printed   = 0;
    long long hits      = 0;     // [STALE]：普通 node 输入的陈旧读
    long long leaf_hits = 0;     // [STALE-LEAF]：leaf 本身是"别的图的 node"且本代际未写
    long long pass      = 0;     // 当前 graph_compute 的代际
    int       sched_depth = 0;   // scheduler 正在驱动 split ⇒ 子 graph_compute 不另起代际
    std::unordered_map<const TensorF32*, long long>  written;   // 张量 → 最近写入代际
    std::unordered_set<const TensorF32*>             node_set;  // 本 pass 图里的 node
    std::unordered_set<const TensorF32*>             leaf_set;  // 本 pass 图里的 leaf（图输入）
    // 故障注入（PPML_STALE_SKIP_MARK=<node 下标>|all）：故意不登记该节点 ⇒ 用于自检
    //   "断言真的会报 [STALE]"（0 命中既可能真干净，也可能断言根本没生效）。
    int               skip_idx  = -1;
    const TensorF32*  skip_ptr  = nullptr;
    bool              skip_all  = false;
    long long         checks    = 0;   // 已执行过的"输入校验"次数（自检：为 0 ⇒ 断言没跑到）
};

StaleState& stale_state() {
    static StaleState s = [] {
        StaleState t;
        t.on       = (std::getenv("PPML_STALE_CHECK") != nullptr);
        t.abort_on = (std::getenv("PPML_STALE_ABORT") != nullptr);
        const char* e = std::getenv("PPML_STALE_MAX");
        t.max_print = e ? std::atoi(e) : 20;
        const char* k = std::getenv("PPML_STALE_SKIP_MARK");
        if (k && std::strcmp(k, "all") == 0) t.skip_all = true;
        else t.skip_idx = k ? std::atoi(k) : -1;
        if (t.on) {
            std::fprintf(stderr,
                         "[STALE] 启用：max_print=%d skip_mark=%s\n", t.max_print,
                         t.skip_all ? "all" : (t.skip_idx >= 0 ? "idx" : "off"));
            std::fflush(stderr);
            // 退出汇总：checks=0 ⇒ 断言根本没跑到（接线问题）；hits>0 ⇒ 有命中
            std::atexit([] {
                StaleState& s2 = stale_state();
                std::fprintf(stderr,
                             "[STALE] 退出汇总：pass=%lld checks=%lld hits=%lld leaf_hits=%lld\n",
                             s2.pass, s2.checks, s2.hits, s2.leaf_hits);
                std::fflush(stderr);
            });
        }
        return t;
    }();
    return s;
}

// scheduler 驱动 split 期间抑制"子 graph_compute 切代际"（异常/提前 return 亦自动复位）。
struct StaleSchedGuard {
    const bool active = stale_state().on;
    StaleSchedGuard()  { if (active) ++stale_state().sched_depth; }
    ~StaleSchedGuard() { if (active) --stale_state().sched_depth; }
};

}  // namespace

void exec_stale_pass_begin(const ComputeGraph* cgraph) {
    StaleState& st = stale_state();
    if (!st.on || !cgraph) return;
    if (st.sched_depth > 0) return;   // scheduler 内部逐 split 调后端 graph_compute ⇒ 不另起代际
    ++st.pass;
    st.node_set.clear();
    st.leaf_set.clear();
    // graph_node/graph_leaf 只有非 const 版本（返回可写 TensorF32*）⇒ 这里只读地取指针。
    ComputeGraph* g = const_cast<ComputeGraph*>(cgraph);
    for (int i = 0; i < g->n_nodes(); ++i) st.node_set.insert(g->graph_node(i));
    for (int i = 0; i < g->n_leafs(); ++i) st.leaf_set.insert(g->graph_leaf(i));
    st.skip_ptr = (st.skip_idx >= 0 && st.skip_idx < g->n_nodes())
                      ? g->graph_node(st.skip_idx) : nullptr;
    if (std::getenv("PPML_STALE_VERBOSE") && st.pass <= 5) {
        std::fprintf(stderr, "[STALE] pass=%lld nodes=%d leafs=%d skip_ptr=%p\n",
                     st.pass, g->n_nodes(), g->n_leafs(), (void*)st.skip_ptr);
        std::fflush(stderr);
    }
}

void exec_stale_mark(const TensorF32* t) {
    StaleState& st = stale_state();
    if (!st.on || !t) return;
    if (st.skip_all || t == st.skip_ptr) return;   // 故障注入自检：故意不登记
    st.written[t] = st.pass;
}

void exec_stale_check(const TensorF32* node, const char* backend_name, int node_idx) {
    StaleState& st = stale_state();
    if (!st.on || !node) return;
    ++st.checks;
    // 本节点的 src 必须在本代际被写过（常量/view/cpy/参数 例外）
    //   【2026-10-20 补充 STALE-LEAF】原来"本图 leaf"整类免检 ⇒ 漏掉一种真实形态：
    //   compute_and_read 子图的 leaf 恰恰是**主图尚未计算的图节点**（op != OP_NONE），
    //   子图会拿上一代际的 buffer 当输入算（实测混合 CUDA 的 SE3 per-block 子图即如此）。
    //   现在：只有"真外部输入"（op == OP_NONE，见上面的 continue）才免检；
    //   leaf 若本身是别的图的 node，则与普通 node 同标准 ⇒ 命中打 [STALE-LEAF]。
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        const TensorF32* s = node->src[j];
        if (!s) continue;
        if (s->op == OP_NONE) continue;                 // 参数/常量/外部输入
        if (s->view_src) continue;                      // 零拷贝别名：由根负责
        if (s->flag & TENSOR_FLAG_PARAM) continue;      // 参数
        auto it = st.written.find(s);
        const long long w = (it == st.written.end()) ? -1 : it->second;
        if (w == st.pass) continue;
        const bool is_leaf = st.leaf_set.count(s) != 0;
        if (is_leaf) ++st.leaf_hits; else ++st.hits;
        if (st.printed < st.max_print) {
            ++st.printed;
            std::fprintf(stderr,
                         "%s backend=%s node=%d op=%d src%d(op=%d numel=%lld flag=%d) "
                         "最近写入代际=%lld 当前=%lld dims=[%lld,%lld,%lld,%lld]%s\n",
                         is_leaf ? "[STALE-LEAF]" : "[STALE]",
                         backend_name ? backend_name : "?", node_idx, (int)node->op, j,
                         (int)s->op, (long long)s->numel(), (int)s->flag, w, st.pass,
                         (long long)s->shape().dim(0), (long long)s->shape().dim(1),
                         (long long)s->shape().dim(2), (long long)s->shape().dim(3),
                         is_leaf ? "（本图 leaf，但它是别的图的 node 且本代际未写）" : "");
            std::fflush(stderr);
        }
        if (st.abort_on) std::abort();
    }
}

void exec_stale_report(const char* where) {
    StaleState& st = stale_state();
    if (!st.on || (st.hits == 0 && st.leaf_hits == 0)) return;
    std::fprintf(stderr, "[STALE] 汇总(%s)：命中=%lld（其中 leaf=%lld，明细仅前 %d 条）\n",
                 where ? where : "?", st.hits + st.leaf_hits, st.leaf_hits, st.max_print);
    std::fflush(stderr);
}

// ============================================================
// exec_probe_op — 定点算子探针（诊断，默认关；2026-10-20）
// ============================================================
//   动机：CPU 与混合 CUDA 路径**图构型不同**（混合模式下 SE3 per_block 走独立子图，
//   节点编号完全对不上），无法用"同一 node_idx"逐节点对比。本探针按 **op** 对齐：
//   每执行一个指定 op 的节点，就打印它的 src[0..2]/dst 的 maxabs ⇒ 同一逻辑算子在两条
//   路径上的**输入/输出量级**可直接并排比。
//   开关：PPML_PROBE_OP=8,51,33（逗号分隔 op 编号）；PPML_PROBE_MAX=40（打印上限）。
namespace {
struct ProbeCfg {
    bool on = false;
    bool ops[512] = {false};
    int  max_print = 40;
    int  printed   = 0;
};

ProbeCfg& probe_cfg() {
    static ProbeCfg c = [] {
        ProbeCfg t;
        const char* e = std::getenv("PPML_PROBE_OP");
        if (!e || !*e) return t;
        t.on = true;
        const char* p = e;
        while (*p) {
            char* end = nullptr;
            long v = std::strtol(p, &end, 10);
            if (end == p) { ++p; continue; }
            if (v >= 0 && v < 512) t.ops[v] = true;
            p = (*end == ',') ? end + 1 : end;
        }
        const char* m = std::getenv("PPML_PROBE_MAX");
        t.max_print = m ? std::atoi(m) : 40;
        return t;
    }();
    return c;
}

// 单张量 maxabs / 最小非零幅值（device 走分块 D2H，复用 exec_scan_tensor）
double probe_maxabs(const TensorF32* t, double* minpos = nullptr) {
    if (minpos) *minpos = 0.0;
    if (!t) return -1.0;
    long long nn = 0, ni = 0, fi = -1;
    double mx = 0.0, mp = 1e300;
    float fb = 0.0f;
    if (!exec_scan_tensor(t, 1e30f, &nn, &ni, &mx, &fb, &fi, nullptr, nullptr, &mp)) return -1.0;
    if (minpos) *minpos = (mp < 1e299) ? mp : 0.0;
    return mx;
}
}  // namespace

void exec_probe_op(const TensorF32* node, const char* backend_name, int node_idx) {
    ProbeCfg& c = probe_cfg();
    if (!c.on || !node) return;
    const int op = static_cast<int>(node->op);
    if (op < 0 || op >= 512 || !c.ops[op]) return;
    if (c.max_print >= 0 && c.printed >= c.max_print) return;
    ++c.printed;
    double min0 = 0.0, min1 = 0.0;
    const double max0 = probe_maxabs(node->src[0], &min0);
    const double max1 = probe_maxabs(node->src[1], &min1);
    std::fprintf(stderr,
                 "[PROBE] #%d backend=%s node=%d op=%d numel=%lld dims=[%lld,%lld,%lld,%lld] "
                 "dst_max=%.4e | src0(op=%d,numel=%lld,max=%.4e) "
                 "src1(op=%d,numel=%lld,max=%.4e,min_nonzero=%.4e)\n",
                 c.printed, backend_name ? backend_name : "?", node_idx, op,
                 (long long)node->numel(),
                 (long long)node->shape().dim(0), (long long)node->shape().dim(1),
                 (long long)node->shape().dim(2), (long long)node->shape().dim(3),
                 probe_maxabs(node),
                 node->src[0] ? (int)node->src[0]->op : -1,
                 node->src[0] ? (long long)node->src[0]->numel() : -1, max0,
                 node->src[1] ? (int)node->src[1]->op : -1,
                 node->src[1] ? (long long)node->src[1]->numel() : -1, max1, min1);
    std::fflush(stderr);
}

// ============================================================
// backend_tensor_copy — 对标 ggml_backend_tensor_copy
// 将 src tensor 的数据拷贝到 dst tensor（跨后端自动处理 CPU↔GPU）
// ============================================================
static bool backend_tensor_copy_impl(const TensorF32* src, TensorF32* dst) {
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
            // 【2026-09-19 P1】优先异步 H2D：pinned 源 ⇒ copy 流 async + 默认流等事件 ⇒ **无需 host 等待** ✓
            //   （源已在 pinned（例如 Step 1b 的 arena）⇒ 零拷贝直接用 ✓；否则先 CPU memcpy 进 pinned ✓）
            if (is_cuda_device_tensor(dst) && staging_enabled()) {
                const void* src_host = src->data();
                void* pin = nullptr;
                if (staging_is_pinned(src_host)) {
                    pin = const_cast<void*>(src_host);
                } else {
                    pin = staging_alloc((int64_t)nbytes);
                    if (pin) std::memcpy(pin, src_host, nbytes);
                }
                if (pin) {
                    void* dst_dev = static_cast<uint8_t*>(dst->buffer_->data()) + dst->buffer_offs_;
                    if (staging_h2d(dst_dev, pin, (int64_t)nbytes) == 0) return true;
                }
            }
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
            // 【2026-09-19 P1】优先异步 D2H：目标本身在 pinned ⇒ 直接异步拷入；否则先到 pinned 再 memcpy ✓
            if (is_cuda_device_tensor(src) && staging_enabled()) {
                float* dst_host = reinterpret_cast<float*>(dst->data());
                void*  pin = nullptr;
                bool   copy_back = false;
                if (staging_is_pinned(dst_host)) {
                    pin = dst_host;                     // 目标已在 arena ⇒ 直接拷，省一次 memcpy ✓
                } else {
                    pin = staging_alloc((int64_t)nbytes);
                    copy_back = true;
                }
                if (pin) {
                    const void* src_dev =
                        static_cast<const uint8_t*>(src->buffer_->data()) + src->buffer_offs_;
                    if (staging_d2h(pin, src_dev, (int64_t)nbytes) == 0 && staging_wait() == 0) {
                        if (copy_back) std::memcpy(dst_host, pin, nbytes);
                        return true;
                    }
                }
            }
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

// 公开入口：包一层以便在**成功拷贝**后登记"该张量本代际已写"（生产者代际断言用）。
//   cpy（OP_DUP 跨后端拷贝）不入图 nodes，不经 dispatch ⇒ 必须在这里补登记，
//   否则它的消费者会被 [STALE] 误报。
bool backend_tensor_copy(const TensorF32* src, TensorF32* dst) {
    const bool ok = backend_tensor_copy_impl(src, dst);
    if (ok) exec_stale_mark(dst);
    return ok;
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

    // 【2026-10-20 修复：reserved_buffers_ 释放前先解绑 cpy 张量 —— 悬垂指针 / 假别名根因】
    //   reserved_buffers_ 是 vector<unique_ptr<Buffer>> ⇒ clear() 会 **delete** Buffer 对象，
    //   但 copy_tensor_map_ 里的 cpy 张量仍持有指向它们的 data_/buffer_ ✗。后果有二：
    //     ① 下一次 pass_assign_leafs 里 `leaf->buffer_->is_host()` 是 use-after-free 的虚调用
    //        （实测 SIGSEGV，rip=0x400，栈 pass_assign_leafs ← split_graph）✗；
    //     ② 步骤 7 的重分配条件是 `cpy->data() == nullptr` —— data_ 悬垂非空 ⇒ **不会重分配**
    //        ⇒ cpy 继续写已释放的内存，而新图的新 Buffer 可能拿到同一段地址 ⇒ 两个张量共享
    //        物理内存，表现为"某节点写完后，下游读到的却是更靠后节点的值"（执行期 NaN 的形态）✗。
    //   修法：清表之前把所有 cpy 张量的绑定复位（参数除外——参数由 param_buffers_ 持有）。
    {
        int unbound = 0;
        for (auto& kv : copy_tensor_map_) {
            TensorF32* cpy = kv.second;
            if (!cpy || (cpy->flag & TENSOR_FLAG_PARAM)) continue;
            if (cpy->buffer_ != nullptr || cpy->data() != nullptr) {
                cpy->bind_data(nullptr);
                cpy->buffer_      = nullptr;
                cpy->buffer_offs_ = 0;
                aliasguard::unregister(cpy);   // 解绑 ⇒ 撤销假别名登记
                ++unbound;
            }
        }
        if (unbound > 0 && getenv("GRAPH_DEBUG_GALLOCR")) {
            fprintf(stderr, "[sched] reserved 释放前解绑 cpy 张量 %d 个（防悬垂/假别名）\n", unbound);
        }
    }
    reserved_buffers_.clear();
    // 【刻意不 drop_pool("cpy")】
    //   上面已把**正常** cpy 全部解绑并逐条 unregister；仍留在 "cpy" pool 里的登记 =
    //   data_ 悬垂非空、没被解绑的 cpy（即 2026-10-20 注释里那种"两个张量共享物理内存"的隐患）。
    //   保留它们，等后续 gallocr/新 cpy 绑到同一地址时立刻触发 [ALIAS]，并指出是哪条陈旧登记。
    aliasguard::report("sched::reserved_clear");
    // 必须先 release 上一轮 gallocr buffer：Gallocr::alloc 每次 push 新 buffer 到
    //    ba.buffers（只有 release() 才 delete）。混合多样本下每个样本都走 reserve+alloc，
    //    若不先释放，旧 buffer 永不回收 → RSS 每样本 +5~6GB 单调增长直至 OOM。
    //    实测（2026-08-25）：纯 CPU 多样本 RSS 稳定 ~4.9GB；混合模式每样本 +5GB。
    gallocr_.release();

    // 【2026-10-20 修复】release() 之后，图里仍可能有张量持有**已 delete 的 Buffer**：
    //   release 只按自己快照（nodes_/leaves_）里的 ni.buffer 复位，且**跳过 PARAM**；
    //   一旦某叶子/节点不在快照内（如新图新增、或由 run_path 绑到 persistent buffer），
    //   它的 buffer_ 就悬垂 ✗。下一次 pass_assign_leafs 里 `leaf->buffer_->is_host()`
    //   是 use-after-free 虚调用（实测 SIGSEGV，rip=0x400）。
    //   这里按【图本身】兜底复位：非 PARAM 且 buffer_ 非空 → 解绑（reserve/alloc 会重绑）。
    if (current_graph_) {
        int fixed = 0;
        auto reset_dangling = [&](TensorF32* t) {
            if (!t || (t->flag & TENSOR_FLAG_PARAM)) return;
            if (t->buffer_ != nullptr) {
                t->bind_data(nullptr);
                t->buffer_      = nullptr;
                t->buffer_offs_ = 0;
                aliasguard::unregister(t);   // 解绑 ⇒ 撤销假别名登记
                ++fixed;
            }
        };
        for (int i = 0; i < current_graph_->n_nodes(); ++i) reset_dangling(current_graph_->graph_node(i));
        for (int i = 0; i < current_graph_->n_leafs(); ++i) reset_dangling(current_graph_->graph_leaf(i));
        if (fixed > 0 && getenv("GRAPH_DEBUG_GALLOCR")) {
            fprintf(stderr, "[sched] release 后兜底解绑 %d 个悬垂张量\n", fixed);
        }
    }

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
                // 【2026-10-20】假别名断言：cpy 张量的分配区间单独登记（pool="cpy"）。
                //   本步分配的 Buffer 与 gallocr arena 是两个独立 Buffer，正常不重叠；
                //   若这里报 [ALIAS]，说明某条**陈旧登记**（未被解绑的旧 cpy / 悬垂指针）
                //   仍占着这段地址 —— 正是"下游读到更靠后节点内容"的假别名形态。
                aliasguard::register_range(cpy, cpy->data(),
                                           GGML_PAD(buft->get_alloc_size(cpy), buft->get_alignment()),
                                           "cpy");
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
    // 【2026-09-17】通知各后端"新图开始"：远端用它在切图前**自动计算**节点区间
    //   （PPML_REMOTE_NODES=auto ⇒ 不用手填节点号 ✓）；其他后端默认无操作 ✓
    for (int b = 0; b < n_backends_; ++b) {
        if (backends_[b]) backends_[b]->on_graph_begin(graph);
    }
    n_splits_ = 0;
    splits_.clear();
    n_graph_inputs_ = 0;
    backend_map_.clear();
    copy_tensor_map_.clear();

    // 显存预算：GPU 空闲显存扣 20% 余量作预算；0 = 不限（无 GPU 或显存探测失败）。
    gpu_reserved_bytes_ = 0;
    // 【2026-09-22】H2D cpy 记账清零（每次构建重新统计 ✓）；顺带回显上一次的预留量（便于观察 ✓）
    if (getenv("GRAPH_DEBUG_SCHED") && (host_cpy_reserved_ > 0 || !host_cpy_bytes_.empty())) {
        fprintf(stderr, "[sched] H2D cpy 预留（上一构建）: %.2f MB / %zu 个 host 源\n",
                (double)host_cpy_reserved_ / 1048576.0, host_cpy_bytes_.size());
    }
    host_cpy_bytes_.clear();
    host_cpy_reserved_ = 0;
    if (n_backends_ > 1) {
        // 仅当存在非 CPU 后端（CUDA）时启用预算
        bool has_gpu = false;
        for (int b = 0; b < n_backends_; b++) {
            if (backends_[b]->vram_budgeted()) { has_gpu = true; break; }   // 远端点不算 GPU ✓
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
            // 【2026-09-17 关键修复（DUP 机制正解）】host 数据叶子（参数/常量/输入，
            //   buffer_==null 或 host buffer）**只能放 host 后端（CPU）** ✗：
            //   非 host 后端（GPU / remote）不能直接持有 host 叶子——否则 remote 在 slim=1 下
            //   本地 staging 从不写 ⇒ get_tensor 取不回 ⇒ 刷 `数据不可得 … op=0 is_remote=0` 卡死 ✗。
            //   正确做法：host 叶子留 CPU（数据本就 host），build_splits 为非 host 消费者
            //   **建 DUP cpy**（remote 侧上传 / GPU 侧 H2D）✓ —— 这才是"省内存"的正解。
            //   原代码只对 vram_budgeted()（GPU）做 host_data 检查，remote(vram_budgeted=false) 漏了 ✗。
            const bool host_data = (leaf->buffer_ == nullptr) || leaf->buffer_->is_host();
            if (host_data && !backends_[b]->buffer_type()->is_host()) continue;
            if (backends_[b]->vram_budgeted()) {
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

            // 【2026-09-19 ①-B】op=0 且**数据确实在 host**（无 data 的裸常量也按 host 处理）⇒ 钉死 CPU ✓。
            //   ⚠️ 必须自己查数据位置，**不能用 node_is_host_producer**：它对**所有** op=0 都返回 true ✗
            //      （Backend.cpp:100 先判 op==OP_NONE ✗），会把 device 常驻的权重/常量也拉回 host ✗
            //      —— 实测这么做会让 epoch 19.9s→28.8s 且 loss 漂移 ⚠️。device 常驻的 op=0 保持原逻辑 ✓。
            if (node->op == OP_NONE) {
                // 【2026-09-22】改用构建期冻结的居住地判定（sched_data_is_host ✓）：原先直接看
                //   运行期 buffer_ 状态 ⇒ 同配置两次 run 可能给出不同归属 ✗（偶发 CUDA 裸读 host ✗）。
                const bool host_data = sched_data_is_host(node);
                if (host_data) {
                    if (tensor_backend_id(node) == -1) backend_map_[node] = n_backends_ - 1;  // CPU（最低优先级）
                    continue;
                }
            }

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
                set_backend_if_supported(node, cur_backend_id, i);
            }
        }
    }

    // ===== 向上扩展（从底向上遍历）=====
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes() - 1; i >= 0; i--) {
            TensorF32* node = graph->graph_node(i);
            if (is_view_op(node->op)) continue;

            // 【2026-09-19 ①-B】同上（向上扩展）：op=0 且数据在 host ⇒ 钉 CPU ✓；
            //   device 常驻的 op=0 保持原逻辑 ✓（不能用 node_is_host_producer ✗ 它把所有 op=0 都当 host ✗）
            if (node->op == OP_NONE) {
                // 【2026-09-22】同上（向上扩展）：构建期冻结判定 ✓
                const bool host_data = sched_data_is_host(node);
                if (host_data) {
                    if (tensor_backend_id(node) == -1) backend_map_[node] = n_backends_ - 1;
                    continue;
                }
            }

            int node_id = tensor_backend_id(node);

            if (node_id != -1) {
                if (node_id == n_backends_ - 1) {
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = node_id;
                }
            } else if (cur_backend_id != -1) {
                set_backend_if_supported(node, cur_backend_id, i);
            }
        }
    }
}

void BackendScheduler::pass_fill_unassigned(ComputeGraph * graph) {
    for (int i = 0; i < graph->n_nodes(); i++) {
        TensorF32* node = graph->graph_node(i);
        if (sched_verbose()) {
            // 诊断（GRAPH_DEBUG_SCHED_VERBOSE=1）：检测栈/堆地址范围的 node，定位悬垂指针
            uintptr_t a = (uintptr_t)node;
            int in_stack = (a >= 0x700000000000ULL && a <= 0x800000000000ULL);
            // 【2026-09-19】补 data/buffer 信息：判断 op=0 节点是"host 参数/常量"还是"device 常驻" ✓
            fprintf(stderr,
                "[sched] node[%d]=%p op=%d ndim=%d numel=%lld flag=0x%x data=%p buf=%p buf_host=%d %s\n",
                i, (void*)node, (int)node->op, (int)node->shape().ndim(),
                (long long)node->numel(), (unsigned)node->flag,
                (void*)node->data(), (void*)node->buffer_,
                node->buffer_ ? (node->buffer_->is_host() ? 1 : 0) : -1,
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
            if (sched_verbose()) {
                fprintf(stderr,
                    "[sched] pass_fill node=%p op=%d n_bk=%d best0=%d bk0=%p v0=%p bk1=%p v1=%p\n",
                    (void*)node, (int)node->op, n_backends_, best_backend,
                    (void*)(n_backends_>0?backends_[0]:nullptr),
                    (n_backends_>0 && backends_[0])?*(void**)backends_[0]:nullptr,
                    (void*)(n_backends_>1?backends_[1]:nullptr),
                    (n_backends_>1 && backends_[1])?*(void**)backends_[1]:nullptr);
            }
            for (int b = 0; b < n_backends_; b++) {
                if (sched_verbose()) {
                    fprintf(stderr, "[sched]   try b=%d node=%p bk=%p v=%p\n",
                            b, (void*)node,
                            (void*)backends_[b],
                            backends_[b] ? *(void**)backends_[b] : nullptr);
                }
                backends_[b]->set_sched_index(i);   // ★ 区间选择（远端 PPML_REMOTE_NODES ✓）
                if (backends_[b]->supports_op(node)) {
                    best_backend = b;  // backends_ 已按 priority 降序 → 第一个即最高
                    break;
                }
            }
            // 若选中的是 GPU 且显存预算不足 → 回落 CPU（最后后端）。
            //   远端后端 vram_budgeted()==false ⇒ 不参与（否则大批区间节点会被预算逐出 ✗）
            if (backends_[best_backend]->vram_budgeted() &&
                !gpu_assign_if_affordable(node, best_backend)) {
                best_backend = n_backends_ - 1;  // CPU
            }

            // （已回滚 2026-08-24：OUT_PROD/OUTER_PROD_BACK 强制 CPU 引入更多跨后端 H2D，loss 更不稳。
            //   跨后端需系统性修 Step 1/1b 的 D2H/H2D buffer 生命周期，见 memory。）

            // ===== 混合训练正确性护栏（避免段错误）=====
            // 仅广播 op 强制回落 CPU：CUDA elemwise kernel 用扁平 idx<n 索引，
            // 不支持广播（src numel != dst numel 越界）。host 源不再整节点禁用 GPU，
            // 改由 build_splits 插入 H2D 拷贝 + graph_compute 兜底处理。
            //   注：远端后端 vram_budgeted()==false ⇒ 跳过本护栏 ✓（服务端是 CPU 执行器，
            //   广播语义正确 ✓；否则区间会被广播 op 切碎成很多小 split ✗）
            if (backends_[best_backend]->vram_budgeted()) {
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
                    OP_SCALE, OP_SQR, OP_SQRT, OP_LOG,
                    OP_SIN, OP_COS, OP_LEAKY_RELU, OP_UNARY, OP_CLAMP,
                    OP_FILL, OP_ARANGE, OP_GLU,
                    // 门控融合 act_mul（2026-10-18）：kernel 内对 g 做 i % g_numel 广播，
                    //   与 OP_MUL 同语义 ⇒ 前缀对齐可留 GPU，否则回落 CPU（见下方 kernel_has_bcast）
                    OP_ACT_MUL, OP_ACT_MUL_BACK_X, OP_ACT_MUL_BACK_G,
                    // 2026-10-18 移除：OP_SUM / OP_MEAN（全元素归约，src numel≠dst 是**归约语义**
                    //   而非广播；CUDA sum/mean kernel 本就按 src numel 归约）、
                    //   OP_OUTER_PROD / OP_OUTER_PROD_MEAN / OP_TRI_MUL（CUDABackend::supports_op
                    //   返回 false，永远到不了 GPU，放这里只是死条目）。
                    -1
                };
                bool is_elemwise = false;
                for (int k = 0; kElemwiseBroadcastOps[k] >= 0; k++) {
                    if (node->op == (tensor_op)kElemwiseBroadcastOps[k]) { is_elemwise = true; break; }
                }
                bool real_broadcast = broadcast && is_elemwise;
                // (2') 2026-10-18：ADD/SUB/MUL/DIV 的 CUDA kernel 已在**体内**做广播
                //   （bcast_elemwise_cuda：src[i % src_numel]，与 CPU kernel_elemwise 同语义，
                //    该 kernel 2026-08-31 落地，晚于本护栏 2026-08-22 ⇒ 上面的旧注释已过期）。
                //   只要每个 src 都是"最内维前缀对齐"或满尺寸，就允许留在 GPU；
                //   其余（如外层维广播 [1,M]→[D,M]）仍回落 CPU，避免静默错值。
                if (real_broadcast) {
                    const tensor_op eop = node->op;
                    // OP_ADD1：src[1] 是标量（numel=1），bcast_prefix_aligned 对"1 元素源"恒真；
                    //   其 CUDA kernel 读 src[1][0]（标量由 build_splits 的 H2D cpy 搬到 device）。
                    const bool kernel_has_bcast =
                        (eop == OP_ADD || eop == OP_SUB || eop == OP_MUL || eop == OP_DIV ||
                         eop == OP_ADD1 ||
                         // act_mul 前向/反向 wrt x：g(src1/src2) 用 i % g_numel 广播
                         eop == OP_ACT_MUL || eop == OP_ACT_MUL_BACK_X);
                    if (kernel_has_bcast) {
                        bool aligned = true;
                        for (int j = 0; j < GGML_MAX_SRC; ++j) {
                            TensorF32* s = node->src[j];
                            if (!s) continue;
                            if ((long long)s->numel() == (long long)node->numel()) continue;
                            if (!bcast_prefix_aligned(node, s)) { aligned = false; break; }
                        }
                        if (aligned) real_broadcast = false;   // kernel 能承接 ⇒ 不驱逐
                    }
                }
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
        // 【2026-09-19 标签修正】`backends_` 按 priority **降序**排序 ⇒ **下标 0 = 最高优先级（GPU/CUDA）**，
        //   最后一个下标 = CPU（`n_backends_-1`）✓。原打印把 [0] 当 CPU、把 [1..] 当 GPU ⇒ **与实际相反** ✗
        //   （此前分析被它误导过 ✓）⇒ 现在直接打印后端名字，杜绝猜 ✓。
        fprintf(stderr, "[sched] split_graph: n_backends=%d（下标 0 = 最高优先级）\n", n_backends_);
        for (int b = 0; b < n_backends_; b++) {
            fprintf(stderr,
                    "[sched]   backend[%d] name=%s priority=%d vram_budgeted=%d op_count=%d last_op=%lld\n",
                    b, backends_[b] ? backends_[b]->get_name() : "?",
                    backends_[b] ? backends_[b]->priority() : -1,
                    backends_[b] ? (int)backends_[b]->vram_budgeted() : -1,
                    sched_backend_cnt_[b], sched_backend_op_last_[b]);
        }
        // 【2026-09-19】预算逐出汇总：哪些 op 被"预算不足"逼回 CPU（前 6 个）✓
        if (g_budget_refuse_cnt_ > 0) {
            fprintf(stderr, "[sched] budget 拒绝总数=%ld，按 op 前 6：",
                    g_budget_refuse_cnt_);
            for (int rank = 0; rank < 6; ++rank) {
                long best = 0; int best_op = -1;
                for (int o = 1; o < 160; ++o) {
                    if (g_budget_refuse_by_op_[o] > best) { best = g_budget_refuse_by_op_[o]; best_op = o; }
                }
                if (best_op < 0) break;
                fprintf(stderr, " op=%d×%ld", best_op, best);
                g_budget_refuse_by_op_[best_op] = -1;   // 已输出，避免重复
            }
            fprintf(stderr, "\n");
        }
    }
}

// ============================================================
// build_splits — Pass 5: 按 backend 边界切分子图 + 创建跨后端拷贝节点
// 对标 ggml_backend_sched_split_graph 的 pass 5
// ============================================================
void BackendScheduler::build_splits(ComputeGraph* graph) {
    int n_nodes = graph->n_nodes();

    // ===== Step 1: 跳过开头的 view op / op=0，确定第一个 split 的 backend =====
    //   【2026-09-19 ①-A】op=0（参数/常量/输入，无 kernel）与 view 一样**不参与 backend 选择** ✗：
    //   否则"开头一串 op=0"会先起一个只装 no-op 的 split ✗（实测有 824 个 1 节点孤岛 ✓）。
    int i = 0;
    for (; i < n_nodes; i++) {
        TensorF32* node = graph->graph_node(i);
        if (is_view_op(node->op) || node->op == OP_NONE) continue;
        break;
    }
    if (i >= n_nodes) return;  // 全是 view / op=0

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
        //   【2026-09-19 ①-A】op=0 对切分**透明**：no-op 节点不因自己的归属切开 split ✗。
        //   安全性（已逐条核对 ✓）：
        //     · 两个后端的 dispatch 都 skip OP_NONE（CUDABackend.cpp:419 / CPUBackend.cpp:367 ✓）⇒
        //       它被并进 CUDA split 也只是个 no-op，不会被任何 kernel 读 ✗；
        //     · 跨端拷贝按"src 数据实际位置 + backend 归属"判定（下方 3c）⇒ 与 split 边界无关 ✓；
        //     · 归属 / buffer / 预算一概不动 ⇒ alloc 与拷贝语义与原实现完全一致 ✓。
        //   注：`split_backend_id` 只用于**是否切分**的判断；真起新 split 时仍写 node_backend_id ✓。
        //   开关：PPML_SCHED_OP0_MERGE=0 关闭本行为（A/B 用，默认开 ✓）。
        static const bool op0_merge = []() {
            const char* e = std::getenv("PPML_SCHED_OP0_MERGE");
            return !(e && *e && std::atoi(e) == 0);
        }();
        const int split_backend_id =
            (op0_merge && node->op == OP_NONE) ? cur_backend_id : node_backend_id;
        if (split_backend_id != cur_backend_id || need_new_split) {
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
            // 【2026-10-20】诊断：进入 3c 时 src 已经是 OP_DUP ⇒ 说明它不是本 pass 刚建的，
            //   而是上一轮 split（或被模型代码）留下的 cpy。这种情况下 target 可能属于别的后端。
            static const bool dbg_cpy_src = (std::getenv("PPML_DEBUG_CPY_SRC") != nullptr);
            if (dbg_cpy_src && src->op == OP_DUP) {
                fprintf(stderr, "[cpy-src] node=%d op=%d j=%d cur_bk=%d src=%p src0=%p src0_op=%d src_buf_host=%d\n",
                        i, (int)node->op, j, cur_backend_id, (void*)src, (void*)src->src[0],
                        src->src[0] ? (int)src->src[0]->op : -1,
                        (src->buffer_ ? (int)src->buffer_->is_host() : -1));
            }

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
            //   【2026-09-22 改造】判据改为**构建期冻结**的 sched_data_is_host ✓（原先用运行期
            //   `src->buffer_` 状态 ⇒ 同配置两次 run 判据不同 ⇒ 偶发漏建 cpy ⇒ GPU 裸读 host ✗）。
            bool src_data_host = sched_data_is_host(src);
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
                    if (dbg_cpy_src) {
                        fprintf(stderr, "[cpy-new] node=%d op=%d j=%d cur_bk=%d src=%p src_op=%d src_bk=%d cpy=%p\n",
                                i, (int)node->op, j, cur_backend_id, (void*)src, (int)src->op,
                                src_backend_id, (void*)cpy);
                    }

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
        // 【2026-09-19】合并上限**按后端区分**（原实现 CPU/GPU/远端共用 64 ✗）：
        //   · CPU 段：仍限 64 节点 —— 原注释的理由（"防止 CPU 大 split 阻塞线程池"）**只对 CPU 成立** ✓
        //   · 非 CPU（GPU / REMOTE 等）：**不限制**（0 = 无上限）——
        //     这类后端的痛点是"**每个 split 的固定开销**"：一次全设备同步（实测 1,844 次 / 754 ms）
        //     + 边界 staging 拷贝（11,767 次 cudaMemcpy / 2.36 s），而节点粒度交替让一次
        //     graph_compute 产生 4,015 个 split ⇒ 放开上限能直接把它们压回几十~几百个 ✓
        //   · 语义不变：只合并**连续同 backend** 段；跨后端边界与执行顺序完全不动 ✓
        //   · CPU 路径与原先**逐字等价**（cap=64 时判据与旧代码同一表达式 ✓）
        //   可用 PPML_MERGE_MAX_NODES_CPU / PPML_MERGE_MAX_NODES_OTHER 覆盖（0 = 不限制，便于 A/B ✓）
        auto merge_cap_of = [this](int backend_id) -> int {
            Backend* b = (backend_id >= 0 && backend_id < n_backends_) ? backends_[backend_id] : nullptr;
            const bool is_cpu = (b && std::strcmp(b->get_name(), "CPU") == 0);
            const char* env = std::getenv(is_cpu ? "PPML_MERGE_MAX_NODES_CPU"
                                                 : "PPML_MERGE_MAX_NODES_OTHER");
            if (env && *env) return std::atoi(env);
            return is_cpu ? 64 : 0;   // 0 = 无上限
        };
        std::vector<SplitInfo> merged;
        merged.reserve(static_cast<size_t>(n_splits_));
        const int n_pre_merge = n_splits_;   // 诊断：合并前的 split 数 ✓
        int n_merge_events = 0;              // 诊断：实际发生的合并次数 ✓
        SplitInfo acc = splits_[0];
        for (int si = 1; si < n_splits_; si++) {
            SplitInfo& cur = splits_[si];
            const int cap = merge_cap_of(acc.backend_id);   // 上限按**当前累积段**的后端定 ✓
            if (cur.backend_id == acc.backend_id &&
                (cap <= 0 || (cur.i_end - acc.i_start) <= cap)) {
                acc.i_end = cur.i_end;   // 同 backend：直接扩展范围
                ++n_merge_events;
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
        if (getenv("GRAPH_DEBUG_SCHED")) {
            fprintf(stderr,
                    "[sched] split_graph: pre-merge n_splits=%d → after merge n_splits=%d"
                    "（合并 %d 次）\n",
                    n_pre_merge, n_splits_, n_merge_events);
        }
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

    // 生产者代际断言：整张图（全部 split）共用一个代际；子 graph_compute 被 sched_depth 抑制。
    exec_stale_pass_begin(current_graph_);
    StaleSchedGuard stale_guard;

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
            // 【P1】先收集（**先不 bind**），等一次成功后再统一 bind ⇒ 失败可整体回落 ✓
            std::vector<std::pair<TensorF32*, float*>> pending_stage;
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
                    // 【2026-09-19 P1】优先：pinned arena + 异步 D2H ⇒ **直接把 pinned 指针 bind 上去** ✓
                    //   好处：① 不再 per-split 新建 host_scratch_ vector ✗ ② 少一次拷贝（CPU kernel 直接读 pinned ✓）
                    //   同步：本 split 全部 issue 完后**统一等一次**（见下方 staging_wait ✓），不逐张量等 ✓
                    if (is_cuda_device_tensor(src) && staging_enabled()) {
                        const int64_t bytes = n * (int64_t)sizeof(float);
                        void* pin = staging_alloc(bytes);
                        if (pin) {
                            const void* src_dev =
                                static_cast<const uint8_t*>(src->buffer_->data()) + src->buffer_offs_;
                            if (staging_d2h(pin, src_dev, bytes) == 0) {
                                pending_stage.emplace_back(src, reinterpret_cast<float*>(pin));
                                continue;                                     // 等统一 bind ✓
                            }
                        }
                    }
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
            // 【2026-09-19 P1】本 split 的异步 D2H 已全部入队 ⇒ **统一等一次**（host 侧唯一等待 ✓）；
            //   等成功后才 bind pinned 指针（失败 ⇒ 这些张量整体回落阻塞拷贝，不留半成品状态 ✗）
            if (!pending_stage.empty()) {
                if (staging_wait() == 0) {
                    for (auto& pr : pending_stage) {
                        host_stage_.emplace_back(pr.first, pr.first->data());   // 沿用既有恢复逻辑 ✓
                        pr.first->bind_data(pr.second);                         // ★ 直接 bind pinned ✓
                    }
                } else {
                    std::fprintf(stderr,
                                 "[STAGING] D2H 等待失败 ⇒ 本 split %zu 个输入回落阻塞拷贝 ✓\n",
                                 pending_stage.size());
                    for (auto& pr : pending_stage) {
                        TensorF32* t = pr.first;
                        const int64_t n = t->numel();
                        host_stage_.emplace_back(t, t->data());
                        std::vector<float>& h2 = host_scratch_.emplace_back(static_cast<size_t>(n));
                        t->buffer_->get_tensor(t, h2.data(), t->buffer_offs_,
                                               static_cast<size_t>(n) * sizeof(float));
                        t->bind_data(h2.data());
                    }
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
            // 【2026-09-19】加打印本 split 的 **op 序列**（最多 12 个，超出显示 …+N）
            //   用途：定位"尾部那些**单独的 CUDA op**"到底是哪些 op（= 减少 CPU/GPU 交替的抓手 ✓）
            char ops_buf[160];
            int  ops_off = 0;
            const int n_nodes_here = sp.i_end - sp.i_start;
            const int n_show = n_nodes_here > 12 ? 12 : n_nodes_here;
            ops_buf[0] = '\0';
            for (int k = 0; k < n_show; ++k) {
                TensorF32* nd = current_graph_->graph_node(sp.i_start + k);
                ops_off += snprintf(ops_buf + ops_off, sizeof(ops_buf) - (size_t)ops_off,
                                    "%s%d", k ? "," : "", nd ? (int)nd->op : -1);
                if (ops_off >= (int)sizeof(ops_buf) - 8) break;
            }
            if (n_nodes_here > n_show) {
                snprintf(ops_buf + ops_off, sizeof(ops_buf) - (size_t)ops_off,
                         ",…+%d", n_nodes_here - n_show);
            }
            // 【2026-09-19】backend 后面直接跟名字（下标 0 = 最高优先级 = GPU/CUDA ✓，最后 = CPU ✓）
            const char* be_name = (sp.backend_id >= 0 && sp.backend_id < n_backends_ && backends_[sp.backend_id])
                                      ? backends_[sp.backend_id]->get_name() : "?";
            fprintf(stderr,
                    "[sched] COMPUTE split=%d backend=%d(%s) i=[%d,%d) nodes=%d scalar(numel=1)=%d ops=[%s]\n",
                    si, sp.backend_id, be_name, sp.i_start, sp.i_end, n_nodes_here, n_scalar, ops_buf);
        }
        int sub_n_nodes = sp.i_end - sp.i_start;
        TensorF32** saved_nodes = current_graph_->nodes;
        int saved_n_nodes = current_graph_->n_nodes_;

        // 临时替换为子图范围
        current_graph_->nodes = saved_nodes + sp.i_start;
        current_graph_->n_nodes_ = sub_n_nodes;

        // 混训：scheduler 已通过 reserve_graph_memory 预分配全部张量，
        // 让后端跳过自身 gallocr（避免两套 gallocr re-bind 冲突）。
        // 【2026-09-22】GPU split 前置守卫（运行期核验，专治"构建后状态漂移"✗）★
        //   构建期已按 sched_data_is_host 冻结判定并建 H2D cpy ✓，但极端情况下（staging rebind /
        //   参数被重新绑回 host）仍可能出现"输入此刻在 host"✗ ⇒ 若直接 launch，CUDA 内核会裸读
        //   host 指针 ⇒ `illegal memory access`（040350 的 exit 134 根因 ✗，且会污染 CUDA 上下文 ✗）。
        //   本守卫在 **launch 之前**用 is_device_pointer 核验：发现 host 输入 ⇒ 放弃本 split 并走
        //   既有 CPU 回退（返回 ABORTED ✓），而不是让内核崩 ✗。
        //   开关：PPML_SCHED_HOST_GUARD=0 关闭（A/B 用 ✓，默认开）。
        if (!backend->buffer_type()->is_host()) {
            static const bool guard_on = []() {
                const char* e = std::getenv("PPML_SCHED_HOST_GUARD");
                return !(e && *e && std::atoi(e) == 0);
            }();
            if (guard_on) {
                static long guard_hits = 0;
                TensorF32* bad_t = nullptr;
                int bad_k = -1, bad_j = -1;
                // 【2026-10-20 修复】此守卫运行时 current_graph_->nodes 已被改成
                //   `saved_nodes + sp.i_start`（见上面"临时替换为子图范围"），
                //   n_nodes_ = 子图长度 ⇒ 这里必须用**相对下标** [0, n_nodes_)。
                //   原实现用绝对 [i_start, i_end) 会二次偏移，实际扫的是
                //   saved_nodes[2*i_start .. 2*i_start+len)（别的 split，常是 CPU 段），
                //   一旦扫到 CPU 节点的 host 输入就误判"本 GPU split 有 host 输入"并
                //   ABORTED 整图 ⇒ 全图回退 CPU（长样本 "CUDA" 跑的其实是 CPU 的根因）。
                for (int k = 0; k < current_graph_->n_nodes_ && !bad_t; ++k) {
                    TensorF32* nd = current_graph_->graph_node(k);
                    if (!nd || is_view_op(nd->op) || nd->op == OP_NONE) continue;   // 无 kernel ✓
                    for (int j = 0; j < GGML_MAX_SRC; ++j) {
                        TensorF32* s = nd->src[j];
                        if (!s || !s->data()) continue;        // 未计算 ✓
                        if (is_device_pointer(s, s->data())) continue;
                        bad_t = s;
                        bad_k = sp.i_start + k;   // 相对 → 绝对，便于对回整图
                        bad_j = j;
                        break;
                    }
                }
                if (bad_t) {
                    if (guard_hits++ < 20) {
                        // 【2026-10-20】归因信息：bad_t 是"谁"、来自哪条 cpy、为什么没被 Step 1 拷到 device。
                        //   Step 1 的跳过条件是 src_bid==backend 或 tensor_buffer_compatible(src,backend)；
                        //   这两条都以 backend_map_（构建期）为准，运行期数据在 host 时就会漏拷。
                        const TensorF32* bs = (bad_t->op == OP_DUP) ? bad_t->src[0] : nullptr;
                        bool in_copy_map = false;
                        int  map_target  = -1;
                        const TensorF32* map_src = nullptr;
                        for (auto& kv : copy_tensor_map_) {
                            if (kv.second == bad_t) {
                                in_copy_map = true;
                                map_target  = kv.first.second;
                                map_src     = kv.first.first;
                                break;
                            }
                        }
                        TensorF32* consumer = current_graph_->graph_node(bad_k);
                        fprintf(stderr,
                                "[CUDA-GUARD] split=%d backend=%d i=[%d,%d) 输入 op=%d numel=%lld data=%p 在 host ⇒ "
                                "放弃本 split（避免裸读 host ✗，转 CPU 回退 ✓）\n",
                                si, sp.backend_id, sp.i_start, sp.i_end, (int)bad_t->op,
                                (long long)bad_t->numel(), (const void*)bad_t->data());
                        fprintf(stderr,
                                "[CUDA-GUARD]   消费节点 k=%d op=%d src_idx=%d | bad: buf=%p host=%d view_src=%p\n",
                                bad_k, consumer ? (int)consumer->op : -1, bad_j,
                                (void*)bad_t->buffer_,
                                (bad_t->buffer_ ? (int)bad_t->buffer_->is_host() : -1),
                                (void*)bad_t->view_src);
                        if (bs) {
                            fprintf(stderr,
                                    "[CUDA-GUARD]   bad 是 cpy(OP_DUP): src0 op=%d numel=%lld data=%p buf=%p host=%d "
                                    "device_ptr=%d backend_id=%d\n",
                                    (int)bs->op, (long long)bs->numel(), (const void*)bs->data(),
                                    (void*)bs->buffer_,
                                    (bs->buffer_ ? (int)bs->buffer_->is_host() : -1),
                                    (bs->data() ? (int)is_device_pointer(bs, bs->data()) : -1),
                                    tensor_backend_id(const_cast<TensorF32*>(bs), -1));
                        }
                        fprintf(stderr,
                                "[CUDA-GUARD]   copy_map: entry=%d target=%d src=%p(%s) | compatible(bad,backend)=%d "
                                "backend_map(bad)=%d sched_host(bad)=%d\n",
                                (int)in_copy_map, map_target, (const void*)map_src,
                                (map_src == bs ? "same_as_src0" : "DIFF"),
                                (int)tensor_buffer_compatible(bad_t, sp.backend_id),
                                tensor_backend_id(bad_t, -1),
                                (int)sched_data_is_host(bad_t));
                    }
                    if (saved_nodes) current_graph_->nodes = saved_nodes;
                    current_graph_->n_nodes_ = saved_n_nodes;
                    restore_graph_nodes();
                    return Status::ABORTED;
                }
            }
        }

        backend->set_skip_alloc(true);
        Status st = backend->graph_compute(current_graph_);
        backend->set_skip_alloc(false);

        // 指纹对拍（PPML_HASH_SOFTMAX=1，诊断用：远端执行 vs 本地执行的数值透明度）
        //   位置选在"每个 split 刚算完"——此时前向激活尚未被 backward 覆盖。
        //   读值走 buffer_->get_tensor ⇒ 远端 split 会自动 fetch，因此打印的是"客户端真正拿到的值"。
        if (getenv("PPML_HASH_SOFTMAX")) {
            // 统一的"读张量字节 + FNV-1a"（view 沿 view_src 回溯；有 buffer 走 get_tensor——
            //   远端 split 会自动 fetch，因此打印的就是客户端真正拿到/送出的字节）
            auto hash_tensor = [](TensorF32* t, uint64_t* out_h, size_t* out_nb) -> bool {
                TensorF32* real = t;
                int guard = 0;
                while (real && !real->buffer_ && !real->data() && real->view_src && guard++ < 64) {
                    real = real->view_src;
                }
                if (!real || real->nbytes() <= 0) return false;
                std::vector<uint8_t> b((size_t)real->nbytes());
                if (real->buffer_) {
                    real->buffer_->get_tensor(real, b.data(), real->buffer_offs_, b.size());
                } else if (real->data()) {
                    std::memcpy(b.data(), real->data(), b.size());
                } else {
                    return false;
                }
                uint64_t h = 1469598103934665603ull;
                for (uint8_t x : b) { h ^= x; h *= 1099511628211ull; }
                *out_h = h; *out_nb = b.size();
                return true;
            };
            for (int i = sp.i_start; i < sp.i_end; i++) {
                TensorF32* nd = current_graph_->graph_node(i);
                if (!nd || nd->op != OP_SOFT_MAX) continue;
                uint64_t h = 0, hi = 0; size_t nb = 0, nbi = 0;
                if (nd->src[0] && hash_tensor(nd->src[0], &hi, &nbi)) {
                    fprintf(stderr, "[HASH-IN] nid=%d backend=%d nb=%zu fnv=%016llx\n",
                            i, sp.backend_id, nbi, (unsigned long long)hi);
                }
                if (hash_tensor(nd, &h, &nb)) {
                    fprintf(stderr, "[HASH] split=%d backend=%d nid=%d nb=%zu fnv=%016llx\n",
                            si, sp.backend_id, i, nb, (unsigned long long)h);
                }
            }
        }

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
        // 【2026-09-19 P1】本 split 结束（pinned 指针已全部恢复成 device 指针 ✓）⇒ 复位 arena 供下次复用 ✓
        staging_split_end();

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
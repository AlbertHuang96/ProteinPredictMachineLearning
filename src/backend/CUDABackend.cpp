#include "ppml/Backend.h"
#include "ppml/ComputeGraph.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>

namespace ppml {

// ===== CUDABufferType 实现 =====
CUDABufferType* CUDABufferType::instance(int device_id) {
    // 每 device 一个单例
    static CUDABufferType s_instances[8] = {
        CUDABufferType(0), CUDABufferType(1), CUDABufferType(2), CUDABufferType(3),
        CUDABufferType(4), CUDABufferType(5), CUDABufferType(6), CUDABufferType(7)
    };
    if (device_id < 0 || device_id >= 8) device_id = 0;
    return &s_instances[device_id];
}

void* CUDABufferType::alloc(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

void CUDABufferType::free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

// ===== CUDABackend 构造/析构 =====
CUDABackend::CUDABackend(int device_id) : device_id_(device_id) {
    cudaSetDevice(device_id_);
}

CUDABackend::~CUDABackend() {
    // 不拥有任何资源，buffer 由 DefaultBuffer 管理
}

// ===== buffer_type / supports_buffer_type =====
const BufferType* CUDABackend::buffer_type() const {
    return CUDABufferType::instance(device_id_);
}

bool CUDABackend::supports_buffer_type(const BufferType* buft) const {
    if (!buft) return false;

    // CUDA backend 支持：
    // 1. 同 device 的 GPU buffer
    // 2. host buffer（用于跨后端拷贝）
    if (buft->is_host()) return true;

    const CUDABufferType* cuda_buft = dynamic_cast<const CUDABufferType*>(buft);
    if (cuda_buft && cuda_buft->device_id() == device_id_) return true;

    return false;
}

// ===== supports_op — 仅声明有 .cu kernel 实际实现的 op =====
bool CUDABackend::supports_op(TensorF32* node) const {
    const TensorF32* src0 = node->src[0];
    const TensorF32* src1 = node->src[1];

    // 诊断二分：PPML_CUDA_DISABLE_OPS 指定要强制回落 CPU 的 op（数字，逗号/空格分隔）。
    // 用于定位哪个 CUDA kernel 有越界/异步崩溃：逐个把可疑 op 禁掉，看崩溃是否消失。
    if (const char* dis = getenv("PPML_CUDA_DISABLE_OPS")) {
        char buf[128];
        int n = (int)strlen(dis);
        if (n > (int)sizeof(buf) - 1) n = (int)sizeof(buf) - 1;
        memcpy(buf, dis, (size_t)n); buf[n] = '\0';
        char* tok = strtok(buf, ", \t");
        while (tok) {
            int v = atoi(tok);
            if (v == (int)node->op) return false;
            tok = strtok(nullptr, ", \t");
        }
    }

    // no-op / view ops 始终支持（不需要 kernel）
    if (node->op == OP_NONE    || node->op == OP_RESHAPE ||
        node->op == OP_VIEW    || node->op == OP_PERMUTE ||
        node->op == OP_TRANSPOSE) {
        return true;
    }

    switch (node->op) {
        // ===== 有完整 .cu kernel 实现的 op =====
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
            return true;

        case OP_MUL_MAT:
            if (!src1) return true;
            return src1->type == TENSOR_TYPE_F32;

        case OP_OUT_PROD:
            if (!src0 || !src1) return false;
            return src0->type == TENSOR_TYPE_F32 &&
                   src1->type == TENSOR_TYPE_F32 &&
                   node->type == TENSOR_TYPE_F32;

        case OP_CONCAT: {
            // N-ary concat：遍历所有 src。
            // 注意：concat 不支持广播语义，非拼接维必须与 dst 一致，
            // 该约束在 kernel_concat_cuda 内强制校验。
            int n_src = 0;
            for (const auto* s : node->src) if (s) n_src++;
            if (n_src < 2) return false;
            for (const auto* s : node->src) {
                if (!s || s->type != TENSOR_TYPE_F32) return false;
            }
            return node->type == TENSOR_TYPE_F32;
        }

        case OP_SOFT_MAX:
            return true;

        case OP_EDGE_GATHER_ROWS:
        case OP_PER_EDGE_MATMUL:
        case OP_SCATTER_ADD:
        case OP_PER_EDGE_MATMUL_BACK_KERNEL:
        case OP_PER_EDGE_MATMUL_BACK_GATHERED:
            return true;

        case OP_SOFT_MAX_BACK:
            if (!src0 || !src1) return false;
            return src0->type == TENSOR_TYPE_F32 &&
                   src1->type == TENSOR_TYPE_F32;

        case OP_NORM:
            return true;

        case OP_NORM_BACK:
            return true;

        // ===== kernel 为空函数体或 NOT_SUPPORTED，暂不支持 =====
        // OP_DUP      → kernel_dup_cuda 空函数体，无实现
        // OP_ADD1     → kernel_add1_cuda 返回 NOT_SUPPORTED
        // OP_SCALE    → kernel_scale_cuda 返回 NOT_SUPPORTED
        // OP_SUM      → kernel_sum_cuda 返回 NOT_SUPPORTED
        // OP_MEAN     → kernel_mean_cuda 返回 NOT_SUPPORTED
        // OP_CPY      → dispatch_node 中无 case
        // OP_SET_ROWS → dispatch_node 中无 case
        // UNARY_OP_*  → kernel_relu/gelu/sigmoid/silu/tanh/exp_cuda 均为空函数体

        // ===== 未实现的 op =====
        case OP_DUP:
        case OP_ADD1:
        case OP_SCALE:
        case OP_SUM:
        case OP_MEAN:
        case OP_CPY:
        case OP_SET_ROWS:
        case OP_RMS_NORM:
        case OP_GET_ROWS_BACK:
        case OP_FLASH_ATTN_EXT:
        case OP_FLASH_ATTN_BACK:
        case OP_CROSS_ENTROPY_LOSS:
        case OP_UNARY:  // 所有 unary op kernel 均为空函数体
        default:
            return false;
    }
}

// ===== synchronize =====
void CUDABackend::synchronize() {
    cudaSetDevice(device_id_);
    cudaDeviceSynchronize();
}

// ===== graph_compute =====
Status CUDABackend::graph_compute(ComputeGraph* cgraph) {
    cudaSetDevice(device_id_);

    // ---- no_alloc 延迟分配：对 data_==nullptr 的中间节点分配 GPU buffer ----
    // 混训时 scheduler 已预分配，置 skip_alloc_=true 跳过，避免两套 gallocr 冲突。
    if (!skip_alloc_) {
        gallocr_.release();
        bool need_alloc = false;
        for (int i = 0; i < cgraph->n_nodes(); ++i) {
            if (cgraph->graph_node(i)->data() == nullptr) { need_alloc = true; break; }
        }
        if (!need_alloc) {
            for (int i = 0; i < cgraph->n_leafs(); ++i) {
                if (cgraph->graph_leaf(i)->data() == nullptr) { need_alloc = true; break; }
            }
        }
        if (need_alloc) {
            gallocr_.set_n_backends(1);
            gallocr_.backends()[0].buft = const_cast<BufferType*>(buffer_type());
            auto backend_id_of = [](TensorF32*) -> int { return 0; };
            if (!gallocr_.reserve(cgraph, backend_id_of, 1) ||
                !gallocr_.alloc(cgraph, backend_id_of, 1)) {
                return Status::ALLOC_FAILED;
            }
        }
    }

    int cuda_dispatch_cnt = 0;
    int cuda_dispatch_op = -1;
    for (int node_n = 0; node_n < cgraph->n_nodes(); node_n++) {
        TensorF32* node = cgraph->graph_node(node_n);

        // 跳过 no-op
        if (node->op == OP_NONE || node->op == OP_VIEW ||
            node->op == OP_RESHAPE || node->op == OP_PERMUTE ||
            node->op == OP_TRANSPOSE) {
            continue;
        }

        // 解析 view src：view（view_src 共享源数据）被跳过 dispatch 后 data() 为 null，
        // 但其作为本 op 的 src 时 kernel 要读数据。沿 view_src 链解析到底层数据载体，
        // 让本 op 的 kernel 拿到真实 device 指针（否则读 null → [CUDA-ERR]/illegal）。
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            TensorF32* src = node->src[s];
            if (!src || !src->view_src) continue;
            TensorF32* base = src;
            int guard = 0;
            while (base->view_src && guard++ < 64) base = base->view_src;
            if (base->data()) {
                src->bind_data(base->data());
                src->buffer_      = base->buffer_;
                src->buffer_offs_ = base->buffer_offs_;
            }
        }

        // 所有数据应在 GPU 上（由 reserve_graph_memory 保证）。
        // 防御：用 cudaPointerGetAttributes 权威判定指针是否属于 device 内存；
        // 若是 host 指针（跨后端拷贝/视图链遗漏），kernel 会把 host 地址当 device 读 -> illegal address。
        auto is_cuda_dev = [](const void* p) -> bool {
            if (!p) return false;
            cudaPointerAttributes attr{};
            cudaError_t e = cudaPointerGetAttributes(&attr, p);
            if (e != cudaSuccess) return false;          // 非 device 指针（host/unified 异常）
            return (attr.type == cudaMemoryTypeDevice ||
                    attr.type == cudaMemoryTypeManaged);
        };
        if (getenv("GRAPH_DEBUG_CUDA_OP")) {
            // 辅助：打印某个指针的 cudaPointerGetAttributes 详情，区分
            // "真正的 device 分配" 与 "悬垂/错位 device 地址"。
            auto dbg_ptr = [](const char* tag, const void* p, const TensorF32* t) {
                if (!p) { fprintf(stderr, "    %s=NULL\n", tag); return; }
                cudaPointerAttributes attr{};
                cudaError_t e = cudaPointerGetAttributes(&attr, p);
                const char* dev_kind = "unknown";
                if (e == cudaSuccess) {
                    if (attr.type == cudaMemoryTypeDevice) dev_kind = "device";
                    else if (attr.type == cudaMemoryTypeManaged) dev_kind = "managed";
                    else if (attr.type == cudaMemoryTypeHost) dev_kind = "host";
                    else dev_kind = "unregistered";
                } else {
                    dev_kind = "attr_err";
                }
                // buffer_ 基址（host 端看到的字段）与 data() 是否一致范围
                fprintf(stderr,
                        "    %s=%p type=%s dev=%d devPtr=%p hostPtr=%p | tensor.buffer_=%p "
                        "buffer.is_host=%d buffer_offs=%zu buffer_size=%zu\n",
                        tag, p, dev_kind, (e==cudaSuccess?attr.device:-1),
                        (e==cudaSuccess?(void*)attr.devicePointer:nullptr),
                        (e==cudaSuccess?(void*)attr.hostPointer:nullptr),
                        (void*)(t && t->buffer_ ? t->buffer_ : nullptr),
                        (t && t->buffer_ ? (int)t->buffer_->is_host() : -1),
                        (size_t)(t ? t->buffer_offs_ : 0),
                        (size_t)(t && t->buffer_ ? t->buffer_->size() : 0));
            };
            fprintf(stderr, "[cuda-dbg] op=%d node=%d numel=%lld\n",
                    (int)node->op, node_n, (long long)node->numel());
            dbg_ptr("dst", node->data(), node);
            if (node->src[0]) dbg_ptr("src0", node->src[0]->data(), node->src[0]);
            if (node->src[1]) dbg_ptr("src1", node->src[1]->data(), node->src[1]);
        }
        {
            bool bad = false;
            int bad_src = -1;
            // data() 是 device 指针但 buffer_ 标记为 host（或 buffer_ 的 data 基址与 data() 不在
            // 同一 device 分配）=> 不一致，通常是 view/buffer 复用错位导致 data() 指向悬垂 device 段。
            auto inconsistent = [is_cuda_dev](const TensorF32* t) -> bool {
                if (!t || !t->data() || !is_cuda_dev(t->data())) return false;
                if (t->buffer_ && t->buffer_->is_host()) return true;  // device data 却挂在 host buffer
                return false;
            };
            if (inconsistent(node)) {
                fprintf(stderr,
                        "[CUDA-ERR] GPU op=%d (node %d) INCONSISTENT: dst->data=%p is device "
                        "but buffer_->is_host=true (view/buffer reuse mismatch). buffer_=%p\n",
                        (int)node->op, node_n, (void*)node->data(),
                        (void*)(node->buffer_ ? node->buffer_ : nullptr));
                return Status::NOT_SUPPORTED;
            }
            if (!node->data() || !is_cuda_dev(node->data())) { bad = true; bad_src = -1; }
            if (!bad) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    TensorF32* s = node->src[j];
                    if (!s) continue;
                    if (inconsistent(s)) {
                        fprintf(stderr,
                                "[CUDA-ERR] GPU op=%d (node %d) INCONSISTENT: src%d->data=%p is device "
                                "but buffer_->is_host=true (view/buffer reuse mismatch). buffer_=%p\n",
                                (int)node->op, node_n, j, (void*)s->data(),
                                (void*)(s->buffer_ ? s->buffer_ : nullptr));
                        return Status::NOT_SUPPORTED;
                    }
                    if (!s->data() || !is_cuda_dev(s->data())) {
                        bad = true; bad_src = j; break;
                    }
                }
            }
            if (bad) {
                fprintf(stderr,
                        "[CUDA-ERR] GPU op=%d (node %d) has HOST/INVALID pointer input "
                        "(src=%d dst=%s) — cross-backend copy/view rewiring missed. "
                        "node->data=%p\n",
                        (int)node->op, node_n, bad_src,
                        node->data() ? "ok" : "null", (void*)node->data());
                // 不再 dispatch（避免段错误），返回错误让上层回退/报错
                return Status::NOT_SUPPORTED;
            }
            // 缓冲区边界检查：dst/src 所需字节必须落在各自 buffer 范围内，
            // 否则 kernel 越界写 -> 段错误。这能区分"gallocr 分配过小"与"shape 不匹配"。
            auto fits = [](const TensorF32* t, int* bad_dim) -> bool {
                if (!t || !t->buffer_) return true;  // 无 buffer（叶子外部内存）跳过
                size_t need = (size_t)t->numel() * sizeof(float);
                size_t avail = t->buffer_->size() - t->buffer_offs_;
                if (need > avail) {
                    if (bad_dim) *bad_dim = (int)t->numel();
                    return false;
                }
                return true;
            };
            int bad_num = 0;
            bool buf_bad = false;
            if (!fits(node, &bad_num)) { buf_bad = true; }
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                if (node->src[j] && !fits(node->src[j], &bad_num)) { buf_bad = true; break; }
            }
            if (buf_bad) {
                fprintf(stderr,
                        "[CUDA-ERR] GPU op=%d (node %d) buffer OVERFLOW: tensor needs "
                        "%d floats but buffer has less (gallocr under-allocated or view offs wrong). "
                        "dst_buf=%zu offs=%zu src0_buf=%zu offs=%zu\n",
                        (int)node->op, node_n, bad_num,
                        node->buffer_ ? node->buffer_->size() : 0, node->buffer_offs_,
                        node->src[0] && node->src[0]->buffer_ ? node->src[0]->buffer_->size() : 0,
                        node->src[0] ? node->src[0]->buffer_offs_ : 0);
                return Status::NOT_SUPPORTED;
            }
        }

        // 直接 dispatch
        ComputeParams params{};
        if (getenv("GRAPH_DEBUG_CUDA_OP")) {
            // 打印每次 dispatch 的 op 与各输入/输出 numel，定位 shape 不匹配导致的越界。
            long long dn = node->numel();
            long long s0 = node->src[0] ? node->src[0]->numel() : -1;
            long long s1 = node->src[1] ? node->src[1]->numel() : -1;
            long long s2 = node->src[2] ? node->src[2]->numel() : -1;
            fprintf(stderr,
                    "[cuda-op] dispatch op=%d node=%d dst_numel=%lld src0=%lld src1=%lld src2=%lld\n",
                    (int)node->op, node_n, dn, s0, s1, s2);
        }
        Status st = dispatch_node(node, &params);
        if (st != Status::SUCCESS) {
            return st;
        }
        // 捕获异步 CUDA 错误（如 kernel 拿到 host 指针当 device 读 -> illegal address）。
        // 先查 launch 错误（廉价的 cudaGetLastError 始终做）；执行错误需同步才能捕获，
        // 用 GRAPH_DEBUG_CUDA 控制（每 kernel 同步很慢，仅定位段错误时开）。
        {
            cudaError_t kerr = cudaGetLastError();
            if (kerr != cudaSuccess) {
                fprintf(stderr,
                        "[CUDA-ERR] kernel launch failed at op=%d (node %d): %s\n",
                        (int)node->op, node_n, cudaGetErrorString(kerr));
                return Status::ABORTED;
            }
            if (getenv("GRAPH_DEBUG_CUDA")) {
                cudaError_t serr = cudaDeviceSynchronize();
                if (serr != cudaSuccess) {
                    fprintf(stderr,
                            "[CUDA-ERR] kernel execution failed at op=%d (node %d): %s\n",
                            (int)node->op, node_n, cudaGetErrorString(serr));
                return Status::ABORTED;
                }
            }
        }
        // 诊断（GRAPH_DEBUG_CUDA=1）：统计实际 dispatch 的 CUDA kernel 次数
        if (getenv("GRAPH_DEBUG_CUDA")) {
            cuda_dispatch_cnt++;
            cuda_dispatch_op = (int)node->op;
        }
    }

    // 诊断（GRAPH_DEBUG_CUDA=1）：确认 CUDA kernel 是否真的被调用（非 0 即 GPU 在执行计算）
    if (getenv("GRAPH_DEBUG_CUDA")) {
        fprintf(stderr, "[cuda] graph_compute: dispatched %d CUDA kernels (last op=%d)\n",
                cuda_dispatch_cnt, cuda_dispatch_op);
    }

    return Status::SUCCESS;
}

} // namespace ppml

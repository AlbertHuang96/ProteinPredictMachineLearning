#include "ppml/Backend.h"
#include "ppml/ComputeGraph.h"
#include <cuda_runtime.h>

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

    for (int node_n = 0; node_n < cgraph->n_nodes(); node_n++) {
        TensorF32* node = cgraph->graph_node(node_n);

        // 跳过 no-op
        if (node->op == OP_NONE || node->op == OP_VIEW ||
            node->op == OP_RESHAPE || node->op == OP_PERMUTE ||
            node->op == OP_TRANSPOSE) {
            continue;
        }

        // 所有数据已在 GPU 上（由 reserve_graph_memory 保证），直接 dispatch
        ComputeParams params{};
        Status st = dispatch_node(node, &params);
        if (st != Status::SUCCESS) {
            return st;
        }
    }

    return Status::SUCCESS;
}

} // namespace ppml


#include "ppml/Backend.h"
#include "ppml/ComputeGraph.h"
#include "ppml/FAPE.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>

//#include <omp.h>

// ===== 开发用 GPU 加速矩阵乘（方案 B，临时）=====
// 说明：仅在"矩阵乘"节点上临时调用 CUDA 内核以加速训练（本机 GPU=RTX2050 4GB）。
//   开关：环境变量 PPML_MUL_MAT_GPU=1 时启用；默认关闭（走 CPU 朴素矩阵乘）。
//   做法：每次 MUL_MAT 把 A/B 拷到 GPU → mul_mat_cuda 计算 → 结果拷回 host。
//   注意：此路径有 host↔device 拷贝开销，且每节点独立分配 GPU buffer，仅适合开发期
//         验证 GPU 加速效果。**正式方案是 BackendScheduler 混合调度**（GPU buffer 复用、
//         跨后端自动拷贝），本路径应在其就绪后移除。
// 项目要求 CUDA（CMake find_package(CUDAToolkit REQUIRED)），ppml_core 链接 cudart。
// 开发用 GPU 矩阵乘开关见上方注释；无 CUDA 环境由运行时 cudaGetDeviceCount 兜底回退 CPU。
#include <cuda_runtime.h>
namespace ppml {
// mul_mat_cuda 实现于 src/cuda/CUDAKernels.cu（host 函数，launch blockTileGEMM kernel）
extern void mul_mat_cuda(float* A, float* B, float* C, int M, int K, int N);
// ===== 禁用内部 GPU 自跑路径 =====
// mul_mat 应完全由 BackendScheduler 分配：要 GPU 时 scheduler 会 dispatch 到 CUDABackend
// (kernel_mul_mat_cuda)，无需 CPU kernel 内部再 cudaMalloc 自跑一次。内部旁路(PPML_MUL_MAT_GPU)
// 假设 a/b 是 host 指针，混训时可能读到 device 指针 → 段错误，且 barrier 与 scheduler 冲突。
// 故强制关闭（保留代码作参考）。
static const bool g_mul_mat_gpu_enabled = false;

// ===== unary op 计算函数前向声明 =====
static void compute_forward_abs(ComputeParams* p, TensorF32* dst);
static void compute_forward_sgn(ComputeParams* p, TensorF32* dst);
static void compute_forward_neg(ComputeParams* p, TensorF32* dst);
static void compute_forward_step(ComputeParams* p, TensorF32* dst);
static void compute_forward_relu(ComputeParams* p, TensorF32* dst);
static void compute_forward_gelu(ComputeParams* p, TensorF32* dst);
static void compute_forward_gelu_quick(ComputeParams* p, TensorF32* dst);
static void compute_forward_silu(ComputeParams* p, TensorF32* dst);
static void compute_forward_tanh(ComputeParams* p, TensorF32* dst);
static void compute_forward_elu(ComputeParams* p, TensorF32* dst);
static void compute_forward_sigmoid(ComputeParams* p, TensorF32* dst);
static void compute_forward_hardsigmoid(ComputeParams* p, TensorF32* dst);
static void compute_forward_hardswish(ComputeParams* p, TensorF32* dst);
static void compute_forward_exp(ComputeParams* p, TensorF32* dst);
static void compute_forward_log(ComputeParams* p, TensorF32* dst);
static void compute_forward_sqrt(ComputeParams* p, TensorF32* dst);
static void compute_forward_sin(ComputeParams* p, TensorF32* dst);
static void compute_forward_cos(ComputeParams* p, TensorF32* dst);
static void compute_forward_fape(ComputeParams* p, TensorF32* dst);
static void compute_forward_fape_back(ComputeParams* p, TensorF32* dst);

// ===== dispatch =====

// 运行时开关：PPML_CPU_BUFFER_AWARE=1 时，CPU kernel 容忍 device(即 GPU)输入。
// 大型机（显存充足、混训 CPU+GPU）开启；本机默认关 → 纯 CPU 行为完全不变。
// 实现：在 dispatch 层把 device buffer 的 src 经 get_tensor(D2H) 暂存到 host，
//       临时 bind_data 到 host scratch，跑完 kernel 后恢复原 device 指针。
//       不依赖 scheduler 的跨后端拷贝（避免拷贝缺口），直接让 CPU kernel 读 host。
static bool g_cpu_buffer_aware_init = false;
static bool g_cpu_buffer_aware = false;
static bool cpu_buffer_aware_enabled() {
    if (!g_cpu_buffer_aware_init) {
        g_cpu_buffer_aware_init = true;
        const char* e = getenv("PPML_CPU_BUFFER_AWARE");
        g_cpu_buffer_aware = (e != nullptr) && (atoi(e) != 0);
    }
    return g_cpu_buffer_aware;
}

// 判断 data 指针是否指向 device 内存。
// 复用 Backend.cpp 中的共享实现（ppml::is_device_pointer）。
// 有 buffer_ 时看 is_host()；无 buffer_（如裸 device 指针）时用 4 字节 D2H 探测兜底。

// 暂存一个 device 内存 src 到 host scratch，返回需恢复的原 device 指针（nullptr 表示无需恢复）
// 仅在 buffer-aware 模式调用。src->data() 是 device 指针（无论有无 buffer_）都暂存。
static float* stage_device_src(TensorF32* src, std::vector<float>& scratch) {
    float* data = src->data();
    if (!is_device_pointer(src, data)) return nullptr;   // host，无需暂存
    const int64_t n = src->numel();
    scratch.resize(static_cast<size_t>(n));
    if (src->buffer_) {
        src->buffer_->get_tensor(src, scratch.data(), src->buffer_offs_,
                                 static_cast<size_t>(n) * sizeof(float));  // D2H
    } else {
        cudaMemcpy(scratch.data(), data, static_cast<size_t>(n) * sizeof(float),
                   cudaMemcpyDeviceToHost);   // 无 buffer 裸 device 指针 D2H
    }
    src->bind_data(scratch.data());   // 临时让 kernel 读 host
    return data;
}

// 实际 dispatch 主体：执行节点 kernel（不感知 buffer 来源）。
Status CPUBackend::dispatch_body(TensorF32* node, ComputeParams* p) {
    switch (node->op) {
        case OP_NONE:   break;
        case OP_DUP:    kernel_dup(node);            break;
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:    kernel_elemwise(node, p);    break;
        case OP_SQR:    kernel_sqr(node, p);         break;
        case OP_SQRT:   compute_forward_sqrt(p, node); break;  // 独立 sqrt 逐元素(op=10)
        case OP_LOG:    compute_forward_log(p, node);  break;  // 独立 log 逐元素(op=11)
        case OP_ADD1:   kernel_add1(node, p);        break;
        case OP_SCALE:  kernel_scale(node, p);       break;
        case OP_MUL_MAT:   kernel_mul_mat(node, p);  break;
        case OP_OUT_PROD:  kernel_out_prod(node, p); break;
        // online softmax
        case OP_SOFT_MAX:  kernel_softmax(node, p);  break;
        case OP_SOFT_MAX_BACK: kernel_softmax_back(node, p); break;
        case OP_RMS_NORM:  kernel_rms_norm(node, p); break;
        case OP_NORM:      kernel_norm(node, p);     break;
        case OP_NORM_BACK: kernel_norm_back(node, p); break;
        case OP_SUM:    kernel_sum(node, p);         break;
        case OP_SUM_ROWS: kernel_sum_rows(node, p);  break;
        case OP_MEAN:   kernel_mean(node, p);        break;
        case OP_MAX_ALL: kernel_max_all(node, p);    break;
        case OP_RELU_BACK: kernel_relu_back(node, p); break;
        case OP_REPEAT:      kernel_repeat(node, p);      break;
        case OP_REPEAT_BACK: kernel_repeat_back(node, p); break;
        case OP_CONCAT:      kernel_concat(node, p);      break;
        case OP_CONCAT_BACK: kernel_concat_back(node, p); break;
        // 方案 B shape op：reshape/view/cont/cpy 整块拷贝；permute/transpose 重排
        case OP_RESHAPE:
        case OP_VIEW:
        case OP_CONT:
        case OP_CPY:     kernel_cpy(node, p);         break;
        case OP_PERMUTE:
        case OP_TRANSPOSE: kernel_permute(node, p);   break;
        case OP_GET_ROWS:      kernel_get_rows(node, p);       break;
        case OP_GET_ROWS_BACK: kernel_get_rows_back(node, p);  break;
        case OP_SET_ROWS:      kernel_set_rows(node, p);       break;
        case OP_EDGE_GATHER_ROWS: kernel_edge_gather_rows(node, p); break;
        case OP_PER_EDGE_MATMUL:  kernel_per_edge_matmul (node, p); break;
        case OP_SCATTER_ADD:      kernel_scatter_add     (node, p); break;
        case OP_PER_EDGE_MATMUL_BACK_KERNEL:   kernel_per_edge_matmul_back_kernel(node, p); break;
        case OP_PER_EDGE_MATMUL_BACK_GATHERED: kernel_per_edge_matmul_back_gathered(node, p); break;
        case OP_FAPE:      compute_forward_fape(p, node);      break;
        case OP_FAPE_BACK: compute_forward_fape_back(p, node); break;
        case OP_TRI_MUL:   kernel_tri_mul(node, p);            break;
        case OP_TRI_MUL_BACK: kernel_tri_mul_back(node, p);    break;
        case OP_OUTER_PROD_MEAN: kernel_outer_prod_mean(node, p); break;
        case OP_OUTER_PROD_MEAN_BACK: kernel_outer_prod_mean_back(node, p); break;
        case OP_OUTER_PROD:      kernel_outer_prod(node, p);      break;
        case OP_OUTER_PROD_BACK: kernel_outer_prod_back(node, p); break;
        case OP_UNARY:  {
            const unary_op uop = get_unary_op(node);
            if (getenv("GRAPH_DEBUG_UNARY")) {
                fprintf(stderr, "[unary] uop=%d numel=%lld\n", (int)uop, (long long)node->numel());
            }
            switch (uop) {
                case UNARY_OP_ABS:
                    compute_forward_abs(p, node);        break;
                case UNARY_OP_SGN:
                    compute_forward_sgn(p, node);        break;
                case UNARY_OP_NEG:
                    compute_forward_neg(p, node);        break;
                case UNARY_OP_STEP:
                    compute_forward_step(p, node);       break;
                case UNARY_OP_RELU:
                    compute_forward_relu(p, node);       break;
                case UNARY_OP_GELU:
                    compute_forward_gelu(p, node);       break;
                case UNARY_OP_GELU_QUICK:
                    compute_forward_gelu_quick(p, node); break;
                case UNARY_OP_SILU:
                    compute_forward_silu(p, node);       break;
                case UNARY_OP_TANH:
                    compute_forward_tanh(p, node);       break;
                case UNARY_OP_ELU:
                    compute_forward_elu(p, node);        break;
                case UNARY_OP_SIGMOID:
                    compute_forward_sigmoid(p, node);    break;
                case UNARY_OP_HARDSIGMOID:
                    compute_forward_hardsigmoid(p, node);break;
                case UNARY_OP_HARDSWISH:
                    compute_forward_hardswish(p, node);  break;
                case UNARY_OP_EXP:
                    compute_forward_exp(p, node);        break;
                case UNARY_OP_LOG:
                    compute_forward_log(p, node);        break;
                case UNARY_OP_SQRT:
                    compute_forward_sqrt(p, node);       break;
                case UNARY_OP_SIN:
                    compute_forward_sin(p, node);        break;
                case UNARY_OP_COS:
                    compute_forward_cos(p, node);        break;
                default:
                    p->threadpool->ec = Status::NOT_SUPPORTED;
                    break;
            }
        } break;
        default:
            p->threadpool->ec = Status::NOT_SUPPORTED;
            break;
    }
    // GRAPH_DEBUG_KERNEL=1：逐个 op 检查输出值域，定位第一个产生 NaN/巨大值(>1e6) 的 kernel
    if (p->ith == 0 && p->threadpool->ec == Status::SUCCESS && node->op != OP_NONE && node->data()) {
        static const char* kdbg = getenv("GRAPH_DEBUG_KERNEL");
        if (kdbg) {
            const int64_t nelt = node->numel();
            if (nelt > 0) {
                const float* dd = node->data();
                float mn = dd[0], mx = dd[0]; bool nan = false; int64_t nnan = 0;
                // 只扫描前若干元素 + 全扫找 nan（避免大张量全扫太慢，但需找 nan 必须全扫）
                int64_t scan = nelt < 4096 ? nelt : 4096;
                for (int64_t i = 0; i < scan; i++) { float v = dd[i]; if (v != v) { nan = true; nnan++; } else { if (v<mn) mn=v; if (v>mx) mx=v; } }
                // 大张量额外全扫 nan
                if (nelt >= 4096) {
                    int64_t nnan2 = 0;
                    for (int64_t i = 0; i < nelt; i++) if (dd[i] != dd[i]) nnan2++;
                    if (nnan2) { nan = true; nnan = nnan2; }
                }
                static int kern_cnt = 0;   // 只打印前 15 条，避免刷屏看不到第一条
                // GRAPH_DEBUG_KERNEL_DETAIL=1 时阈值降到 1e3，暴露 msa_emb_ 等较小但异常的中间量（定位 msa 特征巨大起点）
                const float kern_thresh = getenv("GRAPH_DEBUG_KERNEL_DETAIL") ? 1e3f : 1e6f;
                if (nan || mx > kern_thresh || (mn < -kern_thresh)) {
                    if (kern_cnt >= 15) { /* 超过 15 条不再打印，但记录已发现异常 */ return p->threadpool->ec; }
                    kern_cnt++;
                    fprintf(stderr, "[kern#%d] op=%d numel=%lld min=%.6g max=%.6g nan=%d nnan=%lld src0_op=%d src1_op=%d ndim=%d",
                            kern_cnt, (int)node->op, (long long)nelt, (double)mn, (double)mx,
                            (nan?1:0), (long long)nnan,
                            (node->src[0] ? (int)node->src[0]->op : -1),
                            (node->src[1] ? (int)node->src[1]->op : -1),
                            (int)node->shape().ndim());
                    if (node->shape().ndim() >= 1 && node->shape().ndim() <= 4) {
                        fprintf(stderr, " dims=[");
                        for (int d = 0; d < node->shape().ndim(); ++d)
                            fprintf(stderr, "%s%lld", (d ? "," : ""), (long long)node->shape().dims[d]);
                        fprintf(stderr, "]");
                    }
                    // 打印 src0/src1 值域（若存在且有 data），定位输入是否巨大
                    for (int si = 0; si < 2; si++) {
                        const TensorF32* sp = node->src[si];
                        if (!sp || !sp->data()) continue;
                        const int64_t sn = sp->numel();
                        if (sn <= 0) continue;
                        float smn = sp->data()[0], smx = sp->data()[0]; bool snan = false;
                        int64_t sscan = sn < 4096 ? sn : 4096;
                        for (int64_t q = 0; q < sscan; q++) { float v = sp->data()[q]; if (v != v) { snan = true; break; } if (v < smn) smn = v; if (v > smx) smx = v; }
                        fprintf(stderr, " src%d[op=%d", si, (int)sp->op);
                        if (sp->shape().ndim() >= 1 && sp->shape().ndim() <= 4) {
                            fprintf(stderr, " d=[");
                            for (int dd = 0; dd < sp->shape().ndim(); ++dd) fprintf(stderr, "%s%lld", (dd?",":""), (long long)sp->shape().dims[dd]);
                            fprintf(stderr, "]");
                        }
                        // 定位：打印 src 是否为参数(PARAM)、是否被 gallocr 绑定(buffer_)、及数据指针，
                        // 用于判断 kernel 读的 src 是不是真实参数（若被 buffer 绑定/覆盖则读到污染值）。
                        fprintf(stderr, " flag=0x%x buf=%d ptr=%p", (unsigned)sp->flag,
                                (sp->buffer_ ? 1 : 0), (const void*)sp->data());
                        fprintf(stderr, " min=%.6g max=%.6g nan=%d]", (double)smn, (double)smx, snan ? 1 : 0);
                    }
                    // 追溯 src0 的来源（src0->src[0]->op），定位巨大输入来自哪个 op
                    if (node->src[0] && node->src[0]->src[0]) {
                        const TensorF32* g0 = node->src[0]->src[0];
                        fprintf(stderr, " src0_src0_op=%d", (int)g0->op);
                        if (g0->src[0]) fprintf(stderr, "(src=%d)", (int)g0->src[0]->op);
                    }
                    fprintf(stderr, "\n");
                }
            }
        }
    }
    return p->threadpool->ec;
}

// ===== 统一入口：CPU 节点遇到 device(即 GPU) 输入时自动暂存为 host =====
// 不再依赖 PPML_CPU_BUFFER_AWARE 开关：只要检测到本 node 有 device 输入就走
// buffer-aware 暂存路径；否则走原纯 CPU 路径（行为完全不变，无额外开销）。
// 这是混合训练(CPU+GPU)的兜底——scheduler 跨后端拷贝缺口导致 CPU kernel 直接
// 读到 GPU device 指针 → SIGSEGV，此处保证任何 device 输入都先 D2H 暂存。
static bool node_has_device_input(TensorF32* node) {
    for (int s = 0; s < GGML_MAX_SRC; s++) {
        TensorF32* src = node->src[s];
        if (!src) continue;
        // 确定性判定：只看 buffer_（device tensor 必有 buffer_ 且非 host）。避免用 is_device_pointer
        // 的 cudaMemcpy 探测——多线程并发探测结果可能不一致，导致同一 node 部分线程走 buffer-aware、
        // 部分走纯 CPU → dispatch 路径/barrier 计数不匹配 → 崩溃（PPML_N_THREADS=1 不崩、多线程崩）。
        if (src->buffer_ && !src->buffer_->is_host()) return true;
    }
    return false;
}

Status CPUBackend::dispatch_node(TensorF32 * node, ComputeParams * p) {
    // 崩溃定位（GRAPH_DEBUG_DISPATCH=1，线程 0）：打印每个待调度节点的序号/op/src 指针，
    // 在进入任何 kernel 前。崩溃前最后一行 = 即将执行的故障节点（无论哪种 kernel）。
    // 线程 0 与 worker 带 barrier 逐节点同步，故线程 0 会先打印再进入 kernel。
    if (getenv("GRAPH_DEBUG_DISPATCH") && p->ith == 0) {
        fprintf(stderr,
                "[DISP] node=%p op=%d | src0{op=%d data=%p buf=%p} src1{op=%d data=%p buf=%p}"
                " dst{op=%d data=%p buf=%p}\n",
                (void*)node, (int)node->op,
                (node->src[0] ? (int)node->src[0]->op : -1),
                (node->src[0] ? (void*)node->src[0]->data() : nullptr),
                (node->src[0] ? (void*)node->src[0]->buffer_ : nullptr),
                (node->src[1] ? (int)node->src[1]->op : -1),
                (node->src[1] ? (void*)node->src[1]->data() : nullptr),
                (node->src[1] ? (void*)node->src[1]->buffer_ : nullptr),
                (int)node->op, (void*)node->data(), (void*)node->buffer_);
    }

    // 默认纯 CPU 路径：本 node 无 device 输入时直接执行，行为与旧版完全一致。
    // 有 device 输入（混合训练跨后端缺口）时走 buffer-aware 暂存，避免段错误。
    // 注：view src 的 data() 解析已移到 CPUBackend::graph_compute 提交前（单线程），避免多线程
    // 写共享 src->data()/buffer_ 的竞态（崩溃位置漂移）。
    if (!cpu_buffer_aware_enabled() && !node_has_device_input(node)) {
        return dispatch_body(node, p);
    }

    // ---- buffer-aware：暂存 device 输入为 host，跑 kernel，恢复 ----
    // node 是 CPU 后端节点，node->data() 为 host；仅 src 可能是 device buffer。
    // 多线程：thread0 完成 D2H 暂存后 barrier，所有线程跑 kernel，再 barrier 恢复。
    float* orig_data[GGML_MAX_SRC];
    for (int s = 0; s < GGML_MAX_SRC; s++) orig_data[s] = nullptr;
    std::vector<std::vector<float>> scratch(GGML_MAX_SRC);

    if (p->ith == 0) {
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            TensorF32* src = node->src[s];
            if (!src) continue;
            orig_data[s] = stage_device_src(src, scratch[s]);
        }
    }
    if (p->nth > 1) p->threadpool->barrier_wait();   // 等 thread0 暂存完成

    Status st = dispatch_body(node, p);              // 执行 kernel（读暂存后的 host src）

    if (p->nth > 1) p->threadpool->barrier_wait();   // 等所有线程 kernel 完成
    if (p->ith == 0) {
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (orig_data[s]) node->src[s]->bind_data(orig_data[s]);  // 恢复原 device 指针
        }
    }
    return st;
}

// ===== elemwise =====
void CPUBackend::kernel_elemwise(TensorF32 * node, ComputeParams * p) {
    // 混训崩溃定位：elemwise 裸读 src/dst。若 src[0]/src[1] 为 null，或指针指向 device/悬垂，
    // 这里解引用即 SIGSEGV。必须先把 src 判空再解引用，否则诊断自身先在 src[0]->data() 崩掉。
    // 所有线程带 barrier 逐 node 同步 → 线程 0 也处理崩溃节点，在解引用前打印 → 最后一个
    // [ELEM-PTR] 行就是故障节点。GRAPH_DEBUG_ELEM=1 时线程 0 对每个 elemwise 节点打印。
    if (getenv("GRAPH_DEBUG_ELEM") && p->ith == 0) {
        TensorF32* s0 = node->src[0];
        TensorF32* s1 = node->src[1];
        TensorF32* s2 = node->src[2];
        fprintf(stderr,
                "[ELEM-PTR] op=%d n=%lld src0=%s num0=%lld src1=%s num1=%lld src2=%s num2=%lld | src0{op=%d buf=%p host=%d data=%p flag=%d}"
                " src1{op=%d buf=%p host=%d data=%p flag=%d} dst{op=%d buf=%p host=%d data=%p flag=%d}%s\n",
                (int)node->op, (long long)node->numel(),
                s0 ? "ok" : "NULL", s0 ? (long long)s0->numel() : -1,
                s1 ? "ok" : "NULL", s1 ? (long long)s1->numel() : -1,
                s2 ? "ok" : "NULL", s2 ? (long long)s2->numel() : -1,
                s0 ? (int)s0->op : -1, s0 ? (void*)s0->buffer_ : nullptr,
                (s0 && s0->buffer_ ? (int)s0->buffer_->is_host() : -1),
                s0 ? (void*)s0->data() : nullptr, s0 ? (int)s0->flag : -1,
                s1 ? (int)s1->op : -1, s1 ? (void*)s1->buffer_ : nullptr,
                (s1 && s1->buffer_ ? (int)s1->buffer_->is_host() : -1),
                s1 ? (void*)s1->data() : nullptr, s1 ? (int)s1->flag : -1,
                (int)node->op, (void*)(node->buffer_),
                (node->buffer_ ? (int)node->buffer_->is_host() : -1),
                (void*)node->data(), (int)node->flag,
                s2 ? " src2{...}" : "");
    }

    // ⚠️ 判空防护：src[0]/src[1] 为 null（图构建错误/悬垂节点）或 data() 未分配时，
    // 直接解引用会 SIGSEGV。所有线程一致 return（与 kernel_mul_mat 一致，避免 barrier 失步）。
    if (!node->src[0] || !node->src[1] || !node->data()) {
        if (p->ith == 0) {
            fprintf(stderr,
                    "[ELEM-NULL] op=%d n=%lld src0=%p src1=%p dst_data=%p\n",
                    (int)node->op, (long long)node->numel(),
                    (void*)node->src[0], (void*)node->src[1], (void*)node->data());
        }
        return;
    }
    const float * a = node->src[0]->data();
    const float * b = node->src[1]->data();
    float * d = node->data();
    int64_t n = node->numel();
    if (!a || !b || !d) {
        if (p->ith == 0) {
            fprintf(stderr, "[ELEM-NULL2] op=%d n=%lld a=%p b=%p d=%p\n",
                    (int)node->op, (long long)n, (const void*)a, (const void*)b, (const void*)d);
        }
        return;
    }

    // 开发诊断（GRAPH_DEBUG_LOSS）：标量 add（total 链累加），打印两输入值与地址，定位 nan 来源
    if (p->ith == 0 && n == 1 && getenv("GRAPH_DEBUG_LOSS") && node->op == OP_ADD) {
        fprintf(stderr, "[add] numel=1 a_val=%f a_addr=%p b_val=%f b_addr=%p d_addr=%p\n",
                a[0], (const void*)a, b[0], (const void*)b, (const void*)d);
    }

    // 广播支持：当 src 形状是 dst 形状去掉若干"前导最内维"的后缀时（如 a=[2,7,N], b=[7,N]），
    // src 沿 dst 的前导维重复。扁平序下最内维在前，故 src 索引 = i % src_numel。
    // 合法广播必须满足 n 是 an/bn 的整数倍；若不整除（图构建错位：view/unsqueeze/repeat 后形状
    // 未对齐），原实现走全等分支 a[i]/b[i] 越界读 → SIGSEGV。此处改为无条件安全索引
    // （a_full 时 a[i]，否则 a[i % an]），非法广播时数值可能错但不崩，并打印 [ELEM-BCAST] 暴露节点。
    const int64_t an = node->src[0]->numel();
    const int64_t bn = node->src[1]->numel();
    const bool a_full = (an == n);
    const bool b_full = (bn == n);
    const bool bad_bcast =
        (!a_full && (an <= 0 || n % an != 0)) || (!b_full && (bn <= 0 || n % bn != 0));
    if (bad_bcast && p->ith == 0 && getenv("GRAPH_DEBUG_BCAST")) {
        const Shape& sa = node->src[0]->shape();
        const Shape& sb = node->src[1]->shape();
        const Shape& sd = node->shape();
        TensorF32* s0 = node->src[0];
        TensorF32* s0src = (s0 && s0->src[0]) ? s0->src[0] : nullptr;
        TensorF32* s0src2 = (s0src && s0src->src[0]) ? s0src->src[0] : nullptr;
        TensorF32* s1 = node->src[1];
        TensorF32* s1src = (s1 && s1->src[0]) ? s1->src[0] : nullptr;
        TensorF32* s1src2 = (s1src && s1src->src[0]) ? s1src->src[0] : nullptr;
        TensorF32* s1src3 = (s1src2 && s1src2->src[0]) ? s1src2->src[0] : nullptr;
        fprintf(stderr,
                "[ELEM-BCAST] op=%d n=%lld an=%lld bn=%lld "
                "| src0{op=%d ndim=%d dims=[%lld,%lld,%lld,%lld] numel=%lld src1op=%d src2op=%d} "
                "src1{op=%d ndim=%d dims=[%lld,%lld,%lld,%lld] numel=%lld s1_src0op=%d s1_src1op=%d s1_src2op=%d} "
                "dst{ndim=%d dims=[%lld,%lld,%lld,%lld]}\n",
                (int)node->op, (long long)n, (long long)an, (long long)bn,
                (int)s0->op, (int)sa.ndim(),
                (long long)(sa.ndim()>0?sa.dims[0]:-1), (long long)(sa.ndim()>1?sa.dims[1]:-1),
                (long long)(sa.ndim()>2?sa.dims[2]:-1), (long long)(sa.ndim()>3?sa.dims[3]:-1),
                (long long)sa.numel(),
                (s0src ? (int)s0src->op : -1),
                (s0src2 ? (int)s0src2->op : -1),
                (int)s1->op, (int)sb.ndim(),
                (long long)(sb.ndim()>0?sb.dims[0]:-1), (long long)(sb.ndim()>1?sb.dims[1]:-1),
                (long long)(sb.ndim()>2?sb.dims[2]:-1), (long long)(sb.ndim()>3?sb.dims[3]:-1),
                (long long)sb.numel(),
                (s1src ? (int)s1src->op : -1),
                (s1src2 ? (int)s1src2->op : -1),
                (s1src3 ? (int)s1src3->op : -1),
                (int)sd.ndim(),
                (long long)(sd.ndim()>0?sd.dims[0]:-1), (long long)(sd.ndim()>1?sd.dims[1]:-1),
                (long long)(sd.ndim()>2?sd.dims[2]:-1), (long long)(sd.ndim()>3?sd.dims[3]:-1));
    }

    switch (node->op) {
        case OP_ADD:
            for (int64_t i = p->ith; i < n; i += p->nth)
                d[i] = a[a_full ? i : i % an] + b[b_full ? i : i % bn];
            if (getenv("GRAPH_DEBUG_ELEM_NAN") && node->numel() <= 4096) {
                int cnt = 0;
                // 统计 a/b 中 NaN 数（判断 node#8 dispatch 时 src 是否已含 NaN）
                int na = 0, nb = 0;
                for (int64_t i = 0; i < n; ++i) {
                    if (std::isnan(a_full ? a[i] : a[i % an])) na++;
                    if (std::isnan(b_full ? b[i] : b[i % bn])) nb++;
                }
                for (int64_t i = 0; i < n; ++i) if (std::isnan(d[i])) {
                    if (cnt < 3) fprintf(stderr, "[elem-add] nan@%lld a=%f b=%f src0op=%d src1op=%d\n",
                        (long long)i, a_full ? a[i] : a[i % an], b_full ? b[i] : b[i % bn],
                        (node->src[0]?(int)node->src[0]->op:-1),
                        (node->src[1]?(int)node->src[1]->op:-1));
                    cnt++;
                }
                if (cnt > 0) fprintf(stderr, "[elem-add] node_n? op=%d total_nan=%d/%lld a_nan=%d b_nan=%d self=%p src0=%p src1=%p\n",
                    (int)node->op, cnt, (long long)n, na, nb,
                    (const void*)d, (const void*)a, (const void*)b);
            }
            break;
        case OP_SUB:
            for (int64_t i = p->ith; i < n; i += p->nth)
                d[i] = a[a_full ? i : i % an] - b[b_full ? i : i % bn];
            break;
        case OP_MUL:
            for (int64_t i = p->ith; i < n; i += p->nth)
                d[i] = a[a_full ? i : i % an] * b[b_full ? i : i % bn];
            break;
        case OP_DIV:
            for (int64_t i = p->ith; i < n; i += p->nth)
                d[i] = a[a_full ? i : i % an] / b[b_full ? i : i % bn];
            break;
    }
}

// ===== mul_mat =====
void CPUBackend::kernel_mul_mat(TensorF32 * node, ComputeParams * p) {
    ThreadPool * tp = p->threadpool;
    int M = static_cast<int>(node->shape().dims[1]);
    int N = static_cast<int>(node->shape().dims[0]);
    int K = static_cast<int>(node->src[0]->shape().dims[0]);
    const float * a = node->src[0]->data();
    const float * b = node->src[1]->data();
    float * d = node->data();
    // 混训崩溃定位：mul_mat 裸读 src/dst 指针，若参数被搬到 device/悬垂/null，这里解引用即 SIGSEGV。
    // 打印 src 的 data()/buffer_/is_host/flag，区分"参数被 device 化"vs"buffer 复用悬垂"vs"null"。
    if (!a || !b || !d) {
        if (p->ith == 0) {
            fprintf(stderr,
                    "[MULMAT-NULL] M=%d N=%d K=%d a=%p b=%p d=%p | src0{op=%d buf=%p host=%d data=%p flag=%d}"
                    " src1{op=%d buf=%p host=%d data=%p flag=%d} dst{op=%d buf=%p host=%d data=%p flag=%d}\n",
                    M, N, K, (const void*)a, (const void*)b, (const void*)d,
                    (int)node->src[0]->op, (void*)(node->src[0]->buffer_),
                    (node->src[0]->buffer_ ? (int)node->src[0]->buffer_->is_host() : -1),
                    (void*)node->src[0]->data(), (int)node->src[0]->flag,
                    (int)node->src[1]->op, (void*)(node->src[1]->buffer_),
                    (node->src[1]->buffer_ ? (int)node->src[1]->buffer_->is_host() : -1),
                    (void*)node->src[1]->data(), (int)node->src[1]->flag,
                    (int)node->op, (void*)(node->buffer_),
                    (node->buffer_ ? (int)node->buffer_->is_host() : -1),
                    (void*)node->data(), (int)node->flag);
            fprintf(stderr, "[MULMAT-NULL] aborting mul_mat node to avoid SIGSEGV\n");
        }
        // ⚠️ 所有线程必须一致 return：kernel 内部有 barrier（GPU/CPU 路径各 2 次），若只线程 0
        // return 而 worker 继续走 barrier → barrier 次数不匹配 → 线程失步 → 后续 kernel_elemwise 崩。
        // 所有线程都 return → 每线程跳过的 barrier 次数一致 → 同步不破坏。
        return;
    }
    if (getenv("GRAPH_DEBUG_MULMAT") && (int64_t)M * N * K > 200000000LL && p->ith == 0) {
        std::fprintf(stderr, "[MULMAT-BIG] M=%d N=%d K=%d ops=%lld a={%lld,%lld,%lld,%lld} b={%lld,%lld,%lld,%lld} d={%lld,%lld,%lld,%lld}\n",
            M, N, K, (long long)((int64_t)M * N * K),
            (long long)(node->src[0]->shape().dims.size()>0?node->src[0]->shape().dims[0]:-1),
            (long long)(node->src[0]->shape().dims.size()>1?node->src[0]->shape().dims[1]:-1),
            (long long)(node->src[0]->shape().dims.size()>2?node->src[0]->shape().dims[2]:-1),
            (long long)(node->src[0]->shape().dims.size()>3?node->src[0]->shape().dims[3]:-1),
            (long long)(node->src[1]->shape().dims.size()>0?node->src[1]->shape().dims[0]:-1),
            (long long)(node->src[1]->shape().dims.size()>1?node->src[1]->shape().dims[1]:-1),
            (long long)(node->src[1]->shape().dims.size()>2?node->src[1]->shape().dims[2]:-1),
            (long long)(node->src[1]->shape().dims.size()>3?node->src[1]->shape().dims[3]:-1),
            (long long)(node->shape().dims.size()>0?node->shape().dims[0]:-1),
            (long long)(node->shape().dims.size()>1?node->shape().dims[1]:-1),
            (long long)(node->shape().dims.size()>2?node->shape().dims[2]:-1),
            (long long)(node->shape().dims.size()>3?node->shape().dims[3]:-1));
    }


    // ===== 开发用 GPU 加速矩阵乘（方案 B，临时，见文件顶部注释）=====
    // 开关 PPML_MUL_MAT_GPU=1。首次调用探测 GPU 是否存在（缓存到 thread-safe static）；
    // 无 GPU 时直接走 CPU 路径（与原始行为完全等价，不引入额外 barrier）。
    // 有 GPU 时流程：所有线程 barrier → ith==0 尝试 GPU（结果写 current_chunk）→ 所有线程
    // barrier → 若成功全部返回（内部 barrier 恰 2 次，与 CPU 路径一致）；cudaMalloc 失败时
    // 回退 CPU（概率极低，开发用可接受轻微 barrier 次数差异）。
    if (g_mul_mat_gpu_enabled) {
        static const bool gpu_avail = []() {
            int ndev = 0;
            return (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0);
        }();
        if (gpu_avail) {
            // barrier #1：所有线程到达，准备 GPU 尝试
            if (p->ith == 0) tp->current_chunk.store(0);
            tp->barrier_wait();

            if (p->ith == 0) {
                int gpu_ok = 0;
                if (M > 0 && N > 0 && K > 0) {
                    size_t bA = (size_t)M * K * sizeof(float);
                    size_t bB = (size_t)N * K * sizeof(float);
                    size_t bC = (size_t)M * N * sizeof(float);
                    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
                    if (cudaMalloc(&dA, bA) == cudaSuccess &&
                        cudaMalloc(&dB, bB) == cudaSuccess &&
                        cudaMalloc(&dC, bC) == cudaSuccess) {
                        cudaMemcpy(dA, a, bA, cudaMemcpyHostToDevice);
                        cudaMemcpy(dB, b, bB, cudaMemcpyHostToDevice);
                        mul_mat_cuda(dA, dB, dC, M, K, N);
                        cudaDeviceSynchronize();
                        cudaMemcpy(d, dC, bC, cudaMemcpyDeviceToHost);
                        gpu_ok = 1;
                    }
                    if (dA) cudaFree(dA);
                    if (dB) cudaFree(dB);
                    if (dC) cudaFree(dC);
                }
                tp->current_chunk.store(gpu_ok);
            }
            // barrier #2：等 ith==0 完成 GPU 尝试
            tp->barrier_wait();
            if (tp->current_chunk.load() == 1) {
                // GPU 成功：所有线程一致返回（内部 barrier 恰 2 次，与 CPU 路径对齐）
                return;
            }
            // GPU 失败(cudaMalloc 失败)：落到下方 CPU 朴素路径
        }
    }

    // ===== CPU 朴素矩阵乘（原逻辑；GPU 未启用/无 GPU/GPU 失败时执行）=====
    // GRAPH_DEBUG_KERNEL=1：检查 mul_mat 输入 a/b 值域（定位 emb_t1d_ 巨大是输入问题还是 kernel 问题）
    if (getenv("GRAPH_DEBUG_KERNEL") && p->ith == 0) {
        bool abad = false; float amn = 0, amx = 0; int64_t anan = 0;
        const int64_t anelt = (int64_t)M * K;
        if (anelt > 0 && a) { amn = a[0]; amx = a[0]; for (int64_t q = 0; q < anelt; q++) { float v = a[q]; if (v != v) { anan++; abad = true; } else { if (v < amn) amn = v; if (v > amx) amx = v; } } if (amx > 1e6f || amn < -1e6f) abad = true; }
        if (abad) fprintf(stderr, "[mulmat-in-a] M=%d K=%d numel=%lld min=%.6g max=%.6g nan=%lld\n", M, K, (long long)anelt, (double)amn, (double)amx, (long long)anan);
        bool bbad = false; float bmn = 0, bmx = 0; int64_t bnan = 0;
        const int64_t bnelt = (int64_t)N * K;
        if (bnelt > 0 && b) { bmn = b[0]; bmx = b[0]; for (int64_t q = 0; q < bnelt; q++) { float v = b[q]; if (v != v) { bnan++; bbad = true; } else { if (v < bmn) bmn = v; if (v > bmx) bmx = v; } } if (bmx > 1e6f || bmn < -1e6f) bbad = true; }
        if (bbad) fprintf(stderr, "[mulmat-in-b] N=%d K=%d numel=%lld min=%.6g max=%.6g nan=%lld\n", N, K, (long long)bnelt, (double)bmn, (double)bmx, (long long)bnan);
    }
    if (p->ith == 0) tp->current_chunk.store(0);
    tp->barrier_wait();

    // a (M * K), b (K * N), d (M * N)
    // 注意：sum 必须在每个输出列 j 前清零（点积按列独立累加）。
    //   bug 修复(2026-08-19)：原实现 sum 只在行 i 处重置，导致跨 j 持续累加，
    //   所有 Linear/mul_mat 输出被污染 → 各 head 输出常数、loss 全 0。
    while (true) {
        int i = tp->current_chunk.fetch_add(1);
        if (i >= M) break;
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) sum += a[i * K + k] * b[j * K + k];
            d[j + i * N] = sum;
        }
    }
    tp->barrier_wait();
}

// ===== out_prod =====
// 对标 ggml_compute_forward_out_prod_f32
// 数学: dst[i0, i1, i2, i3] = sum_{i01} src0[i0, i01, i2, i3] * src1[i1, i01, i2, i3]
// dst shape: (ne00, ne10, ne12, ne13)  — 实际就是 A @ B^T 在倒数第二维上收缩
// 当前项目 4D 约定: dims = (ne0, ne1, ne2, ne3), ne0 是最内维
void CPUBackend::kernel_out_prod(TensorF32 * node, ComputeParams * p) {
    ThreadPool * tp = p->threadpool;

    const TensorF32 * src0 = node->src[0];
    const TensorF32 * src1 = node->src[1];

    const int64_t ne00 = src0->shape().dims[0];  // src0 inner dim
    const int64_t ne01 = src0->shape().dims[1];  // src0 K dim (contraction dim)
    const int64_t ne02 = (src0->shape().ndim() > 2) ? src0->shape().dims[2] : 1;
    const int64_t ne03 = (src0->shape().ndim() > 3) ? src0->shape().dims[3] : 1;

    const int64_t ne10 = src1->shape().dims[0];  // src1 inner dim
    const int64_t ne11 = src1->shape().dims[1];  // src1 K dim (contraction dim, == ne01)
    const int64_t ne12 = (src1->shape().ndim() > 2) ? src1->shape().dims[2] : 1;
    const int64_t ne13 = (src1->shape().ndim() > 3) ? src1->shape().dims[3] : 1;
    (void)ne02; (void)ne03; (void)ne12; (void)ne13;  // unused for now

    // dst shape: (ne00, ne10, max(ne02,ne12), max(ne03,ne13))
    const int64_t ne0 = node->shape().dims[0];  // == ne00
    const int64_t ne1 = node->shape().dims[1];  // == ne10
    const int64_t ne2 = (node->shape().ndim() > 2) ? node->shape().dims[2] : 1;
    const int64_t ne3 = (node->shape().ndim() > 3) ? node->shape().dims[3] : 1;

    const float * src0_data = src0->data();
    const float * src1_data = src1->data();
    float * dst_data  = node->data();

    // ===== thread 0: 清零 dst =====
    if (p->ith == 0) {
        for (int64_t i = 0; i < ne0 * ne1 * ne2 * ne3; i++) {
            dst_data[i] = 0.0f;
        }
    }
    tp->barrier_wait();

    // ===== 并行化: 按 dst 的 (ne1, ne2, ne3) 维分配到线程 =====
    // total rows in dst
    const int64_t nr = ne1 * ne2 * ne3;

    // rows per thread
    const int64_t dr = (nr + p->nth - 1) / p->nth;

    // row range for this thread
    const int64_t ir0 = dr * p->ith;
    const int64_t ir1 = (ir0 + dr < nr) ? (ir0 + dr) : nr;

    // really?  had to double check for this --- albert
    // ===== GQA (Group Query Attention) 支持 =====
    // dps2 = ne2 / ne02, 当 src0 的 ne02 < ne12 时, 共享 K/V heads
    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    // ===== block-tiling 参数 (对标 ggml blck_0 / blck_1) =====
    const int64_t blck_0 = 32;  // K 维 block 大小 (对标 GGML_VEC_MAD_UNROLL)
    const int64_t blck_1 = 16;  // 输出行 block 大小

    // ===== 主循环: 双层 block tiling =====
    for (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        const int64_t bir1 = (bir + blck_1 < ir1) ? (bir + blck_1) : ir1;

        for (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            const int64_t bne01 = (bi01 + blck_0 < ne01) ? (bi01 + blck_0) : ne01;

            for (int64_t ir = bir; ir < bir1; ir++) {
                // 反解 dst 索引 (i1, i2, i3)
                const int64_t i3 = ir / (ne2 * ne1);
                const int64_t i2 = (ir - i3 * ne2 * ne1) / ne1;
                const int64_t i1 = ir - i3 * ne2 * ne1 - i2 * ne1;

                // GQA: 将 dst 的 (i2, i3) 映射回 src0 的对应 dim
                const int64_t i02 = i2 / dps2;
                const int64_t i03 = i3 / dps3;

                // src1 的 (i2, i3) 与 dst 一致
                const int64_t i12 = i2;
                const int64_t i13 = i3;

                // dst 行基地址
                float * d_row = dst_data + (i1 * ne0 + i2 * ne0 * ne1 + i3 * ne0 * ne1 * ne2);

                // 沿 K 维 (ne01) 累加
                for (int64_t i01 = bi01; i01 < bne01; i01++) {
                    // src0 行: (i01, i02, i03) → src0 的第 i01 行
                    // 布局: src0[i01, i02, i03] 对应 data[i01*ne00 + i02*ne00*ne01 + i03*ne00*ne01*ne02]
                    const float * s0 = src0_data + (i01 * ne00 + i02 * ne00 * ne01 + i03 * ne00 * ne01 * ne02);

                    // src1 元素: (i1, i01, i12, i13)
                    const float * s1_row = src1_data + (i01 * ne10 + i12 * ne10 * ne11 + i13 * ne10 * ne11 * ne12);
                    float s1_val = s1_row[i1];  // src1[i1, i01, i12, i13]

                    // d_row[i0] += s0[i0] * s1_val  for i0 in [0, ne00)
                    for (int64_t i0 = 0; i0 < ne00; i0++) {
                        d_row[i0] += s0[i0] * s1_val;
                    }
                }
            }
        }
    }

    tp->barrier_wait();
}

// ===== triangle multiplication =====
// outgoing=true:  einsum('bikd,bjkd->bijd', left, right/L)
// outgoing=false: einsum('bkid,bkjd->bijd', left, right/L)
//
// 【ggml 布局（dims[0]=最内维）】  —— 与 kernel_out_prod / kernel_outer_prod_mean 一致
//   outgoing:
//     left  [D, I, K, B]：left[d,i,k,b] = data[((d*I + i)*K + k)*B + b]
//     right [D, J, K, B]：right[d,j,k,b] = data[((d*J + j)*K + k)*B + b]
//     dst   [D, I, J, B]：dst[d,i,j,b]  = data[((d*I + i)*J + j)*B + b]
//     dst[d,i,j,b] = (1/L) * sum_k left[d,i,k,b] * right[d,j,k,b]
//   incoming:
//     left  [D, K, I, B]：left[d,k,i,b] = data[((d*K + k)*I + i)*B + b]
//     right [D, K, J, B]：right[d,k,j,b] = data[((d*K + k)*J + j)*B + b]
//     dst   [D, I, J, B]
//     dst[d,i,j,b] = (1/L) * sum_k left[d,k,i,b] * right[d,k,j,b]
//
// 注意：此前 kernel 误用值布局（B=dst.dims[0], D=dst.dims[3]），而图张量是 ggml 布局
//   （特征维最内），导致 outgoing/incoming 索引错乱 → 已改为 ggml 布局。
void CPUBackend::kernel_tri_mul(TensorF32 * node, ComputeParams * p) {
    ThreadPool * tp = p->threadpool;

    const TensorF32 * src0 = node->src[0];  // left
    const TensorF32 * src1 = node->src[1];  // right
    TensorF32       * dst  = node;

    const int64_t D = dst->shape().dims[0];  // 最内特征维
    const int64_t I = dst->shape().dims[1];
    const int64_t J = dst->shape().dims[2];
    const int64_t B = dst->shape().dims[3];

    float L;
    bool  outgoing;
    memcpy(&L,        node->op_params,      sizeof(float));
    memcpy(&outgoing, node->op_params + 4,  sizeof(bool));

    const float * left_data  = static_cast<const float*>(src0->data());
    const float * right_data = static_cast<const float*>(src1->data());
    float       * dst_data   = static_cast<float*>(dst->data());

    const float inv_L = 1.0f / L;
    const int64_t total = B * I * J * D;
    const int64_t per  = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    for (int64_t idx = start; idx < end; idx++) {
        int64_t tmp = idx;
        int64_t b   = tmp % B; tmp /= B;
        int64_t j   = tmp % J; tmp /= J;
        int64_t i   = tmp % I;
        int64_t d   = tmp / I;

        float sum = 0.0f;
        if (outgoing) {
            // 收缩 k = src0.dims[2]（left/right 的 dims[2] 相同）
            const int64_t K = src0->shape().dims[2];
            for (int64_t k = 0; k < K; k++) {
                const float lv = left_data[((d * I + i) * K + k) * B + b];
                const float rv = right_data[((d * J + j) * K + k) * B + b];
                sum += lv * rv;
            }
        } else {
            // incoming: k = src0.dims[1]（left/right 的 dims[1] 相同）
            const int64_t K = src0->shape().dims[1];
            for (int64_t k = 0; k < K; k++) {
                const float lv = left_data[((d * K + k) * I + i) * B + b];
                const float rv = right_data[((d * K + k) * J + j) * B + b];
                sum += lv * rv;
            }
        }
        float v = sum * inv_L;
        // 数值护栏：三角乘是 NaN/Inf 高发点（left/right 经 sigmoid gate 后仍可能因上游
        // softmax 溢出带入非有限值）。这里做 finite-clamp，避免 NaN 沿 pair 链污染 distogram
        // head 与下游 block，导致整图 loss 变 NaN（问题4）。clamp 到 ±1e4 对 pair 表征无实质影响。
        if (!(v == v) || v > 1e4f || v < -1e4f) {
            v = (v != v) ? 0.0f : (v > 1e4f ? 1e4f : -1e4f);
        }
        dst_data[idx] = v;
    }

    tp->barrier_wait();
}

// ===== outer_product_mean (msa2pair) =====
// einsum('bikd,bjkd->bijd(de)', left, right/N) —— 收缩 seq 维 N（dims[2]），
//   且特征维做笛卡尔积：D × D → D*D（AF2 outer_product_mean 语义）。
// ggml 布局（dims[0]=最内维）:
//   left  [D, L, N, B]：left[d1,i,n,b] = left_data[((d1*L + i)*N + n)*B + b]
//   right [D, L, N, B]：right[d2,j,n,b] = right_data[((d2*L + j)*N + n)*B + b]
//   dst   [D*D, L, L, B]：dst[(d1*D+d2), i, j, b]
//       = (1/N) * sum_n left[d1,i,n,b] * right[d2,j,n,b]
// 索引：dst[(((d1*D+d2)*L + i)*L + j)*B + b]
// op_params[0] 存 N（float 位模式，实际以 shape 为准）。
void CPUBackend::kernel_outer_prod_mean(TensorF32 * node, ComputeParams * p) {
    ThreadPool * tp = p->threadpool;

    const TensorF32 * src0 = node->src[0];  // left [D,L,N,B]
    const TensorF32 * src1 = node->src[1];  // right [D,L,N,B]
    TensorF32       * dst  = node;          // [D*D, L, L, B]

    const int64_t D = src0->shape().dims[0];
    const int64_t L = src0->shape().dims[1];
    const int64_t N = src0->shape().dims[2];
    const int64_t B = src0->shape().dims[3];
    const int64_t D2 = D * D;

    const float inv_N = 1.0f / float(N);   // 以实际 N 为准

    const float * left_data  = static_cast<const float*>(src0->data());
    const float * right_data = static_cast<const float*>(src1->data());
    float       * dst_data   = static_cast<float*>(dst->data());

    const int64_t total = D2 * L * L * B;
    const int64_t per  = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    for (int64_t idx = start; idx < end; idx++) {
        int64_t tmp = idx;
        int64_t b   = tmp % B; tmp /= B;
        int64_t j   = tmp % L; tmp /= L;
        int64_t i   = tmp % L;
        int64_t d2  = tmp % D; tmp /= D;
        int64_t d1  = tmp / D;

        float sum = 0.0f;
        for (int64_t n = 0; n < N; n++) {
            const float lv = left_data[((d1 * L + i) * N + n) * B + b];
            const float rv = right_data[((d2 * L + j) * N + n) * B + b];
            sum += lv * rv;
        }
        dst_data[idx] = sum * inv_N;
    }

    tp->barrier_wait();
}

// ===== outer_product_mean 反向 =====
// 前向: dst[(d1*D+d2), i, j, b] = (1/N)*sum_n left[d1,i,n,b]*right[d2,j,n,b]
//   dL/dleft [d1,i,n,b] = (1/N) * sum_{d2,j} grad[(d1*D+d2), i, j, b] * right[d2,j,n,b]
//   dL/dright[d2,j,n,b] = (1/N) * sum_{d1,i} grad[(d1*D+d2), i, j, b] * left[d1,i,n,b]
// 节点 src: [0]=grad [D*D,L,L,B], [1]=left [D,L,N,B], [2]=right [D,L,N,B]
//          [3]=grad_left [D,L,N,B], [4]=grad_right [D,L,N,B]
void CPUBackend::kernel_outer_prod_mean_back(TensorF32 * node, ComputeParams * p) {
    ThreadPool * tp = p->threadpool;

    const TensorF32 * grad  = node->src[0];  // [D*D,L,L,B]
    const TensorF32 * left  = node->src[1];  // [D,L,N,B]
    const TensorF32 * right = node->src[2];  // [D,L,N,B]
    TensorF32 * grad_left  = node->src[3];   // [D,L,N,B]
    TensorF32 * grad_right = node->src[4];   // [D,L,N,B]

    if (!grad_left || !grad_right) { tp->barrier_wait(); return; }

    const int64_t D = left->shape().dims[0];
    const int64_t L = left->shape().dims[1];
    const int64_t N = left->shape().dims[2];
    const int64_t B = left->shape().dims[3];

    const float inv_N = 1.0f / float(N);

    const float * gdata  = static_cast<const float*>(grad->data());
    const float * ldata  = static_cast<const float*>(left->data());
    const float * rdata  = static_cast<const float*>(right->data());
    float * gl_out = static_cast<float*>(grad_left->data());
    float * gr_out = static_cast<float*>(grad_right->data());

    // dL/dleft [d1,i,n,b] = (1/N)*sum_{d2,j} grad[(d1*D+d2),i,j,b]*right[d2,j,n,b]
    {
        const int64_t total = D * L * N * B;
        const int64_t per  = (total + p->nth - 1) / p->nth;
        const int64_t start = per * p->ith;
        const int64_t end   = (start + per < total) ? (start + per) : total;
        for (int64_t idx = start; idx < end; idx++) {
            int64_t tmp = idx;
            int64_t b   = tmp % B; tmp /= B;
            int64_t n   = tmp % N; tmp /= N;
            int64_t i   = tmp % L;
            int64_t d1  = tmp / L;
            float sum = 0.0f;
            for (int64_t j = 0; j < L; j++) {
                for (int64_t d2 = 0; d2 < D; d2++) {
                    const float gv = gdata[(((d1 * D + d2) * L + i) * L + j) * B + b];
                    const float rv = rdata[((d2 * L + j) * N + n) * B + b];
                    sum += gv * rv;
                }
            }
            gl_out[idx] = sum * inv_N;
        }
    }

    // dL/dright [d2,j,n,b] = (1/N)*sum_{d1,i} grad[(d1*D+d2), i, j, b]*left[d1,i,n,b]
    {
        const int64_t total = D * L * N * B;
        const int64_t per  = (total + p->nth - 1) / p->nth;
        const int64_t start = per * p->ith;
        const int64_t end   = (start + per < total) ? (start + per) : total;
        for (int64_t idx = start; idx < end; idx++) {
            int64_t tmp = idx;
            int64_t b   = tmp % B; tmp /= B;
            int64_t n   = tmp % N; tmp /= N;
            int64_t j   = tmp % L;
            int64_t d2  = tmp / L;
            float sum = 0.0f;
            for (int64_t i = 0; i < L; i++) {
                for (int64_t d1 = 0; d1 < D; d1++) {
                    const float gv = gdata[(((d1 * D + d2) * L + i) * L + j) * B + b];
                    const float lv = ldata[((d1 * L + i) * N + n) * B + b];
                    sum += gv * lv;
                }
            }
            gr_out[idx] = sum * inv_N;
        }
    }

    tp->barrier_wait();
}

// ===== outer_product (pair2pair gate，纯外积，无收缩) =====
// 前向: gate[(d1*D+d2), i, j, b] = left[d1,i,b] * right[d2,j,b]（特征笛卡尔积 D×D→D*D）
// ggml 布局（dims[0]=最内维）:
//   left  [D, L, B]：left[d1,i,b] = left_data[((d1*L + i)*B + b)]
//   right [D, L, B]：right[d2,j,b] = right_data[((d2*L + j)*B + b)]
//   dst   [D*D, L, L, B]：dst[(((d1*D+d2)*L + i)*L + j)*B + b]
void CPUBackend::kernel_outer_prod(TensorF32 * node, ComputeParams * p) {
    ThreadPool * tp = p->threadpool;

    const TensorF32 * src0 = node->src[0];  // left [D,L,B]
    const TensorF32 * src1 = node->src[1];  // right [D,L,B]
    TensorF32       * dst  = node;          // [D*D, L, L, B]

    const int64_t D = src0->shape().dims[0];
    const int64_t L = src0->shape().dims[1];
    const int64_t B = src0->shape().dims[2];
    const int64_t D2 = D * D;

    const float * left_data  = static_cast<const float*>(src0->data());
    const float * right_data = static_cast<const float*>(src1->data());
    float       * dst_data   = static_cast<float*>(dst->data());

    const int64_t total = D2 * L * L * B;
    const int64_t per  = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    for (int64_t idx = start; idx < end; idx++) {
        int64_t tmp = idx;
        int64_t b   = tmp % B; tmp /= B;
        int64_t j   = tmp % L; tmp /= L;
        int64_t i   = tmp % L;
        int64_t d2  = tmp % D; tmp /= D;
        int64_t d1  = tmp / D;

        const float lv = left_data[((d1 * L + i) * B + b)];
        const float rv = right_data[((d2 * L + j) * B + b)];
        dst_data[idx] = lv * rv;
    }

    tp->barrier_wait();
}

// ===== outer_product 反向 =====
// 前向: dst[(d1*D+d2), i, j, b] = left[d1,i,b]*right[d2,j,b]
//   dL/dleft [d1,i,b] = sum_{d2,j} grad[(d1*D+d2), i, j, b] * right[d2,j,b]
//   dL/dright[d2,j,b] = sum_{d1,i} grad[(d1*D+d2), i, j, b] * left[d1,i,b]
// 节点 src: [0]=grad [D*D,L,L,B], [1]=left [D,L,B], [2]=right [D,L,B]
//          [3]=grad_left [D,L,B], [4]=grad_right [D,L,B]
void CPUBackend::kernel_outer_prod_back(TensorF32 * node, ComputeParams * p) {
    ThreadPool * tp = p->threadpool;

    const TensorF32 * grad  = node->src[0];  // [D*D,L,L,B]
    const TensorF32 * left  = node->src[1];  // [D,L,B]
    const TensorF32 * right = node->src[2];  // [D,L,B]
    TensorF32 * grad_left  = node->src[3];   // [D,L,B]
    TensorF32 * grad_right = node->src[4];   // [D,L,B]

    if (!grad_left || !grad_right) { tp->barrier_wait(); return; }

    const int64_t D = left->shape().dims[0];
    const int64_t L = left->shape().dims[1];
    const int64_t B = left->shape().dims[2];

    const float * gdata  = static_cast<const float*>(grad->data());
    const float * ldata  = static_cast<const float*>(left->data());
    const float * rdata  = static_cast<const float*>(right->data());
    float * gl_out = static_cast<float*>(grad_left->data());
    float * gr_out = static_cast<float*>(grad_right->data());

    // dL/dleft [d1,i,b] = sum_{d2,j} grad[(d1*D+d2), i, j, b]*right[d2,j,b]
    {
        const int64_t total = D * L * B;
        const int64_t per  = (total + p->nth - 1) / p->nth;
        const int64_t start = per * p->ith;
        const int64_t end   = (start + per < total) ? (start + per) : total;
        for (int64_t idx = start; idx < end; idx++) {
            int64_t tmp = idx;
            int64_t b   = tmp % B; tmp /= B;
            int64_t i   = tmp % L;
            int64_t d1  = tmp / L;
            float sum = 0.0f;
            for (int64_t j = 0; j < L; j++) {
                for (int64_t d2 = 0; d2 < D; d2++) {
                    const float gv = gdata[(((d1 * D + d2) * L + i) * L + j) * B + b];
                    const float rv = rdata[((d2 * L + j) * B + b)];
                    sum += gv * rv;
                }
            }
            gl_out[idx] = sum;
        }
    }
    tp->barrier_wait();

    // dL/dright [d2,j,b] = sum_{d1,i} grad[(d1*D+d2), i, j, b]*left[d1,i,b]
    {
        const int64_t total = D * L * B;
        const int64_t per  = (total + p->nth - 1) / p->nth;
        const int64_t start = per * p->ith;
        const int64_t end   = (start + per < total) ? (start + per) : total;
        for (int64_t idx = start; idx < end; idx++) {
            int64_t tmp = idx;
            int64_t b   = tmp % B; tmp /= B;
            int64_t j   = tmp % L;
            int64_t d2  = tmp / L;
            float sum = 0.0f;
            for (int64_t i = 0; i < L; i++) {
                for (int64_t d1 = 0; d1 < D; d1++) {
                    const float gv = gdata[(((d1 * D + d2) * L + i) * L + j) * B + b];
                    const float lv = ldata[((d1 * L + i) * B + b)];
                    sum += gv * lv;
                }
            }
            gr_out[idx] = sum;
        }
    }
    tp->barrier_wait();
}

// backward: dL/dleft 和 dL/dright 分别对 left 和 right 求导
void CPUBackend::kernel_tri_mul_back(TensorF32 * node, ComputeParams * p) {
    // grad from upstream
    const TensorF32 * grad = node->src[0];  // dL/ddst: [D, I, J, B]（ggml）
    const TensorF32 * left  = node->src[1]; // left
    const TensorF32 * right = node->src[2]; // right
    TensorF32 * grad_left  = node->src[3];  // dL/dleft（形状同 left）
    TensorF32 * grad_right = node->src[4];  // dL/dright（形状同 right）

    if (!grad_left || !grad_right) return;

    // ggml 布局：特征维最内
    const int64_t D = left->shape().dims[0];
    const int64_t B = left->shape().dims[3];

    float L;
    bool  outgoing;
    memcpy(&L,        node->op_params,      sizeof(float));
    memcpy(&outgoing, node->op_params + 4,  sizeof(bool));

    // I/J 的位置取决于 outgoing：
    //   outgoing: left [D,I,K,B] → I=left.dims[1]；right [D,J,K,B] → J=right.dims[1]
    //   incoming: left [D,K,I,B] → I=left.dims[2]；right [D,K,J,B] → J=right.dims[2]
    const int64_t I = outgoing ? left->shape().dims[1]  : left->shape().dims[2];
    const int64_t J = outgoing ? right->shape().dims[1] : right->shape().dims[2];

    const float inv_L = 1.0f / L;
    const float * grad_data = static_cast<const float*>(grad->data());
    const float * right_data = static_cast<const float*>(right->data());
    const float * left_data  = static_cast<const float*>(left->data());
    float * gleft_data  = static_cast<float*>(grad_left->data());
    float * gright_data = static_cast<float*>(grad_right->data());

    ThreadPool * tp = p->threadpool;

    if (outgoing) {
        // forward: dst[d,i,j,b] = (1/L)*sum_k left[d,i,k,b]*right[d,j,k,b]
        //   left [D,I,K,B], right [D,J,K,B], K = left.dims[2]
        const int64_t K = left->shape().dims[2];
        // dL/dleft[d,i,k,b] = (1/L)*sum_j grad[d,i,j,b]*right[d,j,k,b]
        {
            const int64_t total = D * I * K * B;
            const int64_t per  = (total + p->nth - 1) / p->nth;
            const int64_t start = per * p->ith;
            const int64_t end   = (start + per < total) ? (start + per) : total;
            for (int64_t idx = start; idx < end; idx++) {
                int64_t tmp = idx;
                int64_t b   = tmp % B; tmp /= B;
                int64_t k   = tmp % K; tmp /= K;
                int64_t i   = tmp % I;
                int64_t d   = tmp / I;
                float sum = 0.0f;
                for (int64_t j = 0; j < J; j++) {
                    const float gv = grad_data[((d * I + i) * J + j) * B + b];
                    const float rv = right_data[((d * J + j) * K + k) * B + b];
                    sum += gv * rv;
                }
                gleft_data[idx] = sum * inv_L;
            }
        }
        tp->barrier_wait();
        // dL/dright[d,j,k,b] = (1/L)*sum_i grad[d,i,j,b]*left[d,i,k,b]
        {
            const int64_t total = D * J * K * B;
            const int64_t per  = (total + p->nth - 1) / p->nth;
            const int64_t start = per * p->ith;
            const int64_t end   = (start + per < total) ? (start + per) : total;
            for (int64_t idx = start; idx < end; idx++) {
                int64_t tmp = idx;
                int64_t b   = tmp % B; tmp /= B;
                int64_t k   = tmp % K; tmp /= K;
                int64_t j   = tmp % J;
                int64_t d   = tmp / J;
                float sum = 0.0f;
                for (int64_t i = 0; i < I; i++) {
                    const float gv = grad_data[((d * I + i) * J + j) * B + b];
                    const float lv = left_data[((d * I + i) * K + k) * B + b];
                    sum += gv * lv;
                }
                gright_data[idx] = sum * inv_L;
            }
        }
        tp->barrier_wait();
    } else {
        // incoming: forward dst[d,i,j,b] = (1/L)*sum_k left[d,k,i,b]*right[d,k,j,b]
        //   left [D,K,I,B], right [D,K,J,B], K = left.dims[1]
        const int64_t K = left->shape().dims[1];
        // dL/dleft[d,k,i,b] = (1/L)*sum_j grad[d,i,j,b]*right[d,k,j,b]
        {
            const int64_t total = D * K * I * B;
            const int64_t per  = (total + p->nth - 1) / p->nth;
            const int64_t start = per * p->ith;
            const int64_t end   = (start + per < total) ? (start + per) : total;
            for (int64_t idx = start; idx < end; idx++) {
                int64_t tmp = idx;
                int64_t b   = tmp % B; tmp /= B;
                int64_t i   = tmp % I; tmp /= I;
                int64_t k   = tmp % K;
                int64_t d   = tmp / K;
                float sum = 0.0f;
                for (int64_t j = 0; j < J; j++) {
                    const float gv = grad_data[((d * I + i) * J + j) * B + b];
                    const float rv = right_data[((d * K + k) * J + j) * B + b];
                    sum += gv * rv;
                }
                gleft_data[idx] = sum * inv_L;
            }
        }
        tp->barrier_wait();
        // dL/dright[d,k,j,b] = (1/L)*sum_i grad[d,i,j,b]*left[d,k,i,b]
        {
            const int64_t total = D * K * J * B;
            const int64_t per  = (total + p->nth - 1) / p->nth;
            const int64_t start = per * p->ith;
            const int64_t end   = (start + per < total) ? (start + per) : total;
            for (int64_t idx = start; idx < end; idx++) {
                int64_t tmp = idx;
                int64_t b   = tmp % B; tmp /= B;
                int64_t j   = tmp % J; tmp /= J;
                int64_t k   = tmp % K;
                int64_t d   = tmp / K;
                float sum = 0.0f;
                for (int64_t i = 0; i < I; i++) {
                    const float gv = grad_data[((d * I + i) * J + j) * B + b];
                    const float lv = left_data[((d * K + k) * I + i) * B + b];
                    sum += gv * lv;
                }
                gright_data[idx] = sum * inv_L;
            }
        }
        tp->barrier_wait();
    }
}

// ===== softmax =====
void CPUBackend::kernel_softmax(TensorF32 * node, ComputeParams * p) {
    int D    = static_cast<int>(node->shape().dims[0]);
    int rows = static_cast<int>(node->numel() / D);
    int per  = (rows + p->nth - 1) / p->nth;
    int start = p->ith * per;
    int end   = std::min(start + per, rows);
    float * src = node->src[0]->data();
    float * dst = node->data();

    for (int r = start; r < end; r++) {
        float * sr = src + r * D, * dr = dst + r * D;
        // 标准两遍 softmax（bug 修复 2026-08-19）：
        //   原 online 版本漏加 d=0 项（sum 少算 exp(sr[0]-mx)），且 dr[0] 从不写入，
        //   对常数输入输出 1/(D-1)（如 0.0454=1/22），导致 loss 退化/异常。
        float mx = sr[0];
        for (int d = 1; d < D; d++) if (sr[d] > mx) mx = sr[d];
        float sum = 0;
        for (int d = 0; d < D; d++) sum += expf(sr[d] - mx);
        float inv = 1.0f / (sum + 1e-9f);
        for (int d = 0; d < D; d++) dr[d] = expf(sr[d] - mx) * inv;
    }
}

// ===== rms_norm =====
void CPUBackend::kernel_rms_norm(TensorF32 * node, ComputeParams * p) {
    int D    = static_cast<int>(node->shape().dims[0]);
    int rows = static_cast<int>(node->numel() / D);
    int per  = (rows + p->nth - 1) / p->nth;
    int start = p->ith * per, end = std::min(start + per, rows);
    float * src = node->src[0]->data(), * dst = node->data();
    float eps   = reinterpret_cast<float &>(node->op_params[0]);

    for (int r = start; r < end; r++) {
        float * sr = src + r * D, * dr = dst + r * D;
        float ss = 0;
        for (int d = 0; d < D; d++) ss += sr[d] * sr[d];
        float inv = 1.0f / sqrtf(ss / D + eps);
        for (int d = 0; d < D; d++) dr[d] = sr[d] * inv;
    }
}

// layer norm (OP_NORM) kernel implementation
void CPUBackend::kernel_norm(TensorF32 * node, ComputeParams * p) {
    int D    = static_cast<int>(node->shape().dims[0]);
    int rows = static_cast<int>(node->numel() / D);
    int per  = (rows + p->nth - 1) / p->nth;
    int start = p->ith * per, end = std::min(start + per, rows);
    float * src = node->src[0]->data(), * dst = node->data();
    float eps   = reinterpret_cast<float&>(node->op_params[0]);
    float * mean_buf = node->src[1] ? node->src[1]->data() : nullptr;
    float * rstd_buf = node->src[2] ? node->src[2]->data() : nullptr;

    for (int r = start; r < end; r++) {
        float * sr = src + r * D, * dr = dst + r * D;
        float mean = 0.0f;
        for (int d = 0; d < D; d++) mean += sr[d];
        mean /= D;

        float var = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = sr[d] - mean;
            var += diff * diff;
        }
        var /= D;

        float inv_std = 1.0f / sqrtf(var + eps);
        for (int d = 0; d < D; d++) dr[d] = (sr[d] - mean) * inv_std;

        // int rows = a->numel() / D;
        // r = start, end 
        // range check?
        // cache for backward
        if (mean_buf) mean_buf[r] = mean;
        if (rstd_buf) rstd_buf[r] = inv_std;
    }
}

// layer norm backward (OP_NORM_BACK) kernel
void CPUBackend::kernel_norm_back(TensorF32 * node, ComputeParams * p) {
    // dL_dx = rstd/D * (D * dL_dy - sum(dL_dy) - y_norm * sum(dL_dy * y_norm))
    // src[0] = dL_dy (upstream gradient, aka dout)
    // src[1] = x     (original input, aka inp)
    // src[2] = mean  (cached during forward)
    // src[3] = rstd  (cached during forward)
    int C    = reinterpret_cast<int&>(node->op_params[0]);   // feature dim
    int rows = reinterpret_cast<int&>(node->op_params[1]);   // batch * seq

    float * dout = node->src[0]->data();   // dL_dy
    float * inp  = node->src[1]->data();   // x (原始输入)
    float * mean = node->src[2]->data();
    float * rstd = node->src[3]->data();
    float * dinp = node->data();           // dL_dx

    for (int t = 0; t < rows; t++) {
        // 定位 dout / inp / dinp
        float * dout_bt  = dout + t * C;
        float * inp_bt   = inp  + t * C;
        float * dinp_bt  = dinp + t * C;
        float   mean_bt  = mean[t];
        float   rstd_bt  = rstd[t];

        // norm_bt[i] = (inp[i] - mean_bt) * rstd_bt
        // dnorm[i] = dout[i] (无 weight 的 layer norm)
        float dnorm_mean      = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = 0; i < C; i++) {
            float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i  = dout_bt[i];
            dnorm_mean      += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean      /= C;
        dnorm_norm_mean /= C;

        for (int i = 0; i < C; i++) {
            float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i  = dout_bt[i];
            // dinp = (dnorm - dnorm_mean - norm_bti * dnorm_norm_mean) * rstd_bt
            float dval = dnorm_i - dnorm_mean - norm_bti * dnorm_norm_mean;
            dinp_bt[i] = dval * rstd_bt;
        }
    }
}

// softmax backward (OP_SOFT_MAX_BACK) kernel
void CPUBackend::kernel_softmax_back(TensorF32 * node, ComputeParams * p) {
    // dL/dx_i = y_i * (dL/dy_i - sum_j(y_j * dL/dy_j))
    // src[0] = dL/dy (upstream gradient)
    // src[1] = y     (softmax forward output)
    int D    = reinterpret_cast<int&>(node->op_params[0]);
    int rows = reinterpret_cast<int&>(node->op_params[1]);

    int per   = (rows + p->nth - 1) / p->nth;
    int start = p->ith * per;
    int end   = std::min(start + per, rows);

    const float * grad   = node->src[0]->data();
    const float * output = node->src[1]->data();
    float *       dst    = node->data();

    for (int r = start; r < end; r++) {
        const float * gr = grad   + r * D;
        const float * yr = output + r * D;
        float *       dr = dst    + r * D;

        // Step 1: compute sum_j(y_j * dL/dy_j)
        float dgf_dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dgf_dot += yr[d] * gr[d];
        }

        // Step 2: dL/dx_i = y_i * (dL/dy_i - dgf_dot)
        for (int d = 0; d < D; d++) {
            dr[d] = yr[d] * (gr[d] - dgf_dot);
        }
    }
}

// ===== unary =====
void CPUBackend::kernel_silu(TensorF32 * node) {
    float * s = node->src[0]->data(), * d = node->data();
    for (int64_t i = 0; i < node->numel(); i++)
        d[i] = s[i] / (1.0f + expf(-s[i]));
}
void CPUBackend::kernel_gelu(TensorF32 * node) { (void)node; }
void CPUBackend::kernel_relu(TensorF32 * node) { (void)node; }

void CPUBackend::kernel_dup(TensorF32 * node) {
    std::memcpy(node->data(), node->src[0]->data(), node->numel() * sizeof(float));
}

// ===== cpy (整块拷贝): dst = src0 =====
// 用于 reshape/view/cont/cpy：元素内存顺序不变，dst 是新分配内存，必须整块 memcpy。
// 方案 B 下 Tensor 无 nb[]，这些 shape op 不能像 ggml 那样共享 data 指针。
void CPUBackend::kernel_cpy(TensorF32 * node, ComputeParams * p) {
    if (p->ith != 0) {
        p->threadpool->barrier_wait();
        return;
    }
    // 零拷贝 view：view 共享源数据（view_src 已置），不拷贝、不写独立 buffer。
    // 直接把 node 的 data()/buffer_/buffer_offs_ 解析为 src0 的（源已计算时 data 有效）。
    if (node->op == OP_VIEW) {
        TensorF32* src_t = node->src[0];
        if (src_t && src_t->data()) {
            node->bind_data(src_t->data());
            node->buffer_      = src_t->buffer_;
            node->buffer_offs_ = src_t->buffer_offs_;
        }
        p->threadpool->barrier_wait();
        return;
    }
    // 缓冲感知拷贝：CPU/CUDA 混训时，src 或 dst 可能在 device buffer。
    // 裸 memcpy 会把 device 指针当 host 读 → 段错误。统一经 buffer get_tensor/set_tensor。
    TensorF32 * src_t = node->src[0];
    const size_t bytes = static_cast<size_t>(node->numel()) * sizeof(float);
    const bool src_dev = src_t->buffer_ && !src_t->buffer_->is_host();
    const bool dst_dev = node->buffer_ && !node->buffer_->is_host();

    if (src_dev && dst_dev) {
        std::vector<float> tmp(static_cast<size_t>(node->numel()));
        src_t->buffer_->get_tensor(src_t, tmp.data(), src_t->buffer_offs_, bytes);   // D2H
        node->buffer_->set_tensor(node, tmp.data(), node->buffer_offs_, bytes);      // H2D
    } else if (src_dev) {
        src_t->buffer_->get_tensor(src_t, node->data(), src_t->buffer_offs_, bytes); // D2H
    } else if (dst_dev) {
        node->buffer_->set_tensor(node, src_t->data(), node->buffer_offs_, bytes);   // H2D
    } else {
        std::memcpy(node->data(), src_t->data(), bytes);
    }
    p->threadpool->barrier_wait();
}

// ===== permute (维度重排, 实际搬数据) =====
// 方案 B：Tensor 无 nb，按 op_params 的 dims 映射把数据重排到连续行主序。
//   src 各维大小 sd[i]；dst 各维大小 dd[i]。
//   dst 的第 p 维来自 src 的第 dims[p] 维（op_params 存的映射）。
//   对 dst 每个连续元素 idx，反解 dst 坐标(j0,j1,j2,j3) → src 坐标 i_{dims[p]}=j_p
//   → src 线性偏移(行主序 dims[0]最内)，写入 dst[idx]。
// 用于 OP_PERMUTE / OP_TRANSPOSE（transpose 构造器已把交换映射写入 op_params）。
// 并行安全：各线程写 dst 不同 idx，无 data race。
void CPUBackend::kernel_permute(TensorF32 * node, ComputeParams * p) {
    const TensorF32 * a = node->src[0];
    TensorF32       * dst = node;

    const int ndim = a->shape().ndim();
    // 从 op_params 读 dims 映射（permute/transpose 构造器已写入）
    int dims[4] = {0, 1, 2, 3};
    for (int i = 0; i < ndim && i < 4; i++) dims[i] = (int)node->op_params[i];
    // 防御：op_params 未初始化（垃圾）会导致 i[dims[p]] 越界写 → SIGSEGV。
    for (int i = 0; i < ndim && i < 4; i++) {
        if (dims[i] < 0 || dims[i] >= ndim) {
            fprintf(stderr, "[permute-FAIL] node=%p ndim=%d dims=[%d %d %d %d] "
                            "op_params=[%lld %lld %lld %lld] numel=%lld\n",
                    (void*)node, ndim, dims[0], dims[1], dims[2], dims[3],
                    (long long)node->op_params[0], (long long)node->op_params[1],
                    (long long)node->op_params[2], (long long)node->op_params[3],
                    (long long)node->numel());
            return;  // 跳过非法 permute，避免越界写
        }
    }

    int64_t sd[4] = {1, 1, 1, 1};   // src 各维大小
    int64_t dd[4] = {1, 1, 1, 1};   // dst 各维大小
    for (int i = 0; i < ndim && i < 4; i++) { sd[i] = a->shape().dims[i]; dd[i] = dst->shape().dims[i]; }

    const float * s_data = a->data();
    float       * d_data = dst->data();

    const int64_t total = dst->numel();
    const int64_t per   = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    for (int64_t idx = start; idx < end; idx++) {
        // 反解 dst 坐标 (j0,j1,j2,j3)，dims[0]=最内维
        int64_t t = idx;
        const int64_t j0 = t % dd[0]; t /= dd[0];
        const int64_t j1 = t % dd[1]; t /= dd[1];
        const int64_t j2 = t % dd[2]; t /= dd[2];
        const int64_t j3 = t;

        // src 坐标: i_{dims[p]} = j_p
        int64_t i[4] = {0, 0, 0, 0};
        const int64_t jv[4] = {j0, j1, j2, j3};
        for (int p = 0; p < ndim && p < 4; p++) i[dims[p]] = jv[p];

        // src 线性偏移（行主序 dims[0] 最内）
        const int64_t off = ((i[3] * sd[2] + i[2]) * sd[1] + i[1]) * sd[0] + i[0];
        d_data[idx] = s_data[off];
    }
}

// ===== sqr (逐元素平方): dst[i] = src[i] * src[i] =====
// 独立 op（非 unary）。按线程分片并行。ggml 中该 op 用 n_tasks=1。
void CPUBackend::kernel_sqr(TensorF32 * node, ComputeParams * p) {
    const float * src = node->src[0]->data();
    float       * dst = node->data();

    const int64_t total = node->numel();
    const int64_t per   = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    for (int64_t i = start; i < end; i++) dst[i] = src[i] * src[i];
}

// ===== scale (标量乘法): dst = src0 * s =====
// s 以 float 位模式存于 op_params[0]（由 scale() 构造器写入）
void CPUBackend::kernel_scale(TensorF32 * node, ComputeParams * p) {
    const float s = reinterpret_cast<const float&>(node->op_params[0]);
    const float * src = node->src[0]->data();
    float       * dst = node->data();

    const int64_t total = node->numel();
    const int64_t per   = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    // 开发诊断（GRAPH_DEBUG_LOSS）：标量 scale（total 链），打印输入值与 buffer 地址，定位 nan 来源
    if (p->ith == 0 && total == 1 && getenv("GRAPH_DEBUG_LOSS")) {
        fprintf(stderr, "[scale] numel=1 src0_val=%f src0_addr=%p s=%f dst_addr=%p\n",
                src[0], (const void*)src, s, (const void*)dst);
    }

    for (int64_t i = start; i < end; i++) dst[i] = src[i] * s;
}

// ===== add1 (加标量): dst = src0 + b =====
// b 是标量张量 node->src[1]（读其 data()[0]）
void CPUBackend::kernel_add1(TensorF32 * node, ComputeParams * p) {
    const float b    = node->src[1]->data()[0];
    const float * src = node->src[0]->data();
    float       * dst = node->data();

    const int64_t total = node->numel();
    const int64_t per   = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    for (int64_t i = start; i < end; i++) dst[i] = src[i] + b;
}

// ===== sum (全元素求和): dst[0] = Σ all src0 =====
// 输出为标量 (1,)。不同线程对同一 dst[0] 累加会 data race，
// 故在 thread 0 上串行 reduce（与 FAPE/get_rows_back 的 thread-0-only 模式一致）。
void CPUBackend::kernel_sum(TensorF32 * node, ComputeParams * p) {
    if (p->ith != 0) {
        p->threadpool->barrier_wait();
        return;
    }
    const float * src = node->src[0]->data();
    const int64_t n   = node->src[0]->numel();
    float s = 0.0f;
    for (int64_t i = 0; i < n; i++) s += src[i];
    node->data()[0] = s;
    p->threadpool->barrier_wait();
}

// ===== sum_rows (沿最内维 dims[0] 求和) =====
// 语义: dst[0, i1, i2, i3] = Σ_{i0} src[i0, i1, i2, i3]，输出形状 {1, dims[1], dims[2], dims[3]}。
// 用于 Q mean 等"沿最内维规约、保留其余维度"的场景（graph 布局 dims[0]=最内维）。
// 并行安全: 每个 dst 元素独立对一段连续的最内维归约，各线程处理互不重叠的 dst 子集。
void CPUBackend::kernel_sum_rows(TensorF32 * node, ComputeParams * p) {
    const TensorF32 * src = node->src[0];
    TensorF32       * dst = node;

    const int64_t ne0 = src->shape().dims[0];
    const int64_t ne1 = (src->shape().ndim() > 1) ? src->shape().dims[1] : 1;
    const int64_t ne2 = (src->shape().ndim() > 2) ? src->shape().dims[2] : 1;
    const int64_t ne3 = (src->shape().ndim() > 3) ? src->shape().dims[3] : 1;

    const float * src_data = src->data();
    float       * dst_data = dst->data();

    // 独立输出元素数（不含被归约的最内维 ne0）
    const int64_t total = ne1 * ne2 * ne3;
    const int64_t per   = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    for (int64_t idx = start; idx < end; idx++) {
        int64_t t = idx;
        const int64_t i1 = t % ne1; t /= ne1;
        const int64_t i2 = t % ne2; t /= ne2;
        const int64_t i3 = t;

        // src 行基地址: (i3, i2, i1) 起始的连续 ne0 个元素
        const float * row = src_data + ((i3 * ne2 + i2) * ne1 + i1) * ne0;
        float s = 0.0f;
        for (int64_t i0 = 0; i0 < ne0; i0++) s += row[i0];
        dst_data[idx] = s;
    }

    p->threadpool->barrier_wait();
}

// ===== mean (全元素平均): dst[0] = Σ src0 / n =====
// 输出为标量 (1,)。同样 thread 0 串行 reduce。
void CPUBackend::kernel_mean(TensorF32 * node, ComputeParams * p) {
    if (p->ith != 0) {
        p->threadpool->barrier_wait();
        return;
    }
    const float * src = node->src[0]->data();
    const int64_t n   = node->src[0]->numel();
    float s = 0.0f;
    for (int64_t i = 0; i < n; i++) s += src[i];
    node->data()[0] = (n > 0) ? (s / static_cast<float>(n)) : 0.0f;
    p->threadpool->barrier_wait();
}

// ===== max_all (全局最大值，标量) =====
// 用于 softmax 数值稳定：e' = e - max_all(e)，避免 exp 溢出。
void CPUBackend::kernel_max_all(TensorF32 * node, ComputeParams * p) {
    if (p->ith != 0) {
        p->threadpool->barrier_wait();
        return;
    }
    const float * src = node->src[0]->data();
    const int64_t n   = node->src[0]->numel();
    float m = -INFINITY;
    for (int64_t i = 0; i < n; i++) {
        float v = src[i];
        if (!std::isnan(v) && v > m) m = v;
    }
    node->data()[0] = (n > 0) ? m : 0.0f;
    p->threadpool->barrier_wait();
}

// ===== relu 反向：d(relu(x))/dx = grad * step(x) = grad * (x > 0) =====
void CPUBackend::kernel_relu_back(TensorF32 * node, ComputeParams * p) {
    const TensorF32 * grad = node->src[0];
    const TensorF32 * x    = node->src[1];
    TensorF32       * dst  = node;
    if (!grad || !x || !grad->data() || !x->data() || !dst->data()) return;
    const int64_t n = dst->numel();
    const int64_t per = (n + p->nth - 1) / p->nth;
    const int64_t i_lo = per * p->ith;
    const int64_t i_hi = (i_lo + per < n) ? (i_lo + per) : n;
    const float * g = grad->data();
    const float * xd = x->data();
    float * d = dst->data();
    for (int64_t i = i_lo; i < i_hi; ++i) {
        d[i] = (xd[i] > 0.0f) ? g[i] : 0.0f;
    }
}

// ===== repeat (广播) =====
// 语义: 将 src0 广播到 node(=src1) 的形状。src0 的尾部维与 dst 尾部维对齐，
//       src0 维度数少于 dst 时，缺失的前导维按 1 处理（即广播）。
//       src0 某维 == dst 该维 或 src0 该维 == 1 时合法。
// 使用场景: sum/mean 反向 add1_or_set 首次设置 grad 时，把标量 grad 广播成 src 形状。
void CPUBackend::kernel_repeat(TensorF32 * node, ComputeParams * p) {
    const TensorF32 * src0 = node->src[0];
    TensorF32       * dst  = node;

    const int nd_src = src0->shape().ndim();
    const int nd_dst = dst->shape().ndim();
    if (getenv("GRAPH_DEBUG_REPEAT")) {
        const int64_t t = (int64_t)dst->shape().numel();
        if (t > 10000000) {   // 只打印超大 repeat（numel>1000万）
            std::fprintf(stderr, "[REPEAT-BIG] dst={%lld,%lld,%lld,%lld} numel=%lld src={%lld,%lld,%lld,%lld} numel=%lld\n",
                (long long)(nd_dst>0?dst->shape().dims[0]:-1),
                (long long)(nd_dst>1?dst->shape().dims[1]:-1),
                (long long)(nd_dst>2?dst->shape().dims[2]:-1),
                (long long)(nd_dst>3?dst->shape().dims[3]:-1),
                (long long)t,
                (long long)(nd_src>0?src0->shape().dims[0]:-1),
                (long long)(nd_src>1?src0->shape().dims[1]:-1),
                (long long)(nd_src>2?src0->shape().dims[2]:-1),
                (long long)(nd_src>3?src0->shape().dims[3]:-1),
                (long long)src0->shape().numel());
        }
    }

    const int64_t ne0[4] = {
        src0->shape().dims[0],
        (nd_src > 1) ? src0->shape().dims[1] : 1,
        (nd_src > 2) ? src0->shape().dims[2] : 1,
        (nd_src > 3) ? src0->shape().dims[3] : 1
    };
    const int64_t nd0 = dst->shape().dims[0];
    const int64_t nd1 = (nd_dst > 1) ? dst->shape().dims[1] : 1;
    const int64_t nd2 = (nd_dst > 2) ? dst->shape().dims[2] : 1;
    const int64_t nd3 = (nd_dst > 3) ? dst->shape().dims[3] : 1;

    const float * src_data = src0->data();
    float       * dst_data = dst->data();

    const int64_t total = nd0 * nd1 * nd2 * nd3;
    const int64_t per   = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    for (int64_t idx = start; idx < end; idx++) {
        int64_t t = idx;
        const int64_t i3 = t % nd3; t /= nd3;
        const int64_t i2 = t % nd2; t /= nd2;
        const int64_t i1 = t % nd1; t /= nd1;
        const int64_t i0 = t;

        // 尾部对齐的 src0 索引（缺失维取模 1 = 0）
        const int64_t s0 = i0 % ne0[0];
        const int64_t s1 = i1 % ne0[1];
        const int64_t s2 = i2 % ne0[2];
        const int64_t s3 = i3 % ne0[3];

        dst_data[idx] = src_data[((s3 * ne0[2] + s2) * ne0[1] + s1) * ne0[0] + s0];
    }
}

// ===== repeat_back (归约，repeat 的逆操作) =====
// 语义: 把 src0(大形状，被 repeat 广播的结果/梯度) 归约回 node(小形状)。
//       dst[j] = Σ_{k} src0[j + k*dd] ，其中对每个维度 d：
//         - dd[d] = dst 该维大小，sd[d] = src0 该维大小（尾部对齐），
//         - k 遍历 [0, sd[d]/dd[d])。
//       src0 前导维数多于 dst 时，src0 多出的前导维是纯广播维（dst 对应维=1），全部累加。
// 使用场景: OP_ADD/OP_MUL/OP_REPEAT 反向中 repeat_back(grad, src)。
// 并行安全: 每个 dst 元素独立归约，各线程处理互不重叠的 dst 子集，无 data race。
void CPUBackend::kernel_repeat_back(TensorF32 * node, ComputeParams * p) {
    const TensorF32 * src0 = node->src[0];
    TensorF32       * dst  = node;

    const int nd_src = src0->shape().ndim();
    const int nd_dst = dst->shape().ndim();

    // 尾部对齐维度。dst 的维 d (0=最内维) 对应 src0 的维 (d + off)，off = nd_src - nd_dst。
    // dst 缺失前导维（d >= nd_dst）按 1 处理；src0 缺失前导维（d+off >= nd_src）按 1 处理。
    const int off = nd_src - nd_dst;

    int64_t dd[4] = {1, 1, 1, 1};   // dst 各维
    int64_t sd[4] = {1, 1, 1, 1};   // src0 各维
    for (int d = 0; d < nd_dst && d < 4; d++) dd[d] = dst->shape().dims[d];
    for (int d = 0; d < nd_src && d < 4; d++) sd[d] = src0->shape().dims[d];

    // 归约次数: 每维 src0 大小 / dst 大小（尾部对齐）。sd/dd 应为非负整数。
    int64_t rd[4] = {1, 1, 1, 1};
    for (int d = 0; d < 4; d++) {
        const int s_idx = d + off;
        const int64_t sdim = (s_idx >= 0 && s_idx < 4) ? sd[s_idx] : 1;
        const int64_t ddim = dd[d];
        rd[d] = (ddim > 0) ? (sdim / ddim) : 1;
    }

    const float * src_data = src0->data();
    float       * dst_data = dst->data();

    const int64_t total = dd[0] * dd[1] * dd[2] * dd[3];
    const int64_t per   = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    for (int64_t idx = start; idx < end; idx++) {
        int64_t t = idx;
        const int64_t j0 = t % dd[0]; t /= dd[0];
        const int64_t j1 = t % dd[1]; t /= dd[1];
        const int64_t j2 = t % dd[2]; t /= dd[2];
        const int64_t j3 = t;

        float sum = 0.0f;
        for (int64_t k0 = 0; k0 < rd[0]; k0++) {
            const int64_t s0 = j0 + k0 * dd[0];
            for (int64_t k1 = 0; k1 < rd[1]; k1++) {
                const int64_t s1 = j1 + k1 * dd[1];
                for (int64_t k2 = 0; k2 < rd[2]; k2++) {
                    const int64_t s2 = j2 + k2 * dd[2];
                    for (int64_t k3 = 0; k3 < rd[3]; k3++) {
                        const int64_t s3 = j3 + k3 * dd[3];
                        sum += src_data[((s3 * sd[2] + s2) * sd[1] + s1) * sd[0] + s0];
                    }
                }
            }
        }
        dst_data[idx] = sum;
    }
}

// ===== concat =====
// 对标 ggml_compute_forward_concat_f32（项目仅 F32 连续数据，故无需 block_size / nb stride 处理）
// 支持任意数量 src：沿 op_params[0] 指定的 dim 拼接。
// 内存布局 row-major，ne0 最内维。每个 src 在非拼接维上与 dst 同形状或为广播/截断，
// 这里按 src 自身形状映射（当 src 维度小于 dst 时越界元素跳过）。
void CPUBackend::kernel_concat(TensorF32 * node, ComputeParams * p) {
    const int dim = node->op_params[0];
    const int nd  = node->shape().ndim();
    if (dim < 0 || dim >= 4) { p->threadpool->ec = Status::NOT_SUPPORTED; return; }

    const int64_t ne0 = nd > 0 ? node->shape().dims[0] : 1;
    const int64_t ne1 = nd > 1 ? node->shape().dims[1] : 1;
    const int64_t ne2 = nd > 2 ? node->shape().dims[2] : 1;
    const int64_t ne3 = nd > 3 ? node->shape().dims[3] : 1;

    // src 为固定大小数组(GGML_MAX_SRC)，仅统计非空输入，空位跳过
    int n_src = 0;
    std::array<const TensorF32*, GGML_MAX_SRC> srcs{};
    for (int s = 0; s < GGML_MAX_SRC; s++) {
        if (node->src[s]) srcs[n_src++] = node->src[s];
    }
    // 单源 concat 是 identity（GMABSE3 对单度 fiber 的 concat_ptr(k_nodes,0) 会只有 1 个源，
    //    若直接 NOT_SUPPORTED 会报错）。n_src==1 时 dst 与 src 同形状，直接拷贝。
    //    （本会话 SE3 测试 `node#269 op=22 单源` 即此根因）
    if (n_src < 1) { p->threadpool->ec = Status::NOT_SUPPORTED; return; }
    if (n_src == 1) {
        const TensorF32* src0 = srcs[0];
        if (src0 && src0->data() && node->data()) {
            std::memcpy(node->data(), src0->data(), node->nbytes());
        }
        return;
    }

    // 各 src 在 dim 维的长度与累积起点
    std::vector<int64_t> len(n_src), start(n_src, 0);
    for (int s = 0; s < n_src; s++) len[s] = srcs[s]->shape().dims[dim];
    for (int s = 1; s < n_src; s++) start[s] = start[s - 1] + len[s - 1];

    float * d = node->data();

    const int64_t total = ne0 * ne1 * ne2 * ne3;
    const int64_t per   = (total + p->nth - 1) / p->nth;
    const int64_t i_lo  = per * p->ith;
    const int64_t i_hi  = (i_lo + per < total) ? (i_lo + per) : total;

    for (int64_t idx = i_lo; idx < i_hi; idx++) {
        int64_t t  = idx;
        const int64_t i0 = t % ne0; t /= ne0;
        const int64_t i1 = t % ne1; t /= ne1;
        const int64_t i2 = t % ne2; t /= ne2;
        const int64_t i3 = t;

        // dim 维全局索引（决定元素所属 src 及 src 内索引）
        int64_t gd;
        switch (dim) {
            case 0:  gd = i0; break;
            case 1:  gd = i1; break;
            case 2:  gd = i2; break;
            default: gd = i3; break;
        }

        // 线性定位所属 src
        int s = 0;
        for (int k = 0; k < n_src; k++) {
            if (gd < start[k] + len[k]) { s = k; break; }
        }
        const int64_t local = gd - start[s];

        const TensorF32* src = srcs[s];
        const int64_t s0 = src->shape().ndim() > 0 ? src->shape().dims[0] : 1;
        const int64_t s1 = src->shape().ndim() > 1 ? src->shape().dims[1] : 1;
        const int64_t s2 = src->shape().ndim() > 2 ? src->shape().dims[2] : 1;
        const int64_t s3 = src->shape().ndim() > 3 ? src->shape().dims[3] : 1;

        const int64_t a0 = (dim == 0) ? local : i0;
        const int64_t a1 = (dim == 1) ? local : i1;
        const int64_t a2 = (dim == 2) ? local : i2;
        const int64_t a3 = (dim == 3) ? local : i3;

        if (a0 >= s0 || a1 >= s1 || a2 >= s2 || a3 >= s3) continue;

        d[idx] = src->data()[((a3 * s2 + a2) * s1 + a1) * s0 + a0];
    }
}

// ===== concat 反向：从 concat 输出梯度中切出某 src 的梯度段 =====
// 语义: concat_back(grad, src, dim, offset)
//   src[0]=grad：concat 输出梯度（形状在 dim 上 = 各 src 之和）
//   src[1]=src：参考（目标形状 = 该 src 的形状）
//   op_params[0]=dim, op_params[1]=offset（该 src 在 dim 上的起始偏移）
// dst(node)：形状与 src 相同，取 grad 的 [offset, offset+dim_len) 段。
// 原理：ggml 行主序 dims[0] 最内，dim 维 stride = prod(dims[0..dim-1])，
//       dst 线性索引 k 在 grad 中 = k + offset*stride（仅 dim 维索引平移）。
void CPUBackend::kernel_concat_back(TensorF32 * node, ComputeParams * p) {
    const TensorF32 * grad   = node->src[0];
    const TensorF32 * src    = node->src[1];
    TensorF32       * dst    = node;
    if (!grad || !src || !grad->data() || !dst->data()) return;
    const int64_t dim    = node->op_params[0];
    const int64_t offset = node->op_params[1];

    int64_t stride = 1;
    for (int64_t d = 0; d < dim && d < grad->shape().ndim(); ++d) stride *= grad->shape().dims[d];

    const int64_t total = dst->numel();
    const int64_t per   = (total + p->nth - 1) / p->nth;
    const int64_t i_lo  = per * p->ith;
    const int64_t i_hi  = (i_lo + per < total) ? (i_lo + per) : total;
    const float * g = grad->data();
    float * d = dst->data();
    for (int64_t k = i_lo; k < i_hi; ++k) {
        d[k] = g[k + offset * stride];
    }
}

// ===== set_rows (scatter 写入指定行) =====
// 语义: set_rows(a, b, c)
//   a(src[0]): 目标张量 (N, M)，含已有值
//   b(src[1]): 行索引 (K,)（float 编码的整数）
//   c(src[2]): 值源 (K, M)，每行对应一个要写入的行
// dst(node): (N, M) — 先整体拷贝 a（保留未覆盖行），再把 c 的第 k 行覆写到 b[k] 指定的行。
// 对应 ggml set_rows 的 scatter 语义（但 PPML 版本 a 为已含值的目标，仅覆盖 b 指定行）。
void CPUBackend::kernel_set_rows(TensorF32 * node, ComputeParams * p) {
    const TensorF32 * a = node->src[0];   // 目标 (N, M)
    const TensorF32 * b = node->src[1];   // 行索引 (K,)
    const TensorF32 * c = node->src[2];   // 值源 (K, M)
    TensorF32       * dst = node;         // (N, M)

    const int64_t N = a->shape().dims[0];   // 行数
    const int64_t M = a->shape().dims[1];   // 列数（每行长度）
    const int64_t K = b->shape().dims[0];   // 索引数

    const float * a_data = static_cast<const float*>(a->data());
    const float * b_data = static_cast<const float*>(b->data());
    const float * c_data = static_cast<const float*>(c->data());
    float       * d_data = static_cast<float*>(dst->data());

    // ===== 安全版：thread 0 串行完成全部（拷贝 + 查重 + scatter）=====
    // 全部在 thread 0 上执行，其他线程 barrier 后返回。绝对无 data race，
    // 且不写全局 ThreadPool::ec，避免跨 graph 复用线程池时污染后续节点状态。
    if (p->ith != 0) {
        p->threadpool->barrier_wait();
        return;
    }

    // 1) 拷贝 a 全量 → dst（保留未覆盖行）
    std::memcpy(d_data, a_data, N * M * sizeof(float));

    // 2) 预扫 b 强制"无重复索引"
    //    set_rows 语义是"每行被 scatter 一次"；重复索引会导致结果未定义。
    //    检测到重复 → 跳过 scatter，仅保留 a 的拷贝（不设全局错误状态）。
    bool dup = false;
    std::vector<uint8_t> seen(static_cast<size_t>(N), 0);
    for (int64_t k = 0; k < K; k++) {
        const int64_t i1 = static_cast<int64_t>(b_data[k]);
        if (i1 < 0 || i1 >= N) continue;   // 越界索引：clamp 阶段处理
        if (seen[static_cast<size_t>(i1)]) { dup = true; break; }
        seen[static_cast<size_t>(i1)] = 1;
    }

    // 3) 无重复才执行 scatter
    if (!dup) {
        for (int64_t k = 0; k < K; k++) {
            int64_t i1 = static_cast<int64_t>(b_data[k]);
            if (i1 < 0) i1 = 0;
            if (i1 >= N) i1 = N - 1;
            std::memcpy(d_data + i1 * M, c_data + k * M, M * sizeof(float));
        }
    }

    p->threadpool->barrier_wait();
}

// ===== get_rows (embedding 查表前向) =====
// 前向: y = W[idx], 即按 idx 从 W 中取行
// src0: W (N, M) 权重表; src1: idx (K,) 行索引 (float-encoded ints)
// dst:  y (K, M)  — 逐行拷贝 W[idx[k]] → y[k]
void CPUBackend::kernel_get_rows(TensorF32 * node, ComputeParams * p) {
    const TensorF32 * W   = node->src[0];
    const TensorF32 * idx = node->src[1];
    TensorF32       * dst = node;

    const int64_t N = W->shape().dims[0];      // 权重表行数 (= 词表大小)
    const int64_t M = W->shape().dims[1];      // 每行长度 (= 嵌入维度)
    const int64_t K = idx->shape().dims[0];    // 索引数量

    const float * W_data   = static_cast<const float*>(W->data());
    const float * idx_data = static_cast<const float*>(idx->data());
    float       * d_data   = static_cast<float*>(dst->data());

    const int64_t total = K;
    const int64_t per   = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    for (int64_t k = start; k < end; k++) {
        int64_t i = static_cast<int64_t>(idx_data[k]);
        if (i < 0) i = 0;
        if (i >= N) i = N - 1;
        // 逐元素拷贝第 i 行到输出第 k 行 (row-major)
        std::memcpy(d_data + k * M, W_data + i * M, M * sizeof(float));
    }
}

// ===== get_rows_back (embedding 查表反向) =====
// 前向: y = W[idx], 反向 dL/dW[i,:] = sum_{k: idx[k]==i} dy[k,:]
// src0: dy (K, M) upstream gradient
// src1: idx (K,) row indices (float-encoded ints)
// src2: W (N, M) 权重表 (仅用于取形状 N, M)
// dst:  dW (N, M) — 先清零，再按 idx 散点累加
void CPUBackend::kernel_get_rows_back(TensorF32 * node, ComputeParams * p) {
    const TensorF32 * dy  = node->src[0];
    const TensorF32 * idx = node->src[1];
    const TensorF32 * W   = node->src[2];
    TensorF32       * dst = node;

    const int64_t N = W->shape().dims[0];      // 权重表行数 (= 词表大小)
    const int64_t M = W->shape().dims[1];      // 每行长度 (= 嵌入维度)
    const int64_t K = idx->shape().dims[0];    // 索引数量

    const float * dy_data   = static_cast<const float*>(dy->data());
    const float * idx_data  = static_cast<const float*>(idx->data());
    float       * dW_data   = static_cast<float*>(dst->data());

    // ===== 仅在 thread 0 上执行清零 + 散点累加 =====
    // 注意: 同一 token (i) 可能出现在多个 k, 必须 += 累加而非覆盖;
    //       不同 k 可能映射到同一 i, 若跨线程并行会 data race, 故串行在 thread 0 完成.
    if (p->ith == 0) {
        std::memset(dW_data, 0, N * M * sizeof(float));

        for (int64_t k = 0; k < K; k++) {
            int64_t i = static_cast<int64_t>(idx_data[k]);
            if (i < 0 || i >= N) continue;   // 越界索引丢弃
            for (int64_t d = 0; d < M; d++) {
                dW_data[i * M + d] += dy_data[k * M + d];
            }
        }
    }

    p->threadpool->barrier_wait();
}

// ===== edge_gather_rows (SE3 消息传递：按边源节点索引取行) =====
// 前向: dst[e,:] = node_feat[src_idx[e],:]
// ggml 布局：node_feat dims=[C, N]，src_idx (E,)，dst dims=[C, E]
// src0: node_feat (N, C) 节点特征；src1: src_idx (E,)；dst: (E, C)
// 语义与 get_rows 一致（gather），独立 op 便于 SE3 消息阶段显式表达与反向散点。
void CPUBackend::kernel_edge_gather_rows(TensorF32 * node, ComputeParams * p) {
    const TensorF32 * node_feat = node->src[0];  // dims=[C, N]
    const TensorF32 * src_idx   = node->src[1];  // (E,)
    TensorF32       * dst       = node;          // dims=[C, E]

    const int64_t C = node_feat->shape().dims[0];
    const int64_t N = node_feat->shape().dims[1];
    const int64_t E = src_idx->shape().dims[0];

    const float * nf_data  = static_cast<const float*>(node_feat->data());
    const float * idx_data = static_cast<const float*>(src_idx->data());
    float       * d_data   = static_cast<float*>(dst->data());

    const int64_t total = E;
    const int64_t per   = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    for (int64_t e = start; e < end; e++) {
        int64_t i = static_cast<int64_t>(idx_data[e]);
        if (i < 0) i = 0;
        if (i >= N) i = N - 1;
        // 每行连续 C 个特征（C 最内维）
        std::memcpy(d_data + e * C, nf_data + i * C, C * sizeof(float));
    }
}

// ===== per_edge_matmul (SE3 消息传递：逐边矩阵乘) =====
// 前向: dst[e,:] = kernel[e] @ gathered[e,:]
// ggml 布局：kernel dims=[K, M, E]，gathered dims=[K, E]，dst dims=[M, E]
// src0: kernel (E, M, K) 每条边独立卷积核矩阵 (M×K)
// src1: gathered (E, K) 该边源节点特征；dst: (E, M)
void CPUBackend::kernel_per_edge_matmul(TensorF32 * node, ComputeParams * p) {
    const TensorF32 * kernel   = node->src[0];  // dims=[K, M, E]
    const TensorF32 * gathered = node->src[1];  // dims=[K, E]
    TensorF32       * dst      = node;          // dims=[M, E]

    const int64_t K = kernel->shape().dims[0];
    const int64_t M = kernel->shape().dims[1];
    const int64_t E = kernel->shape().dims[2];

    const float * k_data = static_cast<const float*>(kernel->data());
    const float * g_data = static_cast<const float*>(gathered->data());
    float       * d_data = static_cast<float*>(dst->data());

    const int64_t total = E;
    const int64_t per   = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    for (int64_t e = start; e < end; e++) {
        const float * K_e = k_data + e * M * K;       // (M, K) row-major
        const float * g_e = g_data + e * K;           // (K,)
        float       * d_e = d_data + e * M;           // (M,)
        for (int64_t r = 0; r < M; r++) {
            float val = 0.0f;
            for (int64_t c = 0; c < K; c++) {
                val += K_e[r * K + c] * g_e[c];
            }
            d_e[r] = val;
        }
    }
}

// ===== scatter_add (SE3 消息传递：边消息散点累加到目标节点) =====
// 前向: dst[tgt[e],:] += msg[e,:]
// ggml 布局：msg dims=[M, E]，tgt_idx (E,)，dst dims=[M, N]
// src0: msg (E, M) 边消息；src1: tgt_idx (E,)；dst: (N, M)，N = op_params[0]
// 先清零再累加。多个边可指向同一目标节点，必须 += 而非覆盖，故串行在 thread 0 完成。
void CPUBackend::kernel_scatter_add(TensorF32 * node, ComputeParams * p) {
    const TensorF32 * msg         = node->src[0];  // dims=[M, E]
    const TensorF32 * tgt_idx     = node->src[1];  // (E,)
    TensorF32       * dst         = node;          // dims=[M, N]

    const int64_t N = node->op_params[0];
    const int64_t M = msg->shape().dims[0];
    const int64_t E = msg->shape().dims[1];

    const float * m_data  = static_cast<const float*>(msg->data());
    const float * idx_data = static_cast<const float*>(tgt_idx->data());
    float       * d_data  = static_cast<float*>(dst->data());

    // 串行：清零 + 累加，避免同目标节点多边并行 data race
    if (p->ith == 0) {
        std::memset(d_data, 0, N * M * sizeof(float));
        for (int64_t e = 0; e < E; e++) {
            int64_t i = static_cast<int64_t>(idx_data[e]);
            if (i < 0 || i >= N) continue;   // 越界丢弃
            for (int64_t c = 0; c < M; c++) {
                d_data[i * M + c] += m_data[e * M + c];
            }
        }
    }

    p->threadpool->barrier_wait();
}

// ===== per_edge_matmul_back_kernel (per_edge_matmul 反向 wrt kernel) =====
// 前向: dst[e,:] = kernel[e] @ gathered[e,:]
// 反向: dkernel[e,r,c] = grad[e,r] * gathered[e,c]  (逐边外积)
// ggml 布局：grad dims=[M,E]，gathered dims=[K,E]，dst dims=[K,M,E]
void CPUBackend::kernel_per_edge_matmul_back_kernel(TensorF32 * node, ComputeParams * p) {
    const TensorF32 * grad     = node->src[0];  // dims=[M, E]
    const TensorF32 * gathered = node->src[1];  // dims=[K, E]
    TensorF32       * dst      = node;          // dims=[K, M, E]

    const int64_t M = grad->shape().dims[0];
    const int64_t E = grad->shape().dims[1];
    const int64_t K = gathered->shape().dims[0];

    const float * g_data = static_cast<const float*>(grad->data());
    const float * f_data = static_cast<const float*>(gathered->data());
    float       * d_data = static_cast<float*>(dst->data());

    const int64_t total = E;
    const int64_t per   = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    for (int64_t e = start; e < end; e++) {
        const float * g_e = g_data + e * M;   // (M,)
        const float * f_e = f_data + e * K;   // (K,)
        float       * d_e = d_data + e * M * K; // (M, K) row-major
        for (int64_t r = 0; r < M; r++) {
            const float gr = g_e[r];
            for (int64_t c = 0; c < K; c++) {
                d_e[r * K + c] = gr * f_e[c];
            }
        }
    }
}

// ===== per_edge_matmul_back_gathered (per_edge_matmul 反向 wrt gathered) =====
// 前向: dst[e,:] = kernel[e] @ gathered[e,:]
// 反向: dgathered[e,c] = sum_r kernel[e,r,c] * grad[e,r]
// ggml 布局：grad dims=[M,E]，kernel dims=[K,M,E]，dst dims=[K,E]
void CPUBackend::kernel_per_edge_matmul_back_gathered(TensorF32 * node, ComputeParams * p) {
    const TensorF32 * grad   = node->src[0];  // dims=[M, E]
    const TensorF32 * kernel = node->src[1];  // dims=[K, M, E]
    TensorF32       * dst    = node;          // dims=[K, E]

    const int64_t M = grad->shape().dims[0];
    const int64_t E = grad->shape().dims[1];
    const int64_t K = kernel->shape().dims[0];

    const float * g_data = static_cast<const float*>(grad->data());
    const float * k_data = static_cast<const float*>(kernel->data());
    float       * d_data = static_cast<float*>(dst->data());

    const int64_t total = E;
    const int64_t per   = (total + p->nth - 1) / p->nth;
    const int64_t start = per * p->ith;
    const int64_t end   = (start + per < total) ? (start + per) : total;

    for (int64_t e = start; e < end; e++) {
        const float * g_e = g_data + e * M;   // (M,)
        const float * k_e = k_data + e * M * K; // (M, K) row-major
        float       * d_e = d_data + e * K;   // (K,)
        for (int64_t c = 0; c < K; c++) {
            float val = 0.0f;
            for (int64_t r = 0; r < M; r++) {
                val += k_e[r * K + c] * g_e[r];
            }
            d_e[c] = val;
        }
    }
}

void CPUBackend::kernel_sigmoid(TensorF32 * node, ComputeParams * p) {
    TensorF32* output = node->src[0];
    float* data = output->data();
    int64_t n = output->numel();

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
    (void)p;
}

// ===== unary op 计算函数实现 =====

static void compute_forward_abs(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = fabsf(s[i]);
    }
}

static void compute_forward_sgn(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = (s[i] > 0.0f) ? 1.0f : ((s[i] < 0.0f) ? -1.0f : 0.0f);
    }
}

static void compute_forward_neg(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = -s[i];
    }
}

static void compute_forward_step(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = (s[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

static void compute_forward_relu(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = (s[i] > 0.0f) ? s[i] : 0.0f;
    }
}

static void compute_forward_gelu(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    const float C1 = 0.044715f;
    const float C2 = sqrtf(2.0f / M_PI);
    for (int64_t i = p->ith; i < n; i += p->nth) {
        float x = s[i];
        float inner = C2 * (x + C1 * x * x * x);
        d[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

static void compute_forward_gelu_quick(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        float x = s[i];
        d[i] = x * (1.0f / (1.0f + expf(-1.702f * x)));
    }
}

static void compute_forward_silu(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = s[i] / (1.0f + expf(-s[i]));
    }
}

static void compute_forward_tanh(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = tanhf(s[i]);
    }
}

static void compute_forward_elu(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        float x = s[i];
        d[i] = (x > 0.0f) ? x : (expf(x) - 1.0f);
    }
}

static void compute_forward_sigmoid(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = 1.0f / (1.0f + expf(-s[i]));
    }
}

static void compute_forward_hardsigmoid(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        float x = s[i];
        if (x <= -3.0f)       d[i] = 0.0f;
        else if (x >= 3.0f)   d[i] = 1.0f;
        else                    d[i] = x / 6.0f + 0.5f;
    }
}

static void compute_forward_hardswish(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        float x = s[i];
        if (x <= -3.0f)       d[i] = 0.0f;
        else if (x >= 3.0f)   d[i] = x;
        else                    d[i] = x * (x + 3.0f) / 6.0f;
    }
}

static void compute_forward_exp(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = expf(s[i]);
    }
}

static void compute_forward_log(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = logf(s[i]);
    }
}

static void compute_forward_sqrt(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = sqrtf(s[i]);
    }
}

static void compute_forward_sin(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = sinf(s[i]);
    }
}

static void compute_forward_cos(ComputeParams* p, TensorF32* dst) {
    TensorF32* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = cosf(s[i]);
    }
}

// ============================================================
// OP_FAPE forward kernel — Frame Aligned Point Error
// ============================================================
//
// 输入 (via dst->src[]):
//   src[0]: pred_coords        [N_atoms, 3]      — 预测坐标
//   src[1]: true_coords        [N_atoms, 3]      — 真实坐标
//   src[2]: frame_atom_indices [N_frames, 3]     — 每帧 3 原子全局索引 (float-encoded ints)
//   src[3]: frames_mask        [N_frames]        — 帧有效性 mask
//   src[4]: positions_mask     [N_atoms]         — 原子位置 mask
//
// op_params:
//   [0..1]: d_clamp (float)
//   [2..3]: epsilon (float)
//   [4..5]: length_scale (float)
//
// 输出: dst = scalar (1,) — FAPE loss
//
// 6 步流水线:
//   Step 1: Gather frame atoms from coords
//   Step 2: Gram-Schmidt → T_inv per frame
//   Step 3: Transform all atoms to local frames
//   Step 4: Pairwise Euclidean distances
//   Step 5: Clamp + apply masks
//   Step 6: Reduce to scalar
static void compute_forward_fape(ComputeParams* p, TensorF32* dst) {
    ThreadPool* tp = p->threadpool;

    // Only thread 0 does the computation (single scalar output)
    if (p->ith != 0) {
        tp->barrier_wait();
        return;
    }

    TensorF32* src0 = dst->src[0];  // pred_coords  [N_atoms, 3]
    TensorF32* src1 = dst->src[1];  // true_coords  [N_atoms, 3]
    TensorF32* src2 = dst->src[2];  // frame_atom_indices [N_frames, 3]
    TensorF32* src3 = dst->src[3];  // frames_mask  [N_frames]
    TensorF32* src4 = dst->src[4];  // positions_mask [N_atoms]

    float d_clamp     = reinterpret_cast<float&>(dst->op_params[0]);
    float epsilon     = reinterpret_cast<float&>(dst->op_params[2]);
    float length_scale = reinterpret_cast<float&>(dst->op_params[4]);

    float* pred_coords   = src0->data();
    float* true_coords   = src1->data();
    float* frame_indices = src2->data();
    float* frames_mask   = src3->data();
    float* positions_mask = src4->data();

    int64_t N_atoms  = src0->shape().dims[1];   // number of rows
    int64_t N_frames = src2->shape().dims[1];   // = N_frames

    // Pre-compute T_inv for each frame (both pred and true)
    // T_inv = [R^T | -R^T*A] stored as 12 floats per frame:
    //   R^T col0 (3 floats), R^T col1 (3), R^T col2 (3), translation (3)
    std::vector<float> T_inv_pred(N_frames * 12);
    std::vector<float> T_inv_true(N_frames * 12);

    for (int64_t n = 0; n < N_frames; n++) {
        float fm = frames_mask[n];

        // Default: identity (no rotation, zero translation)
        for (int k = 0; k < 12; k++) {
            T_inv_pred[n * 12 + k] = 0.0f;
            T_inv_true[n * 12 + k] = 0.0f;
        }
        T_inv_pred[n * 12 + 0] = 1.0f;  // R^T col0 x
        T_inv_pred[n * 12 + 4] = 1.0f;  // R^T col1 y
        T_inv_pred[n * 12 + 8] = 1.0f;  // R^T col2 z
        T_inv_true[n * 12 + 0] = 1.0f;
        T_inv_true[n * 12 + 4] = 1.0f;
        T_inv_true[n * 12 + 8] = 1.0f;

        if (fm == 0.0f) continue;

        // Gather frame atoms (Step 1)
        int idx_A = (int)frame_indices[n * 3 + 0];
        int idx_B = (int)frame_indices[n * 3 + 1];
        int idx_C = (int)frame_indices[n * 3 + 2];

        // pred frame atoms
        float pAx = pred_coords[idx_A * 3 + 0], pAy = pred_coords[idx_A * 3 + 1], pAz = pred_coords[idx_A * 3 + 2];
        float pBx = pred_coords[idx_B * 3 + 0], pBy = pred_coords[idx_B * 3 + 1], pBz = pred_coords[idx_B * 3 + 2];
        float pCx = pred_coords[idx_C * 3 + 0], pCy = pred_coords[idx_C * 3 + 1], pCz = pred_coords[idx_C * 3 + 2];

        // true frame atoms
        float tAx = true_coords[idx_A * 3 + 0], tAy = true_coords[idx_A * 3 + 1], tAz = true_coords[idx_A * 3 + 2];
        float tBx = true_coords[idx_B * 3 + 0], tBy = true_coords[idx_B * 3 + 1], tBz = true_coords[idx_B * 3 + 2];
        float tCx = true_coords[idx_C * 3 + 0], tCy = true_coords[idx_C * 3 + 1], tCz = true_coords[idx_C * 3 + 2];

        // Gram-Schmidt for pred (Step 2)
        {
            // v1 = B-A, v2 = C-A
            float v1x = pBx - pAx, v1y = pBy - pAy, v1z = pBz - pAz;
            float v2x = pCx - pAx, v2y = pCy - pAy, v2z = pCz - pAz;

            // e1 = normalize(v1)  —— 退化坐标(v1≈0)时 n1→0，加 eps 避免 0/0=NaN
            float n1 = sqrtf(v1x * v1x + v1y * v1y + v1z * v1z + epsilon);
            float e1x = v1x / n1, e1y = v1y / n1, e1z = v1z / n1;

            // e2 = normalize(v2 - (v2·e1)*e1)  —— 三点共线/退化时 u2≈0，加 eps
            float dot = v2x * e1x + v2y * e1y + v2z * e1z;
            float u2x = v2x - dot * e1x, u2y = v2y - dot * e1y, u2z = v2z - dot * e1z;
            float n2 = sqrtf(u2x * u2x + u2y * u2y + u2z * u2z + epsilon);
            float e2x = u2x / n2, e2y = u2y / n2, e2z = u2z / n2;

            // e3 = e1 × e2
            float e3x = e1y * e2z - e1z * e2y;
            float e3y = e1z * e2x - e1x * e2z;
            float e3z = e1x * e2y - e1y * e2x;

            // R^T columns = e1, e2, e3
            T_inv_pred[n * 12 + 0] = e1x; T_inv_pred[n * 12 + 1] = e1y; T_inv_pred[n * 12 + 2] = e1z;
            T_inv_pred[n * 12 + 3] = e2x; T_inv_pred[n * 12 + 4] = e2y; T_inv_pred[n * 12 + 5] = e2z;
            T_inv_pred[n * 12 + 6] = e3x; T_inv_pred[n * 12 + 7] = e3y; T_inv_pred[n * 12 + 8] = e3z;
            // translation = -R^T * A
            T_inv_pred[n * 12 + 9]  = -(e1x * pAx + e1y * pAy + e1z * pAz);
            T_inv_pred[n * 12 + 10] = -(e2x * pAx + e2y * pAy + e2z * pAz);
            T_inv_pred[n * 12 + 11] = -(e3x * pAx + e3y * pAy + e3z * pAz);
        }

        // Gram-Schmidt for true
        {
            float v1x = tBx - tAx, v1y = tBy - tAy, v1z = tBz - tAz;
            float v2x = tCx - tAx, v2y = tCy - tAy, v2z = tCz - tAz;

            float n1 = sqrtf(v1x * v1x + v1y * v1y + v1z * v1z + epsilon);
            float e1x = v1x / n1, e1y = v1y / n1, e1z = v1z / n1;

            float dot = v2x * e1x + v2y * e1y + v2z * e1z;
            float u2x = v2x - dot * e1x, u2y = v2y - dot * e1y, u2z = v2z - dot * e1z;
            float n2 = sqrtf(u2x * u2x + u2y * u2y + u2z * u2z + epsilon);
            float e2x = u2x / n2, e2y = u2y / n2, e2z = u2z / n2;

            float e3x = e1y * e2z - e1z * e2y;
            float e3y = e1z * e2x - e1x * e2z;
            float e3z = e1x * e2y - e1y * e2x;

            T_inv_true[n * 12 + 0] = e1x; T_inv_true[n * 12 + 1] = e1y; T_inv_true[n * 12 + 2] = e1z;
            T_inv_true[n * 12 + 3] = e2x; T_inv_true[n * 12 + 4] = e2y; T_inv_true[n * 12 + 5] = e2z;
            T_inv_true[n * 12 + 6] = e3x; T_inv_true[n * 12 + 7] = e3y; T_inv_true[n * 12 + 8] = e3z;
            T_inv_true[n * 12 + 9]  = -(e1x * tAx + e1y * tAy + e1z * tAz);
            T_inv_true[n * 12 + 10] = -(e2x * tAx + e2y * tAy + e2z * tAz);
            T_inv_true[n * 12 + 11] = -(e3x * tAx + e3y * tAy + e3z * tAz);
        }
    }

    // Steps 3-6: Transform + distance + clamp + mask + reduce
    float sum_loss    = 0.0f;
    float sum_fm      = 0.0f;  // sum of frames_mask
    float sum_pm      = 0.0f;  // sum of positions_mask

    for (int64_t n = 0; n < N_frames; n++) {
        float fm = frames_mask[n];
        if (fm == 0.0f) continue;
        sum_fm += fm;

        float* Tp = &T_inv_pred[n * 12];
        float* Tt = &T_inv_true[n * 12];

        for (int64_t j = 0; j < N_atoms; j++) {
            float pm = positions_mask[j];
            if (pm == 0.0f) continue;

            // Transform pred atom j to frame n's local coords
            float px = pred_coords[j * 3 + 0], py = pred_coords[j * 3 + 1], pz = pred_coords[j * 3 + 2];
            float lpx = Tp[0]*px + Tp[1]*py + Tp[2]*pz  + Tp[9];
            float lpy = Tp[3]*px + Tp[4]*py + Tp[5]*pz  + Tp[10];
            float lpz = Tp[6]*px + Tp[7]*py + Tp[8]*pz  + Tp[11];

            // Transform true atom j to frame n's local coords
            float tx = true_coords[j * 3 + 0], ty = true_coords[j * 3 + 1], tz = true_coords[j * 3 + 2];
            float ltx = Tt[0]*tx + Tt[1]*ty + Tt[2]*tz  + Tt[9];
            float lty = Tt[3]*tx + Tt[4]*ty + Tt[5]*tz  + Tt[10];
            float ltz = Tt[6]*tx + Tt[7]*ty + Tt[8]*tz  + Tt[11];

            // Step 4: Euclidean distance
            float dx = lpx - ltx, dy = lpy - lty, dz = lpz - ltz;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz + epsilon);

            // Step 5: Clamp and apply masks
            if (dist > d_clamp) dist = d_clamp;
            float masked = dist * fm * pm;

            sum_loss += masked;
            sum_pm += pm;
        }
    }

    // Step 6: Normalize
    float denom = sum_fm * sum_pm + epsilon;
    float loss = sum_loss / denom / length_scale;

    // Write scalar output
    dst->data()[0] = loss;

    if (getenv("GRAPH_DEBUG_LOSS")) {
        fprintf(stderr, "[fape] computed loss=%f denom=%f sum_loss=%f N_atoms=%lld N_frames=%lld\n",
                loss, denom, sum_loss, (long long)N_atoms, (long long)N_frames);
    }

    tp->barrier_wait();
}

// ============================================================
// OP_FAPE_BACK kernel — FAPE loss backward pass
// ============================================================
//
// Computes dL/d(pred_coords) — gradient of FAPE loss w.r.t. predicted coordinates.
//
// Input (via dst->src[]):
//   src[0]: grad              scalar (1,) — upstream gradient dL/dL_fape (= 1.0 at root)
//   src[1]: pred_coords       [N_atoms, 3]  — forward pred_coords
//   src[2]: true_coords       [N_atoms, 3]  — forward true_coords
//   src[3]: frame_atom_indices[N_frames, 3] — frame atom global indices
//   src[4]: frames_mask       [N_frames]    — frame mask
//   src[5]: positions_mask    [N_atoms]     — position mask
//
// op_params: same as OP_FAPE (d_clamp, epsilon, length_scale)
//
// Output: dst = [N_atoms, 3] — gradient w.r.t. pred_coords
//
// Backward strategy (Phase 1 — simplified):
//   Back-propagate through Steps 4-6 only:
//   Step 6 backward: dL/dmasked = upstream / (sum_fm*sum_pm + eps) / length_scale
//   Step 5 backward: clamp gate = (e < d_clamp); dL/de = dL/dmasked * gate * fm * pm
//   Step 4 backward: dL/d(local_pred) = dL/de * (local_pred - local_true) / (e + eps)
//   Step 3 backward: dL/d(global_pred) = R_pred @ dL/d(local_pred)
//   (Gram-Schmidt backward through T_inv is deferred to Phase 2)
static void compute_forward_fape_back(ComputeParams* p, TensorF32* dst) {
    ThreadPool* tp = p->threadpool;

    if (p->ith != 0) {
        tp->barrier_wait();
        return;
    }

    TensorF32* grad_scalar     = dst->src[0];  // upstream gradient (1,)
    TensorF32* pred_coords_t   = dst->src[1];  // [N_atoms, 3]
    TensorF32* true_coords_t   = dst->src[2];  // [N_atoms, 3]
    TensorF32* frame_indices_t = dst->src[3];  // [N_frames, 3]
    TensorF32* frames_mask_t   = dst->src[4];  // [N_frames]
    TensorF32* positions_mask_t= dst->src[5];  // [N_atoms]

    float d_clamp     = reinterpret_cast<float&>(dst->op_params[0]);
    float epsilon     = reinterpret_cast<float&>(dst->op_params[2]);
    float length_scale = reinterpret_cast<float&>(dst->op_params[4]);

    float* pred_coords   = pred_coords_t->data();
    float* true_coords   = true_coords_t->data();
    float* frame_indices = frame_indices_t->data();
    float* frames_mask   = frames_mask_t->data();
    float* positions_mask = positions_mask_t->data();
    float* d_pred        = dst->data();  // output gradient

    float upstream = grad_scalar->data()[0];  // dL/dL_fape

    int64_t N_atoms  = pred_coords_t->shape().dims[1];
    int64_t N_frames = frame_indices_t->shape().dims[1];

    // Zero output gradient
    for (int64_t i = 0; i < N_atoms * 3; i++) d_pred[i] = 0.0f;

    // ---- Step 1+2: Recompute T_inv for each frame (same as forward) ----
    std::vector<float> T_inv_pred(N_frames * 12);

    for (int64_t n = 0; n < N_frames; n++) {
        float fm = frames_mask[n];

        for (int k = 0; k < 12; k++) T_inv_pred[n * 12 + k] = 0.0f;
        T_inv_pred[n * 12 + 0] = 1.0f;
        T_inv_pred[n * 12 + 4] = 1.0f;
        T_inv_pred[n * 12 + 8] = 1.0f;

        if (fm == 0.0f) continue;

        int idx_A = (int)frame_indices[n * 3 + 0];
        int idx_B = (int)frame_indices[n * 3 + 1];
        int idx_C = (int)frame_indices[n * 3 + 2];

        float pAx = pred_coords[idx_A * 3 + 0], pAy = pred_coords[idx_A * 3 + 1], pAz = pred_coords[idx_A * 3 + 2];
        float pBx = pred_coords[idx_B * 3 + 0], pBy = pred_coords[idx_B * 3 + 1], pBz = pred_coords[idx_B * 3 + 2];
        float pCx = pred_coords[idx_C * 3 + 0], pCy = pred_coords[idx_C * 3 + 1], pCz = pred_coords[idx_C * 3 + 2];

        // Gram-Schmidt  —— 退化坐标加 eps 避免 0/0=NaN
        float v1x = pBx - pAx, v1y = pBy - pAy, v1z = pBz - pAz;
        float v2x = pCx - pAx, v2y = pCy - pAy, v2z = pCz - pAz;

        float n1 = sqrtf(v1x * v1x + v1y * v1y + v1z * v1z + epsilon);
        float e1x = v1x / n1, e1y = v1y / n1, e1z = v1z / n1;

        float dot = v2x * e1x + v2y * e1y + v2z * e1z;
        float u2x = v2x - dot * e1x, u2y = v2y - dot * e1y, u2z = v2z - dot * e1z;
        float n2 = sqrtf(u2x * u2x + u2y * u2y + u2z * u2z + epsilon);
        float e2x = u2x / n2, e2y = u2y / n2, e2z = u2z / n2;

        float e3x = e1y * e2z - e1z * e2y;
        float e3y = e1z * e2x - e1x * e2z;
        float e3z = e1x * e2y - e1y * e2x;

        // R^T (rotation transpose): columns are e1, e2, e3
        T_inv_pred[n * 12 + 0] = e1x; T_inv_pred[n * 12 + 1] = e1y; T_inv_pred[n * 12 + 2] = e1z;
        T_inv_pred[n * 12 + 3] = e2x; T_inv_pred[n * 12 + 4] = e2y; T_inv_pred[n * 12 + 5] = e2z;
        T_inv_pred[n * 12 + 6] = e3x; T_inv_pred[n * 12 + 7] = e3y; T_inv_pred[n * 12 + 8] = e3z;
        T_inv_pred[n * 12 + 9]  = -(e1x * pAx + e1y * pAy + e1z * pAz);
        T_inv_pred[n * 12 + 10] = -(e2x * pAx + e2y * pAy + e2z * pAz);
        T_inv_pred[n * 12 + 11] = -(e3x * pAx + e3y * pAy + e3z * pAz);
    }

    // ---- Same for true_coords ----
    std::vector<float> T_inv_true(N_frames * 12);
    for (int64_t n = 0; n < N_frames; n++) {
        float fm = frames_mask[n];

        for (int k = 0; k < 12; k++) T_inv_true[n * 12 + k] = 0.0f;
        T_inv_true[n * 12 + 0] = 1.0f;
        T_inv_true[n * 12 + 4] = 1.0f;
        T_inv_true[n * 12 + 8] = 1.0f;

        if (fm == 0.0f) continue;

        int idx_A = (int)frame_indices[n * 3 + 0];
        int idx_B = (int)frame_indices[n * 3 + 1];
        int idx_C = (int)frame_indices[n * 3 + 2];

        float tAx = true_coords[idx_A * 3 + 0], tAy = true_coords[idx_A * 3 + 1], tAz = true_coords[idx_A * 3 + 2];
        float tBx = true_coords[idx_B * 3 + 0], tBy = true_coords[idx_B * 3 + 1], tBz = true_coords[idx_B * 3 + 2];
        float tCx = true_coords[idx_C * 3 + 0], tCy = true_coords[idx_C * 3 + 1], tCz = true_coords[idx_C * 3 + 2];

        float v1x = tBx - tAx, v1y = tBy - tAy, v1z = tBz - tAz;
        float v2x = tCx - tAx, v2y = tCy - tAy, v2z = tCz - tAz;

        float n1 = sqrtf(v1x * v1x + v1y * v1y + v1z * v1z + epsilon);
        float e1x = v1x / n1, e1y = v1y / n1, e1z = v1z / n1;

        float dot = v2x * e1x + v2y * e1y + v2z * e1z;
        float u2x = v2x - dot * e1x, u2y = v2y - dot * e1y, u2z = v2z - dot * e1z;
        float n2 = sqrtf(u2x * u2x + u2y * u2y + u2z * u2z + epsilon);
        float e2x = u2x / n2, e2y = u2y / n2, e2z = u2z / n2;

        float e3x = e1y * e2z - e1z * e2y;
        float e3y = e1z * e2x - e1x * e2z;
        float e3z = e1x * e2y - e1y * e2x;

        T_inv_true[n * 12 + 0] = e1x; T_inv_true[n * 12 + 1] = e1y; T_inv_true[n * 12 + 2] = e1z;
        T_inv_true[n * 12 + 3] = e2x; T_inv_true[n * 12 + 4] = e2y; T_inv_true[n * 12 + 5] = e2z;
        T_inv_true[n * 12 + 6] = e3x; T_inv_true[n * 12 + 7] = e3y; T_inv_true[n * 12 + 8] = e3z;
        T_inv_true[n * 12 + 9]  = -(e1x * tAx + e1y * tAy + e1z * tAz);
        T_inv_true[n * 12 + 10] = -(e2x * tAx + e2y * tAy + e2z * tAz);
        T_inv_true[n * 12 + 11] = -(e3x * tAx + e3y * tAy + e3z * tAz);
    }

    // ---- Step 6 backward: compute normalization denominator ----
    float sum_fm = 0.0f, sum_pm = 0.0f;
    for (int64_t n = 0; n < N_frames; n++) if (frames_mask[n] != 0.0f) sum_fm += frames_mask[n];
    for (int64_t j = 0; j < N_atoms; j++)  if (positions_mask[j] != 0.0f) sum_pm += positions_mask[j];

    float denom = sum_fm * sum_pm + epsilon;
    // dL/dmasked = upstream / denom / length_scale
    float d_masked = upstream / denom / length_scale;

    // ---- Steps 5,4,3 backward: accumulate gradients ----
    for (int64_t n = 0; n < N_frames; n++) {
        float fm = frames_mask[n];
        if (fm == 0.0f) continue;

        float* Tp = &T_inv_pred[n * 12];  // R^T_pred, t_pred
        float* Tt = &T_inv_true[n * 12];  // R^T_true,  t_true

        // Extract rotation matrices (R_pred, R_true) from T_inv
        // T_inv stores [R^T | -R^T*A], so:
        //   local = R^T * global + t
        //   R is columns 0,1,2 of R^T read row-wise
        // For backward: d(global) = R @ d(local) = (R^T)^T @ d(local)
        // Since R^T is stored, R is just its transpose.
        // But R^T columns are stored contiguously, so R row i = R^T column i

        for (int64_t j = 0; j < N_atoms; j++) {
            float pm = positions_mask[j];
            if (pm == 0.0f) continue;

            // Transform pred atom j to local coords (same as forward Step 3)
            float px = pred_coords[j * 3 + 0], py = pred_coords[j * 3 + 1], pz = pred_coords[j * 3 + 2];
            float lpx = Tp[0]*px + Tp[1]*py + Tp[2]*pz  + Tp[9];
            float lpy = Tp[3]*px + Tp[4]*py + Tp[5]*pz  + Tp[10];
            float lpz = Tp[6]*px + Tp[7]*py + Tp[8]*pz  + Tp[11];

            float tx = true_coords[j * 3 + 0], ty = true_coords[j * 3 + 1], tz = true_coords[j * 3 + 2];
            float ltx = Tt[0]*tx + Tt[1]*ty + Tt[2]*tz  + Tt[9];
            float lty = Tt[3]*tx + Tt[4]*ty + Tt[5]*tz  + Tt[10];
            float ltz = Tt[6]*tx + Tt[7]*ty + Tt[8]*tz  + Tt[11];

            // Step 4: distance
            float dx = lpx - ltx, dy = lpy - lty, dz = lpz - ltz;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz + epsilon);

            // Step 5: clamp gate
            if (dist >= d_clamp) continue;  // gradient is 0 when clamped

            // Step 4 backward: dL/d(local_pred) = dL/de * (local_pred - local_true) / (dist + eps)
            // dL/de = d_masked * fm * pm (from Step 5)
            float d_dist = d_masked * fm * pm;
            float inv_dist = 1.0f / (dist + epsilon);
            float d_lpx = d_dist * dx * inv_dist;
            float d_lpy = d_dist * dy * inv_dist;
            float d_lpz = d_dist * dz * inv_dist;

            // Step 3 backward: d(global) = R @ d(local)
            // R = transpose of R^T
            // R^T is stored as: [e1x e1y e1z] [e2x e2y e2z] [e3x e3y e3z]
            // (columns 0,1,2 stored as rows)
            // R = [e1x e2x e3x]
            //     [e1y e2y e3y]
            //     [e1z e2z e3z]
            float d_px = Tp[0]*d_lpx + Tp[3]*d_lpy + Tp[6]*d_lpz;
            float d_py = Tp[1]*d_lpx + Tp[4]*d_lpy + Tp[7]*d_lpz;
            float d_pz = Tp[2]*d_lpx + Tp[5]*d_lpy + Tp[8]*d_lpz;

            d_pred[j * 3 + 0] += d_px;
            d_pred[j * 3 + 1] += d_py;
            d_pred[j * 3 + 2] += d_pz;
        }
    }

    tp->barrier_wait();
}

} // namespace ppml

#include "ppml/Backend.h"
#include "ppml/ComputeGraph.h"
#include "ppml/FlashAttn.h"     // 【Step ⑤】FlashAttnConfig + flash_attn_{forward,backward}_cuda 声明 ✓
#include <cstring>
#include <vector>

namespace ppml {

// ===== CUDA kernel forward declarations (implemented in src/cuda/CUDAKernels.cu) =====
extern void layernorm_forward_cuda(
    float * out,   float * mean,  float * rstd,
    const float * inp, const float * weight, const float * bias,
    int B, int T, int C, int block_size);

extern void layernorm_backward_cuda(
    float * dinp, float * dweight, float * dbias,
    const float * dout, const float * inp,
    const float * weight, const float * mean, const float * rstd,
    int B, int T, int C, int block_size);

extern void elementwise_add_cuda(float * A, float * B, float * C, int N, int block_size);
extern void elementwise_sub_cuda(float * A, float * B, float * C, int N, int block_size);
extern void elementwise_mul_cuda(float * A, float * B, float * C, int N, int block_size);
extern void elementwise_div_cuda(float * A, float * B, float * C, int N, int block_size);

// 广播逐元素 op（实现于 src/cuda/CUDAKernels.cu）：dst[i] = op(a[i%an], b[i%bn])
// 对齐 CPU kernel_elemwise 的"尾部对齐后缀"广播语义；eop: 0=add 1=sub 2=mul 3=div
extern void bcast_elemwise_cuda(
    const float * a, const float * b, float * dst,
    int64_t n, int64_t an, int64_t bn, int eop);

// OP_ADD1 前向（实现于 src/cuda/CUDAKernels.cu）：dst[i] = src[i] + b（b 标量，D2H 读取）
extern void add1_cuda(const float * src, float * dst, float b, int64_t n);

// OP_DUP/OP_CPY/OP_CONT 整块拷贝（实现于 src/cuda/CUDAKernels.cu）：四分支 buffer-aware
extern void copy_tensor_cuda(const float * src, float * dst, int64_t n,
                             bool src_dev, bool dst_dev);

// OP_FAPE 前向（实现于 src/cuda/CUDAKernels.cu）：FAPE 结构损失 → 标量 [1]
extern void fape_cuda(const float* pred, const float* truth,
                      const float* frame_idx, const float* fmask, const float* pmask,
                      float* dst, int64_t N_atoms, int64_t N_frames,
                      float d_clamp, float epsilon, float length_scale);

// OP_PERMUTE/OP_TRANSPOSE 前向（实现于 src/cuda/CUDAKernels.cu）：通用维度重排
extern void permute_cuda(const float * src, float * dst, int64_t total, int ndim,
                         int32_t d0, int32_t d1, int32_t d2, int32_t d3,
                         int64_t s0, int64_t s1, int64_t s2, int64_t s3,
                         int64_t dd0, int64_t dd1, int64_t dd2, int64_t dd3);

// OP_TRANSPOSE 2D 专用（实现于 src/cuda/CUDAKernels.cu）：tile + XOR swizzle
extern void transpose_cuda(const float * src, float * dst, int64_t M, int64_t N);

// Tensor Core / MMA 硬件能力检测（实现于 src/cuda/CUDAKernels.cu）
//   out_major/out_minor: compute capability。返回 true 表示 sm_80+（TF32/F16 MMA 可用）。
extern bool cutlass_hw_supported(int* out_major, int* out_minor);

// TF32 tensor-core GEMM 性能测试（实现于 src/cuda/CUDAKernels.cu）
//   C[M,N] = A[M,K] * B[N,K]ᵀ（B 行主序 [N][K]）。返回 0=成功，ms_out 为平均耗时。
extern int tf32_gemm_bench_cuda(const float* A, const float* B, float* C,
                                int M, int K, int N, float* ms_out);
extern void softmax_cuda(float * input, float * output, int M, int N, int block_size);
extern void softmax_backward_cuda(
    const float * grad, const float * output, float * dst,
    int M, int N, int block_size, float scale);

extern void mul_mat_cuda(float* A, float* B, float* C, int M, int K, int N);

// 【2026-09-22 Step ⑤】Flash Attention 融合算子（实现见 src/cuda/FlashAttnKernel.cu ✓）
//   返回 0 = 已执行；1 = 不支持（无设备 / head_dim 超上限 / 参数非法）⇒ *st 上报 ✗
extern int flash_attn_forward_cuda(const FlashAttnConfig& cfg, const float* Q, const float* K, const float* V,
                                   const float* bias, float* O, float* LSE);
extern int flash_attn_backward_cuda(const FlashAttnConfig& cfg, const float* Q, const float* K, const float* V,
                                    const float* bias, const float* O, const float* dO, const float* LSE,
                                    float* dQ, float* dK, float* dV, float* dbias);

// OP_MUL_MAT fp16 输入（前期支持，2026-09-10）：A/B 按 16 位 half 存储，kernel 内转 fp32
// 累加，输出仍 fp32。M/K/N 语义与 mul_mat_cuda 完全一致（B 以 [N][K] 转置存储）。
extern void mul_mat_cuda_f16(const void* A, const void* B, float* C, int M, int K, int N);

// 【2026-09-22 Step 1】TF32 tensor-core GEMM（smem + cp.async 双缓冲流水）
//   C[M,N] = A[M,K] * B[N,K]ᵀ（fp32 输入 → 硬件 tf32 截断；fp32 累加）。
//   返回 0 = 已执行；1 = 不支持（< sm_80+）或尺寸非法 ⇒ 调用方回落 mul_mat_cuda（SIMT）。
extern int mul_mat_mma_tf32_smem(const void* A, const void* B, float* C, int M, int K, int N);

// 【2026-09-22 Step 0】MUL_MAT 统计（PPML_MULMAT_STATS=1）：CUDA event 计时，begin/end 夹一次 launch
extern void mulmat_stats_cuda_begin(int M, int K, int N);
extern void mulmat_stats_cuda_end(const char* path);

// OP_MUL_MAT fp16 tensor-core（mma.m16n8k16, fp32 累加，2026-09-10）
//   返回 0 = 已执行；1 = 硬件 < sm_80（调用方回落 mul_mat_cuda_f16）。
extern int mul_mat_mma_f16(const void* A, const void* B, float* C, int M, int K, int N);

// OP_MUL_MAT fp16 tensor-core 流水版（smem + ldmatrix + cp.async 双缓冲，2026-09-10）
//   返回 0 = 已执行；1 = 不支持（< sm_80 / 尺寸非法）→ 回落简化 MMA 或 SIMT。
extern int mul_mat_mma_f16_smem(const void* A, const void* B, float* C, int M, int K, int N);

extern void out_prod_cuda(
    const float* src0, const float* src1, float* dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    int64_t ne0,  int64_t ne1,  int64_t ne2,  int64_t ne3);

// Unary ops（RELU/SQRT/EXP 等，实现于 src/cuda/CUDAKernels.cu）
extern void unary_cuda(const float * src, float * dst, int N, int uop, int block_size);

// OP_SUM/OP_MEAN 全元素归约（实现于 src/cuda/CUDAKernels.cu）：dst 须已清零
extern void sum_cuda(const float * src, float * dst, long long n);
extern void mean_cuda(const float * src, float * dst, long long n);

// OP_MAX_ALL 全元素归约（实现于 src/cuda/CUDAKernels.cu）：max → 标量，跳 NaN
extern void max_all_cuda(const float * src, float * dst, long long n);

// OP_SUM_ROWS 沿最内维归约（实现于 src/cuda/CUDAKernels.cu）：每行一个 block，
// grid=nrows，dst[row] = Σ_{col<ncols} src[row*ncols+col]
extern void sum_rows_cuda(const float * src, float * dst, int64_t ncols, int64_t nrows);

// OP_RELU_BACK（实现于 src/cuda/CUDAKernels.cu）：dst[i] = (x[i]>0) ? grad[i] : 0
extern void relu_back_cuda(const float * grad, const float * x, float * dst, int64_t n);

// OP_REPEAT_BACK 归约（实现于 src/cuda/CUDAKernels.cu）：src(大) → dst(小)，dst[j]=Σ src[j+k*dd]
// 仅支持同 ndim（src 各维 = dst 各维 × 整数重复因子），跨 ndim 广播由 CPU 回落。
extern void repeat_back_cuda(
    const float * src, float * dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne0,  int64_t ne1,  int64_t ne2,  int64_t ne3);

// OP_REPEAT 前向（实现于 src/cuda/CUDAKernels.cu）：dst[j] = src[s]，s_d = j_d % src_dims[d]
// 简化版：一维 grid-stride + 尾部对齐取模，dst 各维须为 src 各维整数倍；
// src 缺维/维=1 由 kernel 内"缺维取模 1"统一处理（跨 ndim 广播）。
extern void repeat_cuda(
    const float * src, float * dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne0,  int64_t ne1,  int64_t ne2,  int64_t ne3);

// OP_SET_ROWS 前向（实现于 src/cuda/CUDAKernels.cu）：散点覆写，对齐 CPU kernel_set_rows。
// dst = a 全量拷贝（保留未覆盖行），再把 c[k] 覆写到 b[k] 指定行（越界钳制）。
// a/c/dst 严格 2D(N,M)，b 为 (K,) float 编码索引。重复索引预扫在包装函数内（宿主 D2H）。
extern void set_rows_cuda(
    const float * a, const float * b, const float * c, float * dst,
    int64_t N, int64_t M, int64_t K);

// OP_SCALE 前向（实现于 src/cuda/CUDAKernels.cu）：dst[i] = scale * x[i]
extern void scale_cuda(const float * x, float * dst, float scale, int64_t nelements);

// OP_CLAMP 前向（实现于 src/cuda/CUDAKernels.cu）：dst[i] = clamp(x[i], lo, hi)
extern void clamp_cuda(const float * x, float * dst, int N, float lo, float hi, int block_size);

// OP_RMS_NORM（实现于 src/cuda/CUDAKernels.cu）：沿 dims[0] 归一化 dst=x/sqrt(mean(x²)+eps)
// 前置: ncols % 32 == 0，否则 CUDA 侧直接 return，由调度回落 CPU
extern void rms_norm_cuda(const float * x, float * dst,
                          int64_t ncols, int64_t nrows,
                          float eps, cudaStream_t stream);

// OP_GET_ROWS（实现于 src/cuda/CUDAKernels.cu）：embedding 查表
//   W (N,M) 行主序, idx(K,) float-encoded 行索引 → dst(K,M) 行主序
//   越界钳制: i<0→0, i>=N→N-1 (对齐 CPU kernel_get_rows)
extern void get_rows_cuda(const float * W, const float * idx, float * dst,
                          int64_t N, int64_t M, int64_t K, cudaStream_t stream);

// OP_GET_ROWS_BACK（实现于 src/cuda/CUDAKernels.cu）：embedding 查表反向
//   dy(K,M) 梯度, idx(K,) 行索引 → dW(N,M) 先清零再散点累加 (atomicAdd)
//   越界: i<0||i>=N 丢弃 (对齐 CPU kernel_get_rows_back)
extern void get_rows_back_cuda(const float * dy, const float * idx, float * dW,
                               int64_t N, int64_t M, int64_t K, cudaStream_t stream);

// N-ary concat：srcs/start/len 均须为 device 指针（见 concat_nary_cuda）。
// 注意：concat 不支持广播语义，非拼接维必须与 dst 一致。
extern void concat_nary_cuda(
    const float* const* srcs, int n_src,
    const int64_t* start, const int64_t* len,
    float* dst, int dim,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

// SE3 消息传递三件套（方案 B）+ per_edge_matmul 反向核（实现于 src/cuda/SE3MessagePassingKernel.cu）
extern void edge_gather_rows_cuda(
    const float* node_feat, const float* src_idx, float* dst,
    int N, int C, int E);
extern void per_edge_matmul_cuda(
    const float* kernel, const float* gathered, float* dst,
    int M, int K, int E);
extern void scatter_add_cuda(
    const float* msg, const float* tgt_idx, float* dst,
    int N, int M, int E);
extern void per_edge_matmul_back_kernel_cuda(
    const float* grad, const float* gathered, float* dst,
    int M, int K, int E);
extern void per_edge_matmul_back_gathered_cuda(
    const float* grad, const float* kernel, float* dst,
    int M, int K, int E);

// ============================================================
// CUDABackend::dispatch_node
// ============================================================

Status CUDABackend::dispatch_node(TensorF32 * node, ComputeParams * p) {
    // CUDA 后端无线程池（单流异步 kernel），错误统一经局部 Status 上报。
    Status st = Status::SUCCESS;
    switch (node->op) {
        case OP_NONE:   break;

        case OP_DUP:
            // 整块拷贝 dst=src0（buffer-aware：D2D/H2D/D2H 四分支）
            kernel_dup_cuda(node);
            break;
        case OP_CPY:
        case OP_CONT:
            // 整块拷贝（含 OP_VIEW 零拷贝共享 src0）
            kernel_cpy_cuda(node, &st);
            break;

        case OP_FAPE:
            kernel_fape_cuda(node, &st);
            break;

        case OP_PERMUTE:
            // 通用维度重排（任意 dims 映射）
            kernel_permute_cuda(node, &st);
            break;

        case OP_TRANSPOSE:
            // 2D 用 tile XOR swizzle 专用 kernel；3D/4D 回落通用 permute
            kernel_transpose_cuda(node, &st);
            break;

        // ===== 有完整 .cu kernel 的 op =====
        case OP_ADD:
            kernel_elemwise_add_cuda(node, p);
            break;
        case OP_SUB:
            kernel_elemwise_sub_cuda(node, p);
            break;
        case OP_MUL:
            kernel_elemwise_mul_cuda(node, p);
            break;
        case OP_DIV:
            kernel_elemwise_div_cuda(node, p);
            break;

        case OP_MUL_MAT:
            kernel_mul_mat_cuda(node, p);
            break;

        case OP_OUT_PROD:
            kernel_out_prod_cuda(node, p);
            break;

        case OP_CONCAT:
            kernel_concat_cuda(node, &st);
            break;

        case OP_EDGE_GATHER_ROWS:
            kernel_edge_gather_rows_cuda(node, p);
            break;
        case OP_PER_EDGE_MATMUL:
            kernel_per_edge_matmul_cuda(node, p);
            break;
        case OP_SCATTER_ADD:
            kernel_scatter_add_cuda(node, p);
            break;
        case OP_PER_EDGE_MATMUL_BACK_KERNEL:
            kernel_per_edge_matmul_back_kernel_cuda(node, p);
            break;
        case OP_PER_EDGE_MATMUL_BACK_GATHERED:
            kernel_per_edge_matmul_back_gathered_cuda(node, p);
            break;

        case OP_SOFT_MAX:
            kernel_softmax_cuda(node, p);
            break;

        case OP_SOFT_MAX_BACK:
            kernel_softmax_back_cuda(node, p);
            break;

        // 【2026-09-22 Step ⑤】Flash Attention 融合算子（PPML_FLASH_ATTN=1 时由 Attention.cpp 建图 ✓）
        //   supports_op 已按 head_dim 上限把关（前向 d≤64 / 反向 d≤32 ✓）⇒ 超限的节点会派给 CPU ✓
        case OP_FLASH_ATTN_EXT:
            kernel_flash_attn_ext_cuda(node, p, &st);
            break;

        case OP_FLASH_ATTN_BACK:
            kernel_flash_attn_back_cuda(node, p, &st);
            break;

        case OP_NORM: {
            int C    = static_cast<int>(node->shape().dims[0]);
            int rows = static_cast<int>(node->numel() / C);

            float * out   = node->data();
            float * mean  = node->src[1] ? node->src[1]->data() : nullptr;
            float * rstd  = node->src[2] ? node->src[2]->data() : nullptr;
            const float * inp   = node->src[0]->data();
            const float * weight = nullptr;
            const float * bias   = nullptr;

            layernorm_forward_cuda(out, mean, rstd, inp, weight, bias, rows, 1, C, 256);
        } break;

        case OP_RMS_NORM: {
            // 沿 dims[0] (最内维) 归一化。dst = src / sqrt(mean(src^2)+eps)
            // supports_op 已保证: F32 + dims[0] % 32 == 0 (CUDA kernel warp 每行 32 线程瓜分)
            int C    = static_cast<int>(node->shape().dims[0]);
            int rows = static_cast<int>(node->numel() / C);
            const float * inp = node->src[0]->data();
            float * out       = node->data();
            const float eps   = reinterpret_cast<const float&>(node->op_params[0]);
            rms_norm_cuda(inp, out, C, rows, eps, /*stream=*/0);
        } break;

        case OP_NORM_BACK:
            kernel_norm_back_cuda(node, p);
            break;

        case OP_UNARY: {
            const unary_op uop = get_unary_op(node);
            kernel_unary_cuda(node, uop, p);
            break;
        }

        case OP_CLAMP:
            kernel_clamp_cuda(node, &st);
            break;

        case OP_SUM:
            kernel_sum_cuda(node, &st);
            break;

        case OP_MEAN:
            kernel_mean_cuda(node, &st);
            break;

        case OP_MAX_ALL:
            kernel_max_all_cuda(node, &st);
            break;

        case OP_SUM_ROWS:
            kernel_sum_rows_cuda(node, &st);
            break;

        case OP_RELU_BACK:
            kernel_relu_back_cuda(node, &st);
            break;

        case OP_ADD1:
            kernel_add1_cuda(node, &st);
            break;

        case OP_SQR:
            // 独立 op（非 unary）：dst = src^2（uop=16）
            if (!node->src[0] || !node->src[0]->data() || !node->data()) {
                st = Status::NOT_SUPPORTED;
                break;
            }
            unary_cuda(node->src[0]->data(), node->data(),
                       (int)node->numel(), 16, 256);
            break;

        case OP_SQRT:
            // 独立 op（非 unary）：dst = sqrt(src)（uop=15）
            if (!node->src[0] || !node->src[0]->data() || !node->data()) {
                st = Status::NOT_SUPPORTED;
                break;
            }
            unary_cuda(node->src[0]->data(), node->data(),
                       (int)node->numel(), 15, 256);
            break;

        case OP_LOG:
            // 独立 op（非 unary）：dst = log(src)（uop=14）
            if (!node->src[0] || !node->src[0]->data() || !node->data()) {
                st = Status::NOT_SUPPORTED;
                break;
            }
            unary_cuda(node->src[0]->data(), node->data(),
                       (int)node->numel(), 14, 256);
            break;

        case OP_SCALE:
            kernel_scale_cuda(node, &st);
            break;

        case OP_REPEAT:
            kernel_repeat_cuda(node, &st);
            break;

        case OP_REPEAT_BACK:
            kernel_repeat_back_cuda(node, &st);
            break;

        case OP_GET_ROWS: {
            // embedding 查表: W (N,M), idx(K,) → dst (K,M)
            // 布局对齐 CPU kernel_get_rows (CPUKernels.cpp:1928)
            const TensorF32* W   = node->src[0];
            const TensorF32* idx = node->src[1];
            if (!W || !idx || !W->data() || !idx->data()) {
                st = Status::NOT_SUPPORTED;
                break;
            }
            const int64_t N = W->shape().dims[0];
            const int64_t M = W->shape().dims[1];
            const int64_t K = idx->shape().dims[0];
            get_rows_cuda(W->data(), idx->data(), node->data(),
                          N, M, K, /*stream=*/0);
            st = Status::SUCCESS;
        } break;

        case OP_SET_ROWS: {
            // 散点覆写: a(N,M) 拷贝 + b[k] 行覆写 c[k]（对齐 CPU kernel_set_rows）
            // src0=a 目标, src1=b 行索引(K,), src2=c 值源(K,M), dst=(N,M)
            const TensorF32* a = node->src[0];
            const TensorF32* b = node->src[1];
            const TensorF32* c = node->src[2];
            if (!a || !b || !c || !a->data() || !b->data() || !c->data()) {
                st = Status::NOT_SUPPORTED;
                break;
            }
            const int64_t N = a->shape().dims[0];
            const int64_t M = a->shape().dims[1];
            const int64_t K = b->shape().dims[0];
            set_rows_cuda(a->data(), b->data(), c->data(), node->data(),
                          N, M, K);
            st = Status::SUCCESS;
        } break;

        case OP_GET_ROWS_BACK: {
            // 反向: dy(K,M), idx(K,) → dW(N,M) 清零+散点累加
            // 对齐 CPU kernel_get_rows_back (CPUKernels.cpp:1961)
            // src0=dy, src1=idx, src2=W(取形状 N,M), dst=dW
            const TensorF32* dy  = node->src[0];
            const TensorF32* idx = node->src[1];
            const TensorF32* W   = node->src[2];
            if (!dy || !idx || !W || !dy->data() || !idx->data()) {
                st = Status::NOT_SUPPORTED;
                break;
            }
            const int64_t N = W->shape().dims[0];
            const int64_t M = W->shape().dims[1];
            const int64_t K = idx->shape().dims[0];
            get_rows_back_cuda(dy->data(), idx->data(), node->data(),
                               N, M, K, /*stream=*/0);
            st = Status::SUCCESS;
        } break;

        // ===== 无实现的 op：统一返回 NOT_SUPPORTED =====
        default:
            if (getenv("GRAPH_DEBUG_CUDA_OP")) {
                fprintf(stderr, "[cuda-dispatch] UNSUPPORTED op=%d numel=%lld src0=%p src1=%p\n",
                        (int)node->op, (long long)node->numel(),
                        (void*)(node->src[0] ? node->src[0]->data() : nullptr),
                        (void*)(node->src[1] ? node->src[1]->data() : nullptr));
            }
            st = Status::NOT_SUPPORTED;
            break;
    }
    return st;
}

void CUDABackend::kernel_unary_cuda(TensorF32 * node, unary_op uop, ComputeParams * p) {
    (void)p;
    int N = static_cast<int>(node->numel());
    float * dst = node->data();
    const float * src = node->src[0]->data();
    unary_cuda(src, dst, N, static_cast<int>(uop), 256);
}

// ============================================================
// CUDA kernel stubs (待 src/cuda/*.cu 实现后替换)
// ============================================================

void CUDABackend::kernel_elemwise_add_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const int64_t n  = node->numel();
    const int64_t an = node->src[0]->numel();
    const int64_t bn = node->src[1]->numel();
    bcast_elemwise_cuda(node->src[0]->data(), node->src[1]->data(), node->data(),
                        n, an, bn, /*eop=*/0);
}

void CUDABackend::kernel_elemwise_sub_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const int64_t n  = node->numel();
    const int64_t an = node->src[0]->numel();
    const int64_t bn = node->src[1]->numel();
    bcast_elemwise_cuda(node->src[0]->data(), node->src[1]->data(), node->data(),
                        n, an, bn, /*eop=*/1);
}

void CUDABackend::kernel_elemwise_mul_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const int64_t n  = node->numel();
    const int64_t an = node->src[0]->numel();
    const int64_t bn = node->src[1]->numel();
    bcast_elemwise_cuda(node->src[0]->data(), node->src[1]->data(), node->data(),
                        n, an, bn, /*eop=*/2);
}

void CUDABackend::kernel_elemwise_div_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const int64_t n  = node->numel();
    const int64_t an = node->src[0]->numel();
    const int64_t bn = node->src[1]->numel();
    bcast_elemwise_cuda(node->src[0]->data(), node->src[1]->data(), node->data(),
                        n, an, bn, /*eop=*/3);
}

// doublecheck:
//原始 kernel 假设 B 是标准 row-major B[K][N]，访问 B[r * N + c]。但 PPML-Cpp 的 CPU 版本中 B 以转置形式存储：b[j * K + k]（即 B[N][K]）。
//因此加载 Bs 时改为：
void CUDABackend::kernel_mul_mat_cuda(TensorF32 * node, ComputeParams * p) {
    // node->shape().dims: output shape (N, M)  → d is (M, N) stored row-major
    // node->src[0]: A (M × K), row-major
    // node->src[1]: B stored transposed as (N × K), i.e. B[j * K + k]
    int M = static_cast<int>(node->shape().dims[1]);
    int N = static_cast<int>(node->shape().dims[0]);
    int K = static_cast<int>(node->src[0]->shape().dims[0]);

    float * A = node->src[0]->data();
    float * B = node->src[1]->data();
    float * C = node->data();

    // 防御（int8 量化准备，2026-09-10）：量化输入尚无 kernel（supports_op 已返回 false）；
    //   万一到达则打印并返回，避免把块结构当 float 读。
    if (is_quantized_type(node->src[0]->type) || is_quantized_type(node->src[1]->type)) {
        fprintf(stderr, "[MULMAT-QUANT] quantized input not implemented yet (src0=%d src1=%d), skip\n",
                (int)node->src[0]->type, (int)node->src[1]->type);
        return;
    }
    // 前期 fp16 支持（2026-09-10）：A/B 同为 F16 → half 输入 kernel（内部转 fp32 累加）。
    //   data() 返回 float*，但实际指向 16 位 half 数据（type_size_bytes()==2），按 void* 传递。
    //   优先 tensor-core（m16n8k16, fp32 累加）；硬件 < sm_80 时回落 SIMT half 版。
    if (node->src[0]->type == TENSOR_TYPE_F16 && node->src[1]->type == TENSOR_TYPE_F16) {
        const void* A16 = reinterpret_cast<const void*>(A);
        const void* B16 = reinterpret_cast<const void*>(B);
        // 优先级：流水版（smem+ldmatrix+cp.async）→ 简化 MMA → SIMT fp16
        const char* path = "tensor-core-smem-pipeline";
        mulmat_stats_cuda_begin(M, K, N);
        int rc = mul_mat_mma_f16_smem(A16, B16, C, M, K, N);
        if (rc != 0) {
            path = "tensor-core-mma-simple";
            rc = mul_mat_mma_f16(A16, B16, C, M, K, N);
        }
        if (rc != 0) {
            path = "fallback-simt-f16";
            mul_mat_cuda_f16(A16, B16, C, M, K, N);
        }
        mulmat_stats_cuda_end(path);
        if (getenv("GRAPH_DEBUG_MMA") && p && p->ith == 0) {
            fprintf(stderr, "[MMA] M=%d K=%d N=%d path=%s\n", M, K, N, path);
        }
        return;
    }

    // ===== 【2026-09-22】F32 → TF32 tensor core（Step 1）=====
    //   开关 PPML_MUL_MAT_TC=1（默认 0 = 纯 SIMT，与历史行为完全一致 ✓）。
    //   按用户要求**不设尺寸阈值**（小形状可能更慢，已接受 ✓）；仅当"非 sm_80+ / 尺寸非法"时
    //   回落 SIMT（由 mul_mat_mma_tf32_smem 内部判定并返回 1 ✓）。
    //   ⚠️ 反向的 mul_mat（compute_backward 也是用 mul_mat 拼的）同样经过这里 ⇒ 前后向一致 ✓。
    static const bool tc_on = []() {
        const char* e = std::getenv("PPML_MUL_MAT_TC");
        return e && e[0] && e[0] != '0';     // "0" = 关（默认）；其它值 = 开（TF32）
    }();
    if (tc_on) {
        mulmat_stats_cuda_begin(M, K, N);
        const int rc_tc = mul_mat_mma_tf32_smem(A, B, C, M, K, N);
        mulmat_stats_cuda_end(rc_tc == 0 ? "tf32-smem" : "tf32-unavailable");
        if (rc_tc == 0) {
            if (getenv("GRAPH_DEBUG_MMA") && p && p->ith == 0) {
                fprintf(stderr, "[MMA] M=%d K=%d N=%d path=tf32-smem\n", M, K, N);
            }
            return;
        }
    }

    mulmat_stats_cuda_begin(M, K, N);
    mul_mat_cuda(A, B, C, M, K, N);
    mulmat_stats_cuda_end("simt-128x128");
    if (getenv("GRAPH_DEBUG_MMA") && p && p->ith == 0) {
        fprintf(stderr, "[MMA] M=%d K=%d N=%d path=simt-128x128\n", M, K, N);
    }
}

void CUDABackend::kernel_softmax_cuda(TensorF32 * node, ComputeParams * p) {
    // node shape: (M, N) — M rows, N classes per row
    int N = static_cast<int>(node->shape().dims[0]);
    int M = static_cast<int>(node->numel() / N);
    float * input  = node->src[0]->data();
    float * output = node->data();
    // block size = 256
    softmax_cuda(input, output, M, N, 256);
}

void CUDABackend::kernel_softmax_back_cuda(TensorF32 * node, ComputeParams * p) {
    // src[0] = grad (upstream gradient dL/dy)
    // src[1] = output (softmax forward output y)
    int N = static_cast<int>(node->shape().dims[0]);
    int M = static_cast<int>(node->numel() / N);
    const float * grad   = node->src[0]->data();
    const float * output = node->src[1]->data();
    float *       dst    = node->data();
    // scale = 1.0f (standard softmax backward, no scaling)
    softmax_backward_cuda(grad, output, dst, M, N, 256, 1.0f);
}

// ============================================================
// 【2026-09-22 Step ⑤】Flash Attention 融合算子 —— CUDA 侧（包装 Step ②/③ 的 host 函数 ✓）
//   srcs/dst/op_params 约定与 CPU 侧同名函数完全一致（见 CPUKernels.cpp 注释 ✓）
//   布局：[d,L,h,B]（flash 布局 ✓，由 ComputeGraphHelper::flash_attn_ext 的 permute 保证 ✓）
//   rc != 0 只可能是"supports_op 的上限判定与实际不符"（不该发生 ✗）⇒ *st 上报，由 graph_compute 中止 ✓
// ============================================================
namespace {
FlashAttnConfig flash_attn_cfg_of_cuda(const TensorF32* const qkv[3], const TensorF32* node) {
    FlashAttnConfig cfg;
    const TensorF32* q = qkv[0];
    cfg.B = static_cast<int>(q->shape().dim(3));
    cfg.h = static_cast<int>(q->shape().dim(2));
    cfg.L = static_cast<int>(q->shape().dim(1));
    cfg.d = static_cast<int>(q->shape().dim(0));
    cfg.scale       = reinterpret_cast<const float&>(node->op_params[0]);
    cfg.causal      = node->op_params[1] != 0;
    cfg.softmax_eps = reinterpret_cast<const float&>(node->op_params[2]);
    cfg.Br          = node->op_params[3] > 0 ? node->op_params[3] : 64;
    cfg.Bc          = node->op_params[4] > 0 ? node->op_params[4] : 64;
    return cfg;
}
}  // namespace

void CUDABackend::kernel_flash_attn_ext_cuda(TensorF32 * node, ComputeParams * /*p*/, Status * st) {
    const TensorF32* qkv[3] = { node->src[0], node->src[1], node->src[2] };
    if (!qkv[0] || !qkv[1] || !qkv[2]) { if (st) *st = Status::NOT_SUPPORTED; return; }
    const FlashAttnConfig cfg = flash_attn_cfg_of_cuda(qkv, node);
    const float* bias = node->src[3] ? node->src[3]->data() : nullptr;
    float*       lse  = node->src[4] ? node->src[4]->data() : nullptr;
    const int rc = flash_attn_forward_cuda(cfg,
        qkv[0]->data(), qkv[1]->data(), qkv[2]->data(), bias, node->data(), lse);
    if (st) *st = (rc == 0) ? Status::SUCCESS : Status::NOT_SUPPORTED;
}

void CUDABackend::kernel_flash_attn_back_cuda(TensorF32 * node, ComputeParams * /*p*/, Status * st) {
    const TensorF32* qkv[3] = { node->src[0], node->src[1], node->src[2] };
    const TensorF32* O   = node->src[4];
    const TensorF32* dO  = node->src[5];
    const TensorF32* lse = node->src[6];
    if (!qkv[0] || !qkv[1] || !qkv[2] || !O || !dO || !lse) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    const FlashAttnConfig cfg = flash_attn_cfg_of_cuda(qkv, node);
    const float* bias = node->src[3] ? node->src[3]->data() : nullptr;
    float* dq = node->src[7] ? node->src[7]->data() : nullptr;
    float* dk = node->src[8] ? node->src[8]->data() : nullptr;
    float* dv = node->src[9] ? node->src[9]->data() : nullptr;
    float* db = (node->op_params[5] != 0) ? node->data() : nullptr;   // dbias 写 dst ✓
    const int rc = flash_attn_backward_cuda(cfg,
        qkv[0]->data(), qkv[1]->data(), qkv[2]->data(), bias,
        O->data(), dO->data(), lse->data(), dq, dk, dv, db);
    if (st) *st = (rc == 0) ? Status::SUCCESS : Status::NOT_SUPPORTED;
}

void CUDABackend::kernel_norm_back_cuda(TensorF32 * node, ComputeParams * p) {
    // src[0] = dout (upstream grad), src[1] = inp (original input)
    // src[2] = mean (cached), src[3] = rstd (cached)
    int C    = reinterpret_cast<int&>(node->op_params[0]);
    int rows = reinterpret_cast<int&>(node->op_params[1]);

    float *       dinp = node->data();
    const float * dout = node->src[0]->data();
    const float * inp  = node->src[1]->data();
    const float * mean = node->src[2]->data();
    const float * rstd = node->src[3]->data();

    //weight/bias 传 nullptr，因为 OP_NORM 不带 affine，gamma/beta 的梯度由上层 out_prod + add_impl 图节点自动求导。
    // OP_NORM_BACK 只计算 dinp，不计算 dweight/dbias（由上层 mul/add 图节点求导）
    layernorm_backward_cuda(
        dinp, nullptr, nullptr,   // dinp, dweight, dbias
        dout, inp, nullptr, mean, rstd,
        rows, 1, C, 256);
}

// 指针类型探测：CUDA 11+ 用 cudaPointerGetAttributes 的 type 字段。
// 与 Backend.cpp 的 is_device_pointer（cudaMemcpy 探测）相比无副作用（不污染错误状态）。
// 覆盖三种来源：gallocr 分配的 device buffer、bind_data 的 cudaMalloc 指针、host 常量。
static bool cuda_ptr_is_device(const float* p) {
    if (!p) return false;
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) return false;
    return attr.type == cudaMemoryTypeDevice;
}

void CUDABackend::kernel_dup_cuda(TensorF32 * node) {
    // OP_DUP：dst = src0 整块拷贝（对齐 CPU kernel_dup + buffer-aware 安全化）。
    // 混合调度下 src/dst 可能在 host 或 device，用指针类型探测选拷贝方向。
    if (!node->src[0] || !node->src[0]->data() || !node->data()) return;
    const bool src_dev = cuda_ptr_is_device(node->src[0]->data());
    const bool dst_dev = cuda_ptr_is_device(node->data());
    copy_tensor_cuda(node->src[0]->data(), node->data(), node->numel(), src_dev, dst_dev);
}

void CUDABackend::kernel_cpy_cuda(TensorF32 * node, Status* st) {
    // OP_CPY / OP_CONT：整块拷贝（对齐 CPU kernel_cpy）。OP_VIEW 零拷贝共享 src0。
    if (!node->src[0]) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    // OP_VIEW：不拷贝，直接共享 src0 的 data/buffer/offs（对齐 CPU kernel_cpy 的 view 分支）
    if (node->op == OP_VIEW) {
        TensorF32* src_t = node->src[0];
        if (src_t && src_t->data()) {
            node->bind_data(src_t->data());
            node->buffer_      = src_t->buffer_;
            node->buffer_offs_ = src_t->buffer_offs_;
        }
        if (st) *st = Status::SUCCESS;
        return;
    }
    if (!node->src[0]->data() || !node->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    const bool src_dev = cuda_ptr_is_device(node->src[0]->data());
    const bool dst_dev = cuda_ptr_is_device(node->data());
    copy_tensor_cuda(node->src[0]->data(), node->data(), node->numel(), src_dev, dst_dev);
    if (st) *st = Status::SUCCESS;
}

void CUDABackend::kernel_scale_cuda(TensorF32 * node, Status* st) {
    // OP_SCALE：dst = src * s（s 存 op_params[0] float 位模式，对齐 CPU kernel_scale）。
    // ggml scale_f32 参考版带 bias 参数，项目 scale() helper 无 bias（仅标量乘），故省略。
    if (!node->src[0] || !node->src[0]->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    const float s   = reinterpret_cast<const float&>(node->op_params[0]);
    const float * src = node->src[0]->data();
    float       * dst = node->data();

    scale_cuda(src, dst, s, node->numel());
    if (st) *st = Status::SUCCESS;
}

void CUDABackend::kernel_clamp_cuda(TensorF32 * node, Status* st) {
    // OP_CLAMP：dst[i] = clamp(src[i], lo, hi)。lo/hi 以 float 位模式存 op_params[0]/[1]
    // （由 clamp() 构造器写入，对齐 CPU kernel_clamp）。NaN 透传，与 CPU 语义一致。
    if (!node->src[0] || !node->src[0]->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    const float lo = reinterpret_cast<const float&>(node->op_params[0]);
    const float hi = reinterpret_cast<const float&>(node->op_params[1]);
    clamp_cuda(node->src[0]->data(), node->data(),
               (int)node->numel(), lo, hi, 256);
    if (st) *st = Status::SUCCESS;
}

void CUDABackend::kernel_add1_cuda(TensorF32 * node, Status* st) {
    // OP_ADD1：dst = src + b（b 为 src[1] 标量张量，device 指针 → D2H 读 b[0]，
    // 对齐 CPU kernel_add1 的 host 读取语义）。
    if (!node->src[0] || !node->src[1] || !node->src[0]->data() || !node->src[1]->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    float b = 0.f;
    cudaMemcpy(&b, node->src[1]->data(), sizeof(float), cudaMemcpyDeviceToHost);  // 同步小拷贝
    add1_cuda(node->src[0]->data(), node->data(), b, node->numel());
    if (st) *st = Status::SUCCESS;
}

void CUDABackend::kernel_sum_cuda(TensorF32 * node, Status* st) {
    // OP_SUM：全元素求和 → 标量 [1]
    // src0 全元素 grid-stride + warp shuffle 两级规约，block 间 atomicAdd 到 dst[0]。
    if (!node->src[0] || !node->src[0]->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    // dst 必须先清零（atomicAdd 累加）。对标 out_prod_cuda 的 cudaMemset 模式。
    cudaMemset(node->data(), 0, node->nbytes());
    const float* src = node->src[0]->data();
    const long long n = node->src[0]->numel();
    sum_cuda(src, node->data(), n);
    if (st) *st = Status::SUCCESS;
}

void CUDABackend::kernel_relu_back_cuda(TensorF32 * node, Status* st) {
    // OP_RELU_BACK：dst[i] = (x[i]>0) ? grad[i] : 0（src0=grad, src1=x，
    // 对齐 CPU kernel_relu_back；x=0 边界梯度置 0，与 PyTorch 同约定）。
    if (!node->src[0] || !node->src[1] ||
        !node->src[0]->data() || !node->src[1]->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    relu_back_cuda(node->src[0]->data(), node->src[1]->data(),
                   node->data(), node->numel());
    if (st) *st = Status::SUCCESS;
}

void CUDABackend::kernel_fape_cuda(TensorF32 * node, Status* st) {
    // OP_FAPE 前向：FAPE 结构损失（对齐 CPU compute_forward_fape）。
    // src0=pred_coords [3,N_atoms] src1=true_coords [3,N_atoms]
    // src2=frame_indices [3,N_frames] src3=frames_mask [1,N_frames] src4=positions_mask [1,N_atoms]
    if (!node->src[0] || !node->src[1] || !node->src[2] ||
        !node->src[3] || !node->src[4] ||
        !node->src[0]->data() || !node->src[1]->data() || !node->src[2]->data() ||
        !node->src[3]->data() || !node->src[4]->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    const int64_t N_atoms  = node->src[0]->shape().dims[1];
    const int64_t N_frames = node->src[2]->shape().dims[1];
    const float d_clamp    = reinterpret_cast<const float&>(node->op_params[0]);
    const float epsilon    = reinterpret_cast<const float&>(node->op_params[2]);
    const float length_scale = reinterpret_cast<const float&>(node->op_params[4]);

    fape_cuda(node->src[0]->data(), node->src[1]->data(), node->src[2]->data(),
              node->src[3]->data(), node->src[4]->data(),
              node->data(), N_atoms, N_frames, d_clamp, epsilon, length_scale);
    if (st) *st = Status::SUCCESS;
}

void CUDABackend::kernel_transpose_cuda(TensorF32 * node, Status* st) {
    // OP_TRANSPOSE：2D 用 tile+XOR swizzle 专用 kernel（对齐用户 transposeSharedSwizzling）；
    //   3D/4D（交换倒数两维）回落通用 permute（用 op_params 映射）。
    if (!node->src[0] || !node->src[0]->data() || !node->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    const TensorF32* a = node->src[0];
    if (a->shape().ndim() == 2) {
        // M=行数=dims[1]，N=行长度=dims[0]（dims[0]=最内）
        const int64_t M = a->shape().dims[1];
        const int64_t N = a->shape().dims[0];
        transpose_cuda(a->data(), node->data(), M, N);
        if (st) *st = Status::SUCCESS;
    } else {
        kernel_permute_cuda(node, st);   // 3D/4D 走通用映射
    }
}

void CUDABackend::kernel_permute_cuda(TensorF32 * node, Status* st) {
    // OP_PERMUTE/OP_TRANSPOSE：通用维度重排（对齐 CPU kernel_permute）。
    // op_params 存 dims 映射（int32）；dst 第 p 维 = src 第 dims[p] 维。
    if (!node->src[0] || !node->src[0]->data() || !node->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    const TensorF32* a = node->src[0];
    const int ndim = a->shape().ndim();
    int32_t dims[4] = {0, 1, 2, 3};
    for (int i = 0; i < ndim && i < 4; i++) dims[i] = node->op_params[i];
    // 防御：op_params 未初始化（垃圾）会导致 i[dims[p]] 越界 → SIGSEGV
    for (int i = 0; i < ndim && i < 4; i++) {
        if (dims[i] < 0 || dims[i] >= ndim) {
            if (st) *st = Status::NOT_SUPPORTED;
            return;
        }
    }
    int64_t sd[4] = {1, 1, 1, 1};
    int64_t dd[4] = {1, 1, 1, 1};
    for (int i = 0; i < ndim && i < 4; i++) {
        sd[i] = a->shape().dims[i];
        dd[i] = node->shape().dims[i];
    }
    permute_cuda(a->data(), node->data(), node->numel(), ndim,
                 dims[0], dims[1], dims[2], dims[3],
                 sd[0], sd[1], sd[2], sd[3],
                 dd[0], dd[1], dd[2], dd[3]);
    if (st) *st = Status::SUCCESS;
}

void CUDABackend::kernel_max_all_cuda(TensorF32 * node, Status* st) {
    // OP_MAX_ALL：全元素归约 max → 标量 [1]（对齐 CPU kernel_max_all：跳 NaN）。
    if (!node->src[0] || !node->src[0]->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    max_all_cuda(node->src[0]->data(), node->data(), node->src[0]->numel());
    if (st) *st = Status::SUCCESS;
}

void CUDABackend::kernel_sum_rows_cuda(TensorF32 * node, Status* st) {
    // OP_SUM_ROWS：沿最内维 dims[0] 归约，输出 {1, dims[1..3]}（对齐 CPU kernel_sum_rows）。
    // block-per-row：ncols = ne0（行长度），nrows = ne1*ne2*ne3（行数）。
    if (!node->src[0] || !node->src[0]->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    const TensorF32* src = node->src[0];
    const int64_t ne0 = src->shape().dims[0];
    const int64_t ne1 = (src->shape().ndim() > 1) ? src->shape().dims[1] : 1;
    const int64_t ne2 = (src->shape().ndim() > 2) ? src->shape().dims[2] : 1;
    const int64_t ne3 = (src->shape().ndim() > 3) ? src->shape().dims[3] : 1;

    sum_rows_cuda(src->data(), node->data(), ne0, ne1 * ne2 * ne3);
    if (st) *st = Status::SUCCESS;
}

void CUDABackend::kernel_mean_cuda(TensorF32 * node, Status* st) {
    // OP_MEAN：全元素平均 → 标量 [1]
    // 复用 OP_SUM 两级规约 kernel，最终标量乘 1/N（Σ(val_b/N) = Σ(val_b)/N）。
    if (!node->src[0] || !node->src[0]->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    // dst 必须先清零（atomicAdd 累加）。对标 out_prod_cuda 的 cudaMemset 模式。
    cudaMemset(node->data(), 0, node->nbytes());
    const float* src = node->src[0]->data();
    const long long n = node->src[0]->numel();
    mean_cuda(src, node->data(), n);
    if (st) *st = Status::SUCCESS;
}

void CUDABackend::kernel_repeat_back_cuda(TensorF32 * node, Status* st) {
    // OP_REPEAT_BACK：dst[j] = Σ_k src[j + k*dd]（repeat 的逆，梯度和）。
    // 与 CPU kernel_repeat_back 对齐；仅支持同 ndim（src 各维 = dst 各维 × 整数重复因子），
    // 跨 ndim 广播（src 前导维多于 dst）由 supports_op 判 false 回落 CPU。
    if (!node->src[0] || !node->src[0]->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    const TensorF32* src0 = node->src[0];
    TensorF32*       dst  = node;

    const int64_t ne00 = src0->shape().dims[0];
    const int64_t ne01 = (src0->shape().ndim() > 1) ? src0->shape().dims[1] : 1;
    const int64_t ne02 = (src0->shape().ndim() > 2) ? src0->shape().dims[2] : 1;
    const int64_t ne03 = (src0->shape().ndim() > 3) ? src0->shape().dims[3] : 1;

    const int64_t ne0 = dst->shape().dims[0];
    const int64_t ne1 = (dst->shape().ndim() > 1) ? dst->shape().dims[1] : 1;
    const int64_t ne2 = (dst->shape().ndim() > 2) ? dst->shape().dims[2] : 1;
    const int64_t ne3 = (dst->shape().ndim() > 3) ? dst->shape().dims[3] : 1;

    // 整除前置校验（同 ndim 下应成立；否则回落 CPU 避免除零/越界）
    if (ne0 <= 0 || ne1 <= 0 || ne2 <= 0 || ne3 <= 0 ||
        ne00 % ne0 != 0 || ne01 % ne1 != 0 || ne02 % ne2 != 0 || ne03 % ne3 != 0) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }

    repeat_back_cuda(
        src0->data(), dst->data(),
        ne00, ne01, ne02, ne03,
        ne0,  ne1,  ne2,  ne3);
    if (st) *st = Status::SUCCESS;
}

void CUDABackend::kernel_repeat_cuda(TensorF32 * node, Status* st) {
    // OP_REPEAT 前向：dst[j] = src[s]，s_d = j_d % src_dims[d]。
    // 与 CPU kernel_repeat 对齐；src[1] 仅为形状模板（不读数据）。
    // 跨 ndim 广播（src 缺维/维=1）由 kernel 内"缺维取模 1"统一处理。
    if (!node->src[0] || !node->src[0]->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    const TensorF32* src0 = node->src[0];
    TensorF32*       dst  = node;

    const int64_t ne00 = src0->shape().dims[0];
    const int64_t ne01 = (src0->shape().ndim() > 1) ? src0->shape().dims[1] : 1;
    const int64_t ne02 = (src0->shape().ndim() > 2) ? src0->shape().dims[2] : 1;
    const int64_t ne03 = (src0->shape().ndim() > 3) ? src0->shape().dims[3] : 1;

    const int64_t ne0 = dst->shape().dims[0];
    const int64_t ne1 = (dst->shape().ndim() > 1) ? dst->shape().dims[1] : 1;
    const int64_t ne2 = (dst->shape().ndim() > 2) ? dst->shape().dims[2] : 1;
    const int64_t ne3 = (dst->shape().ndim() > 3) ? dst->shape().dims[3] : 1;

    // 校验（2026-09-06）：CPU kernel_repeat (CPUKernels.cpp:1666-1680) 是逐维取模
    //   s_d = i_d % ne0[d]（尾部对齐广播），无任何 dst≥src / 整除要求，永不越界。
    //   CUDA repeat_f32_kernel 与之逐行等价。此前 9/4 的 "dst 每维 ≥ src" 校验仍
    //   拒绝合法形状（如 src[256,51]→dst[64,51,8]，dst 最内维反而更小，CPU 照算）→
    //   dispatch NOT_SUPPORTED → GPU split 整体中断回退 CPU。取模天然把坐标钳制在
    //   src 界内，故删除一切维度大小比较，仅保留基本存在性检查。
    if (ne0 <= 0 || ne1 <= 0 || ne2 <= 0 || ne3 <= 0 ||
        ne00 <= 0 || ne01 <= 0 || ne02 <= 0 || ne03 <= 0) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }

    repeat_cuda(
        src0->data(), dst->data(),
        ne00, ne01, ne02, ne03,
        ne0,  ne1,  ne2,  ne3);
    if (st) *st = Status::SUCCESS;
}

void CUDABackend::kernel_set_rows_cuda(TensorF32 * node, Status* st) {
    // OP_SET_ROWS 前向：dst = a 全量拷贝 + 按 b[k] 覆写 c[k] 行（散点覆写）。
    // 对齐 CPU kernel_set_rows (CPUKernels.cpp:1873)；严格 2D(N,M)。
    // 重复索引预扫在 set_rows_cuda 内（宿主侧 D2H，对齐 CPU "有重复则跳过 scatter"）。
    const TensorF32* a = node->src[0];
    const TensorF32* b = node->src[1];
    const TensorF32* c = node->src[2];
    if (!a || !b || !c || !a->data() || !b->data() || !c->data()) {
        if (st) *st = Status::NOT_SUPPORTED;
        return;
    }
    const int64_t N = a->shape().dims[0];
    const int64_t M = a->shape().dims[1];
    const int64_t K = b->shape().dims[0];

    set_rows_cuda(a->data(), b->data(), c->data(), node->data(),
                  N, M, K);
    if (st) *st = Status::SUCCESS;
}

void CUDABackend::kernel_relu_cuda(TensorF32 * node) { (void)node; }
void CUDABackend::kernel_sigmoid_cuda(TensorF32 * node) { (void)node; }
void CUDABackend::kernel_silu_cuda(TensorF32 * node) { (void)node; }
void CUDABackend::kernel_tanh_cuda(TensorF32 * node) { (void)node; }
void CUDABackend::kernel_exp_cuda(TensorF32 * node) { (void)node; }

void CUDABackend::kernel_out_prod_cuda(TensorF32 * node, ComputeParams * p) {
    const TensorF32* src0 = node->src[0];
    const TensorF32* src1 = node->src[1];

    const int64_t ne00 = src0->shape().dims[0];
    const int64_t ne01 = src0->shape().dims[1];
    const int64_t ne02 = (src0->shape().ndim() > 2) ? src0->shape().dims[2] : 1;
    const int64_t ne03 = (src0->shape().ndim() > 3) ? src0->shape().dims[3] : 1;

    const int64_t ne10 = src1->shape().dims[0];
    const int64_t ne11 = src1->shape().dims[1];
    const int64_t ne12 = (src1->shape().ndim() > 2) ? src1->shape().dims[2] : 1;
    const int64_t ne13 = (src1->shape().ndim() > 3) ? src1->shape().dims[3] : 1;

    const int64_t ne0 = node->shape().dims[0];
    const int64_t ne1 = node->shape().dims[1];
    const int64_t ne2 = (node->shape().ndim() > 2) ? node->shape().dims[2] : 1;
    const int64_t ne3 = (node->shape().ndim() > 3) ? node->shape().dims[3] : 1;

    // GQA: ne2/ne02, ne3/ne03 暂不计算 dps 参数
    // const int64_t dps2 = ne2 / ne02;
    // const int64_t dps3 = ne3 / ne03;

    // 先将 dst 清零（对标 CPU 版 thread 0 的清零逻辑）
    cudaMemset(node->data(), 0, node->nbytes());

    out_prod_cuda(
        src0->data(), src1->data(), node->data(),
        ne00, ne01, ne02, ne03,
        ne10, ne11, ne12, ne13,
        ne0,  ne1,  ne2,  ne3);
}

void CUDABackend::kernel_edge_gather_rows_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const TensorF32* node_feat = node->src[0];  // dims=[C, N]
    const TensorF32* src_idx   = node->src[1];  // (E,)
    const int C = static_cast<int>(node_feat->shape().dims[0]);
    const int N = static_cast<int>(node_feat->shape().dims[1]);
    const int E = static_cast<int>(src_idx->shape().dims[0]);
    edge_gather_rows_cuda(node_feat->data(), src_idx->data(), node->data(), N, C, E);
}

void CUDABackend::kernel_per_edge_matmul_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const TensorF32* kernel   = node->src[0];  // dims=[K, M, E]
    const TensorF32* gathered = node->src[1];  // dims=[K, E]
    const int K = static_cast<int>(kernel->shape().dims[0]);
    const int M = static_cast<int>(kernel->shape().dims[1]);
    const int E = static_cast<int>(kernel->shape().dims[2]);
    per_edge_matmul_cuda(kernel->data(), gathered->data(), node->data(), M, K, E);
}

void CUDABackend::kernel_scatter_add_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const TensorF32* msg      = node->src[0];  // dims=[M, E]
    const TensorF32* tgt_idx  = node->src[1];  // (E,)
    const int N = node->op_params[0];
    const int M = static_cast<int>(msg->shape().dims[0]);
    const int E = static_cast<int>(msg->shape().dims[1]);
    //  诊断（GRAPH_DEBUG_CUDA_SCATTER=1）：打印 scatter_add 实际 launch 参数与指针状态，
    //    定位 [CUDA-ERR] op=108(OP_SCATTER_ADD) invalid argument 根因（grid 超限 / 指针非法）。
    if (getenv("GRAPH_DEBUG_CUDA_SCATTER")) {
        const int gz = (int)(((int64_t)N * M + 255) / 256);
        const int ge = (int)(((int64_t)E + 255) / 256);
        fprintf(stderr,
                "[scatter-add-dbg] N=%d M=%d E=%d gz=%d ge=%d | dst=%p msg=%p tgt=%p "
                "dst_buf=%p(is_host=%d) msg_buf=%p(is_host=%d) tgt_buf=%p(is_host=%d)\n",
                N, M, E, gz, ge,
                (void*)node->data(), (void*)msg->data(), (void*)tgt_idx->data(),
                (void*)(node->buffer_?node->buffer_:nullptr), (node->buffer_?(int)node->buffer_->is_host():-1),
                (void*)(msg->buffer_?msg->buffer_:nullptr), (msg->buffer_?(int)msg->buffer_->is_host():-1),
                (void*)(tgt_idx->buffer_?tgt_idx->buffer_:nullptr), (tgt_idx->buffer_?(int)tgt_idx->buffer_->is_host():-1));
        //  指针属性检查：确认 data() 在 GPU launch 时是 device 指针（cudaMemoryTypeDevice=2）。
        //    若 msg/tgt/dst 的 data() 是 host 或 unregistered（跨后端 data()/buffer_ 不一致），
        //    CUDA kernel 读 host 指针 → invalid argument / illegal address。
        cudaPointerAttributes pa_dst{}, pa_msg{}, pa_tgt{};
        cudaPointerGetAttributes(&pa_dst, node->data());
        cudaPointerGetAttributes(&pa_msg, msg->data());
        cudaPointerGetAttributes(&pa_tgt, tgt_idx->data());
        fprintf(stderr,
                "[scatter-ptr] dst type=%d(2=dev,1=host,3=managed,0=unreg) msg type=%d tgt type=%d\n",
                (int)pa_dst.type, (int)pa_msg.type, (int)pa_tgt.type);
    }
    //  诊断（GRAPH_DEBUG_CUDA_SCATTER=1）：进入 scatter 前先清一次错误——
    //    区分"H2D 拷贝等前置遗留错误"（被 cudaGetLastError 残留捕获）与 scatter 自身 launch 错误。
    if (getenv("GRAPH_DEBUG_CUDA_SCATTER")) {
        cudaError_t prior = cudaGetLastError();
        if (prior != cudaSuccess) {
            fprintf(stderr, "[scatter-prior-ERR] BEFORE scatter launch: %s\n",
                    cudaGetErrorString(prior));
        }
    }
    scatter_add_cuda(msg->data(), tgt_idx->data(), node->data(), N, M, E);
    //  诊断（GRAPH_DEBUG_CUDA_SCATTER=1）：launch 后立即查错误——确认 scatter 自身
    //    launch 是否真的失败（node 0 的 [CUDA-ERR] op=108 来源）。
    if (getenv("GRAPH_DEBUG_CUDA_SCATTER")) {
        cudaError_t ler = cudaGetLastError();
        if (ler != cudaSuccess) {
            fprintf(stderr,
                    "[scatter-launch-ERR] N=%d M=%d E=%d gz=%d: %s\n",
                    N, M, E,
                    E > 0 ? ((E + 255) / 256) : 1, cudaGetErrorString(ler));
            cudaGetLastError();  // 清错误
        }
    }
}

void CUDABackend::kernel_per_edge_matmul_back_kernel_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const TensorF32* grad     = node->src[0];  // dims=[M, E]
    const TensorF32* gathered = node->src[1];  // dims=[K, E]
    const int M = static_cast<int>(grad->shape().dims[0]);
    const int E = static_cast<int>(grad->shape().dims[1]);
    const int K = static_cast<int>(gathered->shape().dims[0]);
    per_edge_matmul_back_kernel_cuda(grad->data(), gathered->data(), node->data(), M, K, E);
}

void CUDABackend::kernel_per_edge_matmul_back_gathered_cuda(TensorF32 * node, ComputeParams * p) {
    (void)p;
    const TensorF32* grad   = node->src[0];  // dims=[M, E]
    const TensorF32* kernel = node->src[1];  // dims=[K, M, E]
    const int M = static_cast<int>(grad->shape().dims[0]);
    const int E = static_cast<int>(grad->shape().dims[1]);
    const int K = static_cast<int>(kernel->shape().dims[0]);
    per_edge_matmul_back_gathered_cuda(grad->data(), kernel->data(), node->data(), M, K, E);
}

void CUDABackend::kernel_concat_cuda(TensorF32 * node, Status* st) {
    const int dim = node->op_params[0];
    if (dim < 0 || dim >= 4) { if (st) *st = Status::NOT_SUPPORTED; return; }

    // src 为固定大小数组(GGML_MAX_SRC)，仅统计非空输入，空位跳过
    int n_src = 0;
    TensorF32* srcs_tmp[GGML_MAX_SRC];
    for (int s = 0; s < GGML_MAX_SRC; s++) {
        if (node->src[s]) srcs_tmp[n_src++] = node->src[s];
    }
    if (n_src < 2) { if (st) *st = Status::NOT_SUPPORTED; return; }

    // ============================================================
    // concat 不支持广播语义：强制要求所有 src 的非拼接维与 dst 一致
    //（即 dst 任意非拼接维不得大于 src 对应维）。
    // 不满足则直接置 NOT_SUPPORTED，避免在 CUDA kernel 内发生越界读。
    // ============================================================
    const int nd = node->shape().ndim();
    for (int d = 0; d < 4; d++) {
        if (d == dim) continue; // 拼接维可不同，由 start/len 表处理
        const int64_t dst_d = (d < nd) ? node->shape().dims[d] : 1;
        for (int s = 0; s < n_src; s++) {
            const int      sd    = srcs_tmp[s]->shape().ndim();
            const int64_t  src_d = (d < sd) ? srcs_tmp[s]->shape().dims[d] : 1;
            if (src_d != dst_d) {
                if (st) *st = Status::NOT_SUPPORTED;
                return;
            }
        }
    }

    // dst 形状
    const int64_t ne0 = node->shape().ndim() > 0 ? node->shape().dims[0] : 1;
    const int64_t ne1 = node->shape().ndim() > 1 ? node->shape().dims[1] : 1;
    const int64_t ne2 = node->shape().ndim() > 2 ? node->shape().dims[2] : 1;
    const int64_t ne3 = node->shape().ndim() > 3 ? node->shape().dims[3] : 1;

    // 构造 host 侧 start/len 偏移表与 src 指针数组
    std::vector<int64_t>    len(n_src), start(n_src, 0);
    std::vector<const float*> srcs(n_src);
    for (int s = 0; s < n_src; s++) {
        len[s]   = srcs_tmp[s]->shape().dims[dim];
        srcs[s]  = srcs_tmp[s]->data();
    }
    for (int s = 1; s < n_src; s++) start[s] = start[s - 1] + len[s - 1];

    // 拷贝到 device
    const float** d_srcs  = nullptr;
    int64_t*      d_start = nullptr;
    int64_t*      d_len   = nullptr;
    cudaMalloc(&d_srcs,  sizeof(float*)   * n_src);
    cudaMalloc(&d_start, sizeof(int64_t)  * n_src);
    cudaMalloc(&d_len,   sizeof(int64_t)  * n_src);
    cudaMemcpy(d_srcs,  srcs.data(),  sizeof(float*)  * n_src, cudaMemcpyHostToDevice);
    cudaMemcpy(d_start, start.data(), sizeof(int64_t) * n_src, cudaMemcpyHostToDevice);
    cudaMemcpy(d_len,   len.data(),   sizeof(int64_t) * n_src, cudaMemcpyHostToDevice);

    concat_nary_cuda(
        d_srcs, n_src, d_start, d_len, node->data(), dim,
        ne0, ne1, ne2, ne3);

    cudaFree(d_srcs);
    cudaFree(d_start);
    cudaFree(d_len);
}

} // namespace ppml

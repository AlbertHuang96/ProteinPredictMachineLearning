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
    const TensorF32* src2 = node->src[2];
    const TensorF32* src3 = node->src[3];
    const TensorF32* src4 = node->src[4];

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
    //  OP_PERMUTE/OP_TRANSPOSE 已有真重排 kernel（2026-09-01），在下方 switch 支持。
    if (node->op == OP_NONE    || node->op == OP_RESHAPE ||
        node->op == OP_VIEW) {
        return true;
    }

    switch (node->op) {
        // ===== 有完整 .cu kernel 实现的 op =====
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
            return true;

        case OP_CLAMP:
            if (!src0) return false;
            return src0->type == TENSOR_TYPE_F32 && node->type == TENSOR_TYPE_F32;

        case OP_SQR:
        case OP_SQRT:
        case OP_LOG:
            // 2026-09-01: 独立逐元素 op（非 unary）CUDA 接入：dst=src^2 / sqrt / log。
            //   前置: F32 + src0 存在。
            if (!src0) return false;
            return src0->type == TENSOR_TYPE_F32 && node->type == TENSOR_TYPE_F32;

        case OP_PERMUTE:
        case OP_TRANSPOSE:
            // 2026-09-01: 通用维度重排 CUDA kernel（对齐 CPU kernel_permute，
            //   op_params 存 dims 映射）。此前无 kernel → graph_compute skip 执行 +
            //   supports_op false 强制回落 CPU（loss 链因此无法全 GPU）。现实现后支持。
            if (!src0) return false;
            return src0->type == TENSOR_TYPE_F32 && node->type == TENSOR_TYPE_F32;

        case OP_MUL_MAT: {
            // 前期 fp16 支持（2026-09-10）：A/B 同为 F16 可调度到 GPU（kernel 内转 fp32 累加），
            //   输出仍 F32。混合类型或输出非 F32 一律回落 CPU。
            if (!src1) return true;
            // GEMM epilogue 融合（2026-10-18）：bias 只在 SIMT f32 路径实现 ⇒ 带 bias 时
            //   禁止量化/F16 权重走 GPU（否则 kernel 会跳过 bias，静默错值）→ 回落 CPU。
            if (node->src[2] &&
                ((src0 && src0->type != TENSOR_TYPE_F32) || src1->type != TENSOR_TYPE_F32)) {
                return false;
            }
            // int8 分块量化（2026-09-10 准备）：kernel 未就绪 → 不参与 GPU 调度（回落 CPU，
            //   而 CPU 侧同样返回 false → 该节点整体不被任何后端支持，需等 kernel 落地）。
            if (is_quantized_type(src1->type) || (src0 && is_quantized_type(src0->type))) {
                // 形状合法性：量化张量的最内维须为块元素数整数倍（量化块不能跨行）
                const bool shape_ok =
                    (src0 != nullptr) &&
                    (!is_quantized_type(src0->type) || src0->quant_shape_valid()) &&
                    (!is_quantized_type(src1->type) || src1->quant_shape_valid());
                // W8A8（2026-10-17）：只支持「量化权重 × F32 激活」，且**只允许一侧量化**。
                //   量化权重在 src0（ggml 约定 mul_mat(W,x)）或 src1（本仓库 LinearLayer 的
                //   mul_mat(x,W)）都接受 —— 激活由 kernel 内即时量化成 Q8_1。
                const bool q0 = is_quantized_type(src0->type);
                const bool q1 = is_quantized_type(src1->type);
                if (q0 == q1) return false;                      // 两侧都量化 / 都不量化（前已处理）
                const TensorF32* wsrc = q0 ? src0 : src1;        // 量化权重
                const TensorF32* asrc = q0 ? src1 : src0;        // F32 激活
                return QUANT_MUL_MAT_CUDA_READY && shape_ok &&
                       is_quantized_type_supported(wsrc->type) &&
                       asrc->type == TENSOR_TYPE_F32 &&
                       node->type == TENSOR_TYPE_F32;
            }
            const bool ok_f32 = (src0 && src0->type == TENSOR_TYPE_F32 &&
                                 src1->type == TENSOR_TYPE_F32);
            const bool ok_f16 = (src0 && src0->type == TENSOR_TYPE_F16 &&
                                 src1->type == TENSOR_TYPE_F16);
            return (ok_f32 || ok_f16) && node->type == TENSOR_TYPE_F32;
        }

        case OP_OUT_PROD:
            if (!src0 || !src1) return false;
            return src0->type == TENSOR_TYPE_F32 &&
                   src1->type == TENSOR_TYPE_F32 &&
                   node->type == TENSOR_TYPE_F32;

        case OP_CONCAT: {
            // 【2026-10-24】PPML_CUDA_NO_CONCAT=1：强制 CONCAT 回落 CPU。
            //   实测（RTX 2050 / L=51 / MSA=8 / 3 blocks）：训练图里 split=444 是一个以
            //   CONCAT 打头的 CUDA split，launch 后报非法访存；kernel 自身的 [concat-oob]
            //   与 host 指针守卫都没触发 ⇒ 需要这个开关做 A/B 定位（也作为安全回落 ✓）。
            if (getenv("PPML_CUDA_NO_CONCAT") && std::strcmp(getenv("PPML_CUDA_NO_CONCAT"), "1") == 0) {
                return false;
            }
            // N-ary concat：遍历所有 src。
            // 注意：concat 不支持广播语义，非拼接维必须与 dst 一致，
            // 该约束在 kernel_concat_cuda 内强制校验。
            //   ⚠ none 槽（src[2..GGML_MAX_SRC-1] 默认 nullptr）必须跳过：早期写成
            //     `if (!s || s->type != F32) return false`，任何 n-ary 节点（空槽存在）
            //     都被判成不支持 ⇒ CONCAT 永远回落 CPU + 单后端模式刷 [CUDA-UNSUPPORTED]
            //     误报（2026-10-24 修）。
            //   ⚠ 单源 concat（n_src==1）不是非法输入：GMABSE3/GSE3Res 对**单度 fiber**
            //     调 concat_ptr(k_nodes, 0) 时就只有 1 个源，其语义是 identity（dst 与 src
            //     同形，builder 的 ne[dim] 只累加了一个 src）。CPU kernel_concat 早已按
            //     「memcpy 直通」处理；原判据写成 n_src >= 2 ⇒ 实测训练图里 472 个该节点
            //     （3.74e7 元素）被判不支持、整片回落 CPU（2026-10-24 修，与 CPU 对齐 ✓）。
            int n_src = 0;
            for (const auto* s : node->src) {
                if (!s) continue;
                if (s->type != TENSOR_TYPE_F32) return false;
                ++n_src;
            }
            // PPML_CUDA_NO_CONCAT1=1：只把**单源（identity）** concat 赶回 CPU，用于判别
            //   「concat 的 CPU 实现数值偏差」vs「混合调度换分区本身就改变数值」——
            //   identity 拷贝在数学上不可能改值，若 loss 因此变动，责任只在分区/暂存侧。
            if (n_src == 1) {
                const char* nc1 = std::getenv("PPML_CUDA_NO_CONCAT1");
                if (nc1 && std::atoi(nc1) == 1) return false;
            } else {
                const char* ncn = std::getenv("PPML_CUDA_NO_CONCATN");   // 多源
                if (ncn && std::atoi(ncn) == 1) return false;
            }
            return n_src >= 1 && node->type == TENSOR_TYPE_F32;
        }

        // 【2026-10-24】concat 反向：dst[k] = grad[k + offset*stride]（平移拷贝）
        //   实测训练图里有 2886 个 CONCAT_BACK 节点、1.09e8 元素，原来全落 CPU。
        //   判据只需类型 + 越界可行性（不看指针，放置期数据可能尚未分配 ✓）
        case OP_CONCAT_BACK: {
            // A/B 开关（与 PPML_CUDA_NO_CONCAT 对称）：强制回落 CPU，用于定位数值差异 ✓
            if (getenv("PPML_CUDA_NO_CONCAT_BACK") &&
                std::strcmp(getenv("PPML_CUDA_NO_CONCAT_BACK"), "1") == 0) {
                return false;
            }
            TensorF32* g = node->src[0];
            if (!g || g->type != TENSOR_TYPE_F32) return false;
            if (node->type != TENSOR_TYPE_F32) return false;
            const int dim = node->op_params[0];
            const int offset = node->op_params[1];
            if (dim < 0 || dim >= 4 || offset < 0) return false;
            // 越界可行性（与 kernel 侧同式）：最大索引 = Σ (ne[d] - (d==dim) + offset) * gs[d]
            int64_t ne[4] = {1, 1, 1, 1};
            for (int d = 0; d < node->shape().ndim() && d < 4; ++d) {
                ne[d] = node->shape().dims[static_cast<size_t>(d)];
            }
            int64_t gs[4] = {1, 1, 1, 1};
            for (int d = 1; d < g->shape().ndim() && d < 4; ++d) {
                gs[d] = gs[d - 1] * g->shape().dims[static_cast<size_t>(d - 1)];
            }
            int64_t max_idx = 0;
            for (int d = 0; d < 4; ++d) {
                const int64_t hi = (d == dim) ? (ne[d] + offset) : ne[d];
                max_idx += (hi > 0 ? hi - 1 : 0) * gs[d];
            }
            return node->numel() > 0 && max_idx < g->numel();
        }

        case OP_SOFT_MAX:
            return true;

        case OP_EDGE_GATHER_ROWS:
        case OP_PER_EDGE_MATMUL:
        case OP_SCATTER_ADD:
        case OP_PER_EDGE_MATMUL_BACK_KERNEL:
        case OP_PER_EDGE_MATMUL_BACK_GATHERED:
            //  诊断开关（PPML_CUDA_NO_SCATTER=1）：强制 SCATTER_ADD 回落 CPU，
            //    验证 [CUDA-ERR] op=108(OP_SCATTER_ADD) invalid argument 是否为
            //    混合 SE3 链数值不稳的根因（若 CUDA-ERR 消失且 loss 正常 → 是根因；
            //    若 loss 仍爆炸 → 非根因，是 SE3 前向放大）。
            if (node->op == OP_SCATTER_ADD &&
                getenv("PPML_CUDA_NO_SCATTER") &&
                std::strcmp(getenv("PPML_CUDA_NO_SCATTER"), "1") == 0) {
                return false;
            }
            //  诊断开关（PPML_CUDA_NO_PEM=1）：强制 PER_EDGE_MATMUL 回落 CPU（± 反向两核）。
            //    动机：混合调度下强制 CONCAT 落 CPU 会让 split 结构剧变（84→683），预算逐出集合
            //    随之改变 —— 实测 A(13.88) 里 op=107 被逐出 944 次走 CPU，B(14.58) 里它改跑 GPU。
            //    用它把"GPU 侧 PER_EDGE_MATMUL 是否数值不对"一次问清。
            if ((node->op == OP_PER_EDGE_MATMUL ||
                 node->op == OP_PER_EDGE_MATMUL_BACK_KERNEL ||
                 node->op == OP_PER_EDGE_MATMUL_BACK_GATHERED) &&
                getenv("PPML_CUDA_NO_PEM") &&
                std::strcmp(getenv("PPML_CUDA_NO_PEM"), "1") == 0) {
                return false;
            }
            return true;

        case OP_SOFT_MAX_BACK:
            if (!src0 || !src1) return false;
            return src0->type == TENSOR_TYPE_F32 &&
                   src1->type == TENSOR_TYPE_F32;

        case OP_GET_ROWS:
            // 2026-08-29: CUDA kernel (get_rows_cuda) 已实现, embedding 查表。
            // 前置: W/idx/dst 均 F32; idx 须 1D; W 须 2D (N,M)。
            // 越界钳制 (i<0→0, i>=N→N-1) 已在 kernel 内处理, 无需额外约束。
            if (!src0 || !src1) return false;
            if (src0->type != TENSOR_TYPE_F32 || src1->type != TENSOR_TYPE_F32 ||
                node->type != TENSOR_TYPE_F32) return false;
            return src1->shape().ndim() == 1 && src0->shape().ndim() == 2;

        case OP_GET_ROWS_BACK:
            // 2026-08-29: CUDA kernel (get_rows_back_cuda) 已实现, 反向散点累加。
            // 前置: dy/idx/W 均 F32; idx 须 1D; W 须 2D (N,M)。越界丢弃已在 kernel 内处理。
            if (!src0 || !src1 || !src2) return false;
            if (src0->type != TENSOR_TYPE_F32 || src1->type != TENSOR_TYPE_F32 ||
                src2->type != TENSOR_TYPE_F32 || node->type != TENSOR_TYPE_F32) return false;
            return src1->shape().ndim() == 1 && src2->shape().ndim() == 2;

        case OP_NORM:
            return true;

        case OP_NORM_BACK:
            return true;

        case OP_RMS_NORM:
            // 2026-08-28: CUDA kernel (rms_norm_cuda) 已实现, 沿 dims[0] 归一化。
            // 前置: F32 + 最内维长度 % 32 == 0 (warp 每行 32 线程瓜分)。
            // 非 32 倍数由 supports_op 返回 false → 整个 op 回落到 CPU 后端,
            // 不能在 CUDA dispatch 内联 CPU 计算 (device 指针不可在 host 解引用)。
            if (!src0 || src0->type != TENSOR_TYPE_F32) return false;
            return node->shape().dims[0] % 32 == 0;

        case OP_SUM:
        case OP_MEAN:
            // 2026-08-25：OP_SUM/OP_MEAN 全元素归约已实现（warp shuffle 两级规约 + atomicAdd，
            // mean 复用同一 kernel 最终标量乘 1/N）
            return true;

        case OP_REPEAT:
            // 2026-08-31：OP_REPEAT 前向 CUDA kernel（简化版：一维 grid-stride +
            // 尾部对齐取模，对齐 CPU kernel_repeat）。仅需 F32 + src0 存在；
            // 跨 ndim 广播（src 缺维/维=1）由 kernel 内"缺维取模 1"统一处理，
            // 无需 REPEAT_BACK 的同 ndim/整除限制（整除性由 repeat() helper 保证，
            // dispatch 内仍作防御性校验）。
            if (!src0) return false;
            return src0->type == TENSOR_TYPE_F32 && node->type == TENSOR_TYPE_F32;

        case OP_REPEAT_BACK: {
            // 2026-08-27：OP_REPEAT_BACK CUDA kernel（每线程一 dst 元素，串行累加所有重复拷贝）。
            // 2026-10-24：新增跨 ndim 通用版 repeat_back4_cuda（四重循环 + 维度对齐规则完全
            //   照抄 CPU kernel_repeat_back，含 src 前导广播维整体累加）⇒ 不再要求 nd_src==nd_dst。
            //   实测该情形占残差大头（正向 372 / 反向 372 个节点，numel=57621）。
            if (!src0) return false;
            if (src0->type != TENSOR_TYPE_F32 || node->type != TENSOR_TYPE_F32) return false;
            const int nd_src = src0->shape().ndim();
            const int nd_dst = node->shape().ndim();
            if (nd_src > 4 || nd_dst > 4) return false;   // kernel 只解 4 维（与 CPU 一致）
            // 分流在 dispatch：同 ndim 且各维整数倍 → 原 kernel（数值逐位不变）；
            // 跨 ndim / 非整数倍 → 通用 4 维 kernel（维度对齐与归约次数照抄 CPU）。
            return true;
        }

        case OP_SET_ROWS:
            // 2026-08-31: OP_SET_ROWS CUDA kernel（散点覆写，对齐 CPU kernel_set_rows）。
            // 前置: a/c/dst 均 F32; a 须 2D(N,M); b 须 1D(K,); c 须 2D 且 (K,M)。
            if (!src0 || !src1 || !src2) return false;
            if (src0->type != TENSOR_TYPE_F32 || src1->type != TENSOR_TYPE_F32 ||
                src2->type != TENSOR_TYPE_F32 || node->type != TENSOR_TYPE_F32) return false;
            if (src0->shape().ndim() != 2 || src1->shape().ndim() != 1 || src2->shape().ndim() != 2)
                return false;
            return src0->shape().dims[0] == src2->shape().dims[0] &&  // N
                   src0->shape().dims[1] == src2->shape().dims[1];     // M

        case OP_SCALE:
            // 2026-08-31: OP_SCALE CUDA kernel（dst = src*s，s 在 op_params[0]）。
            // 前置: F32 + src0 存在。
            if (!src0) return false;
            return src0->type == TENSOR_TYPE_F32 && node->type == TENSOR_TYPE_F32;

        case OP_ADD1:
            // OP_ADD1 CUDA kernel（dst = src + b）。
            // 2026-10-19: b 有两种形态 ——
            //   ① src[1]==nullptr：建图期常量已烤进 op_params[0]（add1_impl），kernel 作参数收；
            //   ② src[1] 为 1 元素标量张量：kernel 直接读其 device 指针（不再 D2H 回读）。
            // 前置: F32 + src0 存在。
            if (!src0) return false;
            if (src0->type != TENSOR_TYPE_F32 || node->type != TENSOR_TYPE_F32) return false;
            if (!src1) return true;                       // 烤入常量
            return src1->type == TENSOR_TYPE_F32 && src1->numel() == 1;

        case OP_MAX_ALL:
            // 2026-08-31: OP_MAX_ALL CUDA kernel（全元素归约 max → 标量，跳 NaN）。
            // 前置: F32 + src0 存在。
            if (!src0) return false;
            return src0->type == TENSOR_TYPE_F32 && node->type == TENSOR_TYPE_F32;

        case OP_SUM_ROWS:
            // 2026-08-31: OP_SUM_ROWS CUDA kernel（block-per-row，沿最内维归约）。
            // 前置: F32 + src0 存在。
            if (!src0) return false;
            return src0->type == TENSOR_TYPE_F32 && node->type == TENSOR_TYPE_F32;

        case OP_RELU_BACK:
            // 2026-08-31: OP_RELU_BACK CUDA kernel（relu 梯度：x>0 透传 grad，否则 0）。
            // 前置: F32 + src0/src1 存在。
            if (!src0 || !src1) return false;
            return src0->type == TENSOR_TYPE_F32 && src1->type == TENSOR_TYPE_F32 &&
                   node->type == TENSOR_TYPE_F32;

        case OP_ACT_MUL:
        case OP_ACT_MUL_BACK_X:
        case OP_ACT_MUL_BACK_G:
            // 门控融合（2026-10-18）：F32 逐元素；src1(src2) 可小于 dst（kernel 内 i % n 广播，
            //   语义与 OP_MUL 的 CUDA 广播核一致）。act 种类在 op_params[0]。
            if (!src0 || !src1) return false;
            return src0->type == TENSOR_TYPE_F32 && src1->type == TENSOR_TYPE_F32 &&
                   node->type == TENSOR_TYPE_F32;

        case OP_FAPE:
            // 2026-09-01: OP_FAPE CUDA kernel（FAPE 结构损失）。前置: F32 + 5 输入存在
            //   （pred/true coords、frame_indices、frames_mask、positions_mask）。
            if (!src0 || !src1 || !src2 || !src3 || !src4) return false;
            return src0->type == TENSOR_TYPE_F32 && node->type == TENSOR_TYPE_F32;

        // ===== pair/MSA 专用算子（2026-10-24，kernel 在 src/cuda/PairKernels.cu）=====
        case OP_TRI_MUL:
            // 三角乘法：left/right 与 dst 均 4D F32
            //   outgoing left[D,I,K,B]·right[D,J,K,B]→dst[D,I,J,B]；incoming 为 left[D,K,I,B]
            if (!src0 || !src1) return false;
            if (src0->type != TENSOR_TYPE_F32 || src1->type != TENSOR_TYPE_F32 ||
                node->type != TENSOR_TYPE_F32) return false;
            return src0->shape().ndim() == 4 && src1->shape().ndim() == 4 &&
                   node->shape().ndim() == 4;

        case OP_TRI_MUL_BACK:
            // 多输出反向：grad_left(src[3]) / grad_right(src[4]) 至少其一，kernel 内一并写
            if (!src0 || !src1 || !src2) return false;
            if (!node->src[3] && !node->src[4]) return false;
            return src0->type == TENSOR_TYPE_F32 && node->type == TENSOR_TYPE_F32;

        case OP_OUTER_PROD_MEAN:
            // msa2pair：left/right [D,L,N,B] → dst [D*D,L,L,B]
            if (!src0 || !src1) return false;
            if (src0->type != TENSOR_TYPE_F32 || src1->type != TENSOR_TYPE_F32 ||
                node->type != TENSOR_TYPE_F32) return false;
            return src0->shape().ndim() == 4 && src1->shape().ndim() == 4 &&
                   src0->shape().dims[0] > 0 && src0->shape().dims[2] > 0;

        case OP_OUTER_PROD_MEAN_BACK:
            if (!src0 || !src1 || !src2) return false;
            if (!node->src[3] && !node->src[4]) return false;
            return src0->type == TENSOR_TYPE_F32 && node->type == TENSOR_TYPE_F32;

        case OP_OUTER_PROD:
            // pair2pair 门控：left/right [D,L,B] → dst [D*D,L,L,B]
            if (!src0 || !src1) return false;
            if (src0->type != TENSOR_TYPE_F32 || src1->type != TENSOR_TYPE_F32 ||
                node->type != TENSOR_TYPE_F32) return false;
            return src0->shape().ndim() == 3 && src1->shape().ndim() == 3;

        case OP_OUTER_PROD_BACK:
            if (!src0 || !src1 || !src2) return false;
            if (!node->src[3] && !node->src[4]) return false;
            return src0->type == TENSOR_TYPE_F32 && node->type == TENSOR_TYPE_F32;

        case OP_FAPE_BACK:
            // FAPE 反向（Phase 1）：src[0]=上游标量 … src[5]=positions_mask，共 6 个输入
            if (!src0 || !src1 || !src2 || !src3 || !src4 || !node->src[5]) return false;
            if (src1->shape().ndim() != 2 || src3->shape().ndim() != 2) return false;
            return src0->type == TENSOR_TYPE_F32 && node->type == TENSOR_TYPE_F32;

        case OP_DUP:
        case OP_CPY:
        case OP_CONT:
            // 2026-09-01: OP_DUP/OP_CPY/OP_CONT CUDA kernel（整块拷贝，buffer-aware，
            //   经 cudaPointerGetAttributes 选 D2D/H2D/D2H）。跨后端 cpy 节点不再回落 CPU。
            //   前置: F32 + src0 存在。
            if (!src0) return false;
            return src0->type == TENSOR_TYPE_F32 && node->type == TENSOR_TYPE_F32;

        // ===== kernel 为空函数体或 NOT_SUPPORTED，暂不支持 =====
        // UNARY_OP_*  → kernel_relu/gelu/sigmoid/silu/tanh/exp_cuda 均为空函数体

        // ===== Flash Attention（Step ⑤）：按 head_dim 上限决定留 CUDA 还是回落 CPU =====
        //   ⚠️ 上限必须与 src/cuda/FlashAttnKernel.cu 内的一致：前向 d ≤ 64、反向 d ≤ 32（寄存器约束 ✗）
        //   超限 ⇒ 返回 false ⇒ 调度器（Backend.cpp set_backend_if_supported）自动把该节点派给 CPU ✓
        //   也可用现成的 PPML_CUDA_DISABLE_OPS=<op 号> 强制本 op 回落（做 A/B 用 ✓）
        case OP_FLASH_ATTN_EXT: {
            if (!src0 || !src1 || !src2) return false;
            if (src0->type != TENSOR_TYPE_F32) return false;
            const int d = static_cast<int>(src0->shape().dim(0));   // flash 布局 dims[0] = d ✓
            // ⚠️【②-v2】前向改 lane 切分后要求 d 是 4 的倍数（每 lane 的切片要 16B 对齐 ✓）
            //   不满足 ⇒ 返回 false ⇒ 调度器把该节点派给 CPU ✓（CPU 版无此限制 ✓）
            return d > 0 && d <= 64 && (d % 4) == 0;
        }
        case OP_FLASH_ATTN_BACK: {
            if (!src0) return false;
            if (src0->type != TENSOR_TYPE_F32) return false;
            const int d = static_cast<int>(src0->shape().dim(0));
            // ⚠️【②-v2】反向同样 lane 切分 ⇒ 也要求 d 是 4 的倍数 ✓（不满足 ⇒ 派给 CPU ✓）
            return d > 0 && d <= 32 && (d % 4) == 0;
        }

        // ===== 未实现的 op =====
        case OP_CROSS_ENTROPY_LOSS:
        default:
            return false;

        case OP_UNARY: {
            // 2026-08-24：unary CUDA 默认禁用（跨后端 H2D 未可靠 → chi 链 relu 读 CPU 指针当 device
            //   → chi loss 巨大 185万）。2026-08-31 已验证开启后训练正常（2 epoch 数值与关闭一致）→
            //   默认开启，无需环境变量；PPML_CUDA_UNARY=0 可显式回退禁用。
            if (getenv("PPML_CUDA_UNARY") && std::strcmp(getenv("PPML_CUDA_UNARY"), "0") == 0) {
                return false;
            }
            // 仅支持已在 unary_cuda kernel 实现的 subtype
            const unary_op uop = get_unary_op(node);
            switch (uop) {
                case UNARY_OP_ABS:
                case UNARY_OP_RELU:
                case UNARY_OP_GELU:
                case UNARY_OP_SILU:
                case UNARY_OP_TANH:
                case UNARY_OP_SIGMOID:
                case UNARY_OP_EXP:
                case UNARY_OP_LOG:
                case UNARY_OP_SQRT:
                    return true;
                default:
                    return false;
            }
        }
    }
}

// ===== synchronize =====
void CUDABackend::synchronize() {
    cudaSetDevice(device_id_);
    cudaDeviceSynchronize();
}

// ===== graph_compute =====
// ============================================================
// 单后端全 GPU：host 输入的设备影子（2026-10-23）
// ============================================================
//   scheduler 模式下，build_splits 的 3c 分支会为"host 生产者 → GPU 消费者"建
//   H2D cpy 节点（Backend.cpp:2449-2490）；**单后端全 GPU 路径没有 scheduler**，
//   没有人做这次搬运 ⇒ 图内独有的 host 叶子（GET_ROWS 的权重 W、常量 mask 等）
//   在 kernel 里是 host 指针 ⇒ 守卫判 HOST/INVALID 直接 NOT_SUPPORTED，
//   上层把该节点算作"未执行"，loss 变成垃圾（实测 13.6 → -3.00，无任何报错）。
//   这里补上这次搬运：每个 host 输入建一份设备影子，compute 结束释放并恢复原字段。
bool CUDABackend::ensure_host_shadow(TensorF32* t) {
    if (!t || !t->data()) return false;
    // 已是 device ⇒ 无需影子（有 buffer 看 is_host；无 buffer 用属性探测兜底）
    if (t->buffer_) {
        if (!t->buffer_->is_host()) return true;
    } else {
        cudaPointerAttributes attr{};
        const bool probe_ok =
            (cudaPointerGetAttributes(&attr, t->data()) == cudaSuccess) &&
            (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged);
        if (probe_ok) return true;
        cudaGetLastError();   // 清掉探测产生的错误，避免污染后续 kernel 检查
        // 探测失败 = host 指针（叶子 data_ 由 bind_leaf_data 绑到 host，buffer_ 为空）
        // ⇒ **继续往下建影子**（不能 return false：op=41 的 src1 索引张量正是这种形态）
    }

    auto it = host_shadow_.find(t);
    if (it == host_shadow_.end()) {
        size_t bytes = static_cast<size_t>(t->nbytes());
        if (bytes == 0) bytes = static_cast<size_t>(t->numel()) * sizeof(float);
        if (bytes == 0) return false;
        void* dev = nullptr;
        if (cudaMalloc(&dev, bytes) != cudaSuccess) { cudaGetLastError(); return false; }
        if (cudaMemcpy(dev, t->data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError();
            cudaFree(dev);
            return false;
        }
        host_shadow_[t] = dev;
        host_shadow_saved_.push_back(HostShadowSaved{t, t->data(), t->buffer_, t->buffer_offs_});
        if (getenv("GRAPH_DEBUG_CUDA_SHADOW")) {
            fprintf(stderr, "[cuda-shadow] tensor=%p op=%d numel=%lld bytes=%zu → dev=%p\n",
                    (void*)t, (int)t->op, (long long)t->numel(), bytes, dev);
        }
        it = host_shadow_.find(t);
    }
    // 重绑到影子：buffer_ 必须清空 —— 否则守卫的 INCONSISTENT 判定
    // （"data 是 device 但 buffer_->is_host()==true"）会把这份影子误报成 view 错位。
    t->bind_data(it->second);
    t->buffer_      = nullptr;
    t->buffer_offs_ = 0;
    return true;
}

void CUDABackend::release_host_shadows() {
    for (auto& s : host_shadow_saved_) {
        if (!s.t) continue;
        s.t->bind_data(s.data);
        s.t->buffer_      = s.buf;
        s.t->buffer_offs_ = s.offs;
    }
    for (auto& kv : host_shadow_) {
        if (kv.second) cudaFree(kv.second);
    }
    host_shadow_.clear();
    host_shadow_saved_.clear();
}

// 全量算子名表（由 build/gen_names.awk 从 include/ppml/Tensor.h 的 enum tensor_op 生成，
//   2026-10-24 补齐：原表只覆盖十来个"CUDA 无 kernel"的算子，其余打印成 "?"，
//   导致 [CUDA-UNSUPPORTED] / [OP-CHECK] 难以对号 ✓）
static const char* cuda_op_name(int op) {
    switch (op) {
        case OP_NONE:                        return "NONE";
        case OP_DUP:                         return "DUP";
        case OP_ADD:                         return "ADD";
        case OP_ADD_ID:                      return "ADD_ID";
        case OP_ADD1:                        return "ADD1";
        case OP_ACC:                         return "ACC";
        case OP_SUB:                         return "SUB";
        case OP_MUL:                         return "MUL";
        case OP_DIV:                         return "DIV";
        case OP_SQR:                         return "SQR";
        case OP_SQRT:                        return "SQRT";
        case OP_LOG:                         return "LOG";
        case OP_SIN:                         return "SIN";
        case OP_COS:                         return "COS";
        case OP_SUM:                         return "SUM";
        case OP_SUM_ROWS:                    return "SUM_ROWS";
        case OP_CUMSUM:                      return "CUMSUM";
        case OP_MEAN:                        return "MEAN";
        case OP_ARGMAX:                      return "ARGMAX";
        case OP_COUNT_EQUAL:                 return "COUNT_EQUAL";
        case OP_REPEAT:                      return "REPEAT";
        case OP_REPEAT_BACK:                 return "REPEAT_BACK";
        case OP_CONCAT:                      return "CONCAT";
        case OP_SILU_BACK:                   return "SILU_BACK";
        case OP_NORM:                        return "NORM";
        case OP_NORM_BACK:                   return "NORM_BACK";
        case OP_RMS_NORM:                    return "RMS_NORM";
        case OP_RMS_NORM_BACK:               return "RMS_NORM_BACK";
        case OP_GROUP_NORM:                  return "GROUP_NORM";
        case OP_L2_NORM:                     return "L2_NORM";
        case OP_MUL_MAT:                     return "MUL_MAT";
        case OP_MUL_MAT_ID:                  return "MUL_MAT_ID";
        case OP_OUT_PROD:                    return "OUT_PROD";
        case OP_SCALE:                       return "SCALE";
        case OP_SET:                         return "SET";
        case OP_CPY:                         return "CPY";
        case OP_CONT:                        return "CONT";
        case OP_RESHAPE:                     return "RESHAPE";
        case OP_VIEW:                        return "VIEW";
        case OP_PERMUTE:                     return "PERMUTE";
        case OP_TRANSPOSE:                   return "TRANSPOSE";
        case OP_GET_ROWS:                    return "GET_ROWS";
        case OP_GET_ROWS_BACK:               return "GET_ROWS_BACK";
        case OP_SET_ROWS:                    return "SET_ROWS";
        case OP_DIAG:                        return "DIAG";
        case OP_DIAG_MASK_INF:               return "DIAG_MASK_INF";
        case OP_DIAG_MASK_ZERO:              return "DIAG_MASK_ZERO";
        case OP_SOFT_MAX:                    return "SOFT_MAX";
        case OP_SOFT_MAX_BACK:               return "SOFT_MAX_BACK";
        case OP_ROPE:                        return "ROPE";
        case OP_ROPE_BACK:                   return "ROPE_BACK";
        case OP_CLAMP:                       return "CLAMP";
        case OP_CONV_TRANSPOSE_1D:           return "CONV_TRANSPOSE_1D";
        case OP_IM2COL:                      return "IM2COL";
        case OP_IM2COL_BACK:                 return "IM2COL_BACK";
        case OP_IM2COL_3D:                   return "IM2COL_3D";
        case OP_COL2IM_1D:                   return "COL2IM_1D";
        case OP_CONV_2D:                     return "CONV_2D";
        case OP_CONV_3D:                     return "CONV_3D";
        case OP_CONV_2D_DW:                  return "CONV_2D_DW";
        case OP_CONV_TRANSPOSE_2D:           return "CONV_TRANSPOSE_2D";
        case OP_POOL_1D:                     return "POOL_1D";
        case OP_POOL_2D:                     return "POOL_2D";
        case OP_POOL_2D_BACK:                return "POOL_2D_BACK";
        case OP_UPSCALE:                     return "UPSCALE";
        case OP_PAD:                         return "PAD";
        case OP_PAD_REFLECT_1D:              return "PAD_REFLECT_1D";
        case OP_ROLL:                        return "ROLL";
        case OP_ARANGE:                      return "ARANGE";
        case OP_TIMESTEP_EMBEDDING:          return "TIMESTEP_EMBEDDING";
        case OP_ARGSORT:                     return "ARGSORT";
        case OP_TOP_K:                       return "TOP_K";
        case OP_LEAKY_RELU:                  return "LEAKY_RELU";
        case OP_TRI:                         return "TRI";
        case OP_FILL:                        return "FILL";
        case OP_FLASH_ATTN_EXT:              return "FLASH_ATTN_EXT";
        case OP_FLASH_ATTN_BACK:             return "FLASH_ATTN_BACK";
        case OP_SSM_CONV:                    return "SSM_CONV";
        case OP_SSM_SCAN:                    return "SSM_SCAN";
        case OP_WIN_PART:                    return "WIN_PART";
        case OP_WIN_UNPART:                  return "WIN_UNPART";
        case OP_GET_REL_POS:                 return "GET_REL_POS";
        case OP_ADD_REL_POS:                 return "ADD_REL_POS";
        case OP_RWKV_WKV6:                   return "RWKV_WKV6";
        case OP_GATED_LINEAR_ATTN:           return "GATED_LINEAR_ATTN";
        case OP_RWKV_WKV7:                   return "RWKV_WKV7";
        case OP_SOLVE_TRI:                   return "SOLVE_TRI";
        case OP_GATED_DELTA_NET:             return "GATED_DELTA_NET";
        case OP_UNARY:                       return "UNARY";
        case OP_MAP_CUSTOM1:                 return "MAP_CUSTOM1";
        case OP_MAP_CUSTOM2:                 return "MAP_CUSTOM2";
        case OP_MAP_CUSTOM3:                 return "MAP_CUSTOM3";
        case OP_CUSTOM:                      return "CUSTOM";
        case OP_CROSS_ENTROPY_LOSS:          return "CROSS_ENTROPY_LOSS";
        case OP_CROSS_ENTROPY_LOSS_BACK:     return "CROSS_ENTROPY_LOSS_BACK";
        case OP_OPT_STEP_ADAMW:              return "OPT_STEP_ADAMW";
        case OP_OPT_STEP_SGD:                return "OPT_STEP_SGD";
        case OP_GLU:                         return "GLU";
        case OP_FAPE:                        return "FAPE";
        case OP_FAPE_BACK:                   return "FAPE_BACK";
        case OP_TRI_MUL:                     return "TRI_MUL";
        case OP_TRI_MUL_BACK:                return "TRI_MUL_BACK";
        case OP_OUTER_PROD_MEAN:             return "OUTER_PROD_MEAN";
        case OP_OUTER_PROD_MEAN_BACK:        return "OUTER_PROD_MEAN_BACK";
        case OP_OUTER_PROD:                  return "OUTER_PROD";
        case OP_OUTER_PROD_BACK:             return "OUTER_PROD_BACK";
        case OP_EDGE_GATHER_ROWS:            return "EDGE_GATHER_ROWS";
        case OP_PER_EDGE_MATMUL:             return "PER_EDGE_MATMUL";
        case OP_SCATTER_ADD:                 return "SCATTER_ADD";
        case OP_PER_EDGE_MATMUL_BACK_KERNEL: return "PER_EDGE_MATMUL_BACK_KERNEL";
        case OP_PER_EDGE_MATMUL_BACK_GATHERED:   return "PER_EDGE_MATMUL_BACK_GATHERED";
        case OP_CONCAT_BACK:                 return "CONCAT_BACK";
        case OP_MAX_ALL:                     return "MAX_ALL";
        case OP_RELU_BACK:                   return "RELU_BACK";
        case OP_ACT_MUL:                     return "ACT_MUL";
        case OP_ACT_MUL_BACK_X:              return "ACT_MUL_BACK_X";
        case OP_ACT_MUL_BACK_G:              return "ACT_MUL_BACK_G";
        default:                         return "?";
    }
}

// 公开入口（见 Backend.h）：诊断代码据此把 op 编号打印成名字 ✓
const char* CUDABackend::op_name(int op) { return cuda_op_name(op); }

Status CUDABackend::graph_compute(ComputeGraph* cgraph) {
    cudaSetDevice(device_id_);
    // 诊断：标记"当前图"（子图编号见 exec_set_current_graph 注释）
    exec_set_current_graph(cgraph);

    // 生产者代际断言（PPML_STALE_CHECK=1）：每轮执行起一个新代际，并登记本图 node/leaf。
    //   scheduler 驱动 split 时 sched_depth>0 ⇒ 此处不另起代际（由 scheduler 统一起）。
    exec_stale_pass_begin(cgraph);

    // ---- no_alloc 延迟分配：对 data_==nullptr 的中间节点分配 GPU buffer ----
    // 混训时 scheduler 已预分配，置 skip_alloc_=true 跳过，避免两套 gallocr 冲突。
    if (!skip_alloc_) {
        // ---- needs_realloc 接入（2026-08-31，单后端 GPU）：布局未变则复用 buffer ----
        // 同 CPU：can_reuse（已分配过 && has_snapshot && !needs_realloc）成立则复用
        // （不 release、不重建，只重绑 data）。注意 can_reuse 须在 release 前判断——
        // 原逻辑先 release 再按 data()==nullptr 判定，release 已清空 data() 恒 need_alloc。
        // 回退开关：PPML_NO_GALLOCR_REUSE=1 时走原逻辑（每轮 release + 判定 + 重建）。
        bool gallocr_reuse = true;
        if (getenv("PPML_NO_GALLOCR_REUSE")) {
            gallocr_reuse = (std::strcmp(getenv("PPML_NO_GALLOCR_REUSE"), "1") != 0);
        }
        if (gallocr_reuse && gallocr_.can_reuse(cgraph)) {
            gallocr_.backends()[0].buft = const_cast<BufferType*>(buffer_type());
            auto backend_id_of = [](TensorF32*) -> int { return 0; };
            if (!gallocr_.alloc(cgraph, backend_id_of, 1)) {
                return Status::ALLOC_FAILED;
            }
        } else {
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
    }

    // 设备影子作用域：任何返回路径（含 NOT_SUPPORTED / ALLOC 失败）都会恢复张量原字段
    struct HostShadowScope {
        CUDABackend* self;
        ~HostShadowScope() { self->release_host_shadows(); }
    } host_shadow_scope{this};

    int cuda_dispatch_cnt = 0;
    int cuda_dispatch_op = -1;
    for (int node_n = 0; node_n < cgraph->n_nodes(); node_n++) {
        TensorF32* node = cgraph->graph_node(node_n);

        //  前置异步错误检查（GRAPH_DEBUG_CUDA_ASYNC=1）：CUDA 错误是异步滞留的——
        //    launch 返回时 kernel 可能未执行完，错误到下一个 cudaGetLastError 才暴露。
        //    此前 [CUDA-ERR] op=108(node 0) 打印的是"检查点节点"而非真正失败的 kernel。
        //    此处在本节点 launch 前检查，捕获上一个节点的真实失败。配合 cudaDeviceSynchronize
        //    确保错误已发生（慢，仅诊断用）。
        if (getenv("GRAPH_DEBUG_CUDA_ASYNC")) {
            cudaError_t perr = cudaDeviceSynchronize();
            if (perr != cudaSuccess) {
                fprintf(stderr,
                        "[CUDA-ASYNC-ERR] PRIOR kernel failed before node_n=%d op=%d: %s\n",
                        node_n, (int)node->op, cudaGetErrorString(perr));
                cudaGetLastError();  // 清错误
            }
        }

        // 跳过 no-op（OP_PERMUTE/OP_TRANSPOSE 已有真 kernel，2026-09-01 起不再跳过）
        if (node->op == OP_NONE || node->op == OP_VIEW ||
            node->op == OP_RESHAPE) {
            continue;
        }

        // 解析 view src：view（view_src 共享源数据）被跳过 dispatch 后 data() 为 null，
        // 但其作为本 op 的 src 时 kernel 要读数据。沿 view_src 链解析到底层数据载体，
        // 让本 op 的 kernel 拿到真实 device 指针（否则读 null → [CUDA-ERR]/illegal）。
        // 【2026-10-24 方案 A】兼容"只置 src[0]、没置 view_src"的 OP_VIEW/OP_RESHAPE
        //   （reshape()/unsqueeze() 就是这么建的）：这类节点被 scheduler 当 view 跳过 backend
        //   指派 ⇒ gallocr 按默认 CPU 分配**私有 host buffer**，由 CPU split 的 kernel_cpy 写入。
        //   若其消费者在本 CUDA split 内，内核直接读它 = 裸读 host ⇒
        //   [CUDA-GUARD] 放弃整图（1AKE L=214 exit 3 的根因）。
        //   这里把它与 view_src 视图同路解析到**根**，拿根的真实 device 指针（数据来源相同，
        //   只是省掉那次"拷到 host 再被 GPU 读回"的往返）。
        auto resolve_view_root = [](TensorF32* t) -> TensorF32* {
            TensorF32* r = t;
            int g = 0;
            while (r && g++ < 64) {
                TensorF32* nxt = r->view_src;
                if (!nxt && (r->op == OP_VIEW || r->op == OP_RESHAPE)) nxt = r->src[0];
                if (!nxt) break;
                r = nxt;
            }
            return r;
        };
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            TensorF32* src = node->src[s];
            if (!src) continue;
            TensorF32* base = resolve_view_root(src);
            if (!base || base == src || !base->data()) continue;
            src->bind_data(base->data());
            src->buffer_      = base->buffer_;
            src->buffer_offs_ = base->buffer_offs_;
        }

        // ---- host 叶子 → 设备影子（仅单后端全 GPU；scheduler 由 build_splits 建 H2D cpy）----
        //   skip_alloc_=true 表示由 scheduler 驱动 ⇒ 跨端搬运已由图中的 cpy 节点负责，
        //   这里再补一次会变成双份搬运（且掩盖 cpy 漏建的问题）⇒ 只在单后端模式生效。
        if (!skip_alloc_) {
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                TensorF32* src = node->src[s];
                if (!src || !src->data()) continue;
                ensure_host_shadow(src);   // 失败 ⇒ 保持原样，由下面的守卫给出明确报错
            }
        }

        // ---- 单后端模式：无 CPU 回落的显式护栏（2026-10-23）----
        //   实测（PPML_CUDA_SCHED=0 + PPML_CUDA_MIN_FREE_MB=1024）：op=102 (OP_TRI_MUL_BACK)
        //   在 CUDA 后端**没有 kernel**，dispatch 返回 NOT_SUPPORTED ⇒ 该节点不执行、
        //   图被静默截断 ⇒ loss 从 13.6 变 -3.00（无任何报错）。scheduler 模式靠
        //   supports_op 把它派给 CPU 所以不会暴露；单后端模式必须显式说明。
        if (!skip_alloc_ && node->op != OP_NONE && !supports_op(node)) {
            static int n_warn = 0;
            if (n_warn++ < 8) {
                fprintf(stderr,
                        "[CUDA-UNSUPPORTED] op=%d(%s) (node %d) 无 CUDA kernel —— 单后端全 GPU 模式"
                        "**没有 CPU 回落**，该节点不会被执行（后续 loss 不可信）。"
                        "改用 PPML_CUDA_SCHED=1 让调度器把它派给 CPU。\n",
                        (int)node->op, cuda_op_name((int)node->op), node_n);
                fflush(stderr);
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
                // 2026-10-17：改为按实际字节数 nbytes()（量化张量 = 块数×块字节，远小于 numel*4）。
                //   原来一律 numel*4：int8 权重/量化叶子会被判 "buffer OVERFLOW"（误报），
                //   因为它的 alloc_size 只有 34/32 B/元素。
                size_t need = t->nbytes();
                if (need == 0) need = (size_t)t->numel() * sizeof(float);   // 形状非法时保守估计
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
        //  诊断（GRAPH_DEBUG_BACKNODE=1）：GPU split 里出现 op>=100（_BACK）时的节点详情。
        //    确认 op=108 OUTER_PROD_MEAN_BACK 是否真的被 GPU dispatch、其 src 形状/数据状态。
        if (getenv("GRAPH_DEBUG_BACKNODE") && (int)node->op >= 100 && (int)node->op <= 120) {
            fprintf(stderr,
                    "[cuda-backnode] GPU dispatch op=%d node_n=%d ndim=%d dims=[%lld,%lld,%lld,%lld] "
                    "data=%p buf=%p offs=%zu view_src=%p src0={op=%d data=%p buf=%p offs=%zu} "
                    "src1={op=%d data=%p buf=%p offs=%zu} src2={op=%d data=%p buf=%p offs=%zu} "
                    "src3={op=%d data=%p} src4={op=%d data=%p}\n",
                    (int)node->op, node_n, (int)node->shape().ndim(),
                    (long long)(node->shape().ndim()>0?node->shape().dims[0]:-1),
                    (long long)(node->shape().ndim()>1?node->shape().dims[1]:-1),
                    (long long)(node->shape().ndim()>2?node->shape().dims[2]:-1),
                    (long long)(node->shape().ndim()>3?node->shape().dims[3]:-1),
                    (void*)node->data(), (void*)node->buffer_, (size_t)node->buffer_offs_, (void*)node->view_src,
                    (node->src[0]?(int)node->src[0]->op:-1), (void*)(node->src[0]?node->src[0]->data():nullptr),
                    (void*)(node->src[0]?node->src[0]->buffer_:nullptr), (size_t)(node->src[0]?node->src[0]->buffer_offs_:0),
                    (node->src[1]?(int)node->src[1]->op:-1), (void*)(node->src[1]?node->src[1]->data():nullptr),
                    (void*)(node->src[1]?node->src[1]->buffer_:nullptr), (size_t)(node->src[1]?node->src[1]->buffer_offs_:0),
                    (node->src[2]?(int)node->src[2]->op:-1), (void*)(node->src[2]?node->src[2]->data():nullptr),
                    (void*)(node->src[2]?node->src[2]->buffer_:nullptr), (size_t)(node->src[2]?node->src[2]->buffer_offs_:0),
                    (node->src[3]?(int)node->src[3]->op:-1), (void*)(node->src[3]?node->src[3]->data():nullptr),
                    (node->src[4]?(int)node->src[4]->op:-1), (void*)(node->src[4]?node->src[4]->data():nullptr));
        }
        // 生产者代际断言（PPML_STALE_CHECK=1）：消费前校验 src 是否本代际写过
        exec_stale_check(node, "CUDA", node_n);
        // 输入侧爆炸检查（PPML_EXEC_NAN_IN=1）：爆炸点前移到 kernel 之前
        exec_nan_check_in(node, "CUDA", node_n);
        // per_edge_matmul 三件套：半写判据 + 输入/输出统计（PPML_PEM_STAT=1）
        exec_pem_shape_check(node, "CUDA", node_n);
        exec_pem_stat(node, "CUDA", node_n, "in");
        Status st = dispatch_node(node, &params);
        exec_stale_mark(node);   // 本节点算完 ⇒ 记为"本代际已写"（CUDA 单线程，无需 ith 门控）
        if (getenv("GRAPH_DEBUG_CUDA_OP")) {
            fprintf(stderr, "[cuda-dispatch] op=%d st=%d\n", (int)node->op, (int)st);
        }
        if (st != Status::SUCCESS) {
            return st;
        }
        // 执行期 NaN/爆炸检查（PPML_EXEC_NAN_CHECK=1）：紧跟本节点 kernel，按执行顺序
        //   报告第一个坏节点 ⇒ 直接定位"第一个产生 NaN/Inf 的 CUDA kernel"。
        exec_nan_check(node, "CUDA", node_n);
        exec_zeropage_check(node, "CUDA", node_n);   // 零页探针（PPML_ZEROPAGE_CHECK=1）
        exec_pem_stat(node, "CUDA", node_n, "out");   // per_edge_matmul 实际写出内容（PPML_PEM_STAT=1）
        exec_probe_op(node, "CUDA", node_n);   // 定点算子探针（PPML_PROBE_OP=...）
        exec_view_probe(node, "CUDA", node_n);   // view 别名消费探针（PPML_EXEC_VIEW_PROBE=1）
        exec_ab_check(node, "CUDA", node_n);     // 单节点 CPU/GPU 数值 A/B（PPML_AB_CHECK=1）
        // 捕获异步 CUDA 错误（如 kernel 拿到 host 指针当 device 读 -> illegal address）。
        // 先查 launch 错误（廉价的 cudaGetLastError 始终做）；执行错误需同步才能捕获，
        // 用 GRAPH_DEBUG_CUDA 控制（每 kernel 同步很慢，仅定位段错误时开）。
        {
            cudaError_t kerr = cudaGetLastError();
            if (kerr != cudaSuccess) {
                //  区分 launch 错误 vs 执行期错误：launch 错误由本节点 kernel 引起（可立即查），
                //    执行期错误（illegal address）需同步才暴露——用 cudaDeviceSynchronize 确认。
                //    此前 [CUDA-ERR] op=108 (node 0) 可能是"上一个 kernel 执行期错误"在下一个
                //    检查点暴露。同步后能打印真正的错误源。
                fprintf(stderr,
                        "[CUDA-ERR] kernel launch failed at op=%d (node %d): %s\n",
                        (int)node->op, node_n, cudaGetErrorString(kerr));
                cudaGetLastError();  // 清错
                if (getenv("GRAPH_DEBUG_CUDA_SYNC")) {
                    cudaError_t serr = cudaDeviceSynchronize();
                    if (serr != cudaSuccess) {
                        fprintf(stderr,
                                "[CUDA-SYNC-ERR] exec error after op=%d (node %d): %s\n",
                                (int)node->op, node_n, cudaGetErrorString(serr));
                        cudaGetLastError();
                    } else {
                        fprintf(stderr,
                                "[CUDA-SYNC-OK] no exec error after op=%d (node %d) — pure launch error\n",
                                (int)node->op, node_n);
                    }
                }
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

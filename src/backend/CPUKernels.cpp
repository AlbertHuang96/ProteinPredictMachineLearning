
#include "rfaa/Backend.h"
#include "rfaa/ComputeGraph.h"
#include "rfaa/FAPE.h"
#include <cstring>
#include <cmath>

//#include <omp.h>

namespace rfaa {

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
Status CPUBackend::dispatch_node(TensorF32 * node, ComputeParams * p) {
    switch (node->op) {
        case OP_NONE:   break;
        case OP_DUP:    kernel_dup(node);            break;
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:    kernel_elemwise(node, p);    break;
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
        case OP_MEAN:   kernel_mean(node, p);        break;
        case OP_FAPE:      compute_forward_fape(p, node);      break;
        case OP_FAPE_BACK: compute_forward_fape_back(p, node); break;
        case OP_TRI_MUL:   kernel_tri_mul(node, p);            break;
        case OP_TRI_MUL_BACK: kernel_tri_mul_back(node, p);    break;
        case OP_UNARY:  {
            const unary_op uop = get_unary_op(node);
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
    return p->threadpool->ec;
}

// ===== elemwise =====
void CPUBackend::kernel_elemwise(TensorF32 * node, ComputeParams * p) {
    const float * a = node->src[0]->data();
    const float * b = node->src[1]->data();
    float * d = node->data();
    int64_t n = node->numel();

    switch (node->op) {
        case OP_ADD:
            for (int64_t i = p->ith; i < n; i += p->nth) d[i] = a[i] + b[i];
            break;
        case OP_SUB:
            for (int64_t i = p->ith; i < n; i += p->nth) d[i] = a[i] - b[i];
            break;
        case OP_MUL:
            for (int64_t i = p->ith; i < n; i += p->nth) d[i] = a[i] * b[i];
            break;
        case OP_DIV:
            for (int64_t i = p->ith; i < n; i += p->nth) d[i] = a[i] / b[i];
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

    if (p->ith == 0) tp->current_chunk.store(0);
    tp->barrier_wait();

    // a (M * K), b (K * N), d (M * N)
    while (true) {
        int i = tp->current_chunk.fetch_add(1);
        if (i >= M) break;
        float sum = 0;
        for (int j = 0; j < N; j++) {
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
// left:  (B, I, K, D) for outgoing, (B, K, I, D) for incoming
// right: (B, J, K, D) for outgoing, (B, K, J, D) for incoming
// dst:   (B, I, J, D)
void CPUBackend::kernel_tri_mul(TensorF32 * node, ComputeParams * p) {
    ThreadPool * tp = p->threadpool;

    const TensorF32 * src0 = node->src[0];  // left
    const TensorF32 * src1 = node->src[1];  // right
    TensorF32       * dst  = node;

    const int64_t B = dst->shape().dims[0];
    const int64_t I = dst->shape().dims[1];  // = src0 dim[1]
    const int64_t J = dst->shape().dims[2];  // = src1 dim[1]
    const int64_t D = dst->shape().dims[3];  // inner dim
    const int64_t K = src0->shape().dims[2]; // contraction dim

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
        int64_t tmp  = idx;
        int64_t d    = tmp % D;  tmp /= D;
        int64_t j    = tmp % J;  tmp /= J;
        int64_t i    = tmp % I;
        int64_t b    = tmp / I;

        float sum = 0.0f;
        for (int64_t k = 0; k < K; k++) {
            float lv, rv;
            if (outgoing) {
                // left(b,i,k,d) * right(b,j,k,d)
                lv = left_data[((b * I + i) * K + k) * D + d];
                rv = right_data[((b * J + j) * K + k) * D + d];
            } else {
                // left(b,k,i,d) * right(b,k,j,d)
                lv = left_data[((b * K + k) * I + i) * D + d];
                rv = right_data[((b * K + k) * J + j) * D + d];
            }
            sum += lv * rv;
        }
        dst_data[idx] = sum * inv_L;
    }

    tp->barrier_wait();
}

// backward: dL/dleft 和 dL/dright 分别对 left 和 right 求导
void CPUBackend::kernel_tri_mul_back(TensorF32 * node, ComputeParams * p) {
    // grad from upstream
    const TensorF32 * grad = node->src[0];  // dL/ddst: (B, I, J, D)
    const TensorF32 * left  = node->src[1]; // left
    const TensorF32 * right = node->src[2]; // right
    TensorF32 * grad_left  = node->src[3];  // dL/dleft
    TensorF32 * grad_right = node->src[4];  // dL/dright

    if (!grad_left || !grad_right) return;

    const int64_t B = left->shape().dims[0];
    const int64_t I = left->shape().dims[1];
    const int64_t J = right->shape().dims[1];
    const int64_t D = left->shape().dims[3];
    const int64_t K = left->shape().dims[2];

    float L;
    bool  outgoing;
    memcpy(&L,        node->op_params,      sizeof(float));
    memcpy(&outgoing, node->op_params + 4,  sizeof(bool));

    const float inv_L = 1.0f / L;
    const float * grad_data = static_cast<const float*>(grad->data());
    const float * right_data = static_cast<const float*>(right->data());
    const float * left_data  = static_cast<const float*>(left->data());
    float * gleft_data  = static_cast<float*>(grad_left->data());
    float * gright_data = static_cast<float*>(grad_right->data());

    ThreadPool * tp = p->threadpool;

    // dL/dleft: for outgoing: einsum('bijd,bjkd->bikd', grad, right/L)
    //           for incoming: einsum('bijd,bkjd->bkid', grad, right/L)
    {
        const int64_t total = B * I * K * D;
        const int64_t per  = (total + p->nth - 1) / p->nth;
        const int64_t start = per * p->ith;
        const int64_t end   = (start + per < total) ? (start + per) : total;
        for (int64_t idx = start; idx < end; idx++) {
            int64_t tmp = idx;
            const int64_t d = tmp % D; tmp /= D;
            const int64_t k = tmp % K; tmp /= K;
            const int64_t i = tmp % I;
            const int64_t b = tmp / I;
            float sum = 0.0f;
            for (int64_t j = 0; j < J; j++) {
                float gv = grad_data[((b * I + i) * J + j) * D + d];
                float rv;
                if (outgoing)
                    rv = right_data[((b * J + j) * K + k) * D + d];
                else
                    rv = right_data[((b * K + k) * J + j) * D + d];
                sum += gv * rv;
            }
            gleft_data[idx] = sum * inv_L;
        }
    }
    tp->barrier_wait();

    // dL/dright: for outgoing: einsum('bijd,bikd->bjkd', grad, left/L)
    //            for incoming: einsum('bijd,bkid->bkjd', grad, left/L)
    {
        const int64_t total = B * J * K * D;
        const int64_t per  = (total + p->nth - 1) / p->nth;
        const int64_t start = per * p->ith;
        const int64_t end   = (start + per < total) ? (start + per) : total;
        for (int64_t idx = start; idx < end; idx++) {
            int64_t tmp = idx;
            const int64_t d = tmp % D; tmp /= D;
            const int64_t k = tmp % K; tmp /= K;
            const int64_t j = tmp % J;
            const int64_t b = tmp / J;
            float sum = 0.0f;
            for (int64_t i = 0; i < I; i++) {
                float gv = grad_data[((b * I + i) * J + j) * D + d];
                float lv;
                if (outgoing)
                    lv = left_data[((b * I + i) * K + k) * D + d];
                else
                    lv = left_data[((b * K + k) * I + i) * D + d];
                sum += gv * lv;
            }
            gright_data[idx] = sum * inv_L;
        }
    }
    tp->barrier_wait();
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
        float mx = sr[0];
        float sum = 0;
        for (int d = 1; d < D; d++) {
            float mx_prev = mx;
            if (sr[d] > mx) {
                mx = sr[d];
                sum = sum * expf(mx_prev - mx) + expf(sr[d] - mx);
                // online softmax
            }
            else {
                sum += expf(sr[d] - mx);
            }
        }
        for (int d = 1; d < D; d++) {
            dr[d] = expf(sr[d] - mx) / sum;
        }

        //for (int d = 0; d < D; d++) { dr[d] = expf(sr[d] - mx); sum += dr[d]; }
        //for (int d = 0; d < D; d++) dr[d] /= sum;
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

void CPUBackend::kernel_scale(TensorF32 * node, ComputeParams * p) { (void)node; (void)p; }
void CPUBackend::kernel_add1(TensorF32 * node, ComputeParams * p)  { (void)node; (void)p; }
void CPUBackend::kernel_sum(TensorF32 * node, ComputeParams * p)   { (void)node; (void)p; }
void CPUBackend::kernel_mean(TensorF32 * node, ComputeParams * p)  { (void)node; (void)p; }

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

            // e1 = normalize(v1)
            float n1 = sqrtf(v1x * v1x + v1y * v1y + v1z * v1z);
            float e1x = v1x / n1, e1y = v1y / n1, e1z = v1z / n1;

            // e2 = normalize(v2 - (v2·e1)*e1)
            float dot = v2x * e1x + v2y * e1y + v2z * e1z;
            float u2x = v2x - dot * e1x, u2y = v2y - dot * e1y, u2z = v2z - dot * e1z;
            float n2 = sqrtf(u2x * u2x + u2y * u2y + u2z * u2z);
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

            float n1 = sqrtf(v1x * v1x + v1y * v1y + v1z * v1z);
            float e1x = v1x / n1, e1y = v1y / n1, e1z = v1z / n1;

            float dot = v2x * e1x + v2y * e1y + v2z * e1z;
            float u2x = v2x - dot * e1x, u2y = v2y - dot * e1y, u2z = v2z - dot * e1z;
            float n2 = sqrtf(u2x * u2x + u2y * u2y + u2z * u2z);
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

        // Gram-Schmidt
        float v1x = pBx - pAx, v1y = pBy - pAy, v1z = pBz - pAz;
        float v2x = pCx - pAx, v2y = pCy - pAy, v2z = pCz - pAz;

        float n1 = sqrtf(v1x * v1x + v1y * v1y + v1z * v1z);
        float e1x = v1x / n1, e1y = v1y / n1, e1z = v1z / n1;

        float dot = v2x * e1x + v2y * e1y + v2z * e1z;
        float u2x = v2x - dot * e1x, u2y = v2y - dot * e1y, u2z = v2z - dot * e1z;
        float n2 = sqrtf(u2x * u2x + u2y * u2y + u2z * u2z);
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

        float n1 = sqrtf(v1x * v1x + v1y * v1y + v1z * v1z);
        float e1x = v1x / n1, e1y = v1y / n1, e1z = v1z / n1;

        float dot = v2x * e1x + v2y * e1y + v2z * e1z;
        float u2x = v2x - dot * e1x, u2y = v2y - dot * e1y, u2z = v2z - dot * e1z;
        float n2 = sqrtf(u2x * u2x + u2y * u2y + u2z * u2z);
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

} // namespace rfaa

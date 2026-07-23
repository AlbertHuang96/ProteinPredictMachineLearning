
#include "rfaa/Backend.h"
#include "rfaa/ComputeGraph.h"
#include <cstring>
#include <cmath>

//#include <omp.h>

namespace rfaa {

// ===== unary op 计算函数前向声明 =====
static void compute_forward_abs(ComputeParams* p, Tensor* dst);
static void compute_forward_sgn(ComputeParams* p, Tensor* dst);
static void compute_forward_neg(ComputeParams* p, Tensor* dst);
static void compute_forward_step(ComputeParams* p, Tensor* dst);
static void compute_forward_relu(ComputeParams* p, Tensor* dst);
static void compute_forward_gelu(ComputeParams* p, Tensor* dst);
static void compute_forward_gelu_quick(ComputeParams* p, Tensor* dst);
static void compute_forward_silu(ComputeParams* p, Tensor* dst);
static void compute_forward_tanh(ComputeParams* p, Tensor* dst);
static void compute_forward_elu(ComputeParams* p, Tensor* dst);
static void compute_forward_sigmoid(ComputeParams* p, Tensor* dst);
static void compute_forward_hardsigmoid(ComputeParams* p, Tensor* dst);
static void compute_forward_hardswish(ComputeParams* p, Tensor* dst);
static void compute_forward_exp(ComputeParams* p, Tensor* dst);
static void compute_forward_log(ComputeParams* p, Tensor* dst);
static void compute_forward_sqrt(ComputeParams* p, Tensor* dst);
static void compute_forward_sin(ComputeParams* p, Tensor* dst);
static void compute_forward_cos(ComputeParams* p, Tensor* dst);

// ===== dispatch =====
Status CPUBackend::dispatch_node(Tensor * node, ComputeParams * p) {
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
void CPUBackend::kernel_elemwise(Tensor * node, ComputeParams * p) {
    float * a = node->src[0]->data();
    float * b = node->src[1]->data();
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
void CPUBackend::kernel_mul_mat(Tensor * node, ComputeParams * p) {
    ThreadPool * tp = p->threadpool;
    int M = static_cast<int>(node->dims()[1]);
    int N = static_cast<int>(node->dims()[0]);
    int K = static_cast<int>(node->src[0]->dims()[0]);
    float * a = node->src[0]->data();
    float * b = node->src[1]->data();
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
void CPUBackend::kernel_out_prod(Tensor * node, ComputeParams * p) {
    ThreadPool * tp = p->threadpool;

    const Tensor * src0 = node->src[0];
    const Tensor * src1 = node->src[1];

    const int64_t ne00 = src0->dims()[0];  // src0 inner dim
    const int64_t ne01 = src0->dims()[1];  // src0 K dim (contraction dim)
    const int64_t ne02 = (src0->shape().ndim() > 2) ? src0->dims()[2] : 1;
    const int64_t ne03 = (src0->shape().ndim() > 3) ? src0->dims()[3] : 1;

    const int64_t ne10 = src1->dims()[0];  // src1 inner dim
    const int64_t ne11 = src1->dims()[1];  // src1 K dim (contraction dim, == ne01)
    const int64_t ne12 = (src1->shape().ndim() > 2) ? src1->dims()[2] : 1;
    const int64_t ne13 = (src1->shape().ndim() > 3) ? src1->dims()[3] : 1;

    // dst shape: (ne00, ne10, max(ne02,ne12), max(ne03,ne13))
    const int64_t ne0 = node->dims()[0];  // == ne00
    const int64_t ne1 = node->dims()[1];  // == ne10
    const int64_t ne2 = (node->shape().ndim() > 2) ? node->dims()[2] : 1;
    const int64_t ne3 = (node->shape().ndim() > 3) ? node->dims()[3] : 1;

    float * src0_data = src0->data();
    float * src1_data = src1->data();
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
                    float * s0 = src0_data + (i01 * ne00 + i02 * ne00 * ne01 + i03 * ne00 * ne01 * ne02);

                    // src1 元素: (i1, i01, i12, i13)
                    float * s1_row = src1_data + (i01 * ne10 + i12 * ne10 * ne11 + i13 * ne10 * ne11 * ne12);
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

// ===== softmax =====
void CPUBackend::kernel_softmax(Tensor * node, ComputeParams * p) {
    int D    = static_cast<int>(node->dims()[0]);
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
void CPUBackend::kernel_rms_norm(Tensor * node, ComputeParams * p) {
    int D    = static_cast<int>(node->dims()[0]);
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
void CPUBackend::kernel_norm(Tensor * node, ComputeParams * p) {
    int D    = static_cast<int>(node->dims()[0]);
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
void CPUBackend::kernel_norm_back(Tensor * node, ComputeParams * p) {
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
void CPUBackend::kernel_softmax_back(Tensor * node, ComputeParams * p) {
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
void CPUBackend::kernel_silu(Tensor * node) {
    float * s = node->src[0]->data(), * d = node->data();
    for (int64_t i = 0; i < node->numel(); i++)
        d[i] = s[i] / (1.0f + expf(-s[i]));
}
void CPUBackend::kernel_gelu(Tensor * node) { /* ... */ }
void CPUBackend::kernel_relu(Tensor * node) { /* ... */ }

void CPUBackend::kernel_dup(Tensor * node) {
    std::memcpy(node->data(), node->src[0]->data(), node->numel() * sizeof(float));
}

void CPUBackend::kernel_scale(Tensor * node, ComputeParams * p) { /* ... */ }
void CPUBackend::kernel_add1(Tensor * node, ComputeParams * p)  { /* ... */ }
void CPUBackend::kernel_sum(Tensor * node, ComputeParams * p)   { /* ... */ }
void CPUBackend::kernel_mean(Tensor * node, ComputeParams * p)  { /* ... */ }

void CPUBackend::kernel_sigmoid(Tensor * node, ComputeParams * p) {
    Tensor* output = node->src[0];
    float* data = output->data();
    int64_t n = node->src[0]->numel();

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }

    return output;
}

// ===== unary op 计算函数实现 =====

static void compute_forward_abs(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = fabsf(s[i]);
    }
}

static void compute_forward_sgn(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = (s[i] > 0.0f) ? 1.0f : ((s[i] < 0.0f) ? -1.0f : 0.0f);
    }
}

static void compute_forward_neg(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = -s[i];
    }
}

static void compute_forward_step(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = (s[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

static void compute_forward_relu(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = (s[i] > 0.0f) ? s[i] : 0.0f;
    }
}

static void compute_forward_gelu(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
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

static void compute_forward_gelu_quick(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        float x = s[i];
        d[i] = x * (1.0f / (1.0f + expf(-1.702f * x)));
    }
}

static void compute_forward_silu(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = s[i] / (1.0f + expf(-s[i]));
    }
}

static void compute_forward_tanh(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = tanhf(s[i]);
    }
}

static void compute_forward_elu(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        float x = s[i];
        d[i] = (x > 0.0f) ? x : (expf(x) - 1.0f);
    }
}

static void compute_forward_sigmoid(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = 1.0f / (1.0f + expf(-s[i]));
    }
}

static void compute_forward_hardsigmoid(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
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

static void compute_forward_hardswish(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
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

static void compute_forward_exp(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = expf(s[i]);
    }
}

static void compute_forward_log(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = logf(s[i]);
    }
}

static void compute_forward_sqrt(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = sqrtf(s[i]);
    }
}

static void compute_forward_sin(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = sinf(s[i]);
    }
}

static void compute_forward_cos(ComputeParams* p, Tensor* dst) {
    Tensor* src0 = dst->src[0];
    float* d = dst->data();
    float* s = src0->data();
    int64_t n = dst->numel();
    for (int64_t i = p->ith; i < n; i += p->nth) {
        d[i] = cosf(s[i]);
    }
}

} // namespace rfaa

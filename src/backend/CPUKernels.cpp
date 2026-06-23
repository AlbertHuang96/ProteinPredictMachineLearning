
#include "rfaa/Backend.h"
#include <cstring>
#include <cmath>

//#include <omp.h>

namespace rfaa {

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
        case OP_SOFT_MAX:  kernel_softmax(node, p);  break;
        case OP_RMS_NORM:  kernel_rms_norm(node, p); break;
        //case OP_SILU:   kernel_silu(node);           break;
        //case OP_GELU:   kernel_gelu(node);           break;
        //case OP_RELU:   kernel_relu(node);           break;
        case OP_SUM:    kernel_sum(node, p);         break;
        case OP_MEAN:   kernel_mean(node, p);        break;
        case OP_UNARY:  kernel_sigmoid(node, p);     break;
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

} // namespace rfaa

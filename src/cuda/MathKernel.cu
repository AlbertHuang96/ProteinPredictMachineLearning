#include <cuda_runtime.h>

namespace ppml {

// 通用数学运算 CUDA 核

// 元素级 ReLU
__global__ void kernel_relu(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// 元素级 GELU
__global__ void kernel_gelu(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// 元素级 Sigmoid
__global__ void kernel_sigmoid(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// 元素级 Tanh
__global__ void kernel_tanh(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

// Dropout (训练时)
__global__ void kernel_dropout(const float* input, float* output, 
                                const float* mask, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * mask[idx] * scale;
    }
}

// 广播加法
__global__ void kernel_broadcast_add(const float* a, const float* b, float* c,
                                      int batch, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * dim) return;
    
    int d = idx % dim;
    c[idx] = a[idx] + b[d];
}

// 矩阵转置
__global__ void kernel_transpose(const float* input, float* output,
                                  int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// 批量矩阵乘法 (BMM)
__global__ void kernel_bmm(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int b = blockIdx.z;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (m >= M || n >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[(b * M + m) * K + k] * B[(b * K + k) * N + n];
    }
    C[(b * M + m) * N + n] = sum;
}

} // namespace ppml

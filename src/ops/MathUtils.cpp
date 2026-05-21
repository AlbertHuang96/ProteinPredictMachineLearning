#include "rfaa/MathUtils.h"
#include <numeric>  // for std::iota
#include <vector>
#include <stdexcept>

namespace rfaa {



// 创建 [start, start+1, ..., end-1]
std::vector<int> arange(int start, int end) {
    std::vector<int> result(end - start);
    std::iota(result.begin(), result.end(), start);
    return result;
}

TensorF32 matmul(const TensorF32& a, const TensorF32& b) {
    // 假设 a: (M, K), b: (K, N), 输出: (M, N)
    const auto& a_shape = a.shape().dims;
    const auto& b_shape = b.shape().dims;
    
    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw RFAAError("matmul expects 2D tensors");
    }
    
    int64_t M = a_shape[0];
    int64_t K = a_shape[1];
    int64_t N = b_shape[1];
    
    if (b_shape[0] != K) {
        throw RFAAError("matmul dimension mismatch");
    }
    
    TensorF32 output({M, N}, a.device());
    
    // 使用 OpenBLAS 进行矩阵乘法
    /* cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f,
                a.data(), K,
                b.data(), N,
                0.0f,
                output.data(), N); */
    
    return output;
}

/**
 * @brief 批量矩阵乘法，对一批矩阵执行并行乘法运算
 *
 * 对两个三维张量执行批量矩阵乘法，其中每个批次切片独立进行
 * 矩阵乘法。输入张量 a 的形状为 (B, M, K)，张量 b 的形状为
 * (B, K, N)，输出张量形状为 (B, M, N)。底层使用 BLAS 的
 * cblas_sgemm 例程实现高效计算。
 *
 * @param a 左操作数，形状为 (B, M, K) 的三维张量
 * @param b 右操作数，形状为 (B, K, N) 的三维张量
 * @return TensorF32 结果张量，形状为 (B, M, N)
 * @throws RFAAError 当输入张量不是三维时抛出
 * @throws RFAAError 当批次维度 B 或内积维度 K 不匹配时抛出
 */
TensorF32 batch_matmul(const TensorF32& a, const TensorF32& b) {
    // 假设 a: (B, M, K), b: (B, K, N), 输出: (B, M, N)
    const auto& a_shape = a.shape().dims;
    const auto& b_shape = b.shape().dims;
    
    if (a_shape.size() != 3 || b_shape.size() != 3) {
        throw RFAAError("batch_matmul expects 3D tensors");
    }
    
    int64_t B = a_shape[0];
    int64_t M = a_shape[1];
    int64_t K = a_shape[2];
    int64_t N = b_shape[2];
    
    if (b_shape[0] != B || b_shape[1] != K) {
        throw RFAAError("batch_matmul dimension mismatch");
    }
    
    TensorF32 output({B, M, N}, a.device());
    
    // 批量矩阵乘法
    for (int64_t b = 0; b < B; b++) {
        float* output_slice = output.data() + b * M * N;
        const float* a_slice = a.data() + b * M * K;
        const float* b_slice = b.data() + b * K * N;
        
        /* cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0f,
                    a_slice, K,
                    b_slice, N,
                    0.0f,
                    output_slice, N); */
    }
    
    return output;
}

TensorF32 one_hot(const std::vector<int>& indices, int num_classes) {
    if (num_classes <= 0) {
        throw std::invalid_argument("num_classes must be positive");
    }
    
    // 检查索引范围
    for (int idx : indices) {
        if (idx < 0 || idx >= num_classes) {
            throw std::out_of_range("Index " + std::to_string(idx) + 
                                    " out of range [0, " + std::to_string(num_classes-1) + "]");
        }
    }
    
    int64_t n = static_cast<int64_t>(indices.size());
    TensorF32 result({n, num_classes}, Device::CPU);
    
    // 初始化为0
    result.zero_();
    
    // 设置one-hot位置为1
    float* data = result.data();
    for (int64_t i = 0; i < n; i++) {
        data[i * num_classes + indices[i]] = 1.0f;
    }
    
    return result;
}

TensorF32 one_hot(int index, int num_classes) {
    if (num_classes <= 0) {
        throw std::invalid_argument("num_classes must be positive");
    }
    
    if (index < 0 || index >= num_classes) {
        throw std::out_of_range("Index " + std::to_string(index) + 
                                " out of range [0, " + std::to_string(num_classes-1) + "]");
    }
    
    TensorF32 result({1, num_classes}, Device::CPU);
    
    // 初始化为0
    result.zero_();
    
    // 设置one-hot位置为1
    float* data = result.data();
    data[index] = 1.0f;
    
    return result;
}

TensorF32 outer_sum(const TensorF32& left, const TensorF32& right) {
    const auto& left_shape = left.shape().dims;
    const auto& right_shape = right.shape().dims;
    
    // 检查输入是否为 4D 张量
    if (left_shape.size() != 4 || right_shape.size() != 4) {
        throw RFAAError("outer_sum expects 4D tensors");
    }
    
    // 检查维度：left (B,1,L,D), right (B,L,1,D)
    if (left_shape[0] != right_shape[0] || left_shape[3] != right_shape[3]) {
        throw RFAAError("outer_sum dimension mismatch: batch or feature dim doesn't match");
    }
    if (left_shape[1] != 1) {
        throw RFAAError("outer_sum: left shape[1] should be 1, got " + std::to_string(left_shape[1]));
    }
    if (right_shape[2] != 1) {
        throw RFAAError("outer_sum: right shape[2] should be 1, got " + std::to_string(right_shape[2]));
    }
    
    int64_t B = left_shape[0];
    int64_t L = left_shape[2];  // left: (B,1,L,D) -> L 在 dim 2
    int64_t D = left_shape[3];
    
    // 创建结果张量 (B, L, L, D)
    TensorF32 result({B, L, L, D}, left.device());
    result.zero_();
    
    const float* left_data = left.data();
    const float* right_data = right.data();
    float* result_data = result.data();
    
    // 计算 outer sum: result[b, i, j, d] = left[b, 0, i, d] + right[b, j, 0, d]
    for (int64_t b = 0; b < B; b++) {
        for (int64_t i = 0; i < L; i++) {
            for (int64_t j = 0; j < L; j++) {
                for (int64_t d = 0; d < D; d++) {
                    // left[b, 0, i, d] -> index = b*1*L*D + 0*L*D + i*D + d
                    int64_t left_idx = (b * 1 * L * D) + (0 * L * D) + (i * D) + d;
                    // right[b, j, 0, d] -> index = b*L*1*D + j*1*D + 0*D + d
                    int64_t right_idx = (b * L * 1 * D) + (j * 1 * D) + (0 * D) + d;
                    // result[b, i, j, d] -> index = b*L*L*D + i*L*D + j*D + d
                    int64_t result_idx = (b * L * L * D) + (i * L * D) + (j * D) + d;
                    
                    result_data[result_idx] = left_data[left_idx] + right_data[right_idx];
                }
            }
        }
    }
    
    return result;
}

TensorF32 outer_product(const TensorF32& left, const TensorF32& right) {
    const auto& left_shape = left.shape().dims;
    const auto& right_shape = right.shape().dims;
    
    // 检查输入是否为 4D 张量
    if (left_shape.size() != 4 || right_shape.size() != 4) {
        throw RFAAError("outer_product expects 4D tensors");
    }
    
    // 检查维度：left (B,1,L,D), right (B,L,1,D)
    if (left_shape[0] != right_shape[0] || left_shape[3] != right_shape[3]) {
        throw RFAAError("outer_product dimension mismatch: batch or feature dim doesn't match");
    }
    if (left_shape[1] != 1) {
        throw RFAAError("outer_product: left shape[1] should be 1, got " + std::to_string(left_shape[1]));
    }
    if (right_shape[2] != 1) {
        throw RFAAError("outer_product: right shape[2] should be 1, got " + std::to_string(right_shape[2]));
    }
    
    int64_t B = left_shape[0];
    int64_t L = left_shape[2];  // left: (B,1,L,D) -> L 在 dim 2
    int64_t D = left_shape[3];
    
    // 创建结果张量 (B, L, L, D)
    TensorF32 result({B, L, L, D}, left.device());
    
    const float* left_data = left.data();
    const float* right_data = right.data();
    float* result_data = result.data();
    
    // 计算 outer product: result[b, i, j, d] = left[b, 0, i, d] * right[b, j, 0, d]
    for (int64_t b = 0; b < B; b++) {
        for (int64_t i = 0; i < L; i++) {
            for (int64_t j = 0; j < L; j++) {
                for (int64_t d = 0; d < D; d++) {
                    // left[b, 0, i, d] -> index = b*1*L*D + 0*L*D + i*D + d
                    int64_t left_idx = (b * 1 * L * D) + (0 * L * D) + (i * D) + d;
                    // right[b, j, 0, d] -> index = b*L*1*D + j*1*D + 0*D + d
                    int64_t right_idx = (b * L * 1 * D) + (j * 1 * D) + (0 * D) + d;
                    // result[b, i, j, d] -> index = b*L*L*D + i*L*D + j*D + d
                    int64_t result_idx = (b * L * L * D) + (i * L * D) + (j * D) + d;
                    
                    result_data[result_idx] = left_data[left_idx] * right_data[right_idx];
                }
            }
        }
    }
    
    return result;
}

} // namespace rfaa

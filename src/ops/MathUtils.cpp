#include "rfaa/Ops.h"
//#include <cblas.h>

namespace rfaa {

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

} // namespace rfaa

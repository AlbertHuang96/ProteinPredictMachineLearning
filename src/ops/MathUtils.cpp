#include "rfaa/MathUtils.h"
#include <numeric>  // for std::iota
#include <vector>
#include <stdexcept>
#include <cassert>

namespace rfaa {



// 创建 [start, start+1, ..., end-1]
std::vector<int> arange(int start, int end) {
    std::vector<int> result(end - start);
    std::iota(result.begin(), result.end(), start);
    return result;
}


TensorF32 mean(const TensorF32& input, int dim) {
    if (dim < 0) {
        dim += input.shape().ndim();
    }
    if (dim < 0 || dim >= input.shape().ndim()) {
        throw RFAAError("mean: dim " + std::to_string(dim) + " out of range");
    }
    
    // 计算输出形状
    Shape output_shape;
    for (int i = 0; i < input.shape().ndim(); ++i) {
        if (i != dim) {
            output_shape.dims.push_back(input.shape().dims[i]);
        }
    }
    
    TensorF32 output(output_shape, input.device());
    
    // 计算stride
    int ndim = input.shape().ndim();
    std::vector<int64_t> stride(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        stride[i] = stride[i + 1] * input.shape().dims[i + 1];
    }
    
    const float* src = input.data();
    float* dst = output.data();
    
    int64_t dim_size = input.shape().dims[dim];
    float inv_dim_size = 1.0f / static_cast<float>(dim_size);
    
    int64_t total_output = output.numel();
    
    // 使用简单的嵌套循环（针对4D情况优化）
    if (ndim == 4 && dim == 1) {
        // 常见情况: Q.shape = (B, N, L, D), mean(dim=1) -> (B, L, D)
        int64_t B = input.shape().dims[0];
        int64_t N = input.shape().dims[1];
        int64_t L = input.shape().dims[2];
        int64_t D = input.shape().dims[3];
        
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t l = 0; l < L; ++l) {
                for (int64_t d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    for (int64_t n = 0; n < N; ++n) {
                        int64_t src_idx = ((b * N + n) * L + l) * D + d;
                        sum += src[src_idx];
                    }
                    int64_t dst_idx = (b * L + l) * D + d;
                    dst[dst_idx] = sum / N;
                }
            }
        }
    } else {
        // 通用情况
        for (int64_t idx = 0; idx < total_output; ++idx) {
            // 计算输出坐标
            int64_t tmp = idx;
            std::vector<int64_t> out_coord(ndim - 1, 0);
            int out_dim = 0;
            for (int i = 0; i < ndim; ++i) {
                if (i == dim) continue;
                out_coord[out_dim] = tmp % output_shape.dims[out_dim];
                tmp /= output_shape.dims[out_dim];
                out_dim++;
            }
            
            // 计算源基准索引
            int64_t src_base = 0;
            out_dim = 0;
            for (int i = 0; i < ndim; ++i) {
                if (i == dim) {
                    src_base += 0 * stride[i];
                } else {
                    src_base += out_coord[out_dim] * stride[i];
                    out_dim++;
                }
            }
            
            // 求和平均
            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                sum += src[src_base + d * stride[dim]];
            }
            dst[idx] = sum * inv_dim_size;
        }
    }
    
    return output;
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

TensorF32 triangle_mult(
    const TensorF32& left,
    const TensorF32& right,
    float L,
    bool outgoing
) {
    if (outgoing) {
        // outgoing: einsum('bikd,bjkd->bijd', left, right/L)
        // left: (B, I, K, D)
        // right: (B, J, K, D)
        // output: (B, I, J, D)
        
        int64_t B = left.shape().dims[0];
        int64_t I = left.shape().dims[1];
        int64_t K = left.shape().dims[2];
        int64_t D = left.shape().dims[3];
        
        int64_t J = right.shape().dims[1];
        
        assert(right.shape().dims[0] == B);
        assert(right.shape().dims[2] == K);
        assert(right.shape().dims[3] == D);
        
        TensorF32 output({B, I, J, D}, left.device());
        float* out_data = output.data();
        const float* left_data = left.data();
        const float* right_data = right.data();
        
        float inv_L = 1.0f / L;
        
        // Compute einsum
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t i = 0; i < I; ++i) {
                for (int64_t j = 0; j < J; ++j) {
                    for (int64_t d = 0; d < D; ++d) {
                        float sum = 0.0f;
                        for (int64_t k = 0; k < K; ++k) {
                            int64_t left_idx = ((b * I + i) * K + k) * D + d;
                            int64_t right_idx = ((b * J + j) * K + k) * D + d;
                            sum += left_data[left_idx] * right_data[right_idx];
                        }
                        int64_t out_idx = ((b * I + i) * J + j) * D + d;
                        out_data[out_idx] = sum * inv_L;
                    }
                }
            }
        }
        
        return output;
    } else {
        // incoming: einsum('bkid,bkjd->bijd', left, right/L)
        // left: (B, K, I, D)
        // right: (B, K, J, D)
        // output: (B, I, J, D)
        
        int64_t B = left.shape().dims[0];
        int64_t K = left.shape().dims[1];
        int64_t I = left.shape().dims[2];
        int64_t D = left.shape().dims[3];
        
        int64_t J = right.shape().dims[2];
        
        assert(right.shape().dims[0] == B);
        assert(right.shape().dims[1] == K);
        assert(right.shape().dims[3] == D);
        
        TensorF32 output({B, I, J, D}, left.device());
        float* out_data = output.data();
        const float* left_data = left.data();
        const float* right_data = right.data();
        
        float inv_L = 1.0f / L;
        
        // Compute einsum
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t i = 0; i < I; ++i) {
                for (int64_t j = 0; j < J; ++j) {
                    for (int64_t d = 0; d < D; ++d) {
                        float sum = 0.0f;
                        for (int64_t k = 0; k < K; ++k) {
                            int64_t left_idx = ((b * K + k) * I + i) * D + d;
                            int64_t right_idx = ((b * K + k) * J + j) * D + d;
                            sum += left_data[left_idx] * right_data[right_idx];
                        }
                        int64_t out_idx = ((b * I + i) * J + j) * D + d;
                        out_data[out_idx] = sum * inv_L;
                    }
                }
            }
        }
        
        return output;
    }

    
    //dim=-1: 沿着最后一个维度拼接（-1表示最后一个维度）
TensorF32 concat(const std::vector<TensorF32>& tensors, int dim) {
    if (tensors.empty()) {
        throw RFAAError("cat: cannot concatenate empty list of tensors");
    }
    
    // 获取第一个张量的形状和设备
    const Shape& first_shape = tensors[0].shape();
    int ndim = first_shape.ndim();
    
    // 处理负维度
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw RFAAError("cat: dim " + std::to_string(dim) + " out of range");
    }
    
    // 检查所有张量形状是否兼容
    for (size_t i = 1; i < tensors.size(); ++i) {
        const Shape& shape = tensors[i].shape();
        if (shape.ndim() != ndim) {
            throw RFAAError("cat: all tensors must have same number of dimensions");
        }
        for (int d = 0; d < ndim; ++d) {
            if (d != dim && shape.dims[d] != first_shape.dims[d]) {
                throw RFAAError("cat: shape mismatch at dim " + std::to_string(d));
            }
        }
    }
    
    // 计算输出形状
    Shape output_shape = first_shape;
    for (size_t i = 1; i < tensors.size(); ++i) {
        output_shape.dims[dim] += tensors[i].shape().dims[dim];
    }
    
    // 创建输出张量
    Device device = tensors[0].device();
    TensorF32 output(output_shape, device);
    
    // 计算 stride
    std::vector<int64_t> stride(ndim, 1);
    for (int d = ndim - 2; d >= 0; --d) {
        stride[d] = stride[d + 1] * output_shape.dims[d + 1];
    }
    
    // 当前拼接位置的偏移量
    int64_t offset = 0;
    
    for (size_t t = 0; t < tensors.size(); ++t) {
        const TensorF32& src = tensors[t];
        const float* src_data = src.data();
        float* dst_data = output.data();
        
        int64_t src_size = src.shape().dims[dim];
        int64_t total_outer = 1;
        for (int d = 0; d < dim; ++d) {
            total_outer *= output_shape.dims[d];
        }
        
        // 拷贝数据
        for (int64_t outer = 0; outer < total_outer; ++outer) {
            int64_t dst_offset = outer * stride[dim] + offset * stride[dim + 1];
            int64_t src_offset = outer * src_size * stride[dim + 1];
            int64_t block_size = src_size * stride[dim + 1];
            std::memcpy(dst_data + dst_offset, src_data + src_offset, block_size * sizeof(float));
        }
        
        offset += src_size;
    }
    
    return output;
}

} // namespace rfaa

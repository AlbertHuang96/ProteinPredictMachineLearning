#include "ppml/Ops.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <omp.h>

namespace ppml {

TensorF32 softmax(const TensorF32& input, int dim) {
    if (dim < 0) dim += input.shape().ndim();
    
    // 创建输出张量
    TensorF32 output({input.shape().dims[0], input.shape().dims[1], input.shape().dims[2], input.shape().dims[3]}, input.device());
    output.copy_from(input);
    const auto& shape = input.shape().dims;
    
    // 计算 stride
    std::vector<int64_t> stride(shape.size());
    stride[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= shape[i];
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < shape.size(); i++) inner_size *= shape[i];
    
    int64_t dim_size = shape[dim];
    
    const float* input_data = input.data();
    float* output_data = output.data();
    
    // 对每个切片计算 softmax
    #pragma omp parallel for
    for (int64_t outer = 0; outer < outer_size; outer++) {
        for (int64_t inner = 0; inner < inner_size; inner++) {
            int64_t base_idx = outer * stride[dim] * dim_size + inner;
            
            // 找到最大值 (数值稳定性)
            float max_val = input_data[base_idx];
            for (int64_t d = 1; d < dim_size; d++) {
                max_val = std::max(max_val, input_data[base_idx + d * stride[dim]]);
            }
            
            // 计算 exp(x - max) 和 sum
            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; d++) {
                float exp_val = std::exp(input_data[base_idx + d * stride[dim]] - max_val);
                output_data[base_idx + d * stride[dim]] = exp_val;
                sum += exp_val;
            }
            
            // 归一化
            float inv_sum = 1.0f / (sum + 1e-12f);
            for (int64_t d = 0; d < dim_size; d++) {
                output_data[base_idx + d * stride[dim]] *= inv_sum;
            }
        }
    }
    
    return output;
}

TensorF32 softmax_forward(const TensorF32& input, int dim) {
    return softmax(input, dim);
}

void softmax_backward(const TensorF32& grad_output, const TensorF32& output, 
                      TensorF32& grad_input, int dim) {
    if (dim < 0) dim += grad_output.shape().ndim();
    
    const auto& shape = grad_output.shape().dims;
    
    // 计算 stride
    std::vector<int64_t> stride(shape.size());
    stride[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= shape[i];
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < shape.size(); i++) inner_size *= shape[i];
    
    int64_t dim_size = shape[dim];
    
    const float* grad_out_data = grad_output.data();
    const float* out_data = output.data();
    float* grad_in_data = grad_input.data();
    
    // 初始化 grad_input 为 0
    grad_input.zero_();
    
    // 计算梯度: dL/dx_i = y_i * (dL/dy_i - sum(y_j * dL/dy_j))
    #pragma omp parallel for
    for (int64_t outer = 0; outer < outer_size; outer++) {
        for (int64_t inner = 0; inner < inner_size; inner++) {
            int64_t base_idx = outer * stride[dim] * dim_size + inner;
            
            // 计算 sum(y_j * dL/dy_j)
            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; d++) {
                int64_t idx = base_idx + d * stride[dim];
                sum += out_data[idx] * grad_out_data[idx];
            }
            
            // 计算梯度
            for (int64_t d = 0; d < dim_size; d++) {
                int64_t idx = base_idx + d * stride[dim];
                grad_in_data[idx] = out_data[idx] * (grad_out_data[idx] - sum);
            }
        }
    }
}

} // namespace ppml

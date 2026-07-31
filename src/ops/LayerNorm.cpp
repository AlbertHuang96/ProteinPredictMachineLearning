#include "rfaa/Ops.h"
#include <cmath>
#include <omp.h>

namespace rfaa {

TensorF32 layer_norm(const TensorF32& input, float eps) {
    // 假设最后一个维度是特征维度
    const auto& shape = input.shape().dims;
    int64_t last_dim = shape.back();
    int64_t num_features = input.numel() / last_dim;
    
    TensorF32 output(input.shape(), input.device());
    float* data = output.data();
    const float* input_data = input.data();
    
    #pragma omp parallel for
    for (int64_t i = 0; i < num_features; i++) {
        // 计算均值
        float sum = 0.0f;
        for (int64_t j = 0; j < last_dim; j++) {
            sum += input_data[i * last_dim + j];
        }
        float mean = sum / last_dim;
        
        // 计算方差
        float var_sum = 0.0f;
        for (int64_t j = 0; j < last_dim; j++) {
            float diff = input_data[i * last_dim + j] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / last_dim;
        
        // 归一化
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int64_t j = 0; j < last_dim; j++) {
            data[i * last_dim + j] = (input_data[i * last_dim + j] - mean) * inv_std;
        }
    }
    
    return output;
}

TensorF32 layer_norm(const TensorF32& input, const TensorF32& gamma, 
                    const TensorF32& beta, float eps) {
    // 假设最后一个维度是特征维度
    const auto& shape = input.shape().dims;
    int64_t last_dim = shape.back();
    int64_t num_features = input.numel() / last_dim;
    
    const float* gamma_data = gamma.data();
    const float* beta_data = beta.data();
    
    TensorF32 output(input.shape(), input.device());
    float* data = output.data();
    const float* input_data = input.data();
    
    #pragma omp parallel for
    for (int64_t i = 0; i < num_features; i++) {
        // 计算均值
        float sum = 0.0f;
        for (int64_t j = 0; j < last_dim; j++) {
            sum += input_data[i * last_dim + j];
        }
        float mean = sum / last_dim;
        
        // 计算方差
        float var_sum = 0.0f;
        for (int64_t j = 0; j < last_dim; j++) {
            float diff = input_data[i * last_dim + j] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / last_dim;
        
        // 归一化并应用缩放和偏移
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int64_t j = 0; j < last_dim; j++) {
            data[i * last_dim + j] = (input_data[i * last_dim + j] - mean) * inv_std * gamma_data[j] + beta_data[j];
        }
    }
    
    return output;
}

} // namespace rfaa

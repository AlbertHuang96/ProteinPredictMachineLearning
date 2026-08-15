/* #include "ppml/Ops.h"
#include <cmath>
#include <omp.h>

namespace ppml {

TensorF32 relu(const TensorF32& input) {
    TensorF32 output = input;
    float* data = output.data();
    int64_t n = input.numel();
    
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        data[i] = std::max(0.0f, data[i]);
    }
    
    return output;
}

TensorF32 relu6(const TensorF32& input) {
    TensorF32 output = input;
    float* data = output.data();
    int64_t n = input.numel();
    
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        data[i] = std::max(0.0f, std::min(6.0f, data[i]));
    }
    
    return output;
}

TensorF32 gelu(const TensorF32& input) {
    TensorF32 output = input;
    float* data = output.data();
    int64_t n = input.numel();
    
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        float x = data[i];
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        data[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
    
    return output;
}

TensorF32 swish(const TensorF32& input) {
    TensorF32 output = input;
    float* data = output.data();
    int64_t n = input.numel();
    
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        data[i] = data[i] / (1.0f + std::exp(-data[i]));
    }
    
    return output;
}

TensorF32 sigmoid(const TensorF32& input) {
    TensorF32 output = input;
    float* data = output.data();
    int64_t n = input.numel();
    
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
    
    return output;
}

TensorF32 tanh(const TensorF32& input) {
    TensorF32 output = input;
    float* data = output.data();
    int64_t n = input.numel();
    
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        data[i] = std::tanh(data[i]);
    }
    
    return output;
}

} // namespace ppml
 */
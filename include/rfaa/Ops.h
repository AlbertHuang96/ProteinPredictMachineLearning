#pragma once
#include "Tensor.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace rfaa {

// Softmax 操作
/* TensorF32 softmax(const TensorF32& input, int dim = -1);
TensorF32 softmax_forward(const TensorF32& input, int dim = -1);
void softmax_backward(const TensorF32& grad_output, const TensorF32& output, 
                      TensorF32& grad_input, int dim = -1);

// 激活函数
TensorF32 relu(const TensorF32& input);
TensorF32 relu6(const TensorF32& input);
TensorF32 gelu(const TensorF32& input);
TensorF32 swish(const TensorF32& input);
TensorF32 sigmoid(const TensorF32& input);
TensorF32 tanh(const TensorF32& input);

// Layer Normalization
TensorF32 layer_norm(const TensorF32& input, float eps = 1e-5);
TensorF32 layer_norm(const TensorF32& input, const TensorF32& gamma, 
                    const TensorF32& beta, float eps = 1e-5); */


// 工具函数
// Note: matmul and batch_matmul are now in MathUtils.h

} // namespace rfaa

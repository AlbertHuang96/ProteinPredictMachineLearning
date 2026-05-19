#pragma once

#include "Tensor.h"
#include <vector>

namespace rfaa {

/**
 * @brief Create a vector containing integers from start to end-1
 * 
 * Equivalent to Python's torch.arange(start, end) or numpy.arange(start, end)
 * 
 * @param start Start value (inclusive)
 * @param end End value (exclusive)
 * @return std::vector<int> Vector containing [start, start+1, ..., end-1]
 */
std::vector<int> arange(int start, int end);

/**
 * @brief Matrix multiplication
 * 
 * Computes the matrix product of two 2D tensors.
 * Equivalent to PyTorch's torch.matmul(a, b) for 2D tensors.
 * 
 * @param a Left operand, shape (M, K)
 * @param b Right operand, shape (K, N)
 * @return TensorF32 Result tensor, shape (M, N)
 * @throws RFAAError If inputs are not 2D tensors
 * @throws RFAAError If dimensions don't match (a.shape[1] != b.shape[0])
 */
TensorF32 matmul(const TensorF32& a, const TensorF32& b);

/**
 * @brief Batch matrix multiplication
 * 
 * Computes the batch matrix multiplication of two 3D tensors.
 * Each batch slice is multiplied independently.
 * Equivalent to PyTorch's torch.bmm(a, b).
 * 
 * @param a Left operand, shape (B, M, K)
 * @param b Right operand, shape (B, K, N)
 * @return TensorF32 Result tensor, shape (B, M, N)
 * @throws RFAAError If inputs are not 3D tensors
 * @throws RFAAError If batch dimension B or inner dimension K doesn't match
 */
TensorF32 batch_matmul(const TensorF32& a, const TensorF32& b);

} // namespace rfaa

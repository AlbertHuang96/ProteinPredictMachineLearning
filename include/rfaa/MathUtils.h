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
 * @brief Compute mean along a dimension
 * 
 * @param input Input tensor
 * @param dim Dimension along which to compute mean
 * @return TensorF32 Output tensor with dim dimension reduced
 */
TensorF32 mean(const TensorF32& input, int dim);


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

/**
 * @brief One-hot encoding for categorical data
 * 
 * Converts a vector of indices to one-hot encoded tensor.
 * Equivalent to PyTorch's torch.nn.functional.one_hot() or tf.one_hot().
 * 
 * Example:
 *   indices = [0, 2, 1] with num_classes = 4
 *   output = [[1, 0, 0, 0],
 *             [0, 0, 1, 0],
 *             [0, 1, 0, 0]]
 * 
 * @param indices Vector of indices (each index should be in [0, num_classes-1])
 * @param num_classes Number of classes (length of one-hot vector, default 8)
 * @return TensorF32 One-hot encoded tensor, shape (indices.size(), num_classes)
 * @throws RFAAError If any index is out of range [0, num_classes-1]
 */
TensorF32 one_hot(const std::vector<int>& indices, int num_classes = 8);

/**
 * @brief One-hot encoding for a single index 
 * 
 * Converts a single index to one-hot encoded vector.
 * 
 * @param index Index to encode (should be in [0, num_classes-1])
 * @param num_classes Number of classes (length of one-hot vector, default 8)
 * @return TensorF32 One-hot encoded tensor, shape (1, num_classes)
 * @throws RFAAError If index is out of range [0, num_classes-1]
 */
TensorF32 one_hot(int index, int num_classes = 8);

/**
 * @brief One-hot encoding for sequence data
 * 
 * Converts a sequence of indices to one-hot encoded tensor.
 * Equivalent to PyTorch's F.one_hot() for sequence data.
 * 
 * Example:
 *   seq = [[0, 2, 1], [3, 1, 0]] with num_classes = 5
 *   output shape = (2, 3, 5)
 *   output[b, i, c] = 1 if seq[b, i] == c else 0
 * 
 * @param seq Input sequence tensor, shape (B, L) with integer indices in [0, num_classes-1]
 * @param num_classes Number of classes (default 21 for amino acids)
 * @return TensorF32 One-hot encoded tensor, shape (B, L, num_classes)
 * @throws RFAAError If seq is not 2D tensor
 * @throws RFAAError If any index is out of range [0, num_classes-1]
 */
TensorF32 one_hot_seq(const TensorF32& seq, int num_classes = 21);

/**
 * @brief Outer sum of two tensors (via broadcasting)
 * 
 * Computes the outer sum of two 4D tensors, equivalent to PyTorch:
 *   result = left[:, 1, :, :] + right[:, :, 1, :]
 * 
 * where:
 *   - left has shape (B, 1, L, D) - query embedding
 *   - right has shape (B, L, 1, D) - token embedding
 *   - result has shape (B, L, L, D) - pair representation
 * 
 * This creates a pair representation where:
 *   result[b, i, j, d] = left[b, 0, i, d] + right[b, j, 0, d]
 * 
 * @param left Left tensor, shape (B, 1, L, D)
 * @param right Right tensor, shape (B, L, 1, D)
 * @return TensorF32 Result tensor, shape (B, L, L, D)
 * @throws RFAAError If inputs are not 4D tensors
 * @throws RFAAError If batch dimension B or feature dimension D doesn't match
 * @throws RFAAError If left shape[1] != 1 or right shape[2] != 1
 */
TensorF32 outer_sum(const TensorF32& left, const TensorF32& right);

/**
 * @brief Outer product of two tensors (element-wise multiplication via broadcasting)
 * 
 * Computes the outer product of two 4D tensors, equivalent to PyTorch:
 *   result = left[:, 1, :, :] * right[:, :, 1, :]
 * 
 * where:
 *   - left has shape (B, 1, L, D) - query embedding
 *   - right has shape (B, L, 1, D) - token embedding
 *   - result has shape (B, L, L, D) - pair representation
 * 
 * @param left Left tensor, shape (B, 1, L, D)
 * @param right Right tensor, shape (B, L, 1, D)
 * @return TensorF32 Result tensor, shape (B, L, L, D)
 * @throws RFAAError If inputs are not 4D tensors
 * @throws RFAAError If batch dimension B or feature dimension D doesn't match
 * @throws RFAAError If left shape[1] != 1 or right shape[2] != 1
 */
TensorF32 outer_product(const TensorF32& left, const TensorF32& right);

} // namespace rfaa

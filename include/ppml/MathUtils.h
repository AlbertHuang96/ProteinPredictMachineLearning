#pragma once

#include "Tensor.h"
#include <vector>

namespace ppml {

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
 * @throws PPMLError If inputs are not 2D tensors
 * @throws PPMLError If dimensions don't match (a.shape[1] != b.shape[0])
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
 * @throws PPMLError If inputs are not 3D tensors
 * @throws PPMLError If batch dimension B or inner dimension K doesn't match
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
 * @throws PPMLError If any index is out of range [0, num_classes-1]
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
 * @throws PPMLError If index is out of range [0, num_classes-1]
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
 * @throws PPMLError If seq is not 2D tensor
 * @throws PPMLError If any index is out of range [0, num_classes-1]
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
 * @throws PPMLError If inputs are not 4D tensors
 * @throws PPMLError If batch dimension B or feature dimension D doesn't match
 * @throws PPMLError If left shape[1] != 1 or right shape[2] != 1
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
 * @throws PPMLError If inputs are not 4D tensors
 * @throws PPMLError If batch dimension B or feature dimension D doesn't match
 * @throws PPMLError If left shape[1] != 1 or right shape[2] != 1
 */
TensorF32 outer_product(const TensorF32& left, const TensorF32& right);

/**
 * @brief Triangle multiplication for pair features
 * 
 * Outgoing:  einsum('bikd,bjkd->bijd', left, right/L)
 * Incoming:  einsum('bkid,bkjd->bijd', left, right/L)
 * 
 * @param left     (B, I, K, D) for outgoing, (B, K, I, D) for incoming
 * @param right    (B, J, K, D) for outgoing, (B, K, J, D) for incoming
 * @param L        normalization factor (sequence length)
 * @param outgoing true for outgoing, false for incoming
 * @return         (B, I, J, D)
 */
TensorF32 triangle_mult(const TensorF32& left, const TensorF32& right, float L, bool outgoing);

/**
 * @brief msa2pair outer-product-mean（值版）。
 *
 * einsum('bikd,bjkd->bijd(de)', left, right/N) — 收缩 seq 维 N（dims[1]），
 * 保留残基 i,j，且特征维做笛卡尔积 D×D→D*D（与 msa2pair_out_proj_ 输入 D*D 匹配）。
 *
 *   left  (B, N, L, D)   —— dims[1]=N 为被收缩的序列维，dims[2]=L 为残基
 *   right (B, N, L, D)
 *   dst   (B, L, L, D*D) —— dst[b,i,j,(d1*D+d2)] = (1/N)*sum_n left[b,n,i,d1]*right[b,n,j,d2]
 *
 * 注：这与 outer_product((B,1,L,D),(B,L,1,D)) 不同——后者是纯外积且特征维不扩展。
 * 此前 msa2pair 误用 outer_product(left(B,N,L,D), right(B,N,L,D)) 造成签名不符，应改用本函数。
 *
 * @param left  Left tensor, shape (B, N, L, D)
 * @param right Right tensor, shape (B, N, L, D)
 * @return Result tensor, shape (B, L, L, D*D)
 * @throws PPMLError If inputs are not 4D tensors, or batch/feature dims mismatch
 */
TensorF32 outer_product_mean(const TensorF32& left, const TensorF32& right);

/**
 * @brief pair2pair gate 的 outer product（值版，纯外积，特征维笛卡尔积）。
 *
 * gate[(d1*D+d2), i, j, b] = left[b,i,d1] * right[b,j,d2] —— 无收缩，特征维 D×D→D*D。
 *
 *   left  (B, L, D)   —— dims[1]=L 为残基
 *   right (B, L, D)
 *   dst   (B, L, L, D*D) —— 与 pair2pair_gate_proj_ 输入 D*D 匹配
 *
 * 注：这与 outer_product((B,1,L,D),(B,L,1,D)) 不同——后者不扩展特征维（输出 D 而非 D*D），
 * 与 gate_proj 的 D*D 输入不匹配；此前值版 gate 误用 `outer_product((B,L,16),(B,L,16))`
 * 既签名不符（3D vs 4D）又语义不符（特征维不扩展），应改用本函数。
 *
 * @param left  Left tensor, shape (B, L, D)
 * @param right Right tensor, shape (B, L, D)
 * @return Result tensor, shape (B, L, L, D*D)
 * @throws PPMLError If inputs are not 3D tensors, or batch/feature dims mismatch
 */
TensorF32 outer_product_cartesian(const TensorF32& left, const TensorF32& right);

} // namespace ppml

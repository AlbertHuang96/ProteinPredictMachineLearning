#pragma once

#include "Tensor.h"
#include <random>

namespace rfaa {

/**
 * @brief Dropout layer with optional broadcast dimension
 * 
 * This is a dropout layer that supports broadcasting along a specific dimension.
 * During training, it applies dropout with probability p_drop and scales the
 * output by 1/(1-p_drop) to maintain the expected value.
 * During evaluation, it acts as identity.
 * 
 * Equivalent to PyTorch Dropout with broadcast support.
 */
class Dropout {
public:
    /**
     * @brief Construct a new Dropout layer
     * 
     * @param broadcast_dim Dimension along which to broadcast the dropout mask.
     *                      If -1 (default), each element is dropped independently.
     *                      Similar to Python's broadcast_dim=None.
     * @param p_drop Dropout probability (default: 0.15)
     */
    explicit Dropout(int broadcast_dim = -1, float p_drop = 0.15f);
    
    /**
     * @brief Set training mode
     * 
     * @param training If true, apply dropout. If false, act as identity.
     */
    void set_training(bool training);
    
    /**
     * @brief Check if in training mode
     * 
     * @return true if in training mode
     */
    bool is_training() const;
    
    /**
     * @brief Forward pass
     * 
     * @param x Input tensor
     * @return TensorF32 Output tensor (same shape as input)
     */
    TensorF32 forward(const TensorF32& x);
    
private:
    int broadcast_dim_;   // -1 means no broadcast (independent dropout per element)
    float p_drop_;        // Dropout probability
    bool training_;       // Training mode flag
    
    // Random number generator for Bernoulli sampling
    std::mt19937 rng_;
    std::bernoulli_distribution dist_;
};

} // namespace rfaa

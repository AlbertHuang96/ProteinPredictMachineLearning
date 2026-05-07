#include "rfaa/Model.h"
#include <random>

namespace rfaa {

// Embedding 层实现

class EmbeddingLayer {
public:
    EmbeddingLayer(int num_embeddings, int embedding_dim)
        : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
        // Xavier 初始化
        weights_ = zeros<float>({num_embeddings, embedding_dim}, Device::CPU);
        std::random_device rd;
        std::mt19937 gen(rd());
        float scale = sqrtf(2.0f / (num_embeddings + embedding_dim));
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (int i = 0; i < num_embeddings * embedding_dim; ++i) {
            weights_.data()[i] = dist(gen);
        }
    }
    
    TensorF32 forward(const TensorF32& indices) {
        // indices: (B, L) 整数索引
        // output: (B, L, D)
        int B = indices.shape().dims[0];
        int L = indices.shape().dims[1];
        
        TensorF32 output({B, L, embedding_dim_}, indices.device());
        
        // CPU 实现
        if (indices.device() == Device::CPU) {
            for (int b = 0; b < B; ++b) {
                for (int l = 0; l < L; ++l) {
                    int idx = static_cast<int>(indices.data()[b * L + l]);
                    idx = std::max(0, std::min(idx, num_embeddings_ - 1));
                    std::memcpy(
                        output.data() + (b * L + l) * embedding_dim_,
                        weights_.data() + idx * embedding_dim_,
                        embedding_dim_ * sizeof(float)
                    );
                }
            }
        }
        // CUDA 实现应调用 kernel
        
        return output;
    }
    
private:
    int num_embeddings_;
    int embedding_dim_;
    TensorF32 weights_;
};

class LinearLayer {
public:
    LinearLayer(int in_features, int out_features, bool bias = true)
        : in_features_(in_features), out_features_(out_features), has_bias_(bias) {
        weight_ = zeros<float>({out_features, in_features}, Device::CPU);
        if (bias) {
            bias_ = zeros<float>({out_features}, Device::CPU);
        }
        
        // 初始化
        std::random_device rd;
        std::mt19937 gen(rd());
        float scale = sqrtf(2.0f / in_features);
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (int i = 0; i < out_features * in_features; ++i) {
            weight_.data()[i] = dist(gen);
        }
    }
    
    TensorF32 forward(const TensorF32& x) {
        // x: (..., in_features)
        // output: (..., out_features)
        
        int batch = x.shape().numel() / in_features_;
        TensorF32 output(Shape({batch, out_features_}), x.device());
        
        // 简化的矩阵乘法 (实际应使用 cuBLAS)
        for (int b = 0; b < batch; ++b) {
            for (int o = 0; o < out_features_; ++o) {
                float sum = has_bias_ ? bias_.data()[o] : 0.0f;
                for (int i = 0; i < in_features_; ++i) {
                    sum += x.data()[b * in_features_ + i] * weight_.data()[o * in_features_ + i];
                }
                output.data()[b * out_features_ + o] = sum;
            }
        }
        
        // 恢复原始形状 (最后维变为 out_features)
        auto out_shape = x.shape();
        out_shape.dims.back() = out_features_;
        return output.view(out_shape);
    }
    
private:
    int in_features_, out_features_;
    bool has_bias_;
    TensorF32 weight_;
    TensorF32 bias_;
};

// LayerNorm
class LayerNorm {
public:
    LayerNorm(int normalized_shape, float eps = 1e-5)
        : normalized_shape_(normalized_shape), eps_(eps) {
        gamma_ = zeros<float>({normalized_shape}, Device::CPU);
        beta_ = zeros<float>({normalized_shape}, Device::CPU);
        
        // 初始化为 gamma=1, beta=0
        for (int i = 0; i < normalized_shape; ++i) {
            gamma_.data()[i] = 1.0f;
        }
    }
    
    TensorF32 forward(const TensorF32& x) {
        // 在最后一个维度上做 LayerNorm
        int batch = x.shape().numel() / normalized_shape_;
        TensorF32 output(x.shape(), x.device());
        
        for (int b = 0; b < batch; ++b) {
            // 计算均值
            float mean = 0.0f;
            for (int i = 0; i < normalized_shape_; ++i) {
                mean += x.data()[b * normalized_shape_ + i];
            }
            mean /= normalized_shape_;
            
            // 计算方差
            float var = 0.0f;
            for (int i = 0; i < normalized_shape_; ++i) {
                float diff = x.data()[b * normalized_shape_ + i] - mean;
                var += diff * diff;
            }
            var /= normalized_shape_;
            
            // 归一化
            float inv_std = 1.0f / sqrtf(var + eps_);
            for (int i = 0; i < normalized_shape_; ++i) {
                float normalized = (x.data()[b * normalized_shape_ + i] - mean) * inv_std;
                output.data()[b * normalized_shape_ + i] = 
                    normalized * gamma_.data()[i] + beta_.data()[i];
            }
        }
        
        return output;
    }
    
private:
    int normalized_shape_;
    float eps_;
    TensorF32 gamma_;
    TensorF32 beta_;
};

} // namespace rfaa

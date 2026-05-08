#include "Tensor.h"

class EmbeddingLayer {
public:
// num_embeddings: 词汇表大小 (NAATOKENS)
// embedding_dim: 嵌入向量维度 (D_STATE)
    EmbeddingLayer(int num_embeddings, int embedding_dim);

    TensorF32 forward(const TensorF32& indices);

}

class LinearLayer {
public:
    LinearLayer(int in_features, int out_features, bool bias = true);
    TensorF32 forward(const TensorF32& x);
}

class LayerNorm {
public:
    LayerNorm(int normalized_shape, float eps = 1e-5);
    TensorF32 forward(const TensorF32& x);
}
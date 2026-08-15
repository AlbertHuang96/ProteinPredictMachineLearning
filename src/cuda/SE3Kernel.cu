#include "ppml/SE3Transformer.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace ppml {

// SE3 Transformer CUDA 核函数

// 径向基函数 (RBF) 编码
__global__ void kernel_rbf_encode(const float* distances, float* rbf_feat,
                                   int B, int L, int n_rbf) {
    int b = blockIdx.z;
    int i = blockIdx.y;
    int j = blockIdx.x;
    int k = threadIdx.x;
    
    if (k >= n_rbf) return;
    
    float d = distances[((b * L + i) * L + j)];
    float mu = 2.0f + k * 20.0f / n_rbf;  // RBF 中心
    float gamma = 10.0f;
    
    rbf_feat[(((b * L + i) * L + j) * n_rbf + k)] = expf(-gamma * (d - mu) * (d - mu));
}

// 向量叉积 (l1 x l1 -> l1)
__global__ void kernel_cross_product(const float* a, const float* b, float* c,
                                      int n, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    for (int d = 0; d < dim; ++d) {
        int base = idx * 3 * dim + d;
        float ax = a[base];
        float ay = a[base + dim];
        float az = a[base + 2 * dim];
        float bx = b[base];
        float by = b[base + dim];
        float bz = b[base + 2 * dim];
        
        c[base] = ay * bz - az * by;
        c[base + dim] = az * bx - ax * bz;
        c[base + 2 * dim] = ax * by - ay * bx;
    }
}

// 点积 (l1 · l1 -> l0)
__global__ void kernel_dot_product(const float* a, const float* b, float* c,
                                    int n, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    for (int d = 0; d < dim; ++d) {
        int base = idx * 3 * dim + d;
        c[idx * dim + d] = a[base] * b[base] 
                         + a[base + dim] * b[base + dim]
                         + a[base + 2 * dim] * b[base + 2 * dim];
    }
}

// SE3 等变卷积 (简化版)
__global__ void kernel_se3_conv(const float* nodes, const float* edges,
                                 const float* coords,
                                 float* out_nodes, float* out_coords,
                                 int n_nodes, int node_dim, int edge_dim) {
    int node_i = blockIdx.x;
    if (node_i >= n_nodes) return;
    
    // 聚合邻居信息
    for (int node_j = 0; node_j < n_nodes; ++node_j) {
        float edge_weight = edges[node_i * n_nodes + node_j];
        
        // 标量特征聚合
        for (int d = 0; d < node_dim; ++d) {
            out_nodes[node_i * node_dim + d] += 
                edge_weight * nodes[node_j * node_dim + d];
        }
        
        // 向量特征聚合 (等变性)
        for (int d = 0; d < 3; ++d) {
            out_coords[(node_i * 3 + d)] += 
                edge_weight * coords[(node_j * 3 + d)];
        }
    }
}

// 坐标更新：应用旋转和平移
__global__ void kernel_update_coords(const float* coords, const float* rot,
                                      const float* trans, float* new_coords,
                                      int B, int L) {
    int b = blockIdx.y;
    int l = blockIdx.x;
    int atom = threadIdx.x;
    
    if (atom >= 3) return;  // 3 atoms (N, CA, C)
    
    int base = ((b * L + l) * 3 + atom) * 3;
    
    // 旋转变换 (3x3)
    float x = coords[base];
    float y = coords[base + 1];
    float z = coords[base + 2];
    
    int rot_base = ((b * L + l) * 3 + atom) * 9;
    float nx = rot[rot_base] * x + rot[rot_base + 3] * y + rot[rot_base + 6] * z;
    float ny = rot[rot_base + 1] * x + rot[rot_base + 4] * y + rot[rot_base + 7] * z;
    float nz = rot[rot_base + 2] * x + rot[rot_base + 5] * y + rot[rot_base + 8] * z;
    
    // 平移
    int trans_base = ((b * L + l) * 3 + atom) * 3;
    new_coords[base] = nx + trans[trans_base];
    new_coords[base + 1] = ny + trans[trans_base + 1];
    new_coords[base + 2] = nz + trans[trans_base + 2];
}

} // namespace ppml

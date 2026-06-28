#ifndef RFAA_SE3_TRANSFORMER_H
#define RFAA_SE3_TRANSFORMER_H

#include "Tensor.h"
#include "Model.h"
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <array>

namespace rfaa {

// 前向声明
class Tensor;

namespace se3 {

// ============================================================================
// 数学工具函数声明
// ============================================================================

// 计算球谐基 eijk 和 R_ij (简化版本)
void compute_basis(const std::vector<float>& positions, 
                   const std::vector<float>& orientations,
                   std::vector<float>& eijk, 
                   std::vector<float>& R_ij,
                   int batch_size, 
                   int n_nodes);

// 计算向量 r 和距离 d (简化版本)
void compute_r(const std::vector<float>& positions,
              std::vector<float>& r,
              std::vector<float>& d,
              int batch_size,
              int n_nodes);

// 同时计算基和 r (简化版本)
void get_basis_and_r(const std::vector<float>& positions,
                     const std::vector<float>& orientations,
                     std::vector<float>& eijk,
                     std::vector<float>& R_ij,
                     std::vector<float>& r,
                     std::vector<float>& d,
                     int batch_size,
                     int n_nodes);

// 计算矩阵 A 的零空间 (简化版本)
std::vector<float> null_project(const std::vector<float>& A, int m, int n);

// 生成 Q_J 矩阵 (简化版本)
std::vector<std::vector<float>> get_Q_J(int J_max);

// 笛卡尔坐标转球坐标 (简化版本)
void cart2spher(const std::vector<float>& x,
                std::vector<float>& rho,
                std::vector<float>& theta,
                std::vector<float>& phi,
                int n);

// SO(3) 不可约表示 Wigner D 矩阵 (简化版本)
std::vector<float> irr_repr(int l, float theta, float phi, int n_angles);

} // namespace se3

// ============================================================================
// SE3 等变神经网络组件
// ============================================================================

// Fiber 定义：表示 SE(3) 群的不可约表示的多重性
struct Fiber {
    std::vector<int> multiplicities;  // 每个不可约表示的多重性
    std::vector<int> degrees;         // 每个不可约表示的度数 l
    
    Fiber() = default;
    Fiber(const std::vector<int>& mults, const std::vector<int>& degs);
    
    size_t size() const { return multiplicities.size(); }
    int total_multiplicity() const;
};

// SE3 特征：存储不同类型特征的容器
struct SE3Features {
    std::vector<Tensor> features;  // 特征列表，每个特征对应一个不可约表示
    
    SE3Features() = default;
    SE3Features(const Fiber& fiber, const Tensor& prototype, int batch_size);
    
    void zero_();
    void add_(const SE3Features& other);
    SE3Features operator+(const SE3Features& other) const;
};

// SE3 基：存储预计算的基函数
struct SE3Basis {
    std::vector<Tensor> basis;  // 基函数列表
    
    SE3Basis() = default;
    void compute(const Tensor& positions, const Tensor& orientations, int J_max);
};

// GConvSE3：SE(3) 等变图卷积层
class GConvSE3 {
public:
    // 构造函数
    GConvSE3(const Fiber& fiber_in, 
             const Fiber& fiber_out, 
             int J_max = 2);
    
    // 前向传播
    SE3Features forward(const SE3Features& x, 
                       const SE3Basis& basis,
                       const Tensor& edge_index);
    
    // 获取参数
    std::vector<Tensor> parameters() const;
    
private:
    Fiber fiber_in_;
    Fiber fiber_out_;
    int J_max_;
    
    // 权重参数（简化：实际需要为每个 (l_in, l_out) 对设置权重）
    std::vector<Tensor> weights_;
    std::vector<Tensor> biases_;
};

// GNormSE3：SE(3) 等变归一化层
class GNormSE3 {
public:
    GNormSE3(const Fiber& fiber, float eps = 1e-5f);
    
    SE3Features forward(const SE3Features& x);
    
private:
    Fiber fiber_;
    float eps_;
    std::vector<Tensor> scales_;
    std::vector<Tensor> biases_;
};

// GSE3Res：SE(3) 等变残差块
class GSE3Res {
public:
    GSE3Res(const Fiber& fiber, int J_max = 2, float dropout = 0.0f);
    
    SE3Features forward(const SE3Features& x,
                       const SE3Basis& basis,
                       const Tensor& edge_index,
                       bool training = false);
    
private:
    Fiber fiber_;
    int J_max_;
    float dropout_;
    
    std::unique_ptr<GConvSE3> conv1_;
    std::unique_ptr<GNormSE3> norm1_;
    std::unique_ptr<GConvSE3> conv2_;
    std::unique_ptr<GNormSE3> norm2_;
};

// GNormBias：SE(3) 等变偏置
class GNormBias {
public:
    GNormBias(const Fiber& fiber);
    
    SE3Features forward(const SE3Features& x);
    
private:
    Fiber fiber_;
    std::vector<Tensor> biases_;
};

// TFN：张量场网络层（简化版本）
class TFN {
public:
    TFN(const Fiber& fiber_in,
        const Fiber& fiber_out,
        int J_max = 2,
        bool use_layer_norm = true);
    
    SE3Features forward(const SE3Features& x,
                       const SE3Basis& basis,
                       const Tensor& edge_index);
    
private:
    Fiber fiber_in_;
    Fiber fiber_out_;
    int J_max_;
    bool use_layer_norm_;
    
    std::unique_ptr<GConvSE3> conv_;
    std::unique_ptr<GNormSE3> norm_;
    std::unique_ptr<GNormBias> bias_;
};

// SE3Transformer：完整的 SE(3) 等变 Transformer
class SE3Transformer {
public:
    // 构造函数
    SE3Transformer(int num_layers = 6,
                  int hidden_dim = 128,
                  int J_max = 2,
                  float dropout = 0.1f,
                  int num_heads = 8);
    
    // 前向传播
    SE3Features forward(const SE3Features& x,
                       const Tensor& positions,
                       const Tensor& orientations,
                       const Tensor& edge_index,
                       bool training = false);
    
    // 获取所有参数
    std::vector<Tensor> parameters() const;
    
    // 设置参数
    void set_parameters(const std::vector<Tensor>& params);
    
    // 保存到文件
    bool save(const std::string& filepath) const;
    
    // 从文件加载
    bool load(const std::string& filepath);
    
private:
    int num_layers_;
    int hidden_dim_;
    int J_max_;
    float dropout_;
    int num_heads_;
    
    // 输入投影
    std::unique_ptr<TFN> input_proj_;
    
    // Transformer 层
    std::vector<std::unique_ptr<GSE3Res>> layers_;
    
    // 输出投影
    std::unique_ptr<TFN> output_proj_;
    
    // 输入/输出 Fiber 定义
    Fiber fiber_in_;
    Fiber fiber_hidden_;
    Fiber fiber_out_;
    
    // 初始化权重
    void init_weights();
    
    // 计算注意力（简化版本）
    Tensor compute_attention(const SE3Features& x,
                            const Tensor& positions,
                            const Tensor& edge_index);
};

} // namespace rfaa

#endif // RFAA_SE3_TRANSFORMER_H

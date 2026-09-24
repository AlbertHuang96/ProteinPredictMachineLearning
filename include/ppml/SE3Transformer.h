#ifndef PPML_SE3_TRANSFORMER_H
#define PPML_SE3_TRANSFORMER_H

#include "Tensor.h"
#include "Embedding.h"
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <array>
#include <map>
#include <set>
#include <unordered_map>

namespace ppml {

// 前向声明
template<typename T> class Tensor;

// SE3 配置 (与 PPMLConfig 对齐)
struct SE3Config {
    int num_degrees   = 2;   // SE(3) 表示的 degree 数
    int num_channels  = 32;  // 通道数
    int div           = 4;   // 除法因子
    int n_heads       = 4;   // 注意力头数
    int n_layers      = 2;   // GCN 层数
    int hidden_dim    = 128; // 隐藏层维度
    int node_dim      = 288; // 节点特征维度 (D_MSA + D_STATE)
    int edge_dim      = 193; // 边特征维度 (D_PAIR + 64 + 1)
    int l0_in_feats   = 32;  // degree-0 输入特征
    int l0_out_feats  = 32;  // degree-0 输出特征
    int l1_in_feats   = 16;  // degree-1 边特征
    std::vector<int> l0_features = {32};  // degree-0 每层特征
    std::vector<int> l1_features = {3};   // degree-1 每层特征
};

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

// ============================================================================
// 球谐函数预计算类 (带 Legendre 多项式缓存)
// ============================================================================

class SphericalHarmonics {
public:
    SphericalHarmonics() { clear(); }

    // 清除缓存：每次计算新 θ 时调用
    void clear();

    // 获取连带勒让德多项式 P_l^m(x), x = cos(θ)
    // 带备忘录 (memoization), 同一 x 下低阶值被复用
    double lpmv(int l, int m, double x);

    // 获取单个 tesseral (实) 球谐函数 Y_l^m(θ, φ)
    double get_element(int l, int m, double theta, double phi);

    // 获取度数为 l 的全部球谐函数 [Y_l^{-l}, ..., Y_l^l]
    // 返回 (2l+1) 个值, 索引为 m+l
    std::vector<double> get(int l, double theta, double phi);

private:
    // 缓存: leg_cache_[(l,m)] = P_l^m(x) 在当前 x=cos(θ) 下的值
    // 哈希实现: 将 (l,m) 打包为 int64_t key
    std::unordered_map<int64_t, double> leg_cache_;

    static int64_t make_key(int l, int m) {
        return (static_cast<int64_t>(l) << 32) | static_cast<int64_t>(m + 1024);
    }

    // 辅助函数
    double semifactorial(int x);       // x!!
    double pochhammer(int x, int k);   // (x)_k
    double neg_lpmv(int l, int m, double y);  // 负阶修正因子
};


// 图构建结果
struct GraphData {
    TensorI64 edge_index;  // (2, num_edges) 源节点和目标节点索引
    TensorF32 edge_d;      // (num_edges, 3) 边距离向量 (CA坐标差)
    TensorF32 edge_w;      // (num_edges, E) 边 pair 特征

    GraphData() = default;
};

// 构建图：根据坐标、pair 特征和残基索引生成消息传递图
// - xyz:   (B, L, 3, 3) 骨架坐标 (N, CA, C)
// - pair:  (B, L, L, E) Trunk 输出的 pair 特征
// - idx:   (B, L) 残基索引
// - top_k: 每个节点的最大空间近邻数
// - kmin:  序列相邻阈值 (|i-j| < kmin 总是连边)
GraphData make_graph(const TensorF32& xyz,
                     const TensorF32& pair,
                     const TensorI64& idx,
                     int top_k = 64,
                     int kmin = 9);

// ============================================================
// 【refined topo pass：L1 / L2】（2026-09-23）—— 环境变量门控（见 PPML.cpp 的 PPML_SE3_TOPO=l1|l2 ✓）
//   背景：per_block 每个 block 边界都用**最新坐标**重跑 kNN ⇒ 边集随坐标漂移（edge flapping ✗）：
//     ① 微积分：G(x) 是分片常值 ⇒ 不连续、a.e. 梯度为 0（拓扑选择**不可微** ✗）；
//     ② margin 引理：|Δd_ij| ≤ ‖δ_i‖+‖δ_j‖ ≤ 2Δ ⇒ **Δ < m_i/4**（双侧）/ **Δ_i < m_i/2**（单侧）时不翻边 ✓
//        （m_i = d_(k+1) − d_(k) ✓；推导见 TopoPassRefinement.md 附录 A.1.3 ✓）
//   L1 = **冻结边索引**（首次边界算一次）+ 每边界用**实时坐标**重算几何（edge_d / edge_w ✓），成本 O(E) ✓
//   L2 = 在 L1 基础上，只对"越出 margin 信任域"的节点做**局部** kNN 重算（Ω 门控 + 候选集 C_i ✓），
//        并默认对 Ω 做**邻居封闭**（Ω ← Ω ∪ N(E(x⁰), Ω) ✓）—— 否则"邻居移动导致自己翻边"会被漏掉 ✗
//        候选集 C_i = 冻结出邻居 ∪ 其二跳（与 python_scripts/topo_policy_replay.py 的 local_repair 同构 ✓）
//   ⚠️ 契约：只在 `cfg` 与 `st` 都非空且 (l1||l2) 时走 refined ✓；否则**行为与 make_graph 逐位一致** ✓
//   ⚠️ `TopoRefineState` 应放在 **每次 forward 的局部**（PPMLModel::forward_graph ✓）：
//      它持有 x⁰ 快照 + 冻结边索引 + margin/κ + 邻接表，跨 block 边界存活、随 forward 自然重置 ✓
// ============================================================
struct TopoRefineCfg {
    bool  l1      = false;   // 冻结边索引 + 实时几何 ✓
    bool  l2      = false;   // margin 门控局部重算 ✓
    bool  l3      = false;   // ★ 阻尼 + 信任域基准 B = x⁰ + clamp(γ(x−x⁰), ±κ) ✓（见下 ✓）
    bool  closure = true;    // L2 的 Ω 邻居封闭（默认开 ✓，附录 A.4.3 推论 3 ✓）
    float sigma   = 0.5f;    // κ_i = σ·m_i/2 ✓（σ≤1/2 = "可证不翻边"；σ≈2~4 = "追踪真值" ✓）
    float gamma   = 1.0f;    // L3 的回放阻尼 ✓（γ=1 纯 clamp；γ<1 滞后跟踪 ✓；γ=0 ⇒ 冻结基准 ✓）
    bool  debug   = false;   // 打印 [TOPO-L1/L2/L3] 统计 ✓
};

struct TopoRefineState {
    bool ready = false;
    int  B = 0, L = 0, top_k = 0, kmin = 0;
    int64_t E0 = 0;                                  // |E(x⁰)| ✓
    std::vector<int64_t> src, tgt;                   // 冻结边集 E(x⁰)（host 副本 ✓：GraphData 是 move-only ✗）
    std::vector<std::vector<int64_t>> adj_out;       // 冻结出邻接表（L2 候选集/邻居封闭用 ✓）
    std::vector<uint64_t> keys;                      // E(x⁰) 的 (src,tgt) 排序键（算对称差 = D 分子 ✓）
    std::vector<float>   x0;                         // x⁰ 的整份骨架坐标 (B*L*3*3) ✓
    std::vector<float>   margin;                     // m_i（B*L；k ≥ L−1 ⇒ +inf ✓）
    std::vector<float>   kappa;                      // κ_i = σ·m_i/2 ✓
    int      calls = 0;                              // refined 调用次数（首次建立快照 ✓）
    int64_t  last_omega = 0, last_diff = 0;          // 上次 |Ω| / |E_t Δ E(x⁰)|（诊断 ✓）
    void reset() { *this = TopoRefineState(); }
};

GraphData make_graph_refined(const TensorF32& xyz,
                             const TensorF32& pair,
                             const TensorI64& idx,
                             int top_k,
                             int kmin,
                             const TopoRefineCfg& cfg,
                             TopoRefineState& st);

// 获取键合邻居信息
// - idx: (B, L) 残基索引
// - 返回: (B, L, L, 1)
//   +1: j 是 i 的下一个残基, -1: j 是 i 的上一个残基, 0: 非键合
TensorF32 get_bonded_neigh(const TensorI64& idx);

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
    std::vector<TensorF32> features;  // 特征列表，每个特征对应一个不可约表示
    
    SE3Features() = default;
    SE3Features(const Fiber& fiber, const TensorF32& prototype, int batch_size);
    SE3Features(SE3Features&&) = default;
    SE3Features& operator=(SE3Features&&) = default;
    // 禁止拷贝
    SE3Features(const SE3Features&) = delete;
    SE3Features& operator=(const SE3Features&) = delete;
    
    void zero_();
    void add_(const SE3Features& other);
    SE3Features operator+(const SE3Features& other) const;
};

// SE3 基：存储预计算的基函数
// SE3 基：对标 Python get_basis(G, max_degree, ...)
// 预计算每条边上的 Y_J(theta, phi) 并按需返回 CG 耦合后的 basis
struct SE3Basis {
    // edge_Y[J]: (E, 2J+1) — 预计算的 Y_J 球谐函数, J = 0..J_max
    std::vector<TensorF32> edge_Y;
    int J_max_ = 0;

    // get_basis 缓存: key=(d_in, d_out) → (E, 1, 2*d_out+1, 1, 2*d_in+1, num_freq)
    mutable std::map<std::pair<int,int>, TensorF32> cache_;  // mutable: const get_basis 可修改

    SE3Basis() = default;

    // 对标: r_ij = get_spherical_from_cartesian_torch(edge_d)
    //       Y   = precompute_sh(r_ij, 2*max_degree)
    // edge_d: (E, 3) 笛卡尔相对位移向量
    void compute(const TensorF32& edge_d, int J_max);

    // 对标 Python get_basis 中的双层循环:
    //   for d_in, d_out: K_Js = [Y[J] @ Q_J for J in range(...)]
    //                   basis = stack(K_Js, -1).view(...)
    // 返回: (E, 1, 2*d_out+1, 1, 2*d_in+1, 2*min(d_in,d_out)+1)
    const TensorF32& get_basis(int d_in, int d_out) const;

    // Clebsch-Gordan 变换矩阵 Q_J(J, d_in, d_out)
    // 形状: (2*d_out+1, 2*d_in+1, 2*J+1)
    // 对标 Python: _basis_transformation_Q_J(J, d_in, d_out)
    static TensorF32 q_matrix(int J, int d_in, int d_out);
};


// ============================================================================
// RadialFunc: NN 参数化的径向剖线函数
// 对标 Python: RadialFunc(num_freq, in_dim, out_dim, edge_dim)
// ============================================================================
class RadialFunc {
public:
    // 构造函数
    // - num_freq: 输出频率数 (来自 Clebsch-Gordan 分解)
    // - in_dim:   输入通道数 nc_in
    // - out_dim:  输出通道数 nc_out
    // - edge_dim: 边嵌入维度
    RadialFunc(int num_freq, int in_dim, int out_dim, int edge_dim = 0);

    // 前向传播
    // x: (E, edge_dim+1) 边特征 (距离 + 嵌入)
    // 返回: (E, out_dim, 1, in_dim, 1, num_freq) 六维径向权重
    TensorF32 forward(const TensorF32& x);

    // 图模式前向：x 为图节点 (E, edge_dim+1) → R (E, out_dim, in_dim, num_freq)
    // 布局 dims=[num_freq, in_dim, out_dim, E]（dims[0]=最内维），去掉值版的 "1" 维。
    // 使用 mul_mat / add_impl / mul / scale / relu 等已有图 op；BN 用逐通道 affine（gamma/beta repeat 广播）。
    TensorF32* forward_graph(TensorF32* x);

    // 获取所有参数
    std::vector<TensorF32*> parameters();

private:
    int num_freq_;      // 频率数
    int in_dim_;        // 输入通道数
    int out_dim_;       // 输出通道数
    int edge_dim_;      // 边嵌入维度
    int mid_dim_ = 32;  // 隐藏层维度

    // 三层 MLP: (edge_dim+1) → 32 → 32 → num_freq*in_dim*out_dim
    LinearLayer* linear1_ = nullptr;  // (edge_dim+1) → 32
    LinearLayer* linear2_ = nullptr;  // 32 → 32
    LinearLayer* linear3_ = nullptr;  // 32 → num_freq*in_dim*out_dim

    // BN (简化: 存储 gamma/beta 做逐通道仿射, 对标 BN1d)
    TensorF32* bn1_gamma_ = nullptr;  // (32,)
    TensorF32* bn1_beta_  = nullptr;
    TensorF32* bn2_gamma_ = nullptr;  // (32,)
    TensorF32* bn2_beta_  = nullptr;

    // 简化 BN forward: 沿 batch 维归一化 + affine
    TensorF32 bn_forward(const TensorF32& x, TensorF32* gamma, TensorF32* beta);
};

// ============================================================================
// PairwiseConv: 两个单度特征之间的 SE(3)-等变卷积
// 对标 Python: PairwiseConv(degree_in, nc_in, degree_out, nc_out, edge_dim)
// ============================================================================
class PairwiseConv {
public:
    // 构造函数
    // - degree_in:  输入特征度数
    // - nc_in:      输入通道数
    // - degree_out: 输出特征度数
    // - nc_out:     输出通道数
    // - edge_dim:   边嵌入维度
    PairwiseConv(int degree_in, int nc_in,
                 int degree_out, int nc_out,
                 int edge_dim = 0);

    // 前向传播
    // feat:  (E, edge_dim+1) 边特征
    // basis: 预计算球谐基, slice for (degree_in, degree_out)
    //        形状 (E, 1, 1, 1, d_out, num_freq)
    // 返回: (E, d_out·nc_out, d_in·nc_in) 等变卷积核矩阵
    TensorF32 forward(const TensorF32& feat, const TensorF32& basis);

    // 图模式前向：feat 为图节点 (E, edge_dim+1)，basis 为常量 (E,1,d_out,1,d_in,num_freq)
    // 返回 kernel 图节点 (E, out*d_out, in*d_in)，dims=[in*d_in, out*d_out, E]。
    // 用 per_edge_matmul 在固定 (mo,mi) 块上逐边做 R_co ⊗ basis 收缩，再 concat 组装。
    TensorF32* forward_graph(TensorF32* feat, const TensorF32& basis);

    // 获取参数
    std::vector<TensorF32*> parameters();

private:
    int degree_in_, nc_in_;    // 输入度数和通道数
    int degree_out_, nc_out_;  // 输出度数和通道数
    int num_freq_;             // 2*min(degree_in, degree_out) + 1
    int d_out_;                // 2*degree_out + 1
    int d_in_;                 // 2*degree_in + 1
    int edge_dim_;

    RadialFunc rp_;  // 径向剖线函数
};

// ============================================================================
// GConvSE3Partial: SE(3)-等变图卷积 (节点 → 边, partial 形式)
// 对标 Python: GConvSE3Partial(f_in, f_out, edge_dim, x_ij)
//
// Partial convolution = 不做输入通道求和, 保留 (c_in, c_out) 独立结果
// 适用于 attention 机制的 value embedding 计算
// ============================================================================
class GConvSE3Partial {
public:
    // 构造函数
    // - f_in:  输入 Fiber
    // - f_out: 输出 Fiber
    // - edge_dim: 边嵌入维度
    // - x_ij:     "" / "cat" / "add" 相对位置处理方式
    GConvSE3Partial(const Fiber& f_in, const Fiber& f_out,
                    int edge_dim = 0, const std::string& x_ij = "");

    // 前向传播: 返回每条边上的消息 (边级 SE3Features)
    // - h:          节点特征
    // - edge_index: (2, E) 边索引 [src 行, tgt 行]
    // - edge_d:     (E, 3) 边距离向量 (从 make_graph 输出)
    // - edge_w:     (E, edge_dim) 边 pair 特征 (可选, nullptr 表示无)
    // - basis:      预计算球谐基
    // 返回: SE3Features 各度消息, features[i] 形状 (E, m_out_i, 2*d_out_i+1)
    SE3Features forward(const SE3Features& h,
                        const TensorI64& edge_index,
                        const TensorF32& edge_d,
                        const TensorF32* edge_w,
                        const SE3Basis& basis);

    // 图模式前向：返回每个输出度的边消息图节点。
    // 输入（均为图节点）：
    //   h_nodes: 每输入度一个节点特征图节点 (N, m_in*d_dim_in) 扁平
    //   edge_src_idx / edge_tgt_idx: (E,) 源/目标节点 id（float-encoded int）
    //   edge_d: (E,3) 边位移图节点；edge_w: (E,edge_dim) 边特征图节点或 nullptr
    //   basis: 预计算球谐基（值版常量，kernel 生成时转常量叶子）
    // 返回: out[i] = (E, m_out*d_dim_out) 扁平边消息（对应 f_out_.degrees[i]）。
    std::vector<TensorF32*> forward_graph(
        const std::vector<TensorF32*>& h_nodes,
        TensorF32* edge_src_idx,
        TensorF32* edge_tgt_idx,
        TensorF32* edge_d,
        TensorF32* edge_w,
        const SE3Basis& basis);

    // 获取参数
    std::vector<TensorF32*> parameters();

private:
    Fiber f_in_;         // 输入 Fiber (可能被 x_ij 修改)
    Fiber f_in_orig_;    // 原始输入 Fiber
    Fiber f_out_;
    int edge_dim_;
    std::string x_ij_;

    // 度对 → PairwiseConv: kernel_unary_[pair_key(di, d_out)] = PairwiseConv(di, mi, d_out, mo)
    std::map<std::pair<int,int>, PairwiseConv*> kernel_unary_;

    // 消息传递: 对特定输出度 d_out 计算边消息
    // kernels_map: 度对 key → kernel 矩阵
    TensorF32 udf_u_mul_e(int d_out,
                          const SE3Features& h,
                          const TensorI64& edge_index,
                          const TensorF32& edge_d,
                          const std::map<std::pair<int,int>, TensorF32>& kernels_map);
};


class G1x1SE3 {
public:
    G1x1SE3(const Fiber& f_in, const Fiber& f_out);

    SE3Features forward(const SE3Features& x);

    // 图模式前向：节点级 1x1 等变线性（逐度通道混合，不跨度、不经边）。
    // x_nodes: 每输入度一个扁平节点特征图节点 (N, m_in*d_dim) dims=[m_in*d_dim, N]。
    // 返回: out[i] = (N, m_out*d_dim) 扁平图节点 dims=[m_out*d_dim, N]（对应 f_out_ 中出现的度）。
    // 用"块对角权重"mul_mat：W_expanded[(mo*d_dim+dd),(mi*d_dim+dd)]=W[mo,mi]（其余0），
    // 使 out[n,mo,dd]=Σ_mi W[mo,mi]*x[n,mi,dd]（等变：只混合通道，不混合 Wigner 分量 dd）。
    std::vector<TensorF32*> forward_graph(const std::vector<TensorF32*>& x_nodes);

private:
    Fiber f_in_, f_out_;
    // 每个度一个线性层: weights_[degree] 形状 (m_out, m_in)
    // 对标 Python: self.transform[str(d_out)]
    std::unordered_map<int, LinearLayer*> weights_;
};

// GNormSE3：SE(3) 等变归一化层
class GNormSE3 {
public:
    GNormSE3(const Fiber& fiber, float eps = 1e-5f);
    
    SE3Features forward(const SE3Features& x);
    
private:
    Fiber fiber_;
    float eps_;
    std::vector<TensorF32> scales_;
    std::vector<TensorF32> biases_;
};

// ============================================================================
// GMABSE3: SE(3)-等变多头自注意力块
// 对标 Python: GMABSE3(f_value, f_key, n_heads)
// ============================================================================
class GMABSE3 {
public:
    // 构造函数
    // - f_value: 值特征的 Fiber
    // - f_key:   键/查询的 Fiber
    // - n_heads: 注意力头数
    GMABSE3(const Fiber& f_value, const Fiber& f_key, int n_heads);

    // 前向传播
    // - v:         值特征 (边级, 来自 GConvSE3Partial)
    // - k:         键特征 (边级, 来自 GConvSE3Partial)
    // - q:         查询特征 (节点级, 来自 G1x1SE3)
    // - edge_index:(2, E) 边索引
    // 返回: SE3Features (节点级输出)
    SE3Features forward(const SE3Features& v,
                        const SE3Features& k,
                        const SE3Features& q,
                        const TensorI64& edge_index);

    // 图模式前向（节点级输出）。所有输入/输出为扁平图节点，ggml 布局 dims[0]=最内维。
    //   v_nodes: 每 f_value_ 度一个边特征图节点 dims=[m*d_dim, E]（GConvSE3Partial 输出）
    //   k_nodes: 每 f_key_  度一个边特征图节点 dims=[m*d_dim, E]
    //   q_nodes: 每 f_key_  度一个节点特征图节点 dims=[m*d_dim, N]（G1x1SE3 输出）
    //   edge_tgt_idx: (E,) 目标节点 id（float-encoded int）
    //   N: 节点数
    // 返回: out[i] = (N, m*d_dim) 聚合节点特征图节点 dims=[m*d_dim, N]（对应 f_value_.degrees[i]）。
    std::vector<TensorF32*> forward_graph(
        const std::vector<TensorF32*>& v_nodes,
        const std::vector<TensorF32*>& k_nodes,
        const std::vector<TensorF32*>& q_nodes,
        TensorF32* edge_tgt_idx,
        int N);

private:
    Fiber f_value_, f_key_;
    int n_heads_;
    int N_;  // 节点数 (由 forward 确定)

    // fiber2head: (X, m, d_dim) → (X, n_heads, m/n_heads, d_dim)
    TensorF32 fiber2head(const TensorF32& feat, int m, int d_dim);

    // head2fiber: 逆操作 → (X, m, d_dim)
    TensorF32 head2fiber(const TensorF32& feat, int m, int d_dim);

    // 边级点积: k[e] · q[tgt_node] → (E, n_heads)
    TensorF32 e_dot_v(const TensorF32& k_edge, const TensorF32& q_node,
                      const TensorI64& edge_index, int64_t E);

    // edge_softmax: 对每条边的入边做 softmax
    TensorF32 edge_softmax(const TensorF32& e, const TensorI64& edge_index,
                           int64_t E);
};

// ============================================================================
// GSE3Res: SE(3)-等变残差注意力块
// 对标 Python: GSE3Res(f_in, f_out, edge_dim, div, n_heads, learnable_skip, skip, selfint, x_ij)
// ============================================================================
class GSE3Res {
public:
    // 构造函数
    // - f_in:    输入 Fiber
    // - f_out:   输出 Fiber
    // - edge_dim:边嵌入维度
    // - div:     注意力空间通道压缩比 (默认 4)
    // - n_heads: 注意力头数
    // - skip:    跳跃连接方式 "cat" / "sum"
    // - x_ij:    相对位置处理 ""/"cat"/"add"
    GSE3Res(const Fiber& f_in, const Fiber& f_out,
            int edge_dim = 0, int div = 4, int n_heads = 1,
            const std::string& skip = "cat", const std::string& x_ij = "");

    // 前向传播
    // - h:          节点特征
    // - edge_index: (2, E)
    // - edge_d:     (E, 3)
    // - edge_w:     (E, edge_dim) 可选
    // - basis:      预计算球谐基
    SE3Features forward(const SE3Features& h,
                        const TensorI64& edge_index,
                        const TensorF32& edge_d,
                        const TensorF32* edge_w,
                        const SE3Basis& basis);

    // 图模式前向（节点级）。所有输入/输出为扁平图节点，ggml 布局 dims[0]=最内维。
    //   h_nodes: 每 f_in_ 度一个节点特征图节点 dims=[m*d_dim, N]
    //   edge_src_idx / edge_tgt_idx: (E,) 源/目标节点 id
    //   edge_d: (E,3) dims=[3,E]；edge_w: (E,edge_dim) dims=[edge_dim,E] 或 nullptr
    //   basis: 预计算球谐基（值版常量）
    //   N: 节点数
    // 返回: out[i] = (N, m*d_dim) 扁平节点特征图节点 dims=[m*d_dim, N]（对应 f_out_.degrees[i]）。
    std::vector<TensorF32*> forward_graph(
        const std::vector<TensorF32*>& h_nodes,
        TensorF32* edge_src_idx,
        TensorF32* edge_tgt_idx,
        TensorF32* edge_d,
        TensorF32* edge_w,
        const SE3Basis& basis,
        int N);

    // 获取参数
    std::vector<TensorF32*> parameters();

private:
    Fiber f_in_, f_out_;
    int edge_dim_, div_, n_heads_;
    std::string skip_, x_ij_;

    // 中间 Fiber: f_mid_out (输出度 ÷ div 通道), f_mid_in (仅 f_in 存在的度)
    Fiber f_mid_out_;
    Fiber f_mid_in_;
    // skip=='cat' 时的拼接 Fiber（cat_map 合并 f_mid_out 与 f_in 各度通道数）;
    // 记录度顺序供 forward_graph 按相同顺序 concat 后喂给 out_proj_。
    Fiber cat_fiber_;

    // 子模块
    GConvSE3Partial* v_proj_ = nullptr;  // 值投影
    GConvSE3Partial* k_proj_ = nullptr;  // 键投影
    G1x1SE3*         q_proj_ = nullptr;  // 查询投影
    GMABSE3*         attn_   = nullptr;  // 多头注意力
    G1x1SE3*         out_proj_ = nullptr; // 输出投影
};


// GNormBias：SE(3) 等变偏置
class GNormBias {
public:
    GNormBias(const Fiber& fiber);

    SE3Features forward(const SE3Features& x);

    // 图模式前向：对每度节点特征图节点做等变非线性。
    // 输入 x_nodes[i] = (N, m*d_dim) 扁平图节点 dims=[m*d_dim, N]（对应 fiber_.degrees[i]）。
    // 返回 out[i] = (N, m*d_dim) 扁平图节点 dims=[m*d_dim, N]。
    // 计算：norm=sqrt(clamp(sum_{dd} v², eps))；t=ReLU(norm+bias)；out = v * (t/norm)。
    std::vector<TensorF32*> forward_graph(const std::vector<TensorF32*>& x_nodes);

private:
    Fiber fiber_;

    std::unordered_map<int, TensorF32> bias_;  // bias_[degree]: (1, m) 逐通道偏置
    // 防除零 eps。原先 1e-12 太小：当某节点 degree-1 特征范数 norm→0（图版 forward_graph
    // 经 view+sum_rows 重算 norm，与 mul 用的原 x 非同一路径，norm 近零时 scale3=bias/eps
    // 爆成 ~1e12·bias，沿 SE3 offset 链 MUL_MAT/ROPE 溢出成 inf→NaN。改 1e-5（AF2 标准量级）
    // 将 scale3 与梯度钳制在 ~1e5 内，消除 NaN 级联；norm 正常时 1e-5 相对 norm=O(1) 可忽略。
    float eps_ = 1e-5f;  // 防止除零

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
                       const TensorI64& edge_index);
    
private:
    Fiber fiber_in_;
    Fiber fiber_out_;
    int J_max_;
    bool use_layer_norm_;
    
    //GConvSE3Partial
    //std::unique_ptr<GConvSE3> conv_;
    std::unique_ptr<GNormSE3> norm_;
    std::unique_ptr<GNormBias> bias_;
};

// SE3Transformer：完整的 SE(3) 等变 Transformer
// 对标 Python: SE3Transformer + _build_gcn
//
// Block 结构: [GSE3Res + GNormBias] × num_layers + [GSE3Res (output)]
//   - 隐藏层: div=4 通道压缩注意力
//   - 输出层: div=1 全通道 + GNormBias 非线性
class SE3Transformer {
public:
    // 构造函数
    // - fiber_in:  输入 Fiber  (如 {0:32, 1:3})
    // - fiber_mid: 隐藏层 Fiber (如 {0:32, 1:3})
    // - fiber_out: 输出 Fiber  (如 {0:32, 1:3})
    // - num_layers:残差块 + Norm 的重复次数
    // - edge_dim:  边嵌入维度
    // - div:       注意力通道压缩比 (隐藏层=4, 输出层=1)
    // - n_heads:   注意力头数
    // - x_ij:      是否将相对位置拼接为度1特征
    SE3Transformer(const Fiber& fiber_in, const Fiber& fiber_mid,
                   const Fiber& fiber_out,
                   int num_layers = 2, int edge_dim = 32,
                   int div = 4, int n_heads = 4, bool x_ij = false);
    
    // 便捷构造：从 SE3Config 构建默认 Fiber
    explicit SE3Transformer(const SE3Config& cfg);

    // 前向传播
    // - h:          输入节点特征 SE3Features
    // - edge_index: (2, E) 边索引
    // - edge_d:     (E, 3) 边距离向量
    // - edge_w:     (E, edge_dim) 边 pair 特征 (可选)
    // - basis:      预计算球谐基 (from positions, J_max)
    SE3Features forward(const SE3Features& h,
                        const TensorI64& edge_index,
                        const TensorF32& edge_d,
                        const TensorF32* edge_w,
                        const SE3Basis& basis);

    // 图模式前向（节点级）。所有输入/输出为扁平图节点，ggml 布局 dims[0]=最内维。
    //   h_nodes: 每 fiber_in_ 度一个节点特征图节点 dims=[m*d_dim, N]
    //            （顺序须与 fiber_in_.degrees 一致；当前构造 x_ij=false 无需附加相对位置通道）
    //   edge_src_idx / edge_tgt_idx: (E,) 源/目标节点 id（float-encoded int）
    //   edge_d: (E,3) dims=[3,E]；edge_w: (E,edge_dim) dims=[edge_dim,E] 或 nullptr
    //   basis: 预计算球谐基（值版常量）
    //   N: 节点数
    // 返回: out[i] = (N, m*d_dim) 扁平节点特征图节点 dims=[m*d_dim, N]（对应 fiber_out_.degrees[i]）。
    // 逐块执行 blocks_：GSE3Res（skip='cat'）→ GNormBias（输出层 norm==nullptr 跳过）。
    std::vector<TensorF32*> forward_graph(
        const std::vector<TensorF32*>& h_nodes,
        TensorF32* edge_src_idx,
        TensorF32* edge_tgt_idx,
        TensorF32* edge_d,
        TensorF32* edge_w,
        const SE3Basis& basis,
        int N);

    // 获取参数
    std::vector<TensorF32*> parameters();

    // 输入/输出 Fiber 访问器（供图模式接线时确定 node 度/通道）
    const Fiber& fiber_in()  const { return fiber_in_; }
    const Fiber& fiber_out() const { return fiber_out_; }

private:
    int num_layers_, edge_dim_, div_, n_heads_;
    bool x_ij_;

    Fiber fiber_in_, fiber_mid_, fiber_out_;

    // Block 结构: [GSE3Res + GNormBias] × N + [GSE3Res]
    struct Block {
        GSE3Res*   gcn  = nullptr;   // 等变残差注意力块
        GNormBias* norm = nullptr;   // 等变非线性 (输出 block 为 nullptr)
    };
    std::vector<Block> blocks_;

    // 对标 Python _build_gcn: 构建 block 列表
    void build_gcn();
};


} // namespace ppml

#endif // PPML_SE3_TRANSFORMER_H

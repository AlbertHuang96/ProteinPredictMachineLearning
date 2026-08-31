#include "ppml/SE3Transformer.h"
#include "ppml/Tensor.h"
#include "ppml/ComputeGraph.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

namespace ppml {

// ============================================================================
// se3 命名空间：数学工具函数实现
// ============================================================================

namespace se3 {

struct GraphData;

// 计算球谐基 eijk 和 R_ij (简化实现)
/* void compute_basis(const std::vector<float>& positions, 
                   const std::vector<float>& orientations,
                   std::vector<float>& eijk, 
                   std::vector<float>& R_ij,
                   int batch_size, 
                   int n_nodes) {
    // 简化实现：实际应计算球谐函数的 Clebsch-Gordan 系数
    // 这里仅分配内存并初始化为 0
    
    int basis_dim = 9;  // 简化：假设基的维度为 9
    eijk.resize(batch_size * n_nodes * n_nodes * basis_dim, 0.0f);
    R_ij.resize(batch_size * n_nodes * n_nodes * 9, 0.0f);  // 3x3 旋转矩阵
    
    // 1. 计算相对位置向量
    // 2. 计算球谐函数 Y_lm
    // 3. 计算 Clebsch-Gordan 系数
    // 4. 组合得到 eijk 和 R_ij
} */

// 计算向量 r 和距离 d (简化实现)
/* void compute_r(const std::vector<float>& positions,
              std::vector<float>& r,
              std::vector<float>& d,
              int batch_size,
              int n_nodes) {
    // 简化实现：计算节点间的相对位置和距离
    
    r.resize(batch_size * n_nodes * n_nodes * 3, 0.0f);  // 3D 向量
    d.resize(batch_size * n_nodes * n_nodes, 0.0f);      // 距离
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < n_nodes; ++i) {
            for (int j = 0; j < n_nodes; ++j) {
                int idx_ij = (b * n_nodes * n_nodes + i * n_nodes + j) * 3;
                int idx_i = b * n_nodes * 3 + i * 3;
                int idx_j = b * n_nodes * 3 + j * 3;
                
                // 计算相对位置 r_ij = pos_j - pos_i
                float rx = positions[idx_j] - positions[idx_i];
                float ry = positions[idx_j + 1] - positions[idx_i + 1];
                float rz = positions[idx_j + 2] - positions[idx_i + 2];
                
                r[idx_ij] = rx;
                r[idx_ij + 1] = ry;
                r[idx_ij + 2] = rz;
                
                // 计算距离 d_ij = ||r_ij||
                float dist = std::sqrt(rx * rx + ry * ry + rz * rz);
                d[b * n_nodes * n_nodes + i * n_nodes + j] = dist;
            }
        }
    }
}

// 同时计算基和 r (简化实现)
void get_basis_and_r(const std::vector<float>& positions,
                     const std::vector<float>& orientations,
                     std::vector<float>& eijk,
                     std::vector<float>& R_ij,
                     std::vector<float>& r,
                     std::vector<float>& d,
                     int batch_size,
                     int n_nodes) {
    // 计算 r 和 d
    compute_r(positions, r, d, batch_size, n_nodes);
    
    // 计算基
    compute_basis(positions, orientations, eijk, R_ij, batch_size, n_nodes);
} */

// 计算矩阵 A 的零空间 (简化实现)
std::vector<float> null_project(const std::vector<float>& A, int m, int n) {
    // 简化实现：实际应使用 SVD 计算零空间
    // 这里返回单位矩阵作为占位符
    
    std::vector<float> result(m * n, 0.0f);
    for (int i = 0; i < std::min(m, n); ++i) {
        result[i * n + i] = 1.0f;
    }
    
    // 1. 对 A 进行 SVD 分解：A = U * S * V^T
    // 2. 找到 S 中接近 0 的奇异值对应的 V 的列
    // 3. 这些列构成零空间
    
    return result;
}

// 生成 Q_J 矩阵 (简化实现)
/* std::vector<std::vector<float>> get_Q_J(int J_max) {
    // 简化实现：实际应计算 Clebsch-Gordan 系数的 Q_J 矩阵
    // 这里返回空矩阵作为占位符
    
    std::vector<std::vector<float>> Q_J(J_max + 1);
    for (int J = 0; J <= J_max; ++J) {
        int dim = 2 * J + 1;  // SO(3) 不可约表示的维度
        Q_J[J].resize(dim * dim, 0.0f);
        
        // 简化：设置为单位矩阵
        for (int i = 0; i < dim; ++i) {
            Q_J[J][i * dim + i] = 1.0f;
        }
    }
    
    // 需要计算 Clebsch-Gordan 系数 C_{l1,m1,l2,m2}^{J,M}
    
    return Q_J;
} */

// 笛卡尔坐标转球坐标 (简化实现)
/* void cart2spher(const std::vector<float>& x,
                std::vector<float>& rho,
                std::vector<float>& theta,
                std::vector<float>& phi,
                int n) {
    rho.resize(n);
    theta.resize(n);
    phi.resize(n);
    
    for (int i = 0; i < n; ++i) {
        float x_val = x[i * 3];
        float y_val = x[i * 3 + 1];
        float z_val = x[i * 3 + 2];
        
        // 计算 rho (距离)
        rho[i] = std::sqrt(x_val * x_val + y_val * y_val + z_val * z_val);
        
        // 计算 theta (极坐标角 [0, pi])
        if (rho[i] > 1e-8f) {
            theta[i] = std::acos(z_val / rho[i]);
        } else {
            theta[i] = 0.0f;
        }
        
        // 计算 phi (方位角 [0, 2*pi])
        if (std::abs(x_val) > 1e-8f || std::abs(y_val) > 1e-8f) {
            phi[i] = std::atan2(y_val, x_val);
            if (phi[i] < 0) phi[i] += 2 * M_PI;
        } else {
            phi[i] = 0.0f;
        }
    }
} */

// SO(3) 不可约表示 Wigner D 矩阵 (简化实现)
/* std::vector<float> irr_repr(int l, float theta, float phi, int n_angles) {
    // 简化实现：实际应计算 Wigner D 矩阵 D^l(R)
    // 这里返回单位矩阵作为占位符
    
    int dim = 2 * l + 1;
    std::vector<float> result(dim * dim * n_angles, 0.0f);
    
    // 简化：设置为单位矩阵
    for (int a = 0; a < n_angles; ++a) {
        for (int i = 0; i < dim; ++i) {
            result[a * dim * dim + i * dim + i] = 1.0f;
        }
    }
    
    // D^l_{m,m'}(theta, phi) = e^{-i*m*phi} * d^l_{m,m'}(theta)
    // 其中 d^l 是小 Wigner d 矩阵
    
    return result;
} */

// ============================================================================
// SphericalHarmonics 实现：连带勒让德多项式 + 实球谐函数
// 对标 Python: SphericalHarmonics 类, 带 leg 缓存
// ============================================================================

void SphericalHarmonics::clear() {
    leg_cache_.clear();
}

// ---------------------------------------------------------------------------
// semifactorial(x) — 双阶乘 x!!
// x!! = x · (x-2) · (x-4) · ... · (1 或 2)
// 例: 5!! = 5·3·1 = 15,  6!! = 6·4·2 = 48
// ---------------------------------------------------------------------------
double SphericalHarmonics::semifactorial(int x) {
    double y = 1.0;
    for (int n = x; n > 1; n -= 2) {
        y *= static_cast<double>(n);
    }
    return y;
}

// ---------------------------------------------------------------------------
// pochhammer(x, k) — 上升阶乘 (x)_k
// (x)_k = x · (x+1) · (x+2) · ... · (x+k-1)
// 例: (3)_4 = 3·4·5·6 = 360
// ---------------------------------------------------------------------------
double SphericalHarmonics::pochhammer(int x, int k) {
    double xf = static_cast<double>(x);
    for (int n = x + 1; n < x + k; ++n) {
        xf *= static_cast<double>(n);
    }
    return xf;
}

// ---------------------------------------------------------------------------
// neg_lpmv(l, m, y) — 负阶修正因子
// 当 m < 0 时, P_l^m = (-1)^m / (l+m+1)_{-2m} · P_l^{|m|}
// 公式: P_l^{-m}(x) = (-1)^m · (l-m)!/(l+m)! · P_l^m(x)
// ---------------------------------------------------------------------------
double SphericalHarmonics::neg_lpmv(int l, int m, double y) {
    if (m < 0) {
        // (-1)^m / pochhammer(l+m+1, -2m)
        double sign = (std::abs(m) % 2 == 0) ? 1.0 : -1.0;  // (-1)^m
        y *= sign / pochhammer(l + m + 1, -2 * m);
    }
    return y;
}

// ---------------------------------------------------------------------------
// lpmv(l, m, x) — 连带勒让德多项式 P_l^m(x), 带 Condon-Shortley 相位
// 对标 Python: SphericalHarmonics.lpmv()
//
// 缓存策略:
//   leg_cache_ 以 (l,m) 为键缓存 P_l^m(x) 在当前 x 下的值
//   当 m_abs < l 时, 递归调用自身求解 P_{l-1}^m 和 P_{l-2}^m
//   这些低阶值被缓存, 后续不同 (l',m) 的查询直接复用
//
// 递推公式:
//   P_l^m(x) = ((2l-1)/(l-m)) · x · P_{l-1}^m(x)
//            - ((l+m-1)/(l-m)) · P_{l-2}^m(x)
// ---------------------------------------------------------------------------
double SphericalHarmonics::lpmv(int l, int m, double x) {
    int m_abs = std::abs(m);

    // ---- 查缓存 ----
    int64_t key = make_key(l, m);
    auto it = leg_cache_.find(key);
    if (it != leg_cache_.end()) {
        return it->second;  // 命中, 直接返回
    }

    // ---- 阶 > 度 → 0 ----
    if (m_abs > l) {
        leg_cache_[key] = 0.0;
        return 0.0;
    }

    // ---- 度 0 → 恒为 1 ----
    if (l == 0) {
        leg_cache_[key] = 1.0;
        return 1.0;
    }

    // ---- 边界条件: m_abs == l → P_l^l ----
    // P_l^l(x) = (-1)^l · (2l-1)!! · (1-x²)^{l/2}
    if (m_abs == l) {
        double sign = (m_abs % 2 == 0) ? 1.0 : -1.0;  // (-1)^l
        double y = sign * semifactorial(2 * m_abs - 1);
        // (1-x²)^{l/2}, 注意浮点除法
        y *= std::pow(1.0 - x * x, static_cast<double>(m_abs) / 2.0);
        // 负阶修正
        y = neg_lpmv(l, m, y);
        leg_cache_[key] = y;
        return y;
    }

    // ---- 递归: 先确保 P_{l-1}^m 已缓存 ----
    // Python: self.lpmv(l-1, m, x)  ← 这会触发 P_{l-1}^m 的计算并缓存
    lpmv(l - 1, m, x);  // 副作用: leg_cache_ 中填入 (l-1, m)

    // ---- 递推公式 ----
    // P_l^m = ((2l-1)/(l-m_abs)) · x · P_{l-1}^m
    double y = ((2.0 * l - 1.0) / static_cast<double>(l - m_abs))
             * x * leg_cache_[make_key(l - 1, m_abs)];

    // 当 l - m_abs > 1 时, 需要减去 P_{l-2}^m 项
    if (l - m_abs > 1) {
        // - ((l+m_abs-1)/(l-m_abs)) · P_{l-2}^m
        y -= ((static_cast<double>(l + m_abs - 1)) / static_cast<double>(l - m_abs))
           * leg_cache_[make_key(l - 2, m_abs)];
    }

    // ---- 负阶修正 ----
    if (m < 0) {
        y = neg_lpmv(l, m, y);
    }

    // ---- 存入缓存 ----
    leg_cache_[key] = y;
    return y;
}

// ---------------------------------------------------------------------------
// get_element(l, m, theta, phi) — 单个 tesseral 球谐函数
// Y_l^m(θ, φ) = N_l^m · P_l^{|m|}(cosθ) · { 1, cos(mφ), sin(|m|φ) }
//
// 归一化:
//   N_l^m = √((2l+1)/(4π) · 2 · (l-|m|)!/(l+|m|)!)
//   N_l^0 = √((2l+1)/(4π))
//
// 相位 (Condon-Shortley): 包含在 P_l^m 中
// ---------------------------------------------------------------------------
double SphericalHarmonics::get_element(int l, int m, double theta, double phi) {
    // 参数合法性检查
    // assert(std::abs(m) <= l);

    int m_abs = std::abs(m);

    // 归一化常数: √((2l+1)/(4π))
    double N = std::sqrt((2.0 * l + 1.0) / (4.0 * M_PI));

    // 连带勒让德多项式 P_l^{|m|}(cosθ)
    double x = std::cos(theta);
    double leg = lpmv(l, m_abs, x);

    // 方位角部分
    double Y;
    if (m == 0) {
        Y = N * leg;
    } else if (m > 0) {
        Y = std::cos(static_cast<double>(m) * phi) * leg;
    } else {
        Y = std::sin(static_cast<double>(m_abs) * phi) * leg;
    }

    // 方肩修正: √(2 / (l-|m|+1)_{2|m|})
    // = √(2 · (l-|m|)!/(l+|m|)!)
    N *= std::sqrt(2.0 / pochhammer(l - m_abs + 1, 2 * m_abs));
    Y *= N;

    return Y;
}

// ---------------------------------------------------------------------------
// get(l, theta, phi) — 获取度数为 l 的全部实球谐函数
// 返回 vector<double> 长度 2l+1, 索引 m+l 对应 Y_l^m
//
// 对标 Python: SphericalHarmonics.get(l, theta, phi)
// 先 clear() 清缓存, 然后在循环中调用 get_element
// 同一 cosθ 的多次 lpmv 调用将复用缓存
// ---------------------------------------------------------------------------
std::vector<double> SphericalHarmonics::get(int l, double theta, double phi) {
    // 清空上一轮的缓存 (因为 cosθ 不同了)
    clear();

    std::vector<double> results(2 * l + 1, 0.0);
    for (int m = -l; m <= l; ++m) {
        results[m + l] = get_element(l, m, theta, phi);
    }
    return results;
}


TensorF32 get_bonded_neigh(const TensorI64& idx) {
    // 获取维度
    const auto& shape = idx.shape();
    int B = static_cast<int>(shape.dims[0]);
    int L = static_cast<int>(shape.dims[1]);

    const int64_t* idx_data = idx.data();

    // 输出: (B, L, L, 1)
    TensorF32 result(Shape({B, L, L, 1}), Device::CPU);
    float* result_data = result.data();

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < L; ++i) {
            for (int j = 0; j < L; ++j) {
                int64_t diff = idx_data[b * L + j] - idx_data[b * L + i];
                float val = 0.0f;

                if (diff == 1) {
                    val = 1.0f;    // j 是 i 的下一个残基 (+方向)
                } else if (diff == -1) {
                    val = -1.0f;   // j 是 i 的上一个残基 (-方向)
                }

                int out_idx = (b * L + i) * L + j;  // 最后一维大小为 1
                result_data[out_idx] = val;
            }
        }
    }

    return result;
}

GraphData make_graph(const TensorF32& xyz,
                     const TensorF32& pair,
                     const TensorI64& idx,
                     int top_k,
                     int kmin) {
    // ---- 获取维度 ----
    const auto& xyz_shape = xyz.shape();
    int B = static_cast<int>(xyz_shape.dims[0]);
    int L = static_cast<int>(xyz_shape.dims[1]);
    // xyz: (B, L, 3, 3) — 3 个骨架原子, 各 3D 坐标

    const auto& pair_shape = pair.shape();
    //  图模式（edge_w 图化）下，make_graph 只算拓扑（edge_index/edge_d），pair 可为空
    // （edge_w 由主图 pair 图节点经 edge_gather_rows 在 run_se3_graph 内图化提取）。
    // 空 pair 时 E=0，跳过 edge_w 填充。
    const bool has_pair = pair.numel() > 0 && pair.data() != nullptr;
    int E = has_pair ? static_cast<int>(pair_shape.dims[3]) : 0;  // pair 特征维度

    const float*   xyz_data  = xyz.data();   // xyz[b, i, atom, coord]
    const float*   pair_data = has_pair ? pair.data() : nullptr;  // pair[b, i, j, e]
    const int64_t* idx_data  = idx.data();   // idx[b, i]

    //    partial_sort(begin+actual_top_k) 与 dists[k] 越界 → heap-buffer-overflow
    //    破坏堆 → 后续 free 报 corrupted double-linked list。这是值版 forward 崩溃根因。
    int actual_top_k = std::min(top_k, L - 1);

    // ---- 第 1 步：计算 CA 距离矩阵 D 和序列间隔 sep ----
    // D:     (B, L, L)  CA 原子欧氏距离
    // sep:   (B, L, L)  |idx_j - idx_i|
    std::vector<float> D(B * L * L);
    std::vector<int>   sep(B * L * L);

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < L; ++i) {
            // CA 坐标在原子索引 1: xyz[b, i, 1, :]
            int base_i = ((b * L + i) * 3 + 1) * 3;  // 3 atoms * 3 coords

            for (int j = 0; j < L; ++j) {
                int base_j = ((b * L + j) * 3 + 1) * 3;
                int dij_idx = b * L * L + i * L + j;

                if (i == j) {
                    // 对角线用大值屏蔽, 模拟 +eye*999.9
                    D[dij_idx]   = 999.9f;
                    sep[dij_idx] = 999;
                } else {
                    float dx = xyz_data[base_j]     - xyz_data[base_i];
                    float dy = xyz_data[base_j + 1] - xyz_data[base_i + 1];
                    float dz = xyz_data[base_j + 2] - xyz_data[base_i + 2];
                    D[dij_idx] = std::sqrt(dx * dx + dy * dy + dz * dz);

                    int s = std::abs(static_cast<int>(
                        idx_data[b * L + j] - idx_data[b * L + i]));
                    sep[dij_idx] = s;
                }
            }
        }
    }

    // ---- 第 2 步：收集满足条件的边 ----
    // 条件: top-k 空间近邻  OR  序列间隔 < kmin
    std::vector<int64_t> edge_src;
    std::vector<int64_t> edge_tgt;
    std::vector<float>   edge_d_x, edge_d_y, edge_d_z;
    std::vector<float>   edge_w;  // 展平存储, 每条边 E 个值

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < L; ++i) {
            // --- 2a. 收集 (距离, j) 对, 排除 i==j ---
            std::vector<std::pair<float, int>> dists;
            dists.reserve(L - 1);
            for (int j = 0; j < L; ++j) {
                if (i == j) continue;
                int dij_idx = b * L * L + i * L + j;
                dists.push_back({ D[dij_idx], j });
            }

            // --- 2b. 部分排序取 top-k 最近邻 ---
            std::partial_sort(
                dists.begin(),
                dists.begin() + actual_top_k,
                dists.end());

            // 标记哪些 j 在 top-k 中
            std::vector<bool> in_topk(L, false);
            for (int k = 0; k < actual_top_k; ++k) {
                in_topk[dists[k].second] = true;
            }

            // --- 2c. 遍历所有 j, 添加满足条件的边 ---
            int base_i_ca = ((b * L + i) * 3 + 1) * 3;
            for (int j = 0; j < L; ++j) {
                if (i == j) continue;
                int s = sep[b * L * L + i * L + j];

                if (in_topk[j] || s < kmin) {
                    // 全局节点索引: 将所有 batch 展平
                    edge_src.push_back(static_cast<int64_t>(b * L + i));
                    edge_tgt.push_back(static_cast<int64_t>(b * L + j));

                    // 边距离向量: xyz[b, j, 1, :] - xyz[b, i, 1, :]
                    int base_j_ca = ((b * L + j) * 3 + 1) * 3;
                    edge_d_x.push_back(xyz_data[base_j_ca]     - xyz_data[base_i_ca]);
                    edge_d_y.push_back(xyz_data[base_j_ca + 1] - xyz_data[base_i_ca + 1]);
                    edge_d_z.push_back(xyz_data[base_j_ca + 2] - xyz_data[base_i_ca + 2]);

                    // 边 pair 特征: pair[b, i, j, :]
                    for (int e = 0; e < E; ++e) {
                        int pair_idx = ((b * L + i) * L + j) * E + e;
                        edge_w.push_back(pair_data[pair_idx]);
                    }
                }
            }
        }
    }

    // ---- 第 3 步：构造输出张量 ----
    int64_t num_edges = static_cast<int64_t>(edge_src.size());

    GraphData graph;
    graph.edge_index = TensorI64(Shape({ 2, num_edges }), Device::CPU);
    graph.edge_d     = TensorF32(Shape({ num_edges, 3 }), Device::CPU);
    graph.edge_w     = TensorF32(Shape({ num_edges, E }), Device::CPU);

    // 填充 edge_index: 第 0 行 src, 第 1 行 tgt
    int64_t* ei_data = graph.edge_index.data();
    for (int64_t e = 0; e < num_edges; ++e) {
        ei_data[e]                  = edge_src[e];   // row 0
        ei_data[num_edges + e]      = edge_tgt[e];   // row 1
    }

    // 填充 edge_d: (num_edges, 3)
    float* ed_data = graph.edge_d.data();
    for (int64_t e = 0; e < num_edges; ++e) {
        ed_data[e * 3]     = edge_d_x[e];
        ed_data[e * 3 + 1] = edge_d_y[e];
        ed_data[e * 3 + 2] = edge_d_z[e];
    }

    // 填充 edge_w: (num_edges, E)
    float* ew_data = graph.edge_w.data();
    for (int64_t e = 0; e < num_edges; ++e) {
        for (int f = 0; f < E; ++f) {  // f 代表 feature
            ew_data[e * E + f] = edge_w[e * E + f];
        }
    }

    return graph;
}

} // namespace se3

// ============================================================================
// Fiber 实现
// ============================================================================

Fiber::Fiber(const std::vector<int>& mults, const std::vector<int>& degs)
    : multiplicities(mults), degrees(degs) {
    if (multiplicities.size() != degrees.size()) {
        throw std::invalid_argument("Multiplicities and degrees must have the same size");
    }
}

int Fiber::total_multiplicity() const {
    int total = 0;
    for (int mult : multiplicities) {
        total += mult;
    }
    return total;
}

// ============================================================================
// SE3Features 实现
// ============================================================================

SE3Features::SE3Features(const Fiber& fiber, const TensorF32& prototype, int batch_size) {
    // 根据 fiber 创建特征张量
    features.clear();
    features.reserve(fiber.size());
    
    for (size_t i = 0; i < fiber.size(); ++i) {
        int l = fiber.degrees[i];
        int mult = fiber.multiplicities[i];
        int dim = 2 * l + 1;  // SO(3) 不可约表示的维度
        
        int n_nodes = prototype.numel() / (3);  // 假设 prototype 是位置张量
        
        features.emplace_back(TensorF32({batch_size, n_nodes, mult, dim}, prototype.device()));
        std::fill(features.back().data(), features.back().data() + features.back().numel(), 0.0f);
    }
}

void SE3Features::zero_() {
    for (auto& feat : features) {
        // feat.zero_();
    }
}

void SE3Features::add_(const SE3Features& other) {
    if (features.size() != other.features.size()) {
        throw std::invalid_argument("Features size mismatch");
    }
    
    for (size_t i = 0; i < features.size(); ++i) {
        // features[i] = features[i] + other.features[i];
    }
}

SE3Features SE3Features::operator+(const SE3Features& other) const {
    SE3Features result;
    return result;
}

// ============================================================================
// Q_J(d_out, d_in, J) — 对标 Python _basis_transformation_Q_J
// 形状: (2*d_out+1, 2*d_in+1, 2*J+1)
// Q_J[m_out, m_in, m_J] = ClebschGordan(d_out, m_out-d_out, d_in, m_in-d_in, J, m_J-J)
// ============================================================================
TensorF32 SE3Basis::q_matrix(int J, int d_in, int d_out) {
    int d_o = 2 * d_out + 1;
    int d_i = 2 * d_in  + 1;
    int d_J = 2 * J + 1;

    TensorF32 Q({d_o, d_i, d_J}, Device::CPU);
    Q.zero_();
    float* q = Q.data();

    // 仅对常用低度数 (0, 1, 2) 提供精确 CG 系数
    // 高阶需要完整的 3j-symbol 计算

    // ---------- (d_in=0, d_out, J=d_out) ----------
    // CG: C(J, M, 0, 0, J, M) = 1, 其他 = 0
    if (d_in == 0 && J == d_out) {
        for (int mo = 0; mo < d_o; ++mo) {
            int m_J_idx = mo;  // M = m_out 对应 Y_J 的 m_J = mo
            q[(mo * d_i + 0) * d_J + m_J_idx] = 1.0f;
        }
        return Q;
    }

    // ---------- (d_in, d_out=0, J=d_in) ----------
    if (d_out == 0 && J == d_in) {
        for (int mi = 0; mi < d_i; ++mi) {
            int m_J_idx = mi;
            q[(0 * d_i + mi) * d_J + m_J_idx] = 1.0f;
        }
        return Q;
    }

    // ---------- (d_in=1, d_out=1) ----------
    // 旧实现硬编码 *3/*5 与 mo=3/4，当 d_o 或 d_J < 5（如 d_out=0）时越界写 → ASAN heap-buffer-overflow。
    if (d_in == 1 && d_out == 1) {
        // CG(1,m1, 1,m2, J,M) where M = m1+m2, for J = 0, 1, 2
        auto m_val = [](int idx) { return idx - 1; };  // idx: 0,1,2 → -1,0,1

        if (J == 0) {
            // CG(1,m, 1,-m, 0,0) = (-1)^(1-m) / √3
            float inv_sqrt3 = 1.0f / std::sqrt(3.0f);
            for (int mi = 0; mi < d_i; ++mi) {
                int m = m_val(mi);
                int mo = 0 - m + 1;  // mo = -m 的存储索引
                float sign = ((1 - m) % 2 == 0) ? 1.0f : -1.0f;
                q[(mo * d_i + mi) * d_J + 0] = sign * inv_sqrt3;
            }
            return Q;
        }

        if (J == 1) {
            // CG(1,m1, 1,m2, 1,M) with M=m1+m2
            // 标准 CG 表（m1,m2 ∈ {-1,0,1}，mo=存储索引 m1，mi=存储索引 m2，m_J=存储索引 m1+m2）
            // m1=-1: C(1,-1,1,0,1,-1) = -1/√2,  C(1,-1,1,1,1,0) = 1/√2
            // m1=0:  C(1,0,1,-1,1,-1)= 1/√2,   C(1,0,1,1,1,1) = -1/√2
            // m1=1:  C(1,1,1,-1,1,0) = 1/√2,   C(1,1,1,0,1,1) = -1/√2
            float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
            // 存储索引 mo/mi: m_val 0→-1, 1→0, 2→1; m_J: 值 M ∈ {-1,0,1} → 索引 M+1
            // (mo=-1,mi=0)→M=-1: idx mo=0, mi=1, mJ=0
            q[(0 * d_i + 1) * d_J + 0] = -inv_sqrt2;
            // (mo=-1,mi=1)→M=0:  idx mo=0, mi=2, mJ=1
            q[(0 * d_i + 2) * d_J + 1] =  inv_sqrt2;
            // (mo=0,mi=-1)→M=-1: idx mo=1, mi=0, mJ=0
            q[(1 * d_i + 0) * d_J + 0] =  inv_sqrt2;
            // (mo=0,mi=1)→M=1:  idx mo=1, mi=2, mJ=2
            q[(1 * d_i + 2) * d_J + 2] = -inv_sqrt2;
            // (mo=1,mi=-1)→M=0: idx mo=2, mi=0, mJ=1
            q[(2 * d_i + 0) * d_J + 1] =  inv_sqrt2;
            // (mo=1,mi=0)→M=1:  idx mo=2, mi=1, mJ=2
            q[(2 * d_i + 1) * d_J + 2] = -inv_sqrt2;
            return Q;
        }

        if (J == 2) {
            // CG(1,m1, 1,m2, 2,M) where M=m1+m2
            // 存储索引: mo/mi: m_val 0→-1,1→0,2→1; m_J: M ∈ {-2..2} → 索引 M+2
            float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
            float inv_sqrt6 = 1.0f / std::sqrt(6.0f);
            float sqrt_2_3  = std::sqrt(2.0f / 3.0f);

            q[(0 * d_i + 0) * d_J + 0] = 1.0f;        // (-1,-1)→M=-2
            q[(0 * d_i + 1) * d_J + 1] = inv_sqrt2;   // (-1,0)→M=-1
            q[(1 * d_i + 0) * d_J + 1] = inv_sqrt2;   // (0,-1)→M=-1
            q[(0 * d_i + 2) * d_J + 2] = inv_sqrt6;   // (-1,1)→M=0
            q[(1 * d_i + 1) * d_J + 2] = sqrt_2_3;    // (0,0)→M=0
            q[(2 * d_i + 0) * d_J + 2] = inv_sqrt6;   // (1,-1)→M=0
            q[(1 * d_i + 2) * d_J + 3] = inv_sqrt2;   // (0,1)→M=1
            q[(2 * d_i + 1) * d_J + 3] = inv_sqrt2;   // (1,0)→M=1
            q[(2 * d_i + 2) * d_J + 4] = 1.0f;        // (1,1)→M=2
            return Q;
        }
    }

    // ---- 高阶度的 fallback: 单位近似 ----
    for (int k = 0; k < std::min(d_o, std::min(d_i, d_J)); ++k) {
        q[(k * d_i + k) * d_J + k] = 1.0f;
    }
    return Q;
}

const TensorF32& SE3Basis::get_basis(int d_in, int d_out) const {
    // ---- 检查缓存 ----
    auto key = std::make_pair(d_in, d_out);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second;
    }

    int64_t E = edge_Y[0].shape().dims[0];         // 边数
    int d_out_dim = 2 * d_out + 1;                  // m_out 分量数
    int d_in_dim  = 2 * d_in  + 1;                  // m_in  分量数
    int num_freq  = 2 * std::min(d_in, d_out) + 1;  // J 的个数

    // ---- 对标 Python: 预分配 K_Js 列表 ----
    // 收集所有 J 对应的 K_J = Y[J] @ Q_J.J.T
    // K_Js[j] 形状: (E, d_out_dim * d_in_dim)
    std::vector<TensorF32> K_Js;
    K_Js.reserve(num_freq);

    for (int j = 0; j < num_freq; ++j) {
        int J = std::abs(d_in - d_out) + j;  // J ∈ [|d_in-d_out|, d_in+d_out]

        if (J > J_max_) {
            // 超出预计算范围 → 补 0
            TensorF32 zero_K({E, d_out_dim * d_in_dim}, edge_Y[0].device());
            zero_K.zero_();
            K_Js.push_back(std::move(zero_K));
            continue;
        }

        // ---- 获取 Q_J 矩阵 ----
        // Python: Q_J = _basis_transformation_Q_J(J, d_in, d_out).float().T.to(device)
        // Q_J 形状: (d_out_dim, d_in_dim, 2J+1) → .T → (2J+1, d_out_dim*d_in_dim)
        TensorF32 Q_J = q_matrix(J, d_in, d_out);  // (d_out_dim, d_in_dim, 2J+1)

        // ---- 对标 Python: K_J = torch.matmul(Y[J], Q_J) ----
        // Y[J]: (E, 2J+1)
        // Q_J:  (2J+1, d_out_dim * d_in_dim)  [after .T and reshape]
        // K_J:  (E, d_out_dim * d_in_dim)
        const float* y_data = edge_Y[J].data();
        int d_J = 2 * J + 1;

        // 将 Q_J 从 (d_out_dim, d_in_dim, d_J) 转置为 (d_J, d_out_dim * d_in_dim)
        // 然后做矩阵乘法
        TensorF32 Q_J_T({d_J, d_out_dim * d_in_dim}, edge_Y[0].device());
        float* q_data = Q_J_T.data();
        const float* q_src = Q_J.data();
        for (int mo = 0; mo < d_out_dim; ++mo) {
            for (int mi = 0; mi < d_in_dim; ++mi) {
                for (int m = 0; m < d_J; ++m) {
                    // Q_J[mo, mi, m] → Q_J_T[m, mo*d_in_dim + mi]
                    q_data[m * d_out_dim * d_in_dim + mo * d_in_dim + mi] =
                        q_src[(mo * d_in_dim + mi) * d_J + m];
                }
            }
        }

        // Y[J] @ Q_J_T: (E, 2J+1) @ (2J+1, D) → (E, D)
        int D = d_out_dim * d_in_dim;
        TensorF32 K_J({E, D}, edge_Y[0].device());
        float* kj_data = K_J.data();
        for (int64_t e = 0; e < E; ++e) {
            for (int k = 0; k < D; ++k) {
                float val = 0.0f;
                for (int m = 0; m < d_J; ++m) {
                    val += y_data[e * d_J + m] * q_data[m * D + k];
                }
                kj_data[e * D + k] = val;
            }
        }
        K_Js.push_back(std::move(K_J));
    }

    // ---- 对标 Python: torch.stack(K_Js, -1).view(*size) ----
    // size = (-1, 1, 2*d_out+1, 1, 2*d_in+1, num_freq)
    int D = d_out_dim * d_in_dim;
    TensorF32 result({E, 1, d_out_dim, 1, d_in_dim, num_freq}, edge_Y[0].device());
    float* r_data = result.data();

    for (int64_t e = 0; e < E; ++e) {
        for (int mo = 0; mo < d_out_dim; ++mo) {
            for (int mi = 0; mi < d_in_dim; ++mi) {
                for (int j = 0; j < num_freq; ++j) {
                    int kj_flat_idx = mo * d_in_dim + mi;
                    float val = K_Js[j].data()[e * D + kj_flat_idx];
                    // result[e, 0, mo, 0, mi, j]
                    int64_t r_idx = ((((e * 1 + 0) * d_out_dim + mo) * 1 + 0) * d_in_dim + mi) * num_freq + j;
                    r_data[r_idx] = val;
                }
            }
        }
    }

    // 注意: 不可用 cache_[key].copy_from(result) —— operator[] 默认构造 1 元素 Tensor。
    // 用移动赋值替换。
    cache_[key] = std::move(result);
    return cache_[key];
}

// ============================================================================
// SE3Basis 实现
// 对标 Python:
//   r_ij = get_spherical_from_cartesian_torch(cloned_d)
//   Y = precompute_sh(r_ij, 2*max_degree)
// ============================================================================

void SE3Basis::compute(const TensorF32& edge_d, int J_max) {
    // edge_d: (E, 3) — 每条边的笛卡尔相对位移 (dx, dy, dz)
    const auto& shape = edge_d.shape();
    int64_t E = shape.dims[0];
    int max_Y_degree = 2 * J_max;  // Python: precompute_sh(r_ij, 2*max_degree)
    J_max_ = max_Y_degree;

    const float* d_data = edge_d.data();
    se3::SphericalHarmonics sh;

    // 预分配: edge_Y[J] = (E, 2J+1), J = 0..2*J_max
    edge_Y.resize(max_Y_degree + 1);
    for (int J = 0; J <= max_Y_degree; ++J) {
        edge_Y[J] = TensorF32({E, 2 * J + 1}, edge_d.device());
    }

    // ---- 逐边计算球谐函数 ----
    // 对标: get_spherical_from_cartesian_torch + precompute_sh
    for (int64_t e = 0; e < E; ++e) {
        float dx = d_data[e * 3];
        float dy = d_data[e * 3 + 1];
        float dz = d_data[e * 3 + 2];

        // ---- cart2spher: 笛卡尔 → 球坐标 ----
        // 对标: utils_steerable.get_spherical_from_cartesian_torch(cloned_d)
        float rho = std::sqrt(dx * dx + dy * dy + dz * dz);
        double theta_cart = 0.0;
        double phi_cart   = 0.0;

        if (rho > 1e-8f) {
            theta_cart = std::acos(static_cast<double>(dz) / rho);
            phi_cart   = std::atan2(static_cast<double>(dy), static_cast<double>(dx));
            if (phi_cart < 0.0) phi_cart += 2.0 * M_PI;
        }

        // ---- 球谐函数的 theta/phi ----
        // Python 参考: theta = math.pi - r_ij[..., i_beta]
        //              phi   = r_ij[..., i_alpha]
        double theta_sh = M_PI - theta_cart;
        double phi_sh   = phi_cart;

        // ---- 计算所有 J 的 Y_J(theta, phi) ----
        // 对标 Python: Y = precompute_sh(r_ij, 2*max_degree)
        // SphericalHarmonics::get(J, theta, phi) 返回真实实球谐 [Y_J^{-J}..Y_J^J]
        // （索引 m+J）。此前此处用了全零占位导致所有 basis=0、SE3 卷积核恒 0，
        // 等变/不变性几何基础失效；现接上真实球谐计算。
        // 注：sh.get() 内部每调用一次就 clear() 一次缓存，故对每条边、每个 J 各调一次，
        // 与 Python 逐边 precompute_sh 语义一致（每边独立的 θ/φ）。
        for (int J = 0; J <= max_Y_degree; ++J) {
            std::vector<double> Y = sh.get(J, theta_sh, phi_sh);
            float* y_data = edge_Y[J].data();
            int d_J = 2 * J + 1;
            for (int m = 0; m < d_J; ++m) {
                y_data[e * d_J + m] = static_cast<float>(Y[m]);
            }
        }
    }

    // 清除 get_basis 缓存（数据已变）
    cache_.clear();
}


G1x1SE3::G1x1SE3(const Fiber& f_in, const Fiber& f_out)
    : f_in_(f_in), f_out_(f_out) {
    // 构建 f_in 的 degree → multiplicity 快速查找表
    std::unordered_map<int, int> in_mult;
    for (size_t i = 0; i < f_in_.size(); ++i) {
        in_mult[f_in_.degrees[i]] = f_in_.multiplicities[i];
    }

    // 为每个输出度创建线性层
    // Python: for m_out, d_out in self.f_out.structure:
    //           m_in = self.f_in.structure_dict[d_out]
    //           self.transform[str(d_out)] = nn.Parameter(...)
    for (size_t i = 0; i < f_out_.size(); ++i) {
        int d_out = f_out_.degrees[i];
        int m_out = f_out_.multiplicities[i];

        auto it = in_mult.find(d_out);
        if (it == in_mult.end()) {
            // 跳过 f_in 中不存在的度（不应发生，但安全起见）
            continue;
        }
        int m_in = it->second;  // 输入该度的通道数

        // 创建权重矩阵: (m_out × m_in), Xavier 初始化
        LinearLayer* W = LinearLayer::create(m_in, m_out, /*bias=*/false, /*se3=*/true);
        // create() 内部已经做了 Xavier 初始化 (见 Embedding.h 注释)
        weights_[d_out] = W;
    }
}

SE3Features G1x1SE3::forward(const SE3Features& x) {
    // Python: output = {}
    //         for k, v in features.items():
    //             if str(k) in self.transform.keys():
    //                 output[k] = torch.matmul(self.transform[str(k)], v)
    //         return output

    SE3Features output;

    // 建立 f_in degree → 输入特征索引 的映射
    // SE3Features 构造时按 Fiber.degrees 顺序排列 features[i]
    // 所以 features[0] ↔ f_in_.degrees[0], features[1] ↔ f_in_.degrees[1], ...
    std::unordered_map<int, size_t> in_idx;
    for (size_t i = 0; i < f_in_.size(); ++i) {
        in_idx[f_in_.degrees[i]] = i;
    }

    // 遍历输出 Fiber 的每个度
    output.features.resize(f_out_.size());

    for (size_t i = 0; i < f_out_.size(); ++i) {
        int d_out = f_out_.degrees[i];
        int m_out_d = f_out_.multiplicities[i];
        int d_dim_d = 2 * d_out + 1;

        // 查找权重矩阵
        auto w_it = weights_.find(d_out);
        if (w_it == weights_.end()) {
            // 该度无权重：输出全零 (N, m_out, d_dim)，保持 shape（下游 GMABSE3 读 features[i]，
            // 默认 numel=1 Tensor 会越界）。用 x 的 N 维度（若 x 空则跳过）。
            if (x.features.empty()) continue;
            TensorF32 zero({x.features[0].shape().dims[0], m_out_d, d_dim_d}, x.features[0].device());
            std::fill(zero.data(), zero.data() + zero.numel(), 0.0f);
            output.features[i] = std::move(zero);
            continue;
        }

        // 查找输入特征的索引
        auto idx_it = in_idx.find(d_out);
        if (idx_it == in_idx.end()) {
            // 输入无该度特征：输出全零（同上有界）
            if (x.features.empty()) continue;
            TensorF32 zero({x.features[0].shape().dims[0], m_out_d, d_dim_d}, x.features[0].device());
            std::fill(zero.data(), zero.data() + zero.numel(), 0.0f);
            output.features[i] = std::move(zero);
            continue;
        }

        const TensorF32& v = x.features[idx_it->second];  // 输入特征张量 (N, m_in, d_dim)
        LinearLayer* W = w_it->second;                   // 权重矩阵 (m_out × m_in)

        //    特征按扁平 (batch=N*d_dim, in=m_in) 处理——Wigner 分量 d_dim 与 multiplicity m
        //    混在一起且 in_features_=m_in 与实际特征维 m_in*d_dim 不符 → batch 计算错 → 越界写。
        //    正确语义（对齐图版块对角）：对每个 Wigner 分量 dd 独立做 out=W@v[...,dd]（只混 m）。
        //    即 out[n, mo, dd] = sum_{mi} W[mo,mi] * v[n, mi, dd]。
        const int64_t N = v.shape().dims[0];
        const int64_t d_dim = v.shape().dims[2];
        //    w_dims={in,out}）→ dims[0]=in, dims[1]=out。
        //    m_out 必须以输入张量 v 的实际通道数为准（v.shape().dims[1]），
        //    不能假设等于权重 in_features：skip='cat' 时 out_proj_ 输入是 cat 后的
        //    combined (N, m_total, d_dim)，m_total = m_mid + m_in 可能大于权重 m_in。
        //    若不一致则截断（权重不足列按 0）。
        const int64_t m_w_in  = W->weight()->shape().dims[0];   // in_features
        const int64_t m_out = W->weight()->shape().dims[1];     // out_features
        const int64_t m_in  = v.shape().dims[1];   // 实际输入通道数
        const float* w_data = W->weight()->data();   // (m_out, m_w_in)
        TensorF32 out({N, m_out, d_dim}, v.device());
        float* o_data = out.data();
        const float* v_data = v.data();
        for (int64_t n = 0; n < N; ++n)
            for (int64_t mo = 0; mo < m_out; ++mo)
                for (int64_t dd = 0; dd < d_dim; ++dd) {
                    float s = 0.0f;
                    for (int64_t mi = 0; mi < m_in; ++mi) {
                        const float wv = (mi < m_w_in) ? w_data[mo * m_w_in + mi] : 0.0f;
                        s += wv * v_data[(n * m_in + mi) * d_dim + dd];
                    }
                    o_data[(n * m_out + mo) * d_dim + dd] = s;
                }
        output.features[i] = std::move(out);
    }

    return output;
}


// ============================================================================
// RadialFunc 实现
// ============================================================================

RadialFunc::RadialFunc(int num_freq, int in_dim, int out_dim, int edge_dim)
    : num_freq_(num_freq), in_dim_(in_dim), out_dim_(out_dim), edge_dim_(edge_dim) {
    int input_dim = edge_dim_ + 1;  // +1 是因为 edge features 会拼接距离标量

    // 对标 Python Sequential: Linear → BN → ReLU → Linear → BN → ReLU → Linear
    linear1_ = LinearLayer::create(input_dim, mid_dim_, /*bias=*/true, /*se3=*/true);
    linear2_ = LinearLayer::create(mid_dim_, mid_dim_, /*bias=*/true, /*se3=*/true);
    linear3_ = LinearLayer::create(mid_dim_, num_freq_ * in_dim_ * out_dim_, /*bias=*/true, /*se3=*/true);

    // BN 参数: gamma 初始化为 1, beta 初始化为 0
    //  用 new_param_tensor（context 管理）+ TENSOR_FLAG_PARAM：图模式下这些参数
    //    会进计算图，需有正确的 op=OP_NONE（免 dispatch 报 NOT_SUPPORTED）且标记
    //    PARAM（否则 build_backward_expand 不给它们梯度累加器 → graph_get_grad=nullptr，
    //    方案1 反向测试拿不到 BN 梯度）。
    bn1_gamma_ = new TensorF32(Shape({mid_dim_}), Device::CPU);
    bn1_beta_  = new TensorF32(Shape({mid_dim_}), Device::CPU);
    bn2_gamma_ = new TensorF32(Shape({mid_dim_}), Device::CPU);
    bn2_beta_  = new TensorF32(Shape({mid_dim_}), Device::CPU);
    bn1_gamma_->flag = TENSOR_FLAG_PARAM | TENSOR_FLAG_SE3;
    bn1_beta_->flag  = TENSOR_FLAG_PARAM | TENSOR_FLAG_SE3;
    bn2_gamma_->flag = TENSOR_FLAG_PARAM | TENSOR_FLAG_SE3;
    bn2_beta_->flag  = TENSOR_FLAG_PARAM | TENSOR_FLAG_SE3;

    for (int i = 0; i < mid_dim_; ++i) {
        bn1_gamma_->data()[i] = 1.0f;
        bn1_beta_->data()[i]  = 0.0f;
        bn2_gamma_->data()[i] = 1.0f;
        bn2_beta_->data()[i]  = 0.0f;
    }

    // Kaiming 初始化权重 (对标 nn.init.kaiming_uniform_)
    // LinearLayer::create 内部已做 Xavier/He 初始化
}

// 简化 BN forward: 沿 dim=0 做 mean-std 归一化 + affine
// x: (N, C), gamma/beta: (C,)
TensorF32 RadialFunc::bn_forward(const TensorF32& x, TensorF32* gamma, TensorF32* beta) {
    int N = static_cast<int>(x.shape().dims[0]);
    int C = static_cast<int>(x.shape().dims[1]);

    const float* src   = x.data();
    const float* g_ptr = gamma->data();
    const float* b_ptr = beta->data();

    TensorF32 out({N, C}, x.device());
    float* dst = out.data();

    // 逐通道计算 mean 和 var, 然后归一化 + affine
    for (int c = 0; c < C; ++c) {
        // 计算 mean
        float mean = 0.0f;
        for (int n = 0; n < N; ++n) {
            mean += src[n * C + c];
        }
        mean /= static_cast<float>(N);

        // 计算 var
        float var = 0.0f;
        for (int n = 0; n < N; ++n) {
            float diff = src[n * C + c] - mean;
            var += diff * diff;
        }
        var = var / static_cast<float>(N) + 1e-5f;

        // 归一化 + affine
        float inv_std = 1.0f / std::sqrt(var);
        for (int n = 0; n < N; ++n) {
            dst[n * C + c] = g_ptr[c] * (src[n * C + c] - mean) * inv_std + b_ptr[c];
        }
    }
    return out;
}

TensorF32 RadialFunc::forward(const TensorF32& x) {
    // x: (E, edge_dim+1)

    // ---- Layer 1: Linear → BN → ReLU ----
    TensorF32 h = linear1_->forward(x);         // (E, 32)
    h = bn_forward(h, bn1_gamma_, bn1_beta_);   // BN

    // ReLU: max(0, x)
    {
        float* d = h.data();
        for (int64_t i = 0; i < h.numel(); ++i) {
            if (d[i] < 0.0f) d[i] = 0.0f;
        }
    }

    // ---- Layer 2: Linear → BN → ReLU ----
    h = linear2_->forward(h);                    // (E, 32)
    h = bn_forward(h, bn2_gamma_, bn2_beta_);   // BN

    {
        float* d = h.data();
        for (int64_t i = 0; i < h.numel(); ++i) {
            if (d[i] < 0.0f) d[i] = 0.0f;
        }
    }

    // ---- Layer 3: Linear → reshape ----
    // Linear(32 → num_freq·nc_in·nc_out)
    TensorF32 y = linear3_->forward(h);          // (E, num_freq*in_dim*out_dim)

    // reshape → (E, out_dim, 1, in_dim, 1, num_freq)
    // Python: y.view(-1, self.out_dim, 1, self.in_dim, 1, self.num_freq)
    //  不能用 y = y.view(...)：view 不拥有数据，move 赋值先释放自身再接管悬垂指针。
    int64_t E = y.shape().dims[0];
    TensorF32 y_r({E, out_dim_, 1, in_dim_, 1, num_freq_}, y.device());
    y_r.copy_from(y);   // 行优先扁平拷贝，reshape 安全
    return y_r;
}

std::vector<TensorF32*> RadialFunc::parameters() {
    std::vector<TensorF32*> params;
    params.push_back(linear1_->weight());
    params.push_back(linear1_->bias());
    params.push_back(bn1_gamma_);
    params.push_back(bn1_beta_);
    params.push_back(linear2_->weight());
    params.push_back(linear2_->bias());
    params.push_back(bn2_gamma_);
    params.push_back(bn2_beta_);
    params.push_back(linear3_->weight());
    params.push_back(linear3_->bias());
    return params;
}

// 图模式前向：x: (E, edge_dim+1) 图节点 → R: (E, out, in, num_freq) 图节点
// 布局 dims=[num_freq, in, out, E]（dims[0]=最内维）。
// BN 在图上用逐通道 affine（gamma*x+beta, repeat 广播）近似（值版是按 batch 统计的 BN，
// 这里用参数作逐通道缩放偏移以保持可导）。若需严格对齐值版需额外 batch-norm-over-rows op。
TensorF32* RadialFunc::forward_graph(TensorF32* x) {
    // ---- Layer 1: Linear → BN(affine) → ReLU ----
    TensorF32* h = linear1_->forward_graph(x);          // (E, 32)
    h = add_impl(mul(h, repeat(bn1_gamma_, h)), repeat(bn1_beta_, h), /*inplace=*/false);
    h = relu(h);
    // ---- Layer 2: Linear → BN(affine) → ReLU ----
    h = linear2_->forward_graph(h);                     // (E, 32)
    h = add_impl(mul(h, repeat(bn2_gamma_, h)), repeat(bn2_beta_, h), /*inplace=*/false);
    h = relu(h);
    // ---- Layer 3: Linear → reshape (E, num_freq*in*out) → R dims=[num_freq, in, out, E] ----
    TensorF32* y = linear3_->forward_graph(h);          // (E, num_freq*in*out)
    const int64_t E = y->shape().dims[1];
    //    （dims[0]=num_freq 最内，与 PairwiseConv 的 permute{2,0,1,3} 和
    //    view(R_co, Shape({E,in,nf})) 布局对齐）。原代码 Shape({E,out,in,nf}) 让
    //    dims[0]=E 最内，导致 PairwiseConv 读 R 错位 → kernel 值巨大 → exp 溢出 NaN
    //    （本会话 SE3 测试 `node_n=508 op=8 div` 溢出即此根因）。
    return view(y, Shape({num_freq_, in_dim_, out_dim_, E}));
}

// ============================================================================
// PairwiseConv 实现
// ============================================================================

PairwiseConv::PairwiseConv(int degree_in, int nc_in,
                           int degree_out, int nc_out,
                           int edge_dim)
    : degree_in_(degree_in), nc_in_(nc_in),
      degree_out_(degree_out), nc_out_(nc_out),
      edge_dim_(edge_dim)
    // N_freq = 2*min(l_in, l_out) + 1   ← Clebsch-Gordan 分解
    , num_freq_(2 * std::min(degree_in_, degree_out_) + 1)
    // d_out = 2*l_out + 1   ← 输出 Wigner D-矩阵维度
    , d_out_(2 * degree_out_ + 1)
    // d_in = 2*l_in + 1   ← 输入 Wigner D-矩阵维度
    , d_in_(2 * degree_in_ + 1)
    // 径向剖线函数
    , rp_(num_freq_, nc_in_, nc_out_, edge_dim_) {
}

TensorF32 PairwiseConv::forward(const TensorF32& feat, const TensorF32& basis) {
    // Python:
    //   R = self.rp(feat)
    //   kernel = torch.sum(R * basis[f'{self.degree_in},{self.degree_out}'], -1)
    //   return kernel.view(kernel.shape[0], self.d_out*self.nc_out, -1)

    // Step 1: 径向权重
    // R: (E, nc_out, 1, nc_in, 1, num_freq)
    TensorF32 R = rp_.forward(feat);

    int64_t E = R.shape().dims[0];
    int64_t N_freq = num_freq_;

    const float* r_data = R.data();
    const float* b_data = basis.data();

    // 替换 PairwiseConv::forward 中 Step 2 的注释和索引:

// Step 2: R * basis → sum over last dim (num_freq)
// R:     (E, nc_out, 1, nc_in, 1, num_freq)
// basis: (E, 1, d_out_dim, 1, d_in_dim, num_freq)   ← 修正: 增加 d_in_dim 维度
// 广播:  R(E, nc_out, 1, nc_in, 1, N_freq)
//      * basis(E, 1, d_out, 1, d_in, N_freq)
//     → (E, nc_out, d_out, nc_in, d_in, N_freq)
// sum(-1) → (E, nc_out, d_out, nc_in, d_in)

// 重写 kernel 计算循环: kernel[e, co, mo, ci, mi] = sum_j R[...] * basis[...]
    TensorF32 kernel({E, nc_out_, d_out_, nc_in_, d_in_}, feat.device());
    float* k_data = kernel.data();
    std::memset(k_data, 0, E * nc_out_ * d_out_ * nc_in_ * d_in_ * sizeof(float));

    for (int64_t e = 0; e < E; ++e) {
        for (int co = 0; co < nc_out_; ++co) {
            for (int mo = 0; mo < d_out_; ++mo) {
                for (int ci = 0; ci < nc_in_; ++ci) {
                    for (int mi = 0; mi < d_in_; ++mi) {
                        float val = 0.0f;
                        for (int j = 0; j < N_freq; ++j) {
                            // R[e, co, 0, ci, 0, j]
                            int64_t r_idx = ((e * nc_out_ + co) * 1 + 0) * nc_in_ * 1 * N_freq
                                      + (ci * 1 + 0) * N_freq + j;
                            // basis[e, 0, mo, 0, mi, j]
                            int64_t b_idx = ((((e * 1 + 0) * d_out_ + mo) * 1 + 0) * d_in_ + mi) * N_freq + j;
                            val += r_data[r_idx] * b_data[b_idx];
                        }
                        // kernel[e, co, mo, ci, mi]
                        k_data[(((e * nc_out_ + co) * d_out_ + mo) * nc_in_ + ci) * d_in_ + mi] = val;
                    }
                }
            }
        }
    }

// Step 3: reshape → (E, d_out*nc_out, d_in*nc_in) — 现在 d_in 直接来自 kernel
// kernel shape: (E, nc_out, d_out, nc_in, d_in)
// → permute and reshape: (E, d_out*nc_out, d_in*nc_in)
    TensorF32 kernel_out({E, d_out_ * nc_out_, d_in_ * nc_in_}, feat.device());
    float* ko_data = kernel_out.data();

    for (int64_t e = 0; e < E; ++e) {
        for (int co = 0; co < nc_out_; ++co) {
            for (int mo = 0; mo < d_out_; ++mo) {
                for (int ci = 0; ci < nc_in_; ++ci) {
                    for (int mi = 0; mi < d_in_; ++mi) {
                        int64_t k_idx = (((e * nc_out_ + co) * d_out_ + mo) * nc_in_ + ci) * d_in_ + mi;
                        int64_t row = co * d_out_ + mo;
                        int64_t col = ci * d_in_ + mi;
                        ko_data[(e * d_out_ * nc_out_ + row) * d_in_ * nc_in_ + col] = k_data[k_idx];
                    }
                }
            }
        }
    }

    return kernel_out;


    // 中间结果: (E, nc_out, nc_in, d_out, num_freq)  ← 去掉维度 2,4 的 1
    // 为简化，直接计算收缩
    /* TensorF32 kernel({E, nc_out_, nc_in_, d_out_}, feat.device());
    float* k_data = kernel.data();
    std::memset(k_data, 0, E * nc_out_ * nc_in_ * d_out_ * sizeof(float));

    // kernel[e, c_out, c_in, m] = sum_j R[e,c_out,0,c_in,0,j] * basis[e,0,0,0,m,j]
    for (int64_t e = 0; e < E; ++e) {
        for (int co = 0; co < nc_out_; ++co) {
            for (int ci = 0; ci < nc_in_; ++ci) {
                for (int m = 0; m < d_out_; ++m) {
                    float val = 0.0f;
                    for (int j = 0; j < N_freq; ++j) {
                        // R[e, co, 0, ci, 0, j]
                        int64_t r_idx = ((e * nc_out_ + co) * 1 + 0) * nc_in_ * 1 * N_freq
                                      + (ci * 1 + 0) * N_freq + j;
                        // basis[e, 0, 0, 0, m, j]
                        int64_t b_idx = ((((e * 1 + 0) * 1 + 0) * 1 + 0) * d_out_ + m) * N_freq + j;
                        val += r_data[r_idx] * b_data[b_idx];
                    }
                    k_data[((e * nc_out_ + co) * nc_in_ + ci) * d_out_ + m] = val;
                }
            }
        }
    } */

    // Step 3: reshape → (E, d_out·nc_out, d_in·nc_in)
    // kernel 当前: (E, nc_out, nc_in, d_out)
    // 先 permute 或重组维度: (E, nc_out, d_out) × (nc_in) 然后 reshape
    // Python 等效: kernel.view(E, d_out*nc_out, -1) 其中 -1 = d_in*nc_in
    //
    // kernel: (E, nc_out, nc_in, d_out)
    // 目标:   (E, d_out*nc_out, d_in*nc_in)
    // 即: 将 nc_out 和 d_out 合并, nc_in 和 d_in 合并 (d_in 未显式出现是因为
    //     GConvSE3 中通过消息传递聚合邻居特征时隐含了 d_in 维度)

    // 重组: 按行优先内存顺序, (nc_out, nc_in, d_out) → reshape 到 (d_out*nc_out, nc_in)
    // 但 d_in 未在 kernel 中出现, 需在 reshape 时假定 nc_in 隐含 d_in
    // Python 的 -1 会自动推导为 E*d_out*nc_out*d_in*nc_in / (E*d_out*nc_out) = d_in*nc_in

    // 将 kernel 展平重组: 先展成 (E, d_out*nc_out, nc_in) 然后乘上隐含的 d_in
    // 实际 Python 中 view 做了: kernel(E,nc_out,nc_in,d_out) → (E, d_out*nc_out, d_in*nc_in)
    // 但这需要假定原 kernel 的 nc_in 维度实际代表 d_in*nc_in

    // 当前简化: 直接将 kernel reshape 为输出形状
    //int64_t out_rows = d_out_ * nc_out_;
    //int64_t out_cols = d_in_ * nc_in_;

    //TensorF32 kernel_out({E, out_rows, out_cols}, feat.device());
    //float* ko_data = kernel_out.data();
    //std::memset(ko_data, 0, E * out_rows * out_cols * sizeof(float));

    // 将 kernel[e, co, ci, m] 映射到 kernel_out[e, co*m, ci*m]
    // 实际映射: kernel_out[e, co*d_out + m, ci*d_in + ?] 
    // 因为 d_in 未在 kernel 中出现, 需要从输入的 d_in 维度获得
    /* for (int64_t e = 0; e < E; ++e) {
        for (int co = 0; co < nc_out_; ++co) {
            for (int ci = 0; ci < nc_in_; ++ci) {
                for (int m = 0; m < d_out_; ++m) {
                    int64_t k_idx = ((e * nc_out_ + co) * nc_in_ + ci) * d_out_ + m;
                    float val = k_data[k_idx];

                    // 映射到 (E, d_out*nc_out, d_in*nc_in)
                    // 行: co*d_out + m
                    // 列: 对每个 d_in 分量复制 (因为径向权重不依赖 d_in 分量)
                    for (int n = 0; n < d_in_; ++n) {
                        int64_t row = co * d_out_ + m;
                        int64_t col = ci * d_in_ + n;
                        ko_data[(e * out_rows + row) * out_cols + col] = val;
                    }
                }
            }
        }
    }

    return kernel_out; */
}

std::vector<TensorF32*> PairwiseConv::parameters() {
    return rp_.parameters();
}

// 图模式前向：生成等变卷积核图节点。
// feat: (E, edge_dim+1) 图节点 dims=[edge_dim+1, E]
// basis: 值版常量 (E,1,d_out,1,d_in,num_freq)，内存 index (e,mo,mi,j)=((e*d_out+mo)*d_in+mi)*num_freq+j
// 返回 kernel: dims=[C=in*d_in, R=out*d_out, E]，kernel[e,r=(co,mo),c=(ci,mi)] = Σ_j R[e,co,ci,j]*basis[e,mo,mi,j]
//
// 思路（全部用已有/新图 op，ggml 布局 dims[0]=最内维）：
//   R = RadialFunc_graph(feat) → dims=[num_freq, in, out, E]
//   对固定 co：R_co (E,in,num_freq) dims=[nf,in,E] 用 permute+view+get_rows 提取
//   对固定 (co,mo)、遍历 mi：block_mi = per_edge_matmul(R_co, basis_mo_mi) → (E,in)
//   concat mi → 行 r；unsqueeze + concat r → kernel (E, R, C) dims=[C,R,E]
TensorF32* PairwiseConv::forward_graph(TensorF32* feat, const TensorF32& basis) {
    const int64_t E      = feat->shape().dims[1];
    const int64_t out    = nc_out_;
    const int64_t in     = nc_in_;
    const int64_t dout   = d_out_;
    const int64_t din    = d_in_;
    const int64_t nf     = num_freq_;

    // 1) 径向权重 R (E,out,in,nf) dims=[nf,in,out,E]
    TensorF32* R = rp_.forward_graph(feat);

    // 2) permute R → (out, nf, in, E)：把被切片的输出通道 co 移到最内 dims[0]（get_rows 按 dims[0] 选行），
    //    其余保持 nf,in,E 顺序（nf 最内，便于后续 view 回 [nf,in,E]）。
    //    再 view → 2D (out, E*in*nf)：dims[0]=out（行，供 get_rows 按 co 选），dims[1]=E*in*nf（行内长，nf 最内）。
    //      把待切 co 放 dims[1]（get_rows 当行内长），方向反了；配合本次 get_rows 修复（按 a.dims[0] 选行、
    //      输出 a.dims[1] 长）会切错行并输出错误长度。现改为 co 在 dims[0]、行内长在 dims[1]。
    TensorF32* R_perm = permute(R, std::vector<int>{2, 0, 1, 3});
    const int64_t E_in_nf = E * in * nf;
    TensorF32* R_2d = view(R_perm, Shape({out, E_in_nf}));   // dims=[out, E*in*nf]

    // 收集所有输出行 r=(co,mo)
    std::vector<TensorF32*> kernel_rows;   // 每个 unsqueeze 后 dims=[in*din, 1, E]
    for (int64_t co = 0; co < out; co++) {
        // R_co = R[:,co,:,:] → (E,in,nf) dims=[nf,in,E]
        TensorF32* co_leaf = constant_scalar(static_cast<float>(co));
        TensorF32* R_co_flat = get_rows(R_2d, co_leaf);          // (1, E*in*nf)
        //    per_edge_matmul 读 kernel->dims[2] 作为 E。原 Shape({E,in,nf}) 让 dims[0]=E、
        //    dims[2]=nf，per_edge_matmul 会把 nf 当 E → 索引错乱 → kernel 值巨大 → exp 溢出 NaN。
        TensorF32* R_co = view(R_co_flat, Shape({nf, in, E}));   // dims=[nf,in,E]

        for (int64_t mo = 0; mo < dout; mo++) {
            std::vector<TensorF32*> row_blocks;   // 每个 dims=[in, E]，按 mi 拼 → [in*din, E]
            for (int64_t mi = 0; mi < din; mi++) {
                // 常量叶子 basis_mo_mi (E,nf) dims=[nf,E]，basis_mo_mi[e,j]=basis[e,mo,mi,j]
                std::vector<float> bdat(E * nf);
                for (int64_t e = 0; e < E; e++) {
                    for (int64_t j = 0; j < nf; j++) {
                        bdat[e * nf + j] = basis.data()[((e * dout + mo) * din + mi) * nf + j];
                    }
                }
                TensorF32* b_col = constant_tensor({nf, E}, bdat.data());   // dims=[nf,E]
                // block_mi[e,ci] = Σ_j R[e,co,ci,j]*basis[e,mo,mi,j]  (E,in) dims=[in,E]
                TensorF32* block = per_edge_matmul(R_co, b_col);
                row_blocks.push_back(block);
            }
            // 拼 mi → kernel 行 r 的 (E, in*din) dims=[in*din, E]
            TensorF32* kernel_row = concat_ptr(row_blocks, 0);
            // unsqueeze(dim=1) → dims=[in*din, 1, E]
            TensorF32* row_3d = unsqueeze(kernel_row, 1);
            kernel_rows.push_back(row_3d);
        }
    }

    // 3) 拼所有行 r → kernel (E, out*dout, in*din) dims=[in*din, out*dout, E]
    TensorF32* kernel = concat_ptr(kernel_rows, 1);
    return kernel;
}

// ============================================================================
// GConvSE3Partial 实现
// ============================================================================

GConvSE3Partial::GConvSE3Partial(const Fiber& f_in, const Fiber& f_out,
                                 int edge_dim, const std::string& x_ij)
    : f_in_orig_(f_in), f_out_(f_out), edge_dim_(edge_dim), x_ij_(x_ij) {
    // ---- 处理 x_ij: 拼接相对位置作为额外度1通道 ----
    // Python:
    //   if x_ij == 'cat':
    //       self.f_in = Fiber.combine(f_in, Fiber(structure=[(1,1)]))
    //   else:
    //       self.f_in = f_in
    if (x_ij_ == "cat") {
        // 查找度 1 的位置, 将其 multiplicity +1
        std::vector<int> mults = f_in_orig_.multiplicities;
        std::vector<int> degs  = f_in_orig_.degrees;
        bool found = false;
        for (size_t i = 0; i < degs.size(); ++i) {
            if (degs[i] == 1) {
                mults[i] += 1;  // 增加一个通道给相对位置
                found = true;
                break;
            }
        }
        if (!found) {
            // 原 f_in 无度 1, 添加 (1, 1)
            mults.push_back(1);
            degs.push_back(1);
        }
        f_in_ = Fiber(mults, degs);
    } else {
        f_in_ = f_in_orig_;
    }

    // ---- 为每对 (d_in, d_out) 创建 PairwiseConv ----
    // Python:
    //   for (mi, di) in self.f_in.structure:
    //       for (mo, do) in self.f_out.structure:
    //           self.kernel_unary[f'({di},{do})'] = PairwiseConv(di, mi, do, mo, edge_dim)
    for (size_t i = 0; i < f_in_.size(); ++i) {
        int d_in = f_in_.degrees[i];
        int m_in = f_in_.multiplicities[i];
        for (size_t j = 0; j < f_out_.size(); ++j) {
            int d_out = f_out_.degrees[j];
            int m_out = f_out_.multiplicities[j];

            auto key = std::make_pair(d_in, d_out);
            kernel_unary_[key] = new PairwiseConv(d_in, m_in, d_out, m_out, edge_dim_);
        }
    }
}

// ---------------------------------------------------------------------------
// udf_u_mul_e: 对特定输出度执行消息传递
// Python 对应: GConvSE3Partial.udf_u_mul_e(d_out) 返回的 fnc(edges)
//
// 对每条边 e = (src_node, tgt_node):
//   msg = Σ_{d_in} kernel_{(d_in,d_out)} @ h_src[d_in]
// 其中 kernel 矩阵 (m_out·d_dim_out × m_in·d_dim_in)
// ---------------------------------------------------------------------------
TensorF32 GConvSE3Partial::udf_u_mul_e(
        int d_out,
        const SE3Features& h,
        const TensorI64& edge_index,
        const TensorF32& edge_d,
        const std::map<std::pair<int,int>, TensorF32>& kernels_map) {

    // edge_index: (2, E), row 0 = src, row 1 = tgt
    // 边数 E
    int64_t E = edge_index.shape().dims[1];
    const int64_t* ei_data = edge_index.data();
    // ei_data[idx] = src, ei_data[E + idx] = tgt

    int d_dim_out = 2 * d_out + 1;  // 输出 Wigner D-矩阵维度

    // 查找 d_out 在 f_out_ 中的 multiplicity
    int m_out = 0;
    for (size_t j = 0; j < f_out_.size(); ++j) {
        if (f_out_.degrees[j] == d_out) {
            m_out = f_out_.multiplicities[j];
            break;
        }
    }

    // 输出消息: (E, m_out, d_dim_out)
    TensorF32 msg({E, m_out, d_dim_out}, h.features[0].device());
    float* msg_data = msg.data();
    std::memset(msg_data, 0, E * m_out * d_dim_out * sizeof(float));

    // ---- 累加来自每个输入度 d_in 的消息 ----
    for (size_t i = 0; i < f_in_.size(); ++i) {
        int d_in   = f_in_.degrees[i];
        int m_in   = f_in_.multiplicities[i];
        int d_dim_in = 2 * d_in + 1;

        // 查找该度对的 kernel 矩阵
        auto k_it = kernels_map.find(std::make_pair(d_in, d_out));
        if (k_it == kernels_map.end()) continue;
        const TensorF32& kernel = k_it->second;  // (E, m_out*d_dim_out, m_in*d_dim_in)
        const float* k_data = kernel.data();
        int64_t k_rows = m_out * d_dim_out;
        int64_t k_cols = m_in * d_dim_in;

        // 获取输入节点特征: h.features[i] → (N, m_in, d_dim_in)
        const TensorF32& src_feat_all = h.features[i];  // (N, m_in, d_dim_in)
        const float* src_data = src_feat_all.data();
        int64_t N_per_dim = m_in * d_dim_in;  // stride per node

        // 对每条边做: kernel[e] @ src[edge_index[0,e]]
        // src 展开为 (m_in*d_dim_in, 1) 列向量
        for (int64_t e = 0; e < E; ++e) {
            int64_t src_node = ei_data[e];  // 源节点 ID

            // 获取源节点特征: (m_in*d_dim_in) 维向量
            const float* src_vec = src_data + src_node * N_per_dim;

            // 获取该边的 kernel: (m_out*d_dim_out × m_in*d_dim_in) 矩阵
            const float* K_e = k_data + e * k_rows * k_cols;

            // 矩阵乘法: msg[e] += K_e @ src_vec
            // msg[e]: (m_out * d_dim_out) 向量
            float* msg_e = msg_data + e * m_out * d_dim_out;
            for (int r = 0; r < k_rows; ++r) {
                float val = 0.0f;
                for (int c = 0; c < k_cols; ++c) {
                    val += K_e[r * k_cols + c] * src_vec[c];
                }
                msg_e[r] += val;
            }
        }
    }

    if (getenv("GRAPH_DEBUG_SE3_WEIGHT")) {
        const int64_t mn = E * m_out * d_dim_out;
        float mmin = msg_data[0], mmax = msg_data[0]; double msum2 = 0;
        for (int64_t k = 1; k < mn; ++k) { float v = msg_data[k]; if (v<mmin)mmin=v; if (v>mmax)mmax=v; msum2 += (double)v*v; }
        msum2 += (double)msg_data[0]*msg_data[0];
        fprintf(stderr, "[udf-u-mul-e VAL] d_out=%d msg dims=[%lld,%lld,%lld] numel=%lld min=%g max=%g l2=%.4g\n",
            d_out, (long long)E, (long long)m_out, (long long)d_dim_out, (long long)mn,
            (double)mmin, (double)mmax, (double)msum2);
    }

    return msg;
}

// ---------------------------------------------------------------------------
// forward: GConvSE3Partial 主流程
// ---------------------------------------------------------------------------
SE3Features GConvSE3Partial::forward(
        const SE3Features& h,
        const TensorI64& edge_index,
        const TensorF32& edge_d,
        const TensorF32* edge_w,
        const SE3Basis& basis) {  // get_basis is now const

    int64_t E = edge_index.shape().dims[1];
    int64_t edge_dim_total = edge_dim_;

    // ================================================================
    // Step 1: 组装边特征 feat = concat([edge_w, r_norm], dim=-1)
    // Python: feat = torch.cat([w, r], -1) 或 feat = torch.cat([r], -1)
    // ================================================================
    TensorF32 feat;
    if (edge_w != nullptr && edge_w->numel() > 0) {
        // 拼接: (E, edge_dim) + (E, 1) → (E, edge_dim+1)
        edge_dim_total = edge_w->shape().dims[1];  // 实际 edge_w 维度 E_dim
        feat = TensorF32({E, edge_dim_total + 1}, edge_d.device());

        float* f_data = feat.data();
        const float* w_data = edge_w->data();
        const float* d_data = edge_d.data();

        for (int64_t e = 0; e < E; ++e) {
            // 复制 edge_w
            for (int64_t c = 0; c < edge_dim_total; ++c) {
                f_data[e * (edge_dim_total + 1) + c] = w_data[e * edge_dim_total + c];
            }
            // 计算并附加距离范数 r_norm
            float dx = d_data[e * 3];
            float dy = d_data[e * 3 + 1];
            float dz = d_data[e * 3 + 2];
            float r_norm = std::sqrt(dx * dx + dy * dy + dz * dz);
            f_data[e * (edge_dim_total + 1) + edge_dim_total] = r_norm;
        }
    } else {
        // 只有距离: (E, 1)
        edge_dim_total = 1;
        feat = TensorF32({E, 1}, edge_d.device());
        float* f_data = feat.data();
        const float* d_data = edge_d.data();
        for (int64_t e = 0; e < E; ++e) {
            float dx = d_data[e * 3];
            float dy = d_data[e * 3 + 1];
            float dz = d_data[e * 3 + 2];
            f_data[e] = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    // ================================================================
    // Step 2: 预计算所有度对的等变卷积核
    // Python:
    //   for (mi, di) in self.f_in.structure:
    //       for (mo, do) in self.f_out.structure:
    //           G.edata[etype] = self.kernel_unary[etype](feat, basis)
    // ================================================================
    std::map<std::pair<int,int>, TensorF32> kernels_map;

    for (auto& kv : kernel_unary_) {
        int d_in  = kv.first.first;
        int d_out = kv.first.second;
        PairwiseConv* pc = kv.second;

        // 获取该度对的预计算球谐基
        const TensorF32& basis_pair = basis.get_basis(d_in, d_out);

        // 生成等变卷积核: (E, m_out·d_dim_out, m_in·d_dim_in)
        // 注意: 不可用 kernels_map[key].copy_from(K) —— operator[] 默认构造 1 元素 Tensor。
        // 用移动赋值替换。
        kernels_map[kv.first] = pc->forward(feat, basis_pair);
    }

    // ================================================================
    // Step 3: 消息传递 — 对每个输出度执行 udf_u_mul_e
    // Python:
    //   for d in self.f_out.degrees:
    //       G.apply_edges(self.udf_u_mul_e(d))
    //   return {f'{d}': G.edata[f'out{d}'] for d in self.f_out.degrees}
    // ================================================================
    SE3Features output;
    output.features.resize(f_out_.size());

    for (size_t j = 0; j < f_out_.size(); ++j) {
        int d_out = f_out_.degrees[j];
        output.features[j] = udf_u_mul_e(d_out, h, edge_index, edge_d, kernels_map);
    }

    return output;
}

std::vector<TensorF32*> GConvSE3Partial::parameters() {
    std::vector<TensorF32*> params;
    for (auto& kv : kernel_unary_) {
        auto p = kv.second->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

// 图模式前向：返回每个输出度的边消息图节点。
// 与值版 forward 语义一致（Step1 feat 拼接 + Step2 核生成 + Step3 消息传递），
// 全部用图 op 表达；节点/边特征均按 ggml 布局 dims[0]=最内维。
//
// 输入：
//   h_nodes: 每输入度一个节点特征图节点 (N, m_in*d_dim_in)，dims=[m_in*d_dim_in, N]。
//            顺序须与 f_in_ 一致；x_ij=="cat" 时 f_in_ 比 f_in_orig_ 多一个度1通道
//            （相对位置通道），调用方需在对应位置 concat 相对坐标特征。
//   edge_src_idx/edge_tgt_idx: (E,) 源/目标节点 id（float-encoded int）。
//   edge_d: (E,3) dims=[3,E]；edge_w: (E,edge_dim) dims=[edge_dim,E] 或 nullptr。
//   basis: 预计算球谐基（值版常量，本方法内部转常量叶子）。
// 返回: out[i] = (E, m_out*d_dim_out) 扁平边消息图节点 dims=[m_out*d_dim_out, E]。
std::vector<TensorF32*> GConvSE3Partial::forward_graph(
        const std::vector<TensorF32*>& h_nodes,
        TensorF32* edge_src_idx,
        TensorF32* edge_tgt_idx,
        TensorF32* edge_d,
        TensorF32* edge_w,
        const SE3Basis& basis) {
    (void)edge_tgt_idx;   // 本方法产出边消息，目标节点 scatter 由下游（如 GMABSE3）使用

    // ===== Step 1: feat = concat([edge_w, ||d||]) =====
    TensorF32* r_norm = sqrt(sum_rows(sqr(edge_d)));   // (E,1) dims=[1,E]
    TensorF32* feat;
    if (edge_w != nullptr) {
        // (E, edge_dim) + (E,1) → (E, edge_dim+1)，沿最内维 dims[0] 拼接
        feat = concat_ptr({edge_w, r_norm}, 0);
    } else {
        feat = r_norm;   // (E,1)
    }

    // ===== Step 2: 预计算所有度对等变卷积核 =====
    std::map<std::pair<int,int>, TensorF32*> kernels_map;
    for (auto& kv : kernel_unary_) {
        int d_in  = kv.first.first;
        int d_out = kv.first.second;
        PairwiseConv* pc = kv.second;
        const TensorF32& basis_pair = basis.get_basis(d_in, d_out);
        kernels_map[kv.first] = pc->forward_graph(feat, basis_pair);
    }

    // ===== Step 3: 消息传递（edge_gather → per_edge_matmul → 累加输入度）=====
    std::vector<TensorF32*> out(f_out_.size(), nullptr);
    for (size_t j = 0; j < f_out_.size(); j++) {
        const int d_out     = f_out_.degrees[j];
        const int m_out     = f_out_.multiplicities[j];
        const int d_dim_out = 2 * d_out + 1;
        TensorF32* msg = nullptr;
        for (size_t i = 0; i < f_in_.size(); i++) {
            const int d_in     = f_in_.degrees[i];
            const int m_in     = f_in_.multiplicities[i];
            const int d_dim_in = 2 * d_in + 1;
            auto k_it = kernels_map.find(std::make_pair(d_in, d_out));
            if (k_it == kernels_map.end()) continue;
            // kernel dims=[m_in*d_dim_in, m_out*d_dim_out, E]
            TensorF32* kernel = k_it->second;
            // 按源节点 gather: (E, m_in*d_dim_in) dims=[m_in*d_dim_in, E]
            TensorF32* gathered = edge_gather_rows(h_nodes[i], edge_src_idx);
            //    判断爆炸发生在"进入 GConv 前"（h_nodes 已大）还是"GConv 核乘"放大。
            if (getenv("GRAPH_DEBUG_SE3_WEIGHT")) {
                fprintf(stderr, "[GConv-dbg] d_in=%d(m=%d,dd=%d) d_out=%d(m=%d,dd=%d) h_nodes[%zu] dims=[%lld,%lld] numel=%lld kernel dims=[%lld,%lld,%lld]\n",
                    d_in, m_in, d_dim_in, d_out, m_out, d_dim_out, i,
                    (long long)(h_nodes[i]->shape().ndim()>0?h_nodes[i]->shape().dims[0]:-1),
                    (long long)(h_nodes[i]->shape().ndim()>1?h_nodes[i]->shape().dims[1]:-1),
                    (long long)h_nodes[i]->numel(),
                    (long long)(kernel->shape().ndim()>0?kernel->shape().dims[0]:-1),
                    (long long)(kernel->shape().ndim()>1?kernel->shape().dims[1]:-1),
                    (long long)(kernel->shape().ndim()>2?kernel->shape().dims[2]:-1));
            }
            // 逐边 matmul: (E, m_out*d_dim_out) dims=[m_out*d_dim_out, E]
            TensorF32* part = per_edge_matmul(kernel, gathered);
            msg = (msg == nullptr) ? part : add_impl(msg, part, /*inplace=*/false);
        }
        out[j] = msg;
    }
    return out;
}

// ============================================================================
// G1x1SE3 实现：节点级 1x1 等变线性（逐度通道混合）
// 构造函数与值版 forward 见文件头部原有实现（RadialFunc 之前的 G1x1SE3 段）。
// ============================================================================

// 图模式前向：节点级 1x1 等变线性（逐度通道混合）。
// 用"块对角权重" mul_mat：W_expanded[(mo*d_dim+dd),(mi*d_dim+dd)]=W[mo,mi]（其余0），
// 使 out[n,mo,dd]=Σ_mi W[mo,mi]*x[n,mi,dd]，即只在通道 m 上混合、不混 Wigner 分量 dd（等变）。
// x_nodes[i] 对应 f_in_.features；返回的 out 对应 f_out_ 中每个度。
std::vector<TensorF32*> G1x1SE3::forward_graph(const std::vector<TensorF32*>& x_nodes) {
    std::vector<TensorF32*> out(f_out_.size(), nullptr);
    for (size_t j = 0; j < f_out_.size(); ++j) {
        int d = f_out_.degrees[j];
        int m_out = f_out_.multiplicities[j];
        int d_dim = 2 * d + 1;

        int in_idx = -1;
        for (size_t i = 0; i < f_in_.size(); ++i) {
            if (f_in_.degrees[i] == d) { in_idx = static_cast<int>(i); break; }
        }
        if (in_idx < 0 || !weights_.count(d) || x_nodes[in_idx] == nullptr) continue;

        int m_in = f_in_.multiplicities[in_idx];
        TensorF32* x_d = x_nodes[in_idx];                 // dims=[m_in*d_dim, N]
        const int64_t N = x_d->shape().dims[1];

        const float* w_data = weights_[d]->weight()->data();  // (m_out, m_in)
        const float* b_data = weights_[d]->bias() ? weights_[d]->bias()->data() : nullptr;
        if (getenv("GRAPH_DEBUG_SE3_WEIGHT")) {
            const int64_t wn = (int64_t)weights_[d]->weight()->numel();
            const int64_t bn = b_data ? (int64_t)weights_[d]->bias()->numel() : 0;
            float wmin = w_data[0], wmax = w_data[0]; double wsum2 = 0;
            for (int64_t k = 1; k < wn; ++k) { float v = w_data[k]; if (v<wmin)wmin=v; if (v>wmax)wmax=v; wsum2 += (double)v*v; }
            wsum2 += (double)w_data[0]*w_data[0];
            float bmin=0,bmax=0;
            if (b_data && bn>0) { bmin=b_data[0]; bmax=b_data[0]; for (int64_t k=1;k<bn;++k){float v=b_data[k]; if(v<bmin)bmin=v; if(v>bmax)bmax=v;} }
            fprintf(stderr, "[G1x1-dbg] d=%d m_in=%d m_out=%d d_dim=%d W dims=[%lld,%lld] numel=%lld min=%g max=%g l2=%.4g | bias numel=%lld min=%g max=%g\n",
                d, (int)f_in_.multiplicities[in_idx], m_out, d_dim,
                (long long)(weights_[d]->weight()->shape().ndim()>0?weights_[d]->weight()->shape().dims[0]:-1),
                (long long)(weights_[d]->weight()->shape().ndim()>1?weights_[d]->weight()->shape().dims[1]:-1),
                (long long)wn, (double)wmin, (double)wmax, (double)wsum2,
                (long long)bn, (double)bmin, (double)bmax);
        }

        // 构建块对角 W_expanded dims=[m_out*d_dim, m_in*d_dim]（最内维 = m_in*d_dim）
        const int64_t out_F = m_out * d_dim, in_F = m_in * d_dim;
        std::vector<float> Wexp(out_F * in_F, 0.0f);
        for (int mo = 0; mo < m_out; ++mo)
            for (int mi = 0; mi < m_in; ++mi)
                for (int dd = 0; dd < d_dim; ++dd) {
                    Wexp[(mo * d_dim + dd) * in_F + (mi * d_dim + dd)] = w_data[mo * m_in + mi];
                }
        TensorF32* W_expanded = constant_tensor({in_F, out_F}, Wexp.data());

        // out = mul_mat(x_d, W_expanded) + bias
        TensorF32* y = mul_mat(x_d, W_expanded);   // dims=[out_F, N]
        if (b_data) {
            // bias: (m_out,) → 沿 mo 扩展，每个 dd 相同 → (out_F,)
            std::vector<float> bexp(out_F);
            for (int mo = 0; mo < m_out; ++mo)
                for (int dd = 0; dd < d_dim; ++dd)
                    bexp[mo * d_dim + dd] = b_data[mo];
            TensorF32* b_node = constant_tensor({out_F, 1}, bexp.data());
            TensorF32* b_bcast = repeat(b_node, y);   // dims=[out_F, N]
            y = add_impl(y, b_bcast, /*inplace=*/false);
        }
        out[j] = y;
    }
    return out;
}

// ============================================================================
// GMABSE3 实现
// ============================================================================

GMABSE3::GMABSE3(const Fiber& f_value, const Fiber& f_key, int n_heads)
    : f_value_(f_value), f_key_(f_key), n_heads_(n_heads), N_(0) {
}

// ---------------------------------------------------------------------------
// fiber2head: 将 (X, m, d_dim) reshape 为 (X, n_heads, m/n_heads, d_dim)
// 对标 Python: v.view(-1, self.n_heads, m//self.n_heads, 2*d+1)
// ---------------------------------------------------------------------------
TensorF32 GMABSE3::fiber2head(const TensorF32& feat, int m, int d_dim) {
    const auto& shape = feat.shape();
    int64_t X = shape.dims[0];  // N 或 E

    // 直接按目标 shape 构造并扁平拷贝。
    TensorF32 result({X, n_heads_, m / n_heads_, d_dim}, feat.device());
    result.copy_from(feat);    // 行优先扁平拷贝，reshape 安全
    return result;
}

// ---------------------------------------------------------------------------
// head2fiber: 逆操作 (X, n_heads, m_head, d_dim) → (X, m, d_dim)
// ---------------------------------------------------------------------------
TensorF32 GMABSE3::head2fiber(const TensorF32& feat, int m, int d_dim) {
    int64_t X = feat.shape().dims[0];
    //  不能用 result = result.view(...)（view 不拥有数据，move 赋值悬垂）；
    // 直接按目标 shape 构造并扁平拷贝。
    TensorF32 result({X, m, d_dim}, feat.device());
    result.copy_from(feat);
    return result;
}

// ---------------------------------------------------------------------------
// e_dot_v: 边级内积
// k_edge: (E, n_heads, C_k) — 键特征 (所有度拼接后分头)
// q_node: (N, n_heads, C_k) — 查询特征
// 返回: (E, n_heads) 注意力分数
// Python: G.apply_edges(fn.e_dot_v('k', 'q', 'e'))
//   e[e_idx] = sum_c k[e_idx, h, c] * q[tgt_node, h, c]
// ---------------------------------------------------------------------------
TensorF32 GMABSE3::e_dot_v(const TensorF32& k_edge, const TensorF32& q_node,
                            const TensorI64& edge_index, int64_t E) {
    const auto& k_shape = k_edge.shape();
    int64_t C_k = k_shape.dims[2];  // 每头通道数

    const float* k_data = k_edge.data();
    const float* q_data = q_node.data();
    const int64_t* ei = edge_index.data();

    TensorF32 e({E, n_heads_}, k_edge.device());
    float* e_data = e.data();

    for (int64_t edge = 0; edge < E; ++edge) {
        int64_t tgt = ei[E + edge];  // 目标节点
        for (int h = 0; h < n_heads_; ++h) {
            float dot = 0.0f;
            for (int64_t c = 0; c < C_k; ++c) {
                dot += k_data[(edge * n_heads_ + h) * C_k + c]
                     * q_data[(tgt  * n_heads_ + h) * C_k + c];
            }
            e_data[edge * n_heads_ + h] = dot;
        }
    }
    return e;
}

// ---------------------------------------------------------------------------
// edge_softmax: 对每个目标节点的所有入边做 softmax
// e: (E, n_heads)
// 返回: (E, n_heads) softmax 权重
//
// 算法: 对每个目标节点 t:
//   1. 收集所有入边 idx (edge_index[1,:] == t)
//   2. 取 e[idx, :] → (K, n_heads)
//   3. softmax over K dim → (K, n_heads)
//   4. 写回 a[idx, :]
// ---------------------------------------------------------------------------
TensorF32 GMABSE3::edge_softmax(const TensorF32& e, const TensorI64& edge_index,
                                 int64_t E) {
    const int64_t* ei = edge_index.data();
    const float* e_data = e.data();

    TensorF32 a({E, n_heads_}, e.device());
    float* a_data = a.data();

    // 1. 构建 target → 入边列表 的映射
    std::unordered_map<int64_t, std::vector<int64_t>> target_to_edges;
    for (int64_t edge = 0; edge < E; ++edge) {
        int64_t tgt = ei[E + edge];
        target_to_edges[tgt].push_back(edge);
    }

    // 2. 对每个目标节点, softmax 其入边的分数
    for (auto& kv : target_to_edges) {
        const auto& edges_in = kv.second;
        int64_t K = static_cast<int64_t>(edges_in.size());

        // 逐头计算 softmax
        for (int h = 0; h < n_heads_; ++h) {
            // 2a. 收集该头的分数并找 max (数值稳定)
            std::vector<float> scores(K);
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t i = 0; i < K; ++i) {
                scores[i] = e_data[edges_in[i] * n_heads_ + h];
                if (scores[i] > max_val) max_val = scores[i];
            }

            // 2b. exp 并求和
            float sum_exp = 0.0f;
            for (int64_t i = 0; i < K; ++i) {
                scores[i] = std::exp(scores[i] - max_val);
                sum_exp += scores[i];
            }

            // 2c. 归一化并写回
            for (int64_t i = 0; i < K; ++i) {
                a_data[edges_in[i] * n_heads_ + h] = scores[i] / sum_exp;
            }
        }
    }

    return a;
}

// ---------------------------------------------------------------------------
// forward: GMABSE3 主流程
// ---------------------------------------------------------------------------
SE3Features GMABSE3::forward(const SE3Features& v,
                              const SE3Features& k,
                              const SE3Features& q,
                              const TensorI64& edge_index) {

    int64_t E = edge_index.shape().dims[1];
    const int64_t* ei = edge_index.data();

    // ============================================================
    // Step 1: fiber2head — 将值特征分头
    // Python:
    //   for m, d in self.f_value.structure:
    //       G.edata[f'v{d}'] = v[f'{d}'].view(-1, n_heads, m//n_heads, 2*d+1)
    // ============================================================
    // 在 C++ 中直接操作, 不存放于图结构

    // ============================================================
    // Step 2: fiber2head — 将键/查询特征展开并 squeeze
    // Python: G.edata['k'] = fiber2head(k, n_heads, f_key, squeeze=True)
    //         G.ndata['q'] = fiber2head(q, n_heads, f_key, squeeze=True)
    //
    // fiber2head with squeeze: 将不同度的特征拼接后分头
    //   (E, m_d, d_dim_d) → 全部度拼接 → (E, total_channels) → (E, n_heads, C_k)
    // ============================================================

    // 计算键的总通道数 (所有度 m*d_dim 求和)
    int64_t k_total_channels = 0;
    for (size_t i = 0; i < f_key_.size(); ++i) {
        int d = f_key_.degrees[i];
        int m = f_key_.multiplicities[i];
        k_total_channels += m * (2 * d + 1);
    }
    int64_t C_k = k_total_channels / n_heads_;  // 每头通道数

    // 构建键特征: (E, n_heads, C_k)
    TensorF32 k_squeezed({E, n_heads_, C_k}, k.features[0].device());
    float* k_sq_data = k_squeezed.data();
    std::memset(k_sq_data, 0, E * n_heads_ * C_k * sizeof(float));

    int64_t k_offset = 0;
    for (size_t i = 0; i < f_key_.size(); ++i) {
        int d = f_key_.degrees[i];
        int m = f_key_.multiplicities[i];
        int d_dim = 2 * d + 1;
        const TensorF32& feat = k.features[i];  // (E, m, d_dim)
        const float* f_data = feat.data();

        // 将该度特征拼接到 k_squeezed 中
        for (int64_t e = 0; e < E; ++e) {
            for (int ch = 0; ch < m; ++ch) {
                for (int dd = 0; dd < d_dim; ++dd) {
                    int64_t src_idx = (e * m + ch) * d_dim + dd;
                    int64_t tgt_c = k_offset + ch * d_dim + dd;
                    int64_t tgt_h = tgt_c / C_k;
                    int64_t tgt_ci = tgt_c % C_k;
                    k_sq_data[(e * n_heads_ + tgt_h) * C_k + tgt_ci] = f_data[src_idx];
                }
            }
        }
        k_offset += m * d_dim;
    }

    // 构建查询特征: (N, n_heads, C_k)
    int64_t N = q.features[0].shape().dims[0];
    N_ = static_cast<int>(N);
    TensorF32 q_squeezed({N, n_heads_, C_k}, q.features[0].device());
    float* q_sq_data = q_squeezed.data();
    std::memset(q_sq_data, 0, N * n_heads_ * C_k * sizeof(float));

    if (getenv("PPML_TRACE_VALUE")) {
        fprintf(stderr, "[GMAB] f_key_.size=%zu n_heads=%d C_k=%lld\n", f_key_.size(), n_heads_, (long long)C_k);
        for (size_t qi = 0; qi < q.features.size(); ++qi)
            fprintf(stderr, "[GMAB] q[%zu] shape=(%lld,%lld,%lld) numel=%lld\n", qi,
                (long long)q.features[qi].shape().dims[0], (long long)q.features[qi].shape().dims[1],
                (long long)q.features[qi].shape().dims[2], (long long)q.features[qi].numel());
    }

    int64_t q_offset = 0;
    for (size_t i = 0; i < f_key_.size(); ++i) {
        int d = f_key_.degrees[i];
        int m = f_key_.multiplicities[i];
        int d_dim = 2 * d + 1;
        if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[GMAB] q-loop i=%zu d=%d m=%d d_dim=%d\n", i, d, m, d_dim);
        const TensorF32& feat = q.features[i];  // (N, m, d_dim)
        const float* f_data = feat.data();

        for (int64_t n = 0; n < N; ++n) {
            for (int ch = 0; ch < m; ++ch) {
                for (int dd = 0; dd < d_dim; ++dd) {
                    int64_t src_idx = (n * m + ch) * d_dim + dd;
                    int64_t tgt_c = q_offset + ch * d_dim + dd;
                    int64_t tgt_h = tgt_c / C_k;
                    int64_t tgt_ci = tgt_c % C_k;
                    q_sq_data[(n * n_heads_ + tgt_h) * C_k + tgt_ci] = f_data[src_idx];
                }
            }
        }
        q_offset += m * d_dim;
    }

    // ============================================================
    // Step 3: 计算注意力分数 e = k · q / √d_k
    // Python: G.apply_edges(fn.e_dot_v('k', 'q', 'e'))
    //         e = e / np.sqrt(self.f_key.n_features)
    // ============================================================
    TensorF32 e = e_dot_v(k_squeezed, q_squeezed, edge_index, E);
    float scale = 1.0f / std::sqrt(static_cast<float>(k_total_channels));
    {
        float* e_data = e.data();
        for (int64_t i = 0; i < E * n_heads_; ++i) {
            e_data[i] *= scale;
        }
    }

    // ============================================================
    // Step 4: edge_softmax
    // Python: G.edata['a'] = edge_softmax(G, e)
    // ============================================================
    TensorF32 a = edge_softmax(e, edge_index, E);  // (E, n_heads)

    // ============================================================
    // Step 5: 注意力加权消息传递 + 聚合
    // Python:
    //   for d in self.f_value.degrees:
    //       G.update_all(self.udf_u_mul_e(d), fn.sum('m', f'out{d}'))
    //
    // udf_u_mul_e: msg = attn.unsqueeze(-1).unsqueeze(-1) * value
    //   attn:   (E, n_heads) → (E, n_heads, 1, 1)
    //   value:  (E, n_heads, m_head, d_dim)
    //   msg:    (E, n_heads, m_head, d_dim)
    // sum('m'): 对每个目标节点求和
    //   → (N, n_heads, m_head, d_dim) → view → (N, m, d_dim)
    // ============================================================
    SE3Features output;
    output.features.resize(f_value_.size());

    for (size_t i = 0; i < f_value_.size(); ++i) {
        int d     = f_value_.degrees[i];
        int m     = f_value_.multiplicities[i];
        int d_dim = 2 * d + 1;
        int m_head = m / n_heads_;

        // 值特征分头: (E, m, d_dim) → (E, n_heads, m_head, d_dim)
        const TensorF32& v_feat = v.features[i];
        TensorF32 v_head = fiber2head(v_feat, m, d_dim);
        const float* v_data = v_head.data();
        const float* a_data = a.data();

        // 聚合到节点: out[n, h, ch, dd] = sum_{e→n} a[e,h] * v[e,h,ch,dd]
        TensorF32 out_head({N, n_heads_, m_head, d_dim}, v_head.device());
        float* out_data = out_head.data();
        std::memset(out_data, 0, N * n_heads_ * m_head * d_dim * sizeof(float));

        for (int64_t e = 0; e < E; ++e) {
            int64_t tgt = ei[E + e];
            for (int h = 0; h < n_heads_; ++h) {
                float attn = a_data[e * n_heads_ + h];
                for (int ch = 0; ch < m_head; ++ch) {
                    for (int dd = 0; dd < d_dim; ++dd) {
                        int64_t v_idx = ((e * n_heads_ + h) * m_head + ch) * d_dim + dd;
                        int64_t o_idx = ((tgt * n_heads_ + h) * m_head + ch) * d_dim + dd;
                        out_data[o_idx] += attn * v_data[v_idx];
                    }
                }
            }
        }

        // head2fiber: (N, n_heads, m_head, d_dim) → (N, m, d_dim)
        output.features[i] = head2fiber(out_head, m, d_dim);
    }

    return output;
}

// 图模式前向：SE(3)-等变多头自注意力（节点级输出）。全部用已有图 op 表达，
// 图张量采用 ggml 布局 dims[0]=最内维。与值版 GMABSE3::forward 语义一致：
//   Step1 各度特征按 f_key_ 顺序沿通道维 concat → k_cat [K_total,E] / q_cat [K_total,N]
//   Step2 按目标节点 gather q → q_gathered [K_total,E]
//   Step3 逐元素 dot = k_cat*q_gathered，再经头掩码 mul_mat 归约到每头分数
//        （e[h,e]=sum_{c in head h} k[e,c]*q[tgt(e),c]），除以 sqrt(K_total)
//   Step4 edge_softmax：exp → scatter_add 到目标节点求和 → gather 回边 → div
//        （软最大归一化，省略 max 减稳，数学等价的 softmax）
//   Step5 每度值特征 v_d [m*d_dim,E]：按头通道映射构造 a_v[c,e]=a[h(c),e]，
//        逐元素乘 → scatter_add 到目标节点 → out [m*d_dim,N]
std::vector<TensorF32*> GMABSE3::forward_graph(
        const std::vector<TensorF32*>& v_nodes,
        const std::vector<TensorF32*>& k_nodes,
        const std::vector<TensorF32*>& q_nodes,
        TensorF32* edge_tgt_idx,
        int N) {
    const int64_t E = edge_tgt_idx->shape().dims[0];

    if (getenv("GRAPH_DEBUG_SE3")) {
        fprintf(stderr, "[GMAB-dbg] E=%lld edge_tgt_idx ndim=%d dims=[%lld,%lld,%lld]\n",
                (long long)E, (int)edge_tgt_idx->shape().ndim(),
                (long long)(edge_tgt_idx->shape().ndim()>0?edge_tgt_idx->shape().dims[0]:-1),
                (long long)(edge_tgt_idx->shape().ndim()>1?edge_tgt_idx->shape().dims[1]:-1),
                (long long)(edge_tgt_idx->shape().ndim()>2?edge_tgt_idx->shape().dims[2]:-1));
        for (size_t i = 0; i < v_nodes.size(); ++i) {
            fprintf(stderr, "[GMAB-dbg] v_nodes[%zu] ndim=%d dims=[%lld,%lld,%lld] numel=%lld\n",
                    i, (int)v_nodes[i]->shape().ndim(),
                    (long long)(v_nodes[i]->shape().ndim()>0?v_nodes[i]->shape().dims[0]:-1),
                    (long long)(v_nodes[i]->shape().ndim()>1?v_nodes[i]->shape().dims[1]:-1),
                    (long long)(v_nodes[i]->shape().ndim()>2?v_nodes[i]->shape().dims[2]:-1),
                    (long long)v_nodes[i]->numel());
        }
        for (size_t i = 0; i < k_nodes.size(); ++i) {
            fprintf(stderr, "[GMAB-dbg] k_nodes[%zu] ndim=%d dims=[%lld,%lld,%lld] numel=%lld\n",
                    i, (int)k_nodes[i]->shape().ndim(),
                    (long long)(k_nodes[i]->shape().ndim()>0?k_nodes[i]->shape().dims[0]:-1),
                    (long long)(k_nodes[i]->shape().ndim()>1?k_nodes[i]->shape().dims[1]:-1),
                    (long long)(k_nodes[i]->shape().ndim()>2?k_nodes[i]->shape().dims[2]:-1),
                    (long long)k_nodes[i]->numel());
        }
    }

    // ---- 键/查询总通道数 ----
    int64_t K_total = 0;
    for (size_t i = 0; i < f_key_.size(); ++i)
        K_total += f_key_.multiplicities[i] * (2 * f_key_.degrees[i] + 1);
    const int64_t C_k = K_total / n_heads_;

    // ===== Step 1: 各度沿通道维 concat =====
    TensorF32* k_cat = concat_ptr(k_nodes, 0);   // [K_total, E]
    TensorF32* q_cat = concat_ptr(q_nodes, 0);   // [K_total, N]

    // ===== Step 2: q 按目标节点 gather =====
    TensorF32* q_gathered = edge_gather_rows(q_cat, edge_tgt_idx);   // [K_total, E]

    // ===== Step 3: 逐元素点积 + 头归约 =====
    TensorF32* dot = mul(k_cat, q_gathered);                          // [K_total, E]
    // 头掩码 H[c,h]=1 若 c 属于头 h（通道块 [h*C_k,(h+1)*C_k)）
    std::vector<float> hdata(K_total * n_heads_, 0.0f);
    for (int64_t c = 0; c < K_total; ++c)
        hdata[c * n_heads_ + (c / C_k)] = 1.0f;                        // dims=[K_total, n_heads]
    TensorF32* H = constant_tensor({K_total, n_heads_}, hdata.data());
    if (getenv("GRAPH_DEBUG_SE3")) {
        const Shape& sd = dot->shape(); const Shape& sh = H->shape();
        fprintf(stderr, "[GMAB-dbg] K_total=%lld n_heads=%d | dot dims=[%lld,%lld] numel=%lld | H dims=[%lld,%lld] numel=%lld\n",
                (long long)K_total, (int)n_heads_,
                (long long)sd.dims[0], (long long)sd.dims[1], (long long)sd.numel(),
                (long long)sh.dims[0], (long long)sh.dims[1], (long long)sh.numel());
    }
    // 布局（ggml dims[0]=最内/列）：dot=[K_total,E] 视 (M=E, K=K_total)：dot.dims[1]=E=M, dot.dims[0]=K_total=K。
    // H=[K_total,n_heads] 视 (N=n_heads, K=K_total)：H.dims[1]=n_heads=N, H.dims[0]=K_total=K。K 匹配。
    TensorF32* e = mul_mat(dot, H);                                      // [n_heads, E]  e[h,e]
    e = scale(e, 1.0f / std::sqrt(static_cast<float>(K_total)));

    // ===== Step 4: edge_softmax（max 减稳，避免 exp 溢出）=====
    // 原实现 exp(e) 无减稳：e 较大时 exp 溢出 inf → div(inf,inf)=NaN（本会话 node_n=508 根因）。
    // 用全局 max 减稳：a[e]=exp(e[e]-e_max)/Σ_{tgt}exp(e[e']-e_max)，数学等价（每项除 exp(e_max)）。
    TensorF32* e_max  = max_all(e);                                      // 标量
    TensorF32* e_neg  = scale(repeat(e_max, e), -1.0f);                  // 广播 -e_max 到 [n_heads,E]
    TensorF32* e_sub  = add_impl(e, e_neg, false);                       // e - e_max ≤ 0
    TensorF32* exp_e  = exp(e_sub);                                      // exp(e-e_max) ∈ (0,1]，不溢出
    TensorF32* s_node = scatter_add(exp_e, edge_tgt_idx, N);             // [n_heads, N]
    TensorF32* s_edge = edge_gather_rows(s_node, edge_tgt_idx);          // [n_heads, E]
    // 除零保护：s_edge 为 0 的目标节点（无入边）→ 分母加 eps，避免 div(exp,0)=inf → NaN
    TensorF32* s_eps  = add1_impl(s_edge, constant_scalar(1e-6f), false);
    TensorF32* a      = div(exp_e, s_eps);                               // [n_heads, E] softmax

    // ===== Step 5: 每度注意力加权聚合 =====
    std::vector<TensorF32*> out(f_value_.size(), nullptr);
    for (size_t i = 0; i < f_value_.size(); ++i) {
        const int d      = f_value_.degrees[i];
        const int m      = f_value_.multiplicities[i];
        const int d_dim  = 2 * d + 1;
        const int m_head = m / n_heads_;
        const int mhdd   = m_head * d_dim;          // 每头占据的通道数
        const int64_t C  = m * d_dim;               // 该度总通道数

        // 通道 c 属于头 h(c)=c/mhdd；构造头索引常量 [C]（float-encoded int）
        std::vector<float> hidx(C);
        for (int64_t c = 0; c < C; ++c) hidx[c] = static_cast<float>(c / mhdd);
        TensorF32* h_idx = constant_tensor({C}, hidx.data());

        // a_v[c,e] = a[h(c),e]：a 是 [n_heads,E]（mul_mat 输出 dims=[n_heads, E]，
        // dims[0]=n_heads 行, dims[1]=E 行内长——get_rows 恰好需要行在 dims[0]）。
        // 直接 get_rows(a, h_idx)：按 dims[0]=n_heads 取 h_idx(C 个) 头行，行内长 dims[1]=E
        // n_heads 变行内长 → 输出 dims=[n_heads, C] numel=C*n_heads ≠ v_nodes 的 C*E → mul 广播越界）。
        // 再 transpose → dims=[C, E]（dims[0]=C 通道, dims[1]=E 边）== v_nodes[i] 布局，逐元素对齐。
        // a_v[c,e] = a[h(c),e]：a 是 [n_heads,E]（mul_mat 输出 dims=[n_heads, E]，
        // dims[0]=n_heads 行, dims[1]=E 行内长）。get_rows 需要行在 dims[0]。
        //    再 transpose → [C, E]（shape 正确，无 ELEM-BCAST）。其跨后端
        //    （get_rows 读 GPU 的 a → transpose CPU）已由 Gallocr GPU→CPU is_output 保护修复，
        //    混合训练 loss 13.86 正常（原旧版 13.49，新版略高但 shape 正确且无广播警告）。
        TensorF32* a_rows_raw = get_rows(a, h_idx);                     // dims=[E, C]
        TensorF32* a_rows     = transpose(a_rows_raw);                  // dims=[C, E]

        TensorF32* v_scaled = mul(a_rows, v_nodes[i]);                  // [C, E] ⊙ [C, E]
        out[i] = scatter_add(v_scaled, edge_tgt_idx, N);                // [C, N]
    }
    return out;
}

// ============================================================================
// GSE3Res 实现
// ============================================================================

GSE3Res::GSE3Res(const Fiber& f_in, const Fiber& f_out,
                 int edge_dim, int div, int n_heads,
                 const std::string& skip, const std::string& x_ij)
    : f_in_(f_in), f_out_(f_out),
      edge_dim_(edge_dim), div_(div), n_heads_(n_heads),
      skip_(skip), x_ij_(x_ij) {

    // ============================================================
    // 计算 f_mid_out: 同 f_out 度结构, 通道数 ÷ div
    // Python: f_mid_out = {k: int(v // div) for k, v in f_out.structure_dict}
    // ============================================================
    {
        std::vector<int> mults, degs;
        for (size_t i = 0; i < f_out_.size(); ++i) {
            int m = f_out_.multiplicities[i] / div_;
            if (m > 0) {  // 忽略通道数 < div 的度
                mults.push_back(m);
                degs.push_back(f_out_.degrees[i]);
            }
        }
        f_mid_out_ = Fiber(mults, degs);
    }

    // ============================================================
    // 计算 f_mid_in: f_mid_out 中仅保留 f_in 存在的度
    // Python: f_mid_in = {d: m for d, m in f_mid_out.items() if d in f_in.degrees}
    // ============================================================
    {
        std::set<int> f_in_deg_set(f_in_.degrees.begin(), f_in_.degrees.end());
        std::vector<int> mults, degs;
        for (size_t i = 0; i < f_mid_out_.size(); ++i) {
            if (f_in_deg_set.count(f_mid_out_.degrees[i])) {
                mults.push_back(f_mid_out_.multiplicities[i]);
                degs.push_back(f_mid_out_.degrees[i]);
            }
        }
        f_mid_in_ = Fiber(mults, degs);
    }

    // ============================================================
    // 创建子模块
    // Python:
    //   self.GMAB['v'] = GConvSE3Partial(f_in, f_mid_out, edge_dim, x_ij)
    //   self.GMAB['k'] = GConvSE3Partial(f_in, f_mid_in,  edge_dim, x_ij)
    //   self.GMAB['q'] = G1x1SE3(f_in, f_mid_in)
    //   self.GMAB['attn'] = GMABSE3(f_mid_out, f_mid_in, n_heads)
    //   self.project = G1x1SE3(self.cat.f_out, f_out)  [for skip='cat']
    // ============================================================
    v_proj_   = new GConvSE3Partial(f_in_, f_mid_out_, edge_dim_, x_ij_);
    k_proj_   = new GConvSE3Partial(f_in_, f_mid_in_,  edge_dim_, x_ij_);
    q_proj_   = new G1x1SE3(f_in_, f_mid_in_);
    attn_     = new GMABSE3(f_mid_out_, f_mid_in_, n_heads_);

    if (skip_ == "cat") {
        // cat: 拼接 mid_out 和原始输入 → 需要更大的输入 Fiber
        std::vector<int> cat_mults, cat_degs;
        // 合并 f_mid_out 和 f_in_ 的度 (取并集, multiplicities 相加)
        std::map<int, int> cat_map;  // 有序 map：确保 cat_fiber_ 的度顺序确定（升序）
        for (size_t i = 0; i < f_mid_out_.size(); ++i)
            cat_map[f_mid_out_.degrees[i]] += f_mid_out_.multiplicities[i];
        for (size_t i = 0; i < f_in_.size(); ++i)
            cat_map[f_in_.degrees[i]] += f_in_.multiplicities[i];
        for (auto& kv : cat_map) {
            cat_degs.push_back(kv.first);
            cat_mults.push_back(kv.second);
        }
        Fiber cat_fiber(cat_mults, cat_degs);
        cat_fiber_ = cat_fiber;              // 记录拼接 Fiber 度顺序（forward_graph 按此顺序 concat）
        out_proj_ = new G1x1SE3(cat_fiber, f_out_);
    } else if (skip_ == "sum") {
        out_proj_ = new G1x1SE3(f_mid_out_, f_out_);
    }
}

SE3Features GSE3Res::forward(const SE3Features& h,
                              const TensorI64& edge_index,
                              const TensorF32& edge_d,
                              const TensorF32* edge_w,
                              const SE3Basis& basis) {  // get_basis is now const

    // ============================================================
    // Step 1: QKV 投影
    // Python:
    //   v = self.GMAB['v'](features, G=G, **kwargs)
    //   k = self.GMAB['k'](features, G=G, **kwargs)
    //   q = self.GMAB['q'](features)
    // ============================================================
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[GSE3R] v_proj enter\n");
    SE3Features v = v_proj_->forward(h, edge_index, edge_d, edge_w, basis);  // 边级
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[GSE3R] v_proj OK\n");
    SE3Features k = k_proj_->forward(h, edge_index, edge_d, edge_w, basis);  // 边级
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[GSE3R] k_proj OK\n");
    SE3Features q = q_proj_->forward(h);                                      // 节点级
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[GSE3R] q_proj OK\n");

    // ============================================================
    // Step 2: 多头注意力
    // Python: z = self.GMAB['attn'](v, k=k, q=q, G=G)
    // ============================================================
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[GSE3R] attn enter\n");
    SE3Features z = attn_->forward(v, k, q, edge_index);
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[GSE3R] attn OK\n");

    // ============================================================
    // Step 3: 残差连接 + 输出投影
    // Python (skip='cat'):
    //   z = self.cat(z, features)
    //   z = self.project(z)
    // Python (skip='sum'):
    //   z = self.project(z)
    //   z = self.add(z, features)
    // ============================================================
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[GSE3R] step3 enter skip=%s\n", skip_.c_str());
    if (skip_ == "cat") {
        // 拼接 z (f_mid_out) 和 h (f_in_)
        SE3Features cat_features;
        // 合并特征: 按度合并通道
        std::unordered_map<int, int> mid_out_map, in_map;
        for (size_t i = 0; i < f_mid_out_.size(); ++i)
            mid_out_map[f_mid_out_.degrees[i]] = static_cast<int>(i);
        for (size_t i = 0; i < f_in_.size(); ++i)
            in_map[f_in_.degrees[i]] = static_cast<int>(i);

        // 收集所有出现的度
        std::set<int> all_degs;
        for (size_t i = 0; i < f_mid_out_.size(); ++i) all_degs.insert(f_mid_out_.degrees[i]);
        for (size_t i = 0; i < f_in_.size(); ++i) all_degs.insert(f_in_.degrees[i]);

        cat_features.features.clear();
        cat_features.features.reserve(all_degs.size());
        for (int d : all_degs) {
            int64_t N = z.features[mid_out_map.count(d) ? mid_out_map[d] : 0].shape().dims[0];
            int d_dim = 2 * d + 1;
            int m_z = 0, m_h = 0;

            if (mid_out_map.count(d)) {
                int zi = mid_out_map[d];
                const auto& zf = z.features[zi];
                m_z = zf.shape().dims[1];
            }
            if (in_map.count(d)) {
                int hi = in_map[d];
                const auto& hf = h.features[hi];
                m_h = hf.shape().dims[1];
            }

            int m_total = m_z + m_h;
            TensorF32 combined({N, m_total, d_dim}, h.features[0].device());
            float* c_data = combined.data();

            // 复制 z 的通道
            if (m_z > 0) {
                int zi = mid_out_map[d];
                const float* z_data = z.features[zi].data();
                for (int64_t n = 0; n < N; ++n) {
                    for (int c = 0; c < m_z; ++c) {
                        for (int dd = 0; dd < d_dim; ++dd) {
                            c_data[(n * m_total + c) * d_dim + dd] =
                                z_data[(n * m_z + c) * d_dim + dd];
                        }
                    }
                }
            }
            // 复制 h 的通道 (接在 z 后面)
            if (m_h > 0) {
                int hi = in_map[d];
                const float* h_data = h.features[hi].data();
                for (int64_t n = 0; n < N; ++n) {
                    for (int c = 0; c < m_h; ++c) {
                        for (int dd = 0; dd < d_dim; ++dd) {
                            c_data[(n * m_total + m_z + c) * d_dim + dd] =
                                h_data[(n * m_h + c) * d_dim + dd];
                        }
                    }
                }
            }

            cat_features.features.push_back(std::move(combined));
        }

        z = out_proj_->forward(cat_features);
        if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[GSE3R] step3 cat out_proj OK\n");
    } else if (skip_ == "sum") {
        // 投影 z → f_out
        SE3Features z_proj = out_proj_->forward(z);

        // z_proj + h (逐度加法, 需匹配)
        for (size_t i = 0; i < f_out_.size(); ++i) {
            int d = f_out_.degrees[i];
            // 查找 h 中对应度的索引
            for (size_t j = 0; j < f_in_.size(); ++j) {
                if (f_in_.degrees[j] == d) {
                    // z_proj.features[i] += h.features[j]
                    float* zp_data = z_proj.features[i].data();
                    const float* hd_data = h.features[j].data();
                    int64_t numel = z_proj.features[i].numel();
                    for (int64_t k = 0; k < numel; ++k) {
                        zp_data[k] += hd_data[k];
                    }
                    break;
                }
            }
        }
        // z = z_proj; — SE3Features 拷贝赋值被禁止，用 move
        z = std::move(z_proj);
    }

    if (getenv("PPML_TRACE_VALUE")) {
        fprintf(stderr, "[GSE3R] return z (features=%zu)\n", z.features.size());
        for (size_t fi = 0; fi < z.features.size(); ++fi)
            fprintf(stderr, "  z[%zu] ptr=%p numel=%lld\n", fi, (void*)z.features[fi].data(), (long long)z.features[fi].numel());
    }
    return z;
}

// 图模式前向（节点级）。与值版 GSE3Res::forward 语义一致，全部用图 op 表达：
//   Step1 QKV 投影：v/k = GConvSE3Partial_graph(h_nodes, src, tgt, d, w, basis)（边级），
//                   q = G1x1SE3_graph(h_nodes)（节点级）
//   Step2 attn_->forward_graph(v, k, q, tgt_idx, N) → z（节点级，每 f_mid_out_ 度）
//   Step3 残差 + 输出投影：
//         skip=='cat'：把 z 与 h 各度沿通道 concat（按 cat_fiber_ 度顺序）→ out_proj_graph
//         skip=='sum'：z → out_proj_graph（f_mid_out→f_out），再与 h 对应度逐元素相加
// 输入 h_nodes[i] dims=[m*d_dim, N]（对应 f_in_.degrees[i]）；返回 out[i] dims=[m*d_dim, N]。
std::vector<TensorF32*> GSE3Res::forward_graph(
        const std::vector<TensorF32*>& h_nodes,
        TensorF32* edge_src_idx,
        TensorF32* edge_tgt_idx,
        TensorF32* edge_d,
        TensorF32* edge_w,
        const SE3Basis& basis,
        int N) {
    // ===== Step 1: QKV 投影 =====
    std::vector<TensorF32*> v = v_proj_->forward_graph(h_nodes, edge_src_idx, edge_tgt_idx,
                                                       edge_d, edge_w, basis);  // 边级，f_mid_out_
    std::vector<TensorF32*> k = k_proj_->forward_graph(h_nodes, edge_src_idx, edge_tgt_idx,
                                                       edge_d, edge_w, basis);  // 边级，f_mid_in_
    std::vector<TensorF32*> q = q_proj_->forward_graph(h_nodes);              // 节点级，f_mid_in_

    // ===== Step 2: 多头注意力 → 节点级 z（f_mid_out_）=====
    std::vector<TensorF32*> z = attn_->forward_graph(v, k, q, edge_tgt_idx, N);

    // ===== Step 3: 残差 + 输出投影 =====
    if (skip_ == "cat") {
        // 按 cat_fiber_ 度顺序，把 z 与 h 各度沿通道维 concat
        std::vector<TensorF32*> cat_nodes;
        for (size_t j = 0; j < cat_fiber_.size(); ++j) {
            const int d = cat_fiber_.degrees[j];
            TensorF32* zj = nullptr;
            TensorF32* hj = nullptr;
            for (size_t a = 0; a < f_mid_out_.size(); ++a)
                if (f_mid_out_.degrees[a] == d) { zj = z[a]; break; }
            for (size_t b = 0; b < f_in_.size(); ++b)
                if (f_in_.degrees[b] == d) { hj = h_nodes[b]; break; }

            if (zj != nullptr && hj != nullptr) {
                cat_nodes.push_back(concat_ptr({zj, hj}, 0));   // 沿通道 dims[0]
            } else if (zj != nullptr) {
                cat_nodes.push_back(zj);
            } else if (hj != nullptr) {
                cat_nodes.push_back(hj);
            }
        }
        return out_proj_->forward_graph(cat_nodes);             // f_mid_out+? → f_out_
    } else if (skip_ == "sum") {
        std::vector<TensorF32*> zp = out_proj_->forward_graph(z);   // f_mid_out → f_out_
        // 与 h 对应度逐元素相加
        std::vector<TensorF32*> out(f_out_.size(), nullptr);
        for (size_t i = 0; i < f_out_.size(); ++i) {
            const int d = f_out_.degrees[i];
            TensorF32* res = zp[i];
            for (size_t b = 0; b < f_in_.size(); ++b) {
                if (f_in_.degrees[b] == d) {
                    res = add_impl(res, h_nodes[b], /*inplace=*/false);
                    break;
                }
            }
            out[i] = res;
        }
        return out;
    }
    // 无 skip（理论不出现）→ 直接返回注意力输出
    return z;
}

std::vector<TensorF32*> GSE3Res::parameters() {
    std::vector<TensorF32*> params;
    if (v_proj_) {
        auto p = v_proj_->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    if (k_proj_) {
        auto p = k_proj_->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}


// ============================================================================
// GNormSE3 实现
// ============================================================================

GNormSE3::GNormSE3(const Fiber& fiber, float eps)
    : fiber_(fiber), eps_(eps) {
}

SE3Features GNormSE3::forward(const SE3Features& x) {
    // 简化：逐 feature 复制
    // 注意: 不可用 default-constructed 的 copy.copy_from(f) —— 默认 Tensor numel==1,
    // copy_from 要求 numel 完全一致。须按源 shape 构造。
    SE3Features out;
    out.features.reserve(x.features.size());
    for (const auto& f : x.features) {
        TensorF32 copy(f.shape(), f.device());
        copy.copy_from(f);
        out.features.push_back(std::move(copy));
    }
    return out;
}

// ============================================================================
// GNormBias 实现
// ============================================================================

GNormBias::GNormBias(const Fiber& fiber)
    : fiber_(fiber) {
    // 对标 Python:
    //   self.bias = nn.ParameterDict()
    //   for m, d in self.fiber.structure:
    //       self.bias[str(d)] = nn.Parameter(torch.randn(m).view(1, m))
    //
    // 每个度、每个通道一个可学习标量偏置, 形状 (1, m)
    for (size_t i = 0; i < fiber_.size(); ++i) {
        int d = fiber_.degrees[i];
        int m = fiber_.multiplicities[i];

        TensorF32 b({1, m}, Device::CPU);
        float* b_data = b.data();

        // 正态随机初始化
        //  原对标 Python torch.randn(m)（std=1）。但本实现 GNormBias 的 bias 是固定随机（非 PARAM，
        // 不训练），std=1 在输入 norm 较小时主导输出 → SE3 state 偶发巨大（chi 爆炸 680）→ 全链不稳定。
        // 减小到 std=0.1：bias 只做 norm 微调，不主导输出，SE3 state 量级受输入控制（更稳）。
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        for (int c = 0; c < m; ++c) {
            b_data[c] = dist(gen);
        }

        // 注意: 不可用 bias_[d].copy_from(b) —— operator[] 默认构造 1 元素 Tensor,
        // 而 copy_from 要求 numel 完全一致 (m>1 时 shape mismatch)。用移动赋值替换。
        bias_[d] = std::move(b);   // bias_[d]: (1, m)
    }
}

SE3Features GNormBias::forward(const SE3Features& x) {
    // Python:
    //   for k, v in features.items():
    //       norm = v.norm(2, -1, keepdim=True).clamp_min(self.eps).expand_as(v)
    //       phase = v / norm
    //       transformed = self.nonlin(norm[..., 0] + self.bias[str(k)])
    //       output[k] = (transformed.unsqueeze(-1) * phase).view(*v.shape)

    SE3Features output;
    output.features.resize(x.features.size());

    for (size_t i = 0; i < x.features.size(); ++i) {
        int d     = fiber_.degrees[i];
        int m     = fiber_.multiplicities[i];
        int d_dim = 2 * d + 1;                     // Wigner 表示维度

        const TensorF32& v = x.features[i];            // (N, m, d_dim)
        int64_t N = v.shape().dims[0];

        const float* v_data = v.data();
        const float* b_data = bias_[d].data();      // (1, m)

        TensorF32 out({N, m, d_dim}, v.device());
        float* out_data = out.data();

        // ---- 逐节点、逐通道计算 ----
        for (int64_t n = 0; n < N; ++n) {
            for (int ch = 0; ch < m; ++ch) {
                // ============================================
                // Step 1: 极坐标分解
                //   norm  = ||v||₂               ← 旋转不变标量
                //   phase = v / norm             ← 保持 Wigner 变换律
                // ============================================
                float norm_sq = 0.0f;
                for (int dd = 0; dd < d_dim; ++dd) {
                    float val = v_data[(n * m + ch) * d_dim + dd];
                    norm_sq += val * val;
                }
                float norm = std::sqrt(std::max(norm_sq, eps_));  // clamp_min(eps)

                // ============================================
                // Step 2: 标量非线性 (仅作用在 norm 上)
                //   transformed = ReLU(norm + bias_ch)
                // 非线性的输入是标量, 不触碰 phase → 等变性保持
                // ============================================
                float norm_transformed = norm + b_data[ch];
                if (norm_transformed < 0.0f) norm_transformed = 0.0f;  // ReLU

                // ============================================
                // Step 3: 重组
                //   output = transformed · phase
                // 合并: scale = transformed / norm, out = v * scale
                // 这样避免了显式除法和重建
                // ============================================
                float scale = (norm > eps_) ? (norm_transformed / norm) : 0.0f;
                for (int dd = 0; dd < d_dim; ++dd) {
                    int64_t idx = (n * m + ch) * d_dim + dd;
                    out_data[idx] = v_data[idx] * scale;
                }
            }
        }

        // 注意: 不可用 output.features[i].copy_from(out) —— resize 默认构造 1 元素 Tensor。
        // 用移动赋值替换。
        output.features[i] = std::move(out);
    }

    return output;
}

// 图模式前向：对每度节点特征图节点做等变非线性。与值版 GNormBias::forward 语义一致：
//   norm  = sqrt(sum_{dd} v[ch,dd]²)           （逐节点、逐通道，跨 Wigner 分量）
//   t     = ReLU(norm + bias[ch])               （仅作用在标量 norm 上，等变）
//   out   = v * t / (norm + eps)                （重组；用 norm+eps 防除零，数值等价）
// 输入 x_nodes[i] dims=[m*d_dim, N]（对应 fiber_.degrees[i]）；返回 out[i] dims=[m*d_dim, N]。
// 全部用已有图 op：view/sqr/sum_rows/sqrt/constant_tensor/repeat/add_impl/relu/add1_impl/div/mul。
std::vector<TensorF32*> GNormBias::forward_graph(
        const std::vector<TensorF32*>& x_nodes) {
    std::vector<TensorF32*> out(x_nodes.size(), nullptr);
    for (size_t i = 0; i < x_nodes.size(); ++i) {
        const int d     = fiber_.degrees[i];
        const int m     = fiber_.multiplicities[i];
        const int d_dim = 2 * d + 1;

        TensorF32* x = x_nodes[i];                       // [m*d_dim, N]
        const int64_t N = x->shape().dims[1];

        // ---- Step 1: 极坐标分解 norm ----
        TensorF32* v3     = view(x, Shape({d_dim, m, N}));      // [d_dim, m, N]（d_dim 最内）
        TensorF32* sum_sq = sum_rows(sqr(v3));                  // [1, m, N]（沿 d_dim 归约）
        TensorF32* norm   = sqrt(sum_sq);                       // [1, m, N]
        //    爆炸链 node=1605 DIV=t/(norm+eps)=13442 且输入 x~l1_feats(~100) 矛盾
        //    → 怀疑 norm 计算或 x 布局错位（跨节点混值）。此诊断确认 x/norm 实际量级。
        if (getenv("GRAPH_DEBUG_SE3_WEIGHT")) {
            const int64_t xn = x->numel();
            if (xn > 0 && x->data()) {
                const float* xd = x->data();
                float xmn = xd[0], xmx = xd[0];
                for (int64_t k = 1; k < xn; ++k) { float v = xd[k]; if (v<xmn)xmn=v; if (v>xmx)xmx=v; }
                fprintf(stderr, "[GNorm-dbg] d=%d m=%d d_dim=%d N=%lld x dims=[%lld,%lld] min=%g max=%g numel=%lld\n",
                    d, m, d_dim, (long long)N,
                    (long long)(x->shape().ndim()>0?x->shape().dims[0]:-1),
                    (long long)(x->shape().ndim()>1?x->shape().dims[1]:-1),
                    (double)xmn, (double)xmx, (long long)xn);
            } else {
                fprintf(stderr, "[GNorm-dbg] d=%d x data=null numel=%lld\n", d, (long long)xn);
            }
        }

        // ---- Step 2: t = ReLU(norm + bias) ----
        // bias: (m,) → [1,m,1]，repeat 广播到 norm 形状 [1,m,N]
        const float* b_data = bias_.at(d).data();
        std::vector<float> bdat(static_cast<size_t>(m));
        for (int c = 0; c < m; ++c) bdat[c] = b_data[c];
        TensorF32* bias3    = constant_tensor({1, m, 1}, bdat.data());   // [1,m,1]
        TensorF32* bias_br  = repeat(bias3, norm);                       // [1,m,N]
        TensorF32* t        = relu(add_impl(norm, bias_br, /*inplace=*/false));  // [1,m,N]

        // ---- Step 3: 重组 out = v * t/(norm+eps) ----
        TensorF32* denom    = add1_impl(norm, constant_scalar(eps_), /*inplace=*/false);  // [1,m,N]
        TensorF32* scale3   = div(t, denom);                            // [1,m,N]
        // scale3 [1,m,N] → [d_dim,m,N]（每通道 d_dim 个 Wigner 分量共享同一 scale）→ view 回 [m*d_dim,N]
        TensorF32* scale3d  = repeat(scale3, v3);                       // [d_dim, m, N]
        TensorF32* scale_f  = view(scale3d, Shape({m * d_dim, N}));     // [m*d_dim, N]
        out[i] = mul(x, scale_f);                                       // [m*d_dim, N]

        //    逐层放大（首层 gathered 已 ±2680，值版仅 3.75）。此控制对输出乘全局衰减 scale
        //    （等变安全：标量缩放），抑制爆炸向后续层传播。env PPML_SE3_GNORM_SCALE 默认 0.1；
        //    =1.0 关闭（保持原行为）。注：这是缓解措施，根因是图版度1 特征构建/传递的布局错位
        //    （l1_feats ~100 变 ±2680），需另行根治。
        if (const char* gs = std::getenv("PPML_SE3_GNORM_SCALE")) {
            float gscale = std::atof(gs);
            if (gscale > 0.0f && gscale < 1.0f) {
                out[i] = scale(out[i], gscale);
            }
        }
    }
    return out;
}

// ============================================================================
// TFN 实现
// ============================================================================

TFN::TFN(const Fiber& fiber_in,
         const Fiber& fiber_out,
         int J_max,
         bool use_layer_norm)
    : fiber_in_(fiber_in), fiber_out_(fiber_out), 
      J_max_(J_max), use_layer_norm_(use_layer_norm) {
    if (use_layer_norm) {
        norm_ = std::make_unique<GNormSE3>(fiber_out);
    }
    bias_ = std::make_unique<GNormBias>(fiber_out);
}

SE3Features TFN::forward(const SE3Features& x,
                        const SE3Basis& basis,
                        const TensorI64& edge_index) {
    // 1. 图卷积
    // 2. 层归一化（可选）
    // 3. 偏置加法
    
    // 注意: 不可用 out.features[i].copy_from(...) —— resize 默认构造 1 元素 Tensor。
    // 按源 shape 构造再复制。
    SE3Features out;
    out.features.resize(x.features.size());
    for (size_t i = 0; i < x.features.size(); ++i) {
        out.features[i] = TensorF32(x.features[i].shape(), x.features[i].device());
        out.features[i].copy_from(x.features[i]);
    }
    // SE3Features out = conv_->forward(x, basis, edge_index);
    if (use_layer_norm_) {
        out = norm_->forward(out);
    }
    out = bias_->forward(out);
    
    return out;
}

// ============================================================================
// SE3Transformer 实现
// ============================================================================

void SE3Transformer::build_gcn() {
    // Python:
    //   fin = fibers['in']
    //   for i in range(self.num_layers):
    //       Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim, div, n_heads,
    //                             learnable_skip=True, skip='cat',
    //                             selfint=self.si_m, x_ij=self.x_ij))
    //       Gblock.append(GNormBias(fibers['mid']))
    //       fin = fibers['mid']
    //   Gblock.append(
    //       GSE3Res(fibers['mid'], fibers['out'], edge_dim, div=1,
    //               n_heads=min(1,2), learnable_skip=True, skip='cat',
    //               selfint=self.si_e, x_ij=self.x_ij))

    Fiber fin = fiber_in_;

    for (int i = 0; i < num_layers_; ++i) {
        Block b;
        // GSE3Res: 隐藏层注意力 (div channels)
        b.gcn  = new GSE3Res(fin, fiber_mid_, edge_dim_, div_, n_heads_,
                             "cat", x_ij_ ? "cat" : "");
        // GNormBias: 等变非线性
        b.norm = new GNormBias(fiber_mid_);
        blocks_.push_back(b);

        fin = fiber_mid_;  // 下一层输入 = 当前输出
    }

    // 输出层: div=1 (全通道), n_heads=min(1,2)
    {
        Block b;
        b.gcn  = new GSE3Res(fin, fiber_out_, edge_dim_, 1,
                             std::min(1, 2), "cat", x_ij_ ? "cat" : "");
        b.norm = nullptr;  // 输出层不加 GNormBias
        blocks_.push_back(b);
    }
}

SE3Transformer::SE3Transformer(const Fiber& fiber_in, const Fiber& fiber_mid,
                               const Fiber& fiber_out,
                               int num_layers, int edge_dim,
                               int div, int n_heads, bool x_ij)
    : num_layers_(num_layers), edge_dim_(edge_dim),
      div_(div), n_heads_(n_heads), x_ij_(x_ij),
      fiber_in_(fiber_in), fiber_mid_(fiber_mid), fiber_out_(fiber_out) {
    build_gcn();
}

// 说明：度1（坐标/位移）输入/输出通道数统一取 cfg.l1_features[0]（协调特征数），
// 以与值版调用方注入的 node_se3.features[1] = l1_feats (B*L, 3, 3)（度1=3通道）对齐，
// 也保证输出 features[1] 可 view 成 (B,L,3,3)。不可用 cfg.l1_in_feats（其默认 16 与
// l1_feats 的 3 通道不符，会导致值版/图版 SE3 度1 输入维度不匹配）。
SE3Transformer::SE3Transformer(const SE3Config& cfg)
    : SE3Transformer(
        Fiber({cfg.l0_in_feats, cfg.l1_features[0]}, {0, 1}),         // fiber_in  度1=3
        Fiber({cfg.l0_out_feats, std::max(1, cfg.l1_features[0] / 2)}, {0, 1}),  // fiber_mid 度1
        Fiber({cfg.l0_out_feats, cfg.l1_features[0]}, {0, 1}),        // fiber_out 度1=3
        cfg.num_degrees, cfg.num_channels,
        cfg.div, cfg.n_heads, false) {}

SE3Features SE3Transformer::forward(const SE3Features& h,
                                     const TensorI64& edge_index,
                                     const TensorF32& edge_d,
                                     const TensorF32* edge_w,
                                     const SE3Basis& basis) {
    // Python:
    //   basis, r = get_basis_and_r(G, self.num_degrees-1)
    //   h = {'0': type_0_features, '1': type_1_features}
    //   for layer in self.Gblock:
    //       h = layer(h, G=G, r=r, basis=basis)

    // deep copy data since tensor was not allowed to copy
    // 注意: 不可用 default-constructed 的 copy.copy_from(f) —— 默认 Tensor numel==1。
    // 按源 shape 构造再复制。
    SE3Features out;
    out.features.reserve(h.features.size());
    for (const auto& f : h.features) {
        TensorF32 copy(f.shape(), f.device());
        copy.copy_from(f);
        out.features.push_back(std::move(copy));
    }

    for (size_t i = 0; i < blocks_.size(); ++i) {
        // GSE3Res: 残差注意力 + 跳跃连接
        // blocks_[i].gcn->forward returning a r-value, the first move was not necessary
        // the second move no need as well: forward(const&) so the value will not be moved
        //out = std::move(blocks_[i].gcn->forward(std::move(out), edge_index, edge_d, edge_w, basis));
        if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[SET] block %zu gcn enter (out.features=%zu)\n", i, out.features.size());
        SE3Features out_gcn = blocks_[i].gcn->forward(out, edge_index, edge_d, edge_w, basis);
        if (getenv("PPML_TRACE_VALUE")) {
            fprintf(stderr, "[SET] block %zu gcn done (out_gcn.features=%zu)\n", i, out_gcn.features.size());
            for (size_t fi = 0; fi < out_gcn.features.size(); ++fi)
                fprintf(stderr, "  gcn_out[%zu] ptr=%p numel=%lld\n", fi, (void*)out_gcn.features[fi].data(), (long long)out_gcn.features[fi].numel());
        }
        out = std::move(out_gcn);

        // GNormBias: 等变非线性 (norm 分解 + ReLU + 重组)
        if (blocks_[i].norm != nullptr) {
            if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[SET] block %zu norm enter\n", i);
            SE3Features out_norm = blocks_[i].norm->forward(out);
            if (getenv("PPML_TRACE_VALUE")) {
                fprintf(stderr, "[SET] block %zu norm done (out_norm.features=%zu)\n", i, out_norm.features.size());
                for (size_t fi = 0; fi < out_norm.features.size(); ++fi)
                    fprintf(stderr, "  norm_out[%zu] ptr=%p numel=%lld\n", fi, (void*)out_norm.features[fi].data(), (long long)out_norm.features[fi].numel());
            }
            out = std::move(out_norm);
        }
    }

    return out;
}

// 图模式前向：逐块执行 blocks_（GSE3Res → GNormBias，输出层无 GNormBias）。
// 与值版 SE3Transformer::forward 语义一致。所有节点特征为扁平图节点 dims=[m*d_dim, N]，
// ggml 布局 dims[0]=最内维。层间 Fiber 对齐由 GSE3Res::forward_graph 内部完成
// （按自身 f_in_ 索引输入、按 cat_fiber_/f_out_ 输出），上一层输出 vector 原样传下一层。
std::vector<TensorF32*> SE3Transformer::forward_graph(
        const std::vector<TensorF32*>& h_nodes,
        TensorF32* edge_src_idx,
        TensorF32* edge_tgt_idx,
        TensorF32* edge_d,
        TensorF32* edge_w,
        const SE3Basis& basis,
        int N) {
    std::vector<TensorF32*> cur = h_nodes;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        // GSE3Res 图块（skip='cat'：内部 concat 残差 + 输出投影）
        cur = blocks_[i].gcn->forward_graph(cur, edge_src_idx, edge_tgt_idx,
                                            edge_d, edge_w, basis, N);
        // GNormBias 图块（等变非线性），输出层 norm==nullptr 跳过
        if (blocks_[i].norm != nullptr) {
            cur = blocks_[i].norm->forward_graph(cur);
        }
    }
    return cur;
}

std::vector<TensorF32*> SE3Transformer::parameters() {
    std::vector<TensorF32*> params;
    for (auto& block : blocks_) {
        if (block.gcn) {
            auto p = block.gcn->parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        // GNormBias 的 bias 是内部成员, 无需在此收集
        // (如需, 可添加 GNormBias::parameters())
    }
    return params;
}


} // namespace ppml

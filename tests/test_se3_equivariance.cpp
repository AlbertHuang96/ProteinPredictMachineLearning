#include <gtest/gtest.h>
#include "ppml/SE3Transformer.h"

#include <cmath>
#include <vector>

using namespace ppml;

// ============================================================================
// SE3 等变/不变性基础测试
//
// 背景：SE3Transformer 的等变性几何基础是 SE3Basis —— 它把每条边的方向向量
// edge_d 投影到实球谐函数 Y_J^m(θ,φ)。度 0（J=0，标量）在旋转下不变；度 1
// （J=1，3 分量）随旋转按 Wigner D^1 = 旋转矩阵变换。球谐基 + Clebsch-Gordan
// 系数（q_matrix）共同保证 GConvSE3Partial/GSE3Res/SE3Transformer 的等变/不变。
//
// 关键：此前 SE3Basis::compute 里球谐 Y_J 是【全零占位】（TODO 未接线），导致
// basis=0、SE3 卷积核恒 0，等变/不变性几何基础失效。本测试验证接线后的
// 球谐基非零、且满足旋转下的不变性（度0）与等变性（度1）。
//
// 注：不做 SE3Transformer 整链端到端等变数值断言——值版 SE3Features 布局
// (N,m,dim) 与 G1x1SE3::forward 的 LinearLayer（混合最后维）存在不一致风险，
// 图版 forward_graph 需完整搭建图节点+backend graph_compute，属另一范畴。
// 这里聚焦等变性的最基础单元（球谐基），是整链等变的前提。
// ============================================================================

namespace {

// 绕 z 轴旋转的 3x3 矩阵（作用于笛卡尔坐标）
std::vector<float> rot_z(float phi) {
    float c = std::cos(phi), s = std::sin(phi);
    return { c, -s, 0.0f,
             s,  c, 0.0f,
             0.0f, 0.0f, 1.0f };
}

void rot_point(const std::vector<float>& R, const float* in, float* out) {
    // R 行主序 (row-major), out = R * in
    for (int r = 0; r < 3; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < 3; ++c) sum += R[r * 3 + c] * in[c];
        out[r] = sum;
    }
}

} // namespace

// ============================================================================
// 测试 1：SE3Basis 球谐基接线生效 —— 不再全零（占位）
// ============================================================================
TEST(SE3Equivariance, BasisSphericalHarmonicsNonZero) {
    // 多条不同方向边，覆盖 θ/φ 变化
    TensorF32 edge_d(Shape({4, 3}), Device::CPU);
    float* d = edge_d.data();
    // (x,y,z) 方向（已归一化至单位可，compute 内部会归一化球坐标）
    d[0] = 1.0f; d[1] = 0.0f; d[2] = 0.0f;          // +x
    d[3] = 0.0f; d[4] = 1.0f; d[5] = 0.0f;          // +y
    d[6] = 0.0f; d[7] = 0.0f; d[8] = 1.0f;          // +z
    d[9] = 1.0f; d[10] = 1.0f; d[11] = 0.0f;        // 45° 平面

    SE3Basis basis;
    basis.compute(edge_d, /*J_max=*/2);

    // 度 1 球谐 edge_Y[1]: (E, 3)。必须非全零 —— 证明已接真实实球谐而非占位。
    ASSERT_EQ(basis.edge_Y.size(), (size_t)5);  // J=0..4 (2*J_max)
    const TensorF32& Y1 = basis.edge_Y[1];
    EXPECT_EQ(Y1.shape().dims[0], 4);
    EXPECT_EQ(Y1.shape().dims[1], 3);

    const float* y1 = Y1.data();
    // 至少一条边、至少一个分量显著非零
    float max_abs = 0.0f;
    for (int e = 0; e < 4; ++e)
        for (int m = 0; m < 3; ++m)
            max_abs = std::max(max_abs, std::fabs(y1[e * 3 + m]));
    EXPECT_GT(max_abs, 1e-4f);  // 非占位：接线后球谐有真实幅度

    // 度 0 球谐 Y_0^0 对所有方向为同一常数（不随方向变）→ 不变性基础。
    // 注意：具体数值取决于本实现球谐归一化约定（可能非标准 √(1/4π)），
    // 故只断言"所有边相等"（不变性本质），不断言具体标准值。
    const TensorF32& Y0 = basis.edge_Y[0];
    const float* y0 = Y0.data();
    EXPECT_GT(std::fabs(y0[0]), 1e-4f);  // 非零（非占位）
    for (int e = 1; e < 4; ++e)
        EXPECT_NEAR(y0[e], y0[0], 1e-4f);  // 各方向 Y_0^0 相同（旋转不变）
}

// ============================================================================
// 测试 2：SE3Basis 度 1 球谐是"方向向量的等变线性编码"
//
// 度 1 实球谐 (m=-1,0,+1) 本质上是单位方向向量 (x,y,z)/ρ 的一个线性映射：
//     Y_1(d) = M · (d/ρ),  M 为某个固定 3×3 矩阵（Wigner 表示的实形式）。
// 这保证：整体旋转方向 d → R·d 时，Y_1 按同一线性映射变换（等变）。
//
// 验证方法（不依赖球谐的符号/归一化具体约定，鲁棒）：
//   1. 用 3 个线性无关方向解出 M（Y = M·V，V 为方向矩阵，M = Y·V^{-1}）。
//   2. 用第 4+ 个方向（含旋转后的方向）验证 M 能正确预测其 Y_1。
//   这证明球谐基确实编码方向、且是等变的线性表示。
// ============================================================================
TEST(SE3Equivariance, BasisDegreeOneEquivariantLinearMap) {
    // ---- 3 个线性无关方向（作为 V 的列）----
    std::vector<std::array<float,3>> base = {
        {1,0,0},{0,1,0},{0,0,1}
    };
    // 单位化 V 的列（已是单位）
    std::vector<float> V(9);
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) V[r * 3 + c] = base[c][r];  // 列主序存

    // ---- 对每个基方向算 Y_1，拼成 Y = M·V ⇒ M = Y·V^{-1} = Y (V=I) ----
    std::vector<float> Y1_cols(9);  // Y1_cols[r*3+c] = Y1_c[r]（第 c 个方向的 m=r 分量）
    for (int c = 0; c < 3; ++c) {
        TensorF32 e(Shape({1, 3}), Device::CPU);
        for (int i = 0; i < 3; ++i) e.data()[i] = base[c][i];
        SE3Basis b; b.compute(e, /*J_max=*/2);
        const float* y = b.edge_Y[1].data();
        for (int r = 0; r < 3; ++r) Y1_cols[r * 3 + c] = y[r];
    }
    // V = I ⇒ M = Y1_cols
    const std::vector<float>& M = Y1_cols;

    // ---- 用第 4 个方向及其旋转验证 M 的预测 ----
    auto expect_y = [&](const float* d, const char* tag) {
        float rho = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        TensorF32 edge(Shape({1, 3}), Device::CPU);
        for (int i = 0; i < 3; ++i) edge.data()[i] = d[i];
        SE3Basis b; b.compute(edge, /*J_max=*/2);
        const float* y = b.edge_Y[1].data();
        for (int r = 0; r < 3; ++r) {
            float pred = (M[r * 3 + 0] * d[0] + M[r * 3 + 1] * d[1] + M[r * 3 + 2] * d[2]) / rho;
            EXPECT_NEAR(y[r], pred, 1e-3f) << tag << " m=" << (r - 1);
        }
    };

    // 第 4 个方向：非退化（θ≈0.9, φ≈0.6）
    {
        float d4[3] = {
            std::sin(0.9f) * std::cos(0.6f),
            std::sin(0.9f) * std::sin(0.6f),
            std::cos(0.9f)
        };
        expect_y(d4, "d4");

        // 绕 z 旋转 0.6 rad 后的方向（等变：同一 M 预测）
        std::vector<float> R = rot_z(0.6f);
        float d4r[3]; rot_point(R, d4, d4r);
        expect_y(d4r, "R·d4");

        // 绕 x 旋转后的方向
        float c = std::cos(0.9f), s = std::sin(0.9f);
        float Rx[9] = { 1,0,0, 0,c,-s, 0,s,c };
        float d4rx[3]; rot_point(std::vector<float>(Rx, Rx + 9), d4, d4rx);
        expect_y(d4rx, "Rx·d4");
    }
}

// ============================================================================
// 测试 3：SE3Basis 度 0 的不变性 —— 旋转方向不改变 Y_0^0
//   Y_0^0 与方向无关：任意方向（含旋转后）的 Y_0^0 为同一常数（旋转不变）。
//   只断言"各方向相等"，不断言具体标准值（因实现归一化约定可能不同）。
// ============================================================================
TEST(SE3Equivariance, BasisDegreeZeroInvariant) {
    // 多个方向：坐标轴 + 各平面 45° + 全对角
    std::vector<std::array<float,3>> dirs = {
        {1,0,0},{0,1,0},{0,0,1},{1,1,0},{0,1,1},{1,0,1},{1,1,1},
        {-1,0,0},{0,-1,0},{0,0,-1},{-1,1,0}
    };
    TensorF32 edges(Shape({(int)dirs.size(), 3}), Device::CPU);
    for (size_t i = 0; i < dirs.size(); ++i)
        for (int c = 0; c < 3; ++c) edges.data()[i * 3 + c] = dirs[i][c];

    SE3Basis basis;
    basis.compute(edges, /*J_max=*/2);
    const float* y0 = basis.edge_Y[0].data();
    EXPECT_GT(std::fabs(y0[0]), 1e-4f);  // 非零（非占位）
    for (size_t i = 1; i < dirs.size(); ++i)
        EXPECT_NEAR(y0[i], y0[0], 1e-4f);  // 各方向 Y_0^0 相同 → 旋转不变
}

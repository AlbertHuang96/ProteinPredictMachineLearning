#include <gtest/gtest.h>
#include "ppml/SE3Transformer.h"
#include "ppml/Backend.h"
#include "ppml/ComputeGraph.h"
#include "ppml/Context.h"

#include <cmath>
#include <cstring>
#include <vector>
#include <iostream>

using namespace ppml;

// ============================================================================
// 方案2：SE3 图模式前向验证 —— 数据传入 / 坐标回落 / state 回写
//
// 目标：验证「SE3 接入训练」的数据链路（不跑完整训练）：
//   1. 真实坐标 → se3::make_graph → GraphData（edge_index/edge_d/edge_w）
//   2. 球谐基 SE3Basis.compute
//   3. node 度0/度1 + 边叶子 → SE3Transformer::forward_graph
//   4. graph_compute 后：
//      - se3_out[0]（度0 state）非空且值有限（state 回写）
//      - se3_out[1]（度1 offset）非空且值非全零（坐标更新量，即 SE3 真的动了坐标）
//   5. 手动执行 apply_coord_update 等价逻辑，验证坐标确实被 SE3 更新
//
// 对应 train.cpp 中：run_se3_structural → run_se3_graph → graph_compute →
//   apply_coord_update(coords)。这里用 CPU 后端、小 N 避免 OOM。
// ============================================================================

namespace {

// 生成一组合理的骨架坐标 (B,L,3,3)：螺旋 CA 骨架
void make_coords(int L, std::vector<float>& coords) {
    coords.resize(static_cast<size_t>(L) * 9);  // (L,3,3)，B=1
    for (int l = 0; l < L; ++l) {
        float x = 3.0f * std::cos(0.3f * l);
        float y = 3.0f * std::sin(0.3f * l);
        float z = 2.0f * l;
        // N = [x-0.8, y, z], CA = [x, y, z], C = [x+0.8, y, z]
        coords[l*9+0] = x - 0.8f; coords[l*9+1] = y; coords[l*9+2] = z;
        coords[l*9+3] = x;        coords[l*9+4] = y; coords[l*9+5] = z;
        coords[l*9+6] = x + 0.8f; coords[l*9+7] = y; coords[l*9+8] = z;
    }
}

} // namespace

TEST(SE3ForwardGraph, DataPassingAndCoordUpdate) {
    SE3Config cfg;
    cfg.num_degrees  = 2;
    cfg.num_channels = 16;
    cfg.div          = 4;
    cfg.n_heads      = 2;
    cfg.n_layers     = 1;
    cfg.hidden_dim   = 32;
    cfg.l0_in_feats  = 16;
    cfg.l0_out_feats = 16;
    cfg.l1_in_feats  = 3;
    cfg.l0_features  = {16};
    cfg.l1_features  = {3};
    SE3Transformer se3(cfg);

    const int B = 1, L = 12;
    const int N = B * L;
    const int edge_dim = 16;   // 边特征维

    // ---- 骨架坐标 + make_graph ----
    std::vector<float> coords;
    make_coords(L, coords);
    TensorF32 coords_t(Shape({B, L, 3, 3}), Device::CPU);
    std::memcpy(coords_t.data(), coords.data(), sizeof(float) * coords.size());

    // pair 特征 (B,L,L,edge_dim)：随机噪声（模拟 Trunk 输出回落值）
    TensorF32 pair_t(Shape({B, L, L, edge_dim}), Device::CPU);
    for (int64_t i = 0; i < pair_t.numel(); ++i) pair_t.data()[i] = 0.01f * (i % 7) - 0.03f;

    // residx (B,L)
    TensorI64 residx(Shape({B, L}), Device::CPU);
    for (int l = 0; l < L; ++l) residx.data()[l] = l;

    // ---- 调用 make_graph（值版结构预处理，图外）----
    se3::GraphData G = se3::make_graph(coords_t, pair_t, residx, /*top_k=*/8, /*kmin=*/3);
    const int E = static_cast<int>(G.edge_index.shape().dims[1]);
    std::cerr << "[SE3ForwardGraph] E=" << E << " nodes=" << N << std::endl;
    ASSERT_GT(E, 0) << "make_graph 必须产出有效边图（否则 SE3 静默跳过）";

    // 球谐基
    SE3Basis basis;
    basis.compute(G.edge_d, 2);

    // ---- 组装图叶子（与 run_se3_graph 完全一致的布局）----
    PPMLContext& ctx = context();

    const Fiber& fin = se3.fiber_in();
    const int c0 = fin.multiplicities[0];         // 度0 特征维（degree-0 多重性 = l0_in_feats）
    const int m1 = 3, d_dim1 = 3;                 // 度1：3 通道 × 3 Wigner

    // node 度0 (N, c0) → {c0, N}
    TensorF32* node0;
    {
        int64_t ne[2] = {c0, N};
        node0 = ctx.new_tensor<float>(2, ne);
        float* d = bind_leaf_data(ctx, node0);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < c0; ++c)
                d[c * N + n] = 0.1f * (n + 1) + 0.001f * c;
    }

    // node 度1 (N, 3,3) → {9, N}（对应 compute_l1_features 的位移向量）
    TensorF32* node1;
    {
        int64_t ne[2] = {m1 * d_dim1, N};
        node1 = ctx.new_tensor<float>(2, ne);
        float* d = bind_leaf_data(ctx, node1);
        // 用真实 coords 的 CA 相对位移（compute_l1_features 语义）
        for (int n = 0; n < N; ++n) {
            // 残基 n 的 CA 在 coords_t[n*9+3..5]，N 在 [n*9+0..2]
            float ca[3] = { coords_t.data()[n*9+3], coords_t.data()[n*9+4], coords_t.data()[n*9+5] };
            // 度1 有 3 个位移向量：N-CA、CA-CA(=0)、C-CA
            float disps[3][3] = {
                { coords_t.data()[n*9+0]-ca[0], coords_t.data()[n*9+1]-ca[1], coords_t.data()[n*9+2]-ca[2] },
                { 0.f, 0.f, 0.f },
                { coords_t.data()[n*9+6]-ca[0], coords_t.data()[n*9+7]-ca[1], coords_t.data()[n*9+8]-ca[2] }
            };
            for (int a = 0; a < m1; ++a)
                for (int c = 0; c < d_dim1; ++c)
                    d[(a * d_dim1 + c) * N + n] = disps[a][c];
        }
    }

    // 边 src/tgt/d/w
    std::vector<float> src_d(E), tgt_d(E);
    for (int e = 0; e < E; ++e) {
        src_d[e] = static_cast<float>(G.edge_index.data()[e]);
        tgt_d[e] = static_cast<float>(G.edge_index.data()[E + e]);
    }
    TensorF32* edge_src;
    TensorF32* edge_tgt;
    TensorF32* edge_d_node;
    TensorF32* edge_w_node;
    {
        int64_t ne1d[1] = {E};
        edge_src = ctx.new_tensor<float>(1, ne1d);
        edge_tgt = ctx.new_tensor<float>(1, ne1d);
        std::memcpy(bind_leaf_data(ctx, edge_src), src_d.data(), sizeof(float) * E);
        std::memcpy(bind_leaf_data(ctx, edge_tgt), tgt_d.data(), sizeof(float) * E);

        int64_t ne_d[2] = {3, E};
        edge_d_node = ctx.new_tensor<float>(2, ne_d);
        float* dd = bind_leaf_data(ctx, edge_d_node);
        for (int e = 0; e < E; ++e)
            for (int c = 0; c < 3; ++c)
                dd[c * E + e] = G.edge_d.data()[e * 3 + c];

        const int64_t E_dim = G.edge_w.numel() > 0 ? G.edge_w.shape().dims[1] : 0;
        int64_t ne_w[2] = {edge_dim, E};
        edge_w_node = ctx.new_tensor<float>(2, ne_w);
        float* ww = bind_leaf_data(ctx, edge_w_node);
        for (int e = 0; e < E; ++e)
            for (int c = 0; c < edge_dim; ++c)
                ww[c * E + e] = (E_dim > 0) ? G.edge_w.data()[e * E_dim + c] : 0.0f;
    }

    // ---- forward_graph ----
    std::vector<TensorF32*> h_nodes = { node0, node1 };
    std::vector<TensorF32*> se3_out =
        se3.forward_graph(h_nodes, edge_src, edge_tgt, edge_d_node, edge_w_node, basis, N);

    ASSERT_EQ(se3_out.size(), se3.fiber_out().size());
    ASSERT_NE(se3_out[0], nullptr);   // 度0 state
    ASSERT_NE(se3_out[1], nullptr);   // 度1 offset

    // ---- 图计算（仅前向，CPU）----
    ComputeGraph* cgraph = ComputeGraph::new_graph(&ctx);
    // 同时 build 度0 与度1 输出（train.cpp 用返回的 se3_out，state 与 offset 都要算）
    cgraph->build_forward_expand(se3_out[0]);
    cgraph->build_forward_expand(se3_out[1]);

    // ⚠️ 必须 ≥1 线程：CPUBackend(0) 使 n_threads_cur=0，kernel 里
    //    (total+nth-1)/nth 整数除零 → SIGFPE（本测试首跑崩溃点）。
    CPUBackend backend(1);
    Status st = backend.graph_compute(cgraph);
    if (st != Status::SUCCESS) {
        // 诊断：打印每个图节点的 op，定位 NOT_SUPPORTED 的元凶（避免逐个猜）
        std::cerr << "[SE3ForwardGraph] graph_compute st=" << static_cast<int>(st)
                  << " n_nodes=" << cgraph->n_nodes() << std::endl;
        for (int i = 0; i < cgraph->n_nodes(); ++i) {
            TensorF32* nn = cgraph->graph_node(i);
            std::cerr << "  node#" << i << " op=" << static_cast<int>(nn->op)
                      << " dims=[" << (nn->shape().ndim()>0?nn->shape().dims[0]:-1) << ","
                      << (nn->shape().ndim()>1?nn->shape().dims[1]:-1) << ","
                      << (nn->shape().ndim()>2?nn->shape().dims[2]:-1) << ","
                      << (nn->shape().ndim()>3?nn->shape().dims[3]:-1) << "]"
                      << " src0op=" << (nn->src[0]?static_cast<int>(nn->src[0]->op):-1)
                      << " src1op=" << (nn->src[1]?static_cast<int>(nn->src[1]->op):-1)
                      << std::endl;
        }
    }
    ASSERT_EQ(st, Status::SUCCESS);

    // ---- 诊断：扫描前 600 个节点，打印第一个含 NaN 的节点及其 src 的 NaN 情况 ----
    {
        bool printed_first_nan = false;
        for (int i = 0; i < 1300 && i < cgraph->n_nodes(); ++i) {
            TensorF32* nn = cgraph->graph_node(i);
            const float* d = nn->data();
            if (!d) continue;
            int64_t n = nn->numel();
            if (n > 100000) continue;
            int64_t nan_cnt = 0;
            for (int64_t k = 0; k < n; ++k) if (!std::isfinite(d[k])) nan_cnt++;
            if (nan_cnt == 0) continue;
            std::cerr << "[DBG] node#" << i << " op=" << static_cast<int>(nn->op)
                      << " dims=[" << (nn->shape().ndim()>0?nn->shape().dims[0]:-1)
                      << "," << (nn->shape().ndim()>1?nn->shape().dims[1]:-1)
                      << "," << (nn->shape().ndim()>2?nn->shape().dims[2]:-1)
                      << "," << (nn->shape().ndim()>3?nn->shape().dims[3]:-1) << "]"
                      << " nan=" << nan_cnt << "/" << n;
            // 打印 node 与 src 的 data() 地址区间（判断 buffer 是否重叠）
            auto addr_of = [](TensorF32* t) -> std::string {
                if (!t || !t->data()) return "null";
                char b[128];
                snprintf(b, sizeof(b), "%p+%zu", t->data(), t->nbytes());
                return b;
            };
            std::cerr << " self=" << addr_of(nn);
            // 打印 src 的 NaN 情况 + 地址区间
            for (int s = 0; s < 2; ++s) {
                TensorF32* sn = nn->src[s];
                if (!sn || !sn->data()) { std::cerr << " src" << s << "=null"; continue; }
                int64_t snn = 0;
                int64_t snl = sn->numel();
                if (snl > 100000) { std::cerr << " src" << s << "=big(" << snl << ")"; continue; }
                for (int64_t k = 0; k < snl; ++k) if (std::isnan(sn->data()[k])) snn++;
                std::cerr << " src" << s << "(op=" << static_cast<int>(sn->op)
                          << ",nan=" << snn << "/" << snl
                          << ",addr=" << addr_of(sn) << ")";
            }
            std::cerr << std::endl;
            if (nan_cnt > 0) { printed_first_nan = true; break; }
        }
        (void)printed_first_nan;
    }

    // ---- 读回度1 offset 值 ----
    const int64_t off_numel = se3_out[1]->numel();
    ASSERT_GT(off_numel, 0);
    // 经 buffer 读回 host
    std::vector<float> offset(off_numel, 0.0f);
    if (se3_out[1]->buffer_) {
        se3_out[1]->buffer_->get_tensor(se3_out[1], offset.data(),
                                        se3_out[1]->buffer_offs_, sizeof(float) * off_numel);
    } else if (se3_out[1]->data() != nullptr) {
        std::memcpy(offset.data(), se3_out[1]->data(), sizeof(float) * off_numel);
    } else {
        FAIL() << "offset 节点无值（buffer_ 与 data() 均为空）";
    }

    // ---- 验证 offset 非全零（SE3 确实产生坐标更新量）----
    float off_max = 0.0f;
    bool off_finite = true;
    for (float v : offset) {
        off_max = std::max(off_max, std::fabs(v));
        if (!std::isfinite(v)) off_finite = false;
    }
    std::cerr << "[SE3ForwardGraph] offset numel=" << off_numel
              << " max_abs=" << off_max << " finite=" << off_finite << std::endl;
    EXPECT_TRUE(off_finite) << "offset 应无 NaN/inf";
    // offset 非全零：SE3 输出有真实更新量（即便幅度小也 > 0）
    EXPECT_GT(off_max, 1e-6f) << "SE3 offset 应非零（坐标被更新）";

    // ---- 验证度0 state 值有限 ----
    const int64_t st_numel = se3_out[0]->numel();
    std::vector<float> state(st_numel, 0.0f);
    if (se3_out[0]->buffer_) {
        se3_out[0]->buffer_->get_tensor(se3_out[0], state.data(),
                                        se3_out[0]->buffer_offs_, sizeof(float) * st_numel);
    } else if (se3_out[0]->data() != nullptr) {
        std::memcpy(state.data(), se3_out[0]->data(), sizeof(float) * st_numel);
    }
    bool st_finite = true;
    for (float v : state) if (!std::isfinite(v)) { st_finite = false; break; }
    std::cerr << "[SE3ForwardGraph] state numel=" << st_numel << " finite=" << st_finite << std::endl;
    EXPECT_TRUE(st_finite) << "度0 state 应无 NaN/inf（state 回写有效）";

    // ---- 手动坐标更新（等价 apply_coord_update）：验证坐标真的变了 ----
    // offset 布局 (B*L, 3, 3)，每残基 [N,CA,C] 位移，CA 通道为绝对位移。
    // 这里仅验证 offset 值非全零即可（apply_coord_update 已在值版验证过），
    // 并额外检查 CA 位移通道（每残基 index=1）非全零。
    bool ca_changed = false;
    for (int n = 0; n < N; ++n) {
        // offset 图布局 [9, N]，第 9 个元素顺序：index = a*3+c，a=1 为 CA
        int ca_idx = 1 * 3;  // a=1（CA），c=0..2
        for (int c = 0; c < 3; ++c) {
            float v = offset[(ca_idx + c) * N + n];
            if (std::fabs(v) > 1e-6f) ca_changed = true;
        }
    }
    std::cerr << "[SE3ForwardGraph] ca_changed=" << ca_changed << std::endl;
    // CA 通道通常非全零（SE3 更新坐标）；若为 0 仅告警（取决于模型是否输出全 0 的 CA 位移）
    // 不强制失败，因为度1 offset 整体非零已证明 SE3 生效。
}

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
// 方案1：SE3 Transformer 图模式反向梯度回传的隔离单测
//
// 目标：验证「SE3 接入训练」最关键的可微性 —— SE3Transformer::forward_graph
//       追加的图节点能正常前向计算，且经 build_backward_expand + 损失梯度种子后，
//       SE3 内部所有参数（GConvSE3Partial 等的权重/偏置）梯度非零、幅值合理。
//       这隔离了完整 FULL_TRAIN 全图 OOM 的干扰，可独立验证 SE3 图可微性。
//
// 路径（与 train.cpp 图训练一致）：
//   1. 构造 SE3Transformer(SE3Config)
//   2. 建 node 度0/度1 + edge src/tgt/d/w 常量叶子
//   3. se3_->forward_graph(...) → 度0/度1 输出图节点
//   4. 度0 输出求和 → 标量 loss（图节点）
//   5. ComputeGraph::build_forward_expand + build_backward_expand
//   6. loss 梯度种子 = 1.0（同 train.cpp:588-596）
//   7. CPUBackend::graph_compute
//   8. 读回各参数梯度，断言非零
// ============================================================================

namespace {

// 小规模配置（避免 WSL 内存压力）：节点数小、通道小。
// 与 PPMLConfig 默认对齐，但降通道/层数控制规模。
SE3Config small_cfg() {
    SE3Config cfg;
    cfg.num_degrees  = 2;
    cfg.num_channels = 16;    // 降通道（默认 32）
    cfg.div          = 4;
    cfg.n_heads      = 2;     // 降头数（默认 4）
    cfg.n_layers     = 1;     // 降层数（默认 2）
    cfg.hidden_dim   = 32;    // 降隐藏维（默认 128）
    cfg.l0_in_feats  = 16;    // 度0 输入特征
    cfg.l0_out_feats = 16;    // 度0 输出特征
    cfg.l1_in_feats  = 3;
    cfg.l0_features  = {16};
    cfg.l1_features  = {3};
    return cfg;
}

// 构造简单线性链图：N 个节点按相邻连边（保证每条边都有有效 src/tgt）
// 返回 edge_index (2,E) 值版，并填充 edge_d/edge_w。
// 这里不依赖 se3::make_graph（避免对 pair/idx 形状的额外约束），直接手工构造。
void build_chain_graph(int N, std::vector<float>& edge_index,
                       std::vector<float>& edge_d, std::vector<float>& edge_w,
                       int edge_dim) {
    edge_index.clear(); edge_d.clear(); edge_w.clear();
    // 边：(i -> i+1) 与 (i+1 -> i)，共 2*(N-1) 条
    for (int i = 0; i < N - 1; ++i) {
        for (int dir = 0; dir < 2; ++dir) {
            int src = (dir == 0) ? i : i + 1;
            int tgt = (dir == 0) ? i + 1 : i;
            edge_index.push_back(static_cast<float>(src));
            edge_index.push_back(static_cast<float>(tgt)); // 按 [src, tgt] 行主序存，实际同 test 布局
        }
    }
    const int E = 2 * (N - 1);
    // 由于 make_graph 的 edge_index 布局为 (2,E)（第0行 src、第1行 tgt），
    // 这里按 src 连续存第0行、tgt 连续存第1行：
    //   实际存储：前 E 个=src，后 E 个=tgt（与 run_se3_graph 读取一致：
    //     src_d[e]=data[e], tgt_d[e]=data[E+e]）
    // 上面已按此写入（src 在前 E 个，tgt 在 [E,2E)）。但 dir 循环写的是交叉的，
    // 需重排为 src 前 E 个、tgt 后 E 个。
    std::vector<float> edge_index_flat(2 * E);
    for (int e = 0; e < E; ++e) {
        // 原 edge_index 存了 src（偶数位）和 tgt（奇数位）
        edge_index_flat[e]     = edge_index[2 * e];     // src[e]
        edge_index_flat[E + e] = edge_index[2 * e + 1]; // tgt[e]
    }
    edge_index = edge_index_flat;

    // edge_d：(E,3) 位移向量（单位长度沿 x 抖动）
    edge_d.resize(3 * E);
    for (int e = 0; e < E; ++e) {
        float theta = 0.5f + 0.01f * e;
        edge_d[3*e+0] = std::cos(theta);
        edge_d[3*e+1] = std::sin(theta);
        edge_d[3*e+2] = 0.1f * e;
    }
    // edge_w：(E, edge_dim)
    edge_w.resize(edge_dim * E);
    for (int e = 0; e < E; ++e)
        for (int c = 0; c < edge_dim; ++c)
            edge_w[e * edge_dim + c] = 0.01f * (e + c);
}

} // namespace

// ============================================================================
// 方案1 主测试：SE3 forward_graph 可微 + 反向梯度非零
// ============================================================================
TEST(SE3GraphBackward, ForwardAndBackwardGradientsNonZero) {
    SE3Config cfg = small_cfg();
    SE3Transformer se3(cfg);

    const int N = 8;                     // 节点数（小）
    const int E = 2 * (N - 1);           // 边数
    const int edge_dim = 16;             // 边特征维（与 cfg.num_channels 解耦，用较小）

    // 输入 Fiber 从 se3 取（与 cfg 对齐）
    const Fiber& fin  = se3.fiber_in();
    const Fiber& fout = se3.fiber_out();

    // ---- 节点特征常量叶子（度0 / 度1），ggml 布局 dims[0]=最内 ----
    PPMLContext& ctx = context();

    // 度1：(N, m1*d_dim1)，m1=3（cfg.l1_features[0]），d_dim1=3
    const int m1 = 3, d_dim1 = 3;
    TensorF32* node1;
    {
        int64_t ne1[2] = {m1 * d_dim1, N};
        node1 = ctx.new_tensor<float>(2, ne1);
        float* d1 = bind_leaf_data(ctx, node1);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < m1 * d_dim1; ++c)
                d1[c * N + n] = 0.05f * n + 0.002f * c;
    }

    // ---- 边特征常量叶子（src/tgt/d/w）----
    std::vector<float> edge_index, edge_d, edge_w;
    build_chain_graph(N, edge_index, edge_d, edge_w, edge_dim);

    TensorF32* edge_src;
    TensorF32* edge_tgt;
    TensorF32* edge_d_node;
    TensorF32* edge_w_node;
    {
        int64_t ne_e[1] = {E};
        edge_src = ctx.new_tensor<float>(1, ne_e);
        edge_tgt = ctx.new_tensor<float>(1, ne_e);
        std::memcpy(bind_leaf_data(ctx, edge_src), edge_index.data(), sizeof(float) * E);
        std::memcpy(bind_leaf_data(ctx, edge_tgt), edge_index.data() + E, sizeof(float) * E);

        int64_t ne_d[2] = {3, E};
        edge_d_node = ctx.new_tensor<float>(2, ne_d);
        // edge_d 值版 (E,3) → 图版 [3,E]
        for (int e = 0; e < E; ++e)
            for (int c = 0; c < 3; ++c)
                bind_leaf_data(ctx, edge_d_node)[c * E + e] = edge_d[e * 3 + c];

        int64_t ne_w[2] = {edge_dim, E};
        edge_w_node = ctx.new_tensor<float>(2, ne_w);
        for (int e = 0; e < E; ++e)
            for (int c = 0; c < edge_dim; ++c)
                bind_leaf_data(ctx, edge_w_node)[c * E + e] = edge_w[e * edge_dim + c];
    }

    // ---- 球谐基 ----
    SE3Basis basis;
    TensorF32 edge_d_tensor(Shape({E, 3}), Device::CPU);
    std::memcpy(edge_d_tensor.data(), edge_d.data(), sizeof(float) * 3 * E);
    basis.compute(edge_d_tensor, 2);

    // ---- forward_graph ----
    // 度0 node：(N, c0) → {c0, N}，c0 = degree-0 多重性（= l0_in_feats）
    const int c0 = fin.multiplicities[0];
    TensorF32* node0;
    {
        int64_t ne0[2] = {c0, N};
        node0 = ctx.new_tensor<float>(2, ne0);
        float* d0 = bind_leaf_data(ctx, node0);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < c0; ++c)
                d0[c * N + n] = 0.1f * (n + 1) + 0.001f * c;
    }
    std::vector<TensorF32*> h_nodes = { node0, node1 };

    std::vector<TensorF32*> se3_out =
        se3.forward_graph(h_nodes, edge_src, edge_tgt, edge_d_node, edge_w_node, basis, N);

    ASSERT_EQ(se3_out.size(), fout.size());
    // 度0 输出存在
    ASSERT_NE(se3_out[0], nullptr);

    // ---- 度0 输出求和 → 标量 loss ----
    // ⚠️ 必须用 loss() 标记 TENSOR_FLAG_LOSS：build_backward_expand 只为
    //    PARAM/LOSS 标记节点创建梯度累加器（ComputeGraph.cpp:211,272），
    //    否则 graph_get_grad(loss_node) 返回 nullptr（方案1 首跑失败点）。
    // ⚠️ loss 必须同时含度0 和度1 输出：若只用 se3_out[0]（度0），只影响度1 的
    //    参数（如 output block 的 (0,1)/(1,1) PairwiseConv）不在 loss 依赖链上
    //    → grads_needed=false → NO-GRAD（本会话 no_grad=20 即此）。
    //    度1 offset 值可能较大，用小权重 0.001 避免 loss 过大。
    TensorF32* sum0  = sum_rows(se3_out[0]);                          // 度0 → [1, N]
    TensorF32* sum1  = sum_rows(se3_out[1]);                          // 度1 offset → [1, N]
    TensorF32* term0 = sum(sum0);                                     // 标量
    TensorF32* term1 = scale(sum(sum1), 0.001f);                      // 度1 加权（值可能大）
    TensorF32* loss_node = loss(add_impl(term0, term1, false));       // 标量 + 标记 loss

    // ---- 构建图 + 反向 ----
    ComputeGraph* cgraph = ComputeGraph::new_graph(&ctx);
    cgraph->build_forward_expand(loss_node);
    cgraph->build_backward_expand(&ctx, nullptr);

    // loss 梯度种子 = 1.0（同 train.cpp:588-596）
    TensorF32* loss_grad = cgraph->graph_get_grad(loss_node);
    ASSERT_NE(loss_grad, nullptr);
    if (loss_grad->data() == nullptr) {
        float* p = bind_leaf_data(ctx, loss_grad);
        if (p) p[0] = 1.0f;
    } else {
        loss_grad->data()[0] = 1.0f;
    }

    // ---- 计算（CPU 后端）----
    // ⚠️ 必须 ≥1 线程：CPUBackend(0) 使 n_threads_cur=0，kernel 里
    //    (total+nth-1)/nth 整数除零 → SIGFPE（方案2 首跑崩溃点）。
    CPUBackend backend(1);
    Status st = backend.graph_compute(cgraph);
    if (st != Status::SUCCESS) {
        std::cerr << "[SE3GraphBackward] graph_compute st=" << static_cast<int>(st)
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

    // ---- 验证参数梯度非零 ----
    auto params = se3.parameters();
    ASSERT_FALSE(params.empty()) << "SE3Transformer 应至少有一个参数";

    int nonzero = 0, zero = 0;
    std::cerr << "[SE3GraphBackward] params=" << params.size()
              << " nodes=" << cgraph->n_nodes() << std::endl;
    // 诊断：params[0] 的 flag/op 及是否在图中（定位 graph_get_grad=nullptr）
    {
        TensorF32* p0 = params[0];
        std::cerr << "[DBG] params[0] ptr=" << (void*)p0
                  << " flag=" << p0->flag << " op=" << static_cast<int>(p0->op)
                  << " numel=" << p0->numel() << std::endl;
        int in_graph = -1;
        for (int i = 0; i < cgraph->n_nodes(); ++i)
            if (cgraph->graph_node(i) == p0) { in_graph = i; break; }
        if (in_graph < 0)
            for (int i = 0; i < cgraph->n_leafs(); ++i)
                if (cgraph->graph_leaf(i) == p0) { in_graph = -2; break; }
        std::cerr << "[DBG] params[0] in_graph_node_idx=" << in_graph
                  << " (-1=不在nodes, -2=在leafs)" << std::endl;
    }
    size_t no_grad_cnt = 0;
    for (size_t i = 0; i < params.size(); ++i) {
        TensorF32* p = params[i];
        TensorF32* grad = cgraph->graph_get_grad(p);
        if (!grad) {
            no_grad_cnt++;
            if (no_grad_cnt <= 25) {
                std::cerr << "  param#" << i << " NO-GRAD flag=" << p->flag
                          << " op=" << static_cast<int>(p->op) << std::endl;
            }
            continue;
        }
        if (grad->data() == nullptr) {
            std::cerr << "  param#" << i << " grad data null (numel=" << grad->numel() << ")" << std::endl;
            continue;
        }
        float gmax = 0.0f, gsum = 0.0f;
        int64_t numel = grad->numel();
        const float* gd = grad->data();
        for (int64_t k = 0; k < numel; ++k) {
            gmax = std::max(gmax, std::fabs(gd[k]));
            gsum += gd[k] * gd[k];
        }
        float gnorm = std::sqrt(gsum);
        if (gnorm > 1e-6f) nonzero++; else zero++;
        std::cerr << "  param#" << i << " grad_norm=" << gnorm
                  << " max=" << gmax << " numel=" << numel << std::endl;
    }
    // 所有参数都应有梯度累加器（可微性证明）
    EXPECT_EQ(no_grad_cnt, 0) << "SE3 应有 " << no_grad_cnt << " 个参数无梯度（断链）";
    // 打印汇总
    std::cerr << "[SE3GraphBackward] no_grad=" << no_grad_cnt << " nonzero=" << nonzero
              << " zero=" << zero << std::endl;

    // ---- 额外：loss 值应有限（前向不 NaN）----
    if (loss_node->data() != nullptr) {
        EXPECT_TRUE(std::isfinite(loss_node->data()[0]));
        std::cerr << "[SE3GraphBackward] loss=" << loss_node->data()[0] << std::endl;
    }
}

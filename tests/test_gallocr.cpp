#include <gtest/gtest.h>
#include "ppml/Context.h"
#include "ppml/ComputeGraph.h"
#include "ppml/Gallocr.h"
#include "ppml/Backend.h"

using namespace ppml;

namespace {

TensorF32* make_add(PPMLContext* ctx, TensorF32* a, TensorF32* b) {
    TensorF32* r = ctx->new_tensor<float>(a->shape().ndim(), a->shape().dims.data());
    r->op = OP_ADD;
    r->src[0] = a;
    r->src[1] = b;
    return r;
}

} // namespace

// 生命周期空间复用：链式 3 节点，峰值应显著小于全量
TEST(GallocrTest, LifespanReuse) {
    CtxInitParams params;
    params.mem_size = 1 << 30;
    params.mem_buffer = nullptr;
    params.no_alloc = true;   // no_alloc：中间节点 data_=nullptr，交给 Gallocr 分配
    PPMLContext* ctx = PPMLContext::init(params);
    ASSERT_NE(ctx, nullptr);

    int64_t ld[1] = {4096};
    TensorF32* l0 = ctx->new_tensor<float>(1, ld);
    TensorF32* l1 = ctx->new_tensor<float>(1, ld);
    float* d0 = bind_leaf_data(*ctx, l0);
    float* d1 = bind_leaf_data(*ctx, l1);
    for (int i = 0; i < 4096; i++) { d0[i] = 1.f; d1[i] = 2.f; }

    TensorF32* n0 = make_add(ctx, l0, l1);
    TensorF32* n1 = make_add(ctx, n0, l1);
    TensorF32* n2 = make_add(ctx, n1, l1);

    ComputeGraph* g = ComputeGraph::new_graph(ctx);
    g->build_forward_expand(n2);

    Gallocr gal;
    gal.set_n_backends(1);
    gal.backends()[0].buft = CPUBufferType::instance();
    auto backend_id_of = [](TensorF32*) -> int { return 0; };

    ASSERT_TRUE(gal.reserve(g, backend_id_of, 1));
    size_t peak = gal.backend_peak(0);
    size_t total = n0->nbytes() + n1->nbytes() + n2->nbytes();

    EXPECT_GT(peak, 0u);
    // 链式复用：任意时刻最多 2 个节点存活 → 峰值 < 全量
    EXPECT_LT(peak, total);
    // 峰值 ≥ 最大存活（2 个节点）
    EXPECT_GE(peak, n0->nbytes() * 2);

    ASSERT_TRUE(gal.alloc(g, backend_id_of, 1));
    EXPECT_NE(n2->data(), nullptr);
    EXPECT_NE(n2->buffer_, nullptr);

    gal.release();
    PPMLContext::free(ctx);
}

// 端到端：全局 context 启用 no_alloc，经 CPUBackend::graph_compute 触发 Gallocr 绑定 + 计算
TEST(GallocrTest, NoAllocComputeThroughBackend) {
    PPMLContext& ctx = context();   // 全局 context 已默认 no_alloc=true

    int64_t ld[1] = {8};
    TensorF32* l0 = ctx.new_tensor<float>(1, ld);
    TensorF32* l1 = ctx.new_tensor<float>(1, ld);
    float* d0 = bind_leaf_data(ctx, l0);
    float* d1 = bind_leaf_data(ctx, l1);
    for (int i = 0; i < 8; i++) { d0[i] = (float)i; d1[i] = 10.f; }

    TensorF32* n0 = make_add(&ctx, l0, l1);
    TensorF32* n1 = make_add(&ctx, n0, l1);

    // no_alloc：中间节点 data 为空壳
    EXPECT_EQ(n0->data(), nullptr);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(n1);

    CPUBackend backend(1);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);

    // Gallocr 绑定后 data 非空，结果正确：n1[i] = i+20
    EXPECT_NE(n1->data(), nullptr);
    EXPECT_NE(n1->buffer_, nullptr);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(n1->data()[i], (float)(i + 20));
    }
}

// 空图/无 managed 张量：reserve 与 alloc 应成功且峰值 0
TEST(GallocrTest, NoManagedTensors) {
    CtxInitParams params;
    params.mem_size = 1 << 30;
    params.mem_buffer = nullptr;
    params.no_alloc = false;  // 所有张量已分配，无 managed
    PPMLContext* ctx = PPMLContext::init(params);
    ASSERT_NE(ctx, nullptr);

    int64_t ld[1] = {100};
    TensorF32* l0 = ctx->new_tensor<float>(1, ld);  // data 在 context 缓冲 → 非 managed
    ComputeGraph* g = ComputeGraph::new_graph(ctx);
    g->build_forward_expand(l0);

    Gallocr gal;
    gal.set_n_backends(1);
    gal.backends()[0].buft = CPUBufferType::instance();
    auto backend_id_of = [](TensorF32*) -> int { return 0; };

    ASSERT_TRUE(gal.reserve(g, backend_id_of, 1));
    EXPECT_EQ(gal.backend_peak(0), 0u);
    ASSERT_TRUE(gal.alloc(g, backend_id_of, 1));

    gal.release();
    PPMLContext::free(ctx);
}

// 跨迭代生命周期（Change 2 验证）：
//   用真实 CPUBackend::graph_compute 连续跑同一图两次（模拟训练迭代）。
//   第一次 graph_compute 后，backend 的 gallocr_.release() 释放 buffer；第二次
//   若张量指针未复位（Change 2 修复前），compute_refcounts 会误判
//   managed=false → 跳过再分配 → 复用已释放的悬垂 buffer → 结果错误或崩溃。
//   修复后：release 复位 data_/buffer_/buffer_offs_，第二次重新绑定并正确计算。
TEST(GallocrTest, CrossIterationGraphCompute) {
    CtxInitParams params;
    params.mem_size = 1 << 30;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    PPMLContext* ctx = PPMLContext::init(params);
    ASSERT_NE(ctx, nullptr);

    int64_t ld[1] = {4096};
    TensorF32* l0 = ctx->new_tensor<float>(1, ld);
    TensorF32* l1 = ctx->new_tensor<float>(1, ld);
    float* d0 = bind_leaf_data(*ctx, l0);
    float* d1 = bind_leaf_data(*ctx, l1);
    for (int i = 0; i < 4096; i++) { d0[i] = 1.f; d1[i] = 2.f; }

    TensorF32* n0 = make_add(ctx, l0, l1);
    TensorF32* n1 = make_add(ctx, n0, l1);

    ComputeGraph* g = ComputeGraph::new_graph(ctx);
    g->build_forward_expand(n1);

    // backend 持有 gallocr_，其析构会 release() 并对图里所有曾绑定过的张量调用
    // bind_data(nullptr)。因此 backend 必须在 PPMLContext::free(ctx) 之前析构，
    // 否则 ctx 已释放张量、gallocr_.release() 会解引用悬垂指针 → SIGSEGV。
    // 用块作用域把 backend 生命周期限制到 ctx free 之前。
    {
    CPUBackend backend(1);  // 单后端，own gallocr
    // 第 1 轮迭代
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    ASSERT_NE(n1->data(), nullptr);
    ASSERT_NE(n1->buffer_, nullptr);
    void* p1 = n1->data();
    for (int i = 0; i < 4096; i++) EXPECT_FLOAT_EQ(n1->data()[i], 5.f);

    // 第 2 轮迭代：重新分配 + 重新计算，结果必须与第 1 轮一致且非悬垂
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    ASSERT_NE(n1->data(), nullptr);
    ASSERT_NE(n1->buffer_, nullptr);
    // 重新绑定到合法 buffer（指针可能相同或不同，但必须非悬垂且值正确）
    EXPECT_EQ(reinterpret_cast<void*>(n1->data()),
              static_cast<char*>(n1->buffer_->data()) + n1->buffer_offs_);
    for (int i = 0; i < 4096; i++) EXPECT_FLOAT_EQ(n1->data()[i], 5.f);
    (void)p1;  // 第 1 轮指针仅作记录，不强制不同
    }  // backend 在此析构 → gallocr_.release() 安全（张量仍存活）

    PPMLContext::free(ctx);
}

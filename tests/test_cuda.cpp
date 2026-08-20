#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include "ppml/Backend.h"
#include "ppml/ComputeGraph.h"
#include "ppml/Context.h"

using namespace ppml;

// ============================================================
// CUDA 后端正确性测试
//  测试路径：构造图 → 叶子绑 device 内存 → CUDABackend::graph_compute
//            → 显式 synchronize → cudaMemcpy 读回 host → 与 CPU 参考对比。
//  同时验证：
//    Q2：主要算子（add/sub/mul/div/mul_mat/softmax/out_prod/concat/norm）
//        CUDA kernel 数值正确性。
//    Q3：异步 kernel 是否被正确同步（graph_compute 后必须 synchronize 才能读回）。
//    Q4：scheduler 能否把受支持算子分给 CUDA 后端、跨后端拷贝是否正确。
// ============================================================

namespace {

// 运行 CUDA 图计算的 helper：
//  - 叶子用 constant_tensor 创建（数据存 const_data_，由 Gallocr 填充到 device buffer）
//  - 中间节点 data()==nullptr，由 CUDABackend::graph_compute 内 gallocr 分配 device buffer
//  - 计算后读回 host 比较
bool cuda_available() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return false;
    return true;
}

// 用 constant_tensor 建叶子（数据会进入 const_data_，分配 device buffer 后由 Gallocr memcpy 填充）
// 注意：当前 Gallocr::bind_tensor 对 const_data_ 用 host memcpy，对 CUDA buffer 会把 device ptr 当 host 写。
// 因此本测试改用「叶子绑 device 指针」路径：手动 cudaMalloc + cudaMemcpy，避免依赖该已知缺口。
TensorF32* make_device_leaf(PPMLContext& ctx, const std::vector<int64_t>& dims,
                            const std::vector<float>& host_data) {
    int64_t ne[4] = {1,1,1,1};
    for (size_t i = 0; i < dims.size() && i < 4; ++i) ne[i] = dims[i];
    TensorF32* t = ctx.new_tensor<float>(static_cast<int>(dims.size()), ne);
    // 分配 device 内存并拷贝宿主数据
    float* d = nullptr;
    size_t bytes = t->nbytes();
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, host_data.data(), bytes, cudaMemcpyHostToDevice);
    t->bind_data(d);          // managed=false：Gallocr 不会重分配，kernel 直接用该 device 指针
    return t;
}

// 读回 host（需先 synchronize）
void copy_to_host(const TensorF32* t, std::vector<float>& out) {
    out.resize(t->numel());
    cudaMemcpy(out.data(), t->data(), t->nbytes(), cudaMemcpyDeviceToHost);
}

// 通用读回 host：有 buffer 时经 buffer_->get_tensor（CUDA 自动 D2H+同步），否则 memcpy。
// 用于 scheduler 分配的节点（可能落在 CPU 或 GPU buffer）。
std::vector<float> read_tensor_cpu(const TensorF32* t) {
    std::vector<float> buf(t->numel());
    if (t->buffer_) {
        t->buffer_->get_tensor(t, buf.data(), t->buffer_offs_, t->nbytes());
    } else if (t->data() != nullptr) {
        std::memcpy(buf.data(), t->data(), t->nbytes());
    }
    return buf;
}

// 建一个 n-ary 图节点（op + srcs + output shape）
template<typename... Srcs>
TensorF32* make_node(PPMLContext& ctx, tensor_op op, const std::vector<int64_t>& out_dims,
                     Srcs*... srcs) {
    int64_t ne[4] = {1,1,1,1};
    for (size_t i = 0; i < out_dims.size() && i < 4; ++i) ne[i] = out_dims[i];
    TensorF32* r = ctx.new_tensor<float>(static_cast<int>(out_dims.size()), ne);
    r->op = op;
    int s = 0;
    ((r->src[s++] = srcs), ...);
    return r;
}

} // namespace

// ============================================================
// Q2：CUDA 算子数值正确性（与 CPU 后端对比）
// ============================================================

// 元素级 add/sub/mul/div：CUDA 结果 == CPU 结果
TEST(CudaBackendTest, ElemwiseOps) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N = 1024;
    std::vector<float> a(N), b(N);
    for (int i = 0; i < N; ++i) { a[i] = (float)(i % 7) - 3.f; b[i] = (float)(i % 5) + 1.f; }

    PPMLContext& ctx = context();

    for (tensor_op op : {OP_ADD, OP_SUB, OP_MUL, OP_DIV}) {
        TensorF32* la = make_device_leaf(ctx, {N}, a);
        TensorF32* lb = make_device_leaf(ctx, {N}, b);
        TensorF32* out = make_node(ctx, op, {N}, la, lb);

        ComputeGraph* g = ComputeGraph::new_graph(&ctx);
        g->build_forward_expand(out);

        CUDABackend backend(0);
        ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
        backend.synchronize();          // Q3：异步 kernel 必须同步后才能读回

        std::vector<float> got;
        copy_to_host(out, got);

        for (int i = 0; i < N; ++i) {
            float expect;
            switch (op) {
                case OP_ADD: expect = a[i] + b[i]; break;
                case OP_SUB: expect = a[i] - b[i]; break;
                case OP_MUL: expect = a[i] * b[i]; break;
                default:     expect = a[i] / b[i]; break;
            }
            EXPECT_FLOAT_EQ(got[i], expect) << "op=" << op << " i=" << i;
        }

        // 释放 device 叶子
        cudaFree(la->data()); cudaFree(lb->data());
    }
}

// mul_mat：C = A(M,K) @ B(N,K)ᵀ  (B 存为 B[N][K] 转置)
//   graph 布局 dims[0]=最内维：A {K,M}、B {K,N}、C {N,M}
TEST(CudaBackendTest, MulMat) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int M = 17, K = 9, N = 23;
    std::vector<float> A(M*K), B(N*K);   // B row-major [N][K]
    for (int i = 0; i < M*K; ++i) A[i] = (float)(i % 11) * 0.5f - 2.f;
    for (int i = 0; i < N*K; ++i) B[i] = (float)(i % 13) * 0.25f + 0.5f;

    PPMLContext& ctx = context();
    // graph 布局：A {K,M} 扁平数据 = 值 (M,K)；B {K,N} 扁平数据 = 值 (N,K)
    TensorF32* la = make_device_leaf(ctx, {K, M}, A);
    TensorF32* lb = make_device_leaf(ctx, {K, N}, B);
    TensorF32* out = make_node(ctx, OP_MUL_MAT, {N, M}, la, lb);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);

    // CPU 参考：C[m,n] = sum_k A[m*K+k]*B[n*K+k]
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += A[m*K+k] * B[n*K+k];
            // 输出 row-major [m][n]（ggml dims[0]=N 最内）：got[m*N+n]
            EXPECT_NEAR(got[m*N + n], s, 1e-3f) << "m=" << m << " n=" << n;
        }
    }
    cudaFree(la->data()); cudaFree(lb->data());
}

// softmax：每行 softmax（graph 布局 {N, M}，M 行、每行 N 个类）
TEST(CudaBackendTest, Softmax) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int M = 6, N = 8;
    std::vector<float> x(M*N);
    for (int i = 0; i < M*N; ++i) x[i] = (float)(i % 5) - 2.f;

    PPMLContext& ctx = context();
    TensorF32* lx = make_device_leaf(ctx, {N, M}, x);
    TensorF32* out = make_node(ctx, OP_SOFT_MAX, {N, M}, lx);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);

    for (int m = 0; m < M; ++m) {
        float mx = -INFINITY;
        for (int n = 0; n < N; ++n) mx = std::max(mx, x[m*N+n]);
        float sum = 0.f;
        for (int n = 0; n < N; ++n) sum += std::exp(x[m*N+n] - mx);
        for (int n = 0; n < N; ++n) {
            float e = std::exp(x[m*N+n] - mx) / sum;
            // 输出 row-major [m][n]（ggml dims[0]=N 最内）：got[m*N+n]
            EXPECT_NEAR(got[m*N + n], e, 1e-5f) << "m=" << m << " n=" << n;
        }
    }
    cudaFree(lx->data());
}

// out_prod：dst[i0,i1,i2,i3] = sum_k src0[i0,k,i2,i3]*src1[i1,k,i2,i3]
//   graph 布局 {ne0,ne1,ne2,ne3}；用 2D 简单矩阵验证外积
TEST(CudaBackendTest, OutProd) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // 简单 2D out_prod：src0 {C,K}, src1 {C2,K}（共 K 收缩维）→ dst {C2, C}
    const int C0 = 4, C1 = 5, K = 3;
    std::vector<float> a(C0*K), b(C1*K);
    for (int i = 0; i < C0*K; ++i) a[i] = (float)(i % 7) * 0.5f;
    for (int i = 0; i < C1*K; ++i) b[i] = (float)(i % 9) * 0.25f - 1.f;

    PPMLContext& ctx = context();
    // src0 {ne00=C0, ne01=K}, src1 {ne10=C1, ne11=K}, dst {ne0=C0, ne1=C1}
    TensorF32* la = make_device_leaf(ctx, {C0, K}, a);
    TensorF32* lb = make_device_leaf(ctx, {C1, K}, b);
    TensorF32* out = make_node(ctx, OP_OUT_PROD, {C0, C1}, la, lb);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);

    // 注意：out_prod 按 ggml 布局（dims[0]=最内维）处理所有张量。
    //   src0 {ne00=C0, ne01=K} 存为 src0[i0 + k*C0]（i0 最内）
    //   src1 {ne10=C1, ne11=K} 存为 src1[i1 + k*C1]
    //   dst  {ne0=C0, ne1=C1}  存为 dst[i0 + i1*C0]
    // 因此期望值须用同一布局从原始 a/b 数组反查，而非 C 行主序 a[i0*K+k]。
    for (int i0 = 0; i0 < C0; ++i0) {
        for (int i1 = 0; i1 < C1; ++i1) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += a[i0 + k*C0] * b[i1 + k*C1];
            EXPECT_NEAR(got[i0 + i1*C0], s, 1e-4f) << "i0=" << i0 << " i1=" << i1;
        }
    }
    cudaFree(la->data()); cudaFree(lb->data());
}

// concat：沿 dim 拼接
TEST(CudaBackendTest, Concat) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int D0 = 4, D1 = 3;
    std::vector<float> a(D0), b(D1);
    for (int i = 0; i < D0; ++i) a[i] = (float)i;
    for (int i = 0; i < D1; ++i) b[i] = 100.f + i;

    PPMLContext& ctx = context();
    TensorF32* la = make_device_leaf(ctx, {D0}, a);
    TensorF32* lb = make_device_leaf(ctx, {D1}, b);
    TensorF32* out = make_node(ctx, OP_CONCAT, {D0+D1}, la, lb);
    out->op_params[0] = 0;   // 沿 dim0 拼接

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    for (int i = 0; i < D0; ++i) EXPECT_FLOAT_EQ(got[i], a[i]);
    for (int i = 0; i < D1; ++i) EXPECT_FLOAT_EQ(got[D0+i], b[i]);

    cudaFree(la->data()); cudaFree(lb->data());
}

// LayerNorm（OP_NORM）：每行独立归一化（不带 affine）
TEST(CudaBackendTest, LayerNorm) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int C = 16, R = 4;
    std::vector<float> x(R*C);
    for (int i = 0; i < R*C; ++i) x[i] = (float)(i % 13) * 0.7f - 3.f;

    PPMLContext& ctx = context();
    // OP_NORM src[0]=inp {C,R}，输出 {C,R}；src[1]/src[2] 存 mean/rstd 用 device 叶子
    TensorF32* lin = make_device_leaf(ctx, {C, R}, x);
    TensorF32* mean = make_device_leaf(ctx, {R}, std::vector<float>(R, 0.f));
    TensorF32* rstd = make_device_leaf(ctx, {R}, std::vector<float>(R, 0.f));
    TensorF32* out = make_node(ctx, OP_NORM, {C, R}, lin, mean, rstd);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);

    for (int r = 0; r < R; ++r) {
        float m = 0.f;
        for (int c = 0; c < C; ++c) m += x[r*C+c];
        m /= C;
        float v = 0.f;
        for (int c = 0; c < C; ++c) v += (x[r*C+c]-m)*(x[r*C+c]-m);
        v /= C;
        float s = 1.f / std::sqrt(v + 1e-5f);
        for (int c = 0; c < C; ++c) {
            // 输出 row-major [r][c]（ggml dims[0]=C 最内）：got[r*C+c]
            EXPECT_NEAR(got[r*C + c], (x[r*C+c]-m)*s, 1e-4f) << "r=" << r << " c=" << c;
        }
    }
    cudaFree(lin->data()); cudaFree(mean->data()); cudaFree(rstd->data());
}

// ============================================================
// Q3：异步 kernel 同步语义
// ============================================================
// graph_compute 只 launch 异步 kernel，不隐式同步。
// 读回 host 前必须显式 synchronize() 或做 D2H 同步拷贝。
// 本测试验证：不 synchronize 直接读 host（用 cudaMemcpy 同步）也能拿到结果，
// 因为 cudaMemcpy 本身就同步等待 kernel 完成。
TEST(CudaBackendTest, AsyncKernelMemcpySyncs) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N = 256;
    std::vector<float> a(N, 1.f), b(N, 2.f);
    PPMLContext& ctx = context();
    TensorF32* la = make_device_leaf(ctx, {N}, a);
    TensorF32* lb = make_device_leaf(ctx, {N}, b);
    TensorF32* out = make_node(ctx, OP_ADD, {N}, la, lb);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    // 不调用 synchronize()，直接 cudaMemcpy（同步拷贝会等待 kernel 完成）
    std::vector<float> got;
    copy_to_host(out, got);
    for (int i = 0; i < N; ++i) EXPECT_FLOAT_EQ(got[i], 3.f);

    cudaFree(la->data()); cudaFree(lb->data());
}

// ============================================================
// Q4：scheduler 分配 + 跨后端
// ============================================================
// 注册 CPU + CUDA 两个后端，scheduler 应按 priority 把受支持算子分给 CUDA。
// supports_op 决定算子归属；不支持的算子回落 CPU。
TEST(CudaBackendTest, SchedulerAssignsCuda) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    PPMLContext& ctx = context();
    const int64_t N = 128;
    std::vector<float> a(N), b(N);
    for (int i = 0; i < N; ++i) { a[i] = (float)i; b[i] = 1.f; }

    // host 叶子（bind_leaf_data）：scheduler 把 out 放 GPU 时经 H2D 拷贝喂入。
    int64_t ne1d[1] = {N};
    TensorF32* la = ctx.new_tensor<float>(1, ne1d);
    TensorF32* lb = ctx.new_tensor<float>(1, ne1d);
    std::memcpy(bind_leaf_data(ctx, la), a.data(), sizeof(float)*N);
    std::memcpy(bind_leaf_data(ctx, lb), b.data(), sizeof(float)*N);
    TensorF32* out = make_node(ctx, OP_ADD, {N}, la, lb);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CPUBackend cpu(1);
    CUDABackend cuda(0);

    BackendScheduler sched;
    sched.add_backend(&cpu);    // priority 0
    sched.add_backend(&cuda);   // priority 1 → CUDA 优先

    sched.split_graph(g);

    // Q4：OP_ADD 是 CUDA 支持的算子，应分到 CUDA 后端（backends_[0]，priority 最高）
    EXPECT_EQ(sched.tensor_backend_id(out, -1), 0) << "OP_ADD 应分配到 CUDA";

    ASSERT_TRUE(sched.alloc_splits());
    ASSERT_EQ(sched.graph_compute(), Status::SUCCESS);
    cuda.synchronize();   // Q3：异步 kernel 同步后再读回

    // 读回并验证（CUDA 后端 buffer 的数据经 D2H）
    std::vector<float> got = read_tensor_cpu(out);
    for (int i = 0; i < N; ++i) EXPECT_FLOAT_EQ(got[i], (float)i + 1.f);
}

// scheduler：不受支持的算子应回落 CPU（OP_RMS_NORM 不受 CUDA 支持 → CPU）
TEST(CudaBackendTest, SchedulerFallsBackToCpu) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    PPMLContext& ctx = context();
    const int C = 16, R = 4;   // RMSNorm 每行独立归一化
    std::vector<float> a(R*C);
    for (int i = 0; i < R*C; ++i) a[i] = (float)(i % 11) * 0.5f - 2.f;

    // 叶子绑 host 数据（CPU 路径）；节点 OP_RMS_NORM（CUDA 不支持 → 回落 CPU）
    int64_t nr[2] = {C, R};
    TensorF32* la = ctx.new_tensor<float>(2, nr);
    float* hd = bind_leaf_data(ctx, la);
    std::memcpy(hd, a.data(), sizeof(float)*R*C);

    TensorF32* out = ctx.new_tensor<float>(2, nr);
    out->op = OP_RMS_NORM;
    out->src[0] = la;
    out->op_params[0] = 0;   // eps=0（与 CPU kernel 读取的 op_params[0] 一致）

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CPUBackend cpu(1);
    CUDABackend cuda(0);
    BackendScheduler sched;
    sched.add_backend(&cpu);
    sched.add_backend(&cuda);

    sched.split_graph(g);
    // OP_RMS_NORM 不受 CUDA 支持 → 分配到 CPU（backends_[1]）
    EXPECT_EQ(sched.tensor_backend_id(out, -1), 1) << "OP_RMS_NORM 应回落到 CPU";

    fprintf(stderr, "[DBG] n_splits=%d\n", sched.n_splits());
    ASSERT_TRUE(sched.alloc_splits());
    Status sst = sched.graph_compute();
    fprintf(stderr, "[DBG] graph_compute=%d\n", (int)sst);
    cuda.synchronize();

    // RMSNorm 值校验：out[r*C+c] = x[r*C+c]/sqrt(mean(x[r]^2)+eps)，eps=0
    const float eps = 0.f;
    for (int r = 0; r < R; ++r) {
        float ss = 0.f;
        for (int c = 0; c < C; ++c) ss += a[r*C+c] * a[r*C+c];
        ss /= C;
        float inv = 1.f / std::sqrt(ss + eps);
        for (int c = 0; c < C; ++c) {
            EXPECT_NEAR(out->data()[r*C+c], a[r*C+c] * inv, 1e-4f)
                << "r=" << r << " c=" << c;
        }
    }
}

// scheduler + 显存预算：只把一部分节点放 GPU（不 OOM），其余回落 CPU。
// 两个独立 ADD 链，各输出 4MB；预算 6MB → out1 放 GPU、out2 回落 CPU，
// 且两者数值都正确（跨后端拷贝经 dup + D2H）。
TEST(CudaBackendTest, SchedulerBudgetSubsetPlacement) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N = 1 << 20;   // 1M floats = 4MB
    std::vector<float> a1(N), b1(N), a2(N), b2(N);
    for (int i = 0; i < N; ++i) { a1[i] = 1.f; b1[i] = 2.f; a2[i] = 10.f; b2[i] = 20.f; }

    // 叶子绑 host 数据（bind_leaf_data，buffer_=null 视为 host）。
    // 跨后端拷贝经 backend_tensor_copy：host→GPU 走 H2D、host→CPU 走 host memcpy，均正确。
    PPMLContext& ctx = context();
    int64_t ne1d[1] = {N};
    TensorF32* la1 = ctx.new_tensor<float>(1, ne1d);
    TensorF32* lb1 = ctx.new_tensor<float>(1, ne1d);
    TensorF32* la2 = ctx.new_tensor<float>(1, ne1d);
    TensorF32* lb2 = ctx.new_tensor<float>(1, ne1d);
    std::memcpy(bind_leaf_data(ctx, la1), a1.data(), sizeof(float)*N);
    std::memcpy(bind_leaf_data(ctx, lb1), b1.data(), sizeof(float)*N);
    std::memcpy(bind_leaf_data(ctx, la2), a2.data(), sizeof(float)*N);
    std::memcpy(bind_leaf_data(ctx, lb2), b2.data(), sizeof(float)*N);
    TensorF32* out1 = make_node(ctx, OP_ADD, {N}, la1, lb1);
    TensorF32* out2 = make_node(ctx, OP_ADD, {N}, la2, lb2);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out1);
    g->build_forward_expand(out2);

    CPUBackend cpu(1);
    CUDABackend cuda(0);
    BackendScheduler sched;
    sched.add_backend(&cpu);
    sched.add_backend(&cuda);

    // 强制小预算：out1(4MB) 放 GPU 后预算耗尽，out2(4MB) 回落 CPU。
    setenv("PPML_GPU_BUDGET_MB", "6", 1);
    sched.split_graph(g);
    unsetenv("PPML_GPU_BUDGET_MB");

    // GPU 累计占用不超预算
    EXPECT_LE(sched.gpu_reserved_bytes(), (size_t)6 * 1024 * 1024);
    // out1 放 GPU（backends_[0]），out2 因预算不足回落 CPU（backends_[1]）
    EXPECT_EQ(sched.tensor_backend_id(out1, -1), 0) << "out1 应放 GPU";
    EXPECT_EQ(sched.tensor_backend_id(out2, -1), 1) << "out2 因预算不足应回落 CPU";

    ASSERT_TRUE(sched.alloc_splits());
    ASSERT_EQ(sched.graph_compute(), Status::SUCCESS);
    cuda.synchronize();

    // 结果都正确（out1 在 GPU、out2 在 CPU 均读回）
    std::vector<float> got1 = read_tensor_cpu(out1);
    std::vector<float> got2 = read_tensor_cpu(out2);
    ASSERT_EQ(got1.size(), (size_t)N);
    ASSERT_EQ(got2.size(), (size_t)N);
    for (int i = 0; i < 1000; ++i) {
        EXPECT_FLOAT_EQ(got1[i], 3.f);
        EXPECT_FLOAT_EQ(got2[i], 30.f);
    }
}

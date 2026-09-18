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

// ============================================================
// out_prod **真实 shape** 性能 + 正确性（2026-09-18）
//   ⚠ 热点必须用真实 shape 量：上面的 `OutProd`（4×5×3 微图）**不能当热点代理** ✗（历史教训）。
//   shape 来源 = ncu 实测的热点 launch `(4,1,2601)×(16,16,1)`：
//     z 方向 2601 = ne0*ne1 = 51*51（ne0=ne1=L=51）、收缩维 ne01=51、ne2=16、ne3=64（planes=1024）
//     ⇒ A/B/C 各 ≈10.65 MB（与 ncu 记录 "A+B+C 各 ~10.6 MB" 一致 ✓）；原耗时 46.38 ms / 51.15 ms。
//   直接调 wrapper `out_prod_cuda`（绕开图/gallocr ⇒ 纯 kernel 计时）✓
// ============================================================
namespace ppml {
extern void out_prod_cuda(const float* src0, const float* src1, float* dst,
                          int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                          int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
                          int64_t ne0,  int64_t ne1,  int64_t ne2,  int64_t ne3);
}

TEST(CudaBackendTest, OutProdRealShapePerfAndCorrect) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t L = 51, PL2 = 16, PL3 = 64, K = 51;
    const int64_t ne00 = L, ne01 = K, ne02 = PL2, ne03 = PL3;
    const int64_t ne10 = L, ne11 = K, ne12 = PL2, ne13 = PL3;
    const int64_t ne0  = L, ne1  = L, ne2  = PL2, ne3  = PL3;

    const size_t nA = (size_t)(ne00 * ne01 * ne02 * ne03);
    const size_t nB = (size_t)(ne10 * ne11 * ne12 * ne13);
    const size_t nC = (size_t)(ne0 * ne1 * ne2 * ne3);
    std::vector<float> A(nA), B(nB);
    for (size_t i = 0; i < nA; ++i) A[i] = (float)(i % 13) * 0.0625f - 0.5f;
    for (size_t i = 0; i < nB; ++i) B[i] = (float)(i % 7) * 0.125f - 0.25f;

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    ASSERT_EQ(cudaMalloc(&dA, nA * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dB, nB * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dC, nC * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dA, A.data(), nA * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dB, B.data(), nB * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    cudaMemset(dC, 0, nC * sizeof(float));

    auto run = [&]() {
        ppml::out_prod_cuda(dA, dB, dC,
                            ne00, ne01, ne02, ne03,
                            ne10, ne11, ne12, ne13,
                            ne0,  ne1,  ne2,  ne3);
    };

    // ---- 计时（3 次热身 + 20 次平均）----
    for (int i = 0; i < 3; ++i) run();
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    constexpr int R = 20;
    for (int i = 0; i < R; ++i) run();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    ms /= (float)R;

    const double flops = 2.0 * (double)ne0 * (double)ne1 * (double)ne01
                             * (double)(ne2 * ne3);
    const double useful_mb = (double)(nA + nB + nC) * 4.0 / 1048576.0;
    printf("[outprod-perf] ne0=ne1=%lld K=%lld planes=%lld | %.4f ms | %.2f GFLOPS | "
           "useful %.1f MB (%.2f GB/s useful)\n",
           (long long)ne0, (long long)ne01, (long long)(ne2 * ne3), (double)ms,
           flops / ((double)ms * 1e-3) / 1e9, useful_mb,
           useful_mb / 1024.0 / ((double)ms * 1e-3));

    // ---- 正确性（真实 shape，抽查两个极端平面 × 全部 (i0,i1)）----
    std::vector<float> C(nC);
    ASSERT_EQ(cudaMemcpy(C.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    const int64_t planes[2][2] = {{0, 0}, {ne2 - 1, ne3 - 1}};
    double max_abs = 0.0;
    for (int pi = 0; pi < 2; ++pi) {
        const int64_t i2 = planes[pi][0], i3 = planes[pi][1];
        const size_t s0p = (size_t)(i2 * ne00 * ne01 + i3 * ne00 * ne01 * ne02);
        const size_t s1p = (size_t)(i2 * ne10 * ne11 + i3 * ne10 * ne11 * ne12);
        const size_t dp  = (size_t)(i2 * ne0 * ne1 + i3 * ne0 * ne1 * ne2);
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                float s = 0.f;
                for (int64_t k = 0; k < ne01; ++k) {
                    s += A[s0p + (size_t)(i0 + k * ne00)] * B[s1p + (size_t)(i1 + k * ne10)];
                }
                const double diff = std::fabs((double)s - (double)C[dp + (size_t)(i0 + i1 * ne0)]);
                if (diff > max_abs) max_abs = diff;
            }
        }
    }
    printf("[outprod-check] 真实 shape 抽查 max|Δ| = %.3e（参考 = CPU 按 ggml 布局手算）\n", max_abs);
    EXPECT_LT(max_abs, 1e-3);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
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

// ============================================================
// OP_REPEAT_BACK：归约（repeat 的逆操作，梯度和）
//   dst[j] = Σ_k src[j + k*dd]，每维 k ∈ [0, src_dim/dst_dim)。
//   CUDA kernel 为「原结构：每线程一 dst 元素，串行累加所有重复拷贝」，
//   越界修复：tid0/tid1 越界 return + tid23 grid-stride 覆盖。
//   本测试用多个形状（含非 block 整数倍维度，触发越界线程路径）与 CPU 参考对比。
// ============================================================

// CPU 参考：同 ndim 的 repeat_back（对齐 CPUBackend::kernel_repeat_back）
// src_dims/dst_dims 为 ggml 布局（dims[0] 最内维）。
std::vector<float> repeat_back_reference(
    const std::vector<float>& src,
    const std::vector<int64_t>& src_dims,
    const std::vector<int64_t>& dst_dims) {
    const int64_t sd[4] = {
        src_dims.size() > 0 ? src_dims[0] : 1,
        src_dims.size() > 1 ? src_dims[1] : 1,
        src_dims.size() > 2 ? src_dims[2] : 1,
        src_dims.size() > 3 ? src_dims[3] : 1};
    const int64_t dd[4] = {
        dst_dims.size() > 0 ? dst_dims[0] : 1,
        dst_dims.size() > 1 ? dst_dims[1] : 1,
        dst_dims.size() > 2 ? dst_dims[2] : 1,
        dst_dims.size() > 3 ? dst_dims[3] : 1};
    const int64_t rd[4] = {sd[0]/dd[0], sd[1]/dd[1], sd[2]/dd[2], sd[3]/dd[3]};

    const int64_t total = dd[0]*dd[1]*dd[2]*dd[3];
    std::vector<float> dst(total, 0.f);
    for (int64_t idx = 0; idx < total; ++idx) {
        int64_t t = idx;
        const int64_t j0 = t % dd[0]; t /= dd[0];
        const int64_t j1 = t % dd[1]; t /= dd[1];
        const int64_t j2 = t % dd[2]; t /= dd[2];
        const int64_t j3 = t;
        float sum = 0.f;
        for (int64_t k0 = 0; k0 < rd[0]; k0++) {
            const int64_t s0 = j0 + k0*dd[0];
            for (int64_t k1 = 0; k1 < rd[1]; k1++) {
                const int64_t s1 = j1 + k1*dd[1];
                for (int64_t k2 = 0; k2 < rd[2]; k2++) {
                    const int64_t s2 = j2 + k2*dd[2];
                    for (int64_t k3 = 0; k3 < rd[3]; k3++) {
                        const int64_t s3 = j3 + k3*dd[3];
                        sum += src[((s3*sd[2] + s2)*sd[1] + s1)*sd[0] + s0];
                    }
                }
            }
        }
        dst[idx] = sum;
    }
    return dst;
}

// 运行 CUDA repeat_back 并返回结果
std::vector<float> run_cuda_repeat_back(
    const std::vector<float>& src_data,
    const std::vector<int64_t>& src_dims,
    const std::vector<int64_t>& dst_dims) {
    PPMLContext& ctx = context();
    TensorF32* la = make_device_leaf(ctx, src_dims, src_data);
    TensorF32* out = make_node(ctx, OP_REPEAT_BACK, dst_dims, la);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    EXPECT_EQ(backend.graph_compute(g), Status::SUCCESS);  // helper 非 void，用 EXPECT
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(la->data());
    return got;
}

TEST(CudaBackendTest, RepeatBack2D) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // src [4,12] → dst [2,3]：ne0 重复 2x，ne1 重复 4x。
    // 非 block 整数倍维度（ne0=2, ne1=3）触发越界线程路径。
    const std::vector<int64_t> src_dims = {4, 12};
    const std::vector<int64_t> dst_dims = {2, 3};
    const int64_t n_src = 4*12;
    std::vector<float> src(n_src);
    for (int i = 0; i < n_src; ++i) src[i] = (float)(i % 17) * 0.5f - 3.f;

    std::vector<float> got = run_cuda_repeat_back(src, src_dims, dst_dims);
    std::vector<float> exp = repeat_back_reference(src, src_dims, dst_dims);

    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

TEST(CudaBackendTest, RepeatBack3D) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // src [6,10,3] → dst [3,5,3]：ne0 重复 2x，ne1 重复 2x，ne2 相同。
    const std::vector<int64_t> src_dims = {6, 10, 3};
    const std::vector<int64_t> dst_dims = {3, 5, 3};
    const int64_t n_src = 6*10*3;
    std::vector<float> src(n_src);
    for (int i = 0; i < n_src; ++i) src[i] = (float)(i % 23) * 0.25f + 1.f;

    std::vector<float> got = run_cuda_repeat_back(src, src_dims, dst_dims);
    std::vector<float> exp = repeat_back_reference(src, src_dims, dst_dims);

    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

TEST(CudaBackendTest, RepeatBack4D) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // src [2,3,4,6] → dst [2,3,2,2]：ne2 重复 2x，ne3 重复 3x。
    // 覆盖 tid23 折叠（ne2*ne3=4*6=24 > BLOCK_Z=32? 否, 但 ne2*ne3 非 BLOCK_Z 整数倍）
    const std::vector<int64_t> src_dims = {2, 3, 4, 6};
    const std::vector<int64_t> dst_dims = {2, 3, 2, 2};
    const int64_t n_src = 2*3*4*6;
    std::vector<float> src(n_src);
    for (int i = 0; i < n_src; ++i) src[i] = (float)(i % 31) * 0.1f - 2.f;

    std::vector<float> got = run_cuda_repeat_back(src, src_dims, dst_dims);
    std::vector<float> exp = repeat_back_reference(src, src_dims, dst_dims);

    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

// 训练真实场景：msa 掩码广播 [1,1,N,1] → [D,L,N,B] 的逆归约
// （src 为 repeat 后的梯度 [D,L,N,B]，dst 归约回 [1,1,N,1]）。
TEST(CudaBackendTest, RepeatBackMask4D) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const std::vector<int64_t> src_dims = {256, 12, 8, 2};  // [D,L,N,B]
    const std::vector<int64_t> dst_dims = {1, 1, 8, 2};     // [1,1,N,B]
    const int64_t n_src = 256*12*8*2;
    std::vector<float> src(n_src);
    for (int i = 0; i < n_src; ++i) src[i] = (float)(i % 5) - 2.f;

    std::vector<float> got = run_cuda_repeat_back(src, src_dims, dst_dims);
    std::vector<float> exp = repeat_back_reference(src, src_dims, dst_dims);

    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

// ============================================================
// OP_REPEAT 前向：dst[j] = src[s]，s_d = j_d % src_dims[d]
//   简化版 CUDA kernel（一维 grid-stride + 尾部对齐取模，对齐 CPU kernel_repeat）。
//   覆盖：2D 重复、训练真实掩码 [1,1,N,B]→[D,L,N,B]（内层 B 也被放大，
//   可验证"整块取模"假设不成立时仍正确）、跨 ndim 广播（3D→4D，seq_emb 场景）。
// ============================================================

// CPU 参考（对齐 CPUBackend::kernel_repeat，src_dims/dst_dims 为 ggml 布局 dims[0]=最内）
std::vector<float> repeat_reference(
    const std::vector<float>& src,
    const std::vector<int64_t>& src_dims,
    const std::vector<int64_t>& dst_dims) {
    const int64_t sd[4] = {
        (src_dims.size() > 0) ? src_dims[0] : 1,
        (src_dims.size() > 1) ? src_dims[1] : 1,
        (src_dims.size() > 2) ? src_dims[2] : 1,
        (src_dims.size() > 3) ? src_dims[3] : 1,
    };
    const int64_t dd[4] = {
        (dst_dims.size() > 0) ? dst_dims[0] : 1,
        (dst_dims.size() > 1) ? dst_dims[1] : 1,
        (dst_dims.size() > 2) ? dst_dims[2] : 1,
        (dst_dims.size() > 3) ? dst_dims[3] : 1,
    };
    const int64_t total = dd[0]*dd[1]*dd[2]*dd[3];
    std::vector<float> dst(total);
    for (int64_t idx = 0; idx < total; ++idx) {
        int64_t t = idx;
        const int64_t i3 = t % dd[3]; t /= dd[3];
        const int64_t i2 = t % dd[2]; t /= dd[2];
        const int64_t i1 = t % dd[1]; t /= dd[1];
        const int64_t i0 = t;
        const int64_t s0 = i0 % sd[0];
        const int64_t s1 = i1 % sd[1];
        const int64_t s2 = i2 % sd[2];
        const int64_t s3 = i3 % sd[3];
        dst[idx] = src[((s3 * sd[2] + s2) * sd[1] + s1) * sd[0] + s0];
    }
    return dst;
}

// 运行 CUDA repeat 前向并返回结果（src[1] 为形状模板，不读数据）
std::vector<float> run_cuda_repeat(
    const std::vector<float>& src_data,
    const std::vector<int64_t>& src_dims,
    const std::vector<int64_t>& dst_dims) {
    PPMLContext& ctx = context();
    TensorF32* la = make_device_leaf(ctx, src_dims, src_data);
    int64_t ne[4] = {1,1,1,1};
    for (size_t i = 0; i < dst_dims.size() && i < 4; ++i) ne[i] = dst_dims[i];
    // 形状模板：op=OP_NONE 且无数据 → build_forward_expand 按叶子处理，compute 不读它
    TensorF32* lb = ctx.new_tensor<float>(static_cast<int>(dst_dims.size()), ne);
    TensorF32* out = make_node(ctx, OP_REPEAT, dst_dims, la, lb);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    EXPECT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(la->data());
    return got;
}

TEST(CudaBackendTest, RepeatForward2D) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // src [2,5] → dst [6,5]：dim0 重复 3x（每 2 行一组）。
    const std::vector<int64_t> src_dims = {2, 5};
    const std::vector<int64_t> dst_dims = {6, 5};
    const int64_t n_src = 2*5;
    std::vector<float> src(n_src);
    for (int i = 0; i < n_src; ++i) src[i] = (float)(i % 11) * 0.5f - 2.f;

    std::vector<float> got = run_cuda_repeat(src, src_dims, dst_dims);
    std::vector<float> exp = repeat_reference(src, src_dims, dst_dims);

    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

TEST(CudaBackendTest, RepeatForwardMask4D) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // 训练真实掩码：src [1,1,N,B] → dst [D,L,N,B]。
    // B=2>1 → 全局 flat % src_numel 假设失效，必须逐维取模才正确（本 kernel 的做法）。
    const std::vector<int64_t> src_dims = {1, 1, 8, 2};
    const std::vector<int64_t> dst_dims = {4, 6, 8, 2};
    const int64_t n_src = 1*1*8*2;
    std::vector<float> src(n_src);
    for (int i = 0; i < n_src; ++i) src[i] = (float)(i % 5) - 2.f;

    std::vector<float> got = run_cuda_repeat(src, src_dims, dst_dims);
    std::vector<float> exp = repeat_reference(src, src_dims, dst_dims);

    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

TEST(CudaBackendTest, RepeatForwardSeqBroadcast) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // seq_emb 广播：src [1,L,B] (3D) → dst [d,L,N,B] (4D)，跨 ndim（缺 dim0）。
    // 覆盖 src 缺维时"缺维取模 1 → s0=0"的路径。
    const std::vector<int64_t> src_dims = {1, 5, 2};
    const std::vector<int64_t> dst_dims = {3, 5, 4, 2};
    const int64_t n_src = 1*5*2;
    std::vector<float> src(n_src);
    for (int i = 0; i < n_src; ++i) src[i] = (float)(i % 7) * 0.25f + 1.f;

    std::vector<float> got = run_cuda_repeat(src, src_dims, dst_dims);
    std::vector<float> exp = repeat_reference(src, src_dims, dst_dims);

    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

TEST(CudaBackendTest, RepeatShrinkMix4D) {
    // 2026-09-06 回归：split=23 node=4 真实形状 src[256,51,1,1]→dst[64,51,8,1]。
    // dst 最内维(64) < src 最内维(256)——非"放大"，此前 dst≥src 校验拒绝 → split FAILED
    // 回退 CPU。CPU kernel_repeat 无校验逐维取模，必须与 CUDA 完全一致。
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const std::vector<int64_t> src_dims = {256, 51, 1, 1};
    const std::vector<int64_t> dst_dims = {64, 51, 8, 1};
    const int64_t n_src = 256*51;
    std::vector<float> src(n_src);
    for (int i = 0; i < n_src; ++i) src[i] = (float)(i % 17) * 0.5f - 3.f;

    std::vector<float> got = run_cuda_repeat(src, src_dims, dst_dims);
    std::vector<float> exp = repeat_reference(src, src_dims, dst_dims);

    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

// ============================================================
// OP_SET_ROWS 前向（散点覆写）：dst = a 全量拷贝 + 按 b[k] 覆写 c[k] 行
//   对齐 CPU kernel_set_rows (CPUKernels.cpp:1873)。
//   b 为 (K,) float 编码行索引；越界钳制；b 有重复索引 → 跳过 scatter 仅留 a 拷贝。
// ============================================================

// CPU 参考（镜像 CPUBackend::kernel_set_rows：查重 → 拷贝 → 无重复才散点）
std::vector<float> set_rows_reference(
    const std::vector<float>& a, const std::vector<float>& b,
    const std::vector<float>& c, int64_t N, int64_t M, int64_t K) {
    std::vector<float> dst = a;                       // 1) 拷贝 a 全量
    std::vector<uint8_t> seen((size_t)N, 0);          // 2) 查重（越界索引不参与）
    bool dup = false;
    for (int64_t k = 0; k < K && !dup; ++k) {
        const int64_t i1 = (int64_t)b[(size_t)k];
        if (i1 < 0 || i1 >= N) continue;
        if (seen[(size_t)i1]) dup = true;
        seen[(size_t)i1] = 1;
    }
    if (!dup) {                                       // 3) 无重复才 scatter（钳制）
        for (int64_t k = 0; k < K; ++k) {
            int64_t i1 = (int64_t)b[(size_t)k];
            if (i1 < 0) i1 = 0;
            if (i1 >= N) i1 = N - 1;
            std::memcpy(dst.data() + i1*M, c.data() + k*M, M * sizeof(float));
        }
    }
    return dst;
}

// 运行 CUDA set_rows 并返回结果
std::vector<float> run_cuda_set_rows(
    const std::vector<float>& a, const std::vector<float>& b,
    const std::vector<float>& c, int64_t N, int64_t M, int64_t K) {
    PPMLContext& ctx = context();
    TensorF32* la = make_device_leaf(ctx, {N, M}, a);
    TensorF32* lb = make_device_leaf(ctx, {K}, b);
    TensorF32* lc = make_device_leaf(ctx, {K, M}, c);
    TensorF32* out = make_node(ctx, OP_SET_ROWS, {N, M}, la, lb, lc);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    EXPECT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(la->data()); cudaFree(lb->data()); cudaFree(lc->data());
    return got;
}

TEST(CudaBackendTest, SetRowsBasic) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N = 4, M = 5, K = 2;
    std::vector<float> a(N*M), c(K*M);
    for (int i = 0; i < N*M; ++i) a[i] = (float)(i % 9) * 0.5f - 2.f;   // 目标基值
    for (int i = 0; i < K*M; ++i) c[i] = (float)(i % 7) + 10.f;          // 覆写值
    const std::vector<float> b = {1.f, 3.f};                             // 覆写第 1、3 行

    std::vector<float> got = run_cuda_set_rows(a, b, c, N, M, K);
    std::vector<float> exp = set_rows_reference(a, b, c, N, M, K);

    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_FLOAT_EQ(got[i], exp[i]) << "i=" << i;
    }
}

TEST(CudaBackendTest, SetRowsDupSkipsScatter) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N = 4, M = 3, K = 2;
    std::vector<float> a(N*M), c(K*M);
    for (int i = 0; i < N*M; ++i) a[i] = (float)(i % 5) + 1.f;
    for (int i = 0; i < K*M; ++i) c[i] = (float)(i % 3) - 5.f;
    const std::vector<float> b = {2.f, 2.f};   // 重复索引 → 跳过 scatter，仅留 a 拷贝

    std::vector<float> got = run_cuda_set_rows(a, b, c, N, M, K);
    std::vector<float> exp = set_rows_reference(a, b, c, N, M, K);

    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_FLOAT_EQ(got[i], exp[i]) << "i=" << i;
    }
    // 语义验证：结果必须 == a（scatter 被跳过）
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_FLOAT_EQ(got[i], a[i]) << "i=" << i;
    }
}

TEST(CudaBackendTest, SetRowsClampOutOfRange) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N = 4, M = 5, K = 2;
    std::vector<float> a(N*M), c(K*M);
    for (int i = 0; i < N*M; ++i) a[i] = (float)(i % 8) * 0.25f;
    for (int i = 0; i < K*M; ++i) c[i] = (float)(i % 6) + 20.f;
    const std::vector<float> b = {-2.f, 100.f};   // 越界 → 钳制到行 0 与行 N-1

    std::vector<float> got = run_cuda_set_rows(a, b, c, N, M, K);
    std::vector<float> exp = set_rows_reference(a, b, c, N, M, K);

    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_FLOAT_EQ(got[i], exp[i]) << "i=" << i;
    }
}

// ============================================================
// OP_SCALE 前向：dst = src * s（s 存 op_params[0] float 位模式，对齐 CPU kernel_scale）
//   ggml scale_f32 参考版带 bias，项目 scale() helper 无 bias（仅标量乘）。
// ============================================================
std::vector<float> run_cuda_scale(const std::vector<float>& x, float s) {
    PPMLContext& ctx = context();
    const int64_t N = (int64_t)x.size();
    TensorF32* la = make_device_leaf(ctx, {N}, x);
    TensorF32* out = make_node(ctx, OP_SCALE, {N}, la);
    reinterpret_cast<float&>(out->op_params[0]) = s;   // scale() helper 的写入方式

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    EXPECT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(la->data());
    return got;
}

TEST(CudaBackendTest, ScaleBasic) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N = 4096;   // >256 → grid-stride 多轮
    std::vector<float> x(N);
    for (int i = 0; i < N; ++i) x[i] = (float)(i % 13) * 0.5f - 3.f;
    const float s = 2.5f;

    std::vector<float> got = run_cuda_scale(x, s);
    ASSERT_EQ(got.size(), x.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_FLOAT_EQ(got[i], x[i] * s) << "i=" << i;
    }
}

TEST(CudaBackendTest, ScaleNegativeAndScalar) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // 负标量 + 非 256 倍数（触发边界线程）
    {
        const int64_t N = 257;
        std::vector<float> x(N);
        for (int i = 0; i < N; ++i) x[i] = (float)(i % 7) - 4.f;
        const float s = -0.75f;
        std::vector<float> got = run_cuda_scale(x, s);
        ASSERT_EQ(got.size(), x.size());
        for (size_t i = 0; i < got.size(); ++i) {
            EXPECT_FLOAT_EQ(got[i], x[i] * s) << "i=" << i;
        }
    }
    // numel=1（训练 loss 链标量 scale 场景）
    {
        std::vector<float> x = {3.14f};
        const float s = 0.1f;
        std::vector<float> got = run_cuda_scale(x, s);
        EXPECT_FLOAT_EQ(got[0], 3.14f * 0.1f);
    }
}

// ============================================================
// 广播逐元素 op（OP_ADD/SUB/MUL/DIV）：dst[i] = op(a[i%an], b[i%bn])
//   对齐 CPU kernel_elemwise 的"尾部对齐后缀"广播语义（CPUKernels.cpp:447）。
// ============================================================
std::vector<float> bcast_add_reference(
    const std::vector<float>& a, const std::vector<float>& b, int64_t n) {
    std::vector<float> dst(n);
    const int64_t an = (int64_t)a.size();
    const int64_t bn = (int64_t)b.size();
    for (int64_t i = 0; i < n; ++i) dst[i] = a[i % an] + b[i % bn];
    return dst;
}

std::vector<float> run_cuda_bcast_add(
    const std::vector<float>& a, const std::vector<int64_t>& a_dims,
    const std::vector<float>& b, const std::vector<int64_t>& b_dims,
    const std::vector<int64_t>& out_dims) {
    PPMLContext& ctx = context();
    TensorF32* la = make_device_leaf(ctx, a_dims, a);
    TensorF32* lb = make_device_leaf(ctx, b_dims, b);
    TensorF32* out = make_node(ctx, OP_ADD, out_dims, la, lb);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    EXPECT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(la->data()); cudaFree(lb->data());
    return got;
}

TEST(CudaBackendTest, BcastAdd2D) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // a=[2,7] 全量 + b=[7] 尾部对齐广播 → dst[2,7]
    const std::vector<int64_t> a_dims = {2, 7};
    const std::vector<int64_t> b_dims = {7};
    const std::vector<int64_t> out_dims = {2, 7};
    std::vector<float> a(14), b(7);
    for (int i = 0; i < 14; ++i) a[i] = (float)(i % 9) * 0.5f - 2.f;
    for (int i = 0; i < 7;  ++i) b[i] = (float)(i % 5) + 1.f;

    std::vector<float> got = run_cuda_bcast_add(a, a_dims, b, b_dims, out_dims);
    std::vector<float> exp = bcast_add_reference(a, b, 14);
    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

TEST(CudaBackendTest, BcastAdd4D) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // a=[3,5,4,2] 全量 + b=[5,4,2]（缺前导最内维 dim0）→ 沿 dim0 广播
    const std::vector<int64_t> a_dims = {3, 5, 4, 2};
    const std::vector<int64_t> b_dims = {5, 4, 2};
    const std::vector<int64_t> out_dims = {3, 5, 4, 2};
    std::vector<float> a(120), b(40);
    for (int i = 0; i < 120; ++i) a[i] = (float)(i % 11) * 0.25f - 1.f;
    for (int i = 0; i < 40;  ++i) b[i] = (float)(i % 7) - 3.f;

    std::vector<float> got = run_cuda_bcast_add(a, a_dims, b, b_dims, out_dims);
    std::vector<float> exp = bcast_add_reference(a, b, 120);
    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

// OP_ADD1（加标量）：dst = src + b（b 为标量张量）
TEST(CudaBackendTest, Add1Scalar) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N = 1024;
    std::vector<float> x(N);
    for (int i = 0; i < N; ++i) x[i] = (float)(i % 17) * 0.5f - 4.f;
    const std::vector<float> b = {5.f};   // 标量张量

    PPMLContext& ctx = context();
    TensorF32* la = make_device_leaf(ctx, {N}, x);
    TensorF32* lb = make_device_leaf(ctx, {1}, b);
    TensorF32* out = make_node(ctx, OP_ADD1, {N}, la, lb);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);
    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(la->data()); cudaFree(lb->data());

    ASSERT_EQ(got.size(), x.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_FLOAT_EQ(got[i], x[i] + 5.f) << "i=" << i;
    }
}

// ============================================================
// OP_MAX_ALL 前向：全元素归约 max → 标量（参考 sum/mean reduce 结构）
//   对齐 CPU kernel_max_all：跳 NaN（m 初值 -INF，!isnan(v)&&v>m 才更新）。
// ============================================================
std::vector<float> run_cuda_max_all(const std::vector<float>& x) {
    PPMLContext& ctx = context();
    const int64_t N = (int64_t)x.size();
    TensorF32* la = make_device_leaf(ctx, {N}, x);
    TensorF32* out = make_node(ctx, OP_MAX_ALL, {1}, la);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    EXPECT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(la->data());
    return got;
}

TEST(CudaBackendTest, MaxAllBasic) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // 混合正负 + 重复最大值，N>4096 触发多 block + atomic
    const int64_t N = 5000;
    std::vector<float> x(N);
    for (int i = 0; i < N; ++i) x[i] = (float)((i % 23) - 11) * 0.5f;   // ∈ [-5.5, 5.5]
    x[100]   = 7.25f;
    x[4999]  = 7.25f;    // 两个并列最大

    std::vector<float> got = run_cuda_max_all(x);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_FLOAT_EQ(got[0], 7.25f);
}

TEST(CudaBackendTest, MaxAllSkipsNaN) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // 含 NaN：CPU 跳 NaN，结果取剩余元素 max
    {
        std::vector<float> x = {1.f, 5.f, std::nanf(""), 3.f, -2.f};
        std::vector<float> got = run_cuda_max_all(x);
        EXPECT_FLOAT_EQ(got[0], 5.f);
    }
    // 全 NaN：CPU 返回 -INF（m 初值不变）
    {
        std::vector<float> x(256, std::nanf(""));
        std::vector<float> got = run_cuda_max_all(x);
        EXPECT_TRUE(std::isinf(got[0]) && got[0] < 0.f);
    }
}

// ============================================================
// OP_SUM_ROWS 前向：沿最内维 dims[0] 归约，保留其余维（输出 {1, dims[1..3]}）
//   block-per-row（grid=nrows）。对齐 CPU kernel_sum_rows (CPUKernels.cpp:1536)。
// ============================================================

// CPU 参考（镜像 CPUBackend::kernel_sum_rows，src_dims 为 ggml 布局 dims[0]=最内）
std::vector<float> sum_rows_reference(
    const std::vector<float>& src, const std::vector<int64_t>& src_dims) {
    const int64_t ne0 = src_dims[0];
    const int64_t ne1 = (src_dims.size() > 1) ? src_dims[1] : 1;
    const int64_t ne2 = (src_dims.size() > 2) ? src_dims[2] : 1;
    const int64_t ne3 = (src_dims.size() > 3) ? src_dims[3] : 1;
    const int64_t total = ne1 * ne2 * ne3;
    std::vector<float> dst(total);
    for (int64_t idx = 0; idx < total; ++idx) {
        int64_t t = idx;
        const int64_t i1 = t % ne1; t /= ne1;
        const int64_t i2 = t % ne2; t /= ne2;
        const int64_t i3 = t;
        const float* row = src.data() + ((i3 * ne2 + i2) * ne1 + i1) * ne0;
        float s = 0.0f;
        for (int64_t i0 = 0; i0 < ne0; ++i0) s += row[i0];
        dst[idx] = s;
    }
    return dst;
}

// 运行 CUDA sum_rows 并返回结果
std::vector<float> run_cuda_sum_rows(
    const std::vector<float>& src, const std::vector<int64_t>& src_dims,
    const std::vector<int64_t>& out_dims) {
    PPMLContext& ctx = context();
    TensorF32* la = make_device_leaf(ctx, src_dims, src);
    TensorF32* out = make_node(ctx, OP_SUM_ROWS, out_dims, la);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    EXPECT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(la->data());
    return got;
}

TEST(CudaBackendTest, SumRows2D) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // src [5,7]（ne0=5 最内）→ dst {1,7}：7 行，每行 5 个元素求和
    const std::vector<int64_t> src_dims = {5, 7};
    const std::vector<int64_t> out_dims = {1, 7};
    std::vector<float> src(35);
    for (int i = 0; i < 35; ++i) src[i] = (float)(i % 9) * 0.5f - 2.f;

    std::vector<float> got = run_cuda_sum_rows(src, src_dims, out_dims);
    std::vector<float> exp = sum_rows_reference(src, src_dims);

    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

TEST(CudaBackendTest, SumRows4D) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // src [3,4,5,2] → dst {1,4,5,2}：nrows = 4*5*2 = 40 个 block
    const std::vector<int64_t> src_dims = {3, 4, 5, 2};
    const std::vector<int64_t> out_dims = {1, 4, 5, 2};
    std::vector<float> src(120);
    for (int i = 0; i < 120; ++i) src[i] = (float)(i % 13) * 0.25f + 1.f;

    std::vector<float> got = run_cuda_sum_rows(src, src_dims, out_dims);
    std::vector<float> exp = sum_rows_reference(src, src_dims);

    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

TEST(CudaBackendTest, SumRowsLongRow) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // ne0=1024 > 256 → block 内 grid-stride 多轮；ne1=3 → 3 个 block
    const std::vector<int64_t> src_dims = {1024, 3};
    const std::vector<int64_t> out_dims = {1, 3};
    std::vector<float> src(1024 * 3);
    for (int i = 0; i < 1024 * 3; ++i) src[i] = (float)(i % 7) - 3.f;

    std::vector<float> got = run_cuda_sum_rows(src, src_dims, out_dims);
    std::vector<float> exp = sum_rows_reference(src, src_dims);

    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

// ============================================================
// OP_RELU_BACK 反向：dst[i] = (x[i]>0) ? grad[i] : 0（对齐 CPU kernel_relu_back）
//   src0=grad, src1=x。x=0 边界梯度置 0（与 PyTorch 同约定）。
// ============================================================
TEST(CudaBackendTest, ReluBack) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // 正负混合 + 恰好为 0 的边界 + 非 256 倍数（N=300）
    const int64_t N = 300;
    std::vector<float> x(N), g(N);
    for (int i = 0; i < N; ++i) {
        x[i] = (float)((i % 5) - 2);     // ∈ {-2,-1,0,1,2}
        g[i] = (float)(i % 11) * 0.5f;   // 非零梯度
    }

    PPMLContext& ctx = context();
    TensorF32* lg = make_device_leaf(ctx, {N}, g);
    TensorF32* lx = make_device_leaf(ctx, {N}, x);
    TensorF32* out = make_node(ctx, OP_RELU_BACK, {N}, lg, lx);

    ComputeGraph* gph = ComputeGraph::new_graph(&ctx);
    gph->build_forward_expand(out);

    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(gph), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(lg->data()); cudaFree(lx->data());

    for (int64_t i = 0; i < N; ++i) {
        const float expect = (x[i] > 0.0f) ? g[i] : 0.0f;
        EXPECT_FLOAT_EQ(got[i], expect) << "i=" << i;
    }
}

// ============================================================
// OP_SCATTER_ADD 前向（SE3 消息散点累加）：dst[tgt[e],:] += msg[e,:]
//   msg dims=[M,E]（dims[0]=M 最内=每行长度，dims[1]=E=边数），tgt_idx (E,)
//   dst dims=[M,N]（N=node_count，存 op_params[0] 值语义）。
//   CUDA kernel：先清零再 atomicAdd；越界丢弃；同目标多边累加（对齐 CPU）。
// ============================================================

// CPU 参考（镜像 CPUBackend::kernel_scatter_add：清零+串行累加）
std::vector<float> scatter_add_reference(
    const std::vector<float>& msg, const std::vector<float>& tgt_idx,
    int N, int M, int E) {
    std::vector<float> dst((size_t)N * M, 0.f);
    for (int e = 0; e < E; ++e) {
        const int64_t i = (int64_t)tgt_idx[(size_t)e];
        if (i < 0 || i >= N) continue;
        for (int c = 0; c < M; ++c) dst[(size_t)i * M + c] += msg[(size_t)e * M + c];
    }
    return dst;
}

// CUDA 单后端：CUDABackend::graph_compute（gallocr 分配 dst）
TEST(CudaBackendTest, ScatterAdd) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int N = 4, M = 3, E = 6;
    std::vector<float> msg(E * M);
    for (int i = 0; i < E * M; ++i) msg[i] = (float)(i % 7) * 0.5f - 1.f;
    const std::vector<float> tgt = {0.f, 1.f, 2.f, 1.f, 3.f, 0.f};   // 目标 0/1 各两条边 → atomicAdd 累加

    PPMLContext& ctx = context();
    TensorF32* lm = make_device_leaf(ctx, {M, E}, msg);
    TensorF32* li = make_device_leaf(ctx, {(int64_t)E}, tgt);
    TensorF32* out = make_node(ctx, OP_SCATTER_ADD, {M, N}, lm, li);
    out->op_params[0] = (float)N;   // 同 scatter_add helper 的写入方式（int→float 值语义）

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(lm->data()); cudaFree(li->data());

    std::vector<float> exp = scatter_add_reference(msg, tgt, N, M, E);
    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

// 混合调度：scatter_add 支持走 GPU（supports_op true），msg/tgt_idx 为 host 叶子，
// 经 scheduler 跨后端 H2D 拷贝到 GPU buffer；dst 由 scheduler gallocr 分配。
TEST(CudaBackendTest, ScatterAddScheduler) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int N = 4, M = 3, E = 5;
    std::vector<float> msg(E * M);
    for (int i = 0; i < E * M; ++i) msg[i] = (float)(i % 5) * 0.25f + 0.5f;
    const std::vector<float> tgt = {1.f, 2.f, 1.f, 3.f, 0.f};       // 目标 1 两条边

    PPMLContext& ctx = context();
    int64_t md[2] = {M, E};
    TensorF32* lm = ctx.new_tensor<float>(2, md);
    std::memcpy(bind_leaf_data(ctx, lm), msg.data(), sizeof(float) * E * M);
    int64_t id1d[1] = {E};
    TensorF32* li = ctx.new_tensor<float>(1, id1d);
    std::memcpy(bind_leaf_data(ctx, li), tgt.data(), sizeof(float) * E);
    TensorF32* out = make_node(ctx, OP_SCATTER_ADD, {M, N}, lm, li);
    out->op_params[0] = (float)N;

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CPUBackend cpu(1);
    CUDABackend cuda(0);
    BackendScheduler sched;
    sched.add_backend(&cpu);
    sched.add_backend(&cuda);

    sched.split_graph(g);
    EXPECT_EQ(sched.tensor_backend_id(out, -1), 0) << "OP_SCATTER_ADD 应分配到 CUDA";

    ASSERT_TRUE(sched.alloc_splits());
    ASSERT_EQ(sched.graph_compute(), Status::SUCCESS);
    cuda.synchronize();

    std::vector<float> got = read_tensor_cpu(out);
    std::vector<float> exp = scatter_add_reference(msg, tgt, N, M, E);
    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

// 同图第二次 compute（模拟训练迭代）：gallocr needs_realloc 复用路径（不重建 buffer），
// GPU scatter + 复用下 msg/tgt_idx 索引 leaf 仍保活不别名，结果一致。
TEST(CudaBackendTest, ScatterAddSchedulerReuseIter) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int N = 4, M = 3, E = 5;
    std::vector<float> msg(E * M);
    for (int i = 0; i < E * M; ++i) msg[i] = (float)(i % 5) * 0.25f + 0.5f;
    const std::vector<float> tgt = {1.f, 2.f, 1.f, 3.f, 0.f};

    PPMLContext& ctx = context();
    int64_t md[2] = {M, E};
    TensorF32* lm = ctx.new_tensor<float>(2, md);
    std::memcpy(bind_leaf_data(ctx, lm), msg.data(), sizeof(float) * E * M);
    int64_t id1d[1] = {E};
    TensorF32* li = ctx.new_tensor<float>(1, id1d);
    std::memcpy(bind_leaf_data(ctx, li), tgt.data(), sizeof(float) * E);
    TensorF32* out = make_node(ctx, OP_SCATTER_ADD, {M, N}, lm, li);
    out->op_params[0] = (float)N;

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CPUBackend cpu(1);
    CUDABackend cuda(0);
    BackendScheduler sched;
    sched.add_backend(&cpu);
    sched.add_backend(&cuda);

    sched.split_graph(g);
    ASSERT_TRUE(sched.alloc_splits());
    ASSERT_EQ(sched.graph_compute(), Status::SUCCESS);
    cuda.synchronize();

    // 第 1 轮结果
    std::vector<float> got1 = read_tensor_cpu(out);
    // 第 2 轮：同图再 compute（scheduler reserve_graph_memory 走 gallocr can_reuse 复用）
    ASSERT_EQ(sched.graph_compute(), Status::SUCCESS);
    cuda.synchronize();
    std::vector<float> got2 = read_tensor_cpu(out);

    std::vector<float> exp = scatter_add_reference(msg, tgt, N, M, E);
    ASSERT_EQ(got1.size(), exp.size());
    ASSERT_EQ(got2.size(), exp.size());
    for (size_t i = 0; i < exp.size(); ++i) {
        EXPECT_NEAR(got1[i], exp[i], 1e-4f) << "iter1 i=" << i;
        EXPECT_NEAR(got2[i], exp[i], 1e-4f) << "iter2 i=" << i;
    }
}

// PPML_CUDA_NO_SCATTER=1：supports_op 判 false → scatter 回落 CPU（backends_[1]），数值仍正确。
TEST(CudaBackendTest, ScatterAddNoScatterFallsBack) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int N = 4, M = 3, E = 5;
    std::vector<float> msg(E * M);
    for (int i = 0; i < E * M; ++i) msg[i] = (float)(i % 5) * 0.25f + 0.5f;
    const std::vector<float> tgt = {1.f, 2.f, 1.f, 3.f, 0.f};

    PPMLContext& ctx = context();
    int64_t md[2] = {M, E};
    TensorF32* lm = ctx.new_tensor<float>(2, md);
    std::memcpy(bind_leaf_data(ctx, lm), msg.data(), sizeof(float) * E * M);
    int64_t id1d[1] = {E};
    TensorF32* li = ctx.new_tensor<float>(1, id1d);
    std::memcpy(bind_leaf_data(ctx, li), tgt.data(), sizeof(float) * E);
    TensorF32* out = make_node(ctx, OP_SCATTER_ADD, {M, N}, lm, li);
    out->op_params[0] = (float)N;

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CPUBackend cpu(1);
    CUDABackend cuda(0);
    BackendScheduler sched;
    sched.add_backend(&cpu);
    sched.add_backend(&cuda);

    setenv("PPML_CUDA_NO_SCATTER", "1", 1);
    sched.split_graph(g);
    unsetenv("PPML_CUDA_NO_SCATTER");
    EXPECT_EQ(sched.tensor_backend_id(out, -1), 1) << "no-scatter 开启 → 应回落 CPU";

    ASSERT_TRUE(sched.alloc_splits());
    ASSERT_EQ(sched.graph_compute(), Status::SUCCESS);
    cuda.synchronize();

    std::vector<float> got = read_tensor_cpu(out);
    std::vector<float> exp = scatter_add_reference(msg, tgt, N, M, E);
    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], exp[i], 1e-4f) << "i=" << i;
    }
}

// ============================================================
// OP_DUP / OP_CPY / OP_CONT：整块拷贝（buffer-aware，D2D）
//   src 为 bind_data 的 cudaMalloc 指针（buffer_=null），dst 由 gallocr 分配 →
//   经 cudaPointerGetAttributes 识别 device 指针走 cudaMemcpy D2D。
// ============================================================
std::vector<float> run_cuda_dup(const std::vector<float>& x, tensor_op op) {
    PPMLContext& ctx = context();
    const int64_t N = (int64_t)x.size();
    TensorF32* la = make_device_leaf(ctx, {N}, x);
    TensorF32* out = make_node(ctx, op, {N}, la);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    EXPECT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(la->data());
    return got;
}

TEST(CudaBackendTest, DupCpyCont) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N = 4096;   // 跨 block 拷贝
    std::vector<float> x(N);
    for (int i = 0; i < N; ++i) x[i] = (float)(i % 7) * 0.5f - 2.f;

    // OP_DUP
    {
        std::vector<float> got = run_cuda_dup(x, OP_DUP);
        ASSERT_EQ(got.size(), x.size());
        for (size_t i = 0; i < got.size(); ++i) EXPECT_FLOAT_EQ(got[i], x[i]) << "dup i=" << i;
    }
    // OP_CPY
    {
        std::vector<float> got = run_cuda_dup(x, OP_CPY);
        for (size_t i = 0; i < got.size(); ++i) EXPECT_FLOAT_EQ(got[i], x[i]) << "cpy i=" << i;
    }
    // OP_CONT
    {
        std::vector<float> got = run_cuda_dup(x, OP_CONT);
        for (size_t i = 0; i < got.size(); ++i) EXPECT_FLOAT_EQ(got[i], x[i]) << "cont i=" << i;
    }
}

// ============================================================
// OP_FAPE 前向：FAPE 结构损失 → 标量
//   5 输入：pred/true coords [3,N_atoms]、frame_indices [3,N_frames]、
//   frames_mask [1,N_frames]、positions_mask [1,N_atoms]。
//   op_params: [0]=d_clamp [2]=epsilon [4]=length_scale。
// ============================================================

// CPU 参考：复刻 compute_forward_fape 语义（双 T_inv + 距离归约）
float fape_reference(const std::vector<float>& pred, const std::vector<float>& truth,
                     const std::vector<float>& fidx, const std::vector<float>& fmask,
                     const std::vector<float>& pmask, int64_t N_atoms, int64_t N_frames,
                     float d_clamp, float epsilon, float length_scale) {
    std::vector<float> Tinv_pred(N_frames * 12), Tinv_true(N_frames * 12);
    for (int64_t n = 0; n < N_frames; n++) {
        for (int k = 0; k < 12; k++) { Tinv_pred[n*12+k]=0; Tinv_true[n*12+k]=0; }
        Tinv_pred[n*12+0]=Tinv_pred[n*12+4]=Tinv_pred[n*12+8]=1;
        Tinv_true[n*12+0]=Tinv_true[n*12+4]=Tinv_true[n*12+8]=1;
        if (fmask[(size_t)n] == 0.f) continue;
        auto build = [&](const std::vector<float>& coords, std::vector<float>& T) {
            const int a=(int)fidx[(size_t)n*3+0], b=(int)fidx[(size_t)n*3+1], c=(int)fidx[(size_t)n*3+2];
            float Ax=coords[(size_t)a*3],Ay=coords[(size_t)a*3+1],Az=coords[(size_t)a*3+2];
            float Bx=coords[(size_t)b*3],By=coords[(size_t)b*3+1],Bz=coords[(size_t)b*3+2];
            float Cx=coords[(size_t)c*3],Cy=coords[(size_t)c*3+1],Cz=coords[(size_t)c*3+2];
            float v1x=Bx-Ax,v1y=By-Ay,v1z=Bz-Az; float v2x=Cx-Ax,v2y=Cy-Ay,v2z=Cz-Az;
            float n1=sqrtf(v1x*v1x+v1y*v1y+v1z*v1z+epsilon);
            float e1x=v1x/n1,e1y=v1y/n1,e1z=v1z/n1;
            float dot=v2x*e1x+v2y*e1y+v2z*e1z;
            float u2x=v2x-dot*e1x,u2y=v2y-dot*e1y,u2z=v2z-dot*e1z;
            float n2=sqrtf(u2x*u2x+u2y*u2y+u2z*u2z+epsilon);
            float e2x=u2x/n2,e2y=u2y/n2,e2z=u2z/n2;
            float e3x=e1y*e2z-e1z*e2y,e3y=e1z*e2x-e1x*e2z,e3z=e1x*e2y-e1y*e2x;
            T[(size_t)n*12+0]=e1x;T[(size_t)n*12+1]=e1y;T[(size_t)n*12+2]=e1z;
            T[(size_t)n*12+3]=e2x;T[(size_t)n*12+4]=e2y;T[(size_t)n*12+5]=e2z;
            T[(size_t)n*12+6]=e3x;T[(size_t)n*12+7]=e3y;T[(size_t)n*12+8]=e3z;
            T[(size_t)n*12+9]=-(e1x*Ax+e1y*Ay+e1z*Az);
            T[(size_t)n*12+10]=-(e2x*Ax+e2y*Ay+e2z*Az);
            T[(size_t)n*12+11]=-(e3x*Ax+e3y*Ay+e3z*Az);
        };
        build(pred, Tinv_pred); build(truth, Tinv_true);
    }
    float sum_loss=0, sum_fm=0, sum_pm=0;
    for (int64_t n=0;n<N_frames;n++) {
        float fm=fmask[(size_t)n]; if (fm==0) continue; sum_fm+=fm;
        for (int64_t j=0;j<N_atoms;j++) {
            float pm=pmask[(size_t)j]; if (pm==0) continue;
            const float* Tp=&Tinv_pred[(size_t)n*12]; const float* Tt=&Tinv_true[(size_t)n*12];
            float px=pred[(size_t)j*3],py=pred[(size_t)j*3+1],pz=pred[(size_t)j*3+2];
            float lpx=Tp[0]*px+Tp[1]*py+Tp[2]*pz+Tp[9];
            float lpy=Tp[3]*px+Tp[4]*py+Tp[5]*pz+Tp[10];
            float lpz=Tp[6]*px+Tp[7]*py+Tp[8]*pz+Tp[11];
            float tx=truth[(size_t)j*3],ty=truth[(size_t)j*3+1],tz=truth[(size_t)j*3+2];
            float ltx=Tt[0]*tx+Tt[1]*ty+Tt[2]*tz+Tt[9];
            float lty=Tt[3]*tx+Tt[4]*ty+Tt[5]*tz+Tt[10];
            float ltz=Tt[6]*tx+Tt[7]*ty+Tt[8]*tz+Tt[11];
            float dx=lpx-ltx,dy=lpy-lty,dz=lpz-ltz;
            float dist=sqrtf(dx*dx+dy*dy+dz*dz+epsilon);
            if (dist>d_clamp) dist=d_clamp;
            sum_loss+=dist*fm*pm; sum_pm+=pm;
        }
    }
    return sum_loss/(sum_fm*sum_pm+epsilon)/length_scale;
}

TEST(CudaBackendTest, FapeForward) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N_atoms = 5, N_frames = 3;
    std::vector<float> pred(15), truth(15), fidx(9), fmask(3), pmask(5);
    // pred/truth 3D 坐标（含非零位移）
    for (int i = 0; i < 15; ++i) { pred[(size_t)i] = (float)(i % 7) * 0.3f; truth[(size_t)i] = (float)(i % 5) * 0.2f + 1.f; }
    // frame 原子索引（0,1,2 / 1,2,3 / 2,3,4）
    fidx = {0.f,1.f,2.f, 1.f,2.f,3.f, 2.f,3.f,4.f};
    fmask = {1.f, 1.f, 0.f};       // 第 3 帧掩码 0 → 跳过
    pmask = {1.f, 1.f, 1.f, 0.f, 1.f};  // 第 4 原子掩码 0

    const float d_clamp = 10.f, epsilon = 1e-4f, length_scale = 10.f;

    PPMLContext& ctx = context();
    TensorF32* lp = make_device_leaf(ctx, {3, N_atoms}, pred);
    TensorF32* lt = make_device_leaf(ctx, {3, N_atoms}, truth);
    TensorF32* li = make_device_leaf(ctx, {3, N_frames}, fidx);
    TensorF32* lf = make_device_leaf(ctx, {1, N_frames}, fmask);
    TensorF32* lpm= make_device_leaf(ctx, {1, N_atoms}, pmask);
    TensorF32* out = make_node(ctx, OP_FAPE, {1}, lp, lt, li, lf, lpm);
    reinterpret_cast<float&>(out->op_params[0]) = d_clamp;
    reinterpret_cast<float&>(out->op_params[2]) = epsilon;
    reinterpret_cast<float&>(out->op_params[4]) = length_scale;

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);
    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(lp->data()); cudaFree(lt->data()); cudaFree(li->data());
    cudaFree(lf->data()); cudaFree(lpm->data());

    const float exp = fape_reference(pred, truth, fidx, fmask, pmask, N_atoms, N_frames,
                                     d_clamp, epsilon, length_scale);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_NEAR(got[0], exp, 1e-3f * (1.f + std::fabs(exp)));
}

// ============================================================
// OP_PERMUTE / OP_TRANSPOSE：通用维度重排
//   dst 第 p 维 = src 第 dims[p] 维（op_params int32 映射，行主序 dims[0]=最内）。
//   OP_TRANSPOSE 构造器写 dims=[1,0,2,3]；OP_PERMUTE 任意交换。
// ============================================================

// CPU 参考（镜像 kernel_permute 语义）
std::vector<float> permute_reference(
    const std::vector<float>& src, const std::vector<int64_t>& sd,
    const std::vector<int64_t>& dd, const std::vector<int32_t>& dims) {
    const int ndim = (int)dd.size();
    const int64_t total = dd[0] * dd[1] * dd[2] * dd[3];
    std::vector<float> dst(total);
    for (int64_t idx = 0; idx < total; ++idx) {
        int64_t t = idx;
        int64_t jv[4] = {0,0,0,0};
        jv[0] = t % dd[0]; t /= dd[0];
        jv[1] = t % dd[1]; t /= dd[1];
        jv[2] = t % dd[2]; t /= dd[2];
        jv[3] = t;
        int64_t i[4] = {0,0,0,0};
        for (int p = 0; p < ndim; p++) i[dims[(size_t)p]] = jv[p];
        const int64_t off = ((i[3]*sd[2] + i[2])*sd[1] + i[1])*sd[0] + i[0];
        dst[(size_t)idx] = src[(size_t)off];
    }
    return dst;
}

std::vector<float> run_cuda_permute(
    const std::vector<float>& src, const std::vector<int64_t>& sd,
    const std::vector<int64_t>& dd, const std::vector<int32_t>& dims) {
    PPMLContext& ctx = context();
    TensorF32* la = make_device_leaf(ctx, sd, src);
    TensorF32* out = make_node(ctx, OP_PERMUTE, dd, la);
    for (size_t i = 0; i < dims.size(); ++i) out->op_params[i] = dims[i];

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);
    CUDABackend backend(0);
    EXPECT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(la->data());
    return got;
}

TEST(CudaBackendTest, PermuteTranspose2D) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // transpose [3,5]→[5,3]（dims=[1,0,2,3]）
    const std::vector<int64_t> sd = {3, 5};
    const std::vector<int64_t> dd = {5, 3};
    const std::vector<int32_t> dims = {1, 0, 2, 3};
    std::vector<float> src(15);
    for (int i = 0; i < 15; ++i) src[(size_t)i] = (float)(i % 7) * 0.5f;

    std::vector<float> got = run_cuda_permute(src, sd, dd, dims);
    std::vector<float> exp = permute_reference(src, {3,5,1,1}, {5,3,1,1}, dims);
    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_FLOAT_EQ(got[i], exp[i]) << "i=" << i;
    }
}

TEST(CudaBackendTest, Permute4D) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // permute [2,3,4,1]→[4,2,3,1]（dims=[2,0,1,3]）
    const std::vector<int64_t> sd = {2, 3, 4, 1};
    const std::vector<int64_t> dd = {4, 2, 3, 1};
    const std::vector<int32_t> dims = {2, 0, 1, 3};
    std::vector<float> src(24);
    for (int i = 0; i < 24; ++i) src[(size_t)i] = (float)(i % 5) * 0.25f + 1.f;

    std::vector<float> got = run_cuda_permute(src, sd, dd, dims);
    std::vector<float> exp = permute_reference(src, sd, dd, dims);
    ASSERT_EQ(got.size(), exp.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_FLOAT_EQ(got[i], exp[i]) << "i=" << i;
    }
}

// ============================================================
// OP_TRANSPOSE 2D 专用 tile kernel（XOR swizzle）
//   src {N,M}（dims[0]=N 行长度, dims[1]=M 行数）→ dst {M,N}
// ============================================================
std::vector<float> run_cuda_transpose2d(
    const std::vector<float>& src, int64_t M, int64_t N) {
    PPMLContext& ctx = context();
    TensorF32* la = make_device_leaf(ctx, {N, M}, src);       // src 行长度 N、行数 M
    TensorF32* out = make_node(ctx, OP_TRANSPOSE, {M, N}, la); // dst {M,N}
    out->op_params[0] = 1; out->op_params[1] = 0;              // 交换最后两维（2D）
    out->op_params[2] = 2; out->op_params[3] = 3;

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);
    CUDABackend backend(0);
    EXPECT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(la->data());
    return got;
}

TEST(CudaBackendTest, TransposeTile2D) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // 非方阵 + 奇数行（M=7,N=5）：tile 32 被越界 padding 覆盖
    const int64_t M = 7, N = 5;
    std::vector<float> src((size_t)M * N);
    for (int i = 0; i < M * N; ++i) src[(size_t)i] = (float)(i % 9) * 0.5f - 1.f;

    std::vector<float> got = run_cuda_transpose2d(src, M, N);
    // CPU 参考：B[j*M+i] = A[i*N+j]
    std::vector<float> exp((size_t)M * N);
    for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j) exp[(size_t)(j*M + i)] = src[(size_t)(i*N + j)];

    ASSERT_EQ(got.size(), exp.size());
    for (size_t k = 0; k < got.size(); ++k) {
        EXPECT_FLOAT_EQ(got[k], exp[k]) << "k=" << k;
    }
}

TEST(CudaBackendTest, TransposeTileLarge) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    // 大于 tile（64×48 → 2×2 blocks），覆盖多 block 边界
    const int64_t M = 64, N = 48;
    std::vector<float> src((size_t)M * N);
    for (int i = 0; i < M * N; ++i) src[(size_t)i] = (float)(i % 13) * 0.25f;

    std::vector<float> got = run_cuda_transpose2d(src, M, N);
    std::vector<float> exp((size_t)M * N);
    for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j) exp[(size_t)(j*M + i)] = src[(size_t)(i*N + j)];

    ASSERT_EQ(got.size(), exp.size());
    for (size_t k = 0; k < got.size(); ++k) {
        EXPECT_FLOAT_EQ(got[k], exp[k]) << "k=" << k;
    }
}

// ============================================================
// OP_SQR / OP_SQRT / OP_LOG：独立逐元素 op（非 unary）
//   dst = src^2 / sqrt(src) / log(src)
// ============================================================
std::vector<float> run_cuda_unary_elem(tensor_op op, const std::vector<float>& x,
                                       int uop) {
    PPMLContext& ctx = context();
    const int64_t N = (int64_t)x.size();
    TensorF32* la = make_device_leaf(ctx, {N}, x);
    TensorF32* out = make_node(ctx, op, {N}, la);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);
    CUDABackend backend(0);
    EXPECT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);
    cudaFree(la->data());
    return got;
}

TEST(CudaBackendTest, SqrSqrtLog) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N = 4096;
    std::vector<float> x(N);
    for (int i = 0; i < N; ++i) x[(size_t)i] = (float)(i % 17) * 0.5f + 0.1f;   // 全正，log/sqrt 合法

    // SQR: dst = x^2
    {
        std::vector<float> got = run_cuda_unary_elem(OP_SQR, x, 16);
        ASSERT_EQ(got.size(), x.size());
        for (size_t i = 0; i < got.size(); ++i)
            EXPECT_NEAR(got[i], x[i]*x[i], 1e-4f * (1.f + x[i]*x[i])) << "sqr i=" << i;
    }
    // SQRT: dst = sqrt(x)
    {
        std::vector<float> got = run_cuda_unary_elem(OP_SQRT, x, 15);
        for (size_t i = 0; i < got.size(); ++i)
            EXPECT_NEAR(got[i], std::sqrt(x[i]), 1e-4f) << "sqrt i=" << i;
    }
    // LOG: dst = log(x)
    {
        std::vector<float> got = run_cuda_unary_elem(OP_LOG, x, 14);
        for (size_t i = 0; i < got.size(); ++i)
            EXPECT_NEAR(got[i], std::log(x[i]), 1e-4f) << "log i=" << i;
    }
}

// ============================================================
// Tensor Core / TF32 GEMM：硬件检测 + 正确性 + 性能
//   参考用户 CUTLASS 模板（SM80_16x8x8_*_TN），自包含 mma.sync PTX 实现。
// ============================================================
namespace ppml {
extern bool cutlass_hw_supported(int* major, int* minor);
extern int  tf32_gemm_bench_cuda(const float* A, const float* B, float* C,
                                 int M, int K, int N, float* ms_out);
}

TEST(CudaBackendTest, TensorCoreDetect) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    int major = 0, minor = 0;
    const bool ok = ppml::cutlass_hw_supported(&major, &minor);
    printf("[tc] compute capability %d.%d -> %s\n", major, minor,
           ok ? "TF32/F16 MMA supported (CUTLASS sm80+ path OK)"
              : "no MMA (pre-Ampere, SIMT only)");
    // 本机 sm_86 → 应支持；但测试不硬编码 arch，仅打印（若 GPU 是 pre-80 也算通过）
}

TEST(CudaBackendTest, TF32GemmCorrectness) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    int major = 0, minor = 0;
    if (!ppml::cutlass_hw_supported(&major, &minor)) {
        GTEST_SKIP() << "非 sm_80+，无 TF32 MMA";
    }

    const int M = 32, K = 32, N = 32;   // 2×2 blocks（每 block 16×64）→ 64 列
    std::vector<float> A((size_t)M * K), B((size_t)N * K), C((size_t)M * N, 0.f);
    for (int i = 0; i < M * K; ++i) A[(size_t)i] = (float)(i % 7) * 0.25f - 0.5f;
    for (int i = 0; i < N * K; ++i) B[(size_t)i] = (float)(i % 11) * 0.125f + 0.1f;

    float ms = 0.f;
    ASSERT_EQ(ppml::tf32_gemm_bench_cuda(A.data(), B.data(), C.data(), M, K, N, &ms), 0);

    // CPU 参考（tf32 截断近似；1e-2 相对容差容忍 tf32 精度）
    std::vector<float> ref((size_t)M * N, 0.f);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k)
                ref[(size_t)(i*N + j)] += A[(size_t)(i*K + k)] * B[(size_t)(j*K + k)];

    double scale = 0.0;
    for (int i = 0; i < M * N; ++i) scale += std::fabs(ref[(size_t)i]);
    scale = scale / (M * N);
    const float tol = 0.02f * (float)(scale + 1e-3f);
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[(size_t)i], ref[(size_t)i], tol) << "i=" << i;
    }
}

TEST(CudaBackendTest, TF32GemmPerf) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    int major = 0, minor = 0;
    if (!ppml::cutlass_hw_supported(&major, &minor)) {
        GTEST_SKIP() << "非 sm_80+，无 TF32 MMA";
    }

    // 训练规模近似：256×256×256
    const int M = 256, K = 256, N = 256;
    std::vector<float> A((size_t)M * K), B((size_t)N * K), C((size_t)M * N, 0.f);
    for (int i = 0; i < M * K; ++i) A[(size_t)i] = (float)(i % 5) * 0.1f;
    for (int i = 0; i < N * K; ++i) B[(size_t)i] = (float)(i % 9) * 0.1f;

    float ms = 0.f;
    ASSERT_EQ(ppml::tf32_gemm_bench_cuda(A.data(), B.data(), C.data(), M, K, N, &ms), 0);
    const double flops = 2.0 * M * K * N;
    printf("[tf32-gemm] %dx%dx%d: %.4f ms (%.1f GFLOPS)\n",
           M, K, N, (double)ms, flops / (ms * 1e-3) / 1e9);
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
// OP_RMS_NORM：沿 dims[0]（最内维）归一化
//   dst = x / sqrt(mean(x^2) + eps)，无 affine/bias
//   CUDA kernel: 每 block 一行, warp 蝴蝶全归约 (__shfl_xor_sync)
//   C 须为 32 倍数（CUDA kernel 前置），非 32 倍数由 dispatch 内联 CPU 兜底
// ============================================================
TEST(CudaBackendTest, RMSNorm) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int C = 128;   // 32 倍数
    const int R = 6;
    std::vector<float> x(C * R);
    for (int i = 0; i < C*R; ++i) x[i] = (float)(i % 13) * 0.3f - 1.5f;
    const float eps = 1e-5f;

    PPMLContext& ctx = context();
    TensorF32* lin = make_device_leaf(ctx, {C, R}, x);
    TensorF32* out = make_node(ctx, OP_RMS_NORM, {C, R}, lin);
    // op_params[0] = eps (float bit-cast，与 CPU kernel_rms_norm 一致)
    reinterpret_cast<float&>(out->op_params[0]) = eps;

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);

    for (int r = 0; r < R; ++r) {
        float ss = 0.f;
        for (int c = 0; c < C; ++c) ss += x[r*C+c] * x[r*C+c];
        const float inv = 1.f / std::sqrt(ss / C + eps);
        for (int c = 0; c < C; ++c) {
            EXPECT_NEAR(got[r*C + c], x[r*C+c] * inv, 1e-4f) << "r=" << r << " c=" << c;
        }
    }
    cudaFree(lin->data());
}

// 非 32 倍数 C：supports_op 返回 false（CUDA kernel 前置不满足）。
// 注意: 这里直接调 CUDABackend::graph_compute，不经过 scheduler 跨后端回落，
//      故输出保持 0 是预期（回落到 CPU 是混合调度层的事，见 SchedulerFallsBackToCpu）。
//      本测试只验证 supports_op 判 false，不验证数值。
TEST(CudaBackendTest, RMSNormNonMultiple32Unsupported) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int C = 100;   // 非 32 倍数
    const int R = 4;
    std::vector<float> x(C * R);
    for (int i = 0; i < C*R; ++i) x[i] = (float)(i % 7) * 0.5f - 2.f;

    PPMLContext& ctx = context();
    TensorF32* lin = make_device_leaf(ctx, {C, R}, x);
    TensorF32* out = make_node(ctx, OP_RMS_NORM, {C, R}, lin);
    reinterpret_cast<float&>(out->op_params[0]) = 1e-5f;

    CUDABackend backend(0);
    // supports_op 必须返回 false（CUDA 不支持非 32 倍数 C）
    EXPECT_FALSE(backend.supports_op(out));
    cudaFree(lin->data());
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
// OP_GET_ROWS (embedding 查表) CUDA kernel
//   W (N,M) 行主序, idx(K,) float-encoded 行索引 → dst(K,M)
//   grid: blockIdx.x→行 k, gridDim.y 覆盖 M 维
//      与 ggml 的 dims[0]=最内不同。越界钳制: i<0→0, i>=N→N-1 (对齐 CPU)
// ============================================================
TEST(CudaBackendTest, GetRows) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N = 16;   // 词表大小 (行数)
    const int64_t M = 8;    // 嵌入维 (行内长)
    const int64_t K = 7;    // 索引数
    std::vector<float> W(N * M);
    for (int i = 0; i < N*M; ++i) W[i] = (float)(i % 11) * 0.5f - 2.f;

    // idx 含正常、越界(<0 钳 0)、越界(>=N 钳 N-1)
    std::vector<float> idx = {3.f, 0.f, 15.f, -5.f, 99.f, 8.f, 2.f};

    PPMLContext& ctx = context();
    // W: dims={N, M}; idx: dims={K} (float-encoded); dst: dims={M, K}
    TensorF32* w_t = make_device_leaf(ctx, {N, M}, W);
    TensorF32* i_t = make_device_leaf(ctx, {K}, idx);
    TensorF32* out = make_node(ctx, OP_GET_ROWS, {M, K}, w_t, i_t);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);

    // CPU 参考 (与 CPUKernels.cpp:1928 同逻辑)
    for (int64_t k = 0; k < K; ++k) {
        int64_t i = (int64_t)idx[k];
        if (i < 0)    i = 0;
        if (i >= N)   i = N - 1;
        for (int64_t m = 0; m < M; ++m) {
            EXPECT_NEAR(got[k*M + m], W[i*M + m], 1e-5f)
                << "k=" << k << " m=" << m << " idx=" << idx[k];
        }
    }
    cudaFree(w_t->data()); cudaFree(i_t->data());
}

// 大数据：M 非 256 倍数 → 多 block y 覆盖 + grid-stride 兜底
TEST(CudaBackendTest, GetRowsLargeM) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N = 64;
    const int64_t M = 1000;   // 非 256 倍数, need_y=4 个 y-block
    const int64_t K = 128;
    std::vector<float> W(N * M);
    for (int i = 0; i < N*M; ++i) W[i] = (float)(i % 31) * 0.25f + 0.5f;

    std::vector<float> idx(K);
    for (int64_t k = 0; k < K; ++k) idx[k] = (float)((k * 7) % N);   // 0..N-1 循环

    PPMLContext& ctx = context();
    TensorF32* w_t = make_device_leaf(ctx, {N, M}, W);
    TensorF32* i_t = make_device_leaf(ctx, {K}, idx);
    TensorF32* out = make_node(ctx, OP_GET_ROWS, {M, K}, w_t, i_t);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);

    for (int64_t k = 0; k < K; ++k) {
        const int64_t i = (int64_t)idx[k];
        for (int64_t m = 0; m < M; ++m) {
            EXPECT_NEAR(got[k*M + m], W[i*M + m], 1e-5f)
                << "k=" << k << " m=" << m;
        }
    }
    cudaFree(w_t->data()); cudaFree(i_t->data());
}

// ============================================================
// OP_GET_ROWS_BACK (embedding 查表反向, 方案 A: atomicAdd 散点)
//   dy(K,M) 梯度, idx(K,) 行索引 → dW(N,M) 清零+散点累加
//   同一行被多 k 引用时累加; 越界 (i<0||i>=N) 丢弃 (对齐 CPU kernel_get_rows_back)
// ============================================================
TEST(CudaBackendTest, GetRowsBack) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N = 10;   // 权重表行数
    const int64_t M = 6;    // 每行长度
    const int64_t K = 8;    // grad 行数
    std::vector<float> dy(K * M);
    for (int i = 0; i < K*M; ++i) dy[i] = (float)(i % 7) * 0.5f - 1.f;
    // idx: 含重复行 (0 出现 2 次, 累加验证) + 越界 (-1, 99 丢弃)
    std::vector<float> idx = {3.f, 0.f, 5.f, 0.f, -1.f, 99.f, 2.f, 8.f};
    std::vector<float> W(N * M);   // 仅用于取形状 (N, M)

    PPMLContext& ctx = context();
    TensorF32* dy_t = make_device_leaf(ctx, {K, M}, dy);
    TensorF32* idx_t = make_device_leaf(ctx, {K}, idx);
    TensorF32* w_t = make_device_leaf(ctx, {N, M}, W);
    TensorF32* out = make_node(ctx, OP_GET_ROWS_BACK, {N, M}, dy_t, idx_t, w_t);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);

    // CPU 参考 (与 CPUKernels.cpp:1961 同逻辑)
    std::vector<float> ref(N * M, 0.f);
    for (int64_t k = 0; k < K; ++k) {
        int64_t i = (int64_t)idx[k];
        if (i < 0 || i >= N) continue;   // 越界丢弃
        for (int64_t d = 0; d < M; ++d) ref[i*M + d] += dy[k*M + d];
    }
    for (int64_t i = 0; i < N*M; ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-5f) << "i=" << i;
    }
    cudaFree(dy_t->data()); cudaFree(idx_t->data()); cudaFree(w_t->data());
}

// 大数据: M 非 256 倍数 → 多 y-block + atomicAdd 多线程竞争同一行
TEST(CudaBackendTest, GetRowsBackLarge) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int64_t N = 32;
    const int64_t M = 1000;   // 非 256 倍数
    const int64_t K = 64;
    std::vector<float> dy(K * M);
    for (int i = 0; i < K*M; ++i) dy[i] = (float)(i % 13) * 0.2f + 0.1f;
    std::vector<float> idx(K);
    for (int64_t k = 0; k < K; ++k) idx[k] = (float)((k * 3) % N);   // 高度冲突 (32 行 64 索引)
    std::vector<float> W(N * M, 0.f);

    PPMLContext& ctx = context();
    TensorF32* dy_t = make_device_leaf(ctx, {K, M}, dy);
    TensorF32* idx_t = make_device_leaf(ctx, {K}, idx);
    TensorF32* w_t = make_device_leaf(ctx, {N, M}, W);
    TensorF32* out = make_node(ctx, OP_GET_ROWS_BACK, {N, M}, dy_t, idx_t, w_t);

    ComputeGraph* g = ComputeGraph::new_graph(&ctx);
    g->build_forward_expand(out);

    CUDABackend backend(0);
    ASSERT_EQ(backend.graph_compute(g), Status::SUCCESS);
    backend.synchronize();

    std::vector<float> got;
    copy_to_host(out, got);

    std::vector<float> ref(N * M, 0.f);
    for (int64_t k = 0; k < K; ++k) {
        const int64_t i = (int64_t)idx[k];
        for (int64_t d = 0; d < M; ++d) ref[i*M + d] += dy[k*M + d];
    }
    for (int64_t i = 0; i < N*M; ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-5f) << "i=" << i;
    }
    cudaFree(dy_t->data()); cudaFree(idx_t->data()); cudaFree(w_t->data());
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

// ============================================================
// softmax 原始 kernel 入口（定义于 src/cuda/CUDAKernels.cu）
//   2026-09-11：softmax_cuda（生产入口）已切到 v2；v1 保留对拍；
//   反向 kernel 的 Σ(y·g) 归约依赖 blockReduceSumShuffle（同批修正了该 helper 的广播缺陷）。
// ============================================================
namespace ppml {
void softmax_v1_cuda(float * input, float * output, int M, int N, int block_size);
void softmax_v2_cuda(float * input, float * output, int M, int N, int block_size);
void softmax_backward_cuda(const float * grad, const float * output, float * dst,
                           int M, int N, int block_size, float scale);
}

// ---- v1（smem 树形规约）vs v2（warp shuffle 两级规约）vs CPU 参考 ----
//   N=1000：>blockDim(256) 且非 32 倍数 → 覆盖多轮 stride 循环 + warp 尾部；
//   其中第 2 行整体接近下溢边界（-80），第 3 行含 +inf 级大值 → 覆盖 max 减稳。
TEST(CudaBackendTest, SoftmaxV1V2MatchCpuReference) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int M = 7, N = 1000;
    std::vector<float> x(M * N);
    for (int i = 0; i < M * N; ++i)
        x[i] = 0.01f * (float)((i * 37) % 211) - 1.05f;
    for (int n = 0; n < N; ++n) {
        x[1 * N + n] = -80.0f + 0.001f * (float)n;   // 整行极小值（exp 下溢邻域）
        x[2 * N + n] = 20.0f * (float)(n % 2);       // 交替 0 / 20（max 减稳 + 权重悬殊）
    }

    float *dx = nullptr, *d1 = nullptr, *d2 = nullptr;
    ASSERT_EQ(cudaMalloc(&dx, sizeof(float) * M * N), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d1, sizeof(float) * M * N), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d2, sizeof(float) * M * N), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dx, x.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice), cudaSuccess);

    softmax_v1_cuda(dx, d1, M, N, 256);
    softmax_v2_cuda(dx, d2, M, N, 256);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> h1(M * N), h2(M * N);
    ASSERT_EQ(cudaMemcpy(h1.data(), d1, sizeof(float) * M * N, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h2.data(), d2, sizeof(float) * M * N, cudaMemcpyDeviceToHost), cudaSuccess);

    double worst_v1 = 0.0, worst_v2 = 0.0, worst_v1v2 = 0.0;
    for (int m = 0; m < M; ++m) {
        const float* xr = x.data() + m * N;
        float mx = -INFINITY;
        for (int n = 0; n < N; ++n) mx = std::max(mx, xr[n]);
        double s = 0.0;
        for (int n = 0; n < N; ++n) s += std::exp((double)xr[n] - mx);
        double row_sum1 = 0.0, row_sum2 = 0.0;
        for (int n = 0; n < N; ++n) {
            const double ref = std::exp((double)xr[n] - mx) / s;
            worst_v1    = std::max(worst_v1, std::fabs((double)h1[m * N + n] - ref));
            worst_v2    = std::max(worst_v2, std::fabs((double)h2[m * N + n] - ref));
            worst_v1v2  = std::max(worst_v1v2, std::fabs((double)h1[m * N + n] - (double)h2[m * N + n]));
            row_sum1 += (double)h1[m * N + n];
            row_sum2 += (double)h2[m * N + n];
            EXPECT_NEAR((double)h2[m * N + n], ref, 1e-5) << "v2 m=" << m << " n=" << n;
        }
        EXPECT_NEAR(row_sum1, 1.0, 1e-4) << "v1 第 " << m << " 行和应为 1";
        EXPECT_NEAR(row_sum2, 1.0, 1e-4) << "v2 第 " << m << " 行和应为 1";
    }
    std::cerr << "[SoftmaxV1V2] max|v1-ref|=" << worst_v1
              << " max|v2-ref|=" << worst_v2
              << " max|v1-v2|=" << worst_v1v2 << std::endl;
    EXPECT_LT(worst_v1v2, 1e-6) << "v1/v2 应逐元素一致（归约顺序差异内的舍入）";

    cudaFree(dx); cudaFree(d1); cudaFree(d2);
}

// ---- softmax 反向 kernel：dst[i] = scale * y[i] * (g[i] - Σ_n y[n]·g[n]) ----
//   重点覆盖 blockReduceSumShuffle 的全块广播（原实现只有 tid==0 拿到正确 Σ）。
TEST(CudaBackendTest, SoftmaxBackwardMatchesReference) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA 不可用"; }

    const int M = 5, N = 300;   // N=300 > 256 → 每线程 2 个元素；跨 8 个 warp
    std::vector<float> y(M * N), g(M * N);
    for (int m = 0; m < M; ++m) {
        double s = 0.0;
        std::vector<double> e(N);
        for (int n = 0; n < N; ++n) {
            e[n] = std::exp(0.7 * std::sin(0.11 * (double)(m * N + n)) + 0.3 * (double)(n % 5));
            s += e[n];
        }
        for (int n = 0; n < N; ++n) y[m * N + n] = (float)(e[n] / s);
    }
    for (int i = 0; i < M * N; ++i) g[i] = 0.5f * std::cos(0.07f * (float)i) + 0.1f;

    float *dy = nullptr, *dg = nullptr, *dd = nullptr;
    ASSERT_EQ(cudaMalloc(&dy, sizeof(float) * M * N), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dg, sizeof(float) * M * N), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dd, sizeof(float) * M * N), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dy, y.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dg, g.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice), cudaSuccess);

    const float scale = 1.0f;
    softmax_backward_cuda(dg, dy, dd, M, N, 256, scale);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> hd(M * N);
    ASSERT_EQ(cudaMemcpy(hd.data(), dd, sizeof(float) * M * N, cudaMemcpyDeviceToHost), cudaSuccess);

    double worst = 0.0;
    for (int m = 0; m < M; ++m) {
        double dot = 0.0;
        for (int n = 0; n < N; ++n) dot += (double)y[m * N + n] * (double)g[m * N + n];
        for (int n = 0; n < N; ++n) {
            const double ref = (double)scale * (double)y[m * N + n] * ((double)g[m * N + n] - dot);
            worst = std::max(worst, std::fabs((double)hd[m * N + n] - ref));
            EXPECT_NEAR((double)hd[m * N + n], ref, 1e-5) << "m=" << m << " n=" << n;
        }
    }
    std::cerr << "[SoftmaxBackward] max|kernel-ref|=" << worst << std::endl;

    cudaFree(dy); cudaFree(dg); cudaFree(dd);
}

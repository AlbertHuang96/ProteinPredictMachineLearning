// ============================================================
// remote_probe —— 双机 RemoteBackend 的最小验证工具（2026-09-13）
//
// 三个测试（本机两个进程即可先验通，再上第二台笔记本）：
//   buffer    : RemoteBuffer 往返（set_tensor 上传 / fetch 拉回）= 调度器实际走的搬运路径
//               同时给出 4MB 双向的延迟与折合带宽 —— 对应 PLAN_2NODE.md 阶段 0 的验收项
//   allreduce : 数据并行 Allreduce（客户端 {1,2,3} + 服务端 {10,20,30} ⇒ 双方 {11,22,33}）
//   compute   : 远端执行子图（c = mul_mat(a,b)）→ 按需 fetch → 与本地参考比对
//
// 用法：
//   服务端: ./ppml_remote_probe server --port 2244 [--dp 10,20,30]
//   客户端: ./ppml_remote_probe client --host 127.0.0.1 --port 2244 [--test all|buffer|allreduce|compute]
// ============================================================
#include "ppml/Backend.h"
#include "ppml/ComputeGraph.h"
#include "ppml/Context.h"
#include "ppml/RemoteBackend.h"
#include "ppml/RemoteProtocol.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace ppml;

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(sep, p);
        if (q == std::string::npos) q = s.size();
        if (q > p) out.push_back(s.substr(p, q - p));
        p = q + 1;
    }
    return out;
}

// ---------------- 服务端 ----------------
static int run_server(int port, const std::string& dp_list) {
    RemoteServer srv;
    if (!srv.listen(port)) {
        std::fprintf(stderr, "[probe] 服务端监听 %d 失败\n", port);
        return 1;
    }
    std::fprintf(stderr, "[probe] 服务端监听 :%d，等待客户端…\n", port);
    std::vector<float> dp;
    if (!dp_list.empty()) {
        for (auto& t : split(dp_list, ',')) dp.push_back(std::strtof(t.c_str(), nullptr));
        srv.set_dp_buffer(dp.data(), dp.size());
        std::fprintf(stderr, "[probe] dp_local = {");
        for (float v : dp) std::fprintf(stderr, "%.1f ", v);
        std::fprintf(stderr, "}\n");
    }
    bool ok = srv.serve_forever(/*world_size=*/2);
    std::fprintf(stderr, "[probe] 服务端结束：ok=%d tensors=%zu arena=%.2f MB dp_result={",
                 (int)ok, srv.n_tensors(), (double)srv.arena_bytes() / (1024.0 * 1024.0));
    for (float v : srv.dp_buffer()) std::fprintf(stderr, "%.0f ", v);
    std::fprintf(stderr, "}\n");
    return ok ? 0 : 1;
}

// ---------------- 客户端测试 ----------------
// 1) RemoteBuffer 往返（真实搬运路径）
static int test_buffer_roundtrip(RemoteClient& cli) {
    PPMLContext* ctx = &context();
    const int64_t ne[1] = {1024 * 1024};                 // 4 MB
    TensorF32* t = ctx->new_tensor<float>(1, ne);
    const size_t nb = (size_t)t->nbytes();
    std::vector<float> pat(nb / sizeof(float));
    for (size_t i = 0; i < pat.size(); ++i) pat[i] = (float)(i % 997) - 100.0f;
    float* host = bind_leaf_data(*ctx, t);                // no_alloc 模式：从 ctx scratch 绑定
    std::memcpy(host, pat.data(), nb);

    RemoteBufferType bt(&cli, /*slim=*/false);
    std::unique_ptr<Buffer> buf(bt.new_buffer(0, BufferUsage::COMPUTE));

    const double t0 = now_ms();
    buf->set_tensor(t, host, 0, nb);                      // 上传 TENSOR_SET
    cli.mark_remote(t, cli.id_for(t));
    std::vector<float> back(nb / sizeof(float), -1.0f);
    if (!cli.fetch_tensor(cli.id_for(t), back.data(), nb)) {
        std::fprintf(stderr, "[probe] buffer: fetch 失败\n");
        return 1;
    }
    const double dt = now_ms() - t0;

    int bad = 0;
    for (size_t i = 0; i < pat.size(); ++i) {
        if (back[i] != pat[i]) {
            if (++bad < 4) std::fprintf(stderr, "[probe] buffer: #%zu 不符 %f vs %f\n", i, back[i], pat[i]);
        }
    }
    std::fprintf(stderr, "[probe] buffer 往返: %zu KB 双向 %.2f ms（折合 %.1f MB/s），差异=%d %s\n",
                 nb / 1024, dt, 2.0 * (double)nb / 1048576.0 / (dt / 1000.0), bad,
                 bad == 0 ? "✓" : "✗");
    return bad == 0 ? 0 : 1;
}

// 2) Allreduce（数据并行）
static int test_allreduce(RemoteClient& cli) {
    float buf[3] = {1.0f, 2.0f, 3.0f};
    const double t0 = now_ms();
    if (!cli.allreduce_sum(buf, 3)) { std::fprintf(stderr, "[probe] allreduce 失败\n"); return 1; }
    const double dt = now_ms() - t0;
    const bool ok = (buf[0] == 11.0f && buf[1] == 22.0f && buf[2] == 33.0f);
    std::fprintf(stderr, "[probe] allreduce: 客户端得到 {%.0f,%.0f,%.0f}（期望 {11,22,33}）%s，%.2f ms\n",
                 buf[0], buf[1], buf[2], ok ? "✓" : "✗", dt);
    return ok ? 0 : 1;
}

// 3) 远端执行子图：c = mul_mat(a, b)；a=[3,2] b=[3,2] ⇒ c=[2,2]，c[i,j]=Σ_k a[k,i]*b[k,j]
static int test_compute(std::shared_ptr<RemoteClient> cli) {
    PPMLContext* ctx = &context();
    const int64_t ne_a[2] = {3, 2};
    const int64_t ne_b[2] = {3, 2};
    TensorF32* a = ctx->new_tensor<float>(2, ne_a);
    TensorF32* b = ctx->new_tensor<float>(2, ne_b);
    const float av[6] = {1, 2, 3, 4, 5, 6};        // 按 [3,2]：a[k,i]
    const float bv[6] = {1, 0, 0, 1, 1, 1};        // 按 [3,2]：b[k,j]
    std::memcpy(bind_leaf_data(*ctx, a), av, sizeof(av));
    std::memcpy(bind_leaf_data(*ctx, b), bv, sizeof(bv));

    TensorF32* c = mul_mat(a, b);
    ComputeGraph* cg = ComputeGraph::new_graph(ctx);
    cg->build_forward_expand(c);

    RemoteBackend rb(cli, /*priority=*/0);          // 直接调用，不注册 scheduler
    const double t0 = now_ms();
    Status st = rb.graph_compute(cg);
    const double dt = now_ms() - t0;
    if (st != Status::SUCCESS) {
        std::fprintf(stderr, "[probe] compute: 远端执行失败 status=%d\n", (int)st);
        return 1;
    }

    // 按需拉回结果（与调度器的 backend_tensor_copy Case2 同路径）
    const size_t nb = (size_t)c->nbytes();
    std::vector<float> got(nb / sizeof(float), 0.0f);
    if (!cli->fetch_tensor(cli->id_for(c), got.data(), nb)) {
        std::fprintf(stderr, "[probe] compute: fetch 结果失败（id=%llu）\n",
                     (unsigned long long)cli->id_for(c));
        return 1;
    }

    // 本地参考：**同一张图**在本地 CPU 后端执行（与远端执行走同一套"bump 分配 + skip_alloc"），
    //   这样比的是"远端 == 本地"，不依赖对算子约定的手工推导。
    TensorF32* a2 = ctx->new_tensor<float>(2, ne_a);
    TensorF32* b2 = ctx->new_tensor<float>(2, ne_b);
    std::memcpy(bind_leaf_data(*ctx, a2), av, sizeof(av));
    std::memcpy(bind_leaf_data(*ctx, b2), bv, sizeof(bv));
    TensorF32* c2 = mul_mat(a2, b2);
    ComputeGraph* cg2 = ComputeGraph::new_graph(ctx);
    cg2->build_forward_expand(c2);
    size_t total = 0;
    for (int i = 0; i < cg2->n_nodes(); ++i) total += GGML_PAD((size_t)cg2->graph_node(i)->nbytes(), 64);
    std::unique_ptr<Buffer> big(new DefaultBuffer(CPUBufferType::instance(), total ? total : 64));
    {
        TensorAllocator ta(big.get());
        for (int i = 0; i < cg2->n_nodes(); ++i) ta.alloc(cg2->graph_node(i));
    }
    CPUBackend cpu(4);
    cpu.set_skip_alloc(true);
    Status lst = cpu.graph_compute(cg2);
    cpu.set_skip_alloc(false);
    if (lst != Status::SUCCESS) {
        std::fprintf(stderr, "[probe] compute: 本地参考执行失败 status=%d\n", (int)lst);
        return 1;
    }
    std::vector<float> ref((size_t)c2->numel());
    std::memcpy(ref.data(), c2->data(), ref.size() * sizeof(float));

    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) if (got[i] != ref[i]) ++bad;
    std::fprintf(stderr,
                 "[probe] compute: 远端=[%.1f,%.1f,%.1f,%.1f] 本地=[%.1f,%.1f,%.1f,%.1f] %s"
                 "（远端 %.2f ms，节点 %d 个，发送 %.2f MB）\n",
                 got[0], got[1], got[2], got[3], ref[0], ref[1], ref[2], ref[3],
                 bad == 0 ? "✓" : "✗", dt, cg->n_nodes(),
                 (double)cli->bytes_sent() / 1048576.0);
    return bad == 0 ? 0 : 1;
}

// ---------------- main ----------------
static void usage() {
    std::fprintf(stderr,
        "用法:\n"
        "  ppml_remote_probe server --port 2244 [--dp 10,20,30]\n"
        "  ppml_remote_probe client --host 127.0.0.1 --port 2244 [--test all|buffer|allreduce|compute]\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    const std::string mode = argv[1];
    int port = 2244;
    std::string host = "127.0.0.1";
    std::string test = "all";
    std::string dp;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--port") && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!std::strcmp(argv[i], "--test") && i + 1 < argc) test = argv[++i];
        else if (!std::strcmp(argv[i], "--dp") && i + 1 < argc) dp = argv[++i];
    }

    if (mode == "server") return run_server(port, dp);
    if (mode != "client") { usage(); return 1; }

    auto cli = std::make_shared<RemoteClient>();
    if (!cli->connect(host, port, 2, 1)) {
        std::fprintf(stderr, "[probe] 连不上 %s:%d\n", host.c_str(), port);
        return 1;
    }

    int fails = 0;
    if (test == "all" || test == "buffer")    fails += test_buffer_roundtrip(*cli);
    if (test == "all" || test == "allreduce") fails += test_allreduce(*cli);
    if (test == "all" || test == "compute")   fails += test_compute(cli);

    std::fprintf(stderr, "[probe] 完成：失败项=%d，发送=%.2f MB 接收=%.2f MB（上传 %zu 次，子图 %zu 次）\n",
                 fails, (double)cli->bytes_sent() / 1048576.0,
                 (double)cli->bytes_recv() / 1048576.0, cli->n_uploads(), cli->n_graphs());
    return fails == 0 ? 0 : 1;
}

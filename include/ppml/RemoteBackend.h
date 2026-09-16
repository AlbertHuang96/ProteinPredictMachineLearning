// ============================================================
// RemoteBackend.h —— 把"另一台机器"变成第 N 个后端（2026-09-13）
//
// 目标（对应用户需求）：
//   * 派生 RemoteBackend + RemoteBufferType/RemoteBuffer；
//   * 复用 BackendScheduler 的 split_graph/build_splits（自动按后端切段、自动插 OP_DUP 跨后端拷贝）；
//   * 调用点不变：跨机搬运全部落在 Buffer::set_tensor/get_tensor 的虚函数上；
//   * 注册只需一行：scheduler_->add_backend(remote_backend_.get())
//
// 数据面协议见 RemoteProtocol.h（头：dtype/shape/tensor_id + 数据体）。
//
// 关键语义（v1）：
//   set_tensor(远端 buffer)  = 本地 staging 记账 + 上传对端（TENSOR_SET）
//   get_tensor(远端 buffer)  = 若数据在对端（远端算出来的）→ TENSOR_GET 拉回；否则读本地 staging
//   graph_compute(子图)      = 节点描述（op/dims/op_params/src_id）+ 需要上传的常量/参数 → 对端执行
//                              返回后客户端按需用 get_tensor 拉回边界张量（按需，不预推）
//
// 内存策略：
//   * `slim=false`（默认）：本地仍按 gallocr 峰值分配 staging（行为最保守、绝不越界）；
//   * `slim=true` ：本地只分配极小 staging（真正把激活内存放到对端）；需要外部保证不会对
//                   远端张量做本地原始指针读写（本项目路径已全部走 Buffer 虚函数）。
// ============================================================
#pragma once

#include "ppml/Backend.h"
#include "ppml/ComputeGraph.h"
#include "ppml/RemoteProtocol.h"

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ppml {

class RemoteClient;

// ============================================================
// RemoteBufferType —— "对端内存"的 buffer 类型
//   is_host() = false ⇒ 与 CPU/CUDA 之间都会走 backend_tensor_copy 的 set/get 虚函数路径
// ============================================================
class RemoteBufferType : public BufferType {
public:
    explicit RemoteBufferType(RemoteClient* cli, bool slim = false)
        : cli_(cli), slim_(slim) {}

    const char* get_name() const override { return "REMOTE"; }
    void*  alloc(size_t size) override;
    void   free(void* ptr) override;
    size_t get_alignment() const override { return 64; }
    bool   is_host() const override { return false; }   // 非 host：跨后端拷贝必走 set/get
    Buffer* new_buffer(size_t size, BufferUsage usage = BufferUsage::COMPUTE) override;

    RemoteClient* client() const { return cli_; }
    bool slim() const { return slim_; }

private:
    RemoteClient* cli_ = nullptr;
    bool slim_ = false;
};

// ============================================================
// RemoteBuffer —— 本地 staging + 远端镜像
// ============================================================
class RemoteBuffer : public Buffer {
public:
    RemoteBuffer(RemoteBufferType* bt, size_t size, BufferUsage usage);
    ~RemoteBuffer() override;

    void* data() override { return ptr_; }

    // 跨后端拷贝（backend_tensor_copy Case1/Case3）调用：写入 staging + 上传对端
    void set_tensor(TensorF32* tensor, const void* src, size_t offset, size_t size) override;
    // 跨后端拷贝（Case2/Case3）调用：远端算出的 → 拉取；本地 staging 的 → memcpy
    void get_tensor(const TensorF32* tensor, void* dst, size_t offset, size_t size) override;
    void memset_tensor(TensorF32* tensor, uint8_t value, size_t offset, size_t size) override;
    void clear(uint8_t value) override;
    // 直接 GPU→GPU 不支持：返回 false 让它走 staging（get+set）路径（对我们的语义正确）
    bool cpy_tensor(const TensorF32*, TensorF32*, size_t, size_t, size_t) override { return false; }

    bool owns_ptr() const { return own_; }

private:
    uint8_t* ptr_ = nullptr;
    bool own_ = true;
};

// ============================================================
// RemoteClient —— 一条到对端的连接 + 张量 id 管理 + 子图提交 + Allreduce
// ============================================================
class RemoteClient {
public:
    RemoteClient() = default;
    ~RemoteClient();

    // 作为 worker（rank!=0）连到 rank0；作为 rank0 时用 accept_from()
    bool connect(const std::string& host, int port, int world_size = 2, int rank = 1);
    bool accept_from(rnet::RSocket& listener, int world_size = 2, int rank = 0);
    void close();

    bool connected() const { return sock_.valid(); }
    int  rank() const { return rank_; }
    int  world_size() const { return world_size_; }

    // ---- T1（2026-09-16）：端点记忆 / 断线重连 / 心跳 ----
    //   语义（诚实版，别误解）：
    //     * **迭代边界**（remote_iteration_begin）会先 ping、必要时重连 ⇒ 抖一次网络后**下一轮自动恢复** ✓
    //       （旧行为：连接一断就**永久**失效 ✗，后面所有远端 split 全部失败/回落）
    //     * **迭代中途**断线不做透明重试：那时已上传的输入无法重建（slim 模式下客户端不持有数据 ✗），
    //       所以只标记 io_failed_ 并让本 split 失败（STRICT 时 abort）；连接会重建，下一轮可继续 ✓
    bool ensure_connected();            // 不可用则重连（含退避重试），返回最终是否可用
    bool reconnect();                   // 强制重建连接（内部用）
    bool ping(int timeout_ms = 3000);   // 心跳：PING → PONG（失败即标记连接不可信）
    void set_endpoint(const std::string& host, int port) { host_ = host; port_ = port; }
    uint64_t n_reconnects() const { return n_reconnects_; }
    bool io_failed() const { return io_failed_; }

    // tensor → 稳定 id（同一指针复用同一 id；重新分配后指针变化会得到新 id）
    uint64_t id_for(const TensorF32* t);

    // 上传张量（默认幂等：同一 id 且未 invalidate 时跳过）
    //   force=true ⇒ 无条件重传。用于 set_tensor/memset_tensor 路径：调用方已经确认
    //   "数据被改写"，此时幂等跳过会让对端永久持旧值（静默错值）。见 RemoteBuffer::set_tensor。
    uint64_t upload_tensor(const TensorF32* t, const void* data, size_t nbytes,
                           rnet::RDtype dt = rnet::RDtype::F32, bool force = false);
    // 参数在 optimizer.step() 后调用：下次 graph_compute 重新上传
    void invalidate(const TensorF32* t);
    void invalidate_all();
    // 只作废"激活/输入"类 id（参数/常量保持有效）——迭代边界用，避免每轮重传全部参数
    void invalidate_transient();
    // 标记某 id 为"跨迭代保留"（参数/常量：对端 arena 里的这份要活过 RESULTS_CLEAR）
    void mark_persistent(uint64_t id);
    const std::vector<uint64_t>& persistent_ids() const { return persistent_list_; }
    // 迭代边界：通知对端丢弃上一轮 results_ 与非持久 arena 项（body = persistent_ids()）
    bool clear_results(uint64_t* out_dropped_results = nullptr,
                       uint64_t* out_dropped_arena = nullptr);

    // 拉取远端张量（TENSOR_GET → TENSOR_DATA）
    bool fetch_tensor(uint64_t id, void* dst, size_t nbytes);
    // 标记"该张量的数据在对端"（远端 split 内部节点 / 远端算出的结果）
    void mark_remote(const TensorF32* t, uint64_t id);
    bool is_remote(const TensorF32* t) const;
    uint64_t remote_id_of(const TensorF32* t) const;

    // 提交子图到对端执行（描述 + 输出 id 列表；输出会缓存在对端，按需 fetch）
    bool compute_graph(const std::vector<rnet::RNodeDesc>& nodes,
                       const std::vector<uint64_t>& outs);

    // 数据并行：Allreduce（星型；2 机时等价一次对换）
    bool allreduce_sum(float* buf, size_t n);

    // 统计
    uint64_t bytes_sent() const { return bytes_sent_; }
    uint64_t bytes_recv() const { return bytes_recv_; }
    size_t   n_uploads() const { return n_uploads_; }
    size_t   n_graphs()  const { return n_graphs_; }

private:
    rnet::RSocket sock_;
    int world_size_ = 2;
    int rank_ = 1;
    uint64_t next_id_ = 1;
    std::unordered_map<const TensorF32*, uint64_t> id_map_;
    std::unordered_map<uint64_t, bool> dirty_;      // id → 需要（重）上传
    std::unordered_map<const TensorF32*, uint64_t> remote_ids_;  // "数据在对端"的张量
    std::unordered_set<uint64_t> persistent_ids_;   // 参数/常量 id（跨迭代保留）
    std::vector<uint64_t>        persistent_list_;  // 同上，稳定顺序（发 keep 列表用）
    bool     dp_broken_ = false;   // 阶段 A：allreduce 流已错位 ⇒ 后续 DP 同步一律快速失败（不静默错值）
    // T1（2026-09-16）：端点记忆 + 重连状态
    std::string host_;             // 记住端点 ⇒ 断线可重连（accept_from 侧为空 ⇒ 不重连）
    int      port_ = 0;
    bool     io_failed_ = false;   // 最近一次 I/O 失败 ⇒ 连接不可信（ensure_connected 会重连）
    uint64_t n_reconnects_ = 0;
    uint64_t bytes_sent_ = 0, bytes_recv_ = 0;
    size_t n_uploads_ = 0, n_graphs_ = 0;
};

// ============================================================
// RemoteBackend —— 注册进 BackendScheduler 的"远端设备"
//   priority 故意低于 CPU(0)/CUDA(1)：只有在显式指定/预算允许时才被分配节点。
//   supports_op 默认只放行"纯计算"白名单（由 env PPML_REMOTE_OPS 覆盖成 all）。
// ============================================================
class RemoteBackend : public Backend {
public:
    explicit RemoteBackend(std::shared_ptr<RemoteClient> cli, int priority = -1)
        : cli_(std::move(cli)), buft_(cli_.get(), false), priority_(priority) {}

    const char* get_name() const override { return "REMOTE"; }

    // 把本 split 的子图发到对端执行（节点描述里的 src 用 tensor id 表达）
    Status graph_compute(ComputeGraph* cgraph) override;
    void   synchronize() override {}   // 同步协议：请求-响应式，无需额外同步

    bool supports_op(TensorF32* node) const override;
    const BufferType* buffer_type() const override { return &buft_; }
    bool supports_buffer_type(const BufferType* t) const override { return t == &buft_; }
    int  priority() const override { return priority_; }
    Gallocr& gallocr() override { return gallocr_; }

    RemoteClient* client() const { return cli_.get(); }
    RemoteBufferType& buffer_type_mut() { return buft_; }
    void set_slim(bool v) { buft_ = RemoteBufferType(cli_.get(), v); }

private:
    std::shared_ptr<RemoteClient> cli_;
    RemoteBufferType buft_;
    Gallocr gallocr_;
    int priority_ = -1;
};

// ============================================================
// RemoteServer —— 对端（rank0 或独立张量服务）侧的执行器
//   职责：收 TENSOR_SET（存入 arena）、收 GRAPH_COMPUTE（按节点描述重建 ComputeGraph 并用本地
//         后端算）、按 TENSOR_GET 回传、参与 ALLREDUCE（星型 root）。
// ============================================================
class RemoteServer {
public:
    // 监听端口；serve_forever() 处理**一条**连接直到 BYE（2 机场景足够）
    bool listen(int port);
    bool serve_forever(int world_size = 2, int max_messages = 100000);

    // arena 相关（诊断用）
    size_t n_tensors() const { return arena_.size(); }
    size_t arena_bytes() const { return arena_bytes_; }
    bool   use_cuda() const { return use_cuda_; }
    void   set_use_cuda(bool v) { use_cuda_ = v; }

    // ---- 阶段 A（peer 双端训练）：按调用次序登记的"本端贡献" + 求和结果回放 ----
    //   pair（左右两栏都跑训练）时，本端也有真实梯度/权重 ⇒ 必须与对端**同一参数各调用一次**、
    //   顺序一致（handle_allreduce 按第 k 次调用配对）。启用后 handle_allreduce 走队列；
    //   未启用 ⇒ 退回旧 dp_local_ 平铺语义（独立张量服务路径不受影响）。
    void enable_dp_queue();
    bool dp_queue_enabled() const;
    void push_dp_contribution(const float* p, size_t n);
    bool pop_dp_contribution(std::vector<float>& out);          // 阻塞（有界），服务线程用
    void publish_dp_result(const std::vector<float>& r);        // 服务线程回放给训练线程
    bool wait_dp_result(float* out, size_t n, int timeout_sec = 60);  // 训练线程取回第 k 次结果

    // 数据并行：本端作为 rank0 时"自己那份"待求和缓冲（star allreduce 的 root 贡献）
    void set_dp_buffer(const float* p, size_t n) {
        dp_local_.assign(p, p ? p + n : p);
    }
    const std::vector<float>& dp_buffer() const { return dp_local_; }

private:
    bool handle_tensor_set(rnet::RSocket& s, rnet::RHeader& h);
    bool handle_tensor_get(rnet::RSocket& s, rnet::RHeader& h);
    bool handle_graph_compute(rnet::RSocket& s, rnet::RHeader& h);
    bool handle_allreduce(rnet::RSocket& s, rnet::RHeader& h, int world_size);
    bool handle_results_clear(rnet::RSocket& s, rnet::RHeader& h);

    rnet::RSocket listener_;
    // arena：id → 数据（宿主内存；GRAPH_COMPUTE 时作为输入注入）
    std::unordered_map<uint64_t, std::vector<uint8_t>> arena_;
    std::unordered_map<uint64_t, std::vector<int64_t>> arena_dims_;
    std::unordered_map<uint64_t, rnet::RDtype> arena_dtype_;
    // 上次 GRAPH_COMPUTE 产生的节点结果（id → 数据）
    std::unordered_map<uint64_t, std::vector<uint8_t>> results_;
    std::deque<std::vector<uint64_t>> recent_ids_;   // 最近若干次执行的节点 id（字节上限兜底淘汰）
    // 【2026-09-14 晚】主淘汰机制改成"客户端在迭代边界发 RESULTS_CLEAR"；
    //   下面的 K / 字节上限只作为兜底（客户端崩溃/未调用 hook 时不至于把对端撑爆）。
    size_t keep_splits_ = 4096;                      // K，可用 PPML_REMOTE_KEEP_SPLITS 覆盖
    size_t keep_bytes_  = 2ull * 1024 * 1024 * 1024; // 结果内存上限，PPML_REMOTE_KEEP_MB 覆盖
    size_t results_bytes_ = 0;
    std::vector<float> dp_local_;          // rank0 自己那份（数据并行 star allreduce，旧路径）
    // 阶段 A：贡献/结果队列（定义在 .cpp，避免头文件引入 <mutex>/<condition_variable>）
    struct DpQueue;
    std::shared_ptr<DpQueue> dpq_;
    size_t arena_bytes_ = 0;
    bool use_cuda_ = false;
};

// ============================================================
// 便捷：按环境变量创建 RemoteBackend（供 PPMLModel::ensure_backend_ready 一行接入）
//   PPML_REMOTE_HOST=<peer ip>  PPML_REMOTE_PORT=2244  PPML_REMOTE_RANK=1|0  PPML_REMOTE_SLIM=1
// 返回 nullptr 表示未启用。
// ============================================================
std::shared_ptr<RemoteClient> remote_make_client_from_env();
RemoteBackend* remote_make_backend_from_env();   // 内部持有单例，注册用

// 数据并行开关（PPML_DP=1）：返回与管线并行复用的对端连接；
// 未启用 / 连不上时返回 nullptr（调用点静默跳过，单机零开销）。首次调用时建连并打日志。
RemoteClient* remote_dp_client_if_enabled();

// 数据并行辅助：对图里所有 PARAM 的梯度做 Allreduce 平均（在 optimizer.step() 之前调用）
//   返回处理过的参数个数（0 = 未启用/无梯度）。
int remote_dp_allreduce_grads(ComputeGraph* cgraph, RemoteClient* cli);

// 数据并行（阶段 A：参数分片 / ZeRO-1 式）：optimizer.step() **之后**调用。
//   每个 rank 只更新 owner 的参数；本函数把"owner 的新值"广播给对端：
//   非 owner 侧该参数置 0 后 allreduce ⇒ sum 恒等于唯一 owner 的新值 ⇒ 两端逐位一致（无需 /world）。
//   owner 规则与 AdamW 分片一致：参数全局序号 idx % world（world/rank 取自 cli，未连接返回 0）。
int remote_dp_broadcast_owned_params(ComputeGraph* cgraph, RemoteClient* cli);

// ---- 阶段 A：rank0（acceptor / star root）侧的 DP 同步（见 PLAN_SHARDING.md §5.5）----
//   rank0 是被连的一方，不走客户端 UP；它 push 本端贡献 → 服务线程收到对端 UP 后求和回放
//   → 训练线程 wait_dp_result 取回结果。两函数与 worker 侧的同名动作**位置/顺序一一对应**：
//     remote_dp_grads_root  ←→ remote_dp_allreduce_grads（clip 之前）
//     remote_dp_params_root ←→ remote_dp_broadcast_owned_params（step 之后）
void remote_dp_register_root_server(RemoteServer* srv);
bool remote_dp_root_mode();
int  remote_dp_grads_root(ComputeGraph* cgraph, int world, int my_rank);   // clip 之前
int  remote_dp_params_root(ComputeGraph* cgraph, int world, int my_rank);  // optimizer.step() 之后

// ---- 迭代边界 hook（train.cpp 调用；未启用远端时零开销）----
// 每个样本 forward **之前**调用一次：
//   ① 作废"激活/输入"类上传（参数/常量保留）——同指针跨迭代复用时防止对端拿旧数据；
//   ② 通知对端丢弃上一轮结果（只保留参数/常量 id）——保证 forward+backward 全程可取回，且内存有界。
// 返回 true = 真的发了消息（诊断用）。
bool remote_iteration_begin();

// optimizer.step() **之后**调用一次：把本图 PARAM 标为待重传 ⇒ 下一轮 compute 自动同步新权重。
//   返回被作废的参数个数。
int remote_after_optimizer_step(ComputeGraph* cgraph);

}  // namespace ppml

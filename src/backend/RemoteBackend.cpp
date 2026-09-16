// ============================================================
// RemoteBackend.cpp —— 远端后端实现（2026-09-13）
//   数据面：RemoteProtocol（TCP + 定长头 + 数据体）
//   接入面：BackendScheduler（split_graph 自动切段 + OP_DUP 跨后端拷贝）⇒ 本文件只实现虚函数
// ============================================================
#include "ppml/RemoteBackend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace ppml {

// ============================================================
// 诊断探针（PPML_REMOTE_TRACE=1）—— 2026-09-14 为定位 "TENSOR_GET 未命中 ⇒ abort" 而加
//   把每个 tensor_id 的一生打出来：诞生(NEWID) → 上传(SET) → 标记远端(MARK) → 取回(GET)。
//   任何"标了 remote 却没真送到对端"的情况会立刻暴露成 SET-SKIP / INPUT 行。
//   未设环境变量时只有一个 static bool，零开销。
// ============================================================
static bool rtrace() {
    static const bool on = (std::getenv("PPML_REMOTE_TRACE") != nullptr);
    return on;
}

// 【加固 2026-09-16】RSS（/proc/self/statm 第 2 字段 = resident pages；x86_64 页 4KB）
//   用途：服务端每条消息打一行，出问题时能直接看出"是内存爆了还是别的原因"。
static double cur_rss_mb() {
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return -1.0;
    long total = 0, resident = -1;
    if (std::fscanf(f, "%ld %ld", &total, &resident) != 2) resident = -1;
    std::fclose(f);
    if (resident < 0) return -1.0;
    return (double)resident * 4096.0 / 1048576.0;
}

// 【加固 2026-09-16】严格模式（PPML_REMOTE_STRICT=1）：远端失败**立即终止**，不再静默回落本地。
//   为什么需要：默认行为下"对端死了"仍会继续跑（调度器回落 CPU 单后端），跑完 loss 与单机一致 ⇒
//   极易误判"远端等价已验证" ✗。验收远端时务必开它。
static bool rstrict() {
    static const bool on = (std::getenv("PPML_REMOTE_STRICT") != nullptr &&
                            std::atoi(std::getenv("PPML_REMOTE_STRICT")) != 0);
    return on;
}

static void rstrict_fail(const char* what) {
    std::fprintf(stderr, "\n[REMOTE] FATAL: %s（PPML_REMOTE_STRICT=1 ⇒ 立即终止，避免误判为远端已生效）\n", what);
    std::fflush(stderr);
    std::abort();
}
static void rt_dims(const TensorF32* t, char* out, size_t n) {
    if (!t) { std::snprintf(out, n, "-"); return; }
    int w = 0;
    const int nd = t->shape().ndim();
    for (int i = 0; i < nd && i < 4 && w < (int)n; ++i) {
        w += std::snprintf(out + w, n - (size_t)w, "%s%lld", i ? "," : "",
                           (long long)t->shape().dim(i));
    }
    if (w == 0) std::snprintf(out, n, "-");
}

// ============================================================
// RemoteBufferType
// ============================================================
void* RemoteBufferType::alloc(size_t size) {
    // 本地 staging：默认与请求等大（最保守）；slim 模式只留 1MB 触发位（真内存放在对端）
    const size_t want = slim_ ? (size == 0 ? 0 : (size_t)(1u << 20)) : size;
    if (want == 0) return nullptr;
    return std::malloc(want);
}

void RemoteBufferType::free(void* ptr) {
    std::free(ptr);
}

Buffer* RemoteBufferType::new_buffer(size_t size, BufferUsage usage) {
    return new RemoteBuffer(this, size, usage);
}

// ============================================================
// RemoteBuffer
// ============================================================
RemoteBuffer::RemoteBuffer(RemoteBufferType* bt, size_t size, BufferUsage usage)
    : Buffer(bt, size, usage) {
    if (size > 0) {
        RemoteBufferType* rbt = static_cast<RemoteBufferType*>(bt);
        ptr_ = static_cast<uint8_t*>(rbt->alloc(size));
        if (!ptr_) {
            throw PPMLError("RemoteBuffer: staging allocation failed");
        }
    }
}

RemoteBuffer::~RemoteBuffer() {
    if (ptr_ && buft_) buft_->free(ptr_);
    ptr_ = nullptr;
}

void RemoteBuffer::set_tensor(TensorF32* tensor, const void* src, size_t offset, size_t size) {
    RemoteBufferType* rbt = static_cast<RemoteBufferType*>(buft_);
    RemoteClient* cli = rbt ? rbt->client() : nullptr;
    const size_t nbytes = tensor ? (size_t)tensor->nbytes() : size;

    // 【2026-09-14 修正】调度器给的 offset 是"该张量在所属 buffer 内的偏移"
    //   （backend_tensor_copy 传的是 dst->buffer_offs_），**不是 0**！
    //   所以"整张量更新"的判据是 offset == tensor->buffer_offs_ && size == nbytes。
    //   （旧判据 `offset == 0` 会让所有落在 arena 里非零偏移的张量都拒绝上传 ✗
    //     —— 实测表现为对端报 "缺输入张量 id=2"。）
    const size_t t_offs = tensor ? (size_t)tensor->buffer_offs_ : 0;
    const bool whole = tensor && (offset == t_offs) && (size == nbytes);

    // 1) 本地 staging 记账（slim 模式下不写）
    if (!rbt->slim() && ptr_ && offset + size <= size_) {
        std::memcpy(ptr_ + offset, src, size);
    }

    // 2) 上传到对端（跨机搬运的真正落点）
    //   【2026-09-14 晚 修正】force=true：走到 set_tensor 就意味着"这份数据被改写了"，
    //   旧的幂等跳过（dirty_ 未置位就 return）会让对端永久持旧值 ⇒ 静默错值。
    if (cli && cli->connected() && whole) {
        const uint64_t upid = cli->upload_tensor(tensor, src, nbytes, rnet::RDtype::F32,
                                                 /*force=*/true);
        const uint64_t tid  = cli->id_for(tensor);
        cli->mark_remote(tensor, tid);
        if (tensor && (tensor->flag & (TENSOR_FLAG_PARAM | TENSOR_FLAG_CONST))) {
            cli->mark_persistent(tid);   // 参数/常量：对端 arena 里这份要活过迭代边界
        }
        if (rtrace()) {
            char sh[64]; rt_dims(tensor, sh, sizeof(sh));
            std::fprintf(stderr, "[RT] SET id=%llu op=%d nb=%zu shape=[%s] ret=%llu t_offs=%zu\n",
                         (unsigned long long)tid, tensor ? (int)tensor->op : -1, nbytes, sh,
                         (unsigned long long)upid, t_offs);
        }
    } else if (cli && cli->connected() && tensor) {
        std::fprintf(stderr, "[REMOTE] warn: 非整张量 set_tensor 未上传"
                             "（offset=%zu tensor_offs=%zu size=%zu nbytes=%zu）\n",
                     offset, t_offs, size, nbytes);
    }
}

void RemoteBuffer::get_tensor(const TensorF32* tensor, void* dst, size_t offset, size_t size) {
    RemoteBufferType* rbt = static_cast<RemoteBufferType*>(buft_);
    RemoteClient* cli = rbt ? rbt->client() : nullptr;

    // 【2026-09-14 修正】同 set_tensor：offset 是"张量在 buffer 内的偏移"(= buffer_offs_)，
    //   不是 0。旧判据 `offset == 0` 会让"远端算出的张量"取回失败并**静默回落到本地 staging
    //   （陈旧数据）** —— 实测表现为 loss 与单机不一致（17.05 vs 14.60/32.44）而全程无报错。
    const size_t t_offs = tensor ? (size_t)tensor->buffer_offs_ : 0;
    const bool whole = tensor && (offset == t_offs) && (size == (size_t)tensor->nbytes());

    // 1) 远端算出来的张量：按需拉取（唯一正确来源）
    if (cli && cli->connected() && tensor && cli->is_remote(tensor) && whole) {
        const uint64_t rid = cli->remote_id_of(tensor);
        if (rtrace()) {
            char sh[64]; rt_dims(tensor, sh, sizeof(sh));
            std::fprintf(stderr, "[RT] GET id=%llu op=%d nb=%zu shape=[%s] (fetch 前)\n",
                         (unsigned long long)rid, (int)tensor->op, size, sh);
        }
        if (cli->fetch_tensor(rid, dst, size)) return;
        // 取不回就**中止**：继续跑等于拿陈旧/垃圾数据训练（静默污染比崩溃更糟）
        std::fprintf(stderr,
                     "[REMOTE][FATAL] 远端张量取回失败 id=%llu size=%zu —— 中止以避免静默污染\n",
                     (unsigned long long)rid, size);
        std::abort();
    }

    // 2) 本地 staging（仅用于"本地已有一份正确拷贝"的张量，如上传后的 cpy 原件）
    if (ptr_ && offset + size <= size_) {
        std::memcpy(dst, ptr_ + offset, size);
        return;
    }
    std::fprintf(stderr, "[REMOTE] get_tensor: 既非远端张量也无本地 staging"
                         "（tensor=%p offset=%zu tensor_offs=%zu size=%zu slim=%d）\n",
                 (const void*)tensor, offset, t_offs, size, (int)(rbt ? rbt->slim() : 0));
}

void RemoteBuffer::memset_tensor(TensorF32* tensor, uint8_t value, size_t offset, size_t size) {
    if (ptr_ && offset + size <= size_) {
        std::memset(ptr_ + offset, value, size);
    }
    // 远端侧：整张量清零时同步上传一份零数据（判据同 set_tensor：offset == buffer_offs_）
    RemoteBufferType* rbt = static_cast<RemoteBufferType*>(buft_);
    RemoteClient* cli = rbt ? rbt->client() : nullptr;
    const size_t t_offs = tensor ? (size_t)tensor->buffer_offs_ : 0;
    if (cli && cli->connected() && tensor && offset == t_offs &&
        size == (size_t)tensor->nbytes()) {
        std::vector<uint8_t> zeros(size, value);
        cli->upload_tensor(tensor, zeros.data(), size, rnet::RDtype::F32, /*force=*/true);
        const uint64_t tid = cli->id_for(tensor);
        cli->mark_remote(tensor, tid);
        if (tensor->flag & (TENSOR_FLAG_PARAM | TENSOR_FLAG_CONST)) cli->mark_persistent(tid);
    }
}

void RemoteBuffer::clear(uint8_t value) {
    if (ptr_) std::memset(ptr_, value, size_);
}

// ============================================================
// RemoteClient
// ============================================================
RemoteClient::~RemoteClient() { close(); }

void RemoteClient::close() {
    if (sock_.valid()) {
        rnet::RHeader bye;
        bye.kind = (uint32_t)rnet::RKind::BYE;
        sock_.send_header(bye);
        sock_.close();
    }
}

// ============================================================
// T1（2026-09-16）：重连 / 心跳
// ============================================================
static bool rreconnect_enabled() {
    static const bool on = [] {
        const char* s = std::getenv("PPML_REMOTE_RECONNECT");
        return !(s && std::atoi(s) == 0);            // 默认开
    }();
    return on;
}
static int rconnect_retry() {
    static const int n = [] {
        const char* s = std::getenv("PPML_REMOTE_CONNECT_RETRY");
        int v = s ? std::atoi(s) : 5;
        return v < 1 ? 1 : v;                        // 默认 5 次（0.5s→5s 退避）
    }();
    return n;
}
static int rtimeout_ms() {
    static const int v = [] {
        const char* s = std::getenv("PPML_REMOTE_TIMEOUT_MS");
        const int t = s ? std::atoi(s) : 0;
        return t > 0 ? t : 600000;
    }();
    return v;
}

bool RemoteClient::reconnect() {
    if (host_.empty() || port_ <= 0) {
        std::fprintf(stderr, "[REMOTE] 重连失败：本端没有记住端点（accept 侧不需要重连）\n");
        return false;
    }
    const int retry = rconnect_retry();
    for (int i = 1; i <= retry; ++i) {
        std::fprintf(stderr, "[REMOTE] 重连尝试 %d/%d → %s:%d …\n", i, retry, host_.c_str(), port_);
        sock_.close();
        if (connect(host_, port_, world_size_, rank_)) {
            ++n_reconnects_;
            // 新连接 = 对端状态全新：清掉"数据在对端"记账 + 全部标脏（下次用到时重传）
            remote_ids_.clear();
            invalidate_all();
            std::fprintf(stderr, "[REMOTE] 重连成功（第 %llu 次）：已清空远端记账并标记全部待重传 ✓\n",
                         (unsigned long long)n_reconnects_);
            return true;
        }
        int ms = 500 << (i - 1);                     // 0.5s, 1s, 2s, 4s … 上限 5s
        if (ms > 5000) ms = 5000;
        struct timespec ts{ms / 1000, (long)(ms % 1000) * 1000000L};
        nanosleep(&ts, nullptr);
    }
    std::fprintf(stderr, "[REMOTE] 重连失败（%d 次）✗ ⇒ 本轮远端不可用\n", retry);
    return false;
}

bool RemoteClient::ensure_connected() {
    if (sock_.valid() && !io_failed_) return true;
    if (!rreconnect_enabled()) return false;
    if (std::getenv("GRAPH_DEBUG_REMOTE") || rtrace()) {
        std::fprintf(stderr, "[REMOTE] ensure_connected: 连接不可用（valid=%d io_failed=%d）⇒ 重连\n",
                     (int)sock_.valid(), (int)io_failed_);
    }
    return reconnect();
}

bool RemoteClient::ping(int timeout_ms) {
    if (!ensure_connected()) return false;
    sock_.set_timeout_ms(timeout_ms);                // 心跳只等一小会儿（别用 10 分钟的主超时）
    rnet::RHeader h;
    h.kind = (uint32_t)rnet::RKind::PING;
    bool ok = sock_.send_header(h);
    if (ok) {
        rnet::RHeader rh;
        ok = sock_.recv_header(rh) && rh.kind == (uint32_t)rnet::RKind::PONG;
    }
    sock_.set_timeout_ms(rtimeout_ms());             // 恢复主超时
    if (!ok) {
        io_failed_ = true;
        std::fprintf(stderr, "[REMOTE] ping 失败 ⇒ 标记连接不可信（本轮结束会重连）\n");
    }
    return ok;
}

bool RemoteClient::connect(const std::string& host, int port, int world_size, int rank) {
    world_size_ = world_size;
    rank_ = rank;
    host_ = host;                       // T1：记住端点 ⇒ 断线可重连
    port_ = port;
    if (!sock_.connect_to(host, port)) return false;
    rnet::RHeader h;
    h.kind = (uint32_t)rnet::RKind::HELLO;
    h.aux0 = (uint64_t)world_size;
    h.aux1 = (uint64_t)rank;
    if (!sock_.send_header(h)) return false;
    rnet::RHeader ack;
    if (!sock_.recv_header(ack)) return false;
    if (ack.kind != (uint32_t)rnet::RKind::HELLO_ACK) {
        std::fprintf(stderr, "[REMOTE] 握手失败：收到 %s\n", rnet::kind_name(ack.kind));
        return false;
    }
    std::fprintf(stderr, "[REMOTE] 已连接 %s:%d（world=%d rank=%d，对端 rank=%d）\n",
                 host.c_str(), port, world_size, rank, (int)ack.aux1);
    io_failed_ = false;                 // T1：新连接视为健康
    return true;
}

bool RemoteClient::accept_from(rnet::RSocket& listener, int world_size, int rank) {
    world_size_ = world_size;
    rank_ = rank;
    if (!listener.accept_one(sock_)) return false;
    rnet::RHeader h;
    if (!sock_.recv_header(h)) return false;
    if (h.kind != (uint32_t)rnet::RKind::HELLO) {
        std::fprintf(stderr, "[REMOTE] 期望 HELLO，收到 %s\n", rnet::kind_name(h.kind));
        return false;
    }
    rnet::RHeader ack;
    ack.kind = (uint32_t)rnet::RKind::HELLO_ACK;
    ack.aux0 = (uint64_t)world_size;
    ack.aux1 = 0;   // 我是 rank0
    return sock_.send_header(ack);
}

uint64_t RemoteClient::id_for(const TensorF32* t) {
    auto it = id_map_.find(t);
    if (it != id_map_.end()) return it->second;
    uint64_t id = next_id_++;
    id_map_[t] = id;
    dirty_[id] = true;
    if (rtrace()) {
        char sh[64]; rt_dims(t, sh, sizeof(sh));
        std::fprintf(stderr,
                     "[RT] NEWID id=%llu ptr=%p op=%d type=%d nb=%zu shape=[%s] view_src=%p buf=%p\n",
                     (unsigned long long)id, (const void*)t, t ? (int)t->op : -1,
                     t ? (int)t->type : -1, t ? (size_t)t->nbytes() : 0, sh,
                     t ? (const void*)t->view_src : nullptr,
                     (t ? (const void*)t->buffer_ : nullptr));
    }
    return id;
}

uint64_t RemoteClient::upload_tensor(const TensorF32* t, const void* data, size_t nbytes,
                                     rnet::RDtype dt, bool force) {
    const uint64_t id = id_for(t);

    // 幂等：已上传且未被 invalidate ⇒ 跳过（参数/常量张量的稳态路径，避免每步重传）
    //   force=true 时无条件发送（set_tensor/memset_tensor 路径：数据已确认被改写）
    auto d = dirty_.find(id);
    const bool need = force || (d == dirty_.end()) || d->second;
    if (!need) {
        if (rtrace()) {
            std::fprintf(stderr, "[RT] SET-SKIP(幂等，未真发网络) id=%llu nb=%zu ptr=%p\n",
                         (unsigned long long)id, nbytes, (const void*)t);
        }
        return id;
    }
    int64_t dims[4] = {1, 1, 1, 1};
    int ndim = t ? t->shape().ndim() : 1;
    if (t) {
        for (int i = 0; i < ndim && i < 4; ++i) dims[i] = t->shape().dim(i);
    }
    if (rtrace()) {
        // 【加固 2026-09-16】"发送前"先打一行：失败时最后一行的 nb/op 就是现场（原来只在成功后打）
        std::fprintf(stderr, "[RT] SET id=%llu nb=%.2f MB op=%d ptr=%p\n",
                     (unsigned long long)id, (double)nbytes / 1048576.0, t ? (int)t->op : -1, (const void*)t);
    }
    if (!sock_.send_data((uint32_t)rnet::RKind::TENSOR_SET, id, dt, ndim, dims, data, nbytes)) {
        io_failed_ = true;   // T1：连接不可信 ⇒ 迭代边界会重连
        std::fprintf(stderr, "[REMOTE] TENSOR_SET 发送失败 (id=%llu, %.2f MB)\n",
                     (unsigned long long)id, (double)nbytes / 1048576.0);
        if (rstrict()) rstrict_fail("TENSOR_SET 发送失败（对端已断）");
        return 0;
    }
    rnet::RHeader ack;
    if (!sock_.recv_header(ack) || ack.kind != (uint32_t)rnet::RKind::TENSOR_SET_ACK) {
        io_failed_ = true;   // T1
        std::fprintf(stderr, "[REMOTE] TENSOR_SET 未收到 ACK (id=%llu, %.2f MB, op=%d)\n",
                     (unsigned long long)id, (double)nbytes / 1048576.0, t ? (int)t->op : -1);
        if (rstrict()) rstrict_fail("TENSOR_SET 未收到 ACK（对端可能已死）");
        return 0;
    }
    dirty_[id] = false;
    bytes_sent_ += nbytes;
    n_uploads_++;
    if (rtrace()) {
        // 上传字节指纹 ⇒ 与本地跑同一张量的 [HASH-IN] 对拍，判定"送错"还是"算错"
        const uint8_t* p8 = static_cast<const uint8_t*>(data);
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < nbytes; ++i) { h ^= p8[i]; h *= 1099511628211ull; }
        std::fprintf(stderr, "[RT] SET-OK id=%llu nb=%zu op=%d ptr=%p fnv=%016llx\n",
                     (unsigned long long)id, nbytes, t ? (int)t->op : -1, (const void*)t,
                     (unsigned long long)h);
    }
    return id;
}

void RemoteClient::invalidate(const TensorF32* t) {
    auto it = id_map_.find(t);
    if (it != id_map_.end()) dirty_[it->second] = true;
}

void RemoteClient::invalidate_all() {
    for (auto& kv : dirty_) kv.second = true;
}

// 只作废"激活/输入"类 id：参数/常量（persistent）保持有效 ⇒ 迭代边界不必重传全部权重
void RemoteClient::invalidate_transient() {
    for (auto& kv : dirty_) {
        if (!persistent_ids_.count(kv.first)) kv.second = true;
    }
}

void RemoteClient::mark_persistent(uint64_t id) {
    if (persistent_ids_.insert(id).second) persistent_list_.push_back(id);
}

// 迭代边界：把"上一轮的对端结果"全部丢掉，只保留 keep（参数/常量）对应的 arena 项
bool RemoteClient::clear_results(uint64_t* out_dropped_results, uint64_t* out_dropped_arena) {
    if (!sock_.valid()) return false;
    const size_t n = persistent_list_.size();
    rnet::RHeader h;
    h.kind   = (uint32_t)rnet::RKind::RESULTS_CLEAR;
    h.aux0   = (uint64_t)n;
    h.nbytes = n * sizeof(uint64_t);
    if (!sock_.send_header(h)) return false;
    if (n && !sock_.send_all(persistent_list_.data(), n * sizeof(uint64_t))) return false;
    rnet::RHeader rh;
    if (!sock_.recv_header(rh)) return false;
    if (rh.kind != (uint32_t)rnet::RKind::RESULTS_CLEAR_ACK) {
        std::fprintf(stderr, "[REMOTE] RESULTS_CLEAR 未收到 ACK（收到 %s）\n",
                     rnet::kind_name(rh.kind));
        return false;
    }
    if (out_dropped_results) *out_dropped_results = rh.aux0;
    if (out_dropped_arena)   *out_dropped_arena   = rh.aux1;
    return true;
}

bool RemoteClient::fetch_tensor(uint64_t id, void* dst, size_t nbytes) {
    rnet::RHeader h;
    h.kind      = (uint32_t)rnet::RKind::TENSOR_GET;
    h.tensor_id = id;
    h.nbytes    = nbytes;
    if (!sock_.send_header(h)) { io_failed_ = true; return false; }
    rnet::RHeader rh;
    if (!sock_.recv_header(rh)) { io_failed_ = true; return false; }
    if (rh.kind != (uint32_t)rnet::RKind::TENSOR_DATA || rh.tensor_id != id || rh.nbytes != nbytes) {
        io_failed_ = true;
        std::fprintf(stderr, "[REMOTE] TENSOR_GET 失败：%s id=%llu nbytes=%llu (期望 %llu)\n",
                     rnet::kind_name(rh.kind), (unsigned long long)rh.tensor_id,
                     (unsigned long long)rh.nbytes, (unsigned long long)nbytes);
        return false;
    }
    if (!sock_.recv_all(dst, nbytes)) { io_failed_ = true; return false; }
    bytes_recv_ += nbytes;
    if (rtrace()) {
        // 指纹：远端算出来的张量逐字节 FNV-1a ⇒ 可与单机跑同一张量的哈希直接对比
        const uint8_t* p = static_cast<const uint8_t*>(dst);
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < nbytes; ++i) { h ^= p[i]; h *= 1099511628211ull; }
        std::fprintf(stderr, "[RT] FETCH-HASH id=%llu nb=%zu fnv=%016llx\n",
                     (unsigned long long)id, nbytes, (unsigned long long)h);
    }
    return true;
}

void RemoteClient::mark_remote(const TensorF32* t, uint64_t id) {
    if (t) remote_ids_[t] = id;
}

bool RemoteClient::is_remote(const TensorF32* t) const {
    return remote_ids_.find(t) != remote_ids_.end();
}

uint64_t RemoteClient::remote_id_of(const TensorF32* t) const {
    auto it = remote_ids_.find(t);
    return it == remote_ids_.end() ? 0 : it->second;
}

bool RemoteClient::compute_graph(const std::vector<rnet::RNodeDesc>& nodes,
                                 const std::vector<uint64_t>& outs) {
    rnet::RHeader h;
    h.kind   = (uint32_t)rnet::RKind::GRAPH_COMPUTE;
    h.aux0   = (uint64_t)nodes.size();
    h.aux1   = (uint64_t)outs.size();
    h.nbytes = nodes.size() * sizeof(rnet::RNodeDesc) + outs.size() * sizeof(uint64_t);
    if (!sock_.send_header(h)) { io_failed_ = true; return false; }
    if (!nodes.empty() &&
        !sock_.send_all(nodes.data(), nodes.size() * sizeof(rnet::RNodeDesc))) { io_failed_ = true; return false; }
    if (!outs.empty() &&
        !sock_.send_all(outs.data(), outs.size() * sizeof(uint64_t))) { io_failed_ = true; return false; }
    rnet::RHeader rh;
    if (!sock_.recv_header(rh)) { io_failed_ = true; return false; }
    if (rh.kind == (uint32_t)rnet::RKind::ERROR_MSG) {
        std::fprintf(stderr, "[REMOTE] 对端执行报错（aux0=%llu）\n", (unsigned long long)rh.aux0);
        return false;
    }
    if (rh.kind != (uint32_t)rnet::RKind::GRAPH_RESULT) {
        std::fprintf(stderr, "[REMOTE] 期望 GRAPH_RESULT，收到 %s\n", rnet::kind_name(rh.kind));
        return false;
    }
    bytes_sent_ += nodes.size() * sizeof(rnet::RNodeDesc);
    n_graphs_++;
    return true;
}

bool RemoteClient::allreduce_sum(float* buf, size_t n) {
    if (world_size_ <= 1) return true;
    const size_t bytes = n * sizeof(float);
    // worker 侧：上传本端缓冲 → 等 root 广播求和结果
    //   （root 侧"自己那份"由远端进程贡献，见 RemoteServer::handle_allreduce / set_dp_buffer）
    if (!sock_.send_data((uint32_t)rnet::RKind::ALLREDUCE_UP, 0, rnet::RDtype::F32,
                         1, nullptr, buf, bytes)) return false;
    rnet::RHeader h;
    // 【2026-09-14 深夜】失败必须**响亮**：UP 已发出而回包读失败时，流已经错位，
    //   继续跑会让下一次 allreduce 读到上一次的回包 ⇒ 静默错值（本轮实测：loss 爆到 818779 + 崩溃）。
    //   ⇒ 这里明确报错并标记"本轮 DP 已不可信"，调用方不得把 0 当成"没参数"继续训练。
    if (!sock_.recv_header(h)) {
        std::fprintf(stderr,
                     "[REMOTE-DP] FATAL: allreduce 回包读取失败（流已错位，本次 DP 同步不可信；"
                     "请检查对端是否在等我方贡献/是否已死锁）\n");
        dp_broken_ = true;
        return false;
    }
    if (h.kind != (uint32_t)rnet::RKind::ALLREDUCE_DOWN || h.nbytes != bytes) {
        std::fprintf(stderr, "[REMOTE-DP] FATAL: allreduce 意外回包 %s nbytes=%llu（期望 %zu）⇒ 流已错位\n",
                     rnet::kind_name(h.kind), (unsigned long long)h.nbytes, bytes);
        dp_broken_ = true;
        return false;
    }
    if (!sock_.recv_all(buf, bytes)) {
        dp_broken_ = true;
        std::fprintf(stderr, "[REMOTE-DP] FATAL: allreduce 回包体读取失败 ⇒ 流已错位\n");
        return false;
    }
    return true;
}

// ============================================================
// RemoteBackend
// ============================================================
bool RemoteBackend::supports_op(TensorF32* node) const {
    if (!node) return false;
    const char* list = std::getenv("PPML_REMOTE_OPS");
    if (!list || !*list) return false;   // 默认不分配任何节点（安全：需要显式开启）
    if (std::strcmp(list, "all") == 0) return true;
    // 形如 "1,7,33"：tensor_op 数值列表
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s", list);
    char* save = nullptr;
    for (char* tok = ::strtok_r(buf, ",", &save); tok; tok = ::strtok_r(nullptr, ",", &save)) {
        if (std::atoi(tok) == (int)node->op) return true;
    }
    return false;
}

Status RemoteBackend::graph_compute(ComputeGraph* cg) {
    if (!cg || cg->n_nodes() <= 0) return Status::SUCCESS;
    if (!cli_ || !cli_->connected()) {
        std::fprintf(stderr, "[REMOTE] graph_compute: 未连接对端\n");
        if (rstrict()) rstrict_fail("远端后端被分到节点，但连接不可用");
        return Status::ABORTED;
    }

    std::vector<rnet::RNodeDesc> descs;
    std::vector<uint64_t> outs;

    // 本 split 内所有节点的 id
    std::unordered_map<const TensorF32*, uint64_t> ids;
    ids.reserve((size_t)cg->n_nodes() * 2 + 16);
    for (int i = 0; i < cg->n_nodes(); ++i) {
        TensorF32* n = cg->graph_node(i);
        if (!n) continue;
        ids[n] = cli_->id_for(n);
    }

    // 收集"外部输入"（不属于本 split 的 src）：跨机搬运的入口
    std::vector<const TensorF32*> inputs;
    for (int i = 0; i < cg->n_nodes(); ++i) {
        TensorF32* n = cg->graph_node(i);
        if (!n) continue;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            TensorF32* s = n->src[j];
            if (!s || ids.count(s)) continue;
            if (std::find(inputs.begin(), inputs.end(), s) != inputs.end()) continue;
            inputs.push_back(s);
            uint64_t sid = cli_->id_for(s);
            ids[s] = sid;

            // a) 跨后端拷贝节点（OP_DUP）：数据已由 RemoteBuffer::set_tensor 上传过 ⇒ 标记为远端
            // b) 叶子（参数/常量/输入）：上传一次（参数在 optimizer.step() 后由 invalidate 触发重传）
            uint64_t up_ret = 0;
            bool marked = false;
            if (s->op == OP_DUP) {
                cli_->mark_remote(s, sid);
                marked = true;
            } else if (s->data() && s->nbytes() > 0) {
                up_ret = cli_->upload_tensor(s, s->data(), (size_t)s->nbytes());
                // 参数/常量叶子：对端那份要活过迭代边界（激活/输入则每轮作废重传）
                if (s->flag & (TENSOR_FLAG_PARAM | TENSOR_FLAG_CONST)) {
                    cli_->mark_persistent(sid);
                }
            }
            if (rtrace()) {
                char sh[64]; rt_dims(s, sh, sizeof(sh));
                std::fprintf(stderr,
                             "[RT] INPUT id=%llu op=%d nb=%zu shape=[%s] data=%p buf=%p "
                             "view_src=%p is_dup=%d up_ret=%llu marked=%d\n",
                             (unsigned long long)sid, (int)s->op, (size_t)s->nbytes(), sh,
                             (const void*)s->data(), (const void*)s->buffer_,
                             (const void*)s->view_src, (int)(s->op == OP_DUP),
                             (unsigned long long)up_ret, (int)marked);
            }
        }
    }

    // 节点描述（src 全部用 id 表达；不传任何本地指针）
    descs.resize((size_t)cg->n_nodes());
    for (int i = 0; i < cg->n_nodes(); ++i) {
        TensorF32* n = cg->graph_node(i);
        rnet::RNodeDesc& d = descs[(size_t)i];
        if (!n) continue;
        d.op   = (uint32_t)n->op;
        d.type = (uint32_t)n->type;
        d.ndim = n->shape().ndim();
        for (int k = 0; k < 4; ++k) d.dims[k] = n->shape().dim(k);
        std::memcpy(d.op_params, n->op_params, sizeof(d.op_params));
        d.id = ids[n];
        int nsrc = 0;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            TensorF32* s = n->src[j];
            if (!s) continue;
            if (j < rnet::R_MAX_SRC) {
                d.src_ids[j] = ids[s];
                nsrc = j + 1;
            }
        }
        d.n_src = nsrc;
    }

    // 输出：本 split 中被"外部"消费的节点（保守起见，取该 split 的最后 1/4 节点 + 所有无消费者）
    //   说明：v1 直接把**所有节点**登记为可按需 fetch 的候选（对端结果全留在 arena 直到下次 compute），
    //         这里 outs 只用于让对端知道"优先保留"哪些（当前实现保留全部）。
    for (int i = 0; i < cg->n_nodes(); ++i) {
        TensorF32* n = cg->graph_node(i);
        if (n) outs.push_back(ids[n]);
    }

    if (!cli_->compute_graph(descs, outs)) {
        std::fprintf(stderr, "[REMOTE] graph_compute 失败：对端未回结果（nodes=%d）⇒ 调度器大概率回落本地\n",
                     cg->n_nodes());
        if (rstrict()) rstrict_fail("远端 graph_compute 失败");
        return Status::ABORTED;
    }

    // 本 split 的所有节点数据此后都在对端：get_tensor 会走 TENSOR_GET（按需拉取边界张量）
    for (int i = 0; i < cg->n_nodes(); ++i) {
        TensorF32* n = cg->graph_node(i);
        if (n) cli_->mark_remote(n, ids[n]);
    }
    if (std::getenv("GRAPH_DEBUG_REMOTE")) {
        std::fprintf(stderr, "[REMOTE] graph_compute: nodes=%d inputs=%zu outs=%zu (sent=%.2f MB)\n",
                     cg->n_nodes(), inputs.size(), outs.size(),
                     (double)cli_->bytes_sent() / (1024.0 * 1024.0));
    }
    return Status::SUCCESS;
}

// ============================================================
// RemoteServer
// ============================================================
bool RemoteServer::listen(int port) {
    // 【2026-09-14 晚】主淘汰机制 = 客户端在迭代边界发 RESULTS_CLEAR（见 handle_results_clear）。
    //   下面两个参数只作兜底：客户端没调 hook / 崩了时，防止对端结果无界增长。
    //   注意 keep_splits_ 太小会重现 OPS=47 的老问题（K=16 < 一轮 forward 的远端 split 数
    //   ⇒ backward 取 forward 激活时未命中 abort），默认给足。
    if (const char* k = std::getenv("PPML_REMOTE_KEEP_SPLITS")) {
        const int v = std::atoi(k);
        if (v > 0) keep_splits_ = (size_t)v;
    }
    if (const char* m = std::getenv("PPML_REMOTE_KEEP_MB")) {
        const long v = std::atol(m);
        if (v > 0) keep_bytes_ = (size_t)v * 1024 * 1024;
    }
    return listener_.listen_on(port);
}

bool RemoteServer::handle_tensor_set(rnet::RSocket& s, rnet::RHeader& h) {
    // 【加固 2026-09-16】原实现 `std::vector<uint8_t> body(h.nbytes)` 未捕获异常 ⇒ 大张量/内存不足时
    //   std::bad_alloc ⇒ std::terminate ⇒ 进程直接消失（客户端只看到 "peer closed"，对端零线索 ✗）。
    //   现在：捕获 + 回 ERROR_MSG(405) 告知调用方，服务继续存活。
    std::vector<uint8_t> body;
    try {
        body.resize((size_t)h.nbytes);
    } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "[REMOTE-SRV] !!! 分配 %.1f MB 失败（id=%llu，RSS=%.0f MB）：%s ⇒ 回 ERROR_MSG(405)\n",
                     (double)h.nbytes / 1048576.0, (unsigned long long)h.tensor_id, cur_rss_mb(), e.what());
        rnet::RHeader err;
        err.kind      = (uint32_t)rnet::RKind::ERROR_MSG;
        err.tensor_id = h.tensor_id;
        err.aux0      = 405;
        return s.send_header(err);
    }
    if (h.nbytes && !s.recv_all(body.data(), h.nbytes)) return false;
    arena_[h.tensor_id] = std::move(body);
    arena_bytes_ += h.nbytes;
    std::vector<int64_t> dims;
    for (int i = 0; i < h.ndim && i < 4; ++i) dims.push_back(h.dims[i]);
    arena_dims_[h.tensor_id] = std::move(dims);
    arena_dtype_[h.tensor_id] = (rnet::RDtype)h.dtype;
    rnet::RHeader ack;
    ack.kind      = (uint32_t)rnet::RKind::TENSOR_SET_ACK;
    ack.tensor_id = h.tensor_id;
    return s.send_header(ack);
}

bool RemoteServer::handle_tensor_get(rnet::RSocket& s, rnet::RHeader& h) {
    auto it = results_.find(h.tensor_id);
    if (it == results_.end()) it = arena_.find(h.tensor_id);
    if (it == results_.end()) {
        std::fprintf(stderr, "[REMOTE-SRV] TENSOR_GET 未命中 id=%llu\n",
                     (unsigned long long)h.tensor_id);
        rnet::RHeader err;
        err.kind = (uint32_t)rnet::RKind::ERROR_MSG;
        err.aux0 = 404;
        return s.send_header(err);
    }
    return s.send_data((uint32_t)rnet::RKind::TENSOR_DATA, h.tensor_id, rnet::RDtype::F32,
                       1, nullptr, it->second.data(), it->second.size());
}

// 迭代边界：丢弃上一轮算出的结果，只保留 keep（参数/常量）id 的 arena 项
bool RemoteServer::handle_results_clear(rnet::RSocket& s, rnet::RHeader& h) {
    const size_t n = (size_t)h.aux0;
    std::vector<uint64_t> keep(n);
    if (n && !s.recv_all(keep.data(), n * sizeof(uint64_t))) return false;
    std::unordered_set<uint64_t> keep_set(keep.begin(), keep.end());

    const uint64_t dropped_results = (uint64_t)results_.size();
    results_.clear();
    recent_ids_.clear();
    results_bytes_ = 0;

    uint64_t dropped_arena = 0;
    for (auto it = arena_.begin(); it != arena_.end();) {
        if (keep_set.count(it->first)) { ++it; continue; }
        arena_bytes_ -= it->second.size();
        arena_dims_.erase(it->first);
        arena_dtype_.erase(it->first);
        it = arena_.erase(it);
        ++dropped_arena;
    }

    rnet::RHeader ack;
    ack.kind = (uint32_t)rnet::RKind::RESULTS_CLEAR_ACK;
    ack.aux0 = dropped_results;
    ack.aux1 = dropped_arena;
    if (rtrace()) {
        std::fprintf(stderr,
                     "[RT-SRV] RESULTS_CLEAR: keep=%zu 丢弃 results=%llu arena=%llu "
                     "（arena 余 %zu 项 / %.2f MB）\n",
                     n, (unsigned long long)dropped_results, (unsigned long long)dropped_arena,
                     arena_.size(), (double)arena_bytes_ / (1024.0 * 1024.0));
    }
    return s.send_header(ack);
}

bool RemoteServer::handle_allreduce(rnet::RSocket& s, rnet::RHeader& h, int world_size) {
    // 星型 root：收 worker 的缓冲 → 加上本端贡献 → 广播
    const size_t n = h.nbytes / sizeof(float);
    std::vector<float> buf(n);
    if (!s.recv_all(buf.data(), h.nbytes)) return false;
    // 阶段 A：优先取"按调用次序登记的贡献"（peer 双端训练时本端也有真实梯度/权重）；
    //   未启用队列（旧路径：独立张量服务 + set_dp_buffer）⇒ 退回平铺 dp_local_ 语义（兼容 ✓）。
    std::vector<float> local;
    if (dp_queue_enabled()) {
        if (!pop_dp_contribution(local)) {
            std::fprintf(stderr, "[REMOTE-SRV] allreduce 贡献超时（队列为空）⇒ 本次按 0 贡献\n");
            local.assign(n, 0.0f);
        }
        if (local.size() != n) {
            std::fprintf(stderr, "[REMOTE-SRV] allreduce 贡献尺寸不匹配：local=%zu 收到=%zu ⇒ 按 0 贡献\n",
                         local.size(), n);
            local.assign(n, 0.0f);
        }
    } else {
        local.assign(n, 0.0f);
        for (size_t i = 0; i < n && i < dp_local_.size(); ++i) local[i] = dp_local_[i];
    }
    for (size_t i = 0; i < n; ++i) buf[i] += local[i];
    dp_local_ = buf;   // 本端也拿到求和结果
    if (dp_queue_enabled()) publish_dp_result(buf);   // rank0 训练线程 wait_dp_result 取回
    for (int r = 1; r < world_size; ++r) {
        if (!s.send_data((uint32_t)rnet::RKind::ALLREDUCE_DOWN, 0, rnet::RDtype::F32,
                         1, nullptr, buf.data(), h.nbytes)) return false;
    }
    return true;
}

bool RemoteServer::handle_graph_compute(rnet::RSocket& s, rnet::RHeader& h) {
    const uint32_t n_nodes = (uint32_t)h.aux0;
    const uint32_t n_outs  = (uint32_t)h.aux1;
    std::vector<uint8_t> body(h.nbytes);
    if (h.nbytes && !s.recv_all(body.data(), h.nbytes)) return false;
    if (n_nodes == 0) {
        rnet::RHeader ok;
        ok.kind = (uint32_t)rnet::RKind::GRAPH_RESULT;
        return s.send_header(ok);
    }
    const rnet::RNodeDesc* nd = reinterpret_cast<const rnet::RNodeDesc*>(body.data());
    (void)n_outs;   // v1：输出集合由"边界节点"推导（见下方 consumed 判定），不依赖客户端声明

    PPMLContext* ctx = &context();
    // 【2026-09-14 晚 修正】每次执行都新建 ComputeGraph/张量却从不回收 ⇒ 上下文对象数单调增长，
    //   几十次远端子图后必崩（实测 "Context memory exhausted" abort，约 74 次）。
    //   结果已拷进 results_/返回客户端，图与张量可安全回收 ⇒ RAII 回退到入口水位（覆盖所有 return）。
    struct ObjReset {
        PPMLContext* c = nullptr;
        void* mark = nullptr;
        ~ObjReset() { if (c && mark) c->reset_objects_to(mark); }
    } obj_reset{ctx, ctx->mark_objects()};
    ComputeGraph* cg = ComputeGraph::new_graph(ctx);
    std::unordered_map<uint64_t, TensorF32*> by_id;

    // 1) 输入张量：从 arena 取出形状，建"空壳"节点（数据在下面统一分配后回填）
    std::unordered_set<uint64_t> defined;
    for (uint32_t i = 0; i < n_nodes; ++i) defined.insert(nd[i].id);
    std::vector<uint64_t> input_ids;
    for (uint32_t i = 0; i < n_nodes; ++i) {
        for (int j = 0; j < nd[i].n_src; ++j) {
            uint64_t sid = nd[i].src_ids[j];
            if (!sid || defined.count(sid) || by_id.count(sid)) continue;
            auto dit = arena_dims_.find(sid);
            if (arena_.find(sid) == arena_.end() || dit == arena_dims_.end()) {
                std::fprintf(stderr, "[REMOTE-SRV] 缺输入张量 id=%llu\n", (unsigned long long)sid);
                rnet::RHeader err;
                err.kind = (uint32_t)rnet::RKind::ERROR_MSG;
                err.aux0 = 400;
                return s.send_header(err);
            }
            int64_t dims[4] = {1, 1, 1, 1};
            const int ndim = (int)dit->second.size();
            for (int k = 0; k < ndim && k < 4; ++k) dims[k] = dit->second[k];
            by_id[sid] = ctx->new_tensor<float>(ndim > 0 ? ndim : 1, dims);
            input_ids.push_back(sid);
        }
    }

    // 2) 按描述重建节点（只传结构，不传指针）
    for (uint32_t i = 0; i < n_nodes; ++i) {
        const rnet::RNodeDesc& d = nd[i];
        TensorF32* t = ctx->new_tensor<float>((int)(d.ndim > 0 ? d.ndim : 1), d.dims);
        t->op   = (tensor_op)d.op;
        t->type = (tensor_type)d.type;
        std::memcpy(t->op_params, d.op_params, sizeof(d.op_params));
        for (int j = 0; j < d.n_src && j < GGML_MAX_SRC; ++j) {
            if (!d.src_ids[j]) continue;
            auto it = by_id.find(d.src_ids[j]);
            if (it == by_id.end()) {
                std::fprintf(stderr, "[REMOTE-SRV] src id=%llu 未知（节点 %u）\n",
                             (unsigned long long)d.src_ids[j], i);
                rnet::RHeader err;
                err.kind = (uint32_t)rnet::RKind::ERROR_MSG;
                err.aux0 = 401;
                return s.send_header(err);
            }
            t->src[j] = it->second;
        }
        by_id[d.id] = t;
    }

    // 3) 建图（所有节点都 expand，保证后续按需 fetch 任意节点都有效）
    for (uint32_t i = 0; i < n_nodes; ++i) {
        auto it = by_id.find(nd[i].id);
        if (it != by_id.end()) cg->build_forward_expand(it->second);
    }

    // 3b) **自建 bump 分配**（与 BackendScheduler::reserve_graph_memory 同模式）：
    //     统一给所有张量分配 host buffer 并绑定，然后回填输入数据，最后用 skip_alloc 执行。
    //     若让本地 backend 自己跑 gallocr，它会把输入张量重绑到新 buffer → 数据丢失（实测全 0）。
    size_t total = 0;
    for (auto& kv : by_id) total += GGML_PAD((size_t)kv.second->nbytes(), 64);
    if (total == 0) total = 64;
    auto big_buf = std::make_unique<DefaultBuffer>(CPUBufferType::instance(), total);
    {
        TensorAllocator ta(big_buf.get());
        for (auto& kv : by_id) {
            if (!ta.alloc(kv.second)) {
                std::fprintf(stderr, "[REMOTE-SRV] 分配失败（total=%zu）\n", total);
                rnet::RHeader err;
                err.kind = (uint32_t)rnet::RKind::ERROR_MSG;
                err.aux0 = 403;
                return s.send_header(err);
            }
        }
    }
    for (uint64_t sid : input_ids) {
        auto ait = arena_.find(sid);
        auto tit = by_id.find(sid);
        if (ait == arena_.end() || tit == by_id.end() || !tit->second->data()) continue;
        std::memcpy(tit->second->data(), ait->second.data(), ait->second.size());
    }

    // v1：服务端固定用本地 CPU 后端执行（对端机器可后续换 CUDA：set_use_cuda(true) 走同一接口）
    static std::unique_ptr<Backend> srv_backend;
    if (!srv_backend) {
        const char* nt = std::getenv("PPML_REMOTE_SRV_THREADS");
        srv_backend = std::make_unique<CPUBackend>(nt ? std::atoi(nt) : 4);
        std::fprintf(stderr, "[REMOTE-SRV] 执行后端 = CPU(%d 线程)\n",
                     nt ? std::atoi(nt) : 4);
    }
    srv_backend->set_skip_alloc(true);     // 分配已由上面完成，禁止后端再跑 gallocr
    Status st = srv_backend->graph_compute(cg);
    srv_backend->set_skip_alloc(false);
    if (st != Status::SUCCESS) {
        std::fprintf(stderr, "[REMOTE-SRV] 本地执行失败 status=%d\n", (int)st);
        rnet::RHeader err;
        err.kind = (uint32_t)rnet::RKind::ERROR_MSG;
        err.aux0 = 402;
        return s.send_header(err);
    }

    // 4) 结果落 arena —— **保留本次执行的每一个节点**。
    //    【2026-09-14 修正】此前只保留"边界输出"（split 内无消费者）是为了省内存，但那与
    //    客户端的语义冲突：客户端把 split 内**所有**节点都标记为"数据在对端"，跨 split 的
    //    消费者（例如 backward 节点要读 forward 的中间激活）随后就可能取不到 ⇒ 取回失败。
    //    现在全量保留，并用"最近 K 次执行"的 LRU 控制内存（K 由 PPML_REMOTE_KEEP_SPLITS 控制，
    //    默认 16；一次执行的节点输出通常只有几 MB）。
    std::vector<uint64_t> this_ids;
    this_ids.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; ++i) {
        const uint64_t nid = nd[i].id;
        auto it = by_id.find(nid);
        if (it == by_id.end() || !it->second) continue;
        TensorF32* t = it->second;
        // 【2026-09-14 修正】view 节点自身没有 buffer_/data_（数据在 view_src 链底层）：
        //   直接判空会跳过它 ⇒ 客户端之后按 id 取回时"未命中"（实测 id=4 触发 abort）。
        //   这里回溯到真实载体读取，长度以本节点的 nbytes 为准（本项目 view 均为同字节视图）。
        const TensorF32* real = t;
        int guard = 0;
        while (real && !real->buffer_ && !real->data() && real->view_src && guard++ < 64) {
            real = real->view_src;
        }
        const size_t nb = (size_t)t->nbytes();
        std::vector<uint8_t> out(nb);
        if (real && real->buffer_) {
            if (rtrace() && real != t) {
                std::fprintf(stderr, "[RT-SRV] VIEW-BACK id=%llu op=%d nb=%zu real=%p\n",
                             (unsigned long long)nid, (int)t->op, nb, (const void*)real);
            }
            real->buffer_->get_tensor(real, out.data(), real->buffer_offs_, nb);
        } else if (real && real->data()) {
            std::memcpy(out.data(), real->data(), nb);
        } else {
            // 实在取不到（例如常量叶子）：留给客户端报"未命中"
            if (rtrace()) {
                char sh[64]; rt_dims(t, sh, sizeof(sh));
                std::fprintf(stderr,
                             "[RT-SRV] SKIP(无 buffer/无 data) id=%llu op=%d nb=%zu shape=[%s] "
                             "flags=%d real=%p real_buf=%p real_data=%p\n",
                             (unsigned long long)nid, (int)t->op, nb, sh, (int)t->flag,
                             (const void*)real, (const void*)(real ? real->buffer_ : nullptr),
                             (const void*)(real ? real->data() : nullptr));
            }
            continue;
        }
        // 判据（PPML_REMOTE_TRACE）：同一份输入，对端 kernel 输出 vs 手写参考 softmax 的 max|Δ|
        //   Δ 大  ⇒ 对端 kernel 的问题（形状/轴/实现）；
        //   Δ≈0   ⇒ kernel 没问题 ⇒ 差异来自**输入**（上传/注入阶段的字节不同）。
        if (rtrace() && t->op == OP_SOFT_MAX && t->src[0] && t->src[0]->nbytes() > 0) {
            TensorF32* in = t->src[0];
            TensorF32* rin = in;
            int g = 0;
            while (rin && !rin->buffer_ && !rin->data() && rin->view_src && g++ < 64) rin = rin->view_src;
            std::vector<float> x((size_t)in->numel());
            bool ok_in = false;
            if (rin && rin->buffer_) {
                rin->buffer_->get_tensor(rin, x.data(), rin->buffer_offs_, x.size() * sizeof(float));
                ok_in = true;
            } else if (rin && rin->data()) {
                std::memcpy(x.data(), rin->data(), x.size() * sizeof(float));
                ok_in = true;
            }
            const int64_t D = t->shape().dim(0);
            if (ok_in && D > 0 && (int64_t)x.size() == t->numel()) {
                const int64_t rows = (int64_t)x.size() / D;
                const float* y = reinterpret_cast<const float*>(out.data());
                double maxd = 0.0;
                for (int64_t r = 0; r < rows; ++r) {
                    const float* xr = x.data() + r * D;
                    const float* yr = y + r * D;
                    float m = -INFINITY;
                    for (int64_t k = 0; k < D; ++k) m = std::max(m, xr[k]);
                    float s = 0.0f;
                    for (int64_t k = 0; k < D; ++k) s += std::exp(xr[k] - m);
                    for (int64_t k = 0; k < D; ++k) {
                        const double ref = (double)std::exp(xr[k] - m) / (double)s;
                        maxd = std::max(maxd, std::fabs(ref - (double)yr[k]));
                    }
                }
                // 输入字节指纹：与客户端上传时的 [RT] SET-OK fnv、以及单机的 [HASH-IN] 三方对拍
                uint64_t hin = 1469598103934665603ull;
                for (size_t bi = 0; bi < x.size() * sizeof(float); ++bi) {
                    hin ^= reinterpret_cast<const uint8_t*>(x.data())[bi];
                    hin *= 1099511628211ull;
                }
                std::fprintf(stderr,
                             "[RT-SRV] SOFTMAX-RECHECK id=%llu D=%lld rows=%lld max_abs_diff=%.3e "
                             "in_fnv=%016llx\n",
                             (unsigned long long)nid, (long long)D, (long long)rows, maxd,
                             (unsigned long long)hin);
            }
        }
        // 指纹（PPML_REMOTE_TRACE）：对端自己算出的值 ⇒ 与客户端 FETCH-HASH / 单机 HASH 三方对拍
        if (rtrace()) {
            uint64_t hh = 1469598103934665603ull;
            for (size_t bi = 0; bi < nb; ++bi) { hh ^= out[bi]; hh *= 1099511628211ull; }
            std::fprintf(stderr, "[RT-SRV] OUT-HASH id=%llu op=%d nb=%zu fnv=%016llx\n",
                         (unsigned long long)nid, (int)t->op, nb, (unsigned long long)hh);
        }
        // 【2026-09-14 晚】同一 id 可能被重算（同 split 多次执行）⇒ 先扣掉旧字节再记新的
        auto prev = results_.find(nid);
        if (prev != results_.end()) results_bytes_ -= prev->second.size();
        results_[nid] = std::move(out);
        results_bytes_ += nb;
        this_ids.push_back(nid);
    }
    // 兜底淘汰（主机制是客户端 RESULTS_CLEAR）：超 K 次执行 或 超字节上限时，从最老的一次开始丢
    recent_ids_.push_back(std::move(this_ids));
    while (!recent_ids_.empty() &&
           (recent_ids_.size() > keep_splits_ || results_bytes_ > keep_bytes_)) {
        for (uint64_t old : recent_ids_.front()) {
            auto it = results_.find(old);
            if (it == results_.end()) continue;
            results_bytes_ -= it->second.size();
            results_.erase(it);
            if (rtrace()) {
                std::fprintf(stderr, "[RT-SRV] EVICT id=%llu（兜底窗口 keep_splits=%zu bytes=%zu）\n",
                             (unsigned long long)old, keep_splits_, results_bytes_);
            }
        }
        recent_ids_.pop_front();
    }
    rnet::RHeader ok;
    ok.kind = (uint32_t)rnet::RKind::GRAPH_RESULT;
    ok.aux0 = (uint64_t)results_.size();
    if (std::getenv("GRAPH_DEBUG_REMOTE")) {
        std::fprintf(stderr, "[REMOTE-SRV] graph_compute ok: nodes=%u results=%zu arena=%.2f MB\n",
                     n_nodes, results_.size(), (double)arena_bytes_ / (1024.0 * 1024.0));
    }
    return s.send_header(ok);
}

// 启动屏障标记：服务线程 accept 到连接后置 true；训练线程（root）据此等待对端就位（bool 写实际原子 ✓）。
//   注意：必须定义在 serve_forever 之前（它在文件中更靠前）✗。
static volatile bool g_dp_peer_seen = false;

bool RemoteServer::serve_forever(int world_size, int max_messages) {
    // 【修复 2026-09-15】accept 循环：原实现只服务**一条**连接，recv 失败即 `return` ✗
    //   ⇒ 多 epoch 场景下 epoch1 的 DP 一结束、客户端断开，rank0 的服务就消失
    //   （对端随即看到 `Connection reset by peer` / `Broken pipe`）⇒ epoch2 的 DP 全部失败 ✗。
    //   现在：连接断开后**继续 accept 下一个连接**（listener 常驻），直到 BYE 或 listener 坏掉。
    int accept_fail = 0;
    bool first_conn = true;    // 【T1】用于"客户端重连后清空上一轮 results"
    for (;;) {
        rnet::RSocket conn;
        if (!listener_.accept_one(conn)) {
            if (++accept_fail > 1000) return false;      // 连续失败过多（listener 已坏）才退出
            struct timespec ts{0, 50 * 1000 * 1000};     // 50ms，避免忙等
            nanosleep(&ts, nullptr);
            continue;
        }
        accept_fail = 0;
        g_dp_peer_seen = true;   // 启动屏障：通知 root 训练线程"对端已连接"
        if (!first_conn) {
            // 【T1 2026-09-16】客户端重连：上一轮的 results_ 已无意义（id 不会再被引用；客户端已 invalidate_all）
            //   ⇒ 必须立刻丢掉，否则新连接的第一次取回可能命中**旧结果**（静默错值 ✗）
            std::fprintf(stderr, "[REMOTE-SRV] 客户端重连：清空上一轮 results（%zu 条 / %.1f MB）\n",
                         results_.size(), (double)results_bytes_ / 1048576.0);
            results_.clear();
            recent_ids_.clear();
            results_bytes_ = 0;
        }
        first_conn = false;
        std::fprintf(stderr, "[REMOTE-SRV] 客户端已连接（world=%d）\n", world_size);
        for (int msg = 0; msg < max_messages; ++msg) {
            rnet::RHeader h;
            if (!conn.recv_header(h)) {
                // 对端断开（正常结束一个 epoch 的 DP、或崩溃）⇒ 回到外层继续 accept，不结束服务
                std::fprintf(stderr, "[REMOTE-SRV] 连接断开，继续等待下一个连接（多 epoch 需要）\n");
                break;
            }
            // 【加固 2026-09-16】每条消息一行（PPML_REMOTE_TRACE=1）：出问题时最后一行就是死因现场
            if (rtrace()) {
                std::fprintf(stderr, "[RT-SRV] msg=%s id=%llu nb=%.2f MB rss=%.0f MB arena=%.1f MB results=%zu\n",
                             rnet::kind_name(h.kind), (unsigned long long)h.tensor_id,
                             (double)h.nbytes / 1048576.0, cur_rss_mb(),
                             (double)arena_bytes_ / 1048576.0, results_.size());
            }
        bool conn_broken = false;
        try {
        switch ((rnet::RKind)h.kind) {
            case rnet::RKind::HELLO: {
                rnet::RHeader ack;
                ack.kind = (uint32_t)rnet::RKind::HELLO_ACK;
                ack.aux0 = (uint64_t)world_size;
                ack.aux1 = 0;
                if (!conn.send_header(ack)) conn_broken = true;
                break;
            }
            case rnet::RKind::PING: {
                rnet::RHeader pong;
                pong.kind = (uint32_t)rnet::RKind::PONG;
                if (!conn.send_header(pong)) conn_broken = true;
                break;
            }
            case rnet::RKind::TENSOR_SET:
                if (!handle_tensor_set(conn, h)) conn_broken = true;
                break;
            case rnet::RKind::TENSOR_GET:
                if (!handle_tensor_get(conn, h)) conn_broken = true;
                break;
            case rnet::RKind::GRAPH_COMPUTE:
                if (!handle_graph_compute(conn, h)) conn_broken = true;
                break;
            case rnet::RKind::ALLREDUCE_UP:
                if (!handle_allreduce(conn, h, world_size)) conn_broken = true;
                break;
            case rnet::RKind::RESULTS_CLEAR:
                if (!handle_results_clear(conn, h)) conn_broken = true;
                break;
            case rnet::RKind::BYE:
                std::fprintf(stderr, "[REMOTE-SRV] BYE（tensors=%zu, arena=%.2f MB）\n",
                             arena_.size(), (double)arena_bytes_ / (1024.0 * 1024.0));
                return true;
            default:
                // 未知消息：只记录并跳过（原实现直接 return false ⇒ 杀掉常驻服务 ✗，多 epoch 不可接受）
                std::fprintf(stderr, "[REMOTE-SRV] 未知消息 %s（%u）⇒ 跳过\n",
                             rnet::kind_name(h.kind), h.kind);
                break;
        }
        } catch (const std::exception& e) {
            // 【加固 2026-09-16】异常**不许杀掉常驻服务**（原实现会 std::terminate ⇒ 客户端只见 peer closed ✗）
            std::fprintf(stderr,
                         "[REMOTE-SRV] !!! 处理 %s（id=%llu nb=%.2f MB）抛异常：%s\n",
                         rnet::kind_name(h.kind), (unsigned long long)h.tensor_id,
                         (double)h.nbytes / 1048576.0, e.what());
            conn_broken = true;
        } catch (...) {
            std::fprintf(stderr, "[REMOTE-SRV] !!! 处理 %s 抛未知异常 ⇒ 断开本连接\n",
                         rnet::kind_name(h.kind));
            conn_broken = true;
        }
        if (conn_broken) {
            // 只断这一条连接，服务继续 accept（多 epoch / 客户端重启用）
            std::fprintf(stderr, "[REMOTE-SRV] 本连接终止 ⇒ 继续等待下一个连接（RSS=%.0f MB）\n", cur_rss_mb());
            conn.close();
            break;
        }
        }   // msg 循环
        // 本连接结束（recv 失败 / 达到 max_messages）⇒ 回到外层 accept 继续服务（多 epoch 必需）
    }       // accept 循环（无限，直到 BYE 或 listener 坏）
    return true;
}

// ============================================================
// env 便捷构造
// ============================================================
std::shared_ptr<RemoteClient> remote_make_client_from_env() {
    const char* host = std::getenv("PPML_REMOTE_HOST");
    if (!host) return nullptr;
    const int port = std::getenv("PPML_REMOTE_PORT") ? std::atoi(std::getenv("PPML_REMOTE_PORT")) : 2244;
    const int rank = std::getenv("PPML_REMOTE_RANK") ? std::atoi(std::getenv("PPML_REMOTE_RANK")) : 1;
    const int world = std::getenv("PPML_REMOTE_WORLD") ? std::atoi(std::getenv("PPML_REMOTE_WORLD")) : 2;
    auto cli = std::make_shared<RemoteClient>();
    if (!cli->connect(host, port, world, rank)) {
        // 【T1 2026-09-16】STRICT 下"连不上"必须响亮失败：否则整轮其实在本地跑，却以为远端生效了 ✗
        if (rstrict()) rstrict_fail("远端后端初始化失败：connect 不上对端");
        return nullptr;
    }
    return cli;
}

RemoteBackend* remote_make_backend_from_env() {
    static std::shared_ptr<RemoteClient> cli;
    static std::unique_ptr<RemoteBackend> be;
    // 【修复 2026-09-15】失败也要**记忆**：原实现 `!be` 时每次调用都重试 connect ✗，
    //   而 `remote_iteration_begin()/remote_after_optimizer_step()` 每轮迭代都会调本函数
    //   ⇒ 只设了 PPML_REMOTE_HOST 而对端不存在时，会持续发起半开（EINPROGRESS）连接 ✗
    //   ⇒ 实测：单进程 + `PPML_REMOTE_HOST` + 2 epoch **必崩**（与 DP/分片无关，是既存 bug）。
    //   这里改为"一次失败即本进程内不再重试"（与 remote_dp_client_if_enabled 的 static tried 语义一致）。
    static bool tried_fail = false;
    if (!be) {
        if (tried_fail) return nullptr;
        cli = remote_make_client_from_env();
        if (!cli) {
            tried_fail = true;
            std::fprintf(stderr,
                         "[REMOTE] 远端建连失败 ⇒ 本进程内**不再重试**（避免每轮迭代发起半开连接；"
                         "如需远端能力请确认 PPML_REMOTE_HOST/PORT 与对端已就绪）\n");
            return nullptr;
        }
        const bool slim = std::getenv("PPML_REMOTE_SLIM") &&
                          std::atoi(std::getenv("PPML_REMOTE_SLIM")) != 0;

        // 优先级（决定 scheduler 把节点分给谁）：
        //   * 只有显式给了 PPML_REMOTE_OPS（= 真打算把某些 op 放远端）时才高于 CPU(0)/CUDA(1)，
        //     默认 100，可用 PPML_REMOTE_PRIORITY 覆盖；
        //   * 未给 PPML_REMOTE_OPS ⇒ -1（最低），且 supports_op 全 false ⇒ 完全不抢节点（惰性注册）。
        int prio = -1;
        if (const char* ops = std::getenv("PPML_REMOTE_OPS")) {
            if (*ops) {
                const char* p = std::getenv("PPML_REMOTE_PRIORITY");
                prio = p ? std::atoi(p) : 100;
            }
        }
        be = std::make_unique<RemoteBackend>(cli, prio);
        be->set_slim(slim);
        std::fprintf(stderr, "[REMOTE] RemoteBackend 已创建（slim=%d, ops=%s, priority=%d）\n",
                     (int)slim,
                     std::getenv("PPML_REMOTE_OPS") ? std::getenv("PPML_REMOTE_OPS") : "none",
                     prio);
    }
    return be.get();
}

// 数据并行：PPML_DP=1 时返回与管线并行复用的对端连接（首次调用建连；失败返回 nullptr ⇒ 调用点静默跳过）
RemoteClient* remote_dp_client_if_enabled() {
    const char* dp = std::getenv("PPML_DP");
    if (!dp || std::atoi(dp) == 0) return nullptr;
    static std::shared_ptr<RemoteClient> dp_cli;
    // 【修复 2026-09-15】去掉"失败即锁死"（static tried）：对端（acceptor）可能晚于本端启动，
    //   首次 connect 失败（EINPROGRESS/refused）后就再也不重试 ⇒ 本端整场跳过 allreduce ✗
    //   （实测：按 rank 分片时 rank1 在训练开头就 connect，而 rank0 要等第一次 optimizer init 才 listen ✗）。
    //   改为：尚未连上就**每次调用重试**（connect 到未监听端口是廉价快速失败，不会挂）。
    if (!dp_cli || !dp_cli->connected()) {
        dp_cli = remote_make_client_from_env();   // 与 PP 共用 PPML_REMOTE_HOST/PORT
        if (dp_cli) {
            std::fprintf(stderr, "[REMOTE-DP] 数据并行已启用（world=%d rank=%d）；"
                                 "Allreduce 位置在 clip 之前\n",
                         dp_cli->world_size(), dp_cli->rank());
        } else {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr, "[REMOTE-DP] PPML_DP=1 但暂未连上对端"
                                     "（将随调用重试；需设 PPML_REMOTE_HOST/PORT）\n");
            }
        }
    }
    return dp_cli.get();
}

// ============================================================
// 数据并行：对 PARAM 梯度做 Allreduce 平均（在 optimizer.step() 之前调用一次）
// ============================================================
// ============================================================
// 阶段 A：rank0（acceptor / star root）侧的 DP 同步
//   背景（硬约束，见 PLAN_SHARDING.md §5.5）：`RemoteServer::handle_allreduce` 按"第 k 次调用"配对
//   本端贡献 ⇒ 两端必须对同一参数各调用一次、顺序一致；rank0 不走客户端 UP 路径（它是被连的一方），
//   而是：push_dp_contribution(本端值) → 服务线程收到对端 UP 后求和并回放 → 本端 wait_dp_result 取回。
//   两端遍历顺序都是"参数全局序号"（与 AdamW::init_from_graph 同规则）⇒ 配对天然一致 ✓。
// ============================================================
static RemoteServer* g_dp_root_srv = nullptr;

void remote_dp_register_root_server(RemoteServer* srv) { g_dp_root_srv = srv; }
bool remote_dp_root_mode() { return g_dp_root_srv != nullptr; }

// ---- RemoteServer 的贡献/结果队列（定义在 .cpp，头文件只持 shared_ptr 前向声明）----
struct RemoteServer::DpQueue {
    std::mutex mu;
    std::condition_variable cv_contrib;   // 训练线程 push → 服务线程 pop
    std::condition_variable cv_result;    // 服务线程 publish → 训练线程 wait
    std::deque<std::vector<float>> contrib;
    std::deque<std::vector<float>> done;
    bool on = false;
};
void RemoteServer::enable_dp_queue() {
    if (!dpq_) dpq_ = std::make_shared<DpQueue>();
    std::lock_guard<std::mutex> lk(dpq_->mu);
    dpq_->on = true;
}
bool RemoteServer::dp_queue_enabled() const {
    if (!dpq_) return false;
    std::lock_guard<std::mutex> lk(dpq_->mu);
    return dpq_->on;
}
void RemoteServer::push_dp_contribution(const float* p, size_t n) {
    if (!dpq_) return;
    std::vector<float> v(p, p + n);
    std::lock_guard<std::mutex> lk(dpq_->mu);
    dpq_->contrib.push_back(std::move(v));
    dpq_->cv_contrib.notify_one();
}
bool RemoteServer::pop_dp_contribution(std::vector<float>& out) {
    if (!dpq_) return false;
    std::unique_lock<std::mutex> lk(dpq_->mu);
    // 有界等待：对端 UP 可能先到、本端贡献随后到（同一轮，间隔很小）；旧路径不会进这里
    if (!dpq_->cv_contrib.wait_for(lk, std::chrono::seconds(60),
                                   [&] { return !dpq_->contrib.empty(); })) {
        return false;
    }
    out = std::move(dpq_->contrib.front());
    dpq_->contrib.pop_front();
    return true;
}
void RemoteServer::publish_dp_result(const std::vector<float>& r) {
    if (!dpq_) return;
    std::lock_guard<std::mutex> lk(dpq_->mu);
    dpq_->done.push_back(r);
    dpq_->cv_result.notify_one();
}
bool RemoteServer::wait_dp_result(float* out, size_t n, int timeout_sec) {
    if (!dpq_) return false;
    std::unique_lock<std::mutex> lk(dpq_->mu);
    if (!dpq_->cv_result.wait_for(lk, std::chrono::seconds(timeout_sec),
                                  [&] { return !dpq_->done.empty(); })) {
        return false;
    }
    std::vector<float>& r = dpq_->done.front();
    if (r.size() != n) { dpq_->done.pop_front(); return false; }
    std::memcpy(out, r.data(), n * sizeof(float));
    dpq_->done.pop_front();
    return true;
}

// 读/写一个张量的宿主副本（兼容 backend buffer / 裸 CPU）
static bool dp_read_tensor(TensorF32* t, std::vector<float>& host) {
    const int64_t n = t->numel();
    if (n <= 0) return false;
    host.resize((size_t)n);
    if (t->buffer_) {
        t->buffer_->get_tensor(t, host.data(), t->buffer_offs_, (size_t)n * sizeof(float));
    } else if (t->data()) {
        std::memcpy(host.data(), t->data(), (size_t)n * sizeof(float));
    } else {
        return false;
    }
    return true;
}
static void dp_write_tensor(TensorF32* t, const std::vector<float>& host) {
    // 诊断开关：跳过写回（用于判定"写回是否写进已释放/上一代的存储"这类 epoch 边界崩溃）
    if (std::getenv("PPML_DP_NO_WRITEBACK")) return;
    const size_t bytes = host.size() * sizeof(float);
    // 同代校验（最小版）：目标必须有可用存储；否则**不写**并响亮报错（避免写进已释放内存造成延迟崩溃）
    if (!t->buffer_ && !t->data()) {
        std::fprintf(stderr, "[DP-SHARD] FATAL: 写回目标无存储（numel=%lld）⇒ 跳过写回（疑似跨代/已释放）\n",
                     (long long)t->numel());
        return;
    }
    if (t->buffer_) {
        t->buffer_->set_tensor(t, host.data(), t->buffer_offs_, bytes);
    } else {
        std::memcpy(t->data(), host.data(), bytes);
    }
}

// 阶段 A：**相位显式握手**（2026-09-14 深夜，教训：靠两侧各自推理太脆弱）。
//   每个相位开始时交换 [phase_id, 本侧参数个数] 并求和校验：
//     ① 两侧都必须执行且仅执行一次 ⇒ 即使某一侧该相位"为空"也能保持配对（不会再吃掉下一个相位的 UP）；
//     ② 求和结果应为 [2*phase_id, 2*count] ⇒ 不一致立刻响亮报错并停止同步（不静默错值）。
static bool dp_phase_begin(ComputeGraph* cgraph, RemoteClient* cli, bool is_root, int phase_id,
                           int world, int my_rank) {
    if (!cgraph || world <= 1) return true;
    int local_count = 0;
    for (int i = 0; i < cgraph->n_nodes(); ++i) {
        TensorF32* n = cgraph->graph_node(i);
        if (!n || !(n->flag & TENSOR_FLAG_PARAM)) continue;
        if (!cgraph->graph_get_grad(n)) continue;   // 与同步循环同口径（PARAM 且有 grad）
        ++local_count;
    }
    float buf[2] = {(float)phase_id, (float)local_count};
    bool ok = false;
    if (is_root) {
        if (!g_dp_root_srv) return true;
        g_dp_root_srv->push_dp_contribution(buf, 2);
        ok = g_dp_root_srv->wait_dp_result(buf, 2);
    } else if (cli && cli->connected()) {
        ok = cli->allreduce_sum(buf, 2);
    } else {
        return true;
    }
    if (!ok) {
        std::fprintf(stderr, "[DP-SHARD] FATAL: 相位%d 握手失败（对端无响应/流错位）\n", phase_id);
        return false;
    }
    const float expect_id = 2.0f * (float)phase_id;
    const float expect_cnt = 2.0f * (float)local_count;
    if (std::getenv("GRAPH_DEBUG_REMOTE")) {
        // 成功也打印（此前成功是静默的 ⇒ 看不到两端调用序列，无法定位"谁多推了一次"）
        static int s_ok_no = 0;
        std::fprintf(stderr, "[DP-SHARD] phase#%d OK: phase=%d count=%d rank=%d world=%d sum=[%g,%g]\n",
                     ++s_ok_no, phase_id, local_count, my_rank, world, (double)buf[0], (double)buf[1]);
    }
    if (std::fabs(buf[0] - expect_id) > 0.5f || std::fabs(buf[1] - expect_cnt) > 0.5f) {
        std::fprintf(stderr,
                     "[DP-SHARD] FATAL: 相位%d 两侧不一致（本侧 count=%d；求和 id=%g cnt=%g，期望 %g/%g）"
                     "⇒ 相位错配，停止本次同步（rank=%d world=%d）\n",
                     phase_id, local_count, (double)buf[0], (double)buf[1],
                     (double)expect_id, (double)expect_cnt, my_rank, world);
        return false;
    }
    return true;
}

// 内部：root 侧一轮遍历（is_grad=true 同步梯度平均；false 同步 owner 新权重）
static int dp_roundtrip_root(ComputeGraph* cgraph, bool is_grad, int world, int my_rank) {
    if (!cgraph || !g_dp_root_srv || world <= 1) return 0;
    // 启动屏障：等对端连接后才进入第一轮 allreduce（否则首轮 wait_dp_result 在对端未就位前超时 ⇒ 弃轮 ✗）
    if (!g_dp_peer_seen) {
        std::fprintf(stderr, "[REMOTE-DP] root 等待对端连接（启动屏障）...\n");
        for (int i = 0; i < 6000 && !g_dp_peer_seen; ++i) {
            struct timespec ts{0, 100 * 1000 * 1000};   // 100ms
            nanosleep(&ts, nullptr);
        }
        if (!g_dp_peer_seen) {
            std::fprintf(stderr, "[REMOTE-DP] root 等对端连接 600s 超时 ⇒ 放弃本轮同步\n");
            return 0;
        }
        std::fprintf(stderr, "[REMOTE-DP] root 已检测到对端连接，继续同步\n");
    }
    if (!dp_phase_begin(cgraph, nullptr, /*is_root=*/true, is_grad ? 1 : 2, world, my_rank)) return 0;
    int n_done = 0, idx = 0;
    for (int i = 0; i < cgraph->n_nodes(); ++i) {
        TensorF32* node = cgraph->graph_node(i);
        if (!node || !(node->flag & TENSOR_FLAG_PARAM)) continue;
        TensorF32* g = cgraph->graph_get_grad(node);
        if (!g) continue;
        const int owner = idx % world;
        ++idx;
        TensorF32* src = is_grad ? g : node;
        std::vector<float> host;
        // ★不可读也必须占位（补 0 到 numel）：本端少发一轮 ⇒ 与对端**流错位一格** ✗
        //   （2026-09-14 深夜实测：相位握手报 "sum id=3"=1+2，即两侧相位错开一轮的根因）
        if (!dp_read_tensor(src, host)) host.assign((size_t)std::max<int64_t>(src->numel(), 0), 0.0f);
        if (host.empty()) continue;   // numel==0：两侧同口径（对端也发不出数据）
        // 权重同步：非 owner 提交全 0（占位保持配对，最终结果 = 唯一 owner 的新值）；梯度：双方全量参与
        if (!is_grad && owner != my_rank) std::fill(host.begin(), host.end(), 0.0f);
        g_dp_root_srv->push_dp_contribution(host.data(), host.size());
        if (!g_dp_root_srv->wait_dp_result(host.data(), host.size())) {
            std::fprintf(stderr, "[REMOTE-DP] root wait_dp_result 超时（param idx=%d）\n", idx - 1);
            return n_done;
        }
        if (is_grad) {
            const float inv = 1.0f / (float)world;
            for (float& v : host) v *= inv;
        }
        dp_write_tensor(src, host);
        ++n_done;
    }
    if (std::getenv("GRAPH_DEBUG_REMOTE")) {
        std::fprintf(stderr, "[REMOTE-DP] root 同步完成：%s=%d world=%d rank=%d\n",
                     is_grad ? "grads" : "params", n_done, world, my_rank);
    }
    return n_done;
}

int remote_dp_grads_root(ComputeGraph* cgraph, int world, int my_rank) {
    if (std::getenv("GRAPH_DEBUG_REMOTE")) {
        std::fprintf(stderr, "[DP-SHARD] >>> grads_root 进入（rank=%d world=%d）\n", my_rank, world);
    }
    return dp_roundtrip_root(cgraph, /*is_grad=*/true, world, my_rank);
}
int remote_dp_params_root(ComputeGraph* cgraph, int world, int my_rank) {
    if (std::getenv("GRAPH_DEBUG_REMOTE")) {
        std::fprintf(stderr, "[DP-SHARD] >>> params_root 进入（rank=%d world=%d）\n", my_rank, world);
    }
    return dp_roundtrip_root(cgraph, /*is_grad=*/false, world, my_rank);
}

int remote_dp_allreduce_grads(ComputeGraph* cgraph, RemoteClient* cli) {
    if (!cgraph || !cli || !cli->connected()) return 0;
    const int world = cli->world_size();
    if (world <= 1) return 0;
    if (!dp_phase_begin(cgraph, cli, /*is_root=*/false, 1, world, cli->rank())) return 0;   // 相位1=梯度

    int n_done = 0;
    for (int i = 0; i < cgraph->n_nodes(); ++i) {
        TensorF32* node = cgraph->graph_node(i);
        if (!node || !(node->flag & TENSOR_FLAG_PARAM)) continue;
        TensorF32* g = cgraph->graph_get_grad(node);
        if (!g) continue;
        const int64_t n = g->numel();
        if (n <= 0) continue;

        std::vector<float> host((size_t)n);
        if (g->buffer_) {
            g->buffer_->get_tensor(g, host.data(), g->buffer_offs_, (size_t)n * sizeof(float));
        } else if (g->data()) {
            std::memcpy(host.data(), g->data(), (size_t)n * sizeof(float));
        } else {
            // 不可读也要参与（补 0 占位）：跳过一轮会与对端流错位 ✗
            std::fill(host.begin(), host.end(), 0.0f);
        }

        if (!cli->allreduce_sum(host.data(), (size_t)n)) return n_done;
        const float inv = 1.0f / (float)world;
        for (int64_t k = 0; k < n; ++k) host[(size_t)k] *= inv;

        if (g->buffer_) {
            g->buffer_->set_tensor(g, host.data(), g->buffer_offs_, (size_t)n * sizeof(float));
        } else if (g->data()) {
            std::memcpy(g->data(), host.data(), (size_t)n * sizeof(float));
        }
        ++n_done;
    }
    if (std::getenv("GRAPH_DEBUG_REMOTE")) {
        std::fprintf(stderr, "[REMOTE-DP] allreduce 完成：params=%d world=%d\n", n_done, world);
    }
    return n_done;
}

// ============================================================
// 数据并行（阶段 A：参数分片）——optimizer.step() 之后同步"owner 的新值"
//   设计要点：复用已有的 allreduce_sum 原语（星型/两机对换），无需新协议：
//     ① 参数全局序号 idx 与 AdamW::init_from_graph 的计数规则**完全一致**（PARAM 且有 grad 才 +1）；
//     ② 非 owner 侧把自己那份（旧值）置 0，只有 owner 提交新值 ⇒ sum == owner 的新值；
//     ③ 因此**不做 /world**，两端写回同一个值 ⇒ 逐位一致 ⇒ 下一轮 forward 相同。
//   若跳过本步：非 owner 侧会带着旧权重继续训练 ⇒ 两端发散（阶段 A 的核心风险，见 PLAN_SHARDING.md:44）。
// ============================================================
int remote_dp_broadcast_owned_params(ComputeGraph* cgraph, RemoteClient* cli) {
    if (!cgraph) return 0;
    if (!cli || !cli->connected()) {
        // 阶段 A：rank0/acceptor 的 root 模式（cli 为空）⇒ 走贡献队列（与对端同名动作配对）
        if (g_dp_root_srv) {
            const int world = std::getenv("PPML_REMOTE_WORLD") ? std::atoi(std::getenv("PPML_REMOTE_WORLD")) : 2;
            const int rank  = std::getenv("PPML_REMOTE_RANK")  ? std::atoi(std::getenv("PPML_REMOTE_RANK"))  : 0;
            if (std::getenv("GRAPH_DEBUG_REMOTE")) {
                std::fprintf(stderr, "[DP-SHARD] params: 走 ROOT 队列（cli 空，root=%p）\n", (void*)g_dp_root_srv);
            }
            return dp_roundtrip_root(cgraph, /*is_grad=*/false, world, rank);
        }
        if (std::getenv("GRAPH_DEBUG_REMOTE")) {
            std::fprintf(stderr, "[DP-SHARD] params: cli=%p 未连接 且无 root ⇒ 返回 0（本次不同步）\n",
                         (void*)cli);
        }
        return 0;
    }
    if (std::getenv("GRAPH_DEBUG_REMOTE")) {
        std::fprintf(stderr, "[DP-SHARD] params: 走 WORKER 路径（rank=%d world=%d）\n",
                     cli->rank(), cli->world_size());
    }
    // 相位2=参数（此处 world/my_rank 的局部声明还没引入 ⇒ 直接从 cli 取；world<=1 由握手内部处理）
    if (!dp_phase_begin(cgraph, cli, /*is_root=*/false, 2, cli->world_size(), cli->rank())) return 0;
    const int world = cli->world_size();
    if (world <= 1) return 0;
    const int my_rank = cli->rank();

    int n_done = 0, idx = 0;
    for (int i = 0; i < cgraph->n_nodes(); ++i) {
        TensorF32* node = cgraph->graph_node(i);
        if (!node || !(node->flag & TENSOR_FLAG_PARAM)) continue;
        TensorF32* g = cgraph->graph_get_grad(node);
        if (!g) continue;                       // 与 AdamW 一致：无 grad 的参数不计序号、不同步
        const int owner = idx % world;
        ++idx;

        const int64_t n = node->numel();
        if (n <= 0) continue;
        std::vector<float> host((size_t)n);
        if (node->buffer_) {
            node->buffer_->get_tensor(node, host.data(), node->buffer_offs_, (size_t)n * sizeof(float));
        } else if (node->data()) {
            std::memcpy(host.data(), node->data(), (size_t)n * sizeof(float));
        } else {
            // 不可读也要参与（补 0 占位）：跳过一轮会与对端流错位 ✗（见 dp_phase_begin 注释）
            std::fill(host.begin(), host.end(), 0.0f);
        }
        // ★配对要求（阶段 A 关键）：**两端必须对同一个参数各调用一次 allreduce**（顺序也一致），
        //   否则服务端 handle_allreduce 是按"第 k 次调用"配对贡献的 ⇒ 会把参数 k 的贡献加到参数 m 上 ✗。
        //   非 owner 侧提交全 0（不贡献值，但占位保持配对）。
        if (owner != my_rank) {
            std::fill(host.begin(), host.end(), 0.0f);
        }
        if (!cli->allreduce_sum(host.data(), (size_t)n)) return n_done;
        if (node->buffer_) {
            node->buffer_->set_tensor(node, host.data(), node->buffer_offs_, (size_t)n * sizeof(float));
        } else if (node->data()) {
            std::memcpy(node->data(), host.data(), (size_t)n * sizeof(float));
        }
        ++n_done;
    }
    if (std::getenv("GRAPH_DEBUG_REMOTE")) {
        std::fprintf(stderr, "[DP-SHARD] params: 循环结束 synced=%d / 扫过参数 idx=%d（world=%d rank=%d）\n",
                     n_done, idx, world, my_rank);
    }
    return n_done;
}

// ============================================================
// 迭代边界 hook（train.cpp 调用；未启用远端时零开销）
//   为什么必须成对存在：
//     forward 阶段对端产出激活 → backward 阶段才取回（跨整个 forward+backward），
//     而"最近 K 次执行"的 LRU 会在一轮 forward 内就把最早的结果挤掉 ⇒ 取回未命中 abort。
//     ⇒ 保留窗口的语义必须与"训练迭代"对齐，而不是与"split 执行次数"对齐。
//   同时解决"同指针跨迭代复用 + 幂等上传"导致的静默旧值：transient 一律作废重传。
// ============================================================
bool remote_iteration_begin() {
    RemoteBackend* be = remote_make_backend_from_env();
    if (!be || !be->client()) return false;
    RemoteClient* cli = be->client();
    // 【T1 2026-09-16】迭代边界是**唯一安全的重连点**：此刻两端之间没有"半成品状态"需要保留
    //   （本端下面马上 invalidate_transient() + RESULTS_CLEAR 把对端清空重来 ✓）。
    //   旧行为：连接断了就永久失效 ⇒ 后面所有远端 split 全部失败/回落 ✗（网络抖一下的代价）。
    if (!cli->connected() || cli->io_failed()) {
        if (!cli->ensure_connected()) {
            std::fprintf(stderr, "[REMOTE] 迭代边界重连失败 ⇒ 本轮远端不可用（调度器将回落本地）\n");
            return false;
        }
    } else if (!cli->ping()) {
        std::fprintf(stderr, "[REMOTE] 迭代边界心跳失败 ⇒ 尝试重连 …\n");
        if (!cli->ensure_connected()) return false;
    }
    cli->invalidate_transient();   // 激活/输入：作废重传；参数/常量保留（step 后单独作废）
    uint64_t dr = 0, da = 0;
    const bool ok = cli->clear_results(&dr, &da);
    if (std::getenv("GRAPH_DEBUG_REMOTE") || rtrace()) {
        std::fprintf(stderr,
                     "[REMOTE] 迭代边界：keep(参数/常量)=%zu 对端丢弃 results=%llu arena=%llu ok=%d\n",
                     cli->persistent_ids().size(), (unsigned long long)dr,
                     (unsigned long long)da, (int)ok);
    }
    return ok;
}

int remote_after_optimizer_step(ComputeGraph* cgraph) {
    if (!cgraph) return 0;
    RemoteBackend* be = remote_make_backend_from_env();
    if (!be || !be->client() || !be->client()->connected()) return 0;
    RemoteClient* cli = be->client();
    int n = 0;
    for (int i = 0; i < cgraph->n_nodes(); ++i) {
        TensorF32* node = cgraph->graph_node(i);
        if (!node || !(node->flag & TENSOR_FLAG_PARAM)) continue;
        // invalidate 只对"曾经上传过 id"的参数生效（没上过对端的自动 no-op）
        cli->invalidate(node);
        ++n;
    }
    if ((std::getenv("GRAPH_DEBUG_REMOTE") || rtrace()) && n > 0) {
        std::fprintf(stderr, "[REMOTE] step 后作废参数 %d 个（下轮 compute 自动重传新权重）\n", n);
    }
    return n;
}

}  // namespace ppml

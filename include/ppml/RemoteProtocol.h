// ============================================================
// RemoteProtocol.h —— 双机训练用的最小网络协议（2026-09-13）
//
// 设计目标（对应 experiments/dist/PLAN_2NODE.md §B）：
//   1) **张量搬运**：头部固定 = {magic, version, kind, tensor_id, dtype, ndim, dims[4], nbytes}
//      + 数据体（连续 bytes）。tensor_id 由服务端 arena 分配并回传，客户端后续按 id 引用。
//   2) **子图执行**：节点描述 = {op, type, ndim, dims[4], op_params[64], n_src, src_ids[]}
//      —— 只传"结构 + 标识"，绝不传指针（data_/buffer_/view_src 都是本地指针）。
//   3) **Allreduce**：为数据并行准备（星型：worker→rank0 求和→广播；2 机时等价于一次对换）。
//
// 说明：两台机器都是 x86_64 + 同版本代码 ⇒ 直接按本机字节序/结构体布局发送（带 magic+version 校验）。
//       真要跨架构时把 send/recv 换成显式序列化即可，接口不变。
// ============================================================
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace ppml {
namespace rnet {

// ---------------- 协议常量 ----------------
constexpr uint32_t R_MAGIC   = 0x50524D54u;   // 'PRMT'
constexpr uint32_t R_VERSION = 1u;

enum class RKind : uint32_t {
    HELLO         = 1,   // 握手：客户端 → 服务端 {world_size, rank}
    HELLO_ACK     = 2,   // 握手回包：服务端 → 客户端 {assigned_rank}
    PING          = 3,   // 延迟探测（无体）
    PONG          = 4,
    TENSOR_SET    = 5,   // 张量数据上传（客户端→服务端）；服务端回 TENSOR_SET_ACK 带 id
    TENSOR_SET_ACK= 6,
    TENSOR_GET    = 7,   // 请求下载（客户端→服务端，头里只有 id）
    TENSOR_DATA   = 8,   // 下载数据（服务端→客户端，头+体）
    GRAPH_COMPUTE = 9,   // 客户端→服务端：节点描述 + 输出 id 列表
    GRAPH_RESULT  = 10,  // 服务端→客户端：每个输出 id 的数据（TENSOR_DATA 复用亦可）
    ALLREDUCE_UP  = 11,  // 数据并行：worker 上传待求和缓冲
    ALLREDUCE_DOWN= 12,  // 数据并行：rank0 广播求和结果
    ERROR_MSG     = 13,
    BYE           = 14,
    // 迭代边界（客户端→服务端，2026-09-14 加）：丢弃上一轮算出的结果，只保留 body 里列出的 id
    //   body = uint64[n]（keep ids，一般是参数/常量）；aux0 = n。
    //   服务端回 RESULTS_CLEAR_ACK{aux0=丢掉的 results 条数, aux1=丢掉的 arena 张量数}。
    //   为什么需要它：原先用"最近 K 次执行"的 LRU 淘汰，一轮 forward(18 个远端 split) 就会把最早的
    //   结果挤掉，而 backward 还要取 forward 的激活 ⇒ TENSOR_GET 未命中 abort（OPS=47 的真因）。
    RESULTS_CLEAR = 15,
    RESULTS_CLEAR_ACK = 16,
};

enum class RDtype : uint32_t {
    F32 = 0, I64 = 1, U8 = 2,
};

// ---------------- 定长消息头（"头：dtype/shape/tensor_id"）----------------
struct RHeader {
    uint32_t magic   = R_MAGIC;
    uint32_t version = R_VERSION;
    uint32_t kind    = 0;          // RKind
    uint32_t dtype   = 0;          // RDtype
    uint64_t tensor_id = 0;        // 张量标识（服务端 arena 分配 / 客户端参数空间）
    uint64_t nbytes  = 0;          // 数据体字节数
    int64_t  ndim    = 0;
    int64_t  dims[4] = {1, 1, 1, 1};
    uint64_t aux0    = 0;          // 用途随 kind：世界大小 / 输出个数 / 节点个数 …
    uint64_t aux1    = 0;          // 用途随 kind：rank / 偏移 / …
};
static_assert(sizeof(RHeader) == 88, "RHeader 必须是紧凑定长（跨机一致性）");

// ---------------- 节点描述（远端执行子图）----------------
constexpr int R_MAX_SRC = 10;     // 与 GGML_MAX_SRC 对齐

struct RNodeDesc {
    uint32_t op   = 0;            // tensor_op
    uint32_t type = 0;            // tensor_type
    int64_t  ndim = 0;
    int64_t  dims[4] = {1, 1, 1, 1};
    int32_t  n_src = 0;
    int32_t  _pad  = 0;
    uint64_t id       = 0;        // 本节点 id（客户端分配）
    uint64_t src_ids[R_MAX_SRC] = {0};
    int32_t  op_params[64] = {0};
};

// 一次 GRAPH_COMPUTE 请求的头部信息（节点/输出在紧随的 body 里）
struct RGraphReqInfo {
    uint32_t n_nodes = 0;
    uint32_t n_outs  = 0;
    uint64_t body_bytes = 0;      // n_nodes*sizeof(RNodeDesc) + n_outs*sizeof(uint64_t)
};

// ---------------- TCP 连接（阻塞式，够用且好调试）----------------
class RSocket {
public:
    RSocket() = default;
    ~RSocket();
    RSocket(const RSocket&) = delete;
    RSocket& operator=(const RSocket&) = delete;
    RSocket(RSocket&& o) noexcept;
    RSocket& operator=(RSocket&& o) noexcept;

    // 服务端：监听 + 接受一个连接
    bool listen_on(int port, int backlog = 4);
    bool accept_one(RSocket& out);
    // 客户端：连接
    bool connect_to(const std::string& host, int port, int timeout_ms = 5000);

    bool valid() const { return fd_ >= 0; }
    void close();
    int  fd() const { return fd_; }

    // 收发（返还 false = 对端关闭/出错）
    bool send_all(const void* data, size_t n);
    bool recv_all(void* data, size_t n);

    // 便捷：头 + 体
    bool send_header(const RHeader& h);
    bool recv_header(RHeader& h);

    // 头（kind + 张量元信息）+ 数据体
    bool send_data(uint32_t kind, uint64_t tensor_id, RDtype dt,
                   int ndim, const int64_t* dims,
                   const void* data, size_t nbytes);
    // 默认 kind = TENSOR_DATA 的简化版
    bool send_bytes(const void* data, size_t n, uint64_t tensor_id, RDtype dt,
                    int ndim = 1, const int64_t* dims = nullptr);
    // 收头 + 体（体长度由头里的 nbytes 决定）
    bool recv_bytes(std::vector<uint8_t>& out, RHeader& h);

    // 小工具：设置 5s 收发超时（0 = 不限）
    void set_timeout_ms(int ms);

private:
    int fd_ = -1;
};

// ---------------- Allreduce（星型：rank0 汇总后广播）----------------
// world_size=2 时退化为"一次对换 + 求和"，即最省的一轮。
// role: is_root=true 表示本端是 rank0（聚合端）。
// 语义：allreduce_sum（不做平均；调用方自行除以 world_size 或按需缩放）。
bool allreduce_sum(RSocket& sock, float* buf, size_t n, bool is_root, int world_size);

// ---------------- 常用小工具 ----------------
size_t dtype_size(RDtype dt);
const char* kind_name(uint32_t kind);

}  // namespace rnet
}  // namespace ppml

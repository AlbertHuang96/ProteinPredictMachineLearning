// ============================================================
// RemoteProtocol.cpp —— 双机协议实现（TCP + 定长头 + 数据体）
// 详见 include/ppml/RemoteProtocol.h 的说明与 experiments/dist/PLAN_2NODE.md §B.5/§B.6
// ============================================================
#include "ppml/RemoteProtocol.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace ppml {
namespace rnet {

// ---------------- 小工具 ----------------
size_t dtype_size(RDtype dt) {
    switch (dt) {
        case RDtype::F32: return 4;
        case RDtype::I64: return 8;
        case RDtype::U8:  return 1;
    }
    return 0;
}

const char* kind_name(uint32_t kind) {
    switch ((RKind)kind) {
        case RKind::HELLO:          return "HELLO";
        case RKind::HELLO_ACK:      return "HELLO_ACK";
        case RKind::PING:           return "PING";
        case RKind::PONG:           return "PONG";
        case RKind::TENSOR_SET:     return "TENSOR_SET";
        case RKind::TENSOR_SET_ACK: return "TENSOR_SET_ACK";
        case RKind::TENSOR_GET:     return "TENSOR_GET";
        case RKind::TENSOR_DATA:    return "TENSOR_DATA";
        case RKind::GRAPH_COMPUTE:  return "GRAPH_COMPUTE";
        case RKind::GRAPH_RESULT:   return "GRAPH_RESULT";
        case RKind::ALLREDUCE_UP:   return "ALLREDUCE_UP";
        case RKind::ALLREDUCE_DOWN: return "ALLREDUCE_DOWN";
        case RKind::ERROR_MSG:      return "ERROR_MSG";
        case RKind::BYE:            return "BYE";
        case RKind::RESULTS_CLEAR:     return "RESULTS_CLEAR";
        case RKind::RESULTS_CLEAR_ACK: return "RESULTS_CLEAR_ACK";
    }
    return "?";
}

// ---------------- RSocket ----------------
RSocket::~RSocket() { close(); }

RSocket::RSocket(RSocket&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }

RSocket& RSocket::operator=(RSocket&& o) noexcept {
    if (this != &o) { close(); fd_ = o.fd_; o.fd_ = -1; }
    return *this;
}

void RSocket::close() {
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

void RSocket::set_timeout_ms(int ms) {
    if (fd_ < 0 || ms <= 0) return;
    struct timeval tv{};
    tv.tv_sec  = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

static void set_tcp_nodelay(int fd) {
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

bool RSocket::listen_on(int port, int backlog) {
    close();
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) return false;
    int one = 1;
    ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons((uint16_t)port);
    if (::bind(fd_, (sockaddr*)&addr, sizeof(addr)) != 0) {
        std::fprintf(stderr, "[rnet] bind(%d) failed: %s\n", port, std::strerror(errno));
        close();
        return false;
    }
    if (::listen(fd_, backlog) != 0) {
        std::fprintf(stderr, "[rnet] listen(%d) failed: %s\n", port, std::strerror(errno));
        close();
        return false;
    }
    return true;
}

bool RSocket::accept_one(RSocket& out) {
    if (fd_ < 0) return false;
    sockaddr_in peer{};
    socklen_t len = sizeof(peer);
    int c = ::accept(fd_, (sockaddr*)&peer, &len);
    if (c < 0) {
        std::fprintf(stderr, "[rnet] accept failed: %s\n", std::strerror(errno));
        return false;
    }
    set_tcp_nodelay(c);
    out.close();
    out.fd_ = c;
    return true;
}

bool RSocket::connect_to(const std::string& host, int port, int timeout_ms) {
    close();
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) return false;
    set_timeout_ms(timeout_ms);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        std::fprintf(stderr, "[rnet] bad host '%s'\n", host.c_str());
        close();
        return false;
    }
    if (::connect(fd_, (sockaddr*)&addr, sizeof(addr)) != 0) {
        std::fprintf(stderr, "[rnet] connect(%s:%d) failed: %s\n",
                     host.c_str(), port, std::strerror(errno));
        close();
        return false;
    }
    set_tcp_nodelay(fd_);
    set_timeout_ms(0);   // 后续阻塞等待不限时（由上层控制）
    return true;
}

bool RSocket::send_all(const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t left = n;
    while (left > 0) {
        ssize_t w = ::send(fd_, p, left, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "[rnet] send failed: %s\n", std::strerror(errno));
            return false;
        }
        p += w; left -= (size_t)w;
    }
    return true;
}

bool RSocket::recv_all(void* data, size_t n) {
    uint8_t* p = static_cast<uint8_t*>(data);
    size_t left = n;
    while (left > 0) {
        ssize_t r = ::recv(fd_, p, left, 0);
        if (r == 0) { std::fprintf(stderr, "[rnet] peer closed\n"); return false; }
        if (r < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "[rnet] recv failed: %s\n", std::strerror(errno));
            return false;
        }
        p += r; left -= (size_t)r;
    }
    return true;
}

bool RSocket::send_header(const RHeader& h) { return send_all(&h, sizeof(h)); }

bool RSocket::recv_header(RHeader& h) {
    if (!recv_all(&h, sizeof(h))) return false;
    if (h.magic != R_MAGIC || h.version != R_VERSION) {
        std::fprintf(stderr, "[rnet] bad header magic/version (0x%08x v%u)\n", h.magic, h.version);
        return false;
    }
    return true;
}

bool RSocket::send_data(uint32_t kind, uint64_t tensor_id, RDtype dt,
                        int ndim, const int64_t* dims,
                        const void* data, size_t nbytes) {
    RHeader h;
    h.kind      = kind;
    h.dtype     = (uint32_t)dt;
    h.tensor_id = tensor_id;
    h.nbytes    = nbytes;
    h.ndim      = ndim;
    for (int i = 0; i < 4; ++i) h.dims[i] = (dims && i < ndim) ? dims[i] : 1;
    if (!send_header(h)) return false;
    return nbytes == 0 ? true : send_all(data, nbytes);
}

bool RSocket::send_bytes(const void* data, size_t n, uint64_t tensor_id, RDtype dt,
                         int ndim, const int64_t* dims) {
    return send_data((uint32_t)RKind::TENSOR_DATA, tensor_id, dt, ndim, dims, data, n);
}

bool RSocket::recv_bytes(std::vector<uint8_t>& out, RHeader& h) {
    if (!recv_header(h)) return false;
    out.resize(h.nbytes);
    if (h.nbytes == 0) return true;
    return recv_all(out.data(), h.nbytes);
}

// ---------------- Allreduce（星型：rank0 汇总 + 广播）----------------
// 拓扑：worker 与 rank0 之间各有一条 RSocket（本接口一次处理"本端 ↔ 对端"）。
//   is_root=true  : 收 world_size-1 个 worker 的缓冲（同一 socket 上顺序到达）
//   is_root=false : 发自己的缓冲 → 等待广播
// 语义：求和（不平均）。平均由调用方负责（/world_size）。
bool allreduce_sum(RSocket& sock, float* buf, size_t n, bool is_root, int world_size) {
    const size_t bytes = n * sizeof(float);
    if (world_size <= 1) return true;

    if (is_root) {
        std::vector<float> tmp(n);
        std::vector<float> acc(n, 0.0f);
        for (int r = 1; r < world_size; ++r) {
            RHeader h;
            if (!sock.recv_header(h)) return false;
            if (h.kind != (uint32_t)RKind::ALLREDUCE_UP || h.nbytes != bytes) {
                std::fprintf(stderr, "[rnet] allreduce: unexpected msg %s nbytes=%llu (expect %zu)\n",
                             kind_name(h.kind), (unsigned long long)h.nbytes, bytes);
                return false;
            }
            if (!sock.recv_all(tmp.data(), bytes)) return false;
            for (size_t i = 0; i < n; ++i) acc[i] += tmp[i];
        }
        for (size_t i = 0; i < n; ++i) buf[i] += acc[i];
        // 广播（每个 worker 一份；2 机时只有 1 份）
        for (int r = 1; r < world_size; ++r) {
            if (!sock.send_data((uint32_t)RKind::ALLREDUCE_DOWN, 0, RDtype::F32,
                                1, nullptr, buf, bytes)) return false;
        }
        return true;
    }

    if (!sock.send_data((uint32_t)RKind::ALLREDUCE_UP, 0, RDtype::F32,
                        1, nullptr, buf, bytes)) return false;
    RHeader h;
    if (!sock.recv_header(h)) return false;
    if (h.kind != (uint32_t)RKind::ALLREDUCE_DOWN || h.nbytes != bytes) {
        std::fprintf(stderr, "[rnet] allreduce: unexpected reply %s nbytes=%llu\n",
                     kind_name(h.kind), (unsigned long long)h.nbytes);
        return false;
    }
    return sock.recv_all(buf, bytes);
}

}  // namespace rnet
}  // namespace ppml

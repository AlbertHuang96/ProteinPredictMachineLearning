// ============================================================
// StagingPool.cu —— pinned staging 池 + 异步边界拷贝（P1 2026-09-19 / P2 双缓冲 2026-09-19）
//
// 动机（实测，见 .codebuddy/memory/2026-09-19.md）：
//   混合调度下每次"跨后端边界"都用**阻塞** cudaMemcpy ✗
//     · `backend_tensor_copy` Case1/2（Backend.cpp:536-538 / :562-563）
//     · Step 1b 的 host_scratch_ 暂存（Backend.cpp 内，**per-split 新建 vector** ✗）
//   pageable 拷贝由驱动做（隐式同步 + 内部 staging）⇒ 实测 D2H 占 memcpy 时间 84.7% ✗。
//
// ── P1：pinned arena + 非阻塞拷贝流 + 事件排序 ────────────────────────────────
//   staging_alloc()  —— 从 pinned arena bump 一块（128B 对齐 ✓）
//   staging_d2h()    —— 默认流 record → copy 流 wait → async(D2H)（异步入队 ✓）
//   staging_h2d()    —— copy 流 async(H2D) → record → 默认流 wait（设备侧排序 ⇒ 无 host 等待 ✓）
//   staging_wait()   —— 一次 host 等待（等 copy 流上已入队的全部拷贝 ✓）
//
// ── P2：**双缓冲 arena**（本文件当前版本）───────────────────────────────────
//   P1 的正确性靠"split 末 drain（排空拷贝流）"来保证 ✗ —— 因为 H2D 的**源**、D2H 的**目的地**
//   都在 arena 里：若下一个 split 的 alloc 覆盖了仍在飞的拷贝缓冲区 ⇒ **静默错值** ✗
//   （P1 实测就踩到：loss 12.57 → 15.32 ✗；加 drain 后恢复 ✓，但 drain 把"每 split 一次 host 等待"
//    又加回来 ⇒ 收益被吃掉一半 ✗）。
//   P2 改为 **2 块 arena 轮换**（split i 用 A、i+1 用 B、i+2 回到 A ✓）：
//     · split 末**只 record 一个完成事件**（不阻塞 ✓✓）
//     · 切换/复用某块前，用 **cudaEventQuery 非阻塞查询**它上一批拷贝是否已完成 ✓
//         已完成 ⇒ 直接复用（零等待 ✓✓）；未完成 ⇒ 才同步等它（实测极少 ✓）
//   ⇒ 既有正确性（不会覆盖在飞缓冲 ✓），又没有 drain ✗
//
// 开关/限制：
//   PPML_STAGING_ASYNC=0     ⇒ 全部回落原阻塞路径（A/B 用 ✓）
//   PPML_STAGING_MAX_MB=32   ⇒ **每块** arena 上限（默认 32 ⇒ 两块合计 ≤64 MB ✓，WSL pinned 稀缺 ⚠️）
//   GRAPH_DEBUG_STAGING=1    ⇒ 每 split 打印用量/计数/回落/等待 ✓
//
// ⚠️ 关键约束（写错即静默错值）：
//   ① `cudaStreamNonBlocking` 的 copy 流**不会**与 legacy 默认流隐式同步 ⇒ 全靠事件排序 ✓
//   ② **只有 split 的首分配**（used==0）才允许扩容/换块 ⇒ 否则会搬走本 split 已发出的指针 ✗
//   ③ 复用/释放某块 arena 前，必须确认它上一批拷贝已完成（**先 query 再 wait** ✓）
//   ④ 事件环轮转：避免对"尚未被消费"的事件重复 record ✗
// ============================================================
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace ppml {

namespace {

std::mutex   g_mu;                     // 兜底（正常只有 thread0 调 ✓）
bool         g_inited = false;
bool         g_ok     = false;         // 异步 staging 是否可用

cudaStream_t g_copy   = nullptr;       // 专用拷贝流（nonBlocking ✓）
constexpr int kRingN  = 16;
cudaEvent_t  g_ring[kRingN] = {};      // 轮转事件环（跨流排序用 ✓）
int          g_ring_next = 0;
cudaEvent_t  g_done   = nullptr;       // staging_wait 用

// ---- 双缓冲 arena（P2）----
constexpr int kArenaN = 2;
struct Arena {
    void*       ptr      = nullptr;    // cudaHostAlloc 的 pinned 基址
    size_t      cap      = 0;          // 字节容量（grow-only ✓）
    size_t      used     = 0;          // 本 split 的 bump 偏移
    cudaEvent_t busy     = nullptr;    // 该块最后一批拷贝的完成事件（复用前 query ✓）
    bool        busy_valid = false;    // busy 是否已被 record（首次使用前为 false ✓）
};
Arena        g_arena[kArenaN];
int          g_cur = 0;                // 当前 split 用哪一块
size_t       g_cap_max = 0;            // **每块**上限（PPML_STAGING_MAX_MB）
size_t       g_need_hint = 0;          // 曾出现"中途不足"的总量提示（下次首分配扩到位 ✓）

// 诊断
bool  g_dbg = false;
bool  g_dirty = false;                 // 本 split 是否入队过拷贝（决定 split 末是否 record ✓）
long  g_n_d2h = 0, g_n_h2d = 0, g_n_fallback = 0, g_n_grow = 0, g_n_busy_wait = 0;
long long g_b_d2h = 0, g_b_h2d = 0;

inline size_t align128(size_t n) { return (n + 127u) & ~(size_t)127u; }

void free_arena_locked(Arena& a) {
    if (a.ptr) { cudaFreeHost(a.ptr); a.ptr = nullptr; }
    a.cap = 0;
    a.used = 0;
    a.busy_valid = false;
}

// 确认某块 arena 上一批拷贝已完成（先非阻塞 query，必要时才同步等 ✓）
bool arena_idle_locked(Arena& a) {
    if (!a.busy_valid || !a.busy) return true;
    const cudaError_t q = cudaEventQuery(a.busy);
    if (q == cudaSuccess) { a.busy_valid = false; return true; }
    if (q == cudaErrorNotReady) {
        ++g_n_busy_wait;
        const cudaError_t w = cudaEventSynchronize(a.busy);
        cudaGetLastError();                 // 清残留（含 NotReady ✓）
        a.busy_valid = false;
        if (w != cudaSuccess && g_dbg) {
            std::fprintf(stderr, "[STAGING] arena 等待在飞拷贝失败：%s\n", cudaGetErrorString(w));
        }
        return true;
    }
    cudaGetLastError();
    a.busy_valid = false;
    return true;
}

bool ensure_init_locked() {
    if (g_inited) return g_ok;
    g_inited = true;

    if (const char* e = std::getenv("PPML_STAGING_ASYNC")) {
        if (*e && std::atoi(e) == 0) return g_ok = false;   // 显式关闭（A/B ✓）
    }
    g_dbg = (std::getenv("GRAPH_DEBUG_STAGING") != nullptr);

    size_t max_mb = 32;                                     // **每块**上限（两块合计 ≤64 MB ✓）
    if (const char* m = std::getenv("PPML_STAGING_MAX_MB")) {
        const int v = std::atoi(m);
        if (v > 0) max_mb = (size_t)v;
    }
    g_cap_max = max_mb * 1024u * 1024u;

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
        cudaGetLastError();
        return g_ok = false;
    }
    if (cudaStreamCreateWithFlags(&g_copy, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        g_copy = nullptr;
        return g_ok = false;
    }
    bool evs_ok = true;
    for (int i = 0; i < kRingN; ++i) {
        if (cudaEventCreateWithFlags(&g_ring[i], cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            evs_ok = false;
            break;
        }
    }
    if (evs_ok && cudaEventCreateWithFlags(&g_done, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        evs_ok = false;
    }
    for (int i = 0; evs_ok && i < kArenaN; ++i) {
        if (cudaEventCreateWithFlags(&g_arena[i].busy, cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            evs_ok = false;
        }
    }
    if (!evs_ok) {
        cudaStreamDestroy(g_copy);
        g_copy = nullptr;
        return g_ok = false;
    }
    g_ok = true;
    std::fprintf(stderr,
                 "[STAGING] pinned 双缓冲 staging 已启用（每块上限 %.0f MB，两块合计 ≤%.0f MB，"
                 "nonBlocking 拷贝流 ✓）\n",
                 (double)max_mb, (double)(2 * max_mb));
    return g_ok;
}

inline cudaEvent_t ring_take() {
    cudaEvent_t e = g_ring[g_ring_next];
    g_ring_next = (g_ring_next + 1) % kRingN;
    return e;
}

} // namespace

// ------------------------------------------------------------
// 可用性 / 指针归属 / 分配 / 复位
// ------------------------------------------------------------
int staging_enabled() {
    std::lock_guard<std::mutex> lk(g_mu);
    return ensure_init_locked() ? 1 : 0;
}

int staging_is_pinned(const void* p) {
    if (!p) return 0;
    std::lock_guard<std::mutex> lk(g_mu);
    const char* c = static_cast<const char*>(p);
    for (int i = 0; i < kArenaN; ++i) {
        if (g_arena[i].ptr && c >= (const char*)g_arena[i].ptr &&
            c < (const char*)g_arena[i].ptr + g_arena[i].cap) {
            return 1;
        }
    }
    return 0;
}

void* staging_alloc(int64_t bytes) {
    if (bytes <= 0) return nullptr;
    std::lock_guard<std::mutex> lk(g_mu);
    if (!ensure_init_locked()) return nullptr;

    Arena& a = g_arena[g_cur];
    const size_t need = align128((size_t)bytes);

    // 本 split 的**首分配**：可以安全扩容/换块（不会搬走已发出的指针 ✓）
    if (a.used == 0) {
        // ⓪ ★关键把关（P2 正确性所在）：这块 arena 上次用完时 record 的完成事件可能还在飞；
        //    先非阻塞 query（已完成 ⇒ 零等待 ✓；未完成 ⇒ 才同步等，实测极少 ✓），
        //    否则本 split 的 alloc 会覆盖在飞拷贝的缓冲 ⇒ **静默错值** ✗（P1 就是这么炸的 ✗）
        arena_idle_locked(a);
        // ① 若上次出现过"中途不足"（g_need_hint）⇒ 一次扩到位 ✓（否则尾部大 split 反复回落 ✗）
        if (g_need_hint) {
            const size_t want_hint = g_need_hint + g_need_hint / 8 + (size_t)(1u << 20);
            if (want_hint > a.cap && want_hint <= g_cap_max) {
                arena_idle_locked(a);                 // 释放前必须确认无在飞拷贝 ✓
                free_arena_locked(a);
                if (cudaHostAlloc(&a.ptr, want_hint, cudaHostAllocDefault) == cudaSuccess) {
                    a.cap = want_hint;
                    ++g_n_grow;
                    if (g_dbg) {
                        std::fprintf(stderr,
                                     "[STAGING] arena[%d] 按上次用量扩容 → %.2f MB（第 %ld 次）\n",
                                     g_cur, (double)a.cap / 1048576.0, g_n_grow);
                    }
                } else {
                    cudaGetLastError();
                    a.ptr = nullptr;
                    a.cap = 0;
                }
            }
            g_need_hint = 0;                          // 提示已消费 ✓
        }
        // ② 常规扩容：首分配就装不下
        if (a.used + need > a.cap) {
            if (need > g_cap_max) {
                ++g_n_fallback;
                if (g_dbg) {
                    std::fprintf(stderr,
                                 "[STAGING] 单次需求 %.1f MB > 每块上限 %.1f MB ⇒ 回落阻塞拷贝 ✓\n",
                                 (double)need / 1048576.0, (double)g_cap_max / 1048576.0);
                }
                return nullptr;
            }
            size_t want = need + need / 4 + (size_t)(1u << 20);
            if (want > g_cap_max) want = g_cap_max;
            arena_idle_locked(a);
            free_arena_locked(a);
            if (cudaHostAlloc(&a.ptr, want, cudaHostAllocDefault) != cudaSuccess) {
                cudaGetLastError();
                a.ptr = nullptr;
                a.cap = 0;
                ++g_n_fallback;
                if (g_dbg) {
                    std::fprintf(stderr, "[STAGING] cudaHostAlloc(%zu B) 失败 ⇒ 回落阻塞拷贝 ✓\n", want);
                }
                return nullptr;
            }
            a.cap = want;
            ++g_n_grow;
            if (g_dbg) {
                std::fprintf(stderr, "[STAGING] arena[%d] 扩容 → %.2f MB（第 %ld 次）\n",
                             g_cur, (double)a.cap / 1048576.0, g_n_grow);
            }
        }
    }

    if (a.used + need > a.cap) {
        // 中途不足 ⇒ 记下总量，下次首分配一次扩到位 ✓（然后本次回落阻塞拷贝 ✓）
        if (a.used + need > g_need_hint) g_need_hint = a.used + need;
        ++g_n_fallback;
        if (g_dbg) {
            std::fprintf(stderr,
                         "[STAGING] arena[%d] 中途不足（used=%.1f MB need=%.1f KB）⇒ 本次回落阻塞拷贝 ✓\n",
                         g_cur, (double)a.used / 1048576.0, (double)need / 1024.0);
        }
        return nullptr;
    }

    void* p = (char*)a.ptr + a.used;
    a.used += need;
    return p;
}

// ------------------------------------------------------------
// 异步拷贝（入队）
// ------------------------------------------------------------
int staging_d2h(void* dst_pinned, const void* src_dev, int64_t bytes) {
    if (!dst_pinned || !src_dev || bytes <= 0) return 1;
    std::lock_guard<std::mutex> lk(g_mu);
    if (!ensure_init_locked()) return 1;
    cudaEvent_t e = ring_take();
    if (cudaEventRecord(e, nullptr) != cudaSuccess) { cudaGetLastError(); return 1; }   // 生产者在默认流 ✓
    if (cudaStreamWaitEvent(g_copy, e, 0) != cudaSuccess) { cudaGetLastError(); return 1; }
    if (cudaMemcpyAsync(dst_pinned, src_dev, (size_t)bytes, cudaMemcpyDeviceToHost, g_copy) != cudaSuccess) {
        cudaGetLastError();
        return 1;
    }
    ++g_n_d2h;
    g_b_d2h += bytes;
    g_dirty = true;         // split 末需要 record 完成事件（复用该块 arena 前要 query ✓）
    return 0;
}

int staging_h2d(void* dst_dev, const void* src_pinned, int64_t bytes) {
    if (!dst_dev || !src_pinned || bytes <= 0) return 1;
    std::lock_guard<std::mutex> lk(g_mu);
    if (!ensure_init_locked()) return 1;
    if (cudaMemcpyAsync(dst_dev, src_pinned, (size_t)bytes, cudaMemcpyHostToDevice, g_copy) != cudaSuccess) {
        cudaGetLastError();
        return 1;
    }
    cudaEvent_t e = ring_take();
    if (cudaEventRecord(e, g_copy) != cudaSuccess) { cudaGetLastError(); return 1; }
    if (cudaStreamWaitEvent(nullptr, e, 0) != cudaSuccess) { cudaGetLastError(); return 1; }  // 设备侧排序 ✓
    ++g_n_h2d;
    g_b_h2d += bytes;
    g_dirty = true;
    return 0;
}

int staging_wait() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_ok) return 1;
    if (cudaEventRecord(g_done, g_copy) != cudaSuccess) { cudaGetLastError(); return 1; }
    if (cudaEventSynchronize(g_done) != cudaSuccess) { cudaGetLastError(); return 1; }
    cudaGetLastError();     // 清残留
    return 0;
}

// split 末：**不 drain** ✗（P2）——只 record 本块 arena 的完成事件 + 切到另一块 ✓
//   切块时对"即将使用的那块"做一次非阻塞 query：已完成 ⇒ 零等待 ✓；未完成 ⇒ 才同步等（极少 ✓）
void staging_split_end() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_ok) return;
    Arena& a = g_arena[g_cur];
    if (g_dirty && a.busy) {
        if (cudaEventRecord(a.busy, g_copy) == cudaSuccess) a.busy_valid = true;
    }
    const size_t used = a.used;
    a.used = 0;                       // 该块本轮用完（复用前由 query 把关 ✓）
    g_dirty = false;
    g_cur = (g_cur + 1) % kArenaN;    // ★ 双缓冲轮换 ✓
    if (g_dbg) {
        std::fprintf(stderr,
                     "[STAGING] split 结束：用 %.1f MB（arena[%d]）→ 切到 arena[%d]｜"
                     "累计 D2H=%ld 次/%.1f MB，H2D=%ld 次/%.1f MB，回落=%ld，在飞等待=%ld\n",
                     (double)used / 1048576.0, (g_cur + kArenaN - 1) % kArenaN, g_cur,
                     g_n_d2h, (double)g_b_d2h / 1048576.0,
                     g_n_h2d, (double)g_b_h2d / 1048576.0,
                     g_n_fallback, g_n_busy_wait);
    }
}

} // namespace ppml

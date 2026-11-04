#ifndef PPML_ALIAS_GUARD_H
#define PPML_ALIAS_GUARD_H

// ============================================================
// AliasGuard —— "同一地址被两个活张量占用" 断言（诊断，默认关）
// ============================================================
//   背景（2026-10-20 训练诊断）：
//     CUDA 混合调度下原始梯度 1e16~1e19（纯 CPU 仅 1e4），且同一配置多次运行量级不同；
//     逐节点执行轨迹抓到 `MUL_MAT max=1.85e+02 → SQR max=1.93e-07`（sqr 的输入是 view(x)，
//     而同层 sqrt(1.93e-7)=4.39e-4 与 SQR 自洽）⇒ 消费 view 别名的 kernel 读到的
//     **不是生产者的内容**。`PPML_CUDA_DISABLE_OPS` 任意一组都能"治好"，即对 split 布局敏感
//     ⇒ 属"假别名"（两段本不该共享的物理内存被同时使用）。
//
//   本模块在**分配/绑定**处登记 (地址, 字节数)，登记时对同 pool + 跨 pool 的已登记区间做
//   区间重叠检查：一旦新张量落在另一个**仍存活**张量的区间上，立即打印 `[ALIAS]`（可选 abort）。
//
//   维护点（必须成对，否则只会多报/漏报，不会误伤正确路径）：
//     · Gallocr::bind_tensor  → register(pool="gallocr")     ← 分配后绑定
//     · Gallocr::free_node    → unregister                    ← 区间还给 talloc（可复用，合法）
//     · Gallocr::reserve 的 reset / release → drop_pool("gallocr")
//     · BackendScheduler::reserve_graph_memory 步骤7 → register(pool="cpy")
//     · 上述 cpy 解绑处 → unregister / reserved_buffers_ 释放后 drop_pool("cpy")
//     · transfer_params_to_backend → register(pool="param")，param_buffers_ 释放后 drop_pool
//
//   合法别名（不检查）：view 张量本身（view_src != nullptr，与根共享内存是本项目的设计）。
//
//   开关：PPML_ALIAS_GUARD=1（开启）；PPML_ALIAS_ABORT=1（命中即 abort，便于 gdb 抓栈）；
//         PPML_ALIAS_GUARD_MAX=N（明细打印上限，默认 20）。
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

#include "ppml/Tensor.h"

namespace ppml {
namespace aliasguard {

struct Entry {
    const TensorF32* t = nullptr;
    size_t           bytes = 0;
    const char*      pool  = "?";
    long long        numel = 0;
    int              op    = -1;
    bool             live  = true;
};

struct State {
    bool      on           = false;
    bool      abort_on_hit = false;
    int       print_max    = 20;
    int       printed      = 0;
    long long hits         = 0;
    long long regs         = 0;
    std::map<std::string, std::map<uintptr_t, Entry>>           pools;      // pool → (addr → entry)
    std::map<const TensorF32*, std::pair<std::string, uintptr_t>> by_tensor;  // 张量 → (pool, addr)
};

inline State& state() {
    static State s = [] {
        State st;
        st.on           = (std::getenv("PPML_ALIAS_GUARD") != nullptr);
        st.abort_on_hit = (std::getenv("PPML_ALIAS_ABORT") != nullptr);
        const char* e   = std::getenv("PPML_ALIAS_GUARD_MAX");
        st.print_max    = e ? std::atoi(e) : 20;
        return st;
    }();
    return s;
}

inline void unregister(const TensorF32* t) {
    State& st = state();
    if (!st.on || !t) return;
    auto it = st.by_tensor.find(t);
    if (it == st.by_tensor.end()) return;
    auto pit = st.pools.find(it->second.first);
    if (pit != st.pools.end()) {
        auto e = pit->second.find(it->second.second);
        if (e != pit->second.end() && e->second.t == t) pit->second.erase(e);
    }
    st.by_tensor.erase(it);
}

// 登记一个张量的地址区间；若与任何已登记的存活张量重叠 ⇒ 打印 [ALIAS]（可选 abort）。
inline void register_range(const TensorF32* t, const void* addr, size_t bytes, const char* pool) {
    State& st = state();
    if (!st.on || !t || !addr || bytes == 0) return;
    // view 与根共享内存是本项目设计（零拷贝别名），不参与"独立占用"检查
    if (t->view_src != nullptr) return;

    unregister(t);   // 重绑（同一张量换地址）先撤旧登记

    const uintptr_t lo = reinterpret_cast<uintptr_t>(addr);
    const uintptr_t hi = lo + bytes;

    for (auto& kv : st.pools) {
        auto& m = kv.second;
        // 候选 1：≤ lo 的最近一条（其尾可能越过 lo）
        auto it = m.upper_bound(lo);
        if (it != m.begin()) {
            auto prev = std::prev(it);
            const uintptr_t pw = prev->first + prev->second.bytes;
            if (lo < pw) {
                ++st.hits;
                if (st.printed < st.print_max) {
                    ++st.printed;
                    const Entry& e = prev->second;
                    std::fprintf(stderr,
                        "[ALIAS] 同一地址被两个活张量占用：新 pool=%s t=%p op=%d numel=%lld "
                        "[%p,+%zu) | 已占用 pool=%s t=%p op=%d numel=%lld [%p,+%zu) live=%d\n",
                        pool, (const void*)t, (int)t->op, (long long)t->numel(), (void*)lo, bytes,
                        e.pool, (const void*)e.t, e.op, e.numel,
                        (void*)prev->first, e.bytes, (int)e.live);
                    std::fflush(stderr);
                }
                if (st.abort_on_hit) std::abort();
            }
        }
        // 候选 2：≥ lo 的最近一条（其头可能落在 [lo, hi) 内）
        it = m.lower_bound(lo);
        if (it != m.end() && it->first < hi) {
            ++st.hits;
            if (st.printed < st.print_max) {
                ++st.printed;
                const Entry& e = it->second;
                std::fprintf(stderr,
                    "[ALIAS] 同一地址被两个活张量占用：新 pool=%s t=%p op=%d numel=%lld "
                    "[%p,+%zu) | 已占用 pool=%s t=%p op=%d numel=%lld [%p,+%zu) live=%d\n",
                    pool, (const void*)t, (int)t->op, (long long)t->numel(), (void*)lo, bytes,
                    e.pool, (const void*)e.t, e.op, e.numel,
                    (void*)it->first, e.bytes, (int)e.live);
                std::fflush(stderr);
            }
            if (st.abort_on_hit) std::abort();
        }
    }

    Entry e;
    e.t = t; e.bytes = bytes; e.pool = pool;
    e.numel = (long long)t->numel(); e.op = (int)t->op; e.live = true;
    st.pools[pool][lo] = e;
    st.by_tensor[t]   = std::make_pair(std::string(pool), lo);
    ++st.regs;
}

inline void drop_pool(const char* pool) {
    State& st = state();
    if (!st.on || !pool) return;
    auto pit = st.pools.find(pool);
    if (pit == st.pools.end()) return;
    for (auto& kv : pit->second) {
        auto it = st.by_tensor.find(kv.second.t);
        if (it != st.by_tensor.end() && it->second.first == pool) st.by_tensor.erase(it);
    }
    st.pools.erase(pit);
}

// 汇总（在 release/收尾处调用）：有命中才打印，避免污染正常日志
inline void report(const char* where) {
    State& st = state();
    if (!st.on || st.hits == 0) return;
    std::fprintf(stderr,
                 "[ALIAS] 汇总(%s)：命中=%lld（明细仅前 %d 条）｜累计登记=%lld\n",
                 where ? where : "?", st.hits, st.print_max, st.regs);
    std::fflush(stderr);
}

}  // namespace aliasguard
}  // namespace ppml

#endif  // PPML_ALIAS_GUARD_H

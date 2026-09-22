// ============================================================
// MulMatStats.h — MUL_MAT 形状分布 + 耗时统计（Step 0，2026-09-22）
//
// 目的：回答"MUL_MAT 在 wall 里占多少、形状分布如何"，用于决定 tensor-core
//       （TF32）接入的收益上界；不依赖 nsys/ncu（CPU/GPU 两侧都能统计）。
//
// 用法：PPML_MULMAT_STATS=1 ./ppml_train ...   （默认关闭，关闭时开销≈一次 static bool）
//   输出（程序退出时一次，走 atexit）：
//     [MULMAT-STAT] 合计 calls=… 累计=… ms | wall(自首个 MUL_MAT 起)=… ms ⇒ 占比 …%
//     [MULMAT-STAT]   CUDA/tf32-smem        calls=… 累计=… ms (…%)
//     [MULMAT-STAT]   CPU/simt-naive        calls=… 累计=… ms (…%)
//     [MULMAT-STAT] top 形状（按累计耗时，M K N）：…
//
// 设计要点：
//   · 只做"累加 + 退出时打印"，调用点开销 = 一次 map 插入（仅开关打开时）；
//   · 跨线程安全（CPU kernel 在 threadpool worker 上调用 ⇒ 加锁）；
//   · CUDA 侧需要 event 汇总（异步），故提供 finalizer 机制：由 src/cuda/CUDAKernels.cu
//     注册 cudaEvent 汇总回调，在 dump 之前统一喂数据（顺序与注册先后无关）。
//   · 故意泄漏单例（new 不 delete）：保证 atexit 阶段对象仍存活（避免静态析构顺序坑）。
// ============================================================
#pragma once

// ⚠️ 本头会被 src/cuda/CUDAKernels.cu 直接 include ⇒ **禁止**引入 <algorithm> / <functional>：
//   nvcc 11.5 + GCC 11.4 的组合在这两个头里报
//   `bits/std_function.h:435 error: parameter packs not expanded with '...'` ✗（实测探针确认）。
//   （本文件因此用自写的选择排序，不用 std::sort。）
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ppml {

inline bool mulmat_stats_enabled() {
    static const bool en = []() {
        const char* s = std::getenv("PPML_MULMAT_STATS");
        return s && *s && std::strcmp(s, "0") != 0;
    }();
    return en;
}

struct MulMatShapeStat {
    long   calls = 0;
    double ms    = 0.0;
};

struct MulMatBucket {
    long   calls = 0;
    double ms    = 0.0;
    std::map<unsigned long long, MulMatShapeStat> shapes;   // key = M<<40 | K<<20 | N
};

struct MulMatStats {
    std::mutex mu;
    std::map<std::string, MulMatBucket> buckets;   // key = "<where>/<path>"
    std::chrono::steady_clock::time_point t_first{};
    bool started = false;
};

inline MulMatStats& mulmat_stats() {
    static MulMatStats* s = new MulMatStats();   // 故意泄漏（见文件头说明）
    return *s;
}

inline unsigned long long mulmat_shape_key(int M, int K, int N) {
    return ((unsigned long long)(M & 0xfffff) << 40) |
           ((unsigned long long)(K & 0xfffff) << 20) |
           ((unsigned long long)(N & 0xfffff));
}

// 外部"收尾回调"（如把 CUDA event 汇总成 ms 后喂进来）；在 dump 之前调用
inline std::vector<void (*)()>& mulmat_stats_finalizers() {
    static std::vector<void (*)()>* v = new std::vector<void (*)()>();
    return *v;
}
inline void mulmat_stats_add_finalizer(void (*f)()) {
    for (void (*g)() : mulmat_stats_finalizers()) if (g == f) return;
    mulmat_stats_finalizers().push_back(f);
}

inline void mulmat_stats_dump() {
    if (!mulmat_stats_enabled()) return;
    MulMatStats& s = mulmat_stats();
    std::lock_guard<std::mutex> lk(s.mu);

    const double wall_ms = s.started
        ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s.t_first).count()
        : 0.0;

    long   calls = 0;
    double ms    = 0.0;
    for (auto& kv : s.buckets) { calls += kv.second.calls; ms += kv.second.ms; }

    std::fprintf(stderr, "\n[MULMAT-STAT] ===== MUL_MAT 形状/耗时统计（PPML_MULMAT_STATS=1）=====\n");
    std::fprintf(stderr,
                 "[MULMAT-STAT] 合计 calls=%ld 累计=%.1f ms | wall(自首个 MUL_MAT 起)=%.0f ms ⇒ 占比 %.1f%%\n",
                 calls, ms, wall_ms, wall_ms > 0.0 ? 100.0 * ms / wall_ms : 0.0);
    for (auto& kv : s.buckets) {
        std::fprintf(stderr, "[MULMAT-STAT]   %-24s calls=%-7ld 累计=%10.1f ms (%.1f%%)  不同形状=%zu\n",
                     kv.first.c_str(), kv.second.calls, kv.second.ms,
                     wall_ms > 0.0 ? 100.0 * kv.second.ms / wall_ms : 0.0,
                     kv.second.shapes.size());
    }

    // top 形状（按累计耗时；跨 where/path 汇总）
    struct Row { std::string key; unsigned long long sk; long calls; double ms; };
    std::vector<Row> rows;
    for (auto& kv : s.buckets)
        for (auto& sh : kv.second.shapes)
            rows.push_back({kv.first, sh.first, sh.second.calls, sh.second.ms});
    // 选择排序（避免 <algorithm>；只在退出时跑一次，规模很小 ✓）
    for (size_t i = 0; i + 1 < rows.size(); ++i) {
        size_t best = i;
        for (size_t j = i + 1; j < rows.size(); ++j)
            if (rows[j].ms > rows[best].ms) best = j;
        if (best != i) { Row t = rows[i]; rows[i] = rows[best]; rows[best] = t; }
    }

    std::fprintf(stderr, "[MULMAT-STAT] top 形状（按累计耗时；M=输出行数 K=收缩维 N=输出列数）：\n");
    for (size_t i = 0; i < rows.size() && i < 12; ++i) {
        const unsigned long long k = rows[i].sk;
        const int M = (int)((k >> 40) & 0xfffff);
        const int K = (int)((k >> 20) & 0xfffff);
        const int N = (int)(k & 0xfffff);
        std::fprintf(stderr,
                     "[MULMAT-STAT]   #%-2zu %-24s M=%-6d K=%-6d N=%-6d calls=%-7ld 累计=%10.1f ms\n",
                     i + 1, rows[i].key.c_str(), M, K, N, rows[i].calls, rows[i].ms);
    }
    std::fprintf(stderr, "[MULMAT-STAT] ===== end =====\n");
}

inline void mulmat_stats_finish() {
    for (void (*f)() : mulmat_stats_finalizers()) f();   // 先汇总（CUDA event 等）
    mulmat_stats_dump();
}

inline void mulmat_stats_register_atexit() {
    static const bool once = []() { std::atexit(mulmat_stats_finish); return true; }();
    (void)once;
}

inline void mulmat_stats_mark_start() {
    MulMatStats& s = mulmat_stats();
    std::lock_guard<std::mutex> lk(s.mu);
    if (!s.started) { s.t_first = std::chrono::steady_clock::now(); s.started = true; }
}

// where: "CPU" | "CUDA"；path: 具体路径名（simt-naive / simt-128x128 / tf32-smem / f16-smem …）
inline void mulmat_stats_record(const char* where, const char* path,
                                int M, int K, int N, double ms) {
    if (!mulmat_stats_enabled()) return;
    mulmat_stats_register_atexit();
    mulmat_stats_mark_start();

    MulMatStats& s = mulmat_stats();
    const std::string key = std::string(where) + "/" + path;
    std::lock_guard<std::mutex> lk(s.mu);
    MulMatBucket& b = s.buckets[key];
    b.calls += 1;
    b.ms    += ms;
    MulMatShapeStat& sh = b.shapes[mulmat_shape_key(M, K, N)];
    sh.calls += 1;
    sh.ms    += ms;
}

}  // namespace ppml

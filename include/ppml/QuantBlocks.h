// QuantBlocks.h — int8 分块量化（Q8_0 / Q8_1）块结构 + 尺寸换算 + host 参考实现
// ============================================================
// 本文件只含「非计算」支持代码（2026-09-10 int8 量化准备）：
//   * 块结构（与 ggml 二进制兼容，便于后续直接读 GGUF 的 Q8_0 权重）
//   * 块元素数/块字节数换算（供 Tensor::nbytes / Gallocr / cpy 使用）
//   * host 端参考量化/反量化（给测试与后续 kernel 做数值对照；不是 CUDA kernel）
//   * `QUANT_MUL_MAT_KERNELS_READY` 开关：kernel 未就绪前不放开调度
//
// 约定（与 ggml 一致）：
//   Q8_0 = 对称量化：每 32 元素一块 { half d; int8_t qs[32]; }              (34 B)
//          d = amax/127，q = round(x/d) ∈ [-127,127]（不用 -128）
//   Q8_1 = 对称量化 + 块内 sum：每 32 元素一块 { half2 ds; int8_t qs[32]; } (36 B)
//          ds.x = d（scale）；ds.y = s = d * Σq_i（= 反量化后的块和，与 ggml 一致）
//          —— sum 是给「Q8_1 激活 × 带 zero-point 的量化权重」点积做修正项用的
//          （纯 Q8_0×Q8_0 对称组合不需要它）。
//
// ⚠️ 尺寸/对齐注意：34/36 字节 ⇒ **不是 16 字节对齐**，CUDA 侧不能用 cp.async/ldmatrix
//    直接搬量化块；后续 kernel 应把块反量化成 half/float 写进 smem 再走 tensor core
//    （即「W8A16 反量化 → 复用 fp16 smem+ldmatrix+cp.async 流水」路线）。
// ============================================================
#pragma once
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>

namespace ppml {

static constexpr int QK8_0 = 32;   // Q8_0 每块元素数
static constexpr int QK8_1 = 32;   // Q8_1 每块元素数

struct alignas(2) block_q8_0 {
    __half  d;                 // delta（scale）
    int8_t  qs[QK8_0];         // quants
};

struct alignas(4) block_q8_1 {
    __half2 ds;                // ds.x = d（scale），ds.y = s = d*Σq（ggml 语义：反量化后块和）
    int8_t  qs[QK8_1];
};

static_assert(sizeof(block_q8_0) == 2 + QK8_0, "block_q8_0 必须为紧凑 34B（与 ggml 兼容）");
static_assert(sizeof(block_q8_1) == 4 + QK8_1, "block_q8_1 必须为紧凑 36B（与 ggml 兼容）");

// 量化 mul_mat 的 kernel 就绪状态（**分后端**，2026-10-17 拆分）。
//   CPU : false —— 只有 host 参考实现（dmmv 参考留在测试里），未放开调度，
//         否则量化输入会进 F32/F16 kernel 按 float 读 34B 块 → 错误/越界。
//   CUDA: true  —— 两条路径按 N 选择（PPML_MMQ / PPML_MMQ_MIN_N 可覆盖）：
//         · MMQ  = mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32（sm_80+，N >= 16，主路径）
//         · MMVQ = __dp4a（N 小 / sm_80 以下 / PPML_MMQ=0）
//         量化权重可在 **src0 或 src1**（另一侧必须是 F32 激活，kernel 内即时量化成 Q8_1
//         到 scratch），dst = F32。要求 K % QK8_0 == 0（量化块不能跨行，quant_shape_valid 保证）。
//         对齐实况：Q8_0 块步长 34B ⇒ qs 只有 2B 对齐（每 4 个 lane 需 2 条 LDG.E.U16）；
//         Q8_1 块步长 36B + qs 偏移 4 ⇒ 永远 4B 对齐（单条 32 位读）。
static constexpr bool QUANT_MUL_MAT_CPU_READY  = false;
static constexpr bool QUANT_MUL_MAT_CUDA_READY = true;

// ------------------------------------------------------------------
// host 端参考量化（对称·每块 amax/127）。返回 false ⇒ 元素数不是 QK 的整数倍。
// ------------------------------------------------------------------
inline bool quantize_q8_0_host(const float* src, block_q8_0* dst, int64_t n) {
    if (n % QK8_0 != 0) return false;
    const int64_t nb = n / QK8_0;
    for (int64_t b = 0; b < nb; ++b) {
        const float* x = src + b * QK8_0;
        float amax = 0.f;
        for (int i = 0; i < QK8_0; ++i) amax = std::fmax(amax, std::fabs(x[i]));
        const float d   = amax / 127.0f;
        const float inv = (amax == 0.0f) ? 0.0f : 1.0f / d;    // 全零块：d=0 且 q 全 0
        dst[b].d = __float2half(d);
        for (int i = 0; i < QK8_0; ++i) {
            float q = std::round(x[i] * inv);
            q = q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q);
            dst[b].qs[i] = static_cast<int8_t>(q);
        }
    }
    return true;
}

inline bool quantize_q8_1_host(const float* src, block_q8_1* dst, int64_t n) {
    if (n % QK8_1 != 0) return false;
    const int64_t nb = n / QK8_1;
    for (int64_t b = 0; b < nb; ++b) {
        const float* x = src + b * QK8_1;
        // amax = max|x|；vmax = 取到 amax 的那个「带符号」值（ggml 语义）
        float amax = 0.f, vmax = 0.f;
        for (int i = 0; i < QK8_1; ++i) {
            const float v = x[i];
            if (std::fabs(v) > amax) { amax = std::fabs(v); vmax = v; }
        }
        // 激活量化器：d = max / -128（不是 amax/127）。
        //   · 把最大幅值映射到 -128 —— int8 里唯一"多出来"的码位 ⇒ 吃满 [-128,127] 全区间；
        //   · 除数是 2 的幂 ⇒ 二进制浮点除法精确（只改指数），d 的误差仅来自 fp16 存储；
        //   · d 可带符号；代价是「反号极值」需要 +128 → 必须饱和到 127（下面 clamp）。
        // 全零块显式取 +0.0（否则 0/-128 = -0.0，与 device 侧的 +0.0 字节不同）
        const float d   = (amax == 0.0f) ? 0.0f : vmax / -128.0f;
        const float inv = (d == 0.0f) ? 0.0f : 1.0f / d;
        int32_t qsum = 0;
        for (int i = 0; i < QK8_1; ++i) {
            float q = std::round(x[i] * inv);
            q = q > 127.0f ? 127.0f : (q < -128.0f ? -128.0f : q);
            const int8_t qi = static_cast<int8_t>(q);
            dst[b].qs[i] = qi;
            qsum += qi;
        }
        // ds.y = s = d * Σq_i（ggml 语义：反量化后的块和；不是原始 float 块和）
        dst[b].ds = __floats2half2_rn(d, d * static_cast<float>(qsum));
    }
    return true;
}

// ==================================================================
// Q8_0R：32B 对齐的 repack 布局（2026-10-17，运行时专用，**不落盘**）
// ------------------------------------------------------------------
// 动机：交错 block_q8_0 的步长是 34B ⇒ qs 只有 2B 对齐 ⇒ CUDA 侧每个 4 字节
//   只能拆成 2 条 LDG.E.U16（实测 MMVQ 每 32 元素块 16 条 u16）。repack 后：
//     qs 平面：块 b（= m*nblk_row + ib）的 32 个 int8 落在 [b*32, b*32+32)
//     d  平面：紧跟 qs 平面之后，块 b 的 half 落在 [nblocks*32 + b*2, +2)
//   ⇒ 基址 16B 对齐时每个 32B 组都 32B 对齐 ⇒ 2 × LDG.E.128（或 4 × u32），
//     总字节数仍是 34B/32 元素（零额外开销）。
// ==================================================================
inline size_t q8_0_repacked_bytes(int64_t numel) {
    return static_cast<size_t>(numel / QK8_0) * (QK8_0 + sizeof(__half));
}

// 交错块 → repack（dst 需 16B 对齐、容量 = q8_0_repacked_bytes(numel)）
inline void repack_q8_0_host(const block_q8_0* src, int64_t nblocks, void* dst) {
    int8_t* qs = static_cast<int8_t*>(dst);
    __half* d  = reinterpret_cast<__half*>(qs + nblocks * QK8_0);
    for (int64_t b = 0; b < nblocks; ++b) {
        std::memcpy(qs + b * QK8_0, src[b].qs, QK8_0);
        d[b] = src[b].d;
    }
}

// repack → 交错块（host 参考/测试用）
inline void unrepack_q8_0_host(const void* src, int64_t nblocks, block_q8_0* dst) {
    const int8_t* qs = static_cast<const int8_t*>(src);
    const __half* d  = reinterpret_cast<const __half*>(qs + nblocks * QK8_0);
    for (int64_t b = 0; b < nblocks; ++b) {
        std::memcpy(dst[b].qs, qs + b * QK8_0, QK8_0);
        dst[b].d = d[b];
    }
}

// ------------------------------------------------------------------
// half2 分量读取（host 端；__low2half/__high2half 是 device-only intrinsic）
// ------------------------------------------------------------------
inline float half2_low_to_float(const __half2 h) {
    __half h0;
    std::memcpy(&h0, &h, sizeof(__half));
    return __half2float(h0);
}
inline float half2_high_to_float(const __half2 h) {
    __half h1;
    std::memcpy(&h1, reinterpret_cast<const char*>(&h) + sizeof(__half), sizeof(__half));
    return __half2float(h1);
}

// ------------------------------------------------------------------
// host 端参考反量化（供测试/对照使用）
// ------------------------------------------------------------------
inline void dequantize_q8_0_host(const block_q8_0* src, float* dst, int64_t n) {
    const int64_t nb = n / QK8_0;
    for (int64_t b = 0; b < nb; ++b) {
        const float d = __half2float(src[b].d);
        for (int i = 0; i < QK8_0; ++i) dst[b * QK8_0 + i] = d * static_cast<float>(src[b].qs[i]);
    }
}

inline void dequantize_q8_1_host(const block_q8_1* src, float* dst, int64_t n) {
    const int64_t nb = n / QK8_1;
    for (int64_t b = 0; b < nb; ++b) {
        const float d = half2_low_to_float(src[b].ds);
        for (int i = 0; i < QK8_1; ++i) dst[b * QK8_1 + i] = d * static_cast<float>(src[b].qs[i]);
    }
}

} // namespace ppml

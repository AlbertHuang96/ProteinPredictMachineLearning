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
//          ds.x = d（scale）；ds.y = 该块 32 个原始 float 之和
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
    __half2 ds;                // ds.x = d（scale），ds.y = 块内 32 个原始 float 的 sum
    int8_t  qs[QK8_1];
};

static_assert(sizeof(block_q8_0) == 2 + QK8_0, "block_q8_0 必须为紧凑 34B（与 ggml 兼容）");
static_assert(sizeof(block_q8_1) == 4 + QK8_1, "block_q8_1 必须为紧凑 36B（与 ggml 兼容）");

// 量化 mul_mat 的 kernel 是否已就绪。
//   准备阶段 = false：只放开「尺寸计量 / work size / 类型谓词 / host 参考实现」，
//   **不放开调度**（否则会进 F32/F16 kernel 按 float 读块数据 → 错误/越界）。
//   CPU/CUDA int8 kernel 落地后把这里置 true，supports_op 即自动启用。
static constexpr bool QUANT_MUL_MAT_KERNELS_READY = false;

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
        float amax = 0.f, sum = 0.f;
        for (int i = 0; i < QK8_1; ++i) { amax = std::fmax(amax, std::fabs(x[i])); sum += x[i]; }
        const float d   = amax / 127.0f;
        const float inv = (amax == 0.0f) ? 0.0f : 1.0f / d;
        dst[b].ds = __floats2half2_rn(d, sum);                 // x = d, y = sum
        for (int i = 0; i < QK8_1; ++i) {
            float q = std::round(x[i] * inv);
            q = q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q);
            dst[b].qs[i] = static_cast<int8_t>(q);
        }
    }
    return true;
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

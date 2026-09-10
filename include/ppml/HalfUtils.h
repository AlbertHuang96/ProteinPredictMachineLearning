// HalfUtils.h — fp16↔fp32 位运算转换（host 通用；device 直接用 __half2float）
// ============================================================
// 背景（mul_mat fp16 前期支持）：
//   项目图节点统一是 Tensor<float>*，用 type == TENSOR_TYPE_F16 标记"该节点数据按
//   16 位 half 存放"（2 字节/元素，见 Tensor::type_size_bytes()）。kernel 读出后转 fp32
//   做累加（fp32 累加，保留数值精度），输出仍 F32。
// 本文件提供与编译选项无关的纯位运算转换：
//   * 不依赖 F16C / cuda_fp16.h（host 侧无 CUDA 头也能用）
//   * 算法同 ggml（Fabian Giesen 版本），支持 subnormal / ±0 / inf / NaN
//   * 纯函数、无查表、无全局状态；编译需默认 round-to-nearest（勿开 -ffast-math）
// ============================================================
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

namespace ppml {

// ---- 位双关（避免严格别名 UB；编译器会优化为无开销）----
inline float    fp32_from_bits(uint32_t u) { float f;   std::memcpy(&f, &u, sizeof(f)); return f; }
inline uint32_t fp32_to_bits  (float f)    { uint32_t u; std::memcpy(&u, &f, sizeof(u)); return u; }

// fp16(16 位位模式) → fp32：无损（half 的全部取值都可精确表示）
//   正规路径：把 half 的 e/m 摆进 fp32 位置并人为抬高指数 112，再乘 2^-112 还原；
//   非正规路径(e=0)：magic-bias 减法 → mant * 2^-24。
inline float fp16_to_fp32(uint16_t h) {
    const uint32_t w     = static_cast<uint32_t>(h) << 16;  // half 摆到 fp32 高 16 位
    const uint32_t sign  = w & 0x80000000u;
    const uint32_t two_w = w + w;                           // <<1：指数落到 bit31..27（sign 移出）
    // 正规：exp 域 = 224 + e → 真实指数多算了 112，故乘 2^-112
    const float normalized = fp32_from_bits((two_w >> 4) + (0xE0u << 23)) * 0x1.0p-112f;
    // 非正规：|(126<<23) + mant| - 0.5 = mant * 2^-24
    const float denormalized = fp32_from_bits((two_w >> 17) | (126u << 23)) - 0.5f;
    const uint32_t result = sign |
        ((two_w < (1u << 27)) ? fp32_to_bits(denormalized) : fp32_to_bits(normalized));
    return fp32_from_bits(result);
}

// fp32 → fp16(16 位位模式)：round-to-nearest-even（借一次 fp32 加法完成舍入）
//   base = (|f| * 2^112) * 2^-110：先放大用于溢出探测（大数→inf），再缩回（净 ×4）；
//   magic 指数构造后相加，使 |f| 尾数被右移 13 = 23-10 位，硬件在 fp32 加法里完成 half 舍入。
inline uint16_t fp32_to_fp16(float f) {
    const float base = (std::fabs(f) * 0x1.0p+112f) * 0x1.0p-110f;
    const uint32_t w      = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign   = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;             // 下限 clamp 到 2^-14（half 最小正规）
    const uint32_t bits = fp32_to_bits(fp32_from_bits((bias >> 1) + 0x07800000u) + base);
    const uint32_t nonsign = ((bits >> 13) & 0x00007C00u) + (bits & 0x00000FFFu);  // 加法让进位传播
    return static_cast<uint16_t>((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

} // namespace ppml

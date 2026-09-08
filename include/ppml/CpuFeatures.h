// CpuFeatures.h — 运行时 CPU 特性检测（x86/x64 cpuid + XCR0）
// 用途：程序启动时打印当前 CPU 是否支持 SSE/AVX/AVX2/AVX-512(F/DQ/BW/VL)/FMA 等，
//       为后续 SIMD kernel 分派提供依据。非 x86 平台返回全 false（x86=false）。
#pragma once
#include <cstdint>
#include <cstdio>

namespace ppml {

struct CpuFeatures {
    bool x86 = false;                    // 是否 x86/x64（非 x86 平台全 false）
    // ---- SSE 系列 ----
    bool sse = false, sse2 = false, sse3 = false, ssse3 = false;
    bool sse4_1 = false, sse4_2 = false;
    // ---- AVX 系列（含 OS 使能 XCR0 校验：硬件位 + OS 保存 YMM/ZMM 状态都满足才算可用）----
    bool avx = false, avx2 = false;
    bool avx512f = false, avx512dq = false, avx512bw = false, avx512vl = false;
    // ---- 其它 ----
    bool fma = false, f16c = false, bmi1 = false, bmi2 = false;
    char brand[64] = {0};                // CPU 型号串（0x80000002..04），非 x86/失败为空
};

// 检测当前 CPU 特性（每次调用实时 cpuid，开销微秒级；启动时调用一次即可）
CpuFeatures detect_cpu_features();

// 打印检测结果 + 当前编译单元（本 TU）的静态 ISA 基线宏
void print_cpu_features(FILE* out = stdout);

} // namespace ppml

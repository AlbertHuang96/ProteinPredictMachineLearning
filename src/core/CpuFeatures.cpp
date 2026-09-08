// CpuFeatures.cpp — 运行时 CPU 特性检测实现
//   - cpuid 只报告"硬件位"；AVX/AVX-512 是否能真正使用还取决于 OS 是否保存 YMM/ZMM 状态
//     （XCR0 位），故 AVX 相关位一律叠加 XCR0/OSXSAVE 校验（避免非法指令陷阱）。
//   - 平台：x86/x64（GCC/Clang 用 <cpuid.h> + 内联 xgetbv；MSVC 用 __cpuidex/_xgetbv）。
//     其它架构编译为无操作（x86=false，全部 false）。
#include "ppml/CpuFeatures.h"
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define PPML_X86 1
#else
#  define PPML_X86 0
#endif

#if PPML_X86
#  if defined(__GNUC__) || defined(__clang__)
#    include <cpuid.h>
#  elif defined(_MSC_VER)
#    include <intrin.h>
#  endif
#endif

namespace ppml {
namespace {

#if PPML_X86
static inline void cpuid_leaf(unsigned leaf, unsigned subleaf,
                              unsigned& a, unsigned& b, unsigned& c, unsigned& d) {
#if defined(__GNUC__) || defined(__clang__)
    __cpuid_count(leaf, subleaf, a, b, c, d);
#elif defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    a = static_cast<unsigned>(regs[0]);
    b = static_cast<unsigned>(regs[1]);
    c = static_cast<unsigned>(regs[2]);
    d = static_cast<unsigned>(regs[3]);
#else
    a = b = c = d = 0;
#endif
}

// 读 XCR0（需 OSXSAVE=1 后才能调用）：XCR0.1/2 → AVX 状态；XCR0.5/6/7 → AVX-512 状态
static inline unsigned long long xgetbv0() {
#if defined(_MSC_VER)
    return static_cast<unsigned long long>(_xgetbv(0));
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, edx;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" /* xgetbv */ : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
#else
    return 0;
#endif
}
#endif // PPML_X86

} // namespace

CpuFeatures detect_cpu_features() {
    CpuFeatures f;  // 全部默认 false / brand 空
#if !PPML_X86
    return f;
#else
    f.x86 = true;
    unsigned a = 0, b = 0, c = 0, d = 0;

    cpuid_leaf(0, 0, a, b, c, d);
    const unsigned max_leaf = a;
    if (max_leaf < 1) return f;   // 无 leaf1（极老 CPU）：保底返回

    // ---- leaf 1：基础特性 + OSXSAVE ----
    cpuid_leaf(1, 0, a, b, c, d);
    const unsigned ecx1 = c, edx1 = d;
    f.sse     = ((edx1 >> 25) & 1) != 0;
    f.sse2    = ((edx1 >> 26) & 1) != 0;
    f.sse3    = ((ecx1 >>  0) & 1) != 0;
    f.ssse3   = ((ecx1 >>  9) & 1) != 0;
    f.sse4_1  = ((ecx1 >> 19) & 1) != 0;
    f.sse4_2  = ((ecx1 >> 20) & 1) != 0;
    const bool osxsave  = ((ecx1 >> 27) & 1) != 0;
    const bool cpu_avx  = ((ecx1 >> 28) & 1) != 0;
    const bool cpu_fma  = ((ecx1 >> 12) & 1) != 0;
    const bool cpu_f16c = ((ecx1 >> 29) & 1) != 0;

    // ---- leaf 7 subleaf 0：AVX2 / AVX-512 / BMI ----
    unsigned ebx7 = 0;
    if (max_leaf >= 7) {
        cpuid_leaf(7, 0, a, b, c, d);
        ebx7 = b;
    }
    const bool cpu_avx2      = ((ebx7 >>  5) & 1) != 0;
    const bool cpu_avx512f   = ((ebx7 >> 16) & 1) != 0;
    const bool cpu_avx512dq  = ((ebx7 >> 17) & 1) != 0;
    const bool cpu_avx512bw  = ((ebx7 >> 30) & 1) != 0;
    const bool cpu_avx512vl  = ((ebx7 >> 31) & 1) != 0;
    f.bmi1 = ((ebx7 >>  3) & 1) != 0;
    f.bmi2 = ((ebx7 >>  8) & 1) != 0;

    // ---- OS 状态校验（XCR0）----
    const unsigned long long xcr0    = osxsave ? xgetbv0() : 0ULL;
    const bool os_avx    = (xcr0 & 0x6ULL)  == 0x6ULL;    // XMM + YMM
    const bool os_avx512 = (xcr0 & 0xE6ULL) == 0xE6ULL;   // + opmask + ZMM_hi + Hi16_ZMM
    f.avx = cpu_avx && os_avx;
    f.avx2     = f.avx && cpu_avx2;
    const bool avx512_ok = cpu_avx && os_avx512;          // AVX-512 隐含需 OS 保存 AVX 状态
    f.avx512f  = avx512_ok && cpu_avx512f;
    f.avx512dq = avx512_ok && cpu_avx512dq;
    f.avx512bw = avx512_ok && cpu_avx512bw;
    f.avx512vl = avx512_ok && cpu_avx512vl;
    f.fma  = cpu_fma  && os_avx;                          // FMA 涉及 YMM
    f.f16c = cpu_f16c && os_avx;

    // ---- CPU 型号串（0x80000002..0x80000004）----
    cpuid_leaf(0x80000000u, 0, a, b, c, d);
    if (a >= 0x80000004u) {
        char raw[49];
        for (unsigned leaf = 0x80000002u; leaf <= 0x80000004u; ++leaf) {
            cpuid_leaf(leaf, 0, a, b, c, d);
            std::memcpy(raw + static_cast<int>(leaf - 0x80000002u) * 16 +  0, &a, 4);
            std::memcpy(raw + static_cast<int>(leaf - 0x80000002u) * 16 +  4, &b, 4);
            std::memcpy(raw + static_cast<int>(leaf - 0x80000002u) * 16 +  8, &c, 4);
            std::memcpy(raw + static_cast<int>(leaf - 0x80000002u) * 16 + 12, &d, 4);
        }
        raw[48] = '\0';
        // 去首尾空格
        char* s = raw;
        while (*s == ' ') ++s;
        char* e = s + std::strlen(s);
        while (e > s && e[-1] == ' ') *--e = '\0';
        std::snprintf(f.brand, sizeof(f.brand), "%s", s);
    }
    return f;
#endif
}

void print_cpu_features(FILE* out) {
    if (!out) out = stdout;
    const CpuFeatures f = detect_cpu_features();
#define PPML_YN(x) ((x) ? "yes" : "no")
    fprintf(out, "[CPU] x86=%s brand=\"%s\"\n",
            PPML_YN(f.x86), f.brand[0] ? f.brand : "(n/a)");
    fprintf(out, "[CPU] SSE=%s SSE2=%s SSE3=%s SSSE3=%s SSE4.1=%s SSE4.2=%s\n",
            PPML_YN(f.sse), PPML_YN(f.sse2), PPML_YN(f.sse3), PPML_YN(f.ssse3),
            PPML_YN(f.sse4_1), PPML_YN(f.sse4_2));
    fprintf(out, "[CPU] AVX=%s AVX2=%s FMA=%s F16C=%s BMI1=%s BMI2=%s\n",
            PPML_YN(f.avx), PPML_YN(f.avx2), PPML_YN(f.fma), PPML_YN(f.f16c),
            PPML_YN(f.bmi1), PPML_YN(f.bmi2));
    fprintf(out, "[CPU] AVX512F=%s AVX512DQ=%s AVX512BW=%s AVX512VL=%s\n",
            PPML_YN(f.avx512f), PPML_YN(f.avx512dq),
            PPML_YN(f.avx512bw), PPML_YN(f.avx512vl));
    // 本编译单元静态基线（由 CMake/编译器选项决定；当前默认 -O2 无 -march → 通常仅基础 SSE）
    fprintf(out, "[CPU-BUILD] compiled-ISA(this TU):");
#ifdef __AVX512F__
    fprintf(out, " avx512f");
#endif
#ifdef __AVX2__
    fprintf(out, " avx2");
#endif
#ifdef __AVX__
    fprintf(out, " avx");
#endif
#ifdef __FMA__
    fprintf(out, " fma");
#endif
#ifdef __F16C__
    fprintf(out, " f16c");
#endif
#ifdef __SSE4_2__
    fprintf(out, " sse4_2");
#endif
#ifdef __SSE4_1__
    fprintf(out, " sse4_1");
#endif
#ifdef __SSSE3__
    fprintf(out, " ssse3");
#endif
#ifdef __SSE2__
    fprintf(out, " sse2");
#endif
    fprintf(out, "\n");
#undef PPML_YN
}

} // namespace ppml

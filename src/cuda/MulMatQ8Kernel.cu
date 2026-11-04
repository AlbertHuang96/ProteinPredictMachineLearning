// MulMatQ8Kernel.cu —— int8 权重（Q8_0/Q8_1）× F32 激活的矩阵乘法（W8A8，2026-10-17）
// ============================================================
// 数据流（两条路径共用）：
//   1) B（F32 激活，[K,N]，K 最内维连续）→ quantize_rows_q8_1_cuda → scratch（block_q8_1）
//   2) A（Q8_0/Q8_1 权重，[M,K]）原样读取，**不反量化成 float**
//   3) 每 32 个 K 做一次整数点积（int8×int8→int32），再乘 (d_w * d_a) 累加到 fp32
//      —— 即「scale 后置」：整数累加精确，浮点只在每块结算一次（32 项一组）。
//
// 两条 kernel 路径（按 N 选择，见 mul_mat_q8_cuda）：
//   · MMVQ = mul_mat_vec_q8_kernel：1 thread = 1 输出元素。
//       N 小（per_block / 逐 token，M 大 N 小）时用它 —— 没有跨线程协作开销。
//       访存实况（nvcc 11.5 / sm_80 SASS 实测）：权重侧每 32 元素块 **16 条 LDG.E.U16**
//       （8 个 int × 2 个半字），激活侧 8 条 LDG.E（32 位）；**没有任何 128 位向量读**。
//       根因：block_q8_0 步长 34B ⇒ qs 只有 2B 对齐，无法按 32/128 位读。
//   · Q8_0R = 权重在绑定时重排成「qs 平面（32B 组对齐）+ d 平面」的副本（字节数不变）：
//       · mul_mat_mmq_q8r_kernel：64(M)×64(N) 块瓦片 + smem 双缓冲 + 16B 向量装填，
//         A 片段复用 8×（旧版 2×）——**当前主路径**；
//       · mul_mat_vec_q8r_kernel：dp4a 版（小 N / 对照）。
//   · MMQ  = mul_mat_mmq_q8_kernel：mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32，
//       warp 级 16(M)×8(N) 瓦片 × NT 个子块，N 大时是主路径（tensor core int8）。
//       K 方向每 32 元素 = 1 个量化块 = 2 个 mma（k16 半块）→ int32 精确累加 → 块末乘
//       (d_w·d_a) 结算。与 MMVQ 数学等价（同一顺序的浮点结算 ⇒ 期望逐位一致，见单测）。
//
// 布局契约（与 src/backend/CUDAKernels.cpp::kernel_mul_mat_cuda 一致）：
//   A: [K,M] 图布局 ⇒ 权重行 m 的 K 个元素 = A[m*K .. m*K+K)，块号 A_blk[m*(K/32)+ib]
//   B: [K,N] 图布局 ⇒ 列 n 的 K 个元素 = B[n*K .. )，块号 Bq[n*(K/32)+ib]
//   C: [N,M] 图布局，内存按 M 行 N 列存：C[m*N + n]（与 CPU kernel 的 d[j + i*N] 一致）
//   ⇒ C[m,n] = Σ_k A[m,k]·B[n,k]，即 mma 的 A(M×K row-major) × B(K×N col-major)：
//      本仓库 B 的每个 n 行沿 K 连续 ⇒ 恰好是 col-major 的 B，无需转置 ✓
//
// 精度：int8×int8 的积 ≤ 16129，32 项和 ≤ 516k，int32 精确；误差只来自 scale 与 fp16 存储。
// 对齐（重要）：
//   · block_q8_1 是 alignas(4)、36B ⇒ 任意块号的 qs 都 4B 对齐，可直接 (const int*) 读 ✓
//   · block_q8_0 是 alignas(2)、34B ⇒ 块号奇数时 qs 落在 2 (mod 4)
//     ⇒ 不能直接 (const int*)qs（未对齐 32 位访问是 UB / 可能 misaligned address）
//     ⇒ 用 2B 对齐安全的 u16 读 + 拼接（每 4 个 lane 2 条 LDG.E.U16）
//   TODO(perf)：上线前把权重在绑定时重排成 36B/32B 对齐副本，才能用 int4/ldmatrix 向量读；
//              当前实现以「正确性优先」，MMQ 已把 A 的读放大压到每块 4 个字。
// ============================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "ppml/QuantBlocks.h"

namespace ppml {

int quantize_rows_q8_1_cuda(const float* x, block_q8_1* y, int64_t ne, cudaStream_t stream);

namespace {

// 取块 scale：Q8_0 在 d；Q8_1 在 ds.x（ds.y 是该块的 sum，对称权重下用不到）
__device__ __forceinline__ float blk_scale(const block_q8_0& b) { return __half2float(b.d); }
__device__ __forceinline__ float blk_scale(const block_q8_1& b) { return __half2float(__low2half(b.ds)); }

// 把 32 个 int8 读成 8 个 int（每字 4 lane），只依赖 2B 对齐
__device__ __forceinline__ void load_q8_quants(const int8_t* __restrict__ qs,
                                               int* __restrict__ wq) {
    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(qs);   // 2B 对齐 ✓
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t lo = static_cast<uint32_t>(p16[2 * j]);
        const uint32_t hi = static_cast<uint32_t>(p16[2 * j + 1]);
        wq[j] = static_cast<int>(lo | (hi << 16));
    }
}

// ---------- MMQ 用的「4 个 lane」读取（byte_off 必为 4 的倍数） ----------
// Q8_0：块步长 34B ⇒ qs 可能落在 2 (mod 4) ⇒ 拆成两条 2B 读
__device__ __forceinline__ int load_quant4(const block_q8_0* __restrict__ b, int byte_off) {
    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(b->qs + byte_off);
    const uint32_t lo = static_cast<uint32_t>(p16[0]);
    const uint32_t hi = static_cast<uint32_t>(p16[1]);
    return static_cast<int>(lo | (hi << 16));
}
// Q8_1：块步长 36B、qs 偏移 4 ⇒ 永远 4B 对齐 ⇒ 单条 32 位读
__device__ __forceinline__ int load_quant4(const block_q8_1* __restrict__ b, int byte_off) {
    return *reinterpret_cast<const int*>(b->qs + byte_off);
}

// ---------- int8 tensor core：m16n8k16 ----------
//   D/C(16×8, s32) = A(16×16, s8 row-major) × B(16×8, s8 col-major)
//   每 lane 持有：A = 2 × b32（各 4 个 int8），B = 1 × b32（4 个 int8），C/D = 4 × s32
//   lane 分解：g = lane>>2 (0..7)，t = lane&3 (0..3)
//     a0 = A[g][4t .. 4t+3]        a1 = A[g+8][4t .. 4t+3]
//     b0 = B[4t .. 4t+3][g]        （k 行 4t..4t+3，n 列 g）
//     c0 = C[g][2t]   c1 = C[g][2t+1]   c2 = C[g+8][2t]   c3 = C[g+8][2t+1]
//   ⚠️ 注意 B 的列号是 g、C 的列号是 2t —— 两者都铺满 8 列但映射不同（PTX 规定）。
__device__ __forceinline__ void mma_s8_m16n8k16(int (&c)[4], int a_lo, int a_hi, int b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a_lo), "r"(a_hi), "r"(b));
#else
    (void)c; (void)a_lo; (void)a_hi; (void)b;   // arch < sm_80：本 kernel 不会被启动
#endif
}

// 参考级 MMVQ：1 thread = 1 个输出元素 (m, n)。
//   N 小（per_block / 逐 token）时用它；大 batch 走 MMQ（mma.s8）。
// c_trans = 0：C[m*N + n]（量化权重在 src0，dst dims = {N, M}）
// c_trans = 1：C[n*M + m]（量化权重在 src1，dst dims = {M, N} —— LinearLayer 的 mul_mat(x, W)）
template <typename BLK>
__global__ void mul_mat_vec_q8_kernel(const BLK* __restrict__ A,
                                      const block_q8_1* __restrict__ Bq,
                                      float* __restrict__ C,
                                      int M, int N, int nblk, int c_trans) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y;
    if (m >= M || n >= N) return;

    const BLK* arow = A + static_cast<size_t>(m) * nblk;
    const block_q8_1* brow = Bq + static_cast<size_t>(n) * nblk;

    float acc = 0.0f;
    for (int ib = 0; ib < nblk; ++ib) {
        int wq[8];
        load_q8_quants(arow[ib].qs, wq);
        const int* __restrict__ va = reinterpret_cast<const int*>(brow[ib].qs);   // 4B 对齐 ✓
        int sumi = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
        #pragma unroll
        for (int j = 0; j < 8; ++j) sumi = __dp4a(wq[j], va[j], sumi);
#else
        for (int j = 0; j < 8; ++j) {
            for (int t = 0; t < 4; ++t) {
                int a = (wq[j] >> (8 * t)) & 0xFF;
                int b = (va[j] >> (8 * t)) & 0xFF;
                if (a & 0x80) a -= 256;
                if (b & 0x80) b -= 256;
                sumi += a * b;
            }
        }
#endif
        // ★ 每 32 个 K 结算一次：scale 逐块不同，不能把 int32 一路累到底
        acc += static_cast<float>(sumi) * (blk_scale(arow[ib]) * blk_scale(brow[ib]));
    }
    if (c_trans) C[static_cast<size_t>(n) * M + m] = acc;   // 权重在 src1：token 主序
    else         C[static_cast<size_t>(m) * N + n] = acc;
}

// MMQ 主路径：warp 负责 16(M) × (8*NT)(N) 的输出瓦片。
//   · 每个量化块（32 个 K）= 2 个 mma（k16 半块），共用同一个 int32 累加器
//   · 块末一次性乘 (d_w(行) × d_a(列)) 结算 ⇒ 与 MMVQ 完全同序 ⇒ 期望逐位一致
//   · A 的 2 个字（行 g / g+8）在所有 NT 个子块间复用；B 每子块 1 个字
template <typename BLK, int NT>
__global__ void mul_mat_mmq_q8_kernel(const BLK* __restrict__ A,
                                      const block_q8_1* __restrict__ Bq,
                                      float* __restrict__ C,
                                      int M, int N, int nblk, int c_trans) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g    = lane >> 2;      // 0..7：M 行号偏移 / B 的 N 列号
    const int t    = lane & 3;       // 0..3：K 方向 4 个 int8 一组 / C 的列号 2t

    const int m0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 16;
    const int n0 = blockIdx.y * (8 * NT);

    const int m_lo = m0 + g;
    const int m_hi = m0 + g + 8;
    const bool ok_lo = (m_lo < M);
    const bool ok_hi = (m_hi < M);
    // 越界行夹到第 0 行（读到的是合法内存，结果在写出时丢弃）——避免 OOB
    const int row_lo = ok_lo ? m_lo : 0;
    const int row_hi = ok_hi ? m_hi : 0;

    float acc[NT][4];
    #pragma unroll
    for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

    const BLK* __restrict__ a_lo = A + static_cast<size_t>(row_lo) * nblk;
    const BLK* __restrict__ a_hi = A + static_cast<size_t>(row_hi) * nblk;

    for (int ib = 0; ib < nblk; ++ib) {
        const float dw_lo = ok_lo ? blk_scale(a_lo[ib]) : 0.f;
        const float dw_hi = ok_hi ? blk_scale(a_hi[ib]) : 0.f;

        int c[NT][4];
        #pragma unroll
        for (int j = 0; j < NT; ++j) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0;

        #pragma unroll
        for (int h = 0; h < 2; ++h) {                       // 2 个 k16 半块 = 1 个量化块
            const int off = h * 16 + t * 4;                 // 块内 int8 字节偏移（4 的倍数 ✓）
            const int fa_lo = load_quant4(a_lo + ib, off);
            const int fa_hi = load_quant4(a_hi + ib, off);
            #pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int n = n0 + j * 8 + g;                // B 列号 = groupID
                if (n < N) {
                    const block_q8_1* brow = Bq + static_cast<size_t>(n) * nblk + ib;
                    const int fb = *reinterpret_cast<const int*>(brow->qs + off);   // 4B 对齐 ✓
                    mma_s8_m16n8k16(c[j], fa_lo, fa_hi, fb);
                }
            }
        }

        // 块末结算：c 是该 32 个 K 的精确整数和
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int n_c0 = n0 + j * 8 + t * 2;             // C 列号 = 2t
            const int n_c1 = n_c0 + 1;
            float da0 = 0.f, da1 = 0.f;
            if (n_c0 < N) da0 = blk_scale(Bq[static_cast<size_t>(n_c0) * nblk + ib]);
            if (n_c1 < N) da1 = blk_scale(Bq[static_cast<size_t>(n_c1) * nblk + ib]);
            acc[j][0] += static_cast<float>(c[j][0]) * (dw_lo * da0);
            acc[j][1] += static_cast<float>(c[j][1]) * (dw_lo * da1);
            acc[j][2] += static_cast<float>(c[j][2]) * (dw_hi * da0);
            acc[j][3] += static_cast<float>(c[j][3]) * (dw_hi * da1);
        }
    }

    // 写出：c_trans=0 → C[m*N+n]（权重 src0）；c_trans=1 → C[n*M+m]（权重 src1，token 主序）
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int n_c0 = n0 + j * 8 + t * 2;
        const int n_c1 = n_c0 + 1;
        if (c_trans) {
            if (ok_lo && n_c0 < N) C[static_cast<size_t>(n_c0) * M + m_lo] = acc[j][0];
            if (ok_lo && n_c1 < N) C[static_cast<size_t>(n_c1) * M + m_lo] = acc[j][1];
            if (ok_hi && n_c0 < N) C[static_cast<size_t>(n_c0) * M + m_hi] = acc[j][2];
            if (ok_hi && n_c1 < N) C[static_cast<size_t>(n_c1) * M + m_hi] = acc[j][3];
        } else {
            if (ok_lo && n_c0 < N) C[static_cast<size_t>(m_lo) * N + n_c0] = acc[j][0];
            if (ok_lo && n_c1 < N) C[static_cast<size_t>(m_lo) * N + n_c1] = acc[j][1];
            if (ok_hi && n_c0 < N) C[static_cast<size_t>(m_hi) * N + n_c0] = acc[j][2];
            if (ok_hi && n_c1 < N) C[static_cast<size_t>(m_hi) * N + n_c1] = acc[j][3];
        }
    }
#endif
}

// ============================================================
// Q8_0R（32B 对齐 repack）专用路径（2026-10-17）
//   布局：qs 平面 [M][nblk*32]（每组 32B 对齐）+ d 平面 [M][nblk]，总字节 = 34B/32 元素。
//   ⇒ 权重侧从「每 4 字节 2 条 LDG.E.U16」变成「每 32B 组 1 条 LDG.E.128 ×2」。
// ============================================================

// MMVQ 版（小 N / 对照）：每块 8 条 32 位读（原来 16 条 u16）
__global__ void mul_mat_vec_q8r_kernel(const int8_t* __restrict__ Aq,
                                       const __half*  __restrict__ Ad,
                                       const block_q8_1* __restrict__ Bq,
                                       float* __restrict__ C,
                                       int M, int N, int nblk, int c_trans) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y;
    if (m >= M || n >= N) return;

    const int8_t*      arow = Aq + static_cast<size_t>(m) * (nblk * QK8_0);
    const __half*      drow = Ad + static_cast<size_t>(m) * nblk;
    const block_q8_1*  brow = Bq + static_cast<size_t>(n) * nblk;

    float acc = 0.0f;
    for (int ib = 0; ib < nblk; ++ib) {
        const int* __restrict__ qa = reinterpret_cast<const int*>(arow + ib * QK8_0);  // 4B 对齐 ✓
        const int* __restrict__ va = reinterpret_cast<const int*>(brow[ib].qs);        // 4B 对齐 ✓
        int sumi = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
        #pragma unroll
        for (int j = 0; j < 8; ++j) sumi = __dp4a(qa[j], va[j], sumi);
#else
        for (int j = 0; j < 8; ++j) {
            for (int q = 0; q < 4; ++q) {
                int a = (qa[j] >> (8 * q)) & 0xFF;
                int b = (va[j] >> (8 * q)) & 0xFF;
                if (a & 0x80) a -= 256;
                if (b & 0x80) b -= 256;
                sumi += a * b;
            }
        }
#endif
        acc += static_cast<float>(sumi) *
               (__half2float(drow[ib]) * blk_scale(brow[ib]));
    }
    if (c_trans) C[static_cast<size_t>(n) * M + m] = acc;
    else         C[static_cast<size_t>(m) * N + n] = acc;
}

// ---------- MMQ 主路径：块瓦片 64(M) × 64(N) + smem 双缓冲 ----------
//   128 线程 = 4 warp；每 warp 负责 16 行 × 全部 64 列（NT=8）⇒ A 片段复用 8×
//   （旧版每 warp 只有 16 列 ⇒ 复用 2×）。装填用 16B 向量读（uint4），
//   与计算重叠（staging 下一块，计算当前块）。
constexpr int kQ8rTileM = 64;
constexpr int kQ8rTileN = 64;
constexpr int kQ8rNT    = 8;     // 每 warp 的 8 列子块（8 × 8 = 64 列）

__global__ __launch_bounds__(128)
void mul_mat_mmq_q8r_kernel(const int8_t* __restrict__ Aq,
                            const __half*  __restrict__ Ad,
                            const block_q8_1* __restrict__ Bq,
                            float* __restrict__ C,
                            int M, int N, int nblk, int c_trans) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    // B 块按 9 个 32 位字存（word0 = ds，word1..8 = 32B qs），保证 4B 对齐
    __shared__ int8_t   sAq[2][kQ8rTileM][QK8_0];
    __shared__ __half   sAd[2][kQ8rTileM];
    __shared__ uint32_t sB [2][kQ8rTileN][9];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int t    = lane & 3;

    const int m0 = blockIdx.x * kQ8rTileM;
    const int n0 = blockIdx.y * kQ8rTileN;

    // 装填一块到 buf：A（2 × 16B/行，128 线程各 1 个 uint4）+ A scale + B（9 words/列）
    auto stage = [&](int buf, int ib) {
        {
            const int row  = tid >> 1;                 // 0..63
            const int half = tid & 1;                  // 0/1
            const int mrow = m0 + row;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (mrow < M) {
                const int8_t* p = Aq + static_cast<size_t>(mrow) * (nblk * QK8_0)
                                     + ib * QK8_0 + half * 16;
                v = *reinterpret_cast<const uint4*>(p);      // 16B 对齐 ✓
            }
            *reinterpret_cast<uint4*>(&sAq[buf][row][half * 16]) = v;
        }
        if (tid < kQ8rTileM) {
            const int mrow = m0 + tid;
            sAd[buf][tid] = (mrow < M) ? Ad[static_cast<size_t>(mrow) * nblk + ib]
                                       : __float2half(0.f);
        }
        for (int i = tid; i < kQ8rTileN * 9; i += 128) {
            const int col  = i / 9;
            const int w    = i - col * 9;
            const int ncol = n0 + col;
            uint32_t v = 0;
            if (ncol < N) {
                const block_q8_1* b = Bq + static_cast<size_t>(ncol) * nblk + ib;
                v = reinterpret_cast<const uint32_t*>(b)[w];   // 36B 块 4B 对齐 ✓
            }
            sB[buf][col][w] = v;
        }
    };

    float acc[kQ8rNT][4];
    #pragma unroll
    for (int j = 0; j < kQ8rNT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

    stage(0, 0);
    __syncthreads();

    const int row_lo = warp * 16 + g;
    const int row_hi = row_lo + 8;

    for (int ib = 0; ib < nblk; ++ib) {
        const int buf = ib & 1;
        if (ib + 1 < nblk) stage(buf ^ 1, ib + 1);      // 与下面这段计算重叠

        const float dw_lo = __half2float(sAd[buf][row_lo]);
        const float dw_hi = __half2float(sAd[buf][row_hi]);

        // 两个 pass × 4 个 N 子块：c[4][4] 复用，压住寄存器（acc 32 + c 16）
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            int c[4][4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0;

            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int off = h * 16 + t * 4;
                const int a_lo = *reinterpret_cast<const int*>(&sAq[buf][row_lo][off]);  // 4B 对齐 ✓
                const int a_hi = *reinterpret_cast<const int*>(&sAq[buf][row_hi][off]);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = (pass * 4 + j) * 8 + g;               // B 列号 = groupID
                    const int b = static_cast<int>(sB[buf][col][(4 + off) >> 2]);
                    mma_s8_m16n8k16(c[j], a_lo, a_hi, b);
                }
            }

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int jj   = pass * 4 + j;
                const int col0 = jj * 8 + t * 2;                          // C 列号 = 2t
                const int col1 = col0 + 1;
                const int n_0  = n0 + col0;
                const int n_1  = n0 + col1;
                const float da0 = (n_0 < N)
                    ? __half2float(__ushort_as_half(static_cast<unsigned short>(sB[buf][col0][0] & 0xFFFFu)))
                    : 0.f;
                const float da1 = (n_1 < N)
                    ? __half2float(__ushort_as_half(static_cast<unsigned short>(sB[buf][col1][0] & 0xFFFFu)))
                    : 0.f;
                acc[jj][0] += static_cast<float>(c[j][0]) * (dw_lo * da0);
                acc[jj][1] += static_cast<float>(c[j][1]) * (dw_lo * da1);
                acc[jj][2] += static_cast<float>(c[j][2]) * (dw_hi * da0);
                acc[jj][3] += static_cast<float>(c[j][3]) * (dw_hi * da1);
            }
        }
        __syncthreads();     // 本轮读者完成 ⇒ 下一轮可覆盖本缓冲
    }

    const int m_lo = m0 + row_lo;
    const int m_hi = m0 + row_hi;
    const bool ok_lo = (m_lo < M);
    const bool ok_hi = (m_hi < M);

    #pragma unroll
    for (int j = 0; j < kQ8rNT; ++j) {
        const int n_0 = n0 + j * 8 + t * 2;
        const int n_1 = n_0 + 1;
        if (c_trans) {
            if (ok_lo && n_0 < N) C[static_cast<size_t>(n_0) * M + m_lo] = acc[j][0];
            if (ok_lo && n_1 < N) C[static_cast<size_t>(n_1) * M + m_lo] = acc[j][1];
            if (ok_hi && n_0 < N) C[static_cast<size_t>(n_0) * M + m_hi] = acc[j][2];
            if (ok_hi && n_1 < N) C[static_cast<size_t>(n_1) * M + m_hi] = acc[j][3];
        } else {
            if (ok_lo && n_0 < N) C[static_cast<size_t>(m_lo) * N + n_0] = acc[j][0];
            if (ok_lo && n_1 < N) C[static_cast<size_t>(m_lo) * N + n_1] = acc[j][1];
            if (ok_hi && n_0 < N) C[static_cast<size_t>(m_hi) * N + n_0] = acc[j][2];
            if (ok_hi && n_1 < N) C[static_cast<size_t>(m_hi) * N + n_1] = acc[j][3];
        }
    }
#endif
}

// ---------- 激活量化 scratch（grow-only，避免每次 mul_mat 都 cudaMalloc） ----------
block_q8_1* g_act_scratch        = nullptr;
size_t      g_act_scratch_blocks = 0;
std::mutex  g_act_scratch_mtx;

block_q8_1* ensure_scratch(size_t blocks) {
    if (blocks <= g_act_scratch_blocks) return g_act_scratch;
    if (g_act_scratch) {
        cudaFree(g_act_scratch);
        g_act_scratch = nullptr;
        g_act_scratch_blocks = 0;
    }
    block_q8_1* p = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&p), blocks * sizeof(block_q8_1)) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    g_act_scratch = p;
    g_act_scratch_blocks = blocks;
    return p;
}

int env_int(const char* name, int def) {
    const char* s = std::getenv(name);
    return s ? std::atoi(s) : def;
}

// 设备算力（main.minor）；< 8.0 时 mma.s8（m16n8k16）不可用 ⇒ 回落 MMVQ
int device_cc_major(int dev) {
    static int cached = -1;
    if (cached < 0) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
            cudaGetLastError();
            major = 0;
        }
        cached = major;
    }
    return cached;
}

} // namespace

// A: 量化权重（wtype 指定 Q8_0 / Q8_1）；B: F32 激活 [K,N]；C: 输出
//   c_trans=0 → C[m*N + n]（量化权重在 src0，dst dims = {N, M}）
//   c_trans=1 → C[n*M + m]（量化权重在 src1，dst dims = {M, N}，即 mul_mat(x, W)）
// path_out（可空）：1 = MMQ(mma.s8)，0 = MMVQ(dp4a)，便于调用方打点/统计。
int mul_mat_q8_cuda(const void* A, const float* B, float* C,
                    int M, int N, int K, int wtype, int c_trans, int* path_out) {
    if (path_out) *path_out = 0;
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) return 1;
    if ((K % QK8_0) != 0) return 2;                    // 量化块不跨行
    const int nblk = K / QK8_0;

    std::lock_guard<std::mutex> lock(g_act_scratch_mtx);   // scratch 复用 ⇒ 同进程内串行化
    block_q8_1* qa = ensure_scratch(static_cast<size_t>(nblk) * static_cast<size_t>(N));
    if (!qa) return 3;

    if (quantize_rows_q8_1_cuda(B, qa, static_cast<int64_t>(K) * N, /*stream=*/0) != 0) return 4;

    // ===== 路径选择（2026-10-17）：N 大 + sm_80+ → MMQ(mma.s8)，否则 MMVQ(dp4a) =====
    //   Q8_0R（repack）例外：它的 MMVQ 版要读两个平面（qs / d 分离）⇒ 实测比交错版
    //   MMVQ 慢约 40%，而 smem MMQ 在 N=64..8192 全都快一个数量级 ⇒ 除非显式
    //   PPML_MMQ=0，否则 repack 权重一律走 MMQ（小 N 只是列瓦片填不满，仍然正确）。
    const int min_n   = env_int("PPML_MMQ_MIN_N", 16);
    const bool mmq_on = (env_int("PPML_MMQ", 1) != 0);
    const bool use_mmq = mmq_on && (device_cc_major(0) >= 8) &&
                         (N >= min_n || wtype == 2);

    if (use_mmq) {
        if (wtype == 2) {            // 2 = Q8_0R（32B 对齐 repack）：smem 双缓冲主路径
            const int8_t* Aq = static_cast<const int8_t*>(A);
            const size_t  nb_all = static_cast<size_t>(M) * static_cast<size_t>(nblk);
            const __half* Ad = reinterpret_cast<const __half*>(Aq + nb_all * QK8_0);
            dim3 grid((M + kQ8rTileM - 1) / kQ8rTileM,
                      (N + kQ8rTileN - 1) / kQ8rTileN);
            mul_mat_mmq_q8r_kernel<<<grid, 128>>>(Aq, Ad, qa, C, M, N, nblk, c_trans);
            if (cudaGetLastError() != cudaSuccess) return 6;
            if (path_out) *path_out = 1;
            return 0;
        }
        constexpr int kWarps = 4;    // 每 block 4 个 warp
        constexpr int kNT    = 2;    // 每 warp 16 列（2 × 8）
        const int rows_per_block = 16 * kWarps;
        dim3 grid((M + rows_per_block - 1) / rows_per_block,
                  (N + 8 * kNT - 1) / (8 * kNT));
        if (wtype == 0) {
            mul_mat_mmq_q8_kernel<block_q8_0, kNT><<<grid, 32 * kWarps>>>(
                static_cast<const block_q8_0*>(A), qa, C, M, N, nblk, c_trans);
        } else if (wtype == 1) {
            mul_mat_mmq_q8_kernel<block_q8_1, kNT><<<grid, 32 * kWarps>>>(
                static_cast<const block_q8_1*>(A), qa, C, M, N, nblk, c_trans);
        } else {
            return 5;
        }
        if (cudaGetLastError() != cudaSuccess) return 6;
        if (path_out) *path_out = 1;
        return 0;
    }

    constexpr int kThreads = 128;
    dim3 grid((M + kThreads - 1) / kThreads, N);
    if (wtype == 2) {                       // 2 = Q8_0R（repack，4B 对齐 ⇒ 每块 8 条 32 位读）
        const int8_t* Aq = static_cast<const int8_t*>(A);
        const size_t  nb_all = static_cast<size_t>(M) * static_cast<size_t>(nblk);
        const __half* Ad = reinterpret_cast<const __half*>(Aq + nb_all * QK8_0);
        mul_mat_vec_q8r_kernel<<<grid, kThreads>>>(Aq, Ad, qa, C, M, N, nblk, c_trans);
    } else if (wtype == 0) {                // 0 = Q8_0 权重（34B 块）
        mul_mat_vec_q8_kernel<block_q8_0><<<grid, kThreads>>>(
            static_cast<const block_q8_0*>(A), qa, C, M, N, nblk, c_trans);
    } else if (wtype == 1) {                // 1 = Q8_1 权重（36B 块，qs 在 ds 之后）
        mul_mat_vec_q8_kernel<block_q8_1><<<grid, kThreads>>>(
            static_cast<const block_q8_1*>(A), qa, C, M, N, nblk, c_trans);
    } else {
        return 5;
    }
    if (path_out) *path_out = 0;
    return (cudaGetLastError() == cudaSuccess) ? 0 : 6;
}

} // namespace ppml

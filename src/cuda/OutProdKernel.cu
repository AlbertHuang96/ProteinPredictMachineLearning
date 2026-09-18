#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace ppml {

// ============================================================
// Helper
// ============================================================
static inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// ============================================================
// out_prod CUDA Kernel —— **线程重映射 + 按平面局部化 block**（2026-09-18）
//
// 数学: dst[i0,i1,i2,i3] = sum_{k} src0[i0,k,i2,i3] * src1[i1,k,i2,i3]
//
// ── 为什么改（ncu 定量根因，见 .codebuddy/memory/2026-09-14.md）────────────────
//   旧映射把**批维**放进了 threadIdx：`i3 = blockIdx.x*blockDim.x + threadIdx.x`、
//   `i2 = blockIdx.y*blockDim.y + threadIdx.y`，而 i3/i2 在 src0/src1 里的步长是
//   `ne00*ne01*ne02` / `ne10*ne11*ne12`（**最外层**）✗
//   ⇒ 同一 warp 的 32 个线程落在 32 条互不相邻的 cache line 上 ⇒ 非合并访问。
//   实测（真实 shape ne0=ne1=51/K=51/planes=1024）：L1 命中 0.84%、Compute 0.74%、
//   流量 2.19 GB 而有用数据仅 ~32 MB（**~68× 浪费**）、ncu OPT 判"87% 冗余 sectors"。
//
// ── 新映射（谁放进 threadIdx，取决于**谁是 stride=1**）────────────────────────
//   threadIdx.x  → **i0**   ★ i0 在 src0 与 dst 中 **stride=1**（最内维）
//                            ⇒ 同 warp 连续线程访问**连续地址**（每次迭代 32×4B = 128B
//                              = 4 个 sector 全用满）✓
//   threadIdx.y  → **i1**   ★ i1 在 src1 中 **stride=1**；且同一 warp 内 i1 相同
//                            ⇒ src1 的读是**广播**（全 warp 同一地址 ⇒ 1 次事务）✓
//   blockIdx.z   → 平面 (i2,i3)  ★ 按平面局部化：一个 block 只碰一个平面的数据，
//                                  且带 grid-stride ⇒ 平面数超 gridDim.z(65535) 也不丢 ✓
//   blockIdx.x/y → i0/i1 的 tile 序号（同样带 grid-stride ⇒ 任何 shape 都不丢数据 ✓）
//   K 维         在每线程内**串行**：步长只决定"每次迭代跳多远"，而**每次迭代内**
//                warp 访问的仍是连续的 32 个 i0 ⇒ 依旧满利用 ✓
//
// ── 数值一致性 ──────────────────────────────────────────────────────────────
//   每个输出元素的累加顺序与旧实现**完全一致**（k = 0..ne01-1 依次累加）⇒ 结果逐位相同 ✓
//   （所以这次优化不会改变 loss；改完可直接用 loss 做回归判据 ✓）
//
// ── 历史（新实现同样保留）────────────────────────────────────────────────────
//   CUDA 硬限制 gridDim.y/z ≤ 65535（旧版曾因 ne0*ne1 折叠进 z 而"invalid configuration
//   argument"）⇒ 新实现的三个方向**都**做 grid-stride + host 侧 clamp，不丢数据 ✓
// ============================================================
#if 0  /* ===== 旧实现（原映射，2026-09-18 起停用；原样保留作对照，不参与编译）===== */

// ---------- 以下为被替换的旧注释（保留以便对照） ----------
// Grid 映射方案 (高维张量 → CUDA grid):
//   - dim_{n-1}, dim_n → 2D grid (blockIdx.y, blockIdx.x)
//   - dim1 ... dim_{n-2} → 折叠入 blockIdx.z 线性编码
//
//   对于 4D 情况 (n=4):
//     blockIdx.z = i0 * ne1 + i1          ← dim1=i0, dim2=i1
//     blockIdx.y = i2                      ← dim_{n-1}=i2
//     blockIdx.x = i3                      ← dim_n=i3
//
//   每 thread 负责 dst 一个元素，沿 K 维串行累加
//
// 【2026-09-11 越界修复】原实现把 (i0,i1) 全部塞进 blockIdx.z、把 i2 塞进 blockIdx.y，
//   但 CUDA 硬限制 gridDim.z ≤ 65535、gridDim.y ≤ 65535：
//     - ne0*ne1 > 65535（如 L*L / T*D 等乘积）→ 启动即报
//       "invalid configuration argument"（与 op=32 OP_OUT_PROD 的实测报错一致）；
//     - ne2 > 65535*BLOCK_Y → 同样非法。
//   现改为 z / y 两个方向都做 grid-stride 循环（grid 维度由 host 侧 clamp 到合法值），
//   并补上原缺失的 z_idx ≥ ne0*ne1 上界检查。
// ============================================================

__global__ void kernel_out_prod(
    const float* __restrict__ src0,
    const float* __restrict__ src1,
    float* __restrict__ dst,
    // ---- src0 shape (ne00=inner dim, ne01=contraction dim) ----
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    // ---- src1 shape (ne10=inner dim, ne11=contraction dim, == ne01) ----
    const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
    // ---- dst shape (ne0==ne00, ne1==ne10, ne2, ne3) ----
    const int64_t ne0,  const int64_t ne1,  const int64_t ne2,  const int64_t ne3)
{
    // ================================================================
    // Step 1: 最后两维 — dim_n = i3 由 blockIdx.x/threadIdx.x 提供
    //   dim_{n-1} = i2 由 blockIdx.y/threadIdx.y 提供 + grid-stride 循环（见下）
    // ================================================================
    const int64_t i3 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i3 >= ne3) return;

    const int64_t n_z      = ne0 * ne1;                                   // (i0,i1) 组合数
    const int64_t y_stride = static_cast<int64_t>(gridDim.y) * blockDim.y;
    const int64_t i2_0     = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    for (int64_t z_idx = blockIdx.z; z_idx < n_z; z_idx += gridDim.z) {
    // ================================================================
    // Step 2: 前 n-2 维 — 从 z_idx 反解 (逐层剥洋葱)
    //
    //   对于 4D: n-2 = 2, dim1 = i0, dim2 = i1
    //   z_idx 编码:   z_idx = i0 * ne1 + i1
    //
    //   内存寻址公式:
    //     memoryId = dim1 * (d2*d3*...*dn)
    //              + dim2 * (d3*d4*...*dn)
    //              + ...
    //              + dim_{n-2} * (d_{n-1} * dn)
    //              + dim_{n-1} * dn
    //              + dim_n
    //
    //   这里: d1=ne0, d2=ne1, d3=ne2, d4=ne3
    //         i0*(ne1*ne2*ne3) + i1*(ne2*ne3) + i2*ne3 + i3
    // ================================================================
    const int64_t i0    = z_idx / ne1;        // dim1: z_idx 除以 d2(=ne1) 的商
    const int64_t i1    = z_idx % ne1;        // dim2: 余数

    for (int64_t i2 = i2_0; i2 < ne2; i2 += y_stride) {
    // ================================================================
    // Step 3: GQA 映射 (Group Query Attention) — 暂不启用
    //   将 dst 的 (i2, i3) 映射回 src0 的对应维度
    //   当 src0 的 head 数 < src1 时，共享 K/V heads
    // ================================================================
    // const int64_t i02 = i2 / dps2;
    // const int64_t i03 = i3 / dps3;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    // ================================================================
    // Step 4: 预计算基地址 (减少循环内乘法)
    //
    //   src0 布局: [ne00][ne01][ne02][ne03] row-major, ne00 最内维
    //     offset = i0 + i01*ne00 + i02*ne00*ne01 + i03*ne00*ne01*ne02
    //
    //   src1 布局: [ne10][ne11][ne12][ne13] row-major, ne10 最内维
    //     offset = i1 + i01*ne10 + i2*ne10*ne11  + i3*ne10*ne11*ne12
    // ================================================================
    const int64_t s0_outer = i0 + i02 * ne00 * ne01 + i03 * ne00 * ne01 * ne02;
    const int64_t s1_outer = i1 + i2  * ne10 * ne11 + i3  * ne10 * ne11 * ne12;

    // ================================================================
    // Step 5: 沿 contraction dim (ne01) 串行累加
    // ================================================================
    float sum = 0.0f;
    for (int64_t i01 = 0; i01 < ne01; ++i01) {
        const float val0 = src0[s0_outer + i01 * ne00];   // s0[i0, i01]
        const float val1 = src1[s1_outer + i01 * ne10];   // s1[i1, i01]
        sum += val0 * val1;
    }

    // ================================================================
    // Step 6: 写回 dst
    //   dst 布局: [ne0][ne1][ne2][ne3] row-major, ne0 最内维
    //     offset = i0 + i1*ne0 + i2*ne0*ne1 + i3*ne0*ne1*ne2
    // ================================================================
    dst[i0 + i1 * ne0 + i2 * ne0 * ne1 + i3 * ne0 * ne1 * ne2] = sum;
    } // for i2
    } // for z_idx
}
#endif  /* ===== 旧实现结束（其后为 2026-09-18 重映射版）===== */

// ============================================================
// 重映射版 kernel：threadIdx.x→i0、threadIdx.y→i1、blockIdx.z→平面(i2,i3)
//   · i0 在 src0/dst 中 stride=1 ⇒ 同 warp 连续线程 = 连续地址 ✓
//   · i1 在 src1 中 stride=1，且 warp 内 i1 相同 ⇒ src1 广播 ✓
//   · 平面局部化到 blockIdx.z ⇒ 一个 block 只在一个平面内工作 ✓
//   · 三个方向都带 grid-stride ⇒ 任何 shape 都不丢数据、不受 gridDim.y/z ≤ 65535 限制 ✓
// ============================================================
__global__ void kernel_out_prod_remapped(
    const float* __restrict__ src0,
    const float* __restrict__ src1,
    float* __restrict__ dst,
    // ---- src0 shape (ne00=inner dim, ne01=contraction dim) ----
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    // ---- src1 shape (ne10=inner dim, ne11=contraction dim, == ne01) ----
    const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
    // ---- dst shape (ne0==ne00, ne1==ne10, ne2, ne3) ----
    const int64_t ne0,  const int64_t ne1,  const int64_t ne2,  const int64_t ne3)
{
    const int64_t n_planes  = ne2 * ne3;
    const int64_t x_stride  = static_cast<int64_t>(gridDim.x) * blockDim.x;   // i0 方向
    const int64_t y_stride  = static_cast<int64_t>(gridDim.y) * blockDim.y;   // i1 方向
    const int64_t i1_base   = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int64_t i0_base   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // ---- 平面循环（blockIdx.z + grid-stride）：按平面局部化 ✓ ----
    for (int64_t plane = blockIdx.z; plane < n_planes; plane += gridDim.z) {
        const int64_t i2 = plane % ne2;
        const int64_t i3 = plane / ne2;

        // 平面基址（把 i0/i1 之外的部分预先算好，循环内只做加法）
        const int64_t s0_plane = i2 * ne00 * ne01 + i3 * ne00 * ne01 * ne02;
        const int64_t s1_plane = i2 * ne10 * ne11 + i3 * ne10 * ne11 * ne12;
        const int64_t d_plane  = i2 * ne0 * ne1 + i3 * ne0 * ne1 * ne2;

        for (int64_t i1 = i1_base; i1 < ne1; i1 += y_stride) {
            // ★ warp 内所有线程 i1 相同（threadIdx.y 不随 lane 变）⇒ 下面每次读都是**广播**
            const float* __restrict__ b_col = src1 + s1_plane + i1;
            for (int64_t i0 = i0_base; i0 < ne0; i0 += x_stride) {
                // ★ warp 内 lane 0..31 ↔ i0, i0+1, ... ⇒ src0 侧**连续地址**
                const float* __restrict__ a_row = src0 + s0_plane + i0;
                float sum = 0.0f;
                for (int64_t k = 0; k < ne01; ++k) {          // 累加顺序与旧实现一致 ⇒ 逐位相同 ✓
                    sum += a_row[k * ne00] * b_col[k * ne10];
                }
                dst[d_plane + i1 * ne0 + i0] = sum;           // warp 内连续 i0 ⇒ 写回也合并 ✓
            }
        }
    }
}

// ============================================================
// Host wrapper — 计算 grid/block 维度并启动 kernel
// ============================================================
void out_prod_cuda(
    const float* src0, const float* src1, float* dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    int64_t ne0,  int64_t ne1,  int64_t ne2,  int64_t ne3)
{
    // 【2026-09-18 重映射】block 形状按"谁是 stride=1"选：
    //   BLOCK_X → i0（src0/dst 的 stride=1 维；取 32 = warp 宽 ⇒ 一个 warp 恰好覆盖一行连续 i0 ✓）
    //   BLOCK_Y → i1（src1 的 stride=1 维；warp 内 i1 相同 ⇒ src1 广播 ✓）
    constexpr int BLOCK_X = 32;   // i0（合并维）
    constexpr int BLOCK_Y = 8;    // i1（广播维）
    constexpr int64_t MAX_GRID_XY_Z = 65535;   // CUDA: gridDim.y / gridDim.z 上限（x 上限 2^31-1）

    if (ne0 <= 0 || ne1 <= 0 || ne2 <= 0 || ne3 <= 0) {
        fprintf(stderr,
                "[OUTPROD-ERR] 非法 dims: dst=[%lld,%lld,%lld,%lld] src0=[%lld,%lld,%lld,%lld] "
                "src1=[%lld,%lld,%lld,%lld] (跳过启动)\n",
                (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3,
                (long long)ne00, (long long)ne01, (long long)ne02, (long long)ne03,
                (long long)ne10, (long long)ne11, (long long)ne12, (long long)ne13);
        return;
    }

    // 【2026-09-18 重映射】grid：x/y = i0/i1 的 tile 数，z = **平面数**(ne2*ne3)
    //   （旧版把 (i0,i1) 折叠进 blockIdx.z、把批维放 threadIdx ⇒ 非合并访问 ✗）
    const int64_t n_planes = ne2 * ne3;
    int64_t gx = (ne0 + BLOCK_X - 1) / BLOCK_X;
    int64_t gy = (ne1 + BLOCK_Y - 1) / BLOCK_Y;
    int64_t gz = n_planes;

    // clamp 到 CUDA 硬限制（kernel 内三个方向都是 grid-stride ⇒ clamp 不丢数据 ✓）
    if (gy > MAX_GRID_XY_Z || gz > MAX_GRID_XY_Z) {
        fprintf(stderr,
                "[OUTPROD-CLAMP] grid 越限已 clamp（kernel 内 grid-stride 覆盖全部）: "
                "ne1=%lld (gy=%lld) ne2*ne3=%lld (gz=%lld) dst=[%lld,%lld,%lld,%lld]\n",
                (long long)ne1, (long long)gy, (long long)n_planes, (long long)gz,
                (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3);
    }
    if (gz > MAX_GRID_XY_Z) gz = MAX_GRID_XY_Z;
    if (gy > MAX_GRID_XY_Z) gy = MAX_GRID_XY_Z;

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid (static_cast<unsigned>(gx), static_cast<unsigned>(gy), static_cast<unsigned>(gz));

    kernel_out_prod_remapped<<<grid, block>>>(
        src0, src1, dst,
        ne00, ne01, ne02, ne03,
        ne10, ne11, ne12, ne13,
        ne0,  ne1,  ne2,  ne3);
}

} // namespace ppml

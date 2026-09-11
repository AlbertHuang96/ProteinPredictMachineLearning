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
// out_prod CUDA Kernel
//
// 数学: dst[i0,i1,i2,i3] = sum_{k} src0[i0,k,i2,i3] * src1[i1,k,i2,i3]
//
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

// ============================================================
// Host wrapper — 计算 grid/block 维度并启动 kernel
// ============================================================
void out_prod_cuda(
    const float* src0, const float* src1, float* dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    int64_t ne0,  int64_t ne1,  int64_t ne2,  int64_t ne3)
{
    // BLOCK_X → dim_n (ne3),  BLOCK_Y → dim_{n-1} (ne2)
    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 16;
    constexpr int64_t MAX_GRID_XY_Z = 65535;   // CUDA: gridDim.y / gridDim.z 上限

    if (ne0 <= 0 || ne1 <= 0 || ne2 <= 0 || ne3 <= 0) {
        fprintf(stderr,
                "[OUTPROD-ERR] 非法 dims: dst=[%lld,%lld,%lld,%lld] src0=[%lld,%lld,%lld,%lld] "
                "src1=[%lld,%lld,%lld,%lld] (跳过启动)\n",
                (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3,
                (long long)ne00, (long long)ne01, (long long)ne02, (long long)ne03,
                (long long)ne10, (long long)ne11, (long long)ne12, (long long)ne13);
        return;
    }

    const int64_t n01 = ne0 * ne1;                        // (i0,i1) 组合数 → blockIdx.z
    int64_t gx = ceil_div(static_cast<int64_t>(ne3), BLOCK_X);
    int64_t gy = ceil_div(static_cast<int64_t>(ne2), BLOCK_Y);
    int64_t gz = n01;

    // 2026-09-11：clamp 到 CUDA 硬限制（kernel 内已改成 grid-stride，clamp 不丢数据）
    if (gz > MAX_GRID_XY_Z || gy > MAX_GRID_XY_Z) {
        fprintf(stderr,
                "[OUTPROD-CLAMP] grid 越限已 clamp: ne0*ne1=%lld (z=%lld) ne2=%lld (y=%lld) "
                "dst=[%lld,%lld,%lld,%lld]\n",
                (long long)n01, (long long)gz, (long long)ne2, (long long)gy,
                (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3);
    }
    if (gz > MAX_GRID_XY_Z) gz = MAX_GRID_XY_Z;
    if (gy > MAX_GRID_XY_Z) gy = MAX_GRID_XY_Z;

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid (static_cast<unsigned>(gx), static_cast<unsigned>(gy), static_cast<unsigned>(gz));

    kernel_out_prod<<<grid, block>>>(
        src0, src1, dst,
        ne00, ne01, ne02, ne03,
        ne10, ne11, ne12, ne13,
        ne0,  ne1,  ne2,  ne3);
}

} // namespace ppml

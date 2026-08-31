#include <cuda_runtime.h>
#include <cstdint>

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
    // Step 1: 最后两维 — 直接由 2D grid (x, y) 提供
    //   dim_n     = i3 = ne3 → blockIdx.x * blockDim.x + threadIdx.x
    //   dim_{n-1} = i2 = ne2 → blockIdx.y * blockDim.y + threadIdx.y
    // ================================================================
    const int64_t i3 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t i2 = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (i3 >= ne3 || i2 >= ne2) return;

    // ================================================================
    // Step 2: 前 n-2 维 — 从 blockIdx.z 反解 (逐层剥洋葱)
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
    const int64_t z_idx = blockIdx.z;         // 线性编码值
    const int64_t i0    = z_idx / ne1;        // dim1: z_idx 除以 d2(=ne1) 的商
    const int64_t i1    = z_idx % ne1;        // dim2: 余数

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
    //   dst[i0,i1,i2,i3] += src0[i0, k, i02, i03] * src1[i1, k, i2, i3]
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

    const int grid_x = ceil_div(static_cast<int>(ne3), BLOCK_X);  // dim_n
    const int grid_y = ceil_div(static_cast<int>(ne2), BLOCK_Y);  // dim_{n-1}
    const int grid_z = static_cast<int>(ne0 * ne1);               // 前 n-2 维折叠

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid (grid_x,  grid_y,  grid_z);

    kernel_out_prod<<<grid, block>>>(
        src0, src1, dst,
        ne00, ne01, ne02, ne03,
        ne10, ne11, ne12, ne13,
        ne0,  ne1,  ne2,  ne3);
}

} // namespace ppml

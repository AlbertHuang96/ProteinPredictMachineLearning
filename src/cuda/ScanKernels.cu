// ============================================================
// ScanKernels.cu — int32 inclusive prefix sum（ScanThenFan 两遍 fan-out）
// ============================================================
// 实现来源：用户提供的 PTX 参考（ScanWarp / ScanBlock / ScanAndWritePartSumKernel /
//           AddBaseSumKernel / ScanThenFan），host 封装在此文件宿主区。
//
// 语义：inclusive scan，output[i] = input[0] + ... + input[i]（int32 精确，无浮点误差）。
//
// 结构（两遍 fan）：
//   ScanAndWritePartSumKernel: 每 block 1024 元素做 block scan，输出局部 scan +
//                              各 block 的和（block 间无依赖）
//   递归 ScanThenFan:          扫描 block 和数组（part_num 收缩，part_size=1024 递归）
//   AddBaseSumKernel:          每 block 加排他前缀（part[part_i-1]）
// 共享内存：动态 32*int32（warp 部分和），ScanWarp 用 shfl.sync.up.b32 + 谓词
//   捕获"源 lane 无效"（lane < delta 时 p=false，不累加）——zero-padding 语义。
// ============================================================
#include <cuda_runtime.h>
#include <cstdint>
// 注意：不要 include <algorithm>/<cstdio> 等宿主 STL 头——nvcc 在 device 编译路径
// 会对 std::min 等模板报 "parameter packs not expanded with '...'"（CUDAKernels.cu 同规避）。

namespace ppml_scan {

static inline size_t scan_min(size_t a, size_t b) { return a < b ? a : b; }

// ============================================================
// warp inclusive scan — PTX shfl.sync.up.b32 + 谓词保护
//   p 在 srcLane 无效（lane < delta）时为 false → @p add 跳过 → padding 0 语义
// ============================================================
__device__ __forceinline__ int32_t ScanWarp(int32_t val) {
    int32_t result;
    asm("{"
        ".reg .s32 r<5>;"
        ".reg .pred p<5>;"
        "shfl.sync.up.b32 r0|p0, %1, 1, 0, -1;"
        "@p0 add.s32 r0, r0, %1;"
        "shfl.sync.up.b32 r1|p1, r0, 2, 0, -1;"
        "@p1 add.s32 r1, r1, r0;"
        "shfl.sync.up.b32 r2|p2, r1, 4, 0, -1;"
        "@p2 add.s32 r2, r2, r1;"
        "shfl.sync.up.b32 r3|p3, r2, 8, 0, -1;"
        "@p3 add.s32 r3, r3, r2;"
        "shfl.sync.up.b32 r4|p4, r3, 16, 0, -1;"
        "@p4 add.s32 r4, r4, r3;"
        "mov.s32 %0, r4;"
        "}"
        : "=r"(result)
        : "r"(val));
    return result;
}

// ============================================================
// block inclusive scan（每线程 1 元素）— 三阶段
//   warp_sum 为动态共享内存（launch 时传 32*sizeof(int32)）
// ============================================================
__device__ __forceinline__ int32_t ScanBlock(int32_t val) {
    int32_t warp_id = threadIdx.x >> 5;
    int32_t lane = threadIdx.x & 31;
    extern __shared__ int32_t warp_sum[];
    // scan each warp
    val = ScanWarp(val);
    __syncthreads();
    // write sum of each warp to warp_sum
    if (lane == 31) {
        warp_sum[warp_id] = val;
    }
    __syncthreads();
    // use a single warp to scan warp_sum
    if (warp_id == 0) {
        warp_sum[lane] = ScanWarp(warp_sum[lane]);
    }
    __syncthreads();
    // add base
    if (warp_id > 0) {
        val += warp_sum[warp_id - 1];
    }
    __syncthreads();
    return val;
}

// ============================================================
// 第 1 遍：每 block 局部 scan + 写 block 和到 part[]
// ============================================================
__global__ void ScanAndWritePartSumKernel(const int32_t* input, int32_t* part,
                                          int32_t* output, size_t n,
                                          size_t part_num) {
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        size_t index = part_i * blockDim.x + threadIdx.x;
        int32_t val = index < n ? input[index] : 0;
        val = ScanBlock(val);
        __syncthreads();
        if (index < n) {
            output[index] = val;
        }
        if (threadIdx.x == blockDim.x - 1) {
            part[part_i] = val;
        }
    }
}

// ============================================================
// 第 3 遍：每 block 加排他前缀（part[part_i-1]，part 已为 inclusive）
// ============================================================
__global__ void AddBaseSumKernel(int32_t* part, int32_t* output, size_t n,
                                 size_t part_num) {
    for (size_t part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
        if (part_i == 0) {
            continue;
        }
        size_t index = part_i * blockDim.x + threadIdx.x;
        if (index < n) {
            output[index] += part[part_i - 1];
        }
    }
}

// ============================================================
// host：两遍 fan scan（递归扫描 part 数组）
//   input/output/buffer 均为 device 指针；buffer 需容纳递归各层 part 数组
//   （总大小 < 2*part_num，见宿主封装）。
// ============================================================
void ScanThenFan(const int32_t* input, int32_t* buffer, int32_t* output,
                 size_t n) {
    size_t part_size = 1024;  // tuned
    size_t part_num = (n + part_size - 1) / part_size;
    size_t block_num = scan_min(part_num, 128);
    // use buffer[0:part_num] to save the metric of part
    int32_t* part = buffer;
    // after following step, part[i] = part_sum[i]
    size_t shm_size = 32 * sizeof(int32_t);
    ScanAndWritePartSumKernel<<<block_num, part_size, shm_size>>>(
        input, part, output, n, part_num);
    if (part_num >= 2) {
        // after following step
        // part[i] = part_sum[0] + part_sum[1] + ... + part_sum[i]
        ScanThenFan(part, buffer + part_num, part, part_num);
        // make final result
        AddBaseSumKernel<<<block_num, part_size>>>(part, output, n, part_num);
    }
}

// ============================================================
// 宿主接口：分配 device 内存 + 拷贝 + 计时，返回 0 成功
//   ms_out 非空时返回 10 轮平均耗时（ms，不含 H2D/D2H）
// ============================================================
int scan_int32_inclusive_cuda(const int32_t* in, int32_t* out, size_t n,
                              float* ms_out) {
    if (n == 0) return 0;
    const size_t part_size = 1024;
    const size_t part_num = (n + part_size - 1) / part_size;
    // 递归各层 part 数组总大小 < 2*part_num（等比收缩），+64 兜底
    const size_t buf_cap = 2 * part_num + 64;

    int32_t *d_in = nullptr, *d_out = nullptr, *d_buf = nullptr;
    if (cudaMalloc(&d_in,  n * sizeof(int32_t))     != cudaSuccess) return 1;
    if (cudaMalloc(&d_out, n * sizeof(int32_t))     != cudaSuccess) return 1;
    if (cudaMalloc(&d_buf, buf_cap * sizeof(int32_t)) != cudaSuccess) return 1;
    cudaMemcpy(d_in, in, n * sizeof(int32_t), cudaMemcpyHostToDevice);

    auto launch = [&]() {
        ScanThenFan(d_in, d_buf, d_out, n);
    };

    launch();   // warmup + 正确性
    cudaError_t err = cudaDeviceSynchronize();

    if (ms_out) {
        cudaEvent_t e_s, e_e;
        cudaEventCreate(&e_s);
        cudaEventCreate(&e_e);
        cudaEventRecord(e_s);
        constexpr int R = 10;
        for (int r = 0; r < R; r++) launch();
        cudaEventRecord(e_e);
        cudaEventSynchronize(e_e);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e_s, e_e);
        *ms_out = ms / R;
        cudaEventDestroy(e_s);
        cudaEventDestroy(e_e);
    }

    cudaMemcpy(out, d_out, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_buf);
    return (err == cudaSuccess) ? 0 : 1;
}

} // namespace ppml_scan

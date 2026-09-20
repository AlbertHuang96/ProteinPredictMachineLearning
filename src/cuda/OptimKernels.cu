// ============================================================
// OptimKernels.cu —— 优化器/梯度裁剪的 GPU kernel（2026-09-19 新增）
//
// 动机（实测，见 .codebuddy/memory/2026-09-19.md）：
//   nsys 显示 dev 配置 1 epoch 有 **11,767 次 cudaMemcpy / 2.36 s（64.4% wall）**，其中
//   **D2H 4,235 次占 memcpy 时间的 84.7%**，中位 41 KB（= 参数级）⇒ 来源就是
//   `AdamW::step` 与 `GradientClipper` 的**逐参数 D2H/H2D 往返**：
//     AdamW.cpp:102-132  每个参数：读 w(D2H) + 读 g(D2H) + 写 w(H2D)      × 540 参数/step
//     GradientClipper.cpp:105/157-164/203-208/314-334 每个参数每个 loss 再来一遍 × 5 loss
//   而 pageable 内存的 cudaMemcpy 还会**隐式做一次全设备同步** ✗ ⇒ 代价 = 同步 + 传输。
//
// 本文件把这几件事变成"**数据留在显存里的就地 kernel**"：
//   · adamw_step_cuda        —— w/g/m/v 全在 device，一次 launch 完成整步更新 ✓
//   · grad_stats_cuda        —— Σg²（double）、max|g|、NaN 计数（供 clip 判据/诊断）✓
//   · grad_scale_cuda        —— 就地缩放 + NaN/Inf→0（与 CPU 判据逐字一致 ✓）
//   · grad_accumulate_cuda   —— 累加器 += 梯度（per-loss 累加，留在 device ✓）
//   · optim_fill_cuda        —— 就地填常数（1.0/0.0，配合 loss 激活/清零 ✓）
//   · optim_copy_d2d/h2d/d2h —— 显存↔内存搬运的薄封装（把 CUDA 调用收在本文件 ✓）
//
// ⚠️ 数值一致性：AdamW 的每个元素运算**顺序与 CPU 版逐字相同**
//   （m→v→m_hat/v_hat→update→w-=update→(可选)w-=lr*wd*w，用更新后的 w）
//   ⇒ 同一输入下逐位一致 ✓（可拿 loss 做回归判据 ✓）
//   ⚠️ 唯一例外：**范数**（grad_stats_cuda）用 device double 规约，加法顺序与 CPU 不同
//   ⇒ clip 的 scale 因子可能有末位差异 ⇒ loss 允许 1e-6 量级差异（实测：无差异，见回归 ✓）
// ============================================================
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace ppml {

// ============================================================
// 1) AdamW 一步更新（就地）
//    与 src/core/AdamW.cpp:111-129 的逐元素逻辑逐字对应：
//      m = b1*m + (1-b1)*g
//      v = b2*v + (1-b2)*g*g
//      w -= lr*ls * (m/bc1) / (sqrt(v/bc2) + eps)
//      if (wd != 0 && !no_wd) w -= lr*ls * wd * w      ← 注意用的是**已更新**的 w ✓
// ============================================================
__global__ void adamw_step_kernel(float* __restrict__ w,
                                 const float* __restrict__ g,
                                 float* __restrict__ m,
                                 float* __restrict__ v,
                                 const int64_t n,
                                 const float lr_ls,        // = lr * lr_scale（保持 CPU 的结合顺序 ✓）
                                 const float beta1,
                                 const float beta2,
                                 const float eps,
                                 const float wd,
                                 const float bc1,
                                 const float bc2,
                                 const int   no_weight_decay)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float gi = g[i];
    const float mi = beta1 * m[i] + (1.0f - beta1) * gi;
    const float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;

    const float m_hat = mi / bc1;
    const float v_hat = vi / bc2;
    const float update = lr_ls * m_hat / (sqrtf(v_hat) + eps);

    float wi = w[i] - update;
    if (wd != 0.0f && !no_weight_decay) {
        wi -= lr_ls * wd * wi;      // 与 CPU 同：用更新后的 w ✓
    }
    w[i] = wi;
}

// 返回 0 = 已执行
int adamw_step_cuda(float* w, const float* g, float* m, float* v, int64_t n,
                    float lr, float lr_scale, float beta1, float beta2, float eps,
                    float wd, float bc1, float bc2, int no_weight_decay)
{
    if (!w || !g || !m || !v || n <= 0) return 1;
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    if (blocks > 2147483647LL) return 1;                 // gridDim.x 上限
    adamw_step_kernel<<<(int)blocks, threads>>>(w, g, m, v, n,
                                                lr * lr_scale, beta1, beta2, eps,
                                                wd, bc1, bc2, no_weight_decay);
    return 0;
}

// ============================================================
// 2) 梯度统计：Σg²（double）、max|g|、NaN 计数
//    语义与 src/core/GradientClipper.cpp:96-149 相同：
//      · NaN **不计入** sum/max，只计数 ✓
//      · 累加用 double ✓
//    做法：每 block 出一份部分和 → 拷回 host（几 KB）→ host 端 double 汇总 ✓
// ============================================================
__global__ void grad_stats_kernel(const float* __restrict__ g,
                                  const int64_t n,
                                  double* __restrict__ out)   // [3*blocks]: sum, max, nnan
{
    const int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    double lsum = 0.0, lmax = 0.0;
    long long lnan = 0;
    for (int64_t i = tid; i < n; i += stride) {
        const float v = g[i];
        if (v != v) { ++lnan; continue; }        // NaN 只计数 ✓
        const double d = (double)v;
        lsum += d * d;
        const double a = (d < 0.0) ? -d : d;
        if (a > lmax) lmax = a;
    }

    __shared__ double sh_sum[256];
    __shared__ double sh_max[256];
    __shared__ long long sh_nan[256];
    const int t = (int)threadIdx.x;
    sh_sum[t] = lsum; sh_max[t] = lmax; sh_nan[t] = lnan;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) {
            sh_sum[t] += sh_sum[t + s];
            if (sh_max[t + s] > sh_max[t]) sh_max[t] = sh_max[t + s];
            sh_nan[t] += sh_nan[t + s];
        }
        __syncthreads();
    }
    if (t == 0) {
        out[blockIdx.x]                    = sh_sum[0];
        out[gridDim.x + blockIdx.x]        = sh_max[0];
        out[2 * gridDim.x + blockIdx.x]    = (double)sh_nan[0];
    }
}

// 持久化的小缓冲（避免每次调用都 cudaMalloc ✓）
static double* s_stats_buf = nullptr;
static int     s_stats_cap = 0;

int grad_stats_cuda(const float* g, int64_t n,
                    double* out_sum_sq, double* out_maxabs, long long* out_nnan)
{
    if (out_sum_sq) *out_sum_sq = 0.0;
    if (out_maxabs) *out_maxabs = 0.0;
    if (out_nnan)   *out_nnan   = 0;
    if (!g || n <= 0) return 0;

    int blocks = (int)((n + 255) / 256);
    if (blocks > 512) blocks = 512;               // 上限 512 block（够用且 host 汇总便宜 ✓）
    if (blocks < 1) blocks = 1;
    if (blocks > s_stats_cap) {
        if (s_stats_buf) { cudaFree(s_stats_buf); s_stats_buf = nullptr; s_stats_cap = 0; }
        if (cudaMalloc((void**)&s_stats_buf, sizeof(double) * 3 * blocks) != cudaSuccess) {
            s_stats_buf = nullptr;
            return 1;
        }
        s_stats_cap = blocks;
    }

    grad_stats_kernel<<<blocks, 256>>>(g, n, s_stats_buf);
    if (cudaGetLastError() != cudaSuccess) return 1;

    double host[3 * 512];
    if (blocks > 512) return 1;
    if (cudaMemcpy(host, s_stats_buf, sizeof(double) * 3 * blocks,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 1;
    }
    double ssum = 0.0, smax = 0.0;
    long long snan = 0;
    for (int b = 0; b < blocks; ++b) {
        ssum += host[b];
        if (host[blocks + b] > smax) smax = host[blocks + b];
        snan += (long long)host[2 * blocks + b];
    }
    if (out_sum_sq) *out_sum_sq = ssum;
    if (out_maxabs) *out_maxabs = smax;
    if (out_nnan)   *out_nnan   = snan;
    return 0;
}

// ============================================================
// 3) 就地缩放 + NaN/Inf→0（对应 GradientClipper.cpp:153-166 的 scale_param_grads ✓）
//    判据逐字一致：v != v || v > 3.0e38f || v < -3.0e38f ⇒ 置 0，否则 v *= scale ✓
// ============================================================
__global__ void grad_scale_kernel(float* __restrict__ g, const int64_t n, const float scale)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = g[i];
    if (v != v || v > 3.0e38f || v < -3.0e38f) { g[i] = 0.0f; return; }
    g[i] = v * scale;
}

int grad_scale_cuda(float* g, int64_t n, float scale)
{
    if (!g || n <= 0) return 1;
    const int64_t blocks = (n + 255) / 256;
    if (blocks > 2147483647LL) return 1;
    grad_scale_kernel<<<(int)blocks, 256>>>(g, n, scale);
    return 0;
}

// ============================================================
// 4) 累加：acc[i] += g[i]（对应 accumulate_per_loss_gradients 的 per-loss 累加 ✓）
// ============================================================
__global__ void grad_accumulate_kernel(float* __restrict__ acc,
                                       const float* __restrict__ g, const int64_t n)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] += g[i];
}

int grad_accumulate_cuda(float* acc, const float* g, int64_t n)
{
    if (!acc || !g || n <= 0) return 1;
    const int64_t blocks = (n + 255) / 256;
    if (blocks > 2147483647LL) return 1;
    grad_accumulate_kernel<<<(int)blocks, 256>>>(acc, g, n);
    return 0;
}

// ============================================================
// 5) 就地填常数（loss 激活 1.0 / 清零 0.0 ✓）
// ============================================================
__global__ void optim_fill_kernel(float* __restrict__ dst, const int64_t n, const float val)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = val;
}

int optim_fill_cuda(float* dst, int64_t n, float val)
{
    if (!dst || n <= 0) return 1;
    if (val == 0.0f) {                       // 0 用 memset 更快 ✓
        return (cudaMemset(dst, 0, (size_t)n * sizeof(float)) == cudaSuccess) ? 0 : 1;
    }
    const int64_t blocks = (n + 255) / 256;
    if (blocks > 2147483647LL) return 1;
    optim_fill_kernel<<<(int)blocks, 256>>>(dst, n, val);
    return 0;
}

// ============================================================
// 6) 显存/搬运薄封装（让 .cpp 侧不必 include cuda_runtime ✓）
// ============================================================
void* optim_alloc_cuda(int64_t bytes)
{
    if (bytes <= 0) return nullptr;
    void* p = nullptr;
    if (cudaMalloc(&p, (size_t)bytes) != cudaSuccess) return nullptr;
    return p;
}

void optim_free_cuda(void* p)
{
    if (p) cudaFree(p);
}

int optim_h2d_cuda(void* dst, const void* src, int64_t bytes)
{
    if (!dst || !src || bytes <= 0) return 1;
    return (cudaMemcpy(dst, src, (size_t)bytes, cudaMemcpyHostToDevice) == cudaSuccess) ? 0 : 1;
}

int optim_d2h_cuda(void* dst, const void* src, int64_t bytes)
{
    if (!dst || !src || bytes <= 0) return 1;
    return (cudaMemcpy(dst, src, (size_t)bytes, cudaMemcpyDeviceToHost) == cudaSuccess) ? 0 : 1;
}

int optim_d2d_cuda(void* dst, const void* src, int64_t bytes)
{
    if (!dst || !src || bytes <= 0) return 1;
    return (cudaMemcpy(dst, src, (size_t)bytes, cudaMemcpyDeviceToDevice) == cudaSuccess) ? 0 : 1;
}

// 当前 CUDA 是否可用（AdamW/GradientClipper 用它在无 GPU 时保持原 CPU 路径 ✓）
int optim_cuda_available()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n > 0 ? 1 : 0;
}

} // namespace ppml

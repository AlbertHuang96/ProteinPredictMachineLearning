#include "ppml/GradientClipper.h"
#include "ppml/Tensor.h"

#include <cmath>
#include <cstring>
#include <cassert>

namespace ppml {

// ============================================================
// 【2026-09-19】裁剪的 GPU 路径（把"读整张量回 host 算范数/缩放"换成显存内就地运算）
//   背景（实测）：dev 1 epoch `cudaMemcpy` 11,767 次 / 2.36 s，**D2H 占 84.7%**，
//   中位 41 KB（= 参数级）；本文件原来每个参数 × 每个 loss 都做 D2H(+H2D) ✗
//   （5 个 loss × 540 参数 ⇒ 上千次往返，且 pageable 拷贝自带**隐式全设备同步** ✗）。
//   现在：梯度在显存时 ⇒
//     · 范数/最大值/NaN 计数 → `grad_stats_cuda`（只回传 3 个 double ✓，不再搬整个张量）
//     · 缩放 + NaN/Inf→0      → `grad_scale_cuda`（就地，判据与 CPU 逐字一致 ✓）
//     · per-loss 累加器        → 放显存（`grad_accumulate_cuda`），最后 D2D 写回梯度 ✓
//   数值：范数是 device double 规约（语义同 CPU 的 double 累加 ✓；加法顺序可能不同 ⇒
//   scale 因子末位可有差异，实测 loss 无差异 ✓）。
// ============================================================
extern int  grad_stats_cuda(const float* g, int64_t n, double* sum_sq, double* maxabs,
                            long long* nnan);
extern int  grad_scale_cuda(float* g, int64_t n, float scale);
extern int  grad_accumulate_cuda(float* acc, const float* g, int64_t n);
extern int  optim_fill_cuda(float* dst, int64_t n, float val);
extern void* optim_alloc_cuda(int64_t bytes);
extern void  optim_free_cuda(void* p);
extern int   optim_d2d_cuda(void* dst, const void* src, int64_t bytes);
extern int   optim_cuda_available();

// ---- 张量读写工具：兼容 backend buffer / 裸 CPU / CUDA 裸张量 ----
// 当全图在 CUDA 上时，参数/梯度 buffer_ 指向 device 内存，grad->data() 是 device 指针，
// 不能直接当 host 读写；统一经 buffer_->get_tensor（D2H 同步拷贝）/ set_tensor（H2D）访问。
namespace {
// 【2026-09-19】设备判定与裸指针（GPU 路径用；host/裸 CPU 张量返回 false ⇒ 走原路径 ✓）
bool tensor_on_device(const TensorF32* t) {
    return t && t->buffer_ && !t->buffer_->is_host() && t->data() != nullptr;
}
float* dev_f32(TensorF32* t) { return reinterpret_cast<float*>(t->data()); }

std::vector<float> read_tensor_values(const TensorF32* t) {
    std::vector<float> buf(static_cast<size_t>(t->numel()));
    const size_t bytes = static_cast<size_t>(t->numel()) * sizeof(float);
    if (t->buffer_) {
        // 从 backend buffer 读取（CPU buffer 直接 memcpy；CUDA buffer 内部自动 D2H 同步）
        t->buffer_->get_tensor(t, buf.data(), t->buffer_offs_, bytes);
    } else if (t->device() == Device::CPU) {
        if (!t->data()) {
            // 梯度 data 未分配（SE3 部分参数无梯度路径 / graph_compute 未算出）→ 返回空（跳过）
            if (getenv("GRAPH_DEBUG_GRAD")) {
                std::fprintf(stderr, "[GRAD-NULL] grad data null, numel=%lld\n",
                    (long long)t->numel());
            }
            return {};
        }
        std::memcpy(buf.data(), t->data(), bytes);
    } else {
        // CUDA 且无 buffer：拷贝一份 CPU 张量读取
        TensorF32 tcpu = t->cpu();
        std::memcpy(buf.data(), tcpu.data(), bytes);
    }
    return buf;
}

void write_tensor_values(TensorF32* t, const std::vector<float>& vals) {
    const size_t bytes = static_cast<size_t>(t->numel()) * sizeof(float);
    if (t->buffer_) {
        t->buffer_->set_tensor(t, vals.data(), t->buffer_offs_, bytes);
    } else {
        std::memcpy(t->data(), vals.data(), bytes);
    }
}

void fill_tensor_values(TensorF32* t, float val) {
    const int64_t n = t->numel();
    // 【2026-09-19】device ⇒ 就地填充（0 走 cudaMemset / 其它走 fill kernel ✓，不再造 host 向量 ✓）
    if (tensor_on_device(t) && optim_fill_cuda(dev_f32(t), n, val) == 0) return;
    if (t->buffer_) {
        std::vector<float> v(static_cast<size_t>(n), val);
        t->buffer_->set_tensor(t, v.data(), t->buffer_offs_, static_cast<size_t>(n) * sizeof(float));
    } else if (t->device() == Device::CPU) {
        std::fill(t->data(), t->data() + n, val);
    } else {
        std::vector<float> v(static_cast<size_t>(n), val);
        write_tensor_values(t, v);
    }
}
} // namespace

/// @brief 激活单个 loss 节点：将其 upstream grad 设为 1.0，其他 loss 设为 0.0
static void activate_single_loss(
    ComputeGraph* cgraph,
    TensorF32* active_loss,
    const std::vector<TensorF32*>& all_losses)
{
    for (auto* loss_node : all_losses) {
        if (!loss_node) continue;
        TensorF32* grad = cgraph->graph_get_grad(loss_node);
        if (!grad) continue;
        float val = (loss_node == active_loss) ? 1.0f : 0.0f;
        // buffer 感知：CUDA 上写 device 内存（D2H 经 set_tensor）
        fill_tensor_values(grad, val);
    }
}

/// @brief 清空所有参数的梯度缓冲区（grad_accs）。参数是 node 非 leaf，遍历 n_nodes()。
static void clear_param_grads(ComputeGraph* cgraph) {
    for (int i = 0; i < cgraph->n_nodes(); i++) {
        TensorF32* node = cgraph->graph_node(i);
        if (!(node->flag & TENSOR_FLAG_PARAM)) continue;
        TensorF32* grad = cgraph->graph_get_grad(node);
        if (grad) {
            fill_tensor_values(grad, 0.0f);
        }
    }
}

/// @brief 计算所有参数梯度的 L2 范数平方
/// @param buffers 参数缓冲区列表
/// @return 总 L2 范数
static float compute_total_grad_norm(const std::vector<TensorF32*>& param_list, ComputeGraph* cgraph) {
    double norm_sq = 0.0;  // double 防溢出
    // 诊断：记录 L2 贡献最大的参数（scale 之前，定位爆炸源，避免被 clip 缩小后误判）
    struct PInfo { int node_idx = -1; int64_t numel = 0; double l2 = 0; double maxabs = 0; long nnan = 0; };
    std::vector<PInfo> pinfos;
    pinfos.reserve(param_list.size());
    for (auto* param : param_list) {
        TensorF32* grad = cgraph->graph_get_grad(param);
        if (!grad) continue;
        const int64_t n = grad->numel();
        double pl2 = 0.0; double pmax = 0.0; long nnan = 0;
        if (tensor_on_device(grad)) {
            // 【2026-09-19】device ⇒ 只回传 3 个统计量（Σg²/max|g|/NaN 数），**不搬整张量** ✓
            double s2 = 0.0, mx = 0.0;
            long long nn = 0;
            if (grad_stats_cuda(dev_f32(grad), n, &s2, &mx, &nn) == 0) {
                pl2 = s2; pmax = mx; nnan = (long)nn;
                norm_sq += s2;                      // 与 CPU 同：跨参数用 double 累加 ✓
            } else {
                continue;                           // 统计失败 ⇒ 跳过该参数（保守 ✓）
            }
        } else {
            std::vector<float> g = read_tensor_values(grad);  // D2H（若 device）
            for (int64_t j = 0; j < n; j++) {
                const double v = static_cast<double>(g[static_cast<size_t>(j)]);
                if (v != v) { ++nnan; continue; }   // NaN 不计入 l2/max，只统计数量
                norm_sq += v * v;
                pl2 += v * v;
                const double a = std::fabs(v);
                if (a > pmax) pmax = a;
            }
        }
        PInfo pi; pi.numel = n; pi.l2 = pl2; pi.maxabs = pmax; pi.nnan = nnan;
        pinfos.push_back(pi);
    }
    const float total = static_cast<float>(std::sqrt(norm_sq));
    // 仅当范数异常（爆炸/NaN）时，在 scale 之前打印 top-8 贡献参数
    if (total > 100000.0f || std::isnan(total)) {
        auto find_idx = [&](const TensorF32* p) -> int {
            for (int i = 0; i < cgraph->n_nodes(); ++i)
                if (cgraph->graph_node(i) == p) return i;
            return -1;
        };
        std::vector<std::pair<double,int>> ranked;
        for (size_t k = 0; k < pinfos.size(); ++k) ranked.push_back({pinfos[k].l2, (int)k});
        std::sort(ranked.begin(), ranked.end(),
                  [](const std::pair<double,int>& a, const std::pair<double,int>& b){ return a.first > b.first; });
        std::fprintf(stderr, "[GRAD-NORM] total=%.4e params=%zu\n", total, pinfos.size());
        const int m = (int)ranked.size() < 8 ? (int)ranked.size() : 8;
        for (int q = 0; q < m; ++q) {
            const PInfo& pi = pinfos[ranked[q].second];
            const int nidx = find_idx(param_list[ranked[q].second]);
            std::fprintf(stderr, "  [GRAD-NORM] rank=%d node_idx=%d numel=%lld l2=%.4e max_abs=%.4e nnan=%ld",
                         q, nidx, (long long)pi.numel, pi.l2, pi.maxabs, pi.nnan);
        TensorF32* pn = (nidx >= 0 && nidx < cgraph->n_nodes()) ? cgraph->graph_node(nidx) : nullptr;
        if (pn && pn->shape().ndim() >= 1 && pn->shape().ndim() <= 4) {
            std::fprintf(stderr, " dims=[");
            for (int d = 0; d < pn->shape().ndim(); ++d)
                std::fprintf(stderr, "%s%lld", (d ? "," : ""), (long long)pn->shape().dims[d]);
            std::fprintf(stderr, "] op=%d", (int)pn->op);
        }
        std::fprintf(stderr, "\n");
        }
    }
    return total;
}

/// @brief 等比例缩放所有参数梯度
/// 这里把非有限元素置 0（等价"该元素不贡献梯度"），打破"梯度 NaN→参数 NaN→前向 NaN"恶性循环。
static void scale_param_grads(const std::vector<TensorF32*>& param_list, ComputeGraph* cgraph, float scale) {
    for (auto* param : param_list) {
        TensorF32* grad = cgraph->graph_get_grad(param);
        if (!grad) continue;
        // 【2026-09-19】device ⇒ 就地缩放（判据与下面 CPU 版逐字一致：NaN/Inf→0 ✓），零往返 ✓
        if (tensor_on_device(grad) &&
            grad_scale_cuda(dev_f32(grad), grad->numel(), scale) == 0) {
            continue;
        }
        std::vector<float> g = read_tensor_values(grad);  // D2H（若 device）
        const int64_t n = grad->numel();
        for (int64_t j = 0; j < n; j++) {
            float& v = g[static_cast<size_t>(j)];
            if (v != v || v > 3.0e38f || v < -3.0e38f) { v = 0.0f; continue; }  // NaN/Inf→0
            v *= scale;
        }
        write_tensor_values(grad, g);  // H2D 写回
    }
}

/// @brief 收集所有带梯度的参数节点。
/// 注意: 参数 (TENSOR_FLAG_PARAM) 在 build_forward_impl 中归类为 node 而非 leaf
///       (ComputeGraph.cpp:137: OP_NONE && !PARAM 才作 leaf)，故必须遍历 n_nodes()。
static std::vector<TensorF32*> collect_params(ComputeGraph* cgraph) {
    std::vector<TensorF32*> params;
    for (int i = 0; i < cgraph->n_nodes(); i++) {
        TensorF32* node = cgraph->graph_node(i);
        if (node->flag & TENSOR_FLAG_PARAM) {
            TensorF32* grad = cgraph->graph_get_grad(node);
            if (grad) {
                params.push_back(node);
            }
        }
    }
    return params;
}

// ============================================================
// 公共 API
// ============================================================

float clip_grad_norm(ComputeGraph* cgraph, float max_norm) {
    if (max_norm <= 0.0f) return 0.0f;

    auto params = collect_params(cgraph);
    if (params.empty()) return 0.0f;

    // [2026-09-07 Epoch2 NaN 止损] 显式清零含 NaN 的参数梯度：
    //   compute_total_grad_norm 跳过 NaN（不计入 norm）→ 若整体 norm 未超 clip 阈值，
    //   scale_param_grads 不会被调用 → NaN 梯度原样进入 AdamW → 参数 NaN → Epoch2 全链 NaN
    //   （实测：混合 per_block Epoch1 后 SE3 embed_x/embed_e 权重 NaN，Epoch2 loss=-nan；
    //    CPU per_block 无此问题 → GPU 反向链个别参数梯度 NaN）。
    for (auto* param : params) {
        TensorF32* grad = cgraph->graph_get_grad(param);
        if (!grad) continue;
        // 【2026-09-19】device ⇒ 用统计量判 NaN（只回传计数 ✓），命中则**整张清零**（memset ✓）
        if (tensor_on_device(grad)) {
            const int64_t n = grad->numel();
            double s2 = 0.0, mx = 0.0;
            long long nn = 0;
            if (grad_stats_cuda(dev_f32(grad), n, &s2, &mx, &nn) != 0) continue;
            if (nn > 0) {
                optim_fill_cuda(dev_f32(grad), n, 0.0f);
                if (getenv("GRAPH_DEBUG_GRAD_NAN")) {
                    fprintf(stderr, "[GRAD-NAN-CLR] param cleared (NaN grad->0) [device, nnan=%lld]\n", nn);
                }
            }
            continue;
        }
        std::vector<float> g = read_tensor_values(grad);  // D2H（若 device）
        bool has_nan = false;
        for (const float v : g) { if (v != v) { has_nan = true; break; } }
        if (has_nan) {
            for (float& v : g) v = 0.0f;
            write_tensor_values(grad, g);  // H2D 写回
            if (getenv("GRAPH_DEBUG_GRAD_NAN")) {
                fprintf(stderr, "[GRAD-NAN-CLR] param cleared (NaN grad->0)\n");
            }
        }
    }

    // SE3 梯度尺度（~1e13）远大于主图/head 梯度（~1e5），若与主图一起做全局比例 clip，
    // SE3 主导 scale → msa head 等正常参数被压到极小（1e-9）→ 权重几乎不动 → msa/chi 不学习
    // （实测开 SE3 时 msa=3.13 恒定，关 SE3 时 msa 下降）。
    // 方案：SE3 参数单独 clip 到 max_norm*0.1（更小阈值），非 SE3 参数用全局 clip（max_norm）。
    // 这样 msa head 等正常梯度不受 SE3 主导拖累。
    std::vector<TensorF32*> se3_params, main_params;
    for (auto* p : params) {
        if (p->flag & TENSOR_FLAG_SE3) se3_params.push_back(p);
        else main_params.push_back(p);
    }

    // 1) 主图/head 参数（非 SE3）：全局 clip（正常阈值）
    float main_norm = 0.0f;
    if (!main_params.empty()) {
        main_norm = compute_total_grad_norm(main_params, cgraph);
        if (main_norm > max_norm && main_norm > 1e-12f) {
            scale_param_grads(main_params, cgraph, max_norm / main_norm);
        }
    }

    // 2) SE3 参数：单独 clip（阈值 = max_norm*0.1，防主导）
    const float se3_max = max_norm * 0.1f;
    float se3_norm = 0.0f;
    if (!se3_params.empty()) {
        se3_norm = compute_total_grad_norm(se3_params, cgraph);
        if (se3_norm > se3_max && se3_norm > 1e-12f) {
            scale_param_grads(se3_params, cgraph, se3_max / se3_norm);
        }
    }

    // 返回实际**应用**（clip 后）的总体范数，作为收敛诊断指标。
    // 之前返回 main_norm + se3_norm（raw，SE3 可达 1e10+）→ 误导"grad_norm 爆炸"。
    // clip 后：主图 ≤ max_norm，SE3 ≤ se3_max，二者之和即真实步长量级。
    const float applied_main = std::min(main_norm, max_norm);
    const float applied_se3 = std::min(se3_norm, se3_max);
    return applied_main + applied_se3;
}

float accumulate_per_loss_gradients(
    ComputeGraph*                     cgraph,
    Backend*                          backend,
    const std::vector<TensorF32*>&    loss_nodes,
    const std::vector<float>&         clip_norms,
    const PerLossClipConfig&          config,
    std::vector<LossGradientInfo>*    out_info)
{
    assert(loss_nodes.size() == clip_norms.size());
    assert(cgraph && backend);

    // ---- Step 0: 收集所有可训练参数及其梯度缓冲区 ----
    auto param_list = collect_params(cgraph);
    if (param_list.empty()) return 0.0f;

    // 为每个参数分配一个累加器
    //   【2026-09-19】梯度在显存时 ⇒ 累加器也放**显存**（否则每个 loss 都要 D2H+H2D ✗）✓
    struct ParamAccum {
        TensorF32* param = nullptr;
        std::vector<float> accum;   // host 累加器（梯度在 host 时用）
        float*  d_accum = nullptr;  // device 累加器（指向下面 arena 内的一段 ✓）
        int64_t dev_off = 0;        // 在 arena 内的元素偏移
        int64_t n = 0;
    };
    std::vector<ParamAccum> accumulators;
    accumulators.reserve(param_list.size());
    int64_t dev_total = 0;
    for (auto* p : param_list) {
        TensorF32* grad = cgraph->graph_get_grad(p);
        ParamAccum a;
        a.param = p;
        a.n = grad ? grad->numel() : 0;
        if (a.n > 0 && tensor_on_device(grad)) {
            a.dev_off = dev_total;             // 先记偏移，分配后再落真实地址 ✓
            dev_total += a.n;
        } else {
            a.accum.assign(static_cast<size_t>(a.n), 0.0f);
        }
        accumulators.push_back(std::move(a));
    }
    // 一次 cudaMalloc 拿整块 arena（避免逐参数分配 ✗）+ RAII 释放 ✓
    struct ArenaGuard {
        void* base = nullptr;
        ~ArenaGuard() { if (base) optim_free_cuda(base); }
    } arena;
    if (dev_total > 0) {
        arena.base = optim_alloc_cuda(dev_total * (int64_t)sizeof(float));
        if (!arena.base) {                     // 分配失败 ⇒ 全部退回 host 累加器（保守 ✓）
            for (auto& a : accumulators) {
                if (a.n > 0 && a.dev_off < dev_total) {
                    a.d_accum = nullptr;
                    a.accum.assign(static_cast<size_t>(a.n), 0.0f);
                }
            }
            dev_total = 0;
        } else {
            float* d_base = reinterpret_cast<float*>(arena.base);
            for (auto& a : accumulators) {
                if (a.n <= 0) continue;
                if (a.accum.empty()) {         // device 累加器 ✓
                    a.d_accum = d_base + a.dev_off;
                    optim_fill_cuda(a.d_accum, a.n, 0.0f);   // 清零初始化 ✓
                }
            }
        }
    }

    // ---- Step 1: 逐 loss 执行 backward + per-loss clip + accumulate ----
    std::vector<LossGradientInfo> info_vec;

    for (size_t k = 0; k < loss_nodes.size(); k++) {
        TensorF32* loss_node = loss_nodes[k];
        if (!loss_node) continue;

        float per_loss_max = clip_norms[k];

        // 1a. 激活当前 loss，抑制其他 loss
        activate_single_loss(cgraph, loss_node, loss_nodes);

        // 1b. 清空参数梯度（上一轮残留）
        clear_param_grads(cgraph);

        // 1c. 执行 backward compute
        backend->graph_compute(cgraph);

        // 1d. 计算 per-loss norm
        float loss_norm = compute_total_grad_norm(param_list, cgraph);
        bool was_clipped = false;

        // 1e. per-loss clip
        if (loss_norm > per_loss_max && per_loss_max > 0.0f && loss_norm > 1e-12f) {
            float s = per_loss_max / loss_norm;
            scale_param_grads(param_list, cgraph, s);
            was_clipped = true;
        }

        // 1f. 累加到 persistent accumulator
        for (size_t a = 0; a < accumulators.size(); a++) {
            TensorF32* grad = cgraph->graph_get_grad(accumulators[a].param);
            if (!grad) continue;
            // 【2026-09-19】device 累加器 ⇒ 显存内累加，零往返 ✓
            if (accumulators[a].d_accum && tensor_on_device(grad) &&
                grad_accumulate_cuda(accumulators[a].d_accum, dev_f32(grad), accumulators[a].n) == 0) {
                continue;
            }
            std::vector<float> gdata = read_tensor_values(grad);  // D2H（若 device）
            int64_t n = grad->numel();
            for (int64_t j = 0; j < n; j++) {
                accumulators[a].accum[static_cast<size_t>(j)] += gdata[static_cast<size_t>(j)];
            }
        }

        // 1g. 记录诊断信息
        LossGradientInfo info;
        info.loss_node    = loss_node;
        info.raw_norm     = loss_norm;
        info.clipped_norm = was_clipped ? per_loss_max : loss_norm;
        info.was_clipped  = was_clipped;
        info_vec.push_back(info);
    }

    // ---- Step 2: 将累加结果写回 cgraph 的参数梯度 ----
    for (size_t a = 0; a < accumulators.size(); a++) {
        TensorF32* grad = cgraph->graph_get_grad(accumulators[a].param);
        if (!grad) continue;
        // 【2026-09-19】device 累加器 ⇒ 显存内 D2D 写回（零 H2D ✓）
        if (accumulators[a].d_accum && tensor_on_device(grad) &&
            optim_d2d_cuda(dev_f32(grad), accumulators[a].d_accum,
                           accumulators[a].n * (int64_t)sizeof(float)) == 0) {
            continue;
        }
        write_tensor_values(grad, accumulators[a].accum);  // H2D（若 device）
    }

    // ---- Step 3: 全局裁剪 ----
    float total_norm = clip_grad_norm(cgraph, config.global_max_norm);

    // ---- Step 4: 输出诊断 ----
    if (out_info) {
        *out_info = std::move(info_vec);
    }

    return total_norm;
}

float apply_per_loss_clip(
    ComputeGraph*           cgraph,
    Backend*                backend,
    TensorF32*              loss_fape,
    TensorF32*              loss_chi,
    TensorF32*              loss_distogram,
    TensorF32*              loss_msa,
    TensorF32*              loss_conf,
    const PerLossClipConfig& config,
    std::vector<LossGradientInfo>* out_info)
{
    // 构建 loss_nodes 和 clip_norms 向量
    // 顺序: FAPE, Chi, Distogram, MSA, Conf
    struct LossEntry {
        TensorF32* node;
        float      max_norm;
        const char* name;
    };

    LossEntry entries[] = {
        {loss_fape,      config.fape_max_norm,      "FAPE"},
        {loss_chi,       config.chi_max_norm,       "Chi"},
        {loss_distogram, config.distogram_max_norm, "Distogram"},
        {loss_msa,       config.msa_max_norm,       "MSA"},
        {loss_conf,      config.conf_max_norm,      "Conf"},
    };

    std::vector<TensorF32*> loss_nodes;
    std::vector<float>      clip_norms;
    loss_nodes.reserve(5);
    clip_norms.reserve(5);

    for (const auto& entry : entries) {
        if (entry.node) {
            loss_nodes.push_back(entry.node);
            clip_norms.push_back(entry.max_norm);
        }
    }

    std::vector<LossGradientInfo> info;
    float result = accumulate_per_loss_gradients(
        cgraph, backend, loss_nodes, clip_norms, config, &info);

    // 注入 loss name
    for (auto& inf : info) {
        for (const auto& entry : entries) {
            if (inf.loss_node == entry.node) {
                inf.loss_name = entry.name;
                break;
            }
        }
    }

    if (out_info) {
        *out_info = std::move(info);
    }

    return result;
}

} // namespace ppml

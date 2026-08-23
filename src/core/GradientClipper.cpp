#include "ppml/GradientClipper.h"
#include "ppml/Tensor.h"

#include <cmath>
#include <cstring>
#include <cassert>

namespace ppml {

// ============================================================
// 内部辅助函数
// ============================================================

// ---- 张量读写工具：兼容 backend buffer / 裸 CPU / CUDA 裸张量 ----
// 当全图在 CUDA 上时，参数/梯度 buffer_ 指向 device 内存，grad->data() 是 device 指针，
// 不能直接当 host 读写；统一经 buffer_->get_tensor（D2H 同步拷贝）/ set_tensor（H2D）访问。
namespace {
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
        std::vector<float> g = read_tensor_values(grad);  // D2H（若 device）
        const int64_t n = grad->numel();
        double pl2 = 0.0; double pmax = 0.0; long nnan = 0;
        for (int64_t j = 0; j < n; j++) {
            const double v = static_cast<double>(g[static_cast<size_t>(j)]);
            if (v != v) { ++nnan; continue; }   // NaN 不计入 l2/max，只统计数量
            norm_sq += v * v;
            pl2 += v * v;
            const double a = std::fabs(v);
            if (a > pmax) pmax = a;
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
/// ⚠️ 若梯度含 NaN/Inf，直接 *=scale 会把 NaN 传播进 Adam 更新 → 参数变 NaN → 下一 epoch 前向全 NaN。
/// 这里把非有限元素置 0（等价"该元素不贡献梯度"），打破"梯度 NaN→参数 NaN→前向 NaN"恶性循环。
static void scale_param_grads(const std::vector<TensorF32*>& param_list, ComputeGraph* cgraph, float scale) {
    for (auto* param : param_list) {
        TensorF32* grad = cgraph->graph_get_grad(param);
        if (!grad) continue;
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

    // ⚠️ 分离 SE3 参数（TENSOR_FLAG_SE3）单独 clip（2026-08-22 修复）：
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

    // 为每个参数分配一个 CPU 端持久化累加器
    struct ParamAccum {
        TensorF32* param;
        std::vector<float> accum;  // CPU-side persistent accumulator
    };
    std::vector<ParamAccum> accumulators;
    accumulators.reserve(param_list.size());
    for (auto* p : param_list) {
        TensorF32* grad = cgraph->graph_get_grad(p);
        int64_t n = grad ? grad->numel() : 0;
        accumulators.push_back({p, std::vector<float>(static_cast<size_t>(n), 0.0f)});
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

        // 1f. 累加到 persistent accumulator（CPU 端）
        for (size_t a = 0; a < accumulators.size(); a++) {
            TensorF32* grad = cgraph->graph_get_grad(accumulators[a].param);
            if (!grad) continue;
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

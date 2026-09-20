#include "ppml/AdamW.h"
#include "ppml/Backend.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>

namespace ppml {

// ============================================================
// 【2026-09-19】优化器 GPU 路径（把逐参数的 D2H/H2D 往返换成显存内就地 kernel）
//   背景（实测）：dev 1 epoch 里 `cudaMemcpy` 11,767 次 / 2.36 s，其中 **D2H 占 84.7%**
//   且中位 41 KB（= 参数级）—— 来源正是本文件的"读 w / 读 g / 写 w"三步往返 ✗
//   （pageable 拷贝还会**隐式做一次全设备同步** ✗ ⇒ 代价 = 同步 + 传输）。
//   现在：参数/梯度都在显存时 ⇒ 分配 device 版 m/v，整步 1 个 kernel，**零往返** ✓
//   · kernel 逐元素运算顺序与下面 CPU 版**逐字相同** ⇒ 数值逐位一致 ✓（回归判据 ✓）
//   · host m/v 保留为 checkpoint 镜像，只在 export/import_momentum 时同步 ✓
//   · 关闭开关：PPML_OPTIM_GPU=0（回到原 CPU 路径，便于 A/B ✓）
// ============================================================
// 注意：必须在 **ppml 命名空间**声明（不是匿名 namespace ✗）—— 定义在 src/cuda/OptimKernels.cu
//   里的 ppml::xxx ✓，放进匿名 namespace 会变成另一个符号 ⇒ 链接期 undefined reference ✗
extern int   adamw_step_cuda(float* w, const float* g, float* m, float* v, int64_t n,
                             float lr, float lr_scale, float beta1, float beta2, float eps,
                             float wd, float bc1, float bc2, int no_weight_decay);
extern void* optim_alloc_cuda(int64_t bytes);
extern void  optim_free_cuda(void* p);
extern int   optim_h2d_cuda(void* dst, const void* src, int64_t bytes);
extern int   optim_d2h_cuda(void* dst, const void* src, int64_t bytes);
extern int   optim_cuda_available();

namespace {
// 参数/梯度是否都在显存（host buffer 或裸 CPU 张量一律走原 CPU 路径 ✓）
bool tensor_on_device(const TensorF32* t) {
    return t && t->buffer_ && !t->buffer_->is_host() && t->data() != nullptr;
}

bool gpu_optim_enabled() {
    static const bool on = [] {
        const char* s = std::getenv("PPML_OPTIM_GPU");
        if (s && *s && std::atoi(s) == 0) return false;   // 显式关闭（A/B 用 ✓）
        return true;
    }();
    return on && optim_cuda_available() != 0;
}
} // namespace

AdamW::~AdamW() {
    // 回收 GPU 路径分配过的 m/v（没用过就是 nullptr，无副作用 ✓）
    for (auto& s : states_) {
        if (s.d_m) { optim_free_cuda(s.d_m); s.d_m = nullptr; }
        if (s.d_v) { optim_free_cuda(s.d_v); s.d_v = nullptr; }
    }
}

AdamW::AdamW(float lr, float weight_decay, float beta1, float beta2,
             float eps, bool bias_correction, float lora_lr_scale)
    : lr_(lr),
      weight_decay_(weight_decay),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      bias_correction_(bias_correction),
      lora_lr_scale_(lora_lr_scale)
{}

void AdamW::init_from_graph(ComputeGraph* cgraph) {
    states_.clear();
    n_owned_ = 0;
    // 参数 (TENSOR_FLAG_PARAM) 在 build_forward_impl 中被归类为 node 而非 leaf
    // (ComputeGraph.cpp:137: OP_NONE && !PARAM 才作 leaf)，故遍历 n_nodes()。
    // 分片（阶段 A）：idx = 参数全局序号（含被跳过的），owner = idx % world ⇒ 两端一致。
    int idx = 0;
    for (int i = 0; i < cgraph->n_nodes(); i++) {
        TensorF32* node = cgraph->graph_node(i);
        if (!(node->flag & TENSOR_FLAG_PARAM)) continue;          // 非参数，跳过
        TensorF32* grad = cgraph->graph_get_grad(node);
        if (!grad) continue;                                      // 无梯度（未参与 loss），跳过
        const int owner = (world_ > 1) ? (idx % world_) : 0;
        ++idx;
        if (world_ > 1 && owner != my_rank_) continue;            // 非本 rank owner：不建状态（省 m/v）
        const int64_t n = grad->numel();
        ParamState s;
        s.param            = node;
        s.owner_rank       = owner;
        s.no_weight_decay  = (node->flag & TENSOR_FLAG_NO_WEIGHT_DECAY) != 0;
        // SE3 等变参数：梯度尺度与主图不匹配（offset→坐标→FAPE 链放大），用分层小 lr。
        // 2026-08-24: 0.1→0.01（新版 a_rows 正确 shape 后 SE3 权重梯度 l2~1e25，
        //   0.1 分层仍致权重漂移/epoch2 爆炸；降到 0.01 减缓更新步长）。
        s.se3_lr_scale     = (node->flag & TENSOR_FLAG_SE3)  ? 0.01f : 1.0f;
        s.lora_lr_scale    = (node->flag & TENSOR_FLAG_LORA) ? lora_lr_scale_ : 1.0f;
        s.m.assign(static_cast<size_t>(n), 0.0f);
        s.v.assign(static_cast<size_t>(n), 0.0f);
        states_.push_back(std::move(s));
    }
}

// ---- 内部工具：把张量数据读到 CPU 向量（兼容 backend buffer / 裸 CPU / CUDA 裸张量）----
namespace {
std::vector<float> read_tensor_values(const TensorF32* t) {
    std::vector<float> buf(static_cast<size_t>(t->numel()));
    const size_t bytes = static_cast<size_t>(t->numel()) * sizeof(float);
    if (t->buffer_) {
        // 从 backend buffer 读取（CPU buffer 直接 memcpy；CUDA buffer 内部自动 H2D）
        t->buffer_->get_tensor(t, buf.data(), t->buffer_offs_, bytes);
    } else if (t->device() == Device::CPU) {
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
} // namespace

void AdamW::step(ComputeGraph* cgraph) {
    if (states_.empty()) return;

    step_count_++;
    const int   t = step_count_;
    const float b1 = beta1_, b2 = beta2_;
    const float lr = lr_, eps = eps_, wd = weight_decay_;
    const bool  bc = bias_correction_;

    // 偏差校正分母（避免逐参数重复求幂）
    const float bc1 = bc ? (1.0f - std::pow(b1, static_cast<float>(t))) : 1.0f;
    const float bc2 = bc ? (1.0f - std::pow(b2, static_cast<float>(t))) : 1.0f;

    const bool gpu_ok = gpu_optim_enabled();
    // 诊断（GRAPH_DEBUG_OPTIM=1）：本步有多少参数走了 GPU 路径（不设则零开销 ✓）
    static const bool dbg_optim = (std::getenv("GRAPH_DEBUG_OPTIM") != nullptr);
    int n_gpu_params = 0, n_cpu_params = 0;

    for (auto& s : states_) {
        TensorF32* grad = cgraph->graph_get_grad(s.param);
        if (!grad) continue;

        // 参数与梯度尺寸应一致；否则跳过（防御）
        const int64_t n = grad->numel();
        if (n != s.param->numel()) continue;

        // ---- 【2026-09-19】GPU 路径：w/g/m/v 全在显存 ⇒ 1 个 kernel 完成整步，零 H2D/D2H ✓ ----
        if (gpu_ok && tensor_on_device(s.param) && tensor_on_device(grad)) {
            const int64_t bytes = n * (int64_t)sizeof(float);
            if (!s.d_m || s.dev_n != n) {                  // 惰性分配 / 尺寸变化 ⇒ 重分配
                if (s.d_m) { optim_free_cuda(s.d_m); s.d_m = nullptr; }
                if (s.d_v) { optim_free_cuda(s.d_v); s.d_v = nullptr; }
                s.dev_n     = 0;
                s.dev_valid = false;
                s.d_m = optim_alloc_cuda(bytes);
                s.d_v = optim_alloc_cuda(bytes);
                if (s.d_m && s.d_v) s.dev_n = n;
            }
            if (s.d_m && s.d_v && s.dev_n == n) {
                if (!s.dev_valid) {                        // 首次 / import 之后：host→device 上传一次 ✓
                    s.dev_valid = (optim_h2d_cuda(s.d_m, s.m.data(), bytes) == 0 &&
                                   optim_h2d_cuda(s.d_v, s.v.data(), bytes) == 0);
                }
                if (s.dev_valid) {
                    const float lr_scale = s.se3_lr_scale * s.lora_lr_scale;
                    const int rc = adamw_step_cuda(
                        reinterpret_cast<float*>(s.param->data()),
                        reinterpret_cast<const float*>(grad->data()),
                        reinterpret_cast<float*>(s.d_m),
                        reinterpret_cast<float*>(s.d_v),
                        n, lr, lr_scale, b1, b2, eps, wd, bc1, bc2,
                        s.no_weight_decay ? 1 : 0);
                    if (rc == 0) {
                        s.host_stale = true;   // host m/v 已落后（export 时按需 D2H ✓）
                        ++n_gpu_params;
                        continue;              // ★ 跳过下面的 CPU 往返 ✓
                    }
                }
            }
            // 分配/上传失败 ⇒ 保守落到下面的 CPU 路径 ✓
        }

        // 读取参数与梯度的 CPU 副本
        std::vector<float> w = read_tensor_values(s.param);
        std::vector<float> g = read_tensor_values(grad);

        float* m = s.m.data();
        float* v = s.v.data();
        const bool nod = s.no_weight_decay;
        // SE3 与 LoRA 分层 lr 可叠加（不同 flag 不冲突）：最终缩放 = 两者乘积
        const float lr_scale = s.se3_lr_scale * s.lora_lr_scale;

        for (int64_t i = 0; i < n; ++i) {
            const float gi = g[static_cast<size_t>(i)];

            // m = beta1*m + (1-beta1)*g
            m[i] = b1 * m[i] + (1.0f - b1) * gi;
            // v = beta2*v + (1-beta2)*g^2
            v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;

            // decoupled AdamW 更新：
            //   p -= lr * lr_scale * m_hat / (sqrt(v_hat) + eps) + lr * lr_scale * wd * p
            //   weight_decay 不进 m/v，且 no_weight_decay 参数跳过 wd 项
            const float m_hat = m[i] / bc1;
            const float v_hat = v[i] / bc2;
            const float update = lr * lr_scale * m_hat / (std::sqrt(v_hat) + eps);
            w[static_cast<size_t>(i)] -= update;
            if (wd != 0.0f && !nod) {
                w[static_cast<size_t>(i)] -= lr * lr_scale * wd * w[static_cast<size_t>(i)];
            }
        }

        // 写回参数
        write_tensor_values(s.param, w);
        ++n_cpu_params;
    }

    if (dbg_optim) {
        std::fprintf(stderr,
                     "[OPTIM] step %d：GPU 参数=%d / CPU 参数=%d"
                     "（GPU 路径=显存内就地 kernel ⇒ 零 H2D/D2H ✓）\n",
                     step_count_, n_gpu_params, n_cpu_params);
        // 为什么没走 GPU？抽样打印前 2 个参数的 param/grad 是否在显存 ✓（定位 placement 问题）
        int shown = 0;
        for (auto& s : states_) {
            if (shown >= 2) break;
            TensorF32* g = cgraph->graph_get_grad(s.param);
            std::fprintf(stderr,
                         "[OPTIM]   样本 param buf=%p is_host=%d data=%p | grad buf=%p is_host=%d data=%p\n",
                         (void*)s.param->buffer_,
                         s.param->buffer_ ? (int)s.param->buffer_->is_host() : -1,
                         (const void*)s.param->data(),
                         (void*)(g ? g->buffer_ : nullptr),
                         (g && g->buffer_) ? (int)g->buffer_->is_host() : -1,
                         (const void*)(g ? g->data() : nullptr));
            ++shown;
        }
    }
}

// ---- 断点续训：导出 m/v 动量 (与 params 同序) ----
void AdamW::export_momentum(const std::vector<TensorF32*>& params,
                            const std::vector<std::string>& param_names,
                            std::vector<std::vector<float>>& ms,
                            std::vector<std::vector<float>>& vs) const {
    ms.clear(); vs.clear();
    ms.reserve(params.size()); vs.reserve(params.size());
    for (TensorF32* p : params) {
        // 按指针在 states_ 中找该参数状态
        const ParamState* st = nullptr;
        for (const auto& s : states_) {
            if (s.param == p) { st = &s; break; }
        }
        if (st) {
            // 【2026-09-19】GPU 路径下 device m/v 才是权威值 ⇒ 先把 host 镜像刷新回来 ✓
            ParamState& sm = const_cast<ParamState&>(*st);
            if (sm.d_m && sm.d_v && sm.dev_valid && sm.host_stale &&
                sm.dev_n == (int64_t)sm.m.size()) {
                const int64_t bytes = sm.dev_n * (int64_t)sizeof(float);
                if (optim_d2h_cuda(sm.m.data(), sm.d_m, bytes) == 0 &&
                    optim_d2h_cuda(sm.v.data(), sm.d_v, bytes) == 0) {
                    sm.host_stale = false;
                }
            }
            ms.push_back(sm.m);
            vs.push_back(sm.v);
        } else {
            ms.emplace_back();   // 无状态 → 空 (save 侧跳过)
            vs.emplace_back();
        }
        (void)param_names;
    }
}

// ---- 断点续训：恢复 m/v 动量 (须在 init_from_graph 之后) ----
void AdamW::import_momentum(const std::vector<TensorF32*>& params,
                            const std::vector<std::string>& param_names,
                            const std::map<std::string, std::vector<float>>& raw_tensors) {
    for (size_t i = 0; i < params.size(); ++i) {
        TensorF32* p = params[i];
        if (i >= param_names.size()) break;
        // 按指针在 states_ 中找
        ParamState* st = nullptr;
        for (auto& s : states_) {
            if (s.param == p) { st = &s; break; }
        }
        if (!st) continue;

        const std::string key_m = "opt.m." + param_names[i];
        const std::string key_v = "opt.v." + param_names[i];
        auto itm = raw_tensors.find(key_m);
        auto itv = raw_tensors.find(key_v);
        // 尺寸必须与当前状态一致 (m/v 与参数 numel 相同)；否则跳过 (保持 0)
        if (itm != raw_tensors.end() && itm->second.size() == st->m.size())
            st->m = itm->second;
        if (itv != raw_tensors.end() && itv->second.size() == st->v.size())
            st->v = itv->second;

        // 【2026-09-19】若已分配 device m/v ⇒ 立刻把恢复值传上去
        //   （否则下次 GPU step 会把显存里的旧值当权威值用 ✗）
        if (st->d_m && st->d_v && st->dev_n == (int64_t)st->m.size() && st->dev_n > 0) {
            const int64_t bytes = st->dev_n * (int64_t)sizeof(float);
            if (optim_h2d_cuda(st->d_m, st->m.data(), bytes) == 0 &&
                optim_h2d_cuda(st->d_v, st->v.data(), bytes) == 0) {
                st->dev_valid  = true;
                st->host_stale = false;
            } else {
                st->dev_valid = false;   // 上传失败 ⇒ 下次 step 会重试（用 host 值 ✓）
            }
        }
    }
}

void AdamW::zero_grad(ComputeGraph* cgraph) {
    for (int i = 0; i < cgraph->n_nodes(); i++) {
        TensorF32* node = cgraph->graph_node(i);
        if (!(node->flag & TENSOR_FLAG_PARAM)) continue;
        TensorF32* grad = cgraph->graph_get_grad(node);
        if (!grad) continue;
        std::vector<float> z(static_cast<size_t>(grad->numel()), 0.0f);
        write_tensor_values(grad, z);
    }
}

} // namespace ppml

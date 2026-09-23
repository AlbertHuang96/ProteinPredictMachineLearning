#include "ppml/Model.h"
#include "ppml/Embedding.h"
#include "ppml/PositionalEncoding.h"
#include "ppml/MathUtils.h"
#include <iostream>
#include <iomanip>          // std::fixed / std::setprecision
#include <algorithm>        // std::max

#include "ppml/Dropout.h"
#include "ppml/Context.h"
#include "ppml/RemoteBackend.h"   // 双机：远端后端注册（PPML_REMOTE_HOST 时启用，2026-09-14）
#include <cuda_runtime.h>   // cudaGetDeviceCount / cudaMemGetInfo 用于 GPU/显存探测

namespace ppml {

namespace {
// ===== proj_state_add_to_query_row 图版（掩码广播 add）=====
// 语义: msa[:,0,:,:] += proj_state  (msa 第 0 条序列 query 行注入)
//   msa        : 图 [D_MSA, L, N, B]   (值 (B,N,L,D_MSA))
//   proj_state : 图 [D_MSA, L, B]      (值 (B,L,D_MSA)，state2msa_linear 输出)
// 返回: 新的 msa 图节点（n==0 处 += proj_state，其余 n 不变）。
//
// 为什么不用 get_rows/set_rows：CPU kernel 只支持严格 2D(N,M)（行沿 dims[0]，每行
// 只 memcpy dims[1] 个 float），而目标 query 行在 dims[2]（N 序列维），无法正确
// 4D 切片。改为掩码广播，全部复用已验证 kernel：
//   unsqueeze / view / repeat / mul / add_impl。
TensorF32* query_row_add_graph(TensorF32* msa, TensorF32* proj_state) {
    const int64_t D = msa->shape().dims[0];  // D_MSA
    const int64_t L = msa->shape().dims[1];
    const int64_t N = msa->shape().dims[2];
    const int64_t B = msa->shape().dims[3];

    // 1) proj_state [D,L,B] -> [D,L,1,B]（OP_RESHAPE -> kernel_cpy，行主序不变）
    TensorF32* ps_unsq = unsqueeze(proj_state, 2);

    // 2) 目标形状占位节点（repeat 只取 b->shape()，不读数据）
    int64_t tgt_dims[] = {D, L, N, B};
    TensorF32* target = context().new_tensor<float>(4, tgt_dims);

    // 3) 常量掩码 [N] = [1,0,0,...] -> view 为 [1,1,N,1]
    int64_t mask_dims[] = {N};
    TensorF32* n_mask = context().new_tensor<float>(1, mask_dims);
    n_mask->flag = 0;                        // 常量，不可训练
    float* md = bind_leaf_data(context(), n_mask);
    for (int64_t i = 0; i < N; i++) md[i] = (i == 0) ? 1.0f : 0.0f;
    TensorF32* n_mask4 = view(n_mask, Shape{1, 1, N, 1});

    // 4) 掩码广播到 [D,L,N,B]：kernel_repeat 尾部对齐取模，仅在 n==0 处=1
    TensorF32* mask_r = repeat(n_mask4, target);
    // 5) proj_state 广播到 [D,L,N,B]（所有 n 相同）
    TensorF32* ps_r   = repeat(ps_unsq, target);
    // 6) addend = proj_state * mask -> n!=0 处清零
    TensorF32* addend = mul(ps_r, mask_r);
    // 7) msa + addend（同形 [D,L,N,B] 逐元素加）
    return add_impl(msa, addend, /*inplace=*/false);
}

// ===== 值版基础操作（值版 forward 用；图 helper add_impl/mul/sigmoid 返回图节点 data()=nullptr，
//       值版 copy_from 读 null 崩。这些是纯值逐元素实现）=====
// 加法：dst = a + b（同形逐元素；shape 不同时按扁平 numel 匹配，要求 numel 一致）
TensorF32 add_value(const TensorF32& a, const TensorF32& b) {
    TensorF32 out(a.shape(), a.device());
    const float* ap = a.data();
    const float* bp = b.data();
    float* op = out.data();
    const int64_t n = a.numel();
    for (int64_t i = 0; i < n; i++) op[i] = ap[i] + bp[i];
    return out;
}
// 乘法：dst = a * b（同形逐元素）
TensorF32 mul_value(const TensorF32& a, const TensorF32& b) {
    TensorF32 out(a.shape(), a.device());
    const float* ap = a.data();
    const float* bp = b.data();
    float* op = out.data();
    const int64_t n = a.numel();
    for (int64_t i = 0; i < n; i++) op[i] = ap[i] * bp[i];
    return out;
}
// sigmoid：dst = 1/(1+exp(-x))
TensorF32 sigmoid_value(const TensorF32& x) {
    TensorF32 out(x.shape(), x.device());
    const float* xp = x.data();
    float* op = out.data();
    const int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) op[i] = 1.0f / (1.0f + std::exp(-xp[i]));
    return out;
}
// relu：dst = max(x, 0)
TensorF32 relu_value(const TensorF32& x) {
    TensorF32 out(x.shape(), x.device());
    const float* xp = x.data();
    float* op = out.data();
    const int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) op[i] = xp[i] > 0.0f ? xp[i] : 0.0f;
    return out;
}
// own_data_=false 的共享张量，move 赋值先 deallocate() 释放自身数据再接管悬垂指针 → use-after-free。
TensorF32 reshape_value(const TensorF32& x, const Shape& s) {
    if (s.numel() != x.shape().numel())
        throw PPMLError("reshape_value numel mismatch");
    TensorF32 out(s, x.device());
    out.copy_from(x);   // 行优先扁平拷贝，reshape 安全
    return out;
}

// ===== 把值张量包装为图节点 leaf（图模式 forward_graph 用）=====
// 布局约定: 值 row-major (B, ..., C) 最内维 C 连续，与图 ggml dims[0]=C 存储兼容，
// 故可直接把扁平数据拷入图节点。dims 传图维度（dims[0]=最内维），即值维度的逆序。
// 例如: 值 (B,N,L,164) → 图 dims {164, L, N, B}。
TensorF32* wrap_input_as_leaf(const TensorF32& t, const std::vector<int64_t>& dims) {
    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < dims.size() && i < 4; i++) ne[i] = dims[i];
    TensorF32* leaf = context().new_tensor<float>(static_cast<int>(dims.size()), ne);
    float* dst = bind_leaf_data(context(), leaf);
    const float* srcp = (t.device() == Device::CUDA) ? nullptr : t.data();
    if (t.device() == Device::CUDA) {
        TensorF32 tcpu = t.cpu();
        srcp = tcpu.data();
        std::memcpy(dst, tcpu.data(), tcpu.numel() * sizeof(float));
    } else {
        std::memcpy(dst, t.data(), t.numel() * sizeof(float));
    }
    // GRAPH_DEBUG_KERNEL=1：检查输入 leaf 值域，定位哪个输入含巨大值/NaN（程序此前无输入有效检查）
    if (getenv("GRAPH_DEBUG_KERNEL")) {
        const int64_t nelt = t.numel();
        if (srcp && nelt > 0) {
            bool nan = false; float mn = srcp[0], mx = srcp[0]; int64_t nnan = 0;
            int64_t scan = nelt < 4096 ? nelt : 4096;
            for (int64_t i = 0; i < scan; i++) { float v = srcp[i]; if (v != v) { nan = true; nnan++; } else { if (v<mn) mn=v; if (v>mx) mx=v; } }
            if (nelt >= 4096) { int64_t c = 0; for (int64_t i=0;i<nelt;i++) if (srcp[i]!=srcp[i]) c++; if (c){nan=true;nnan=c;} }
            if (nan || mx > 1e6f || mn < -1e6f) {
                fprintf(stderr, "[inleaf] numel=%lld min=%.6g max=%.6g nan=%d nnan=%lld\n",
                        (long long)nelt, (double)mn, (double)mx, nan?1:0, (long long)nnan);
            }
        }
    }
    return leaf;
}

// ===== 图节点回落为值张量（build + compute + 读 data）=====
// 图节点 dims [C, L, N, B]（dims[0]=最内维）扁平数据 == 值张量 (B,N,L,C) 扁平数据，
// 存储兼容，直接 memcpy 到目标值张量。
// CUDA 后端：kernel 异步，node->data() 是 device 指针，须先 synchronize 再经 buffer D2H 读回。
void compute_and_read(TensorF32* node, TensorF32& dst,
                      ComputeGraph* cgraph, Backend* backend,
                      const std::vector<TensorF32*>* keep_nodes = nullptr) {
    if (!node) return;
    // 崩溃定位（GRAPH_DEBUG_DISPATCH=1）：打印每次值回落的调用点（build 前 / graph_compute 前），
    // 区分崩溃在"图构建"还是"kernel 执行"。numel=1 的是 loss 累加等小量。
    if (getenv("GRAPH_DEBUG_DISPATCH")) {
        fprintf(stderr, "[CAR] op=%d numel=%lld -> build_forward_expand\n",
                (int)node->op, (long long)node->numel());
    }
    cgraph->build_forward_expand(node);
    if (getenv("GRAPH_DEBUG_DISPATCH")) {
        fprintf(stderr, "[CAR] op=%d numel=%lld -> graph_compute (n_nodes=%d)\n",
                (int)node->op, (long long)node->numel(), (int)cgraph->n_nodes());
    }
    backend->graph_compute(cgraph);
    if (node->data() != nullptr) {
        size_t bytes = static_cast<size_t>(node->numel()) * sizeof(float);
        if (dst.numel() != (size_t)node->numel()) {
            // 目标值张量为 move-only，无法 operator=；用 placement-new 重建。
            // 值张量布局 (B,...,C) 最内维 C = 图 dims[0]，故值 shape = 图 dims 逆序。
            const Shape& g = node->shape();
            std::vector<int64_t> v;
            for (int i = (int)g.ndim() - 1; i >= 0; --i) v.push_back(g.dims[i]);
            dst.~TensorF32();
            new (&dst) TensorF32(Shape(v), Device::CPU);
        }
        if (dst.numel() == (size_t)node->numel()) {
            const bool on_device = (node->buffer_ != nullptr && !node->buffer_->is_host());
            if (on_device) {
                // CUDA：先同步等待异步 kernel 完成，再经 buffer 做 D2H 拷贝到 host。
                backend->synchronize();
                node->buffer_->get_tensor(node, dst.data(), node->buffer_offs_, bytes);
            } else {
                std::memcpy(dst.data(), node->data(), bytes);
            }
        }
    }
    // ===== 关键：清理子图节点残留的 buffer_/data 引用（2026-08-23）=====
    // SE3 offset 回落（compute_and_read 用独立 se3_backend_）的 graph_compute 结束后，
    // gallocr 已释放该后端 buffer，但中间节点 buffer_/data() 指针未清空 → 悬垂。
    // 主图 build_forward_expand 会复用这些节点（coords_graph 引用链），split_graph 的
    // node_is_host_producer 检查 base->buffer_->is_host() 时对悬垂 GPU buffer 做虚调用 → SIGSEGV。
    // 值已读回 dst，子图不再需要 buffer；主图最终 compute 由 bind_tensor 无条件 rebind，故直接清零安全。
    //   否则 gallocr 会把它们误判为 managed 需分配 → 大量 WARN 且参数数据丢失。
    for (int i = 0; i < cgraph->n_nodes(); i++) {
        TensorF32* nd = cgraph->graph_node(i);
        if (!nd) continue;
        if (nd->flag & TENSOR_FLAG_PARAM) continue;  // 参数保留 data
        // 2026-09-07 方案A：keep_nodes（SE3 输入值 leaf）跳过清理——它们是无上游可重算的
        // 值拷贝常量 leaf，被清 data 后主图 coords_graph 重算 SE3 时数据不可恢复。
        if (keep_nodes) {
            bool hit = false;
            for (const TensorF32* k : *keep_nodes) { if (k == nd) { hit = true; break; } }
            if (hit) continue;
        }
        nd->buffer_      = nullptr;
        nd->buffer_offs_ = 0;
        nd->bind_data(nullptr);  // 清 data_（值已拷回 dst）
    }
}

// ===== 诊断（2026-09-12）：per-block 拓扑 / 坐标 dump =====
//   开关 PPML_DUMP_TOPO_DIR=<dir>：把每个 block 边界 make_graph 的拓扑（edge_index/edge_d）
//   与 apply_coord_update 产出的 coords 落盘 → 离线量化"逐步中间结构 vs 最终结构"的拓扑差
//   （kNN 成员重合率 / edge_d 差异 / coords 漂移）。默认关闭、零开销。
//   文件格式（小端）：
//     topo_<kind>_<seq>.bin : int64[4]={B,L,E,0} + int64[2E] edge_index + float[3E] edge_d
//                             + float[B*L*9] coords（= 生成该拓扑所用的 coords）
//     coord_<seq>.bin       : int64[4]={B,L,3,3} + float[B*L*9]（apply_coord_update 输出）
namespace {
bool dump_write_file(const char* dir, const char* name,
                     const std::vector<unsigned char>& buf) {
    if (!dir || !name) return false;
    std::string path = std::string(dir) + "/" + name;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "[DUMP-ERR] open %s failed\n", path.c_str()); return false; }
    if (!buf.empty()) std::fwrite(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    std::fprintf(stderr, "[DUMP] %s bytes=%zu\n", path.c_str(), buf.size());
    return true;
}

void dump_se3_topo(const char* kind, const se3::GraphData& G, const TensorF32& coords) {
    const char* dir = std::getenv("PPML_DUMP_TOPO_DIR");
    if (!dir || !coords.data() || coords.shape().ndim() < 2) return;
    static int s_iter = 0, s_ref = 0;
    const int seq = (std::strcmp(kind, "iter") == 0) ? s_iter++ : s_ref++;
    const int64_t B = coords.shape().dims[0];
    const int64_t L = coords.shape().dims[1];
    const int64_t E = (G.edge_index.numel() > 0 && G.edge_index.shape().ndim() > 1)
                          ? G.edge_index.shape().dims[1] : 0;
    std::vector<unsigned char> buf;
    const int64_t hdr[4] = {B, L, E, 0};
    auto put = [&](const void* p, size_t n) {
        if (!p || !n) return;
        const unsigned char* q = static_cast<const unsigned char*>(p);
        buf.insert(buf.end(), q, q + n);
    };
    put(hdr, sizeof(hdr));
    if (E > 0) {
        put(G.edge_index.data(), sizeof(int64_t) * 2 * E);
        put(G.edge_d.data(),     sizeof(float)   * 3 * E);
    }
    put(coords.data(), sizeof(float) * B * L * 9);
    char nm[64];
    std::snprintf(nm, sizeof(nm), "topo_%s_%02d.bin", kind, seq);
    dump_write_file(dir, nm, buf);
}

void dump_coords(const TensorF32& xyz) {
    const char* dir = std::getenv("PPML_DUMP_TOPO_DIR");
    if (!dir || !xyz.data() || xyz.numel() <= 0 || xyz.shape().ndim() < 4) return;
    static int s_coord = 0;
    const int seq = s_coord++;
    std::vector<unsigned char> buf;
    const int64_t hdr[4] = {xyz.shape().dims[0], xyz.shape().dims[1],
                            xyz.shape().dims[2], xyz.shape().dims[3]};
    buf.insert(buf.end(), reinterpret_cast<const unsigned char*>(hdr),
               reinterpret_cast<const unsigned char*>(hdr) + sizeof(hdr));
    const unsigned char* d = reinterpret_cast<const unsigned char*>(xyz.data());
    buf.insert(buf.end(), d, d + sizeof(float) * xyz.numel());
    char nm[64];
    std::snprintf(nm, sizeof(nm), "coord_%02d.bin", seq);
    dump_write_file(dir, nm, buf);
}
} // namespace
} // namespace

// ===== SE3 offset scale：可学习全局标量参数 =====
namespace {
    // 基于前期 sweep：0.0003 尖峰、0.003 上行、0.001 最优 → 安全区间留 [1e-4, 5e-3]
    const float kSe3ScaleLo = 1e-4f;
    const float kSe3ScaleHi = 5e-3f;
}

TensorF32* PPMLModel::se3_scale_tensor() {
    // 是否启用学习：多样本训练 或 显式开启
    static const bool kLearnScale =
        (getenv("PPML_MULTI_SAMPLE") && std::string(getenv("PPML_MULTI_SAMPLE")) == "1") ||
        (getenv("PPML_SE3_LEARN_SCALE") && std::string(getenv("PPML_SE3_LEARN_SCALE")) == "1");

    if (!kLearnScale) {
        // 冻结常量：env PPML_SE3_GRAPH_SCALE 覆盖，缺省 1e-3
        static float frozen = 1e-3f;
        if (getenv("PPML_SE3_GRAPH_SCALE")) {
            frozen = std::strtof(getenv("PPML_SE3_GRAPH_SCALE"), nullptr);
        }
        return constant_tensor({1}, &frozen);
    }

    // 懒创建全局共享的 log_scale PARAM（init = log(1e-3)，落在安全区间中部）
    if (!se3_log_scale_param_) {
        int64_t dims1[1] = {1};
        TensorF32* p = context().new_tensor<float>(1, dims1);
        p->op = OP_NONE;
        p->flag = TENSOR_FLAG_PARAM | TENSOR_FLAG_SE3;
        float* pd = bind_leaf_data(context(), p);
        pd[0] = std::log(1e-3f);
        se3_log_scale_param_ = p;
    }
    TensorF32* log_s = se3_log_scale_param_;
    TensorF32* scale  = exp(log_s);                       // 恒正，梯度可回传
    // 可微硬 clamp（relu 实现，clamp/tanh/sigmoid 反向缺失）：
    //   clamped = lo + relu( (hi-lo) - relu(hi - scale) )
    TensorF32* hi_minus_s = sub(constant_tensor({1}, &kSe3ScaleHi), scale);
    TensorF32* inner = relu(hi_minus_s);                  // relu(hi - scale)
    TensorF32* span_minus = sub(constant_tensor({1}, &kSe3ScaleHi), constant_tensor({1}, &kSe3ScaleLo));
    TensorF32* gate = sub(span_minus, inner);             // (hi-lo) - relu(hi - scale)
    TensorF32* relu_gate = relu(gate);
    TensorF32* clamped = add_impl(constant_tensor({1}, &kSe3ScaleLo), relu_gate, false);
    return clamped;
}

void PPMLModel::se3_scale_report() const {
    static const bool kLearnScale =
        (getenv("PPML_MULTI_SAMPLE") && std::string(getenv("PPML_MULTI_SAMPLE")) == "1") ||
        (getenv("PPML_SE3_LEARN_SCALE") && std::string(getenv("PPML_SE3_LEARN_SCALE")) == "1");
    if (!kLearnScale) {
        float frozen = 1e-3f;
        if (getenv("PPML_SE3_GRAPH_SCALE")) frozen = std::strtof(getenv("PPML_SE3_GRAPH_SCALE"), nullptr);
        std::cout << "  [SE3-SCALE] frozen(const)=" << std::fixed << std::setprecision(6) << frozen
                  << std::defaultfloat
                  << " (learn=" << (kLearnScale ? "on" : "off") << ")" << std::endl;
        return;
    }
    if (se3_log_scale_param_ && se3_log_scale_param_->data()) {
        float log_s = se3_log_scale_param_->data()[0];
        float s = std::exp(log_s);
        std::cout << "  [SE3-SCALE] learn=on log_scale=" << std::fixed
                  << std::setprecision(6) << log_s
                  << " -> scale=" << s
                  << " (clamp=[" << kSe3ScaleLo << "," << kSe3ScaleHi << "])"
                  << std::defaultfloat << std::endl;
    } else {
        std::cout << "  [SE3-SCALE] learn=on (param not materialized yet)" << std::endl;
    }
}

PPMLConfig::PPMLConfig() {
    // 默认 SE3 配置
    se3_config.node_dim = D_MSA + D_STATE;  // 288
    se3_config.edge_dim = D_PAIR + 64 + 1;  // 193 (pair + rbf + seqsep)
    se3_config.hidden_dim = 128;
    se3_config.n_layers = 2;
    se3_config.n_heads = 4;
    se3_config.l0_features = {32};   // state 输出
    se3_config.l1_features = {3};    // 坐标更新
    // 度1 输入通道数统一与 l1_features[0]=3 对齐（值版 l1_feats 为 3 通道）；
    // SE3Transformer(SE3Config) 构造器据此构建 fiber_in 度1=3，避免默认 16 与 3 不匹配。
    se3_config.l1_in_feats = 3;
}

// IterBlock 实现
IterBlock::IterBlock(const PPMLConfig& config, bool update_msa_pair)
    : config_(config), update_msa_pair_(update_msa_pair) {
    // 旧值初始化 (保留注释):
    // 所有 LinearLayer/LayerNorm 值成员已移除, 子模块现由 PPMLModel 创建后通过 set_sub_modules() 注入
    // pos_enc_ 由 PPMLModel 构造后通过 set_pos_enc() 注入
}

void IterBlock::proj_state_add_to_query_row(TensorF32& msa, const TensorF32& proj_state) {
    // state -> msa[:,0]
    // msa[:, 0] += proj(state)  (B,L,32) -> (B,L,256)


    // projected state (B, L, 256)
    // query_row += state_proj
    // msa (B, N, L, D) -> N = 0 the target seq to predict
    // query_row (B, L, D)
    int B = msa.shape().dims[0];
    int L = msa.shape().dims[2];
    int D = msa.shape().dims[3];
    
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            for (int d = 0; d < D; d++) {
                // 计算索引
                size_t idx = b * L * D + l * D + d;
                
                // 修改值
                msa.data()[idx] += proj_state.data()[b * L * D_MSA + l * D_MSA + d];
            }
        }
    }
    
}

TensorF32 IterBlock::compute_rbf_feature(const TensorF32& coords)
{
    // Python equivalent:
    // cas = xyz[:, :, 1].contiguous()
    // rbf_feat = rbf(torch.cdist(cas, cas))
    
    // coords: (B, L, A, 3) where A=3 (N, CA, C)
    // Return: (B, L, L, 64) - RBF feature
    
    int B = coords.shape().dims[0];
    int L = coords.shape().dims[1];
    int A = coords.shape().dims[2];  // num atoms
    int D = coords.shape().dims[3];  // 3 for x,y,z
    
    // Step1: cas = coords[:, :, 1] -> (B, L, 3)
    // Select CA atom (index 1 on atom dimension)
    //TensorF32 cas = coords.select(2, 1);  // shape: (B, L, 3)
    
    TensorF32 cas({B, L, 3}, coords.device());
    const float* src = coords.data();
    float* dst = cas.data();

    // Each CA entry is 3 floats (x,y,z), but spaced A*3 = 9 floats apart
    const int64_t src_stride = A * D;  // 9 floats between consecutive residues
    const int64_t dst_stride = D;      // 3 floats between consecutive residues

    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            // Source: at (b, l, 1, 0) - CA atom, x coordinate
            const float* src_ptr = src + b * L * A * D + l * A * D + 1 * D;
            // Destination: at (b, l, 0)
            float* dst_ptr = dst + b * L * D + l * D;
        
            // Copy 3 floats (x, y, z) for this residue
            std::memcpy(dst_ptr, src_ptr, 3 * sizeof(float));
        }
    }

    
    // Ensure contiguous: create new tensor and copy
    TensorF32 cas_copy({B, L, D}, coords.device());
    cas_copy.copy_from(cas);
    
    // Step2: Compute pairwise distances - torch.cdist(cas, cas)
    // Input: (B, L, 3), Output: (B, L, L)
    TensorF32 dists({B, L, L}, coords.device());
    
    const float* cas_data = cas_copy.data();
    float* dists_data = dists.data();
    
    // For each batch
    #pragma omp parallel for
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                // Compute Euclidean distance between cas[b,i,:] and cas[b,j,:]
                float dx = cas_data[b * L * D + i * D + 0] - cas_data[b * L * D + j * D + 0];
                float dy = cas_data[b * L * D + i * D + 1] - cas_data[b * L * D + j * D + 1];
                float dz = cas_data[b * L * D + i * D + 2] - cas_data[b * L * D + j * D + 2];
                float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                dists_data[b * L * L + i * L + j] = dist;
            }
        }
    }
    
    // Step3: RBF expansion
    // rbf(x) = exp(-(x - center_i)^2 / (2*width^2)) for i in num_rbf
    // Standard: 64 RBF centers from 0 to 20 Angstroms
    const int num_rbf = 64;
    const float rbf_min = 0.0f;
    const float rbf_max = 20.0f;
    
    TensorF32 rbf_feature({B, L, L, num_rbf}, coords.device());
    float* rbf_data = rbf_feature.data();
    
    float rbf_step = (rbf_max - rbf_min) / (num_rbf - 1);
    
    #pragma omp parallel for
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                float dist = dists_data[b * L * L + i * L + j];
                for (int k = 0; k < num_rbf; k++) {
                    float center = rbf_min + k * rbf_step;
                    float diff = dist - center;
                    // Gaussian RBF with width = step
                    rbf_data[b * L * L * num_rbf + i * L * num_rbf + j * num_rbf + k] = 
                        std::exp(-diff * diff / (2.0f * rbf_step * rbf_step));
                }
            }
        }
    }
    
    return rbf_feature;
}

TensorF32 IterBlock::compute_l1_features(const TensorF32& coords) {
    // Python: l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
    //         l1_feats = l1_feats.reshape(B*L, -1, 3)
    // coords: (B, L, 3, 3)  — 3 atoms (N, CA, C) × 3 xyz
    // output: (B*L, 3, 3)   — 各原子相对 CA 的位移向量

    int B = coords.shape().dims[0];
    int L = coords.shape().dims[1];
    int A = coords.shape().dims[2];  // 3 atoms
    int D = coords.shape().dims[3];  // 3 xyz

    TensorF32 l1_feats({B * L, A, D}, coords.device());

    const float* src = coords.data();
    float* dst = l1_feats.data();

    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            // CA 坐标 (atom index 1)
            float ca_x = src[b * L * A * D + l * A * D + 1 * D + 0];
            float ca_y = src[b * L * A * D + l * A * D + 1 * D + 1];
            float ca_z = src[b * L * A * D + l * A * D + 1 * D + 2];

            for (int a = 0; a < A; a++) {
                int out_idx = (b * L + l) * A * D + a * D;
                dst[out_idx + 0] = src[b * L * A * D + l * A * D + a * D + 0] - ca_x;
                dst[out_idx + 1] = src[b * L * A * D + l * A * D + a * D + 1] - ca_y;
                dst[out_idx + 2] = src[b * L * A * D + l * A * D + a * D + 2] - ca_z;
            }
        }
    }

    return l1_feats;
}

void IterBlock::forward(TensorF32& msa, TensorF32& pair, 
                        TensorF32& state, 
                        const TensorF32& seq1hot,
                        const TensorF32& coords,
                        const TensorF32& bond_feats,
                        const TensorF32& dist_matrix,
                        const TensorF32& same_chain,
                        const TensorI64& residx) {
    // residx 转 float 供 PositionalEncoding 使用
    TensorF32 residx_f32(residx.shape());
    if (residx.numel() > 0) {
        for (int64_t i = 0; i < residx.numel(); ++i)
            residx_f32.data()[i] = static_cast<float>(residx.data()[i]);
    }
    
    if (update_msa_pair_) {

        // ------------ 1D track update ------------
        // ===== Step 1: msa2msa =====

        // state -> msa[:,0]
        // msa[:, 0] += proj(state)  (B,L,32) -> (B,L,256)
        // CUDA optimize
        {
            //auto query_row = msa.select(1, 0);  // (B, L, 256)
            // query_row += Linear(state) ...

            // the supplemental said that a layernorm and then a linear?
            // layernorm first
            // 旧栈上变量: LayerNorm state_norm(D_STATE); → state2msa_norm_
            // 值版：forward_exec（值 LayerNorm），forward 是图版返回图节点 data()=nullptr
            TensorF32 state_normed = state2msa_norm_->forward_exec(state);
            
            // 旧栈上变量: LinearLayer linear(D_STATE, D_MSA); → state2msa_linear_
            const auto proj_state = state2msa_linear_->forward(state_normed);
            proj_state_add_to_query_row(msa, proj_state);
        }

        TensorF32 rbf_feature;
        {
            // pair2msa: 将 pair 转换为 attention bias 注入 msa row attention
            // pair -> attention bias
        
            TensorF32 pair_biased;
       
            //// (B, L, 3, 3) - 初始 Ca 坐标 (可选)
            // compute the RBF feature to inject into pair bias
            TensorF32 rbf = compute_rbf_feature(coords);
            // 对齐图版：rbf(64) 经 pair2pair_rbf_proj_(64→128) 投影到 D_PAIR 再加到 normed pair。
            // 旧代码 rbf(64)+pos_out(128) 形状错配且 pos_enc 值版返回全 0（图版无 pos_out 路径）。
            rbf_feature = pair2pair_rbf_proj_->forward(rbf);   // (B,L,L,128)
            // 旧栈上变量: LayerNorm pair_layernorm(D_PAIR); → pair2msa_norm_
            pair_biased = pair2msa_norm_->forward_exec(pair);

            pair_biased = add_value(pair_biased, rbf_feature);

            // update msa query row with state from SE3 output
            // state → msa[:,0] 已在 Step 1 (msa2msa) 完成，无需重复

            // a problem: tensor reshape?
            // MSA Row Attention with bias
            msa = msa_row_attn_->forward(msa, pair_biased);
            // RF2 code dropout(row_attn_out, 0.15);
        
            // MSA Column Attention
            msa = msa_col_attn_->forward(msa);
        
            // FeedForward
            msa = msa_ff_->forward(msa);
        }
        // ------------ 1D track update ------------
        
        // to update pair : 2D track update

        // msa2pair: how the pair is updated from the msa?
        // ===== Step 2: msa2pair =====
        // Outer Product Mean
        // msa (B,N,L,256) -> Linear -> (B,N,L,16)
        // outer product + mean -> (B,L,L,256) -> Linear -> (B,L,L,128)
        {
            // msa2pair: einsum('bikd,bjkd->bijd', left, right/N) — 收缩 seq 维 N(dims[1])
            // 值版须用 outer_product_mean（outer_product 签名 (B,1,L,D)×(B,L,1,D) 是纯外积，
            // 与此处的"外积+对 N 求均值"不符，且输入形状 (B,N,L,16) 不匹配 → 修正为 outer_product_mean）。
            TensorF32 msa_normed = msa2pair_norm_->forward_exec(msa);
            TensorF32 left  = msa2pair_left_proj_->forward(msa_normed);   // (B,N,L,16)
            TensorF32 right = msa2pair_right_proj_->forward(msa_normed);  // (B,N,L,16)
            // dst[b,i,j,d] = (1/N)*sum_n left[b,n,i,d]*right[b,n,j,d] → (B,L,L,16)
            TensorF32 pair_update = outer_product_mean(left, right);
            pair_update = msa2pair_out_proj_->forward(pair_update);  // (B,L,L,128)
            pair = add_value(pair, pair_update);  // residual（值版加法）
        }
        
        // Triangle Multiplication
        //pair = pair + drop_row(tri_mul_out_->forward(pair));
        //pair = pair + drop_row(tri_mul_in_->forward(pair));
        Dropout drop_row(1, 0.15);
        auto tri_out = drop_row.forward(tri_mul_out_->forward(pair, true));
        pair = add_value(pair, tri_out);
        auto tri_in = drop_row.forward(tri_mul_in_->forward(pair, false));
        pair = add_value(pair, tri_in);

        // ===== Step 3: pair2pair =====
        // state outer product -> gate
        {
            // rbf_feature 已在 pair2msa 段经 pair2pair_rbf_proj_(64→128) 投影为 (B,L,L,128)，
            // 图版同样复用该投影结果（无二次投影）。
            // 旧栈上变量: LayerNorm state_norm(D_STATE); → pair2pair_state_norm_
            TensorF32 state_normed = pair2pair_state_norm_->forward_exec(state);
            // 旧栈上变量: LinearLayer left_proj(D_STATE, 16); → pair2pair_left_proj_
            // 旧栈上变量: LinearLayer right_proj(D_STATE, 16); → pair2pair_right_proj_
            // different weights for left and right?
            TensorF32 left = pair2pair_left_proj_->forward(state_normed);   // (B,L,16)
            TensorF32 right = pair2pair_right_proj_->forward(state_normed); // (B,L,16)
            // gate 纯外积，特征笛卡尔积 (B,L,L,256)，与 gate_proj 输入 256 匹配
            TensorF32 gate = outer_product_cartesian(left, right);  // (B,L,L,256)
            // 旧栈上变量: LinearLayer gate_proj(16 * 16, D_PAIR); → pair2pair_gate_proj_
            // d_hidden_gate = 16
            gate = pair2pair_gate_proj_->forward(gate);  // (B,L,L,128)
            gate = sigmoid_value(gate);  // (B,L,L,128) -> (B,L,L,128) gate values between 0 and 1
            rbf_feature = mul_value(rbf_feature, gate);  // element-wise gating
            // left = Linear(state, 32->16)
            // right = Linear(state, 32->16)
            // gate = sigmoid(left x right -> Linear -> 128)
            //auto gate = state;  // get_gate
            //gate.copy_from(state);  // 简化，实际需要计算 gate
            // rbf_feat 经 gate 过滤注入 pair
            // pair += gate * rbf_feat
        
            // to update pair
            // Biased Axial Attention (row/col)
            Dropout drop_row(1, 0.15);
            Dropout drop_col(2, 0.15);
            auto row_out = drop_row.forward(pair_row_attn_->forward(pair, rbf_feature));
            pair = add_value(pair, row_out);
            auto col_out = drop_col.forward(pair_col_attn_->forward(pair, rbf_feature));
            pair = add_value(pair, col_out);
            // FeedForward (pair_ff) + residual
            auto pair_ff_out = pair_ff_->forward(pair);
            pair = add_value(pair, pair_ff_out);
        }

    }
    
    // 3D track update
    // ===== Step 4: str2str (SE3 Transformer) =====
    {
        // node features: msa[:,0] + state -> concat -> Linear
        /* auto msa_query = msa.select(1, 0);  // (B, L, 256)
        
        // edge features: pair + rbf + seqsep -> concat -> Linear
        //auto edges = pair;  // concat with rbf, seqsep
        TensorF32 edges;
        edges.copy_from(pair);
        // SE3 Transformer
        auto se3_out = se3_->forward(msa_query, edges, coords);
        
        // 重建 state (完全替换)
        state = se3_out.l0.view({state.shape().dims[0], state.shape().dims[1], D_STATE});
        
        // 更新坐标
        auto updated_coords = struct_update_->update_coords(coords, se3_out.l1);
        
        // 预测侧链扭转角
        auto alpha = struct_update_->predict_torsion(msa_query, state); */

            // 3D track update
    // ===== Step 4: str2str (SE3 Transformer) =====
    // Python:
    //   msa = self.norm_msa(msa)
    //   pair = self.norm_pair(pair)
    //   w_seq = self.encoder_seq(msa).reshape(B,L,1,N).permute(0,3,1,2)
    //   msa = w_seq * msa
    //   msa = msa.sum(dim=1)                          ← sum over sequences
    //   msa = torch.cat((msa, seq1hot), dim=-1)
    //   msa = self.norm_node(self.embed_x(msa))
    //   pair = self.norm_edge(self.embed_e(pair))
    //   G = make_graph(xyz, pair, idx, top_k=top_k)
    //   l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
    //   shift = self.se3(G, msa.reshape(B*L,-1,1), l1_feats)
    //   state = shift['0']; offset = shift['1']
    
        int B = msa.shape().dims[0];
        int N = msa.shape().dims[1];
        int L = msa.shape().dims[2];

        // ---- Step 4a: LayerNorm on msa & pair（值版：forward_exec，forward 是图版返回图节点）----
        TensorF32 msa_normed  = norm_msa_3d_->forward_exec(msa);   // (B, N, L, 256)
        TensorF32 pair_normed = norm_pair_3d_->forward_exec(pair);  // (B, L, L, 128)

        // ---- Step 4b: 序列加权求和 ----
        // encoder_seq: 学习每条序列的权重, shape (N,) → softmax → 加权求和
        // 简化实现: equal-weight mean over N sequences
        TensorF32 msa_sum({B, L, D_MSA}, msa_normed.device());
        float* sum_data = msa_sum.data();
        const float* msa_data = msa_normed.data();
        std::memset(sum_data, 0, B * L * D_MSA * sizeof(float));

        for (int b = 0; b < B; ++b) {
            for (int n = 0; n < N; ++n) {
                for (int l = 0; l < L; ++l) {
                    for (int d = 0; d < D_MSA; ++d) {
                        int64_t src = ((b * N + n) * L + l) * D_MSA + d;
                        int64_t dst = (b * L + l) * D_MSA + d;
                        sum_data[dst] += msa_data[src] / static_cast<float>(N);
                    }
                }
            }
        }

        // ---- Step 4c: cat(msa_sum, seq1hot) → embed → norm（值版：手写 concat）----
        const int64_t NODE_3D_IN = D_MSA + 21;   // 256 + 21 = 277
        TensorF32 node_cat({B, L, NODE_3D_IN}, msa_sum.device());
        float* cat_data = node_cat.data();
        const float* s_data = msa_sum.data();
        const float* onehot_data = seq1hot.data();
        for (int b = 0; b < B; ++b) {
            for (int l = 0; l < L; ++l) {
                for (int d = 0; d < D_MSA; ++d) {
                    cat_data[(b * L + l) * NODE_3D_IN + d] =
                        s_data[(b * L + l) * D_MSA + d];
                }
                for (int d = 0; d < 21; ++d) {
                    cat_data[(b * L + l) * NODE_3D_IN + D_MSA + d] =
                        onehot_data[(b * L + l) * 21 + d];
                }
            }
        }

        // Linear(277 → 32) → LayerNorm（值版 forward_exec）→ (B, L, 32)
        TensorF32 node_emb = embed_x_->forward(node_cat);
        TensorF32 node_out = norm_node_3d_->forward_exec(node_emb);

        // ---- Step 4d: pair embedding ----
        // Linear(128 → 32) → LayerNorm（值版 forward_exec）→ (B, L, L, 32)
        TensorF32 pair_emb = embed_e_->forward(pair_normed);
        TensorF32 edge_out = norm_edge_3d_->forward_exec(pair_emb);

        // ---- Step 4e: 构建图 ----
        se3::GraphData G = se3::make_graph(coords, edge_out, residx, 64, 9);

        // ---- Step 4f: l1 特征 (位移向量) ----
        TensorF32 l1_feats = compute_l1_features(coords);  // (B*L, 3, 3)

        // ---- Step 4g: 组装 SE3Features 输入 ----
        // node_out: (B, L, 32) → reshape to (B*L, 32, 1) 作为 degree-0
        // l1_feats: (B*L, 3, 3) 作为 degree-1
        SE3Features node_se3;
        node_se3.features.resize(2);
        node_se3.features[0] = node_out.view({B * L, ITER_NODE_3D_OUT, 1});
        //    先按 l1_feats 形状重建再拷贝（同 FullBlock 修复）。
        node_se3.features[1].~TensorF32();
        new (&node_se3.features[1]) TensorF32(l1_feats.shape(), l1_feats.device());
        node_se3.features[1].copy_from(l1_feats);

        // ---- Step 4h: 预计算球谐基 ----
        SE3Basis basis;
        //basis.compute(coords, TensorF32() /*orient*/, 2 /*J_max*/);
        basis.compute(G.edge_d, 2);  // J_max=2, 边向量来自 graph


        // ---- Step 4i: SE3 Transformer forward ----
        SE3Features se3_out = se3_->forward(
            node_se3, G.edge_index, G.edge_d, &G.edge_w, basis);

        // ---- Step 4j: 提取输出 ----
        // state: degree-0 → (B*L, D_STATE) → (B, L, D_STATE)
        if (state.numel() != static_cast<int64_t>(B * L * D_STATE)) {
            state.~TensorF32();
            new (&state) TensorF32(Shape({B, L, D_STATE}), se3_out.features[0].device());
        }
        state.copy_from(se3_out.features[0].view({B, L, D_STATE}));

        // offset: degree-1 → (B*L, 3, 3) → (B, L, 3, 3)
        TensorF32 offset = se3_out.features[1].view({B, L, 3, 3});

        // ---- Step 4k: 坐标更新 ----
        // CA_new = xyz[:,:,1] + offset[:,:,1]
        // N_new  = CA_new + offset[:,:,0]
        // C_new  = CA_new + offset[:,:,2]
        // 【2026-09-12 修复】与图版 IterBlock::apply_coord_update 统一：单步位移 = clamp(offset×0.03, ±3Å)。
        //   原因：本处原实现直接加**原始 offset**（无 scale、无 clamp），而随机初始化下 offset 可达
        //   O(1e3~1e8) ⇒ 每 block 坐标漂移数百~数千 Å（实测 CA 均值 747Å、典型边长 1828Å，正常应 ~19Å，
        //   见 Experiment.md §5）。后果：值版 Pass1（两遍管线的旧实现）产出的拓扑基准被打爆。
        //   env PPML_SE3_MAX_STEP 覆盖最大单步（≤0 = 不 clamp，保留旧行为）。
        static constexpr float SE3_OFFSET_SCALE = 0.03f;
        float kMaxStep = 3.0f;
        if (const char* _s = std::getenv("PPML_SE3_MAX_STEP")) kMaxStep = std::atof(_s);
        const bool do_clamp = (kMaxStep > 0.0f);
        auto step_of = [&](float off) -> float {
            float d = off * SE3_OFFSET_SCALE;
            if (do_clamp) { if (d >  kMaxStep) return  kMaxStep; if (d < -kMaxStep) return -kMaxStep; }
            return d;
        };
        const float* xyz_data = coords.data();
        const float* off_data = offset.data();

        TensorF32 xyz_new({B, L, 3, 3}, coords.device());
        float* xyz_out = xyz_new.data();

        for (int b = 0; b < B; ++b) {
            for (int l = 0; l < L; ++l) {
                int base = (b * L + l) * 9;  // 3 atoms × 3 coords = 9

                // 原始 CA 坐标
                float ca_x0 = xyz_data[base + 3];
                float ca_y0 = xyz_data[base + 4];
                float ca_z0 = xyz_data[base + 5];

                // δCA (单步位移：scale + clamp)
                float dca_x = step_of(off_data[base + 3]);
                float dca_y = step_of(off_data[base + 4]);
                float dca_z = step_of(off_data[base + 5]);

                // 更新后 CA
                float ca_x_new = ca_x0 + dca_x;
                float ca_y_new = ca_y0 + dca_y;
                float ca_z_new = ca_z0 + dca_z;

                // N = CA_new + δN
                xyz_out[base + 0] = ca_x_new + step_of(off_data[base + 0]);
                xyz_out[base + 1] = ca_y_new + step_of(off_data[base + 1]);
                xyz_out[base + 2] = ca_z_new + step_of(off_data[base + 2]);

                // CA = CA_new
                xyz_out[base + 3] = ca_x_new;
                xyz_out[base + 4] = ca_y_new;
                xyz_out[base + 5] = ca_z_new;

                // C = CA_new + δC
                xyz_out[base + 6] = ca_x_new + step_of(off_data[base + 6]);
                xyz_out[base + 7] = ca_y_new + step_of(off_data[base + 7]);
                xyz_out[base + 8] = ca_z_new + step_of(off_data[base + 8]);
            }
        }

        //    先按源 shape 重建（同 FullBlock 修复）。
        if (xyz_new_.numel() != xyz_new.numel()) {
            xyz_new_.~TensorF32();
            new (&xyz_new_) TensorF32(xyz_new.shape(), xyz_new.device());
        }
        xyz_new_.copy_from(xyz_new);
    }

    
}

// ===== IterBlock::forward_graph (图模式) =====
// 处理 msa/pair 两条 track (1D/2D 注意力 + FF) + SE3(3D) track。
// 输入/输出均为图节点指针 (ggml 布局 dims[0]=最内维):
//   msa   : 值 (B,N,L,D_MSA) = 图 [D_MSA, L, N, B]
//   pair  : 值 (B,L,L,D_PAIR) = 图 [D_PAIR, L, L, B]
//   rbf   : 值 (B,L,L,D_RBF)  = 图 [D_RBF, L, L, B]  (RBF + pos_enc 注入)
//   state : 值 (B,L,D_STATE)  = 图 [D_STATE, L, B]；SE3 通过引用回写更新
// 返回: 更新后的 pair 图节点；msa、state 通过引用回写。
// 注: 当 coords/seq1hot 提供时，末尾追加 SE3(3D) track（见 run_se3_graph）。
TensorF32* IterBlock::forward_graph(TensorF32*& msa, TensorF32*& pair,
                                    TensorF32* rbf, TensorF32*& state,
                                    const TensorF32* coords,
                                    const TensorI64* residx,
                                    const TensorF32* seq1hot) {
    // ------------ 1D track: msa2msa ------------
    // Step 1: state -> msa[:,0] (query row 注入，图版：掩码广播 add)
    //   proj_state [D_MSA, L, B] = Linear(LayerNorm(state))
    TensorF32* proj_state = state2msa_linear_->forward_graph(
        state2msa_norm_->forward(state));
    msa = query_row_add_graph(msa, proj_state);   // msa[:,0,:,:] += proj_state

    // pair -> pair_biased (msa row attention bias): pair2msa_norm(pair) + emb_rbf(rbf)
    //   RF2AA MSAPairStr2MSA：rbf_feat 先经 emb_rbf (D_RBF=64 → D_PAIR=128) 线性投影到 d_pair，
    //   再加到 layer-normed pair 上作为 row attention bias。
    TensorF32* pair_biased = add_impl(
        pair2msa_norm_->forward(pair),
        pair2pair_rbf_proj_->forward_graph(rbf), /*inplace=*/false);

    // MSA Row Attention (with bias)
    msa = msa_row_attn_->forward_graph(msa, pair_biased);
    // MSA Column Attention
    msa = msa_col_attn_->forward_graph(msa);
    // FeedForward
    msa = msa_ff_->forward_graph(msa);

    // ------------ 2D track: msa2pair (outer-product-mean) ------------
    // einsum('bikd,bjkd->bijd(de)', left, right/N) — 收缩 seq 维 N（dims[2]），特征笛卡尔积 D×D→D*D。
    // msa 图 [D_MSA,L,N,B]：N = dims[2]（seq），L = dims[1]（残基）。
    // 用专用图 op OP_OUTER_PROD_MEAN（out_prod 只收缩 dims[1]，无法收缩 N 维）。
    const int64_t Nseq_msa2pair = msa->shape().dims[2];
    auto msa_normed  = msa2pair_norm_->forward(msa);                 // [D_MSA, L, N, B]
    auto left        = msa2pair_left_proj_->forward_graph(msa_normed);   // [16, L, N, B]
    auto right       = msa2pair_right_proj_->forward_graph(msa_normed);  // [16, L, N, B]
    auto pair_update = outer_product_mean(left, right, static_cast<int>(Nseq_msa2pair));  // [256,L,L,B]
    pair_update      = msa2pair_out_proj_->forward_graph(pair_update);   // [D_PAIR,L,L,B]
    pair             = add_impl(pair, pair_update, /*inplace=*/false);   // residual

    // Triangle Multiplication (out/in) + dropout + residual
    // 图 dropout 用 Dropout::forward_graph（random mask 常量叶子 + mul），值版语义一致。
    {
        Dropout drop_row(1, 0.15);
        TensorF32* tri_out = drop_row.forward_graph(tri_mul_out_->forward_graph(pair, /*bOutgoing=*/true));
        pair = add_impl(pair, tri_out, /*inplace=*/false);
        TensorF32* tri_in = drop_row.forward_graph(tri_mul_in_->forward_graph(pair, /*bOutgoing=*/false));
        pair = add_impl(pair, tri_in, /*inplace=*/false);
    }

    // ------------ pair2pair ------------
    // rbf -> rbf_proj (bias 注入 pair row/col attention)
    TensorF32* rbf_proj = pair2pair_rbf_proj_->forward_graph(rbf);   // [128, L, L, B]
    // gate = sigmoid(gate_proj(outer_product(state)))
    // state 图 [D_STATE, L, B]；gate 用 OP_OUTER_PROD（纯外积，特征笛卡尔积 D*D→256）
    TensorF32* state_normed  = pair2pair_state_norm_->forward(state);           // [D_STATE, L, B]
    TensorF32* gate_left  = pair2pair_left_proj_->forward_graph(state_normed);  // [16, L, B]
    TensorF32* gate_right = pair2pair_right_proj_->forward_graph(state_normed); // [16, L, B]
    TensorF32* gate       = outer_product_graph(gate_left, gate_right);         // [256, L, L, B]
    gate = pair2pair_gate_proj_->forward_graph(gate);                           // [128, L, L, B]
    gate = sigmoid(gate);                                                       // [0,1]
    rbf_proj = mul(rbf_proj, gate);                                             // element-wise gate

    // Biased Axial Attention (row/col) + residual
    pair = add_impl(pair, pair_row_attn_->forward_graph(pair, rbf_proj),
                    /*inplace=*/false);
    pair = add_impl(pair, pair_col_attn_->forward_graph(pair, rbf_proj),
                    /*inplace=*/false);
    // FeedForward (pair_ff) + residual — 值版 forward 同步补齐（见 IterBlock::forward）
    pair = add_impl(pair, pair_ff_->forward_graph(pair), /*inplace=*/false);

    // ------------ 3D track: SE3 Transformer ------------
    // SE3 图块由"训练入口"驱动（图外值回落）：
    //   forward_graph 只构建 msa/pair/rbf 图节点并返回 pair；SE3 的结构常量（make_graph → G、
    //   basis）依赖"更新后 pair 的值"，而 pair 在此为图节点、未 graph_compute，故不能在
    //   forward_graph 内部构造 G/basis。
    //   训练入口在每个 block 边界需：
    //     1) graph_compute(pair) 得到 pair_value；
    //     2) 调用 run_se3_structural(msa, pair, rbf, state, pair_value, coords, residx, seq1hot)，
    //        其内部 Phase A 回落 pair_value → embed_e_ → norm_edge_3d_ → make_graph → basis；
    //        Phase B 调用 run_se3_graph 追加可微 SE3 图节点并把 state 回写为图节点；
    //     3) graph_compute 返回的 offset 图节点（se3_out[1]）后，调用
    //        apply_coord_update(offset_value, coords) 更新骨架坐标 → xyz_new_。
    //   当未提供结构输入（coords/seq1hot==nullptr）时跳过 SE3，等价纯 1D/2D track。
    //(void)coords; (void)residx; (void)seq1hot;

    return pair;
}

// SE3(3D) track 图模式子流程（可微部分用图 op，结构常量由调用方以 G/basis 注入）。
//   node 度0 = norm_node_3d(embed_x(cat(msa 沿 Nseq 维均值, seq1hot)))  → 图节点 [32,B*L]
//   node 度1 = l1_feats (compute_l1_features(coords))                    → 常量叶子
//   edge     = G.edge_index / G.edge_d / G.edge_w → src/tgt/d/w 常量叶子
//   basis    = 调用方预计算（SE3Basis.compute(G.edge_d, 2)）
//   out      = se3_->forward_graph({node0,node1}, src, tgt, d, w, basis, N=B*L)
//   state    = out[0].view({B,L,D_STATE})（引用回写）
// 说明：结构预处理（make_graph 需 edge_out 值张量）属"图外"，由训练入口在 block 边界
//       回落值计算后传入 G/basis/coords；此处仅做可微 node 嵌入 + se3_ 调用。
// 返回 se3_out 图节点：[0]=state(度0), [1]=offset(度1，坐标更新用，训练入口 graph_compute 后
//      回落值 + apply_coord_update 更新骨架坐标)。
std::vector<TensorF32*> IterBlock::run_se3_graph(TensorF32*& msa, TensorF32*& pair, TensorF32* rbf,
                                                 TensorF32*& state,
                                                 const se3::GraphData& G, const SE3Basis& basis,
                                                 const TensorF32& coords, const TensorF32& seq1hot) {
    //(void)pair; (void)rbf;
    const int B = seq1hot.shape().dims[0];
    const int L = seq1hot.shape().dims[1];
    const int64_t N = static_cast<int64_t>(B) * L;   // 图节点数

    // ---- node 度0：msa 沿 Nseq 维均值 → cat(seq1hot) → embed_x_ → norm_node_3d_ ----
    // msa 图节点 [D_MSA, L, Nseq, B]：permute 把 Nseq 移到最内 dims[0] 后 sum_rows 归约
    const int64_t Nseq = msa->shape().dims[2];
    const int64_t msa_feat = msa->shape().dims[0];  // 实际特征维（IterBlock=256, FullBlock=64）
    TensorF32* msa_p = permute(msa, std::vector<int>{2, 0, 1, 3});    // [Nseq, msa_feat, L, B]
    TensorF32* msa_s = sum_rows(msa_p);                              // [1, msa_feat, L, B]
    TensorF32* msa_m = scale(msa_s, 1.0f / static_cast<float>(Nseq)); // 均值
    TensorF32* msa_v = view(msa_m, Shape({msa_feat, L, B}));         // [msa_feat, L, B]
    // FullBlock 的 msa_full 特征维为 D_MSA_FULL=64，而 embed_x_ 为 D_MSA+21=277 维（值版
    // FullBlock::forward 的 SE3 也用 D_MSA=256 累加 msa_sum）。故 64 维 msa 须 pad 到 D_MSA=256，
    // 使 node_cat=[256+21=277, L, B] 与 embed_x_ 输入匹配。IterBlock(256) 无需 pad。
    if (msa_feat < D_MSA) {
        const int64_t pad_len = D_MSA - msa_feat;   // 256-64=192
        std::vector<float> pad_zero(static_cast<size_t>(pad_len) * L * B, 0.0f);
        TensorF32* pad = constant_tensor({pad_len, L, B}, pad_zero.data());
        msa_v = concat_ptr({msa_v, pad}, 0);        // [D_MSA, L, B]
    }
    TensorF32* s1h   = constant_tensor({21, L, B}, seq1hot.data());  // [21, L, B]
    TensorF32* node_cat = concat_ptr({msa_v, s1h}, 0);               // [D_MSA+21=277, L, B]
    TensorF32* node_emb = embed_x_->forward_graph(node_cat);         // [32, L, B]
    TensorF32* node_nrm = norm_node_3d_->forward(node_emb);          // [32, L, B]
    TensorF32* node0 = view(node_nrm, Shape({ITER_NODE_3D_OUT, N})); // [32, B*L]（n=b*L+l）

    // ---- node 度1：l1_feats 常量叶子 [3*d_dim1, B*L]，与值版 node_se3.features[1] 对齐 ----
    // 值版固定度1输入 = l1_feats (B*L, 3, 3)（3 通道位移向量，d_dim1=3）。
    // SE3Transformer(SE3Config) 构造器已把 fiber_in 度1 通道数取 cfg.l1_features[0]=3，
    // 故此处直接填 3 通道的 9 个元素，无需补零，与值版严格一致。
    TensorF32 l1 = compute_l1_features(coords);                      // (B*L, 3, 3)
    const int m1     = 3;                                             // 度1 通道数（值版 l1_feats 固定 3）
    const int d_dim1 = 3;                                             // 度1 → 2*1+1
    //    ggml dims[0]=最内维 → 图节点 [m1*d_dim1, N] 数据序应为 [节点][特征]（节点外层）。
    //    kernel_mul_mat 读 a[i*K+k]（i=节点行, k=特征K）要求 [节点][特征] 序。
    //    原序导致 G1x1SE3 度1 输出错位放大 ~26 倍（l1_feats ~100 → ±2680，值版仅 3.75）。
    //    修复后 l1data[n*(m1*d_dim1) + (a*d_dim1+c)] 与值版 (N, m, d_dim)（n 外层, Wigner 最内）一致。
    std::vector<float> l1data(static_cast<size_t>(m1) * d_dim1 * N, 0.0f);
    for (int64_t n = 0; n < N; ++n)
        for (int a = 0; a < m1; ++a)
            for (int c = 0; c < d_dim1; ++c)
                l1data[static_cast<size_t>(n) * (m1 * d_dim1) + (a * d_dim1 + c)] =
                    l1.data()[n * 9 + a * 3 + c];
    TensorF32* node1 = constant_tensor({m1 * d_dim1, N}, l1data.data());

    // ---- 边特征常量叶子（由调用方值版 make_graph 注入）----
    const int64_t E = G.edge_index.numel() > 0 ? G.edge_index.shape().dims[1] : 0;
    if (E <= 0) {
        // 无有效边图（结构常量未注入），SE3 图块无法执行；仅回写 state=输入（等价跳过）。
        return {};
    }
    std::vector<float> src_d(static_cast<size_t>(E)), tgt_d(static_cast<size_t>(E));
    for (int64_t e = 0; e < E; ++e) {
        src_d[e] = static_cast<float>(G.edge_index.data()[e]);
        tgt_d[e] = static_cast<float>(G.edge_index.data()[E + e]);
    }
    TensorF32* edge_src = constant_tensor({E}, src_d.data());
    TensorF32* edge_tgt = constant_tensor({E}, tgt_d.data());
    // edge_d: (E,3) → [3,E]（结构常量，用 host coords 值版算）
    std::vector<float> dd(static_cast<size_t>(3 * E));
    for (int64_t e = 0; e < E; ++e)
        for (int c = 0; c < 3; ++c) dd[static_cast<size_t>(c) * E + e] = G.edge_d.data()[e * 3 + c];
    TensorF32* edge_d = constant_tensor({3, E}, dd.data());
    // ===== edge_w 图化（同一 autograd 图）=====
    // 【阶段1.5 重构】edge_w 不再用值版回落（pair_value→embed_e_→make_graph 常量注入），
    // 改为从主图 pair 图节点直接图化提取：pair [D_PAIR,L,L,B] → norm_pair_3d_ → embed_e_ →
    // norm_edge_3d_ → [ITER_EDGE_3D_OUT,L,L,B] → view [ITER_EDGE_3D_OUT, L*L*B] →
    // edge_gather_rows(edge_feat_flat, edge_pair_idx) → [ITER_EDGE_3D_OUT, E]。
    // 这样 pair→edge_w 全程可微（pair 梯度回传 SE3），且彻底消除 compute_and_read(pair_value) 回落。
    // edge_pair_idx[e] = b*L*L + i*L + j（b=src/L, i=src%L, j=tgt%L，与 make_graph 的
    // pair[b,i,j,:] 索引 (b*L+i)*L+j 一致）。
    const int64_t LLB   = static_cast<int64_t>(L) * L * B;     // 节点空间 L*L*B
    TensorF32* edge_feat_g = norm_edge_3d_->forward(
        embed_e_->forward_graph(norm_pair_3d_->forward(pair))); // [ITER_EDGE_3D_OUT,L,L,B]
    TensorF32* edge_feat_flat = view(edge_feat_g,
        Shape({edge_feat_g->shape().dims[0], LLB}));            // [ITER_EDGE_3D_OUT, L*L*B]
    // 边索引常量（host 值：b*L*L + src%L*L + tgt%L）
    std::vector<float> pair_idx(E);
    for (int64_t e = 0; e < E; ++e) {
        const int64_t src = static_cast<int64_t>(G.edge_index.data()[e]);       // b*L+i
        const int64_t tgt = static_cast<int64_t>(G.edge_index.data()[E + e]);   // b*L+j
        const int64_t b = src / L, i = src % L, j = tgt % L;
        pair_idx[e] = static_cast<float>(b * L * L + i * L + j);
    }
    TensorF32* edge_idx_leaf = constant_tensor({E}, pair_idx.data());
    TensorF32* edge_w = edge_gather_rows(edge_feat_flat, edge_idx_leaf);  // [E_dim, E] 图节点

    // ---- SE3 Transformer forward_graph ----
    std::vector<TensorF32*> h_nodes = {node0, node1};
    std::vector<TensorF32*> se3_out = se3_->forward_graph(
        h_nodes, edge_src, edge_tgt, edge_d, edge_w, basis, static_cast<int>(N));

    // ---- state 回写：度0 → (B,L,D_STATE) 图 [D_STATE, L, B] ----
    // se3_out[0] 为 [32, B*L]（度0 输出），节点序 n=b*L+l → view [D_STATE, L, B]
    state = view(se3_out[0], Shape({D_STATE, L, B}));

    // 返回 se3_out：se3_out[1] 为度1 offset 图节点 [3*3, B*L]（坐标更新需"图外"回落其值）。
    // 训练入口在此后 graph_compute，再用 apply_coord_update 把 offset 值叠加到 coords → xyz_new_。
    return se3_out;
}

// ===== 训练入口驱动：SE3 图块（同一 autograd 图，无回落）=====
// Phase A（拓扑结构常量）：make_graph(coords, 空pair, residx) → G（edge_index/edge_d 仅依赖
//   host coords 值，天然有值，不回落）。basis.compute(G.edge_d, 2)。此阶段非可微（离散拓扑）。
// Phase B（可微图块）：调用 run_se3_graph——edge_w 从主图 pair 图节点图化提取
//   （norm_pair_3d_/embed_e_/norm_edge_3d_/edge_gather_rows），与 SE3 前向同一 autograd 图。
//   state 经引用回写为图节点（度0）。返回的 offset 图节点由训练入口 graph_compute 后做坐标更新。
// 注：pair_value 参数已移除——不再需要图外回落（跨 cgraph 冲突根源）。
std::vector<TensorF32*> IterBlock::run_se3_structural(TensorF32*& msa, TensorF32*& pair, TensorF32* rbf,
                                                      TensorF32*& state,
                                                      const TensorF32& coords,
                                                      const TensorI64& residx,
                                                      const TensorF32& seq1hot,
                                                      const se3::TopoRefineCfg*   topo_cfg,
                                                      se3::TopoRefineState*       topo_st) {
    // ---- Phase A: make_graph 拓扑（edge_index/edge_d 只依赖 host coords）----
    // 空 pair（numel=0）传给 make_graph：edge_w 留空（run_se3_graph 内图化），拓扑照常。
    TensorF32 empty_pair;   // 默认构造 numel=1 data=nullptr → has_pair=false
    // 【refined topo pass L1/L2】cfg/st 非空 ⇒ 冻结边索引（L1）+ margin 门控局部重算（L2）✓；
    //   两者都为空 ⇒ 与原来逐位一致 ✓（见 SE3Transformer.h 的契约 ✓）
    se3::GraphData G = (topo_cfg && topo_st)
        ? se3::make_graph_refined(coords, empty_pair, residx, 64, 9, *topo_cfg, *topo_st)
        : se3::make_graph(coords, empty_pair, residx, 64, 9);
    SE3Basis basis;
    basis.compute(G.edge_d, 2);
    // 无有效边图（拓扑空）时跳过 SE3
    if (G.edge_index.numel() <= 0) {
        if (getenv("GRAPH_DEBUG_COORD")) {
            std::fprintf(stderr, "[SE3-SKIP] run_se3_structural: empty edge_index, skip SE3\n");
        }
        return {};
    }
    // [诊断 2026-09-12] dump 本 block 实际使用的拓扑（edge_index/edge_d + 生成它的 coords）
    dump_se3_topo("iter", G, coords);

    // ---- Phase B: run_se3_graph（可微图块，edge_w 图化，state 回写）----
    // 返回 se3_out：se3_out[1]（offset 图节点）由训练入口 graph_compute 后调用 apply_coord_update。
    return run_se3_graph(msa, pair, rbf, state, G, basis, coords, seq1hot);
}

// ===== 坐标更新（图外值回落）=====
// offset_value: 度1 SE3 输出，值布局 (B*L, 3, 3)（[N,CA,C] 相对 CA 位移；CA 通道为绝对位移）。
// 与值版 Step4k 一致：CA_new = coords_CA + offset[:,:,1]；N_new = CA_new + offset[:,:,0]；
// C_new = CA_new + offset[:,:,2]。结果写 xyz_new_。
void IterBlock::apply_coord_update(const TensorF32& offset_value, const TensorF32& coords) {
    const int B = coords.shape().dims[0];
    const int L = coords.shape().dims[1];
    if (getenv("GRAPH_DEBUG_COORD")) {
        std::fprintf(stderr, "[COORD] this=%p coords={%lld,%lld,%lld,%lld} numel=%lld offset={%lld,%lld,%lld,%lld} numel=%lld xyz_new_={%lld,%lld,%lld,%lld} numel=%lld\n",
            (void*)this,
            (long long)(coords.shape().dims.size()>0?coords.shape().dims[0]:-1),
            (long long)(coords.shape().dims.size()>1?coords.shape().dims[1]:-1),
            (long long)(coords.shape().dims.size()>2?coords.shape().dims[2]:-1),
            (long long)(coords.shape().dims.size()>3?coords.shape().dims[3]:-1),
            (long long)coords.numel(),
            (long long)(offset_value.shape().dims.size()>0?offset_value.shape().dims[0]:-1),
            (long long)(offset_value.shape().dims.size()>1?offset_value.shape().dims[1]:-1),
            (long long)(offset_value.shape().dims.size()>2?offset_value.shape().dims[2]:-1),
            (long long)(offset_value.shape().dims.size()>3?offset_value.shape().dims[3]:-1),
            (long long)offset_value.numel(),
            (long long)(xyz_new_.shape().dims.size()>0?xyz_new_.shape().dims[0]:-1),
            (long long)(xyz_new_.shape().dims.size()>1?xyz_new_.shape().dims[1]:-1),
            (long long)(xyz_new_.shape().dims.size()>2?xyz_new_.shape().dims[2]:-1),
            (long long)(xyz_new_.shape().dims.size()>3?xyz_new_.shape().dims[3]:-1),
            (long long)xyz_new_.numel());
    }
    const float* xyz_data = coords.data();
    const float* off_data = offset_value.data();

    TensorF32 xyz_new({B, L, 3, 3}, coords.device());
    float* xyz_out = xyz_new.data();
    if (getenv("GRAPH_DEBUG_COORD")) {
        std::fprintf(stderr, "[COORD-PTR] coords.data=%p off.data=%p xyz_out=%p B=%d L=%d req=%d\n",
            (void*)xyz_data, (void*)off_data, (void*)xyz_out, B, L, (int)(B*L*9));
    }
    // 防护：offset/coords data 无效（graph_compute 未算 offset 值）时不能做坐标更新
    if (!xyz_data || !off_data || !xyz_out) {
        std::fprintf(stderr, "[COORD-WARN] null data (xyz=%p off=%p out=%p), skip coord update\n",
            (void*)xyz_data, (void*)off_data, (void*)xyz_out);
        return;
    }
    // 坐标推到 ±3.5e10 → FAPE 的 sqr 溢出 → [FWD-NAN] op=9 → fape/conf(lddt 距离) NaN。
    // 参考 RF2AA 对 translation 输出的约束，这里乘 scale 限制单步位移量级（训练早期尤为关键）。
    // 2026-08-22: 0.1 下 chi head 波动剧烈（grad 经 coords→offset→SE3 链放大，[GRAD-NORM]
    // rank=0 SE3 权重 l2=1e13），loss 不收敛；改用 0.03 减小 SE3 对坐标/state 的更新步长。
    // 2026-09-07 RFAA 式单步位移 clamp：SE3 随机初始化下 offset 偶发 O(1e3~1e8)，per_block
    //   每 block 用漂移 coords 重构图 → 正反馈放大（coords 巨大/NaN）。RFAA 原版显式
    //   T=offset/10、R=offset/100 限制步长；此处等价限制"应用后的单步位移"：
    //     d = clamp(offset * SE3_OFFSET_SCALE, ±kMaxStep)
    //   使每 block 位移 ≤ kMaxStep Å（默认 3Å），打破 offset→coords→构图→offset 反馈放大。
    //   env PPML_SE3_MAX_STEP 覆盖；设 ≤0 表示不 clamp（保留原行为）。
    static constexpr float SE3_OFFSET_SCALE = 0.03f;
    float kMaxStep = 3.0f;
    if (const char* _s = std::getenv("PPML_SE3_MAX_STEP")) kMaxStep = std::atof(_s);
    const bool do_clamp = (kMaxStep > 0.0f);
    // 应用单步位移（offset×scale，可选 clamp）
    auto step_of = [&](float off) -> float {
        float d = off * SE3_OFFSET_SCALE;
        if (do_clamp) { if (d >  kMaxStep) return  kMaxStep; if (d < -kMaxStep) return -kMaxStep; }
        return d;
    };
    for (int b = 0; b < B; ++b) {
        for (int l = 0; l < L; ++l) {
            const int base = (b * L + l) * 9;
            const float ca_x0 = xyz_data[base + 3];
            const float ca_y0 = xyz_data[base + 4];
            const float ca_z0 = xyz_data[base + 5];
            const float dca_x = step_of(off_data[base + 3]);
            const float dca_y = step_of(off_data[base + 4]);
            const float dca_z = step_of(off_data[base + 5]);
            const float ca_x_new = ca_x0 + dca_x;
            const float ca_y_new = ca_y0 + dca_y;
            const float ca_z_new = ca_z0 + dca_z;
            xyz_out[base + 0] = ca_x_new + step_of(off_data[base + 0]);
            xyz_out[base + 1] = ca_y_new + step_of(off_data[base + 1]);
            xyz_out[base + 2] = ca_z_new + step_of(off_data[base + 2]);
            xyz_out[base + 3] = ca_x_new;
            xyz_out[base + 4] = ca_y_new;
            xyz_out[base + 5] = ca_z_new;
            xyz_out[base + 6] = ca_x_new + step_of(off_data[base + 6]);
            xyz_out[base + 7] = ca_y_new + step_of(off_data[base + 7]);
            xyz_out[base + 8] = ca_z_new + step_of(off_data[base + 8]);
        }
    }
    // xyz_new_ 是成员，首次调用时为空（numel=0）；copy_from 要求 numel 严格相等，
    // 故 numel 不等时用 placement-new 重建（Tensor 为 move-only，不能 operator=）。
    if (xyz_new_.numel() != xyz_new.numel()) {
        xyz_new_.~TensorF32();
        new (&xyz_new_) TensorF32(xyz_new.shape(), xyz_new.device());
    }
    xyz_new_.copy_from(xyz_new);
    // [诊断 2026-09-12] dump 更新后的 coords（供 per-block 拓扑/结构漂移量化）
    dump_coords(xyz_new_);
}

void FullBlock::forward(TensorF32& msa_full, TensorF32& pair, TensorF32& state, 
                        const TensorF32& seq1hot,
                        const TensorF32& coords,
                        const TensorF32& bond_feats,
                        const TensorF32& dist_matrix,
                        const TensorF32& same_chain,
                        const TensorI64& residx) {
    // residx 转 float 供 PositionalEncoding 使用
    TensorF32 residx_f32(residx.shape());
    if (residx.numel() > 0) {
        for (int64_t i = 0; i < residx.numel(); ++i)
            residx_f32.data()[i] = static_cast<float>(residx.data()[i]);
    }
    // FullBlock 在 IterBlock 的基础上增加了 msa_full 的使用和全局 column attention
    // msa_full 需要在 forward 函数参数中传入，或者在 IterBlock 中存储为成员变量
    
    // ------------ 1D track update ------------
    // ===== Step 1: msa2msa =====
    {
        //auto query_row = msa.select(1, 0);  // (B, L, 256)
        // query_row += Linear(state) ...

        // 旧栈上变量: LayerNorm state_norm(D_STATE); → state2msa_norm_
        // 值版：forward_exec（值 LayerNorm），forward 是图版返回图节点 data()=nullptr
        TensorF32 state_normed = state2msa_norm_->forward_exec(state);
            
        // 旧栈上变量: LinearLayer linear(D_STATE, D_MSA); → state2msa_linear_
        const auto proj_state = state2msa_linear_->forward(state_normed);
        proj_state_add_to_query_row(msa_full, proj_state);
    }
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[FBLK] msa2msa OK (msa_full=%lld)\n", (long long)msa_full.numel());

    TensorF32 rbf_feature;
    {
        // pair2msa: 将 pair 转换为 attention bias 注入 msa row attention
        // pair -> attention bias
        
        TensorF32 pair_biased;
       
        //// (B, L, 3, 3) - 初始 Ca 坐标 (可选)
        // compute the RBF feature to inject into pair bias
        TensorF32 rbf = compute_rbf_feature(coords);
        // 对齐图版：rbf(64) 经 pair2pair_rbf_proj_(64→128) 投影到 D_PAIR 再加到 normed pair。
        // 旧代码 rbf(64)+pos_out(128) 形状错配且 pos_enc 值版返回全 0（图版无 pos_out 路径）。
        rbf_feature = pair2pair_rbf_proj_->forward(rbf);   // (B,L,L,128)
        // 旧栈上变量: LayerNorm pair_layernorm(D_PAIR); → pair2msa_norm_
        pair_biased = pair2msa_norm_->forward_exec(pair);

        pair_biased = add_value(pair_biased, rbf_feature);

        // update msa query row with state from SE3 output
        // DONE already update in the msa2msa

        // a problem: tensor reshape?
        // MSA Row Attention with bias
        msa_full = msa_row_attn_->forward(msa_full, pair_biased);
        // RF2 code dropout(row_attn_out, 0.15);
        
        // MSA Column Attention
        // global attention
        msa_full = msa_global_col_attn_->forward(msa_full);
        
        // FeedForward
        msa_full = msa_ff_->forward(msa_full);
    }
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[FBLK] msa row/col/global/ff OK (msa_full=%lld)\n", (long long)msa_full.numel());
    // ------------ 1D track update ------------

    // msa2pair: einsum('bikd,bjkd->bijd', left, right/N) — 收缩 seq 维 N(dims[1])
    // 值版用 outer_product_mean（原 outer_product 签名 (B,1,L,D)×(B,L,1,D) 不符且为纯外积）。
    {
        TensorF32 msa_normed = msa2pair_norm_->forward_exec(msa_full);
        TensorF32 left  = msa2pair_left_proj_->forward(msa_normed);   // (B,N,L,16)
        TensorF32 right = msa2pair_right_proj_->forward(msa_normed);  // (B,N,L,16)
        TensorF32 pair_update = outer_product_mean(left, right);      // (B,L,L,16)
        pair_update = msa2pair_out_proj_->forward(pair_update);       // (B,L,L,128)
        pair = add_value(pair, pair_update);  // residual（值版加法）
    }
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[FBLK] msa2pair OK (pair=%lld)\n", (long long)pair.numel());

    // Triangle Multiplication
    Dropout drop_row(1, 0.15);
    auto tri_out = drop_row.forward(tri_mul_out_->forward(pair, true));
    pair = add_value(pair, tri_out);
    auto tri_in = drop_row.forward(tri_mul_in_->forward(pair, false));
    pair = add_value(pair, tri_in);
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[FBLK] tri_mul OK\n");

    // ===== Step 3: pair2pair =====
    // state outer product -> gate
    {
        // rbf_feature 已在 pair2msa 段经 pair2pair_rbf_proj_(64→128) 投影为 (B,L,L,128)，
        // 图版同样复用该投影结果（无二次投影）。
        // 旧栈上变量: LayerNorm state_norm(D_STATE); → pair2pair_state_norm_
        TensorF32 state_normed = pair2pair_state_norm_->forward_exec(state);
        // 旧栈上变量: LinearLayer left_proj(D_STATE, 16); → pair2pair_left_proj_
        // 旧栈上变量: LinearLayer right_proj(D_STATE, 16); → pair2pair_right_proj_
        // different weights for left and right?
        TensorF32 left = pair2pair_left_proj_->forward(state_normed);   // (B,L,16)
        TensorF32 right = pair2pair_right_proj_->forward(state_normed); // (B,L,16)
        // gate 纯外积，特征笛卡尔积 (B,L,L,256)，与 gate_proj 输入 256 匹配
        TensorF32 gate = outer_product_cartesian(left, right);  // (B,L,L,256)
        // 旧栈上变量: LinearLayer gate_proj(16 * 16, D_PAIR); → pair2pair_gate_proj_
        // d_hidden_gate = 16
        gate = pair2pair_gate_proj_->forward(gate);  // (B,L,L,128)
        gate = sigmoid_value(gate);  // (B,L,L,128) -> (B,L,L,128) gate values between 0 and 1
        rbf_feature = mul_value(rbf_feature, gate);  // element-wise gating
            
        Dropout drop_row2(1, 0.15);
        Dropout drop_col(2, 0.15);
        if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[FBLK] pair2pair gate OK\n");
        auto row_out2 = drop_row2.forward(pair_row_attn_->forward(pair, rbf_feature));
        pair = add_value(pair, row_out2);
        if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[FBLK] pair row attn OK\n");
        auto col_out2 = drop_col.forward(pair_col_attn_->forward(pair, rbf_feature));
        pair = add_value(pair, col_out2);
        if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[FBLK] pair col attn OK\n");
        // FeedForward (pair_ff) + residual
        auto pair_ff_out = pair_ff_->forward(pair);
        pair = add_value(pair, pair_ff_out);
    }
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[FBLK] pair2pair OK\n");

    // 3D track update — same logic as IterBlock, but uses msa_full
    // ===== Step 4: str2str (SE3 Transformer) =====
    {
        int B = msa_full.shape().dims[0];
        int N = msa_full.shape().dims[1];
        int L = msa_full.shape().dims[2];

        // 值版 SE3：用 forward_exec（值 LayerNorm），不能用图版 forward
        auto msa_normed  = norm_msa_3d_->forward_exec(msa_full);
        auto pair_normed = norm_pair_3d_->forward_exec(pair);

        // 序列加权求和 (simplified: equal-weight mean)
        // 注意：FullBlock 的 msa_full 特征维是 D_MSA_FULL=64（非 D_MSA=256），
        // 按实际特征维累加，避免越界读；embed_x_ 输入需 D_MSA+21=277，下方 concat 补零 pad。
        const int64_t msa_feat = msa_full.shape().dims[3];
        TensorF32 msa_sum({B, L, msa_feat}, msa_normed.device());
        float* sum_data = msa_sum.data();
        const float* msa_data = msa_normed.data();
        std::memset(sum_data, 0, B * L * msa_feat * sizeof(float));
        for (int b = 0; b < B; ++b)
            for (int n = 0; n < N; ++n)
                for (int l = 0; l < L; ++l)
                    for (int d = 0; d < msa_feat; ++d)
                        sum_data[(b * L + l) * msa_feat + d] +=
                            msa_data[((b * N + n) * L + l) * msa_feat + d] / float(N);

        // cat + embed（值版：手写 concat，concat_ptr 是图版返回图节点 data()=nullptr）
        const int64_t NODE_3D_IN = D_MSA + 21;   // 256 + 21 = 277
        TensorF32 node_cat({B, L, NODE_3D_IN}, msa_sum.device());
        float* cat_data = node_cat.data();
        const float* s_data = msa_sum.data();
        const float* onehot_data = seq1hot.data();
        for (int b = 0; b < B; ++b) {
            for (int l = 0; l < L; ++l) {
                int base_cat = (b * L + l) * NODE_3D_IN;
                for (int d = 0; d < msa_feat; ++d)
                    cat_data[base_cat + d] = s_data[(b * L + l) * msa_feat + d];
                for (int d = msa_feat; d < D_MSA; ++d)
                    cat_data[base_cat + d] = 0.0f;   // 64→256 pad
                for (int d = 0; d < 21; ++d)
                    cat_data[base_cat + D_MSA + d] = onehot_data[(b * L + l) * 21 + d];
            }
        }

        auto node_emb = embed_x_->forward(node_cat);
        auto node_out = norm_node_3d_->forward_exec(node_emb);
        auto edge_emb = embed_e_->forward(pair_normed);
        auto edge_out = norm_edge_3d_->forward_exec(edge_emb);
        if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[FBLK] node/edge embed OK (node=%lld edge=%lld)\n",
            (long long)node_out.numel(), (long long)edge_out.numel());

        se3::GraphData G = se3::make_graph(coords, edge_out, residx, 64, 9);
        TensorF32 l1_feats = compute_l1_features(coords);
        if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[FBLK] make_graph OK (E=%lld)\n", (long long)G.edge_index.numel());

        //Fiber fiber_in({NODE_3D_OUT, 3}, {0, 1});
        SE3Features node_se3;
        node_se3.features.resize(2);
        // node_out = it was actually msa input
        node_se3.features[0] = node_out.view({B * L, ITER_NODE_3D_OUT, 1});
        //    先按 l1_feats 形状重建再拷贝（值版 SE3Features 是值类型成员）。
        node_se3.features[1].~TensorF32();
        new (&node_se3.features[1]) TensorF32(l1_feats.shape(), l1_feats.device());
        node_se3.features[1].copy_from(l1_feats);

        SE3Basis basis;
        //basis.compute(coords, TensorF32(), 2);
        basis.compute(G.edge_d, 2);  // J_max=2, 边向量来自 graph
        if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[FBLK] basis OK\n");

        SE3Features se3_out = se3_->forward(
            node_se3, G.edge_index, G.edge_d, &G.edge_w, basis);
        if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[FBLK] se3 forward OK (f0=%lld f1=%lld)\n",
            (long long)se3_out.features[0].numel(), (long long)se3_out.features[1].numel());

        //    state 是外部引用，move 赋值释放其旧数据；若旧数据地址恰好被 se3_out
        //    复用或 se3_out 析构释放 view 指向的 data，后续 state 读悬垂/析构 double-free。
        //    用深拷贝（先重建再 copy_from）使 state/offset 独立于 se3_out。
        if (state.numel() != static_cast<int64_t>(B * L * D_STATE)) {
            state.~TensorF32();
            new (&state) TensorF32(Shape({B, L, D_STATE}), se3_out.features[0].device());
        }
        state.copy_from(se3_out.features[0].view({B, L, D_STATE}));
        TensorF32 offset = se3_out.features[1].view({B, L, 3, 3});

        // Coordinate update (same as IterBlock)
        // 【2026-09-12 修复】同 IterBlock 值版：单步位移统一为 clamp(offset×0.03, ±3Å)
        //   （原实现直接加原始 offset → 每 block 数百~数千 Å 漂移，见 Experiment.md §5）。
        static constexpr float SE3_OFFSET_SCALE = 0.03f;
        float kMaxStep = 3.0f;
        if (const char* _s = std::getenv("PPML_SE3_MAX_STEP")) kMaxStep = std::atof(_s);
        const bool do_clamp = (kMaxStep > 0.0f);
        auto step_of = [&](float off) -> float {
            float d = off * SE3_OFFSET_SCALE;
            if (do_clamp) { if (d >  kMaxStep) return  kMaxStep; if (d < -kMaxStep) return -kMaxStep; }
            return d;
        };
        const float* xyz_data = coords.data();
        const float* off_data = offset.data();

        TensorF32 xyz_new({B, L, 3, 3}, coords.device());
        float* xyz_out = xyz_new.data();
        for (int b = 0; b < B; ++b) {
            for (int l = 0; l < L; ++l) {
                int base = (b * L + l) * 9;

                float ca_x0 = xyz_data[base + 3];
                float ca_y0 = xyz_data[base + 4];
                float ca_z0 = xyz_data[base + 5];

                // δCA (单步位移：scale + clamp)
                float dca_x = step_of(off_data[base + 3]);
                float dca_y = step_of(off_data[base + 4]);
                float dca_z = step_of(off_data[base + 5]);

                // 更新后 CA
                float ca_x_new = ca_x0 + dca_x;
                float ca_y_new = ca_y0 + dca_y;
                float ca_z_new = ca_z0 + dca_z;

                // N = CA_new + δN
                xyz_out[base + 0] = ca_x_new + step_of(off_data[base + 0]);
                xyz_out[base + 1] = ca_y_new + step_of(off_data[base + 1]);
                xyz_out[base + 2] = ca_z_new + step_of(off_data[base + 2]);

                // CA = CA_new
                xyz_out[base + 3] = ca_x_new;
                xyz_out[base + 4] = ca_y_new;
                xyz_out[base + 5] = ca_z_new;

                // C = CA_new + δC
                xyz_out[base + 6] = ca_x_new + step_of(off_data[base + 6]);
                xyz_out[base + 7] = ca_y_new + step_of(off_data[base + 7]);
                xyz_out[base + 8] = ca_z_new + step_of(off_data[base + 8]);
            }
        }

        //    先按源 shape 重建（同 IterBlock::forward 的 placement-new 模式）。
        if (xyz_new_.numel() != xyz_new.numel()) {
            xyz_new_.~TensorF32();
            new (&xyz_new_) TensorF32(xyz_new.shape(), xyz_new.device());
        }
        xyz_new_.copy_from(xyz_new);
    }

}

// ===== FullBlock::forward_graph (图模式) =====
// 与 IterBlock::forward_graph 一致，但 msa_full 用 global column attention。
// 布局约定同 IterBlock::forward_graph；SE3 track 由末尾追加（同 IterBlock，见 run_se3_graph）。
TensorF32* FullBlock::forward_graph(TensorF32*& msa_full, TensorF32*& pair,
                                    TensorF32* rbf, TensorF32*& state,
                                    const TensorF32* coords,
                                    const TensorI64* residx,
                                    const TensorF32* seq1hot) {
    // ------------ 1D track: msa2msa ------------
    // Step 1: state -> msa_full[:,0] (query row 注入，图版：掩码广播 add)
    //   proj_state [D_MSA, L, B] = Linear(LayerNorm(state))
    TensorF32* proj_state = state2msa_linear_->forward_graph(
        state2msa_norm_->forward(state));
    msa_full = query_row_add_graph(msa_full, proj_state);   // msa_full[:,0,:,:] += proj_state

    // pair -> pair_biased (msa row attention bias): pair2msa_norm(pair) + emb_rbf(rbf)
    //   RF2AA MSAPairStr2MSA：rbf_feat 先经 emb_rbf (D_RBF=64 → D_PAIR=128) 线性投影到 d_pair。
    TensorF32* pair_biased = add_impl(
        pair2msa_norm_->forward(pair),
        pair2pair_rbf_proj_->forward_graph(rbf), /*inplace=*/false);

    // MSA Row Attention (with bias)
    msa_full = msa_row_attn_->forward_graph(msa_full, pair_biased);
    // MSA Global Column Attention
    msa_full = msa_global_col_attn_->forward_graph(msa_full);
    // FeedForward
    msa_full = msa_ff_->forward_graph(msa_full);

    // ------------ 2D track: msa2pair (outer-product-mean) ------------
    // einsum('bikd,bjkd->bijd(de)', left, right/N) — 收缩 seq 维 N（dims[2]），特征笛卡尔积 D×D→D*D。
    // msa_full 图 [D_MSA,L,N,B]：N = dims[2]（seq），L = dims[1]（残基）。
    // 用专用图 op OP_OUTER_PROD_MEAN（out_prod 只收缩 dims[1]，无法收缩 N 维）。同 IterBlock::forward_graph。
    const int64_t Nseq_msa2pair_full = msa_full->shape().dims[2];
    auto msa_normed_full  = msa2pair_norm_->forward(msa_full);                  // [D_MSA, L, N, B]
    auto left_full        = msa2pair_left_proj_->forward_graph(msa_normed_full);   // [16, L, N, B]
    auto right_full       = msa2pair_right_proj_->forward_graph(msa_normed_full);  // [16, L, N, B]
    auto pair_update_full = outer_product_mean(left_full, right_full,
                                               static_cast<int>(Nseq_msa2pair_full));  // [256,L,L,B]
    pair_update_full      = msa2pair_out_proj_->forward_graph(pair_update_full);      // [D_PAIR,L,L,B]
    pair                  = add_impl(pair, pair_update_full, /*inplace=*/false);      // residual

    // Triangle Multiplication (out/in) + dropout + residual (同 IterBlock::forward_graph)
    {
        Dropout drop_row(1, 0.15);
        TensorF32* tri_out = drop_row.forward_graph(tri_mul_out_->forward_graph(pair, /*bOutgoing=*/true));
        pair = add_impl(pair, tri_out, /*inplace=*/false);
        TensorF32* tri_in = drop_row.forward_graph(tri_mul_in_->forward_graph(pair, /*bOutgoing=*/false));
        pair = add_impl(pair, tri_in, /*inplace=*/false);
    }

    // ------------ pair2pair ------------
    TensorF32* rbf_proj = pair2pair_rbf_proj_->forward_graph(rbf);   // [128, L, L, B]
    // gate = sigmoid(gate_proj(outer_product(state)))——与 IterBlock::forward_graph 相同
    TensorF32* state_normed  = pair2pair_state_norm_->forward(state);           // [D_STATE, L, B]
    TensorF32* gate_left  = pair2pair_left_proj_->forward_graph(state_normed);  // [16, L, B]
    TensorF32* gate_right = pair2pair_right_proj_->forward_graph(state_normed); // [16, L, B]
    TensorF32* gate       = outer_product_graph(gate_left, gate_right);         // [256, L, L, B]
    gate = pair2pair_gate_proj_->forward_graph(gate);                           // [128, L, L, B]
    gate = sigmoid(gate);                                                       // [0,1]
    rbf_proj = mul(rbf_proj, gate);                                             // element-wise gate

    // Biased Axial Attention (row/col) + residual
    pair = add_impl(pair, pair_row_attn_->forward_graph(pair, rbf_proj),
                    /*inplace=*/false);
    pair = add_impl(pair, pair_col_attn_->forward_graph(pair, rbf_proj),
                    /*inplace=*/false);
    // FeedForward (pair_ff) + residual — 值版 forward 同步补齐（见 FullBlock::forward）
    pair = add_impl(pair, pair_ff_->forward_graph(pair), /*inplace=*/false);

    // ------------ 3D track: SE3 Transformer ------------
    // 同 IterBlock::forward_graph：SE3 由训练入口驱动（图外值回落）。
    // 训练入口在每个 block 边界：graph_compute(pair)→pair_value 后调用
    //   run_se3_structural(msa_full, pair, rbf, state, pair_value, coords, residx, seq1hot)
    //   （内部 Phase A 回落 pair_value→edge_out→make_graph→basis；Phase B run_se3_graph 追加
    //     SE3 图节点并回写 state；返回 offset 图节点），再 graph_compute 出 offset 后调用
    //   apply_coord_update(offset_value, coords) 更新骨架坐标。
    //(void)coords; (void)residx; (void)seq1hot;

    return pair;
}

// RefineBlock 构造函数已在 Model.h 中 inline 定义
/* RefineBlock::RefineBlock(const PPMLConfig& config)
    : IterBlock(config, false)  // update_msa_pair = false, 仅更新结构
    // , norm_msa_(D_MSA), norm_pair_(D_PAIR), norm_state_(D_STATE)
    // , embed_x_(NODE_IN_DIM, NODE_OUT_DIM), norm_node_(NODE_OUT_DIM)
    // , embed_e1_(D_PAIR, N_EDGE_FEATS), norm_edge1_(N_EDGE_FEATS)
    // , embed_e2_(EDGE_IN_DIM2, N_EDGE_FEATS), norm_edge2_(N_EDGE_FEATS)
    // 以上 10 个参数现在由 PPMLModel 创建，通过指针注入
{
} */

/* void RefineBlock::set_seq_info(const TensorF32& seq1hot, const TensorI64& idx) {
    seq1hot_ = seq1hot;
    idx_     = idx;
    has_seq_info_ = true;
} */

void RefineBlock::forward(TensorF32& msa, 
                          TensorF32& pair, 
                          TensorF32& state, 
                          const TensorF32& seq1hot,
                          const TensorF32& coords,
                          const TensorF32& bond_feats,
                          const TensorF32& dist_matrix,
                          const TensorF32& same_chain,
                          const TensorI64& residx) {

    // ---- 获取维度 ----
    const auto& msa_shape = msa.shape();
    int B = static_cast<int>(msa_shape.dims[0]);
    int Nseq = static_cast<int>(msa_shape.dims[1]);
    int L = static_cast<int>(msa_shape.dims[2]);

    // ================================================================
    // Step 1: LayerNorm 归一化三个 track 输入（值版：forward_exec，
    //         forward 是图版返回图节点 data()=nullptr，copy/view 会崩）
    // ================================================================
    // node 输入 = msa 沿 Nseq 维均值（对齐图版 run_se3_graph_refine: msa→mean→norm_msa）
    TensorF32 msa_mean({B, L, D_MSA}, msa.device());
    float* mean_data = msa_mean.data();
    const float* msa_data = msa.data();
    std::memset(mean_data, 0, B * L * D_MSA * sizeof(float));
    for (int b = 0; b < B; ++b)
        for (int n = 0; n < Nseq; ++n)
            for (int l = 0; l < L; ++l)
                for (int d = 0; d < D_MSA; ++d)
                    mean_data[(b * L + l) * D_MSA + d] +=
                        msa_data[((b * Nseq + n) * L + l) * D_MSA + d] / float(Nseq);
    TensorF32 node    = norm_msa_->forward_exec(msa_mean);   // (B, L, 256)
    TensorF32 pair_n  = norm_pair_->forward_exec(pair);      // (B, L, L, 128)
    TensorF32 state_n = norm_state_->forward_exec(state);    // (B, L, 32)

    // ================================================================
    // Step 2: 构建节点特征（值版：手写 concat，concat_ptr 是图版）
    // Python: node = cat((node, seq1hot, state), dim=-1)
    //         node = self.norm_node(self.embed_x(node))
    // ================================================================
    // cat([msa_norm(B,L,256), seq1hot(B,L,21), state_norm(B,L,32)]) → (B, L, 309)
    const int64_t NODE_IN = REFINE_NODE_IN_DIM;   // 309
    TensorF32 node_cat({B, L, NODE_IN}, msa.device());
    float* nc_data = node_cat.data();
    const float* nd_data = node.data();
    const float* onehot_data = seq1hot.data();
    const float* st_data = state_n.data();
    for (int b = 0; b < B; ++b) {
        for (int l = 0; l < L; ++l) {
            int base = (b * L + l) * NODE_IN;
            int base_s = (b * L + l);
            for (int d = 0; d < D_MSA; ++d)
                nc_data[base + d] = nd_data[base_s * D_MSA + d];
            for (int d = 0; d < 21; ++d)
                nc_data[base + D_MSA + d] = onehot_data[base_s * 21 + d];
            for (int d = 0; d < D_STATE; ++d)
                nc_data[base + D_MSA + 21 + d] = st_data[base_s * D_STATE + d];
        }
    }

    // Linear(309 → 32) → LayerNorm（值版 forward_exec）→ (B, L, 32)
    TensorF32 node_emb = embed_x_->forward(node_cat);
    TensorF32 node_out = norm_node_->forward_exec(node_emb);

    // ================================================================
    // Step 3: 构建边特征（两阶段，值版）
    // ================================================================
    // 阶段1: pair (B,L,L,128) → Linear → (B,L,L,32) → LayerNorm → (B,L,L,32)
    TensorF32 pair_emb = embed_e1_->forward(pair_n);
    TensorF32 pair_e1  = norm_edge1_->forward_exec(pair_emb);

    // 获取辅助边特征
    TensorF32 neighbor = se3::get_bonded_neigh(residx);            // (B, L, L, 1)
    TensorF32 rbf_feat = compute_rbf_feature(coords);              // (B, L, L, 64)

    // cat(pair_e1(32), rbf_feat(64), neighbor(1)) → (B,L,L,97) → Linear → LayerNorm
    const int64_t EDGE_IN = REFINE_EDGE_IN_DIM2;   // 97
    TensorF32 pair_cat({B, L, L, EDGE_IN}, pair.device());
    float* pc_data = pair_cat.data();
    const float* e1_data = pair_e1.data();
    const float* rbf_data = rbf_feat.data();
    const float* nbr_data = neighbor.data();
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < L; ++i) {
            for (int j = 0; j < L; ++j) {
                int base = ((b * L + i) * L + j) * EDGE_IN;
                int base_s = (b * L + i) * L + j;
                for (int d = 0; d < N_EDGE_FEATS; ++d)
                    pc_data[base + d] = e1_data[base_s * N_EDGE_FEATS + d];
                for (int d = 0; d < 64; ++d)
                    pc_data[base + N_EDGE_FEATS + d] = rbf_data[base_s * 64 + d];
                pc_data[base + N_EDGE_FEATS + 64] = nbr_data[base_s];
            }
        }
    }
    TensorF32 pair_e2  = embed_e2_->forward(pair_cat);
    TensorF32 edge_out = norm_edge2_->forward_exec(pair_e2);

    // ================================================================
    // Step 4: 构建消息传递图
    // Python: G = make_graph_topk(xyz, pair, idx, top_k=top_k)
    // ================================================================
    se3::GraphData G = se3::make_graph(coords, edge_out, residx,
                            64 /* top_k */, 9 /* kmin */);

    // ================================================================
    // Step 5: 计算 l1 特征（各原子相对 CA 的位移向量）
    // Python: l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
    //         l1_feats = l1_feats.reshape(B*L, -1, 3)
    // 输出: (B*L, 3, 3)  3个原子 × 3个坐标维度
    // ================================================================
    TensorF32 l1_feats = compute_l1_features(coords);  // (B*L, 3, 3)

    // ================================================================
    // Step 6: SE(3) Transformer 前向传播
    // Python: shift = self.se3(G, node.reshape(B*L, -1, 1), l1_feats)
    //
    // node_out: (B, L, 32) → (B*L, 32, 1) 作为 degree-0 标量特征
    // l1_feats: (B*L, 3, 3)               作为 degree-1 向量特征
    // ================================================================
    // 构建 SE3Basis (预计算球谐基)
    SE3Basis basis;
    basis.compute(G.edge_d, 2);  // J_max=2, 边向量来自 graph

    // 构建 SE3Features 输入
    SE3Features node_se3;
    node_se3.features.resize(2);
    node_se3.features[0] = node_out.view({B * L, ITER_NODE_3D_OUT, 1});
    //    先按 l1_feats 形状重建再拷贝（同 IterBlock/FullBlock 修复）。
    node_se3.features[1].~TensorF32();
    new (&node_se3.features[1]) TensorF32(l1_feats.shape(), l1_feats.device());
    node_se3.features[1].copy_from(l1_feats);

    SE3Features se3_out = se3_->forward(
            node_se3, G.edge_index, G.edge_d, &G.edge_w, basis);
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[RBLK] se3_out features=%zu f0=%lld f1=%lld\n",
        se3_out.features.size(),
        (long long)(se3_out.features.size() > 0 ? se3_out.features[0].numel() : -1),
        (long long)(se3_out.features.size() > 1 ? se3_out.features[1].numel() : -1));

        // ---- Step 4j: 提取输出 ----
        // state: degree-0 → (B*L, D_STATE) → (B, L, D_STATE)
        if (state.numel() != static_cast<int64_t>(B * L * D_STATE)) {
            state.~TensorF32();
            new (&state) TensorF32(Shape({B, L, D_STATE}), se3_out.features[0].device());
        }
        state.copy_from(se3_out.features[0].view({B, L, D_STATE}));

        // offset: degree-1 → (B*L, 3, 3) → (B, L, 3, 3)
        TensorF32 offset = se3_out.features[1].view({B, L, 3, 3});
        if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[RBLK] offset extract OK (%lld)\\n", (long long)offset.numel());

        // ---- Step 4k: 坐标更新 ----
        // CA_new = xyz[:,:,1] + offset[:,:,1]
        // N_new  = CA_new + offset[:,:,0]
        // C_new  = CA_new + offset[:,:,2]
        // 【2026-09-12 修复】同 IterBlock/FullBlock 值版：单步位移统一为 clamp(offset×0.03, ±3Å)
        //   （原实现直接加原始 offset → 每 block 数百~数千 Å 漂移，见 Experiment.md §5）。
        static constexpr float SE3_OFFSET_SCALE = 0.03f;
        float kMaxStep = 3.0f;
        if (const char* _s = std::getenv("PPML_SE3_MAX_STEP")) kMaxStep = std::atof(_s);
        const bool do_clamp = (kMaxStep > 0.0f);
        auto step_of = [&](float off) -> float {
            float d = off * SE3_OFFSET_SCALE;
            if (do_clamp) { if (d >  kMaxStep) return  kMaxStep; if (d < -kMaxStep) return -kMaxStep; }
            return d;
        };
    const float* xyz_data = coords.data();
    const float* off_data = offset.data();

    TensorF32 xyz_new({B, L, 3, 3}, coords.device());
    float* xyz_out = xyz_new.data();

    for (int b = 0; b < B; ++b) {
        for (int l = 0; l < L; ++l) {
            int base = (b * L + l) * 9;

            float ca_x0 = xyz_data[base + 3];
            float ca_y0 = xyz_data[base + 4];
            float ca_z0 = xyz_data[base + 5];

                // δCA (单步位移：scale + clamp)
            float dca_x = step_of(off_data[base + 3]);
            float dca_y = step_of(off_data[base + 4]);
            float dca_z = step_of(off_data[base + 5]);

                // 更新后 CA
            float ca_x_new = ca_x0 + dca_x;
            float ca_y_new = ca_y0 + dca_y;
            float ca_z_new = ca_z0 + dca_z;

                // N = CA_new + δN
            xyz_out[base + 0] = ca_x_new + step_of(off_data[base + 0]);
            xyz_out[base + 1] = ca_y_new + step_of(off_data[base + 1]);
            xyz_out[base + 2] = ca_z_new + step_of(off_data[base + 2]);

                // CA = CA_new
            xyz_out[base + 3] = ca_x_new;
            xyz_out[base + 4] = ca_y_new;
            xyz_out[base + 5] = ca_z_new;

                // C = CA_new + δC
            xyz_out[base + 6] = ca_x_new + step_of(off_data[base + 6]);
            xyz_out[base + 7] = ca_y_new + step_of(off_data[base + 7]);
            xyz_out[base + 8] = ca_z_new + step_of(off_data[base + 8]);
        }
    }

    //    先按源 shape 重建（同 IterBlock/FullBlock 修复）。
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[RBLK] before xyz_new_ store: xyz_new_.numel=%lld xyz_new.numel=%lld this=%p\\n",
        (long long)xyz_new_.numel(), (long long)xyz_new.numel(), (void*)&xyz_new_);
    if (xyz_new_.numel() != xyz_new.numel()) {
        xyz_new_.~TensorF32();
        new (&xyz_new_) TensorF32(xyz_new.shape(), xyz_new.device());
    }
    xyz_new_.copy_from(xyz_new);
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[RBLK] after xyz_new_ store: numel=%lld this=%p\\n",
        (long long)xyz_new_.numel(), (void*)&xyz_new_);
    //       orientations, edge_index, training)。
    //       等接口完善后，此处替换为：
    //
    //   // 组装节点特征为 SE3Features（度0=32通道, 度1=3通道）
    //   Fiber fiber_in({32, 3}, {0, 1});
    //   SE3Features node_se3(fiber_in, /*prototype*/..., B*L);
    //   // 填充 node_se3.features[0] = node_out.view({B*L, 32, 1})
    //   // 填充 node_se3.features[1] = l1_feats
    //
    //   SE3Features se3_out = se3_->forward(node_se3, coords,
    //                                       TensorF32() /* orient */,
    //                                       G.edge_index);
    //
    //   // 提取输出
    //   TensorF32 state_vec = se3_out.features[0];  // degree-0 → (B*L, C)
    //   TensorF32 offset    = se3_out.features[1];  // degree-1 → (B*L, 3, 3)

    // ================================================================
    // Step 7: 整理输出
    // Python: state  = shift['0'].reshape(B, L, -1)
    //         offset = shift['1'].reshape(B, L, -1, 3)
    // ================================================================
    //
    // state_new_ = state_vec.view({B, L, D_STATE});        // (B, L, 32)
    // TensorF32 offset = offset_tensor.view({B, L, 3, 3}); // (B, L, 3, 3)
    //
    // // ================================================================
    // // Step 8: 更新骨架坐标
    // // Python:
    // //   CA_new = xyz[:,:,1] + offset[:,:,1]
    // //   N_new  = CA_new + offset[:,:,0]
    // //   C_new  = CA_new + offset[:,:,2]
    // //   xyz_new = torch.stack([N_new, CA_new, C_new], dim=2)
    // //
    // // offset 各通道定义:
    // //   [:, :, 0, :] = δN  (N 原子相对 CA 的位移)
    // //   [:, :, 1, :] = δCA (CA 位移, 绝对; 即 offset[:,:,1] 直接加在 CA 上)
    // //   [:, :, 2, :] = δC  (C 原子相对 CA 的位移)
    // // ================================================================
    //
    // const float* xyz_data = coords.data();
    // const float* off_data = offset.data();
    //
    // xyz_new_ = TensorF32({B, L, 3, 3}, coords.device());
    // float* xyz_out = xyz_new_.data();
    //
    // for (int b = 0; b < B; ++b) {
    //     for (int l = 0; l < L; ++l) {
    //         // 原始 CA 坐标
    //         int ca_idx  = (b * L + l) * 9 + 3;  // atom=1, x
    //         float ca_x0 = xyz_data[ca_idx];
    //         float ca_y0 = xyz_data[ca_idx + 1];
    //         float ca_z0 = xyz_data[ca_idx + 2];
    //
    //         // offset 各分量
    //         int off_base = (b * L + l) * 9;
    //
    //         // δCA (绝对偏移)
    //         float dca_x = off_data[off_base + 3];
    //         float dca_y = off_data[off_base + 4];
    //         float dca_z = off_data[off_base + 5];
    //
    //         // 更新后的 CA
    //         float ca_x_new = ca_x0 + dca_x;
    //         float ca_y_new = ca_y0 + dca_y;
    //         float ca_z_new = ca_z0 + dca_z;
    //
    //         // N = CA_new + δN
    //         xyz_out[off_base + 0] = ca_x_new + off_data[off_base + 0];
    //         xyz_out[off_base + 1] = ca_y_new + off_data[off_base + 1];
    //         xyz_out[off_base + 2] = ca_z_new + off_data[off_base + 2];
    //
    //         // CA = CA_new (绝对位置)
    //         xyz_out[off_base + 3] = ca_x_new;
    //         xyz_out[off_base + 4] = ca_y_new;
    //         xyz_out[off_base + 5] = ca_z_new;
    //
    //         // C = CA_new + δC
    //         xyz_out[off_base + 6] = ca_x_new + off_data[off_base + 6];
    //         xyz_out[off_base + 7] = ca_y_new + off_data[off_base + 7];
    //         xyz_out[off_base + 8] = ca_z_new + off_data[off_base + 8];
    //     }
    // }
    //
    // // 回写 state (通过引用)
    // state.copy_from(state_new_);
}

// ===== RefineBlock::forward_graph (图模式, override) =====
// update_msa_pair_=false：RefineBlock 不改 msa/pair 两条 track，仅做 3D 结构更新。
// 因此本 override 为 pass-through（直接返回 pair，msa/pair/state 保持原图节点引用）。
// 3D 结构更新由训练入口在每个 refine block 边界调用 run_se3_structural_refine 驱动
// （与 IterBlock 由 drive_block_se3 调 run_se3_structural 的驱动模式一致）。
TensorF32* RefineBlock::forward_graph(TensorF32*& msa, TensorF32*& pair,
                                      TensorF32* rbf, TensorF32*& state,
                                      const TensorF32* coords,
                                      const TensorI64* residx,
                                      const TensorF32* seq1hot) {
    // 不做任何 1D/2D track 修改；SE3 结构更新由训练入口驱动。
    (void)msa; (void)rbf; (void)state; (void)coords; (void)residx; (void)seq1hot;
    return pair;
}

// ===== RefineBlock::run_se3_graph_refine (可微 SE3 图块) =====
// 语义对齐值版 RefineBlock::forward 的 SE3 部分，但用图 op 构建可微节点：
//   node 度0 = norm_node_(embed_x_(cat(norm_msa(msa 沿 Nseq 均值), seq1hot, norm_state(state))))
//   node 度1 = l1_feats (compute_l1_features(coords)) 常量叶子
//   edges    = G.edge_index / edge_d / edge_w → src/tgt/d/w 常量叶子
//   basis    = 调用方预计算
//   out      = se3_->forward_graph({node0,node1}, src, tgt, d, w, basis, N=B*L)
//   state    = out[0].view({B,L,D_STATE})（引用回写）
// 返回 se3_out：se3_out[0]=state(度0)、se3_out[1]=offset(度1，训练入口 graph_compute 后
//      回落值 + apply_coord_update 更新骨架坐标)。空 vector = 无有效边图（跳过 SE3）。
std::vector<TensorF32*> RefineBlock::run_se3_graph_refine(TensorF32*& msa, TensorF32*& pair, TensorF32* rbf,
                                                          TensorF32*& state,
                                                          const se3::GraphData& G, const SE3Basis& basis,
                                                          const TensorF32& coords, const TensorF32& seq1hot) {
    (void)pair; (void)rbf;
    const int B = seq1hot.shape().dims[0];
    const int L = seq1hot.shape().dims[1];
    const int64_t N = static_cast<int64_t>(B) * L;   // 图节点数

    // ---- node 度0：msa 沿 Nseq 维均值 → cat(norm_msa(msa_mean), seq1hot, norm_state(state)) ----
    // msa 图节点 [D_MSA, L, Nseq, B]：permute 把 Nseq 移到最内 dims[0] 后 sum_rows 归约得均值。
    const int64_t Nseq = msa->shape().dims[2];
    TensorF32* msa_p = permute(msa, std::vector<int>{2, 0, 1, 3});      // [Nseq, D_MSA, L, B]
    TensorF32* msa_s = sum_rows(msa_p);                                // [1, D_MSA, L, B]
    TensorF32* msa_m = scale(msa_s, 1.0f / static_cast<float>(Nseq));   // 均值
    TensorF32* msa_v = view(msa_m, Shape({D_MSA, L, B}));              // [D_MSA, L, B]
    TensorF32* msa_n = norm_msa_->forward(msa_v);                      // [D_MSA, L, B]
    // seq1hot 常量 [21, L, B]；state 图 [D_STATE, L, B] → norm_state_
    TensorF32* s1h   = constant_tensor({21, L, B}, seq1hot.data());
    TensorF32* st_n  = norm_state_->forward(state);                    // [D_STATE, L, B]
    // concat 特征维 dims[0] → [D_MSA+21+D_STATE=309, L, B]（对齐 REFINE_NODE_IN_DIM）
    TensorF32* node_cat = concat_ptr({msa_n, s1h, st_n}, 0);           // [309, L, B]
    TensorF32* node_emb = embed_x_->forward_graph(node_cat);           // [32, L, B]
    TensorF32* node_nrm = norm_node_->forward(node_emb);               // [32, L, B]
    TensorF32* node0 = view(node_nrm, Shape({REFINE_NODE_OUT_DIM, N})); // [32, B*L]（n=b*L+l）

    // ---- node 度1：l1_feats 常量叶子 [3*d_dim1, B*L]，与值版 node_se3.features[1] 对齐 ----
    // 值版固定度1输入 = l1_feats (B*L, 3, 3)（3 通道位移向量，d_dim1=3）。
    // SE3Transformer(SE3Config) 构造器已把 fiber_in 度1 通道数取 cfg.l1_features[0]=3。
    TensorF32 l1 = compute_l1_features(coords);                        // (B*L, 3, 3)
    const int m1     = 3;                                              // 度1 通道数（值版 l1_feats 固定 3）
    const int d_dim1 = 3;                                              // 度1 → 2*1+1
    //    ggml dims[0]=最内维 → 图节点 [m1*d_dim1, N] 数据序应为 [节点][特征]（节点外层）。
    //    kernel_mul_mat 读 a[i*K+k]（i=节点行, k=特征K）要求 [节点][特征] 序。
    //    原序导致 G1x1SE3 度1 输出错位放大 ~26 倍（l1_feats ~100 → ±2680，值版仅 3.75）。
    //    修复后 l1data[n*(m1*d_dim1) + (a*d_dim1+c)] 与值版 (N, m, d_dim)（n 外层, Wigner 最内）一致。
    std::vector<float> l1data(static_cast<size_t>(m1) * d_dim1 * N, 0.0f);
    for (int64_t n = 0; n < N; ++n)
        for (int a = 0; a < m1; ++a)
            for (int c = 0; c < d_dim1; ++c)
                l1data[static_cast<size_t>(n) * (m1 * d_dim1) + (a * d_dim1 + c)] =
                    l1.data()[n * 9 + a * 3 + c];
    TensorF32* node1 = constant_tensor({m1 * d_dim1, N}, l1data.data());

    // ---- 边特征：拓扑常量（edge_index/edge_d）+ edge_w 图化（主图 pair）----
    const int64_t E = G.edge_index.numel() > 0 ? G.edge_index.shape().dims[1] : 0;
    if (E <= 0) {
        // 无有效边图（结构常量未注入），SE3 图块无法执行；仅回写 state=输入（等价跳过）。
        return {};
    }
    std::vector<float> src_d(static_cast<size_t>(E)), tgt_d(static_cast<size_t>(E));
    for (int64_t e = 0; e < E; ++e) {
        src_d[e] = static_cast<float>(G.edge_index.data()[e]);
        tgt_d[e] = static_cast<float>(G.edge_index.data()[E + e]);
    }
    TensorF32* edge_src = constant_tensor({E}, src_d.data());
    TensorF32* edge_tgt = constant_tensor({E}, tgt_d.data());
    // edge_d: (E,3) → [3,E]（结构常量，用 host coords 值版算）
    std::vector<float> dd(static_cast<size_t>(3 * E));
    for (int64_t e = 0; e < E; ++e)
        for (int c = 0; c < 3; ++c) dd[static_cast<size_t>(c) * E + e] = G.edge_d.data()[e * 3 + c];
    TensorF32* edge_d = constant_tensor({3, E}, dd.data());
    // ===== edge_w 图化（同一 autograd 图）=====
    // 【阶段1.5 重构】从主图 pair 图节点提取（阶段1 简化：pair→norm_pair_→embed_e1_→norm_edge1_，
    // 暂不注入 rbf/neighbor 增强特征，阶段2 补全）。pair [D_PAIR,L,L,B] → [N_EDGE_FEATS,L,L,B]
    // → view [N_EDGE_FEATS, L*L*B] → edge_gather_rows → [N_EDGE_FEATS, E]。
    const int64_t LLB   = static_cast<int64_t>(L) * L * B;     // 节点空间 L*L*B
    TensorF32* edge_feat_g = norm_edge1_->forward(
        embed_e1_->forward_graph(norm_pair_->forward(pair)));  // [N_EDGE_FEATS,L,L,B]
    TensorF32* edge_feat_flat = view(edge_feat_g,
        Shape({edge_feat_g->shape().dims[0], LLB}));           // [N_EDGE_FEATS, L*L*B]
    // 边索引常量（host 值：b*L*L + src%L*L + tgt%L）
    std::vector<float> pair_idx(E);
    for (int64_t e = 0; e < E; ++e) {
        const int64_t src = static_cast<int64_t>(G.edge_index.data()[e]);       // b*L+i
        const int64_t tgt = static_cast<int64_t>(G.edge_index.data()[E + e]);   // b*L+j
        const int64_t b = src / L, i = src % L, j = tgt % L;
        pair_idx[e] = static_cast<float>(b * L * L + i * L + j);
    }
    TensorF32* edge_idx_leaf = constant_tensor({E}, pair_idx.data());
    TensorF32* edge_w = edge_gather_rows(edge_feat_flat, edge_idx_leaf);  // [E_dim, E] 图节点

    // ---- SE3 Transformer forward_graph ----
    std::vector<TensorF32*> h_nodes = {node0, node1};
    std::vector<TensorF32*> se3_out = se3_->forward_graph(
        h_nodes, edge_src, edge_tgt, edge_d, edge_w, basis, static_cast<int>(N));

    // ---- state 回写：度0 → (B,L,D_STATE) 图 [D_STATE, L, B] ----
    // se3_out[0] 为 [32, B*L]（度0 输出），节点序 n=b*L+l → view [D_STATE, L, B]
    state = view(se3_out[0], Shape({D_STATE, L, B}));

    // 返回 se3_out：se3_out[1] 为度1 offset 图节点 [3*3, B*L]（坐标更新需"图外"回落其值）。
    return se3_out;
}

// ===== 训练入口驱动：RefineBlock SE3 图块（图外值回落 + state 回写）=====
// Phase A（结构常量，图外值回落）：pair_value 经值版两阶段边嵌入得 edge_out，再
//   make_graph(coords, edge_out, residx) → G，并 basis.compute(G.edge_d, 2)。
//   RefineBlock 边特征依赖当前坐标（rbf_feat=compute_rbf_feature(coords)）与键合邻居
//   （get_bonded_neigh(residx)），故每 block 用最新坐标重算。此阶段非可微（make_graph 是离散拓扑）。
// Phase B（可微图块）：调用 run_se3_graph_refine，追加 node 嵌入 + se3_->forward_graph 到计算图，
//   state 经引用回写为图节点（度0）。返回的 offset 图节点由训练入口 graph_compute 后做坐标更新。
std::vector<TensorF32*> RefineBlock::run_se3_structural_refine(TensorF32*& msa, TensorF32*& pair, TensorF32* rbf,
                                                               TensorF32*& state,
                                                               const TensorF32& coords,
                                                               const TensorI64& residx,
                                                               const TensorF32& seq1hot,
                                                               const se3::TopoRefineCfg*   topo_cfg,
                                                               se3::TopoRefineState*       topo_st) {
    // ---- Phase A: make_graph 拓扑（edge_index/edge_d 只依赖 host coords）----
    // 【阶段1.5 重构】不再回落 pair 值。edge_w 在 run_se3_graph_refine 内从主图 pair 图化
    // （阶段1 简化：仅 pair→norm_pair_→embed_e1_→norm_edge1_→gather，暂不注入 rbf/neighbor
    //  增强特征；阶段2 补全）。
    TensorF32 empty_pair;   // 默认构造 numel=1 data=nullptr → has_pair=false
    // 【refined topo pass L1/L2】与 IterBlock 共用同一份 topo_st ⇒ 跨 iter/refine 边界的 Ω 门控连续 ✓
    se3::GraphData G = (topo_cfg && topo_st)
        ? se3::make_graph_refined(coords, empty_pair, residx, 64, 9, *topo_cfg, *topo_st)
        : se3::make_graph(coords, empty_pair, residx, 64, 9);
    SE3Basis basis;
    basis.compute(G.edge_d, 2);
    // 无有效边图（拓扑空）时跳过 SE3
    if (G.edge_index.numel() <= 0) {
        if (getenv("GRAPH_DEBUG_COORD")) {
            std::fprintf(stderr, "[SE3-SKIP] run_se3_structural_refine: empty edge_index, skip SE3\n");
        }
        return {};
    }
    // [诊断 2026-09-12] dump 本 refine block 实际使用的拓扑
    dump_se3_topo("ref", G, coords);

    // ---- Phase B: run_se3_graph_refine（可微图块，edge_w 图化，state 回写）----
    return run_se3_graph_refine(msa, pair, rbf, state, G, basis, coords, seq1hot);
}

// ===== GPU 可用性探测 =====
// 返回 true 表示当前环境可用的 CUDA 设备数 > 0 (且 device_id 合法)。
// 用于 ensure_backend_ready 在创建 CUDABackend 前探测: 无 GPU 时给出警告并回退 CPU。
bool cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        // 打印实际失败原因，便于排查 "CUDA 调用问题"（最常见：CUDA 11/12 运行时冲突、
        // libcudart 被 LD_LIBRARY_PATH 指向错误版本、或驱动不可见）。
        fprintf(stderr,
                "[CUDA-AVAIL] cudaGetDeviceCount failed: %s (device_count=%d)\n",
                cudaGetErrorString(err), device_count);
        fprintf(stderr,
                "[CUDA-AVAIL] 提示: 本项目链接 CUDA 11 (libcudart.so.11.0)。若 LD_LIBRARY_PATH 指向了\n"
                "  含其他版本 libcudart 的目录（如 anaconda 的 libcudart），可能加载到不兼容的运行时\n"
                "  或导致设备探测失败。建议: export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu\n"
                "  （系统 11.0 所在目录），并确认 nvidia-smi 能看到 GPU、/dev/nvidia* 可访问。\n");
        return false;
    }
    return true;
}

// ===== GPU 显存可用性探测 =====
// 返回当前 CUDA 设备剩余可用显存字节数；失败返回 0。
// 用于 ensure_backend_ready：CUDA 训练图显存需求很大，若剩余显存不足则提前回退 CPU，
// 避免 graph_compute 中途 ALLOC_FAILED。
size_t cuda_free_memory_bytes() {
    if (!cuda_available()) return 0;
    size_t free_b = 0, total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) return 0;
    return free_b;
}

// 估算模型/图运行所需显存下限（字节）。粗略：参数 + 前向/反向峰值图工作区。
// 训练图峰值目前按"小样本 L=51"经验约 18-19.5GB（CPU gallocr 测得）；这里给出一个
// 可由环境变量覆盖的阈值，作为是否启用 CUDA 的判据。
size_t cuda_min_required_bytes() {
    if (const char* p = getenv("PPML_CUDA_MIN_FREE_MB")) {
        size_t mb = (size_t)std::max(0, atoi(p));
        if (mb > 0) return mb * 1024ULL * 1024ULL;
    }
    // 默认要求至少 4GB 空闲显存才启用 CUDA（RTX 2050 4GB 可跑小样本）。
    return 4ULL * 1024ULL * 1024ULL * 1024ULL;
}

// PPMLModel 实现
PPMLModel::PPMLModel(const PPMLConfig& config) : config_(config) {
    // block 数用 config（支持 PPML_N_EXTRA/PPML_N_MAIN/PPML_N_REFINE 环境变量覆盖），
    // 不用硬编码宏 N_EXTRA_BLOCKS/N_MAIN_BLOCKS/N_REFINE_BLOCKS（否则 config 显示与实际建块数不一致）。
    const int n_extra  = config.n_extra_blocks;
    const int n_main   = config.n_main_blocks;
    const int n_refine = config.n_refine_blocks;
    int n_iter = n_extra + n_main;  // ITER_N_BLOCKS = 12
    int n_refn = n_refine;          // 4

    // ===== IterBlock 3D SE 参数 (每组 6 个, 共 12 组) =====
    for (int i = 0; i < n_iter; ++i) {
        iter_norm_msa_3d_.push_back(LayerNorm::create(D_MSA));                        // 256
        iter_norm_pair_3d_.push_back(LayerNorm::create(D_PAIR));                      // 128
        iter_embed_x_.push_back(LinearLayer::create(ITER_NODE_3D_IN, ITER_NODE_3D_OUT, /*bias=*/true, /*se3=*/true));  // 277→32 SE3
        iter_embed_e_.push_back(LinearLayer::create(D_PAIR, ITER_EDGE_3D_OUT, /*bias=*/true, /*se3=*/true));       // 128→32 SE3
        iter_norm_node_3d_.push_back(LayerNorm::create(ITER_NODE_3D_OUT));             // 32
        iter_norm_edge_3d_.push_back(LayerNorm::create(ITER_EDGE_3D_OUT));             // 32
    }

    // ===== IterBlock forward 内部参数 (每组 12 个, 共 12 组) =====
    for (int i = 0; i < n_iter; ++i) {
        iter_state2msa_norm_.push_back(LayerNorm::create(D_STATE));                         // 32
        iter_state2msa_linear_.push_back(LinearLayer::create(D_STATE, D_MSA));              // 32→256
        iter_pair2msa_norm_.push_back(LayerNorm::create(D_PAIR));                           // 128
        iter_msa2pair_norm_.push_back(LayerNorm::create(D_MSA));                            // 256
        iter_msa2pair_left_proj_.push_back(LinearLayer::create(D_MSA, MSA2PAIR_HIDDEN));    // 256→16
        iter_msa2pair_right_proj_.push_back(LinearLayer::create(D_MSA, MSA2PAIR_HIDDEN));   // 256→16
        iter_msa2pair_out_proj_.push_back(LinearLayer::create(MSA2PAIR_HIDDEN * MSA2PAIR_HIDDEN, D_PAIR)); // 256→128
        iter_pair2pair_rbf_proj_.push_back(LinearLayer::create(D_RBF, D_PAIR));              // 64→128
        iter_pair2pair_state_norm_.push_back(LayerNorm::create(D_STATE));                    // 32
        iter_pair2pair_left_proj_.push_back(LinearLayer::create(D_STATE, PAIR2PAIR_GATE_HIDDEN));  // 32→16
        iter_pair2pair_right_proj_.push_back(LinearLayer::create(D_STATE, PAIR2PAIR_GATE_HIDDEN)); // 32→16
        iter_pair2pair_gate_proj_.push_back(LinearLayer::create(PAIR2PAIR_GATE_HIDDEN * PAIR2PAIR_GATE_HIDDEN, D_PAIR)); // 256→128
    }

    // ===== RefineBlock 3D SE 参数 (每组 10 个, 共 4 组) =====
    for (int i = 0; i < n_refn; ++i) {
        refine_norm_msa_.push_back(LayerNorm::create(D_MSA));                                 // 256
        refine_norm_pair_.push_back(LayerNorm::create(D_PAIR));                              // 128
        refine_norm_state_.push_back(LayerNorm::create(D_STATE));                            // 32
        refine_embed_x_.push_back(LinearLayer::create(REFINE_NODE_IN_DIM, REFINE_NODE_OUT_DIM, /*bias=*/true, /*se3=*/true)); // 309→32 SE3
        refine_norm_node_.push_back(LayerNorm::create(REFINE_NODE_OUT_DIM));                  // 32
        refine_embed_e1_.push_back(LinearLayer::create(D_PAIR, N_EDGE_FEATS, /*bias=*/true, /*se3=*/true));                // 128→32 SE3
        refine_norm_edge1_.push_back(LayerNorm::create(N_EDGE_FEATS));                        // 32
        refine_embed_e2_.push_back(LinearLayer::create(REFINE_EDGE_IN_DIM2, N_EDGE_FEATS, /*bias=*/true, /*se3=*/true));   // 97→32 SE3
        refine_norm_edge2_.push_back(LayerNorm::create(N_EDGE_FEATS));                        // 32
    }

    // ===== attention / sub-module 参数 (per-block create) =====
    AttnConfig msa_ac(config.d_msa, config.n_heads);
    AttnConfig pair_ac(config.d_pair, config.n_heads);

    // 【2026-09-11 维度修正】MSA row/col 注意力投影必须是 d_msa -> d_msa（256→256），
    //   逐头 reshape 成 H 头 × d_k（d_k = D_MSA/N_HEAD = 32），总维度不变。
    //   原写法 `N_HEAD * D_MSA`（256→2048）= 每个头都拿到完整 256 维 → 单块激活 ×8，
    //   24 块(12×2 样本)的前向激活 + 反向梯度把 Gallocr CPU 峰值推到 150GB
    //   （日志：`reserve done: backend=1 peak=... (150.13 GB)` → 168.9GB 分配 → OOM 137）。
    //   依据：① 参考实现 RoseTTAFold `MultiheadAttention` 为 `nn.Linear(d_model,d_model)`
    //   + `.view(..., heads, d_k)`（Transformer.py:56-68）；② 本文件 FullBlock 已是正确约定
    //   （`FMSA_QOUT = 64 = n_msa_head*d_msa_channels`，见下方 push_full_msa_row）；
    //   ③ `[space]` 估算器（18.9GB）也是按 256 维算的，改后两者一致。
    auto push_msa_row = [&]() {
        msa_row_Wq_.push_back(    LinearLayer::create(D_MSA, D_MSA));        // 256→256（逐头 32）
        msa_row_Wk_.push_back(    LinearLayer::create(D_MSA, D_MSA));
        msa_row_Wv_.push_back(    LinearLayer::create(D_MSA, D_MSA));
        msa_row_to_b_.push_back(  LinearLayer::create(D_PAIR, N_HEAD));      // 128→8
        msa_row_to_g_.push_back(  LinearLayer::create(D_MSA, D_MSA));        // 门控 256→256
        msa_row_to_out_.push_back(LinearLayer::create(D_MSA, D_MSA));        // 256→256
    };
    auto push_msa_col = [&]() {
        msa_col_Wq_.push_back(   LinearLayer::create(D_MSA, D_MSA));
        msa_col_Wk_.push_back(   LinearLayer::create(D_MSA, D_MSA));
        msa_col_Wv_.push_back(   LinearLayer::create(D_MSA, D_MSA));
        msa_col_to_b_.push_back( LinearLayer::create(D_PAIR, N_HEAD));
        msa_col_to_g_.push_back( LinearLayer::create(D_MSA, D_MSA));
        msa_col_to_out_.push_back(LinearLayer::create(D_MSA, D_MSA));
    };
    auto push_pair_row = [&]() {
        pair_attn_norm_.push_back(LayerNorm::create(D_PAIR));                             // 128 (AF2 PairAxialAttention 投影前 norm)
        pair_row_Wq_.push_back(   LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));  // 128→256
        pair_row_Wk_.push_back(   LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));
        pair_row_Wv_.push_back(   LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));
        pair_row_to_b_.push_back( LinearLayer::create(D_PAIR, N_HEAD));                   // 128→8
        pair_row_to_g_.push_back( LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));
        pair_row_to_out_.push_back(LinearLayer::create(N_HEAD * D_PAIR_HIDDEN, D_PAIR));   // 256→128
    };
    auto push_pair_col = [&]() {
        pair_col_Wq_.push_back(   LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));
        pair_col_Wk_.push_back(   LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));
        pair_col_Wv_.push_back(   LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));
        pair_col_to_b_.push_back( LinearLayer::create(D_PAIR, N_HEAD));
        pair_col_to_g_.push_back( LinearLayer::create(D_PAIR, N_HEAD * D_PAIR_HIDDEN));
        pair_col_to_out_.push_back(LinearLayer::create(N_HEAD * D_PAIR_HIDDEN, D_PAIR));
    };
    auto push_msa_ff = [&]() {
        msa_ff_norm_.push_back(   LayerNorm::create(D_MSA));                              // 256
        msa_ff_linear1_.push_back(LinearLayer::create(D_MSA, D_MSA * 4));                 // 256→1024
        msa_ff_linear2_.push_back(LinearLayer::create(D_MSA * 4, D_MSA));                 // 1024→256
    };
    auto push_pair_ff = [&]() {
        pair_ff_norm_.push_back(   LayerNorm::create(D_PAIR));                            // 128
        pair_ff_linear1_.push_back(LinearLayer::create(D_PAIR, D_PAIR * 2));              // 128→256
        pair_ff_linear2_.push_back(LinearLayer::create(D_PAIR * 2, D_PAIR));              // 256→128
    };
    // ===== FullBlock(extra) 专属 64 维 MSA 注意力（d_msa_full=64, n_msa_head*n_msa_channels=64）=====
    // RF2AA：FullBlock 处理 msa_full[64,...]，其 row attention/ff/global col attention 均按 64 维。
    constexpr int FMSA = D_MSA_FULL;                 // 64
    constexpr int FMSA_HID = N_HEAD * D_MSA_FULL;    // 8*64=512?  实际 n_msa_head*n_msa_channels=64
    constexpr int FMSA_QOUT = 64;                    // n_msa_head*d_msa_channels = 8*8 = 64
    auto push_full_msa_row = [&]() {
        full_msa_row_Wq_.push_back(     LinearLayer::create(FMSA, FMSA_QOUT));  // 64→64
        full_msa_row_Wk_.push_back(     LinearLayer::create(FMSA, FMSA_QOUT));
        full_msa_row_Wv_.push_back(     LinearLayer::create(FMSA, FMSA_QOUT));
        full_msa_row_to_b_.push_back(   LinearLayer::create(D_PAIR, N_HEAD));   // 128→8
        full_msa_row_to_g_.push_back(   LinearLayer::create(FMSA, FMSA_QOUT));  // 64→64
        full_msa_row_to_out_.push_back( LinearLayer::create(FMSA_QOUT, FMSA));   // 64→64
    };
    auto push_full_msa_ff = [&]() {
        full_msa_ff_norm_.push_back(   LayerNorm::create(FMSA));                              // 64
        full_msa_ff_linear1_.push_back(LinearLayer::create(FMSA, FMSA * 4));                  // 64→256
        full_msa_ff_linear2_.push_back(LinearLayer::create(FMSA * 4, FMSA));                  // 256→64
    };
    auto push_full_global_col = [&]() {
        full_msa_global_col_Wq_.push_back(   LinearLayer::create(FMSA, FMSA_QOUT));  // 64→64 single-head
        full_msa_global_col_Wk_.push_back(   LinearLayer::create(FMSA, FMSA_QOUT));
        full_msa_global_col_Wv_.push_back(   LinearLayer::create(FMSA, FMSA_QOUT));
        full_msa_global_col_to_b_.push_back( LinearLayer::create(D_PAIR, N_HEAD));   // 128→8
        full_msa_global_col_to_g_.push_back( LinearLayer::create(FMSA, FMSA_QOUT));  // 64→64
        full_msa_global_col_to_out_.push_back(LinearLayer::create(FMSA_QOUT, FMSA)); // 64→64
    };
    auto push_full_msa2pair = [&]() {
        full_msa2pair_norm_.push_back(      LayerNorm::create(FMSA));                    // 64
        full_msa2pair_left_proj_.push_back( LinearLayer::create(FMSA, MSA2PAIR_HIDDEN)); // 64→16
        full_msa2pair_right_proj_.push_back(LinearLayer::create(FMSA, MSA2PAIR_HIDDEN)); // 64→16
        full_msa2pair_out_proj_.push_back(  LinearLayer::create(MSA2PAIR_HIDDEN * MSA2PAIR_HIDDEN, D_PAIR)); // 256→128
    };
    auto push_tri = [&](std::vector<LayerNorm*>& ln1, std::vector<LayerNorm*>& ln2,
                         std::vector<LinearLayer*>& l1, std::vector<LinearLayer*>& r1,
                         std::vector<LinearLayer*>& lg, std::vector<LinearLayer*>& rg,
                         std::vector<LinearLayer*>& g,  std::vector<LinearLayer*>& op) {
        constexpr int T = 128;  // D_HIDDEN_TRIMUL
        ln1.push_back(LayerNorm::create(D_PAIR));                                        // 128
        l1.push_back( LinearLayer::create(D_PAIR, T));   r1.push_back( LinearLayer::create(D_PAIR, T));
        lg.push_back( LinearLayer::create(D_PAIR, T));   rg.push_back( LinearLayer::create(D_PAIR, T));
        g.push_back(  LinearLayer::create(D_PAIR, D_PAIR));
        ln2.push_back(LayerNorm::create(T));
        op.push_back( LinearLayer::create(T, D_PAIR));
    };

    // n_iter 份 IterBlock 参数
    for (int i = 0; i < n_iter; ++i) {
        push_msa_row(); push_msa_col(); push_pair_row(); push_pair_col();
        push_msa_ff(); push_pair_ff();
        push_tri(tri_out_layernorm_, tri_out_output_layernorm_,
                 tri_out_left_proj_, tri_out_right_proj_,
                 tri_out_left_gate_, tri_out_right_gate_,
                 tri_out_gate_, tri_out_out_proj_);
        push_tri(tri_in_layernorm_, tri_in_output_layernorm_,
                 tri_in_left_proj_, tri_in_right_proj_,
                 tri_in_left_gate_, tri_in_right_gate_,
                 tri_in_gate_, tri_in_out_proj_);
    }

    // n_extra 份 GlobalColAttention 参数（2026-09-11 同上修正为 d_msa → d_msa）
    for (int i = 0; i < n_extra; ++i) {
        msa_global_col_Wq_.push_back(   LinearLayer::create(D_MSA, D_MSA));
        msa_global_col_Wk_.push_back(   LinearLayer::create(D_MSA, D_MSA));
        msa_global_col_Wv_.push_back(   LinearLayer::create(D_MSA, D_MSA));
        msa_global_col_to_b_.push_back( LinearLayer::create(D_PAIR, N_HEAD));
        msa_global_col_to_g_.push_back( LinearLayer::create(D_MSA, D_MSA));
        msa_global_col_to_out_.push_back(LinearLayer::create(D_MSA, D_MSA));
    }

    // ===== FullBlock(extra) 专属 64 维 MSA 注意力参数（n_extra 份）=====
    for (int i = 0; i < n_extra; ++i) {
        push_full_msa_row();
        push_full_msa_ff();
        push_full_global_col();
        push_full_msa2pair();
    }

    // ===== PositionalEncoding 参数 (每 block 2 个 EmbeddingLayer) =====
    // 必须先于下方 block 构造循环分配 (1575/1649 处按 idx 访问 pos_enc_emb_res_[idx])
    for (int i = 0; i < n_iter; ++i) {
        pos_enc_emb_res_.push_back(EmbeddingLayer::create(65, D_PAIR));     // (65, 128)  residue dist
        pos_enc_emb_atom_.push_back(EmbeddingLayer::create(17, D_PAIR));    // (17, 128)  atom bond dist
    }

    // ===== 创建迭代块 + 注入指针 =====
    // extra_blocks (n_extra) — FullBlock with update_msa_pair=true
    for (int i = 0; i < n_extra; ++i) {
        int idx = i;
        auto block = std::make_unique<FullBlock>(config, true);
        // 注入 3D SE + forward 内部参数
        block->norm_msa_3d_  = iter_norm_msa_3d_[idx]; block->norm_pair_3d_ = iter_norm_pair_3d_[idx];
        block->embed_x_      = iter_embed_x_[idx];     block->embed_e_      = iter_embed_e_[idx];
        block->norm_node_3d_ = iter_norm_node_3d_[idx]; block->norm_edge_3d_ = iter_norm_edge_3d_[idx];
        block->state2msa_norm_       = iter_state2msa_norm_[idx];
        block->state2msa_linear_     = iter_state2msa_linear_[idx];
        block->pair2msa_norm_        = iter_pair2msa_norm_[idx];
        // FullBlock 处理 msa_full[64,...]，msa2pair 用 64 维专属权重
        block->msa2pair_norm_        = full_msa2pair_norm_[idx];
        block->msa2pair_left_proj_   = full_msa2pair_left_proj_[idx];
        block->msa2pair_right_proj_  = full_msa2pair_right_proj_[idx];
        block->msa2pair_out_proj_    = full_msa2pair_out_proj_[idx];
        block->pair2pair_rbf_proj_   = iter_pair2pair_rbf_proj_[idx];
        block->pair2pair_state_norm_ = iter_pair2pair_state_norm_[idx];
        block->pair2pair_left_proj_  = iter_pair2pair_left_proj_[idx];
        block->pair2pair_right_proj_ = iter_pair2pair_right_proj_[idx];
        block->pair2pair_gate_proj_  = iter_pair2pair_gate_proj_[idx];

        // 创建并注入子模块 (layernorm 在 IterBlock::forward 外部完成)
        // 注意：FullBlock 处理 msa_full[64,...]，其 row attention / ff 用 D_MSA_FULL=64 专属权重
        auto msa_row = std::make_unique<MSARowAttention>();
        msa_row->set_params(msa_ac,
            full_msa_row_to_b_[idx], full_msa_row_to_g_[idx], full_msa_row_to_out_[idx],
            full_msa_row_Wq_[idx], full_msa_row_Wk_[idx], full_msa_row_Wv_[idx]);
        // FullBlock 不用 msa_col（用 global col attention），注入 nullptr 即可
        std::unique_ptr<MSAColAttention> msa_col;
        auto msa_ff = std::make_unique<FeedForward>();
        msa_ff->set_params(D_MSA_FULL, D_MSA_FULL * 4, 0.1f,
            full_msa_ff_norm_[idx], full_msa_ff_linear1_[idx], full_msa_ff_linear2_[idx]);
        auto pair_row = std::make_unique<PairRowAttention>();
        pair_row->set_params(pair_ac, pair_attn_norm_[idx],
            pair_row_to_b_[idx], pair_row_to_g_[idx], pair_row_to_out_[idx],
            pair_row_Wq_[idx], pair_row_Wk_[idx], pair_row_Wv_[idx]);
        auto pair_col = std::make_unique<PairColAttention>();
        pair_col->set_params(pair_ac, pair_attn_norm_[idx],
            pair_col_to_b_[idx], pair_col_to_g_[idx], pair_col_to_out_[idx],
            pair_col_Wq_[idx], pair_col_Wk_[idx], pair_col_Wv_[idx]);
        auto pair_ff = std::make_unique<FeedForward>();
        pair_ff->set_params(D_PAIR, D_PAIR * 2, 0.1f, pair_ff_norm_[idx], pair_ff_linear1_[idx], pair_ff_linear2_[idx]);
        auto tri_out = std::make_unique<TriangleMultiplication>();
        tri_out->set_params(D_PAIR,
            tri_out_layernorm_[idx], tri_out_left_proj_[idx], tri_out_right_proj_[idx],
            tri_out_left_gate_[idx], tri_out_right_gate_[idx], tri_out_gate_[idx],
            tri_out_output_layernorm_[idx], tri_out_out_proj_[idx]);
        auto tri_in = std::make_unique<TriangleMultiplication>();
        tri_in->set_params(D_PAIR,
            tri_in_layernorm_[idx], tri_in_left_proj_[idx], tri_in_right_proj_[idx],
            tri_in_left_gate_[idx], tri_in_right_gate_[idx], tri_in_gate_[idx],
            tri_in_output_layernorm_[idx], tri_in_out_proj_[idx]);
        auto se3 = std::make_unique<SE3Transformer>(config.se3_config);

        block->set_sub_modules(
            std::move(msa_row), std::move(msa_col), std::move(msa_ff),
            std::move(pair_row), std::move(pair_col), std::move(pair_ff),
            std::move(tri_out), std::move(tri_in), std::move(se3));

        // PositionalEncoding (per block)
        auto pos_enc = std::make_unique<PositionalEncoding>();
        pos_enc->set_params(-32, 32, 8, config.d_pair, pos_enc_emb_res_[idx], pos_enc_emb_atom_[idx]);
        block->set_pos_enc(std::move(pos_enc));

        // GlobalColAttention (FullBlock only) — 处理 msa_full[64,...]，用 D_MSA_FULL=64 专属权重
        auto gcol = std::make_unique<MSAGlobalColAttention>();
        gcol->set_params(msa_ac,
            full_msa_global_col_to_b_[i], full_msa_global_col_to_g_[i], full_msa_global_col_to_out_[i],
            full_msa_global_col_Wq_[i], full_msa_global_col_Wk_[i], full_msa_global_col_Wv_[i]);
        block->set_global_col_attn(std::move(gcol));

        extra_blocks_.push_back(std::move(block));
    }

    // main_blocks (n_main) — IterBlock with update_msa_pair=true
    for (int i = 0; i < n_main; ++i) {
        int idx = n_extra + i;
        auto block = std::make_unique<IterBlock>(config, true);
        // 注入 3D SE + forward 内部参数
        block->norm_msa_3d_  = iter_norm_msa_3d_[idx]; block->norm_pair_3d_ = iter_norm_pair_3d_[idx];
        block->embed_x_      = iter_embed_x_[idx];     block->embed_e_      = iter_embed_e_[idx];
        block->norm_node_3d_ = iter_norm_node_3d_[idx]; block->norm_edge_3d_ = iter_norm_edge_3d_[idx];
        block->state2msa_norm_       = iter_state2msa_norm_[idx];
        block->state2msa_linear_     = iter_state2msa_linear_[idx];
        block->pair2msa_norm_        = iter_pair2msa_norm_[idx];
        // main_blocks(IterBlock) 处理 256 维 msa，msa2pair 用 256 维权重
        block->msa2pair_norm_        = iter_msa2pair_norm_[idx];
        block->msa2pair_left_proj_   = iter_msa2pair_left_proj_[idx];
        block->msa2pair_right_proj_  = iter_msa2pair_right_proj_[idx];
        block->msa2pair_out_proj_    = iter_msa2pair_out_proj_[idx];
        block->pair2pair_rbf_proj_   = iter_pair2pair_rbf_proj_[idx];
        block->pair2pair_state_norm_ = iter_pair2pair_state_norm_[idx];
        block->pair2pair_left_proj_  = iter_pair2pair_left_proj_[idx];
        block->pair2pair_right_proj_ = iter_pair2pair_right_proj_[idx];
        block->pair2pair_gate_proj_  = iter_pair2pair_gate_proj_[idx];

        // 创建并注入子模块 (layernorm 在 IterBlock::forward 外部完成)
        // main_blocks(IterBlock) 处理 256 维 msa，用 D_MSA=256 权重
        auto msa_row = std::make_unique<MSARowAttention>();
        msa_row->set_params(msa_ac,
            msa_row_to_b_[idx], msa_row_to_g_[idx], msa_row_to_out_[idx],
            msa_row_Wq_[idx], msa_row_Wk_[idx], msa_row_Wv_[idx]);
        auto msa_col = std::make_unique<MSAColAttention>();
        msa_col->set_params(msa_ac,
            msa_col_to_b_[idx], msa_col_to_g_[idx], msa_col_to_out_[idx],
            msa_col_Wq_[idx], msa_col_Wk_[idx], msa_col_Wv_[idx]);
        auto msa_ff = std::make_unique<FeedForward>();
        msa_ff->set_params(D_MSA, D_MSA * 4, 0.1f, msa_ff_norm_[idx], msa_ff_linear1_[idx], msa_ff_linear2_[idx]);
        auto pair_row = std::make_unique<PairRowAttention>();
        pair_row->set_params(pair_ac, pair_attn_norm_[idx],
            pair_row_to_b_[idx], pair_row_to_g_[idx], pair_row_to_out_[idx],
            pair_row_Wq_[idx], pair_row_Wk_[idx], pair_row_Wv_[idx]);
        auto pair_col = std::make_unique<PairColAttention>();
        pair_col->set_params(pair_ac, pair_attn_norm_[idx],
            pair_col_to_b_[idx], pair_col_to_g_[idx], pair_col_to_out_[idx],
            pair_col_Wq_[idx], pair_col_Wk_[idx], pair_col_Wv_[idx]);
        auto pair_ff = std::make_unique<FeedForward>();
        pair_ff->set_params(D_PAIR, D_PAIR * 2, 0.1f, pair_ff_norm_[idx], pair_ff_linear1_[idx], pair_ff_linear2_[idx]);
        auto tri_out = std::make_unique<TriangleMultiplication>();
        tri_out->set_params(D_PAIR,
            tri_out_layernorm_[idx], tri_out_left_proj_[idx], tri_out_right_proj_[idx],
            tri_out_left_gate_[idx], tri_out_right_gate_[idx], tri_out_gate_[idx],
            tri_out_output_layernorm_[idx], tri_out_out_proj_[idx]);
        auto tri_in = std::make_unique<TriangleMultiplication>();
        tri_in->set_params(D_PAIR,
            tri_in_layernorm_[idx], tri_in_left_proj_[idx], tri_in_right_proj_[idx],
            tri_in_left_gate_[idx], tri_in_right_gate_[idx], tri_in_gate_[idx],
            tri_in_output_layernorm_[idx], tri_in_out_proj_[idx]);
        auto se3 = std::make_unique<SE3Transformer>(config.se3_config);

        block->set_sub_modules(
            std::move(msa_row), std::move(msa_col), std::move(msa_ff),
            std::move(pair_row), std::move(pair_col), std::move(pair_ff),
            std::move(tri_out), std::move(tri_in), std::move(se3));

        // PositionalEncoding (per block)
        auto pos_enc = std::make_unique<PositionalEncoding>();
        pos_enc->set_params(-32, 32, 8, config.d_pair, pos_enc_emb_res_[idx], pos_enc_emb_atom_[idx]);
        block->set_pos_enc(std::move(pos_enc));

        main_blocks_.push_back(std::move(block));
    }

    // refine_blocks (4) — RefineBlock with update_msa_pair=false
    for (int i = 0; i < n_refn; ++i) {
        auto block = std::make_unique<RefineBlock>(config, false);
        // 注入 RefineBlock 专属 3D SE 参数
        block->norm_msa_   = refine_norm_msa_[i];        // 256
        block->norm_pair_  = refine_norm_pair_[i];       // 128
        block->norm_state_ = refine_norm_state_[i];      // 32
        block->embed_x_    = refine_embed_x_[i];         // 309→32
        block->norm_node_  = refine_norm_node_[i];       // 32
        block->embed_e1_   = refine_embed_e1_[i];        // 128→32
        block->norm_edge1_ = refine_norm_edge1_[i];      // 32
        block->embed_e2_   = refine_embed_e2_[i];        // 97→32
        block->norm_edge2_ = refine_norm_edge2_[i];      // 32
        // RefineBlock 继承 IterBlock 的 forward 内部参数不需要 (update_msa_pair=false)
        // 但 SE3(3D) track 需要 se3_ 子模块（值版 RefineBlock::forward 的 se3_->forward）。
        // 前 8 个参数传空 unique_ptr 安全（RefineBlock::forward_graph 为 pass-through 不使用）。
        auto se3_r = std::make_unique<SE3Transformer>(config.se3_config);
        block->set_sub_modules(
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            std::move(se3_r));
        refine_blocks_.push_back(std::move(block));
    }

    // ===== embedding / track / template 全局参数 =====
    // 旧栈上变量 (保留注释):
    // BondEmbedding bond_embed(0, D_PAIR); FullEmbedding full_emb(NAATOKENS-1+4, D_MSA_FULL);
    // LinearLayer linear(feat_dim, dim_) in MSATrack
    // EmbeddingLayer embedding(NAATOKENS, D_STATE) in StateTrack
    // EmbeddingLayer emb_left/right(NAATOKENS, D_PAIR) in PairTrack
    // LinearLayer emb_t1d(110,64) in StateTrack::inject_template
    // LinearLayer t1d_proj(80,32) / LayerNorm(64) in PairTrack::templ_stack
    bond_emb_    = LinearLayer::create(8, D_PAIR);                                   // NBYTES → D_PAIR (128)
    full_linear_ = LinearLayer::create(NAATOKENS - 1 + 4, D_MSA_FULL);              // 83 → 64
    full_emb_    = EmbeddingLayer::create(NAATOKENS, D_MSA_FULL);                    // (80, 64)
    msa_emb_              = LinearLayer::create(MSA_LATENT_DIM, D_MSA);              // 164 → 256
    state_emb_            = EmbeddingLayer::create(NAATOKENS, D_STATE);              // (80, 32)
    pair_left_emb_        = EmbeddingLayer::create(NAATOKENS, D_PAIR);              // (80, 128)
    pair_right_emb_       = EmbeddingLayer::create(NAATOKENS, D_PAIR);              // (80, 128)
    pair_init_pos_enc_    = new PositionalEncoding();                                 // pair 初始化 pos enc
    pair_init_pos_enc_emb_res_  = EmbeddingLayer::create(65, D_PAIR);
    pair_init_pos_enc_emb_atom_ = EmbeddingLayer::create(17, D_PAIR);
    pair_init_pos_enc_->set_params(-32, 32, 8, D_PAIR,
                                    pair_init_pos_enc_emb_res_, pair_init_pos_enc_emb_atom_);
    emb_t1d_              = LinearLayer::create(D_T1D + D_TOR, 64);                  // 110 → 64
    proj_t1d_             = LinearLayer::create(64, 64);                             // 64 → 64
    emb_t1d_t2d_          = LinearLayer::create(D_T1D * 2 + D_T2D, D_PAIR);         // 228 → 128（模板 pair=D_PAIR）
    temp_stack_t1d_proj_  = LinearLayer::create(D_T1D, D_STATE);                    // 80 → 32
    temp_stack_norm_      = LayerNorm::create(D_PAIR);                               // 128
    // Template state cross-attention 投影 (CrossAttention 外部注入, 无 bias):
    //   proj_dim = n_head * head_dim = 8 * ceil(max(32,64)/8)=8 → 64
    templ_attn_Wq_        = LinearLayer::create(D_STATE, 64, false);                // 32 → 64
    templ_attn_Wk_        = LinearLayer::create(64, 64, false);                     // 64 → 64
    templ_attn_Wv_        = LinearLayer::create(64, 64, false);                     // 64 → 64
    templ_attn_Wo_        = LinearLayer::create(64, D_STATE, false);                // 64 → 32
    // Template pair→pair cross-attention (CrossAttention(D_PAIR,D_PAIR,8)): proj_dim=128
    templ_pair_attn_Wq_   = LinearLayer::create(D_PAIR, 128, false);                // 128 → 128
    templ_pair_attn_Wk_   = LinearLayer::create(D_PAIR, 128, false);                // 128 → 128
    templ_pair_attn_Wv_   = LinearLayer::create(D_PAIR, 128, false);                // 128 → 128
    templ_pair_attn_Wo_   = LinearLayer::create(128, D_PAIR, false);                // 128 → 128

    // ===== 输出头参数 (全局单份) =====
    // Masked MSA head: LayerNorm(D_MSA) → Linear(D_MSA→D_MSA) → ReLU → Linear(D_MSA→23)
    // 输入 msa 特征 (B,N,L,D_MSA)，输出 logits (B,N,L,23)（每个序列位置/残基的 aatype 预测）
    msa_head_ln_      = LayerNorm::create(D_MSA);        // 256
    msa_head_linear1_ = LinearLayer::create(D_MSA, D_MSA);   // 256 → 256
    msa_head_linear2_ = LinearLayer::create(D_MSA, 23);      // 256 → 23

    // Chi (扭转角) head: LayerNorm(D_STATE) → Linear(D_STATE→D_STATE) → ReLU → Linear(D_STATE→14)
    // 输入 state (B,L,D_STATE)，输出 alpha (B,L,7,2)（omega/phi/psi/chi1-4 未归一化 sin/cos）
    chi_head_ln_      = LayerNorm::create(D_STATE);      // 32
    chi_head_linear1_ = LinearLayer::create(D_STATE, D_STATE);  // 32 → 32
    chi_head_linear2_ = LinearLayer::create(D_STATE, 7 * 2);    // 32 → 14

    // Distogram head: 从 pair 特征投影 4 组 logits (D/Ω/Θ/Φ)
    distogram_pair_ln_ = LayerNorm::create(D_PAIR);       // 128 LayerNorm(pair)，投影前归一化
    distogram_d_head_ = LinearLayer::create(D_PAIR, 60);  // 距离 60 bins
    distogram_o_head_ = LinearLayer::create(D_PAIR, 36);  // Ω 36 bins
    distogram_t_head_ = LinearLayer::create(D_PAIR, 36);  // Θ 36 bins
    distogram_p_head_ = LinearLayer::create(D_PAIR, 18);  // Φ 18 bins

    if (getenv("GRAPH_DEBUG_GALLOCR")) {
        auto dw = [](LinearLayer* h, const char* tag) {
            TensorF32* wt = h->weight();
            const float* w = wt ? wt->data() : nullptr;
            if (!w) { fprintf(stderr, "[headw] %s weight data=null\n", tag); return; }
            bool nan=false; float mn=1e30f,mx=-1e30f; long nnan=0;
            for (int i=0;i<wt->numel();++i){ float v=w[i]; if(v!=v){nan=true;++nnan;} mn=std::min(mn,v); mx=std::max(mx,v);}
            fprintf(stderr, "[headw] %s w_numel=%d nan=%d nnan=%ld min=%f max=%f w[0]=%f w[1]=%f\n",
                    tag, (int)wt->numel(), (int)nan, nnan, mn, mx, w[0], w[1]);
        };
        dw(distogram_d_head_, "dist");
        dw(distogram_o_head_, "omega");
        dw(distogram_t_head_, "theta");
        dw(distogram_p_head_, "phi");
    }

    // pLDDT head: state → lddt logits (B,L,50)
    plddt_head_       = LinearLayer::create(D_STATE, 50);

    // ===== TemplatePairStack 子层创建 =====
    // 直接层
    tps_rbf_proj_   = LinearLayer::create(D_RBF, D_PAIR);                            // 64 → 128
    tps_state_norm_ = LayerNorm::create(D_STATE);                                     // 32
    tps_left_proj_  = LinearLayer::create(D_STATE, 16);                               // 32 → 16
    tps_right_proj_ = LinearLayer::create(D_STATE, 16);                               // 32 → 16
    tps_gate_proj_  = LinearLayer::create(16 * 16, D_RBF);                            // 256 → 64（gate 用于逐元素更新 rbf_feature，须与 D_RBF 同维）
    // TriangleMultiplication out
    tps_tri_out_layernorm_        = LayerNorm::create(D_PAIR);                        // 128
    tps_tri_out_left_proj_        = LinearLayer::create(D_PAIR, 128);
    tps_tri_out_right_proj_       = LinearLayer::create(D_PAIR, 128);
    tps_tri_out_left_gate_        = LinearLayer::create(D_PAIR, 128);
    tps_tri_out_right_gate_       = LinearLayer::create(D_PAIR, 128);
    tps_tri_out_gate_             = LinearLayer::create(D_PAIR, D_PAIR);              // 128 → 128
    tps_tri_out_output_layernorm_ = LayerNorm::create(128);
    tps_tri_out_out_proj_         = LinearLayer::create(128, D_PAIR);
    tps_tri_mul_out_.set_params(D_PAIR,
        tps_tri_out_layernorm_,        tps_tri_out_left_proj_,
        tps_tri_out_right_proj_,       tps_tri_out_left_gate_,
        tps_tri_out_right_gate_,       tps_tri_out_gate_,
        tps_tri_out_output_layernorm_, tps_tri_out_out_proj_);
    // TriangleMultiplication in
    tps_tri_in_layernorm_        = LayerNorm::create(D_PAIR);                        // 128
    tps_tri_in_left_proj_        = LinearLayer::create(D_PAIR, 128);
    tps_tri_in_right_proj_       = LinearLayer::create(D_PAIR, 128);
    tps_tri_in_left_gate_        = LinearLayer::create(D_PAIR, 128);
    tps_tri_in_right_gate_       = LinearLayer::create(D_PAIR, 128);
    tps_tri_in_gate_             = LinearLayer::create(D_PAIR, D_PAIR);              // 128 → 128
    tps_tri_in_output_layernorm_ = LayerNorm::create(128);
    tps_tri_in_out_proj_         = LinearLayer::create(128, D_PAIR);
    tps_tri_mul_in_.set_params(D_PAIR,
        tps_tri_in_layernorm_,        tps_tri_in_left_proj_,
        tps_tri_in_right_proj_,       tps_tri_in_left_gate_,
        tps_tri_in_right_gate_,       tps_tri_in_gate_,
        tps_tri_in_output_layernorm_, tps_tri_in_out_proj_);
    // PairRowAttention (AF2 PairAxialAttention: 投影前 LayerNorm(pair))
    tps_pair_norm_       = LayerNorm::create(D_PAIR);                                // 128
    tps_pair_row_to_b_   = LinearLayer::create(D_PAIR, 8);                           // 128 → 8
    tps_pair_row_to_g_   = LinearLayer::create(D_PAIR, 256);                         // 128 → 256
    tps_pair_row_to_out_ = LinearLayer::create(256, D_PAIR);                         // 256 → 128
    tps_pair_row_Wq_     = LinearLayer::create(D_PAIR, 256);                         // 128 → 256
    tps_pair_row_Wk_     = LinearLayer::create(D_PAIR, 256);
    tps_pair_row_Wv_     = LinearLayer::create(D_PAIR, 256);
    tps_pair_row_attn_.set_params(AttnConfig(D_PAIR, 8), tps_pair_norm_,
        tps_pair_row_to_b_, tps_pair_row_to_g_, tps_pair_row_to_out_,
        tps_pair_row_Wq_,   tps_pair_row_Wk_,   tps_pair_row_Wv_);
    // PairColAttention
    tps_pair_col_to_b_   = LinearLayer::create(D_PAIR, 8);                           // 128 → 8
    tps_pair_col_to_g_   = LinearLayer::create(D_PAIR, 256);                         // 128 → 256
    tps_pair_col_to_out_ = LinearLayer::create(256, D_PAIR);                         // 256 → 128
    tps_pair_col_Wq_     = LinearLayer::create(D_PAIR, 256);                         // 128 → 256
    tps_pair_col_Wk_     = LinearLayer::create(D_PAIR, 256);
    tps_pair_col_Wv_     = LinearLayer::create(D_PAIR, 256);
    tps_pair_col_attn_.set_params(AttnConfig(D_PAIR, 8), tps_pair_norm_,
        tps_pair_col_to_b_, tps_pair_col_to_g_, tps_pair_col_to_out_,
        tps_pair_col_Wq_,   tps_pair_col_Wk_,   tps_pair_col_Wv_);
    // FeedForward
    tps_pair_ff_norm_    = LayerNorm::create(D_PAIR);                                // 128
    tps_pair_ff_linear1_ = LinearLayer::create(D_PAIR, D_PAIR * 2);                  // 128 → 256
    tps_pair_ff_linear2_ = LinearLayer::create(D_PAIR * 2, D_PAIR);                  // 256 → 128
    tps_pair_ff_.set_params(D_PAIR, 2, 0.15f,
        tps_pair_ff_norm_, tps_pair_ff_linear1_, tps_pair_ff_linear2_);
    // TemplatePairStack 本身
    tps_.set_params(
        tps_rbf_proj_, tps_state_norm_,
        tps_left_proj_, tps_right_proj_, tps_gate_proj_,
        &tps_tri_mul_out_, &tps_tri_mul_in_,
        &tps_pair_row_attn_, &tps_pair_col_attn_,
        &tps_pair_ff_);
}

PPMLModel::~PPMLModel() = default;

void PPMLModel::set_seq_info(const TensorF32& seq1hot, const TensorI64& idx) {
    // 值版：成员默认构造 numel=1，须先按源重建再 copy_from
    seq1hot_.~TensorF32();
    new (&seq1hot_) TensorF32(seq1hot.shape(), seq1hot.device());
    seq1hot_.copy_from(seq1hot);
    idx_.~TensorI64();
    new (&idx_) TensorI64(idx.shape(), idx.device());
    idx_.copy_from(idx);
    has_seq_info_ = true;
}

ModelOutput PPMLModel::forward(const ModelInput& input) {
    ModelOutput output;
    
    int B = input.msa_latent.shape().dims[0];
    int N = input.msa_latent.shape().dims[1];
    int L = input.msa_latent.shape().dims[2];
    
    // 初始化 tracks
    msa_track_ = std::make_unique<MSATrack>(N, L, config_.d_msa, device_);
    pair_track_ = std::make_unique<PairTrack>(L, config_.d_pair, device_);
    state_track_ = std::make_unique<StateTrack>(L, config_.d_state, device_);
    
    // 旧: msa_track_->init_from_features(input.msa_latent); → 用 msa_emb_ 直接投影
    msa_track_->representation() = msa_emb_->forward(input.msa_latent);                               // (B,N,L,164→256)
    // 旧: state_track_->init_from_embedding(input.seq_tokens); → 用 state_emb_
    state_track_->representation() = state_emb_->forward_exec(input.seq_tokens);                        // (B,L)→(B,L,32)
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[VFWD] tracks init OK (msa=%lld state=%lld)\n",
        (long long)msa_track_->representation().numel(), (long long)state_track_->representation().numel());
    // 旧: pair_track_->init_from_embedding(...); → 用 pair_left_emb_/pair_right_emb_
    {
        auto left  = pair_left_emb_->forward_exec(input.seq_tokens).unsqueeze(1);            // (B,1,L,128)
        auto right = pair_right_emb_->forward_exec(input.seq_tokens).unsqueeze(2);           // (B,L,1,128)
        auto pair_repr = outer_sum(left, right);                                              // (B,L,L,128)
        // PositionalEncoding: 使用 input 中预计算的 bond_feats/dist_matrix/same_chain/residx
        // idx 需要从 TensorI64 转为 TensorF32
        TensorF32 residx_f32(input.residx.shape());
        for (int64_t i = 0; i < input.residx.numel(); ++i)
            residx_f32.data()[i] = static_cast<float>(input.residx.data()[i]);
        auto pos_out = pair_init_pos_enc_->forward(pair_repr, residx_f32, input.bond_feats, input.dist_matrix, input.same_chain);
        //  值版 forward 用值加法，不能用图 helper add_impl（返回 OP_ADD 图节点 data()=nullptr，
        //    copy_from 读 null 崩，PPML.cpp:2124 SIGSEGV）。pair_repr/pos_out 都是值张量。
        //    Tensor 是 move-only，不能 `TensorF32 pair_init = pair_repr`（拷贝构造已删除），
        //    须先构造空张量再 copy_from 拷贝数据。
        TensorF32 pair_init(pair_repr.shape(), pair_repr.device());
        pair_init.copy_from(pair_repr);
        for (int64_t i = 0; i < pair_init.numel(); ++i)
            pair_init.data()[i] += pos_out.data()[i];
        pair_track_->representation().copy_from(pair_init);
        if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[VFWD] pair init OK\n");
    }

    // msa full embed?
    // msa_full = self.full_emb(msa_full, seq, idx)
    // msa_full was used in the full block
    TensorF32 msa_full;
    if (input.msa_full.numel() > 0) {
        // 旧栈上变量: FullEmbedding full_emb(NAATOKENS - 1 + 4, D_MSA_FULL);
        FullEmbedding full_emb;
        full_emb.set_params(full_linear_, full_emb_, D_MSA_FULL);
        msa_full = full_emb.forward(input.msa_full, input.seq_tokens, TensorF32());
    }

    // bond embed for pair track
    // need to get the bond feats
    // 旧栈上变量: BondEmbedding bond_embed(0, D_PAIR);
    BondEmbedding bond_embed;
    bond_embed.set_params(bond_emb_, D_PAIR);
    // 值版：先按 pair_track_ 形状构造 pair，再加 bond 特征（值加法，不用图 add_impl——
    // add_impl 返回 OP_ADD 图节点 data()=nullptr，copy_from 读 null 崩）。
    TensorF32 pair(pair_track_->representation().shape(), pair_track_->representation().device());
    pair.copy_from(pair_track_->representation());
    auto bond_out = bond_embed.forward(input.bond_feats);
    if (pair.numel() == bond_out.numel()) {
        float* pd = pair.data();
        const float* bd = bond_out.data();
        for (int64_t i = 0; i < pair.numel(); ++i) pd[i] += bd[i];
    }
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[VFWD] embed OK (msa_full=%lld pair=%lld)\n",
        (long long)msa_full.numel(), (long long)pair.numel());
    //bond_feats: (B, L, L, d_init)
    //bond embed: 
    // bond_feats = one_hot(bond_feats)
    // linear(d_bond_type = 5, d_pair = 128)
    // linear(bond_feats.float())

    // recycle embed?
    
    // template embed
    // Template injection
    // cross attention need to reshape
    // state cross attention and pair cross attention
    //  模板判断与图版一致：t1d 是 (B,T,L,80) 4D 且有 T>0 才注入；占位空张量（shape=() numel=1）跳过。
    if (input.t1d.shape().ndim() == 4 && input.t1d.shape().dims[1] > 0 && input.t1d.numel() > 0) {
        // 旧: state_track_->inject_template(input.t1d, input.tor_feat);
        // 旧栈上: LinearLayer emb_t1d(110,64), proj_t1d(64,64) → 用 emb_t1d_/proj_t1d_
        {
            // 值版：先按源形状构造再 copy（tor_feat 可能空，需防护）
            TensorF32 t1d_copy(input.t1d.shape(), input.t1d.device());
            t1d_copy.copy_from(input.t1d);
            TensorF32 tor_copy;
            if (input.tor_feat.numel() > 0 && input.tor_feat.data()) {
                tor_copy.~TensorF32();
                new (&tor_copy) TensorF32(input.tor_feat.shape(), input.tor_feat.device());
                tor_copy.copy_from(input.tor_feat);
            } else {
                new (&tor_copy) TensorF32();  // 空：concat 时跳过
            }
            // 值版：手写 concat（concat_ptr 是图版返回图节点 data()=nullptr），tor_feat 空则跳过
            const int64_t D_T1D_ALL = D_T1D + (input.tor_feat.numel() > 0 ? input.tor_feat.shape().dims[3] : 0);
            TensorF32 t1d_tor(Shape({t1d_copy.shape().dims[0], t1d_copy.shape().dims[1],
                                     t1d_copy.shape().dims[2], D_T1D_ALL}),
                              t1d_copy.device());
            {
                const float* td = t1d_copy.data();
                float* tod = t1d_tor.data();
                const int64_t rows = (int64_t)t1d_copy.shape().dims[0] * t1d_copy.shape().dims[1] * t1d_copy.shape().dims[2];
                for (int64_t r = 0; r < rows; ++r) {
                    for (int d = 0; d < D_T1D; ++d) tod[r * D_T1D_ALL + d] = td[r * D_T1D + d];
                }
                if (input.tor_feat.numel() > 0 && input.tor_feat.data()) {
                    const float* fr = input.tor_feat.data();
                    const int64_t D_tor = input.tor_feat.shape().dims[3];
                    for (int64_t r = 0; r < rows; ++r)
                        for (int d = 0; d < D_tor; ++d) tod[r * D_T1D_ALL + D_T1D + d] = fr[r * D_tor + d];
                }
            }
            TensorF32 t1d_emb = emb_t1d_->forward(t1d_tor);        // (B,T,L,64)
            t1d_emb = proj_t1d_->forward(relu_value(t1d_emb));     // (B,T,L,64)
            // Cross-attention: state as Q, template as K/V
            int B = t1d_emb.shape().dims[0], T = t1d_emb.shape().dims[1];
            auto state_q = state_track_->representation().view({B * L, 1, D_STATE});
            auto t1d_kv = t1d_emb.permute({0, 2, 1, 3}).view({B * L, T, 64});
            CrossAttention cross_attn(D_STATE, 64, 8);  // Q=state(32), KV=template(64), H=head(8)
            auto out = cross_attn.forward(state_q, t1d_kv);  // query, key-value
            // residual connection: state_rep += out_view（值版逐元素加）
            auto& state_rep = state_track_->representation();
            auto out_view = out.view({B, L, D_STATE});
            float* rd = state_rep.data();
            const float* od = out_view.data();
            for (int64_t i = 0; i < state_rep.numel(); ++i) rd[i] += od[i];
        }
        TensorF32 templ_pair = get_templ_emb(input.t1d, input.t2d);  // (B,T,L,L,D_PAIR=128)
        // rbf_feature: 用初始 coords 计算 RBF 特征
        TensorF32 init_coords;
        init_coords.copy_from(input.coords);
        TensorF32 rbf_feature = IterBlock::compute_rbf_feature(init_coords);  // (B, L, L, 64)
        // 旧: pair_track_->templ_stack(templ_pair, rbf_feature, input.t1d);
        // templ_stack 1406-1418
        // 旧栈上: LinearLayer t1d_proj(80,32), LayerNorm(D_PAIR=128), TemplatePairStack
        {
            int T = templ_pair.shape().dims[1];
            TensorF32 t1d_2d = input.t1d.view({B*T, L, D_T1D});
            // (B*T, L, 80) — already a view above, no reshape needed
            templ_pair = reshape_value(templ_pair, Shape({B*T, L, L, D_PAIR}));  // (B*T, L, L, 128) 拥有数据
            // 旧栈上: LinearLayer t1d_proj(D_T1D, D_STATE) → temp_stack_t1d_proj_
            TensorF32 state_proj = temp_stack_t1d_proj_->forward(t1d_2d);      // (B*T, L, 32)
            for (int k = 0; k < 2; ++k) {
                templ_pair = tps_.forward(templ_pair, rbf_feature, state_proj);  // rbf_feature from outer scope
            }
            // 旧栈上: LayerNorm(D_PAIR=128) → temp_stack_norm_（值版 forward_exec）
            templ_pair = temp_stack_norm_->forward_exec(templ_pair);            // (B*T, L, L, 128)
            templ_pair = reshape_value(templ_pair, Shape({B, T, L, L, D_PAIR}));
            pair_track_->inject_template(templ_pair);
        }
    }
    
    // 获取初始表示（值版：先按源形状构造再 copy_from，避免默认构造 numel=1 shape mismatch）
    TensorF32 msa(msa_track_->representation().shape(), msa_track_->representation().device());
    msa.copy_from(msa_track_->representation());
    pair.~TensorF32();
    new (&pair) TensorF32(pair_track_->representation().shape(), pair_track_->representation().device());
    pair.copy_from(pair_track_->representation());
    TensorF32 state(state_track_->representation().shape(), state_track_->representation().device());
    state.copy_from(state_track_->representation());
    TensorF32 coords(input.coords.shape(), input.coords.device());
    coords.copy_from(input.coords);
    
    // need to modify : (already finished)
    // the block in the 4 full block was different from the main block
    // full/extra block use global column attention

    const TensorF32& seq1hot = one_hot_seq(input.seq_tokens, 21);
    set_seq_info(seq1hot, input.residx);

    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[VFWD] before extra blocks (n=%zu)\n", extra_blocks_.size());
    // Extra blocks
    // need to use msa_full
    // and use global column attention as well
    for (auto& block : extra_blocks_) {
        // stop grad
        block->forward(msa_full, pair, state, seq1hot, coords,
                       input.bond_feats, input.dist_matrix, input.same_chain, input.residx);
        coords.copy_from(block->updated_coords());
    }
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[VFWD] extra blocks done (n=%zu)\n", extra_blocks_.size());
    
    // Main blocks
    for (auto& block : main_blocks_) {
        // stop grad
        // chiral grad
        block->forward(msa, pair, state, seq1hot, coords,
                       input.bond_feats, input.dist_matrix, input.same_chain, input.residx);
        coords.copy_from(block->updated_coords());
    }
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[VFWD] main blocks done (n=%zu)\n", main_blocks_.size());
    
    // Refinement blocks (仅更新结构)
    for (auto& block : refine_blocks_) {
        // stop grad
        // chiral grad
        // clash grad
        /* if (block.get()) {
        // seq1hot: 从 input.seq_tokens 生成 one-hot (B, L, 21)
            TensorF32 seq1hot = one_hot_seq(input.seq_tokens, 21);
            TensorI64 idx = input.seq_tokens;  // 或专门的 idx 输入
            refine->set_seq_info(seq1hot, idx);
        } */

        //    不传则默认构造 TensorI64 shape 未初始化 → get_bonded_neigh 读 dims[0] SEGV。
        block->forward(msa, pair, state, seq1hot, coords,
                       input.bond_feats, input.dist_matrix, input.same_chain, input.residx);

        if (block.get()) {
            coords.copy_from(block->updated_coords());
            // state 通过 track 更新，不需要 updated_state()
        }
    }
    if (getenv("PPML_TRACE_VALUE")) fprintf(stderr, "[VFWD] refine blocks done (n=%zu)\n", refine_blocks_.size());
    
    // 输出头（值版：output 字段默认构造 numel=1，须先按源形状重建再 copy_from）
    output.msa.~TensorF32();
    new (&output.msa) TensorF32(msa.shape(), msa.device());
    output.msa.copy_from(msa);
    output.pair.~TensorF32();
    new (&output.pair) TensorF32(pair.shape(), pair.device());
    output.pair.copy_from(pair);
    output.state.~TensorF32();
    new (&output.state) TensorF32(state.shape(), state.device());
    output.state.copy_from(state);
    output.coords.~TensorF32();
    new (&output.coords) TensorF32(coords.shape(), coords.device());
    output.coords.copy_from(coords);

    // ===== Masked MSA head: msa (B,N,L,D_MSA) → logits (B,N,L,23) =====
    //   LayerNorm(D_MSA) → Linear(D_MSA→D_MSA) → ReLU → Linear(D_MSA→23)
    //   msa 值布局 (B,N,L,D_MSA)；LinearLayer::forward 输入须展平为 (B*N*L, D_MSA)
    {
        int msa_B = msa.shape().dims[0];
        int msa_N = msa.shape().dims[1];
        int msa_L = msa.shape().dims[2];
        int64_t flat_rows = (int64_t)msa_B * msa_N * msa_L;

        // 1) LayerNorm (逐 token 归一化, 保持 (B,N,L,D_MSA))——值版 forward_exec
        TensorF32 ln_out = msa_head_ln_->forward_exec(msa);              // (B,N,L,256)

        // 2) view 展平 → Linear1 → ReLU（值版）→ view 回 4D
        TensorF32 ln_flat(Shape({flat_rows, D_MSA}), ln_out.device());
        ln_flat.copy_from(ln_out);
        auto lin1  = msa_head_linear1_->forward(ln_flat);                 // (flat_rows, 256)
        TensorF32 relu1 = relu_value(lin1);
        // 3) view 回 (B,N,L,256) → Linear2 → logits (B*N*L, 23)
        TensorF32 relu4(Shape({msa_B, msa_N, msa_L, D_MSA}), relu1.device());
        relu4.copy_from(relu1);
        auto lin2  = msa_head_linear2_->forward(relu4);                   // (flat_rows, 23)
        // 4) view 回 (B,N,L,23)
        TensorF32 logits4(Shape({msa_B, msa_N, msa_L, 23}), lin2.device());
        logits4.copy_from(lin2);
        output.msa_logits = std::move(logits4);
    }

    // ===== Chi (扭转角) head: state (B,L,D_STATE) → alpha (B,L,7,2) =====
    //   LayerNorm(D_STATE) → Linear(D_STATE→D_STATE) → ReLU → Linear(D_STATE→14)
    //   state 值布局 (B,L,D_STATE)；LinearLayer::forward 输入须展平为 (B*L, D_STATE)
    if (state.numel() > 0) {
        int ch_B = state.shape().dims[0];
        int ch_L = state.shape().dims[1];
        int64_t ch_rows = (int64_t)ch_B * ch_L;

        // 1) LayerNorm (逐 token, 保持 (B,L,D_STATE))——值版 forward_exec
        TensorF32 ch_ln = chi_head_ln_->forward_exec(state);            // (B,L,32)

        // 2) view 扁平 → Linear1 → ReLU（值版）→ view 回 (B,L,32)
        TensorF32 ch_flat(Shape({ch_rows, D_STATE}), ch_ln.device());
        ch_flat.copy_from(ch_ln);
        auto ch_lin1 = chi_head_linear1_->forward(ch_flat);             // (B*L, 32)
        TensorF32 ch_relu = relu_value(ch_lin1);
        // 3) view 回 (B,L,32) → Linear2 → logits (B*L, 14)
        TensorF32 ch_relu4(Shape({ch_B, ch_L, D_STATE}), ch_relu.device());
        ch_relu4.copy_from(ch_relu);
        auto ch_lin2 = chi_head_linear2_->forward(ch_relu4);            // (B*L, 14)
        // 4) view 回 (B,L,7,2)
        TensorF32 alpha4(Shape({ch_B, ch_L, 7, 2}), ch_lin2.device());
        alpha4.copy_from(ch_lin2);
        output.alpha = std::move(alpha4);
    }

    // ===== Distogram head: pair (B,L,L,D_PAIR) → 4 组 logits =====
    //   distogram (B,L,L,60), omega (B,L,L,36), theta (B,L,L,36), phi (B,L,L,18)
    if (pair.numel() > 0) {
        // distogram 头投影前对 pair 做 LayerNorm（防 logits 巨大→softmax 退化→loss 卡死）——值版 forward_exec
        TensorF32 pair_normed = distogram_pair_ln_->forward_exec(pair);
        int dg_B = pair.shape().dims[0];
        int dg_L = pair.shape().dims[1];
        int64_t dg_rows = (int64_t)dg_B * dg_L * dg_L;

        // pair (B,L,L,D_PAIR) → 展平 (B*L*L, D_PAIR) 送入各 head
        TensorF32 pair_flat(Shape({dg_rows, D_PAIR}), pair.device());
        pair_flat.copy_from(pair_normed);

        auto project_logits = [&](LinearLayer* head, int bins) {
            auto logits2 = head->forward(pair_flat);              // (B*L*L, bins)
            TensorF32 logits4(Shape({dg_B, dg_L, dg_L, bins}), logits2.device());
            logits4.copy_from(logits2);
            return logits4;
        };
        output.distogram = std::move(project_logits(distogram_d_head_, 60));
        output.omega     = std::move(project_logits(distogram_o_head_, 36));
        output.theta     = std::move(project_logits(distogram_t_head_, 36));
        output.phi       = std::move(project_logits(distogram_p_head_, 18));
    }

    // ===== pLDDT head: state (B,L,D_STATE) → lddt logits (B,L,50) =====
    if (state.numel() > 0) {
        int pl_B = state.shape().dims[0];
        int pl_L = state.shape().dims[1];
        int64_t pl_rows = (int64_t)pl_B * pl_L;

        TensorF32 pl_flat(Shape({pl_rows, D_STATE}), state.device());
        pl_flat.copy_from(state);
        auto pl_logits2 = plddt_head_->forward(pl_flat);          // (B*L, 50)
        TensorF32 lddt4(Shape({pl_B, pl_L, 50}), pl_logits2.device());
        lddt4.copy_from(pl_logits2);
        output.lddt = std::move(lddt4);
    }

    return output;
}

// ===== PPMLModel::forward_graph (图模式前向，新增入口，不改 forward) =====
// 预处理 embedding 与 block 前向均使用 forward_graph 版本。
// 布局约定（ggml dims[0]=最内维）：
//   msa        : 图 [D_MSA, L, N, B]     = 值 (B, N, L, D_MSA)
//   pair       : 图 [D_PAIR, L, L, B]    = 值 (B, L, L, D_PAIR)
//   state      : 图 [D_STATE, L, B]      = 值 (B, L, D_STATE)
//   rbf        : 图 [D_RBF, L, L, B]     = 值 (B, L, L, D_RBF)
// 输入值张量包装为图 leaf；内部对输出头 graph_compute 回落为 ModelOutput 值张量。
// 注:
//   - PositionalEncoding::forward_graph 当前为占位（返回零图节点），pair 初始化不含位置编码；
//   - SE3 3D track 需"图外值回落"驱动（graph_compute(pair) → run_se3_structural），本入口
//     暂未驱动 SE3（block 的 forward_graph 在无结构输入时只跑 msa/pair 两条 track）。
GraphOutput PPMLModel::forward_graph(const ModelInput& input, bool enable_se3,
                                     const TensorF32* topo_coords) {
    GraphOutput go;

    const int B = input.msa_latent.shape().dims[0];
    const int N = input.msa_latent.shape().dims[1];
    const int L = input.msa_latent.shape().dims[2];
    const int64_t D_init = input.msa_latent.shape().dims[3];   // 164

    // ==== 1. 输入包装为图 leaf (ggml dims[0]=最内维) ====
    TensorF32* msa      = wrap_input_as_leaf(input.msa_latent, {D_init, L, N, B});  // [164,L,N,B]
    // seq_tokens (B,L) row-major → 扁平一维 [B*L]（k=b*L+l），供 get_rows 查表
    TensorF32* seq_flat = wrap_input_as_leaf(input.seq_tokens, {B * L});            // [B*L]
    TensorF32* seq2d    = wrap_input_as_leaf(input.seq_tokens, {L, B});             // [L,B]（PositionalEncoding 占位用）
    TensorF32* bond     = wrap_input_as_leaf(input.bond_feats, {L, L, B});          // [L,L,B]
    TensorF32* dist     = wrap_input_as_leaf(input.dist_matrix, {L, L, B});         // [L,L,B]
    // residx (TensorI64) → float 图 leaf [L,B]
    TensorF32 residx_f32(input.residx.shape());
    for (int64_t i = 0; i < input.residx.numel(); ++i)
        residx_f32.data()[i] = static_cast<float>(input.residx.data()[i]);
    TensorF32* residx = wrap_input_as_leaf(residx_f32, {L, B});                     // [L,B]

    // ==== 2. 预处理 embedding (forward_graph) ====
    // get_rows 查表: [D, B*L] → view 为 block 布局（存储兼容，k=b*L+l 与 [D,L,B] 扁平一致）
    // state = state_emb(seq) → [D_STATE, L, B]
    TensorF32* state = view(state_emb_->forward_graph(seq_flat), Shape{D_STATE, L, B});
    // msa = msa_emb(msa_latent) → [D_MSA, L, N, B]
    msa = msa_emb_->forward_graph(msa);
    // pair = left_emb(seq) ⊕ right_emb(seq) → [D_PAIR, L, L, B]
    TensorF32* left  = view(pair_left_emb_->forward_graph(seq_flat),  Shape{D_PAIR, L, B});   // [D_PAIR,L,B]
    TensorF32* right = view(pair_right_emb_->forward_graph(seq_flat), Shape{D_PAIR, L, B});
    TensorF32* pair  = outer_sum_graph(unsqueeze(left, 1), unsqueeze(right, 2)); // [D_PAIR,L,L,B]
    // PositionalEncoding（图版占位，返回零图节点）— 留后实现
    TensorF32* pos_out = pair_init_pos_enc_->forward_graph(seq2d, residx, bond, dist);
    pair = add_impl(pair, pos_out, /*inplace=*/false);

    // msa_full embedding (FullBlock / extra blocks 用) — 仅当输入有 msa_full
    TensorF32* msa_full = nullptr;
    if (input.msa_full.numel() > 0) {
        const int64_t D_full = input.msa_full.shape().dims[3];   // 83
        const int N_full     = input.msa_full.shape().dims[1];
        TensorF32* msa_full_g = wrap_input_as_leaf(input.msa_full, {D_full, L, N_full, B});
        FullEmbedding full_emb;
        full_emb.set_params(full_linear_, full_emb_, D_MSA_FULL);
        msa_full = full_emb.forward_graph(msa_full_g, seq_flat, residx);
    }

    // ==== 3. rbf 特征注入（值版 compute_rbf_feature → 常量图 leaf）====
    TensorF32 rbf_val;
    if (input.coords.numel() > 0) {
        rbf_val = IterBlock::compute_rbf_feature(input.coords);  // (B,L,L,D_RBF)
    } else {
        rbf_val = zeros<float>({B, L, L, D_RBF}, Device::CPU);
    }
    TensorF32* rbf = constant_tensor({D_RBF, L, L, B}, rbf_val.data());   // [D_RBF,L,L,B]

    // ==== 3.5 模板特征接入 (t1d/t2d/tor_feat) — 值张量包装成图 leaf，全程图节点 ====
    // 仅在输入含模板特征时注入。两分支，遵循 4D 图布局 (dims[0]=最内维)。
    //   state 分支: 用全部 T 模板（T 维作 cross-attn key）
    //   pair 分支: 用 t=0 单模板（值版 PairTrack::inject_template 为 T=1 语义；图基础设施 4D 上限
    //              无法在不新增 5D 归约下遍历多模板，故取首模板）。
    //              (sum/mean 只支持全归约, sum_rows 只沿 dims[1]) 与 5D permute，故暂用 t=0。
    //              后续可：① 新增沿任意维的 reduce op；② 或把 templ_pair [64,L,L,T] 经 permute
    //              重排为 [64,1,1,L*L*T] 作为多模板 kv（T 折叠进 key 长度），query 仍为 B*L*L。
    //              需同步在值版 PairTrack::inject_template 保持语义一致。
    // 模板分支仅当 t1d 为合法 4D 模板特征 (B,T,L,80) 时进入。不能只判 numel()>0：
    // load_from_files 在无模板时可能给 t1d 留 1 元素占位（非空 4D），直接访问 dims[1] 会越界。
    const bool has_tmpl = (getenv("PPML_DISABLE_TEMPLATE") == nullptr
                           && input.t1d.shape().ndim() == 4
                           && input.t1d.shape().dims[1] > 0
                           && input.t1d.numel() > 0);
    if (has_tmpl) {
        // ---- (a) 输入包装: 值张量 → 图 leaf ----
        // state 与 pair 分支统一用 t=0 单模板（CrossAttention 单 key 设计）。
        // 值 (B,L,80)/(B,L,L,68)/(B,L,30) → 图 [80,L,B]/[68,L,L,B]/[30,L,B]
        TensorF32 t1d_t0 = input.t1d.select(1, 0);                              // 值 (B,L,80)
        TensorF32 t2d_t0 = input.t2d.select(1, 0);                              // 值 (B,L,L,68)
        TensorF32* t1d_t0_g = wrap_input_as_leaf(t1d_t0, {D_T1D, L, B});        // [80,L,B]
        TensorF32* t2d_t0_g = wrap_input_as_leaf(t2d_t0, {D_T2D, L, L, B});     // [68,L,L,B]

        // ---- (b) state 分支: template cross-attention (state 为 Q, 模板为 K/V) ----
        // 注: CrossAttention::forward_graph 为单 key (T=1) 设计（merge 时 view 掉 T 维），
        //     故 state 分支也用 t=0 单模板，与 pair 分支一致。多模板 (T>1) 需先扩展
        // concat(t1d_t0, tor_t0) 沿最内维 → [110,L,B]
        TensorF32 tor_t0 = input.tor_feat.select(1, 0);                          // 值 (B,L,30)
        TensorF32* tor_t0_g = wrap_input_as_leaf(tor_t0, {D_TOR, L, B});         // [30,L,B]
        std::vector<TensorF32*> t1d_parts = {t1d_t0_g, tor_t0_g};
        TensorF32* t1d_tor = concat_ptr(t1d_parts, 0);                           // [110,L,B]（沿 dims[0] 最内特征维）
        TensorF32* t1d_emb = emb_t1d_->forward_graph(t1d_tor);                   // [64,L,B]
        t1d_emb = relu(t1d_emb);
        t1d_emb = proj_t1d_->forward_graph(t1d_emb);                             // [64,L,B]
        // kv = 值 (B*L,1,64) = 图 [64,1,1,B*L]（T=1；view 合并 L*B 到 batch）
        TensorF32* t1d_kv = view(t1d_emb, Shape{64, 1, 1, B * L});
        // query = state 值 (B,L,32) → 值 (B*L,1,32) = 图 [32,1,1,B*L]
        TensorF32* state_q = view(state, Shape{D_STATE, 1, 1, B * L});
        // 用模型成员投影的 CrossAttention（持久化参数，供权重加载/优化）
        CrossAttention templ_attn(D_STATE, 64, 8);
        templ_attn.set_params(templ_attn_Wq_, templ_attn_Wk_, templ_attn_Wv_, templ_attn_Wo_);
        TensorF32* templ_out = templ_attn.forward_graph(state_q, t1d_kv);       // [32,1,1,B*L]
        TensorF32* templ_add = view(templ_out, Shape{D_STATE, L, B});            // [32,L,B]
        state = add_impl(state, templ_add, /*inplace=*/false);                   // residual

        // ---- (c) pair 分支: template pair stack (t=0 单模板, 4D) ----
        // get_templ_emb 图化: t2d + repeat(left t1d) + repeat(right t1d) → [228,L,L,B]
        TensorF32* t1d_left  = unsqueeze(t1d_t0_g, 1);                          // [80,1,L,B]
        TensorF32* t1d_right = unsqueeze(t1d_t0_g, 2);                          // [80,L,1,B]
        int64_t tpl_tgt[4] = {D_T1D, L, L, B};
        TensorF32* t1d_l_exp = repeat(t1d_left,  context().new_tensor<float>(4, tpl_tgt));  // [80,L,L,B]
        TensorF32* t1d_r_exp = repeat(t1d_right, context().new_tensor<float>(4, tpl_tgt));  // [80,L,L,B]
        std::vector<TensorF32*> templ_parts = {t2d_t0_g, t1d_l_exp, t1d_r_exp};
        TensorF32* templ_pair = concat_ptr(templ_parts, 0);                     // [228,L,L,B]（沿 dims[0] 最内特征维）
        templ_pair = emb_t1d_t2d_->forward_graph(templ_pair);                   // [D_PAIR=128,L,L,B]
        // state_proj: temp_stack_t1d_proj(t1d_t0) → [32,L,B]
        TensorF32* state_proj = temp_stack_t1d_proj_->forward_graph(t1d_t0_g);  // [32,L,B]
        // TemplatePairStack ×2（rbf 引用，内部 gate 更新；用局部副本指针避免污染主 rbf 常量）
        TensorF32* rbf_tpl = rbf;                                                // [D_RBF,L,L,B]
        for (int k = 0; k < 2; ++k)
            templ_pair = tps_.forward_graph(templ_pair, rbf_tpl, state_proj);    // [D_PAIR=128,L,L,B]
        templ_pair = temp_stack_norm_->forward(templ_pair);                      // [D_PAIR=128,L,L,B]
        // 注入到 pair: pair 为 Q, templ_pair 为 K/V 的 cross-attention（模型成员投影）。
        //   pair 值 (B,L,L,128) → 图 [128,1,1,B*L*L]（query）
        //   templ_pair 值 (B,L,L,128) → 图 [128,1,1,B*L*L]（kv，T 归并到单模板）
        TensorF32* pair_q = view(pair, Shape{D_PAIR, 1, 1, B * L * L});          // [128,1,1,B*L*L]
        TensorF32* tpl_kv = view(templ_pair, Shape{D_PAIR, 1, 1, B * L * L});    // 值 (B*L*L,1,128)
        CrossAttention templ_pair_attn(D_PAIR, D_PAIR, 8);
        templ_pair_attn.set_params(templ_pair_attn_Wq_, templ_pair_attn_Wk_,
                                   templ_pair_attn_Wv_, templ_pair_attn_Wo_);
        TensorF32* pair_out = templ_pair_attn.forward_graph(pair_q, tpl_kv);    // [128,1,1,B*L*L]
        TensorF32* pair_add = view(pair_out, Shape{D_PAIR, L, L, B});            // [128,L,L,B]
        pair = add_impl(pair, pair_add, /*inplace=*/false);                      // 注入模板 pair
    }

    // ==== 4. block 前向 (forward_graph) + SE3 3D track（图外值回落驱动）====
    ensure_backend_ready();
    PPMLContext* ctx = &context();
    Backend* backend = cpu_backend_.get();

    // 【阶段1.5】SE3 offset 回落的独立 backend（独立 gallocr_，不碰主图）。
    // 惰性创建一次并持久复用（同一模型多次 forward 不重建；run_se3=false 时保持空）。
    if (enable_se3 && !se3_backend_) {
        se3_backend_ = CPUBackend::create(1);
    }
    Backend* se3_bk = se3_backend_ ? se3_backend_.get() : backend;

    // SE3 需结构常量：seq1hot（值 (B,L,21)）与链式 coords。
    // 有 coords 时驱动 SE3；无 coords 则纯 1D/2D track（与调用方约定一致）。
    // has_struct: 是否有结构（coords 存在）。rbf 常量仍用 input.coords 计算（在 block 前）。
    // run_se3: 是否真正驱动 SE3 3D track。enable_se3=false 时跳过（供模板注入验证等，
    //   规避 SE3 训练驱动未完成的崩溃；rbf 距离特征不受影响）。
    const bool has_struct = (input.coords.numel() > 0);
    const bool run_se3    = enable_se3 && has_struct;
    // 【开关A/B】SE3 拓扑模式：
    //   PPML_SE3_TOPO=per_block → 开关B：逐 block 用最新 coords 构图（精度高，但需每 block 回落 offset，
    //    跨 cgraph 冲突 → 偶发 NaN）。
    //   PPML_SE3_TOPO 缺省/其它 → 开关A（fixed）：所有 block 用初始 coords 构图（拓扑固定），block 循环后
    //    主图一次 graph_compute 物化全部 offset/state，再从主图 buffer 读 offset 链式更新 coords。
    //    无独立回落 compute → 无跨 cgraph 冲突 → 根治偶发 NaN（代价：拓扑用初始坐标，长程接触边略糙）。
    // 【refined topo pass：L1/L2】（2026-09-23）新增两个模式（设计/数学见 TopoPassRefinement.md §6.3 + 附录 A ✓）：
    //   PPML_SE3_TOPO=l1 → **冻结边索引**（首次边界算一次）+ 每边界用**实时坐标**重算几何 ✓
    //   PPML_SE3_TOPO=l2 → 上述 + **margin 门控局部重算**（Ω = {i : ‖x_i−x⁰_i‖ ≥ κ_i}，κ_i = σ·m_i/2 ✓）
    //   两者都沿用 per_block 的**驱动**逻辑（每边界回落 offset + 链式更新 coords ✓），只把"构图那一步"换掉 ✓
    //   调参：PPML_TOPO_SIGMA（默认 0.5 ✓）；PPML_TOPO_L2_CLOSURE=0 关 Ω 邻居封闭（默认开 ✓）；
    //         PPML_TOPO_DEBUG=1 打印 [TOPO-L1/L2] 快照/|Ω|/对称差统计 ✓
    const char* topo_env = std::getenv("PPML_SE3_TOPO");
    const bool  topo_l1  = run_se3 && topo_env && std::strcmp(topo_env, "l1") == 0;
    const bool  topo_l2  = run_se3 && topo_env && std::strcmp(topo_env, "l2") == 0;
    const bool  topo_l3  = run_se3 && topo_env && std::strcmp(topo_env, "l3") == 0;
    const bool se3_fixed_topo = run_se3 && !topo_l1 && !topo_l2 && !topo_l3 && (
        !topo_env || std::strcmp(topo_env, "per_block") != 0);
    se3::TopoRefineCfg topo_cfg;
    topo_cfg.l3 = topo_l3;
    topo_cfg.l1 = topo_l1 || topo_l3;          // l3 = l1 + l2 + 阻尼基准 ✓
    topo_cfg.l2 = topo_l2 || topo_l3;
    topo_cfg.closure = !(std::getenv("PPML_TOPO_L2_CLOSURE") &&
                         std::strcmp(std::getenv("PPML_TOPO_L2_CLOSURE"), "0") == 0);
    if (const char* _sg = std::getenv("PPML_TOPO_SIGMA")) topo_cfg.sigma = static_cast<float>(std::atof(_sg));
    if (const char* _gm = std::getenv("PPML_TOPO_GAMMA")) topo_cfg.gamma = static_cast<float>(std::atof(_gm));
    topo_cfg.debug = (std::getenv("PPML_TOPO_DEBUG") != nullptr);
    // ★ 状态是**本函数局部** ⇒ 跨 block 边界存活、随每次 forward 自然重置 ✓（x⁰ 快照/冻结索引/margin ✓）
    se3::TopoRefineState topo_st;
    if (run_se3 && (topo_l1 || topo_l2 || topo_l3)) {
        std::fprintf(stderr, "[SE3-TOPO] mode=%s  sigma=%.3f  gamma=%.3f  closure=%d  (refined topo pass ✓；"
                             "求解见 TopoPassRefinement.md 附录 A ✓)\n",
                     topo_l3 ? "l3" : (topo_l1 ? "l1" : "l2"), topo_cfg.sigma, topo_cfg.gamma,
                     (int)topo_cfg.closure);
    }
    if (run_se3 && getenv("GRAPH_DEBUG_COORD")) {
        std::fprintf(stderr, "[SE3-TOPO] mode=%s (fixed=%d per_block=%d l1=%d l2=%d l3=%d)\n",
            se3_fixed_topo ? "fixed" : (topo_l3 ? "l3" : (topo_l1 ? "l1" : (topo_l2 ? "l2" : "per_block"))),
            (int)se3_fixed_topo, (int)(!se3_fixed_topo), (int)topo_l1, (int)topo_l2, (int)topo_l3);
    }
    // 开关A：收集各 block 的 offset 图节点（iter/refine 分开，因 apply_coord_update 类型不同），
    // block 循环后统一 graph_compute + 读值 + 链式更新 coords。
    std::vector<TensorF32*> se3_fixed_offsets_iter_;
    std::vector<TensorF32*> se3_fixed_offsets_ref_;
    std::vector<IterBlock*>  se3_fixed_blks_iter_;
    std::vector<RefineBlock*> se3_fixed_blks_ref_;
    TensorF32 seq1hot = has_struct ? one_hot_seq(input.seq_tokens, 21)
                                   : TensorF32();                     // (B,L,21)
    // SE3 链式坐标：operator= 被禁用，先按形状构造再 copy_from
    // 【开关B/pass1】topo_coords 非空时用它做 SE3 make_graph 的拓扑基准（Pass1 值版逐 block
    //   优化后的精确 coords，RF2 思路）；否则用 input.coords（初始坐标，开关A 行为）。
    const TensorF32& coords_src = (run_se3 && topo_coords && topo_coords->numel() > 0)
        ? *topo_coords : input.coords;
    TensorF32 current_coords = run_se3
        ? TensorF32(coords_src.shape(), Device::CPU)
        : TensorF32();
    if (run_se3) current_coords.copy_from(coords_src);                 // SE3 链式坐标
    const TensorF32* coords_ptr  = run_se3 ? &current_coords : nullptr;
    const TensorI64* residx_ptr  = run_se3 ? &input.residx  : nullptr;
    const TensorF32* seq1hot_ptr = run_se3 ? &seq1hot       : nullptr;

    // 【FAPE 梯度回传】坐标图节点链（开关A：fixed 拓扑下逐 block 用图 op 更新坐标）。
    // 目标：让 FAPE 的 pred 坐标成为可微图节点，梯度经 coords→offset→SE3 回传（原 wrap_value_as_leaf 断链）。
    // 布局：offset 图节点 [9, B*L]（9=3原子×3坐标, 原子最内, 节点次内）；coords 值 (B,L,3,3) 展平与
    // [9, B*L] 布局一致（原子/坐标最内, b/l 外）。更新公式（与值版 apply_coord_update 一致）：
    //   dN = off_CA + off_N;  dCA = off_CA;  dC = off_CA + off_C
    //   coords_new = coords_old + T @ offset，T 为 (9,9) 常量映射矩阵。
    TensorF32* coords_graph = nullptr;
    std::vector<float> T_data(static_cast<size_t>(81), 0.0f);   // 9×9 row-major
    for (int c = 0; c < 3; ++c) {
        T_data[static_cast<size_t>(c) * 9 + static_cast<size_t>(3 + c)] += 1.0f;       // dN 行 += e_CA(列3-5)
        T_data[static_cast<size_t>(c) * 9 + static_cast<size_t>(0 + c)] += 1.0f;       // dN 行 += e_N(列0-2)
        T_data[static_cast<size_t>(3 + c) * 9 + static_cast<size_t>(3 + c)] += 1.0f;   // dCA 行 = e_CA
        T_data[static_cast<size_t>(6 + c) * 9 + static_cast<size_t>(3 + c)] += 1.0f;   // dC 行 += e_CA
        T_data[static_cast<size_t>(6 + c) * 9 + static_cast<size_t>(6 + c)] += 1.0f;   // dC 行 += e_C(列6-8)
    }
    TensorF32* T_const = constant_tensor({9, 9}, T_data.data());                       // [9,9]
    // 图版 FAPE 路径的 offset 缩放：默认值经扫描确定为 1e-3（offset~1e4 × 1e-3 ≈ 10Å/block 的
    // 刚体位移，量级合理、FAPE 梯度良态、loss 收敛平滑）。原值版 apply_coord_update 用 0.03
    // （即 300Å/block 巨型扰动，FAPE 落退化平台、梯度爆炸）。可用 PPML_SE3_GRAPH_SCALE 覆盖。
    float kSe3OffsetScale = 0.001f;
    if (const char* s = std::getenv("PPML_SE3_GRAPH_SCALE")) kSe3OffsetScale = std::atof(s);
    // 图版位移 clamp（2026-09-07）：值版 apply_coord_update 已对单步位移 clamp ±3Å；图版
    // coords_graph 链此前未 clamp → 偶发巨大 offset（1e3~1e8）×scale 直接进入 FAPE/conf 坐标
    // → Epoch 瞬态 loss 巨大。与值版同用 PPML_SE3_MAX_STEP（≤0 表示不 clamp）。
    float kMaxStepGraph = 3.0f;
    if (const char* s = std::getenv("PPML_SE3_MAX_STEP")) kMaxStepGraph = std::atof(s);
    const bool do_clamp_graph = (kMaxStepGraph > 0.0f);

    // ================================================================
    // [方案A step1-3] 值版主干 + SE3 输入值 leaf 化（PPML_SE3_VALUE_DRIVE=1 启用，默认关）
    // 动机：per_block（开关B）下 SE3 每 block 构图需要"该 block 的 msa/pair 值"，当前靠
    //   compute_and_read 在独立 CPU 子图展开/重算整条 backbone（msa/pair/state 主图链）。
    //   值版主干可让 SE3 构图输入改用值 leaf → compute_and_read 子图只算 SE3 网络自身
    //   （不再重算 backbone）→ 简化共享节点 bind/清空复杂度（图更小、更确定）。
    // 注（2026-09-07 排查更新）：此前注释将"共享节点反复 bind/清空"归因为"偶发巨大
    //   offset/NaN"的成因——该因果不成立。实际根因另有：
    //   ① Epoch2 全 NaN = Epoch1 反向个别参数梯度 NaN（clip_grad_norm 跳过 NaN 未触发
    //      scale_param_grads → NaN 梯度进 AdamW → SE3 embed 参数 NaN），已由 clip 前显式
    //      清零含 NaN 参数梯度（止损）解决；
    //   ② SE3 随机初始化下 offset 偶发巨大 O(1e3~1e8) = 度1 输出无约束（正反馈放大），
    //      已由值版/图版单步位移 clamp ±3Å 解决；
    //   ③ Epoch3 偶发瞬态 loss 巨大 = chi head 分量异常（fape 同 step 正常），非
    //      coords/offset 链，止损 + grad clip 保证自恢复。
    // 局限：值版 track 与图版 track 各自 dropout mask 随机不同 → 两链 msa/pair
    //   有 0.15 随机差；SE3 输入走值版链（自洽）。混合调度下曾 segfault（默认关）。
    const bool kValDrive = run_se3 && !se3_fixed_topo && getenv("PPML_SE3_VALUE_DRIVE") &&
                           std::strcmp(getenv("PPML_SE3_VALUE_DRIVE"), "1") == 0;
    TensorF32 vmsa, vpair, vstate;          // 值版主干 msa/pair/state（值 row-major 布局）
    TensorF32 vmsa_full;                    // extra(FullBlock) 用 msa_full 值
    // 值 → 图 leaf（值 dims 逆序即图 ggml dims：值 (B,N,L,D) = 图 [D,L,N,B]）
    auto val_to_graph_leaf = [&](const TensorF32& v) -> TensorF32* {
        if (v.numel() == 0 || !v.data()) return nullptr;
        std::vector<int64_t> gd;
        const Shape& s = v.shape();
        for (int i = (int)s.ndim() - 1; i >= 0; --i) gd.push_back(s.dims[i]);
        return wrap_input_as_leaf(v, gd);
    };
    if (kValDrive) {
        // ---- 初始值：物化主图 msa/pair/state 初始 embedding 图节点（与图版同源同权重）----
        // 用主 backend(cpu) 独立子图 compute，读值后清 COMPUTE flag + buffer/data（主图后续重算）。
        auto materialize_val = [&](TensorF32* g, TensorF32& out) {
            if (!g) return;
            ComputeGraph* cg = ComputeGraph::new_graph(ctx);
            cg->build_forward_expand(g);
            backend->graph_compute(cg);
            for (int gi = 0; gi < cg->n_nodes(); ++gi) { TensorF32* nd = cg->graph_node(gi); if (nd) nd->flag &= ~TENSOR_FLAG_COMPUTE; }
            for (int gi = 0; gi < cg->n_leafs(); ++gi) { TensorF32* nd = cg->graph_leaf(gi); if (nd) nd->flag &= ~TENSOR_FLAG_COMPUTE; }
            if (g->data()) {
                std::vector<int64_t> vd;
                for (int i = (int)g->shape().ndim() - 1; i >= 0; --i) vd.push_back(g->shape().dims[i]);
                out.~TensorF32();
                new (&out) TensorF32(Shape(vd), Device::CPU);
                const size_t bytes = static_cast<size_t>(g->numel()) * sizeof(float);
                if (g->buffer_ && !g->buffer_->is_host()) {
                    backend->synchronize();
                    g->buffer_->get_tensor(g, out.data(), g->buffer_offs_, bytes);
                } else {
                    std::memcpy(out.data(), g->data(), bytes);
                }
            }
            for (int gi = 0; gi < cg->n_nodes(); ++gi) {
                TensorF32* nd = cg->graph_node(gi);
                if (nd && !(nd->flag & TENSOR_FLAG_PARAM)) { nd->buffer_ = nullptr; nd->buffer_offs_ = 0; nd->bind_data(nullptr); }
            }
        };
        materialize_val(msa, vmsa);
        materialize_val(pair, vpair);
        materialize_val(state, vstate);
        if (msa_full != nullptr) materialize_val(msa_full, vmsa_full);
        if (getenv("GRAPH_DEBUG_COORD")) {
            std::fprintf(stderr, "[VAL-DRIVE] init vmsa=%lld vpair=%lld vstate=%lld vmsa_full=%lld\n",
                (long long)vmsa.numel(), (long long)vpair.numel(), (long long)vstate.numel(),
                (long long)vmsa_full.numel());
        }
    }

    // 驱动单个 block 的 SE3（在每个 block 前向之后、下一个 block 之前）：
    //   1) 回落当前 pair 值；2) run_se3_structural 追加可微 SE3 图节点并回写 state；
    //   3) 回落 offset 值 → apply_coord_update → 链式更新 current_coords。
    auto drive_block_se3 = [&](IterBlock* blk, TensorF32*& msa_ref, TensorF32*& pair_ref) {
        if (!run_se3) return;
        // 【阶段1.5 重构】不再回落 pair 值（edge_w 已在 run_se3_graph 内从主图 pair 图化），
        // 拓扑用 host current_coords（开关B：每 block 即时更新；开关A：初始 coords，不更新）。
        std::vector<TensorF32*> se3_out;
        // [方案A] SE3 输入值 leaf（lambda 级，供 run_se3_structural 与 compute_and_read keep 使用）
        TensorF32* vleaf_msa = nullptr;
        TensorF32* vleaf_pair = nullptr;
        if (kValDrive) {
            // [方案A] SE3 构图输入改用值 leaf（值版主干 vmsa/vpair）→ 子图不再重算 backbone
            vleaf_msa  = val_to_graph_leaf(vmsa);
            vleaf_pair = val_to_graph_leaf(vpair);
            se3_out = (vleaf_msa && vleaf_pair) ? blk->run_se3_structural(
                vleaf_msa, vleaf_pair, rbf, state, current_coords, input.residx, seq1hot,
                &topo_cfg, &topo_st)                                   // 【L1/L2】✓（关时零变化 ✓）
                                 : std::vector<TensorF32*>();
        } else {
            se3_out = blk->run_se3_structural(
                msa_ref, pair_ref, rbf, state, current_coords, input.residx, seq1hot,
                &topo_cfg, &topo_st);                                  // 【L1/L2】✓
        }
        // 开关A（fixed）：只构图，收集 offset 图节点，block 循环后统一 compute + 读值更新。
        //   ~2100Å 巨型扰动 → FAPE 发散）。改为循环结束后一次性 build（sum/N，N=block 数），
        //   使图版坐标扰动总量≈单 block 量级，与开关B 一致，FAPE 梯度收敛。
        if (se3_fixed_topo) {
            if (se3_out.size() > 1) {
                se3_fixed_blks_iter_.push_back(blk);
                se3_fixed_offsets_iter_.push_back(se3_out[1]);
            }
            return;
        }
        if (se3_out.size() > 1) {
            // 【FAPE 梯度回传 · 开关B】逐 block 累积可微坐标链：coords_new = coords_old + T@offset。
            // 开关B 拓扑每 block 即时更新（current_coords 已含上一 block 的 offset），故各 block 的
            // offset 是真实的结构精化，应累加（不除 N）。coords_graph 初始化于首 block，布局 [9,B*L]。
            if (!coords_graph) {
                TensorF32 coords_flat = TensorF32(Shape({B * L * 9}), Device::CPU);
                std::memcpy(coords_flat.data(), coords_src.data(), sizeof(float) * coords_src.numel());
                coords_graph = wrap_input_as_leaf(coords_flat, {B * L * 9});
            }
            {
                TensorF32* offset_scaled = mul(mul_mat(T_const, se3_out[1]),
                                               se3_scale_tensor());
                if (do_clamp_graph)
                    offset_scaled = clamp(offset_scaled, -kMaxStepGraph, kMaxStepGraph);
                coords_graph = add_impl(coords_graph, offset_scaled, /*inplace=*/false);
            }
            TensorF32 offset_val;
            // 【阶段1.5】offset 回落用独立 se3_backend_（独立 gallocr_）：其 graph_compute 的
            // release 只释放 se3_backend_ 自己的 buffer，不碰主图 cpu_backend_ 的 msa/pair/state。
            // SE3 子图节点会 bind 到 se3_backend_ buffer（持久存活），主图最终 compute 时由
            // bind_tensor 无条件 rebind 回主 backend（正确覆盖）。
            // [方案A] kValDrive 时 keep 值 leaf（避免被清 data，主图 coords_graph 需重算 SE3）
            { ComputeGraph* cg = ComputeGraph::new_graph(ctx);
              std::vector<TensorF32*> keep;
              if (kValDrive && vleaf_msa && vleaf_pair) { keep.push_back(vleaf_msa); keep.push_back(vleaf_pair); }
              compute_and_read(se3_out[1], offset_val, cg, se3_bk, kValDrive ? &keep : nullptr); }
            blk->apply_coord_update(offset_val, current_coords);
            if (getenv("GRAPH_DEBUG_COORD")) {
                const TensorF32& uc = blk->updated_coords();
                std::fprintf(stderr, "[DRIVE] cur={%lld,%lld,%lld,%lld} numel=%lld upd={%lld,%lld,%lld,%lld} numel=%lld se3_out=%zu\n",
                    (long long)(current_coords.shape().dims.size()>0?current_coords.shape().dims[0]:-1),
                    (long long)(current_coords.shape().dims.size()>1?current_coords.shape().dims[1]:-1),
                    (long long)(current_coords.shape().dims.size()>2?current_coords.shape().dims[2]:-1),
                    (long long)(current_coords.shape().dims.size()>3?current_coords.shape().dims[3]:-1),
                    (long long)current_coords.numel(),
                    (long long)(uc.shape().dims.size()>0?uc.shape().dims[0]:-1),
                    (long long)(uc.shape().dims.size()>1?uc.shape().dims[1]:-1),
                    (long long)(uc.shape().dims.size()>2?uc.shape().dims[2]:-1),
                    (long long)(uc.shape().dims.size()>3?uc.shape().dims[3]:-1),
                    (long long)uc.numel(), se3_out.size());
            }
            const TensorF32& ucc = blk->updated_coords();
            if (getenv("GRAPH_DEBUG_COORD")) {
                std::fprintf(stderr, "[DRIVE-PTR] cur.data=%p cur.numel=%lld upd.data=%p upd.numel=%lld\n",
                    (void*)current_coords.data(), (long long)current_coords.numel(),
                    (void*)ucc.data(), (long long)ucc.numel());
            }
            // 链式坐标更新：仅当 SE3 产出了有效 offset（xyz_new_ numel>1）才更新 current_coords；
            // 若 offset 空（graph_compute 未算出 se3_out[1]，apply_coord_update 跳过），保持原坐标，
            // 避免重建空 current_coords 后 copy_from 悬垂崩（DRIVE-WARN 触发场景）。
            if (ucc.numel() > 1) {
                if (ucc.numel() != current_coords.numel()) {
                    std::fprintf(stderr, "[DRIVE-WARN] blk=%p upd numel=%lld != cur numel=%lld, rebuild cur\n",
                        (void*)blk, (long long)ucc.numel(), (long long)current_coords.numel());
                    current_coords.~TensorF32();
                    new (&current_coords) TensorF32(ucc.shape(), ucc.device());
                }
                current_coords.copy_from(ucc);   // 链式坐标供下一 block
            }
        }
    };

    if (msa_full != nullptr) {
        for (auto& block : extra_blocks_) {   // FullBlock: global column attention
            block->forward_graph(msa_full, pair, rbf, state, coords_ptr, residx_ptr, seq1hot_ptr);
            if (kValDrive && vmsa_full.numel() > 0) {
                // [方案A] 值版主干：track+SE3 值前向，产出 vmsa_full/vpair/vstate（打开 dropout）
                block->forward(vmsa_full, vpair, vstate, seq1hot, current_coords,
                               input.bond_feats, input.dist_matrix, input.same_chain, input.residx);
            }
            drive_block_se3(block.get(), msa_full, pair);
        }
    }
    for (auto& block : main_blocks_) {
        block->forward_graph(msa, pair, rbf, state, coords_ptr, residx_ptr, seq1hot_ptr);
        if (kValDrive && vmsa.numel() > 0) {
            // [方案A] 值版主干：track+SE3 值前向（打开 dropout，与图版同算子同权重）
            block->forward(vmsa, vpair, vstate, seq1hot, current_coords,
                           input.bond_feats, input.dist_matrix, input.same_chain, input.residx);
        }
        drive_block_se3(block.get(), msa, pair);
    }
    // RefineBlock 的 SE3 结构更新 pipeline 与 IterBlock 不同（node=309 含 state、边两段式、
    // update_msa_pair=false），须用 RefineBlock 专属驱动：回落 pair 值 → run_se3_structural_refine
    // （Phase A 值版两阶段边→make_graph→basis；Phase B 可微 SE3 图块、state 回写）→ 回落 offset →
    // apply_coord_update → 链式更新 current_coords。注意不能复用 drive_block_se3（其调用的
    // run_se3_structural 为 IterBlock 非 virtual 版本，会访问 RefineBlock 未注入的 IterBlock SE3 成员）。
    auto drive_refine_block_se3 = [&](RefineBlock* blk, TensorF32*& msa_ref, TensorF32*& pair_ref) {
        if (!run_se3) return;
        // 【阶段1.5 重构】不再回落 pair 值（edge_w 在 run_se3_graph_refine 内从主图 pair 图化）。
        std::vector<TensorF32*> se3_out;
        // [方案A] SE3 输入值 leaf（lambda 级，供 run_se3_structural_refine 与 compute_and_read keep）
        TensorF32* vleaf_msa = nullptr;
        TensorF32* vleaf_pair = nullptr;
        if (kValDrive) {
            // [方案A] SE3 构图输入改用值 leaf（值版主干 vmsa/vpair）→ 子图不再重算 backbone
            vleaf_msa  = val_to_graph_leaf(vmsa);
            vleaf_pair = val_to_graph_leaf(vpair);
            se3_out = (vleaf_msa && vleaf_pair) ? blk->run_se3_structural_refine(
                vleaf_msa, vleaf_pair, rbf, state, current_coords, input.residx, seq1hot,
                &topo_cfg, &topo_st)                                   // 【L1/L2】✓（与 iter 共用状态 ✓）
                                 : std::vector<TensorF32*>();
        } else {
            se3_out = blk->run_se3_structural_refine(
                msa_ref, pair_ref, rbf, state, current_coords, input.residx, seq1hot,
                &topo_cfg, &topo_st);                                  // 【L1/L2】✓
        }
        // 开关A（fixed）：只构图，收集 offset 图节点，block 循环后统一 compute + 读值更新。
        if (se3_fixed_topo) {
            if (se3_out.size() > 1) {
                se3_fixed_blks_ref_.push_back(blk);
                se3_fixed_offsets_ref_.push_back(se3_out[1]);
            }
            return;
        }
        if (se3_out.size() > 1) {
            // 【FAPE 梯度回传 · 开关B】 refining 阶段同样逐 block 累积可微坐标链。
            if (!coords_graph) {
                TensorF32 coords_flat = TensorF32(Shape({B * L * 9}), Device::CPU);
                std::memcpy(coords_flat.data(), coords_src.data(), sizeof(float) * coords_src.numel());
                coords_graph = wrap_input_as_leaf(coords_flat, {B * L * 9});
            }
            {
                TensorF32* offset_scaled = mul(mul_mat(T_const, se3_out[1]),
                                               se3_scale_tensor());
                if (do_clamp_graph)
                    offset_scaled = clamp(offset_scaled, -kMaxStepGraph, kMaxStepGraph);
                coords_graph = add_impl(coords_graph, offset_scaled, /*inplace=*/false);
            }
            TensorF32 offset_val;
            // 【阶段1.5】offset 回落用独立 se3_backend_（见 drive_block_se3 注释）。
            // [方案A] kValDrive 时 keep 值 leaf（避免被清 data，主图 coords_graph 需重算 SE3）
            { ComputeGraph* cg = ComputeGraph::new_graph(ctx);
              std::vector<TensorF32*> keep;
              if (kValDrive && vleaf_msa && vleaf_pair) { keep.push_back(vleaf_msa); keep.push_back(vleaf_pair); }
              compute_and_read(se3_out[1], offset_val, cg, se3_bk, kValDrive ? &keep : nullptr); }
            blk->apply_coord_update(offset_val, current_coords);
            if (getenv("GRAPH_DEBUG_COORD")) {
                const TensorF32& uc = blk->updated_coords();
                std::fprintf(stderr, "[DRIVE] cur={%lld,%lld,%lld,%lld} numel=%lld upd={%lld,%lld,%lld,%lld} numel=%lld se3_out=%zu\n",
                    (long long)(current_coords.shape().dims.size()>0?current_coords.shape().dims[0]:-1),
                    (long long)(current_coords.shape().dims.size()>1?current_coords.shape().dims[1]:-1),
                    (long long)(current_coords.shape().dims.size()>2?current_coords.shape().dims[2]:-1),
                    (long long)(current_coords.shape().dims.size()>3?current_coords.shape().dims[3]:-1),
                    (long long)current_coords.numel(),
                    (long long)(uc.shape().dims.size()>0?uc.shape().dims[0]:-1),
                    (long long)(uc.shape().dims.size()>1?uc.shape().dims[1]:-1),
                    (long long)(uc.shape().dims.size()>2?uc.shape().dims[2]:-1),
                    (long long)(uc.shape().dims.size()>3?uc.shape().dims[3]:-1),
                    (long long)uc.numel(), se3_out.size());
            }
            const TensorF32& ucc = blk->updated_coords();
            if (getenv("GRAPH_DEBUG_COORD")) {
                std::fprintf(stderr, "[DRIVE-PTR] cur.data=%p cur.numel=%lld upd.data=%p upd.numel=%lld\n",
                    (void*)current_coords.data(), (long long)current_coords.numel(),
                    (void*)ucc.data(), (long long)ucc.numel());
            }
            // 链式坐标更新：仅当 SE3 产出了有效 offset（xyz_new_ numel>1）才更新 current_coords；
            // 若 offset 空（graph_compute 未算出 se3_out[1]，apply_coord_update 跳过），保持原坐标，
            // 避免重建空 current_coords 后 copy_from 悬垂崩（DRIVE-WARN 触发场景）。
            if (ucc.numel() > 1) {
                if (ucc.numel() != current_coords.numel()) {
                    std::fprintf(stderr, "[DRIVE-WARN] blk=%p upd numel=%lld != cur numel=%lld, rebuild cur\n",
                        (void*)blk, (long long)ucc.numel(), (long long)current_coords.numel());
                    current_coords.~TensorF32();
                    new (&current_coords) TensorF32(ucc.shape(), ucc.device());
                }
                current_coords.copy_from(ucc);   // 链式坐标供下一 block
            }
        }
    };
    for (auto& block : refine_blocks_) {
        // forward_graph 为 pass-through（不改 msa/pair），SE3 由驱动完成。
        block->forward_graph(msa, pair, rbf, state, coords_ptr, residx_ptr, seq1hot_ptr);
        if (kValDrive && vmsa.numel() > 0) {
            // [方案A] 值版主干：refine 值前向（更新 vstate，SE3 输入用值 leaf）
            block->forward(vmsa, vpair, vstate, seq1hot, current_coords,
                           input.bond_feats, input.dist_matrix, input.same_chain, input.residx);
        }
        drive_refine_block_se3(static_cast<RefineBlock*>(block.get()), msa, pair);
    }

    // ==== 开关A（fixed 拓扑）：统一 compute 所有 block 的 SE3 offset，读值链式更新 coords ====
    // 【设计】开关A 下 drive_block_se3 只构图（用初始 coords 拓扑），不回落 offset。
    // 此处对收集的全部 offset 图节点 build 到**同一个** cgraph，一次 graph_compute 物化全部
    // SE3 offset/state（及其依赖的 msa/pair 子树——SE3 的 node0 依赖主图 msa）。
    //   msa/pair，若用独立 backend compute，会把主图 msa/pair 子图节点 bind 到 se3_bk buffer 并
    //   置 TENSOR_FLAG_COMPUTE → loss 图 compute 时 msa 子图被跳过/读错位 → msa head 输入全 0
    //   → msa loss 恒 ln(23)=3.13（实测 2026-08-22）。用主 backend 则 msa 子树在主图生命周期内
    //   计算，loss 图再 compute 时正常 rebind+重算（与 4.5 段"无 SE3 时主图一次 compute"一致）。
    //   此 cgraph 覆盖全部 offset 节点 → 即主图前向物化；loss 图（train.cpp）后续再 compute 反向。
    if (se3_fixed_topo && (!se3_fixed_offsets_iter_.empty() || !se3_fixed_offsets_ref_.empty())) {
        ComputeGraph* cg = ComputeGraph::new_graph(ctx);
        for (TensorF32* onode : se3_fixed_offsets_iter_) cg->build_forward_expand(onode);
        for (TensorF32* onode : se3_fixed_offsets_ref_) cg->build_forward_expand(onode);
        backend->graph_compute(cg);   // 主 backend：物化全部 offset/state + 依赖的 msa/pair 子树
        // msa/pair 依赖，若残留 flag，loss 图 build_forward_expand(total) 时
        // visit_parents_graph（ComputeGraph.cpp:112）在"已访问"分支会跳过已带 flag 的 src →
        // msa/pair 子图大量节点/叶子不进 loss 图 nodes[] → 不重算 → msa head 输入全 0 →
        // msa loss 恒 ln(23)=3.13（实测 [msa-sub] mul_mat_104448 9 vs 30，且 leaf 亦需清）。
        for (int gi = 0; gi < cg->n_nodes(); ++gi) {
            TensorF32* nd = cg->graph_node(gi);
            if (nd) nd->flag &= ~TENSOR_FLAG_COMPUTE;
        }
        for (int gi = 0; gi < cg->n_leafs(); ++gi) {
            TensorF32* nd = cg->graph_leaf(gi);
            if (nd) nd->flag &= ~TENSOR_FLAG_COMPUTE;
        }
        // 依次读 offset（compute 后 se3_bk buffer 有效），链式更新 current_coords
        for (size_t i = 0; i < se3_fixed_offsets_iter_.size(); ++i) {
            TensorF32* onode = se3_fixed_offsets_iter_[i];
            if (!onode || !onode->data()) continue;
            TensorF32 offset_val;
            offset_val.~TensorF32();
            new (&offset_val) TensorF32(Shape({onode->shape().dims[1], onode->shape().dims[0]}), Device::CPU);
            offset_val.copy_from(*onode);   // 图 [D,B*L] → 值 [B*L,D]
            if (offset_val.numel() > 1) {
                se3_fixed_blks_iter_[i]->apply_coord_update(offset_val, current_coords);
                const TensorF32& ucc = se3_fixed_blks_iter_[i]->updated_coords();
                if (ucc.numel() > 1) {
                    if (ucc.numel() != current_coords.numel()) {
                        current_coords.~TensorF32();
                        new (&current_coords) TensorF32(ucc.shape(), ucc.device());
                    }
                    current_coords.copy_from(ucc);
                }
            }
        }
        for (size_t i = 0; i < se3_fixed_offsets_ref_.size(); ++i) {
            TensorF32* onode = se3_fixed_offsets_ref_[i];
            if (!onode || !onode->data()) continue;
            TensorF32 offset_val;
            offset_val.~TensorF32();
            new (&offset_val) TensorF32(Shape({onode->shape().dims[1], onode->shape().dims[0]}), Device::CPU);
            offset_val.copy_from(*onode);
            if (offset_val.numel() > 1) {
                se3_fixed_blks_ref_[i]->apply_coord_update(offset_val, current_coords);
                const TensorF32& ucc = se3_fixed_blks_ref_[i]->updated_coords();
                if (ucc.numel() > 1) {
                    if (ucc.numel() != current_coords.numel()) {
                        current_coords.~TensorF32();
                        new (&current_coords) TensorF32(ucc.shape(), ucc.device());
                    }
                    current_coords.copy_from(ucc);
                }
            }
        }
        if (getenv("GRAPH_DEBUG_COORD")) {
            std::fprintf(stderr, "[SE3-FIXED] applied %zu iter + %zu refine offsets\n",
                se3_fixed_offsets_iter_.size(), se3_fixed_offsets_ref_.size());
        }
        // 【FAPE 梯度回传】一次性构建 coords_graph = coords_src + (Σ offset×scale)/N，
        // N = block 数。避免逐 block 累加导致 ~N× 巨型坐标扰动（FAPE 发散）。offset 图节点
        // 仍连主图 → 梯度可回流 SE3 参数；除以 N 使图版坐标扰动≈单 block 量级（与开关B 一致）。
        {
            const int N = (int)(se3_fixed_offsets_iter_.size() + se3_fixed_offsets_ref_.size());
            TensorF32 coords_flat = TensorF32(Shape({B * L * 9}), Device::CPU);
            std::memcpy(coords_flat.data(), coords_src.data(), sizeof(float) * coords_src.numel());
            TensorF32* cg_init = wrap_input_as_leaf(coords_flat, {B * L * 9});  // [9, B*L]
            TensorF32* accum = nullptr;
            auto add_one = [&](TensorF32* o) {
                TensorF32* os = mul(mul_mat(T_const, o), se3_scale_tensor());
                if (do_clamp_graph) os = clamp(os, -kMaxStepGraph, kMaxStepGraph);
                accum = accum ? add_impl(accum, os, false) : os;
            };
            for (TensorF32* o : se3_fixed_offsets_iter_) add_one(o);
            for (TensorF32* o : se3_fixed_offsets_ref_) add_one(o);
            if (accum && N > 0) {
                const float invN = 1.0f / (float)N;
                accum = mul(accum, constant_tensor({1}, &invN));  // /N
                coords_graph = add_impl(cg_init, accum, false);
            } else {
                coords_graph = cg_init;
            }
        }
    }

    // ==== 4.5 主干落地（重构：不再独立 compute）====
    // 【历史教训】这里曾对 msa/pair/state 子图做"独立 build_forward_expand + graph_compute + 落地值"，
    // 该方案源于错误认知"head 的 LayerNorm 是值版会立即 compute、污染 go.*"——已证明 LayerNorm::forward
    // 是指针版纯构图（Embedding.cpp:213），不 compute，head 全部 forward_graph 只构图不执行。
    // 独立 compute 的真正危害：
    //   1) 给 msa/pair/state 子图节点置 TENSOR_FLAG_COMPUTE，若不清除则 final loss graph 的
    //      build_forward_expand(total)（train.cpp:1154）会因 visit_parents_graph（ComputeGraph.cpp:112）
    //      跳过这些节点 → 不重算 → buffer 清零 → head 输入全 0 → msa loss 恒 ln(23)；
    //   2) 即使清除标志，跨 cgraph 的 gallocr buffer 生命周期仍与 final compute 冲突（memory 24931582）。
    // 正确做法：不落地、不 compute，go.msa/go.pair/go.state 直接指向图节点本身。head 与 final loss
    // graph 共用同一批图节点，由 train.cpp 一次 build_forward_expand(total)+graph_compute 统一物化。
    // go.* 仅作诊断用（train.cpp 的 scan_node/dump4 均有 data()==nullptr 防护）。
    {
        go.msa   = msa;
        go.pair  = pair;
        go.state = state;

        // 【问题3 即时诊断】compute 后立刻统计 msa/pair/state 的 NaN（此时 buffer 刚物化、
        // 尚未被后续 final loss graph 的 graph_compute 经 gallocr 复用释放）。若此处已 NaN，
        // 说明 4.5 段这个部分子图（只 build msa+state，pair 作为 msa 依赖被连带 build）的
        // 拓扑/别名与 final graph 不一致，导致 state 计算偏差 → 真 NaN；若此处有限而 train.cpp
        // 后续打印 go.state_v 为 NaN，才是 gallocr 释放读后释放的假象。两者区分决定问题3 真因。
        auto diag_nan = [](const char* tag, const TensorF32* n) {
            if (!n || n->numel() == 0 || !n->data()) return;
            long nnan = 0; float mn = 1e30f, mx = -1e30f;
            const float* p = static_cast<const float*>(n->data());
            for (int64_t i = 0; i < n->numel(); ++i) {
                float v = p[i];
                if (v != v) { nnan++; continue; }
                if (v < mn) mn = v; if (v > mx) mx = v;
            }
            fprintf(stderr, "[FWD-FEAT-IMM] %s numel=%lld nnan=%ld min=%.6g max=%.6g\n",
                    tag, (long long)n->numel(), nnan, (double)mn, (double)mx);
        };
        diag_nan("msa", msa);
        diag_nan("pair", pair);
        diag_nan("state", state);
    }

    // ==== 5. 输出头 (forward_graph)：构建可微图节点，供调用方（train.cpp）组装 loss 图 ====
    // 这里只构建图节点、不 graph_compute；调用方对总 loss 图一次 build_forward_expand +
    // build_backward_expand + graph_compute，梯度即可经这些节点回传模型参数。
    // Masked MSA head: LN(D_MSA) → Linear(D_MSA→D_MSA) → ReLU → Linear(D_MSA→23) → [23,L,N,B]
    TensorF32* ln_msa  = msa_head_ln_->forward(msa);
    TensorF32* lin1    = msa_head_linear1_->forward_graph(ln_msa);
    TensorF32* relu1   = relu(lin1);
    TensorF32* logits  = msa_head_linear2_->forward_graph(relu1);

    // 【MSA head 诊断】GRAPH_DEBUG_MSA_HEAD=1：打印 head 权重范数 + 输入链节点信息，
    // 区分 "msa=3.13=ln(23) 恒定" 的根因：
    //   A) head 权重全 0 → logits 全 0 → 均匀 softmax（初始化/加载问题）；
    //   B) head 权重正常但输入链（msa 特征）在 final graph 中为 0 → 图重建/flag 问题；
    //   C) 输入有值、权重有值 → masked_msa_loss 计算问题。
    if (getenv("GRAPH_DEBUG_MSA_HEAD")) {
        auto wstat = [](const char* tag, const TensorF32* w) {
            if (!w || !w->data()) { fprintf(stderr, "[MSA-H] %s null\n", tag); return; }
            double s = 0; double s2 = 0; int nz = 0; float mx = 0;
            for (int64_t i = 0; i < w->numel(); ++i) {
                float v = w->data()[i]; s += v; s2 += (double)v * v;
                if (v != 0.0f) nz++;
                float a = (v < 0) ? -v : v; if (a > mx) mx = a;
            }
            fprintf(stderr, "[MSA-H] %s numel=%lld sum=%.6g l2=%.6g nz=%d maxabs=%.6g v0=%.6g\n",
                    tag, (long long)w->numel(), s, std::sqrt(s2), nz, mx, w->data()[0]);
        };
        wstat("ln_msa.beta", msa_head_ln_->beta());
        wstat("lin1.w", msa_head_linear1_->weight());
        wstat("lin1.b", msa_head_linear1_->bias());
        wstat("lin2.w", msa_head_linear2_->weight());
        wstat("lin2.b", msa_head_linear2_->bias());
        fprintf(stderr, "[MSA-H] msa ndim=%d dims=[%lld,%lld,%lld,%lld] flag=0x%x\n",
                msa->shape().ndim(),
                (long long)msa->shape().dims[0], (long long)msa->shape().dims[1],
                (long long)msa->shape().dims[2], (long long)msa->shape().dims[3],
                (unsigned)msa->flag);
    }

    // Chi head: state [32,L,B] → LN → Linear → ReLU → Linear(→14) → [14,L,B] = (B,L,7,2)
    TensorF32* ch_ln   = chi_head_ln_->forward(state);
    TensorF32* ch1     = chi_head_linear1_->forward_graph(ch_ln);
    TensorF32* ch_relu = relu(ch1);
    TensorF32* alpha   = chi_head_linear2_->forward_graph(ch_relu);   // [14,L,B]

    // Distogram heads: pair → LayerNorm → 4 组 logits（防 logits 巨大→softmax 退化→loss 卡死）
    TensorF32* pair_normed = distogram_pair_ln_->forward(pair);
    TensorF32* dist_d = distogram_d_head_->forward_graph(pair_normed);
    TensorF32* dist_o = distogram_o_head_->forward_graph(pair_normed);
    TensorF32* dist_t = distogram_t_head_->forward_graph(pair_normed);
    TensorF32* dist_p = distogram_p_head_->forward_graph(pair_normed);

    // pLDDT head: state → [50,L,B] = (B,L,50)
    TensorF32* lddt   = plddt_head_->forward_graph(state);

    // ---- 组装 GraphOutput：可微图节点 + coords 回落值 ----
    // 注意：保留 4.5 段落到 go.msa_v/go.pair_v/go.state_v 的持久值指针，
    // 绝不可改回 msa/pair/state 图节点（其 buffer 会被 head 的值版 forward 触发 compute
    // 经 gallocr 释放 → go.* 读 0/NaN，正是原始 NaN 根因）。head 图节点仍用原图节点构建。
    // （go.msa/go.pair/go.state 已在 4.5 段指向持久值，此处不动）
    go.msa_logits = logits;
    go.alpha      = alpha;
    go.lddt       = lddt;
    go.distogram  = dist_d;
    go.omega      = dist_o;
    go.theta      = dist_t;
    go.phi        = dist_p;

    // SE3 更新后的坐标（未驱动 SE3 时即输入；图外量，非可微，供 FAPE/conf 沿用原值版方式）
    // SE3 更新后的坐标（未驱动 SE3 时即输入；图外量，非可微，供 FAPE/conf 沿用原值版方式）
    // go.coords 默认空（numel=0），copy_from 要求 shape 匹配，故先按源形状重建。
    {
        const TensorF32& src = run_se3 ? current_coords : input.coords;
        if (src.numel() > 0) {
            go.coords.~TensorF32();
            new (&go.coords) TensorF32(src.shape(), Device::CPU);
            go.coords.copy_from(src);
        }
    }
    // 【FAPE 梯度回传】坐标图节点（开关A：可微，梯度经 coords→offset→SE3 回传）。
    // 未驱动 SE3 或开关B 时为空（nullptr，train.cpp 的 FAPE 回落 wrap_value_as_leaf 沿用值 coords）。
    go.coords_graph = coords_graph;

    return go;
}

//The t1d feature has shape (B, T, L, d_t1d) where B is batch size,
// T is number of templates, L is sequence length, 
//and d_t1d is the feature dimension that varies by model configuration
TensorF32 PPMLModel::get_templ_emb(const TensorF32& t1d, const TensorF32& t2d) {
    // 值版：unsqueeze + repeat 展开 + concat 手写（repeat/concat_ptr 是图版返回图节点
    // data()=nullptr，值 forward 调 copy_from 会崩）。布局 (B,T,L,L,64+2D) 对齐图版。
    int B = t1d.shape().dims[0];
    int T = t1d.shape().dims[1];
    int L = t1d.shape().dims[2];
    int D = t1d.shape().dims[3];
    // t2d 空占位（numel=1 默认构造）时按 0 特征处理
    const int64_t D_T2D = (t2d.shape().ndim() == 4 && t2d.numel() > 0) ? t2d.shape().dims[3] : 0;  // 68
    const int64_t OUT_D = D_T2D + 2 * D;           // 228

    TensorF32 templ({B, T, L, L, OUT_D}, t1d.device());
    float* od = templ.data();
    const float* t2d_p = t2d.data();
    const float* t1d_p = t1d.data();
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < L; ++i)
                for (int j = 0; j < L; ++j) {
                    int64_t base = ((int64_t)(b * T + t) * L + i) * L + j;
                    int64_t base_c = base * OUT_D;
                    for (int d = 0; d < D_T2D; ++d)
                        od[base_c + d] = t2d_p[base * D_T2D + d];               // t2d
                    const int64_t base_l = ((int64_t)(b * T + t) * L + i) * D;  // left: t1d[b,t,i,:]
                    const int64_t base_r = ((int64_t)(b * T + t) * L + j) * D;  // right: t1d[b,t,j,:]
                    for (int d = 0; d < D; ++d) {
                        od[base_c + D_T2D + d] = t1d_p[base_l + d];
                        od[base_c + D_T2D + D + d] = t1d_p[base_r + d];
                    }
                }
    // Linear(228 → 128) → (B,T,L,L,D_PAIR)
    return emb_t1d_t2d_->forward(templ);
}

void PPMLModel::to(Device device) {
    device_ = device;

    // 确保后端基础设施已初始化
    ensure_backend_ready();
}

void PPMLModel::ensure_backend_ready() {
    if (backend_ready_) return;

    // 1. 创建 CPU Backend（始终存在）
    //    线程数：PPML_N_THREADS 环境变量（默认 4）。多线程 barrier 已修：graph_plan 保证
    //    plan.n_threads 恒等于实际参与线程数（n_threads_），barrier 计数一致，多线程安全。
    //    GPU op 走 CUDA（dispatch 单线程）仍加速。
    if (!cpu_backend_) {
        int n_threads = 4;
        if (const char* p = getenv("PPML_N_THREADS")) {
            int v = atoi(p);
            if (v >= 1) n_threads = v;
        }
        cpu_backend_ = std::make_unique<CPUBackend>(n_threads);
    }

    // 2. 创建 BackendScheduler 并注册后端
    if (!scheduler_) {
        scheduler_ = std::make_unique<BackendScheduler>();
        scheduler_->add_backend(cpu_backend_.get());
        // 3. 若模型目标设备为 CUDA, 先探测 GPU 可用性 + 空闲显存; 不足则警告并回退 CPU
        if (device_ == Device::CUDA) {
            if (!cuda_available()) {
                const bool user_requested =
                    (getenv("PPML_USE_CUDA") != nullptr);
                std::cerr << "[WARN] CUDA device not available; "
                          << "falling back to CPU backend." << std::endl;
                if (user_requested) {
                    std::cerr
                        << "[CUDA-ERR] 你已显式设置 PPML_USE_CUDA=1，但 GPU 探测失败，"
                           "训练将完全在 CPU 上运行（你加在 CUDABackend 中的日志不会打印）。\n"
                        << "[CUDA-ERR] 常见原因与排查:\n"
                        << "  1) LD_LIBRARY_PATH 指向了其他版本的 libcudart（如 anaconda 的 11 变体），"
                           "覆盖了系统 /usr/lib/x86_64-linux-gnu 的 libcudart.so.11.0 -> "
                           "运行前 export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu。\n"
                        << "  2) nvidia-smi / 驱动不可见 -> 确认驱动已加载、nvidia-smi 能列出 GPU。\n"
                        << "  3) 无 GPU 或权限不足 -> 确认设备节点 /dev/nvidia* 可访问。\n"
                        << "[CUDA-ERR] 见上方 [CUDA-AVAIL] 输出的 cudaGetDeviceCount 具体错误。\n";
                }
                device_ = Device::CPU;   // 回退: 模型按 CPU 运行
            } else {
                const size_t free_b = cuda_free_memory_bytes();
                const size_t need_b = cuda_min_required_bytes();

                // 是否启用 scheduler（混合训练：按节点分配，GPU 放得下的放 GPU，放不下的回落 CPU）。
                // 该标志与 train.cpp 中 use_sched 的判定保持一致。
                const bool use_sched_flag =
                    (getenv("PPML_CUDA_SCHED") != nullptr) &&
                    (std::atoi(getenv("PPML_CUDA_SCHED")) != 0);

                if (!use_sched_flag) {
                    // 单后端全 GPU 模式：确实需要整块显存，不足则整体回退 CPU。
                    if (free_b > 0 && free_b < need_b) {
                        std::cerr << "[WARN] CUDA device free VRAM " << (free_b >> 20)
                                  << " MB < required " << (need_b >> 20)
                                  << " MB (single-backend full-GPU mode); "
                                     "falling back to CPU backend." << std::endl;
                        device_ = Device::CPU;
                        }
                } else {
                    // 混合训练（scheduler）：不整体回退。显存不足由 scheduler 的
                    // gpu_vram_budget_（=空闲显存*4/5）逐节点预算控制，放不下的节点
                    // 自动回落 CPU。仅当空闲显存近乎为 0（驱动/设备异常）才整体回退。
                    if (free_b < (256ULL << 20)) {
                        std::cerr << "[WARN] CUDA free VRAM only " << (free_b >> 20)
                                  << " MB (<256MB); falling back to CPU backend." << std::endl;
                        device_ = Device::CPU;
                        }
                    std::cerr << "[INFO] Mixed CUDA training (scheduler): free VRAM "
                              << (free_b >> 20) << " MB, nodes that fit are placed on GPU, "
                                 "the rest spill to CPU. No full fallback." << std::endl;
                }

                if (device_ == Device::CUDA && !cuda_backend_) {
                    // scheduler 按 priority 排序, CUDA 优先调度到 GPU;
                    // 不支持的 op / 预算不足的节点自动跨后端拷贝或回落 CPU
                    cuda_backend_ = std::make_unique<CUDABackend>(0);  // device 0
                    scheduler_->add_backend(cuda_backend_.get());
                }
            }
        }

        // 4. 可选：把"另一台机器"注册为第 N 个后端（仅当 PPML_REMOTE_HOST 设置时启用）。
        //    节点被分配到远端的条件（两个都要满足）：
        //      ① supports_op 命中 PPML_REMOTE_OPS 列表（形如 "32" 或 "30,32" 或 "all"）
        //      ② 该后端 priority 高于 CPU/CUDA（由 remote_make_backend_from_env 按 env 决定）
        //    未设环境变量时本块不执行 ⇒ 单机行为零影响。RemoteBackend 由
        //    remote_make_backend_from_env() 内部静态持有，生命周期覆盖本模型。
        // ⚠️ 2026-09-14 晚 修正：必须**同时**有非空的 PPML_REMOTE_OPS 才注册远端后端。
        //   只设 PPML_REMOTE_HOST（ops 为空）时 priority=-1 = 最低 ⇒ 排在 backends_ 末尾，
        //   而调度器把"未分配节点（view/reshape 等）"默认交给 n_backends_-1 = **它** ✗
        //   ⇒ 实测 74 个单节点子图被发到对端（数值虽透明，但白搬数据 + 对端上下文耗尽 abort）。
        const char* remote_ops = getenv("PPML_REMOTE_OPS");
        if (getenv("PPML_REMOTE_HOST") != nullptr && remote_ops && *remote_ops) {
            if (RemoteBackend* rb = remote_make_backend_from_env()) {
                scheduler_->add_backend(rb);
                std::cerr << "[REMOTE] 远端后端已注册: ops="
                          << (getenv("PPML_REMOTE_OPS") ? getenv("PPML_REMOTE_OPS") : "(none)")
                          << " priority=" << rb->priority()
                          << " slim=" << (int)rb->buffer_type_mut().slim() << std::endl;
            } else {
                std::cerr << "[REMOTE] PPML_REMOTE_HOST 已设置但连接对端失败 —— 忽略远端后端"
                          << std::endl;
            }
        }
    }

    backend_ready_ = true;

    // 4. 后端就绪后将参数迁移到 backend buffer (权重/偏置等)
    //    在模型构造后 / to() 时调用; load_weights 内部也调用 (见下)
    transfer_params_to_backend();
}

Device PPMLModel::device() const {
    return device_;
}

Backend* PPMLModel::active_backend() {
    ensure_backend_ready();
    // CUDA 若就绪则优先（模型目标设备为 CUDA 且 GPU 可用），否则回退 CPU
    if (device_ == Device::CUDA && cuda_backend_) return cuda_backend_.get();
    return cpu_backend_.get();
}

Backend* PPMLModel::active_cpu_backend() {
    ensure_backend_ready();
    return cpu_backend_.get();
}

BackendScheduler* PPMLModel::scheduler() {
    ensure_backend_ready();
    return scheduler_.get();
}

void PPMLModel::train() {
    training_ = true;
}

void PPMLModel::eval() {
    training_ = false;
}

bool PPMLModel::is_training() const {
    return training_;
}

void PPMLModel::load_weights(const std::string& path) {
    std::cout << "Loading weights from: " << path << std::endl;

    // 1. 确保后端已就绪
    ensure_backend_ready();

    // 2. 从文件加载权重数据到各参数 tensor 的 CPU arena
    //    （此时参数数据仍在 context arena 中，data_ 指向 arena 地址）
    //    示例: file.read(linear->weight()->data(), linear->weight()->nbytes());

    // 3. 将参数迁移到 backend buffer
    transfer_params_to_backend();

    std::cout << "Weights loaded and transferred to backend buffer." << std::endl;
}

// ============================================================
// 私有辅助: 收集所有参数 Tensor (weight/bias/gamma/beta), 顺序固定
// 供 transfer_params_to_backend / params() / 保存共用
// ============================================================
void PPMLModel::collect_all_params(std::vector<TensorF32*>& param_tensors) {
    std::vector<std::string> names;  // 无名字版本: 忽略名字
    collect_params_with_names(param_tensors, names);
}

// ============================================================
// 收集所有参数 Tensor 及语义名 (block/attention 等), 顺序与 param_tensors 严格一一对应。
// 命名规范 (对齐 RF2/AlphaFold 惯例):
//   - 全局单份参数: "msa_emb", "state_emb", "pair_left_emb" ...
//   - 模板参数:     "tps.<layer>"
//   - per-block:    "main.{i}.attention.<attn>.<proj>", "extra.{i}.", "refine.{i}." ...
//     (extra = FullBlock, main = IterBlock, 均共享 iter_* 参数, 索引为全局 0..11)
//   - 每个 LinearLayer: <name>.weight / <name>.bias
//   - 每个 LayerNorm:   <name>.gamma / <name>.beta
//   - 每个 Embedding:   <name>.weight
// ============================================================
void PPMLModel::collect_params_with_names(std::vector<TensorF32*>& param_tensors,
                                          std::vector<std::string>& param_names) {
    collect_params_with_names(param_tensors, param_names, nullptr);
}

void PPMLModel::collect_params_with_names(std::vector<TensorF32*>& param_tensors,
                                          std::vector<std::string>& param_names,
                                          std::vector<LinearLayer*>* linear_layers_out) {
    // 辅助 lambda：收集 LinearLayer / LayerNorm / EmbeddingLayer 的参数及名字
    // 注：param_tensors 与 param_names 必须严格一一对应（save_checkpoint 校验数量一致）。
    // 之前用 `if(!param_names.empty())` 守卫名字，导致从空向量开始时名字永远不 push → 数量不匹配 bug。
    auto collect_linear = [&](LinearLayer* ll, const std::string& name) {
        if (!ll) return;
        if (linear_layers_out) linear_layers_out->push_back(ll);   // 每层收集一次（LoRA 用）
        if (ll->weight()) {
            param_tensors.push_back(ll->weight());
            param_names.push_back(name + ".weight");
        }
        if (ll->bias()) {
            param_tensors.push_back(ll->bias());
            param_names.push_back(name + ".bias");
        }
        // LoRA 旁路参数（A/B）：启用后追加（命名 <name>.lora_A/.lora_B，含 rank 维度）
        if (ll->lora_A()) {
            param_tensors.push_back(ll->lora_A());
            param_names.push_back(name + ".lora_A");
        }
        if (ll->lora_B()) {
            param_tensors.push_back(ll->lora_B());
            param_names.push_back(name + ".lora_B");
        }
    };
    auto collect_layernorm = [&](LayerNorm* ln, const std::string& name) {
        if (ln && ln->gamma()) {
            param_tensors.push_back(ln->gamma());
            param_names.push_back(name + ".gamma");
        }
        if (ln && ln->beta()) {
            param_tensors.push_back(ln->beta());
            param_names.push_back(name + ".beta");
        }
    };
    auto collect_embedding = [&](EmbeddingLayer* emb, const std::string& name) {
        if (emb && emb->weight()) {
            param_tensors.push_back(emb->weight());
            param_names.push_back(name + ".weight");
        }
    };

    // embedding / template 全局参数
    collect_linear(msa_emb_, "msa_emb");
    collect_embedding(state_emb_, "state_emb");
    collect_embedding(pair_left_emb_, "pair_left_emb");
    collect_embedding(pair_right_emb_, "pair_right_emb");
    collect_linear(full_linear_, "full_linear");
    collect_embedding(full_emb_, "full_emb");
    collect_linear(bond_emb_, "bond_emb");
    collect_linear(emb_t1d_, "emb_t1d");
    collect_linear(proj_t1d_, "proj_t1d");
    collect_linear(emb_t1d_t2d_, "emb_t1d_t2d");
    collect_linear(temp_stack_t1d_proj_, "temp_stack_t1d_proj");
    collect_layernorm(temp_stack_norm_, "temp_stack_norm");
    // Template state cross-attention 投影
    collect_linear(templ_attn_Wq_, "templ.attn.Wq");
    collect_linear(templ_attn_Wk_, "templ.attn.Wk");
    collect_linear(templ_attn_Wv_, "templ.attn.Wv");
    collect_linear(templ_attn_Wo_, "templ.attn.Wo");
    collect_linear(templ_pair_attn_Wq_, "templ.pair_attn.Wq");
    collect_linear(templ_pair_attn_Wk_, "templ.pair_attn.Wk");
    collect_linear(templ_pair_attn_Wv_, "templ.pair_attn.Wv");
    collect_linear(templ_pair_attn_Wo_, "templ.pair_attn.Wo");

    // ===== 输出头参数 =====
    collect_layernorm(msa_head_ln_, "msa_head.ln");
    collect_linear(msa_head_linear1_, "msa_head.linear1");
    collect_linear(msa_head_linear2_, "msa_head.linear2");
    collect_layernorm(chi_head_ln_, "chi_head.ln");
    collect_linear(chi_head_linear1_, "chi_head.linear1");
    collect_linear(chi_head_linear2_, "chi_head.linear2");
    collect_layernorm(distogram_pair_ln_, "distogram_head.pair_ln");
    collect_linear(distogram_d_head_, "distogram_head.dist");
    collect_linear(distogram_o_head_, "distogram_head.omega");
    collect_linear(distogram_t_head_, "distogram_head.theta");
    collect_linear(distogram_p_head_, "distogram_head.phi");
    collect_linear(plddt_head_, "plddt_head");

    // ===== TemplatePairStack 参数 =====
    collect_linear(tps_rbf_proj_, "tps.rbf_proj");
    collect_layernorm(tps_state_norm_, "tps.state_norm");
    collect_linear(tps_left_proj_, "tps.left_proj");
    collect_linear(tps_right_proj_, "tps.right_proj");
    collect_linear(tps_gate_proj_, "tps.gate_proj");
    // tri_mul_out
    collect_layernorm(tps_tri_out_layernorm_, "tps.tri_mul_out.layernorm");
    collect_linear(tps_tri_out_left_proj_, "tps.tri_mul_out.left_proj");
    collect_linear(tps_tri_out_right_proj_, "tps.tri_mul_out.right_proj");
    collect_linear(tps_tri_out_left_gate_, "tps.tri_mul_out.left_gate");
    collect_linear(tps_tri_out_right_gate_, "tps.tri_mul_out.right_gate");
    collect_linear(tps_tri_out_gate_, "tps.tri_mul_out.gate");
    collect_layernorm(tps_tri_out_output_layernorm_, "tps.tri_mul_out.output_layernorm");
    collect_linear(tps_tri_out_out_proj_, "tps.tri_mul_out.out_proj");
    // tri_mul_in
    collect_layernorm(tps_tri_in_layernorm_, "tps.tri_mul_in.layernorm");
    collect_linear(tps_tri_in_left_proj_, "tps.tri_mul_in.left_proj");
    collect_linear(tps_tri_in_right_proj_, "tps.tri_mul_in.right_proj");
    collect_linear(tps_tri_in_left_gate_, "tps.tri_mul_in.left_gate");
    collect_linear(tps_tri_in_right_gate_, "tps.tri_mul_in.right_gate");
    collect_linear(tps_tri_in_gate_, "tps.tri_mul_in.gate");
    collect_layernorm(tps_tri_in_output_layernorm_, "tps.tri_mul_in.output_layernorm");
    collect_linear(tps_tri_in_out_proj_, "tps.tri_mul_in.out_proj");
    // pair_row_attn
    collect_layernorm(tps_pair_norm_, "tps.attention.pair_norm");
    collect_linear(tps_pair_row_to_b_, "tps.attention.pair_row.to_b");
    collect_linear(tps_pair_row_to_g_, "tps.attention.pair_row.to_g");
    collect_linear(tps_pair_row_to_out_, "tps.attention.pair_row.to_out");
    collect_linear(tps_pair_row_Wq_, "tps.attention.pair_row.Wq");
    collect_linear(tps_pair_row_Wk_, "tps.attention.pair_row.Wk");
    collect_linear(tps_pair_row_Wv_, "tps.attention.pair_row.Wv");
    // pair_col_attn
    collect_linear(tps_pair_col_to_b_, "tps.attention.pair_col.to_b");
    collect_linear(tps_pair_col_to_g_, "tps.attention.pair_col.to_g");
    collect_linear(tps_pair_col_to_out_, "tps.attention.pair_col.to_out");
    collect_linear(tps_pair_col_Wq_, "tps.attention.pair_col.Wq");
    collect_linear(tps_pair_col_Wk_, "tps.attention.pair_col.Wk");
    collect_linear(tps_pair_col_Wv_, "tps.attention.pair_col.Wv");
    // pair_ff
    collect_layernorm(tps_pair_ff_norm_, "tps.pair_ff.norm");
    collect_linear(tps_pair_ff_linear1_, "tps.pair_ff.linear1");
    collect_linear(tps_pair_ff_linear2_, "tps.pair_ff.linear2");

    // attention 参数 (n_iter blocks × 6 LinearLayer × 6 组)
    // extra = FullBlock (索引 0..n_extra-1), main = IterBlock (索引 n_extra..)
    // block 数用 config（支持环境变量覆盖），名字动态生成。
    const int c_n_extra  = config_.n_extra_blocks;
    const int c_n_main   = config_.n_main_blocks;
    const int c_n_iter   = c_n_extra + c_n_main;

    for (int i = 0; i < c_n_iter; ++i) {
        const std::string b = (i < c_n_extra ? "extra." + std::to_string(i)
                                             : "main." + std::to_string(i - c_n_extra));
        // 注意：extra blocks (FullBlock) 用 D_MSA_FULL=64 专属 MSA 行注意力/ff 权重
        const bool is_full = (i < c_n_extra);
        collect_linear(is_full ? full_msa_row_Wq_[i]     : msa_row_Wq_[i],     b + ".attention.msa_row.Wq");
        collect_linear(is_full ? full_msa_row_Wk_[i]     : msa_row_Wk_[i],     b + ".attention.msa_row.Wk");
        collect_linear(is_full ? full_msa_row_Wv_[i]     : msa_row_Wv_[i],     b + ".attention.msa_row.Wv");
        collect_linear(is_full ? full_msa_row_to_b_[i]   : msa_row_to_b_[i],   b + ".attention.msa_row.to_b");
        collect_linear(is_full ? full_msa_row_to_g_[i]   : msa_row_to_g_[i],   b + ".attention.msa_row.to_g");
        collect_linear(is_full ? full_msa_row_to_out_[i] : msa_row_to_out_[i], b + ".attention.msa_row.to_out");

        // 仅 main blocks (i>=4, IterBlock) 用 msa_col（FullBlock 用 global col attention）
        if (!is_full) {
            collect_linear(msa_col_Wq_[i], b + ".attention.msa_col.Wq");
            collect_linear(msa_col_Wk_[i], b + ".attention.msa_col.Wk");
            collect_linear(msa_col_Wv_[i], b + ".attention.msa_col.Wv");
            collect_linear(msa_col_to_b_[i], b + ".attention.msa_col.to_b");
            collect_linear(msa_col_to_g_[i], b + ".attention.msa_col.to_g");
            collect_linear(msa_col_to_out_[i], b + ".attention.msa_col.to_out");
        }

        collect_layernorm(pair_attn_norm_[i], b + ".attention.pair_norm");
        collect_linear(pair_row_Wq_[i], b + ".attention.pair_row.Wq");
        collect_linear(pair_row_Wk_[i], b + ".attention.pair_row.Wk");
        collect_linear(pair_row_Wv_[i], b + ".attention.pair_row.Wv");
        collect_linear(pair_row_to_b_[i], b + ".attention.pair_row.to_b");
        collect_linear(pair_row_to_g_[i], b + ".attention.pair_row.to_g");
        collect_linear(pair_row_to_out_[i], b + ".attention.pair_row.to_out");

        collect_linear(pair_col_Wq_[i], b + ".attention.pair_col.Wq");
        collect_linear(pair_col_Wk_[i], b + ".attention.pair_col.Wk");
        collect_linear(pair_col_Wv_[i], b + ".attention.pair_col.Wv");
        collect_linear(pair_col_to_b_[i], b + ".attention.pair_col.to_b");
        collect_linear(pair_col_to_g_[i], b + ".attention.pair_col.to_g");
        collect_linear(pair_col_to_out_[i], b + ".attention.pair_col.to_out");

        collect_layernorm(is_full ? full_msa_ff_norm_[i]     : msa_ff_norm_[i],     b + ".msa_ff.norm");
        collect_linear(   is_full ? full_msa_ff_linear1_[i]  : msa_ff_linear1_[i],  b + ".msa_ff.linear1");
        collect_linear(   is_full ? full_msa_ff_linear2_[i]  : msa_ff_linear2_[i],  b + ".msa_ff.linear2");

        collect_layernorm(pair_ff_norm_[i], b + ".pair_ff.norm");
        collect_linear(pair_ff_linear1_[i], b + ".pair_ff.linear1");
        collect_linear(pair_ff_linear2_[i], b + ".pair_ff.linear2");

        // TriangleMultiplication out
        collect_layernorm(tri_out_layernorm_[i], b + ".tri_mul_out.layernorm");
        collect_linear(tri_out_left_proj_[i], b + ".tri_mul_out.left_proj");
        collect_linear(tri_out_right_proj_[i], b + ".tri_mul_out.right_proj");
        collect_linear(tri_out_left_gate_[i], b + ".tri_mul_out.left_gate");
        collect_linear(tri_out_right_gate_[i], b + ".tri_mul_out.right_gate");
        collect_linear(tri_out_gate_[i], b + ".tri_mul_out.gate");
        collect_layernorm(tri_out_output_layernorm_[i], b + ".tri_mul_out.output_layernorm");
        collect_linear(tri_out_out_proj_[i], b + ".tri_mul_out.out_proj");

        // TriangleMultiplication in
        collect_layernorm(tri_in_layernorm_[i], b + ".tri_mul_in.layernorm");
        collect_linear(tri_in_left_proj_[i], b + ".tri_mul_in.left_proj");
        collect_linear(tri_in_right_proj_[i], b + ".tri_mul_in.right_proj");
        collect_linear(tri_in_left_gate_[i], b + ".tri_mul_in.left_gate");
        collect_linear(tri_in_right_gate_[i], b + ".tri_mul_in.right_gate");
        collect_linear(tri_in_gate_[i], b + ".tri_mul_in.gate");
        collect_layernorm(tri_in_output_layernorm_[i], b + ".tri_mul_in.output_layernorm");
        collect_linear(tri_in_out_proj_[i], b + ".tri_mul_in.out_proj");

        // IterBlock 3D SE 参数
        collect_linear(iter_embed_x_[i], b + ".embed_x");
        collect_linear(iter_embed_e_[i], b + ".embed_e");
        collect_layernorm(iter_norm_node_3d_[i], b + ".norm_node_3d");
        collect_layernorm(iter_norm_edge_3d_[i], b + ".norm_edge_3d");
        collect_layernorm(iter_norm_msa_3d_[i], b + ".norm_msa_3d");
        collect_layernorm(iter_norm_pair_3d_[i], b + ".norm_pair_3d");

        // IterBlock forward 内部参数
        collect_layernorm(iter_state2msa_norm_[i], b + ".state2msa_norm");
        collect_linear(iter_state2msa_linear_[i], b + ".state2msa_linear");
        collect_layernorm(iter_pair2msa_norm_[i], b + ".pair2msa_norm");
        collect_layernorm(is_full ? full_msa2pair_norm_[i]      : iter_msa2pair_norm_[i],      b + ".msa2pair_norm");
        collect_linear(   is_full ? full_msa2pair_left_proj_[i] : iter_msa2pair_left_proj_[i], b + ".msa2pair_left_proj");
        collect_linear(   is_full ? full_msa2pair_right_proj_[i]: iter_msa2pair_right_proj_[i], b + ".msa2pair_right_proj");
        collect_linear(   is_full ? full_msa2pair_out_proj_[i]  : iter_msa2pair_out_proj_[i],  b + ".msa2pair_out_proj");
        collect_linear(iter_pair2pair_rbf_proj_[i], b + ".pair2pair_rbf_proj");
        collect_layernorm(iter_pair2pair_state_norm_[i], b + ".pair2pair_state_norm");
        collect_linear(iter_pair2pair_left_proj_[i], b + ".pair2pair_left_proj");
        collect_linear(iter_pair2pair_right_proj_[i], b + ".pair2pair_right_proj");
        collect_linear(iter_pair2pair_gate_proj_[i], b + ".pair2pair_gate_proj");
    }

    // MSAGlobalColAttention 参数 (FullBlock only, n_extra) — D_MSA_FULL=64 维
    for (int i = 0; i < c_n_extra; ++i) {
        const std::string b = "extra." + std::to_string(i);
        collect_linear(full_msa_global_col_Wq_[i], b + ".attention.global_col.Wq");
        collect_linear(full_msa_global_col_Wk_[i], b + ".attention.global_col.Wk");
        collect_linear(full_msa_global_col_Wv_[i], b + ".attention.global_col.Wv");
        collect_linear(full_msa_global_col_to_b_[i], b + ".attention.global_col.to_b");
        collect_linear(full_msa_global_col_to_g_[i], b + ".attention.global_col.to_g");
        collect_linear(full_msa_global_col_to_out_[i], b + ".attention.global_col.to_out");
    }

    // PositionalEncoding 参数 (每 block 2 个 EmbeddingLayer, n_iter 组)
    for (int i = 0; i < c_n_iter; ++i) {
        const std::string b = (i < c_n_extra ? "extra." + std::to_string(i)
                                             : "main." + std::to_string(i - c_n_extra));
        collect_embedding(pos_enc_emb_res_[i], b + ".pos_enc_emb_res");
        collect_embedding(pos_enc_emb_atom_[i], b + ".pos_enc_emb_atom");
    }

    // RefineBlock 参数 (n_refine_blocks)
    for (int i = 0; i < (int)refine_norm_msa_.size(); ++i) {
        const std::string b = "refine." + std::to_string(i);
        collect_layernorm(refine_norm_msa_[i], b + ".norm_msa");
        collect_layernorm(refine_norm_pair_[i], b + ".norm_pair");
        collect_layernorm(refine_norm_state_[i], b + ".norm_state");
        collect_linear(refine_embed_x_[i], b + ".embed_x");
        collect_layernorm(refine_norm_node_[i], b + ".norm_node");
        collect_linear(refine_embed_e1_[i], b + ".embed_e1");
        collect_layernorm(refine_norm_edge1_[i], b + ".norm_edge1");
        collect_linear(refine_embed_e2_[i], b + ".embed_e2");
        collect_layernorm(refine_norm_edge2_[i], b + ".norm_edge2");
    }
}

// ===== LoRA 低秩微调 =====
// 收集所有 LinearLayer 指针（与 collect_params_with_names 的遍历顺序一致，每层一次）。
std::vector<LinearLayer*> PPMLModel::collect_linear_layers() {
    std::vector<TensorF32*> tmp;
    std::vector<std::string> names;
    std::vector<LinearLayer*> layers;
    collect_params_with_names(tmp, names, &layers);
    return layers;
}

int PPMLModel::enable_lora_all(int rank, float alpha, const std::string& name_substr) {
    std::vector<TensorF32*> tmp;
    std::vector<std::string> names;
    std::vector<LinearLayer*> layers;
    collect_params_with_names(tmp, names, &layers);

    int n = 0;
    // names 与 layers 一一对应（collect_linear 每层 push 一次 layers，且 weight/bias 顺序）
    // 但 collect_linear 在 weight 前 push layer，故每层对应名字 = 第一个属于它的名字 (name.weight)。
    // 更稳健：用 "x.weight" 前缀匹配。layers 顺序 == 每个 collect_linear 调用顺序，但名字
    // 可能含 weight/bias 两项 → 需按层聚合名字。改为：收集时记录层名，这里用遍历对齐：
    // 由于 layers 数量 = 有 weight 或 bias 的层数（每 collect_linear 一次），而 names 含
    // weight+bias+lora 多项，无法直接一一对齐。故采用双收集：先收集 (layers, 每层首名)。
    // 简化：直接遍历 layers，用 weight 指针在 tmp 中反查所属名字。
    std::vector<std::string> layer_names;
    layer_names.reserve(layers.size());
    for (LinearLayer* ll : layers) {
        // 找到该层 weight 在 tmp 中的索引 → 对应 names 里的名字（去尾 .weight/.bias）
        std::string nm;
        for (size_t i = 0; i < tmp.size(); ++i) {
            if (tmp[i] == (ll->weight() ? ll->weight() : ll->bias())) {
                nm = names[i];
                break;
            }
        }
        // 去掉 ".weight"/".bias" 后缀得层名
        const std::string suffix_w = ".weight";
        const std::string suffix_b = ".bias";
        if (nm.size() > suffix_w.size() && nm.compare(nm.size()-suffix_w.size(), suffix_w.size(), suffix_w) == 0)
            nm = nm.substr(0, nm.size() - suffix_w.size());
        else if (nm.size() > suffix_b.size() && nm.compare(nm.size()-suffix_b.size(), suffix_b.size(), suffix_b) == 0)
            nm = nm.substr(0, nm.size() - suffix_b.size());
        layer_names.push_back(nm);
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        if (!name_substr.empty() && layer_names[i].find(name_substr) == std::string::npos) continue;
        if (layers[i]->lora_rank() > 0) continue;   // 已启用，跳过
        layers[i]->enable_lora(rank, alpha);
        n++;
    }
    if (n > 0) {
        std::cerr << "[lora] enable_lora_all: rank=" << rank << " alpha=" << alpha
                  << " substr=\"" << name_substr << "\" → " << n << " layers" << std::endl;
    }
    return n;
}

void PPMLModel::freeze_all() {
    std::vector<TensorF32*> tmp;
    std::vector<std::string> names;
    std::vector<LinearLayer*> layers;
    collect_params_with_names(tmp, names, &layers);
    for (LinearLayer* ll : layers) ll->freeze();
    std::cerr << "[lora] freeze_all: " << layers.size() << " linear layers frozen" << std::endl;
}

std::vector<TensorF32*> PPMLModel::params() {
    std::vector<TensorF32*> out;
    collect_all_params(out);
    return out;
}

void PPMLModel::transfer_params_to_backend() {
    if (!scheduler_ || !cpu_backend_) return;

    // 幂等保护: 若参数已被分配进 backend buffer (buffer_ != nullptr), 跳过。
    // 这样 ensure_backend_ready / load_weights 多次调用不会重复搬迁覆盖 data_ 指针。
    {
        std::vector<TensorF32*> probe;
        collect_all_params(probe);
        for (auto* t : probe) {
            if (t->data() != nullptr && t->buffer_ != nullptr) {
                return;  // 已搬迁过
            }
        }
    }

    const BufferType* cpu_buft = cpu_backend_->buffer_type();

    // ===== Step 1: 收集所有需要搬迁的参数 tensor =====
    std::vector<TensorF32*> param_tensors;
    collect_all_params(param_tensors);

    // ===== Step 2: 计算总大小并分配 CPU backend buffer =====
    size_t total_size = 0;
    size_t alignment = cpu_buft->get_alignment();

    for (auto* t : param_tensors) {
        if (t->data() != nullptr) {
            total_size += GGML_PAD(t->nbytes(), alignment);
        }
    }

    if (total_size == 0 || param_tensors.empty()) return;

    // 诊断（2026-09-11）：参数总量异常大时逐项打印 —— 定位"超大分配"是否来自参数搬迁
    //   buffer（usage=WEIGHTS）：某个 param 的 nbytes 异常 = shape 被污染 /
    //   把激活型（随 L/E 规模变化）张量当参数注册。
    if (total_size > (8ull << 30)) {
        fprintf(stderr, "[params] transfer: total=%.2f GB (n_params=%zu) —— 单参数 >1GB 列表：\n",
                (double)total_size / (1024.0*1024.0*1024.0), param_tensors.size());
        for (auto* t : param_tensors) {
            if (!t || !t->data() || t->nbytes() <= (1ull << 30)) continue;
            fprintf(stderr, "[params]   nbytes=%.2f GB op=%d ndim=%d dims=[",
                    (double)t->nbytes() / (1024.0*1024.0*1024.0), (int)t->op, t->shape().ndim());
            for (int d = 0; d < t->shape().ndim(); ++d)
                fprintf(stderr, "%s%lld", (d ? "," : ""), (long long)t->shape().dims[d]);
            fprintf(stderr, "]\n");
        }
    }

    // ===== Step 3: 分配 buffer 并搬迁数据（对标 ggml_backend_alloc_ctx_tensors + ggml_backend_tensor_set）=====
    Buffer* param_buf = alloc_buffer(const_cast<BufferType*>(cpu_buft), total_size, BufferUsage::WEIGHTS);
    if (!param_buf) return;

    // 将 buffer 所有权交给 PPMLModel
    param_buffers_.emplace_back(param_buf);

    TensorAllocator tallocr(param_buf);

    for (auto* t : param_tensors) {
        if (t->data() == nullptr) continue;

        // 保存旧数据指针（指向 context arena）
        float* old_data = t->data();
        size_t old_nbytes = t->nbytes();

        // 在 backend buffer 中分配新空间（覆盖 data_）
        if (!tallocr.alloc(t)) {
            // buffer 空间不足（理论上不会发生）
            continue;
        }

        // 对标 ggml_backend_tensor_set：将旧数据拷贝到新 buffer
        param_buf->set_tensor(t, old_data, t->buffer_offs_, old_nbytes);

        // 旧数据在 context arena 中，无法释放，但 data_ 已指向新 buffer
        // 后续可通过 t->buffer_ 和 t->buffer_offs_ 访问
    }
}

void PPMLModel::save_weights(const std::string& path) const {
    // ============================================================================
    // 权重落盘（**保留为工具**，2026-09-15 实现）—— 供"权重逐字节对比"判据使用。
    //   ⚠️ 但它产出的 `*.bin` 是**开发期临时产物**：判据完成后应删除，不要当长期权重格式 ✗
    //     （正式检查点：分片 manifest + 优化器状态 + 断点续训，见 PLAN_SHARDING §2.2 S6）。
    //   目的：让"权重逐字节对比"这条端到端判据可用（单机 vs 远端执行 vs 数据并行分片两端）。
    //   背景：此前本函数是空壳 ⇒ `experiments/dist/probe_weight_diff.sh` 之类的判据无法执行。
    //   格式（自描述、便于跨进程/机器比较，固定顺序来自 collect_params_with_names）：
    //     magic "PPW1" | u32 n | [ u32 name_len | name | u64 numel | f32 data[numel] ] × n
    //   * peer 模式（PPML_ROLE=peer）自动追加 `.rank<N>` 后缀 ⇒ 两端各存一份、互不覆盖 ✓
    //   * 只存参数（不含优化器状态）；**分片下的合并**由"两端权重逐位一致"来判定 ⇒ 无需 manifest 合并器 ✓
    //   ⚠️ 正式方案（分片 manifest + 优化器状态 + 断点续训）见 PLAN_SHARDING §2.2 S6；
    //      **判据完成后请删除本实现与产出的 *.bin**（不要把它当长期权重格式 ✗）。
    // ============================================================================
    std::string out = path;
    const char* role = std::getenv("PPML_ROLE");
    if (role && std::strcmp(role, "peer") == 0) {
        const int rk = std::getenv("PPML_REMOTE_RANK") ? std::atoi(std::getenv("PPML_REMOTE_RANK")) : 0;
        out += ".rank" + std::to_string(rk);
    }
    std::cout << "Saving weights to: " << out << " (DEV-ONLY 判据设施)" << std::endl;

    std::vector<TensorF32*> ps;
    std::vector<std::string> ns;
    const_cast<PPMLModel*>(this)->collect_params_with_names(ps, ns);   // dev 设施：只读收集
    std::FILE* f = std::fopen(out.c_str(), "wb");
    if (!f) {
        std::cerr << "  [save_weights] 打开失败: " << out << std::endl;
        return;
    }
    auto wr = [&](const void* p, size_t n) { if (n) std::fwrite(p, 1, n, f); };
    wr("PPW1", 4);
    const uint32_t n_t = (uint32_t)ps.size();
    wr(&n_t, sizeof(n_t));
    size_t total_bytes = 0;
    for (size_t i = 0; i < ps.size(); ++i) {
        TensorF32* t = ps[i];
        const std::string& nm = (i < ns.size()) ? ns[i] : std::string("<unnamed>");
        const uint32_t nl = (uint32_t)nm.size();
        wr(&nl, sizeof(nl));
        wr(nm.data(), nl);
        const uint64_t n_el = (uint64_t)(t ? t->numel() : 0);
        wr(&n_el, sizeof(n_el));
        if (!t || n_el == 0) continue;
        std::vector<float> buf((size_t)n_el);
        const size_t bytes = (size_t)n_el * sizeof(float);
        if (t->buffer_) {
            t->buffer_->get_tensor(t, buf.data(), t->buffer_offs_, bytes);
        } else if (t->data()) {
            std::memcpy(buf.data(), t->data(), bytes);
        } else {
            std::cerr << "  [save_weights] 参数无数据: " << nm << " ⇒ 写 0" << std::endl;
            std::fill(buf.begin(), buf.end(), 0.0f);
        }
        wr(buf.data(), bytes);
        total_bytes += bytes;
    }
    std::fflush(f);
    std::fclose(f);
    std::cout << "  [save_weights] tensors=" << ps.size() << " payload=" << (total_bytes / (1024 * 1024))
              << " MB" << std::endl;
}

} // namespace ppml

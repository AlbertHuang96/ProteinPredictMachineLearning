// verify_template.cpp
// 验证 forward_graph 的模板特征注入：
//   用 P62891 (L=51, 内存小) 数据，构造【合成模板特征】填充 ModelInput（含模板）与
//   无模板输入（t1d/t2d 空），各跑一次 forward_graph，对比 state/pair 数值。
//   若两者不同 → 模板特征被正确接入计算图；且全程不崩溃 → 接入正确。
//
// 用法: ppml_verify_template
// 说明: 运行在 CPU，小模型(各 1 block)。保留 coords（rbf 需要；L=51 下 SE3 也应可控）。
#include "ppml/Model.h"
#include "ppml/DataLoader.h"
#include "ppml/ComputeGraph.h"
#include "ppml/Context.h"
#include <iostream>
#include <cmath>

using namespace ppml;

// 图节点回落为值张量（等价 PPML.cpp 匿名空间 compute_and_read）
static void compute_read(PPMLModel& model, TensorF32* node, TensorF32& dst) {
    if (!node) return;
    PPMLContext* ctx = &context();
    ComputeGraph* cg = ComputeGraph::new_graph(ctx);
    cg->build_forward_expand(node);
    Backend* backend = model.active_backend();
    backend->graph_compute(cg);
    if (node->data() != nullptr) {
        size_t bytes = static_cast<size_t>(node->numel()) * sizeof(float);
        const Shape& g = node->shape();
        std::vector<int64_t> v;
        for (int i = (int)g.ndim() - 1; i >= 0; --i) v.push_back(g.dims[i]);
        dst.~TensorF32();
        new (&dst) TensorF32(Shape(v), Device::CPU);
        std::memcpy(dst.data(), node->data(), bytes);
    }
}

static float l2_diff(const TensorF32& a, const TensorF32& b) {
    if (a.data() == nullptr || b.data() == nullptr) return -1.0f;
    if (a.numel() != b.numel()) return -1.0f;
    float s = 0.0f, na = 0.0f;
    for (int64_t i = 0; i < a.numel(); ++i) {
        float d = a.data()[i] - b.data()[i];
        s += d * d; na += a.data()[i] * a.data()[i];
    }
    return (na > 0) ? std::sqrt(s / na) : 0.0f;
}

static float abs_energy(const TensorF32& a) {
    if (a.data() == nullptr || a.numel() == 0) return 0.0f;
    float s = 0.0f;
    for (int64_t i = 0; i < a.numel(); ++i) s += std::fabs(a.data()[i]);
    return s;
}

// 构造合成模板特征（非零，验证注入路径）。用与 DataLoader build_template_features 相同的维度。
static void synth_template(ModelInput& mi, int B, int T, int L) {
    { TensorF32 t(Shape{B,T,L,80}, Device::CPU);
      for (int64_t i = 0; i < t.numel(); ++i) t.data()[i] = 0.3f * ((i % 7) + 1) / 7.0f;
      mi.t1d = std::move(t); }
    { TensorF32 t(Shape{B,T,L,30}, Device::CPU);
      for (int64_t i = 0; i < t.numel(); ++i) t.data()[i] = 0.5f * ((i % 5) + 1) / 5.0f;
      mi.tor_feat = std::move(t); }
    { TensorF32 t(Shape{B,T,L,L,68}, Device::CPU);
      for (int64_t i = 0; i < t.numel(); ++i) t.data()[i] = 0.2f * ((i % 3) + 1) / 3.0f;
      mi.t2d = std::move(t); }
    { TensorF32 t(Shape{B,T,L}, Device::CPU);
      for (int64_t i = 0; i < t.numel(); ++i) t.data()[i] = 1.0f;
      mi.template_mask = std::move(t); }
}

int main() {
    // ---- P62891 (L=51) 数据 ----
    const std::string a3m_path   = "data/training_batch_data/P62891_alignment.a3m";
    const std::string fasta_path = "data/P62891.fasta";
    const std::string csv_path   = "data/training_batch_data/4ug0_P62891_mapping.csv";

    PPMLDataLoader loader("", "", 512, 4, 2048);
    std::string seq = PPMLDataLoader::read_fasta_first_sequence(fasta_path);
    int L = static_cast<int>(seq.length());
    const int B = 1, T = 4;

    // 有模板输入：正常加载（无模板目录 → t1d 空），再手动填充合成模板特征
    ModelInput in_t = loader.load_from_files(a3m_path, seq, csv_path, "", "");
    std::cout << "[load] base L=" << L << " t1d=" << in_t.t1d.numel()
              << " t2d=" << in_t.t2d.numel() << " coords=" << in_t.coords.numel() << std::endl;
    synth_template(in_t, B, T, L);
    std::cout << "[load] with-template t1d=" << in_t.t1d.numel()
              << " t2d=" << in_t.t2d.numel() << " tor=" << in_t.tor_feat.numel()
              << " tmpl_mask=" << in_t.template_mask.numel() << std::endl;

    // 无模板输入（t1d 等保持空）
    ModelInput in_nt = loader.load_from_files(a3m_path, seq, csv_path, "", "");
    std::cout << "[load] no-template   t1d=" << in_nt.t1d.numel() << std::endl;

    // ---- 小模型（各 1 block，CPU）----
    PPMLConfig config;
    config.n_extra_blocks = 1;
    config.n_main_blocks  = 1;
    config.n_refine_blocks = 1;
    PPMLModel model(config);
    model.to(Device::CPU);
    model.train();
    std::cout << "[model] created, device=" << (int)model.device() << std::endl;

    // ---- 1) 无模板前向 ----
    std::cout << "=== forward_graph (no-template) ===" << std::endl;
    GraphOutput go_nt = model.forward_graph(in_nt, /*enable_se3=*/false);
    TensorF32 st_nt, pr_nt;
    compute_read(model, go_nt.state, st_nt);
    compute_read(model, go_nt.pair,  pr_nt);
    std::cout << "  state numel=" << (go_nt.state ? go_nt.state->numel() : 0)
              << " pair numel=" << (go_nt.pair ? go_nt.pair->numel() : 0) << std::endl;
    std::cout << "  state abs_energy=" << abs_energy(st_nt)
              << " pair abs_energy=" << abs_energy(pr_nt) << std::endl;

    // ---- 2) 有模板前向 ----
    std::cout << "=== forward_graph (with-template) ===" << std::endl;
    GraphOutput go_t = model.forward_graph(in_t, /*enable_se3=*/false);
    std::cout << "  [diag] go_t.state data=" << (go_t.state ? (void*)go_t.state->data() : 0)
              << " numel=" << (go_t.state ? go_t.state->numel() : 0) << std::endl;
    TensorF32 st_t, pr_t;
    compute_read(model, go_t.state, st_t);
    compute_read(model, go_t.pair,  pr_t);
    std::cout << "  [diag] st_t data=" << (void*)st_t.data() << " numel=" << st_t.numel() << std::endl;
    std::cout << "  state numel=" << (go_t.state ? go_t.state->numel() : 0)
              << " pair numel=" << (go_t.pair ? go_t.pair->numel() : 0) << std::endl;
    std::cout << "  state abs_energy=" << abs_energy(st_t)
              << " pair abs_energy=" << abs_energy(pr_t) << std::endl;

    // ---- 3) 对比 ----
    float d_state = l2_diff(st_t, st_nt);
    float d_pair  = l2_diff(pr_t, pr_nt);
    std::cout << "========================================" << std::endl;
    std::cout << "[RESULT] state rel-diff = " << d_state << std::endl;
    std::cout << "[RESULT] pair  rel-diff = " << d_pair  << std::endl;
    bool injected = (d_state > 1e-6f || d_pair > 1e-6f);
    std::cout << "[VERDICT] template injection "
              << (injected ? "SUCCESS (state/pair differ)" : "FAIL (no effect)") << std::endl;

    return injected ? 0 : 1;
}

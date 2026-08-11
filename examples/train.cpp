#include "rfaa/Model.h"
#include "rfaa/DataLoader.h"
#include "rfaa/ComputeGraph.h"
#include "rfaa/Context.h"
#include "rfaa/PythonBridge.h"
#include "rfaa/ONNXExporter.h"
#include "rfaa/GradientClipper.h"
#include "rfaa/GGUF.h"
#include "rfaa/LDDT.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <sstream>

using namespace rfaa;

// ============================================================================
// 工具: 将值张量 (ModelOutput.coords / ModelInput.true_coords) 包装为图节点 leaf
// 注意: 当前 RFAAModel::forward 基于数值张量计算, 尚未接入 autograd 图。
//       这里将 pred/true coords 拷入 context leaf 节点, 使 fape_loss 等图节点
//       损失函数可以正确构造计算图。待模型接入图后端后, 此包装可移除。
// ============================================================================
TensorF32* wrap_value_as_leaf(const TensorF32& t, const std::vector<int64_t>& dims) {
    int64_t ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < dims.size() && i < 4; i++) ne[i] = dims[i];
    TensorF32* leaf = context().new_tensor<float>(static_cast<int>(dims.size()), ne);
    // 数据按扁平顺序拷入. Tensor 是 move-only (禁拷贝), 故:
    //   - CUDA 上: 用 t.cpu() 生成新的 CPU 张量 (可移动) 再取 data
    //   - CPU 上:  直接取 t.data() 拷入
    if (t.device() == Device::CUDA) {
        TensorF32 tcpu = t.cpu();  // 移动构造, 合法
        std::memcpy(leaf->data(), tcpu.data(), tcpu.numel() * sizeof(float));
    } else {
        std::memcpy(leaf->data(), t.data(), t.numel() * sizeof(float));
    }
    return leaf;
}

// ============================================================================
// 工具: 将参数 Tensor 的数据安全读取到一块 CPU 内存 (方案 b)
//   - 若 tensor 已迁入 backend buffer, 通过 buffer_->get_tensor 读取
//     (自动处理 CPU/GPU buffer 类型)
//   - 否则若 device 为 CPU, 直接 memcpy data_
//   - CUDA 上无 buffer 的裸张量, 退回 t->cpu()
// 返回: 指向连续 CPU 数据的 buffer (vector 生命期由调用方保证存活到写完)
// ============================================================================
std::vector<float> read_tensor_cpu(const TensorF32* t) {
    size_t bytes = t->nbytes();
    std::vector<float> buf(t->numel());

    if (t->buffer_) {
        // 从 backend buffer 读取 (buffer_->get_tensor 按偏移读)
        // DefaultBuffer get_tensor will deal with CPU/GPU buffer type automatically
        t->buffer_->get_tensor(t, buf.data(), t->buffer_offs_, bytes);
    } else if (t->device() == Device::CPU) {
        std::memcpy(buf.data(), t->data(), bytes);
    } else {
        // CUDA 且无 buffer: 用 t->cpu() 生成一份 CPU 拷贝
        // cuda backend synchronize
        TensorF32 tcpu = t->cpu();   // 移动语义, 临时对象
        std::memcpy(buf.data(), tcpu.data(), bytes);
    }
    return buf;
}

// ============================================================================
// 开发功能: 统计所有权重占用内存大小 (仅计算, 不写文件)
// 返回字节数。同时打印每个张量的形状与字节数 (便于调优量化/显存)。
// ============================================================================
size_t estimate_params_memory(const RFAAModel& model) {
    // params() 收集的指针顺序固定, 但此处通过尺寸计算内存;
    // 由于 params() 为非 const, 这里用一个 const 转换不了, 故通过 const_cast 调用
    // (仅读取 nbytes/shape, 不修改任何状态, 安全)
    auto& m = const_cast<RFAAModel&>(model);
    auto params = m.params();

    size_t total_bytes = 0;
    std::cout << "  [ParamsMemory] #tensors=" << params.size() << std::endl;
    size_t i = 0;
    for (auto* t : params) {
        if (!t || t->numel() == 0) continue;
        size_t bytes = t->nbytes();
        total_bytes += bytes;
        /* if (i < 64) {  // 只打印前 64 个, 避免刷屏
            std::cout << "    tensor_" << i << "  shape=(";
            for (int d = 0; d < t->ndim(); d++) {
                if (d) std::cout << ",";
                std::cout << t->dims()[d];
            }
            std::cout << ")  " << bytes << " B" << std::endl;
        } */
        i++;
    }

    double mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
    std::cout << "  [ParamsMemory] total = " << total_bytes << " B = "
              << mb << " MB" << std::endl;
    return total_bytes;
}

// 计算参数总字节数 (供 save_checkpoint 元数据用)
size_t total_bytes_of(const std::vector<TensorF32*>& params) {
    size_t total = 0;
    for (auto* t : params) total += t->nbytes();
    return total;
}

// ============================================================================
// 保存 checkpoint (方案 b): 先把所有参数临时拷贝到 CPU, 再写入 GGUF
//   path: 输出 gguf 文件路径
//   epoch: 当前已完成 epoch (0-based), total_epochs, loss, elapsed_sec
// ============================================================================
void save_checkpoint(RFAAModel& model, const std::string& path,
                     int epoch, int total_epochs, float loss, double elapsed_sec) {
    // 1. 收集参数 (及一一对应的语义名, 含 block/attention 等信息)
    std::vector<TensorF32*> params;
    std::vector<std::string> param_names;
    model.collect_params_with_names(params, param_names);
    if (params.empty()) {
        throw std::runtime_error("save_checkpoint: no parameters to save");
    }
    if (param_names.size() != params.size()) {
        throw std::runtime_error("save_checkpoint: tensor name count mismatch");
    }

    // 2. 方案 b: 临时 CPU 拷贝 — 每张量读成独立 vector, 存活到 save_gguf 完成
    std::vector<std::vector<float>> cpu_copies;
    cpu_copies.reserve(params.size());
    for (auto* p : params) {
        cpu_copies.push_back(read_tensor_cpu(p));
    }

    // 3. 因为 save_gguf 需要 Tensor* 集合, 我们用轻量的非 owning Tensor 视图
    //    指向 CPU 拷贝 (数据连续), 生命周期与 cpu_copies 一致
    std::vector<TensorF32> views;
    views.reserve(params.size());
    std::vector<TensorF32*> gguf_params;
    gguf_params.reserve(params.size());
    for (size_t i = 0; i < params.size(); i++) {
        // 注意: Tensor 禁止拷贝/赋值, 但允许移动; 用 emplace 构造非 owning 视图
        //       view(data, shape, CPU, own=false) 不会拥有数据, 不会 free cpu_copies
        views.emplace_back(params[i]->shape(), cpu_copies[i].data(),
                           Device::CPU, /*own=*/false);
        gguf_params.push_back(&views.back());
    }

    // 4. 写入 GGUF, 内嵌训练状态元数据 + 语义化张量名
    save_gguf(gguf_params, path,
              {
                  {"epoch", float(epoch + 1)},       // 1-based epoch 序号
                  {"total_epochs", float(total_epochs)},
                  {"loss", loss},
                  {"elapsed_sec", float(elapsed_sec)},
                  {"param_bytes", float(total_bytes_of(params))},
              },
              {
                  {"arch", "rfaa_v1"},
                  {"checkpoint", "train"},
              },
              param_names);
}

int main(int argc, char* argv[]) {
    // 1. 初始化 Python 桥接 (用于数据加载和预处理)
    auto& py = PythonBridge::instance();
    if (!py.initialize()) {
        std::cerr << "Failed to initialize Python" << std::endl;
        return 1;
    }
    
    // 2. 创建模型
    RFAAConfig config;
    config.d_msa = 256;
    config.d_pair = 128;
    config.d_state = 32;
    config.n_extra_blocks = 4;
    config.n_main_blocks = 8;
    config.n_refine_blocks = 4;
    
    RFAAModel model(config);
    
    // 3. 转移到 GPU
    model.to(Device::CUDA);
    model.train();
    
    std::cout << "Model created and moved to CUDA" << std::endl;
    
    // 4. 加载预训练权重 (通过 Python 桥接)
    // 暂时没有预训练权重, 先注释掉, 待有权重文件后再启用
    // if (argc > 1) {
    //     std::string weights_path = argv[1];
    //     ProteinTools::load_torch_weights(model, weights_path);
    // }
    
    // ============================================================
    // 数据加载: 通过 RFAADataLoader 从 A3M + CSV mapping 加载
    // CSV 提供真实坐标 (true_coords), 作为 FAPE 的 ground truth
    // hhr 模板文件暂时没有, 放在最后, 默认为空 (暂未接入模板)
    // ============================================================
    // 命令行参数: train <a3m> <fasta> <csv_mapping> [hhr]
    // 例如: train 1a00.a3m data/P04637.fasta data/1a00_P69905_mapping.csv
    //       或带 hhr: train 1a00.a3m data/P04637.fasta data/1a00_P69905_mapping.csv 1a00.hhr
    std::string a3m_path    = (argc > 1) ? argv[1] : "query.a3m";
    std::string fasta_path  = (argc > 2) ? argv[2] : "data/P04637.fasta";
    std::string csv_path    = (argc > 3) ? argv[3] : "data/1a00_P69905_mapping.csv";
    std::string hhr_path    = (argc > 4) ? argv[4] : "";  // hhr 放最后, 默认为空

    // 从 FASTA 文件读取查询序列 (只读第一条)
    std::string sequence;
    try {
        sequence = RFAADataLoader::read_fasta_first_sequence(fasta_path);
        std::cout << "Read query sequence from " << fasta_path
                  << " (L=" << sequence.length() << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to read FASTA: " << e.what() << std::endl;
        return 1;
    }

    RFAADataLoader loader("", "", 512, 4, 2048);
    ModelInput input = loader.load_from_files(a3m_path, sequence, csv_path, hhr_path);
    int L = static_cast<int>(sequence.length());

    std::cout << "Loaded data. L=" << L
              << " true_coords shape=(" << (input.true_coords.numel() > 0 ? 1 : 0)
              << "," << L << ",3,3)" << std::endl;

    // 若没有真实坐标, 用一个占位 coords 供前向使用
    if (input.coords.numel() == 0) {
        input.coords = zeros<float>({1, L, 3, 3}, Device::CPU);
    }
    if (input.true_coords.numel() == 0) {
        input.true_coords = zeros<float>({1, L, 3, 3}, Device::CPU);
    }
    // 模型在 CUDA 上, 把输入搬到 CUDA
    //input.msa_latent = input.msa_latent.to(Device::CUDA);
    //input.seq_tokens = input.seq_tokens.to(Device::CUDA);
    //input.coords     = input.coords.to(Device::CUDA);
    //input.true_coords= input.true_coords.to(Device::CUDA);
    
    // 5. 训练循环
    const int num_epochs = 10;

    // ---- Checkpoint 配置 ----
    // 每隔 checkpoint_interval 个 epoch 保存一次权重 (GGUF)。
    // TODO(增强): 增加命令行参数 --ckpt-interval N 覆盖此默认值,
    //             例: 解析 argc/argv 后 ckpt_interval = atoi(argv[k])。
    int ckpt_interval = 2;                     // 默认每 2 个 epoch 保存一次
    const std::string ckpt_dir = "checkpoints"; // checkpoint 输出目录

    // 可选: 开发期统计所有权重的内存占用 (不写文件)
    // 若要打印初始权重占用的内存, 取消下面一行注释:
    // size_t init_param_bytes = estimate_params_memory(model);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // ===== 计时代码: epoch 级 + 前向/损失阶段子计时 =====
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        float epoch_loss = 0.0f;
        
        // 前向传播
        auto fwd_start = std::chrono::high_resolution_clock::now();
        auto output = model.forward(input);
        auto fwd_end = std::chrono::high_resolution_clock::now();
        auto fwd_ms = std::chrono::duration_cast<std::chrono::milliseconds>(fwd_end - fwd_start).count();
        
        // ============================================================
        // 组装 total_loss 的 5 个组成部分 (调用正确的损失函数)
        //   total_loss(loss_fape, loss_chi, loss_distogram, loss_msa, loss_conf)
        // ============================================================
        const int B = 1;
        const int N_atoms  = B * L * 3;   // 每残基 N/CA/C
        const int N_frames = B * L;

        // --- pred / true coords 展平为 (N_atoms, 3) 图节点 ---
        TensorF32 pred_flat = output.coords.view({N_atoms, 3});
        TensorF32 true_flat = input.true_coords.view({N_atoms, 3});
        TensorF32* pred_node = wrap_value_as_leaf(pred_flat, {N_atoms, 3});
        TensorF32* true_node = wrap_value_as_leaf(true_flat, {N_atoms, 3});

        // --- FAPE frame indices: 每残基一帧 [N, CA, C] ---
        TensorF32* frame_idx = build_frame_atom_indices(B, L);          // (B*L, 3)
        TensorF32* frames_mask    = constant_ones({N_frames});           // (B*L,)
        TensorF32* positions_mask = constant_ones({N_atoms});            // (B*L*3,)

        // 1) FAPE loss
        FAPEConfig fape_cfg;
        fape_cfg.length_scale = 10.0f;
        fape_cfg.d_clamp      = 10.0f;
        fape_cfg.epsilon      = 1e-4f;
        TensorF32* loss_fape_node =
            loss(fape_loss(pred_node, true_node, frame_idx, frames_mask, positions_mask, fape_cfg));

        // 2) Chi (扭转角) loss — 接入 supervised_chi_loss(unnormed, gt, chi_mask, seq_mask)
        //    unnormed : output.alpha (B,L,7,2) → view (B*L,7,2) → wrap 图 {2,7,N}
        //    gt       : input.gt_chi  (B,L,7,2) → view (B*L,7,2) → wrap 图 {2,7,N}
        //    chi_mask : input.chi_mask(B,L,7)   → view (B*L,7)   → wrap 图 {7,N}
        //    seq_mask : 全 1 (B*L,1) → wrap 图 {1,N}
        TensorF32* loss_chi_node = loss(constant_scalar(0.0f));
        if (input.gt_chi.numel() > 0 && output.alpha.numel() > 0) {
            int64_t chi_N = input.gt_chi.shape().dims[0] * input.gt_chi.shape().dims[1]; // B*L
            TensorF32 chi_unnormed = output.alpha.view({chi_N, 7, 2});
            TensorF32 chi_gt       = input.gt_chi.view({chi_N, 7, 2});
            TensorF32 chi_mask_2d  = input.chi_mask.view({chi_N, 7});
            TensorF32 seq_mask_2d  = TensorF32({chi_N, 1}, Device::CPU);
            seq_mask_2d.zero_();
            for (int64_t i = 0; i < chi_N; ++i) seq_mask_2d.data()[i] = 1.0f;  // 全残基有效
            TensorF32* unnormed_node = wrap_value_as_leaf(chi_unnormed, {2, 7, chi_N});
            TensorF32* gt_node       = wrap_value_as_leaf(chi_gt,       {2, 7, chi_N});
            TensorF32* cmask_node    = wrap_value_as_leaf(chi_mask_2d,  {7, chi_N});
            TensorF32* smask_node    = wrap_value_as_leaf(seq_mask_2d,  {1, chi_N});
            loss_chi_node = loss(supervised_chi_loss(unnormed_node, gt_node, cmask_node, smask_node, 0.5f, 0.5f));
        }

        // 3) Distogram loss — 接入 distogram_loss(4 logits, 4 onehot, pair_mask)
        //    布局: 值 (B*L*L, bins) 行主序(bins 最内) → wrap 图 {bins, B*L*L}
        //          pair_mask (B*L*L) → wrap 图 {B*L*L}
        //    distogram_loss 内 sum_rows 沿 dims[0]=bins(最内) 求和 → per-pair CE
        TensorF32* loss_distogram_node = loss(constant_scalar(0.0f));
        if (input.D_onehot.numel() > 0 && output.distogram.numel() > 0) {
            int64_t dg_N = input.D_onehot.shape().dims[0]
                         * input.D_onehot.shape().dims[1]
                         * input.D_onehot.shape().dims[2];  // B*L*L
            TensorF32 dg_dist = output.distogram.view({dg_N, 60});
            TensorF32 dg_omg  = output.omega.view({dg_N, 36});
            TensorF32 dg_tht  = output.theta.view({dg_N, 36});
            TensorF32 dg_phi  = output.phi.view({dg_N, 18});
            TensorF32 dg_D    = input.D_onehot.view({dg_N, 60});
            TensorF32 dg_O    = input.O_onehot.view({dg_N, 36});
            TensorF32 dg_T    = input.T_onehot.view({dg_N, 36});
            TensorF32 dg_P    = input.P_onehot.view({dg_N, 18});
            TensorF32 dg_mask = input.pair_mask.view({dg_N});
            TensorF32* l_dist = wrap_value_as_leaf(dg_dist, {60, dg_N});
            TensorF32* l_omg  = wrap_value_as_leaf(dg_omg,  {36, dg_N});
            TensorF32* l_tht  = wrap_value_as_leaf(dg_tht,  {36, dg_N});
            TensorF32* l_phi  = wrap_value_as_leaf(dg_phi,  {18, dg_N});
            TensorF32* l_D    = wrap_value_as_leaf(dg_D,    {60, dg_N});
            TensorF32* l_O    = wrap_value_as_leaf(dg_O,    {36, dg_N});
            TensorF32* l_T    = wrap_value_as_leaf(dg_T,    {36, dg_N});
            TensorF32* l_P    = wrap_value_as_leaf(dg_P,    {18, dg_N});
            TensorF32* l_pm   = wrap_value_as_leaf(dg_mask, {dg_N});
            loss_distogram_node = loss(distogram_loss(l_dist, l_omg, l_tht, l_phi,
                                                       l_D, l_O, l_T, l_P, l_pm));
        }

        // 4) Masked MSA loss — 接入 masked_msa_loss(logits, true_msa, bert_mask)
        //    logits   : output.msa_logits (B=1, N, L, 23) → view (N, L, 23)
        //               wrap 图 dims={23, L, N} (dims[0]=类别最内), 匹配 masked_msa_loss 期望
        //    true_msa : input.true_msa   (B=1, N, L)      → view (N, L) → 图 {L, N}
        //    bert_mask: input.bert_mask  (B=1, N, L)      → view (N, L) → 图 {L, N}
        //    布局: masked_msa_loss 期望 logits[N_seq,N_res,23], true_msa[N_seq,N_res]
        //          (ggml: logits dims={23,L,N}, true_msa dims={L,N} → N_res=L, N_seq=N)
        TensorF32* loss_msa_node = loss(constant_scalar(0.0f));
        if (input.true_msa.numel() > 0 && output.msa_logits.numel() > 0) {
            int N_seq_msa = static_cast<int>(input.true_msa.shape().dims[1]);  // N_seq
            TensorF32 msa_logits_2d = output.msa_logits.view({N_seq_msa, L, 23});  // (N,L,23)
            TensorF32 msa_true_2d   = input.true_msa.view({N_seq_msa, L});         // (N,L)
            TensorF32 msa_mask_2d   = input.bert_mask.view({N_seq_msa, L});        // (N,L)
            TensorF32* logits_node = wrap_value_as_leaf(msa_logits_2d, {23, L, N_seq_msa});
            TensorF32* true_node   = wrap_value_as_leaf(msa_true_2d,   {L, N_seq_msa});
            TensorF32* mask_node   = wrap_value_as_leaf(msa_mask_2d,   {L, N_seq_msa});
            loss_msa_node = loss(masked_msa_loss(logits_node, true_node, mask_node));
        }

        // 5) Confidence (pLDDT) loss — 接入 plddt_loss(logits, lddt_onehot, ca_mask)
        //    logits    : output.lddt (B,L,50) → view (B*L,50) → 图 {50, B*L}
        //    ca_mask   : input.ca_mask(B,L)   → view (B*L)     → 图 {B*L}
        //    lddt_onehot: 用 output.coords(预测) vs input.true_coords(真实) 动态计算
        //                 (compute_lddt_ca + lddt_to_onehot), → (B*L,50) → 图 {50, B*L}
        TensorF32* loss_conf_node = loss(constant_scalar(0.0f));
        if (input.ca_mask.numel() > 0 && output.lddt.numel() > 0 && output.coords.numel() > 0) {
            int64_t pl_N = output.lddt.shape().dims[0] * output.lddt.shape().dims[1];  // B*L
            // 提取 CA 坐标 (B,L,3,3)→(B*L,3), 真值 CA (先转 CPU 以便 .data() 访问)
            TensorF32 pred_coords_cpu = output.coords.cpu();
            TensorF32 true_coords_cpu = input.true_coords.cpu();
            TensorF32 ca_mask_cpu     = input.ca_mask.cpu();
            std::vector<float> pred_ca(pl_N * 3, 0.0f), true_ca(pl_N * 3, 0.0f);
            std::vector<float> ca_m(pl_N, 0.0f);
            for (int64_t i = 0; i < pl_N; ++i) {
                // CA 原子索引=1
                pred_ca[i*3+0] = pred_coords_cpu.data()[i*9 + 1*3 + 0];
                pred_ca[i*3+1] = pred_coords_cpu.data()[i*9 + 1*3 + 1];
                pred_ca[i*3+2] = pred_coords_cpu.data()[i*9 + 1*3 + 2];
                true_ca[i*3+0] = true_coords_cpu.data()[i*9 + 1*3 + 0];
                true_ca[i*3+1] = true_coords_cpu.data()[i*9 + 1*3 + 1];
                true_ca[i*3+2] = true_coords_cpu.data()[i*9 + 1*3 + 2];
                ca_m[i] = ca_mask_cpu.data()[i];
            }
            const int N_BINS = 50;
            std::vector<float> lddt(pl_N, 0.0f), onehot(pl_N * N_BINS, 0.0f);
            compute_lddt_ca(pred_ca.data(), true_ca.data(), ca_m.data(),
                            static_cast<int>(pl_N), 15.0f, lddt.data());
            lddt_to_onehot(lddt.data(), static_cast<int>(pl_N), N_BINS, onehot.data());

            TensorF32 pl_logits = output.lddt.view({pl_N, N_BINS});
            TensorF32 pl_onehot(Shape({pl_N, N_BINS}), onehot.data(), Device::CPU, false);
            TensorF32 pl_camask = input.ca_mask.view({pl_N});
            TensorF32* l_lddt  = wrap_value_as_leaf(pl_logits, {N_BINS, pl_N});
            TensorF32* l_oh    = wrap_value_as_leaf(pl_onehot, {N_BINS, pl_N});
            TensorF32* l_cam   = wrap_value_as_leaf(pl_camask, {pl_N});
            loss_conf_node = loss(plddt_loss(l_lddt, l_oh, l_cam));
        }

        // --- total_loss: 0.5*FAPE + 0.5*Chi + 0.3*Distogram + 2.0*MSA + 0.01*Conf ---
        TensorF32* total_node = loss(total_loss(
            loss_fape_node, loss_chi_node, loss_distogram_node, loss_msa_node, loss_conf_node));

        // ============================================================
        // 构建 backward 计算图并执行 (全局裁剪)
        // ============================================================
        RFAAContext* ctx = &context();
        ComputeGraph* cgraph = ComputeGraph::new_graph(ctx);
        cgraph->build_forward_expand(total_node);
        cgraph->build_backward_expand(ctx, nullptr);

        // 若已接入 backend, 可在此调用 graph_compute + clip_grad_norm + optimizer
        // backend->graph_compute(cgraph);
        // float grad_norm = clip_grad_norm(cgraph, 0.1f);

        // 读取 total loss 数值 (FAPE 项已正确构造, 其余项为 0)
        // 注: 需要 backend->graph_compute 后 total_node->data() 才有值
        float batch_loss = 0.0f;
        if (total_node->data() != nullptr && total_node->numel() == 1) {
            batch_loss = total_node->data()[0];
        }

        epoch_loss += batch_loss;
        
        // ===== 计时代码: 汇总 epoch 耗时 =====
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            epoch_end - epoch_start).count();
        double epoch_sec = static_cast<double>(epoch_ms) / 1000.0;

        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
                  << " completed in " << epoch_ms << " ms"
                  << " (forward " << fwd_ms << " ms)"
                  << ", loss: " << batch_loss << std::endl;

        // ============================================================
        // Checkpoint 保存: 每隔 ckpt_interval 个 epoch, 以及最后一个 epoch
        // ============================================================
        bool is_ckpt_epoch = (epoch + 1) % ckpt_interval == 0 || (epoch + 1) == num_epochs;
        if (is_ckpt_epoch) {
            // 确保输出目录存在
            std::error_code ec;
            std::filesystem::create_directories(ckpt_dir, ec);

            std::string path = ckpt_dir + "/ckpt_epoch" + std::to_string(epoch + 1) + ".gguf";
            std::cout << "  [Checkpoint] Saving at epoch " << epoch + 1
                      << "/" << num_epochs
                      << " (loss=" << batch_loss
                      << ", epoch_time=" << epoch_ms << " ms) ... ";

            auto ckpt_start = std::chrono::high_resolution_clock::now();
            try {
                // before save: use backend->synchronize force cpu to wait all gpu work finish
                // cudaStreamSynchronize or cudaDeviceSynchronize
                save_checkpoint(model, path, epoch, num_epochs, batch_loss, epoch_sec);
                auto ckpt_end = std::chrono::high_resolution_clock::now();
                auto ckpt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   ckpt_end - ckpt_start).count();

                // 打印保存后的文件大小
                std::uintmax_t fsize = 0;
                if (!ec) fsize = std::filesystem::file_size(path);
                std::cout << "OK -> " << path
                          << " (" << fsize << " B, save took " << ckpt_ms << " ms)"
                          << std::endl;

                // 开发功能: 统计所有权重的内存占用 (在 save_checkpoint 内用同款 CPU 拷贝逻辑)
                estimate_params_memory(model);
            } catch (const std::exception& e) {
                std::cerr << "  [Checkpoint] FAILED: " << e.what() << std::endl;
            }
        }
    }
    
    // 6. 保存最终模型 (仍保留原有 .bin 保存, 以便兼容)
    model.save_weights("rfaa_weights.bin");
    
    // 7. 导出 ONNX
    ONNXExportConfig onnx_config;
    onnx_config.output_path = "rfaa_model.onnx";
    onnx_config.opset_version = 17;
    
    // 设置动态维度
    onnx_config.dynamic_dims = {
        {"batch", 1, 4},
        {"seq_len", 32, 2048},
        {"n_seq", 1, 512},
        {"n_templ", 1, 4}
    };
    
    /* ONNXExporter exporter;
    exporter.export_model(model, onnx_config);
    
    if (exporter.validate(onnx_config.output_path)) {
        std::cout << "ONNX model exported and validated successfully" << std::endl;
        std::cout << exporter.get_model_info(onnx_config.output_path) << std::endl;
    }
     */
    // 8. 清理
    py.finalize();
    
    return 0;
}

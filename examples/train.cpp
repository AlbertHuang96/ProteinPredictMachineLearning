#include "rfaa/Model.h"
#include "rfaa/DataLoader.h"
#include "rfaa/ComputeGraph.h"
#include "rfaa/Context.h"
#include "rfaa/PythonBridge.h"
#include "rfaa/ONNXExporter.h"
#include "rfaa/GradientClipper.h"
#include <iostream>
#include <chrono>
#include <cstring>

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
    if (argc > 1) {
        std::string weights_path = argv[1];
        ProteinTools::load_torch_weights(model, weights_path);
    }
    
    // ============================================================
    // 数据加载: 通过 RFAADataLoader 从 A3M/HHR + CSV mapping 加载
    // CSV 提供真实坐标 (true_coords), 作为 FAPE 的 ground truth
    // ============================================================
    // 命令行参数: train <a3m> <hhr> <sequence> <csv_mapping>
    // 例如: train 1a00.a3m 1a00.hhr "MVLSPADKTNVKAAWGKVG..." data/1a00_P69905_mapping.csv
    std::string a3m_path  = (argc > 2) ? argv[2] : "query.a3m";
    std::string hhr_path  = (argc > 3) ? argv[3] : "query.hhr";
    std::string sequence  = (argc > 4) ? argv[4] : "";
    std::string csv_path  = (argc > 5) ? argv[5] : "data/1a00_P69905_mapping.csv";

    if (sequence.empty()) {
        // 无显式序列时从 CSV 行数推断长度 (demo 场景)
        std::cerr << "No sequence provided; using CSV-derived length. "
                     "Pass <a3m> <hhr> <sequence> <csv> as argv[2..5]." << std::endl;
    }

    RFAADataLoader loader("", "", 512, 4, 2048);
    ModelInput input = loader.load_from_files(a3m_path, hhr_path, sequence, csv_path);
    int L = 0;
    if (!sequence.empty()) {
        L = static_cast<int>(sequence.length());
    } else if (input.true_coords.numel() > 0) {
        L = static_cast<int>(input.true_coords.shape().dims[1]);
    }

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
    input.msa_latent = input.msa_latent.to(Device::CUDA);
    input.seq_tokens = input.seq_tokens.to(Device::CUDA);
    input.coords     = input.coords.to(Device::CUDA);
    input.true_coords= input.true_coords.to(Device::CUDA);
    
    // 5. 训练循环
    const int num_epochs = 10;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        float epoch_loss = 0.0f;
        
        // 前向传播
        auto output = model.forward(input);
        
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

        // 2) Chi (扭转角) loss — 当前无真实扭转角监督, 用标量 0 占位
        //    接入真实 gt 时替换为: supervised_chi_loss(unnormed, gt, chi_mask, seq_mask)
        TensorF32* loss_chi_node       = loss(constant_scalar(0.0f));

        // 3) Distogram loss — 当前无 distogram 标签, 用标量 0 占位
        TensorF32* loss_distogram_node = loss(constant_scalar(0.0f));

        // 4) Masked MSA loss — 当前无 MSA 掩码监督, 用标量 0 占位
        TensorF32* loss_msa_node       = loss(constant_scalar(0.0f));

        // 5) Confidence (pLDDT) loss — 当前无 LDDT 标签, 用标量 0 占位
        TensorF32* loss_conf_node      = loss(constant_scalar(0.0f));

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
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
        
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs 
                  << " completed in " << duration.count() << "s"
                  << ", loss: " << batch_loss << std::endl;
    }
    
    // 6. 保存模型
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

#include "rfaa/Model.h"
#include "rfaa/PythonBridge.h"
#include "rfaa/ONNXExporter.h"
#include "rfaa/GradientClipper.h"
#include <iostream>
#include <chrono>

using namespace rfaa;

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
    
    // 5. 训练循环
    const int num_epochs = 10;
    const int batch_size = 2;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        float epoch_loss = 0.0f;
        int num_batches = 100;  // 示例
        
        for (int batch = 0; batch < num_batches; ++batch) {
            // 通过 Python 获取数据
            // ModelInput input = load_batch_from_python(batch);
            
            // 构造示例输入
            int B = batch_size, N = 64, L = 128, T = 4;
            ModelInput input;
            input.msa_latent = zeros<float>({B, N, L, MSA_LATENT_DIM}, Device::CUDA);
            input.seq_tokens = zeros<float>({B, L}, Device::CUDA);
            input.t1d = zeros<float>({B, T, L, D_T1D}, Device::CUDA);
            input.coords = zeros<float>({B, L, 3, 3}, Device::CUDA);
            
            // 前向传播
            auto output = model.forward(input);
            
            // 计算损失 (实际应通过 Python 或 C++ 实现)
            // loss = compute_loss(output, targets)
            //
            // ============================================================
            // 示例：逐项损失 + 梯度裁剪 (Gradient Clipping)
            // ============================================================
            //
            // 1. 分别计算各 loss 并标记为 LOSS 节点:
            //    auto loss_fape_node      = loss(fape_loss(...));
            //    auto loss_chi_node       = loss(supervised_chi_loss(...));
            //    auto loss_distogram_node = loss(distogram_loss(...));
            //    auto loss_msa_node       = loss(masked_msa_loss(...));
            //    auto loss_conf_node      = loss(plddt_loss(...));
            //
            // 2. Build backward graph:
            //    cgraph->build_backward_expand(ctx, nullptr);
            //
            // 3. Per-loss + global clipping config:
            //    PerLossClipConfig clip_cfg;
            //    clip_cfg.fape_max_norm      = 10.0f;  // Å-scale, 阈值较大
            //    clip_cfg.chi_max_norm       = 1.0f;   // angular, 天然 bounded
            //    clip_cfg.distogram_max_norm = 1.0f;   // CE loss
            //    clip_cfg.msa_max_norm       = 0.5f;   // CE loss
            //    clip_cfg.conf_max_norm      = 0.1f;   // 置信度 loss
            //    clip_cfg.global_max_norm    = 0.1f;   // AF2 standard
            //
            // 4. 执行逐项 backward + clip + accumulate:
            //    std::vector<LossGradientInfo> info;
            //    float total_norm = apply_per_loss_clip(
            //        cgraph, backend.get(),
            //        loss_fape_node, loss_chi_node, loss_distogram_node,
            //        loss_msa_node, loss_conf_node,
            //        clip_cfg, &info);
            //
            // 5. 日志输出（可选）:
            //    for (auto& inf : info) {
            //        std::cout << "[" << inf.loss_name << "] raw_norm=" << inf.raw_norm
            //                  << " clipped_norm=" << inf.clipped_norm
            //                  << (inf.was_clipped ? " [CLIPPED]" : "") << std::endl;
            //    }
            //    std::cout << "Total grad norm after clipping: " << total_norm << std::endl;
            //
            // 6. Optimizer step:
            //    optimizer->step(cgraph);
            //
            // 或者，仅使用全局裁剪（单次 backward）:
            //    cgraph->build_backward_expand(ctx, nullptr);
            //    backend->graph_compute(cgraph);
            //    float grad_norm = clip_grad_norm(cgraph, 0.1f);
            //
            // ============================================================
            
            // 反向传播 (需要实现 autograd 或调用 Python)
            // loss.backward()
            
            epoch_loss += 0.0f;  // 占位
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
        
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs 
                  << " completed in " << duration.count() << "s"
                  << ", avg loss: " << epoch_loss / num_batches << std::endl;
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

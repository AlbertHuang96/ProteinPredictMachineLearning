#include "rfaa/Model.h"
#include "rfaa/ONNXExporter.h"
#include <iostream>

using namespace rfaa;

/* int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <weights_path> <output_onnx_path>" << std::endl;
        return 1;
    }
    
    std::string weights_path = argv[1];
    std::string output_path = argv[2];
    
    // 1. 创建模型
    RFAAConfig config;
    RFAAModel model(config);
    
    // 2. 加载权重
    model.load_weights(weights_path);
    model.eval();
    
    std::cout << "Model loaded from: " << weights_path << std::endl;
    
    // 3. 配置 ONNX 导出
    ONNXExportConfig onnx_config;
    onnx_config.output_path = output_path;
    onnx_config.opset_version = 17;
    
    // 设置动态维度 (batch, seq_len, n_seq, n_templ)
    onnx_config.dynamic_dims = {
        {"batch_size", 1, 8},
        {"seq_len", 16, 2048},
        {"n_seq", 1, 512},
        {"n_templ", 0, 4}
    };
    
    // 大模型使用外部数据
    if (argc > 3 && std::string(argv[3]) == "--external") {
        onnx_config.use_external_data = true;
        onnx_config.external_data_path = output_path + ".data";
    }
    
    // 4. 导出
    ONNXExporter exporter;
    
    try {
        exporter.export_model(model, onnx_config);
        
        // 5. 验证
        if (exporter.validate(output_path)) {
            std::cout << "ONNX export successful!" << std::endl;
            std::cout << exporter.get_model_info(output_path) << std::endl;
            
            // 6. 测试推理
            ONNXRuntime runtime(output_path);
            std::cout << "ONNX Runtime test passed." << std::endl;
        } else {
            std::cerr << "ONNX validation failed!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Export failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} */

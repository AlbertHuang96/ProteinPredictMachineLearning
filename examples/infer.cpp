#include "ppml/Model.h"
#include "ppml/ONNXExporter.h"
#include <iostream>

using namespace ppml;

int main(int argc, char* argv[]) {
    // 1. 创建模型
    PPMLConfig config;
    PPMLModel model(config);
    
    // 2. 加载权重
    if (argc > 1) {
        model.load_weights(argv[1]);
    }
    
    // 3. 推理模式
    model.eval();
    model.to(Device::CUDA);
    
    // 4. 构造输入 (示例)
    int B = 1, N = 256, L = 200, T = 4;
    
    ModelInput input;
    input.msa_latent = zeros<float>({B, N, L, MSA_LATENT_DIM}, Device::CUDA);
    input.seq_tokens = zeros<float>({B, L}, Device::CUDA);
    input.t1d = zeros<float>({B, T, L, D_T1D}, Device::CUDA);
    input.coords = zeros<float>({B, L, 3, 3}, Device::CUDA);
    
    // 填充实际数据...
    
    // 5. 推理
    std::cout << "Running inference..." << std::endl;
    auto output = model.forward(input);
    
    // 6. 输出结果
    std::cout << "Output shapes:" << std::endl;
    std::cout << "  MSA: " << output.msa.to_string() << std::endl;
    std::cout << "  Pair: " << output.pair.to_string() << std::endl;
    std::cout << "  State: " << output.state.to_string() << std::endl;
    std::cout << "  Coords: " << output.coords.to_string() << std::endl;
    std::cout << "  Alpha: " << output.alpha.to_string() << std::endl;
    
    // 7. 保存预测结果
    // output.coords.cpu().save("predicted_coords.npy");
    // output.alpha.cpu().save("predicted_torsion.npy");
    
    // 8. 使用 ONNX Runtime 推理 (可选)
    if (argc > 2) {
        std::string onnx_path = argv[2];
        /* ONNXRuntime runtime(onnx_path);
        
        std::cout << "ONNX Runtime inputs:" << std::endl;
        for (const auto& name : runtime.get_input_names()) {
            std::cout << "  " << name << std::endl;
        } */
        
        // runtime.run({input.msa_latent.cpu(), ...});
    }
    
    return 0;
}

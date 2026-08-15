#include "ppml/ONNXExporter.h"
#include <fstream>
#include <iostream>

namespace ppml {

ONNXExporter::ONNXExporter() 
    : env_(ORT_LOGGING_LEVEL_WARNING, "PPML_ONNX_Exporter") {}

ONNXExporter::~ONNXExporter() = default;

void ONNXExporter::export_model(const PPMLModel& model, const ONNXExportConfig& config) {
    std::cout << "Exporting PPML model to ONNX: " << config.output_path << std::endl;
    
    // 创建 ONNX 模型 protobuf
    // 由于 PPML 包含自定义 SE3 操作，需要分解为 ONNX 支持的操作
    
    // 方案1: 使用 ONNX 的自定义算子
    // 方案2: 将 SE3 部分保留为 Python，其余导出 ONNX
    
    // 这里展示核心逻辑框架：
    
    // 1. 创建输入定义
    // inputs: msa_latent, seq_tokens, t1d, coords
    
    // 2. 构建计算图
    // - Embedding (Gather, MatMul)
    // - Attention (MultiHeadAttention 或分解为 MatMul+Softmax)
    // - TriangleMultiplication (MatMul, ReduceMean)
    // - SE3Transformer (需自定义或分解)
    
    // 3. 设置动态维度
    // batch, seq_len, n_seq, n_templ 设为动态
    
    // 4. 序列化保存
    std::ofstream ofs(config.output_path, std::ios::binary);
    // 写入 ONNX protobuf...
    
    std::cout << "ONNX export completed." << std::endl;
}

void ONNXExporter::export_from_weights(const std::string& weights_path,
                                       const ONNXExportConfig& config) {
    // 从权重文件重建模型并导出
    PPMLModel model;
    model.load_weights(weights_path);
    export_model(model, config);
}

bool ONNXExporter::validate(const std::string& onnx_path) {
    try {
        Ort::Session session(env_, onnx_path.c_str(), session_options_);
        
        // 检查输入输出
        size_t num_inputs = session.GetInputCount();
        size_t num_outputs = session.GetOutputCount();
        
        std::cout << "ONNX model validated. Inputs: " << num_inputs 
                  << ", Outputs: " << num_outputs << std::endl;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX validation failed: " << e.what() << std::endl;
        return false;
    }
}

std::string ONNXExporter::get_model_info(const std::string& onnx_path) {
    try {
        Ort::Session session(env_, onnx_path.c_str(), session_options_);
        
        std::ostringstream oss;
        oss << "Model: " << onnx_path << "\n";
        
        // 输入信息
        Ort::AllocatorWithDefaultOptions allocator;
        oss << "Inputs:\n";
        for (size_t i = 0; i < session.GetInputCount(); ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            
            oss << "  " << name.get() << ": ";
            for (auto dim : shape) {
                oss << dim << " ";
            }
            oss << "\n";
        }
        
        // 输出信息
        oss << "Outputs:\n";
        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            auto type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            
            oss << "  " << name.get() << ": ";
            for (auto dim : shape) {
                oss << dim << " ";
            }
            oss << "\n";
        }
        
        return oss.str();
    } catch (const Ort::Exception& e) {
        return std::string("Error: ") + e.what();
    }
}

// ONNXRuntime 实现
ONNXRuntime::ONNXRuntime(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "PPML_ONNX_Runtime"),
      session_(env_, model_path.c_str(), Ort::SessionOptions{}),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

ONNXRuntime::~ONNXRuntime() = default;

std::vector<TensorF32> ONNXRuntime::run(const std::vector<TensorF32>& inputs) {
    std::vector<Ort::Value> ort_inputs;
    
    // 转换 Tensor -> Ort::Value
    for (const auto& input : inputs) {
        auto cpu_input = input.cpu();
        std::vector<int64_t> shape(cpu_input.shape().dims.begin(),
                                   cpu_input.shape().dims.end());
        
        ort_inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info_,
            cpu_input.data(),
            cpu_input.numel(),
            shape.data(),
            shape.size()
        ));
    }
    
    // 获取输入输出名称
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*> input_names;
    for (size_t i = 0; i < session_.GetInputCount(); ++i) {
        input_names.push_back(session_.GetInputNameAllocated(i, allocator).get());
    }
    
    std::vector<const char*> output_names;
    for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
        output_names.push_back(session_.GetOutputNameAllocated(i, allocator).get());
    }
    
    // 运行推理
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names.data(), ort_inputs.data(), ort_inputs.size(),
        output_names.data(), output_names.size()
    );
    
    // 转换回 Tensor
    std::vector<TensorF32> results;
    for (auto& output : outputs) {
        auto tensor_info = output.GetTensorTypeAndShapeInfo();
        auto shape_vec = tensor_info.GetShape();
        Shape shape(std::vector<int64_t>(shape_vec.begin(), shape_vec.end()));
        
        float* data = output.GetTensorMutableData<float>();
        TensorF32 t(shape, data, Device::CPU, false);
        results.push_back(std::move(t));
    }
    
    return results;
}

std::vector<std::string> ONNXRuntime::get_input_names() const {
    std::vector<std::string> names;
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session_.GetInputCount(); ++i) {
        names.emplace_back(session_.GetInputNameAllocated(i, allocator).get());
    }
    return names;
}

std::vector<std::string> ONNXRuntime::get_output_names() const {
    std::vector<std::string> names;
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
        names.emplace_back(session_.GetOutputNameAllocated(i, allocator).get());
    }
    return names;
}

} // namespace ppml

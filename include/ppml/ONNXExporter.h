#pragma once

#include "Model.h"
//#include <onnxruntime_cxx_api.h>
#include <string>

namespace ppml {

// ONNX 导出配置
struct ONNXExportConfig {
    std::string output_path;           // 输出文件路径
    int opset_version = 17;            // ONNX opset 版本
    bool use_external_data = false;    // 大模型使用外部存储
    std::string external_data_path;    // 外部数据路径
    
    // 动态维度配置
    struct DynamicDim {
        std::string name;
        int min_val;
        int max_val;
    };
    std::vector<DynamicDim> dynamic_dims;
};

// ONNX 模型导出器
/* class ONNXExporter {
public:
    ONNXExporter();
    ~ONNXExporter();
    
    // 从 PPML 模型导出 ONNX
    // 方式1: 直接导出 (通过 tracing)
    void export_model(const PPMLModel& model, const ONNXExportConfig& config);
    
    // 方式2: 从训练好的权重文件导出
    void export_from_weights(const std::string& weights_path, 
                             const ONNXExportConfig& config);
    
    // 验证导出的 ONNX 模型
    bool validate(const std::string& onnx_path);
    
    // 获取 ONNX 模型信息
    std::string get_model_info(const std::string& onnx_path);
    
private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ONNX Runtime 推理封装 (可选)
class ONNXRuntime {
public:
    explicit ONNXRuntime(const std::string& model_path);
    ~ONNXRuntime();
    
    // 运行推理
    std::vector<TensorF32> run(const std::vector<TensorF32>& inputs);
    
    // 获取输入/输出信息
    std::vector<std::string> get_input_names() const;
    std::vector<std::string> get_output_names() const;
    
private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
    
    struct Impl;
    std::unique_ptr<Impl> impl_;
}; */

} // namespace ppml

#pragma once

#include "Model.h"
#include <Python.h>
#include <string>
#include <functional>

namespace ppml {

// Python 桥接：C++ 调用 Python 库
// 用途：数据预处理、MSA 搜索、模板检索、训练可视化等

class PythonBridge {
public:
    // 单例模式
    static PythonBridge& instance();
    
    // 初始化和清理
    bool initialize(const std::string& python_home = "");
    void finalize();
    bool is_initialized() const;
    
    // 执行 Python 代码
    PyObject* run_string(const std::string& code);
    PyObject* run_file(const std::string& path);
    
    // 调用 Python 函数
    PyObject* call_function(const std::string& module_name,
                            const std::string& func_name,
                            PyObject* args = nullptr);
    
    // 模块管理
    bool import_module(const std::string& module_name);
    PyObject* get_module(const std::string& module_name);
    
    // GIL 管理 (多线程安全)
    class GILGuard {
    public:
        GILGuard();
        ~GILGuard();
    private:
        PyGILState_STATE state_;
    };
    
    // 异常处理
    std::string get_last_error() const;
    void clear_error();
    
private:
    PythonBridge() = default;
    ~PythonBridge();
    
    PythonBridge(const PythonBridge&) = delete;
    PythonBridge& operator=(const PythonBridge&) = delete;
    
    bool initialized_ = false;
    std::unordered_map<std::string, PyObject*> modules_;
};

// 高级封装：常用蛋白质工具调用
class ProteinTools {
public:
    // HHblits MSA 搜索
    static TensorF32 run_hhblits(const std::string& sequence, 
                                  const std::string& database_path);
    
    // 模板搜索 (HHsearch)
    static TensorF32 search_templates(const std::string& sequence,
                                       const std::string& pdb_db);
    
    // 二级结构预测 (可选，PPML 本身不需要)
    static TensorF32 predict_ss(const std::string& sequence);
    
    // 使用 PyTorch 加载预训练权重
    static bool load_torch_weights(PPMLModel& model, const std::string& pt_path);
    
    // 使用 PyTorch 训练循环
    static void train_epoch(PPMLModel& model, 
                            const std::vector<ModelInput>& batch,
                            float learning_rate);
    
    // 数据增强 (Python 端实现)
    static ModelInput augment_data(const ModelInput& input);
};

// PyBind11 暴露接口 (供 Python 调用 C++)
#ifdef PPML_BUILD_PYTHON_MODULE
void init_ppml_module(pybind11::module& m);
#endif

} // namespace ppml

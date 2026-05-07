#include "rfaa/PythonBridge.h"
#include <iostream>

namespace rfaa {

PythonBridge& PythonBridge::instance() {
    static PythonBridge instance;
    return instance;
}

PythonBridge::~PythonBridge() {
    if (initialized_) {
        finalize();
    }
}

bool PythonBridge::initialize(const std::string& python_home) {
    if (initialized_) return true;
    
    if (!python_home.empty()) {
        Py_SetPythonHome(Py_DecodeLocale(python_home.c_str(), nullptr));
    }
    
    Py_Initialize();
    initialized_ = Py_IsInitialized();
    
    if (initialized_) {
        // 初始化 numpy 支持
        //import_array1(false);
    }
    
    return initialized_;
}

void PythonBridge::finalize() {
    if (initialized_) {
        for (auto& [name, mod] : modules_) {
            Py_XDECREF(mod);
        }
        modules_.clear();
        Py_Finalize();
        initialized_ = false;
    }
}

bool PythonBridge::is_initialized() const {
    return initialized_;
}

PyObject* PythonBridge::run_string(const std::string& code) {
    GILGuard guard;
    PyObject* result = PyRun_String(code.c_str(), Py_file_input, 
                                     PyDict_New(), PyDict_New());
    if (!result) {
        std::cerr << "Python error: " << get_last_error() << std::endl;
    }
    return result;
}

PyObject* PythonBridge::call_function(const std::string& module_name,
                                      const std::string& func_name,
                                      PyObject* args) {
    GILGuard guard;
    
    PyObject* module = get_module(module_name);
    if (!module) return nullptr;
    
    PyObject* func = PyObject_GetAttrString(module, func_name.c_str());
    if (!func || !PyCallable_Check(func)) {
        Py_XDECREF(func);
        return nullptr;
    }
    
    PyObject* result = PyObject_CallObject(func, args);
    Py_DECREF(func);
    
    if (!result) {
        std::cerr << "Function call error: " << get_last_error() << std::endl;
    }
    
    return result;
}

bool PythonBridge::import_module(const std::string& module_name) {
    GILGuard guard;
    
    PyObject* module = PyImport_ImportModule(module_name.c_str());
    if (!module) {
        std::cerr << "Failed to import module: " << module_name << std::endl;
        std::cerr << get_last_error() << std::endl;
        return false;
    }
    
    modules_[module_name] = module;
    return true;
}

PyObject* PythonBridge::get_module(const std::string& module_name) {
    auto it = modules_.find(module_name);
    if (it != modules_.end()) {
        return it->second;
    }
    
    if (import_module(module_name)) {
        return modules_[module_name];
    }
    return nullptr;
}

std::string PythonBridge::get_last_error() const {
    if (!PyErr_Occurred()) return "";
    
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    
    std::string error_msg;
    if (value) {
        PyObject* str = PyObject_Str(value);
        if (str) {
            error_msg = PyUnicode_AsUTF8(str);
            Py_DECREF(str);
        }
    }
    
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    
    return error_msg;
}

void PythonBridge::clear_error() {
    PyErr_Clear();
}

// GILGuard 实现
PythonBridge::GILGuard::GILGuard() {
    state_ = PyGILState_Ensure();
}

PythonBridge::GILGuard::~GILGuard() {
    PyGILState_Release(state_);
}

// ========== ProteinTools 实现 ==========

TensorF32 ProteinTools::run_hhblits(const std::string& sequence,
                                     const std::string& database_path) {
    auto& py = PythonBridge::instance();
    
    // 调用 Python 的 hhblits 封装
    PyObject* args = PyTuple_Pack(2,
        PyUnicode_FromString(sequence.c_str()),
        PyUnicode_FromString(database_path.c_str())
    );
    
    PyObject* result = py.call_function("rfaa_tools", "run_hhblits", args);
    Py_DECREF(args);
    
    // 转换 Python numpy array -> C++ Tensor
    // ...
    
    return TensorF32();
}

bool ProteinTools::load_torch_weights(RFAAModel& model, const std::string& pt_path) {
    auto& py = PythonBridge::instance();
    
    // 使用 torch.load 读取权重，然后转换到 C++
    std::string code = R"(
import torch
import numpy as np

state_dict = torch.load('" + pt_path + R"(', map_location='cpu')

# 转换并保存为 numpy 格式供 C++ 读取
for name, param in state_dict.items():
    np.save(f'weights/{name}.npy', param.cpu().numpy())
)";
    
    py.run_string(code);
    
    // C++ 端加载 numpy 权重文件
    model.load_weights("weights/");
    
    return true;
}

void ProteinTools::train_epoch(RFAAModel& model,
                               const std::vector<ModelInput>& batch,
                               float learning_rate) {
    auto& py = PythonBridge::instance();
    
    // 调用 PyTorch 优化器或自定义训练逻辑
    // 这里可以混合 C++ 前向 + Python 反向
    
    std::string code = R"(
import torch

# 定义损失函数
def compute_loss(outputs, targets):
    loss = torch.nn.CrossEntropyLoss()
    return loss(outputs, targets)
)";
    
    py.run_string(code);
}

} // namespace rfaa

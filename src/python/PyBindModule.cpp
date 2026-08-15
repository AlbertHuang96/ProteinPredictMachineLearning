#include "ppml/PythonBridge.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>

namespace py = pybind11;
using namespace ppml;

// Tensor <-> numpy array 转换
template<typename T>
py::array_t<T> tensor_to_numpy(const Tensor<T>& t) {
    auto cpu_t = t.cpu();
    std::vector<size_t> shape(t.shape().dims.begin(), t.shape().dims.end());
    std::vector<size_t> strides(t.shape().ndim());
    
    size_t stride = sizeof(T);
    for (int i = t.shape().ndim() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= t.shape().dims[i];
    }
    
    return py::array_t<T>(shape, strides, cpu_t.data());
}

template<typename T>
Tensor<T> numpy_to_tensor(py::array_t<T> arr, Device device = Device::CPU) {
    py::buffer_info info = arr.request();
    std::vector<int64_t> shape(info.shape.begin(), info.shape.end());
    
    Tensor<T> t(Shape(shape), device);
    if (device == Device::CPU) {
        std::memcpy(t.data(), info.ptr, t.nbytes());
    } else {
        // CUDA 拷贝
        cudaMemcpy(t.data(), info.ptr, t.nbytes(), cudaMemcpyHostToDevice);
    }
    return t;
}

// 暴露 ModelInput/ModelOutput
void bind_types(py::module& m) {
    py::class_<Shape>(m, "Shape")
        .def(py::init<std::vector<int64_t>>())
        .def("numel", &Shape::numel)
        .def("ndim", &Shape::ndim);
    
    py::class_<TensorF32>(m, "Tensor")
        .def("shape", &TensorF32::shape)
        .def("to_numpy", [](const TensorF32& t) { return tensor_to_numpy(t); })
        .def("cpu", &TensorF32::cpu)
        .def("cuda", &TensorF32::cuda);
    
        //pybind11/pybind11.h:2209:68: error: use of deleted function 
        /* ‘ppml::Tensor<T>& ppml::Tensor<T>::operator=(const ppml::Tensor<T>&) [with T = float]’
              return cpp_function([pm](T &c, const D &value) { c.*pm = value; }, is_method(hdl)); */
    // Tensor default copy constructor was deleted, so we need to define it for pybind11 ?
    /* py::class_<ModelInput>(m, "ModelInput")
        .def(py::init<>())
        .def_readwrite("msa_latent", &ModelInput::msa_latent)
        .def_readwrite("seq_tokens", &ModelInput::seq_tokens)
        .def_readwrite("t1d", &ModelInput::t1d)
        .def_readwrite("coords", &ModelInput::coords); */
    
    // Tensor default copy constructor was deleted, so we need to define it for pybind11 ?
    /* py::class_<ModelOutput>(m, "ModelOutput")
        .def(py::init<>())
        .def_readwrite("msa", &ModelOutput::msa)
        .def_readwrite("pair", &ModelOutput::pair)
        .def_readwrite("state", &ModelOutput::state)
        .def_readwrite("coords", &ModelOutput::coords)
        .def_readwrite("alpha", &ModelOutput::alpha)
        .def_readwrite("lddt", &ModelOutput::lddt)
        .def_readwrite("distogram", &ModelOutput::distogram)
        .def_readwrite("pae", &ModelOutput::pae); */
}

// 暴露模型
void bind_model(py::module& m) {
    py::class_<PPMLConfig>(m, "PPMLConfig")
        .def(py::init<>())
        .def_readwrite("d_msa", &PPMLConfig::d_msa)
        .def_readwrite("d_pair", &PPMLConfig::d_pair)
        .def_readwrite("d_state", &PPMLConfig::d_state)
        .def_readwrite("n_extra_blocks", &PPMLConfig::n_extra_blocks)
        .def_readwrite("n_main_blocks", &PPMLConfig::n_main_blocks)
        .def_readwrite("n_refine_blocks", &PPMLConfig::n_refine_blocks);
    
    py::class_<PPMLModel>(m, "PPMLModel")
        .def(py::init<const PPMLConfig&>(), py::arg("config") = PPMLConfig{})
        .def("forward", &PPMLModel::forward)
        .def("load_weights", &PPMLModel::load_weights)
        .def("save_weights", &PPMLModel::save_weights)
        .def("to", &PPMLModel::to)
        .def("train", &PPMLModel::train)
        .def("eval", &PPMLModel::eval);
}

// 暴露 ONNX 导出
/* void bind_onnx(py::module& m) {
    py::class_<ONNXExportConfig>(m, "ONNXExportConfig")
        .def(py::init<>())
        .def_readwrite("output_path", &ONNXExportConfig::output_path)
        .def_readwrite("opset_version", &ONNXExportConfig::opset_version);
    
    py::class_<ONNXExporter>(m, "ONNXExporter")
        .def(py::init<>())
        .def("export_model", &ONNXExporter::export_model)
        .def("validate", &ONNXExporter::validate);
} */

// 模块初始化
PYBIND11_MODULE(pyppml, m) {
    m.doc() = "PPML C++ Protein Structure Prediction Framework";
    
    bind_types(m);
    bind_model(m);
    //bind_onnx(m);
    
    // 工具函数
    m.def("numpy_to_tensor", [](py::array_t<float> arr) {
        return numpy_to_tensor(arr, Device::CPU);
    });
    
    m.def("version", []() { return "1.0.0"; });
}

#pragma once

#include "Core.h"
#include <cstring>
#include <cuda_fp16.h>

namespace rfaa {

// 张量接口：CPU/CUDA 统一抽象
template<typename T = float>
class Tensor {
public:
    // 构造/析构
    Tensor() = default;
    Tensor(const Shape& shape, Device device = Device::CPU);
    Tensor(const Shape& shape, T* data, Device device = Device::CPU, bool own = false);
    ~Tensor();
    
    // 禁止拷贝，允许移动
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // 基础属性
    const Shape& shape() const { return shape_; }
    Device device() const { return device_; }
    DType dtype() const;
    int64_t numel() const { return shape_.numel(); }
    size_t nbytes() const { return numel() * sizeof(T); }
    
    // 数据访问
    T* data() { return data_; }
    const T* data() const { return data_; }
    T* data(Device target);  // 获取指定设备的数据（自动拷贝）
    
    // 设备转移
    Tensor to(Device device) const;
    Tensor cpu() const { return to(Device::CPU); }
    Tensor cuda() const { return to(Device::CUDA); }
    
    // 索引访问 (仅 CPU)
    T& operator()(std::initializer_list<int64_t> indices);
    T operator()(std::initializer_list<int64_t> indices) const;
    
    // 视图操作
    Tensor view(const Shape& new_shape) const;
    Tensor slice(int dim, int64_t start, int64_t end) const;
    Tensor select(int dim, int64_t index) const;
    
    // 维度操作
    Tensor unsqueeze(int dim) const;  // 在指定维度插入大小为1的维度
    Tensor permute(const std::vector<int>& dims) const; // 维度重排
    
    // 内存管理
    void zero_();
    void copy_from(const Tensor& other);
    
    // 打印调试用
    std::string to_string() const;
    
private:
    Shape shape_;
    T* data_ = nullptr;
    Device device_ = Device::CPU;
    bool own_data_ = true;
    
    void allocate();
    void deallocate();
};

// 特化常用类型
using TensorF32 = Tensor<float>;
using TensorF16 = Tensor<half>;

// 工厂函数
template<typename T = float>
Tensor<T> zeros(const Shape& shape, Device device = Device::CPU) {
    Tensor<T> t(shape, device);
    t.zero_();
    return t;
}

template<typename T = float>
Tensor<T> from_numpy(T* data, const Shape& shape) {
    return Tensor<T>(shape, data, Device::CPU, false);
}

} // namespace rfaa

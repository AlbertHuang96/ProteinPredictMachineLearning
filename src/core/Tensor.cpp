#include "rfaa/Tensor.h"
#include <cuda_runtime.h>
#include <cstring>
#include <sstream>

namespace rfaa {

// TODO : memory pool
// CPU 内存分配
static void* cpu_alloc(size_t size) {
    void* ptr = nullptr;
    ptr = malloc(size);
    if (!ptr) throw RFAAError("CPU memory allocation failed");
    return ptr;
}

static void cpu_free(void* ptr) {
    free(ptr);
}

// CUDA 内存分配
static void* cuda_alloc(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        throw RFAAError(std::string("CUDA memory allocation failed: ") + 
                       cudaGetErrorString(err));
    }
    return ptr;
}

static void cuda_free(void* ptr) {
    cudaFree(ptr);
}

static void cuda_copy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
    cudaError_t err = cudaMemcpy(dst, src, size, kind);
    if (err != cudaSuccess) {
        throw RFAAError(std::string("CUDA memcpy failed: ") + cudaGetErrorString(err));
    }
}

// Tensor 实现
template<typename T>
Tensor<T>::Tensor(const Shape& shape, Device device) 
    : shape_(shape), device_(device), own_data_(true) {
    allocate();
}

template<typename T>
Tensor<T>::Tensor(const Shape& shape, T* data, Device device, bool own)
    : shape_(shape), data_(data), device_(device), own_data_(own) {}

template<typename T>
Tensor<T>::~Tensor() {
    deallocate();
}

template<typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept 
    : shape_(std::move(other.shape_)),
      data_(other.data_),
      device_(other.device_),
      own_data_(other.own_data_) {
    other.data_ = nullptr;
    other.own_data_ = false;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        shape_ = std::move(other.shape_);
        data_ = other.data_;
        device_ = other.device_;
        own_data_ = other.own_data_;
        other.data_ = nullptr;
        other.own_data_ = false;
    }
    return *this;
}

template<typename T>
void Tensor<T>::allocate() {
    size_t size = nbytes();
    if (device_ == Device::CPU) {
        data_ = static_cast<T*>(cpu_alloc(size));
    } else {
        data_ = static_cast<T*>(cuda_alloc(size));
    }
}

template<typename T>
void Tensor<T>::deallocate() {
    if (data_ && own_data_) {
        if (device_ == Device::CPU) {
            cpu_free(data_);
        } else {
            cuda_free(data_);
        }
    }
    data_ = nullptr;
}

template<typename T>
Tensor<T> Tensor<T>::to(Device target) const {
    Tensor<T> result(shape_, target);
    if (device_ == target) {

        result.copy_from(*this);
        return result;
        //return Tensor<T>(*this);  // 需要拷贝构造，这里简化
    }

    size_t size = nbytes();
    
    if (device_ == Device::CPU && target == Device::CUDA) {
        cuda_copy(result.data_, data_, size, cudaMemcpyHostToDevice);
    } else if (device_ == Device::CUDA && target == Device::CPU) {
        cuda_copy(result.data_, data_, size, cudaMemcpyDeviceToHost);
    }
    
    return result;
}

template<typename T>
T* Tensor<T>::data(Device target) {
    if (device_ == target) return data_;
    // 临时拷贝（注意：这里简化处理，实际应缓存）
    Tensor<T> temp = to(target);
    return temp.data();
}

template<typename T>
void Tensor<T>::zero_() {
    size_t size = nbytes();
    if (device_ == Device::CPU) {
        std::memset(data_, 0, size);
    } else {
        cudaMemset(data_, 0, size);
    }
}

template<typename T>
void Tensor<T>::copy_from(const Tensor& other) {
    if (shape_.numel() != other.shape_.numel()) {
        throw RFAAError("Tensor shape mismatch in copy");
    }
    
    if (device_ == other.device_) {
        size_t size = nbytes();
        if (device_ == Device::CPU) {
            std::memcpy(data_, other.data_, size);
        } else {
            cuda_copy(data_, other.data_, size, cudaMemcpyDeviceToDevice);
        }
    } else {
        // 跨设备拷贝
        Tensor<T> temp = other.to(device_);
        std::memcpy(data_, temp.data_, nbytes());  // 同设备后拷贝
    }
}

template<typename T>
Tensor<T> Tensor<T>::view(const Shape& new_shape) const {
    if (new_shape.numel() != shape_.numel()) {
        throw RFAAError("View shape size mismatch");
    }
    return Tensor<T>(new_shape, data_, device_, false);  // 不拥有数据
}

template<typename T>
Tensor<T> Tensor<T>::slice(int dim, int64_t start, int64_t end) const {
    // 简化实现：仅支持 CPU，返回拷贝
    if (dim < 0 || dim >= shape_.ndim()) {
        throw RFAAError("Invalid slice dimension");
    }
    
    Shape new_shape = shape_;
    new_shape.dims[dim] = end - start;
    
    Tensor<T> result(new_shape, device_);
    // 实际应计算 stride 并拷贝
    // 这里简化...
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::select(int dim, int64_t index) const {
    if (dim < 0 || dim >= shape_.ndim()) {
        throw RFAAError("Invalid select dimension");
    }
    
    Shape new_shape;
    for (int i = 0; i < shape_.ndim(); ++i) {
        if (i != dim) new_shape.dims.push_back(shape_.dims[i]);
    }
    
    // 计算偏移
    int64_t stride = 1;
    for (int i = dim + 1; i < shape_.ndim(); ++i) {
        stride *= shape_.dims[i];
    }
    // ?
    int64_t offset = index * stride;
    
    return Tensor<T>(new_shape, data_ + offset, device_, false);
}

template<typename T>
std::string Tensor<T>::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(";
    for (size_t i = 0; i < shape_.dims.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_.dims[i];
    }
    oss << ", device=" << (device_ == Device::CPU ? "cpu" : "cuda") << ")";
    return oss.str();
}

// 显式实例化
template class Tensor<float>;

template<> DType Tensor<float>::dtype() const { return DType::F32; }

} // namespace rfaa

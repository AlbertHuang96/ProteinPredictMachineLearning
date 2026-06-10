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
Tensor<T> Tensor<T>::unsqueeze(int dim) const {
    // 处理负索引
    if (dim < 0) {
        dim += shape_.ndim() + 1;
    }
    
    if (dim < 0 || dim > shape_.ndim()) {
        throw RFAAError("Invalid unsqueeze dimension: " + std::to_string(dim) + 
                        " for tensor with " + std::to_string(shape_.ndim()) + " dimensions");
    }
    
    // 创建新的形状，在 dim 位置插入大小为 1 的维度
    Shape new_shape;
    for (int i = 0; i < dim; i++) {
        new_shape.dims.push_back(shape_.dims[i]);
    }
    new_shape.dims.push_back(1);  // 插入大小为 1 的维度
    for (int i = dim; i < shape_.ndim(); i++) {
        new_shape.dims.push_back(shape_.dims[i]);
    }
    
    // 返回视图（不拥有数据）
    return Tensor<T>(new_shape, data_, device_, false);
}


template<typename T>
Tensor<T> Tensor<T>::permute(const std::vector<int>& dims) const {
    // 检查dims长度是否与张量维度一致
    if (static_cast<int>(dims.size()) != shape_.ndim()) {
        throw RFAAError("permute: dims size (" + std::to_string(dims.size()) + 
                        ") != tensor ndim (" + std::to_string(shape_.ndim()) + ")");
    }
    
    // 检查dims是否包含0到ndim-1的每个数字（permutation检查）
    std::vector<bool> used(shape_.ndim(), false);
    for (int dim : dims) {
        if (dim < 0 || dim >= shape_.ndim()) {
            throw RFAAError("permute: dim " + std::to_string(dim) + " out of range [0, " + 
                            std::to_string(shape_.ndim() - 1) + "]");
        }
        if (used[dim]) {
            throw RFAAError("permute: dim " + std::to_string(dim) + " repeated");
        }
        used[dim] = true;
    }
    
    // 计算新形状
    Shape new_shape;
    for (int dim : dims) {
        new_shape.dims.push_back(shape_.dims[dim]);
    }
    
    // 创建新张量
    Tensor<T> result(new_shape, device_);
    
    // 拷贝数据（需要按新顺序遍历）
    // 计算旧形状的stride
    std::vector<int64_t> old_stride(shape_.ndim(), 1);
    for (int i = shape_.ndim() - 2; i >= 0; --i) {
        old_stride[i] = old_stride[i + 1] * shape_.dims[i + 1];
    }
    
    // 计算新形状的stride
    std::vector<int64_t> new_stride(shape_.ndim(), 1);
    for (int i = shape_.ndim() - 2; i >= 0; --i) {
        new_stride[i] = new_stride[i + 1] * new_shape.dims[i + 1];
    }
    
    // 遍历新张量的每个元素，计算在旧张量中的位置
    const T* src = data_;
    T* dst = result.data();
    int64_t total = new_shape.numel();
    
    // 使用递归或迭代方式填充数据
    // 这里使用简单的N维循环（最多支持6维）
    if (shape_.ndim() == 4) {
        // 常见情况：4D张量 (B, H, L, L) -> (B, L, H, L)
        int64_t B = new_shape.dims[0];
        int64_t D1 = new_shape.dims[1];
        int64_t D2 = new_shape.dims[2];
        int64_t D3 = new_shape.dims[3];
        
        // 新索引 (nb, n1, n2, n3) -> 旧索引 (ob, o1, o2, o3)
        // 其中 old_dim = dims[new_dim]
        for (int64_t nb = 0; nb < B; ++nb) {
            for (int64_t n1 = 0; n1 < D1; ++n1) {
                for (int64_t n2 = 0; n2 < D2; ++n2) {
                    for (int64_t n3 = 0; n3 < D3; ++n3) {
                        // 计算新索引的线性位置
                        int64_t new_linear = nb * new_stride[0] + n1 * new_stride[1] + 
                                            n2 * new_stride[2] + n3 * new_stride[3];
                        
                        // 计算对应的旧索引
                        int64_t ob = (dims[0] == 0) ? nb : (dims[0] == 1) ? n1 : (dims[0] == 2) ? n2 : n3;
                        int64_t o1 = (dims[1] == 0) ? nb : (dims[1] == 1) ? n1 : (dims[1] == 2) ? n2 : n3;
                        int64_t o2 = (dims[2] == 0) ? nb : (dims[2] == 1) ? n1 : (dims[2] == 2) ? n2 : n3;
                        int64_t o3 = (dims[3] == 0) ? nb : (dims[3] == 1) ? n1 : (dims[3] == 2) ? n2 : n3;
                        
                        int64_t old_linear = ob * old_stride[0] + o1 * old_stride[1] + 
                                            o2 * old_stride[2] + o3 * old_stride[3];
                        
                        dst[new_linear] = src[old_linear];
                    }
                }
            }
        }
    } else {
        // 通用情况：使用递归或栈来遍历
        std::vector<int64_t> new_indices(shape_.ndim(), 0);
        std::vector<int64_t> old_indices(shape_.ndim(), 0);
        
        // 使用迭代方式遍历所有组合
        int64_t* new_idx = new_indices.data();
        int64_t* old_idx = old_indices.data();
        
        for (int64_t linear = 0; linear < total; ++linear) {
            // 计算新索引
            int64_t tmp = linear;
            for (int d = shape_.ndim() - 1; d >= 0; --d) {
                new_idx[d] = tmp / new_stride[d];
                tmp %= new_stride[d];
            }
            
            // 计算旧索引
            for (int d = 0; d < shape_.ndim(); ++d) {
                old_idx[dims[d]] = new_idx[d];
            }
            
            // 计算旧线性索引
            int64_t old_linear = 0;
            for (int d = 0; d < shape_.ndim(); ++d) {
                old_linear += old_idx[d] * old_stride[d];
            }
            
            dst[linear] = src[old_linear];
        }
    }
    
    return result;
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

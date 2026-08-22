#include "ppml/Tensor.h"
#include <cuda_runtime.h>
#include <cstring>
#include <sstream>

namespace ppml {

// TODO : memory pool
// CPU 内存分配
static void* cpu_alloc(size_t size) {
    void* ptr = nullptr;
    ptr = malloc(size);
    if (!ptr) throw PPMLError("CPU memory allocation failed");
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
        throw PPMLError(std::string("CUDA memory allocation failed: ") + 
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
        throw PPMLError(std::string("CUDA memcpy failed: ") + cudaGetErrorString(err));
    }
}

// Tensor 实现
template<typename T>
Tensor<T>::Tensor(const Shape& shape, Device device) 
    : shape_(shape), device_(device), own_data_(true) {
    //  必须初始化 op/flag/src：Tensor() 默认构造不初始化这些成员，
    //    若用普通构造（如 RadialFunc 的 BN 参数 new TensorF32(Shape,Device)）
    //    再进图，op 读到垃圾（32653）→ dispatch_body 无 case → NOT_SUPPORTED。
    op = OP_NONE;
    flag = 0;
    type = TENSOR_TYPE_F32;   //  必须初始化：值构造（如 RadialFunc BN 参数）也进图作 src，
                              //    type 垃圾值 → build_backward_expand 断言失败。
    src.fill(nullptr);
    for (int i = 0; i < GGML_MAX_OP_PARAMS; ++i) op_params[i] = 0;
    allocate();
}

template<typename T>
Tensor<T>::Tensor(const Shape& shape, T* data, Device device, bool own)
    : shape_(shape), data_(data), device_(device), own_data_(own) {
    op = OP_NONE;
    flag = 0;
    type = TENSOR_TYPE_F32;
    src.fill(nullptr);
    for (int i = 0; i < GGML_MAX_OP_PARAMS; ++i) op_params[i] = 0;
}

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
    op       = other.op;
    flag     = other.flag;
    type     = other.type;
    src      = other.src;
    for (int i = 0; i < GGML_MAX_OP_PARAMS; ++i) op_params[i] = other.op_params[i];
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
        op       = other.op;
        flag     = other.flag;
        type     = other.type;
        src      = other.src;
        for (int i = 0; i < GGML_MAX_OP_PARAMS; ++i) op_params[i] = other.op_params[i];
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
        // [COPY-FAIL] 诊断：打印新旧 shape 与 numel，定位值版 forward 的 copy 错配
        fprintf(stderr, "[COPY-FAIL] dst numel=%lld src numel=%lld | dst=(", (long long)shape_.numel(), (long long)other.shape_.numel());
        for (int i = 0; i < shape_.ndim(); ++i) fprintf(stderr, "%lld%s", (long long)shape_.dims[i], i + 1 < shape_.ndim() ? "," : "");
        fprintf(stderr, ") src=(");
        for (int i = 0; i < other.shape_.ndim(); ++i) fprintf(stderr, "%lld%s", (long long)other.shape_.dims[i], i + 1 < other.shape_.ndim() ? "," : "");
        fprintf(stderr, ")\n");
        throw PPMLError("Tensor shape mismatch in copy");
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
        // [VIEW-FAIL] 诊断：打印新旧 shape 与 numel，定位值版 forward 的 view 错配
        fprintf(stderr, "[VIEW-FAIL] old numel=%lld new numel=%lld | old=(", (long long)shape_.numel(), (long long)new_shape.numel());
        for (int i = 0; i < shape_.ndim(); ++i) fprintf(stderr, "%lld%s", (long long)shape_.dims[i], i + 1 < shape_.ndim() ? "," : "");
        fprintf(stderr, ") new=(");
        for (int i = 0; i < new_shape.ndim(); ++i) fprintf(stderr, "%lld%s", (long long)new_shape.dims[i], i + 1 < new_shape.ndim() ? "," : "");
        fprintf(stderr, ")\n");
        throw PPMLError("View shape size mismatch");
    }
    return Tensor<T>(new_shape, data_, device_, false);  // 不拥有数据
}

template<typename T>
Tensor<T> Tensor<T>::slice(int dim, int64_t start, int64_t end) const {
    // 简化实现：仅支持 CPU，返回拷贝
    if (dim < 0 || dim >= shape_.ndim()) {
        throw PPMLError("Invalid slice dimension");
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
        throw PPMLError("Invalid select dimension");
    }
    
    Shape new_shape;
    for (int i = 0; i < shape_.ndim(); ++i) {
        if (i != dim) new_shape.dims.push_back(shape_.dims[i]);
    }
    
    // 正确切片（ggml 布局 dims[0] 最内维）：取 dim 维的 index，深拷贝到独立内存。
    // ⚠️ 旧实现 `Tensor(new_shape, data_+offset, false)` 是错误 view：offset 与 view 形状不匹配，
    //    导致切片数据读到错误区域（巨大值/垃圾），模板分支 emb_t1d_ 由此产生 1.78e6 溢出（[kern] op=30）。
    //    改为按外层步进深拷贝，得到正确的、独立拥有的切片。
    int64_t outer = 1;   // dim 之前维度乘积
    int64_t inner = 1;   // dim 及之后维度乘积
    for (int i = 0; i < dim; ++i)       outer *= shape_.dims[i];
    for (int i = dim; i < shape_.ndim(); ++i) inner *= shape_.dims[i];
    const int64_t dim_size   = shape_.dims[dim];
    const int64_t slice_step = dim_size > 0 ? inner / dim_size : 1;   // 每个 dim 切片（dim 之后）元素数

    Tensor<T> result(new_shape, device_);   // own_data_=true
    T* dst = result.data();
    const T* src_base = data_;
    for (int64_t o = 0; o < outer; ++o) {
        const T* src = src_base + o * inner + index * slice_step;
        std::memcpy(dst, src, slice_step * sizeof(T));
        dst += slice_step;
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::unsqueeze(int dim) const {
    // 处理负索引
    if (dim < 0) {
        dim += shape_.ndim() + 1;
    }
    
    if (dim < 0 || dim > shape_.ndim()) {
        throw PPMLError("Invalid unsqueeze dimension: " + std::to_string(dim) + 
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
    
    // 返回深拷贝（own_data_=true）：
    //  旧实现返回 view（own_data_=false）会触发 use-after-free：
    //    `input.X = input.X.unsqueeze(0)` 自引用 move 时，move 赋值 deallocate() 释放原 data，
    //    而 view 仍指向该已释放内存 → 悬垂（ASAN: heap-use-after-free，DataLoader.cpp:2300/2302/2310）。
    //    故改为深拷贝，使 move 赋值接管的是独立内存，无共享、无悬垂。
    Tensor<T> result(new_shape, device_);   // own_data_=true, allocate()
    std::memcpy(result.data(), data_, numel() * sizeof(T));
    return result;
}


template<typename T>
Tensor<T> Tensor<T>::permute(const std::vector<int>& dims) const {
    // 检查dims长度是否与张量维度一致
    if (static_cast<int>(dims.size()) != shape_.ndim()) {
        throw PPMLError("permute: dims size (" + std::to_string(dims.size()) + 
                        ") != tensor ndim (" + std::to_string(shape_.ndim()) + ")");
    }
    
    // 检查dims是否包含0到ndim-1的每个数字（permutation检查）
    std::vector<bool> used(shape_.ndim(), false);
    for (int dim : dims) {
        if (dim < 0 || dim >= shape_.ndim()) {
            throw PPMLError("permute: dim " + std::to_string(dim) + " out of range [0, " + 
                            std::to_string(shape_.ndim() - 1) + "]");
        }
        if (used[dim]) {
            throw PPMLError("permute: dim " + std::to_string(dim) + " repeated");
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
    //  旧 4D 特化分支索引映射错误（old_idx[j] = new_idx[dims[j]] 反了，正确为
    //    old_idx[dims[j]] = new_idx[j]），对 {0,3,1,2} 等 permutation 越界读。
    //    统一用通用正确实现（最多支持6维）。
    const T* src = data_;
    T* dst = result.data();
    int64_t total = new_shape.numel();
    
    // 通用情况：使用迭代方式遍历所有组合
    std::vector<int64_t> new_indices(shape_.ndim(), 0);
    std::vector<int64_t> old_indices(shape_.ndim(), 0);
    
    // 使用迭代方式遍历所有组合
    int64_t* new_idx = new_indices.data();
    int64_t* old_idx = old_indices.data();
    
    for (int64_t linear = 0; linear < total; ++linear) {
        // 计算新索引（行优先线性分解：从最低维用 % dims[d]）
        //  不能用 tmp / new_stride[d] 从低位分解（错误，会得到超界索引）。
        int64_t tmp = linear;
        for (int d = shape_.ndim() - 1; d >= 0; --d) {
            new_idx[d] = tmp % new_shape.dims[d];
            tmp /= new_shape.dims[d];
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
template class Tensor<int64_t>;

template<> DType Tensor<float>::dtype() const { return DType::F32; }
template<> DType Tensor<int64_t>::dtype() const { return DType::I64; }

} // namespace ppml

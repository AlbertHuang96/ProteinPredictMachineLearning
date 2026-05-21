#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>

namespace rfaa {

// 维度常量 (与 RFAA 对齐)
constexpr int NAATOKENS = 80;           // 统一 token 空间
constexpr int D_RBF = 64;
constexpr int D_MSA_FULL = 64;
constexpr int D_MSA = 256;              // MSA 隐层维度
constexpr int D_PAIR = 128;             // Pair 隐层维度
constexpr int D_PAIR_HIDDEN = 32;
constexpr int D_STATE = 32;             // State 隐层维度
constexpr int D_T1D = 80;               // 模板 1D 特征维度
constexpr int MSA_LATENT_DIM = 164;     // msa_latent 输入维度
constexpr int MSA_FULL_DIM = 83;        // msa_full 输入维度
constexpr int N_HEAD = 8;               // Attention head 数
constexpr int N_EXTRA_BLOCKS = 4;       // Extra blocks 数
constexpr int N_MAIN_BLOCKS = 8;        // Main blocks 数
constexpr int N_REFINE_BLOCKS = 4;      // Refinement blocks 数

// 设备类型
enum class Device { CPU, CUDA };

// 数据类型
enum class DType { F32, F16, BF16 };

// 张量形状
struct Shape {
    std::vector<int64_t> dims;
    
    Shape() = default;
    Shape(std::initializer_list<int64_t> d) : dims(d) {}
    Shape(std::vector<int64_t> d) : dims(std::move(d)) {}
    
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
    
    int ndim() const { return static_cast<int>(dims.size()); }
};

// 运行时异常
class RFAAError : public std::runtime_error {
public:
    explicit RFAAError(const std::string& msg) : std::runtime_error(msg) {}
};

// 前向声明
template<typename T>
class Tensor;

class Model;
class Track;
class MSATrack;
class PairTrack;
class StateTrack;

} // namespace rfaa

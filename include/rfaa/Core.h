#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>

namespace rfaa {

// 维度常量 (与 RFAA 对齐)
constexpr int NAATOKENS = 80;           // 统一 token 空间
constexpr int NPROTAAS = 20;            // 标准蛋白质氨基酸种类
constexpr int NNAPROTAAS = 25;          // 蛋白质 + 核酸氨基酸种类
constexpr int NTOTALDOFS = 20;          // 总自由度（扭转角数）
constexpr int D_RBF = 64;
constexpr int D_MSA_FULL = 64;
constexpr int D_MSA = 256;              // MSA 隐层维度
constexpr int D_PAIR = 128;             // Pair 隐层维度
constexpr int D_PAIR_HIDDEN = 32;
constexpr int D_STATE = 32;             // State 隐层维度
constexpr int D_T1D = 80;               // 模板 1D 特征维度
constexpr int D_T2D = 68;              // 模板 2D 特征维度 (RF2: 61 dist one-hot + 6 orien + 1 mask)
constexpr int D_TOR = 30;               // 侧链扭转角维度
constexpr int MSA_LATENT_DIM = 164;     // msa_latent 输入维度
constexpr int MSA_FULL_DIM = 83;        // msa_full 输入维度
constexpr int N_HEAD = 8;               // Attention head 数
constexpr int N_EXTRA_BLOCKS = 4;       // Extra blocks 数
constexpr int N_MAIN_BLOCKS = 8;      // Main blocks 数  8 or 32 ?
constexpr int N_REFINE_BLOCKS = 4;      // Refinement blocks 数

constexpr int N_L0_IN_FEATS = 32;
constexpr int N_EDGE_FEATS = 32;

// 3D SE 输入维度常量 (由基础维度组合)
constexpr int ITER_NODE_3D_IN   = D_MSA + 21;             // 256 + 21 = 277   IterBlock 3D node input
constexpr int ITER_NODE_3D_OUT  = N_L0_IN_FEATS;          // 32                IterBlock 3D node output
constexpr int ITER_EDGE_3D_OUT  = N_EDGE_FEATS;           // 32                IterBlock 3D edge output
constexpr int REFINE_NODE_IN_DIM  = D_MSA + 21 + D_STATE;  // 256+21+32 = 309  RefineBlock node input
constexpr int REFINE_NODE_OUT_DIM = N_L0_IN_FEATS;         // 32               RefineBlock node output
constexpr int REFINE_EDGE_IN_DIM2 = N_EDGE_FEATS + 64 + 1;  // 32+64+1 = 97    RefineBlock edge input stage2
constexpr int ITER_N_BLOCKS = N_EXTRA_BLOCKS + N_MAIN_BLOCKS; // 12 = 4+8

// 共享 forward 内部参数维度
constexpr int MSA2PAIR_HIDDEN = 16;          // msa2pair outer product hidden dim
constexpr int PAIR2PAIR_GATE_HIDDEN = 16;    // pair2pair gate outer product hidden dim

// 设备类型
enum class Device { CPU, CUDA };

// 数据类型
enum class DType { F32, F16, BF16, I64, I32, I16, I8 };

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

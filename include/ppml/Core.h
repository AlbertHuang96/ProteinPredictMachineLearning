#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdlib>   // std::getenv / std::strtoul（ppml_rng_seed）
#include <random>    // std::random_device（ppml_rng_seed）

namespace ppml {

// ===== 可复现随机种子（诊断/对比实验用，2026-09-12）=====
//   PPML_SEED=<uint> 已设置 → 返回 PPML_SEED + salt（同一 salt 每次运行相同 ⇒ 权重初始化与
//     dropout 掩码可复现）；未设置 → std::random_device{}()（与既有行为完全一致：每次运行不同）。
// 动机：train.cpp 不加载权重（每次运行随机初始化），跨 run 对比不同管线/模式时，随机初始化噪声
//   （loss ±2 量级）会淹没被测差异（如 Pass1 实现方式对 loss 的影响）。固定种子后可做有意义的对比。
inline uint32_t ppml_rng_seed(uint32_t salt = 0) {
    const char* s = std::getenv("PPML_SEED");
    if (s == nullptr) return std::random_device{}();
    static const uint32_t base = static_cast<uint32_t>(std::strtoul(s, nullptr, 10));
    return base + salt;
}

// 维度常量 (与 PPML 对齐)
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

    // 安全访问第 i 维（2026-09-11）：超出 ndim 时返回 1（ggml 语义：缺失维视为 1）。
    //   背景：dims 是 std::vector，`dims[i]`（i ≥ ndim）越界是 UB —— 读到堆垃圾 → 形状爆炸
    //   → 曾出现 168.9GB 超大分配 / CUDA grid 越限（invalid configuration argument）。
    //   约定：所有"可能不足 4 维"的读写一律用 dim(i)，禁止裸 dims[i]。
    int64_t dim(int i) const {
        return (i >= 0 && i < static_cast<int>(dims.size())) ? dims[i] : 1;
    }
};

// 运行时异常
class PPMLError : public std::runtime_error {
public:
    explicit PPMLError(const std::string& msg) : std::runtime_error(msg) {}
};

// 前向声明
template<typename T>
class Tensor;

class Model;
class Track;
class MSATrack;
class PairTrack;
class StateTrack;

} // namespace ppml

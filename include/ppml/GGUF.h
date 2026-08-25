#pragma once 

#include <cstdint>
#include <string>
#include <vector>
#include <map>

#include "ppml/Tensor.h"

namespace ppml {

    enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// ========== 量化类型 (与 GGML 对齐) ==========
enum class GGMLQuantType : uint32_t {
    F32  = 0,
    F16  = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    I8   = 24,
    I16  = 25,
    I32  = 26,
    I64  = 27,
    BF16 = 30,
};

// 计算"布局指纹" (FNV-1a)：仅基于 (语义名, 各维形状) 元数据，不读权重数据。
// 顺序敏感：只要参数收集顺序/名字/形状变化，指纹即改变。用于校验
// GGUF 中参数的布局与当前模型定义是否一致，防"顺序改变但 shape 相同"的静默错配。
uint64_t fnv1a_layout_hash(const std::vector<std::string>& tensor_names,
                           const std::vector<std::vector<uint64_t>>& shapes);

// 优化器状态序列化 (断点续训用)：以 GGUF 额外张量保存，命名 "opt.m.<param_name>" /
// "opt.v.<param_name>" (1D, 与参数 numel 相同)。保存时追加在模型参数之后，
// 不参与 layout_hash（因此旧 checkpoint 无 opt 张量也能正常加载 → 优化器从 0 初始化）。
struct GGUFRawTensor {
    std::string name;
    std::vector<float> data;  // 1D 扁平数据
};

// load_gguf: 加载权重并校验布局 (数量/形状/layout_hash)。
//   float_meta_out (可选): 收集文件中的数值型元数据 (如 epoch/sample_pos/resume_epoch)，
//   供断点续训恢复训练进度。
//   raw_tensors_out (可选): 收集文件中参数之外的额外张量 (命名 "opt.m.<name>"/"opt.v.<name>")，
//   供恢复 AdamW m/v 动量。旧文件无额外张量时该 map 为空。
void load_gguf(const std::string& path,
               std::vector<TensorF32*>& params,
               const std::vector<std::string>& tensor_names = {},
               std::map<std::string, float>* float_meta_out = nullptr,
               std::map<std::string, std::vector<float>>* raw_tensors_out = nullptr);

void save_gguf(const std::vector<TensorF32*>& params,
               const std::string& path,
               const std::vector<std::pair<std::string, float>>& float_meta,
               const std::vector<std::pair<std::string, std::string>>& str_meta,
               const std::vector<std::string>& tensor_names = {},
               const std::vector<GGUFRawTensor>& extra_tensors = {});

} // namespace ppml
#pragma once 

#include <cstdint>
#include <string>
#include <vector>

#include "rfaa/Tensor.h"

namespace rfaa {

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

void load_gguf(const std::string& path,
               std::vector<TensorF32*>& params);

void save_gguf(const std::vector<TensorF32*>& params,
               const std::string& path,
               const std::vector<std::pair<std::string, float>>& float_meta,
               const std::vector<std::pair<std::string, std::string>>& str_meta,
               const std::vector<std::string>& tensor_names = {});

} // namespace rfaa
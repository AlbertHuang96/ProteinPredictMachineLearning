#include "rfaa/Tensor.h"
#include "rfaa/GGUF.h"
#include <fstream>
#include <cstring>
#include <cassert>
#include <sstream>
#include <vector>

namespace rfaa {

// ============================================================
// 内部常量/工具
// ============================================================

static constexpr uint32_t GGUF_MAGIC   = 0x46554747;
static constexpr uint32_t GGUF_VERSION = 3;
static constexpr size_t   GGUF_ALIGN   = 32;  // tensor 数据对齐

// v3 用 uint64 存长度
static constexpr size_t SIZE_T_LEN = 8;

static inline size_t align_up(size_t n, size_t align) {
    return (n + align - 1) / align * align;
}

// ============================================================
// 辅助结构：GGUF 输出缓冲
// ============================================================

class GGUFWriter {
public:
    explicit GGUFWriter(std::ostream& os) : os_(os) {}

    void u32(uint32_t v) { os_.write((const char*)&v, 4); }
    void u64(uint64_t v) { os_.write((const char*)&v, 8); }
    void f32(float    v) { os_.write((const char*)&v, 4); }
    void raw(const void* data, size_t n) { os_.write((const char*)data, n); }
    void pad_to(size_t align) {
        size_t pos = tell();
        size_t next = align_up(pos, align);
        while (pos++ < next) os_.put('\0');
    }
    void str(const std::string& s) {
        u64(s.size());
        raw(s.data(), s.size());
    }
    size_t tell() const { return static_cast<size_t>(os_.tellp()); }

private:
    std::ostream& os_;
};

struct GGUFLayout {
    // ===== 各部分字节数 =====
    size_t header_bytes       = 0;   // magic + version + tensor_count + kv_count
    size_t metadata_bytes     = 0;   // 所有 KV pair
    size_t tensor_infos_bytes = 0;   // 所有 tensor 的 name/dims/type/offset
    size_t data_start_offset  = 0;   // 对齐后的数据起点

    // ===== Tensor 数据偏移 (数组) =====
    std::vector<size_t> tensor_offsets;

    // ===== KV 条目大小计算 =====
    static size_t kv_size_float(const std::string& key)    {
        return SIZE_T_LEN + key.size() + 4 + 4;  // str_len + str + type(4) + float(4)
    }
    static size_t kv_size_string(const std::string& key, const std::string& val) {
        return SIZE_T_LEN + key.size() + 4 + SIZE_T_LEN + val.size();  // key_str + type + val_str
    }

    // ===== Tensor info 大小计算 =====
    static size_t tensor_info_size(int n_dims) {
        // name_str + n_dims(4) + dims[](SIZE_T_LEN each) + type(4) + offset(8)
        return SIZE_T_LEN + /*name_len=*/0  // ← name_len 在调用处加
               + 4 + n_dims * SIZE_T_LEN + 4 + 8;
    }
};

static GGUFLayout compute_layout(
    const std::vector<TensorF32*>& params,
    const std::vector<std::pair<std::string, float>>& float_meta,
    const std::vector<std::pair<std::string, std::string>>& str_meta,
    const std::vector<std::string>& tensor_names = {})
{
    GGUFLayout L;

    // ===== 1. Header 固定大小 =====
    L.header_bytes = 4 + 4 + 8 + 8;   // magic + version + tensor_count + kv_count = 24

    // ===== 2. Metadata 大小 =====
    L.metadata_bytes = 0;
    for (auto& [k, v] : float_meta)  L.metadata_bytes += GGUFLayout::kv_size_float(k);
    for (auto& [k, v] : str_meta)    L.metadata_bytes += GGUFLayout::kv_size_string(k, v);

    // ===== 3. Tensor Infos 大小 =====
    L.tensor_infos_bytes = 0;
    for (size_t i = 0; i < params.size(); i++) {
        // 优先使用语义名; 未提供时退回 tensor_{i} 编号, 便于可读
        std::string name;
        if (i < tensor_names.size() && !tensor_names[i].empty())
            name = tensor_names[i];
        else
            name = "tensor_" + std::to_string(i);
        L.tensor_infos_bytes += SIZE_T_LEN + name.size()    // name_len + name
                               + 4                           // n_dims
                               + params[i]->shape().ndim() * SIZE_T_LEN  // dims[]
                               + 4                           // type
                               + 8;                          // offset
    }

    // ===== 4. 数据起点 (对齐) =====
    L.data_start_offset = align_up(
        L.header_bytes + L.metadata_bytes + L.tensor_infos_bytes,
        GGUF_ALIGN);

    // ===== 5. 每个 Tensor 的数据偏移 =====
    L.tensor_offsets.resize(params.size());
    size_t cur = L.data_start_offset;
    for (size_t i = 0; i < params.size(); i++) {
        L.tensor_offsets[i] = cur;
        size_t bytes = params[i]->numel() * sizeof(float);  // F32
        cur = align_up(cur + bytes, GGUF_ALIGN);
    }

    return L;
}

void save_gguf(const std::vector<TensorF32*>& params,
               const std::string& path,
               const std::vector<std::pair<std::string, float>>& float_meta,
               const std::vector<std::pair<std::string, std::string>>& str_meta,
               const std::vector<std::string>& tensor_names)
{
    // ===== 阶段 A: 纯算术计算布局 =====
    GGUFLayout L = compute_layout(params, float_meta, str_meta, tensor_names);

    // ===== 阶段 B: 顺序写入文件 (一次性, 不 seek 回跳) =====
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open: " + path);
    GGUFWriter w(ofs);

    // ------ B1: Header ------
    w.u32(GGUF_MAGIC);
    w.u32(GGUF_VERSION);
    w.u64(params.size());                              // tensor_count
    w.u64(float_meta.size() + str_meta.size());        // kv_count

    // ------ B2: Metadata (float) ------
    for (auto& [k, v] : float_meta) {
        w.str(k);
        w.u32(static_cast<uint32_t>(GGUFValueType::FLOAT32));
        w.f32(v);
    }

    // ------ B3: Metadata (string) ------
    for (auto& [k, v] : str_meta) {
        w.str(k);
        w.u32(static_cast<uint32_t>(GGUFValueType::STRING));
        w.str(v);
    }

    // ------ B4: Tensor Infos ------
    for (size_t i = 0; i < params.size(); i++) {
        TensorF32* t = params[i];

        // 名称 (优先语义名; 否则退回 index 编号)
        std::string tname;
        if (i < tensor_names.size() && !tensor_names[i].empty())
            tname = tensor_names[i];
        else
            tname = "tensor_" + std::to_string(i);
        w.str(tname);

        // 维度
        w.u32(static_cast<uint32_t>(t->shape().ndim()));
        for (int d = 0; d < t->shape().ndim(); d++) {
            w.u64(static_cast<uint64_t>(t->shape().dims[d]));
        }

        // 类型 (F32)
        w.u32(static_cast<uint32_t>(GGMLQuantType::F32));

        // 数据偏移 — 来自预先计算
        w.u64(L.tensor_offsets[i]);
    }

    // ------ B5: 填充到数据起始偏移 ------
    w.pad_to(L.data_start_offset);

    // ------ B6: Tensor 数据 ------
    for (size_t i = 0; i < params.size(); i++) {
        TensorF32* t = params[i];
        size_t bytes = t->numel() * sizeof(float);
        w.raw(t->data(), bytes);
        w.pad_to(GGUF_ALIGN);  // 对齐下一个 tensor
    }

    ofs.close();
}

void load_gguf(const std::string& path,
               std::vector<TensorF32*>& params)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open: " + path);

    // ------ 1. Header ------
    auto r32 = [&]() { uint32_t v; ifs.read((char*)&v,4); return v; };
    auto r64 = [&]() { uint64_t v; ifs.read((char*)&v,8); return v; };
    auto rstr = [&]() { uint64_t len=r64(); std::string s(len,0); ifs.read(&s[0],len); return s; };

    uint32_t magic = r32();
    if (magic != GGUF_MAGIC) throw std::runtime_error("Not a GGUF file");
    uint32_t version = r32();
    if (version != GGUF_VERSION) throw std::runtime_error("Wrong version");

    uint64_t tensor_count = r64();
    uint64_t kv_count     = r64();

    if (tensor_count != params.size())
        throw std::runtime_error("Tensor count mismatch: file=" + 
            std::to_string(tensor_count) + " params=" + std::to_string(params.size()));

    // ------ 2. 跳过 Metadata ------
    for (uint64_t i = 0; i < kv_count; i++) {
        rstr();                           // skip key
        uint32_t vtype = r32();           // type
        switch (static_cast<GGUFValueType>(vtype)) {
            case GGUFValueType::UINT8:  case GGUFValueType::INT8:
            case GGUFValueType::BOOL:   ifs.seekg(1, std::ios::cur); break;
            case GGUFValueType::UINT16: case GGUFValueType::INT16:
                                        ifs.seekg(2, std::ios::cur); break;
            case GGUFValueType::UINT32: case GGUFValueType::INT32:
            case GGUFValueType::FLOAT32: ifs.seekg(4, std::ios::cur); break;
            case GGUFValueType::UINT64: case GGUFValueType::INT64:
            case GGUFValueType::FLOAT64: ifs.seekg(8, std::ios::cur); break;
            case GGUFValueType::STRING:  rstr(); break;
            case GGUFValueType::ARRAY: {
                uint32_t atype = r32();
                uint64_t alen  = r64();
                for (uint64_t j = 0; j < alen; j++) {
                    // 简化: 假设数组元素是 int32
                    ifs.seekg(4, std::ios::cur);
                }
            } break;
        }
    }

    // ------ 3. 读 Tensor Infos (收集偏移) ------
    struct TI { std::string name; std::vector<uint64_t> dims; uint32_t type; uint64_t offset; };
    std::vector<TI> tinfos(tensor_count);

    for (uint64_t i = 0; i < tensor_count; i++) {
        tinfos[i].name = rstr();
        uint32_t nd = r32();
        tinfos[i].dims.resize(nd);
        for (uint32_t d = 0; d < nd; d++) tinfos[i].dims[d] = r64();
        tinfos[i].type   = r32();
        tinfos[i].offset = r64();
    }

    // ------ 4. 定位到每个 offset 读数据 ------
    for (uint64_t i = 0; i < tensor_count; i++) {
        // 校验形状 (轻量)
        TensorF32* t = params[i];
        if (static_cast<uint32_t>(t->shape().ndim()) != tinfos[i].dims.size())
            throw std::runtime_error("Dimension count mismatch for tensor " + std::to_string(i));
        for (size_t d = 0; d < tinfos[i].dims.size(); d++) {
            if (static_cast<uint64_t>(t->shape().dims[d]) != tinfos[i].dims[d])
                throw std::runtime_error("Shape mismatch for tensor " + std::to_string(i) +
                    " dim[" + std::to_string(d) + "]");
        }

        // 定位 + 读
        ifs.seekg(tinfos[i].offset, std::ios::beg);
        size_t bytes = t->numel() * sizeof(float);
        ifs.read(reinterpret_cast<char*>(t->data()), bytes);
    }

    ifs.close();
}

/* void save_model_gguf(RFAAModel& model, const std::string& path, int step) {
    save_gguf(model.params(), path,
        {{"rfaa.step", float(step)},
         {"rfaa.d_msa", float(D_MSA)},
         {"rfaa.d_pair", float(D_PAIR)},
         {"rfaa.d_state", float(D_STATE)}},
        {{"rfaa.arch", "rfaa_v1"}}
    );
} */

/* void load_model_gguf(RFAAModel& model, const std::string& path) {
    load_gguf(path, model.params());
} */

}
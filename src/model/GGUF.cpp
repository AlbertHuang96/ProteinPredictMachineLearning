#include "ppml/Tensor.h"
#include "ppml/GGUF.h"
#include <fstream>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <sstream>
#include <vector>
#include <map>

namespace ppml {

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
// 布局指纹 (FNV-1a 64-bit)
// 仅对 (名字, 各维形状) 元数据做增量哈希，不读权重数据。顺序敏感。
// ============================================================
static constexpr uint64_t FNV_OFFSET_BASIS = 1469598103934665603ULL;
static constexpr uint64_t FNV_PRIME        = 1099511628211ULL;

static inline void fnv_feed(uint64_t& h, uint8_t byte) {
    h ^= byte;
    h *= FNV_PRIME;
}

static inline void fnv_feed_u32(uint64_t& h, uint32_t v) {
    for (int b = 0; b < 4; ++b) fnv_feed(h, static_cast<uint8_t>((v >> (8 * b)) & 0xFF));
}

static inline void fnv_feed_u64(uint64_t& h, uint64_t v) {
    for (int b = 0; b < 8; ++b) fnv_feed(h, static_cast<uint8_t>((v >> (8 * b)) & 0xFF));
}

static inline void fnv_feed_str(uint64_t& h, const std::string& s) {
    for (char c : s) fnv_feed(h, static_cast<uint8_t>(c));
}

// 核心：顺序喂入每个 (名字, 维度数, 各维大小)。名字为空时退回编号。
uint64_t fnv1a_layout_hash(const std::vector<std::string>& tensor_names,
                           const std::vector<std::vector<uint64_t>>& shapes) {
    uint64_t h = FNV_OFFSET_BASIS;
    size_t n = shapes.size();
    for (size_t i = 0; i < n; ++i) {
        std::string name;
        if (i < tensor_names.size() && !tensor_names[i].empty())
            name = tensor_names[i];
        else
            name = "tensor_" + std::to_string(i);

        // 名字
        fnv_feed_str(h, name);
        fnv_feed_u32(h, 0xFF);  // 分隔符，防止名字拼接歧义
        // 维度数 + 各维大小
        fnv_feed_u32(h, static_cast<uint32_t>(shapes[i].size()));
        for (uint64_t d : shapes[i]) fnv_feed_u64(h, d);
        fnv_feed_u32(h, 0xFE);  // 张量间分隔符
    }
    return h;
}

static std::string to_hex64(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v);
    return std::string(buf);
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
    const std::vector<std::string>& tensor_names = {},
    const std::vector<GGUFRawTensor>& extra_tensors = {})
{
    GGUFLayout L;
    const size_t n_all = params.size() + extra_tensors.size();

    // ===== 1. Header 固定大小 =====
    L.header_bytes = 4 + 4 + 8 + 8;   // magic + version + tensor_count + kv_count = 24

    // ===== 2. Metadata 大小 =====
    L.metadata_bytes = 0;
    for (auto& [k, v] : float_meta)  L.metadata_bytes += GGUFLayout::kv_size_float(k);
    for (auto& [k, v] : str_meta)    L.metadata_bytes += GGUFLayout::kv_size_string(k, v);

    // ===== 3. Tensor Infos 大小 =====
    //    模型参数 (params) + 优化器状态 (extra_tensors, 1D)。extra 不参与 layout_hash。
    L.tensor_infos_bytes = 0;
    for (size_t i = 0; i < n_all; i++) {
        std::string name;
        if (i < params.size()) {
            if (i < tensor_names.size() && !tensor_names[i].empty())
                name = tensor_names[i];
            else
                name = "tensor_" + std::to_string(i);
        } else {
            name = extra_tensors[i - params.size()].name;
        }
        const size_t ndim = (i < params.size())
            ? static_cast<size_t>(params[i]->shape().ndim()) : 1;
        L.tensor_infos_bytes += SIZE_T_LEN + name.size()    // name_len + name
                               + 4                           // n_dims
                               + ndim * SIZE_T_LEN           // dims[]
                               + 4                           // type
                               + 8;                          // offset
    }

    // ===== 4. 数据起点 (对齐) =====
    L.data_start_offset = align_up(
        L.header_bytes + L.metadata_bytes + L.tensor_infos_bytes,
        GGUF_ALIGN);

    // ===== 5. 每个 Tensor 的数据偏移 =====
    L.tensor_offsets.resize(n_all);
    size_t cur = L.data_start_offset;
    for (size_t i = 0; i < n_all; i++) {
        L.tensor_offsets[i] = cur;
        const size_t numel = (i < params.size())
            ? static_cast<size_t>(params[i]->numel())
            : extra_tensors[i - params.size()].data.size();
        size_t bytes = numel * sizeof(float);  // F32
        cur = align_up(cur + bytes, GGUF_ALIGN);
    }

    return L;
}

void save_gguf(const std::vector<TensorF32*>& params,
               const std::string& path,
               const std::vector<std::pair<std::string, float>>& float_meta,
               const std::vector<std::pair<std::string, std::string>>& str_meta,
               const std::vector<std::string>& tensor_names,
               const std::vector<GGUFRawTensor>& extra_tensors)
{
    const size_t n_all = params.size() + extra_tensors.size();

    // ===== 阶段 0: 计算布局指纹 (仅模型参数: 名字+形状) 并注入元数据 =====
    //   extra_tensors (优化器 m/v) 不参与 hash → 旧文件 (无 opt) 与 新文件 (有 opt)
    //   的 layout_hash 一致，保证向后兼容。
    std::vector<std::vector<uint64_t>> shapes;
    shapes.reserve(params.size());
    for (auto* t : params) {
        std::vector<uint64_t> dims;
        dims.reserve(static_cast<size_t>(t->shape().ndim()));
        for (int d = 0; d < t->shape().ndim(); ++d)
            dims.push_back(static_cast<uint64_t>(t->shape().dims[d]));
        shapes.push_back(std::move(dims));
    }
    uint64_t layout_hash = fnv1a_layout_hash(tensor_names, shapes);

    std::vector<std::pair<std::string, std::string>> str_meta_in;
    str_meta_in.reserve(str_meta.size() + 1);
    str_meta_in.push_back({"layout_hash", to_hex64(layout_hash)});
    str_meta_in.insert(str_meta_in.end(), str_meta.begin(), str_meta.end());

    // ===== 阶段 A: 纯算术计算布局 =====
    GGUFLayout L = compute_layout(params, float_meta, str_meta_in, tensor_names, extra_tensors);

    // ===== 阶段 B: 顺序写入文件 (一次性, 不 seek 回跳) =====
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open: " + path);
    GGUFWriter w(ofs);

    // ------ B1: Header ------
    w.u32(GGUF_MAGIC);
    w.u32(GGUF_VERSION);
    w.u64(n_all);                                      // tensor_count (含 opt)
    w.u64(float_meta.size() + str_meta_in.size());     // kv_count

    // ------ B2: Metadata (float) ------
    for (auto& [k, v] : float_meta) {
        w.str(k);
        w.u32(static_cast<uint32_t>(GGUFValueType::FLOAT32));
        w.f32(v);
    }

    // ------ B3: Metadata (string) ------
    for (auto& [k, v] : str_meta_in) {
        w.str(k);
        w.u32(static_cast<uint32_t>(GGUFValueType::STRING));
        w.str(v);
    }

    // ------ B4: Tensor Infos ------
    for (size_t i = 0; i < n_all; i++) {
        // 名称
        std::string tname;
        size_t ndim;
        if (i < params.size()) {
            if (i < tensor_names.size() && !tensor_names[i].empty())
                tname = tensor_names[i];
            else
                tname = "tensor_" + std::to_string(i);
            ndim = static_cast<size_t>(params[i]->shape().ndim());
        } else {
            tname = extra_tensors[i - params.size()].name;
            ndim  = 1;   // 优化器状态为 1D
        }
        w.str(tname);
        w.u32(static_cast<uint32_t>(ndim));

        // 维度
        if (i < params.size()) {
            TensorF32* t = params[i];
            for (size_t d = 0; d < ndim; d++)
                w.u64(static_cast<uint64_t>(t->shape().dims[d]));
        } else {
            w.u64(extra_tensors[i - params.size()].data.size());  // [N]
        }

        // 类型 (F32)
        w.u32(static_cast<uint32_t>(GGMLQuantType::F32));

        // 数据偏移 — 来自预先计算
        w.u64(L.tensor_offsets[i]);
    }

    // ------ B5: 填充到数据起始偏移 ------
    w.pad_to(L.data_start_offset);

    // ------ B6: Tensor 数据 ------
    for (size_t i = 0; i < n_all; i++) {
        if (i < params.size()) {
            TensorF32* t = params[i];
            size_t bytes = t->numel() * sizeof(float);
            w.raw(t->data(), bytes);
        } else {
            const std::vector<float>& d = extra_tensors[i - params.size()].data;
            w.raw(d.data(), d.size() * sizeof(float));
        }
        w.pad_to(GGUF_ALIGN);  // 对齐下一个 tensor
    }

    ofs.close();
}

void load_gguf(const std::string& path,
               std::vector<TensorF32*>& params,
               const std::vector<std::string>& tensor_names,
               std::map<std::string, float>* float_meta_out,
               std::map<std::string, std::vector<float>>* raw_tensors_out)
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

    // 文件张量数 ≥ 模型参数数：多余的为优化器状态 (opt.*)。缺失 → 布局错。
    if (tensor_count < params.size())
        throw std::runtime_error("Tensor count mismatch: file=" + 
            std::to_string(tensor_count) + " params=" + std::to_string(params.size()));

    // ------ 2. 解析 Metadata (捕获 layout_hash 供校验 + 数值 meta 供续训/状态恢复) ------
    std::string saved_layout_hash;
    for (uint64_t i = 0; i < kv_count; i++) {
        std::string key = rstr();
        uint32_t vtype = r32();           // type
        switch (static_cast<GGUFValueType>(vtype)) {
            case GGUFValueType::UINT8:  case GGUFValueType::INT8:
            case GGUFValueType::BOOL:   ifs.seekg(1, std::ios::cur); break;
            case GGUFValueType::UINT16: case GGUFValueType::INT16:
                                        ifs.seekg(2, std::ios::cur); break;
            case GGUFValueType::FLOAT32: {
                // 注意: 必须是 bit-copy 到 float, 不能 (float)v32 强转 (会把 0x3F800000
                // = 1.0f 的原始字节读成整数 1065353216)
                float v; ifs.read((char*)&v, 4);
                if (float_meta_out) (*float_meta_out)[key] = v;
                break;
            }
            case GGUFValueType::UINT32: case GGUFValueType::INT32: {
                uint32_t v32; ifs.read((char*)&v32, 4);
                if (float_meta_out) (*float_meta_out)[key] = (float)v32;
                break;
            }
            case GGUFValueType::UINT64: case GGUFValueType::INT64: {
                uint64_t v64; ifs.read((char*)&v64, 8);
                if (float_meta_out) (*float_meta_out)[key] = (float)v64;
                break;
            }
            case GGUFValueType::FLOAT64: {
                double v; ifs.read((char*)&v, 8);
                if (float_meta_out) (*float_meta_out)[key] = (float)v;
                break;
            }
            case GGUFValueType::STRING:
                if (key == "layout_hash") saved_layout_hash = rstr();
                else rstr();
                break;
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
    //  前 params.size() 个 = 模型参数 (校验形状 + 读入)；多余 = 优化器状态 (opt.*)。
    for (uint64_t i = 0; i < tensor_count; i++) {
        if (i < params.size()) {
            // 模型参数：形状校验 + 读入
            TensorF32* t = params[i];
            if (static_cast<uint32_t>(t->shape().ndim()) != tinfos[i].dims.size())
                throw std::runtime_error("Dimension count mismatch for tensor " + std::to_string(i));
            for (size_t d = 0; d < tinfos[i].dims.size(); d++) {
                if (static_cast<uint64_t>(t->shape().dims[d]) != tinfos[i].dims[d])
                    throw std::runtime_error("Shape mismatch for tensor " + std::to_string(i) +
                        " dim[" + std::to_string(d) + "]");
            }
            ifs.seekg(tinfos[i].offset, std::ios::beg);
            size_t bytes = t->numel() * sizeof(float);
            ifs.read(reinterpret_cast<char*>(t->data()), bytes);
        } else {
            // 优化器状态：仅收集 "opt." 前缀的 1D 张量 (AdamW m/v)；其他忽略
            if (!raw_tensors_out) continue;
            if (tinfos[i].name.rfind("opt.", 0) != 0) continue;
            if (tinfos[i].dims.size() != 1) continue;
            std::vector<float> data(static_cast<size_t>(tinfos[i].dims[0]));
            ifs.seekg(tinfos[i].offset, std::ios::beg);
            ifs.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
            (*raw_tensors_out)[tinfos[i].name] = std::move(data);
        }
    }

    // ------ 5. 布局指纹校验 (若文件存有 layout_hash) ------
    if (!saved_layout_hash.empty()) {
        // 用文件中的 (名字, 形状) 重算指纹；⚠️ 只取前 params.size() 个（模型参数），
        // 优化器状态 (opt.*) 不参与 hash（与保存侧一致，否则含 opt 的新文件会误报 mismatch）
        const size_t hash_n = std::min<size_t>(static_cast<size_t>(tensor_count), params.size());
        std::vector<std::string> file_names;
        file_names.reserve(hash_n);
        for (size_t k = 0; k < hash_n; ++k) file_names.push_back(tinfos[k].name);
        std::vector<std::vector<uint64_t>> file_shapes;
        file_shapes.reserve(hash_n);
        for (size_t k = 0; k < hash_n; ++k) file_shapes.push_back(tinfos[k].dims);

        uint64_t cur_hash = fnv1a_layout_hash(file_names, file_shapes);
        std::string cur_hex = to_hex64(cur_hash);

        // 若调用方提供了当前模型的语义名, 额外按名字+形状重算, 用于捕获
        // "顺序改变但 shape 相同" 的静默错配 (名字才是布局的强约束)。
        if (!tensor_names.empty()) {
            std::vector<std::vector<uint64_t>> cur_shapes;
            cur_shapes.reserve(params.size());
            for (auto* t : params) {
                std::vector<uint64_t> dims;
                dims.reserve(static_cast<size_t>(t->shape().ndim()));
                for (int d = 0; d < t->shape().ndim(); ++d)
                    dims.push_back(static_cast<uint64_t>(t->shape().dims[d]));
                cur_shapes.push_back(std::move(dims));
            }
            cur_hash = fnv1a_layout_hash(tensor_names, cur_shapes);
            cur_hex = to_hex64(cur_hash);
        }

        if (cur_hex != saved_layout_hash) {
            ifs.close();
            throw std::runtime_error(
                "Layout mismatch: file=" + saved_layout_hash +
                " current=" + cur_hex +
                " (parameter collection order/shape/name changed vs. model definition)");
        }
    }

    ifs.close();
}

/* void save_model_gguf(PPMLModel& model, const std::string& path, int step) {
    save_gguf(model.params(), path,
        {{"ppml.step", float(step)},
         {"ppml.d_msa", float(D_MSA)},
         {"ppml.d_pair", float(D_PAIR)},
         {"ppml.d_state", float(D_STATE)}},
        {{"ppml.arch", "ppml_v1"}}
    );
} */

/* void load_model_gguf(PPMLModel& model, const std::string& path) {
    load_gguf(path, model.params());
} */

}
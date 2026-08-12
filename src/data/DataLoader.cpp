#include "rfaa/DataLoader.h"
#include "rfaa/DistogramBins.h"
#include <fstream>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cctype>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>

#include <random>

#include <queue>
#include <algorithm>
#include <limits>

#include <filesystem>
#include <set>
#include <map>
#include <array>

// ========== FFindex 数据结构 ==========

struct FFindexEntry {
    std::string name;       // 条目名称（如 "1ABC_A"）
    int64_t offset;         // 在数据文件中的偏移量（字节）
    int64_t length;         // 数据长度（字节）
};

struct FFindexDB {
    std::vector<FFindexEntry> index;  // 索引列表
    std::shared_ptr<void> data;        // 内存映射的数据指针
    size_t data_size;                   // 数据大小（字节）
    std::string data_path;             // 数据文件路径（用于 mmap）
};

// ========== 读取 .ffindex 文件 ==========
std::vector<FFindexEntry> read_index(const std::string& ffindex_filename) {
    std::vector<FFindexEntry> entries;
    
    std::ifstream fh(ffindex_filename);
    if (!fh.is_open()) {
        throw std::runtime_error("Cannot open ffindex file: " + ffindex_filename);
    }
    
    std::string line;
    while (std::getline(fh, line)) {
        if (line.empty()) continue;
        
        // 用 tab 分割: name\toffset\tlength
        size_t tab1 = line.find('\t');
        size_t tab2 = line.find('\t', tab1 + 1);
        
        if (tab1 == std::string::npos || tab2 == std::string::npos) {
            std::cerr << "Warning: Invalid ffindex line: " << line << std::endl;
            continue;
        }
        
        std::string name = line.substr(0, tab1);
        std::string offset_str = line.substr(tab1 + 1, tab2 - tab1 - 1);
        std::string length_str = line.substr(tab2 + 1);
        
        // 移除可能的换行符
        if (!length_str.empty() && (length_str.back() == '\n' || length_str.back() == '\r')) {
            length_str.pop_back();
        }
        
        try {
            FFindexEntry entry;
            entry.name = name;
            entry.offset = std::stoll(offset_str);
            entry.length = std::stoll(length_str);
            entries.push_back(entry);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to parse ffindex line: " << line 
                      << " (" << e.what() << ")" << std::endl;
        }
    }
    
    fh.close();
    return entries;
}

// ========== 内存映射 .ffdata 文件 ==========
// 对应 Python: mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
#ifdef _WIN32
    #include <windows.h>
    
    struct MMapDeleter {
        HANDLE hMap;
        void* data;
        size_t size;
        
        void operator()(void*) {
            if (data) UnmapViewOfFile(data);
            if (hMap) CloseHandle(hMap);
        }
    };
    
    std::shared_ptr<void> mmap_file_read(const std::string& ffdata_filename, size_t& file_size) {
        HANDLE hFile = CreateFileA(
            ffdata_filename.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL
        );
        
        if (hFile == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Cannot open file for mmap: " + ffdata_filename);
        }
        
        // 获取文件大小
        DWORD highSize = 0;
        DWORD lowSize = GetFileSize(hFile, &highSize);
        file_size = (static_cast<size_t>(highSize) << 32) | lowSize;
        
        // 创建文件映射
        HANDLE hMap = CreateFileMappingA(
            hFile,
            NULL,
            PAGE_READONLY,
            0,
            0,
            NULL
        );
        
        CloseHandle(hFile);  // 可以关闭文件句柄，映射仍然有效
        
        if (hMap == NULL) {
            throw std::runtime_error("Failed to create file mapping: " + ffdata_filename);
        }
        
        // 映射视图
        void* data = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (data == NULL) {
            CloseHandle(hMap);
            throw std::runtime_error("Failed to map view of file: " + ffdata_filename);
        }
        
        // 使用自定义 deleter 管理生命周期
        auto deleter = new MMapDeleter{hMap, data, file_size};
        return std::shared_ptr<void>(data, [deleter](void*) {
            (*deleter)(nullptr);
            delete deleter;
        });
    }
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
    
    struct MMapDeleter {
        void* data;
        size_t size;
        
        void operator()(void*) {
            if (data) munmap(data, size);
        }
    };
    
    std::shared_ptr<void> mmap_file_read(const std::string& ffdata_filename, size_t& file_size) {
        int fd = open(ffdata_filename.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Cannot open file for mmap: " + ffdata_filename);
        }
        
        // 获取文件大小
        struct stat st;
        if (fstat(fd, &st) == -1) {
            close(fd);
            throw std::runtime_error("Failed to get file size: " + ffdata_filename);
        }
        file_size = st.st_size;
        
        // mmap
        void* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);  // 可以关闭文件描述符，映射仍然有效
        
        if (data == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap file: " + ffdata_filename);
        }
        
        // 使用自定义 deleter 管理生命周期
        auto deleter = new MMapDeleter{data, file_size};
        return std::shared_ptr<void>(data, [deleter](void*) {
            (*deleter)(nullptr);
            delete deleter;
        });
    }
#endif

// ========== 读取 .ffdata 文件（内存映射）==========
// 对应 Python: read_data() -> mmap.mmap(...)
std::shared_ptr<void> read_data(const std::string& ffdata_filename, size_t& file_size) {
    return mmap_file_read(ffdata_filename, file_size);
}

// ========== 创建 FFindexDB 对象 ==========
FFindexDB load_ffdb(const std::string& db_prefix) {
    // db_prefix 例如: "pdb100_2021Mar03/pdb100_2021Mar03"
    // 会读取: db_prefix + "_pdb.ffindex" 和 db_prefix + "_pdb.ffdata"
    
    FFindexDB ffdb;
    ffdb.data_path = db_prefix + "_pdb.ffdata";
    ffdb.index = read_index(db_prefix + "_pdb.ffindex");
    
    size_t file_size = 0;
    ffdb.data = read_data(ffdb.data_path, file_size);
    ffdb.data_size = file_size;
    
    std::cerr << "Loaded FFDB: " << ffdb.index.size() << " entries, "
              << file_size << " bytes data" << std::endl;
    
    return ffdb;
}

// ========== 按名称查找条目 ==========
const FFindexEntry* get_entry_by_name(const std::string& name, const std::vector<FFindexEntry>& index) {
    for (const auto& entry : index) {
        if (name == entry.name) {
            return &entry;
        }
    }
    return nullptr;
}

// ========== 读取条目数据（返回字符串行）==========
std::vector<std::string> read_entry_lines(const FFindexEntry& entry, const void* data_ptr) {
    // Python: data[entry.offset:entry.offset + entry.length - 1].decode("utf-8").split("\n")
    // 注意：Python slice 不包含 end，所以是 length - 1
    
    const char* data = static_cast<const char*>(data_ptr);
    int64_t start = entry.offset;
    int64_t len = entry.length - 1;  // -1 因为 Python slice 不包含 end
    
    if (start < 0 || start + len < 0) {
        throw std::runtime_error("Entry offset out of bounds: " + entry.name);
    }
    
    // 转换为字符串（假设 UTF-8/ASCII）
    std::string entry_data(data + start, static_cast<size_t>(len));
    
    // 按行分割
    std::vector<std::string> lines;
    std::stringstream ss(entry_data);
    std::string line;
    while (std::getline(ss, line)) {
        // 移除可能的 \r (Windows 换行符)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(std::move(line));
    }
    
    return lines;
}

// ========== 读取条目原始数据 ==========
// 返回指向 mmap 数据的指针（零拷贝）
std::pair<const char*, size_t> read_entry_raw(const FFindexEntry& entry, const void* data_ptr) {
    const char* data = static_cast<const char*>(data_ptr);
    int64_t start = entry.offset;
    int64_t len = entry.length - 1;  // -1 因为 Python slice
    
    return {data + start, static_cast<size_t>(len)};
}

// 氨基酸 3 字母 -> 编号
int aa3_to_num(const std::string& aa3) {
    static const std::unordered_map<std::string, int> map = {
        {"ALA", 0}, {"ARG", 1}, {"ASN", 2}, {"ASP", 3}, {"CYS", 4},
        {"GLN", 5}, {"GLU", 6}, {"GLY", 7}, {"HIS", 8}, {"ILE", 9},
        {"LEU", 10}, {"LYS", 11}, {"MET", 12}, {"PHE", 13}, {"PRO", 14},
        {"SER", 15}, {"THR", 16}, {"TRP", 17}, {"TYR", 18}, {"VAL", 19}
    };
    auto it = map.find(aa3);
    return (it != map.end()) ? it->second : 20;  // 20 = unknown/gap
}

// 氨基酸编号 -> 长格式原子名列表（对应 Python 的 util.aa2long）
// 每个氨基酸有 14 个原子：[N, CA, C, O, CB, ...]
const std::vector<std::string>& aa2long(int aa_num) {
    // 索引：0=N, 1=CA, 2=C, 3=O, 4=CB, 5=..., 13=最后一個侧链原子
    static const std::vector<std::vector<std::string>> AA2LONG = {
        // ALA: N, CA, C, O, CB
        {"N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""},
        // ARG: N, CA, C, O, CB, CG, CD, NE, CZ, NH1, NH2
        {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""},
        // ... 其他氨基酸
        // 这里只列出 ALA 和 ARG 作为示例，完整版本需要所有 20 种氨基酸
    };
    
    // TODO: 完整实现所有 20 种氨基酸的原子列表
    // 暂时返回 ALA 的定义
    return AA2LONG[aa_num];
}

// 返回: (xyz, mask, idx_s)
struct ParsePDBResult {
    rfaa::TensorF32 xyz;      // (N_res, 14, 3)
    rfaa::TensorF32 mask;     // (N_res, 14)
    rfaa::TensorF32 idx_s;    // (N_res,) - 残基序号
};
ParsePDBResult parse_pdb_lines(const std::vector<std::string>& lines) {
    // ========== 1. 提取有 CA 原子的残基索引 ==========
    std::vector<int> idx_s_vec;
    for (const auto& l : lines) {
        if (l.size() < 26) continue;
        if (l.substr(0, 4) != "ATOM") continue;
        
        std::string atom_name = l.substr(12, 4);
        // 去除空格
        atom_name.erase(0, atom_name.find_first_not_of(' '));
        atom_name.erase(atom_name.find_last_not_of(' ') + 1);
        
        if (atom_name == "CA") {
            int resNo = std::stoi(l.substr(22, 4));
            idx_s_vec.push_back(resNo);
        }
    }
    
    int N_res = static_cast<int>(idx_s_vec.size());
    if (N_res == 0) {
        // 返回空结果
        ParsePDBResult result;
        result.xyz = rfaa::TensorF32({0, 14, 3});
        result.mask = rfaa::TensorF32({0, 14});
        result.idx_s = rfaa::TensorF32({0});
        return result;
    }
    
    // ========== 2. 初始化坐标数组 (N_res, 14, 3) 为 NaN ==========
    rfaa::TensorF32 xyz({N_res, 14, 3});
    float* xyz_data = xyz.data();
    int64_t total_elements = N_res * 14 * 3;
    
    // 填充 NaN（用 quiet_NaN）
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    for (int64_t i = 0; i < total_elements; i++) {
        xyz_data[i] = nan_val;
    }
    
    // ========== 3. 填充坐标 ==========
    for (const auto& l : lines) {
        if (l.size() < 54) continue;
        if (l.substr(0, 4) != "ATOM") continue;
        
        // 解析 PDB 行
        int resNo = std::stoi(l.substr(22, 4));
        std::string atom_name = l.substr(12, 4);
        atom_name.erase(0, atom_name.find_first_not_of(' '));
        atom_name.erase(atom_name.find_last_not_of(' ') + 1);
        
        std::string aa3 = l.substr(17, 3);  // 氨基酸 3 字母代码
        
        // 找到残基索引
        auto it = std::find(idx_s_vec.begin(), idx_s_vec.end(), resNo);
        if (it == idx_s_vec.end()) continue;
        int idx = static_cast<int>(std::distance(idx_s_vec.begin(), it));
        
        // 找到原子索引
        int aa_num = aa3_to_num(aa3);
        const std::vector<std::string>& atom_list = aa2long(aa_num);
        
        for (int i_atm = 0; i_atm < 14; i_atm++) {
            if (i_atm < static_cast<int>(atom_list.size()) && atom_list[i_atm] == atom_name) {
                // 解析坐标: x=l[30:38], y=l[38:46], z=l[46:54]
                float x = std::stof(l.substr(30, 8));
                float y = std::stof(l.substr(38, 8));
                float z = std::stof(l.substr(46, 8));
                
                // 填入 xyz[idx, i_atm, :]
                int64_t offset = (idx * 14 + i_atm) * 3;
                xyz_data[offset + 0] = x;
                xyz_data[offset + 1] = y;
                xyz_data[offset + 2] = z;
                break;
            }
        }
    }
    
    // ========== 4. 生成掩码 ==========
    rfaa::TensorF32 mask({N_res, 14});
    float* mask_data = mask.data();
    
    for (int i = 0; i < N_res; i++) {
        for (int j = 0; j < 14; j++) {
            int64_t offset = (i * 14 + j) * 3;
            // mask = not isnan(xyz[..., 0])
            mask_data[i * 14 + j] = !std::isnan(xyz_data[offset + 0]) ? 1.0f : 0.0f;
        }
    }
    
    // ========== 5. 将 NaN 替换为 0.0 ==========
    for (int64_t i = 0; i < total_elements; i++) {
        if (std::isnan(xyz_data[i])) {
            xyz_data[i] = 0.0f;
        }
    }
    
    // ========== 6. 构建 idx_s 张量 ==========
    rfaa::TensorF32 idx_s_tensor({N_res});
    float* idx_s_data = idx_s_tensor.data();
    for (int i = 0; i < N_res; i++) {
        idx_s_data[i] = static_cast<float>(idx_s_vec[i]);
    }
    
    // ========== 返回结果 ==========
    ParsePDBResult result;
    result.xyz = std::move(xyz);
    result.mask = std::move(mask);
    result.idx_s = std::move(idx_s_tensor);
    
    return result;
}

namespace rfaa {


// ============================================================================
// ExternalTools 实现
// ============================================================================

std::string ExternalTools::run_hhblits(
    const std::string& sequence,
    const std::string& database_path,
    int n_iter,
    float e_value
) {
    // 写入临时 FASTA 文件
    std::string temp_fasta = "/tmp/rfaa_query.fasta";
    std::ofstream fasta_file(temp_fasta);
    fasta_file << ">query\n" << sequence << "\n";
    fasta_file.close();
    
    // 运行 HHblits
    std::string output_a3m = "/tmp/rfaa_output.a3m";
    std::string cmd = "hhblits -i " + temp_fasta + 
                     " -d " + database_path +
                     " -n " + std::to_string(n_iter) +
                     " -e " + std::to_string(e_value) +
                     " -oa3m " + output_a3m +
                     " -cpu 4";
    
    int ret = system(cmd.c_str());
    if (ret != 0) {
        throw RFAAError("ExternalTools: hhblits failed with return code " + std::to_string(ret));
    }
    
    return output_a3m;
}

std::string ExternalTools::run_hhsearch(
    const std::string& a3m_path,
    const std::string& database_path,
    int n_templates
) {
    // 运行 HHsearch
    std::string output_hhr = "/tmp/rfaa_output.hhr";
    std::string cmd = "hhsearch -i " + a3m_path + 
                     " -d " + database_path +
                     " -o " + output_hhr +
                     " -z " + std::to_string(n_templates);
    
    int ret = system(cmd.c_str());
    if (ret != 0) {
        throw RFAAError("ExternalTools: hhsearch failed with return code " + std::to_string(ret));
    }
    
    return output_hhr;
}

// ============================================================================
// RFAADataLoader 实现
// ============================================================================

RFAADataLoader::RFAADataLoader(
    const std::string& hhblits_db,
    const std::string& hhsearch_db,
    int max_seqs,
    int max_templates,
    int max_length
) : hhblits_db_(hhblits_db),
    hhsearch_db_(hhsearch_db),
    max_seqs_(max_seqs),
    max_templates_(max_templates),
    max_length_(max_length) {
}

ModelInput RFAADataLoader::load(const std::string& sequence) {
    // Step 1: 运行 HHblits
    std::string a3m_path = ExternalTools::run_hhblits(sequence, hhblits_db_);
    
    // Step 2: 运行 HHsearch
    std::string hhr_path = ExternalTools::run_hhsearch(a3m_path, hhsearch_db_);
    
    // Step 3: 加载数据
    return load_from_files(a3m_path, sequence);
}

// ========== 辅助函数：torch.where(qmap[:,1] == nt) ==========
// 返回 qmap[:,1] == nt 的索引
std::vector<int64_t> where_qmap_equals(const rfaa::TensorF32& qmap, int nt) {
    std::vector<int64_t> result;
    int64_t n = qmap.shape().dims[0];  // 总对齐数
    
    for (int64_t i = 0; i < n; i++) {
        // qmap[i, 1] == nt ?
        if (static_cast<int>(qmap.data()[i * 2 + 1]) == nt) {
            result.push_back(i);
        }
    }
    
    return result;
}

// 计算两个向量的叉积: a x b
rfaa::TensorF32 cross(const rfaa::TensorF32& a, const rfaa::TensorF32& b) {
    // a, b: (..., 3)
    rfaa::TensorF32 result(a.shape());
    float* rdata = result.data();
    const float* adata = a.data();
    const float* bdata = b.data();
    int64_t n = a.numel() / 3;
    
    for (int64_t i = 0; i < n; i++) {
        float ax = adata[i*3+0], ay = adata[i*3+1], az = adata[i*3+2];
        float bx = bdata[i*3+0], by = bdata[i*3+1], bz = bdata[i*3+2];
        rdata[i*3+0] = ay*bz - az*by;
        rdata[i*3+1] = az*bx - ax*bz;
        rdata[i*3+2] = ax*by - ay*bx;
    }
    
    return result;
}

// 计算点对之间的距离矩阵: dist[i,j] = ||a[i] - b[j]||
rfaa::TensorF32 get_pair_dist(const rfaa::TensorF32& a, const rfaa::TensorF32& b) {
    // a: (batch, N, 3), b: (batch, M, 3)
    // 返回: (batch, N, M)
    int batch = static_cast<int>(a.shape().dims[0]);
    int N = static_cast<int>(a.shape().dims[1]);
    int M = static_cast<int>(b.shape().dims[1]);
    
    rfaa::TensorF32 dist({batch, N, M});
    float* ddata = dist.data();
    
    for (int b_idx = 0; b_idx < batch; b_idx++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                float dx = a.data()[(b_idx*N + i)*3 + 0] - b.data()[(b_idx*M + j)*3 + 0];
                float dy = a.data()[(b_idx*N + i)*3 + 1] - b.data()[(b_idx*M + j)*3 + 1];
                float dz = a.data()[(b_idx*N + i)*3 + 2] - b.data()[(b_idx*M + j)*3 + 2];
                ddata[(b_idx*N + i)*M + j] = std::sqrt(dx*dx + dy*dy + dz*dz);
            }
        }
    }
    
    return dist;
}

// 计算二面角 (dihedral angle)
rfaa::TensorF32 get_dih(const rfaa::TensorF32& a, const rfaa::TensorF32& b,
                         const rfaa::TensorF32& c, const rfaa::TensorF32& d) {
    // a,b,c,d: (..., 3)
    // 返回: (...,) - 二面角（弧度）
    // 创建形状去掉最后一维
    std::vector<int64_t> dih_dims = a.shape().dims;
    dih_dims.pop_back();
    rfaa::TensorF32 dih(dih_dims, a.device());
    
    int64_t n = a.numel() / 3;
    float* result = dih.data();
    const float* ad = a.data();
    const float* bd = b.data();
    const float* cd = c.data();
    const float* dd = d.data();
    
    for (int64_t i = 0; i < n; i++) {
        // 向量
        float ab_x = bd[i*3+0] - ad[i*3+0];
        float ab_y = bd[i*3+1] - ad[i*3+1];
        float ab_z = bd[i*3+2] - ad[i*3+2];
        
        float bc_x = cd[i*3+0] - bd[i*3+0];
        float bc_y = cd[i*3+1] - bd[i*3+1];
        float bc_z = cd[i*3+2] - bd[i*3+2];
        
        float cd_x = dd[i*3+0] - cd[i*3+0];
        float cd_y = dd[i*3+1] - cd[i*3+1];
        float cd_z = dd[i*3+2] - cd[i*3+2];
        
        // 法向量
        float n1_x = ab_y*bc_z - ab_z*bc_y;
        float n1_y = ab_z*bc_x - ab_x*bc_z;
        float n1_z = ab_x*bc_y - ab_y*bc_x;
        
        float n2_x = bc_y*cd_z - bc_z*cd_y;
        float n2_y = bc_z*cd_x - bc_x*cd_z;
        float n2_z = bc_x*cd_y - bc_y*cd_x;
        
        // 二面角
        float dot = n1_x*n2_x + n1_y*n2_y + n1_z*n2_z;
        float norm1 = std::sqrt(n1_x*n1_x + n1_y*n1_y + n1_z*n1_z);
        float norm2 = std::sqrt(n2_x*n2_x + n2_y*n2_y + n2_z*n2_z);
        
        float cos_angle = dot / (norm1 * norm2 + 1e-6f);
        cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));
        
        float angle = std::acos(cos_angle);
        
        // 符号由 bc 方向决定
        float cross_x = n1_y*n2_z - n1_z*n2_y;
        float cross_y = n1_z*n2_x - n1_x*n2_z;
        float cross_z = n1_x*n2_y - n1_y*n2_x;
        float sign = bc_x*cross_x + bc_y*cross_y + bc_z*cross_z;
        
        result[i] = (sign > 0) ? angle : -angle;
    }
    
    return dih;
}

// 计算平面角 (angle)
rfaa::TensorF32 get_ang(const rfaa::TensorF32& a, const rfaa::TensorF32& b,
                         const rfaa::TensorF32& c) {
    // a,b,c: (..., 3)
    // 返回: (...,) - 平面角（弧度）
    std::vector<int64_t> ang_dims = a.shape().dims;
    ang_dims.pop_back();
    rfaa::TensorF32 ang(ang_dims, a.device());
    
    int64_t n = a.numel() / 3;
    float* result = ang.data();
    const float* ad = a.data();
    const float* bd = b.data();
    const float* cd = c.data();
    
    for (int64_t i = 0; i < n; i++) {
        // 向量 ba, bc
        float ba_x = ad[i*3+0] - bd[i*3+0];
        float ba_y = ad[i*3+1] - bd[i*3+1];
        float ba_z = ad[i*3+2] - bd[i*3+2];
        
        float bc_x = cd[i*3+0] - bd[i*3+0];
        float bc_y = cd[i*3+1] - bd[i*3+1];
        float bc_z = cd[i*3+2] - bd[i*3+2];
        
        // 夹角
        float dot = ba_x*bc_x + ba_y*bc_y + ba_z*bc_z;
        float norm_ba = std::sqrt(ba_x*ba_x + ba_y*ba_y + ba_z*ba_z);
        float norm_bc = std::sqrt(bc_x*bc_x + bc_y*bc_y + bc_z*bc_z);
        
        float cos_angle = dot / (norm_ba * norm_bc + 1e-6f);
        cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));
        
        result[i] = std::acos(cos_angle);
    }
    
    return ang;
}

// ========== xyz_to_c6d 主函数 ==========
// 对应 Python: xyz_to_c6d(xyz, params=PARAMS)
struct C6DResult {
    rfaa::TensorF32 c6d;   // (batch, nres, nres, 4) - [dist, omega, theta, phi]
    rfaa::TensorF32 mask;  // (batch, nres, nres) - 有效位置掩码
};

C6DResult xyz_to_c6d(const rfaa::TensorF32& xyz, float DMAX = 20.0f) {
    // xyz: (batch, nres, 3, 3) - [N, Ca, C] 坐标
    int batch = static_cast<int>(xyz.shape().dims[0]);
    int nres = static_cast<int>(xyz.shape().dims[1]);
    
    // ========== 1. 提取三个锚原子 ==========
    // N = xyz[:,:,0], Ca = xyz[:,:,1], C = xyz[:,:,2]
    rfaa::TensorF32 N({batch, nres, 3});
    rfaa::TensorF32 Ca({batch, nres, 3});
    rfaa::TensorF32 C({batch, nres, 3});
    
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < nres; i++) {
            for (int k = 0; k < 3; k++) {
                N.data()[(b*nres + i)*3 + k] = xyz.data()[((b*nres + i)*3 + 0)*3 + k];
                Ca.data()[(b*nres + i)*3 + k] = xyz.data()[((b*nres + i)*3 + 1)*3 + k];
                C.data()[(b*nres + i)*3 + k] = xyz.data()[((b*nres + i)*3 + 2)*3 + k];
            }
        }
    }
    
    // ========== 2. 重建 Cb 原子 ==========
    // b = Ca - N, c = C - Ca, a = cross(b, c)
    // Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    // (手动循环实现，因为 Tensor 不支持运算符重载)
    rfaa::TensorF32 Cb({batch, nres, 3});
    for (int b_idx = 0; b_idx < batch; b_idx++) {
        for (int i = 0; i < nres; i++) {
            int offset = (b_idx * nres + i) * 3;
            
            // b = Ca - N
            float bx = Ca.data()[offset+0] - N.data()[offset+0];
            float by = Ca.data()[offset+1] - N.data()[offset+1];
            float bz = Ca.data()[offset+2] - N.data()[offset+2];
            
            // c = C - Ca
            float cx = C.data()[offset+0] - Ca.data()[offset+0];
            float cy = C.data()[offset+1] - Ca.data()[offset+1];
            float cz = C.data()[offset+2] - Ca.data()[offset+2];
            
            // a = cross(b, c)
            float ax = by*cz - bz*cy;
            float ay = bz*cx - bx*cz;
            float az = bx*cy - by*cx;
            
            // Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
            Cb.data()[offset+0] = -0.58273431f*ax + 0.56802827f*bx - 0.54067466f*cx + Ca.data()[offset+0];
            Cb.data()[offset+1] = -0.58273431f*ay + 0.56802827f*by - 0.54067466f*cy + Ca.data()[offset+1];
            Cb.data()[offset+2] = -0.58273431f*az + 0.56802827f*bz - 0.54067466f*cz + Ca.data()[offset+2];
        }
    }
    
    // ========== 3. 计算 6D 坐标 ==========
    rfaa::TensorF32 c6d({batch, nres, nres, 4});
    c6d.zero_();
    
    // dist = get_pair_dist(Cb, Cb)
    rfaa::TensorF32 dist = get_pair_dist(Cb, Cb);
    
    // dist[isnan(dist)] = 999.9
    for (int i = 0; i < dist.numel(); i++) {
        if (std::isnan(dist.data()[i])) {
            dist.data()[i] = 999.9f;
        }
    }
    
    // c6d[...,0] = dist + 999.9*eye(nres)
    for (int b_idx = 0; b_idx < batch; b_idx++) {
        for (int i = 0; i < nres; i++) {
            for (int j = 0; j < nres; j++) {
                float val = dist.data()[(b_idx*nres + i)*nres + j];
                if (i == j) val += 999.9f;
                c6d.data()[((b_idx*nres + i)*nres + j)*4 + 0] = val;
            }
        }
    }
    
    // ========== 4. 找到 dist < DMAX 的位置 ==========
    // b,i,j = where(c6d[...,0] < DMAX)
    std::vector<std::tuple<int,int,int>> valid_positions;
    for (int b_idx = 0; b_idx < batch; b_idx++) {
        for (int i = 0; i < nres; i++) {
            for (int j = 0; j < nres; j++) {
                float d = c6d.data()[((b_idx*nres + i)*nres + j)*4 + 0];
                if (d < DMAX) {
                    valid_positions.emplace_back(b_idx, i, j);
                }
            }
        }
    }
    
    // ========== 5. 计算角度 ==========
    if (!valid_positions.empty()) {
        // 为有效位置创建张量
        int n_valid = static_cast<int>(valid_positions.size());
        
        // 创建索引张量
        rfaa::TensorF32 b_idx_tensor({n_valid});
        rfaa::TensorF32 i_tensor({n_valid});
        rfaa::TensorF32 j_tensor({n_valid});
        
        for (int idx = 0; idx < n_valid; idx++) {
            auto [b_val, i_val, j_val] = valid_positions[idx];
            b_idx_tensor.data()[idx] = static_cast<float>(b_val);
            i_tensor.data()[idx] = static_cast<float>(i_val);
            j_tensor.data()[idx] = static_cast<float>(j_val);
        }
        
        // 提取有效位置的坐标
        // Ca[b,i], Cb[b,i], Cb[b,j], Ca[b,j]
        rfaa::TensorF32 Ca_bi({n_valid, 3});
        rfaa::TensorF32 Cb_bi({n_valid, 3});
        rfaa::TensorF32 Cb_bj({n_valid, 3});
        rfaa::TensorF32 Ca_bj({n_valid, 3});
        rfaa::TensorF32 N_bi({n_valid, 3});
        
        for (int idx = 0; idx < n_valid; idx++) {
            auto [b_val, i_val, j_val] = valid_positions[idx];
            int b_idx = b_val;
            int i_val_int = i_val;
            int j_val_int = j_val;
            
            // Ca[b,i]
            std::memcpy(Ca_bi.data() + idx*3, 
                       Ca.data() + (b_idx*nres + i_val_int)*3, 
                       3*sizeof(float));
            // Cb[b,i]
            std::memcpy(Cb_bi.data() + idx*3, 
                       Cb.data() + (b_idx*nres + i_val_int)*3, 
                       3*sizeof(float));
            // Cb[b,j]
            std::memcpy(Cb_bj.data() + idx*3, 
                       Cb.data() + (b_idx*nres + j_val_int)*3, 
                       3*sizeof(float));
            // Ca[b,j]
            std::memcpy(Ca_bj.data() + idx*3, 
                       Ca.data() + (b_idx*nres + j_val_int)*3, 
                       3*sizeof(float));
            // N[b,i]
            std::memcpy(N_bi.data() + idx*3, 
                       N.data() + (b_idx*nres + i_val_int)*3, 
                       3*sizeof(float));
        }
        
        // 计算角度
        rfaa::TensorF32 omega = get_dih(Ca_bi, Cb_bi, Cb_bj, Ca_bj);  // (n_valid,)
        rfaa::TensorF32 theta = get_dih(N_bi, Ca_bi, Cb_bi, Cb_bj);    // (n_valid,)
        rfaa::TensorF32 phi = get_ang(Ca_bi, Cb_bi, Cb_bj);            // (n_valid,)
        
        // 填入 c6d
        for (int idx = 0; idx < n_valid; idx++) {
            auto [b_val, i_val, j_val] = valid_positions[idx];
            int offset = ((b_val*nres + i_val)*nres + j_val)*4;
            c6d.data()[offset + 1] = omega.data()[idx];  // omega
            c6d.data()[offset + 2] = theta.data()[idx];  // theta
            c6d.data()[offset + 3] = phi.data()[idx];    // phi
        }
    }
    
    // ========== 6. 修复长距离 ==========
    // c6d[...,0][c6d[...,0] >= DMAX] = 999.9
    for (int i = 0; i < c6d.numel(); i += 4) {
        if (c6d.data()[i] >= DMAX) {
            c6d.data()[i] = 999.9f;
        }
    }
    
    // ========== 7. 创建掩码 ==========
    rfaa::TensorF32 mask({batch, nres, nres});
    mask.zero_();
    for (auto [b_val, i_val, j_val] : valid_positions) {
        mask.data()[(b_val*nres + i_val)*nres + j_val] = 1.0f;
    }
    
    // ========== 返回 ==========
    C6DResult result;
    result.c6d = std::move(c6d);
    result.mask = std::move(mask);
    
    return result;
}

// 输入:
//   xyz_t: (B, T, L, 3, 3) - 模板坐标
//   t0d: (B, T, 3) - 模板级特征
// 输出:
//   t2d: (B, T, L, L, 10) - 模板 2D 特征
// old version of RosettaFold
// d_t2d = 10 is old version
// RF2 d_t2d = 68
rfaa::TensorF32 xyz_to_t2d(const rfaa::TensorF32& xyz_t, 
                            const rfaa::TensorF32& t0d, 
                            float DMAX = 20.0f) {
    // ========== 参数检查 ==========
    if (xyz_t.shape().dims.size() != 5 || xyz_t.shape().dims[3] != 3 || xyz_t.shape().dims[4] != 3) {
        throw std::runtime_error("xyz_t must have shape (B, T, L, 3, 3)");
    }
    if (t0d.shape().dims.size() != 3 || t0d.shape().dims[2] != 3) {
        throw std::runtime_error("t0d must have shape (B, T, 3)");
    }
    
    int B = static_cast<int>(xyz_t.shape().dims[0]);
    int T = static_cast<int>(xyz_t.shape().dims[1]);
    int L = static_cast<int>(xyz_t.shape().dims[2]);
    
    // ========== 1. 转换为 6D 坐标 ==========
    // xyz_t.view(B*T, L, 3, 3)
    rfaa::TensorF32 xyz_t_reshaped({B * T, L, 3, 3});
    std::memcpy(xyz_t_reshaped.data(), xyz_t.data(), xyz_t.numel() * sizeof(float));
    
    // 调用 xyz_to_c6d
    C6DResult c6d_result = xyz_to_c6d(xyz_t_reshaped, DMAX);
    // c6d.view(B, T, L, L, 4)
    // mask.view(B, T, L, L)
    rfaa::TensorF32 c6d({B, T, L, L, 4});
    rfaa::TensorF32 mask({B, T, L, L});
    std::memcpy(c6d.data(), c6d_result.c6d.data(), c6d_result.c6d.numel() * sizeof(float));
    std::memcpy(mask.data(), c6d_result.mask.data(), c6d_result.mask.numel() * sizeof(float));
    
    // ========== 2. 归一化距离 ==========
    // dist = c6d[...,:1] * mask / DMAX  -> (B, T, L, L, 1)
    rfaa::TensorF32 dist({B, T, L, L, 1});
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < L; j++) {
                    float c6d_val = c6d.data()[((b*T + t)*L + i)*L*4 + j*4 + 0];
                    float mask_val = mask.data()[((b*T + t)*L + i)*L + j];
                    dist.data()[((b*T + t)*L + i)*L*1 + j*1 + 0] = c6d_val * mask_val / DMAX;
                }
            }
        }
    }
    
    // ========== 3. 编码方向 (sin/cos) ==========
    // orien = cat(sin(c6d[...,1:]), cos(c6d[...,1:])) * mask  -> (B, T, L, L, 6)
    // c6d[...,1:] = omega, theta, phi (3个角度)
    // sin + cos = 6维
    rfaa::TensorF32 orien({B, T, L, L, 6});
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < L; j++) {
                    int c6d_offset = ((b*T + t)*L + i)*L*4 + j*4;
                    int orien_offset = ((b*T + t)*L + i)*L*6 + j*6;
                    float mask_val = mask.data()[((b*T + t)*L + i)*L + j];
                    
                    for (int k = 0; k < 3; k++) {  // omega=1, theta=2, phi=3
                        float angle = c6d.data()[c6d_offset + 1 + k];
                        orien.data()[orien_offset + k*2 + 0] = std::sin(angle) * mask_val;  // sin
                        orien.data()[orien_offset + k*2 + 1] = std::cos(angle) * mask_val;  // cos
                    }
                }
            }
        }
    }
    
    // ========== 4. 扩展 t0d ==========
    // t0d: (B, T, 3) -> (B, T, L, L, 3)
    // t0d.unsqueeze(2).unsqueeze(3).expand(-1,-1,L,L,-1)
    rfaa::TensorF32 t0d_expanded({B, T, L, L, 3});
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < L; j++) {
                    for (int k = 0; k < 3; k++) {
                        t0d_expanded.data()[((b*T + t)*L + i)*L*3 + j*3 + k] = 
                            t0d.data()[(b*T + t)*3 + k];
                    }
                }
            }
        }
    }
    
    // ========== 5. 拼接 ==========
    // t2d = cat(dist, orien, t0d, dim=-1)  -> (B, T, L, L, 10)
    // 10 = 1 (dist) + 6 (orien) + 3 (t0d)
    rfaa::TensorF32 t2d({B, T, L, L, 10});
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < L; j++) {
                    int dst_offset = ((b*T + t)*L + i)*L*10 + j*10;
                    
                    // dist: 1维
                    t2d.data()[dst_offset + 0] = dist.data()[((b*T + t)*L + i)*L*1 + j*1 + 0];
                    
                    // orien: 6维
                    for (int k = 0; k < 6; k++) {
                        t2d.data()[dst_offset + 1 + k] = orien.data()[((b*T + t)*L + i)*L*6 + j*6 + k];
                    }
                    
                    // t0d: 3维
                    for (int k = 0; k < 3; k++) {
                        t2d.data()[dst_offset + 7 + k] = t0d_expanded.data()[((b*T + t)*L + i)*L*3 + j*3 + k];
                    }
                }
            }
        }
    }
    
    return t2d;
}

//using TensorI64 = Tensor<int64_t>;
std::vector<float> linspace(float start, float end, int n) {
    std::vector<float> result;
    result.reserve(n);
    for (int i = 0; i < n; i++) {
        result.push_back(start + static_cast<float>(i) * (end - start) / static_cast<float>(n - 1));
    }
    return result;
}

// ========== 参数结构体 ==========
struct DistParams {
    float DMIN = 1.0f;      // 最小距离
    float DMID = 4.0f;     // 中间距离
    float DMAX = 20.0f;     // 最大距离
    int DBINS1 = 30;        // 近距离 bin 数
    int DBINS2 = 30;        // 远距离 bin 数
    
    int num_classes() const { return DBINS1 + DBINS2 + 1; }
};

// ========== dist_to_bins: 距离离散化为 bin 索引 ==========
// 对应 Python: dist_to_bins(dist, params=PARAMS)
// 输入: dist (...,) - 距离矩阵
// 输出: db (...,) - bin 索引 (long/int64)
rfaa::TensorI64 dist_to_bins(const rfaa::TensorF32& dist, const DistParams& params = DistParams()) {
    // ========== 1. 处理 NaN ==========
    rfaa::TensorF32 dist_clean(dist.shape(), dist.device());
    dist_clean.copy_from(dist);
    for (int64_t i = 0; i < dist_clean.numel(); i++) {
        if (std::isnan(dist_clean.data()[i])) {
            dist_clean.data()[i] = 999.9f;
        }
    }
    
    // ========== 2. 计算 bin 边界 ==========
    // dstep1 = (DMID - DMIN) / DBINS1
    float dstep1 = (params.DMID - params.DMIN) / static_cast<float>(params.DBINS1);
    // dstep2 = (DMAX - DMID) / DBINS2
    float dstep2 = (params.DMAX - params.DMID) / static_cast<float>(params.DBINS2);
    
    // dbins = cat(linspace(DMIN+dstep1, DMID, DBINS1), linspace(DMID+dstep2, DMAX, DBINS2))
    std::vector<float> dbins;
    dbins.reserve(params.DBINS1 + params.DBINS2);
    
    // 第一段: linspace(DMIN+dstep1, DMID, DBINS1)
    for (int i = 0; i < params.DBINS1; i++) {
        float val = params.DMIN + dstep1 + static_cast<float>(i) * (params.DMID - params.DMIN - dstep1) / static_cast<float>(params.DBINS1 - 1);
        // 更准确: linspace(start, end, n) = start + i * (end - start) / (n - 1)
        val = params.DMIN + dstep1 + static_cast<float>(i) * (params.DMID - (params.DMIN + dstep1)) / static_cast<float>(params.DBINS1 - 1);
        dbins.push_back(val);
    }
    
    // 第二段: linspace(DMID+dstep2, DMAX, DBINS2)
    for (int i = 0; i < params.DBINS2; i++) {
        float val = params.DMID + dstep2 + static_cast<float>(i) * (params.DMAX - (params.DMID + dstep2)) / static_cast<float>(params.DBINS2 - 1);
        dbins.push_back(val);
    }
    
    // ========== 3. bucketize: 离散化 ==========
    // torch.bucketize(dist, dbins) -> 返回 dist[i] 应该插入 dbins 的位置
    rfaa::TensorI64 db(dist.shape());  // 假设 TensorI64 是 int64_t 张量
    
    for (int64_t i = 0; i < dist.numel(); i++) {
        float d = dist_clean.data()[i];
        
        // 找到第一个 >= d 的位置 (upper_bound)
        auto it = std::upper_bound(dbins.begin(), dbins.end(), d);
        int64_t idx = std::distance(dbins.begin(), it);
        
        db.data()[i] = idx;
    }
    
    return db;
}

// ========== dist_to_onehot: 距离转 one-hot ==========
// 对应 Python: dist_to_onehot(dist, params=PARAMS)
// 输入: dist (...,) - 距离矩阵
// 输出: onehot (..., num_classes) - one-hot 编码
rfaa::TensorF32 dist_to_onehot(const rfaa::TensorF32& dist, const DistParams& params = DistParams()) {
    // ========== 1. 离散化 ==========
    rfaa::TensorI64 db = dist_to_bins(dist, params);
    int num_classes = params.num_classes();
    
    // ========== 2. One-hot 编码 ==========
    // 输出形状: dist.shape + (num_classes,)
    std::vector<int64_t> out_dims = dist.shape().dims;
    out_dims.push_back(num_classes);
    
    rfaa::TensorF32 onehot(out_dims);
    onehot.zero_();
    
    // 计算总元素数（排除最后一维）
    int64_t total_prefix = dist.numel();
    
    for (int64_t i = 0; i < total_prefix; i++) {
        int64_t class_idx = db.data()[i];
        
        // 边界检查
        if (class_idx < 0 || class_idx >= num_classes) {
            // 超出范围，可能是 999.9 距离 -> 放到最后一个 bin
            class_idx = num_classes - 1;
        }
        
        // onehot[i, class_idx] = 1.0
        onehot.data()[i * num_classes + class_idx] = 1.0f;
    }
    
    return onehot;
}

// ========== 辅助函数：单个距离离散化 ==========
int dist_to_bin_single(float dist, const DistParams& params) {
    // 处理 NaN
    if (std::isnan(dist)) dist = 999.9f;
    
    // 计算 bin 边界（与 dist_to_bins 相同逻辑）
    float dstep1 = (params.DMID - params.DMIN) / static_cast<float>(params.DBINS1);
    float dstep2 = (params.DMAX - params.DMID) / static_cast<float>(params.DBINS2);
    
    // 找到 bin 索引
    if (dist <= params.DMIN + dstep1) {
        return 0;
    } else if (dist <= params.DMID) {
        float bin_width = (params.DMID - (params.DMIN + dstep1)) / static_cast<float>(params.DBINS1 - 1);
        return static_cast<int>((dist - (params.DMIN + dstep1)) / bin_width);
    } else if (dist <= params.DMAX) {
        float bin_width = (params.DMAX - (params.DMID + dstep2)) / static_cast<float>(params.DBINS2 - 1);
        return params.DBINS1 + static_cast<int>((dist - (params.DMID + dstep2)) / bin_width);
    } else {
        return params.num_classes() - 1;  // 最后一个 bin
    }
}

// new version of RF2 t2d = 68
// 输入:
//   xyz_t: (B, T, L, 3, 3) - 模板坐标
//   mask: (B, T, L, L) - 有效对掩码
// 输出:
//   t2d: (B, T, L, L, D) - 模板 2D 特征 (D = num_classes + 6 + 1) = 61 + 7 = 68
rfaa::TensorF32 xyz_to_t2d(
    const rfaa::TensorF32& xyz_t,
    const rfaa::TensorF32& mask,
    const DistParams& params = DistParams()
) {
    // ========== 参数检查 ==========
    if (xyz_t.shape().dims.size() != 5 || xyz_t.shape().dims[3] != 3 || xyz_t.shape().dims[4] != 3) {
        throw std::runtime_error("xyz_t must have shape (B, T, L, 3, 3)");
    }
    if (mask.shape().dims.size() != 4) {
        throw std::runtime_error("mask must have shape (B, T, L, L)");
    }
    
    int B = static_cast<int>(xyz_t.shape().dims[0]);
    int T = static_cast<int>(xyz_t.shape().dims[1]);
    int L = static_cast<int>(xyz_t.shape().dims[2]);
    int num_classes = params.num_classes();  // DBINS1 + DBINS2 + 1 = 61
    
    // ========== 1. 计算 6D 坐标 ==========
    // xyz_t[:,:,:,:3].view(B*T, L, 3, 3)
    rfaa::TensorF32 xyz_reshaped({B*T, L, 3, 3});
    std::memcpy(xyz_reshaped.data(), xyz_t.data(), xyz_t.numel() * sizeof(float));
    
    //auto [c6d_flat, mask_flat] = xyz_to_c6d_simple(xyz_reshaped, params);
    C6DResult xyz_to_c6d_result = xyz_to_c6d(xyz_reshaped, params.DMAX);
    
    // c6d.view(B, T, L, L, 4)
    rfaa::TensorF32 c6d({B, T, L, L, 4});
    std::memcpy(c6d.data(), xyz_to_c6d_result.c6d.data(), xyz_to_c6d_result.c6d.numel() * sizeof(float));
    
    // ========== 2. 距离 one-hot 编码 ==========
    // mask[...,None] -> (B, T, L, L, 1)
    // dist = dist_to_onehot(c6d[...,0]) * mask
    rfaa::TensorF32 dist_onehot({B, T, L, L, num_classes});
    dist_onehot.zero_();
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < L; j++) {
                    float dist_val = c6d.data()[((b*T + t)*L + i)*L*4 + j*4 + 0];
                    float mask_val = mask.data()[(b*T + t)*L*L + i*L + j];
                    
                    // ?
                    // 离散化距离
                    int class_idx = dist_to_bin_single(dist_val, params);
                    class_idx = std::max(0, std::min(num_classes-1, class_idx));
                    
                    // one-hot
                    dist_onehot.data()[((b*T + t)*L + i)*L*num_classes + j*num_classes + class_idx] 
                    = dist_val * mask_val;
                    // ? dist = one hot float * mask
                }
            }
        }
    }
    
    // ========== 3. 方向编码 (sin/cos) ==========
    // orien = cat(sin(c6d[...,1:]), cos(c6d[...,1:])) * mask
    // c6d[...,1:] = omega, theta, phi (3个角度) -> sin+cos = 6维
    rfaa::TensorF32 orien({B, T, L, L, 6});
    orien.zero_();
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < L; j++) {
                    float mask_val = mask.data()[(b*T + t)*L*L + i*L + j];
                    int c6d_offset = ((b*T + t)*L + i)*L*4 + j*4;
                    
                    for (int k = 0; k < 3; k++) {  // omega=1, theta=2, phi=3
                        float angle = c6d.data()[c6d_offset + 1 + k];
                        orien.data()[((b*T + t)*L + i)*L*6 + j*6 + k*2 + 0] = std::sin(angle) * mask_val;
                        orien.data()[((b*T + t)*L + i)*L*6 + j*6 + k*2 + 1] = std::cos(angle) * mask_val;
                    }
                }
            }
        }
    }
    
    // ========== 4. 扩展 mask 到最后维 ==========
    // mask 已经是 (B,T,L,L)，需要扩展为 (B,T,L,L,1)
    rfaa::TensorF32 mask_expanded({B, T, L, L, 1});
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < L; j++) {
                    mask_expanded.data()[((b*T + t)*L + i)*L*1 + j*1 + 0] = 
                        mask.data()[(b*T + t)*L*L + i*L + j];
                }
            }
        }
    }
    
    // ========== 5. 拼接 ==========
    // t2d = cat(dist, orien, mask, dim=-1)
    // 维度: num_classes + 6 + 1 = num_classes + 7
    int D = num_classes + 6 + 1;
    rfaa::TensorF32 t2d({B, T, L, L, D});
    t2d.zero_();
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < L; j++) {
                    int dst_offset = ((b*T + t)*L + i)*L*D + j*D;
                    int src_offset_dist = ((b*T + t)*L + i)*L*num_classes + j*num_classes;
                    int src_offset_orien = ((b*T + t)*L + i)*L*6 + j*6;
                    
                    // dist: num_classes 维
                    for (int k = 0; k < num_classes; k++) {
                        t2d.data()[dst_offset + k] = dist_onehot.data()[src_offset_dist + k];
                    }
                    
                    // orien: 6 维
                    for (int k = 0; k < 6; k++) {
                        t2d.data()[dst_offset + num_classes + k] = orien.data()[src_offset_orien + k];
                    }
                    
                    // mask: 1 维
                    t2d.data()[dst_offset + num_classes + 6] = mask_expanded.data()[((b*T + t)*L + i)*L*1 + j*1 + 0];
                }
            }
        }
    }
    
    return t2d;
}

// ============================================================================
// read_fasta_first_sequence — 从 FASTA 文件读取第一条查询序列
// ============================================================================
std::string RFAADataLoader::read_fasta_first_sequence(const std::string& fasta_path) {
    std::ifstream file(fasta_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open FASTA file: " + fasta_path);
    }

    std::string sequence;
    sequence.reserve(4096);
    bool in_first_record = false;   // 是否已进入第一条序列记录
    bool first_done      = false;   // 第一条序列是否已经读取完毕

    std::string line;
    while (std::getline(file, line)) {
        // 去掉尾部 \r (Windows 换行)
        if (!line.empty() && line.back() == '\r') line.pop_back();

        if (line.empty()) continue;

        if (line[0] == '>') {
            if (first_done) break;        // 遇到第二条记录的头, 结束
            in_first_record = true;       // 开始第一条记录
            continue;
        }

        // 注释行 (FASTA 允许以 ';' 开头)
        if (line[0] == ';') continue;

        if (in_first_record) {
            sequence += line;
        }
        // 头部之前出现的杂散序列行 (严格 FASTA 中不应出现) 忽略
    }

    if (sequence.empty()) {
        throw std::runtime_error("No sequence found in FASTA file: " + fasta_path);
    }

    return sequence;
}

// ============================================================================
// parse_csv_true_coords — 从 CSV mapping 文件解析真实坐标 (ground truth)
// ============================================================================
// CSV 格式 (由 Python 脚本生成):
//   FASTA_Pos,FASTA_AA,PDB_ResNum,PDB_AA,CA_X,CA_Y,CA_Z,All_Atoms_Coords
//   All_Atoms_Coords 示例: "N:(101.60,38.53,-1.96) | CA:(103.06,38.51,-2.16) | C:(...) | ..."
//
// 兼容策略 (不同 PDB 残基原子存储情况不一):
//   1. 优先从 All_Atoms_Coords 提取 N / CA / C 三个骨架原子坐标
//   2. CA 缺失时回退到 CA_X / CA_Y / CA_Z 列 (每个残基必有)
//   3. N 或 C 缺失时用 CA 坐标回退 (避免零坐标导致 FAPE 异常)
// 返回: (B=1, L, 3, 3) — [N, CA, C] × [x, y, z]
TensorF32 RFAADataLoader::parse_csv_true_coords(
    const std::string& csv_path,
    int expected_L)
{
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV mapping file: " + csv_path);
    }

    // ========== 1. 读取所有数据行 (跳过 header) ==========
    struct Atom3D { std::string name; float xyz[3]; };
    struct CsvRow {
        float ca[3];
        std::vector<Atom3D> atoms; // name -> xyz
    };
    std::vector<CsvRow> rows;

    std::string line;
    bool is_header = true;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        // 去除 \r (Windows)
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // ===== 解析一行 (字段可能被引号包裹, 含逗号/空格) =====
        std::vector<std::string> fields;
        std::string cur;
        bool in_quotes = false;
        for (size_t i = 0; i < line.size(); i++) {
            char c = line[i];
            if (c == '"') {
                in_quotes = !in_quotes;
            } else if (c == ',' && !in_quotes) {
                fields.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        fields.push_back(cur);

        // 跳过 header
        if (is_header) {
            is_header = false;
            continue;
        }
        if (fields.size() < 8) continue;  // 缺列则跳过

        CsvRow row;
        row.ca[0] = row.ca[1] = row.ca[2] = 0.0f;

        // CA_X, CA_Y, CA_Z (字段 4,5,6)
        try {
            row.ca[0] = std::stof(fields[4]);
            row.ca[1] = std::stof(fields[5]);
            row.ca[2] = std::stof(fields[6]);
        } catch (...) {
            row.ca[0] = row.ca[1] = row.ca[2] = 0.0f;
        }

        // All_Atoms_Coords (字段 7): "N:(x,y,z) | CA:(x,y,z) | ..."
        const std::string& all_atoms = fields[7];
        // 按 " | " 分割原子条目
        size_t pos = 0;
        while (pos < all_atoms.size()) {
            // 找到下一个 "("
            size_t open = all_atoms.find('(', pos);
            if (open == std::string::npos) break;
            // 原子名 = 从 pos 到 '(' 去空格
            std::string name = all_atoms.substr(pos, open - pos);
            // 去除前后空白
            name.erase(0, name.find_first_not_of(" \t"));
            name.erase(name.find_last_not_of(" \t") + 1);
            // 找到对应的 ')'
            size_t close = all_atoms.find(')', open);
            if (close == std::string::npos) break;
            std::string coord_str = all_atoms.substr(open + 1, close - open - 1);
            // 解析 x,y,z
            float x = 0, y = 0, z = 0;
            int cnt = sscanf(coord_str.c_str(), "%f,%f,%f", &x, &y, &z);
            if (cnt == 3) {
                Atom3D a;
                a.name = name;
                a.xyz[0] = x; a.xyz[1] = y; a.xyz[2] = z;
                row.atoms.push_back(std::move(a));
            }
            pos = close + 1;
        }

        rows.push_back(std::move(row));
    }

    int L = (expected_L > 0) ? expected_L : static_cast<int>(rows.size());
    if (L <= 0) {
        throw std::runtime_error("CSV mapping file contains no valid rows: " + csv_path);
    }

    // ========== 2. 组装 coords (B=1, L, 3, 3) ==========
    TensorF32 coords({1, L, 3, 3}, Device::CPU);
    float* data = coords.data();

    auto find_atom = [&](const CsvRow& row, const char* name) -> const float* {
        for (const auto& a : row.atoms) {
            if (a.name == name) return a.xyz;
        }
        return nullptr;
    };

    for (int l = 0; l < L; l++) {
        const CsvRow& row = (l < static_cast<int>(rows.size())) ? rows[l] : rows.back();

        const float* pN  = find_atom(row, "N");
        const float* pCA = find_atom(row, "CA");
        const float* pC  = find_atom(row, "C");

        // CA 回退到 CA_X/Y/Z 列
        float ca_fallback[3] = {row.ca[0], row.ca[1], row.ca[2]};
        if (!pCA) pCA = ca_fallback;
        if (!pN)  pN  = pCA;  // N 缺失 → CA
        if (!pC)  pC  = pCA;  // C 缺失 → CA

        int64_t base = (l * 3) * 3;  // (l, atom, coord)
        for (int c = 0; c < 3; c++) {
            data[base + 0*3 + c] = pN[c];    // N
            data[base + 1*3 + c] = pCA[c];   // CA
            data[base + 2*3 + c] = pC[c];    // C
        }
    }

    return coords;
}

// ============================================================================
// list_csv_mapping_files — 列出目录下所有 *_mapping_results.csv
// ============================================================================
std::vector<std::string> RFAADataLoader::list_csv_mapping_files(const std::string& dir) {
    namespace fs = std::filesystem;
    std::vector<std::string> files;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return files;  // 非目录, 返回空 (由调用方决定是否报错)
    }
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("_mapping_results.csv") != std::string::npos ||
            lower.find("_mapping.csv") != std::string::npos) {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// ============================================================================
// collect_ground_truth_pdb_ids — 从 ground truth 目录收集 PDB id (小写)
// 用于模板过滤: 扫描 *_mapping_results.csv 前缀与 *.pdb/*.cif 文件名。
// 例: "1c26_mapping_results.csv" → "1c26"; "1C26.pdb" → "1c26"
// ============================================================================
std::set<std::string> RFAADataLoader::collect_ground_truth_pdb_ids(const std::string& dir) {
    namespace fs = std::filesystem;
    std::set<std::string> ids;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return ids;

    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        std::string stem = entry.path().stem().string();  // 去掉扩展名
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        // 优先从 *_mapping_results.csv / *_mapping.csv 提取 pdb 前缀
        if (lower.find("_mapping_results.csv") != std::string::npos ||
            lower.find("_mapping.csv") != std::string::npos) {
            size_t pos = lower.find("_mapping");
            std::string pdb = stem.substr(0, pos);
            if (!pdb.empty()) ids.insert(pdb);
        }
        // 其次从 *.pdb / *.cif 文件名提取
        else if (lower.find(".pdb") != std::string::npos ||
                 lower.find(".cif") != std::string::npos) {
            std::string pdb = stem;
            // 若带链后缀 (如 1abc_A), 取 pdb 前缀
            auto us = pdb.find('_');
            if (us != std::string::npos) pdb = pdb.substr(0, us);
            std::transform(pdb.begin(), pdb.end(), pdb.begin(), ::tolower);
            if (!pdb.empty()) ids.insert(pdb);
        }
    }
    return ids;
}

// ============================================================================
// list_structure_files — 列出目录下所有结构文件 (*.cif / *.pdb)
// 返回: {完整路径, 小写扩展名}
// ============================================================================
std::vector<std::pair<std::string, std::string>> RFAADataLoader::list_structure_files(
    const std::string& dir) {
    namespace fs = std::filesystem;
    std::vector<std::pair<std::string, std::string>> files;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return files;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".cif" || ext == ".pdb") {
            files.emplace_back(entry.path().string(), ext);
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// ============================================================================
// parse_csv_true_coords_multi — 合并多个 CSV mapping 为单个 (1, L, 3, 3) 真实坐标
//
// 背景: 一个 uniprot id (如 P04637) 的查询序列可能被多个 PDB 结构域覆盖,
//       每个 CSV 对应一个 PDB, 记录该 PDB 结构域所覆盖的查询残基坐标。
//       FASTA_Pos 是 1 索引的查询序列位置。
//
// 合并规则:
//   1. 遍历所有 CSV, 将每个残基的 [N, CA, C] 坐标放入其 FASTA_Pos-1 处。
//   2. 重叠残基 (多个 CSV 覆盖同一位置): 取首个覆盖且坐标非全零的 CSV。
//   3. 未被任何 CSV 覆盖的残基坐标保持 0 (供 prepare_ca_mask 排除, 不参与监督)。
// ============================================================================
TensorF32 RFAADataLoader::parse_csv_true_coords_multi(
    const std::vector<std::string>& csv_paths,
    int L)
{
    if (L <= 0) {
        throw std::runtime_error("parse_csv_true_coords_multi: invalid L=" + std::to_string(L));
    }
    // (L, 3, 3): [res][atom 0=N,1=CA,2=C][xyz]
    TensorF32 coords({1, L, 3, 3}, Device::CPU);
    coords.zero_();
    std::vector<char> filled(L, 0);  // 该残基是否已被有效坐标填充

    for (const auto& csv_path : csv_paths) {
        std::ifstream file(csv_path);
        if (!file.is_open()) {
            std::cerr << "[WARN] skip unreadable CSV: " << csv_path << std::endl;
            continue;
        }

        struct Atom3D { std::string name; float xyz[3]; };
        std::string line;
        bool is_header = true;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (line.back() == '\r') line.pop_back();

            // ===== CSV 字段解析 (兼容引号包裹) =====
            std::vector<std::string> fields;
            std::string cur;
            bool in_quotes = false;
            for (size_t i = 0; i < line.size(); i++) {
                char c = line[i];
                if (c == '"') {
                    in_quotes = !in_quotes;
                } else if (c == ',' && !in_quotes) {
                    fields.push_back(cur);
                    cur.clear();
                } else {
                    cur.push_back(c);
                }
            }
            fields.push_back(cur);
            if (is_header) { is_header = false; continue; }
            if (fields.size() < 8) continue;

            // FASTA_Pos (1 索引)
            int fasta_pos = 0;
            try { fasta_pos = std::stoi(fields[0]); } catch (...) { continue; }
            if (fasta_pos < 1 || fasta_pos > L) continue;  // 越界跳过
            int l = fasta_pos - 1;
            if (filled[l]) continue;  // 已被更早 CSV 覆盖, 跳过

            // CA 回退列 (字段 4,5,6)
            float ca_fallback[3] = {0, 0, 0};
            try {
                ca_fallback[0] = std::stof(fields[4]);
                ca_fallback[1] = std::stof(fields[5]);
                ca_fallback[2] = std::stof(fields[6]);
            } catch (...) { /* 保持 0 */ }

            // All_Atoms_Coords (字段 7): 提取 N / CA / C
            float pN[3] = {0,0,0}, pCA[3] = {0,0,0}, pC[3] = {0,0,0};
            bool hasN=false, hasCA=false, hasC=false;
            const std::string& all_atoms = fields[7];
            size_t pos = 0;
            while (pos < all_atoms.size()) {
                size_t open = all_atoms.find('(', pos);
                if (open == std::string::npos) break;
                std::string name = all_atoms.substr(pos, open - pos);
                name.erase(0, name.find_first_not_of(" \t"));
                name.erase(name.find_last_not_of(" \t") + 1);
                size_t close = all_atoms.find(')', open);
                if (close == std::string::npos) break;
                std::string coord_str = all_atoms.substr(open + 1, close - open - 1);
                float x=0,y=0,z=0;
                if (sscanf(coord_str.c_str(), "%f,%f,%f", &x, &y, &z) == 3) {
                    if (name == "N")  { pN[0]=x;pN[1]=y;pN[2]=z; hasN=true; }
                    if (name == "CA") { pCA[0]=x;pCA[1]=y;pCA[2]=z; hasCA=true; }
                    if (name == "C")  { pC[0]=x;pC[1]=y;pC[2]=z; hasC=true; }
                }
                pos = close + 1;
            }

            // CA 回退到 CA_X/Y/Z 列; N/C 缺失回退到 CA
            if (!hasCA) { pCA[0]=ca_fallback[0]; pCA[1]=ca_fallback[1]; pCA[2]=ca_fallback[2]; }
            if (!hasN)  { pN[0]=pCA[0]; pN[1]=pCA[1]; pN[2]=pCA[2]; }
            if (!hasC)  { pC[0]=pCA[0]; pC[1]=pCA[1]; pC[2]=pCA[2]; }

            // 校验: CA 坐标非零才视为有效 (避免占位残基污染)
            if (fabsf(pCA[0]) + fabsf(pCA[1]) + fabsf(pCA[2]) < 1e-6f) continue;

            float* base = coords.data() + (l * 3) * 3;
            base[0*3+0]=pN[0]; base[0*3+1]=pN[1]; base[0*3+2]=pN[2];
            base[1*3+0]=pCA[0]; base[1*3+1]=pCA[1]; base[1*3+2]=pCA[2];
            base[2*3+0]=pC[0]; base[2*3+1]=pC[1]; base[2*3+2]=pC[2];
            filled[l] = 1;
        }
    }

    int covered = 0;
    for (char f : filled) covered += (f != 0);
    std::cout << "[DataLoader] merged " << csv_paths.size() << " CSV mapping files: "
              << covered << "/" << L << " residues covered" << std::endl;
    return coords;
}

// ============================================================================
// parse_template_structure — 解析单个结构文件 (cif/pdb), 提取指定链骨架坐标
//
// 输出 out_coords 为展平 (N_res * 4 * 3): [res][atom 0..3 = N,CA,C,O][xyz 0..2]。
// 只保留骨干重原子完整的标准残基 (与 python extract_template_coords.py 一致)。
// ============================================================================
void RFAADataLoader::parse_template_structure(
    const std::string& path,
    const std::string& chain,
    std::vector<float>& out_coords,
    int& out_nres)
{
    out_coords.clear();
    out_nres = 0;
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // 目标链小写 (用于大小写不敏感比较)
    std::string target_chain = chain;
    std::transform(target_chain.begin(), target_chain.end(), target_chain.begin(), ::tolower);

    // ---- 残基级骨架原子缓存: 按 (chain, resseq) 组织 ----
    // 每个残基: N/CA/C/O 坐标, 记已找到哪些原子
    struct ResKey { std::string chain; std::string resseq; std::string comp; };
    auto key_less = [](const ResKey& a, const ResKey& b) {
        if (a.chain != b.chain) return a.chain < b.chain;
        return a.resseq < b.resseq;
    };
    std::map<ResKey, std::array<float,4*3>, decltype(key_less)> res_atoms(key_less);
    std::map<ResKey, std::array<char,4>, decltype(key_less)> res_found(key_less);

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open structure file: " + path);
    }

    auto store_atom = [&](const std::string& ch, const std::string& resseq,
                          const std::string& comp, const std::string& atom,
                          float x, float y, float z) {
        ResKey k{ch, resseq, comp};
        // 标准氨基酸 (20 AA)
        static const std::string AA = "ARNDCQEGHILKMFPSTWYV";
        if (comp.size() == 1 && AA.find(comp[0]) == std::string::npos) return;
        if (comp.size() == 3) {
            // 三字母转单字母 (常用子集)
            static const char* tri[20] = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU",
                "GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"};
            bool ok = false;
            for (int i = 0; i < 20; i++) if (comp == tri[i]) { ok = true; break; }
            if (!ok) return;
        }
        int atom_idx = -1;
        if (atom == "N") atom_idx = 0;
        else if (atom == "CA") atom_idx = 1;
        else if (atom == "C") atom_idx = 2;
        else if (atom == "O") atom_idx = 3;
        if (atom_idx < 0) return;
        auto& a = res_atoms[k];
        a[atom_idx*3+0]=x; a[atom_idx*3+1]=y; a[atom_idx*3+2]=z;
        res_found[k][atom_idx] = 1;
    };

    std::string line;
    if (ext == ".cif") {
        // ---- CIF: 逐行解析 _atom_site loop ----
        bool in_atom_site = false;
        // col_indices[i] = 第 i 个目标列的列位置 (i: 0=group_PDB,1=label_atom_id,
        // 2=label_comp_id,3=label_asym_id,4=label_seq_id,5/6/7=Cartn_x/y/z)
        std::vector<int> col_indices(8, -1);
        std::vector<std::string> col_headers;
        while (std::getline(file, line)) {
            if (line.back() == '\r') line.pop_back();
            std::istringstream iss(line);
            std::string tok;
            std::vector<std::string> cols;
            while (iss >> tok) cols.push_back(tok);
            if (cols.empty()) continue;

            // 进入 / 退出 atom_site 段落
            if (cols[0].rfind("_atom_site.", 0) == 0) {
                in_atom_site = true;
                col_headers.push_back(cols[0].substr(std::string("_atom_site.").size()));
                // 记录列索引
                int idx = static_cast<int>(col_headers.size()) - 1;
                const std::string& h = col_headers.back();
                if (h == "group_PDB") col_indices[0]=idx;
                else if (h == "label_atom_id") col_indices[1]=idx;
                else if (h == "label_comp_id") col_indices[2]=idx;
                else if (h == "label_asym_id") col_indices[3]=idx;
                else if (h == "label_seq_id") col_indices[4]=idx;
                else if (h == "Cartn_x") col_indices[5]=idx;
                else if (h == "Cartn_y") col_indices[6]=idx;
                else if (h == "Cartn_z") col_indices[7]=idx;
                continue;
            }
            if (in_atom_site && cols[0] == "#") { in_atom_site = false; continue; }
            if (!in_atom_site) continue;
            if (cols[0] == "loop_") continue;
            if (cols.size() < 8) continue;
            if (cols[0] != "ATOM") continue;  // 仅标准原子行

            auto cget = [&](int ci)->std::string{
                if (ci < 0 || ci >= (int)cols.size()) return "";
                return cols[ci];
            };
            std::string atom = cget(col_indices[1]);
            std::string comp = cget(col_indices[2]);
            std::string ch   = cget(col_indices[3]);
            std::string rsq  = cget(col_indices[4]);
            float x=0,y=0,z=0;
            try { x=std::stof(cget(col_indices[5])); y=std::stof(cget(col_indices[6])); z=std::stof(cget(col_indices[7])); }
            catch (...) { continue; }
            std::string chl = ch;
            std::transform(chl.begin(), chl.end(), chl.begin(), ::tolower);
            if (!target_chain.empty() && chl != target_chain) continue;
            store_atom(ch, rsq, comp, atom, x, y, z);
        }
    } else {
        // ---- PDB: 解析 ATOM 记录 (固定列宽) ----
        while (std::getline(file, line)) {
            if (line.size() < 54) continue;
            std::string record = line.substr(0, 6);
            if (record.rfind("ATOM", 0) != 0) continue;
            std::string atom = line.substr(12, 4);
            atom.erase(0, atom.find_first_not_of(" "));
            atom.erase(atom.find_last_not_of(" ") + 1);
            std::string resname = line.substr(17, 3);
            std::string chain = line.substr(21, 1);
            std::string resseq = line.substr(22, 4);
            std::string chl = chain;
            std::transform(chl.begin(), chl.end(), chl.begin(), ::tolower);
            if (!target_chain.empty() && chl != target_chain) continue;
            try {
                float x = std::stof(line.substr(30, 8));
                float y = std::stof(line.substr(38, 8));
                float z = std::stof(line.substr(46, 8));
                store_atom(chain, resseq, resname, atom, x, y, z);
            } catch (...) { continue; }
        }
    }

    // ---- 组装: 仅保留 4 个骨架原子齐全的残基 ----
    for (const auto& kv : res_atoms) {
        const auto& f = res_found[kv.first];
        if (!(f[0] && f[1] && f[2] && f[3])) continue;  // 缺任一骨架原子则跳过
        const auto& a = kv.second;
        for (int i = 0; i < 12; i++) out_coords.push_back(a[i]);
        out_nres++;
    }
}

// ============================================================================
// load_templates_from_dir — 从目录加载模板结构 (cif/pdb), 过滤真实值重复 PDB
// ============================================================================
void RFAADataLoader::load_templates_from_dir(
    const std::string& template_dir,
    const std::set<std::string>& exclude_pdb_ids,
    ModelInput& input,
    int max_templates)
{
    namespace fs = std::filesystem;
    input.template_coords.clear();
    input.template_ids.clear();
    input.template_chains.clear();
    input.template_residue_counts.clear();

    auto files = list_structure_files(template_dir);
    if (files.empty()) {
        std::cout << "[DataLoader] no structure files found in: " << template_dir << std::endl;
        return;
    }

    // 预扫描同目录 <pdb>_<chain>_coords.npy 以推断目标链
    std::map<std::string, std::string> pdb_to_chain;  // pdb(小写) -> chain
    {
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(template_dir, ec)) {
            if (ec) break;
            std::string name = entry.path().filename().string();
            if (name.size() < 12) continue;
            if (name.rfind("_coords.npy") != name.size() - 11) continue;
            std::string stem = entry.path().stem().string();   // "<pdb>_<chain>_coords"
            if (stem.rfind("_coords") != stem.size() - 7) continue;
            std::string base = stem.substr(0, stem.size() - 7); // "<pdb>_<chain>"
            auto us = base.rfind('_');
            if (us == std::string::npos || us == 0 || us == base.size() - 1) continue;
            std::string pdb = base.substr(0, us);
            std::string ch  = base.substr(us + 1);
            std::transform(pdb.begin(), pdb.end(), pdb.begin(), ::tolower);
            pdb_to_chain[pdb] = ch;
        }
    }

    int loaded = 0;
    for (const auto& fp : files) {
        if (loaded >= max_templates) break;
        std::string stem = fs::path(fp.first).stem().string();
        std::string pdb = stem;
        auto us = pdb.find('_');
        if (us != std::string::npos) pdb = pdb.substr(0, us);
        std::transform(pdb.begin(), pdb.end(), pdb.begin(), ::tolower);

        // 过滤: 与真实值 (ground truth) 相同的 PDB id 不再作为模板
        if (exclude_pdb_ids.count(pdb) > 0) {
            std::cout << "[DataLoader] skip template (duplicate of ground truth): "
                      << pdb << std::endl;
            continue;
        }

        std::string chain;
        auto it = pdb_to_chain.find(pdb);
        if (it != pdb_to_chain.end()) chain = it->second;  // 由 _coords.npy 推断, 否则取首链

        std::vector<float> coords;
        int nres = 0;
        try {
            parse_template_structure(fp.first, chain, coords, nres);
        } catch (const std::exception& e) {
            std::cerr << "[WARN] parse template failed: " << fp.first
                      << " (" << e.what() << ")" << std::endl;
            continue;
        }
        if (nres == 0 || coords.empty()) {
            std::cout << "[DataLoader] template yields no complete backbone residues: "
                      << fp.first << std::endl;
            continue;
        }

        TensorF32 t({nres, 4, 3}, Device::CPU);
        std::memcpy(t.data(), coords.data(), coords.size() * sizeof(float));
        input.template_coords.push_back(std::move(t));
        input.template_ids.push_back(pdb);
        input.template_chains.push_back(chain.empty() ? "A" : chain);
        input.template_residue_counts.push_back(nres);
        loaded++;
    }
    std::cout << "[DataLoader] loaded " << loaded << " templates (excluded "
              << exclude_pdb_ids.size() << " ground-truth pdb ids)" << std::endl;
}

ModelInput RFAADataLoader::load_from_files(
    const std::string& a3m_path,
    const std::string& sequence,
    const std::string& csv_path,
    const std::string& template_dir,
    const std::string& hhr_path
) {
    ModelInput input;

    // ============================================================
    // ground truth 真实坐标加载
    // csv_path 可以是:
    //   - 单个 CSV mapping 文件 (单一结构域)
    //   - 一个目录 (含多个 *_mapping_results.csv, 对应多个 PDB 结构域)
    // 当一个 uniprot 序列被多个 PDB 覆盖时, 目录方式会将它们合并为
    // 单个 (1, L, 3, 3) 真实坐标 (重叠残基取首个覆盖)。
    // ============================================================
    int L = static_cast<int>(sequence.length());
    if (!csv_path.empty()) {
        TensorF32 true_coords;
        // 判断是目录还是文件
        bool is_dir = std::filesystem::is_directory(csv_path);
        if (is_dir) {
            auto csv_files = list_csv_mapping_files(csv_path);
            if (csv_files.empty()) {
                std::cerr << "[WARN] no *_mapping_results.csv found in dir: "
                          << csv_path << std::endl;
            } else {
                true_coords = parse_csv_true_coords_multi(csv_files, L);
            }
        } else {
            true_coords = parse_csv_true_coords(csv_path, L);
        }
        if (true_coords.numel() > 0) {
            input.true_coords = std::move(true_coords);
            // 同时用真实坐标初始化 input.coords (推理/初始结构), 复制一份
            input.coords = TensorF32(input.true_coords.shape(), Device::CPU);
            input.coords.copy_from(input.true_coords);
        }
    }

    // ============================================================
    // 模板结构加载 (可选)
    // template_dir: 含 *.cif / *.pdb 的目录。自动过滤掉与真实值
    // (ground truth) 相同 PDB id 的模板 (不再作为模板使用)。
    // 结果存入 input.template_coords / template_ids / chains / residue_counts。
    // ============================================================
    if (!template_dir.empty()) {
        std::set<std::string> exclude_ids;
        if (!csv_path.empty() && std::filesystem::is_directory(csv_path)) {
            exclude_ids = collect_ground_truth_pdb_ids(csv_path);
        }
        load_templates_from_dir(template_dir, exclude_ids, input, max_templates_);
    }
    
    // Step 1: 解析 A3M (含插入计数矩阵)
    std::vector<std::vector<uint8_t>> a3m_raw;
    std::vector<std::vector<uint8_t>> a3m_ins;
    a3m_raw = parse_a3m(a3m_path, &a3m_ins);
    A3MData a3m_data;
    // 将 uint8 token (0-20) 逆向映射回氨基酸字符, 填充 a3m_data.sequences
    // ALPHABET 需与 parse_a3m 保持一致: "ARNDCQEGHILKMFPSTWYV-" (token 20 = gap)
    static const char* A3M_ALPHABET = "ARNDCQEGHILKMFPSTWYV-";
    a3m_data.sequences.clear();
    a3m_data.sequences.reserve(a3m_raw.size());
    for (const auto& row : a3m_raw) {
        std::string seq;
        seq.reserve(row.size());
        for (uint8_t tok : row) {
            // 防御: 越界 token 一律视为 gap
            seq.push_back(tok < 21 ? A3M_ALPHABET[tok] : '-');
        }
        a3m_data.sequences.push_back(std::move(seq));
    }
    // 插入计数矩阵: (N_seq, L), 与 sequences 一一对应
    a3m_data.ins_matrix = std::move(a3m_ins);
    // parse_a3m 已跳过 '>' 头行, 故每行都是合法序列
    a3m_data.num_sequences = static_cast<int>(a3m_data.sequences.size());
    a3m_data.sequence_length = a3m_raw.empty() ? 0 : static_cast<int>(a3m_raw[0].size());
    input.msa_latent = prepare_msa_latent(a3m_data);
    input.msa_full = prepare_msa_full(a3m_data);
    input.seq_tokens = prepare_seq_tokens(sequence);

    // ---- Masked MSA 监督标签 (BERT-style) ----
    // a3m_raw 即 MSA 整数 token (N_seq, L, 0-20)。此处实施 mask 操作:
    //   1. prepare_msa_mask 随机选择 ~15% 位置 (query 行除外) 作为掩码,
    //      产出 true_msa (被掩码位置真实 aatype) 与 bert_mask (1.0=掩码)。
    //   2. 特征侧 (input.msa_latent / msa_full) 中对应被掩码位置的 AA one-hot
    //      应在后续替换为 MASK token — 当前 prepare_msa_latent 未实施特征替换,
    //      仅产出监督标签; 特征替换留待 MSA head 接入时补充。
    prepare_msa_mask(a3m_raw, input.true_msa, input.bert_mask);

    // ---- Chi (扭转角) 监督标签 (从真实坐标计算) ----
    // true_coords (B,L,3,3)=[N,CA,C]。骨架角 (omega/phi/psi) 由 N/CA/C 计算,
    // chi1-4 需侧链原子 (true_coords 不含), mask=0。仅当提供真实坐标时计算。
    if (input.true_coords.numel() > 0) {
        prepare_chi_labels(input.true_coords, input.gt_chi, input.chi_mask);
        // ---- Distogram 监督标签 (从真实坐标 binning) ----
        prepare_distogram_labels(input.true_coords, input.D_onehot, input.O_onehot,
                                 input.T_onehot, input.P_onehot, input.pair_mask);
        // ---- CA 有效掩码 ----
        prepare_ca_mask(input.true_coords, input.ca_mask);
    }

    // read templates (TODO: 需要 FFindexDB 支持)
    // ReadTemplatesResult read_templates_result = read_templates(
    //     sequence.length(), ffdb, hhr_path, atab_fn, max_templates_);
    // TensorF32 xyz_t = std::move(read_templates_result.xyz);
    // xyz_t = xyz_t.unsqueeze(0);
    // TensorF32 t1d = std::move(read_templates_result.f1d);
    // TensorF32 t0d = std::move(read_templates_result.f0d);
    // t1d = t1d.unsqueeze(0);
    // t0d = t0d.unsqueeze(0);
    // TensorF32 t2d = xyz_to_t2d(xyz_t, read_templates_result.masks);
    // input.coords = std::move(xyz_t);
    // input.t1d = std::move(t1d);
    // input.t2d = std::move(t2d);

    /* alpha, _, alpha_mask, _ = util.get_torsions(
            xyz_t.reshape(-1,L,27,3),
            seq_tmp,
            util.torsion_indices,
            util.torsion_can_flip,
            util.reference_angles
        ) */
    /* alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0])) */
    /* alpha[torch.isnan(alpha)] = 0.0 */
    /* alpha = alpha.reshape(1,-1,L,10,2) */
    /* alpha_mask = alpha_mask.reshape(1,-1,L,10,1) */
    /* alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30) */

    // bond_feats, dist_matrix, same_chain, residx — 不依赖模板，可直接计算
    {
        int L = static_cast<int>(sequence.length());
        input.bond_feats = get_protein_bond_feats(L);
        input.bond_feats = input.bond_feats.unsqueeze(0);  // (L, L) → (1, L, L)
        input.dist_matrix = get_bond_distances(get_protein_bond_feats(L));
        input.dist_matrix = input.dist_matrix.unsqueeze(0); // (L, L) → (1, L, L)

        // same_chain: 全 1 (单链场景)
        TensorF32 same_chain({L, L});
        same_chain.data()[0] = 1.0f;  // 触发分配
        same_chain.zero_();
        float* sc_data = same_chain.data();
        for (int i = 0; i < L * L; i++) sc_data[i] = 1.0f;
        input.same_chain = same_chain.unsqueeze(0);  // (1, L, L)

        // residx: 0, 1, 2, ..., L-1
        TensorI64 residx({L});
        int64_t* ri_data = residx.data();
        for (int i = 0; i < L; i++) ri_data[i] = static_cast<int64_t>(i);
        input.residx = residx.unsqueeze(0);  // (1, L)
    }

    // TODO: torsions — 需要 xyz_t
    // input.tor_feat = get_torsions(xyz_t, sequence).torsions;

    // Step 2: 解析 HHR
    /* HHRData hhr_data = parse_hhr(hhr_path);
    TemplateData template_data = hhr_extract_template_features(
        hhr_data, sequence, max_templates_, max_length_
    );
    input.t1d = template_data.t1d;
    input.t2d = template_data.t2d; */
    
    // Step 3: 准备初始坐标 (可选)
    // coords from template?
    //input.coords = prepare_coords(sequence);
    
    return input;
}

ReadTemplatesResult RFAADataLoader::read_templates(
    int qlen,
    const FFindexDB& ffdb,
    const std::string& hhr_fn,
    const std::string& atab_fn,
    int n_templ
) {
    // TODO: 完整实现需要 FFindexDB 完整类型 + parse_templates 完善
    ReadTemplatesResult result;
    result.xyz = rfaa::TensorF32({0, qlen, 3, 3});
    result.f1d = rfaa::TensorF32({0, qlen, 3});
    result.f0d = rfaa::TensorF32({0, 3});
    result.masks = rfaa::TensorF32({0, qlen, 1});
    return result;
}

// Dead code below preserved for future reference
#if 0
    // int npick = std::min(n_templ, static_cast<int>(parsed.ids.size()));
    // ...
    rfaa::TensorF32 xyz({npick, qlen, 3, 3});
    for (int64_t i = 0; i < xyz_size; i++) {
        xyz_data[i] = nan_val;
    }
    
    // f1d: (npick, qlen, 3) - 初始化为 0
    rfaa::TensorF32 f1d({npick, qlen, 3});
    
    // f0d: 先收集到 vector，最后 stack
    std::vector<rfaa::TensorF32> f0d_list;
    
    // ========== 3. 填充数据 ==========
    // sample = range(npick) - 选择前 npick 个模板
    for (int i = 0; i < npick; i++) {
        int nt = i;  // Python: for i, nt in enumerate(sample) - nt 是模板索引
        
        // sel = torch.where(qmap[:,1] == nt)[0]
        std::vector<int64_t> sel = where_qmap_equals(parsed.qmap, nt);
        if (sel.empty()) continue;
        
        // pos = qmap[sel, 0]
        std::vector<int64_t> pos;
        for (int64_t idx : sel) {
            pos.push_back(static_cast<int64_t>(parsed.qmap.data()[idx * 2 + 0]));
        }
        
        // xyz[i, pos] = xyz_t[sel, :3]
        // xyz_t: (total_align, 3) - 假设只有 N,CA,C 坐标
        for (size_t j = 0; j < sel.size(); j++) {
            int64_t sel_idx = sel[j];
            int64_t pos_idx = pos[j];
            
            if (pos_idx < 0 || pos_idx >= qlen) continue;
            
            // xyz_t[sel_idx, :3] -> xyz[i, pos_idx, 0:3, 0:3]
            // 注意：Python 是 xyz_t[sel, :3]，但原始是 (N, 14, 3)
            // 这里 xyz_t 应该是 (N, 3, 3) 或 (N, 3)
            // 假设 xyz_t 是 (N, 3, 3) - [N,CA,C] x [x,y,z]
            for (int atom = 0; atom < 3; atom++) {  // N, CA, C
                for (int coord = 0; coord < 3; coord++) {
                    // 从 parsed.xyz 读取（假设是 (N, 14, 3) 格式）
                    // 或者从 parsed.xyz_t 读取（如果是预处理过的）
                    float val = 0.0f;
                    if (parsed.xyz.shape().dims.size() >= 2) {
                        // parsed.xyz: (N, 14, 3) 或 (N, 3, 3)
                        int64_t src_offset;
                        if (parsed.xyz.shape().dims[1] == 14) {
                            // (N, 14, 3) - 取前3个原子 N=0, CA=1, C=2
                            src_offset = (sel_idx * 14 + atom) * 3 + coord;
                        } else {
                            // (N, 3, 3)
                            src_offset = (sel_idx * 3 + atom) * 3 + coord;
                        }
                        val = parsed.xyz.data()[src_offset];
                    }
                    
                    // xyz[i, pos_idx, atom, coord] = val
                    int64_t dst_offset = ((i * qlen + pos_idx) * 3 + atom) * 3 + coord;
                    xyz_data[dst_offset] = val;
                }
            }
        }
        
        // f1d[i, pos] = t1d[sel, :3]
        for (size_t j = 0; j < sel.size(); j++) {
            int64_t sel_idx = sel[j];
            int64_t pos_idx = pos[j];
            
            if (pos_idx < 0 || pos_idx >= qlen) continue;
            
            for (int k = 0; k < 3; k++) {
                float val = parsed.f1d.data()[sel_idx * 3 + k];
                f1d.data()[((i * qlen + pos_idx) * 3 + k)] = val;
            }
        }
        
        // f0d.append([t0d[nt,0]/100, t0d[nt,4]/100, t0d[nt,5]])
        // t0d: (n_templates, 8) - [Probab, E-value, Score, Aligned_cols, Identities, Similarity, ...]
        rfaa::TensorF32 f0d_row({1, 3});
        f0d_row.data()[0] = parsed.f0d.data()[nt * 8 + 0] / 100.0f;  // Probab/100
        f0d_row.data()[1] = parsed.f0d.data()[nt * 8 + 4] / 100.0f;  // Identities/100
        f0d_row.data()[2] = parsed.f0d.data()[nt * 8 + 5] / 100.0f;  // Similarity / 100
        f0d_list.push_back(std::move(f0d_row));
    }
    
    // ========== 4. Stack f0d ==========
    rfaa::TensorF32 f0d_stacked({npick, 3});
    float* f0d_data = f0d_stacked.data();
    for (int i = 0; i < npick; i++) {
        std::memcpy(f0d_data + i * 3, f0d_list[i].data(), 3 * sizeof(float));
    }
    
    // ========== 5. 返回结果 ==========
    ReadTemplatesResult result;
    result.xyz = std::move(xyz);
    result.masks = std::move(masks);
    result.f1d = std::move(f1d);
    result.f0d = std::move(f0d_stacked);
    result.ids = std::move(parsed.ids);
    
    return result;
}
#endif

std::vector<TemplateHit> RFAADataLoader::parse_atab(const std::string& atab_fn) {
    // 解析 .atab 文件，提取模板命中信息
    std::vector<TemplateHit> hits;
    // TODO: 完整实现
    return hits;
}

TemplateDataInternal RFAADataLoader::parse_templates(
    const std::string& db_prefix,
    const std::string& hhr_fn,
    const std::string& atab_fn,
    int n_templ
) {
    // TODO: FFindexDB 需要完整类型支持（当前仅为前向声明）
    // 完整实现需要: load_ffdb, parse_atab, parse_pdb_lines 等
    TemplateDataInternal result;
    return result;
}

// ========== 2. 解析 .hhr 文件 ==========
HHRData RFAADataLoader::parse_hhr(const std::string& hhr_path) {
    HHRData result;
    std::ifstream file(hhr_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open hhr file: " + hhr_path);
    }
        
    std::string line;
    int hit_idx = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
            
        if (line[0] == '>') {
                // 提取统计数据
            std::string cleaned;
            for (char c : line) {
                if (c == '=' || c == '%') {
                    cleaned += ' ';
                } else {
                    cleaned += c;
                }
            }
                
            std::istringstream iss(cleaned);
            std::vector<std::string> tokens;
            std::string token;
            while (iss >> token) {
                tokens.push_back(token);
            }
                
                // 取索引 1,3,5,7,... (值)
            std::vector<float> stats;
            for (size_t i = 1; i < tokens.size(); i += 2) {
                try {
                    stats.push_back(std::stof(tokens[i]));
                } catch (...) {
                    stats.push_back(0.0f);
                }
            }
                
            if (hit_idx < static_cast<int>(result.hits.size())) {
                result.hits[hit_idx].stats = stats;
            }
            hit_idx++;
        }
    }
    return result;
}

std::vector<std::vector<uint8_t>> RFAADataLoader::parse_a3m(
    const std::string& a3m_path,
    std::vector<std::vector<uint8_t>>* ins_matrix) {
    std::vector<std::vector<uint8_t>> msa;
    std::vector<std::vector<uint8_t>> ins_out;
    
    // 构建 ASCII 到整数的快速查找表 (256 大小)
    uint8_t char_map[256];
    std::fill(std::begin(char_map), std::end(char_map), 20); // 默认是 gap
    
    // 设置映射: A->0, R->1, ... V->19, - ->20
    const char* ALPHABET = "ARNDCQEGHILKMFPSTWYV-";
    for (int i = 0; i < 21; i++) {
        char_map[static_cast<uint8_t>(ALPHABET[i])] = static_cast<uint8_t>(i);
    }
    
    std::ifstream file(a3m_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + a3m_path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line[0] == '>') continue;
        
        // 移除尾部空白
        while (!line.empty() && std::isspace(line.back())) {
            line.pop_back();
        }
        if (line.empty()) continue;
        
        // 转换：移除小写，大写转整数; 同时统计每列的插入计数 (RFAA 算法)
        std::vector<uint8_t> seq;
        std::vector<uint8_t> ins_row;  // 长度 = 清洗后对齐列数
        seq.reserve(line.size());
        ins_row.reserve(line.size());
        for (char c : line) {
            if (std::isupper(static_cast<unsigned char>(c))) {
                seq.push_back(char_map[static_cast<uint8_t>(c)]);
                ins_row.push_back(0);
            } else if (c == '-') {
                seq.push_back(20); // gap
                ins_row.push_back(0);
            }
            // 小写字母: 不进入 seq/ins_row, 作为插入在下方统计
        }
        
        if (!seq.empty()) {
            // 插入计数: 对每个小写 (插入) 位置, collapsed = pos - 出现次序,
            // 相同 collapsed 的连续插入归到同一对齐列, 计数为该列插入长度。
            // 等价于 RFAA data_loader_utils.py: a=pos-arange(pos.size()); i[unique]=count
            std::vector<int> pos;
            for (size_t k = 0; k < line.size(); ++k) {
                if (std::islower(static_cast<unsigned char>(line[k]))) pos.push_back((int)k);
            }
            if (!pos.empty()) {
                std::vector<int> collapsed(pos.size());
                for (size_t i = 0; i < pos.size(); ++i) collapsed[i] = pos[i] - (int)i;
                // collapsed 非递减, 连续相等者归组
                size_t i = 0;
                while (i < collapsed.size()) {
                    int val = collapsed[i];
                    int cnt = 0;
                    while (i < collapsed.size() && collapsed[i] == val) { ++cnt; ++i; }
                    if (val >= 0 && val < (int)ins_row.size()) ins_row[val] = (uint8_t)cnt;
                }
            }
            
            msa.push_back(std::move(seq));
            ins_out.push_back(std::move(ins_row));
        }
    }
    
    if (ins_matrix) *ins_matrix = std::move(ins_out);
    return msa;
}

TensorF32 RFAADataLoader::a3m_to_msa_features(const std::vector<std::vector<uint8_t>>& a3m_data, int max_seqs, int max_length) {

    // 简化实现：返回零张量
    // 实际实现需要：
    // 1. 将序列转换为 one-hot + 删除概率特征
    // 2. 填充/截断到 (max_seqs, max_length, 83)
    
    TensorF32 features({max_seqs, max_length, MSA_FULL_DIM}, Device::CPU);
    features.zero_();
    
    // TODO: 实际特征提取逻辑
    // 对于每个序列：
    //   - one-hot 编码氨基酸 (20 维)
    //   - 删除概率特征 (1 维)
    //   - 其他特征 (62 维)
    //   - 总计 83 维
    
    return features;
}


TemplateData RFAADataLoader::hhr_extract_template_features(
    const HHRData& hhr_data,
    const std::string& query_sequence,
    int num_templates,
    int max_length
) {
    // 简化实现：返回零张量
    // 实际实现需要：
    // 1. 从 PDB 文件加载模板坐标
    // 2. 计算 t1d (模板 1D 特征): (T, L, 80)
    // 3. 计算 t2d (模板 2D 特征): (T, L, L, ...)
    // 4. 提取坐标: (T, L, 3, 3)
    
    int T = std::min(num_templates, hhr_data.num_templates);
    
    TemplateData result;
    result.t1d = zeros<float>({T, max_length, D_T1D}, Device::CPU);
    result.t2d = zeros<float>({T, max_length, max_length, D_T2D}, Device::CPU);
    result.coords = zeros<float>({T, max_length, 3, 3}, Device::CPU);
    
    // TODO: 实际特征提取逻辑
    
    return result;
}

TorsionResult RFAADataLoader::get_torsions(
    const rfaa::TensorF32& xyz_in,           // (B, L, 14, 3)
    const rfaa::TensorF32& seq,              // (B, L)
    const std::vector<std::vector<std::vector<int>>>& torsion_indices, // (21, 7, 4)
    const std::vector<std::vector<bool>>& torsion_can_flip,            // (21, 7)
    const std::vector<std::vector<std::array<float, 2>>>& ref_angles  // (21, 3, 2)
) {
    int B = static_cast<int>(xyz_in.shape().dims[0]);
    int L = static_cast<int>(xyz_in.shape().dims[1]);
    
    // ========== 1. 计算 tors_mask (简化：假设所有都有效) ==========
    rfaa::TensorF32 tors_mask({B, L, 10});
    // fill with 1.0
    float* mask_data = tors_mask.data();
    for (int i = 0; i < tors_mask.numel(); i++) mask_data[i] = 1.0f;
    
    // ========== 2. tors_planar: TYR chi 3 should be planar ==========
    rfaa::TensorF32 tors_planar({B, L, 10});
    tors_planar.zero_();
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            int seq_val = static_cast<int>(seq.data()[b*L + l]);
            if (seq_val == 18) {  // TYR = 18
                tors_planar.data()[(b*L + l)*10 + 5] = 1.0f;  // chi 3 (index 5)
            }
        }
    }
    
    // ========== 3. 理想化坐标 ==========
    rfaa::TensorF32 xyz(xyz_in.shape(), xyz_in.device());
    xyz.copy_from(xyz_in);
    
    // Rs, Ts = rigid_from_3_points(N, Ca, C)
    rfaa::TensorF32 N = xyz.select(2, 0);  // (B, L, 3)
    rfaa::TensorF32 Ca = xyz.select(2, 1);
    rfaa::TensorF32 C = xyz.select(2, 2);
    
    // rigid_from_3_points: 从 N, CA, C 计算旋转矩阵和平移向量
    // Rs: (B, L, 3, 3) 旋转矩阵, Ts: (B, L, 3) 平移向量
    rfaa::TensorF32 Rs({B, L, 3, 3});
    rfaa::TensorF32 Ts({B, L, 3});
    Rs.zero_(); Ts.zero_();
    // TODO: 实现完整的 rigid_from_3_points
    
    // Nideal = [-0.5272, 1.3593, 0.000], Cideal = [1.5233, 0.000, 0.000]
    std::array<float, 3> Nideal = {-0.5272f, 1.3593f, 0.0f};
    std::array<float, 3> Cideal = {1.5233f, 0.0f, 0.0f};
    
    // xyz[...,0,:] = einsum('brij,j->bri', Rs, Nideal) + Ts
    // xyz[...,2,:] = einsum('brij,j->bri', Rs, Cideal) + Ts
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            int rs_offset = (b*L + l) * 9;
            int ts_offset = (b*L + l) * 3;
            
            // Nideal 旋转 + 平移
            float nx = Rs.data()[rs_offset+0]*Nideal[0] + Rs.data()[rs_offset+1]*Nideal[1] + Rs.data()[rs_offset+2]*Nideal[2] + Ts.data()[ts_offset+0];
            float ny = Rs.data()[rs_offset+3]*Nideal[0] + Rs.data()[rs_offset+4]*Nideal[1] + Rs.data()[rs_offset+5]*Nideal[2] + Ts.data()[ts_offset+1];
            float nz = Rs.data()[rs_offset+6]*Nideal[0] + Rs.data()[rs_offset+7]*Nideal[1] + Rs.data()[rs_offset+8]*Nideal[2] + Ts.data()[ts_offset+2];
            
            // Cideal 旋转 + 平移
            float cx = Rs.data()[rs_offset+0]*Cideal[0] + Rs.data()[rs_offset+1]*Cideal[1] + Rs.data()[rs_offset+2]*Cideal[2] + Ts.data()[ts_offset+0];
            float cy = Rs.data()[rs_offset+3]*Cideal[0] + Rs.data()[rs_offset+4]*Cideal[1] + Rs.data()[rs_offset+5]*Cideal[2] + Ts.data()[ts_offset+1];
            float cz = Rs.data()[rs_offset+6]*Cideal[0] + Rs.data()[rs_offset+7]*Cideal[1] + Rs.data()[rs_offset+8]*Cideal[2] + Ts.data()[ts_offset+2];
            
            // 写回 xyz
            xyz.data()[((b*L + l)*14 + 0)*3 + 0] = nx;
            xyz.data()[((b*L + l)*14 + 0)*3 + 1] = ny;
            xyz.data()[((b*L + l)*14 + 0)*3 + 2] = nz;
            
            xyz.data()[((b*L + l)*14 + 2)*3 + 0] = cx;
            xyz.data()[((b*L + l)*14 + 2)*3 + 1] = cy;
            xyz.data()[((b*L + l)*14 + 2)*3 + 2] = cz;
        }
    }
    
    // ========== 4. 计算 torsions ==========
    rfaa::TensorF32 torsions({B, L, 10, 2});
    torsions.zero_();
    
    // 辅助: 获取原子 xyz 坐标指针
    // xyz: (B, L, 14, 3), 原子索引: 0=N, 1=CA, 2=C, 3=O, 4=CB, ...
    const float* xyz_data = xyz.data();
    auto get_atom = [&](int b, int l, int atom_idx) -> std::array<float, 3> {
        int base = ((b*L + l)*14 + atom_idx)*3;
        return {xyz_data[base], xyz_data[base+1], xyz_data[base+2]};
    };

    for (int b = 0; b < B; b++) {
        // ---- omega: torsions[b, :L-1, 0, :] ----
        // omega[i] = dihedral(CA[i], C[i], N[i+1], CA[i+1])
        for (int l = 0; l < L-1; l++) {
            auto ca0 = get_atom(b, l,   1);
            auto c0  = get_atom(b, l,   2);
            auto n1  = get_atom(b, l+1, 0);
            auto ca1 = get_atom(b, l+1, 1);

            float ab_x = ca0[0] - c0[0],  ab_y = ca0[1] - c0[1],  ab_z = ca0[2] - c0[2];
            float bc_x = c0[0]  - n1[0],  bc_y = c0[1]  - n1[1],  bc_z = c0[2]  - n1[2];
            float cd_x = n1[0]  - ca1[0], cd_y = n1[1]  - ca1[1], cd_z = n1[2]  - ca1[2];

            // 法向量 n1_ = ab × bc,  n2_ = bc × cd
            float n1_x = ab_y*bc_z - ab_z*bc_y;
            float n1_y = ab_z*bc_x - ab_x*bc_z;
            float n1_z = ab_x*bc_y - ab_y*bc_x;
            float n2_x = bc_y*cd_z - bc_z*cd_y;
            float n2_y = bc_z*cd_x - bc_x*cd_z;
            float n2_z = bc_x*cd_y - bc_y*cd_x;

            float dot = n1_x*n2_x + n1_y*n2_y + n1_z*n2_z;
            float n1_n = std::sqrt(n1_x*n1_x + n1_y*n1_y + n1_z*n1_z);
            float n2_n = std::sqrt(n2_x*n2_x + n2_y*n2_y + n2_z*n2_z);
            float cos_a = std::max(-1.0f, std::min(1.0f, dot / (n1_n * n2_n + 1e-6f)));
            float ang = std::acos(cos_a);
            // 符号
            float sx = n1_y*n2_z - n1_z*n2_y;
            float sy = n1_z*n2_x - n1_x*n2_z;
            float sz = n1_x*n2_y - n1_y*n2_x;
            float sign = bc_x*sx + bc_y*sy + bc_z*sz;
            if (sign < 0) ang = -ang;

            int out_base = ((b*L + l)*10 + 0)*2;
            torsions.data()[out_base + 0] = std::sin(ang);
            torsions.data()[out_base + 1] = std::cos(ang);
        }

        // ---- phi: torsions[b, 1:, 1, :] ----
        // phi[i] = dihedral(C[i-1], N[i], CA[i], C[i])
        for (int l = 1; l < L; l++) {
            auto c_prev = get_atom(b, l-1, 2);
            auto n      = get_atom(b, l,   0);
            auto ca     = get_atom(b, l,   1);
            auto c      = get_atom(b, l,   2);

            float ab_x = c_prev[0] - n[0],  ab_y = c_prev[1] - n[1],  ab_z = c_prev[2] - n[2];
            float bc_x = n[0]  - ca[0],     bc_y = n[1]  - ca[1],     bc_z = n[2]  - ca[2];
            float cd_x = ca[0] - c[0],      cd_y = ca[1] - c[1],      cd_z = ca[2] - c[2];

            float n1_x = ab_y*bc_z - ab_z*bc_y;
            float n1_y = ab_z*bc_x - ab_x*bc_z;
            float n1_z = ab_x*bc_y - ab_y*bc_x;
            float n2_x = bc_y*cd_z - bc_z*cd_y;
            float n2_y = bc_z*cd_x - bc_x*cd_z;
            float n2_z = bc_x*cd_y - bc_y*cd_x;

            float dot = n1_x*n2_x + n1_y*n2_y + n1_z*n2_z;
            float n1_n = std::sqrt(n1_x*n1_x + n1_y*n1_y + n1_z*n1_z);
            float n2_n = std::sqrt(n2_x*n2_x + n2_y*n2_y + n2_z*n2_z);
            float cos_a = std::max(-1.0f, std::min(1.0f, dot / (n1_n * n2_n + 1e-6f)));
            float ang = std::acos(cos_a);
            float sx = n1_y*n2_z - n1_z*n2_y;
            float sy = n1_z*n2_x - n1_x*n2_z;
            float sz = n1_x*n2_y - n1_y*n2_x;
            float sign = bc_x*sx + bc_y*sy + bc_z*sz;
            if (sign < 0) ang = -ang;

            int out_base = ((b*L + l)*10 + 1)*2;
            torsions.data()[out_base + 0] = std::sin(ang);
            torsions.data()[out_base + 1] = std::cos(ang);
        }

        // ---- psi: torsions[b, :, 2, :] ----
        // psi[i] = -dihedral(N[i], CA[i], C[i], N[i+1])  (最后一残基无 psi)
        for (int l = 0; l < L-1; l++) {
            auto n   = get_atom(b, l,   0);
            auto ca  = get_atom(b, l,   1);
            auto c   = get_atom(b, l,   2);
            auto n_next = get_atom(b, l+1, 0);

            float ab_x = n[0]  - ca[0],  ab_y = n[1]  - ca[1],  ab_z = n[2]  - ca[2];
            float bc_x = ca[0] - c[0],   bc_y = ca[1] - c[1],   bc_z = ca[2] - c[2];
            float cd_x = c[0]  - n_next[0], cd_y = c[1]  - n_next[1], cd_z = c[2]  - n_next[2];

            float n1_x = ab_y*bc_z - ab_z*bc_y;
            float n1_y = ab_z*bc_x - ab_x*bc_z;
            float n1_z = ab_x*bc_y - ab_y*bc_x;
            float n2_x = bc_y*cd_z - bc_z*cd_y;
            float n2_y = bc_z*cd_x - bc_x*cd_z;
            float n2_z = bc_x*cd_y - bc_y*cd_x;

            float dot = n1_x*n2_x + n1_y*n2_y + n1_z*n2_z;
            float n1_n = std::sqrt(n1_x*n1_x + n1_y*n1_y + n1_z*n1_z);
            float n2_n = std::sqrt(n2_x*n2_x + n2_y*n2_y + n2_z*n2_z);
            float cos_a = std::max(-1.0f, std::min(1.0f, dot / (n1_n * n2_n + 1e-6f)));
            float ang = std::acos(cos_a);
            float sx = n1_y*n2_z - n1_z*n2_y;
            float sy = n1_z*n2_x - n1_x*n2_z;
            float sz = n1_x*n2_y - n1_y*n2_x;
            float sign = bc_x*sx + bc_y*sy + bc_z*sz;
            if (sign < 0) ang = -ang;

            // psi = -angle
            ang = -ang;

            int out_base = ((b*L + l)*10 + 2)*2;
            torsions.data()[out_base + 0] = std::sin(ang);
            torsions.data()[out_base + 1] = std::cos(ang);
        }

        // ---- chi1-chi4: torsions[b, :, 3:7, :] ----
        // 从 torsion_indices[seq_val][3+j] 查表获取原子索引
        for (int l = 0; l < L; l++) {
            int seq_val = static_cast<int>(seq.data()[b*L + l]);
            if (seq_val < 0 || seq_val >= static_cast<int>(torsion_indices.size())) continue;

            for (int chi = 0; chi < 4; chi++) {
                int t_idx = 3 + chi;  // torsion type index: 3=chi1, 4=chi2, 5=chi3, 6=chi4
                const auto& idx4 = torsion_indices[seq_val][t_idx];

                // 检查是否有效 (四个原子索引都 >= 0)
                if (idx4.size() < 4) continue;
                bool valid = true;
                for (int k = 0; k < 4; k++) {
                    if (idx4[k] < 0 || idx4[k] >= 14) { valid = false; break; }
                }
                if (!valid) continue;

                auto a0 = get_atom(b, l, idx4[0]);
                auto a1 = get_atom(b, l, idx4[1]);
                auto a2 = get_atom(b, l, idx4[2]);
                auto a3 = get_atom(b, l, idx4[3]);

                float ab_x = a0[0] - a1[0], ab_y = a0[1] - a1[1], ab_z = a0[2] - a1[2];
                float bc_x = a1[0] - a2[0], bc_y = a1[1] - a2[1], bc_z = a1[2] - a2[2];
                float cd_x = a2[0] - a3[0], cd_y = a2[1] - a3[1], cd_z = a2[2] - a3[2];

                float n1_x = ab_y*bc_z - ab_z*bc_y;
                float n1_y = ab_z*bc_x - ab_x*bc_z;
                float n1_z = ab_x*bc_y - ab_y*bc_x;
                float n2_x = bc_y*cd_z - bc_z*cd_y;
                float n2_y = bc_z*cd_x - bc_x*cd_z;
                float n2_z = bc_x*cd_y - bc_y*cd_x;

                float dot = n1_x*n2_x + n1_y*n2_y + n1_z*n2_z;
                float n1_n = std::sqrt(n1_x*n1_x + n1_y*n1_y + n1_z*n1_z);
                float n2_n = std::sqrt(n2_x*n2_x + n2_y*n2_y + n2_z*n2_z);
                float cos_a = std::max(-1.0f, std::min(1.0f, dot / (n1_n * n2_n + 1e-6f)));
                float ang = std::acos(cos_a);
                float sx = n1_y*n2_z - n1_z*n2_y;
                float sy = n1_z*n2_x - n1_x*n2_z;
                float sz = n1_x*n2_y - n1_y*n2_x;
                float sign = bc_x*sx + bc_y*sy + bc_z*sz;
                if (sign < 0) ang = -ang;

                int out_base = ((b*L + l)*10 + t_idx)*2;
                torsions.data()[out_base + 0] = std::sin(ang);
                torsions.data()[out_base + 1] = std::cos(ang);
            }
        }
    }
    // CB bend, CB twist, CG bend 等 (index 7-9) 暂不计算，保持为 0
    
    // ========== 5. 处理 NaN ==========
    // Python: alpha[torch.isnan(alpha)] = 0.0, 即 sin=0, cos=1 (角=0°)
    for (int i = 0; i < torsions.numel(); i += 2) {
        if (std::isnan(torsions.data()[i])) {
            torsions.data()[i] = 0.0f;      // sin = 0
            torsions.data()[i+1] = 1.0f;   // cos = 1 → angle = 0
        }
    }
    
    // ========== 6. torsions_alt = torsions * -1 (可翻转的) ==========
    rfaa::TensorF32 torsions_alt(torsions.shape(), torsions.device());
    torsions_alt.copy_from(torsions);
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            int seq_val = static_cast<int>(seq.data()[b*L + l]);
            for (int t = 0; t < 7; t++) {
                if (t < static_cast<int>(torsion_can_flip[seq_val].size()) && 
                    torsion_can_flip[seq_val][t]) {
                    // 翻转: sin -> -sin, cos -> cos (或者根据实现)
                    torsions_alt.data()[((b*L + l)*10 + (t+3))*2 + 0] *= -1;  // sin 取负
                }
            }
        }
    }
    
    TorsionResult result;
    result.torsions = std::move(torsions);
    result.torsions_alt = std::move(torsions_alt);
    result.tors_mask = std::move(tors_mask);
    result.tors_planar = std::move(tors_planar);
    return result;
}

void RFAADataLoader::init_torsion_indices() {
    for (int i = 0; i < NPROTAAS; i++) {
        const auto& i_l = aa2long[i];
        const auto& i_a = aa2longalt[i];
            
        // ========== 蛋白质 omega/phi/psi ==========
        // omega: [-1, -2, 0, 1]
        torsion_indices[i][0][0] = -1;
        torsion_indices[i][0][1] = -2;
        torsion_indices[i][0][2] = 0;
        torsion_indices[i][0][3] = 1;
            
        // phi: [-2, 0, 1, 2]
        torsion_indices[i][1][0] = -2;
        torsion_indices[i][1][1] = 0;
        torsion_indices[i][1][2] = 1;
        torsion_indices[i][1][3] = 2;
            
        // psi: [0, 1, 2, 3]
        torsion_indices[i][2][0] = 0;
        torsion_indices[i][2][1] = 1;
        torsion_indices[i][2][2] = 2;
        torsion_indices[i][2][3] = 3;
            
        // ========== 蛋白质 chis ==========
        for (int j = 0; j < 4; j++) {
            if (torsions[i][j].empty() || torsions[i][j][0].empty()) {
                // None 或空，跳过
                continue;
            }
                
            for (int k = 0; k < 4; k++) {
                const std::string& a = torsions[i][j][k];
                    
                // 在 i_l 中查找索引
                auto it = std::find(i_l.begin(), i_l.end(), a);
                if (it != i_l.end()) {
                    torsion_indices[i][3 + j][k] = static_cast<int8_t>(std::distance(i_l.begin(), it));
                } else {
                    torsion_indices[i][3 + j][k] = -1;  // 未找到
                }
                    
                // 检查是否可以翻转
                auto it_a = std::find(i_a.begin(), i_a.end(), a);
                if (it_a != i_a.end() && std::distance(i_a.begin(), it_a) != std::distance(i_l.begin(), it)) {
                    torsion_can_flip[i][3 + j] = true;
                }
            }
        }
            
        // ========== CB/CG angles ==========
        // CB ang1: [0, 2, 1, 4]
        torsion_indices[i][7][0] = 0;
        torsion_indices[i][7][1] = 2;
        torsion_indices[i][7][2] = 1;
        torsion_indices[i][7][3] = 4;
            
        // CB ang2: [0, 2, 1, 4]
        torsion_indices[i][8][0] = 0;
        torsion_indices[i][8][1] = 2;
        torsion_indices[i][8][2] = 1;
        torsion_indices[i][8][3] = 4;
            
        // CG ang: [0, 2, 4, 5]
        torsion_indices[i][9][0] = 0;
        torsion_indices[i][9][1] = 2;
        torsion_indices[i][9][2] = 4;
        torsion_indices[i][9][3] = 5;
    }
        
    // ========== HIS 特殊情况 ==========
    torsion_can_flip[8][4] = false;  // HIS chi2 不翻转
        
    // ========== DNA/RNA ==========
    // 假设 use_phospate_frames_for_NA = false (默认)
    for (int i = NPROTAAS; i < NNAPROTAAS; i++) {
        // ribose frame
        // epsilon_prev: [-2, -9, -10, 4]
        torsion_indices[i][10][0] = -2;
        torsion_indices[i][10][1] = -9;
        torsion_indices[i][10][2] = -10;
        torsion_indices[i][10][3] = 4;
            
        // zeta_prev: [-9, -10, 4, 6]
        torsion_indices[i][11][0] = -9;
        torsion_indices[i][11][1] = -10;
        torsion_indices[i][11][2] = 4;
        torsion_indices[i][11][3] = 6;
            
        // alpha: [7, 6, 4, 3]
        torsion_indices[i][12][0] = 7;
        torsion_indices[i][12][1] = 6;
        torsion_indices[i][12][2] = 4;
        torsion_indices[i][12][3] = 3;
            
        // beta: [8, 7, 6, 4]
        torsion_indices[i][13][0] = 8;
        torsion_indices[i][13][1] = 7;
        torsion_indices[i][13][2] = 6;
        torsion_indices[i][13][3] = 4;
            
        // gamma: [9, 8, 7, 6]
        torsion_indices[i][14][0] = 9;
        torsion_indices[i][14][1] = 8;
        torsion_indices[i][14][2] = 7;
        torsion_indices[i][14][3] = 6;
            
        // delta: [2, 9, 8, 7]
        torsion_indices[i][15][0] = 2;
        torsion_indices[i][15][1] = 9;
        torsion_indices[i][15][2] = 8;
        torsion_indices[i][15][3] = 7;
            
        // nu2: [1, 2, 9, 8]
        torsion_indices[i][16][0] = 1;
        torsion_indices[i][16][1] = 2;
        torsion_indices[i][16][2] = 9;
        torsion_indices[i][16][3] = 8;
            
        // nu1: [0, 1, 2, 9]
        torsion_indices[i][17][0] = 0;
        torsion_indices[i][17][1] = 1;
        torsion_indices[i][17][2] = 2;
        torsion_indices[i][17][3] = 9;
            
        // nu0: [2, 1, 0, 8]
        torsion_indices[i][18][0] = 2;
        torsion_indices[i][18][1] = 1;
        torsion_indices[i][18][2] = 0;
        torsion_indices[i][18][3] = 8;
            
        // NA chi: [如果 torsions[i][0] 不为 None]
        if (!torsions[i][0].empty() && !torsions[i][0][0].empty()) {
            const auto& i_l = aa2long[i];
            for (int k = 0; k < 4; k++) {
                const std::string& a = torsions[i][0][k];
                auto it = std::find(i_l.begin(), i_l.end(), a);
                if (it != i_l.end()) {
                    torsion_indices[i][19][k] = static_cast<int8_t>(std::distance(i_l.begin(), it));
                }
            }
        }
    }
}

TensorF32 RFAADataLoader::get_protein_bond_feats(int protein_L) {
    // 创建 L x L 的零矩阵
    TensorF32 bond_feats({protein_L, protein_L});
    bond_feats.zero_();
    
    // 设置相邻残基之间的键（值为 5）
    float* data = bond_feats.data();
    for (int i = 0; i < protein_L - 1; i++) {
        data[i * protein_L + (i + 1)] = 5.0f;  // (i, i+1)
        data[(i + 1) * protein_L + i] = 5.0f;  // (i+1, i)
    }
    
    return bond_feats;
}

// ========== 辅助函数：BFS 计算最短路径 ==========
// 输入: adj (L, L) - 邻接矩阵 (bool)
// 输出: dist (L, L) - 最短路径矩阵
rfaa::TensorF32 compute_shortest_path(const rfaa::TensorF32& adj) {
    int L = static_cast<int>(adj.shape().dims[0]);
    rfaa::TensorF32 dist({L, L});
    float* dist_data = dist.data();
    
    const float* adj_data = adj.data();
    
    // 对每个源点做 BFS
    for (int src = 0; src < L; src++) {
        // 初始化距离
        for (int i = 0; i < L; i++) {
            dist_data[src * L + i] = (i == src) ? 0.0f : std::numeric_limits<float>::infinity();
        }
        
        // BFS 队列
        std::queue<int> q;
        q.push(src);
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            
            // 遍历邻居
            for (int v = 0; v < L; v++) {
                if (adj_data[u * L + v] > 0.5f) {  // 有边
                    if (std::isinf(dist_data[src * L + v])) {
                        dist_data[src * L + v] = dist_data[src * L + u] + 1.0f;
                        q.push(v);
                    }
                }
            }
        }
    }
    
    return dist;
}

TensorF32 RFAADataLoader::get_bond_distances(const rfaa::TensorF32& bond_feats) {
    // Impl
    // ========== 1. 创建邻接矩阵 ==========
    // atom_bonds = (bond_feats > 0) * (bond_feats < 5)
    int L = static_cast<int>(bond_feats.shape().dims[0]);
    rfaa::TensorF32 atom_bonds({L, L});
    float* bonds_data = atom_bonds.data();
    const float* bf_data = bond_feats.data();
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            float val = bf_data[i * L + j];
            bonds_data[i * L + j] = (val > 0.0f && val < 5.0f) ? 1.0f : 0.0f;
        }
    }
    
    // ========== 2. 计算最短路径 ==========
    rfaa::TensorF32 dist_matrix = compute_shortest_path(atom_bonds);
    
    // ========== 3. 处理 inf (可选) ==========
    // 将 inf 替换为 4.0 (如果需要的话)
    float* dm_data = dist_matrix.data();
    for (int i = 0; i < L * L; i++) {
        if (std::isinf(dm_data[i])) {
            dm_data[i] = 4.0f;  // 或者保持 inf，取决于使用场景
        }
    }
    
    return dist_matrix;
    
}

TensorF32 RFAADataLoader::prepare_msa_latent(const A3MData& a3m_data) {
    // 准备 msa_latent: (B, N_clust, L, 164)
    // 参考 RFAA MSAFeaturize (data_loader_utils.py):
    //   msa_seed = cat([msa_clust_onehot(80), msa_clust_profile(80),
    //                   ins_clust(2), term_info(2)], dim=-1)   // 80+80+2+2 = 164
    // - msa_clust_onehot: token 的 one-hot (NAATOKENS=80)
    // - msa_clust_profile: 簇内各位置 AA 频率。当前无聚类 (每个簇仅 1 条序列),
    //   故 profile = 该序列自身的 one-hot。
    // - ins_clust: [seed插入统计, 簇平均插入统计]。无聚类时两者均 = 该序列自身的
    //   插入计数, 经 (2/π)·arctan(ins/3) 变换。数据来自 a3m_data.ins_matrix。
    // - term_info: N端(i==0)/C端(i==L-1) 标记。
    // 行 0 固定为 query 序列 (a3m_data.sequences[0])。
    int B = 1;  // batch size = 1
    int N_clust = std::min(max_seqs_, a3m_data.num_sequences);
    int L = std::min(max_length_, a3m_data.sequence_length);
    if (N_clust <= 0 || L <= 0) {
        return TensorF32({B, std::max(N_clust, 0), std::max(L, 0), MSA_LATENT_DIM},
                         Device::CPU);
    }

    TensorF32 msa_latent({B, N_clust, L, MSA_LATENT_DIM}, Device::CPU);
    msa_latent.zero_();
    float* data = msa_latent.data();

    // A3M 21 字母码 (与 parse_a3m 的 ALPHABET 一致): 索引 0-20, 20 = gap
    static const char* A3M_ALPHABET = "ARNDCQEGHILKMFPSTWYV-";
    const float kInsScale = (2.0f / 3.14159265358979f);  // (2/π), 用于 arctan(ins/3)

    for (int s = 0; s < N_clust; ++s) {
        const std::string& seq = (s < (int)a3m_data.sequences.size())
                                 ? a3m_data.sequences[s] : std::string();
        // 该序列的插入计数 (与 sequences 一一对应)
        const std::vector<uint8_t>* ins =
            (s < (int)a3m_data.ins_matrix.size()) ? &a3m_data.ins_matrix[s] : nullptr;
        // 将 AA 字符串转为 token 索引 (A3M 码)
        std::vector<int> toks(L, 20);  // 默认 gap
        for (int i = 0; i < L; ++i) {
            char c = (i < (int)seq.size()) ? seq[i] : '-';
            for (int a = 0; a < 21; ++a) {
                if (A3M_ALPHABET[a] == c) { toks[i] = a; break; }
            }
        }

        float* row = data + (size_t)s * L * MSA_LATENT_DIM;  // B=1
        for (int i = 0; i < L; ++i) {
            int tok = toks[i];
            if (tok < 0 || tok >= NAATOKENS) tok = NAATOKENS - 1;  // boundary cases
            float* f = row + (size_t)i * MSA_LATENT_DIM;

            // 1) one-hot (80)
            f[tok] = 1.0f;
            // 2) profile (80): 单簇时 = 自身 one-hot
            f[NAATOKENS + tok] = 1.0f;
            // 3) ins_clust (2): (2/π)·arctan(ins/3)。无聚类时 seed 与簇均值相同
            float ins_val = 0.0f;
            if (ins && i < (int)ins->size()) ins_val = static_cast<float>((*ins)[i]);
            float ins_feat = kInsScale * std::atan(ins_val / 3.0f);
            f[2 * NAATOKENS + 0] = ins_feat;  // seed 插入统计
            f[2 * NAATOKENS + 1] = ins_feat;  // 簇平均插入统计
            // 4) term_info (2): 紧跟 ins_clust 之后, 偏移 2*NAATOKENS+2
            if (i == 0)     f[2 * NAATOKENS + 2] = 1.0f;  // N 端
            if (i == L - 1) f[2 * NAATOKENS + 3] = 1.0f;  // C 端
        }
    }

    return msa_latent;
}

// ============================================================================
// prepare_msa_mask — BERT-style Masked MSA 预处理
// ============================================================================
// 输入: msa_tokens (N_seq, L), token 0-20 (20=gap)
// 输出:
//   out_true_msa : (B=1, N_seq, L) — 掩码位置保留真实 aatype token (0-20), 其余置 0
//   out_bert_mask: (B=1, N_seq, L) — 1.0=被掩码位置, 0.0=未掩码
// 实现: 以 mask_frac 概率(默认0.15)均匀随机选择位置作为掩码。
//   query 行(seq 0)不掩码, 保证结构/序列监督稳定。
void RFAADataLoader::prepare_msa_mask(
    const std::vector<std::vector<uint8_t>>& msa_tokens,
    TensorF32& out_true_msa,
    TensorF32& out_bert_mask,
    float mask_frac) {
    const int B = 1;
    // 与 prepare_msa_latent 的 N_clust = min(max_seqs_, num_sequences) 保持一致,
    // 保证 true_msa/bert_mask 的 N_seq 与 msa_logits(来自 msa, N_clust 截断) 对齐。
    const int N_seq = std::min(max_seqs_, static_cast<int>(msa_tokens.size()));
    const int L = (msa_tokens.empty()) ? 0
                  : std::min(max_length_, static_cast<int>(msa_tokens[0].size()));

    out_true_msa  = TensorF32({B, N_seq, L}, Device::CPU);
    out_bert_mask = TensorF32({B, N_seq, L}, Device::CPU);
    out_true_msa.zero_();
    out_bert_mask.zero_();

    // 确定性随机种子 (后续可改为每 epoch 重新采样)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int s = 0; s < N_seq; ++s) {
        for (int i = 0; i < L; ++i) {
            float tok = (i < (int)msa_tokens[s].size()) ? (float)msa_tokens[s][i] : 20.0f; // gap
            int64_t idx = (int64_t)s * L + i;
            out_true_msa.data()[idx] = tok;

            // 掩码: query 行 (s==0) 不掩码; 其余按概率
            if (s != 0 && dist(rng) < mask_frac) {
                out_bert_mask.data()[idx] = 1.0f;
                // 注意: 被掩码位置的特征替换为 MASK token 由调用方在
                //       构建 msa_latent 特征时处理; 这里仅产出监督标签。
            }
        }
    }
}

// ============================================================================
// prepare_chi_labels — Chi (扭转角) 监督标签预处理
// ============================================================================
// 从真实骨架坐标 coords (B,L,3,3)=[N,CA,C] 计算 7 角 (omega,phi,psi,chi1-4) 的
// (sin,cos) 与有效掩码。
//   - 骨架角 omega/phi/psi 仅需 N/CA/C, 由 dihedral 计算, mask=1。
//   - chi1-4 需侧链原子 (CB/CG...), true_coords 不含侧链, 故置 0, mask=0。
//   - 骨架角仅在同残基/相邻残基坐标有效时置 1 (近似: 全部视为有效, 因 coords 已补齐)。
// 输出:
//   out_gt_chi   : (B,L,7,2)  — 7 角 (sin,cos)
//   out_chi_mask : (B,L,7)    — 1.0=有效
void RFAADataLoader::prepare_chi_labels(
    const TensorF32& coords,
    TensorF32& out_gt_chi,
    TensorF32& out_chi_mask) {
    int B = static_cast<int>(coords.shape().dims[0]);
    int L = static_cast<int>(coords.shape().dims[1]);

    out_gt_chi   = TensorF32({B, L, 7, 2}, Device::CPU);
    out_chi_mask = TensorF32({B, L, 7}, Device::CPU);
    out_gt_chi.zero_();
    out_chi_mask.zero_();

    // 标量 dihedral: 四个三维向量 a,b,c,d → 二面角 (弧度), 返回 (sin, cos)
    auto dihedral = [](const float a[3], const float b[3],
                       const float c[3], const float d[3], float& s, float& cs) {
        // b1 = b - a, b2 = c - b, b3 = d - c
        float b1x = b[0]-a[0], b1y = b[1]-a[1], b1z = b[2]-a[2];
        float b2x = c[0]-b[0], b2y = c[1]-b[1], b2z = c[2]-b[2];
        float b3x = d[0]-c[0], b3y = d[1]-c[1], b3z = d[2]-c[2];
        // n1 = b1 × b2, n2 = b2 × b3
        float n1x = b1y*b2z - b1z*b2y, n1y = b1z*b2x - b1x*b2z, n1z = b1x*b2y - b1y*b2x;
        float n2x = b2y*b3z - b2z*b3y, n2y = b2z*b3x - b2x*b3z, n2z = b2x*b3y - b2y*b3x;
        // m1 = n1 × (b2/|b2|)
        float b2n = std::sqrt(b2x*b2x + b2y*b2y + b2z*b2z) + 1e-6f;
        float b2ux = b2x/b2n, b2uy = b2y/b2n, b2uz = b2z/b2n;
        float m1x = n1y*b2uz - n1z*b2uy, m1y = n1z*b2ux - n1x*b2uz, m1z = n1x*b2uy - n1y*b2ux;
        // x = n1·n2, y = m1·n2
        float x = n1x*n2x + n1y*n2y + n1z*n2z;
        float y = m1x*n2x + m1y*n2y + m1z*n2z;
        float ang = std::atan2(y, x);
        s  = std::sin(ang);
        cs = std::cos(ang);
    };

    // 坐标访问辅助: coords[b, l, atom(0=N,1=CA,2=C), c]
    auto atom = [&](int b, int l, int a, int c) -> float {
        if (l < 0 || l >= L) return 0.0f;
        const float* d = coords.data();
        return d[((b*L + l)*3 + a)*3 + c];
    };

    const float* mask_d = out_chi_mask.data();
    for (int b = 0; b < B; ++b) {
        for (int l = 0; l < L; ++l) {
            // 每残基 3 骨架原子坐标
            float N[3]  = {atom(b,l,0,0), atom(b,l,0,1), atom(b,l,0,2)};
            float CA[3] = {atom(b,l,1,0), atom(b,l,1,1), atom(b,l,1,2)};
            float C[3]  = {atom(b,l,2,0), atom(b,l,2,1), atom(b,l,2,2)};

            // 相邻残基原子 (越界用本残基占位, mask=0 由下面控制)
            float Np[3], Cp_1[3];  // N[l+1], C[l-1]
            {
                if (l+1 < L) { Np[0]=atom(b,l+1,0,0); Np[1]=atom(b,l+1,0,1); Np[2]=atom(b,l+1,0,2); }
                else         { Np[0]=N[0]; Np[1]=N[1]; Np[2]=N[2]; }
                if (l-1 >= 0) { Cp_1[0]=atom(b,l-1,2,0); Cp_1[1]=atom(b,l-1,2,1); Cp_1[2]=atom(b,l-1,2,2); }
                else          { Cp_1[0]=C[0]; Cp_1[1]=C[1]; Cp_1[2]=C[2]; }
            }

            // omega[l] = dihedral(CA[l], C[l], N[l+1], CA[l+1])   — 仅 l<L-1 有效
            if (l < L-1) {
                float CAn[3] = {atom(b,l+1,1,0), atom(b,l+1,1,1), atom(b,l+1,1,2)};
                float s, cs;
                dihedral(CA, C, Np, CAn, s, cs);
                int64_t base = ((b*L + l)*7 + 0)*2;
                out_gt_chi.data()[base+0] = s;
                out_gt_chi.data()[base+1] = cs;
                out_chi_mask.data()[(b*L + l)*7 + 0] = 1.0f;
            }
            // phi[l] = dihedral(C[l-1], N[l], CA[l], C[l])   — 仅 l>0 有效
            if (l > 0) {
                float s, cs;
                dihedral(Cp_1, N, CA, C, s, cs);
                int64_t base = ((b*L + l)*7 + 1)*2;
                out_gt_chi.data()[base+0] = s;
                out_gt_chi.data()[base+1] = cs;
                out_chi_mask.data()[(b*L + l)*7 + 1] = 1.0f;
            }
            // psi[l] = dihedral(N[l], CA[l], C[l], N[l+1])   — 仅 l<L-1 有效
            if (l < L-1) {
                float s, cs;
                dihedral(N, CA, C, Np, s, cs);
                int64_t base = ((b*L + l)*7 + 2)*2;
                out_gt_chi.data()[base+0] = s;
                out_gt_chi.data()[base+1] = cs;
                out_chi_mask.data()[(b*L + l)*7 + 2] = 1.0f;
            }
            // chi1-4 (索引 3..6): true_coords 无侧链原子 → mask=0, 值保持 0
            (void)mask_d;
        }
    }
}

// ============================================================================
// prepare_distogram_labels — Distogram 监督标签 (从真实坐标 binning)
// ============================================================================
// 逐 batch 调用 DistogramBins::compute_all_distogram_onehots (coords (L,3,3)),
// 并把距离 one-hot 从 61 bins (含溢出) 压缩为 60 bins (溢出 bin 并入 bin 59),
// 与 distogram_head 输出的 60 bins 对齐。同时构造 pair_mask (残基对均有效=1)。
void RFAADataLoader::prepare_distogram_labels(
    const TensorF32& coords,
    TensorF32& out_D_onehot,
    TensorF32& out_O_onehot,
    TensorF32& out_T_onehot,
    TensorF32& out_P_onehot,
    TensorF32& out_pair_mask) {
    const int B = static_cast<int>(coords.shape().dims[0]);
    const int L = static_cast<int>(coords.shape().dims[1]);

    const int D_BINS = 60, O_BINS = 36, T_BINS = 36, P_BINS = 18;
    out_D_onehot = TensorF32({B, L, L, D_BINS}, Device::CPU);
    out_O_onehot = TensorF32({B, L, L, O_BINS}, Device::CPU);
    out_T_onehot = TensorF32({B, L, L, T_BINS}, Device::CPU);
    out_P_onehot = TensorF32({B, L, L, P_BINS}, Device::CPU);
    out_pair_mask = TensorF32({B, L, L}, Device::CPU);
    out_D_onehot.zero_(); out_O_onehot.zero_(); out_T_onehot.zero_(); out_P_onehot.zero_();

    // 临时 (L,3,3) 坐标 + seq_mask(L,) (全部有效)
    TensorF32 coord_b({L, 3, 3}, Device::CPU);
    std::vector<float> seq_mask(L, 1.0f);

    for (int b = 0; b < B; ++b) {
        // 拷贝当前 batch 的 (L,3,3)
        for (int i = 0; i < L * 9; ++i) {
            coord_b.data()[i] = coords.data()[b * (L * 9) + i];
        }

        // 61-bin D 临时缓冲
        const int D_BINS_RAW = 61;
        std::vector<float> D_raw(L * L * D_BINS_RAW, 0.0f);
        std::vector<float> O_raw(L * L * O_BINS, 0.0f);
        std::vector<float> T_raw(L * L * T_BINS, 0.0f);
        std::vector<float> P_raw(L * L * P_BINS, 0.0f);

        compute_all_distogram_onehots(coord_b, seq_mask.data(),
                                      D_raw.data(), O_raw.data(), T_raw.data(), P_raw.data());

        // 写回输出: D 压缩 61→60 (溢出 bin 60 并入 bin 59)
        for (int l = 0; l < L; ++l) {
            for (int lp = 0; lp < L; ++lp) {
                for (int c = 0; c < D_BINS_RAW; ++c) {
                    float v = D_raw[(l * L + lp) * D_BINS_RAW + c];
                    int c_out = std::min(c, D_BINS - 1);  // 溢出归入最后 bin
                    out_D_onehot.data()[((b * L + l) * L + lp) * D_BINS + c_out] += v;
                }
                for (int c = 0; c < O_BINS; ++c) {
                    out_O_onehot.data()[((b * L + l) * L + lp) * O_BINS + c] = O_raw[(l * L + lp) * O_BINS + c];
                }
                for (int c = 0; c < T_BINS; ++c) {
                    out_T_onehot.data()[((b * L + l) * L + lp) * T_BINS + c] = T_raw[(l * L + lp) * T_BINS + c];
                }
                for (int c = 0; c < P_BINS; ++c) {
                    out_P_onehot.data()[((b * L + l) * L + lp) * P_BINS + c] = P_raw[(l * L + lp) * P_BINS + c];
                }
                out_pair_mask.data()[(b * L + l) * L + lp] = 1.0f;  // 所有残基对有效
            }
        }
    }
}

// ============================================================================
// prepare_ca_mask — CA 原子有效掩码
// ============================================================================
// 从真实骨架坐标 (B,L,3,3) 判断每个残基 CA 是否有效 (坐标不全为零)。
void RFAADataLoader::prepare_ca_mask(
    const TensorF32& coords,
    TensorF32& out_ca_mask) {
    const int B = static_cast<int>(coords.shape().dims[0]);
    const int L = static_cast<int>(coords.shape().dims[1]);
    out_ca_mask = TensorF32({B, L}, Device::CPU);
    out_ca_mask.zero_();

    for (int b = 0; b < B; ++b) {
        for (int l = 0; l < L; ++l) {
            // CA 原子索引 = 1, xyz 3 维; coords[b,l,1,0..2]
            const float* d = coords.data();
            float cx = d[((b * L + l) * 3 + 1) * 3 + 0];
            float cy = d[((b * L + l) * 3 + 1) * 3 + 1];
            float cz = d[((b * L + l) * 3 + 1) * 3 + 2];
            bool valid = (cx != 0.0f || cy != 0.0f || cz != 0.0f);
            out_ca_mask.data()[(b * L + l)] = valid ? 1.0f : 0.0f;
        }
    }
}

TensorF32 RFAADataLoader::prepare_msa_full(const A3MData& a3m_data) {
    // 准备 msa_full: (B, N_extra, L, 83)
    // 参考 RFAA MSAFeaturize (data_loader_utils.py):
    //   msa_extra = cat([msa_extra_onehot(80), ins_extra(1), term_info(2)], dim=-1)  // 80+1+2 = 83
    // - msa_extra_onehot: token 的 one-hot (NAATOKENS=80), 处理完整未聚类 MSA (Track 0)。
    // - ins_extra: 插入统计 (1 维), (2/π)·arctan(ins/3), 数据来自 a3m_data.ins_matrix。
    // - term_info: N端(i==0)/C端(i==L-1) 标记。
    int B = 1;  // batch size = 1
    int N_extra = std::min(max_seqs_, a3m_data.num_sequences);
    int L = std::min(max_length_, a3m_data.sequence_length);
    if (N_extra <= 0 || L <= 0) {
        return TensorF32({B, std::max(N_extra, 0), std::max(L, 0), MSA_FULL_DIM},
                         Device::CPU);
    }

    TensorF32 msa_full({B, N_extra, L, MSA_FULL_DIM}, Device::CPU);
    msa_full.zero_();
    float* data = msa_full.data();

    // A3M 21 字母码 (与 parse_a3m 的 ALPHABET 一致): 索引 0-20, 20 = gap
    static const char* A3M_ALPHABET = "ARNDCQEGHILKMFPSTWYV-";
    const float kInsScale = (2.0f / 3.14159265358979f);  // (2/π), 用于 arctan(ins/3)

    for (int s = 0; s < N_extra; ++s) {
        const std::string& seq = (s < (int)a3m_data.sequences.size())
                                 ? a3m_data.sequences[s] : std::string();
        const std::vector<uint8_t>* ins =
            (s < (int)a3m_data.ins_matrix.size()) ? &a3m_data.ins_matrix[s] : nullptr;
        std::vector<int> toks(L, 20);  // 默认 gap
        for (int i = 0; i < L; ++i) {
            char c = (i < (int)seq.size()) ? seq[i] : '-';
            for (int a = 0; a < 21; ++a) {
                if (A3M_ALPHABET[a] == c) { toks[i] = a; break; }
            }
        }

        float* row = data + (size_t)s * L * MSA_FULL_DIM;  // B=1
        for (int i = 0; i < L; ++i) {
            int tok = toks[i];
            if (tok < 0 || tok >= NAATOKENS) tok = NAATOKENS - 1;  // 防御
            float* f = row + (size_t)i * MSA_FULL_DIM;

            // 1) one-hot (80)
            f[tok] = 1.0f;
            // 2) ins_extra (1): (2/π)·arctan(ins/3), 位于 NAATOKENS (80)
            float ins_val = 0.0f;
            if (ins && i < (int)ins->size()) ins_val = static_cast<float>((*ins)[i]);
            f[NAATOKENS] = kInsScale * std::atan(ins_val / 3.0f);
            // 3) term_info (2): 紧跟 ins_extra 之后, 偏移 NAATOKENS+1
            if (i == 0)     f[NAATOKENS + 1] = 1.0f;  // N 端
            if (i == L - 1) f[NAATOKENS + 2] = 1.0f;  // C 端
        }
    }

    return msa_full;
}

TensorF32 RFAADataLoader::prepare_seq_tokens(const std::string& sequence) {
    // 准备 seq_tokens: (B, L)
    // 将氨基酸字符串转换为 token ID
    int B = 1;
    int L = static_cast<int>(sequence.length());
    
    TensorF32 seq_tokens({B, L}, Device::CPU);
    float* data = seq_tokens.data();
    
    // 简单的氨基酸到 token 的映射 (需要根据 ChemData 调整)
    for (int i = 0; i < L; ++i) {
        char aa = sequence[i];
        int token = 0;  // 默认 token
        
        // 简化映射：A=0, C=1, D=2, ...
        // 实际应该使用 ChemData().aa2long 或类似映射
        switch (aa) {
            case 'A': token = 0; break;
            case 'C': token = 1; break;
            case 'D': token = 2; break;
            case 'E': token = 3; break;
            case 'F': token = 4; break;
            case 'G': token = 5; break;
            case 'H': token = 6; break;
            case 'I': token = 7; break;
            case 'K': token = 8; break;
            case 'L': token = 9; break;
            case 'M': token = 10; break;
            case 'N': token = 11; break;
            case 'P': token = 12; break;
            case 'Q': token = 13; break;
            case 'R': token = 14; break;
            case 'S': token = 15; break;
            case 'T': token = 16; break;
            case 'V': token = 17; break;
            case 'W': token = 18; break;
            case 'Y': token = 19; break;
            default: token = 0; break;
        }
        
        data[i] = static_cast<float>(token);
    }
    
    return seq_tokens;
}

TensorF32 RFAADataLoader::prepare_coords(const std::string& sequence) {
    // 准备初始坐标: (B, L, 3, 3)
    // 简化：返回零张量或随机初始化
    int B = 1;
    int L = static_cast<int>(sequence.length());
    
    TensorF32 coords({B, L, 3, 3}, Device::CPU);
    coords.zero_();
    
    // TODO: 实际实现
    // 可以使用简单的扩展或从 PDB 加载
    
    return coords;
}

} // namespace rfaa

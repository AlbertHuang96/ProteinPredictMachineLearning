#include "rfaa/DataLoader.h"
#include <fstream>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <cmath>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>

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
            (*deleter)();
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
            (*deleter)();
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
    return (it != map.end()) ? it->second : AA_UNKNOWN;
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
    return load_from_files(a3m_path, hhr_path, sequence);
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
    rfaa::TensorF32 dih(a.shape());
    dih.shape().dims.pop_back();  // 移除最后一维
    
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
    rfaa::TensorF32 ang(a.shape());
    ang.shape().dims.pop_back();
    
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
    rfaa::TensorF32 b_vec = Ca - N;  // 需要重载 - 运算符
    rfaa::TensorF32 c_vec = C - Ca;
    rfaa::TensorF32 a_vec = cross(b_vec, c_vec);
    
    // 由于 Tensor 可能不支持运算符重载，这里用循环实现
    // 重新实现 Cb 计算
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
    rfaa::TensorF32 c6d({batch, nres, nres, 4}, 0.0f);
    
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
    rfaa::TensorF32 mask({batch, nres, nres}, 0.0f);
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
    //auto [c6d_flat, mask_flat] = xyz_to_c6d(xyz_t_reshaped, DMAX);
    C6DResult c6d_result = xyz_to_c6d(xyz_t_reshaped, DMAX);
    rfaa::TensorF32 c6d_flat = c6d_result.c6d;
    // c6d.view(B, T, L, L, 4)
    // mask.view(B, T, L, L)
    rfaa::TensorF32 c6d({B, T, L, L, 4});
    rfaa::TensorF32 mask({B, T, L, L});
    std::memcpy(c6d.data(), c6d_flat.data(), c6d_flat.numel() * sizeof(float));
    std::memcpy(mask.data(), mask_flat.data(), mask_flat.numel() * sizeof(float));
    
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
    rfaa::TensorF32 dist_clean = dist;  // 克隆
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
    
    rfaa::TensorF32 onehot(out_dims, 0.0f);
    
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
    rfaa::TensorF32 dist_onehot({B, T, L, L, num_classes}, 0.0f);
    
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
    rfaa::TensorF32 orien({B, T, L, L, 6}, 0.0f);
    
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
    rfaa::TensorF32 t2d({B, T, L, L, D}, 0.0f);
    
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

ModelInput RFAADataLoader::load_from_files(
    const std::string& a3m_path,
    const std::string& hhr_path,
    const std::string& sequence
) {
    ModelInput input;
    
    // Step 1: 解析 A3M
    std::vector<std::vector<uint8_t>> a3m_data = parse_a3m(a3m_path);
    input.msa_latent = prepare_msa_latent(a3m_data);
    input.msa_full = prepare_msa_full(a3m_data);
    input.seq_tokens = prepare_seq_tokens(sequence);

    // read templates
    // TODO: add mask to the import result of read_templates
    ReadTemplatesResult read_templates_result = read_templates(
        sequence.length(), ffdb_, hhr_path, atab_path, max_templates_);
    TensorF32 xyz_t = read_templates_result.xyz;  // (T, L, 3, 3)
    xyz_t = xyz_t.unsqueeze(0);
    TensorF32 t1d = read_templates_result.f1d;  // (T, L, 3)
    TensorF32 t0d = read_templates_result.f0d;  // (T, 3)
    t1d = t1d.unsqueeze(0);
    t0d = t0d.unsqueeze(0);

    t2d = xyz_to_t2d(xyz_t);
    // xyz_to_t2d(xyz, mask, params)
    // mask from the template reading


    input.coords = xyz_t;  // (1, T, L, 3, 3)
    input.t1d = t1d;      // (1, T, L, 3)
    input.t2d = t2d;

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

    input.tor_feat = get_torsions(xyz_t, sequence).torsions;
    
    // int L = len(sequence); ?
    input.bond_feat = get_protein_bond_feats();
    //input.dist_matrix = get_protein_dist_matrix(xyz_t);

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
    int qlen,                      // 查询序列长度
    const FFindexDB& ffdb,         // FFDB 数据库
    const std::string& hhr_fn,     // HHR 文件路径
    const std::string& atab_fn,    // ATAB 文件路径
    int n_templ = 10               // 最大模板数
) {
    // ========== 1. 调用 parse_templates ==========
    // 注意：这里需要先实现 parse_templates，返回 TemplateData
    TemplateDataInternal parsed = parse_templates(ffdb, hhr_fn, atab_fn, n_templ);
    
    int npick = std::min(n_templ, static_cast<int>(parsed.ids.size()));
    if (npick <= 0) {
        // 返回空结果
        ReadTemplatesResult result;
        result.xyz = rfaa::TensorF32({0, qlen, 3, 3});
        result.f1d = rfaa::TensorF32({0, qlen, 3});
        result.f0d = rfaa::TensorF32({0, 3});
        result.masks = rfaa::TensorF32({0, qlen, 1});
        return result;
    }
    
    // ========== 2. 创建全长度张量 ==========
    // xyz: (npick, qlen, 3, 3) - 初始化为 NaN
    rfaa::TensorF32 xyz({npick, qlen, 3, 3});
    float* xyz_data = xyz.data();
    int64_t xyz_size = npick * qlen * 3 * 3;
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    for (int64_t i = 0; i < xyz_size; i++) {
        xyz_data[i] = nan_val;
    }
    
    // f1d: (npick, qlen, 3) - 初始化为 0
    rfaa::TensorF32 f1d({npick, qlen, 3}, 0.0f);
    
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
        f0d_row.data()[2] = parsed.f0d.data()[nt * 8 + 5] / 100.0f;  // Similarity
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

std::vector<TemplateHitInput> RFAADataLoader::parse_atab(const std::string& atab_fn) {
    // 解析 .atab 文件，提取模板命中信息
    std::vector<TemplateHitInput> hits;
    
    std::ifstream file(atab_fn);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open atab file: " + atab_fn);
    }
        
    TemplateHit* current_hit = nullptr;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
            
        if (line[0] == '>') {
                // 新 hit
            hits.emplace_back();
            current_hit = &hits.back();
                // 提取名称（第一个空格前的部分）
            size_t space_pos = line.find(' ', 1);
            current_hit->name = line.substr(1, space_pos - 1);
        } else if (line.find("score") != std::string::npos || 
                line.find("dssp") != std::string::npos) {
            // 跳过 header 行
            continue;
        } else {
            // 数据行: query_idx template_idx score1 score2 score3 ...
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;
            while (iss >> token) {
                tokens.push_back(token);
            }
                
            if (tokens.size() >= 5 && current_hit) {
                int query_idx = std::stoi(tokens[0]);
                int template_idx = std::stoi(tokens[1]);
                float score1 = std::stof(tokens[2]);
                float score2 = std::stof(tokens[3]);
                float score3 = std::stof(tokens[4]);
                    
                current_hit->alignments.emplace_back(query_idx, template_idx);
                current_hit->position_scores.push_back({score1, score2, score3});
            }
        }
    }
    
    return hits;
}

TemplateDataInternal RFAADataLoader::parse_templates(
    const std::string& db_prefix,  // e.g., "pdb100_2021Mar03/pdb100_2021Mar03"
    const std::string& hhr_fn,
    const std::string& atab_fn,
    int n_templ = 10
) {
    // 1. 加载 FFDB
    FFindexDB ffdb = load_ffdb(db_prefix);
    
    // 2. 解析 .atab 和 .hhr（省略，见之前代码）
    std::vector<TemplateHitInput> hits = parse_atab(atab_fn);
    parse_hhr(hhr_fn, hits);
    
    // 3. 从 FFDB 读取模板
    for (auto& hit : hits) {
        // 查找条目
        const FFindexEntry* entry = get_entry_by_name(hit.name, ffdb.index);
        if (entry == nullptr) {
            continue;  // 跳过未找到的模板
        }
        
        // 读取条目数据（PDB 行）
        std::vector<std::string> lines = read_entry_lines(*entry, *ffdb.data);
        
        // 解析 PDB 行 -> 坐标和掩码
        //auto [xyz, mask] = parse_pdb_lines(lines);
        ParsePDBResult pdb_result = parse_pdb_lines(lines);
        hit.xyz = std::move(pdb_result.xyz);  // TensorF32 (N, 3)
        hit.mask = std::move(pdb_result.mask); // TensorF32 (N,)
    }
    
    // 
    // ========== 4. 处理 hits ==========
    std::vector<rfaa::TensorF32> all_xyz;
    std::vector<rfaa::TensorF32> all_mask;
    std::vector<rfaa::TensorF32> all_qmap;
    std::vector<rfaa::TensorF32> all_f0d;
    std::vector<rfaa::TensorF32> all_f1d;
    std::vector<std::string> ids;
    
    int counter = 0;
    
    for (auto& hit : hits) {
        if (hit.stats.empty() || hit.xyz.numel() == 0) {
            continue;
        }
        
        // qi, ti 从 alignments 提取
        int ncol = static_cast<int>(hit.alignments.size());
        if (ncol < 10) {
            continue;
        }
        
        //_,sel1,sel2 = np.intersect1d(ti, data[6], return_indices=True)
        // 创建 qi, ti 张量 (简化：假设所有对齐都有效)
        rfaa::TensorF32 qi({ncol});
        rfaa::TensorF32 ti({ncol});
        for (int i = 0; i < ncol; i++) {
            qi.data()[i] = static_cast<float>(hit.alignments[i].first);
            ti.data()[i] = static_cast<float>(hit.alignments[i].second);
        }
        
        ids.push_back(hit.name);
        
        // ========== f0d: hit.stats -> (1, 8) ==========
        {
            rfaa::TensorF32 f0d_row({1, static_cast<int>(hit.stats.size())});
            for (size_t i = 0; i < hit.stats.size(); i++) {
                f0d_row.data()[i] = hit.stats[i];
            }
            all_f0d.push_back(std::move(f0d_row));
        }
        
        // ========== f1d: position_scores -> (ncol, 3) ==========
        {
            rfaa::TensorF32 f1d_mat({ncol, 3});
            for (int i = 0; i < ncol; i++) {
                f1d_mat.data()[i * 3 + 0] = hit.position_scores[i][0];
                f1d_mat.data()[i * 3 + 1] = hit.position_scores[i][1];
                f1d_mat.data()[i * 3 + 2] = hit.position_scores[i][2];
            }
            all_f1d.push_back(std::move(f1d_mat));
        }

        //xyz.append(data[4][sel2])
        // ========== xyz: hit.xyz -> (ncol, 3) ==========
        // 简化：使用全部坐标
        all_xyz.push_back(hit.xyz);
        
        // ========== mask: hit.mask -> (ncol, 1) ==========
        all_mask.push_back(hit.mask);
        //mask.append(data[5][sel2])
        
        // ========== qmap: [qi-1, counter] -> (ncol, 2) ==========
        {
            rfaa::TensorF32 qmap_mat({ncol, 2});
            for (int i = 0; i < ncol; i++) {
                qmap_mat.data()[i * 2 + 0] = static_cast<float>(hit.alignments[i].first - 1);
                qmap_mat.data()[i * 2 + 1] = static_cast<float>(counter);
            }
            all_qmap.push_back(std::move(qmap_mat));
        }
        
        counter++;
        
        if (counter >= n_templ) {
            break;
        }
    }
    // ========== 5. 堆叠结果 (vstack) ==========
    TemplateDataInternal result;
    
    // ========== mask: vstack ==========
    {
        int64_t total_rows = 0;
        for (const auto& t : all_mask) total_rows += t.shape().dims[0];
        
        result.masks = rfaa::TensorF32({total_rows, 1});
        bool* dst = result.masks.data();
        for (const auto& t : all_mask) {
            int64_t n = t.numel();
            std::memcpy(dst, t.data(), n * sizeof(bool));
            dst += n;
        }
    }

    // ========== xyz: vstack ==========
    {
        int64_t total_rows = 0;
        for (const auto& t : all_xyz) total_rows += t.shape().dims[0];
        
        result.xyz = rfaa::TensorF32({total_rows, 3});
        float* dst = result.xyz.data();
        for (const auto& t : all_xyz) {
            int64_t n = t.numel();
            std::memcpy(dst, t.data(), n * sizeof(float));
            dst += n;
        }
    }
    
    // ========== qmap: vstack ==========
    {
        int64_t total_rows = 0;
        for (const auto& t : all_qmap) total_rows += t.shape().dims[0];
        
        result.qmap = rfaa::TensorF32({total_rows, 2});
        float* dst = result.qmap.data();
        for (const auto& t : all_qmap) {
            int64_t n = t.numel();
            std::memcpy(dst, t.data(), n * sizeof(float));
            dst += n;
        }
    }
    
    // ========== f0d: vstack ==========
    {
        result.f0d = rfaa::TensorF32({counter, 8});
        float* dst = result.f0d.data();
        for (const auto& t : all_f0d) {
            int64_t n = t.numel();
            std::memcpy(dst, t.data(), n * sizeof(float));
            dst += n;
        }
    }
    
    // ========== f1d: vstack ==========
    {
        int64_t total_rows = 0;
        for (const auto& t : all_f1d) total_rows += t.shape().dims[0];
        
        result.f1d = rfaa::TensorF32({total_rows, 3});
        float* dst = result.f1d.data();
        for (const auto& t : all_f1d) {
            int64_t n = t.numel();
            std::memcpy(dst, t.data(), n * sizeof(float));
            dst += n;
        }
    }
    
    result.ids = ids;
    
    return result;
    
}

// ========== 2. 解析 .hhr 文件 ==========
void RFAADataLoader::parse_hhr(const std::string& hhr_fn, std::vector<TemplateHit>& hits)
{
    std::ifstream file(hhr_fn);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open hhr file: " + hhr_fn);
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
                
            if (hit_idx < hits.size()) {
                hits[hit_idx].stats = stats;
            }
            hit_idx++;
        }
    }
}

std::vector<std::vector<uint8_t>> RFAADataLoader::parse_a3m(const std::string& a3m_path) {
    std::vector<std::vector<uint8_t>> msa;
    
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
        
        // 转换：移除小写，大写转整数
        std::vector<uint8_t> seq;
        seq.reserve(line.size());
        for (char c : line) {
            if (std::isupper(c)) {
                seq.push_back(char_map[static_cast<uint8_t>(c)]);
            } else if (c == '-') {
                seq.push_back(20); // gap
            }
            // 小写字母跳过
        }
        
        if (!seq.empty()) {
            msa.push_back(std::move(seq));
        }
    }
    
    return msa;
}

TensorF32 RFAADataLoader::a3m_to_msa_features(const A3MData& a3m_data, int max_seqs, int max_length) {

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


HHRData RFAADataLoader::parse_hhr(const std::string& hhr_path) {
    HHRData result;
    result.num_templates = 0;
    
    std::ifstream file(hhr_path);
    if (!file.is_open()) {
        throw RFAAError("HHRParser: cannot open file: " + hhr_path);
    }
    
    std::string line;
    bool in_hits = false;
    
    while (std::getline(file, line)) {
        // 跳过空行
        if (line.empty()) continue;
        
        // 检测命中区域开始 (通常以数字开头)
        if (!in_hits && std::isdigit(line[0])) {
            in_hits = true;
        }
        
        if (in_hits) {
            // 解析命中行
            // 格式: "  1  pdb_id  chain  prob  evalue  ..."
            std::istringstream iss(line);
            int rank;
            std::string pdb_id, chain_id;
            float prob, evalue;
            int q_start, q_end, t_start, t_end;
            std::string alignment;
            
            iss >> rank >> pdb_id >> chain_id >> prob >> evalue;
            
            HHRData::TemplateHit hit;
            hit.name = pdb_id + "_" + chain_id;
            hit.pdb_id = pdb_id;
            hit.chain_id = chain_id;
            hit.probability = prob;
            hit.evalue = evalue;
            // TODO: 解析比对区域和比对字符串
            
            result.hits.push_back(hit);
            result.num_templates++;
        }
    }
    
    file.close();
    return result;
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
    rfaa::TensorF32 tors_mask({B, L, 10}, 1.0f);
    
    // ========== 2. tors_planar: TYR chi 3 should be planar ==========
    rfaa::TensorF32 tors_planar({B, L, 10}, 0.0f);
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L; l++) {
            int seq_val = static_cast<int>(seq.data()[b*L + l]);
            if (seq_val == 18) {  // TYR = 18
                tors_planar.data()[(b*L + l)*10 + 5] = 1.0f;  // chi 3 (index 5)
            }
        }
    }
    
    // ========== 3. 理想化坐标 ==========
    rfaa::TensorF32 xyz = xyz_in;  // 克隆（这里假设调用者会管理）
    
    // Rs, Ts = rigid_from_3_points(N, Ca, C)
    rfaa::TensorF32 N = xyz.select(2, 0);  // (B, L, 3)
    rfaa::TensorF32 Ca = xyz.select(2, 1);
    rfaa::TensorF32 C = xyz.select(2, 2);
    
    auto [Rs, Ts] = rigid_from_3_points(N, Ca, C);
    
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
    rfaa::TensorF32 torsions({B, L, 10, 2}, 0.0f);
    
    // omega: torsions[:,:-1,0,:] = th_dih(Ca[i], C[i], N[i+1], Ca[i+1])
    for (int b = 0; b < B; b++) {
        for (int l = 0; l < L-1; l++) {
            // 需要提取 Ca[b,l], C[b,l], N[b,l+1], Ca[b,l+1]
            // 简化：假设已有 th_dih 函数支持不同索引
            // 这里省略具体实现...
        }
    }
    
    // phi: torsions[:,1:,1,:] = th_dih(C[i-1], N[i], Ca[i], C[i])
    // psi: torsions[:,:,2,:] = -th_dih(N[i], Ca[i], C[i], N[i+1])
    // chis: torsions[:,:,3:7,:] = th_dih(ti0, ti1, ti2, ti3)
    // CB bend, CB twist, CG bend...
    
    // ========== 5. 处理 NaN ==========
    for (int i = 0; i < torsions.numel(); i += 2) {
        if (std::isnan(torsions.data()[i])) {
            torsions.data()[i] = 1.0f;      // sin = 0 -> angle = 0
            torsions.data()[i+1] = 0.0f;   // cos = 1
        }
    }
    
    // ========== 6. torsions_alt = torsions * -1 (可翻转的) ==========
    rfaa::TensorF32 torsions_alt = torsions;
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
    
    return {torsions, torsions_alt, tors_mask, tors_planar};
}

void init_torsion_indices() {
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
    bond_feats.fill(0.0f);
    
    // 设置相邻残基之间的键（值为 5）
    // 正向: (0,1), (1,2), ..., (L-2, L-1)
    // 反向: (1,0), (2,1), ..., (L-1, L-2)
    for (int i = 0; i < protein_L - 1; i++) {
        //bond_feats.at({i, i + 1}) = 5.0f;
        //bond_feats.at({i + 1, i}) = 5.0f;
    }
    
    return bond_feats;
}

TensorF32 RFAADataLoader::prepare_msa_latent(const A3MData& a3m_data) {
    // 准备 msa_latent: (B, N_clust, L, 164)
    // 简化：返回零张量
    int B = 1;  // batch size = 1
    int N_clust = std::min(max_seqs_, a3m_data.num_sequences);
    int L = std::min(max_length_, a3m_data.sequence_length);
    
    TensorF32 msa_latent({B, N_clust, L, MSA_LATENT_DIM}, Device::CPU);
    msa_latent.zero_();
    
    // TODO: 实际实现
    // 1. 将序列转换为 embedding
    // 2. 添加位置编码
    // 3. 返回 (B, N_clust, L, 164)
    
    return msa_latent;
}

TensorF32 RFAADataLoader::prepare_msa_full(const A3MData& a3m_data) {
    // 准备 msa_full: (B, N_extra, L, 83)
    // 简化：返回零张量
    int B = 1;
    int N_extra = std::min(max_seqs_, a3m_data.num_sequences);
    int L = std::min(max_length_, a3m_data.sequence_length);
    
    TensorF32 msa_full({B, N_extra, L, MSA_FULL_DIM}, Device::CPU);
    msa_full.zero_();
    
    // TODO: 实际实现 (类似 A3MParser::to_msa_features)
    
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

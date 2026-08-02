#pragma once

#include "Model.h"
#include "Tensor.h"
#include <string>
#include <vector>
#include <memory>

namespace rfaa {

// ============================================================================
// 数据结构定义
// ============================================================================

/**
 * @brief A3M 解析结果
 * 
 * 包含从 A3M 文件中解析出的 MSA 数据
 */
struct A3MData {
    std::string query_sequence;                    // 查询序列 (字符串)
    std::vector<std::string> sequences;           // 所有 MSA 序列
    std::vector<float> deletion_probabilities;     // 删除概率 (用于特征计算)
    int num_sequences;                           // 序列数量
    int sequence_length;                         // 序列长度
};

/**
 * @brief HHR 解析结果
 * 
 * 包含从 HHR 文件中解析出的模板搜索结果
 */

struct TemplateHit {
    std::string name;                      // 模板名称
    std::string pdb_id;                   // PDB ID
    std::string chain_id;                 // 链 ID
    float probability;                    // 概率/得分
    float evalue;                       // E-value
    int query_start;                    // 查询起始位置
    int query_end;                      // 查询结束位置
    int template_start;                 // 模板起始位置
    int template_end;                   // 模板结束位置
    std::string alignment;              // 比对字符串
    std::vector<float> stats;           // 统计数据 (Probab, E-value, Score, ...)
};

struct HHRData {
    std::vector<TemplateHit> hits;           // 模板命中列表
    int num_templates;                      // 模板数量
    std::string query_sequence;              // 查询序列
};

struct FFindexDB;  // 前向声明，实现在 DataLoader.cpp

struct TemplateDataInternal {
    rfaa::TensorF32 xyz;      // (total_atoms, 3)
    rfaa::TensorF32 masks;    // (total_atoms, 1)
    rfaa::TensorF32 qmap;     // (total_alignments, 2) - [query_idx, template_hit_idx]
    rfaa::TensorF32 f0d;      // (n_templates, 8) - per-hit stats
    rfaa::TensorF32 f1d;      // (total_alignments, 3) - per-position scores
    std::vector<std::string> ids;  // template names
};

struct TemplateHitInput {
    std::string name;
    std::vector<std::pair<int, int>> alignments;  // (query_idx, template_idx)
    std::vector<std::array<float, 3>> position_scores;  // (score1, score2, score3)
    std::vector<float> stats;  // [Probab, E-value, Score, Aligned_cols, Identities, Similarity, Sum_probs, Template_Neff]
    rfaa::TensorF32 xyz;  // (N_atoms, 3)
    rfaa::TensorF32 mask;  // (N_atoms,)
};

// 对应 Python: read_templates(qlen, ffdb, hhr_fn, atab_fn, n_templ=10)
struct ReadTemplatesResult {
    rfaa::TensorF32 xyz;   // (npick, qlen, 3, 3) - N,CA,C 坐标
    rfaa::TensorF32 masks; // (npick, qlen, 1) - 掩码
    rfaa::TensorF32 f1d;   // (npick, qlen, 3) - 位置特征
    rfaa::TensorF32 f0d;   // (npick, 3) - 全局特征 [Probab/100, Identities/100, Similarity]
    std::vector<std::string> ids;  // 模板 ID
};

/**
 * @brief 模板结构数据
 * 
 * 包含模板的 3D 坐标和特征
 */
struct TemplateData {
    TensorF32 t1d;                        // (T, L, 80) 模板 1D 特征
    TensorF32 t2d;                        // (T, L, L, ...) 模板 2D 特征
    TensorF32 coords;                     // (T, L, 3) 模板坐标
    //TensorF32 masks;                      // (T, L, 3) 模板掩码
    std::vector<std::string> template_names; // 模板名称列表
};

struct TorsionResult {
    rfaa::TensorF32 torsions;      // (B, L, 10, 2)
    rfaa::TensorF32 torsions_alt;  // (B, L, 10, 2)
    rfaa::TensorF32 tors_mask;     // (B, L, 10)
    rfaa::TensorF32 tors_planar;   // (B, L, 10) bool
};


// ============================================================================
// 外部工具调用 (可选，通过系统调用或 Python C API)
// ============================================================================

/**
 * @brief 外部工具包装器
 * 
 * 封装对 HHblits、HHsearch 等外部工具的调用
 */
class ExternalTools {
public:
    /**
     * @brief 运行 HHblits 进行 MSA 搜索
     * 
     * @param sequence 查询序列 (FASTA 格式字符串)
     * @param database_path HHblits 数据库路径
     * @param n_iter 迭代次数
     * @param e_value E-value 阈值
     * @return std::string 输出的 A3M 文件路径
     */
    static std::string run_hhblits(
        const std::string& sequence,
        const std::string& database_path,
        int n_iter = 3,
        float e_value = 0.001f
    );
    
    /**
     * @brief 运行 HHsearch 进行模板搜索
     * 
     * @param a3m_path 输入 A3M 文件路径
     * @param database_path HHsearch 数据库路径 (PDB)
     * @param n_templates 模板数量
     * @return std::string 输出的 HHR 文件路径
     */
    static std::string run_hhsearch(
        const std::string& a3m_path,
        const std::string& database_path,
        int n_templates = 4
    );
    
private:
    ExternalTools() = delete;
};

// ============================================================================
// 主数据加载器
// ============================================================================

/**
 * @brief RFAA 数据加载器
 * 
 * 协调整个数据准备流程：
 * 1. 运行 HHblits 获取 MSA
 * 2. 运行 HHsearch 获取模板
 * 3. 解析 A3M 和 HHR 文件
 * 4. 提取特征并组装 ModelInput
 */
class RFAADataLoader {
public:
    /**
     * @brief 构造数据加载器
     * 
     * @param hhblits_db HHblits 数据库路径
     * @param hhsearch_db HHsearch 数据库路径 (PDB)
     * @param max_seqs 最大 MSA 序列数
     * @param max_templates 最大模板数
     * @param max_length 最大序列长度
     */
    RFAADataLoader(
        const std::string& hhblits_db,
        const std::string& hhsearch_db,
        int max_seqs = 512,
        int max_templates = 4,
        int max_length = 2048
    );
    
    /**
     * @brief 加载数据并准备 ModelInput
     * 
     * @param sequence 查询序列 (氨基酸字符串)
     * @return ModelInput 模型输入数据
     */
    ModelInput load(const std::string& sequence);
    
    /**
     * @brief 从现有文件加载 (不运行外部工具)
     * 
     * @param a3m_path A3M 文件路径
     * @param hhr_path HHR 文件路径
     * @param sequence 查询序列
     * @return ModelInput 模型输入数据
     */
    ModelInput load_from_files(
        const std::string& a3m_path,
        const std::string& hhr_path,
        const std::string& sequence,
        const std::string& csv_path = ""  // 可选: CSV mapping 文件 → true_coords
    );

    // 从 CSV mapping 文件加载真实坐标 (ground truth)
    // 返回: TensorF32 ({B=1, L, 3, 3}) — [N, CA, C] × [x, y, z]
    // 兼容每个残基原子存储情况不一的 CSV (缺失原子回退到 CA)
    static TensorF32 parse_csv_true_coords(
        const std::string& csv_path,
        int expected_L = -1);  // expected_L < 0 表示以 CSV 行数为准

    ReadTemplatesResult read_templates(
    int qlen,                      // 查询序列长度
    const FFindexDB& ffdb,         // FFDB 数据库
    const std::string& hhr_fn,     // HHR 文件路径
    const std::string& atab_fn,    // ATAB 文件路径
    int n_templ = 10               // 最大模板数
    );

    TemplateDataInternal parse_templates(
    const std::string& db_prefix,  // e.g., "pdb100_2021Mar03/pdb100_2021Mar03"
    const std::string& hhr_fn,
    const std::string& atab_fn,
    int n_templ = 10
    );

    std::vector<TemplateHit> parse_atab(const std::string& atab_fn);

    /**
     * @brief 解析 A3M 文件
     * 
     * @param a3m_path A3M 文件路径
     * @return A3MData 解析结果
     */
    static std::vector<std::vector<uint8_t>> parse_a3m(const std::string& a3m_path);

    static TensorF32 a3m_to_msa_features(
        const std::vector<std::vector<uint8_t>>& a3m_data,
        int max_seqs = 512,
        int max_length = 2048
    );

    /**
     * @brief 解析 HHR 文件
     * 
     * @param hhr_path HHR 文件路径
     * @return HHRData 解析结果
     */
    static HHRData parse_hhr(const std::string& hhr_path);
    
    static TemplateData hhr_extract_template_features(
        const HHRData& hhr_data,
        const std::string& query_sequence,
        int num_templates = 4,
        int max_length = 2048
    );
    
private:
    std::string hhblits_db_;
    std::string hhsearch_db_;
    int max_seqs_;
    int max_templates_;
    int max_length_;

    void init_torsion_indices();

    // 扭转角索引: (NAATOKENS, NTOTALDOFS, 4)
    // 使用 int8_t 存储原子索引（可能为负表示跨残基）
    int8_t torsion_indices[NAATOKENS][NTOTALDOFS][4];
    
    // 是否可以翻转: (NAATOKENS, NTOTALDOFS)
    bool torsion_can_flip[NAATOKENS][NTOTALDOFS];
    
    // 氨基酸长格式原子列表 (每个氨基酸最多14个原子)
    // aa2long[i] = ["N","CA","C","O","CB",...]
    std::vector<std::vector<std::string>> aa2long;
    std::vector<std::vector<std::string>> aa2longalt;
    
    // 扭转角定义: torsions[i][j] = [atom1, atom2, atom3, atom4] 或 None
    std::vector<std::vector<std::vector<std::string>>> torsions;

    TensorF32 get_protein_bond_feats(int protein_L);
    TensorF32 get_bond_distances(const TensorF32& bond_feats);
    
    TorsionResult get_torsions(
        const TensorF32& xyz_in,
        const TensorF32& seq,
        const std::vector<std::vector<std::vector<int>>>& torsion_indices,
        const std::vector<std::vector<bool>>& torsion_can_flip,
        const std::vector<std::vector<std::array<float, 2>>>& ref_angles);
    
    // 内部辅助函数
    TensorF32 prepare_msa_latent(const A3MData& a3m_data);
    TensorF32 prepare_msa_full(const A3MData& a3m_data);
    TensorF32 prepare_seq_tokens(const std::string& sequence);
    TensorF32 prepare_coords(const std::string& sequence);  // 初始坐标 (可选)
};

} // namespace rfaa

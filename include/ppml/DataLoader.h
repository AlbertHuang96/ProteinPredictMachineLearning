#pragma once

// ============================================================================
// RF2/PPML 80 类 one-hot token 类型 (ChemicalData::num2aa, NAATOKENS = 20+2+10+1+47)
// RF2/PPML 80 one-hot token types (ChemicalData::num2aa, NAATOKENS = 20+2+10+1+47)
// 这是 aatype / t1d 等 one-hot 特征的单一事实来源, 与 Python rf2aa/chemical.py 对齐:
// This is the single source of truth for one-hot features such as aatype/t1d, aligned with Python rf2aa/chemical.py:
//
//   Range      | Count | Content
//   -----------|-------|----------------------------------------------------------
//   0–19       | 20    | 标准氨基酸: ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE
//             |       |   Standard amino acids: ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE
//             |       |   LEU LYS MET PHE PRO SER THR TRP TYR VAL
//   20         | 1     | UNK  (未知残基)  / UNK (unknown residue)
//   21         | 1     | MAS  (蛋白质 mask token)  / MAS (protein mask token)
//   22–26      | 5     | DNA 碱基 + 未知:  DA DC DG DT DX
//             |       |   DNA bases + unknown: DA DC DG DT DX
//   27–31      | 5     | RNA 碱基 + 未知:  RA RC RG RU RX
//             |       |   RNA bases + unknown: RA RC RG RU RX
//   32         | 1     | HIS_D (仅用于 cart_bonded)  / HIS_D (only used for cart_bonded)
//   33–79      | 47    | 重原子元素 (配体/小分子): Al As Au B Be Br C Ca Cl Co Cr Cu
//             |       |   Heavy-atom element types (ligands/small molecules): Al As Au B Be Br C Ca Cl Co Cr Cu
//             |       |   F Fe Hg I Ir K Li Mg Mn Mo N Ni O Os P Pb Pd Pr Pt Re Rh
//             |       |   Ru S Sb Se Si Sn Tb Te U W V Y Zn ATM
//
//  索引常量: UNKINDEX=20, MASKINDEX=21 (蛋白), MASKINDEXDNA=26, MASKINDEXRNA=31。
//  Index constants: UNKINDEX=20, MASKINDEX=21 (protein), MASKINDEXDNA=26, MASKINDEXRNA=31.
//  类别分组 (nucleic_compatibility_utils.mol_class_3letter):
//  Class grouping (nucleic_compatibility_utils.mol_class_3letter):
//    protein = [0:22]+HIS_D(32),  dna = [22:27],  rna = [27:32],  atom = [33:80]。
//  模板特征 t1d 使用其中前 22 类 (aa one-hot 0:20 + UNK/MASK), 见 build_template_features。
//  The template feature t1d uses the first 22 classes (aa one-hot 0:20 + UNK/MASK), see build_template_features.
// ============================================================================

#include "Model.h"
#include "Tensor.h"
#include <string>
#include <vector>
#include <set>
#include <utility>
#include <memory>

namespace ppml {

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
    std::vector<std::vector<uint8_t>> ins_matrix; // (N_seq, L): 每个对齐列后的插入计数 (与 sequences 对齐)
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
    ppml::TensorF32 xyz;      // (total_atoms, 3)
    ppml::TensorF32 masks;    // (total_atoms, 1)
    ppml::TensorF32 qmap;     // (total_alignments, 2) - [query_idx, template_hit_idx]
    ppml::TensorF32 f0d;      // (n_templates, 8) - per-hit stats
    ppml::TensorF32 f1d;      // (total_alignments, 3) - per-position scores
    std::vector<std::string> ids;  // template names
};

struct TemplateHitInput {
    std::string name;
    std::vector<std::pair<int, int>> alignments;  // (query_idx, template_idx)
    std::vector<std::array<float, 3>> position_scores;  // (score1, score2, score3)
    std::vector<float> stats;  // [Probab, E-value, Score, Aligned_cols, Identities, Similarity, Sum_probs, Template_Neff]
    ppml::TensorF32 xyz;  // (N_atoms, 3)
    ppml::TensorF32 mask;  // (N_atoms,)
};

// 对应 Python: read_templates(qlen, ffdb, hhr_fn, atab_fn, n_templ=10)
struct ReadTemplatesResult {
    ppml::TensorF32 xyz;   // (npick, qlen, 3, 3) - N,CA,C 坐标
    ppml::TensorF32 masks; // (npick, qlen, 1) - 掩码
    ppml::TensorF32 f1d;   // (npick, qlen, 3) - 位置特征
    ppml::TensorF32 f0d;   // (npick, 3) - 全局特征 [Probab/100, Identities/100, Similarity]
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
    ppml::TensorF32 torsions;      // (B, L, 10, 2)
    ppml::TensorF32 torsions_alt;  // (B, L, 10, 2)
    ppml::TensorF32 tors_mask;     // (B, L, 10)
    ppml::TensorF32 tors_planar;   // (B, L, 10) bool
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
 * @brief PPML 数据加载器
 * 
 * 协调整个数据准备流程：
 * 1. 运行 HHblits 获取 MSA
 * 2. 运行 HHsearch 获取模板
 * 3. 解析 A3M 和 HHR 文件
 * 4. 提取特征并组装 ModelInput
 */
class PPMLDataLoader {
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
    PPMLDataLoader(
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
     * @param sequence 查询序列
     * @param csv_path 可选: CSV mapping 文件 → true_coords
     * @param hhr_path 可选: HHR 模板文件 → 默认空 (暂未接入模板)
     * @return ModelInput 模型输入数据
     */
    ModelInput load_from_files(
        const std::string& a3m_path,
        const std::string& sequence,
        const std::string& csv_path = "",   // 可选: CSV mapping 文件 (或含多个 *_mapping_results.csv 的目录) → true_coords
        const std::string& template_dir = "", // 可选: 模板结构目录 (cif/pdb), 自动过滤与真实值重复的 PDB id
        const std::string& hhr_path = ""    // 可选: HHR 模板文件 → 默认空
    );

    /**
     * @brief 从 FASTA 文件读取查询序列
     * 
     * 只读取文件中的第一条序列 (第一个 '>' 头之后的序列行拼接)。
     * 跳过注释行 (以 ';' 开头) 与空行, 忽略除第一个头以外的其他记录。
     * 
     * @param fasta_path FASTA 文件路径
     * @return std::string 第一个序列的氨基酸字符串 (不包含任何头/空白)
     * @throws std::runtime_error 文件无法打开或文件中无任何序列
     */
    static std::string read_fasta_first_sequence(const std::string& fasta_path);

    // 从 CSV mapping 文件加载真实坐标 (ground truth)
    // 返回: TensorF32 ({B=1, L, 3, 3}) — [N, CA, C] × [x, y, z]
    // 兼容每个残基原子存储情况不一的 CSV (缺失原子回退到 CA)
    static TensorF32 parse_csv_true_coords(
        const std::string& csv_path,
        int expected_L = -1);  // expected_L < 0 表示以 CSV 行数为准

    // 列出目录下所有 *_mapping_results.csv 文件 (供多结构域 ground truth 合并)
    static std::vector<std::string> list_csv_mapping_files(const std::string& dir);

    // 合并多个 CSV mapping 为单个 (1, L, 3, 3) 真实坐标。
    // 每个 CSV 代表一个 PDB 结构域, 覆盖查询序列的一段 (FASTA_Pos 为 1 索引)。
    // 重叠残基: 以第一个覆盖该残基的 CSV 为准 (取首个非零有效坐标)。
    // 未被任何 CSV 覆盖的残基坐标保持 0 (供 ca_mask 排除)。
    static TensorF32 parse_csv_true_coords_multi(
        const std::vector<std::string>& csv_paths,
        int L);

    // 从目录收集所有 PDB id (小写, 用于模板过滤): 扫描 csv 文件名 / *.pdb / *.cif
    static std::set<std::string> collect_ground_truth_pdb_ids(const std::string& dir);

    // 列出目录下所有结构文件 (*.cif / *.pdb), 返回 {路径, 扩展名}
    static std::vector<std::pair<std::string, std::string>> list_structure_files(const std::string& dir);

    // 解析单个结构文件 (cif/pdb), 提取指定链的骨架坐标 (N, CA, C, O) 与残基数。
    // out_coords: (N_res*4*3) 展平 [res][atom(0..3)][xyz(0..2)]
    static void parse_template_structure(
        const std::string& path,           // *.cif 或 *.pdb
        const std::string& chain,          // 目标链 id (大小写不敏感), 空则取第一条链
        std::vector<float>& out_coords,
        int& out_nres);

    // 从模板目录加载模板结构, 过滤掉与真实值重复的 PDB id。
    // 填充 input.template_coords / template_ids / template_chains / template_residue_counts。
    // 链: 优先从同目录 <pdb>_<chain>_coords.npy 推断, 否则取文件第一条链。
    static void load_templates_from_dir(
        const std::string& template_dir,
        const std::set<std::string>& exclude_pdb_ids,
        ModelInput& input,
        int max_templates = 4);

    // ===== 模板特征构建 (t1d / t2d / tor_feat / template_mask) =====
    // 从 *_mapped.csv (uniprot_pos → CA 坐标, 即模板残基到全长查询序列的对齐) 构建模板特征,
    // 填充 input.t1d (B,T,L,80) / input.t2d (B,T,L,L,64) / input.tor_feat (B,T,L,30) /
    // input.template_mask (B,T,L)。
    //
    // 处理逻辑:
    //   1. 对每个模板的 *_mapped.csv, 逐行读取 (uniprot_pos, template_orig_pos, x,y,z) →
    //      建立 "查询序列位置 → 模板残基 + CA 坐标" 的对齐映射。
    //   2. t1d: [0:20] aatype one-hot(模板残基, 放到对应查询位置), [20] template_mask,
    //      [21:24] 伪 β(CA) 坐标, [24] has_pseudo_beta, [25:80] 0(扭转角独立在 tor_feat)。
    //   3. tor_feat: 10 个扭转角 × (sin, cos, mask) = 30 (从结构 backbone 计算; 仅有 CA 时置 0)。
    //   4. t2d: 伪 β 距离/方向特征 (CA-based) 填 64 维, 未覆盖残基对全 0。
    //   5. template_mask: 被 mapped 覆盖的残基 = 1, 其余 = 0。
    //   unmapped 位置全部零填充并用 template_mask 标记 (0), 避免把模板坐标错放到 N 端。
    static void build_template_features(
        const std::string& template_dir,
        const std::vector<std::string>& template_ids,   // 已过滤后的模板 id (小写)
        const std::string& query_sequence,              // 全长查询序列
        ModelInput& input,
        int max_templates = 4);

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
     * @param ins_matrix 可选输出: 插入计数矩阵 (N_seq, L), 与返回的 MSA 一一对应
     * @return MSA 整数 token (N_seq, L), 0-20 (20=gap)
     */
    static std::vector<std::vector<uint8_t>> parse_a3m(
        const std::string& a3m_path,
        std::vector<std::vector<uint8_t>>* ins_matrix = nullptr);

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

    // BERT-style Masked MSA 预处理:
    // 从 MSA token (N_seq, L, token 0-20) 随机掩码一部分位置 (默认 ~15%)。
    // 返回: (B=1, N_seq, L) 的 true_msa (被掩码位置的真实 aatype) 与 bert_mask (1.0=掩码)。
    // 掩码位置在 msa_latent 特征中也应被替换为 MASK token (由调用方处理)；
    // 本函数仅负责产出监督标签 true_msa 与 bert_mask。
    void prepare_msa_mask(
        const std::vector<std::vector<uint8_t>>& msa_tokens,
        TensorF32& out_true_msa,
        TensorF32& out_bert_mask,
        float mask_frac = 0.15f);

    // Chi (扭转角) 监督标签预处理:
    // 从真实骨架坐标 coords (B,L,3,3) = [N,CA,C] 计算扭转角 sin/cos 与有效掩码。
    //   out_gt_chi   : (B,L,7,2) — 7 角 (omega,phi,psi,chi1-4) 的 (sin,cos)。
    //                 骨架角 (omega/phi/psi) 由 N/CA/C 计算；chi1-4 需侧链原子,
    //                 true_coords 不含侧链, 故 chi1-4 置 0 (mask=0)。
    //   out_chi_mask : (B,L,7)   — 1.0=有效 (骨架角), 0.0=无效 (chi1-4)。
    static void prepare_chi_labels(
        const TensorF32& coords,            // (B,L,3,3)
        TensorF32& out_gt_chi,              // (B,L,7,2)
        TensorF32& out_chi_mask);           // (B,L,7)

    // Distogram 监督标签预处理:
    // 从真实骨架坐标 coords (B,L,3,3) 计算 4 组 one-hot (D 60/Ω 36/Θ 36/Φ 18)
    // 与 pair_mask (B,L,L)。coords 支持 B>1 (每 batch 独立计算)。
    static void prepare_distogram_labels(
        const TensorF32& coords,            // (B,L,3,3)
        TensorF32& out_D_onehot,            // (B,L,L,60)
        TensorF32& out_O_onehot,            // (B,L,L,36)
        TensorF32& out_T_onehot,            // (B,L,L,36)
        TensorF32& out_P_onehot,            // (B,L,L,18)
        TensorF32& out_pair_mask);          // (B,L,L)

    // CA 原子有效掩码预处理:
    // 从真实骨架坐标 (B,L,3,3) 判断每个残基 CA 是否有效 (坐标非零)。
    static void prepare_ca_mask(
        const TensorF32& coords,            // (B,L,3,3)
        TensorF32& out_ca_mask);            // (B,L)
};

} // namespace ppml

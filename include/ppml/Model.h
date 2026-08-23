#pragma once

#include "Track.h"
#include "Attention.h"
#include "ppml/SE3Transformer.h"
#include "ppml/PositionalEncoding.h"
#include <string>
#include <memory>

#include "ppml/Embedding.h"
#include "ppml/Backend.h"

namespace ppml {

// PPML 模型配置
struct PPMLConfig {
    // 维度
    int d_msa = D_MSA;
    int d_pair = D_PAIR;
    int d_state = D_STATE;
    int d_t1d = D_T1D;
    
    // 结构
    int n_extra_blocks = N_EXTRA_BLOCKS;
    int n_main_blocks = N_MAIN_BLOCKS;
    int n_refine_blocks = N_REFINE_BLOCKS;
    int n_heads = N_HEAD;
    
    // SE3
    SE3Config se3_config;
    
    // 其他
    int max_seq_len = 2048;
    int max_n_seq = 512;
    int max_n_templ = 4;
    
    PPMLConfig();
};

// 模型输入
struct ModelInput {
    TensorF32 msa_latent;    // (B, N_clust, L, 164)
    TensorF32 msa_full;      // (B, N_extra, L, 83) - optional
    TensorF32 seq_tokens;    // (B, L) - 查询序列 token
    TensorF32 t1d;           // (B, T, L, 80) - 模板特征
    TensorF32 t2d;           // (B, T, L, L, 64) - 模板 2D 特征
    TensorF32 coords;        // (B, L, 3, 3) - 初始 Ca 坐标 (可选)
    TensorF32 true_coords;   // (B, L, 3, 3) - 真实坐标 (ground truth, 从 CSV 加载) ← 新增
    TensorF32 tor_feat;      // (B, T, L, 30) - 模板扭转角特征 (10 torsion × sin/cos/mask)
    TensorF32 template_mask; // (B, T, L)   - 模板有效掩码 (1.0=该残基有模板坐标, 0.0=缺失/零填充)
    TensorF32 bond_feats;    // (B, L, L, d_bond) - 键特征
    TensorF32 dist_matrix;   // (B, L, L) - 距离矩阵
    TensorF32 same_chain;    // (B, L, L) - 同链掩码
    TensorI64 residx;        // (B, L) - 残基索引

    // ---- Masked MSA 监督 (BERT-style) ----
    TensorF32 true_msa;      // (B, N_clust, L) - 被掩码位置的真实 aatype token (0-20)
    TensorF32 bert_mask;     // (B, N_clust, L) - 掩码标记 (1.0=被掩码, 0.0=未掩码)

    // ---- Chi (扭转角) 监督 ----
    TensorF32 gt_chi;        // (B, L, 7, 2) - 真实扭转角 (omega,phi,psi,chi1-4) 的 sin/cos
    TensorF32 chi_mask;      // (B, L, 7)    - chi 角有效掩码 (1.0=有效, 0.0=无效)

    // ---- Distogram 监督 (从真实坐标 binning) ----
    TensorF32 D_onehot;      // (B, L, L, 60) - 距离 one-hot (Cβ-Cβ, 60 bins)
    TensorF32 O_onehot;      // (B, L, L, 36) - Ω 二面角 one-hot
    TensorF32 T_onehot;      // (B, L, L, 36) - Θ 二面角 one-hot
    TensorF32 P_onehot;      // (B, L, L, 18) - Φ 平面角 one-hot
    TensorF32 pair_mask;     // (B, L, L)     - pair 有效掩码 (残基对均有效=1)

    // ---- pLDDT 监督 ----
    TensorF32 ca_mask;       // (B, L)        - CA 原子有效掩码 (1.0=有效)

    // ---- 模板结构数据 (从 cif/pdb 目录加载) ----
    // 每个模板为一个独立结构域, 残基数可不同, 故用 vector 存储原始骨架坐标。
    // 已自动过滤掉与真实值 (ground truth) 相同的 PDB id, 不再作为模板。
    std::vector<TensorF32>  template_coords;        // 每个模板: (N_res, 4, 3) = [N, CA, C, O] × [x,y,z]
    std::vector<std::string> template_ids;          // 模板 PDB id (小写, 不含链), 如 "2bim"
    std::vector<std::string> template_chains;       // 模板链 id, 如 "B"
    std::vector<int>         template_residue_counts; // 每个模板的残基数
};

// 模型输出
struct ModelOutput {
    TensorF32 msa;           // (B, N, L, D_MSA)
    TensorF32 pair;          // (B, L, L, D_PAIR)
    TensorF32 state;         // (B, L, D_STATE)
    TensorF32 coords;        // (B, L, 3, 3) - 更新后的坐标
    TensorF32 alpha;         // (B, L, NTOTALDOFS, 2) - 侧链扭转角

    // Masked MSA 预测头 logits (供 masked_msa_loss)
    TensorF32 msa_logits;    // (B, N, L, 23) - MSA head 输出 (每个序列位置/残基的 23 类 aatype logits)
    
    // 辅助输出
    TensorF32 lddt;          // (B, L, 50) - pLDDT logits (每残基 50 bins)
    TensorF32 distogram;     // (B, L, L, 60) - 距离分布 logits
    TensorF32 omega;         // (B, L, L, 36) - Ω 二面角 logits
    TensorF32 theta;         // (B, L, L, 36) - Θ 二面角 logits
    TensorF32 phi;           // (B, L, L, 18) - Φ 平面角 logits
    TensorF32 pae;           // (B, L, L) - 预测对齐误差
};

// 图模式前向输出：携带可微图节点（供 loss 直接组装计算图，梯度可回传模型参数）与回落值。
// 图节点为 ggml 布局（dims[0]=最内维），loss 用 view 到其期望布局后接图：
//   msa        [D_MSA, L, N, B]   → masked_msa_loss 需 {23, L, N}
//   pair       [D_PAIR, L, L, B]  → distogram_loss 上游
//   state      [D_STATE, L, B]    → plddt_loss 上游
//   msa_logits [23, L, N, B]      → view {23, L, N}  (B=1 丢弃批维)
//   alpha      [14, L, B]         → view {2, 7, N}   (N=B*L, 14=7*2, sin/cos 最内)
//   lddt       [50, L, B]         → view {50, L}
//   distogram/omega/theta/phi [bins, L, L, B] → view {bins, L, L}
// coords 为值张量（SE3 图外坐标更新，非可微），FAPE/conf 等坐标相关 loss 沿用原值版方式。
struct GraphOutput {
    TensorF32* msa         = nullptr; // [D_MSA, L, N, B]   最终 block msa 图节点
    TensorF32* pair        = nullptr; // [D_PAIR, L, L, B]  最终 block pair 图节点
    TensorF32* state       = nullptr; // [D_STATE, L, B]    SE3 后 state 图节点
    // 落地值成员：forward_graph 在 head 之前把 msa/pair/state 图节点一次性落到这些持久值，
    // 供 go.msa/go.pair/go.state 指针引用。否则 head 的值版 forward 触发 compute 会经 gallocr
    // 复用释放原图节点 buffer，使 go.* 读到 0/NaN，且图节点 src 悬垂。（见 4.5 段）
    TensorF32 msa_v;
    TensorF32 pair_v;
    TensorF32 state_v;
    TensorF32* msa_logits  = nullptr; // [23, L, N, B]      masked-msa head 图节点
    TensorF32* alpha       = nullptr; // [14, L, B]         chi head 图节点
    TensorF32* lddt        = nullptr; // [50, L, B]         plddt head 图节点
    TensorF32* distogram   = nullptr; // [60, L, L, B]      distogram d head
    TensorF32* omega       = nullptr; // [36, L, L, B]
    TensorF32* theta       = nullptr; // [36, L, L, B]
    TensorF32* phi         = nullptr; // [18, L, L, B]

    // SE3 更新后的骨架坐标（值，图外量；未驱动 SE3 时即输入坐标）
    TensorF32 coords;                  // (B, L, 3, 3)
    // 【FAPE 梯度回传】SE3 更新后的坐标图节点 [9, B*L]（9=3原子×3坐标, 原子最内）。
    // 坐标链: wrap(input.coords) → add(mul_mat(T, offset)) 逐 block，FAPE 用此可微节点，
    // 梯度经 coords→offset→SE3 回传。值 coords 仍供 conf/lddt 计算（compute_lddt_ca 需 host 值）。
    TensorF32* coords_graph = nullptr; // [9, B*L] 图节点
};

// 迭代块 (IterBlock)
class IterBlock {
public:
    IterBlock(const PPMLConfig& config, bool update_msa_pair = true);
    // 多态基类必须有 virtual 析构：RefineBlock 经 make_unique 分配（更大），被存进 unique_ptr<IterBlock>
    // 删除时若走非 virtual 析构会 new-delete-type-mismatch（ASAN）。加 virtual 使 delete 按最派生类型正确回收。
    virtual ~IterBlock() = default;
    
    // 由 PPMLModel 注入子模块 (non-owning pointers 已在之前注入, 这里是 unique_ptr 所有权转移)
    void set_sub_modules(
        std::unique_ptr<MSARowAttention>          msa_row,
        std::unique_ptr<MSAColAttention>          msa_col,
        std::unique_ptr<FeedForward>              msa_ff,
        std::unique_ptr<PairRowAttention>         pair_row,
        std::unique_ptr<PairColAttention>         pair_col,
        std::unique_ptr<FeedForward>              pair_ff,
        std::unique_ptr<TriangleMultiplication>   tri_out,
        std::unique_ptr<TriangleMultiplication>   tri_in,
        std::unique_ptr<SE3Transformer>           se3
    ) {
        msa_row_attn_  = std::move(msa_row);
        msa_col_attn_  = std::move(msa_col);
        msa_ff_        = std::move(msa_ff);
        pair_row_attn_ = std::move(pair_row);
        pair_col_attn_ = std::move(pair_col);
        pair_ff_       = std::move(pair_ff);
        tri_mul_out_   = std::move(tri_out);
        tri_mul_in_    = std::move(tri_in);
        se3_           = std::move(se3);
    }

    void set_pos_enc(std::unique_ptr<PositionalEncoding> p) { pos_enc_ = std::move(p); }
    
    // 执行一个迭代块
    // 输入/输出通过引用修改
    virtual void forward(TensorF32& msa, TensorF32& pair, TensorF32& state, 
                 const TensorF32& seq1hot,
                 const TensorF32& coords,
                 const TensorF32& bond_feats  = TensorF32(),
                 const TensorF32& dist_matrix = TensorF32(),
                 const TensorF32& same_chain  = TensorF32(),
                 const TensorI64& residx       = TensorI64());

    // ===== 图模式前向（msa/pair 两条 track + SE3(3D) track，供训练入口驱动调用）=====
    // 输入/输出均为图节点指针 (ggml 布局 dims[0]=最内维):
    //   msa    : 值 (B,N,L,D_MSA)  = 图 [D_MSA, L, N, B]
    //   pair   : 值 (B,L,L,D_PAIR) = 图 [D_PAIR, L, L, B]
    //   rbf    : 值 (B,L,L,D_RBF)  = 图 [D_RBF, L, L, B]  (RBF + pos_enc 注入)
    //   state  : 值 (B,L,D_STATE)  = 图 [D_STATE, L, B]；SE3 通过引用回写更新
    //   coords : 值 (B,L,3,3) 骨架坐标（固定结构输入，非可微；用于 make_graph/l1_feats/basis）
    //   residx : 值 (B,L) 残基索引（make_graph 用）
    //   seq1hot: 值 (B,L,21) 序列 one-hot（node 输入 concat 用）
    // 返回: 更新后的 pair 图节点；msa、state 通过引用回写。
    // 注: 当 coords/seq1hot 提供时，末尾追加 SE3(3D) track：
    //   node0 = norm_node_3d(embed_x(cat(msa_sum, seq1hot))) 图节点 [32,B*L]；
    //   node1 = l1_feats 常量叶子 [m1*d_dim1,B*L]（m1=SE3 fiber_in 度1通道数，不足补零）；
    //   边 src/tgt/d/w 与 basis 由 make_graph/compute_l1_features/basis.compute 值版常量注入；
    //   state = SE3 输出度0 [B,L,D_STATE]；xyz_new_ = 坐标更新（图外值回落）。
    virtual TensorF32* forward_graph(TensorF32*& msa, TensorF32*& pair,
                                     TensorF32* rbf, TensorF32*& state,
                                     const TensorF32* coords = nullptr,
                                     const TensorI64* residx = nullptr,
                                     const TensorF32* seq1hot = nullptr);

    void proj_state_add_to_query_row(TensorF32& msa, const TensorF32& proj_state);

    // SE3(3D) track 图模式子流程。
    // 可微部分（node 度0：msa 沿 Nseq 均值 + seq1hot → embed_x_ → norm_node_3d_）走图 op；
    // 结构常量由调用方值版预处理后以 G（make_graph 产物）与 basis 注入。
    // 调用 se3_->forward_graph，把 state（度0）经引用回写。
    // 返回 se3_out 图节点：se3_out[0]=state（度0）、se3_out[1]=offset（度1，坐标更新用，图外回落）。
    // msa/pair/rbf 为图节点（最新 track 输出）；G/basis/coords/seq1hot 为结构常量。
    std::vector<TensorF32*> run_se3_graph(TensorF32*& msa, TensorF32*& pair, TensorF32* rbf,
                                          TensorF32*& state,
                                          const se3::GraphData& G, const SE3Basis& basis,
                                          const TensorF32& coords, const TensorF32& seq1hot);

    // 训练入口驱动：SE3 图块的"图外值回落" + 坐标更新。
    //   Phase A（结构常量）：pair 值 → norm_pair_3d_? 否 → embed_e_ → norm_edge_3d_ → make_graph
    //                        → G；basis.compute(G.edge_d, 2)。pair_value 为前一 track graph_compute 后的值。
    //   Phase B（图块）：run_se3_graph(...) 追加 SE3 图节点，state 经引用回写为图节点。
    //   Phase C（坐标更新）：由调用方对返回的 offset 图节点 graph_compute 后调用
    //                        apply_coord_update(offset_value, coords) → xyz_new_。
    // 返回 run_se3_graph 的 se3_out：se3_out[0]=state 度0 图节点、se3_out[1]=offset 度1 图节点
    //   （调用方 graph_compute se3_out[1] 得 offset 值后 apply_coord_update 更新坐标）。空 vector
    //   表示无有效边图（跳过 SE3）。
    std::vector<TensorF32*> run_se3_structural(TensorF32*& msa, TensorF32*& pair, TensorF32* rbf,
                                               TensorF32*& state,
                                               const TensorF32& coords,
                                               const TensorI64& residx,
                                               const TensorF32& seq1hot);

    // 坐标更新（图外值回落）：把 SE3 度1 offset 值叠加到 coords → xyz_new_（值版 Step4k）。
    // offset_value 布局 (B*L, 3, 3)，[N,CA,C] 通道；CA 为绝对位移。供训练入口在 graph_compute
    // 出 offset 值后调用。
    void apply_coord_update(const TensorF32& offset_value, const TensorF32& coords);

    static TensorF32 compute_rbf_feature(const TensorF32& coords);
    static TensorF32 compute_l1_features(const TensorF32& coords);
    
protected:
    // 子模块 (protected 允许 FullBlock/RefineBlock 子类访问)
    std::unique_ptr<MSARowAttention> msa_row_attn_;
    std::unique_ptr<MSAColAttention> msa_col_attn_;
    std::unique_ptr<PairRowAttention> pair_row_attn_;
    std::unique_ptr<PairColAttention> pair_col_attn_;
    std::unique_ptr<FeedForward> msa_ff_;
    std::unique_ptr<FeedForward> pair_ff_;
    std::unique_ptr<TriangleMultiplication> tri_mul_out_;
    std::unique_ptr<TriangleMultiplication> tri_mul_in_;
    std::unique_ptr<SE3Transformer> se3_;
    //std::unique_ptr<StructureUpdate> struct_update_;
    std::unique_ptr<PositionalEncoding> pos_enc_;

private:
    PPMLConfig config_;
    bool update_msa_pair_;

protected:
    // 3D track (子类可访问)
    TensorF32 xyz_new_;              // (B, L, 3, 3)
    TensorF32 state_new_;            // (B, L, D_STATE) 可选，或直接用 state 引用

private:

protected:
    // ---- 3D SE 参数 (non-owning pointer, 由 PPMLModel 创建, FullBlock 子类访问) ----
    LayerNorm*  norm_msa_3d_  = nullptr; // D_MSA (256)
    LayerNorm*  norm_pair_3d_ = nullptr; // D_PAIR (128)
    LinearLayer* embed_x_     = nullptr; // ITER_NODE_3D_IN (277) → ITER_NODE_3D_OUT (32)
    LinearLayer* embed_e_     = nullptr; // D_PAIR (128) → ITER_EDGE_3D_OUT (32)
    LayerNorm*  norm_node_3d_ = nullptr; // ITER_NODE_3D_OUT (32)
    LayerNorm*  norm_edge_3d_ = nullptr; // ITER_EDGE_3D_OUT (32)

    // ---- forward 内部参数 (non-owning pointer, 由 PPMLModel 创建) ----
    LayerNorm*   state2msa_norm_        = nullptr; // D_STATE (32)
    LinearLayer* state2msa_linear_      = nullptr; // D_STATE (32) → D_MSA (256)
    LayerNorm*   pair2msa_norm_         = nullptr; // D_PAIR (128)
    LayerNorm*   msa2pair_norm_         = nullptr; // D_MSA (256)
    LinearLayer* msa2pair_left_proj_    = nullptr; // D_MSA (256) → 16
    LinearLayer* msa2pair_right_proj_   = nullptr; // D_MSA (256) → 16
    LinearLayer* msa2pair_out_proj_     = nullptr; // 16*16 (256) → D_PAIR (128)
    LinearLayer* pair2pair_rbf_proj_    = nullptr; // D_RBF (64) → D_PAIR (128)
    LayerNorm*   pair2pair_state_norm_  = nullptr; // D_STATE (32)
    LinearLayer* pair2pair_left_proj_   = nullptr; // D_STATE (32) → 16
    LinearLayer* pair2pair_right_proj_  = nullptr; // D_STATE (32) → 16
    LinearLayer* pair2pair_gate_proj_   = nullptr; // 16*16 (256) → D_PAIR (128)

    // 额外输入 (由 caller 在 forward 前设置)
    //TensorF32 seq1hot_;              // (B, L, 21)
    //TensorI64 idx_;                  // (B, L)
    //bool has_seq_info_ = false;

public:
    const TensorF32& updated_coords() const { return xyz_new_; }
    //void set_seq_info(const TensorF32& seq1hot, const TensorI64& idx);

    friend class PPMLModel;  // PPMLModel 直接注入 non-owning pointers
};

class FullBlock : public IterBlock {
public:
    explicit FullBlock(const PPMLConfig& config, bool update_msa_pair = true)
        : IterBlock(config, update_msa_pair) {}

    // a virtual dtor
    virtual ~FullBlock() = default;

    void set_global_col_attn(std::unique_ptr<MSAGlobalColAttention> p) {
        msa_global_col_attn_ = std::move(p);
    }

    void forward(TensorF32& msa_full, TensorF32& pair, TensorF32& state, 
                 const TensorF32& seq1hot,
                 const TensorF32& coords,
                 const TensorF32& bond_feats  = TensorF32(),
                 const TensorF32& dist_matrix = TensorF32(),
                 const TensorF32& same_chain  = TensorF32(),
                 const TensorI64& residx       = TensorI64()) override;

    // ===== 图模式前向（msa/pair + SE3；msa 走 global column attention）=====
    // 布局约定同 IterBlock::forward_graph；SE3 track 由 IterBlock::forward_graph 末尾追加。
    TensorF32* forward_graph(TensorF32*& msa_full, TensorF32*& pair,
                             TensorF32* rbf, TensorF32*& state,
                             const TensorF32* coords = nullptr,
                             const TensorI64* residx = nullptr,
                             const TensorF32* seq1hot = nullptr) override;
private:
    std::unique_ptr<MSAGlobalColAttention> msa_global_col_attn_;
};

class RefineBlock : public IterBlock {
public:
    explicit RefineBlock(const PPMLConfig& config, bool update_msa_pair = true)
        : IterBlock(config, update_msa_pair) {}

    // a virtual dtor
    virtual ~RefineBlock() = default;

    void forward(TensorF32& msa, TensorF32& pair, TensorF32& state, 
                 const TensorF32& seq1hot,
                 const TensorF32& coords,
                 const TensorF32& bond_feats  = TensorF32(),
                 const TensorF32& dist_matrix = TensorF32(),
                 const TensorF32& same_chain  = TensorF32(),
                 const TensorI64& residx       = TensorI64()) override;

    // ===== 图模式前向（override：RefineBlock 语义）=====
    // update_msa_pair_=false：RefineBlock 不改 msa/pair，仅做 3D 结构更新。
    // 故本 override 为 pass-through（直接返回 pair），结构更新由训练入口调用
    // run_se3_structural_refine 驱动（见 PPMLModel::forward_graph 的 refine 循环）。
    // 参数布局同 IterBlock::forward_graph。
    TensorF32* forward_graph(TensorF32*& msa, TensorF32*& pair,
                             TensorF32* rbf, TensorF32*& state,
                             const TensorF32* coords = nullptr,
                             const TensorI64* residx = nullptr,
                             const TensorF32* seq1hot = nullptr) override;

    // ===== RefineBlock 专属 SE3 图模式子流程 =====
    // 语义对齐值版 RefineBlock::forward，但用图 op 构建可微部分：
    //   node 度0 = norm_node_(embed_x_(cat(norm_msa(msa 沿 Nseq 均值), seq1hot, norm_state(state))))
    //   node 度1 = l1_feats (compute_l1_features(coords)) 常量叶子
    //   edges    = G.edge_index / edge_d / edge_w → src/tgt/d/w 常量叶子（调用方值版 make_graph 注入）
    //   basis    = 调用方预计算（SE3Basis.compute(G.edge_d, 2)）
    //   out      = se3_->forward_graph({node0,node1}, src, tgt, d, w, basis, N=B*L)
    //   state    = out[0].view({B,L,D_STATE})（引用回写）
    // 返回 se3_out：se3_out[0]=state(度0)、se3_out[1]=offset(度1，坐标更新用，训练入口 graph_compute
    //       后回落值 + apply_coord_update 更新骨架坐标)。空 vector = 无有效边图（跳过 SE3）。
    std::vector<TensorF32*> run_se3_graph_refine(TensorF32*& msa, TensorF32*& pair, TensorF32* rbf,
                                                 TensorF32*& state,
                                                 const se3::GraphData& G, const SE3Basis& basis,
                                                 const TensorF32& coords, const TensorF32& seq1hot);

    // 训练入口驱动：RefineBlock SE3 图块（图外值回落 + state 回写）。
    //   Phase A（结构常量，图外值回落）：pair_value → norm_pair_ → embed_e1_ → norm_edge1_
    //       → cat(rbf_feat=compute_rbf_feature(coords), neighbor=get_bonded_neigh(residx))
    //       → embed_e2_ → norm_edge2_ → make_graph(coords, edge_out, residx, 64, 9) → G；
    //       basis.compute(G.edge_d, 2)。
    //   Phase B（可微图块）：调用 run_se3_graph_refine(...)，state 经引用回写为图节点（度0）。
    // 返回 run_se3_graph_refine 的 se3_out：se3_out[0]=state 度0 图节点、se3_out[1]=offset 度1
    //       图节点（训练入口 graph_compute 后 apply_coord_update 更新坐标）。
    std::vector<TensorF32*> run_se3_structural_refine(TensorF32*& msa, TensorF32*& pair, TensorF32* rbf,
                                                      TensorF32*& state,
                                                      const TensorF32& coords,
                                                      const TensorI64& residx,
                                                      const TensorF32& seq1hot);

    // 设置额外输入（在 forward 调用前设置）
    //void set_seq_info(const TensorF32& seq1hot, const TensorI64& idx);

    // 获取 SE3 更新后的坐标
    const TensorF32& updated_coords() const { return xyz_new_; }
    const TensorF32& updated_state()  const { return state_new_; }

private:
    // ---- LayerNorm 模块 (non-owning pointer) ----
    // 旧值类型 (保留注释):
    // LayerNorm norm_msa_(D_MSA);
    // LayerNorm norm_pair_(D_PAIR);
    // LayerNorm norm_state_(D_STATE);
    LayerNorm* norm_msa_   = nullptr; // D_MSA (256)
    LayerNorm* norm_pair_  = nullptr; // D_PAIR (128)
    LayerNorm* norm_state_ = nullptr; // D_STATE (32)

    // ---- Node 嵌入 (309 → 32) ----
    // 旧值类型 (保留注释):
    // static constexpr int NODE_IN_DIM  = D_MSA + 21 + D_STATE;  // → Core.h REFINE_NODE_IN_DIM
    // static constexpr int NODE_OUT_DIM = N_L0_IN_FEATS;          // → Core.h REFINE_NODE_OUT_DIM
    // LinearLayer embed_x_(NODE_IN_DIM, NODE_OUT_DIM);
    // LayerNorm  norm_node_(NODE_OUT_DIM);
    LinearLayer* embed_x_   = nullptr; // REFINE_NODE_IN_DIM (309) → REFINE_NODE_OUT_DIM (32)
    LayerNorm*   norm_node_ = nullptr; // REFINE_NODE_OUT_DIM (32)

    // ---- Edge 嵌入 第一阶段 (128 → 32) ----
    // 旧值类型 (保留注释):
    // LinearLayer embed_e1_(D_PAIR, N_EDGE_FEATS);
    // LayerNorm  norm_edge1_(N_EDGE_FEATS);
    LinearLayer* embed_e1_   = nullptr; // D_PAIR (128) → N_EDGE_FEATS (32)
    LayerNorm*   norm_edge1_ = nullptr; // N_EDGE_FEATS (32)

    // ---- Edge 嵌入 第二阶段 (32+64+1=97 → 32) ----
    // 旧值类型 (保留注释):
    // static constexpr int EDGE_IN_DIM2 = N_EDGE_FEATS + 64 + 1;  // → Core.h REFINE_EDGE_IN_DIM2
    // LinearLayer embed_e2_(EDGE_IN_DIM2, N_EDGE_FEATS);
    // LayerNorm  norm_edge2_(N_EDGE_FEATS);
    LinearLayer* embed_e2_   = nullptr; // REFINE_EDGE_IN_DIM2 (97) → N_EDGE_FEATS (32)
    LayerNorm*   norm_edge2_ = nullptr; // N_EDGE_FEATS (32)

    // ---- 额外输入（由外部设置）----
    //TensorF32 seq1hot_;       // (B, L, 21) 序列 one-hot
    //TensorI64 idx_;           // (B, L) 残基索引
    //bool      has_seq_info_ = false;

    // ---- 输出缓存 ----
    // ⚠️ 不再定义独立的 xyz_new_/state_new_：updated_coords() 是非虚函数，
    //    block->updated_coords()（静态类型 IterBlock*）会调用基类版本返回基类成员。
    //    派生类若再定义同名成员，RefineBlock::forward 更新派生类成员而读取端拿基类
    //    成员 → 永远拿到默认空张量（[COPY-FAIL] src=()）。统一复用基类成员。

    friend class PPMLModel;  // PPMLModel 直接注入 non-owning pointers
};

// PPML 主模型
class PPMLModel {
public:
    explicit PPMLModel(const PPMLConfig& config = PPMLConfig{});
    ~PPMLModel();

    void set_seq_info(const TensorF32& seq1hot, const TensorI64& idx);
    
    // 前向传播 (值模式, 既有实现, 不改动)
    ModelOutput forward(const ModelInput& input);

    // ===== 图模式前向（新增入口，不改 forward）=====
    // 预处理 embedding 与 block 前向均使用 forward_graph 版本（ggml 布局 dims[0]=最内维）。
    // 返回 GraphOutput：可微图节点（供 loss 组装计算图，反向可回传模型参数）+ coords 回落值。
    // 注:
    //   - PositionalEncoding 的图版本当前为占位（返回零图节点），故 pair 初始化不含位置编码；
    //   - rbf 特征用值版 compute_rbf_feature 后注入为常量图 leaf；
    //   - SE3 3D track 需"图外值回落"驱动（graph_compute(pair)→run_se3_structural→graph_compute
    //     offset→apply_coord_update），本入口在 block 循环边界以相同方式驱动；
    //   - 输出头图节点只构建不 graph_compute，由调用方（如 train.cpp）对总 loss 图一次性计算。
    // topo_coords：可选的外部拓扑坐标（开关B/pass1 两遍 forward 用）——非空时用它初始化
    //   current_coords（SE3 make_graph 的拓扑基准），替代默认的 input.coords（初始坐标）。
    //   典型用法：Pass1 值版 forward 逐 block 更新得到精确 coords → 传给 Pass2 forward_graph。
    GraphOutput forward_graph(const ModelInput& input, bool enable_se3 = true,
                              const TensorF32* topo_coords = nullptr);

    TensorF32 get_templ_emb(const TensorF32& t1d, const TensorF32& t2d);
    
    // 加载/保存权重
    void load_weights(const std::string& path);
    void save_weights(const std::string& path) const;

    // 收集所有参数 Tensor (weight/bias/gamma/beta), 顺序固定, 供保存/加载/统计使用
    // 注意: 返回的是指向内部参数存储的裸指针集合, 仅用于读取; 生命周期由模型管理。
    std::vector<TensorF32*> params();

    // 收集所有参数 Tensor 及与之顺序一一对应的语义名 (block/attention 等), 供 GGUF 保存。
    // param_names[i] 与 param_tensors[i] 严格对应, 不可交错。
    void collect_params_with_names(std::vector<TensorF32*>& param_tensors,
                                   std::vector<std::string>& param_names);
    
    // 设备管理
    void to(Device device);
    Device device() const;

    // 配置访问器 (供训练脚本读取 block 数量等, 不复制)
    const PPMLConfig& config() const { return config_; }
    
    // 训练/推理模式
    void train();
    void eval();
    bool is_training() const;

    // 后端访问器：返回当前可用的执行后端（CUDA 若就绪则优先，否则 CPU）。
    // 供训练循环调用 backend->graph_compute(cgraph) 执行反向/更新。
    Backend* active_backend();

    // 调度器访问器：返回后端调度器（CPU+CUDA 注册后按 priority 分配算子）。
    // 供训练循环用 split_graph + alloc_splits + graph_compute 让受支持算子跑 CUDA、
    // 不支持的自动跨后端拷贝回落 CPU。
    BackendScheduler* scheduler();

    // 【SE3 offset scale · 可学习标量参数】(2026-08-23)
    // 返回 SE3 offset scale 图节点（已 clamp 到安全区间）。
    // 多样本/显式学习模式下为可训练 PARAM（log-space + relu 硬 clamp，梯度可回传）；
    // 否则为冻结常量（env PPML_SE3_GRAPH_SCALE 覆盖，缺省 1e-3）。
    // 该节点在所有 block 间共享同一实例，确保全局一致且优化器能跨图找到它。
    TensorF32* se3_scale_tensor();

    // 仅诊断：打印当前 SE3 scale（学习模式下为 PARAM 值 exp(log_scale)，否则冻结常量）。
    void se3_scale_report() const;

private:
    PPMLConfig config_;
    Device device_ = Device::CPU;
    bool training_ = false;

    // ===== embedding / template 参数 (全局单份) =====
    LinearLayer*     msa_emb_            = nullptr; // (164 → 256)
    EmbeddingLayer*  state_emb_          = nullptr; // (NAATOKENS, D_STATE)
    EmbeddingLayer*  pair_left_emb_      = nullptr; // (NAATOKENS, D_PAIR)
    EmbeddingLayer*  pair_right_emb_     = nullptr; // (NAATOKENS, D_PAIR)
    LinearLayer*     full_linear_        = nullptr; // (83 → D_MSA_FULL)
    EmbeddingLayer*  full_emb_           = nullptr; // (NAATOKENS, D_MSA_FULL)
    LinearLayer*     bond_emb_           = nullptr; // (8 → D_PAIR)
    LinearLayer*     emb_t1d_            = nullptr; // (110 → 64)
    LinearLayer*     proj_t1d_           = nullptr; // (64 → 64)
    LinearLayer*     emb_t1d_t2d_        = nullptr; // (228 → D_PAIR) get_templ_emb；模板 pair 为 D_PAIR=128 维
    LinearLayer*     temp_stack_t1d_proj_ = nullptr; // (80 → 32) templ_stack
    LayerNorm*       temp_stack_norm_    = nullptr; // (D_PAIR=128)
    // Template state cross-attention 投影 (CrossAttention 外部注入, 无 bias):
    //   Q=state(D_STATE=32), KV=template_emb(64), H=8 → proj_dim=64
    LinearLayer*     templ_attn_Wq_      = nullptr; // D_STATE (32) → 64
    LinearLayer*     templ_attn_Wk_      = nullptr; // 64 → 64
    LinearLayer*     templ_attn_Wv_      = nullptr; // 64 → 64
    LinearLayer*     templ_attn_Wo_      = nullptr; // 64 → D_STATE (32)
    // Template pair→pair cross-attention 投影 (CrossAttention(D_PAIR,D_PAIR,8) 外部注入):
    //   Q=pair(D_PAIR=128), KV=templ_pair(D_PAIR=128), H=8 → proj_dim=128
    LinearLayer*     templ_pair_attn_Wq_ = nullptr; // D_PAIR (128) → 128
    LinearLayer*     templ_pair_attn_Wk_ = nullptr; // D_PAIR (128) → 128
    LinearLayer*     templ_pair_attn_Wv_ = nullptr; // D_PAIR (128) → 128
    LinearLayer*     templ_pair_attn_Wo_ = nullptr; // 128 → D_PAIR (128)

    // ===== TemplatePairStack 子层 (全局单份, 2 次 forward 复用) =====
    // 直接 LinearLayer/LayerNorm
    LinearLayer*     tps_rbf_proj_      = nullptr; // D_RBF (64) → D_PAIR (128)
    LayerNorm*       tps_state_norm_    = nullptr; // D_STATE (32)
    LinearLayer*     tps_left_proj_     = nullptr; // D_STATE (32) → 16
    LinearLayer*     tps_right_proj_    = nullptr; // D_STATE (32) → 16
    LinearLayer*     tps_gate_proj_     = nullptr; // 16*16 (256) → D_PAIR (128)
    // TriangleMultiplication out (2 LN + 6 LL)
    LayerNorm*       tps_tri_out_layernorm_        = nullptr;
    LinearLayer*     tps_tri_out_left_proj_        = nullptr;
    LinearLayer*     tps_tri_out_right_proj_       = nullptr;
    LinearLayer*     tps_tri_out_left_gate_        = nullptr;
    LinearLayer*     tps_tri_out_right_gate_       = nullptr;
    LinearLayer*     tps_tri_out_gate_             = nullptr;
    LayerNorm*       tps_tri_out_output_layernorm_ = nullptr;
    LinearLayer*     tps_tri_out_out_proj_         = nullptr;
    // TriangleMultiplication in (2 LN + 6 LL)
    LayerNorm*       tps_tri_in_layernorm_        = nullptr;
    LinearLayer*     tps_tri_in_left_proj_        = nullptr;
    LinearLayer*     tps_tri_in_right_proj_       = nullptr;
    LinearLayer*     tps_tri_in_left_gate_        = nullptr;
    LinearLayer*     tps_tri_in_right_gate_       = nullptr;
    LinearLayer*     tps_tri_in_gate_             = nullptr;
    LayerNorm*       tps_tri_in_output_layernorm_ = nullptr;
    LinearLayer*     tps_tri_in_out_proj_         = nullptr;
    // PairRowAttention (6 LL)
    LayerNorm*       tps_pair_norm_       = nullptr; // D_PAIR (128) LayerNorm(pair)，投影前归一化
    LinearLayer*     tps_pair_row_to_b_   = nullptr;
    LinearLayer*     tps_pair_row_to_g_   = nullptr;
    LinearLayer*     tps_pair_row_to_out_ = nullptr;
    LinearLayer*     tps_pair_row_Wq_     = nullptr;
    LinearLayer*     tps_pair_row_Wk_     = nullptr;
    LinearLayer*     tps_pair_row_Wv_     = nullptr;
    // PairColAttention (6 LL)
    LinearLayer*     tps_pair_col_to_b_   = nullptr;
    LinearLayer*     tps_pair_col_to_g_   = nullptr;
    LinearLayer*     tps_pair_col_to_out_ = nullptr;
    LinearLayer*     tps_pair_col_Wq_     = nullptr;
    LinearLayer*     tps_pair_col_Wk_     = nullptr;
    LinearLayer*     tps_pair_col_Wv_     = nullptr;
    // FeedForward (1 LN + 2 LL)
    LayerNorm*       tps_pair_ff_norm_    = nullptr;
    LinearLayer*     tps_pair_ff_linear1_ = nullptr;
    LinearLayer*     tps_pair_ff_linear2_ = nullptr;
    // 子模块实例 (通过 set_params 注入)
    TriangleMultiplication tps_tri_mul_out_;
    TriangleMultiplication tps_tri_mul_in_;
    PairRowAttention       tps_pair_row_attn_;
    PairColAttention       tps_pair_col_attn_;
    FeedForward            tps_pair_ff_;
    // TemplatePairStack 实例
    TemplatePairStack      tps_;

    // ===== 输出头参数 (全局单份) =====
    // Masked MSA head: LayerNorm(D_MSA) → Linear(D_MSA→D_MSA) → ReLU → Linear(D_MSA→23)
    LayerNorm*   msa_head_ln_      = nullptr;  // D_MSA (256)
    LinearLayer* msa_head_linear1_ = nullptr;  // D_MSA (256) → D_MSA (256)
    LinearLayer* msa_head_linear2_ = nullptr;  // D_MSA (256) → 23

    // Chi (扭转角) head: LayerNorm(D_STATE) → Linear(D_STATE→D_STATE) → ReLU → Linear(D_STATE→7*2)
    // 输出 alpha (B, L, 7, 2) — omega/phi/psi/chi1-4 的未归一化 (sin, cos)
    LayerNorm*   chi_head_ln_      = nullptr;  // D_STATE (32)
    LinearLayer* chi_head_linear1_ = nullptr;  // D_STATE (32) → D_STATE (32)
    LinearLayer* chi_head_linear2_ = nullptr;  // D_STATE (32) → 14

    // Distogram head: 从 pair 特征投影 4 组 logits (D/Ω/Θ/Φ)
    // 输出 distogram (B,L,L,60), omega (B,L,L,36), theta (B,L,L,36), phi (B,L,L,18)
    LayerNorm*   distogram_pair_ln_ = nullptr;  // D_PAIR (128) LayerNorm(pair)，投影前归一化（防 logits 巨大→softmax 退化）
    LinearLayer* distogram_d_head_ = nullptr;  // D_PAIR → 60 (距离 bins)
    LinearLayer* distogram_o_head_ = nullptr;  // D_PAIR → 36 (Ω bins)
    LinearLayer* distogram_t_head_ = nullptr;  // D_PAIR → 36 (Θ bins)
    LinearLayer* distogram_p_head_ = nullptr;  // D_PAIR → 18 (Φ bins)

    // pLDDT head: state → lddt logits (B,L,50)
    LinearLayer* plddt_head_       = nullptr;  // D_STATE → 50

    // ===== attention / sub-module 参数 (per-block, vector) =====
    static constexpr int N_ITER = 12;   // extra(4) + main(8)
    static constexpr int N_GLOB = 4;    // extra_blocks (FullBlock)

    // --- MSARowAttention (6 LL ×12) ---
    // 旧: LinearLayer* msa_row_Wq_ etc.
    std::vector<LinearLayer*> msa_row_Wq_;     // D_MSA (256) → N_HEAD*D_MSA (2048)
    std::vector<LinearLayer*> msa_row_Wk_;     // D_MSA (256) → N_HEAD*D_MSA (2048)
    std::vector<LinearLayer*> msa_row_Wv_;     // D_MSA (256) → N_HEAD*D_MSA (2048)
    std::vector<LinearLayer*> msa_row_to_b_;   // D_PAIR (128) → N_HEAD (8)
    std::vector<LinearLayer*> msa_row_to_g_;   // D_MSA (256) → N_HEAD*D_MSA (2048)
    std::vector<LinearLayer*> msa_row_to_out_; // N_HEAD*D_MSA (2048) → D_MSA (256)

    // --- MSAColAttention (6 LL ×12) ---
    std::vector<LinearLayer*> msa_col_Wq_;     // D_MSA (256) → N_HEAD*D_MSA (2048)
    std::vector<LinearLayer*> msa_col_Wk_;     
    std::vector<LinearLayer*> msa_col_Wv_;     
    std::vector<LinearLayer*> msa_col_to_b_;   // D_PAIR (128) → N_HEAD (8)
    std::vector<LinearLayer*> msa_col_to_g_;   
    std::vector<LinearLayer*> msa_col_to_out_; 

    // --- MSAGlobalColAttention (6 LL ×4, FullBlock only) ---
    std::vector<LinearLayer*> msa_global_col_Wq_;    // D_MSA (256) → D_MSA (256)  single-head
    std::vector<LinearLayer*> msa_global_col_Wk_;    
    std::vector<LinearLayer*> msa_global_col_Wv_;    
    std::vector<LinearLayer*> msa_global_col_to_b_;  // D_PAIR (128) → N_HEAD (8)
    std::vector<LinearLayer*> msa_global_col_to_g_;  
    std::vector<LinearLayer*> msa_global_col_to_out_;

    // --- PairRowAttention (6 LL ×12) ---
    std::vector<LayerNorm*>   pair_attn_norm_;  // D_PAIR (128) LayerNorm(pair)，投影前归一化（row/col 共享）
    std::vector<LinearLayer*> pair_row_Wq_;     // D_PAIR (128) → N_HEAD*D_PAIR_HIDDEN (256)
    std::vector<LinearLayer*> pair_row_Wk_;     
    std::vector<LinearLayer*> pair_row_Wv_;     
    std::vector<LinearLayer*> pair_row_to_b_;   // D_PAIR (128) → N_HEAD (8)
    std::vector<LinearLayer*> pair_row_to_g_;   
    std::vector<LinearLayer*> pair_row_to_out_; 

    // --- PairColAttention (6 LL ×12) ---
    std::vector<LinearLayer*> pair_col_Wq_;     
    std::vector<LinearLayer*> pair_col_Wk_;     
    std::vector<LinearLayer*> pair_col_Wv_;     
    std::vector<LinearLayer*> pair_col_to_b_;   
    std::vector<LinearLayer*> pair_col_to_g_;   
    std::vector<LinearLayer*> pair_col_to_out_; 

    // --- FeedForward msa_ff_ (1 LN + 2 LL ×12) ---
    std::vector<LayerNorm*>   msa_ff_norm_;     // D_MSA (256)
    std::vector<LinearLayer*> msa_ff_linear1_;  // D_MSA (256) → D_MSA*4 (1024)
    std::vector<LinearLayer*> msa_ff_linear2_;  // D_MSA*4 (1024) → D_MSA (256)

    // ===== FullBlock(extra_blocks_) 专属 MSA 注意力权重（D_MSA_FULL=64 维）=====
    // RF2AA 全量 MSA 模块：n_msa_head=8, n_msa_channels=8 → n_head*d_hidden=64=D_MSA_FULL。
    // FullBlock 处理 msa_full[64,L,N,B]，其 row attention / ff / global col attention 均按 64 维。
    // 注意：pair 相关（pair_row/col/ff/tri/msa2pair）仍为 D_PAIR=128，不在此列。
    std::vector<LinearLayer*> full_msa_row_Wq_;     // D_MSA_FULL (64) → 64
    std::vector<LinearLayer*> full_msa_row_Wk_;
    std::vector<LinearLayer*> full_msa_row_Wv_;
    std::vector<LinearLayer*> full_msa_row_to_b_;   // D_PAIR (128) → N_HEAD (8)
    std::vector<LinearLayer*> full_msa_row_to_g_;   // D_MSA_FULL (64) → 64
    std::vector<LinearLayer*> full_msa_row_to_out_; // 64 → D_MSA_FULL (64)
    std::vector<LayerNorm*>   full_msa_ff_norm_;     // D_MSA_FULL (64)
    std::vector<LinearLayer*> full_msa_ff_linear1_;  // 64 → 256
    std::vector<LinearLayer*> full_msa_ff_linear2_;  // 256 → 64
    std::vector<LinearLayer*> full_msa_global_col_Wq_;    // 64 → 64  single-head
    std::vector<LinearLayer*> full_msa_global_col_Wk_;
    std::vector<LinearLayer*> full_msa_global_col_Wv_;
    std::vector<LinearLayer*> full_msa_global_col_to_b_;  // D_PAIR (128) → N_HEAD (8)
    std::vector<LinearLayer*> full_msa_global_col_to_g_;
    std::vector<LinearLayer*> full_msa_global_col_to_out_;
    // FullBlock msa2pair（处理 msa_full[64,...]）— 64 维
    std::vector<LayerNorm*>   full_msa2pair_norm_;          // 64
    std::vector<LinearLayer*> full_msa2pair_left_proj_;     // 64→16
    std::vector<LinearLayer*> full_msa2pair_right_proj_;    // 64→16
    std::vector<LinearLayer*> full_msa2pair_out_proj_;      // 256→D_PAIR(128)

    // --- FeedForward pair_ff_ (1 LN + 2 LL ×12) ---
    std::vector<LayerNorm*>   pair_ff_norm_;     // D_PAIR (128)
    std::vector<LinearLayer*> pair_ff_linear1_;  // D_PAIR (128) → D_PAIR*2 (256)
    std::vector<LinearLayer*> pair_ff_linear2_;  // D_PAIR*2 (256) → D_PAIR (128)

    // --- TriangleMultiplication out (2 LN + 6 LL ×12) ---
    std::vector<LayerNorm*>   tri_out_layernorm_;         // D_PAIR (128)
    std::vector<LinearLayer*> tri_out_left_proj_;         // D_PAIR (128) → D_HIDDEN_TRIMUL (128)
    std::vector<LinearLayer*> tri_out_right_proj_;        
    std::vector<LinearLayer*> tri_out_left_gate_;         
    std::vector<LinearLayer*> tri_out_right_gate_;        
    std::vector<LinearLayer*> tri_out_gate_;              // D_PAIR (128) → D_PAIR (128)
    std::vector<LayerNorm*>   tri_out_output_layernorm_;  // D_HIDDEN_TRIMUL (128)
    std::vector<LinearLayer*> tri_out_out_proj_;          // D_HIDDEN_TRIMUL (128) → D_PAIR (128)

    // --- TriangleMultiplication in (2 LN + 6 LL ×12) ---
    std::vector<LayerNorm*>   tri_in_layernorm_;
    std::vector<LinearLayer*> tri_in_left_proj_;
    std::vector<LinearLayer*> tri_in_right_proj_;
    std::vector<LinearLayer*> tri_in_left_gate_;
    std::vector<LinearLayer*> tri_in_right_gate_;
    std::vector<LinearLayer*> tri_in_gate_;
    std::vector<LayerNorm*>   tri_in_output_layernorm_;
    std::vector<LinearLayer*> tri_in_out_proj_;

    // --- PositionalEncoding (2 EmbeddingLayer ×12, per block) ---
    std::vector<EmbeddingLayer*> pos_enc_emb_res_;   // [0..11] (65, D_PAIR=128)  residue dist embedding
    std::vector<EmbeddingLayer*> pos_enc_emb_atom_;  // [0..11] (17, D_PAIR=128)  atom bond dist embedding

    // ===== 3D SE 参数 (per-block, 由 PPMLModel::create 统一创建后将指针注入 block) =====

    // --- IterBlock 3D SE (每组 6 个, ITER_N_BLOCKS=12 组) ---
    // 旧值类型 (保留注释):
    // LinearLayer embed_x_;  // ITER_NODE_3D_IN → ITER_NODE_3D_OUT
    // LinearLayer embed_e_;  // D_PAIR → ITER_EDGE_3D_OUT
    // LayerNorm  norm_node_3d_; LayerNorm norm_edge_3d_;
    // LayerNorm norm_msa_3d_; LayerNorm norm_pair_3d_;
    std::vector<LinearLayer*> iter_embed_x_;       // [0..11]  ITER_NODE_3D_IN (277) → ITER_NODE_3D_OUT (32)
    std::vector<LinearLayer*> iter_embed_e_;       // [0..11]  D_PAIR (128) → ITER_EDGE_3D_OUT (32)
    std::vector<LayerNorm*>   iter_norm_node_3d_;  // [0..11]  ITER_NODE_3D_OUT (32)
    std::vector<LayerNorm*>   iter_norm_edge_3d_;  // [0..11]  ITER_EDGE_3D_OUT (32)
    std::vector<LayerNorm*>   iter_norm_msa_3d_;   // [0..11]  D_MSA (256)
    std::vector<LayerNorm*>   iter_norm_pair_3d_;  // [0..11]  D_PAIR (128)

    // --- IterBlock forward 内部参数 (每组 12 个, 12 组) ---
    std::vector<LayerNorm*>   iter_state2msa_norm_;       // [0..11]  D_STATE (32)
    std::vector<LinearLayer*> iter_state2msa_linear_;     // [0..11]  D_STATE (32) → D_MSA (256)
    std::vector<LayerNorm*>   iter_pair2msa_norm_;        // [0..11]  D_PAIR (128)
    std::vector<LayerNorm*>   iter_msa2pair_norm_;        // [0..11]  D_MSA (256)
    std::vector<LinearLayer*> iter_msa2pair_left_proj_;   // [0..11]  D_MSA (256) → 16
    std::vector<LinearLayer*> iter_msa2pair_right_proj_;  // [0..11]  D_MSA (256) → 16
    std::vector<LinearLayer*> iter_msa2pair_out_proj_;    // [0..11]  256 → D_PAIR (128)
    std::vector<LinearLayer*> iter_pair2pair_rbf_proj_;   // [0..11]  D_RBF (64) → D_PAIR (128)
    std::vector<LayerNorm*>   iter_pair2pair_state_norm_; // [0..11]  D_STATE (32)
    std::vector<LinearLayer*> iter_pair2pair_left_proj_;  // [0..11]  D_STATE (32) → 16
    std::vector<LinearLayer*> iter_pair2pair_right_proj_; // [0..11]  D_STATE (32) → 16
    std::vector<LinearLayer*> iter_pair2pair_gate_proj_;  // [0..11]  256 → D_PAIR (128)

    // --- RefineBlock 3D SE (每组 10 个, N_REFINE_BLOCKS=4 组) ---
    // 旧值类型 (保留注释):
    // LayerNorm norm_msa_(D_MSA); LayerNorm norm_pair_(D_PAIR); LayerNorm norm_state_(D_STATE);
    // LinearLayer embed_x_(REFINE_NODE_IN_DIM, REFINE_NODE_OUT_DIM); LayerNorm norm_node_(REFINE_NODE_OUT_DIM);
    // LinearLayer embed_e1_(D_PAIR, N_EDGE_FEATS); LayerNorm norm_edge1_(N_EDGE_FEATS);
    // LinearLayer embed_e2_(REFINE_EDGE_IN_DIM2, N_EDGE_FEATS); LayerNorm norm_edge2_(N_EDGE_FEATS);
    std::vector<LayerNorm*>   refine_norm_msa_;     // [0..3]  D_MSA (256)
    std::vector<LayerNorm*>   refine_norm_pair_;    // [0..3]  D_PAIR (128)
    std::vector<LayerNorm*>   refine_norm_state_;   // [0..3]  D_STATE (32)
    std::vector<LinearLayer*> refine_embed_x_;      // [0..3]  REFINE_NODE_IN_DIM (309) → REFINE_NODE_OUT_DIM (32)
    std::vector<LayerNorm*>   refine_norm_node_;    // [0..3]  REFINE_NODE_OUT_DIM (32)
    std::vector<LinearLayer*> refine_embed_e1_;     // [0..3]  D_PAIR (128) → N_EDGE_FEATS (32)
    std::vector<LayerNorm*>   refine_norm_edge1_;   // [0..3]  N_EDGE_FEATS (32)
    std::vector<LinearLayer*> refine_embed_e2_;     // [0..3]  REFINE_EDGE_IN_DIM2 (97) → N_EDGE_FEATS (32)
    std::vector<LayerNorm*>   refine_norm_edge2_;   // [0..3]  N_EDGE_FEATS (32)


    // Tracks
    std::unique_ptr<MSATrack> msa_track_;
    std::unique_ptr<PairTrack> pair_track_;
    std::unique_ptr<StateTrack> state_track_;

    TensorF32 seq1hot_;       // (B, L, 21) 序列 one-hot
    TensorI64 idx_;           // (B, L) 残基索引
    bool      has_seq_info_ = false;
    
    // 迭代块
    std::vector<std::unique_ptr<IterBlock>> extra_blocks_;
    std::vector<std::unique_ptr<IterBlock>> main_blocks_;
    std::vector<std::unique_ptr<IterBlock>> refine_blocks_;

    // ===== Pair 初始化位置编码 =====
    PositionalEncoding* pair_init_pos_enc_       = nullptr;
    EmbeddingLayer*     pair_init_pos_enc_emb_res_  = nullptr; // (65, D_PAIR)
    EmbeddingLayer*     pair_init_pos_enc_emb_atom_ = nullptr; // (17, D_PAIR)
    
    // 输出头
    //struct OutputHeads;
    //std::unique_ptr<OutputHeads> heads_;

    // ===== 后端基础设施 =====
    std::unique_ptr<CPUBackend>       cpu_backend_;
    std::unique_ptr<CUDABackend>      cuda_backend_;
    std::unique_ptr<BackendScheduler> scheduler_;
    // 【阶段1.5】SE3 offset 回落的独立 backend（持久成员，存活过最终 loss graph_compute）。
    // 独立 gallocr_：offset 回落的 graph_compute 只 release se3_backend_ 自己的 buffer，
    // 不碰主图（cpu_backend_）的 msa/pair/state。SE3 子图节点虽会 bind 到 se3_backend_
    // buffer，但主图最终 compute 时 bind_tensor 无条件 rebind 回主 backend（正确覆盖）。
    std::unique_ptr<CPUBackend>       se3_backend_;
    bool backend_ready_ = false;

    // 【SE3 offset scale · 可学习标量参数】(2026-08-23)
    // 全局共享一个 log-space 标量 PARAM：实际 scale = exp(log_scale)，保证恒正；
    // 再经 relu 实现的可微硬 clamp 约束到 [kSe3ScaleLo, kSe3ScaleHi]（基于前期 sweep：
    // 0.0003 尖峰、0.003 上行、0.001 最优 → 留 [1e-4, 5e-3] 余量）。
    // 仅多样本训练(PPML_MULTI_SAMPLE=1)或显式 PPML_SE3_LEARN_SCALE=1 时注册为可训练 PARAM；
    // 否则回落为冻结常量（env PPML_SE3_GRAPH_SCALE 覆盖，缺省 1e-3）。
    TensorF32* se3_log_scale_param_ = nullptr;  // 懒创建，跨 block/epoch 共享同一对象

    // 持有 backend buffer 的所有权（对标 ggml 中 backend 管理的 buffer 列表）
    std::vector<std::unique_ptr<Buffer>> param_buffers_;

    void ensure_backend_ready();
    void transfer_params_to_backend();

    // 收集所有参数 tensor 到指定容器 (weight/bias/gamma/beta), 顺序固定
    void collect_all_params(std::vector<TensorF32*>& param_tensors);
};

} // namespace ppml

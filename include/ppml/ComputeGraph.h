#pragma once
#include "Tensor.h"
#include "HashSet.h"
#include "FAPE.h"

#define GGML_MAX_DIMS           4
#define GGML_MAX_PARAMS         2048
// GGML_MAX_SRC 定义移至 Tensor.h（Tensor.h 的 src 数组依赖它）
#define GGML_MAX_N_THREADS      512
//#define GGML_MAX_OP_PARAMS      64


namespace ppml {

struct FAPEConfig;

// ============================================================
// unary op 枚举 — 对标 ggml_unary_op (去掉 ggml_ 前缀)
// ============================================================
enum unary_op {
    UNARY_OP_ABS = 0,
    UNARY_OP_SGN,
    UNARY_OP_NEG,
    UNARY_OP_STEP,
    UNARY_OP_RELU,
    UNARY_OP_GELU,
    UNARY_OP_GELU_QUICK,
    UNARY_OP_SILU,
    UNARY_OP_TANH,
    UNARY_OP_ELU,
    UNARY_OP_SIGMOID,
    UNARY_OP_HARDSIGMOID,
    UNARY_OP_HARDSWISH,
    UNARY_OP_EXP,
    UNARY_OP_LOG,
    UNARY_OP_SQRT,
    UNARY_OP_SIN,
    UNARY_OP_COS,
    UNARY_OP_COUNT,
};

// 从 op_params[0] 读取 unary op 子类型
inline unary_op get_unary_op(const Tensor<float>* node) {
    return static_cast<unary_op>(node->op_params[0]);
}

// 写入 unary op 子类型到 op_params[0]
inline void set_unary_op(Tensor<float>* node, unary_op op) {
    node->op_params[0] = static_cast<int32_t>(op);
}

    //遍历方式
enum cgraph_eval_order {
    CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
    CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
    CGRAPH_EVAL_ORDER_COUNT
};

    // computation graph
class ComputeGraph {
public:
    // ===== 禁止拷贝/移动（Context 管理生命周期）=====
    ComputeGraph(const ComputeGraph&) = delete;
    ComputeGraph& operator=(const ComputeGraph&) = delete;

    int  size()       const { return size_; }
    int  n_nodes()    const { return n_nodes_; }
    int  n_leafs()    const { return n_leafs_; }

    void graph_clear();
 
    int graph_size();
    //error: deduced class type ‘Tensor’ in function return type
    // use void*
    // or TensorF32
    // fixme: quantize case?
    TensorF32 * graph_node(int i);
    TensorF32 * graph_leaf(int i);
    TensorF32 ** graph_nodes();
    TensorF32 * graph_get_grad(const TensorF32 * node);

    // backward 梯度节点访问（gallocr liveness 修复用）：
    // grads[i] 存前向节点 i 的梯度（可能为 null）；数组大小为 visited_hash_set.size。
    TensorF32 ** graph_grads() { return grads; }
    int          graph_grad_slots() { return static_cast<int>(visited_hash_set.size); }

    void build_forward_expand(TensorF32 * tensor);
    void build_backward_expand(
        struct PPMLContext *  ctx,
        TensorF32  ** grad_accs);

    static ComputeGraph * new_graph_custom(struct PPMLContext * ctx, size_t size, bool grads);

    static ComputeGraph * new_graph(struct PPMLContext * ctx);

    static ComputeGraph * graph_dup(struct PPMLContext * ctx, ComputeGraph * cgraph, bool force_grads);

private:
    // this -- struct ComputeGraph * cgraph
    size_t visit_parents_graph( TensorF32 * node, bool compute);
    void build_forward_impl(TensorF32 * tensor, bool expand, bool compute);
    void graph_cpy(ComputeGraph * src, ComputeGraph * dst);
    void compute_backward(
        struct PPMLContext * ctx, int i, const bool * grads_needed);

    static size_t graph_nbytes(size_t size, bool grads);
    static void * incr_ptr_aligned(void ** p, size_t size, size_t align);

    static void sub_or_set(
        struct PPMLContext * ctx,
        ComputeGraph  * cgraph,
        size_t                isrc,
        TensorF32  * tensor);
    static void add1_or_set(
        struct PPMLContext * ctx,
        ComputeGraph  * cgraph,
        size_t                isrc,
        TensorF32  * tensor);
    static void acc_or_set(
        struct PPMLContext * ctx,
        ComputeGraph  * cgraph,
        size_t                isrc,
        TensorF32  * tensor,
        const  size_t         nb1,
        const  size_t         nb2,
        const  size_t         nb3,
        const  size_t         offset);
    static void add_or_set(
        struct PPMLContext * ctx,
        ComputeGraph  * cgraph,
        size_t                isrc,
        TensorF32  * tensor);

    // 构造只能通过静态工厂
    ComputeGraph() = default;  // placement new 构造

    friend class PPMLContext;        // Context 可以访问私有构造
    friend class BackendScheduler;   // Scheduler 可以临时修改 nodes/n_nodes_ 实现子图 view

    int size_;
    int n_nodes_;
    int n_leafs_;

    TensorF32 ** nodes;
    TensorF32 ** grads;     // the outputs of these tensors are the gradients of the nodes
    TensorF32 ** grad_accs; // accumulators for node gradients
    TensorF32 ** leafs;

    //一个图中存一个hash表，size的大小取决于图中nodes和leafs的数量之和
    struct HashSet visited_hash_set;

    enum cgraph_eval_order order;
};

// ============================================================
// 图节点构造函数（全局函数，公开 API）
// ============================================================

// 1. 算术
TensorF32* add_impl (TensorF32* a, TensorF32* b, bool inplace);
TensorF32* sub (TensorF32* a, TensorF32* b);
TensorF32* mul (TensorF32* a, TensorF32* b);
TensorF32* div (TensorF32* a, TensorF32* b);
TensorF32* add1_impl(TensorF32* a, TensorF32* scalar, bool inplace);
TensorF32* scale(TensorF32* a, float s);
TensorF32* neg (TensorF32* a);

// 2. 矩阵
TensorF32* mul_mat  (TensorF32* a, TensorF32* b);
TensorF32* out_prod (TensorF32* a, TensorF32* b);
TensorF32* transpose(TensorF32* a);

// Triangle Multiplication: left (B,I,K,D) × right (B,J,K,D) → (B,I,J,D)
// outgoing=true: einsum('bikd,bjkd->bijd'), false: einsum('bkid,bkjd->bijd')
TensorF32* triangle_mul(TensorF32* left, TensorF32* right, float L, bool outgoing);

// 3. 激活
TensorF32* softmax(TensorF32* a);
TensorF32* softmax_backward(TensorF32* grad, TensorF32* output);
TensorF32* silu   (TensorF32* a);
TensorF32* gelu   (TensorF32* a);
TensorF32* relu   (TensorF32* a);
TensorF32* relu_back(TensorF32* grad, TensorF32* x);
TensorF32* exp    (TensorF32* a);
TensorF32* leaky_relu(TensorF32* a, float alpha = 0.01f);
TensorF32* sigmoid(TensorF32* a);

// 4. 归一化
TensorF32* rms_norm(TensorF32* a, float eps = 1e-6f);
TensorF32* norm   (TensorF32* a, float eps = 1e-5f);

// 5. 规约
TensorF32* sum     (TensorF32* a);
TensorF32* mean    (TensorF32* a);
TensorF32* max_all (TensorF32* a);
TensorF32* sum_rows(TensorF32* a);

// 6. 形状
TensorF32* view     (TensorF32* a, const Shape& new_shape);
TensorF32* reshape  (TensorF32* a, const Shape& new_shape);
TensorF32* permute  (TensorF32* a, const std::vector<int>& dims);
TensorF32* unsqueeze(TensorF32* a, int dim);
TensorF32* concat     (const std::vector<TensorF32>& tensors, int dim);
TensorF32* concat_ptr (const std::vector<TensorF32*>& tensors, int dim);
TensorF32* concat_back(TensorF32* grad, TensorF32* src, int dim, int64_t offset);
TensorF32* repeat   (TensorF32* a, TensorF32* b);
TensorF32* repeat_back(TensorF32* a, TensorF32* b);
TensorF32* cont     (TensorF32* a);

// 7. 索引
TensorF32* get_rows(TensorF32* a, TensorF32* b);
// get_rows_back(dy, idx, W) — get_rows 的反向：将 dy 按 idx 散点累加回权重表 W
// dy: (K, M), idx: (K,), W: (N, M) → dW: (N, M)
TensorF32* get_rows_back(TensorF32* dy, TensorF32* idx, TensorF32* W);
TensorF32* set_rows(TensorF32* a, TensorF32* b, TensorF32* c);

// 7.5 SE3 消息传递三件套（方案 B：edge_gather / per_edge_matmul / scatter_add）
// ---------------------------------------------------------------------------
// edge_gather_rows(node_feat, edge_src_idx) — 按边源节点索引取行（gather）
//   node_feat: (N, C) 节点特征；edge_src_idx: (E,) 源节点 id（float-encoded int）
//   dst: (E, C) — dst[e,:] = node_feat[edge_src_idx[e],:]
//   ⚠️ 与 get_rows 语义一致（get_rows 本身即该操作），但独立 op 便于显式表达
//      SE3 消息传递阶段，并在 backward 时散点累加回节点梯度（见 scatter_add）。
TensorF32* edge_gather_rows(TensorF32* node_feat, TensorF32* edge_src_idx);

// per_edge_matmul(kernel, gathered) — 逐边矩阵乘（消息生成）
//   kernel: (E, M, K) 每条边独立的卷积核矩阵（row-major 展平 (E*M*K)）
//   gathered: (E, K) 该边源节点特征（edge_gather_rows 输出）
//   dst: (E, M) — dst[e,:] = kernel[e] @ gathered[e,:]
TensorF32* per_edge_matmul(TensorF32* kernel, TensorF32* gathered);

// scatter_add(msg, edge_tgt_idx, node_count) — 边消息散点累加到目标节点
//   msg: (E, M) 边消息；edge_tgt_idx: (E,) 目标节点 id（float-encoded int）
//   node_count: 节点数 N（用于确定 dst 形状，可为常量/占位节点）
//   dst: (N, M) — dst[tgt[e],:] += msg[e,:]
TensorF32* scatter_add(TensorF32* msg, TensorF32* edge_tgt_idx, int node_count);

// per_edge_matmul 的反向（由 compute_backward 内部构造，不直接调用）：
// per_edge_matmul_back_kernel(grad, gathered): grad(E,M)⊗gathered(E,K) → dkernel(E,M,K)
TensorF32* per_edge_matmul_back_kernel(TensorF32* grad, TensorF32* gathered);
// per_edge_matmul_back_gathered(grad, kernel): kernel(E,M,K)ᵀ@grad(E,M) → dgathered(E,K)
TensorF32* per_edge_matmul_back_gathered(TensorF32* grad, TensorF32* kernel);

// one_hot_seq 图版：seq 为一维扁平整数索引图节点 → get_rows(eye, seq)
// 返回 [num_classes, K] 图节点（ggml 布局 dims[0]=最内维）。依赖 get_rows（已实现）。
// ⚠️ seq 需先扁平为一维 (view/reshape kernel 待补时由调用方保证已扁平)
TensorF32* one_hot_seq_graph(TensorF32* seq, int num_classes = 21);

// outer_sum 图版：left [D,1,L,B] + right [D,L,1,B] → [D,L,L,B]（ggml 布局 dims[0]=最内维）
// 用 repeat 广播 + add_impl 组装。依赖 repeat / add_impl（均已有 kernel）。返回图节点。
TensorF32* outer_sum_graph(TensorF32* left, TensorF32* right);

// outer_product_mean 图版（msa2pair）：einsum('bikd,bjkd->bijd(de)', left, right/N)
//   left  [D, L, N, B]（ggml 布局 dims[0]=最内维；N=seq 维在 dims[2]）
//   right [D, L, N, B]
//   dst   [D*D, L, L, B]（特征笛卡尔积）
//   dst[(d1*D+d2), i, j, b] = (1/N) * sum_n left[d1,i,n,b] * right[d2,j,n,b]
// 注意：out_prod 只收缩 dims[1]，无法表达此处收缩 dims[2] 的 N 维，故新增 OP_OUTER_PROD_MEAN。
// N 为收缩维长度（seq 数），由调用方传入（== left->dims[2]）。
TensorF32* outer_product_mean(TensorF32* left, TensorF32* right, int N);

// outer_product_graph（gate 纯外积）：left [D,L,B] × right [D,L,B] → [D*D,L,L,B]
//   gate[(d1*D+d2), i, j, b] = left[d1,i,b] * right[d2,j,b]（无收缩，特征笛卡尔积）
// ggml 布局 dims[0]=最内维。对应值版 `MathUtils::outer_product` 的纯外积语义（但特征维扩展为 D*D）。
TensorF32* outer_product_graph(TensorF32* left, TensorF32* right);

// 8. 特殊
TensorF32* diag_mask_inf (TensorF32* a, int n_past);
TensorF32* diag_mask_zero(TensorF32* a, int n_past);
TensorF32* clamp(TensorF32* a, float min_val, float max_val);
TensorF32* sqr  (TensorF32* a);
TensorF32* sqrt (TensorF32* a);
TensorF32* abs  (TensorF32* a);
TensorF32* log  (TensorF32* a);
TensorF32* sin  (TensorF32* a);
TensorF32* cos  (TensorF32* a);

// 9. 损失
TensorF32* cross_entropy_loss(TensorF32* logits, TensorF32* targets);
TensorF32* torsion_angle_loss(TensorF32* pred, TensorF32* gt, TensorF32* chi_mask);
TensorF32* masked_msa_loss(TensorF32* logits, TensorF32* true_msa, TensorF32* bert_mask);
TensorF32* distogram_loss(
    TensorF32* logits_dist, TensorF32* logits_ω,
    TensorF32* logits_θ,    TensorF32* logits_ϕ,
    TensorF32* D_onehot,    TensorF32* Ω_onehot,
    TensorF32* Θ_onehot,    TensorF32* Φ_onehot,
    TensorF32* pair_mask);
TensorF32* angle_norm_loss(TensorF32* unnormed, TensorF32* seq_mask, float eps = 1e-6f);
TensorF32* supervised_chi_loss(
    TensorF32* unnormed, TensorF32* gt,
    TensorF32* chi_mask,  TensorF32* seq_mask,
    float chi_weight = 1.0f, float angle_norm_weight = 0.01f);
TensorF32* total_loss(
    TensorF32* loss_fape,
    TensorF32* loss_chi,
    TensorF32* loss_distogram,
    TensorF32* loss_msa,
    TensorF32* loss_conf);
TensorF32* plddt_loss(TensorF32* logits, TensorF32* lddt_onehot, TensorF32* ca_mask);
TensorF32* fape_loss(
    TensorF32* pred_coords,
    TensorF32* true_coords,
    TensorF32* frame_atom_indices,
    TensorF32* frames_mask,
    TensorF32* positions_mask,
    const FAPEConfig& config = FAPEConfig{});

// 为 FAPE 构建帧索引图节点。
// 坐标布局假设: (B, L, 3, 3) 展平为 (N_atoms=B*L*3, 3), 每残基 3 原子 [N, CA, C]
//   residue r 的 N = r*3+0, CA = r*3+1, C = r*3+2
// 返回: TensorF32* ({N_frames=B*L, 3}) — 每帧的 3 原子全局索引 (float-encoded int)
TensorF32* build_frame_atom_indices(int B, int L);

// 创建常量张量图节点 (leaf, 数据直接写入)
// shape: dims (变长) → 返回 {dims} 全 1 的 TensorF32* 节点
TensorF32* constant_ones(const std::vector<int64_t>& dims);
// 创建常量标量图节点
TensorF32* constant_scalar(float value);
// 从已有数据创建常量叶子图节点（逐元素拷贝，不参与求导）
TensorF32* constant_tensor(const std::vector<int64_t>& dims, const float* data);
// 动态一次性常量（TENSOR_FLAG_CONST 不置位）：Gallocr 填充后清空 const_data_，
// 用于 dropout 随机掩码等每迭代重建、不应累积宿主内存的叶子。
TensorF32* constant_tensor_dynamic(const std::vector<int64_t>& dims, const float* data);

// 10. 位置编码
TensorF32* rope(TensorF32* a, int n_past, int n_dims = 0);

// 11. 辅助
TensorF32* pad (TensorF32* a, const std::vector<int>& pad_dims);
TensorF32* acc (TensorF32* a, TensorF32* b, size_t nb1, size_t nb2, size_t nb3, size_t offset);
TensorF32* cpy (TensorF32* dst, TensorF32* src);
TensorF32* dup (TensorF32* a);

// 12. 标记
TensorF32* param(TensorF32* a);  // 设为可训练参数
TensorF32* loss (TensorF32* a);  // 设为损失节点

// 13. 常量
TensorF32* arange(float start, float end, float step = 1.0f);

} // namespace ppml
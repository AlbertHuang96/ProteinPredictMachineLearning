#pragma once
#include "Tensor.h"
#include "HashSet.h"

#define GGML_MAX_DIMS           4
#define GGML_MAX_PARAMS         2048
#define GGML_MAX_SRC            10
#define GGML_MAX_N_THREADS      512
#define GGML_MAX_OP_PARAMS      64


namespace rfaa {

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
    TensorF32 ** graph_nodes();
    TensorF32 * graph_get_grad(const TensorF32 * node);

    void build_forward_expand(TensorF32 * tensor);
    void build_backward_expand(
        struct RFAAContext *  ctx,
        TensorF32  ** grad_accs);

    static ComputeGraph * new_graph_custom(struct RFAAContext * ctx, size_t size, bool grads);

    static ComputeGraph * new_graph(struct RFAAContext * ctx);

    static ComputeGraph * graph_dup(struct RFAAContext * ctx, ComputeGraph * cgraph, bool force_grads);

private:
    // this -- struct ComputeGraph * cgraph
    size_t visit_parents_graph( TensorF32 * node, bool compute);
    void build_forward_impl(TensorF32 * tensor, bool expand, bool compute);
    void graph_cpy(ComputeGraph * src, ComputeGraph * dst);
    void compute_backward(
        struct RFAAContext * ctx, int i, const bool * grads_needed);

    static size_t graph_nbytes(size_t size, bool grads);
    static void * incr_ptr_aligned(void ** p, size_t size, size_t align);

    static void sub_or_set(
        struct RFAAContext * ctx,
        ComputeGraph  * cgraph,
        size_t                isrc,
        TensorF32  * tensor);
    static void add1_or_set(
        struct RFAAContext * ctx,
        ComputeGraph  * cgraph,
        size_t                isrc,
        TensorF32  * tensor);
    static void acc_or_set(
        struct RFAAContext * ctx,
        ComputeGraph  * cgraph,
        size_t                isrc,
        TensorF32  * tensor,
        const  size_t         nb1,
        const  size_t         nb2,
        const  size_t         nb3,
        const  size_t         offset);
    static void add_or_set(
        struct RFAAContext * ctx,
        ComputeGraph  * cgraph,
        size_t                isrc,
        TensorF32  * tensor);

    // 构造只能通过静态工厂
    ComputeGraph() = default;  // placement new 构造

    friend class RFAAContext;  // Context 可以访问私有构造

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
TensorF32* add_impl (TensorF32* a, TensorF32* b);
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

// 3. 激活
TensorF32* softmax(TensorF32* a);
TensorF32* silu   (TensorF32* a);
TensorF32* gelu   (TensorF32* a);
TensorF32* relu   (TensorF32* a);
TensorF32* leaky_relu(TensorF32* a, float alpha = 0.01f);

// 4. 归一化
TensorF32* rms_norm(TensorF32* a, float eps = 1e-6f);
TensorF32* norm   (TensorF32* a, float eps = 1e-5f);

// 5. 规约
TensorF32* sum     (TensorF32* a);
TensorF32* mean    (TensorF32* a);
TensorF32* sum_rows(TensorF32* a);

// 6. 形状
TensorF32* view     (TensorF32* a, const Shape& new_shape);
TensorF32* reshape  (TensorF32* a, const Shape& new_shape);
TensorF32* permute  (TensorF32* a, const std::vector<int>& dims);
TensorF32* unsqueeze(TensorF32* a, int dim);
TensorF32* concat   (const std::vector<TensorF32*>& tensors, int dim);
TensorF32* repeat   (TensorF32* a, TensorF32* b);
TensorF32* repeat_back(TensorF32* a, TensorF32* b);
TensorF32* cont     (TensorF32* a);

// 7. 索引
TensorF32* get_rows(TensorF32* a, TensorF32* b);
TensorF32* set_rows(TensorF32* a, TensorF32* b, TensorF32* c);

// 8. 特殊
TensorF32* diag_mask_inf (TensorF32* a, int n_past);
TensorF32* diag_mask_zero(TensorF32* a, int n_past);
TensorF32* clamp(TensorF32* a, float min_val, float max_val);
TensorF32* sqr  (TensorF32* a);
TensorF32* sqrt (TensorF32* a);
TensorF32* log  (TensorF32* a);
TensorF32* sin  (TensorF32* a);
TensorF32* cos  (TensorF32* a);

// 9. 损失
TensorF32* cross_entropy_loss(TensorF32* logits, TensorF32* targets);

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

} // namespace rfaa
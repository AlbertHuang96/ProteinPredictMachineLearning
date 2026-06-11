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
    Tensor * graph_node(int i);
    Tensor ** graph_nodes();
    Tensor * graph_get_grad(const Tensor * node);

    void build_forward_expand(struct Tensor * tensor);
    void build_backward_expand(
        struct RFAAContext *  ctx,
        struct Tensor  ** grad_accs);

    static ComputeGraph * new_graph_custom(struct RFAAContext * ctx, size_t size, bool grads);

    static ComputeGraph * new_graph(struct RFAAContext * ctx);

    static ComputeGraph * graph_dup(struct RFAAContext * ctx, struct ComputeGraph * cgraph, bool force_grads);

private:
    // this -- struct ComputeGraph * cgraph
    size_t visit_parents_graph( Tensor * node, bool compute);
    void build_forward_impl( Tensor * tensor, bool expand, bool compute);
    void graph_cpy(struct ComputeGraph * src, struct ComputeGraph * dst);
    void compute_backward(
        struct RFAAContext * ctx, int i, const bool * grads_needed);

    static size_t ComputeGraph::graph_nbytes(size_t size, bool grads);
    static void * ComputeGraph::incr_ptr_aligned(void ** p, size_t size, size_t align);

    static void sub_or_set(
        struct RFAAContext * ctx,
        struct ComputeGraph  * cgraph,
        size_t                isrc,
        Tensor  * tensor);
    static void add1_or_set(
        struct RFAAContext * ctx,
        struct ComputeGraph  * cgraph,
        size_t                isrc,
        Tensor  * tensor);
    static void acc_or_set(
        struct RFAAContext * ctx,
        struct ComputeGraph  * cgraph,
        size_t                isrc,
        Tensor  * tensor,
        const  size_t         nb1,
        const  size_t         nb2,
        const  size_t         nb3,
        const  size_t         offset);
    static void add_or_set(
        struct RFAAContext * ctx,
        struct ComputeGraph  * cgraph,
        size_t                isrc,
        Tensor  * tensor);

    // 构造只能通过静态工厂
    ComputeGraph() = default;  // placement new 构造

    friend class RFAAContext;  // Context 可以访问私有构造

    int size;
    int n_nodes;
    int n_leafs;

    struct Tensor ** nodes;
    struct Tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    struct Tensor ** grad_accs; // accumulators for node gradients
    struct Tensor ** leafs;

    //一个图中存一个hash表，size的大小取决于图中nodes和leafs的数量之和
    struct HashSet visited_hash_set;

    enum cgraph_eval_order order;
};



} // namespace rfaa
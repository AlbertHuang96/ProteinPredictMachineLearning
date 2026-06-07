
#include "Tensor.h"
#include "HashSet.h"

namespace rfaa {

    //遍历方式
enum cgraph_eval_order {
    CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
    CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
    CGRAPH_EVAL_ORDER_COUNT
};

    // computation graph
struct ComputeGraph {
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
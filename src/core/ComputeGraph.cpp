#include "ComputeGraph.h"


namespace rfaa {


    static size_t visit_parents_graph(struct ComputeGraph * cgraph, struct Tensor * node, bool compute) {
        if (node->op != GGML_OP_NONE && compute) {
            node->flags |= GGML_TENSOR_FLAG_COMPUTE;
        }
 
        const size_t node_hash_pos = hash_find(&cgraph->visited_hash_set, node);
        //GGML_ASSERT(node_hash_pos != GGML_HASHSET_FULL);
 
        if (bitset_get(cgraph->visited_hash_set.used, node_hash_pos)) {
        // already visited
 
        if (compute) {
            // update the compute flag regardless
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                struct Tensor * src = node->src[i];
                if (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    rfaa::visit_parents_graph(cgraph, src, true);
                }
            }
        }
 
        return node_hash_pos;
    }
 
    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.keys[node_hash_pos] = node;
    bitset_set(cgraph->visited_hash_set.used, node_hash_pos);
    cgraph->use_counts[node_hash_pos] = 0;
 
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const int k =
            (cgraph->order == CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? i :
            (cgraph->order == CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;
 
        struct Tensor * src = node->src[k];
        if (src) {
            const size_t src_hash_pos = rfaa::visit_parents_graph(cgraph, src, compute);
 
            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }
 
    if (node->op == GGML_OP_NONE && !(node->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        //GGML_ASSERT(cgraph->n_leafs < cgraph->size);
 
        if (strlen(node->name) == 0) {
            //ggml_format_name(node, "leaf_%d", cgraph->n_leafs);
        }
 
        cgraph->leafs[cgraph->n_leafs] = node;
        cgraph->n_leafs++;
    } else {
        //GGML_ASSERT(cgraph->n_nodes < cgraph->size);
 
        if (strlen(node->name) == 0) {
            //ggml_format_name(node, "node_%d", cgraph->n_nodes);
        }
 
        cgraph->nodes[cgraph->n_nodes] = node;
        cgraph->n_nodes++;
    }
 
    return node_hash_pos;
}


} // namespace rfaa
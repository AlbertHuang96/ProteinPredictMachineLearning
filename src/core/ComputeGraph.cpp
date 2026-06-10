#include "ComputeGraph.h"


namespace rfaa {

void graph_clear(struct ComputeGraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    hash_set_reset(&cgraph->visited_hash_set);
}
 
int graph_size(struct ComputeGraph * cgraph) {
    return cgraph->size;
}
 
struct Tensor * graph_node(struct ComputeGraph * cgraph, int i) {
    if (i < 0) {
        //GGML_ASSERT(cgraph->n_nodes + i >= 0);
        return cgraph->nodes[cgraph->n_nodes + i];
    }
 
    //GGML_ASSERT(i < cgraph->n_nodes);
    return cgraph->nodes[i];
}
 
struct Tensor ** graph_nodes(struct ComputeGraph * cgraph) {
    return cgraph->nodes;
}

struct ComputeGraph * new_graph_custom(struct RFAAContext * ctx, size_t size, bool grads) {
    const size_t obj_size = graph_nbytes(size, grads);
    struct RFAAObject * obj = new_object(ctx, RFAA_OBJECT_TYPE_GRAPH, obj_size);
    struct ComputeGraph * cgraph = (struct ComputeGraph *) ((char *) ctx->mem_buffer + obj->offs);
 
    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = ggml_hash_size(size * 2);
 
    void * p = cgraph + 1;
 
    struct Tensor ** nodes_ptr      =         incr_ptr_aligned(&p, size      * sizeof(struct Tensor *), sizeof(struct Tensor *));
    struct Tensor ** leafs_ptr      =         incr_ptr_aligned(&p, size      * sizeof(struct Tensor *), sizeof(struct Tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_size * sizeof(int32_t), sizeof(int32_t));
    struct Tensor ** hash_keys_ptr  =         incr_ptr_aligned(&p, hash_size * sizeof(struct Tensor *), sizeof(struct Tensor *));
    struct Tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_size * sizeof(struct Tensor *), sizeof(struct Tensor *)) : NULL;
    struct Tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_size * sizeof(struct Tensor *), sizeof(struct Tensor *)) : NULL;
 
    bitset_t * hash_used = incr_ptr_aligned(&p, bitset_size(hash_size) * sizeof(ggml_bitset_t), sizeof(ggml_bitset_t));
 
    // check that we allocated the correct amount of memory
    assert(obj_size == (size_t)((char *)p - (char *)cgraph));
 
    *cgraph = (struct ComputeGraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_size, hash_used, hash_keys_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT,
        /*.uid          =*/ 0,
    };
 
    hash_set_reset(&cgraph->visited_hash_set);
    if (grads) {
        memset(cgraph->grads,     0, hash_size*sizeof(struct Tensor *));
        memset(cgraph->grad_accs, 0, hash_size*sizeof(struct Tensor *));
    }
 
    return cgraph;
}
 
struct ComputeGraph * new_graph(struct RFAAContext * ctx) {
    return new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}

static size_t visit_parents_graph(struct ComputeGraph * cgraph, struct Tensor * node, bool compute) {
    
    if (node->op != OP_NONE && compute) {
        node->flags |= TENSOR_FLAG_COMPUTE;
    }
 
    const size_t node_hash_pos = hash_find(&cgraph->visited_hash_set, node);
    //GGML_ASSERT(node_hash_pos != GGML_HASHSET_FULL);
 
    if (bitset_get(cgraph->visited_hash_set.used, node_hash_pos)) {
    // already visited
 
        if (compute) {
            // update the compute flag regardless
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                struct Tensor * src = node->src[i];
                if (src && ((src->flags & TENSOR_FLAG_COMPUTE) == 0)) {
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
    

    if (node->op == OP_NONE && !(node->flags & TENSOR_FLAG_PARAM)) {
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

static void build_forward_impl(struct ComputeGraph * cgraph, struct Tensor * tensor, bool expand, bool compute) {
    if (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        graph_clear(cgraph);
    }
 
    const int n_old = cgraph->n_nodes;
 
    rfaa::visit_parents_graph(cgraph, tensor, compute);
 
    const int n_new = cgraph->n_nodes - n_old;
    //GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);
 
    if (n_new > 0) {
        // the last added node should always be starting point
        //GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
        assert(cgraph->n_nodes[cgraph->n_nodes - 1] == tensor);
    }
}

void build_forward_expand(struct ComputeGraph * cgraph, struct Tensor * tensor) {
    build_forward_impl(cgraph, tensor, true, true);
}

void build_backward_expand(
        struct RFAAContext *  ctx,
        struct ComputeGraph  *  cgraph,
        struct Tensor  ** grad_accs) {
    //GGML_ASSERT(cgraph->n_nodes > 0);
    //GGML_ASSERT(cgraph->grads);
    //GGML_ASSERT(cgraph->grad_accs);

    assert(cgraph->n_nodes > 0);
    assert(cgraph->grads);
    assert(cgraph->grad_accs);
 
    const int n_nodes_f = cgraph->n_nodes;
 
    memset(cgraph->grads,     0, cgraph->visited_hash_set.size*sizeof(struct Tensor *));
    memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.size*sizeof(struct Tensor *));
    bool * grads_needed = calloc(cgraph->visited_hash_set.size, sizeof(bool));
 
    {
        bool any_params = false;
        bool any_loss   = false;
        for (int i = 0; i < n_nodes_f; ++i) {
            struct Tensor * node = cgraph->nodes[i];
            any_params = any_params || (node->flags & TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (node->flags & TENSOR_FLAG_LOSS);
        }
        //GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        //GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }
 
    for (int i = 0; i < n_nodes_f; ++i) {
        struct Tensor * node = cgraph->nodes[i];
 
        if (node->type == TENSOR_TYPE_I32) {
            continue;
        }
 
        bool node_needs_grad = (node->flags & TENSOR_FLAG_PARAM) || (node->flags & TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        switch (node->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            case OP_IM2COL:      // only used for its shape
            case OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                break;
            case OP_UNARY: {
                const enum ggml_unary_op uop = ggml_get_unary_op(node);
                // SGN and STEP unary ops are piecewise constant
                if (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } break;
 
            // gradients in node->src[1] for one reason or another have no effect on output gradients
            case OP_CPY:           // gradients in CPY target are irrelevant
            case OP_GET_ROWS:      // row indices not differentiable
            case OP_GET_ROWS_BACK: // same as for GET_ROWS
            case OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                break;
 
            default:
                break;
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (!node->src[j] || ignore_src[j] || !grads_needed[hash_find(&cgraph->visited_hash_set, node->src[j])]) {
                continue;
            }
            //GGML_ASSERT(node->src[j]->type == GGML_TYPE_F32 || node->src[j]->type == GGML_TYPE_F16);
            //assert(node->src[j]->type == TENSOR_TYPE_F32 || node->src[j]->type == TENSOR_TYPE_F16);
            node_needs_grad = true;
            break;
        }
        if (!node_needs_grad) {
            continue;
        }
 
        // inplace operations are currently not supported
        ///GGML_ASSERT(!node->view_src || node->op == OP_CPY || node->op == OP_VIEW ||
            //node->op == OP_RESHAPE || node->op == OP_PERMUTE || node->op == OP_TRANSPOSE);
 
        const size_t ihash = hash_find(&cgraph->visited_hash_set, node);
        //GGML_ASSERT(ihash != GGML_HASHSET_FULL);
        //GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_set.used, ihash));
        assert(ihash != HASHSET_FULL);
        assert(bitset_get(cgraph->visited_hash_set.used, ihash));

        if (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } else if (node->flags & TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }
 
    for (int i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        compute_backward(ctx, cgraph, i, grads_needed);
    }
 
    free(grads_needed);
}
 
static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {
    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}
 
static size_t graph_nbytes(size_t size, bool grads) {
    size_t hash_size = hash_size(size * 2);
    void * p = 0;
    incr_ptr_aligned(&p, sizeof(struct ComputeGraph), 1);
    incr_ptr_aligned(&p, size * sizeof(struct Tensor *), sizeof(struct Tensor *)); // nodes
    incr_ptr_aligned(&p, size * sizeof(struct Tensor *), sizeof(struct Tensor *)); // leafs
    incr_ptr_aligned(&p, hash_size * sizeof(int32_t), sizeof(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_size * sizeof(struct Tensor *), sizeof(struct Tensor *)); // hash keys
    if (grads) {
        incr_ptr_aligned(&p, hash_size * sizeof(struct Tensor *), sizeof(struct Tensor *)); // grads
        incr_ptr_aligned(&p, hash_size * sizeof(struct Tensor *), sizeof(struct Tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, bitset_size(hash_size) * sizeof(bitset_t), sizeof(bitset_t));
 
    size_t nbytes = (size_t) p;
    return nbytes;
}


} // namespace rfaa
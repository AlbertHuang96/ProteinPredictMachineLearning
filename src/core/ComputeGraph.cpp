#include "ComputeGraph.h"


namespace rfaa {

void ComputeGraph::graph_clear() {
    this->n_leafs = 0;
    this->n_nodes = 0;
    hash_set_reset(&this->visited_hash_set);
}
 
int ComputeGraph::graph_size() {
    return this->size;
}
 
Tensor * ComputeGraph::graph_node(int i) {
    if (i < 0) {
        //GGML_ASSERT(cgraph->n_nodes + i >= 0);
        return this->nodes[this->n_nodes + i];
    }
 
    //GGML_ASSERT(i < cgraph->n_nodes);
    return this->nodes[i];
}
 
Tensor ** ComputeGraph::graph_nodes() {
    return this->nodes;
}

static ComputeGraph * ComputeGraph::new_graph_custom(struct RFAAContext * ctx, size_t size, bool grads) {
    const size_t obj_size = graph_nbytes(size, grads);
    struct RFAAObject * obj = new_object(ctx, RFAA_OBJECT_TYPE_GRAPH, obj_size);
    ComputeGraph * cgraph = (ComputeGraph *) ((char *) ctx->mem_buffer + obj->offs);
 
    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = hash_size(size * 2);
 
    void * p = cgraph + 1;
 
    Tensor ** nodes_ptr      =         incr_ptr_aligned(&p, size      * sizeof(struct Tensor *), sizeof(struct Tensor *));
    Tensor ** leafs_ptr      =         incr_ptr_aligned(&p, size      * sizeof(struct Tensor *), sizeof(struct Tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_size * sizeof(int32_t), sizeof(int32_t));
    Tensor ** hash_keys_ptr  =         incr_ptr_aligned(&p, hash_size * sizeof(struct Tensor *), sizeof(struct Tensor *));
    Tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_size * sizeof(struct Tensor *), sizeof(struct Tensor *)) : NULL;
    Tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_size * sizeof(struct Tensor *), sizeof(struct Tensor *)) : NULL;
 
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
        memset(cgraph->grads,     0, hash_size*sizeof(Tensor *));
        memset(cgraph->grad_accs, 0, hash_size*sizeof(Tensor *));
    }
 
    return cgraph;
}
 
static ComputeGraph * ComputeGraph::new_graph(struct RFAAContext * ctx) {
    return new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}

static ComputeGraph * ComputeGraph::graph_dup(struct RFAAContext * ctx, struct ComputeGraph * cgraph, bool force_grads) {
    ComputeGraph * result = new_graph_custom(ctx, cgraph->size, cgraph->grads || force_grads);
    graph_cpy(cgraph, result);
    return result;
}

size_t ComputeGraph::visit_parents_graph(Tensor * node, bool compute) {
    
    if (node->op != OP_NONE && compute) {
        node->flags |= TENSOR_FLAG_COMPUTE;
    }
 
    const size_t node_hash_pos = hash_find(&this->visited_hash_set, node);
    //GGML_ASSERT(node_hash_pos != GGML_HASHSET_FULL);
 
    if (bitset_get(this->visited_hash_set.used, node_hash_pos)) {
    // already visited
 
        if (compute) {
            // update the compute flag regardless
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                Tensor * src = node->src[i];
                if (src && ((src->flags & TENSOR_FLAG_COMPUTE) == 0)) {
                    rfaa::visit_parents_graph(src, true);
                }
            }
        }
 
        return node_hash_pos;
    }
 
    // This is the first time we see this node in the current graph.
    this->visited_hash_set.keys[node_hash_pos] = node;
    bitset_set(this->visited_hash_set.used, node_hash_pos);
    this->use_counts[node_hash_pos] = 0;
 
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const int k =
            (this->order == CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? i :
            (this->order == CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;
 
        Tensor * src = node->src[k];
        if (src) {
            const size_t src_hash_pos = rfaa::visit_parents_graph(src, compute);
 
            // Update the use count for this operand.
            this->use_counts[src_hash_pos]++;
        }
    }
    

    if (node->op == OP_NONE && !(node->flags & TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        //GGML_ASSERT(cgraph->n_leafs < cgraph->size);
 
        if (strlen(node->name) == 0) {
            //ggml_format_name(node, "leaf_%d", cgraph->n_leafs);
        }
 
        this->leafs[this->n_leafs] = node;
        this->n_leafs++;
    } else {
        //GGML_ASSERT(cgraph->n_nodes < cgraph->size);
 
        if (strlen(node->name) == 0) {
            //ggml_format_name(node, "node_%d", cgraph->n_nodes);
        }
 
        this->nodes[this->n_nodes] = node;
        this->n_nodes++;
    }
 
    return node_hash_pos;
}

void ComputeGraph::build_forward_impl(Tensor * tensor, bool expand, bool compute) {
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

void ComputeGraph::build_forward_expand(Tensor * tensor) {
    build_forward_impl(tensor, true, true);
}

void ComputeGraph::build_backward_expand(
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
            assert(node->src[j]->type == TENSOR_TYPE_F32 || node->src[j]->type == TENSOR_TYPE_F16);
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
            cgraph->grad_accs[ihash] = new_tensor(ctx, TENSOR_TYPE_F32, GGML_MAX_DIMS, node->ne);
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

Tensor * ComputeGraph::graph_get_grad(const Tensor * node) {
    const size_t igrad = hash_find(&this->visited_hash_set, node);
    return igrad != HASHSET_FULL && bitset_get(this->visited_hash_set.used, igrad) && this->grads ? this->grads[igrad] : NULL;
}

void ComputeGraph::compute_backward(
    struct RFAAContext * ctx, int i, const bool * grads_needed) {
    Tensor * tensor = this->nodes[i];
    Tensor * grad   = graph_get_grad(this, tensor);
 
    if (!grad) {
        return;
    }
 
    Tensor * src0 = tensor->src[0];
    Tensor * src1 = tensor->src[1];
    Tensor * src2 = tensor->src[2];
    HashSet * hash_set = &this->visited_hash_set;
    const size_t isrc0 = src0 ? hash_find(hash_set, src0) : (size_t) -1;
    const size_t isrc1 = src1 ? hash_find(hash_set, src1) : (size_t) -1;
    const size_t isrc2 = src2 ? hash_find(hash_set, src2) : (size_t) -1;
    const bool src0_needs_grads = src0 && isrc0 != HASHSET_FULL && bitset_get(hash_set->used, isrc0) && grads_needed[isrc0];
    const bool src1_needs_grads = src1 && isrc1 != HASHSET_FULL && bitset_get(hash_set->used, isrc1) && grads_needed[isrc1];
    const bool src2_needs_grads = src2 && isrc2 != HASHSET_FULL && bitset_get(hash_set->used, isrc2) && grads_needed[isrc2];
 
    switch (tensor->op) {
        case OP_ADD: {
            if (src0_needs_grads) {
                add_or_set(ctx, cgraph, isrc0, grad);
            }
            if (src1_needs_grads) {
                Tensor * tmp = grad;
                if (!src0->same_shape(src1)) {
                    //tmp = src1->repeat_back(tmp);
                    tmp = repeat_back(tmp, src1);
                }
                add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } break;
        case OP_ADD1: {
            if (src0_needs_grads) {
                add_or_set(ctx, cgraph, isrc0, grad);
            }
            if (src1_needs_grads) {
                add_or_set(ctx, cgraph, isrc1, ggml_mean(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } break;
        case OP_SUB: {
            if (src0_needs_grads) {
                add_or_set(ctx, cgraph, isrc0, grad);
            }
            if (src1_needs_grads) {
                sub_or_set(ctx, cgraph, isrc1, grad);
            }
        } break;
        case OP_MUL: {
            if (src0_needs_grads) {
                add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            if (src1_needs_grads) {
                Tensor * tmp = ggml_mul(ctx, src0, grad);
                if (!tmp->same_shape(src1)) {
                    //tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } break;
        case OP_DIV: {
            if (src0_needs_grads) {
                add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            if (src1_needs_grads) {
                sub_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } break;
        case OP_MUL_MAT: {
            // https://cs231n.github.io/optimization-2/#staged
            // # forward pass
            // s0 = np.random.randn(5, 10)
            // s1 = np.random.randn(10, 3)
            // t = s0.dot(s1)
 
            // # now suppose we had the gradient on t from above in the circuit
            // dt = np.random.randn(*t.shape) # same shape as t
            // ds0 = dt.dot(s1.T) #.T gives the transpose of the matrix
            // ds1 = t.T.dot(dt)
 
            // tensor.shape [m,p,qq,rr]
            // src0.shape   [n,m,q1,r1]
            // src1.shape   [n,p,qq,rr]
 
            if (src0_needs_grads) {
                //GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                //GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                assert(grad->Shape().dims[2] == src1->Shape().dims[2]);
                assert(grad->Shape().dims[3] == src1->Shape().dims[3]);
                Tensor * tmp =
                    ggml_out_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                if (!tmp->same_shape(src0)) {
                    //GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    //GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    //GGML_ASSERT(tmp->ne[3] == 1);
                    assert(tmp->Shape().dims[0] == src0->Shape().dims[0]);
                    assert(tmp->Shape().dims[1] == src0->Shape().dims[1]);
                    assert(tmp->Shape().dims[3] == 1);
 
                    const int64_t nr2 = tmp->Shape().dims[2] / src0->Shape().dims[2];
                    const size_t nb2 = tmp->nb[2] * nr2;
                    const size_t nb3 = tmp->nb[2];
 
                    tmp = ggml_view_4d(ctx, tmp, src0->Shape().dims[0], src0->Shape().dims[1], src0->Shape().dims[2], nr2, tmp->nb[1], nb2, nb3, 0);
                    tmp = repeat_back(ctx, tmp, src0);
                    //tmp = 
                }
                add_or_set(ctx, cgraph, isrc0, tmp);
            }
            if (src1_needs_grads) {
                add_or_set(ctx, cgraph, isrc1,
                        // ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                        //     ggml_cont(ctx,                  // [m,n,q1,r1]
                        //         ggml_transpose(ctx, src0)), // [m,n,q1,r1]
                        //     grad),                          // [m,p,qq,rr]
 
                        // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                        // avoid transpose of src0, rather transpose smaller tensor->grad
                        // and then use ggml_out_prod
                        ggml_out_prod(ctx,      // [n,p,qq,rr]
                            src0,               // [n,m,q1,r1]
                            ggml_transpose(ctx, // [p,m,qq,rr]
                                grad)));        // [m,p,qq,rr]
            }
        } break;
        case OP_NONE: {

        } break;
        case OP_COUNT:
        default: {

        }
    }
}

static void ComputeGraph::add_or_set(
        struct RFAAContext * ctx,
        struct ComputeGraph  * cgraph,
        size_t                isrc,
        Tensor  * tensor) {
    Tensor * src = cgraph->visited_hash_set.keys[isrc];
    //GGML_ASSERT(src);
    assert(src);
    if (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = add_impl(cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } else {
        cgraph->grads[isrc] = tensor;
    }
    //ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    //build_forward_expand(cgraph, cgraph->grads[isrc]);
    cgraph->build_forward_expand(cgraph->grads[isrc]);
}
 
static void ComputeGraph::acc_or_set(
        struct RFAAContext * ctx,
        struct ComputeGraph  * cgraph,
        size_t                isrc,
        Tensor  * tensor,
        const  size_t         nb1,
        const  size_t         nb2,
        const  size_t         nb3,
        const  size_t         offset) {
    Tensor * src = cgraph->visited_hash_set.keys[isrc];
    //GGML_ASSERT(src);
    if (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = acc_impl(cgraph->grads[isrc], tensor, nb1, nb2, nb3, offset, cgraph->grad_accs[isrc]);
    } else {
        //Tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        //cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
    }
    //ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_set.keys[isrc]->name);
    //build_forward_expand(cgraph, cgraph->grads[isrc]);
    cgraph->build_forward_expand(cgraph->grads[isrc]);
}
 
static void ComputeGraph::add1_or_set(
        struct RFAAContext * ctx,
        struct ComputeGraph  * cgraph,
        size_t                isrc,
        Tensor  * tensor) {
    Tensor * src = cgraph->visited_hash_set.keys[isrc];
    //GGML_ASSERT(src);
    assert(src);
    if (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = add1_impl(cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } else {
        //cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);

    }
    //ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    //build_forward_expand(cgraph, cgraph->grads[isrc]);
    cgraph->build_forward_expand(cgraph->grads[isrc]);
}
 
static void ComputeGraph::sub_or_set(
        struct RFAAContext * ctx,
        struct ComputeGraph  * cgraph,
        size_t                isrc,
        Tensor  * tensor) {
    Tensor * src = cgraph->visited_hash_set.keys[isrc];
    //GGML_ASSERT(src);
    assert(src);
    if (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } else {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    //ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    //build_forward_expand(cgraph, cgraph->grads[isrc]);
    cgraph->build_forward_expand(cgraph->grads[isrc]);
}

void ComputeGraph::graph_cpy(struct ComputeGraph * src, struct ComputeGraph * dst) {
    //GGML_ASSERT(dst->size >= src->n_leafs);
    //GGML_ASSERT(dst->size >= src->n_nodes);
    //GGML_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

    assert(dst->size >= src->n_leafs);
    assert(dst->size >= src->n_nodes);
    assert(dst->visited_hash_set.size >= src->visited_hash_set.size);
 
    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->order   = src->order;
 
    for (int i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }
 
    for (int i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }
 
    for (size_t i = 0; i < src->visited_hash_set.size; ++i) {
        // copy all hashset keys (tensors) that are in use
        if (bitset_get(src->visited_hash_set.used, i)) {
            size_t new_hash_pos = hash_insert(&dst->visited_hash_set, src->visited_hash_set.keys[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }
 
    if (dst->grads) {
        memset(dst->grads,     0, dst->visited_hash_set.size*sizeof(struct Tensor *));
        memset(dst->grad_accs, 0, dst->visited_hash_set.size*sizeof(struct Tensor *));
    }
    if (src->grads) {
        //GGML_ASSERT(dst->grads     != NULL);
        //GGML_ASSERT(dst->grad_accs != NULL);
        assert(dst->grads     != NULL);
        assert(dst->grad_accs != NULL);

        for (int i = 0; i < src->n_nodes; ++i) {
            const size_t igrad_src = hash_find(&src->visited_hash_set, src->nodes[i]);
            const size_t igrad_dst = hash_find(&dst->visited_hash_set, dst->nodes[i]);
 
            //GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
            //GGML_ASSERT(bitset_get(src->visited_hash_set.used, igrad_src));
            //GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
            //GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

            assert(igrad_src != HASHSET_FULL);
            assert(bitset_get(src->visited_hash_set.used, igrad_src));
            assert(igrad_dst != HASHSET_FULL);
            assert(bitset_get(dst->visited_hash_set.used, igrad_dst));

 
            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

 
static void * ComputeGraph::incr_ptr_aligned(void ** p, size_t size, size_t align) {
    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}
 
static size_t ComputeGraph::graph_nbytes(size_t size, bool grads) {
    size_t hash_size = hash_size(size * 2);
    void * p = 0;
    ComputeGraph::incr_ptr_aligned(&p, sizeof(ComputeGraph), 1);
    ComputeGraph::incr_ptr_aligned(&p, size * sizeof(Tensor *), sizeof( Tensor *)); // nodes
    ComputeGraph::incr_ptr_aligned(&p, size * sizeof(Tensor *), sizeof( Tensor *)); // leafs
    ComputeGraph::incr_ptr_aligned(&p, hash_size * sizeof(int32_t), sizeof(int32_t)); // use_counts
    ComputeGraph::incr_ptr_aligned(&p, hash_size * sizeof(Tensor *), sizeof( Tensor *)); // hash keys
    if (grads) {
        ComputeGraph::incr_ptr_aligned(&p, hash_size * sizeof( Tensor *), sizeof( Tensor *)); // grads
        ComputeGraph::incr_ptr_aligned(&p, hash_size * sizeof( Tensor *), sizeof( Tensor *)); // grad_accs
    }
    ComputeGraph::incr_ptr_aligned(&p, bitset_size(hash_size) * sizeof(bitset_t), sizeof(bitset_t));
 
    size_t nbytes = (size_t) p;
    return nbytes;
}

void ggml_compute_forward_out_prod(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
 
    const ggml_tensor * src0 = dst->src[0];
 
    switch (src0->type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_out_prod_q_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ABORT("fatal error"); // todo
                // ggml_compute_forward_out_prod_f16_f32(params, dst);
            }
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_out_prod_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}


} // namespace rfaa
#include "rfaa/ComputeGraph.h"
#include "rfaa/Context.h"
#include <cassert>

namespace rfaa {

void ComputeGraph::graph_clear() {
    this->n_leafs_ = 0;
    this->n_nodes_ = 0;
    hash_set_reset(&this->visited_hash_set);
}
 
int ComputeGraph::graph_size() {
    return this->size();
}
 
TensorF32 * ComputeGraph::graph_node(int i) {
    if (i < 0) {
        return this->nodes[this->n_nodes_ + i];
    }
    return this->nodes[i];
}

TensorF32 * ComputeGraph::graph_leaf(int i) {
    if (i < 0) {
        return this->leafs[this->n_leafs_ + i];
    }
    return this->leafs[i];
}
 
TensorF32 ** ComputeGraph::graph_nodes() {
    return this->nodes;
}

ComputeGraph * ComputeGraph::new_graph_custom(struct RFAAContext * ctx, size_t size, bool grads) {
    const size_t obj_size = graph_nbytes(size, grads);
    struct RFAAObject * obj = ctx->new_object(RFAA_OBJECT_TYPE_GRAPH, obj_size);
    ComputeGraph * cgraph = (ComputeGraph *) ((char *) ctx->mem_buffer + obj->offs);
 
    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = bitset_size(size * 2);
 
    void * p = cgraph + 1;
 
    TensorF32 ** nodes_ptr      = (TensorF32 **) incr_ptr_aligned(&p, size      * sizeof(TensorF32 *), sizeof(TensorF32 *));
    TensorF32 ** leafs_ptr      = (TensorF32 **) incr_ptr_aligned(&p, size      * sizeof(TensorF32 *), sizeof(TensorF32 *));
    int32_t             * use_counts_ptr = (int32_t *) incr_ptr_aligned(&p, hash_size * sizeof(int32_t), sizeof(int32_t));
    TensorF32 ** hash_keys_ptr  = (TensorF32 **) incr_ptr_aligned(&p, hash_size * sizeof(TensorF32 *), sizeof(TensorF32 *));
    TensorF32 ** grads_ptr      = grads ? (TensorF32 **) incr_ptr_aligned(&p, hash_size * sizeof(TensorF32 *), sizeof(TensorF32 *)) : NULL;
    TensorF32 ** grad_accs_ptr  = grads ? (TensorF32 **) incr_ptr_aligned(&p, hash_size * sizeof(TensorF32 *), sizeof(TensorF32 *)) : NULL;
 
    bitset_t * hash_used = (bitset_t *) incr_ptr_aligned(&p, bitset_size(hash_size) * sizeof(bitset_t), sizeof(bitset_t));
 
    // check that we allocated the correct amount of memory
    assert(obj_size == (size_t)((char *)p - (char *)cgraph));
 
    cgraph->size_    = size;
    cgraph->n_nodes_ = 0;
    cgraph->n_leafs_ = 0;
    cgraph->nodes    = nodes_ptr;
    cgraph->grads    = grads_ptr;
    cgraph->grad_accs = grad_accs_ptr;
    cgraph->leafs    = leafs_ptr;
    cgraph->order    = CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT;
    cgraph->visited_hash_set.size = hash_size;
    cgraph->visited_hash_set.used = hash_used;
    cgraph->visited_hash_set.keys = (void **)hash_keys_ptr;
 
    hash_set_reset(&cgraph->visited_hash_set);
    if (grads) {
        memset(cgraph->grads,     0, hash_size*sizeof(TensorF32 *));
        memset(cgraph->grad_accs, 0, hash_size*sizeof(TensorF32 *));
    }
 
    return cgraph;
}
 
ComputeGraph * ComputeGraph::new_graph(struct RFAAContext * ctx) {
    return new_graph_custom(ctx, 2048, false);
}

ComputeGraph * ComputeGraph::graph_dup(struct RFAAContext * ctx, struct ComputeGraph * cgraph, bool force_grads) {
    ComputeGraph * result = new_graph_custom(ctx, cgraph->size(), cgraph->grads || force_grads);
    cgraph->graph_cpy(cgraph, result);
    // there is no this pointer in the graph_cpy so it behaves like a static function
    return result;
}

size_t ComputeGraph::visit_parents_graph(TensorF32 * node, bool compute) {
    
    if (node->op != OP_NONE && compute) {
        node->flag |= TENSOR_FLAG_COMPUTE;
    }
 
    const size_t node_hash_pos = hash_find(&this->visited_hash_set, node);
    //GGML_ASSERT(node_hash_pos != GGML_HASHSET_FULL);
 
    if (bitset_get(this->visited_hash_set.used, node_hash_pos)) {
    // already visited
 
        if (compute) {
            // update the compute flag regardless
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                TensorF32 * src = node->src[i];
                if (src && ((src->flag & TENSOR_FLAG_COMPUTE) == 0)) {
                    this->visit_parents_graph(src, true);
                }
            }
        }
 
        return node_hash_pos;
    }
 
    // This is the first time we see this node in the current graph.
    this->visited_hash_set.keys[node_hash_pos] = node;
    bitset_set(this->visited_hash_set.used, node_hash_pos);
    // this->use_counts[node_hash_pos] = 0;  // use_counts not available
    
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const int k =
            (this->order == CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? i :
            (this->order == CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;
 
        TensorF32 * src = node->src[k];
        if (src) {
            this->visit_parents_graph(src, compute);

            // TODO: use_counts tracking not yet implemented
        }
    }
    

    if (node->op == OP_NONE && !(node->flag & TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        //GGML_ASSERT(cgraph->n_leafs < cgraph->size);
 
        this->leafs[this->n_leafs_] = node;
        this->n_leafs_++;
    } else {
        //GGML_ASSERT(cgraph->n_nodes < cgraph->size);
 
        this->nodes[this->n_nodes_] = node;
        this->n_nodes_++;
    }
 
    return node_hash_pos;
}

void ComputeGraph::build_forward_impl(TensorF32 * tensor, bool expand, bool compute) {
    if (!expand) {
        graph_clear();
    }
 
    const int n_old = this->n_nodes_;
 
    visit_parents_graph(tensor, compute);
 
    const int n_new = this->n_nodes_ - n_old;
    //GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);
 
    if (n_new > 0) {
        // the last added node should always be starting point
        assert(this->nodes[this->n_nodes_ - 1] == tensor);
    }
}

void ComputeGraph::build_forward_expand(TensorF32 * tensor) {
    build_forward_impl(tensor, true, true);
}

void ComputeGraph::build_backward_expand(
        struct RFAAContext *  ctx,
        TensorF32  ** grad_accs) {
    //GGML_ASSERT(cgraph->n_nodes > 0);
    //GGML_ASSERT(cgraph->grads);
    //GGML_ASSERT(cgraph->grad_accs);

    assert(this->n_nodes_ > 0);
    assert(this->grads);
    assert(this->grad_accs);
 
    const int n_nodes_f = this->n_nodes_;
 
    memset(this->grads,     0, this->visited_hash_set.size*sizeof(TensorF32 *));
    memset(this->grad_accs, 0, this->visited_hash_set.size*sizeof(TensorF32 *));
    bool * grads_needed = (bool*)calloc(this->visited_hash_set.size, sizeof(bool));
 
    {
        bool any_params = false;
        bool any_loss   = false;
        for (int i = 0; i < n_nodes_f; ++i) {
            TensorF32 * node = this->nodes[i];
            any_params = any_params || (node->flag & TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (node->flag & TENSOR_FLAG_LOSS);
        }
        //GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        //GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }
 
    for (int i = 0; i < n_nodes_f; ++i) {
        TensorF32 * node = this->nodes[i];
 
        if (node->type == TENSOR_TYPE_I32) {
            continue;
        }
 
        bool node_needs_grad = (node->flag & TENSOR_FLAG_PARAM) || (node->flag & TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        switch (node->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            case OP_IM2COL:      // only used for its shape
            case OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                break;
            case OP_UNARY: {
                const enum unary_op uop = get_unary_op(node);
                // SGN and STEP unary ops are piecewise constant
                if (uop == UNARY_OP_SGN || uop == UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } break;
            case OP_NORM: {
                // norm 节点的 src[0] 需要梯度，OP_NORM_BACK 将在 backward 时处理
            } break;
            case OP_NORM_BACK: {
                // OP_NORM_BACK 由 compute_backward 主动构造，不在此标记 ignore
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
            if (!node->src[j] || ignore_src[j] || !grads_needed[hash_find(&this->visited_hash_set, node->src[j])]) {
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
 
        const size_t ihash = hash_find(&this->visited_hash_set, node);
        //GGML_ASSERT(ihash != GGML_HASHSET_FULL);
        //GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_set.used, ihash));
        assert(ihash != HASHSET_FULL);
        assert(bitset_get(this->visited_hash_set.used, ihash));

        if (grad_accs && grad_accs[i]) {
            this->grad_accs[ihash] = grad_accs[i];
            this->grads[ihash]     = this->grad_accs[ihash];
        } else if (node->flag & TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            this->grad_accs[ihash] = ctx->new_tensor<float>(node->shape().ndim(), node->shape().dims.data());
            this->grads[ihash]     = this->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }
 
    for (int i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        compute_backward(ctx, i, grads_needed);
    }
 
    free(grads_needed);
}

TensorF32 * ComputeGraph::graph_get_grad(const TensorF32 * node) {
    const size_t igrad = hash_find(&this->visited_hash_set, (void*)node);
    return igrad != HASHSET_FULL && bitset_get(this->visited_hash_set.used, igrad) && this->grads ? this->grads[igrad] : NULL;
}

void ComputeGraph::compute_backward(
    struct RFAAContext * ctx, int i, const bool * grads_needed) {
    ComputeGraph * cgraph = this;
    TensorF32 * tensor = this->nodes[i];
    TensorF32 * grad   = graph_get_grad(tensor);
 
    if (!grad) {
        return;
    }
 
    TensorF32 * src0 = tensor->src[0];
    TensorF32 * src1 = tensor->src[1];
    TensorF32 * src2 = tensor->src[2];
    HashSet * hash_set = &this->visited_hash_set;
    const size_t isrc0 = src0 ? hash_find(hash_set, src0) : (size_t) -1;
    const size_t isrc1 = src1 ? hash_find(hash_set, src1) : (size_t) -1;
    const size_t isrc2 = src2 ? hash_find(hash_set, src2) : (size_t) -1;
    const bool src0_needs_grads = src0 && isrc0 != HASHSET_FULL && bitset_get(hash_set->used, isrc0) && grads_needed[isrc0];
    const bool src1_needs_grads = src1 && isrc1 != HASHSET_FULL && bitset_get(hash_set->used, isrc1) && grads_needed[isrc1];
    const bool src2_needs_grads = src2 && isrc2 != HASHSET_FULL && bitset_get(hash_set->used, isrc2) && grads_needed[isrc2];
 
    switch (tensor->op) {
        case OP_DUP: {
            if (src0_needs_grads) {
                add_or_set(ctx, cgraph, isrc0, grad);
            }
        } break;
        case OP_ADD: {
            if (src0_needs_grads) {
                add_or_set(ctx, cgraph, isrc0, grad);
            }
            if (src1_needs_grads) {
                TensorF32 * tmp = grad;
                if (!src0->same_shape(*src1)) {
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
                add_or_set(ctx, cgraph, isrc1, mean(grad)); // TODO: should probably be sum instead of mean
            }
        } break;
        case OP_ACC: {
            if (src0_needs_grads) {
                add_or_set(ctx, cgraph, isrc0, grad);
            }
            if (src1_needs_grads) {
                // TODO: extract nb1/nb2/nb3/offset from tensor->op_params and use view_4d + reshape + cont
                // const size_t nb1    = ((int32_t *) tensor->op_params)[0];
                // const size_t nb2    = ((int32_t *) tensor->op_params)[1];
                // const size_t nb3    = ((int32_t *) tensor->op_params)[2];
                // const size_t offset = ((int32_t *) tensor->op_params)[3];
                // struct TensorF32 * tensor_grad_view = view_4d(ctx, grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], nb1, nb2, nb3, offset);
                // add_or_set(ctx, cgraph, isrc1, reshape(cont(tensor_grad_view), src1));
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
                add_or_set(ctx, cgraph, isrc0, mul(grad, src1));
            }
            if (src1_needs_grads) {
                TensorF32 * tmp = mul(src0, grad);
                if (!tmp->same_shape(*src1)) {
                    tmp = repeat_back(tmp, src1);
                }
                add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } break;
        case OP_DIV: {
            if (src0_needs_grads) {
                add_or_set(ctx, cgraph, isrc0, div(grad, src1));
            }
            if (src1_needs_grads) {
                sub_or_set(ctx, cgraph, isrc1, mul(grad, div(tensor, src1)));
            }
        } break;
        case OP_SQR: {
            if (src0_needs_grads) {
                // d(x²)/dx = 2x * grad
                add_or_set(ctx, cgraph, isrc0, scale(mul(src0, grad), 2.0f));
            }
        } break;
        case OP_SQRT: {
            if (src0_needs_grads) {
                // d(sqrt(x))/dx = grad / (2 * sqrt(x)) = 0.5 * grad / tensor
                add_or_set(ctx, cgraph, isrc0, scale(div(grad, tensor), 0.5f));
            }
        } break;
        case OP_LOG: {
            if (src0_needs_grads) {
                // d(log(x))/dx = grad / x
                add_or_set(ctx, cgraph, isrc0, div(grad, src0));
            }
        } break;
        case OP_SIN: {
            if (src0_needs_grads) {
                // d(sin(x))/dx = cos(x) * grad
                add_or_set(ctx, cgraph, isrc0, mul(grad, cos(src0)));
            }
        } break;
        case OP_COS: {
            if (src0_needs_grads) {
                // d(cos(x))/dx = -sin(x) * grad
                sub_or_set(ctx, cgraph, isrc0, mul(grad, sin(src0)));
            }
        } break;
        case OP_SUM: {
            if (src0_needs_grads) {
                add1_or_set(ctx, cgraph, isrc0, grad);
            }
        } break;
        case OP_SUM_ROWS: {
            if (src0_needs_grads) {
                add_or_set(ctx, cgraph, isrc0, repeat(grad, src0));
            }
        } break;
        case OP_MEAN: {
            if (src0_needs_grads) {
                // d(mean(x))/dx = grad / N, broadcast to src0 shape
                float inv_N = 1.0f / src0->numel();
                add1_or_set(ctx, cgraph, isrc0, scale(grad, inv_N));
            }
        } break;
        case OP_REPEAT: {
            if (src0_needs_grads) {
                add_or_set(ctx, cgraph, isrc0, repeat_back(grad, src0));
            }
        } break;
        case OP_REPEAT_BACK: {
            if (src0_needs_grads) {
                add_or_set(ctx, cgraph, isrc0, repeat(grad, src0));
            }
        } break;
        case OP_RMS_NORM: {
            if (src0_needs_grads) {
                // TODO: needs rms_norm_back graph node and kernel
                // float eps;
                // memcpy(&eps, tensor->op_params, sizeof(float));
                // add_or_set(ctx, cgraph, isrc0, rms_norm_back(grad, src0, eps));
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
                assert(grad->shape().dims[2] == src1->shape().dims[2]);
                assert(grad->shape().dims[3] == src1->shape().dims[3]);
                TensorF32 * tmp =
                    out_prod(
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                if (!tmp->same_shape(*src0)) {
                    assert(tmp->shape().dims[0] == src0->shape().dims[0]);
                    assert(tmp->shape().dims[1] == src0->shape().dims[1]);
                    assert(tmp->shape().dims[3] == 1);
                    tmp = repeat_back(tmp, src0);
                }
                add_or_set(ctx, cgraph, isrc0, tmp);
            }
            if (src1_needs_grads) {
                // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                // avoid transpose of src0, rather transpose smaller tensor->grad
                // and then use out_prod
                add_or_set(ctx, cgraph, isrc1,
                        out_prod(
                            src0,               // [n,m,q1,r1]
                            transpose( // [p,m,qq,rr]
                                grad)));        // [m,p,qq,rr]
            }
        } break;
        case OP_SCALE: {
            if (src0_needs_grads) {
                // d(s * a)/da = s * grad
                float s;
                memcpy(&s, tensor->op_params, sizeof(float));
                add_or_set(ctx, cgraph, isrc0, scale(grad, s));
            }
        } break;
        case OP_SET: {
            // TODO: needs op_params for nb1/nb2/nb3/offset, view_4d, and acc_impl
            // const size_t nb1    = ((const int32_t *) tensor->op_params)[0];
            // const size_t nb2    = ((const int32_t *) tensor->op_params)[1];
            // const size_t nb3    = ((const int32_t *) tensor->op_params)[2];
            // const size_t offset = ((const int32_t *) tensor->op_params)[3];
            // if (src0_needs_grads || src1_needs_grads) {
            //     tensor_grad_view = view_4d(ctx, grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], nb1, nb2, nb3, offset);
            // }
            // if (src0_needs_grads) {
            //     struct TensorF32 * tmp = neg(tensor_grad_view);
            //     add_or_set(ctx, cgraph, isrc0, acc_impl(grad, tmp, nb1, nb2, nb3, offset, false));
            // }
            // if (src1_needs_grads) {
            //     add_or_set(ctx, cgraph, isrc1, reshape(cont(tensor_grad_view), src1));
            // }
        } break;
        case OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            if (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                add_or_set(ctx, cgraph, isrc0, reshape(grad, src0->shape()));
            }
            if (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } break;
        case OP_CONT: {
            // same as cpy
            if (src0_needs_grads) {
                add_or_set(ctx, cgraph, isrc0,
                    tensor->same_shape(*src0) ? grad : reshape(grad, src0->shape()));
            }
        } break;
        case OP_RESHAPE: {
            if (src0_needs_grads) {
                TensorF32 * grad_cont = grad->is_contiguous() ? grad : cont(grad);
                add_or_set(ctx, cgraph, isrc0, reshape(grad_cont, src0->shape()));
            }
        } break;
        case OP_VIEW: {
            if (src0_needs_grads) {
                // TODO: needs view_4d with offset/nb1/nb2/nb3 from op_params, then acc_or_set
                // size_t offset;
                // memcpy(&offset, tensor->op_params, sizeof(offset));
                // size_t nb1 = tensor->nb[1];
                // size_t nb2 = tensor->nb[2];
                // size_t nb3 = tensor->nb[3];
                // acc_or_set(ctx, cgraph, isrc0, grad, nb1, nb2, nb3, offset);
            }
        } break;
        case OP_PERMUTE: {
            if (src0_needs_grads) {
                // TODO: needs axes stored in op_params to compute inverse permutation
                // const int32_t * axes = (const int32_t *) tensor->op_params;
                // const int axis0 = axes[0] & 0x3;
                // const int axis1 = axes[1] & 0x3;
                // const int axis2 = axes[2] & 0x3;
                // const int axis3 = axes[3] & 0x3;
                // int axb[4] = {0,0,0,0}; // axes backward (inverse)
                // axb[axis0] = 0; axb[axis1] = 1; axb[axis2] = 2; axb[axis3] = 3;
                // add_or_set(ctx, cgraph, isrc0, permute(grad, {axb[0], axb[1], axb[2], axb[3]}));
            }
        } break;
        case OP_TRANSPOSE: {
            if (src0_needs_grads) {
                add_or_set(ctx, cgraph, isrc0, transpose(grad));
            }
        } break;
        case OP_TRI_MUL: {
            // triangle_mul(left, right, L, outgoing) → grad back to left/right
            // dst = einsum('bikd,bjkd->bijd', left, right/L) (outgoing)
            //      = einsum('bkid,bkjd->bijd', left, right/L) (incoming)
            // dL/dleft:  for outgoing: einsum('bijd,bjkd->bikd', grad, right/L)
            //            for incoming: einsum('bijd,bkjd->bkid', grad, right/L)
            // dL/dright: for outgoing: einsum('bijd,bikd->bjkd', grad, left/L)
            //            for incoming: einsum('bijd,bkid->bkjd', grad, left/L)
            // 当前简化：将 gradient 传递到 left 和 right（需要完整 kernel_tri_mul_back 实现）
            if (src0_needs_grads || src1_needs_grads) {
                // 创建 backward 图节点: OP_TRI_MUL_BACK
                // src: [grad, left, right], dst: [grad_left, grad_right]
                TensorF32* grad_left  = src0_needs_grads ? context().new_tensor<float>(src0->shape().ndim(), src0->shape().dims.data()) : nullptr;
                TensorF32* grad_right = src1_needs_grads ? context().new_tensor<float>(src1->shape().ndim(), src1->shape().dims.data()) : nullptr;

                TensorF32* back_node = context().new_tensor<float>(tensor->shape().ndim(), tensor->shape().dims.data());
                back_node->op      = OP_TRI_MUL_BACK;
                back_node->src[0]  = grad;       // upstream grad
                back_node->src[1]  = src0;       // left
                back_node->src[2]  = src1;       // right
                back_node->src[3]  = grad_left;
                back_node->src[4]  = grad_right;
                memcpy(back_node->op_params,     tensor->op_params,     8);  // L + outgoing

                if (src0_needs_grads && grad_left)
                    add_or_set(ctx, cgraph, isrc0, grad_left);
                if (src1_needs_grads && grad_right)
                    add_or_set(ctx, cgraph, isrc1, grad_right);
            }
        } break;
        case OP_GET_ROWS: {
            if (src0_needs_grads) {
                // TODO: needs get_rows_back graph node and kernel
                // add_or_set(ctx, cgraph, isrc0, get_rows_back(grad, src1, src0));
            }
            if (src1_needs_grads) {
                // noop
            }
        } break;
        case OP_DIAG_MASK_INF: {
            if (src0_needs_grads) {
                // ref: https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992
                // const int n_past = ((const int32_t *) tensor->op_params)[0];
                // add_or_set(ctx, cgraph, isrc0, diag_mask_zero_impl(grad, n_past, false));
            }
        } break;
        case OP_DIAG_MASK_ZERO: {
            if (src0_needs_grads) {
                // const int n_past = ((const int32_t *) tensor->op_params)[0];
                // add_or_set(ctx, cgraph, isrc0, diag_mask_zero_impl(grad, n_past, false));
            }
        } break;
        case OP_SOFT_MAX: {
            // forward: y = softmax(x), 沿最后一维
            // backward: dL/dx_i = y_i * (dL/dy_i - sum_j(y_j * dL/dy_j))
            if (src0_needs_grads) {
                int D    = src0->shape().dims.back();
                int rows = src0->numel() / D;

                // 构造 OP_SOFT_MAX_BACK 节点
                int64_t dx_dims[] = {rows, D};
                TensorF32 * dx = context().new_tensor<float>(2, dx_dims);
                dx->op     = OP_SOFT_MAX_BACK;
                dx->src[0] = grad;             // dL/dy (upstream gradient)
                dx->src[1] = tensor;           // y (forward output = softmax(x))
                reinterpret_cast<int&>(dx->op_params[0]) = D;
                reinterpret_cast<int&>(dx->op_params[1]) = rows;

                add_or_set(ctx, cgraph, isrc0, dx);
            }
        } break;
        case OP_ROPE: {
            if (src0_needs_grads) {
                // TODO: needs rope_back graph node and kernel
                // const int n_dims     = ((const int32_t *) tensor->op_params)[1];
                // const int mode       = ((const int32_t *) tensor->op_params)[2];
                // const int n_ctx_orig = ((const int32_t *) tensor->op_params)[4];
                // float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                // ... read from op_params ...
                // add_or_set(ctx, cgraph, isrc0, rope_back(grad, src1, src2, n_dims, mode, ...));
            }
        } break;
        case OP_NORM: {
            // forward: y = (x - mean) / std, 缓存了 mean(src[1]) 和 rstd(src[2])
            // backward:
            //   dL_dx = rstd/D * (D * dL_dy - sum(dL_dy) - y * sum(dL_dy * y))
            if (src0_needs_grads) {
                int D    = src0->shape().dims.back();
                int rows = src0->numel() / D;

                // 构造 OP_NORM_BACK 节点
                int64_t dx_dims[] = {rows, D};  // 2D shape
                TensorF32 * dx = context().new_tensor<float>(2, dx_dims);
                dx->op     = OP_NORM_BACK;
                dx->src[0] = grad;              // dL_dy
                dx->src[1] = tensor->src[0];    // x (原始输入)
                dx->src[2] = tensor->src[1];    // mean 缓存
                dx->src[3] = tensor->src[2];    // rstd 缓存
                reinterpret_cast<int&>(dx->op_params[0]) = D;
                reinterpret_cast<int&>(dx->op_params[1]) = rows;

                add_or_set(ctx, cgraph, isrc0, dx);
            }
        } break;
        case OP_IM2COL: {
            // TODO: needs im2col_back graph node and kernel
            // if (src1_needs_grads) {
            //     const int32_t s0 = ((const int32_t *) tensor->op_params)[0];
            //     const int32_t s1 = ((const int32_t *) tensor->op_params)[1];
            //     const int32_t p0 = ((const int32_t *) tensor->op_params)[2];
            //     const int32_t p1 = ((const int32_t *) tensor->op_params)[3];
            //     const int32_t d0 = ((const int32_t *) tensor->op_params)[4];
            //     const int32_t d1 = ((const int32_t *) tensor->op_params)[5];
            //     const bool is_2D = ((const int32_t *) tensor->op_params)[6] == 1;
            //     add_or_set(ctx, cgraph, isrc1, im2col_back(grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            // }
        } break;
        case OP_POOL_2D: {
            // TODO: needs pool_2d_back graph node and kernel
            // if (src0_needs_grads) {
            //     const enum ggml_op_pool op = ((const int32_t *) tensor->op_params)[0];
            //     const int32_t k0 = ((const int32_t *) tensor->op_params)[1];
            //     const int32_t k1 = ((const int32_t *) tensor->op_params)[2];
            //     const int32_t s0 = ((const int32_t *) tensor->op_params)[3];
            //     const int32_t s1 = ((const int32_t *) tensor->op_params)[4];
            //     const int32_t p0 = ((const int32_t *) tensor->op_params)[5];
            //     const int32_t p1 = ((const int32_t *) tensor->op_params)[6];
            //     add_or_set(ctx, cgraph, isrc0, pool_2d_back(grad, src0, op, k0, k1, s0, s1, p0, p1));
            // }
        } break;
        case OP_WIN_PART:
        case OP_WIN_UNPART:
        case OP_UNARY: {
            switch (get_unary_op(tensor)) {
                case UNARY_OP_ABS: {
                    if (src0_needs_grads) {
                        // d(abs(x))/dx = sign(x) * grad
                        // TODO: needs sgn() graph node; currently using step()-based approach
                        // add_or_set(ctx, cgraph, isrc0, mul(sgn(src0), grad));
                    }
                } break;
                case UNARY_OP_SGN: {
                    // noop (gradient is zero almost everywhere)
                } break;
                case UNARY_OP_NEG: {
                    if (src0_needs_grads) {
                        sub_or_set(ctx, cgraph, isrc0, grad);
                    }
                } break;
                case UNARY_OP_STEP: {
                    // noop (gradient is zero almost everywhere)
                } break;
                case UNARY_OP_RELU: {
                    if (src0_needs_grads) {
                        // d(relu(x))/dx = step(x) * grad
                        // TODO: needs step() graph node
                        // add_or_set(ctx, cgraph, isrc0, mul(step(src0), grad));
                    }
                } break;
                case UNARY_OP_SILU: {
                    if (src0_needs_grads) {
                        // TODO: needs silu_back graph node and kernel
                        // add_or_set(ctx, cgraph, isrc0, silu_back(grad, src0));
                    }
                } break;
                case UNARY_OP_GELU: {
                    // TODO: needs gelu_back graph node and kernel
                } break;
                case UNARY_OP_GELU_QUICK: {
                    // TODO: needs gelu_quick_back graph node and kernel
                } break;
                case UNARY_OP_TANH: {
                    // TODO: needs tanh_back graph node and kernel
                } break;
                case UNARY_OP_SIGMOID: {
                    // TODO: needs sigmoid_back graph node and kernel
                    // d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x)) * grad = tensor * (1 - tensor) * grad
                } break;
                default: {
                    // unsupported unary op for backward pass
                } break;
            }
        } break;
        case OP_CROSS_ENTROPY_LOSS: {
            if (src0_needs_grads) {
                // TODO: needs cross_entropy_loss_back graph node and kernel
                // add_or_set(ctx, cgraph, isrc0, cross_entropy_loss_back(grad, src0, src1));
            }
            // labels (src1) gradient not implemented
        } break;
        case OP_FAPE: {
            // OP_FAPE backward: construct OP_FAPE_BACK node that computes
            // gradient w.r.t. pred_coords (src[0])
            // Only src[0] (pred_coords) gets gradient;
            // src[1] (true_coords), src[2] (frame_atom_indices),
            // src[3] (frames_mask), src[4] (positions_mask) are fixed.
            if (src0_needs_grads) {
                // Create OP_FAPE_BACK node
                // Output shape = pred_coords shape [N_atoms, 3]
                int64_t ne[4] = {1, 1, 1, 1};
                int ndim = src0->shape().ndim();
                for (int d = 0; d < ndim; d++) ne[d] = src0->shape().dims[d];
                TensorF32* fape_grad = ctx->new_tensor<float>(ndim, ne);

                fape_grad->op     = OP_FAPE_BACK;
                fape_grad->src[0] = grad;          // upstream gradient (scalar)
                fape_grad->src[1] = src0;          // pred_coords (forward)
                fape_grad->src[2] = src1;          // true_coords (forward)
                fape_grad->src[3] = src2;          // frame_atom_indices
                fape_grad->src[4] = src2;          // frames_mask (reuse src2 for now)
                fape_grad->src[5] = src2;          // positions_mask (reuse src2 for now)

                // Copy op_params (d_clamp, epsilon, length_scale)
                memcpy(fape_grad->op_params, tensor->op_params, sizeof(tensor->op_params));

                add_or_set(ctx, cgraph, isrc0, fape_grad);
            }
        } break;
        case OP_GLU: {
            // TODO: needs glu op and glu_back graph nodes and kernels
            // switch (ggml_get_glu_op(tensor)) {
            //     case GGML_GLU_OP_SWIGLU:
            //         if (src0_needs_grads) add_or_set(ctx, cgraph, isrc0, silu_back(mul(grad, src1), src0));
            //         if (src1_needs_grads) add_or_set(ctx, cgraph, isrc1, mul(silu(src0), grad));
            //         break;
            // }
        } break;
        case OP_NONE: {

        } break;
        case OP_COUNT:
        default: {

        }
    }
}

void ComputeGraph::add_or_set(
        struct RFAAContext * ctx,
        struct ComputeGraph  * cgraph,
        size_t                isrc,
        TensorF32  * tensor) {
    TensorF32 * src = static_cast<TensorF32 *>(cgraph->visited_hash_set.keys[isrc]);
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
 
void ComputeGraph::acc_or_set(
        struct RFAAContext * ctx,
        struct ComputeGraph  * cgraph,
        size_t                isrc,
        TensorF32  * tensor,
        const  size_t         nb1,
        const  size_t         nb2,
        const  size_t         nb3,
        const  size_t         offset) {
    TensorF32 * src = static_cast<TensorF32 *>(cgraph->visited_hash_set.keys[isrc]);
    assert(src);
    if (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = acc(cgraph->grads[isrc], tensor, nb1, nb2, nb3, offset);
    } else {
        // FIXME this is going to produce NaN if src contains inf/NaN
        TensorF32 * a_zero = scale(src, 0.0f);
        cgraph->grads[isrc] = acc(a_zero, tensor, nb1, nb2, nb3, offset);
    }
    cgraph->build_forward_expand(cgraph->grads[isrc]);
}
 
void ComputeGraph::add1_or_set(
        struct RFAAContext * ctx,
        struct ComputeGraph  * cgraph,
        size_t                isrc,
        TensorF32  * tensor) {
    TensorF32 * src = static_cast<TensorF32 *>(cgraph->visited_hash_set.keys[isrc]);
    assert(src);
    if (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = add1_impl(cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } else {
        cgraph->grads[isrc] = repeat(tensor, src);
    }
    cgraph->build_forward_expand(cgraph->grads[isrc]);
}
 
void ComputeGraph::sub_or_set(
        struct RFAAContext * ctx,
        struct ComputeGraph  * cgraph,
        size_t                isrc,
        TensorF32  * tensor) {
    TensorF32 * src = static_cast<TensorF32 *>(cgraph->visited_hash_set.keys[isrc]);
    assert(src);
    if (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = sub(cgraph->grads[isrc], tensor);
    } else {
        cgraph->grads[isrc] = neg(tensor);
    }
    cgraph->build_forward_expand(cgraph->grads[isrc]);
}

void ComputeGraph::graph_cpy(struct ComputeGraph * src, struct ComputeGraph * dst) {
    //GGML_ASSERT(dst->size >= src->n_leafs);
    //GGML_ASSERT(dst->size >= src->n_nodes);
    //GGML_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

    assert(dst->size() >= src->n_leafs());
    assert(dst->size() >= src->n_nodes());
    assert(dst->visited_hash_set.size >= src->visited_hash_set.size);
 
    dst->n_leafs_ = src->n_leafs();
    dst->n_nodes_ = src->n_nodes();
    dst->order   = src->order;
 
    for (int i = 0; i < src->n_leafs(); ++i) {
        dst->leafs[i] = src->leafs[i];
    }
 
    for (int i = 0; i < src->n_nodes(); ++i) {
        dst->nodes[i] = src->nodes[i];
    }
 
    for (size_t i = 0; i < src->visited_hash_set.size; ++i) {
        // copy all hashset keys (tensors) that are in use
        if (bitset_get(src->visited_hash_set.used, i)) {
            size_t new_hash_pos = hash_insert(&dst->visited_hash_set, src->visited_hash_set.keys[i]);
            // dst->use_counts not available; skip
            //(void)new_hash_pos;
        }
    }
 
    if (dst->grads) {
        memset(dst->grads,     0, dst->visited_hash_set.size*sizeof(TensorF32 *));
        memset(dst->grad_accs, 0, dst->visited_hash_set.size*sizeof(TensorF32 *));
    }
    if (src->grads) {
        //GGML_ASSERT(dst->grads     != NULL);
        //GGML_ASSERT(dst->grad_accs != NULL);
        assert(dst->grads     != NULL);
        assert(dst->grad_accs != NULL);

        for (int i = 0; i < src->n_nodes(); ++i) {
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

 
void * ComputeGraph::incr_ptr_aligned(void ** p, size_t size, size_t align) {
    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}
 
size_t ComputeGraph::graph_nbytes(size_t size, bool grads) {
    size_t hash_size_val = bitset_size(size * 2);
    void * p = 0;
    ComputeGraph::incr_ptr_aligned(&p, sizeof(ComputeGraph), 1);
    ComputeGraph::incr_ptr_aligned(&p, size * sizeof(TensorF32 *), sizeof(TensorF32 *)); // nodes
    ComputeGraph::incr_ptr_aligned(&p, size * sizeof(TensorF32 *), sizeof(TensorF32 *)); // leafs
    ComputeGraph::incr_ptr_aligned(&p, hash_size_val * sizeof(TensorF32 *), sizeof(TensorF32 *)); // hash keys
    if (grads) {
        ComputeGraph::incr_ptr_aligned(&p, hash_size_val * sizeof(TensorF32 *), sizeof(TensorF32 *)); // grads
        ComputeGraph::incr_ptr_aligned(&p, hash_size_val * sizeof(TensorF32 *), sizeof(TensorF32 *)); // grad_accs
    }
    ComputeGraph::incr_ptr_aligned(&p, bitset_size(hash_size_val) * sizeof(bitset_t), sizeof(bitset_t));
 
    size_t nbytes = (size_t) p;
    return nbytes;
}




} // namespace rfaa
#include "rfaa/Tensor.h"

namespace rfaa {

    static Tensor * add_impl( 
        Tensor  * a,  
        Tensor  * b,  
        bool                  inplace) {  
    //GGML_ASSERT(ggml_can_repeat(b, a)); 
    assert(b->can_repeat(a));
  
    //struct Tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);  
    Tensor * result; 
    result = inplace ? result.view(a->shape()) : result->copy_from(a);
  
    result->op     = OP_ADD;  
    result->src[0] = a;  
    result->src[1] = b;  
  
    return result;  
}

Tensor * repeat_back(
        Tensor  * a,  
        Tensor  * b) {  
    //GGML_ASSERT(ggml_can_repeat(b, a));  
    assert(b->can_repeat(a));
  
    Tensor* result = new_tensor(ctx, a->type, GGML_MAX_DIMS, b->shape().dims.data());  

    assert(b->can_repeat(a));
  
    result->op     = OP_REPEAT_BACK;  
    result->src[0] = a;  
  
    return result;  
}

// add a scalar
Tensor * add1_impl(
        Tensor  * a,
        Tensor  * b,
        bool                  inplace) {
    //GGML_ASSERT(ggml_is_scalar(b));
    //GGML_ASSERT(ggml_is_padded_1d(a));

    assert(b->is_scalar());
    assert(a->is_contiguous());
 
    //struct Tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    struct Tensor * result; 
    result = inplace ? result->view(a->shape()) : result->copy_from(a);
 
    result->op     = OP_ADD1;
    result->src[0] = a;
    result->src[1] = b;
 
    return result;
}



} // namespace rfaa

#include "rfaa/Context.h"
#include "rfaa/Tensor.h"
#include "rfaa/Core.h"
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>

namespace rfaa {

// ========== 静态全局容器 ==========
static struct {
    bool used;
    struct RFAAContext context;
} g_contexts[RFAA_MAX_CONTEXTS];

// ========== init() ==========
RFAAContext* RFAAContext::init(const CtxInitParams& params) {
    // 找个空槽位
    RFAAContext* ctx = nullptr;
    for (int i = 0; i < RFAA_MAX_CONTEXTS; i++) {
        if (!g_contexts[i].used) {
            g_contexts[i].used = true;
            ctx = &g_contexts[i].context;
            break;
        }
    }
    if (!ctx) return nullptr;

    size_t mem_size = params.mem_size;
    if (mem_size == 0) mem_size = RFAA_MEM_ALIGN;
    // 对齐
    mem_size = (mem_size + RFAA_MEM_ALIGN - 1) / RFAA_MEM_ALIGN * RFAA_MEM_ALIGN;

    *ctx = {};
    ctx->mem_size = mem_size;
    ctx->mem_buffer_owned = (params.mem_buffer == nullptr);
    ctx->mem_buffer = params.mem_buffer ? params.mem_buffer : malloc(mem_size);
    ctx->no_alloc = params.no_alloc;
    // objects_begin/end 初始为 NULL

    return ctx;
}

// ========== free() ==========
void RFAAContext::free(RFAAContext* ctx) {
    if (!ctx) return;
    if (ctx->mem_buffer_owned && ctx->mem_buffer) {
        ::free(ctx->mem_buffer);
    }
    // 重置槽位
    for (int i = 0; i < RFAA_MAX_CONTEXTS; i++) {
        if (&g_contexts[i].context == ctx) {
            g_contexts[i].used = false;
            break;
        }
    }
}

// ========== new_object() ==========
RFAAObject* RFAAContext::new_object(enum RFAAObjectType type, size_t size) {
    RFAAObject* obj_cur = objects_end;

    size_t cur_end = 0;
    if (obj_cur) {
        cur_end = obj_cur->offs + obj_cur->size;
    }

    size_t size_aligned = (size + RFAA_MEM_ALIGN - 1) / RFAA_MEM_ALIGN * RFAA_MEM_ALIGN;

    if (cur_end + size_aligned + RFAA_OBJECT_SIZE > mem_size) {
        // 内存不足
        assert(false && "Context memory exhausted");
        return nullptr;
    }

    // obj 放在 mem_buffer + cur_end 处
    RFAAObject* obj_new = (RFAAObject*)((char*)mem_buffer + cur_end);
    *obj_new = {};
    obj_new->offs = cur_end + RFAA_OBJECT_SIZE;
    obj_new->size = size_aligned;
    obj_new->type = type;
    obj_new->next = nullptr;

    if (obj_cur) {
        obj_cur->next = obj_new;
    } else {
        objects_begin = obj_new;
    }
    objects_end = obj_new;
    n_objects++;

    return obj_new;
}

// ========== new_tensor() ==========
template<typename T>
Tensor<T>* RFAAContext::new_tensor(int n_dims, const int64_t* ne) {
    // 1. 计算数据大小
    size_t data_size = sizeof(T);
    for (int i = 0; i < n_dims; i++) data_size *= ne[i];

    // 2. 创建 object
    size_t obj_alloc_size = data_size;  // tensor 结构体不占额外空间
    RFAAObject* obj = new_object(RFAA_OBJECT_TYPE_TENSOR,
                                  RFAA_TENSOR_SIZE + obj_alloc_size);
    if (!obj) return nullptr;

    // 3. tensor 结构体位于 obj->offs 处
    Tensor<T>* result = (Tensor<T>*)((char*)mem_buffer + obj->offs);

    // 4. 数据紧随结构体 (result + 1)
    void* data_ptr = (obj_alloc_size > 0) ? (void*)(result + 1) : nullptr;

    // 5. placement new 初始化 Tensor
    new (result) Tensor<T>();
    result->init_from_context(n_dims, ne, data_ptr);

    return result;
}

// 显式实例化
template Tensor<float>*  RFAAContext::new_tensor<float>(int, const int64_t*);
template Tensor<int64_t>* RFAAContext::new_tensor<int64_t>(int, const int64_t*);

} // namespace rfaa
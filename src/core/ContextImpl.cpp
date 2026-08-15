
#include "ppml/Context.h"
#include "ppml/Tensor.h"
#include "ppml/Core.h"
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <mutex>

#if defined(_WIN32)  
#define WIN32_LEAN_AND_MEAN  
#ifndef NOMINMAX  
    #define NOMINMAX  
#endif  
#include <windows.h>  
#endif

namespace ppml {

// ========== 静态全局容器 ==========
static struct {
    bool used;
    struct PPMLContext context;
} g_contexts[PPML_MAX_CONTEXTS];

 
std::mutex ppml_critical_section_mutex;
 
void ppml_critical_section_start() {
    ppml_critical_section_mutex.lock();
}
 
void ppml_critical_section_end(void) {
    ppml_critical_section_mutex.unlock();
}

#if defined(_MSC_VER) || defined(__MINGW32__)
static int64_t timer_freq, timer_start;
void ppml_time_init(void) {
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;
 
    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    QueryPerformanceCounter(&t);
    timer_start = t.QuadPart;
}
int64_t ppml_time_ms(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t ppml_time_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#else
void ppml_time_init(void) {}
int64_t ppml_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}
 
int64_t ppml_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

// ========== init() ==========
PPMLContext* PPMLContext::init(const CtxInitParams& params) {

    static bool is_first_call = true;
 
    // thread safe init for time system
    ppml_critical_section_start();
 
    if (is_first_call) {
        // initialize time system (required on Windows)
        ppml_time_init();
 
        is_first_call = false;
    }
 
    ppml_critical_section_end();

    // 找个空槽位
    PPMLContext* ctx = nullptr;
    for (int i = 0; i < PPML_MAX_CONTEXTS; i++) {
        if (!g_contexts[i].used) {
            g_contexts[i].used = true;
            ctx = &g_contexts[i].context;
            break;
        }
    }
    if (!ctx) return nullptr;

    size_t mem_size = params.mem_size;
    if (mem_size == 0) mem_size = PPML_MEM_ALIGN;
    // 对齐
    // PPML_MEM_ALIGN == 16 standard alignment for a 64 bit system
    //mem_size = (mem_size + PPML_MEM_ALIGN - 1) / PPML_MEM_ALIGN * PPML_MEM_ALIGN;
    mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(mem_size, PPML_MEM_ALIGN);

    *ctx = {};
    ctx->mem_size = mem_size;
    ctx->mem_buffer_owned = (params.mem_buffer == nullptr);
    ctx->mem_buffer = params.mem_buffer ? params.mem_buffer : malloc(mem_size);
    ctx->no_alloc = params.no_alloc;
    // objects_begin/end 初始为 NULL

    return ctx;
}

// ========== free() ==========
void PPMLContext::free(PPMLContext* ctx) {
    if (!ctx) return;
    if (ctx->mem_buffer_owned && ctx->mem_buffer) {
        ::free(ctx->mem_buffer);
    }
    // 重置槽位
    for (int i = 0; i < PPML_MAX_CONTEXTS; i++) {
        if (&g_contexts[i].context == ctx) {
            g_contexts[i].used = false;
            break;
        }
    }
}

// ========== new_object() ==========
PPMLObject* PPMLContext::new_object(enum PPMLObjectType type, size_t size) {
    PPMLObject* obj_cur = objects_end;

    size_t cur_end = 0;
    if (obj_cur) {
        cur_end = obj_cur->offs + obj_cur->size;
    }

    size_t size_aligned = (size + PPML_MEM_ALIGN - 1) / PPML_MEM_ALIGN * PPML_MEM_ALIGN;

    if (cur_end + size_aligned + PPML_OBJECT_SIZE > mem_size) {
        // 内存不足
        assert(false && "Context memory exhausted");
        return nullptr;
    }

    // obj 放在 mem_buffer + cur_end 处
    PPMLObject* obj_new = (PPMLObject*)((char*)mem_buffer + cur_end);
    *obj_new = {};
    obj_new->offs = cur_end + PPML_OBJECT_SIZE;
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
Tensor<T>* PPMLContext::new_tensor(int n_dims, const int64_t* ne) {
    // 1. 计算数据大小
    size_t data_size = sizeof(T);
    for (int i = 0; i < n_dims; i++) data_size *= ne[i];

    // 2. 创建 object
    size_t obj_alloc_size = data_size;  // tensor 结构体不占额外空间
    PPMLObject* obj = new_object(PPML_OBJECT_TYPE_TENSOR,
                                  PPML_TENSOR_SIZE + obj_alloc_size);
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
template Tensor<float>*  PPMLContext::new_tensor<float>(int, const int64_t*);
template Tensor<int64_t>* PPMLContext::new_tensor<int64_t>(int, const int64_t*);

} // namespace ppml
#pragma once
#include <cstddef>

#include "Tensor.h"

#define GGML_PAD(x, n) (((x) + ((n)-1)) & ~((n)-1))

namespace ppml {

enum PPMLObjectType {
    PPML_OBJECT_TYPE_TENSOR,
    PPML_OBJECT_TYPE_GRAPH,
    PPML_OBJECT_TYPE_WORK_BUFFER
};

struct PPMLObject {
    size_t offs;
    size_t size;

    struct PPMLObject * next;

    enum PPMLObjectType type;

    char padding[4];
};

static constexpr size_t PPML_TENSOR_SIZE = sizeof(struct Tensor<float>);
static constexpr size_t PPML_OBJECT_SIZE = sizeof(struct PPMLObject);
static constexpr size_t PPML_MEM_ALIGN   = 16;
static constexpr int    PPML_MAX_CONTEXTS = 8;

// about the MEM_ALIGN:
// 32 bit:
/* #if UINTPTR_MAX == 0xFFFFFFFF
    #define GGML_MEM_ALIGN 4
#elif defined(__EMSCRIPTEN__)
// emscripten uses max_align_t == 8, so we need GGML_MEM_ALIGN == 8 for 64-bit wasm.
// (for 32-bit wasm, the first conditional is true and GGML_MEM_ALIGN stays 4.)
// ref: https://github.com/ggml-org/llama.cpp/pull/18628
    #define GGML_MEM_ALIGN 8
#else
64 bit:
    #define GGML_MEM_ALIGN 16
#endif */



struct CtxInitParams {
    size_t mem_size;
    void * mem_buffer;
    bool   no_alloc;
};


struct PPMLContext {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    int    n_objects;

    struct PPMLObject * objects_begin;
    struct PPMLObject * objects_end;

    // ===== 新增 API =====
    // 初始化/释放
    static PPMLContext* init(const CtxInitParams& params);
    static void         free(PPMLContext* ctx);

    // 创建张量（核心 API）
    template<typename T>
    Tensor<T>* new_tensor(int n_dims, const int64_t* ne);

    // 计算所需总内存（预分配阶段使用）
    static size_t calc_mem_size(int n_tensors, int total_elements);

    // 获取全局单例（默认 context）
    static PPMLContext& global();
    static void         init_global(const CtxInitParams& params);
    static void         free_global();

    friend class ComputeGraph;

private:
    PPMLObject* new_object(enum PPMLObjectType type, size_t size);
};


// ========== 全局单例实现（内联在头文件中）==========
inline PPMLContext& context() {
    static PPMLContext* g_ctx = nullptr;
    if (!g_ctx) {
        // 如果未初始化，使用默认参数
        CtxInitParams default_params = { 1024 * 1024 * 1024, nullptr, false };  // 1GB
        g_ctx = PPMLContext::init(default_params);
    }
    return *g_ctx;
}

} // namespace ppml
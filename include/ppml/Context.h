#pragma once
#include <cstddef>
#include <vector>

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
    // no_alloc=false：数据直接分配在 context 缓冲区（data_ 指向 (result+1)）。
    // no_alloc=true ：只分配 tensor 结构体，数据 data_=nullptr（空壳），
    //                 由后续 Gallocr/backend buffer 按需分配（节省 context 内存）。
    template<typename T>
    Tensor<T>* new_tensor(int n_dims, const int64_t* ne);

    // 创建参数张量（weight/bias/gamma/beta 等）：no_alloc 时也从宿主暂存区分配
    // 真实数据（data_ 非空），以便构建期 init_weights/transfer_params 直接读写。
    template<typename T>
    Tensor<T>* new_param_tensor(int n_dims, const int64_t* ne);

    // ===== no_alloc 宿主暂存区 =====
    // no_alloc 时，构建期需直接写入数据的 leaf/常量节点从该暂存区获取宿主内存，
    // 计算中间节点不占用（data_=nullptr）。暂存块生命周期由 context 管理。
    // 后续 Gallocr 分配 backend buffer 后可覆盖 data_/buffer_，暂存块在 scratch_free 统一释放。
    void* scratch_alloc(size_t bytes);
    void  scratch_free();            // 释放全部暂存块（PPMLContext::free 调用）
    std::vector<void*> scratch_;     // 已分配的暂存块（owner）

    // 计算所需总内存（预分配阶段使用）
    static size_t calc_mem_size(int n_tensors, int total_elements);

    // ===== 训练期 arena 复用 =====
    // 模型参数在 PPML::create 时率先分配（arena 低地址），每 epoch 的图节点随后分配（高地址）。
    // 记录构建期末水位 mark_objects()，每 epoch 开始 reset_objects_to(mark) 即可释放上一轮图节点、
    // 保留参数，避免 arena 跨 epoch 单调增长耗尽（Context memory exhausted 崩溃）。
    void*  mark_objects();                 // 返回当前 objects_end 指针
    void   reset_objects_to(void* mark);   // 将 objects_end 回退到 mark（释放 mark 之上的对象）

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
        // 全局 context 启用 no_alloc：中间张量只建空壳，由 Gallocr 延迟分配 + 空间复用，
        // 避免 eager 全量分配进固定 context 缓冲区（~40GB 溢出）。
        CtxInitParams default_params = { 1024 * 1024 * 1024, nullptr, true };
        g_ctx = PPMLContext::init(default_params);
    }
    return *g_ctx;
}

// ===== no_alloc 构建期 leaf 数据绑定 =====
// 返回 tensor 可写的宿主数据指针：
//   - no_alloc=false：直接返回 t->data()（new_tensor 已分配在 context 缓冲区）
//   - no_alloc=true ：若 data() 为空，从 context 暂存区分配并绑定，再返回
// 构建期需直接写入数据的 leaf/常量节点（constant_*、wrap_input_as_leaf、掩码等）
// 必须通过此函数获取写指针，否则 no_alloc 时空指针写崩溃。
template<typename T>
inline T* bind_leaf_data(PPMLContext& ctx, Tensor<T>* t) {
    if (t->data()) return t->data();
    if (t->nbytes() == 0) return nullptr;
    void* p = ctx.scratch_alloc(t->nbytes());
    t->bind_data(p);
    return t->data();
}

} // namespace ppml
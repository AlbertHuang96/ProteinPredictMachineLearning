#pragma once
#include <cstddef>
#include <list>
#include <cstdint>

namespace ppml {

// ============================================================
// DynTalloc — 对标 ggml_dyn_tallocr
//
// 在单个 buffer 区域内按"自由块链表 + best-fit + 相邻合并"进行动态分配/释放。
// 用于 no_alloc 图的延迟分配：tensor 沿拓扑序执行时"借/还"同一块 buffer 空间，
// 使同时存活的数据仅占峰值而非全量（空间复用），从而大幅节省内存。
//
// 本类只追踪 offset/size（相对 buffer 基址），不拥有实际内存；
// 底层内存由外部 Buffer（通过 BufferType::alloc）持有。
// ============================================================
class DynTalloc {
public:
    // 自由块
    struct FreeBlock {
        size_t offset;
        size_t size;
        FreeBlock(size_t o, size_t s) : offset(o), size(s) {}
    };

    DynTalloc() = default;
    DynTalloc(size_t size, size_t alignment);

    // 配置（必须在 reset 之前设置）
    void set_size(size_t size)            { size_ = size; }
    void set_alignment(size_t alignment)  { alignment_ = alignment; }

    size_t size()      const { return size_; }
    size_t alignment() const { return alignment_; }
    bool   is_empty()  const { return size_ == 0; }

    // 分配 size 字节，返回偏移到 offset。best-fit（对齐后剩余最小）。
    // 成功返回 true；空间不足返回 false。
    bool   alloc(size_t size, size_t& offset);

    // 释放 [offset, offset+size) 回到自由列表，并与相邻自由块合并。
    // 返回释放的字节数（对齐后）。
    size_t free_bytes(size_t offset, size_t size);

    // 当前已用字节数（= size_ - 各自由块大小之和）
    size_t used_bytes() const;

    // 清空并重置为单一整块自由区域 {0, size_}
    void   reset();

    // 自由块列表（测试/调试用）
    const std::list<FreeBlock>& free_blocks() const { return free_blocks_; }

private:
    size_t             size_      = 0;
    size_t             alignment_ = 0;
    std::list<FreeBlock> free_blocks_;
};

} // namespace ppml

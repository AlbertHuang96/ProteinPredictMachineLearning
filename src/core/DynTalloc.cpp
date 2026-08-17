#include "ppml/DynTalloc.h"

#include <algorithm>
#include <limits>

namespace ppml {

#define GGML_PAD(x, n) (((x) + ((n)-1)) & ~((n)-1))

DynTalloc::DynTalloc(size_t size, size_t alignment)
    : size_(size), alignment_(alignment) {
    reset();
}

bool DynTalloc::alloc(size_t size, size_t& offset) {
    if (size == 0) {
        offset = 0;
        return true;
    }

    // 分配大小按 alignment 对齐（与 free_bytes 对齐一致，保证后续可合并）
    const size_t alloc_size = GGML_PAD(size, alignment_);

    size_t       best_off   = 0;
    size_t       best_rem   = std::numeric_limits<size_t>::max();
    auto         best_it    = free_blocks_.end();

    // best-fit：遍历自由块，取"对齐后剩余最小"的一块
    for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
        const size_t block_begin = it->offset;
        const size_t block_end   = it->offset + it->size;

        const size_t off     = GGML_PAD(block_begin, alignment_);
        if (off > block_end) continue;                       // 对齐后越界
        const size_t remaining = block_end - off;
        if (remaining < alloc_size) continue;                // 放不下

        if (remaining < best_rem) {
            best_rem = remaining;
            best_off = off;
            best_it  = it;
        }
    }

    if (best_it == free_blocks_.end()) {
        return false;  // 无足够连续空间
    }

    // 从 best 块切出 [best_off, best_off + alloc_size)
    const size_t block_end = best_it->offset + best_it->size;
    best_it->offset = best_off + alloc_size;
    best_it->size   = block_end - best_it->offset;

    if (best_it->size == 0) {
        free_blocks_.erase(best_it);   // 块用尽，移除
    }

    offset = best_off;
    return true;
}

size_t DynTalloc::free_bytes(size_t offset, size_t size) {
    if (size == 0) return 0;

    const size_t block_begin = offset;
    const size_t block_end   = offset + GGML_PAD(size, alignment_);

    // 找到第一个 offset >= block_begin 的自由块（保持 offset 升序）
    auto it = free_blocks_.begin();
    while (it != free_blocks_.end() && it->offset < block_begin) ++it;

    // 情况 1：与前一自由块相邻 → 向前合并
    if (it != free_blocks_.begin()) {
        auto prev = std::prev(it);
        if (prev->offset + prev->size == block_begin) {
            prev->size += block_end - block_begin;
            // 若合并后与 it 也相邻 → 一并合并
            if (it != free_blocks_.end() && prev->offset + prev->size == it->offset) {
                prev->size += it->size;
                free_blocks_.erase(it);
            }
            return prev->size;
        }
    }

    // 情况 2：与后一自由块相邻 → 向后合并
    if (it != free_blocks_.end() && block_end == it->offset) {
        it->offset = block_begin;
        it->size  += block_end - block_begin;
        return it->size;
    }

    // 情况 3：无相邻自由块 → 插入新块
    free_blocks_.insert(it, FreeBlock(block_begin, block_end - block_begin));
    return block_end - block_begin;
}

size_t DynTalloc::used_bytes() const {
    size_t free_total = 0;
    for (const auto& b : free_blocks_) {
        free_total += b.size;
    }
    return size_ >= free_total ? size_ - free_total : 0;
}

void DynTalloc::reset() {
    free_blocks_.clear();
    if (size_ > 0) {
        free_blocks_.emplace_back(0, size_);
    }
}

} // namespace ppml

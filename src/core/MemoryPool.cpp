#include "ppml/Core.h"
#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <list>

namespace ppml {

// 内存池实现：减少 CUDA malloc/free 开销

class MemoryPool {
public:
    static MemoryPool& instance();
    
    // 分配内存
    void* allocate(size_t size, Device device);
    
    // 释放内存（回收到池）
    void deallocate(void* ptr, size_t size, Device device);
    
    // 清理未使用的内存
    void trim();
    
    // 获取统计信息
    struct Stats {
        size_t total_allocated;
        size_t total_used;
        size_t pool_hits;
        size_t pool_misses;
    };
    Stats get_stats() const;
    
private:
    MemoryPool() = default;
    ~MemoryPool();
    
    struct Block {
        void* ptr;
        size_t size;
        std::chrono::steady_clock::time_point last_used;
    };
    
    std::mutex mutex_;
    std::unordered_map<size_t, std::list<Block>> cpu_pool_;
    std::unordered_map<size_t, std::list<Block>> cuda_pool_;
    
    Stats stats_;
};

MemoryPool& MemoryPool::instance() {
    static MemoryPool pool;
    return pool;
}

MemoryPool::~MemoryPool() {
    // 释放所有池化内存
    for (auto& [size, blocks] : cpu_pool_) {
        for (auto& block : blocks) {
            free(block.ptr);
        }
    }
    for (auto& [size, blocks] : cuda_pool_) {
        for (auto& block : blocks) {
            cudaFree(block.ptr);
        }
    }
}

void* MemoryPool::allocate(size_t size, Device device) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& pool = (device == Device::CPU) ? cpu_pool_ : cuda_pool_;
    
    // 查找合适大小的块
    auto it = pool.find(size);
    if (it != pool.end() && !it->second.empty()) {
        void* ptr = it->second.front().ptr;
        it->second.pop_front();
        stats_.pool_hits++;
        return ptr;
    }
    
    // 分配新内存
    void* ptr = nullptr;
    if (device == Device::CPU) {
        ptr = malloc(size);
    } else {
        cudaMalloc(&ptr, size);
    }
    
    stats_.pool_misses++;
    stats_.total_allocated += size;
    return ptr;
}

void MemoryPool::deallocate(void* ptr, size_t size, Device device) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& pool = (device == Device::CPU) ? cpu_pool_ : cuda_pool_;
    
    Block block{ptr, size, std::chrono::steady_clock::now()};
    pool[size].push_back(block);
}

void MemoryPool::trim() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto now = std::chrono::steady_clock::now();
    auto max_idle = std::chrono::minutes(5);
    
    for (auto& [size, blocks] : cpu_pool_) {
        blocks.remove_if([&](const Block& b) {
            if (now - b.last_used > max_idle) {
                free(b.ptr);
                return true;
            }
            return false;
        });
    }
    
    for (auto& [size, blocks] : cuda_pool_) {
        blocks.remove_if([&](const Block& b) {
            if (now - b.last_used > max_idle) {
                cudaFree(b.ptr);
                return true;
            }
            return false;
        });
    }
}

MemoryPool::Stats MemoryPool::get_stats() const {
    return stats_;
}

} // namespace ppml

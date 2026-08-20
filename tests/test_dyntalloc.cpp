#include <gtest/gtest.h>
#include "ppml/DynTalloc.h"

using namespace ppml;

// 基础：整块分配 / 释放后可完整复用
TEST(DynTallocTest, BasicAllocFreeReuse) {
    DynTalloc ta(1024, 32);
    EXPECT_EQ(ta.size(), 1024);
    EXPECT_EQ(ta.used_bytes(), 0);

    size_t o1 = 0, o2 = 0, o3 = 0;
    EXPECT_TRUE(ta.alloc(256, o1));
    EXPECT_TRUE(ta.alloc(256, o2));
    EXPECT_TRUE(ta.alloc(256, o3));
    EXPECT_EQ(o1, 0u);
    EXPECT_EQ(o2, 256u);
    EXPECT_EQ(o3, 512u);
    EXPECT_EQ(ta.used_bytes(), 768u);

    // 释放中间一块，应能再次分配
    ta.free_bytes(o2, 256);
    size_t o4 = 0;
    EXPECT_TRUE(ta.alloc(256, o4));
    EXPECT_EQ(o4, 256u);   // 复用释放的块

    // 释放全部，应能重新分配整个区域
    ta.free_bytes(o1, 256);
    ta.free_bytes(o3, 256);
    ta.free_bytes(o4, 256);
    EXPECT_EQ(ta.used_bytes(), 0u);

    size_t o5 = 0;
    EXPECT_TRUE(ta.alloc(1000, o5));
    EXPECT_EQ(o5, 0u);
}

// 相邻释放应合并，避免碎片
TEST(DynTallocTest, AdjacentFreeMerges) {
    DynTalloc ta(1024, 32);
    size_t a = 0, b = 0, c = 0;
    ta.alloc(100, a);  // a = 0     （对齐 32 → 每块 128）
    ta.alloc(100, b);  // b = 128
    ta.alloc(100, c);  // c = 256

    // 初始空闲：[384, 1024)
    ta.free_bytes(b, 100);   // 释放中间 [128,256) → {128,128},{384,640}
    EXPECT_EQ(ta.free_blocks().size(), 2u);

    ta.free_bytes(a, 100);   // 释放左侧 [0,128)，与 [128,256) 合并 → {0,256},{384,640}
    EXPECT_EQ(ta.free_blocks().size(), 2u);
    EXPECT_EQ(ta.free_blocks().front().offset, 0u);
    EXPECT_EQ(ta.free_blocks().front().size, 256u);

    ta.free_bytes(c, 100);   // 释放 [256,384)，先与 {0,256} 前合并，再与 {384,640} 后合并
    EXPECT_EQ(ta.free_blocks().size(), 1u);
    EXPECT_EQ(ta.free_blocks().front().offset, 0u);
    EXPECT_EQ(ta.free_blocks().front().size, 1024u);  // 全部空闲合并回整块 [0,1024)
}

// 空间不足应失败
TEST(DynTallocTest, OutOfMemory) {
    DynTalloc ta(256, 32);
    size_t a = 0, b = 0;
    EXPECT_TRUE(ta.alloc(200, a));    // 200 对齐 32 = 224
    EXPECT_EQ(a, 0u);
    EXPECT_FALSE(ta.alloc(200, b));   // 剩余 32 字节，放不下 224
    EXPECT_TRUE(ta.alloc(16, b));     // 16 对齐 32 = 32，恰好放入尾部
    EXPECT_EQ(b, 224u);
}

// best-fit：优先选择剩余最小的块（减少大块碎片）
TEST(DynTallocTest, BestFit) {
    DynTalloc ta(1024, 32);
    size_t a = 0, b = 0, c = 0, d = 0;
    ta.alloc(128, a);  // [0, 128)
    ta.alloc(512, b);  // [128, 640)
    ta.alloc(128, c);  // [640, 768)

    // 释放 a 和 c，产生 [0,128) 与 [640,768) 两个自由块
    ta.free_bytes(a, 128);
    ta.free_bytes(c, 128);

    // 分配 64 字节，best-fit 应命中较小的块（size=128 的那块）
    EXPECT_TRUE(ta.alloc(64, d));
    EXPECT_TRUE(d == 0u || d == 640u);
    // 剩余大小都还有 64，best-fit 取首个最小 → 0
    EXPECT_EQ(d, 0u);
}

// reset 恢复为整块
TEST(DynTallocTest, Reset) {
    DynTalloc ta(512, 32);
    size_t a = 0;
    ta.alloc(100, a);
    EXPECT_NE(ta.used_bytes(), 0u);
    ta.reset();
    EXPECT_EQ(ta.used_bytes(), 0u);
    EXPECT_EQ(ta.free_blocks().front().size, 512u);
}

#include <gtest/gtest.h>
#include "ppml/Tensor.h"

using namespace ppml;

TEST(TensorTest, BasicConstruction) {
    TensorF32 t({2, 3, 4}, Device::CPU);
    EXPECT_EQ(t.shape().numel(), 24);
    EXPECT_EQ(t.shape().ndim(), 3);
    EXPECT_EQ(t.device(), Device::CPU);
}

TEST(TensorTest, ZeroInitialization) {
    auto t = zeros<float>({10, 10}, Device::CPU);
    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(t.data()[i], 0.0f);
    }
}

TEST(TensorTest, DeviceTransfer) {
    auto cpu_t = zeros<float>({5, 5}, Device::CPU);
    // 如果 CUDA 可用
    auto cuda_t = cpu_t.to(Device::CUDA);
    EXPECT_EQ(cuda_t.device(), Device::CUDA);
    
    auto back_to_cpu = cuda_t.to(Device::CPU);
    EXPECT_EQ(back_to_cpu.device(), Device::CPU);
}

TEST(TensorTest, View) {
    TensorF32 t({2, 3, 4}, Device::CPU);
    auto v = t.view({6, 4});
    EXPECT_EQ(v.shape().numel(), 24);
    EXPECT_EQ(v.shape().ndim(), 2);
}

TEST(TensorTest, Select) {
    TensorF32 t({2, 4, 8}, Device::CPU);
    auto sliced = t.select(1, 0);
    EXPECT_EQ(sliced.shape().ndim(), 2);
    EXPECT_EQ(sliced.shape().dims[0], 2);
    EXPECT_EQ(sliced.shape().dims[1], 8);
}

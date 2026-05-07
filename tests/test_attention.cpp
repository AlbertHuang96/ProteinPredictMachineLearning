#include <gtest/gtest.h>
#include "rfaa/Attention.h"

using namespace rfaa;

TEST(AttentionTest, SelfAttentionForward) {
    AttnConfig config(256, 8);
    SelfAttention attn(config);
    
    TensorF32 x({2, 100, 256}, Device::CPU);
    auto out = attn.forward(x);
    
    EXPECT_EQ(out.shape().dims[0], 2);
    EXPECT_EQ(out.shape().dims[1], 100);
    EXPECT_EQ(out.shape().dims[2], 256);
}

TEST(AttentionTest, MSARowAttention) {
    AttnConfig config(256, 8);
    MSARowAttention attn(config);
    
    TensorF32 msa({2, 64, 100, 256}, Device::CPU);
    TensorF32 pair_bias({2, 100, 100, 8}, Device::CPU);
    
    auto out = attn.forward(msa, pair_bias);
    EXPECT_EQ(out.shape().dims[0], 2);
    EXPECT_EQ(out.shape().dims[1], 64);
    EXPECT_EQ(out.shape().dims[2], 100);
    EXPECT_EQ(out.shape().dims[3], 256);
}

TEST(AttentionTest, MSAColAttention) {
    AttnConfig config(256, 8);
    MSAColAttention attn(config);
    
    TensorF32 msa({2, 64, 100, 256}, Device::CPU);
    auto out = attn.forward(msa);
    
    EXPECT_EQ(out.shape().dims[0], 2);
    EXPECT_EQ(out.shape().dims[1], 64);
    EXPECT_EQ(out.shape().dims[2], 100);
    EXPECT_EQ(out.shape().dims[3], 256);
}

TEST(AttentionTest, TriangleMultiplication) {
    TriangleMultiplication tri_mul(128, TriangleMultiplication::Direction::Outgoing);
    
    TensorF32 pair({2, 50, 50, 128}, Device::CPU);
    auto out = tri_mul.forward(pair);
    
    EXPECT_EQ(out.shape().dims[0], 2);
    EXPECT_EQ(out.shape().dims[1], 50);
    EXPECT_EQ(out.shape().dims[2], 50);
    EXPECT_EQ(out.shape().dims[3], 128);
}

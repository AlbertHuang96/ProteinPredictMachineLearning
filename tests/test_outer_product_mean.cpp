#include <gtest/gtest.h>
#include <cstring>
#include "ppml/MathUtils.h"

using namespace ppml;

// 值版 outer_product_mean: einsum('bikd,bjkd->bijd(de)', left, right/N)
//   left/right (B, N, L, D) → dst (B, L, L, D*D)  特征笛卡尔积 D×D→D*D
//   dst[b,i,j,(d1*D+d2)] = (1/N) * sum_n left[b,n,i,d1] * right[b,n,j,d2]
TEST(OuterProductMeanTest, ValueForwardNumeric) {
    // B=1, N=2, L=2, D=2 → dst (1,2,2,4)
    TensorF32 left({1, 2, 2, 2}, Device::CPU);
    TensorF32 right({1, 2, 2, 2}, Device::CPU);

    // left[b,n,i,d]  (B=0)
    //   n=0: i0=(1,2), i1=(3,4)
    //   n=1: i0=(5,6), i1=(7,8)
    float ld[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::memcpy(left.data(), ld, sizeof(ld));
    // right[b,n,j,d]
    //   n=0: j0=(1,0), j1=(0,1)
    //   n=1: j0=(1,1), j1=(0,0)
    float rd[8] = {1, 0, 0, 1, 1, 1, 0, 0};
    std::memcpy(right.data(), rd, sizeof(rd));

    TensorF32 out = outer_product_mean(left, right);  // (1,2,2,4)
    ASSERT_EQ(out.shape().dims[0], 1);
    ASSERT_EQ(out.shape().dims[1], 2);
    ASSERT_EQ(out.shape().dims[2], 2);
    ASSERT_EQ(out.shape().dims[3], 4);   // D*D = 4

    const float* o = out.data();
    // dst[b,i,j,(d1*D+d2)] = 0.5*( left[b,0,i,d1]*right[b,0,j,d2]
    //                              + left[b,1,i,d1]*right[b,1,j,d2] )
    // idx = ((b*L+i)*L+j)*D2 + (d1*D+d2)
    // i=0,j=0:
    //   d1=0,d2=0: 0.5*(1*1 + 5*1)=3.0
    //   d1=0,d2=1: 0.5*(1*0 + 5*1)=2.5
    //   d1=1,d2=0: 0.5*(2*1 + 6*1)=4.0
    //   d1=1,d2=1: 0.5*(2*0 + 6*1)=3.0
    EXPECT_FLOAT_EQ(o[((0*2+0)*2+0)*4 + (0*2+0)], 3.0f);
    EXPECT_FLOAT_EQ(o[((0*2+0)*2+0)*4 + (0*2+1)], 2.5f);
    EXPECT_FLOAT_EQ(o[((0*2+0)*2+0)*4 + (1*2+0)], 4.0f);
    EXPECT_FLOAT_EQ(o[((0*2+0)*2+0)*4 + (1*2+1)], 3.0f);
    // i=1,j=1:
    //   d1=0,d2=0: 0.5*(3*0 + 7*0)=0.0
    //   d1=1,d2=1: 0.5*(4*1 + 8*0)=2.0
    EXPECT_FLOAT_EQ(o[((0*2+1)*2+1)*4 + (0*2+0)], 0.0f);
    EXPECT_FLOAT_EQ(o[((0*2+1)*2+1)*4 + (1*2+1)], 2.0f);
}

// 验证收缩 seq 维 N（值版）
TEST(OuterProductMeanTest, ContractOverSeqDim) {
    // B=1, N=3, L=2, D=1 → dst (1,2,2,1)
    TensorF32 left({1, 3, 2, 1}, Device::CPU);
    TensorF32 right({1, 3, 2, 1}, Device::CPU);
    // left[b,n,i,0] = n+i : [0,1, 1,2, 2,3]
    for (int n = 0; n < 3; ++n)
        for (int i = 0; i < 2; ++i)
            left.data()[((0*3+n)*2+i)*1+0] = float(n + i);
    for (int idx = 0; idx < 6; ++idx) right.data()[idx] = 1.0f;

    TensorF32 out = outer_product_mean(left, right);  // (1,2,2,1)
    const float* o = out.data();
    // dst[0,i,j,(0*1+0)] = (1/3)*sum_n (n+i)*1 = i+1
    EXPECT_FLOAT_EQ(o[((0*2+0)*2+0)*1 + 0], 1.0f);
    EXPECT_FLOAT_EQ(o[((0*2+0)*2+1)*1 + 0], 1.0f);
    EXPECT_FLOAT_EQ(o[((0*2+1)*2+0)*1 + 0], 2.0f);
    EXPECT_FLOAT_EQ(o[((0*2+1)*2+1)*1 + 0], 2.0f);
}

// 值版 outer_product_cartesian（gate 纯外积）：left/right (B,L,D) → (B,L,L,D*D)
//   dst[b,i,j,(d1*D+d2)] = left[b,i,d1] * right[b,j,d2]
TEST(OuterProductCartesianTest, ValueForwardNumeric) {
    // B=1, L=2, D=2 → dst (1,2,2,4)
    TensorF32 left({1, 2, 2}, Device::CPU);
    TensorF32 right({1, 2, 2}, Device::CPU);
    // left[b,i,d]: i0=(1,2), i1=(3,4)
    float ld[4] = {1, 2, 3, 4};
    std::memcpy(left.data(), ld, sizeof(ld));
    // right[b,j,d]: j0=(1,0), j1=(0,1)
    float rd[4] = {1, 0, 0, 1};
    std::memcpy(right.data(), rd, sizeof(rd));

    TensorF32 out = outer_product_cartesian(left, right);  // (1,2,2,4)
    ASSERT_EQ(out.shape().dims[3], 4);
    const float* o = out.data();
    // idx = ((b*L+i)*L+j)*D2 + (d1*D+d2)
    // i=0,j=0: left[0,0,d1]*right[0,0,d2]
    //   (d1,d2)=(0,0): 1*1=1 ; (0,1): 1*0=0 ; (1,0): 2*1=2 ; (1,1): 2*0=0
    EXPECT_FLOAT_EQ(o[((0*2+0)*2+0)*4 + 0], 1.0f);
    EXPECT_FLOAT_EQ(o[((0*2+0)*2+0)*4 + 1], 0.0f);
    EXPECT_FLOAT_EQ(o[((0*2+0)*2+0)*4 + 2], 2.0f);
    EXPECT_FLOAT_EQ(o[((0*2+0)*2+0)*4 + 3], 0.0f);
    // i=0,j=1: left[0,0,d1]*right[0,1,d2]
    //   (0,0): 1*0=0 ; (0,1): 1*1=1 ; (1,0): 2*0=0 ; (1,1): 2*1=2
    EXPECT_FLOAT_EQ(o[((0*2+0)*2+1)*4 + 1], 1.0f);
    EXPECT_FLOAT_EQ(o[((0*2+0)*2+1)*4 + 3], 2.0f);
}

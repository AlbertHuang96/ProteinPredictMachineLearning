#include <gtest/gtest.h>
#include "ppml/Model.h"

using namespace ppml;

TEST(ModelTest, PPMLConfigDefault) {
    PPMLConfig config;
    EXPECT_EQ(config.d_msa, 256);
    EXPECT_EQ(config.d_pair, 128);
    EXPECT_EQ(config.d_state, 32);
    EXPECT_EQ(config.n_extra_blocks, 4);
    EXPECT_EQ(config.n_main_blocks, 8);
    EXPECT_EQ(config.n_refine_blocks, 4);
}

TEST(ModelTest, ModelCreation) {
    PPMLConfig config;
    config.n_extra_blocks = 1;
    config.n_main_blocks = 1;
    config.n_refine_blocks = 1;
    
    PPMLModel model(config);
    EXPECT_EQ(model.device(), Device::CPU);
    EXPECT_FALSE(model.is_training());
}

TEST(ModelTest, ModelForward) {
    PPMLConfig config;
    config.n_extra_blocks = 1;
    config.n_main_blocks = 1;
    config.n_refine_blocks = 1;
    
    PPMLModel model(config);
    model.eval();
    
    int B = 1, N = 16, L = 32, T = 2;
    ModelInput input;
    input.msa_latent = zeros<float>({B, N, L, MSA_LATENT_DIM}, Device::CPU);
    input.seq_tokens = zeros<float>({B, L}, Device::CPU);
    input.t1d = zeros<float>({B, T, L, D_T1D}, Device::CPU);
    input.coords = zeros<float>({B, L, 3, 3}, Device::CPU);
    // 补全 forward 用到的字段（避免空 Tensor 解引用崩溃，PPML.cpp:2122 residx）
    input.residx = zeros<int64_t>({B, L}, Device::CPU);
    for (int l = 0; l < L; ++l) input.residx.data()[l] = l;
    input.tor_feat = zeros<float>({B, T, L, D_TOR}, Device::CPU);
    input.t2d = zeros<float>({B, T, L, L, D_T2D}, Device::CPU);
    input.bond_feats = zeros<float>({B, L, L, 5}, Device::CPU);
    input.dist_matrix = zeros<float>({B, L, L}, Device::CPU);
    input.same_chain = zeros<float>({B, L, L}, Device::CPU);
    input.template_mask = zeros<float>({B, T, L}, Device::CPU);
    
    auto output = model.forward(input);
    
    EXPECT_EQ(output.msa.shape().dims[0], B);
    EXPECT_EQ(output.pair.shape().dims[0], B);
    EXPECT_EQ(output.state.shape().dims[0], B);
    EXPECT_EQ(output.coords.shape().dims[0], B);
}

TEST(ModelTest, ModelTrainEval) {
    PPMLConfig config;
    PPMLModel model(config);
    
    model.train();
    EXPECT_TRUE(model.is_training());
    
    model.eval();
    EXPECT_FALSE(model.is_training());
}

TEST(ModelTest, DeviceTransfer) {
    PPMLConfig config;
    PPMLModel model(config);
    
    model.to(Device::CUDA);
    EXPECT_EQ(model.device(), Device::CUDA);
}

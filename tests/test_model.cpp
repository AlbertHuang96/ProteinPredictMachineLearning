#include <gtest/gtest.h>
#include "rfaa/Model.h"

using namespace rfaa;

TEST(ModelTest, RFAAConfigDefault) {
    RFAAConfig config;
    EXPECT_EQ(config.d_msa, 256);
    EXPECT_EQ(config.d_pair, 128);
    EXPECT_EQ(config.d_state, 32);
    EXPECT_EQ(config.n_extra_blocks, 4);
    EXPECT_EQ(config.n_main_blocks, 8);
    EXPECT_EQ(config.n_refine_blocks, 4);
}

TEST(ModelTest, ModelCreation) {
    RFAAConfig config;
    config.n_extra_blocks = 1;
    config.n_main_blocks = 1;
    config.n_refine_blocks = 1;
    
    RFAAModel model(config);
    EXPECT_EQ(model.device(), Device::CPU);
    EXPECT_FALSE(model.is_training());
}

TEST(ModelTest, ModelForward) {
    RFAAConfig config;
    config.n_extra_blocks = 1;
    config.n_main_blocks = 1;
    config.n_refine_blocks = 1;
    
    RFAAModel model(config);
    model.eval();
    
    int B = 1, N = 16, L = 32, T = 2;
    ModelInput input;
    input.msa_latent = zeros<float>({B, N, L, MSA_LATENT_DIM}, Device::CPU);
    input.seq_tokens = zeros<float>({B, L}, Device::CPU);
    input.t1d = zeros<float>({B, T, L, D_T1D}, Device::CPU);
    input.coords = zeros<float>({B, L, 3, 3}, Device::CPU);
    
    auto output = model.forward(input);
    
    EXPECT_EQ(output.msa.shape().dims[0], B);
    EXPECT_EQ(output.pair.shape().dims[0], B);
    EXPECT_EQ(output.state.shape().dims[0], B);
    EXPECT_EQ(output.coords.shape().dims[0], B);
}

TEST(ModelTest, ModelTrainEval) {
    RFAAConfig config;
    RFAAModel model(config);
    
    model.train();
    EXPECT_TRUE(model.is_training());
    
    model.eval();
    EXPECT_FALSE(model.is_training());
}

TEST(ModelTest, DeviceTransfer) {
    RFAAConfig config;
    RFAAModel model(config);
    
    model.to(Device::CUDA);
    EXPECT_EQ(model.device(), Device::CUDA);
}

# GPU Backend: CUDA 加速后端 (独立 DLL)

set(GPU_SOURCES
    ../src/cuda/CudaTensor.cu
    ../src/cuda/CUDAKernels.cu
    ../src/cuda/AttentionKernel.cu
    ../src/cuda/SE3Kernel.cu
    ../src/cuda/MathKernel.cu
    ../src/cuda/OutProdKernel.cu
)


# CUDA 动态库
#cuda_add_library
add_library(rfaa_gpu SHARED ${GPU_SOURCES})

target_include_directories(rfaa_gpu PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(rfaa_gpu PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
)

# 编译选项
#set_target_properties(rfaa_gpu PROPERTIES
    #CUDA_SEPARABLE_COMPILATION ON
    #CUDA_ARCHITECTURES "86"
#)

# 2050 Ampere sm_86 
# CUDA_ARCHITECTURES "75;80;86"
#Could not open input file '/usr/lib/x86_64-linux-gnu/librt.a' (target: sm_75)
# set(CMAKE_CUDA_ARCHITECTURES 70)
# nvidia-smi -q | grep Architecture

# DLL 导出宏
#target_compile_definitions(rfaa_gpu PRIVATE RFAA_GPU_BUILD_DLL)

# 安装
#install(TARGETS rfaa_gpu
    #RUNTIME DESTINATION bin
    #LIBRARY DESTINATION lib
#)

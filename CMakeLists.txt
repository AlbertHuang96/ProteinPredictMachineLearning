cmake_minimum_required(VERSION 3.18)
project(ppml VERSION 1.0.0 LANGUAGES CXX CUDA)
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# 选项
option(USE_CUDA "Enable CUDA GPU acceleration" ON)
option(BUILD_CORE_LIB "Build core as static/shared library" ON)
option(BUILD_GPU_BACKEND "Build GPU backend as DLL" ON)
option(BUILD_PYTHON_BRIDGE "Build Python bridge DLL" ON)
option(BUILD_EXAMPLES "Build example applications" ON)
option(BUILD_TESTS "Build unit tests" OFF)
#cmake -DBUILD_TESTS=OFF

# 查找依赖
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)

# ONNX Runtime
#set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
#find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_DIR}/lib)
#find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h PATHS ${ONNXRUNTIME_DIR}/include)

# pybind11
add_subdirectory(third_party/pybind11)

# 全局包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    #${ONNXRUNTIME_INCLUDE}
)

# 编译选项
if(MSVC)
    add_compile_options(/W4)
    # /O2
else()
    add_compile_options(-Wall -fPIC)
    # -O3
endif()

# GPU 架构自动检测
# 优先级：
#   1. 用户显式 -DCMAKE_CUDA_ARCHITECTURES=<arch>  (如 89) 直接采用
#   2. nvidia-smi 探测当前机器 GPU 的 compute capability  (自编译自跑最优)
#   3. 探测失败则回退全家桶 (80;86;89;90)，一份产物兼容多代 Ampere/Ada/Hopper
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE COMPUTE_CAP
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE SMI_RESULT
        )
        # "8.6" -> "86"；"9.0" -> "90"；支持多 GPU 时取第一行
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" COMPUTE_CAP_FIRST "${COMPUTE_CAP}")
        if(SMI_RESULT EQUAL 0 AND COMPUTE_CAP_FIRST MATCHES "^[0-9]+\\.[0-9]+")
            string(REPLACE "." "" CMAKE_CUDA_ARCHITECTURES "${COMPUTE_CAP_FIRST}")
            message(STATUS "自动检测到 CUDA 架构: compute_${CMAKE_CUDA_ARCHITECTURES} (来自 nvidia-smi)")
        else()
            message(STATUS "nvidia-smi 探测失败，回退到多架构全家桶")
        endif()
    else()
        message(STATUS "未找到 nvidia-smi，回退到多架构全家桶")
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
        message(STATUS "CUDA 架构回退为全家桶: 80;86;89;90")
    endif()
endif()
message(STATUS "最终 CUDA 架构: ${CMAKE_CUDA_ARCHITECTURES}")

# 子目录
add_subdirectory(core_lib)
add_subdirectory(gpu_backend)
add_subdirectory(python_bridge)


if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

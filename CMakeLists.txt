cmake_minimum_required(VERSION 3.18)
project(RFAA-Cpp VERSION 1.0.0 LANGUAGES CXX CUDA)
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

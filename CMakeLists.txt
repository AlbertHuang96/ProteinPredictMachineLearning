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
    add_compile_definitions(NOMINMAX _USE_MATH_DEFINES)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
    # /O2
else()
    # 训练性能需要：朴素矩阵乘等在 -O0 下极慢(5-20x)，开 -O2 保留 -g 符号 (RelWithDebInfo 风格)。
    # 若需完整调试可改回 -O0，但训练 forward/backward 会极慢。
    add_compile_options(-Wall -fPIC -O2 -g)
endif()

# ============================================================
# ASan / UBSan 构建（2026-09-11）
#   用途：抓 Shape::dims（std::vector）越界读、堆越界读写等 UB。
#   用法：cmake -S . -B build/asan -DPPML_SANITIZE=address,undefined
#   要点：
#     * C++ 源：直接加 -fsanitize=...（编译 + 链接都加，宿主侧插桩）
#     * .cu 源：nvcc 不认识 -fsanitize，须用 -Xcompiler= 透传给宿主编译器；
#       同时把该 flag 放在 CMAKE_CUDA_FLAGS 里，nvcc 的链接行也会带上 → 混合
#       CXX+CUDA 目标（ppml_core）链接时能找到 ASan 运行时。
#     * 不设 -fno-sanitize-recover，让 UBSan 报错后继续跑，一次收集尽可能多的问题。
# ============================================================
set(PPML_SANITIZE "" CACHE STRING "宿主侧 sanitizer 列表，如 address,undefined（空=关闭）")
if(PPML_SANITIZE)
    message(STATUS "[SANITIZE] enabled: ${PPML_SANITIZE}")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${PPML_SANITIZE}>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
        # 关键补充：libstdc++ 的 operator[] 越界断言。
        #   GCC 的 ASan **抓不到** std::vector 的 `v[i]`（i ≥ size 但 < capacity）——
        #   例如本项目的 `Shape::dims`（vector<int64_t>）在 3 维 shape 上读 dims[3]：
        #   vector 增长 1→2→4，capacity=4 → 该读落在已分配块内，无红区 → ASan 静默。
        #   `_GLIBCXX_ASSERTIONS` 让 operator[] 显式检查 i < size() 并 abort（ABI 兼容）。
        $<$<COMPILE_LANGUAGE:CXX>:-D_GLIBCXX_ASSERTIONS>
    )
    add_link_options($<$<LINK_LANGUAGE:CXX>:-fsanitize=${PPML_SANITIZE}>)

    string(REPLACE "," ";" _ppml_san_list "${PPML_SANITIZE}")
    foreach(_s IN LISTS _ppml_san_list)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fsanitize=${_s}")
    endforeach()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-omit-frame-pointer")
    # .cu 文件里同样可能读 std::vector（如 kernel 里访问 shape().dims[i]）→ 断言开关一并透传
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-D_GLIBCXX_ASSERTIONS")
    message(STATUS "[SANITIZE] CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}")
endif()

# ============================================================
# 环境信息打印（构建日志里的 CPU / GPU / 编译器 / 编译类型）
# ============================================================
# --- CPU ---
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    find_program(LSCPU_CMD lscpu)
    if(LSCPU_CMD)
        execute_process(COMMAND ${LSCPU_CMD} -p OUTPUT_VARIABLE LSCPU_OUT)
        execute_process(COMMAND ${LSCPU_CMD} --version OUTPUT_VARIABLE LSCPU_VER)
        if(LSCPU_OUT)
            list(LENGTH LSCPU_OUT LSCPU_NLINES)
            string(REGEX MATCH "^[0-9]+" LSCPU_CORES "${LSCPU_NLINES}")
            if(NOT LSCPU_CORES)
                set(LSCPU_CORES ${LSCPU_NLINES})
            endif()
        endif()
    endif()
    # CPU 型号（去掉 MHz 频率项，只留型号名）
    find_program(GREP_CMD grep)
    find_program(SED_CMD sed)
    if(LSCPU_CMD AND GREP_CMD AND SED_CMD)
        execute_process(
            COMMAND sh -c "${LSCPU_CMD} | grep -m1 '^Model name:' | sed 's/^Model name:[[:space:]]*//'"
            OUTPUT_VARIABLE CPU_MODEL
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
elseif(APPLE)
    execute_process(COMMAND sysctl -n machdep.cpu.brand_string OUTPUT_VARIABLE CPU_MODEL OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND sysctl -n hw.logicalcpu OUTPUT_VARIABLE LSCPU_CORES OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
if(NOT CPU_MODEL)
    set(CPU_MODEL "unknown")
endif()
message(STATUS "[ENV] CPU: ${CPU_MODEL} | logical cores: ${LSCPU_CORES}")

# --- GPU ---
find_program(NVIDIA_SMI nvidia-smi)
if(NVIDIA_SMI)
    execute_process(COMMAND ${NVIDIA_SMI} --query-gpu=name,memory.total --format=csv,noheader OUTPUT_VARIABLE GPU_INFO OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(GPU_INFO)
        string(REPLACE "\\n" " | " GPU_INFO "${GPU_INFO}")
        string(REGEX REPLACE "\n" " | " GPU_INFO "${GPU_INFO}")
        message(STATUS "[ENV] GPU(s): ${GPU_INFO}")
    else()
        message(STATUS "[ENV] GPU(s): none detected via nvidia-smi")
    endif()
else()
    message(STATUS "[ENV] GPU(s): nvidia-smi not found (no NVIDIA GPU / no driver)")
endif()

# --- 编译器 & 编译类型 ---
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(_cxx_id "GCC ${CMAKE_CXX_COMPILER_VERSION}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_cxx_id "Clang ${CMAKE_CXX_COMPILER_VERSION}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(_cxx_id "MSVC ${CMAKE_CXX_COMPILER_VERSION}")
else()
    set(_cxx_id "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
endif()
message(STATUS "[ENV] C++ compiler: ${_cxx_id}")
message(STATUS "[ENV] Build type: ${CMAKE_BUILD_TYPE} | CUDA enabled: ${USE_CUDA}")

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

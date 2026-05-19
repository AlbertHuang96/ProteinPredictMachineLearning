# Core Library: 静态库/动态库保护核心功能

find_package(CUDAToolkit REQUIRED)

set(CORE_SOURCES
    ../src/core/Tensor.cpp
    ../src/core/MemoryPool.cpp
    ../src/model/Embedding.cpp
    ../src/model/IterBlock.cpp
    ../src/model/RFAA.cpp
    ../src/tracks/MSA.cpp
    ../src/tracks/Pair.cpp
    ../src/tracks/State.cpp
    ../src/modules/Attention.cpp
    ../src/modules/TriangleMul.cpp
    ../src/modules/SE3Transformer.cpp
    ../src/ops/Softmax.cpp
    ../src/ops/Activation.cpp
    ../src/ops/LayerNorm.cpp
    ../src/ops/MathUtils.cpp
    ../src/model/PositionalEncoding.cpp
)

# 静态库 (内部使用)
add_library(rfaa_core_static STATIC ${CORE_SOURCES})
target_include_directories(rfaa_core_static PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(rfaa_core_static PUBLIC
    CUDA::cudart
    CUDA::cublas
    #CUDA::cudnn
)

# 动态库 (对外暴露接口)
add_library(rfaa_core SHARED ${CORE_SOURCES})
target_include_directories(rfaa_core PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(rfaa_core PUBLIC
    CUDA::cudart
    CUDA::cublas
    #CUDA::cudnn
)

# 设置 DLL 导出宏
target_compile_definitions(rfaa_core PRIVATE RFAA_BUILD_DLL)
target_compile_definitions(rfaa_core PUBLIC RFAA_USE_DLL)

# Windows: 导出所有符号
if(MSVC)
    set_target_properties(rfaa_core PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# 安装规则
install(TARGETS rfaa_core
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/rfaa 
    DESTINATION include
)

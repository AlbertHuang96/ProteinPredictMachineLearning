# Python Bridge: C++ ↔ Python 桥接 (独立 DLL)

set(PYTHON_BRIDGE_SOURCES
    ../src/python/PythonBridge.cpp
    ../src/python/PyBindModule.cpp
)

# Python 桥接动态库
add_library(rfaa_python SHARED ${PYTHON_BRIDGE_SOURCES})

target_include_directories(rfaa_python PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(rfaa_python PUBLIC
    rfaa_core_static
    pybind11::module
    ${Python3_LIBRARIES}
)

# 编译定义
target_compile_definitions(rfaa_python PRIVATE RFAA_PYTHON_BUILD_DLL)

# pybind11 模块 (供 Python import)
pybind11_add_module(pyrfaa 
    ../src/python/PyBindModule.cpp
)

target_link_libraries(pyrfaa PRIVATE rfaa_core_static)

# 安装
install(TARGETS rfaa_python pyrfaa
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

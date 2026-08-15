# Python Bridge: C++ ↔ Python 桥接 (独立 DLL)

set(PYTHON_BRIDGE_SOURCES
    ../src/python/PythonBridge.cpp
    ../src/python/PyBindModule.cpp
)

# Python 桥接动态库
add_library(ppml_python SHARED ${PYTHON_BRIDGE_SOURCES})

target_include_directories(ppml_python PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(ppml_python PUBLIC
    ppml_core_static
    pybind11::module
    ${Python3_LIBRARIES}
)

# 编译定义
target_compile_definitions(ppml_python PRIVATE PPML_PYTHON_BUILD_DLL)

# pybind11 模块 (供 Python import)
pybind11_add_module(pyppml 
    ../src/python/PyBindModule.cpp
)

target_link_libraries(pyppml PRIVATE ppml_core_static)

# 安装
install(TARGETS ppml_python pyppml
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

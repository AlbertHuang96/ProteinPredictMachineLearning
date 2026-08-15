# Examples

add_executable(ppml_train train.cpp)
target_link_libraries(ppml_train PRIVATE ppml_core ppml_python)

add_executable(ppml_infer infer.cpp)
target_link_libraries(ppml_infer PRIVATE ppml_core ppml_python)

# 内存测量工具：只构建图 + 引用计数峰值模拟，不执行 compute
add_executable(ppml_measure measure_graph.cpp)
target_link_libraries(ppml_measure PRIVATE ppml_core ppml_python)

#add_executable(ppml_export_onnx export_onnx.cpp)
#target_link_libraries(ppml_export_onnx PRIVATE ppml_core)

# 安装
install(TARGETS ppml_train ppml_infer #ppml_export_onnx
    RUNTIME DESTINATION bin
)

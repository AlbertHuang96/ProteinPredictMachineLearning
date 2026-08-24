# Examples

add_executable(ppml_train train.cpp)
target_link_libraries(ppml_train PRIVATE ppml_core ppml_python)

add_executable(ppml_infer infer.cpp)
target_link_libraries(ppml_infer PRIVATE ppml_core ppml_python)

# 内存测量工具：只构建图 + 引用计数峰值模拟，不执行 compute
add_executable(ppml_measure measure_graph.cpp)
target_link_libraries(ppml_measure PRIVATE ppml_core ppml_python)

# 多样本加载测试：不训练，仅加载数据集前几个样本打印维度
add_executable(ppml_test_dataloader test_dataloader.cpp)
target_link_libraries(ppml_test_dataloader PRIVATE ppml_core)

# 模板注入验证：加载带/无模板数据对比 forward_graph 的 state/pair（CPU）
add_executable(ppml_verify_template verify_template.cpp)
target_link_libraries(ppml_verify_template PRIVATE ppml_core ppml_python)

# const_data_ + TENSOR_FLAG_CONST 机制验证（CPU）
add_executable(ppml_verify_const_data verify_const_data.cpp)
target_link_libraries(ppml_verify_const_data PRIVATE ppml_core)

#add_executable(ppml_export_onnx export_onnx.cpp)
#target_link_libraries(ppml_export_onnx PRIVATE ppml_core)

# 安装
install(TARGETS ppml_train ppml_infer #ppml_export_onnx
    RUNTIME DESTINATION bin
)

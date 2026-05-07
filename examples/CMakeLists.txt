# Examples

add_executable(rfaa_train train.cpp)
target_link_libraries(rfaa_train PRIVATE rfaa_core rfaa_python)

add_executable(rfaa_infer infer.cpp)
target_link_libraries(rfaa_infer PRIVATE rfaa_core rfaa_python)

#add_executable(rfaa_export_onnx export_onnx.cpp)
#target_link_libraries(rfaa_export_onnx PRIVATE rfaa_core)

# 安装
install(TARGETS rfaa_train rfaa_infer #rfaa_export_onnx
    RUNTIME DESTINATION bin
)

# Tests

# 注意: 禁用 find_package(GTest) —— 本机 WSL 的 PATH 含 Windows miniconda 库路径
# (/mnt/c/Users/86422/miniconda3/Library/...), 它会命中 MSVC 编译的 gtest.lib,
# 在 Linux 下链接必然失败(undefined reference)。强制 FetchContent 下载 Linux 版。
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(ppml_tests
    test_tensor.cpp
    test_attention.cpp
    test_softmax_back.cpp
    test_model.cpp
    test_lora.cpp
    test_outer_product_mean.cpp
    test_dyntalloc.cpp
    test_gallocr.cpp
    test_cuda.cpp
    test_scan.cpp
    test_mul_mat_f16.cpp
    test_quant_prep.cpp
    test_se3_equivariance.cpp
    test_se3_graph_backward.cpp
    test_se3_forward_graph.cpp
)

target_link_libraries(ppml_tests PRIVATE
    ppml_core
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(ppml_tests)

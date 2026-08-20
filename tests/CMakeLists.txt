# Tests

find_package(GTest)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(ppml_tests
    test_tensor.cpp
    test_attention.cpp
    test_model.cpp
    test_outer_product_mean.cpp
    test_dyntalloc.cpp
    test_gallocr.cpp
    test_cuda.cpp
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

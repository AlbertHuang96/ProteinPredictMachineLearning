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

add_executable(rfaa_tests
    test_tensor.cpp
    test_attention.cpp
    test_model.cpp
)

target_link_libraries(rfaa_tests PRIVATE
    rfaa_core
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(rfaa_tests)

#pragma once

// DLL 导出宏定义

#ifdef _WIN32
    #ifdef PPML_BUILD_DLL
        #define PPML_API __declspec(dllexport)
    #elif defined(PPML_USE_DLL)
        #define PPML_API __declspec(dllimport)
    #else
        #define PPML_API
    #endif
#else
    #define PPML_API __attribute__((visibility("default")))
#endif

// GPU 后端导出
#ifdef _WIN32
    #ifdef PPML_GPU_BUILD_DLL
        #define PPML_GPU_API __declspec(dllexport)
    #else
        #define PPML_GPU_API __declspec(dllimport)
    #endif
#else
    #define PPML_GPU_API __attribute__((visibility("default")))
#endif

// Python 桥接导出
#ifdef _WIN32
    #ifdef PPML_PYTHON_BUILD_DLL
        #define PPML_PYTHON_API __declspec(dllexport)
    #else
        #define PPML_PYTHON_API __declspec(dllimport)
    #endif
#else
    #define PPML_PYTHON_API __attribute__((visibility("default")))
#endif

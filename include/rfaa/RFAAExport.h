#pragma once

// DLL 导出宏定义

#ifdef _WIN32
    #ifdef RFAA_BUILD_DLL
        #define RFAA_API __declspec(dllexport)
    #elif defined(RFAA_USE_DLL)
        #define RFAA_API __declspec(dllimport)
    #else
        #define RFAA_API
    #endif
#else
    #define RFAA_API __attribute__((visibility("default")))
#endif

// GPU 后端导出
#ifdef _WIN32
    #ifdef RFAA_GPU_BUILD_DLL
        #define RFAA_GPU_API __declspec(dllexport)
    #else
        #define RFAA_GPU_API __declspec(dllimport)
    #endif
#else
    #define RFAA_GPU_API __attribute__((visibility("default")))
#endif

// Python 桥接导出
#ifdef _WIN32
    #ifdef RFAA_PYTHON_BUILD_DLL
        #define RFAA_PYTHON_API __declspec(dllexport)
    #else
        #define RFAA_PYTHON_API __declspec(dllimport)
    #endif
#else
    #define RFAA_PYTHON_API __attribute__((visibility("default")))
#endif

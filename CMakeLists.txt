cmake_minimum_required(VERSION 3.24)

# CMake policy settings for compatibility
cmake_policy(SET CMP0144 NEW)
cmake_policy(SET CMP0146 NEW)

# Check if CUDA should be enabled (default: OFF for CPU Core Module)
option(TR_USE_CUDA "Enable CUDA support" OFF)

# Only enable CUDA language if requested
if(TR_USE_CUDA)
    project(tech_renaissance LANGUAGES CXX CUDA)
else()
    project(tech_renaissance LANGUAGES CXX)
    message(STATUS "Building CPU Core Module - CUDA support disabled")
endif()

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA standards and configuration (only if CUDA is enabled)
if(TR_USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
endif()

# Add cmake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Load user configuration if exists
if(EXISTS "${CMAKE_SOURCE_DIR}/config/user_paths.cmake")
    include("${CMAKE_SOURCE_DIR}/config/user_paths.cmake")
    message(STATUS "Loaded user configuration from config/user_paths.cmake")
endif()

# Configure vcpkg if available
if(DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
elseif(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    message(STATUS "Using vcpkg from: $ENV{VCPKG_ROOT}")
endif()

# 编译器检测 - 支持MSVC和GCC以实现跨平台
if(MSVC)
    message(STATUS "检测到Microsoft Visual C++编译器，版本: ${CMAKE_CXX_COMPILER_VERSION}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "检测到GCC编译器，版本: ${CMAKE_CXX_COMPILER_VERSION}")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13.0")
        message(WARNING "GCC版本过低，建议使用GCC 13.0或更高版本")
    endif()
else()
    message(WARNING "检测到不支持的编译器: ${CMAKE_CXX_COMPILER_ID}")
    message(WARNING "建议使用MSVC或GCC 13.0+")
endif()

# === Finding Dependencies ===

# Find OpenMP for performance optimization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP support: ENABLED for maximum performance")
else()
    message(STATUS "OpenMP support: DISABLED - not found")
endif()

# Find Eigen3 library
message(STATUS "Searching for Eigen3...")
if(DEFINED EIGEN3_INCLUDE_DIR)
    set(EIGEN3_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")
    set(EIGEN3_FOUND TRUE)
    message(STATUS "Using configured Eigen3: ${EIGEN3_INCLUDE_DIR}")
else()
    find_package(Eigen3 QUIET)
    if(Eigen3_FOUND)
        message(STATUS "Found Eigen3: ${Eigen3_INCLUDE_DIRS}")
    else()
        message(WARNING "Eigen3 not found - some optimizations will be disabled")
    endif()
endif()

# CUDA and cuDNN (only if enabled)
if(TR_USE_CUDA)
    # CUDA - Use CUDAToolkit_ROOT from config/user_paths.cmake
    find_package(CUDAToolkit 12.8 REQUIRED)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
        message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
    else()
        message(FATAL_ERROR "CUDA not found. Please install CUDA Toolkit 12.8")
    endif()

    # cuDNN - Use CUDNN_ROOT from config/user_paths.cmake
    if(DEFINED CUDNN_ROOT)
        set(CUDNN_INCLUDE_DIR "${CUDNN_ROOT}/include")
        if(WIN32)
            set(CUDNN_LIBRARY_DIR "${CUDNN_ROOT}/lib/x64")
        else()
            if(EXISTS "${CUDNN_ROOT}/lib64")
                set(CUDNN_LIBRARY_DIR "${CUDNN_ROOT}/lib64")
            else()
                set(CUDNN_LIBRARY_DIR "${CUDNN_ROOT}/lib")
            endif()
        endif()

        # Check if cuDNN files exist
        if(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn.h")
            message(STATUS "Found cuDNN: ${CUDNN_ROOT}")
            set(CUDNN_FOUND TRUE)
        else()
            message(FATAL_ERROR "cuDNN headers not found in ${CUDNN_ROOT}/include")
        endif()
    else()
        message(FATAL_ERROR "CUDNN_ROOT not defined in config/user_paths.cmake")
    endif()
endif()

# 固定指向工作目录，方便所有测试文件查找，避免歧义
add_compile_definitions(WORKSPACE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/workspace")

# 设置workspace目录路径宏
set(WORKSPACE_DIR "${CMAKE_SOURCE_DIR}/workspace")
add_definitions(-DWORKSPACE_DIR="${WORKSPACE_DIR}")

# 确保workspace目录存在
if(NOT EXISTS "${WORKSPACE_DIR}")
    file(MAKE_DIRECTORY "${WORKSPACE_DIR}")
    message(STATUS "Created workspace directory: ${WORKSPACE_DIR}")
else()
    message(STATUS "Workspace directory exists: ${WORKSPACE_DIR}")
endif()

# MSVC特定设置
if(MSVC)
    # 添加Windows特定的宏定义
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DNOMINMAX)
    # 设置运行时库
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    set_property(GLOBAL PROPERTY MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}")

    # 修复PDB文件冲突问题 - 让每个目标管理自己的PDB文件
    add_compile_options(/Zi)  # 启用调试信息
    add_compile_options($<$<CONFIG:Debug>:/FS>)  # Debug模式下启用文件系统PDB

    # 添加UTF-8编码支持
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    message(STATUS "MSVC UTF-8 encoding support enabled for CXX")
    message(STATUS "MSVC PDB settings configured for parallel compilation")
endif()

# GCC特定设置
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC优化标志
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -fopenmp)
        # 移除LTO以提高兼容性，对深度学习框架性能提升有限
        message(STATUS "GCC optimizations enabled: -O3 -march=native -fopenmp")
    endif()
endif()

# 全局编译选项
option(TR_BUILD_TESTS "Build tests" ON)
option(TR_BUILD_UTILS "Build utils" ON)
option(TR_USE_EIGEN "Enable Eigen for CPU optimizations" ${EIGEN3_FOUND})

# 全局Eigen配置
if(TR_USE_EIGEN AND EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIRS})
    add_compile_definitions(TR_USE_EIGEN)
    message(STATUS "Eigen optimizations: ENABLED for CPU backend")
else()
    message(STATUS "Eigen optimizations: DISABLED")
    set(TR_USE_EIGEN OFF)
endif()

# 输出编译器信息
message(STATUS "编译器: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "编译器版本: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")

# 添加子目录
add_subdirectory(src)

if(TR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
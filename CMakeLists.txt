cmake_minimum_required(VERSION 3.20)

# 设置项目
project(tech_renaissance VERSION 1.0.0 LANGUAGES CXX)

# 编译器检测和强制使用MSVC
if(NOT MSVC)
    message(FATAL_ERROR "技术觉醒框架必须使用Microsoft Visual C++ (cl.exe)编译器。检测到的编译器: ${CMAKE_CXX_COMPILER_ID}")
    message(FATAL_ERROR "请安装Visual Studio 2019/2022或Build Tools for Visual Studio")
endif()
message(STATUS "检测到Microsoft Visual C++编译器")

# 设置C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 固定指向工作目录，方便所有测试文件查找，避免歧义
add_compile_definitions(WORKSPACE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/workspace")

# OpenMP配置选项 (兼顾性能和可移植性)
option(ENABLE_OPENMP "Enable OpenMP multi-threading support for better performance" ON)
if(ENABLE_OPENMP)
    message(STATUS "OpenMP support: ENABLED for maximum performance")
else()
    message(STATUS "OpenMP support: DISABLED for embedded platform compatibility")
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

    # 添加UTF-8编码支持（Expert A推荐的完整修复）
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    message(STATUS "MSVC UTF-8 encoding support enabled for both CXX and CUDA")
    message(STATUS "MSVC PDB settings configured for parallel compilation")
endif()

# 全局编译选项
option(TR_USE_PROTOBUF "Use protobuf for serialization/ONNX" OFF)
option(TR_USE_ONNX "Enable ONNX export (requires protobuf)" OFF)

# 配置vcpkg
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
endif()


if(TR_USE_PROTOBUF)
    find_package(Protobuf CONFIG REQUIRED)
endif()

# 尝试启用CUDA支持
option(TR_ENABLE_CUDA "Enable CUDA support" ON)

if(TR_ENABLE_CUDA)
    # 方法1: 从PATH环境变量中查找nvcc
    find_program(NVCC_EXECUTABLE nvcc
        PATHS $ENV{CUDA_PATH}/bin $ENV{CUDA_HOME}/bin
        DOC "NVIDIA CUDA Compiler"
    )

    # 方法2: 如果PATH中没有找到，尝试常见安装路径
    if(NOT NVCC_EXECUTABLE)
        set(POSSIBLE_CUDA_PATHS
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.7"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"
            "/usr/local/cuda"
            "/opt/cuda"
        )

        foreach(CUDA_PATH ${POSSIBLE_CUDA_PATHS})
            if(EXISTS "${CUDA_PATH}/bin/nvcc.exe" OR EXISTS "${CUDA_PATH}/bin/nvcc")
                set(NVCC_EXECUTABLE "${CUDA_PATH}/bin/nvcc")
                break()
            endif()
        endforeach()
    endif()

    # 如果找到了nvcc，尝试启用CUDA语言
    if(NVCC_EXECUTABLE)
        # 测试nvcc是否可以正常运行
        execute_process(
            COMMAND "${NVCC_EXECUTABLE}" --version
            OUTPUT_VARIABLE CUDA_VERSION_OUTPUT
            ERROR_VARIABLE CUDA_VERSION_ERROR
            RESULT_VARIABLE CUDA_VERSION_RESULT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )

        if(CUDA_VERSION_RESULT EQUAL 0)
            message(STATUS "CUDA compiler found: ${NVCC_EXECUTABLE}")
            message(STATUS "CUDA support will be enabled at build time")

            # 尝试获取CUDA工具包信息
            get_filename_component(CUDA_TOOLKIT_ROOT "${NVCC_EXECUTABLE}" DIRECTORY)
            get_filename_component(CUDA_TOOLKIT_ROOT "${CUDA_TOOLKIT_ROOT}" DIRECTORY)
            set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT}" CACHE PATH "CUDA Toolkit Root")

            # 设置CMAKE_CUDA_COMPILER以供backend使用
            set(CMAKE_CUDA_COMPILER "${NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA Compiler")

            message(STATUS "CUDA toolkit root: ${CUDA_TOOLKIT_ROOT}")

            # 设置cuDNN根目录（根据AI_EXP.md经验文档）
            set(CUDNN_ROOT "C:/Program Files/NVIDIA/CUDNN/v8.9.7" CACHE PATH "cuDNN root directory")
            message(STATUS "cuDNN root: ${CUDNN_ROOT}")

            set(TR_ENABLE_CUDA ON CACHE BOOL "Enable CUDA support" FORCE)
        else()
            message(WARNING "CUDA compiler found but not functional")
            message(STATUS "Error: ${CUDA_VERSION_ERROR}")
            message(STATUS "CUDA support will be disabled")
            set(TR_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support" FORCE)
        endif()
    else()
        message(STATUS "CUDA compiler not found - CUDA support will be disabled")
        message(STATUS "To enable CUDA:")
        message(STATUS "  1. Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads")
        message(STATUS "  2. Add CUDA's bin directory to your PATH environment variable")
        message(STATUS "  3. Or use -DCUDA_PATH=/path/to/cuda")
        set(TR_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support" FORCE)
    endif()
endif()

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

# 输出编译器信息
message(STATUS "编译器: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "编译器版本: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")



# 简化cuDNN配置：直接集成FindCudnnStatic逻辑
if(TR_ENABLE_CUDA)
    # 首先找到CUDAToolkit
    find_package(CUDAToolkit 12.0 QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDAToolkit found, configuring cuDNN")

        # 1. 允许用户通过 -DCUDNN_ROOT=xxx 覆盖
        set(CUDNN_ROOT "${CUDNN_ROOT}" CACHE PATH "cuDNN root directory")

        # 2. 查找cuDNN头文件
        find_path(CUDNN_INCLUDE_DIR cudnn.h
                  PATHS "${CUDNN_ROOT}/include"
                  NO_DEFAULT_PATH)
        # 再给一次系统机会（可选）
        find_path(CUDNN_INCLUDE_DIR cudnn.h)

        # 3. 查找cuDNN库文件（Windows 只认 .lib）
        find_library(CUDNN_LIBRARY cudnn
                     PATHS "${CUDNN_ROOT}/lib/x64"
                     NO_DEFAULT_PATH)
        find_library(CUDNN_LIBRARY cudnn)

        # 4. 创建cuDNN静态链接目标
        if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
            add_library(cudnn_static INTERFACE IMPORTED GLOBAL)
            target_include_directories(cudnn_static INTERFACE "${CUDNN_INCLUDE_DIR}")
            target_link_libraries(cudnn_static INTERFACE "${CUDNN_LIBRARY}")
            # 把 CUDA 头文件也带上，防止 nvcc 找不到 helper_cuda.h 之类
            target_link_libraries(cudnn_static INTERFACE CUDA::cudart CUDA::cublas)

            message(STATUS "[SUCCESS] cuDNN static target ready: ${CUDNN_LIBRARY}")
            message(STATUS "cuDNN include directory: ${CUDNN_INCLUDE_DIR}")
        else()
            message(WARNING "[FAIL] cuDNN not found:")
            if(NOT CUDNN_INCLUDE_DIR)
                message(WARNING "  - cuDNN headers not found (tried: ${CUDNN_ROOT}/include)")
            endif()
            if(NOT CUDNN_LIBRARY)
                message(WARNING "  - cuDNN library not found (tried: ${CUDNN_ROOT}/lib/x64)")
            endif()
            message(WARNING "  Please ensure cuDNN is installed and CUDNN_ROOT is set correctly")
            message(WARNING "  Example: -DCUDNN_ROOT=\"C:/Program Files/NVIDIA/CUDNN/v8.9.7\"")
        endif()
    else()
        message(WARNING "[FAIL] CUDAToolkit not found, static-link tests disabled")
    endif()
endif()

# 添加子目录
add_subdirectory(src)
add_subdirectory(tests)
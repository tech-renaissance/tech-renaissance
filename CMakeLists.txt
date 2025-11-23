cmake_minimum_required(VERSION 3.20)

# 设置项目
project(tech_renaissance VERSION 1.0.0 LANGUAGES CXX)

# 编译器检测 - 支持MSVC和GCC以实现跨平台
if(MSVC)
    message(STATUS "检测到Microsoft Visual C++编译器")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "检测到GCC编译器，版本: ${CMAKE_CXX_COMPILER_VERSION}")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.0")
        message(WARNING "GCC版本过低，建议使用GCC 8.0或更高版本")
    endif()
else()
    message(WARNING "检测到不支持的编译器: ${CMAKE_CXX_COMPILER_ID}")
    message(WARNING "建议使用MSVC或GCC 8.0+")
endif()

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

    # 添加UTF-8编码支持
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    message(STATUS "MSVC UTF-8 encoding support enabled for CXX")
    message(STATUS "MSVC PDB settings configured for parallel compilation")
endif()

# 全局编译选项
option(TR_USE_EIGEN "Enable Eigen for CPU optimizations (default: ON)" ON)
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

# 全局Eigen配置
if(TR_USE_EIGEN)
    # 设置Eigen库根目录
    set(EIGEN_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Eigen")

    # 检查Eigen库是否存在
    if(EXISTS "${EIGEN_ROOT_DIR}/Dense")
        set(EIGEN_FOUND TRUE)
        set(EIGEN_INCLUDE_DIR "${EIGEN_ROOT_DIR}")

        # 添加全局包含目录和定义
        include_directories(${EIGEN_INCLUDE_DIR})
        add_compile_definitions(TR_USE_EIGEN)

        message(STATUS "Eigen found: ${EIGEN_ROOT_DIR}")
        message(STATUS "Eigen optimizations: ENABLED for CPU backend")
    else()
        message(STATUS "Eigen not found at: ${EIGEN_ROOT_DIR}")
        message(STATUS "Eigen optimizations: DISABLED, using naive implementations")
        set(TR_USE_EIGEN OFF)
    endif()
else()
    message(STATUS "Eigen optimizations: DISABLED by user")
endif()

# CPU-only Core Module Design
# CUDA support removed for modular architecture and Linux portability
message(STATUS "Building CPU Core Module - CUDA support disabled")

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



# cuDNN and CUDA configurations removed
# Future modular integration will support GPU backends as optional modules

# 添加子目录
add_subdirectory(src)
add_subdirectory(tests)
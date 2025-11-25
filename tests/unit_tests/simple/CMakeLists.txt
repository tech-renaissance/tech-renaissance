# =============================================================================
# 技术觉醒框架 - Simple模块单元测试构建配置
# 版本: 2.4.0
# 日期: 2025-11-25
# 作者: 技术觉醒团队
# 说明: 使用SIMD库的图像处理测试
# =============================================================================

# 启用测试
enable_testing()

# 检查SIMD库 - 使用自动配置的路径
if(DEFINED SIMD_INCLUDE_DIR)
    set(SIMD_FOUND TRUE)
    set(SIMD_INCLUDE_DIR "${SIMD_INCLUDE_DIR}")
    message(STATUS "Using configured SIMD: ${SIMD_INCLUDE_DIR}")

    # 查找SIMD库文件
    if(WIN32)
        find_library(SIMD_LIBRARY
            NAMES simd
            PATHS ${SIMD_INCLUDE_DIR}/../lib
            PATH_SUFFIXES x64-windows x64-windows/lib
            NO_DEFAULT_PATH
        )
        if(NOT SIMD_LIBRARY)
            find_library(SIMD_LIBRARY
                NAMES simd
                PATHS ${SIMD_INCLUDE_DIR}/../..
                PATH_SUFFIXES lib x64-windows/lib
                NO_DEFAULT_PATH
            )
        endif()
    else()
        find_library(SIMD_LIBRARY
            NAMES simd Simd
            PATHS ${SIMD_INCLUDE_DIR}/../..
            PATH_SUFFIXES lib x64-linux/lib
        )
        # Also check vcpkg package directories if not found
        if(NOT SIMD_LIBRARY)
            find_library(SIMD_LIBRARY
                NAMES Simd
                PATHS /root/R/vcpkg-install-project/packages/simd_x64-linux/lib
                NO_DEFAULT_PATH
            )
        endif()
    endif()

    if(SIMD_LIBRARY)
        message(STATUS "Found SIMD library: ${SIMD_LIBRARY}")
    else()
        message(STATUS "SIMD library not found, using header-only mode")
    endif()
else()
    set(SIMD_FOUND FALSE)
    message(STATUS "SIMD library not configured")
endif()

# =============================================================================
# Simple模块测试配置函数
# =============================================================================
function(configure_simple_test test_name source_file)
    add_executable(${test_name} ${source_file})

    # 包含目录
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/stb  # STB库头文件
    )

    # SIMD库配置
    if(SIMD_FOUND)
        target_include_directories(${test_name} PRIVATE ${SIMD_INCLUDE_DIR})
        target_compile_definitions(${test_name} PRIVATE SIMD_FOUND=1)

        # 查找并链接SIMD库
        if(SIMD_LIBRARY)
            target_link_libraries(${test_name} PRIVATE ${SIMD_LIBRARY})

            # 处理动态链接库部署（Windows）
            if(WIN32 AND SIMD_LIBRARY MATCHES "\\.lib$")
                # 获取对应的DLL路径
                get_filename_component(SIMD_LIB_DIR ${SIMD_LIBRARY} DIRECTORY)
                get_filename_component(SIMD_ROOT_DIR ${SIMD_LIB_DIR} DIRECTORY)
                set(SIMD_DLL_PATH "${SIMD_ROOT_DIR}/bin/Simd.dll")

                if(EXISTS ${SIMD_DLL_PATH})
                    # 复制DLL到输出目录
                    add_custom_command(TARGET ${test_name} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${SIMD_DLL_PATH} $<TARGET_FILE_DIR:${test_name}>
                        COMMENT "Copying SIMD DLL to output directory"
                    )
                    message(STATUS "Found SIMD DLL: ${SIMD_DLL_PATH}")
                else()
                    message(WARNING "SIMD DLL not found at: ${SIMD_DLL_PATH}")
                endif()
            endif()
        endif()
    else()
        target_compile_definitions(${test_name} PRIVATE SIMD_FOUND=0)
    endif()

    # 定义PROJECT_ROOT_DIR和WORKSPACE_PATH宏指向项目目录
    target_compile_definitions(${test_name}
        PRIVATE
            PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
            WORKSPACE_PATH="${CMAKE_SOURCE_DIR}/workspace"
    )

    # 高性能编译优化
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        if(MSVC)
            target_compile_options(${test_name} PRIVATE /O2 /arch:AVX2)
        else()
            target_compile_options(${test_name} PRIVATE -O3 -march=native)
        endif()
    endif()

    # 设置标准属性
    set_target_properties(${test_name} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "tests/simple"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    message(STATUS "[SUCCESS] Simple test ${test_name} configured")
endfunction()

# =============================================================================
# Simple模块测试 - SIMD图像处理
# =============================================================================

# SIMD图像处理测试
configure_simple_test(test_simd test_simd.cpp)

# =============================================================================
# 注册测试到CTest
# =============================================================================
function(register_simple_test test_name test_description)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        TIMEOUT 60
        PASS_REGULAR_EXPRESSION "SUCCESS|completed successfully"
        LABELS "simple;simd;image_processing"
    )
endfunction()

# 注册所有Simple测试
register_simple_test(test_simd "SIMD Image Rotation Test")

message(STATUS "Simple tests configured successfully - ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "Total Simple tests: 1")
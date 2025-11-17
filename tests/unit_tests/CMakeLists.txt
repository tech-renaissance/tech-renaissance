# =============================================================================
# 技术觉醒框架 - 单元测试构建配置（优化版）
# 版本: 1.01.02
# 日期: 2025-11-03
# 作者: 技术觉醒团队
# 优化: 简化配置，使用tech_renaissance库的依赖传递机制
# =============================================================================

# 启用测试
enable_testing()

# 检查Eigen库
set(EIGEN_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/Eigen")
if(EXISTS "${EIGEN_ROOT_DIR}/Dense")
    set(EIGEN_FOUND TRUE)
    set(EIGEN_INCLUDE_DIR "${EIGEN_ROOT_DIR}")
    message(STATUS "Found Eigen library: ${EIGEN_INCLUDE_DIR}")
else()
    set(EIGEN_FOUND FALSE)
    message(WARNING "Eigen library not found at ${EIGEN_ROOT_DIR}")
endif()

# 查找OpenMP支持
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(OPENMP_AVAILABLE TRUE)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    set(OPENMP_AVAILABLE FALSE)
    message(WARNING "OpenMP not found, performance may be reduced")
endif()

# 检查CUDA支持（使用根目录的设置）
if(TR_ENABLE_CUDA)
    set(CUDA_BACKEND_ENABLED TRUE)
    message(STATUS "CUDA backend tests enabled (TR_ENABLE_CUDA=${TR_ENABLE_CUDA})")
else()
    set(CUDA_BACKEND_ENABLED FALSE)
    message(STATUS "CUDA backend tests disabled")
endif()

# =============================================================================
# 通用测试配置函数
# =============================================================================
function(configure_simple_test test_name source_file)
    add_executable(${test_name} ${source_file})

    # 链接主库 - 让tech_renaissance自动传递所有依赖
    if(WIN32)
        target_link_libraries(${test_name} PRIVATE tech_renaissance)
    else()
        target_link_libraries(${test_name} PRIVATE tech_renaissance pthread)
    endif()

    # 定义PROJECT_ROOT_DIR宏，指向项目根目录
    target_compile_definitions(${test_name}
        PUBLIC
            PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
    )

    # 设置标准属性
    set_target_properties(${test_name} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "tests"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    message(STATUS "[SUCCESS] ${test_name} configured successfully")
endfunction()

# =============================================================================
# 基础工具测试（简化版）
# =============================================================================
configure_simple_test(test_tr_exception test_tr_exception.cpp)
configure_simple_test(test_logger test_logger.cpp)
configure_simple_test(test_workspace test_workspace.cpp)

# =============================================================================
# 后端测试
# =============================================================================
configure_simple_test(test_cpu_backend test_cpu_backend.cpp)

if(CUDA_BACKEND_ENABLED)
    configure_simple_test(test_cuda_backend test_cuda_backend.cpp)
endif()

# =============================================================================
# 数据类型测试
# =============================================================================
configure_simple_test(test_dtype test_dtype.cpp)
configure_simple_test(test_device test_device.cpp)
configure_simple_test(test_storage test_storage.cpp)
configure_simple_test(test_shape test_shape.cpp)
configure_simple_test(test_tensor test_tensor.cpp)

# =============================================================================
# 张量-后端联合测试
# =============================================================================
configure_simple_test(test_tensor_backend test_tensor_backend.cpp)
configure_simple_test(test_copy test_copy.cpp)
configure_simple_test(test_is_close test_is_close.cpp)
configure_simple_test(test_print test_print.cpp)
configure_simple_test(test_tensor_io test_tensor_io.cpp)

# =============================================================================
# CPU操作测试
# =============================================================================
configure_simple_test(test_cpu_unary test_cpu_unary.cpp)
configure_simple_test(test_cpu_scalar test_cpu_scalar.cpp)
configure_simple_test(test_cpu_broadcast test_cpu_broadcast.cpp)
configure_simple_test(test_cpu_expand test_cpu_expand.cpp)
configure_simple_test(test_cpu_dimension test_cpu_dimension.cpp)
configure_simple_test(test_cpu_pad test_cpu_pad.cpp)
configure_simple_test(test_cpu_create test_cpu_create.cpp)
configure_simple_test(test_cpu_cast test_cpu_cast.cpp)
configure_simple_test(test_cpu_loss test_cpu_loss.cpp)

# =============================================================================
# CPU矩阵运算测试
# =============================================================================
# configure_simple_test(test_cpu_mm test_cpu_mm.cpp)  # 已删除

# =============================================================================
# CPU卷积测试（高性能优化配置）
# =============================================================================
if(EIGEN_FOUND)
    add_executable(test_cpu_conv test_cpu_conv.cpp)

    # 链接主库 - 让tech_renaissance自动传递所有依赖
    if(WIN32)
        target_link_libraries(test_cpu_conv PRIVATE tech_renaissance)
    else()
        target_link_libraries(test_cpu_conv PRIVATE tech_renaissance pthread)
    endif()

    # 包含头文件目录
    target_include_directories(test_cpu_conv
        PRIVATE
            ../../include
            ${EIGEN_INCLUDE_DIR}
            ${EIGEN_INCLUDE_DIR}/src
        )

    # 设置编译选项 - 启用高性能优化
    target_compile_definitions(test_cpu_conv PRIVATE
        EIGEN_DONT_PARALLELIZE  # 禁用自动并行化，使用OpenMP手动控制
    )

    # 启用编译器优化
    if(MSVC)
        # Visual Studio编译器优化 - 只在Release模式下启用高级优化
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(test_cpu_conv PRIVATE
                /O2              # 最高级优化
                /arch:AVX2        # 启用AVX2指令集
                /openmp          # 启用OpenMP支持
            )
        else()
            target_compile_options(test_cpu_conv PRIVATE
                /openmp          # 在Debug模式下只启用OpenMP
            )
        endif()
        # 链接OpenMP库
        if(OPENMP_AVAILABLE)
            target_link_libraries(test_cpu_conv PRIVATE OpenMP::OpenMP_CXX)
        endif()
    else()
        # GCC/Clang编译器优化
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(test_cpu_conv PRIVATE
                -O3              # 最高级优化
                -march=native    # 针对当前CPU架构优化
                -fopenmp         # 启用OpenMP
            )
        else()
            target_compile_options(test_cpu_conv PRIVATE
                -fopenmp         # 在Debug模式下只启用OpenMP
            )
        endif()
        if(OPENMP_AVAILABLE)
            target_link_libraries(test_cpu_conv PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()

    # 设置标准属性
    set_target_properties(test_cpu_conv PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "tests"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    message(STATUS "[SUCCESS] test_cpu_conv configured with high-performance optimization")
else()
    # 回退到简单配置
    configure_simple_test(test_cpu_conv test_cpu_conv.cpp)
    message(WARNING "[WARNING] Eigen not found, using simple configuration for test_cpu_conv")
endif()

# configure_simple_test(test_cpu_gemm test_cpu_gemm.cpp)  # 已删除

# =============================================================================
# CPU切片测试
# =============================================================================
configure_simple_test(test_cpu_slice test_cpu_slice.cpp)

# =============================================================================
# CPU池化测试
# =============================================================================
configure_simple_test(test_cpu_pooling test_cpu_pooling.cpp)

# =============================================================================
# CPU卷积新测试
# =============================================================================
configure_simple_test(test_cpu_conv_new test_cpu_conv_new.cpp)

# =============================================================================
# CPU卷积扩展测试
# =============================================================================
configure_simple_test(test_cpu_conv_extra test_cpu_conv_extra.cpp)

# =============================================================================
# CPU卷积最终测试（高性能优化配置）
# =============================================================================
if(EIGEN_FOUND)
    add_executable(test_cpu_conv_final test_cpu_conv_final.cpp)

    # 链接主库 - 让tech_renaissance自动传递所有依赖
    if(WIN32)
        target_link_libraries(test_cpu_conv_final PRIVATE tech_renaissance)
    else()
        target_link_libraries(test_cpu_conv_final PRIVATE tech_renaissance pthread)
    endif()

    # 定义PROJECT_ROOT_DIR宏，指向项目根目录
    target_compile_definitions(test_cpu_conv_final
        PUBLIC
            PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
            EIGEN_DONT_PARALLELIZE  # 禁用自动并行化，使用OpenMP手动控制
    )

    # 设置编译选项 - 启用高性能优化
    if(MSVC)
        # Visual Studio编译器优化 - 只在Release模式下启用高级优化
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(test_cpu_conv_final PRIVATE
                /O2              # 最高级优化
                /arch:AVX2        # 启用AVX2指令集
                /openmp          # 启用OpenMP支持
            )
        else()
            target_compile_options(test_cpu_conv_final PRIVATE
                /openmp          # 在Debug模式下只启用OpenMP
            )
        endif()
        # 链接OpenMP库
        if(OPENMP_AVAILABLE)
            target_link_libraries(test_cpu_conv_final PRIVATE OpenMP::OpenMP_CXX)
        endif()
    else()
        # GCC/Clang编译器优化
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(test_cpu_conv_final PRIVATE
                -O3              # 最高级优化
                -march=native    # 针对当前CPU架构优化
                -fopenmp         # 启用OpenMP
            )
        else()
            target_compile_options(test_cpu_conv_final PRIVATE
                -fopenmp         # 在Debug模式下只启用OpenMP
            )
        endif()
        if(OPENMP_AVAILABLE)
            target_link_libraries(test_cpu_conv_final PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()

    # 设置标准属性
    set_target_properties(test_cpu_conv_final PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "tests"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    message(STATUS "[SUCCESS] test_cpu_conv_final configured with high-performance optimization")
else()
    # 回退到简单配置
    configure_simple_test(test_cpu_conv_final test_cpu_conv_final.cpp)
    message(WARNING "[WARNING] Eigen not found, using simple configuration for test_cpu_conv_final")
endif()

# =============================================================================
# CPU矩阵乘法最终测试（高性能优化配置）
# =============================================================================
if(EIGEN_FOUND)
    add_executable(test_cpu_mm_final test_cpu_mm_final.cpp)

    # 链接主库 - 让tech_renaissance自动传递所有依赖
    if(WIN32)
        target_link_libraries(test_cpu_mm_final PRIVATE tech_renaissance)
    else()
        target_link_libraries(test_cpu_mm_final PRIVATE tech_renaissance pthread)
    endif()

    # 定义PROJECT_ROOT_DIR宏，指向项目根目录
    target_compile_definitions(test_cpu_mm_final
        PUBLIC
            PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
            EIGEN_DONT_PARALLELIZE  # 禁用自动并行化，使用OpenMP手动控制
    )

    # 设置编译选项 - 启用高性能优化
    if(MSVC)
        # Visual Studio编译器优化 - 只在Release模式下启用高级优化
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(test_cpu_mm_final PRIVATE
                /O2              # 最高级优化
                /arch:AVX2        # 启用AVX2指令集
                /openmp          # 启用OpenMP支持
            )
        else()
            target_compile_options(test_cpu_mm_final PRIVATE
                /openmp          # 在Debug模式下只启用OpenMP
            )
        endif()
        # 链接OpenMP库
        if(OPENMP_AVAILABLE)
            target_link_libraries(test_cpu_mm_final PRIVATE OpenMP::OpenMP_CXX)
        endif()
    else()
        # GCC/Clang编译器优化
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(test_cpu_mm_final PRIVATE
                -O3              # 最高级优化
                -march=native    # 针对当前CPU架构优化
                -fopenmp         # 启用OpenMP
            )
        else()
            target_compile_options(test_cpu_mm_final PRIVATE
                -fopenmp         # 在Debug模式下只启用OpenMP
            )
        endif()
        if(OPENMP_AVAILABLE)
            target_link_libraries(test_cpu_mm_final PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()

    # 设置标准属性
    set_target_properties(test_cpu_mm_final PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "tests"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    message(STATUS "[SUCCESS] test_cpu_mm_final configured with high-performance optimization")
else()
    # 回退到简单配置
    configure_simple_test(test_cpu_mm_final test_cpu_mm_final.cpp)
    message(WARNING "[WARNING] Eigen not found, using simple configuration for test_cpu_mm_final")
endif()


# =============================================================================
# 简单算法测试
# =============================================================================
configure_simple_test(test_naive_conv test_naive_conv.cpp)
configure_simple_test(test_naive_gemm test_naive_gemm.cpp)

# =============================================================================
# 高级功能测试
# =============================================================================
# configure_simple_test(test_new_api test_new_api.cpp)  # 已删除

# =============================================================================
# 内存管理测试
# =============================================================================
configure_simple_test(test_memory_occupation test_memory_occupation.cpp)
configure_simple_test(test_get_memory_size test_get_memory_size.cpp)

# =============================================================================
# 元素访问测试
# =============================================================================
configure_simple_test(test_item_access test_item_access.cpp)

# =============================================================================
# CPU维度操作测试
# =============================================================================
configure_simple_test(test_cpu_dim_ops test_cpu_dim_ops.cpp)

# =============================================================================
# Python接口测试（如果启用）
# =============================================================================
if(TR_BUILD_PYTHON_SESSION)
    configure_simple_test(test_python_session test_python_session.cpp)

    if(TR_PYTHON_MNIST_TEST)
        configure_simple_test(test_pytorch_data test_pytorch_data.cpp)
        configure_simple_test(test_pytorch_mnist test_pytorch_mnist.cpp)
    endif()
endif()

# =============================================================================
# 注册所有测试到CTest
# =============================================================================
function(register_test test_name test_description)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        TIMEOUT 60
        PASS_REGULAR_EXPRESSION "PASSED|SUCCESS|completed successfully"
    )
endfunction()

# 注册基础测试
register_test(test_tr_exception "TR Exception Test")
register_test(test_logger "Logger Test")
register_test(test_workspace "Workspace Test")

# 注册后端测试
register_test(test_cpu_backend "CPU Backend Test")
if(CUDA_BACKEND_ENABLED)
    register_test(test_cuda_backend "CUDA Backend Test")
endif()

# 注册数据类型测试
register_test(test_dtype "DType Test")
register_test(test_device "Device Test")
register_test(test_storage "Storage Test")
register_test(test_shape "Shape Test")
register_test(test_tensor "Tensor Test")

# 注册联合测试
register_test(test_tensor_backend "Tensor-Backend Joint Test")
register_test(test_copy "Copy Test")
register_test(test_is_close "Is Close Test")
register_test(test_print "Print Test")
register_test(test_tensor_io "Tensor I/O Test")

# 注册CPU操作测试
register_test(test_cpu_unary "CPU Unary Test")
register_test(test_cpu_scalar "CPU Scalar Test")
register_test(test_cpu_broadcast "CPU Broadcast Test")
register_test(test_cpu_expand "CPU Expand Test")
register_test(test_cpu_dimension "CPU Dimension Test")
register_test(test_cpu_pad "CPU Pad Test")
register_test(test_cpu_create "CPU Create Test")
register_test(test_cpu_cast "CPU Cast Test")
register_test(test_cpu_loss "CPU Loss Test")

# 注册矩阵运算测试
# register_test(test_cpu_mm "CPU Matrix Multiplication Test")  # 已删除

# 注册卷积测试
register_test(test_cpu_conv "CPU Convolution Test")
# register_test(test_cpu_gemm "CPU GEMM Test")  # 已删除

# 注册切片测试
register_test(test_cpu_slice "CPU Slice Test")

# 注册池化测试
register_test(test_cpu_pooling "CPU Pooling Test")

# 注册卷积新测试
register_test(test_cpu_conv_new "CPU Convolution New Test")

# 注册卷积扩展测试
register_test(test_cpu_conv_extra "CPU Convolution Extra Test")

# 注册卷积最终测试
register_test(test_cpu_conv_final "CPU Convolution Final Test")

# 注册矩阵乘法最终测试
register_test(test_cpu_mm_final "CPU Matrix Multiplication Final Test")

# =============================================================================
# CUDA矩阵乘法最终测试（高性能优化配置）
# =============================================================================
if(TR_ENABLE_CUDA)
    if(EIGEN_FOUND)
        add_executable(test_cuda_mm_final test_cuda_mm_final.cpp)

        # 链接主库 - 让tech_renaissance自动传递所有依赖
        if(WIN32)
            target_link_libraries(test_cuda_mm_final PRIVATE tech_renaissance)
        else()
            target_link_libraries(test_cuda_mm_final PRIVATE tech_renaissance pthread)
        endif()

        # 定义PROJECT_ROOT_DIR宏，指向项目根目录
        target_compile_definitions(test_cuda_mm_final
            PUBLIC
                PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
                EIGEN_DONT_PARALLELIZE  # 禁用自动并行化，使用OpenMP手动控制
        )

        # 设置编译选项 - 启用高性能优化
        if(MSVC)
            # Visual Studio编译器优化 - 只在Release模式下启用高级优化
            if(CMAKE_BUILD_TYPE STREQUAL "Release")
                target_compile_options(test_cuda_mm_final PRIVATE
                    /O2              # 最高级优化
                    /arch:AVX2        # 启用AVX2指令集
                    /openmp          # 启用OpenMP支持
                )
            else()
                target_compile_options(test_cuda_mm_final PRIVATE
                    /openmp          # 在Debug模式下只启用OpenMP
                )
            endif()
            # 链接OpenMP库
            if(OPENMP_AVAILABLE)
                target_link_libraries(test_cuda_mm_final PRIVATE OpenMP::OpenMP_CXX)
            endif()
        else()
            # GCC/Clang编译器优化
            if(CMAKE_BUILD_TYPE STREQUAL "Release")
                target_compile_options(test_cuda_mm_final PRIVATE
                    -O3              # 最高级优化
                    -march=native    # 针对当前CPU架构优化
                    -fopenmp         # 启用OpenMP
                )
            else()
                target_compile_options(test_cuda_mm_final PRIVATE
                    -fopenmp         # 在Debug模式下只启用OpenMP
                )
            endif()
            if(OPENMP_AVAILABLE)
                target_link_libraries(test_cuda_mm_final PRIVATE OpenMP::OpenMP_CXX)
            endif()
        endif()

        # 设置标准属性
        set_target_properties(test_cuda_mm_final PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
            FOLDER "tests"
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
        )

        message(STATUS "[SUCCESS] test_cuda_mm_final configured with high-performance optimization")
    else()
        # 回退到简单配置
        configure_simple_test(test_cuda_mm_final test_cuda_mm_final.cpp)
        message(WARNING "[WARNING] Eigen not found, using simple configuration for test_cuda_mm_final")
    endif()

    # =============================================================================
    # CUDA卷积最终测试（高性能优化配置）
    # =============================================================================
    if(EIGEN_FOUND)
        add_executable(test_cuda_conv_final test_cuda_conv_final.cpp)

        # 链接主库 - 让tech_renaissance自动传递所有依赖
        if(WIN32)
            target_link_libraries(test_cuda_conv_final PRIVATE tech_renaissance)
        else()
            target_link_libraries(test_cuda_conv_final PRIVATE tech_renaissance pthread)
        endif()

        # 包含头文件目录
        target_include_directories(test_cuda_conv_final
            PRIVATE
                ../../include
                ${EIGEN_INCLUDE_DIR}
                ${EIGEN_INCLUDE_DIR}/src
        )

        # 设置编译选项 - 启用高性能优化
        target_compile_definitions(test_cuda_conv_final PRIVATE
            EIGEN_DONT_PARALLELIZE  # 禁用自动并行化，使用OpenMP手动控制
        )

        # 启用编译器优化
        if(MSVC)
            # Visual Studio编译器优化 - 只在Release模式下启用高级优化
            if(CMAKE_BUILD_TYPE STREQUAL "Release")
                target_compile_options(test_cuda_conv_final PRIVATE
                    /O2              # 最高级优化
                    /arch:AVX2        # 启用AVX2指令集
                    /openmp          # 启用OpenMP支持
                )
            else()
                target_compile_options(test_cuda_conv_final PRIVATE
                    /openmp          # 在Debug模式下只启用OpenMP
                )
            endif()
        endif()

        # 设置标准属性
        set_target_properties(test_cuda_conv_final PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
            FOLDER "tests"
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
        )

        message(STATUS "[SUCCESS] test_cuda_conv_final configured with high-performance optimization")
    else()
        # 回退到简单配置
        configure_simple_test(test_cuda_conv_final test_cuda_conv_final.cpp)
        message(WARNING "[WARNING] Eigen not found, using simple configuration for test_cuda_conv_final")
    endif()

    # 转置卷积最终版本测试
    if(EIGEN_FOUND)
        add_executable(test_cuda_tconv_final test_cuda_tconv_final.cpp)

        # 链接主库 - 让tech_renaissance自动传递所有依赖
        if(WIN32)
            target_link_libraries(test_cuda_tconv_final PRIVATE tech_renaissance)
        else()
            target_link_libraries(test_cuda_tconv_final PRIVATE tech_renaissance pthread)
        endif()

        # 包含头文件目录
        target_include_directories(test_cuda_tconv_final
            PRIVATE
                ../../include
                ${EIGEN_INCLUDE_DIR}
                ${EIGEN_INCLUDE_DIR}/src
        )

        # 设置编译选项 - 启用高性能优化
        target_compile_definitions(test_cuda_tconv_final PRIVATE
            EIGEN_DONT_PARALLELIZE  # 禁用自动并行化，使用OpenMP手动控制
        )

        # 启用编译器优化
        if(MSVC)
            # Visual Studio编译器优化 - 只在Release模式下启用高级优化
            if(CMAKE_BUILD_TYPE STREQUAL "Release")
                target_compile_options(test_cuda_tconv_final PRIVATE
                    /O2              # 最高级优化
                    /arch:AVX2        # 启用AVX2指令集
                    /openmp          # 启用OpenMP支持
                )
            else()
                target_compile_options(test_cuda_tconv_final PRIVATE
                    /openmp          # 在Debug模式下只启用OpenMP
                )
            endif()
        endif()

        # 设置标准属性
        set_target_properties(test_cuda_tconv_final PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
            FOLDER "tests"
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
        )

        message(STATUS "[SUCCESS] test_cuda_tconv_final configured with high-performance optimization")
    else()
        # 回退到简单配置
        configure_simple_test(test_cuda_tconv_final test_cuda_tconv_final.cpp)
        message(WARNING "[WARNING] Eigen not found, using simple configuration for test_cuda_tconv_final")
    endif()

    # 性能测试汇总
    if(EIGEN_FOUND)
        add_executable(test_performance test_performance.cpp)

        # 链接主库 - 让tech_renaissance自动传递所有依赖
        if(WIN32)
            target_link_libraries(test_performance PRIVATE tech_renaissance)
        else()
            target_link_libraries(test_performance PRIVATE tech_renaissance pthread)
        endif()

        # 包含头文件目录
        target_include_directories(test_performance
                PRIVATE
                ../../include
                ${EIGEN_INCLUDE_DIR}
                ${EIGEN_INCLUDE_DIR}/src
        )

        # 设置编译选项 - 启用高性能优化
        target_compile_definitions(test_performance PRIVATE
                EIGEN_DONT_PARALLELIZE  # 禁用自动并行化，使用OpenMP手动控制
        )

        # 启用编译器优化
        if(MSVC)
            # Visual Studio编译器优化 - 只在Release模式下启用高级优化
            if(CMAKE_BUILD_TYPE STREQUAL "Release")
                target_compile_options(test_performance PRIVATE
                        /O2              # 最高级优化
                        /arch:AVX2        # 启用AVX2指令集
                        /openmp          # 启用OpenMP支持
                )
            else()
                target_compile_options(test_performance PRIVATE
                        /openmp          # 在Debug模式下只启用OpenMP
                )
            endif()
        endif()

        # 设置标准属性
        set_target_properties(test_performance PROPERTIES
                CXX_STANDARD 17
                CXX_STANDARD_REQUIRED ON
                CXX_EXTENSIONS OFF
                FOLDER "tests"
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
        )

        message(STATUS "[SUCCESS] test_performance configured with high-performance optimization")
    else()
        # 回退到简单配置
        configure_simple_test(test_performance test_performance.cpp)
        message(WARNING "[WARNING] Eigen not found, using simple configuration for test_performance")
    endif()
else()
    message(STATUS "[INFO] CUDA backend tests disabled (TR_ENABLE_CUDA=OFF)")
endif()

# 注册简单算法测试
register_test(test_naive_conv "Naive Convolution Test")
register_test(test_naive_gemm "Naive GEMM Test")

# 注册高级功能测试
# register_test(test_new_api "New API Test")  # 已删除

# 注册内存管理测试
register_test(test_memory_occupation "Memory Occupation Test")
register_test(test_get_memory_size "Get Memory Size Test")

# 注册视图功能测试
configure_simple_test(test_view test_view.cpp)
register_test(test_view "View Functionality Test")

# 注册元素访问测试
register_test(test_item_access "Item Access Test")

# 注册CPU维度操作测试
register_test(test_cpu_dim_ops "CPU Dimension Operations Test")

# 注册Python接口测试
if(TR_BUILD_PYTHON_SESSION)
    register_test(test_python_session "Python Session Test")

    if(TR_PYTHON_MNIST_TEST)
        register_test(test_pytorch_data "PyTorch Data Test")
        register_test(test_pytorch_mnist "PyTorch MNIST Test")
    endif()
endif()

if(TR_BUILD_PYTHON_SESSION)
    configure_simple_test(test_mlp test_mlp.cpp)
    register_test(test_mlp "MLP Test")

    configure_simple_test(test_mlp_module test_mlp_module.cpp)
    register_test(test_mlp_module "MLP Module Test")
endif()

# =============================================================================
# Module测试（新增）
# =============================================================================
configure_simple_test(test_module_gradient test_module_gradient.cpp)
register_test(test_module_gradient "Module Gradient Check Test")

configure_simple_test(test_memory_allocation test_memory_allocation.cpp)
register_test(test_memory_allocation "Memory Allocation Verification Test")

# =============================================================================
# Model测试（新增）
# =============================================================================
configure_simple_test(test_model test_model.cpp)
register_test(test_model "Model Class Test")

message(STATUS "All tests configured successfully (optimized version)")


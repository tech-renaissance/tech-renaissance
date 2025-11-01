# =============================================================================
# 技术觉醒框架 - 单元测试构建配置
# 版本: 1.01.01
# 日期: 2025-10-24
# 作者: 技术觉醒团队
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

# 收集基础测试源文件
set(TEST_SOURCES
    test_tr_exception.cpp
    test_logger.cpp
    test_workspace.cpp
)

# 收集后端测试源文件
set(BACKEND_TEST_SOURCES
    test_cpu_backend.cpp
    test_cuda_backend.cpp
)

# 收集所有源文件（用于后端测试）
set(BACKEND_SOURCES
    ../../src/utils/tr_exception.cpp
    ../../src/utils/logger.cpp
    ../../src/backend/backend.cpp
    ../../src/backend/backend_manager.cpp
    ../../src/backend/cpu_backend.cpp
)

# 检查CUDA支持（使用根目录的设置）
if(TR_ENABLE_CUDA)
    list(APPEND BACKEND_SOURCES ../../src/backend/cuda_backend.cu)
    set(CUDA_BACKEND_ENABLED TRUE)
    message(STATUS "CUDA backend tests enabled (TR_ENABLE_CUDA=${TR_ENABLE_CUDA})")
else()
    set(CUDA_BACKEND_ENABLED FALSE)
    message(STATUS "CUDA not found - CUDA backend tests will be skipped")
endif()

# 为基础测试创建可执行目标
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})

    # 链接基础源文件
    target_sources(${test_name} PRIVATE
        ../../src/utils/tr_exception.cpp
        ../../src/utils/logger.cpp
    )

    # 包含头文件目录
    target_include_directories(${test_name}
        PRIVATE
            ../../include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
    )

    # 设置C++标准
    set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${test_name} PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET ${test_name} PROPERTY CXX_EXTENSIONS OFF)
    set_property(TARGET ${test_name} PROPERTY FOLDER "tests")

    # 链接必要的系统库
    if(WIN32)
        target_link_libraries(${test_name} PRIVATE)
    else()
        target_link_libraries(${test_name} PRIVATE pthread)
    endif()

    # 设置输出目录
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    # 注册测试
    add_test(NAME ${test_name} COMMAND ${test_name})

    # 设置测试属性
    set_tests_properties(${test_name} PROPERTIES
        RUN_SERIAL FALSE  # 允许并行运行测试
        TIMEOUT 30       # 设置超时时间为30秒
    )

    message(STATUS "Test ${test_name} configured successfully")
endforeach()


# CPU后端测试
add_executable(test_cpu_backend
    test_cpu_backend.cpp
)

target_include_directories(test_cpu_backend
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 链接系统库
if(WIN32)
    target_link_libraries(test_cpu_backend PRIVATE tech_renaissance)
else()
    target_link_libraries(test_cpu_backend PRIVATE tech_renaissance pthread)
endif()

# 链接CUDA库（如果可用）
if(CUDA_BACKEND_ENABLED)
    # 使用根目录创建的cudnn_static目标
    if(TARGET cudnn_static)
        target_link_libraries(test_cpu_backend PRIVATE cudnn_static)
        target_compile_definitions(test_cpu_backend PRIVATE TR_ENABLE_CUDA)
        message(STATUS "[SUCCESS] Using cudnn_static target for test_cpu_backend")
    else()
        message(WARNING "[FAIL] cudnn_static target not available, CUDA tests disabled")
    endif()
endif()

# 设置编译选项
target_compile_definitions(test_cpu_backend PRIVATE
    TR_TEST_VERSION_MAJOR=1
    TR_TEST_VERSION_MINOR=01
    TR_TEST_VERSION_PATCH=01
)

set_target_properties(test_cpu_backend PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# CPU单目运算测试
add_executable(test_cpu_unary
    test_cpu_unary.cpp
)

target_include_directories(test_cpu_unary
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 链接系统库
if(WIN32)
    target_link_libraries(test_cpu_unary PRIVATE tech_renaissance)
else()
    target_link_libraries(test_cpu_unary PRIVATE tech_renaissance pthread)
endif()

if(CUDA_BACKEND_ENABLED)
    target_link_libraries(test_cpu_unary PRIVATE cudnn_static)
    target_compile_definitions(test_cpu_unary PRIVATE TR_ENABLE_CUDA)
    message(STATUS "[SUCCESS] Using cudnn_static target for test_cpu_unary")
endif()

target_compile_definitions(test_cpu_unary PRIVATE
    TR_USE_EIGEN=${TR_USE_EIGEN}
)

set_target_properties(test_cpu_unary PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 张量复制功能测试
add_executable(test_copy
    test_copy.cpp
)

target_include_directories(test_copy
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 链接系统库
if(WIN32)
    target_link_libraries(test_copy PRIVATE tech_renaissance)
else()
    target_link_libraries(test_copy PRIVATE tech_renaissance pthread)
endif()

if(CUDA_BACKEND_ENABLED)
    target_link_libraries(test_copy PRIVATE cudnn_static)
    target_compile_definitions(test_copy PRIVATE TR_ENABLE_CUDA)
    message(STATUS "[SUCCESS] Using cudnn_static target for test_copy")
endif()

target_compile_definitions(test_copy PRIVATE
    TR_USE_EIGEN=${TR_USE_EIGEN}
)

set_target_properties(test_copy PROPERTIES
    CXX_STANDARD 17
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# CUDA后端测试（仅在有CUDA时编译）
if(CUDA_BACKEND_ENABLED)
    add_executable(test_cuda_backend
        test_cuda_backend.cpp
    )

    target_include_directories(test_cuda_backend
        PRIVATE
            ../../include
            ${CUDNN_INCLUDE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
    )

    if(WIN32)
        target_link_libraries(test_cuda_backend PRIVATE tech_renaissance)
    else()
        target_link_libraries(test_cuda_backend PRIVATE tech_renaissance pthread)
    endif()

    target_compile_definitions(test_cuda_backend PRIVATE
        TR_USE_CUDA
        TR_ENABLE_CUDA
        TR_TEST_VERSION_MAJOR=1
        TR_TEST_VERSION_MINOR=01
        TR_TEST_VERSION_PATCH=01
    )

    set_target_properties(test_cuda_backend PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "tests"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    # 添加测试到CTest
    add_test(NAME CUDABackendTest COMMAND test_cuda_backend)
    add_test(NAME CUDABackendTestWorkingDir COMMAND test_cuda_backend
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    # 设置测试属性
    set_tests_properties(CUDABackendTest PROPERTIES
        TIMEOUT 120
        PASS_REGULAR_EXPRESSION "SUCCESS"
    )

    message(STATUS "CUDA backend tests configured successfully")
endif()

# 添加后端测试到CTest
add_test(NAME CPUBackendTest COMMAND test_cpu_backend)
add_test(NAME CPUBackendTestWorkingDir COMMAND test_cpu_backend
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# 设置测试属性
set_tests_properties(CPUBackendTest PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION "SUCCESS"
)

# Expert A解决方案：添加4种卷积方案测试的公共函数
if(CUDA_BACKEND_ENABLED)
    # 检查是否可以使用静态链接
    if(TARGET cudnn_static)
        message(STATUS "[SUCCESS] Using cudnn_static target for 4 convolution solutions")

        # Expert A的add_static_cuda_test函数
        function(add_static_cuda_test target_name cpp_file)
            if(NOT TARGET cudnn_static)
                return()   # 没找齐就不生成，不报错
            endif()

            add_executable(${target_name} ${cpp_file})

            # 关键：把 CUDA 语言级别打开，MSVC 才会自动加 /utf-8 并找到 limits.h
            set_property(TARGET ${target_name} PROPERTY CUDA_RUNTIME_LIBRARY static)
            # 注意：不在这里调用enable_language(CUDA)，因为父级已经启用了

            target_link_libraries(${target_name}
                PRIVATE
                    tech_renaissance      # 我们自己的核心库
                    cudnn_static)         # 上面创建的 IMPORTED 目标

            # 包含头文件目录
            target_include_directories(${target_name}
                PRIVATE
                    ../../include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
            )

            # 定义PROJECT_ROOT_DIR宏
            target_compile_definitions(${target_name}
                PRIVATE
                    PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
            )

            # 设置编译属性
            set_target_properties(${target_name} PROPERTIES
                CXX_STANDARD 17
                CXX_STANDARD_REQUIRED ON
                CXX_EXTENSIONS OFF
                FOLDER "tests"
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
            )

            # 把测试注册到 CTest
            add_test(NAME ${target_name} COMMAND ${target_name})
            set_tests_properties(${target_name} PROPERTIES TIMEOUT 120)
        endfunction()

        # 添加卷积和矩阵乘法测试
        add_static_cuda_test(test_cuda_conv test_cuda_conv.cpp)
        # 添加GEMM方案测试（统一标准版本）
        add_static_cuda_test(test_cuda_gemm test_cuda_gemm.cpp)

        message(STATUS "[SUCCESS] 2 tests (1 convolution + 1 GEMM) configured successfully with static linking")

    else()
        message(WARNING "[FAIL] cudnn_static target not available, static-link tests disabled")
    endif()
endif()

# CPU卷积测试 (基于Eigen的方案D)
if(EIGEN_FOUND)
    add_executable(test_cpu_conv test_cpu_conv.cpp)

    # 包含头文件目录
    target_include_directories(test_cpu_conv
        PRIVATE
            ../../include
            ${EIGEN_INCLUDE_DIR}
            ${EIGEN_INCLUDE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
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
        target_link_libraries(test_cpu_conv PRIVATE OpenMP::OpenMP_CXX)
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
        target_link_libraries(test_cpu_conv PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # 链接库
    if(WIN32)
        target_link_libraries(test_cpu_conv PRIVATE tech_renaissance)
    else()
        target_link_libraries(test_cpu_conv PRIVATE tech_renaissance pthread)
    endif()

    # 设置编译属性
    set_target_properties(test_cpu_conv PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "tests"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    # 注册测试到CTest
    add_test(NAME CPUConvTest COMMAND test_cpu_conv)
    set_tests_properties(CPUConvTest PROPERTIES
        TIMEOUT 60
        PASS_REGULAR_EXPRESSION "SUCCESS"
    )

    message(STATUS "[SUCCESS] CPU convolution test (Eigen-based) configured successfully")
else()
    message(WARNING "[FAIL] Eigen not found, CPU convolution test disabled")
endif()

# 查找OpenMP支持
find_package(OpenMP)

# CPU矩阵乘法测试 (基于Eigen库)
if(EIGEN_FOUND)
    add_executable(test_cpu_gemm test_cpu_gemm.cpp)

    # 包含头文件目录
    target_include_directories(test_cpu_gemm
        PRIVATE
            ../../include
            ${EIGEN_INCLUDE_DIR}
            ${EIGEN_INCLUDE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
    )

    # 设置编译选项 - 启用高性能优化
    target_compile_definitions(test_cpu_gemm PRIVATE
        EIGEN_DONT_PARALLELIZE  # 禁用自动并行化，使用OpenMP手动控制
    )

    # 启用编译器优化
    if(MSVC)
        # Visual Studio编译器优化 - 只在Release模式下启用高级优化
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(test_cpu_gemm PRIVATE
                /O2              # 最高级优化
                /arch:AVX2        # 启用AVX2指令集
                /openmp          # 启用OpenMP支持
            )
        else()
            target_compile_options(test_cpu_gemm PRIVATE
                /openmp          # 在Debug模式下只启用OpenMP
            )
        endif()
        # 链接OpenMP库
        target_link_libraries(test_cpu_gemm PRIVATE OpenMP::OpenMP_CXX)
    else()
        # GCC/Clang编译器优化
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(test_cpu_gemm PRIVATE
                -O3              # 最高级优化
                -march=native    # 针对当前CPU架构优化
                -fopenmp         # 启用OpenMP
            )
        else()
            target_compile_options(test_cpu_gemm PRIVATE
                -fopenmp         # 在Debug模式下只启用OpenMP
            )
        endif()
        target_link_libraries(test_cpu_gemm PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # 链接库
    if(WIN32)
        target_link_libraries(test_cpu_gemm PRIVATE tech_renaissance)
    else()
        target_link_libraries(test_cpu_gemm PRIVATE tech_renaissance pthread)
    endif()

    # 设置编译属性
    set_target_properties(test_cpu_gemm PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "tests"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    # 注册测试到CTest
    add_test(NAME CPUGEMMTest COMMAND test_cpu_gemm)
    set_tests_properties(CPUGEMMTest PROPERTIES
        TIMEOUT 120  # 矩阵乘法计算量较大，设置更长的超时时间
        PASS_REGULAR_EXPRESSION "SUCCESS"
    )

    message(STATUS "[SUCCESS] CPU GEMM test (Eigen-based) configured successfully")
else()
    message(WARNING "[FAIL] Eigen not found, CPU GEMM test disabled")
endif()

# CPU朴素卷积测试 (基于C++标准库)
add_executable(test_naive_conv test_naive_conv.cpp)

# 包含头文件目录
target_include_directories(test_naive_conv
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 启用编译器优化 (根据ENABLE_OPENMP选项配置)
if(MSVC)
    # Visual Studio编译器优化 - 只在Release模式下启用高级优化
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(test_naive_conv PRIVATE
            /O2              # 最高级优化
            /arch:AVX2        # 启用AVX2指令集
        )
        if(ENABLE_OPENMP)
            target_compile_options(test_naive_conv PRIVATE /openmp)
        endif()
    else()
        if(ENABLE_OPENMP)
            target_compile_options(test_naive_conv PRIVATE /openmp)
        endif()
    endif()
    # 链接OpenMP库（如果启用）
    if(ENABLE_OPENMP)
        target_link_libraries(test_naive_conv PRIVATE OpenMP::OpenMP_CXX)
    endif()
else()
    # GCC/Clang编译器优化
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(test_naive_conv PRIVATE
            -O3              # 最高级优化
            -march=native    # 针对当前CPU架构优化
        )
        if(ENABLE_OPENMP)
            target_compile_options(test_naive_conv PRIVATE -fopenmp)
        endif()
    else()
        if(ENABLE_OPENMP)
            target_compile_options(test_naive_conv PRIVATE -fopenmp)
        endif()
    endif()
    # 链接OpenMP库（如果启用）
    if(ENABLE_OPENMP)
        target_link_libraries(test_naive_conv PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# 链接库 (naive实现是standalone的，不需要框架库)
if(WIN32)
    # target_link_libraries(test_naive_conv PRIVATE)  # 注释掉，不需要框架库
else()
    # target_link_libraries(test_naive_conv PRIVATE pthread)  # 注释掉，不需要框架库
endif()

# 设置编译属性
set_target_properties(test_naive_conv PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 注册测试到CTest
add_test(NAME NaiveConvTest COMMAND test_naive_conv)
set_tests_properties(NaiveConvTest PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION "SUCCESS"
)

message(STATUS "[SUCCESS] CPU naive convolution test configured successfully")

# CPU朴素矩阵乘法测试 (基于C++标准库)
add_executable(test_naive_gemm test_naive_gemm.cpp)

# 包含头文件目录
target_include_directories(test_naive_gemm
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 启用编译器优化 (根据ENABLE_OPENMP选项配置)
if(MSVC)
    # Visual Studio编译器优化 - 只在Release模式下启用高级优化
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(test_naive_gemm PRIVATE
            /O2              # 最高级优化
            /arch:AVX2        # 启用AVX2指令集
        )
        if(ENABLE_OPENMP)
            target_compile_options(test_naive_gemm PRIVATE /openmp)
        endif()
    else()
        if(ENABLE_OPENMP)
            target_compile_options(test_naive_gemm PRIVATE /openmp)
        endif()
    endif()
    # 链接OpenMP库（如果启用）
    if(ENABLE_OPENMP)
        target_link_libraries(test_naive_gemm PRIVATE OpenMP::OpenMP_CXX)
    endif()
else()
    # GCC/Clang编译器优化
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(test_naive_gemm PRIVATE
            -O3              # 最高级优化
            -march=native    # 针对当前CPU架构优化
        )
        if(ENABLE_OPENMP)
            target_compile_options(test_naive_gemm PRIVATE -fopenmp)
        endif()
    else()
        if(ENABLE_OPENMP)
            target_compile_options(test_naive_gemm PRIVATE -fopenmp)
        endif()
    endif()
    # 链接OpenMP库（如果启用）
    if(ENABLE_OPENMP)
        target_link_libraries(test_naive_gemm PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# 链接库 (naive实现是standalone的，不需要框架库)
if(WIN32)
    # target_link_libraries(test_naive_gemm PRIVATE)  # 注释掉，不需要框架库
else()
    # target_link_libraries(test_naive_gemm PRIVATE pthread)  # 注释掉，不需要框架库
endif()

# 设置编译属性
set_target_properties(test_naive_gemm PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 注册测试到CTest
add_test(NAME NaiveGEMMTest COMMAND test_naive_gemm)
set_tests_properties(NaiveGEMMTest PROPERTIES
    TIMEOUT 120  # 矩阵乘法计算量较大，设置更长的超时时间
    PASS_REGULAR_EXPRESSION "SUCCESS"
)

message(STATUS "[SUCCESS] CPU naive GEMM test configured successfully")

# Shape类测试
add_executable(test_shape test_shape.cpp)

# 包含头文件目录
target_include_directories(test_shape
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 设置编译属性
set_target_properties(test_shape PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 链接库
if(WIN32)
    target_link_libraries(test_shape PRIVATE tech_renaissance)
else()
    target_link_libraries(test_shape PRIVATE tech_renaissance pthread)
endif()

# 注册测试到CTest
add_test(NAME ShapeTest COMMAND test_shape)
set_tests_properties(ShapeTest PROPERTIES
    TIMEOUT 30
    PASS_REGULAR_EXPRESSION "All Tests PASSED"
)

# DType类测试
add_executable(test_dtype test_dtype.cpp)

# 包含头文件目录
target_include_directories(test_dtype
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 设置编译属性
set_target_properties(test_dtype PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 链接库
if(WIN32)
    target_link_libraries(test_dtype PRIVATE tech_renaissance)
else()
    target_link_libraries(test_dtype PRIVATE tech_renaissance pthread)
endif()

# 注册测试到CTest
add_test(NAME DTypeTest COMMAND test_dtype)
set_tests_properties(DTypeTest PROPERTIES
    TIMEOUT 30
    PASS_REGULAR_EXPRESSION "All DType Tests PASSED"
)

message(STATUS "[SUCCESS] DType class test configured successfully")

# Device类测试
add_executable(test_device test_device.cpp)

# 包含头文件目录
target_include_directories(test_device
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 设置编译属性
set_target_properties(test_device PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 链接库
if(WIN32)
    target_link_libraries(test_device PRIVATE tech_renaissance)
else()
    target_link_libraries(test_device PRIVATE tech_renaissance pthread)
endif()

# 注册测试到CTest
add_test(NAME DeviceTest COMMAND test_device)
set_tests_properties(DeviceTest PROPERTIES
    TIMEOUT 30
    PASS_REGULAR_EXPRESSION "All Device Tests PASSED"
)

message(STATUS "[SUCCESS] Device class test configured successfully")

# Storage类测试
add_executable(test_storage test_storage.cpp)

# 包含头文件目录
target_include_directories(test_storage
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 设置编译属性
set_target_properties(test_storage PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 链接库
if(WIN32)
    target_link_libraries(test_storage PRIVATE tech_renaissance)
else()
    target_link_libraries(test_storage PRIVATE tech_renaissance pthread)
endif()

# 注册测试到CTest
add_test(NAME StorageTest COMMAND test_storage)
set_tests_properties(StorageTest PROPERTIES
    TIMEOUT 30
    PASS_REGULAR_EXPRESSION "All Storage Tests PASSED"
)

message(STATUS "[SUCCESS] Storage class test configured successfully")

# Tensor类测试
add_executable(test_tensor test_tensor.cpp)

# 包含头文件目录
target_include_directories(test_tensor
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 设置编译属性
set_target_properties(test_tensor PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 链接库
if(WIN32)
    target_link_libraries(test_tensor PRIVATE tech_renaissance)
else()
    target_link_libraries(test_tensor PRIVATE tech_renaissance pthread)
endif()

# 注册测试到CTest
add_test(NAME TensorTest COMMAND test_tensor)
set_tests_properties(TensorTest PROPERTIES
    TIMEOUT 30
    PASS_REGULAR_EXPRESSION "All Tests PASSED"
)

message(STATUS "[SUCCESS] Tensor class test configured successfully")

# Tensor-Backend联合测试
add_executable(test_tensor_backend test_tensor_backend.cpp)

# 包含头文件目录
target_include_directories(test_tensor_backend
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 设置编译属性
set_target_properties(test_tensor_backend PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 链接库
if(WIN32)
    target_link_libraries(test_tensor_backend PRIVATE tech_renaissance)
else()
    target_link_libraries(test_tensor_backend PRIVATE tech_renaissance pthread)
endif()

# 如果CUDA可用，添加CUDA库支持
if(CUDA_BACKEND_ENABLED)
    # 使用根目录创建的cudnn_static目标
    if(TARGET cudnn_static)
        target_link_libraries(test_tensor_backend PRIVATE cudnn_static)
        target_compile_definitions(test_tensor_backend PRIVATE TR_ENABLE_CUDA)
        message(STATUS "[SUCCESS] Using cudnn_static target for test_tensor_backend")
    else()
        message(WARNING "[FAIL] cudnn_static target not available, CUDA tests disabled")
    endif()
endif()

# 注册测试到CTest
add_test(NAME TensorBackendTest COMMAND test_tensor_backend)
set_tests_properties(TensorBackendTest PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION "Tensor-Backend 联合测试完成"
)

message(STATUS "[SUCCESS] Tensor-Backend joint test configured successfully")

# PythonSession测试（可选组件）
if(TR_BUILD_PYTHON_SESSION)

    add_executable(test_pytorch_data test_pytorch_data.cpp)

    # 包含头文件目录
    target_include_directories(test_pytorch_data
            PRIVATE
            ../../include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
    )

    # 链接库
    if(WIN32)
        target_link_libraries(test_pytorch_data PRIVATE tech_renaissance)
    else()
        target_link_libraries(test_pytorch_data PRIVATE tech_renaissance pthread)
    endif()

    # 设置编译属性
    set_target_properties(test_pytorch_data PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
            FOLDER "tests"
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    # 定义项目根目录路径宏，解决Python脚本路径问题
    target_compile_definitions(test_pytorch_data PRIVATE
            PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
    )

    # 注册测试到CTest
    add_test(NAME PyTorchDataTest COMMAND test_pytorch_data)
    set_tests_properties(PyTorchDataTest PROPERTIES
            TIMEOUT 60  # Python交互测试需要更长时间
            PASS_REGULAR_EXPRESSION "All PyTorch Data tests PASSED"
    )

    message(STATUS "[SUCCESS] PyTorch Data test configured successfully")

    # PythonSession MNIST并行测试
    add_executable(test_pytorch_mnist test_pytorch_mnist.cpp)

    # 包含头文件目录
    target_include_directories(test_pytorch_mnist
        PRIVATE
            ../../include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
    )

    # 链接库
    if(WIN32)
        target_link_libraries(test_pytorch_mnist PRIVATE tech_renaissance)
    else()
        target_link_libraries(test_pytorch_mnist PRIVATE tech_renaissance pthread)
    endif()

    # 定义项目根目录路径宏，解决Python脚本路径问题
    target_compile_definitions(test_pytorch_mnist PRIVATE
        PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
    )

    # 设置编译属性
    set_target_properties(test_pytorch_mnist PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "tests"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    # 注册测试到CTest
    add_test(NAME PythonSessionMNISTTest COMMAND test_pytorch_mnist)
    set_tests_properties(PythonSessionMNISTTest PROPERTIES
        TIMEOUT 90  # MNIST并行测试需要更长时间
        PASS_REGULAR_EXPRESSION "All MNIST tests passed successfully"
    )

    message(STATUS "[SUCCESS] PythonSession MNIST parallel test configured successfully")
else()
    message(STATUS "[INFO] PythonSession support disabled, tests skipped")
endif()

# =============================================================================
# 张量打印功能测试
# =============================================================================
add_executable(test_print test_print.cpp)

# 包含头文件目录
target_include_directories(test_print
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 设置编译选项
target_compile_definitions(test_print PRIVATE
    PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

set_target_properties(test_print PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 链接库
if(WIN32)
    target_link_libraries(test_print PRIVATE tech_renaissance)
else()
    target_link_libraries(test_print PRIVATE tech_renaissance pthread)
endif()

# 如果CUDA可用，添加CUDA库支持
if(CUDA_BACKEND_ENABLED)
    # 使用根目录创建的cudnn_static目标
    if(TARGET cudnn_static)
        target_link_libraries(test_print PRIVATE cudnn_static)
        target_compile_definitions(test_print PRIVATE TR_ENABLE_CUDA)
        message(STATUS "[SUCCESS] Using cudnn_static target for test_print")
    else()
        message(WARNING "[FAIL] cudnn_static target not available, CUDA tests disabled")
    endif()
endif()

# 注册测试到CTest
add_test(NAME TensorPrintTest COMMAND test_print)
set_tests_properties(TensorPrintTest PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION "All Tensor Print tests PASSED"
)

message(STATUS "[SUCCESS] Tensor print test configured successfully")

# =============================================================================
# 张量导入导出功能测试
# =============================================================================
add_executable(test_tensor_io test_tensor_io.cpp)

# 包含头文件目录
target_include_directories(test_tensor_io
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 设置编译属性
set_target_properties(test_tensor_io PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 链接库
if(WIN32)
    target_link_libraries(test_tensor_io PRIVATE tech_renaissance)
else()
    target_link_libraries(test_tensor_io PRIVATE tech_renaissance pthread)
endif()

# 如果CUDA可用，添加CUDA库支持
if(CUDA_BACKEND_ENABLED)
    # 使用根目录创建的cudnn_static目标
    if(TARGET cudnn_static)
        target_link_libraries(test_tensor_io PRIVATE cudnn_static)
        target_compile_definitions(test_tensor_io PRIVATE TR_ENABLE_CUDA)
        message(STATUS "[SUCCESS] Using cudnn_static target for test_tensor_io")
    else()
        message(WARNING "[FAIL] cudnn_static target not available, CUDA tests disabled")
    endif()
endif()

# 注册测试到CTest
add_test(NAME TensorIOTest COMMAND test_tensor_io)
set_tests_properties(TensorIOTest PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION "All tests passed"
)

message(STATUS "[SUCCESS] Tensor I/O test configured successfully")

# =============================================================================
# is_close方法功能测试
# =============================================================================
add_executable(test_is_close test_is_close.cpp)

# 包含头文件目录
target_include_directories(test_is_close
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 设置编译属性
set_target_properties(test_is_close PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 链接库
if(WIN32)
    target_link_libraries(test_is_close PRIVATE tech_renaissance)
else()
    target_link_libraries(test_is_close PRIVATE tech_renaissance pthread)
endif()

# 如果CUDA可用，添加CUDA库支持（用于测试跨设备比较）
if(CUDA_BACKEND_ENABLED)
    # 使用根目录创建的cudnn_static目标
    if(TARGET cudnn_static)
        target_link_libraries(test_is_close PRIVATE cudnn_static)
        target_compile_definitions(test_is_close PRIVATE TR_ENABLE_CUDA)
        message(STATUS "[SUCCESS] Using cudnn_static target for test_is_close")
    else()
        message(WARNING "[FAIL] cudnn_static target not available, CUDA tests disabled")
    endif()
endif()

# 注册测试到CTest
add_test(NAME IsCloseTest COMMAND test_is_close)
set_tests_properties(IsCloseTest PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION "ALL TESTS PASSED"
)

message(STATUS "[SUCCESS] is_close method test configured successfully")

# =============================================================================
# CPU矩阵乘法测试（新的测试文件）
# =============================================================================
if(TR_BUILD_PYTHON_SESSION)
    add_executable(test_cpu_mm test_cpu_mm.cpp)

    # 包含头文件目录
    target_include_directories(test_cpu_mm
        PRIVATE
            ../../include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
    )

    # 如果Eigen可用，添加Eigen支持
    if(EIGEN_FOUND)
        target_include_directories(test_cpu_mm PRIVATE
            ${EIGEN_INCLUDE_DIR}
        )
        target_compile_definitions(test_cpu_mm PRIVATE TR_USE_EIGEN)
        message(STATUS "[SUCCESS] test_cpu_mm configured with Eigen support")
    endif()

    # 链接库
    if(WIN32)
        target_link_libraries(test_cpu_mm PRIVATE tech_renaissance)
    else()
        target_link_libraries(test_cpu_mm PRIVATE tech_renaissance pthread)
    endif()

    # 如果CUDA可用，添加CUDA库支持
    if(CUDA_BACKEND_ENABLED)
        if(TARGET cudnn_static)
            target_link_libraries(test_cpu_mm PRIVATE cudnn_static)
            target_compile_definitions(test_cpu_mm PRIVATE TR_ENABLE_CUDA)
            message(STATUS "[SUCCESS] Using cudnn_static target for test_cpu_mm")
        endif()
    endif()

    # 定义项目根目录路径宏
    target_compile_definitions(test_cpu_mm PRIVATE
        PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
    )

    # 设置编译属性
    set_target_properties(test_cpu_mm PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "tests"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    # 注册测试到CTest
    add_test(NAME CPUMMTest COMMAND test_cpu_mm)
    set_tests_properties(CPUMMTest PROPERTIES
        TIMEOUT 180  # 矩阵乘法测试可能需要较长时间
        PASS_REGULAR_EXPRESSION "ALL TESTS PASSED"
    )

    message(STATUS "[SUCCESS] CPU matrix multiplication test configured successfully")
else()
    message(STATUS "[INFO] test_cpu_mm skipped - Python session support disabled")
endif()

message(STATUS "Unit tests configured successfully (backend tests enabled)")

# ========== 新API测试 ==========
add_executable(test_new_api test_new_api.cpp)

# 链接库
if(WIN32)
    target_link_libraries(test_new_api PRIVATE tech_renaissance)
else()
    target_link_libraries(test_new_api PRIVATE tech_renaissance pthread)
endif()

# 设置编译属性
set_target_properties(test_new_api PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

message(STATUS "[SUCCESS] New API test configured successfully")

# ========== CPU标量运算测试 ==========
add_executable(test_cpu_scalar test_cpu_scalar.cpp)

# 包含头文件目录
target_include_directories(test_cpu_scalar
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 如果Eigen可用，添加Eigen支持
if(EIGEN_FOUND)
    target_include_directories(test_cpu_scalar PRIVATE
        ${EIGEN_INCLUDE_DIR}
    )
    target_compile_definitions(test_cpu_scalar PRIVATE TR_USE_EIGEN)
    message(STATUS "[SUCCESS] test_cpu_scalar configured with Eigen support")
endif()

# 链接库
if(WIN32)
    target_link_libraries(test_cpu_scalar PRIVATE tech_renaissance)
else()
    target_link_libraries(test_cpu_scalar PRIVATE tech_renaissance pthread)
endif()

# 如果CUDA可用，添加CUDA库支持
if(CUDA_BACKEND_ENABLED)
    if(TARGET cudnn_static)
        target_link_libraries(test_cpu_scalar PRIVATE cudnn_static)
        target_compile_definitions(test_cpu_scalar PRIVATE TR_ENABLE_CUDA)
        message(STATUS "[SUCCESS] Using cudnn_static target for test_cpu_scalar")
    endif()
endif()

# 设置编译属性
set_target_properties(test_cpu_scalar PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 注册测试到CTest
add_test(NAME CPUScalarTest COMMAND test_cpu_scalar)
set_tests_properties(CPUScalarTest PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION "All CPU scalar operations completed successfully"
)

message(STATUS "[SUCCESS] CPU scalar operations test configured successfully")

# ========== CPU广播运算测试 ==========
add_executable(test_cpu_broadcast test_cpu_broadcast.cpp)

# 包含头文件目录
target_include_directories(test_cpu_broadcast
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 如果Eigen可用，添加Eigen支持
if(EIGEN_FOUND)
    target_include_directories(test_cpu_broadcast PRIVATE
        ${EIGEN_INCLUDE_DIR}
    )
    target_compile_definitions(test_cpu_broadcast PRIVATE TR_USE_EIGEN)
    message(STATUS "[SUCCESS] test_cpu_broadcast configured with Eigen support")
endif()

# 链接库
if(WIN32)
    target_link_libraries(test_cpu_broadcast PRIVATE tech_renaissance)
else()
    target_link_libraries(test_cpu_broadcast PRIVATE tech_renaissance pthread)
endif()

# 如果CUDA可用，添加CUDA库支持
if(CUDA_BACKEND_ENABLED)
    if(TARGET cudnn_static)
        target_link_libraries(test_cpu_broadcast PRIVATE cudnn_static)
        target_compile_definitions(test_cpu_broadcast PRIVATE TR_ENABLE_CUDA)
        message(STATUS "[SUCCESS] Using cudnn_static target for test_cpu_broadcast")
    endif()
endif()

# 设置编译属性
set_target_properties(test_cpu_broadcast PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 注册测试到CTest
add_test(NAME CPUBroadcastTest COMMAND test_cpu_broadcast)
set_tests_properties(CPUBroadcastTest PROPERTIES
    TIMEOUT 120
    PASS_REGULAR_EXPRESSION "CPU广播运算功能测试完成"
)

message(STATUS "[SUCCESS] CPU broadcast operations test configured successfully")

# ========== CPU扩展运算测试 ==========
add_executable(test_cpu_expand test_cpu_expand.cpp)

# 包含头文件目录
target_include_directories(test_cpu_expand
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 如果Eigen可用，添加Eigen支持
if(EIGEN_FOUND)
    target_include_directories(test_cpu_expand PRIVATE
        ${EIGEN_INCLUDE_DIR}
    )
    target_compile_definitions(test_cpu_expand PRIVATE TR_USE_EIGEN)
    message(STATUS "[SUCCESS] test_cpu_expand configured with Eigen support")
endif()

# 链接库
if(WIN32)
    target_link_libraries(test_cpu_expand PRIVATE tech_renaissance)
else()
    target_link_libraries(test_cpu_expand PRIVATE tech_renaissance pthread)
endif()

# 如果CUDA可用，添加CUDA库支持
if(CUDA_BACKEND_ENABLED)
    if(TARGET cudnn_static)
        target_link_libraries(test_cpu_expand PRIVATE cudnn_static)
        target_compile_definitions(test_cpu_expand PRIVATE TR_ENABLE_CUDA)
        message(STATUS "[SUCCESS] Using cudnn_static target for test_cpu_expand")
    endif()
endif()

# 设置编译属性
set_target_properties(test_cpu_expand PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 注册测试到CTest
add_test(NAME CPUExpandTest COMMAND test_cpu_expand)
set_tests_properties(CPUExpandTest PROPERTIES
    TIMEOUT 120
    PASS_REGULAR_EXPRESSION "CPU expand operations test completed successfully"
)

message(STATUS "[SUCCESS] CPU expand operations test configured successfully")

# ========== CPU维度操作测试 ==========
add_executable(test_cpu_dimension test_cpu_dimension.cpp)

# 包含头文件目录
target_include_directories(test_cpu_dimension
    PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
)

# 如果Eigen可用，添加Eigen支持
if(EIGEN_FOUND)
    target_include_directories(test_cpu_dimension PRIVATE
        ${EIGEN_INCLUDE_DIR}
    )
    target_compile_definitions(test_cpu_dimension PRIVATE TR_USE_EIGEN)
    message(STATUS "[SUCCESS] test_cpu_dimension configured with Eigen support")
endif()

# 链接库
if(WIN32)
    target_link_libraries(test_cpu_dimension PRIVATE tech_renaissance)
else()
    target_link_libraries(test_cpu_dimension PRIVATE tech_renaissance pthread)
endif()

# 如果CUDA可用，添加CUDA库支持
if(CUDA_BACKEND_ENABLED)
    if(TARGET cudnn_static)
        target_link_libraries(test_cpu_dimension PRIVATE cudnn_static)
        target_compile_definitions(test_cpu_dimension PRIVATE TR_ENABLE_CUDA)
        message(STATUS "[SUCCESS] Using cudnn_static target for test_cpu_dimension")
    endif()
endif()

# 设置编译属性
set_target_properties(test_cpu_dimension PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tests"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 注册测试到CTest
add_test(NAME CPUDimensionTest COMMAND test_cpu_dimension)
set_tests_properties(CPUDimensionTest PROPERTIES
    TIMEOUT 120
    PASS_REGULAR_EXPRESSION "CPU dimension operations test completed successfully"
)

message(STATUS "[SUCCESS] CPU dimension operations test configured successfully")

# ========== Python会话测试 ==========
add_executable(test_python_session test_python_session.cpp)

# 链接库
if(WIN32)
    target_link_libraries(test_python_session PRIVATE tech_renaissance)
else()
    target_link_libraries(test_python_session PRIVATE tech_renaissance pthread)
endif()

# 设置编译属性
set_target_properties(test_python_session PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "tests"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

message(STATUS "[SUCCESS] New API test configured successfully")
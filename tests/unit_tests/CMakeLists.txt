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
configure_simple_test(test_cpu_bce test_cpu_bce.cpp)
configure_simple_test(test_cpu_broadcast test_cpu_broadcast.cpp)
configure_simple_test(test_cpu_expand test_cpu_expand.cpp)
configure_simple_test(test_cpu_dimension test_cpu_dimension.cpp)
configure_simple_test(test_cpu_pad test_cpu_pad.cpp)
configure_simple_test(test_cpu_create test_cpu_create.cpp)
configure_simple_test(test_cpu_cast test_cpu_cast.cpp)

# =============================================================================
# CPU矩阵运算测试
# =============================================================================
configure_simple_test(test_cpu_mm test_cpu_mm.cpp)

# =============================================================================
# CPU卷积测试
# =============================================================================
configure_simple_test(test_cpu_conv test_cpu_conv.cpp)
configure_simple_test(test_cpu_gemm test_cpu_gemm.cpp)

# =============================================================================
# 简单算法测试
# =============================================================================
configure_simple_test(test_naive_conv test_naive_conv.cpp)
configure_simple_test(test_naive_gemm test_naive_gemm.cpp)

# =============================================================================
# 高级功能测试
# =============================================================================
configure_simple_test(test_new_api test_new_api.cpp)

# =============================================================================
# 内存管理测试
# =============================================================================
configure_simple_test(test_memory_occupation test_memory_occupation.cpp)
configure_simple_test(test_get_memory_size test_get_memory_size.cpp)

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
register_test(test_cpu_bce "CPU BCE Test")
register_test(test_cpu_broadcast "CPU Broadcast Test")
register_test(test_cpu_expand "CPU Expand Test")
register_test(test_cpu_dimension "CPU Dimension Test")
register_test(test_cpu_pad "CPU Pad Test")
register_test(test_cpu_create "CPU Create Test")
register_test(test_cpu_cast "CPU Cast Test")

# 注册矩阵运算测试
register_test(test_cpu_mm "CPU Matrix Multiplication Test")

# 注册卷积测试
register_test(test_cpu_conv "CPU Convolution Test")
register_test(test_cpu_gemm "CPU GEMM Test")

# 注册简单算法测试
register_test(test_naive_conv "Naive Convolution Test")
register_test(test_naive_gemm "Naive GEMM Test")

# 注册高级功能测试
register_test(test_new_api "New API Test")

# 注册内存管理测试
register_test(test_memory_occupation "Memory Occupation Test")
register_test(test_get_memory_size "Get Memory Size Test")

# 注册Python接口测试
if(TR_BUILD_PYTHON_SESSION)
    register_test(test_python_session "Python Session Test")

    if(TR_PYTHON_MNIST_TEST)
        register_test(test_pytorch_data "PyTorch Data Test")
        register_test(test_pytorch_mnist "PyTorch MNIST Test")
    endif()
endif()

message(STATUS "All tests configured successfully (optimized version)")
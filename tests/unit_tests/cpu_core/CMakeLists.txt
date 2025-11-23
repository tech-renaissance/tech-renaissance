# =============================================================================
# 技术觉醒框架 - CPU Core模块单元测试构建配置
# 版本: 2.0.0
# 日期: 2025-11-23
# 作者: 技术觉醒团队
# 说明: CPU Core Module的测试用例，支持模块化架构和Linux移植
# =============================================================================

# 启用测试
enable_testing()

# 检查Eigen库 - 使用自动配置的路径
if(DEFINED EIGEN3_INCLUDE_DIR)
    set(EIGEN_FOUND TRUE)
    set(EIGEN_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}")
    message(STATUS "Using configured Eigen3: ${EIGEN_INCLUDE_DIR}")
else()
    set(EIGEN_FOUND FALSE)
    message(STATUS "Eigen3 not configured, using naive computation")
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

# =============================================================================
# CPU Core测试配置函数
# =============================================================================
function(configure_cpu_core_test test_name source_file)
    add_executable(${test_name} ${source_file})

    # 链接主库 - 让tech_renaissance自动传递所有依赖
    if(WIN32)
        target_link_libraries(${test_name} PRIVATE tech_renaissance)
    else()
        target_link_libraries(${test_name} PRIVATE tech_renaissance pthread)
    endif()

    # 定义PROJECT_ROOT_DIR宏，指向项目根目录
    target_compile_definitions(${test_name}
        PRIVATE
            PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
    )

    # Eigen优化配置
    if(EIGEN_FOUND)
        target_include_directories(${test_name} PRIVATE ${EIGEN_INCLUDE_DIR})
        target_compile_definitions(${test_name} PRIVATE
            EIGEN_DONT_PARALLELIZE  # 禁用自动并行化，使用OpenMP手动控制
        )
    endif()

    # OpenMP配置
    if(OPENMP_AVAILABLE)
        if(MSVC)
            target_compile_options(${test_name} PRIVATE /openmp)
        else()
            target_compile_options(${test_name} PRIVATE -fopenmp)
            target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()

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
        FOLDER "tests/cpu_core"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    message(STATUS "[SUCCESS] CPU Core test ${test_name} configured")
endfunction()

# =============================================================================
# CPU Core基础测试 - 按实际文件配置
# =============================================================================

# 基础工具测试
configure_cpu_core_test(test_logger test_logger.cpp)
configure_cpu_core_test(test_shape test_shape.cpp)
configure_cpu_core_test(test_storage test_storage.cpp)
configure_cpu_core_test(test_tensor test_tensor.cpp)

# CPU后端测试
configure_cpu_core_test(test_cpu_backend test_tensor_backend.cpp)
configure_cpu_core_test(test_copy test_copy.cpp)
configure_cpu_core_test(test_print test_print.cpp)

# CPU操作测试
configure_cpu_core_test(test_cpu_unary test_cpu_unary.cpp)
configure_cpu_core_test(test_cpu_create test_cpu_create.cpp)
configure_cpu_core_test(test_cpu_cast test_cpu_cast.cpp)
configure_cpu_core_test(test_cpu_broadcast test_cpu_broadcast.cpp)
configure_cpu_core_test(test_cpu_slice test_cpu_slice.cpp)

# 张量I/O测试
configure_cpu_core_test(test_tsr_io_extended test_tsr_io_extended.cpp)

# 视图操作测试
configure_cpu_core_test(test_view test_view.cpp)

# Model和Module测试
configure_cpu_core_test(test_model test_model.cpp)
configure_cpu_core_test(test_mlp_module test_mlp_module.cpp)

# 学习率调度器测试
configure_cpu_core_test(test_lr_schedulers test_lr_schedulers.cpp)

# 性能测试
configure_cpu_core_test(test_performance test_performance.cpp)
configure_cpu_core_test(test_memory_occupation test_memory_occupation.cpp)

# =============================================================================
# 注册测试到CTest
# =============================================================================
function(register_cpu_core_test test_name test_description)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        TIMEOUT 120
        PASS_REGULAR_EXPRESSION "PASSED|SUCCESS|completed successfully"
        LABELS "cpu_core"
    )
endfunction()

# 注册所有CPU Core测试
register_cpu_core_test(test_logger "Logger Test")
register_cpu_core_test(test_shape "Shape Test")
register_cpu_core_test(test_storage "Storage Test")
register_cpu_core_test(test_tensor "Tensor Test")
register_cpu_core_test(test_cpu_backend "CPU Backend Test")
register_cpu_core_test(test_copy "Copy Test")
register_cpu_core_test(test_print "Print Test")
register_cpu_core_test(test_cpu_unary "CPU Unary Test")
register_cpu_core_test(test_cpu_create "CPU Create Test")
register_cpu_core_test(test_cpu_cast "CPU Cast Test")
register_cpu_core_test(test_cpu_broadcast "CPU Broadcast Test")
register_cpu_core_test(test_cpu_slice "CPU Slice Test")
register_cpu_core_test(test_tsr_io_extended "TSR Extended I/O Test")
register_cpu_core_test(test_view "View Test")
register_cpu_core_test(test_model "Model Test")
register_cpu_core_test(test_mlp_module "MLP Module Test")
register_cpu_core_test(test_lr_schedulers "Learning Rate Schedulers Test")
register_cpu_core_test(test_performance "Performance Test")
register_cpu_core_test(test_memory_occupation "Memory Occupation Test")

message(STATUS "CPU Core tests configured successfully - ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "Total CPU Core tests: 20")
# =============================================================================
# 技术觉醒框架 - 集成测试构建配置（CPU Core Module）
# 版本: 2.0.0
# 日期: 2025-11-23
# 作者: 技术觉醒团队
# 说明: 集成测试，基于实际存在的Trainer测试文件
# =============================================================================

# 启用测试
enable_testing()

# =============================================================================
# 集成测试配置函数
# =============================================================================
function(configure_integration_test test_name source_file)
    add_executable(${test_name} ${source_file})

    # 链接主库
    if(WIN32)
        target_link_libraries(${test_name} PRIVATE tech_renaissance)
    else()
        target_link_libraries(${test_name} PRIVATE tech_renaissance pthread)
    endif()

    # 设置标准属性
    set_target_properties(${test_name} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "tests/integration"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )

    message(STATUS "[SUCCESS] Integration test ${test_name} configured")
endfunction()

# =============================================================================
# 集成测试 - 基于实际存在的文件
# =============================================================================

configure_integration_test(test_trainer_sgd test_trainer_sgd.cpp)
configure_integration_test(test_trainer_adam test_trainer_adam.cpp)
configure_integration_test(test_trainer_adamw test_trainer_adamw.cpp)
configure_integration_test(test_task_sgd test_task_sgd.cpp)
configure_integration_test(test_task_adamw test_task_adamw.cpp)

# =============================================================================
# 注册测试到CTest
# =============================================================================
function(register_integration_test test_name test_description)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        TIMEOUT 300
        PASS_REGULAR_EXPRESSION "PASSED|SUCCESS|completed successfully"
        LABELS "integration"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )
endfunction()

# 注册所有集成测试
register_integration_test(test_trainer_sgd "Trainer SGD Integration Test")
register_integration_test(test_trainer_adam "Trainer Adam Integration Test")
register_integration_test(test_trainer_adamw "Trainer AdamW Integration Test")
register_integration_test(test_task_sgd "Task SGD Integration Test")
register_integration_test(test_task_adamw "Task AdamW Integration Test")

message(STATUS "Integration tests configured successfully - CPU Core Module only")
message(STATUS "Total integration tests: 5")
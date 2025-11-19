# 集成测试的CMakeLists.txt

# 设置测试源文件
set(TEST_SOURCES
    test_model_optimizer_integration.cpp
)

# 创建测试可执行文件
add_executable(test_model_optimizer_integration ${TEST_SOURCES})

# 链接必要的库
target_link_libraries(test_model_optimizer_integration
    PRIVATE
    tech_renaissance  # 主框架库
)

# 包含头文件
target_include_directories(test_model_optimizer_integration
    PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# 设置输出目录
set_target_properties(test_model_optimizer_integration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 添加到CTest
add_test(NAME ModelOptimizerIntegrationTest
         COMMAND test_model_optimizer_integration)

# 设置测试工作目录
set_tests_properties(ModelOptimizerIntegrationTest PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
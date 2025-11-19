# 集成测试的CMakeLists.txt

# 设置测试源文件
set(TEST_SOURCES
    test_model_optimizer_integration.cpp
    test_training_sgd.cpp
    test_training_adam.cpp
)

# 创建测试可执行文件
add_executable(test_model_optimizer_integration test_model_optimizer_integration.cpp)
add_executable(test_training_sgd test_training_sgd.cpp)
add_executable(test_training_adam test_training_adam.cpp)
add_executable(test_training_simple test_training_simple.cpp)

# 链接必要的库
target_link_libraries(test_model_optimizer_integration
    PRIVATE
    tech_renaissance  # 主框架库
)

target_link_libraries(test_training_sgd
    PRIVATE
    tech_renaissance  # 主框架库
)

target_link_libraries(test_training_adam
    PRIVATE
    tech_renaissance  # 主框架库
)

target_link_libraries(test_training_simple
    PRIVATE
    tech_renaissance  # 主框架库
)

# 包含头文件
target_include_directories(test_model_optimizer_integration
    PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_include_directories(test_training_sgd
    PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_include_directories(test_training_adam
    PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_include_directories(test_training_simple
    PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# 设置输出目录
set_target_properties(test_model_optimizer_integration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

set_target_properties(test_training_sgd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

set_target_properties(test_training_adam PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

set_target_properties(test_training_simple PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# 添加到CTest
add_test(NAME ModelOptimizerIntegrationTest
         COMMAND test_model_optimizer_integration)

add_test(NAME TrainingAlignmentTestSGD
         COMMAND test_training_sgd)

add_test(NAME TrainingAlignmentTestAdam
         COMMAND test_training_adam)

# 设置测试工作目录
set_tests_properties(ModelOptimizerIntegrationTest PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

set_tests_properties(TrainingAlignmentTestSGD PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

set_tests_properties(TrainingAlignmentTestAdam PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
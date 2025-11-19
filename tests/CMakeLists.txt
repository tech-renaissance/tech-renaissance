# 测试目录的CMakeLists.txt

# 启用测试
enable_testing()

# 添加单元测试
add_subdirectory(unit_tests)

# 添加集成测试
add_subdirectory(integration_tests)

message(STATUS "All tests configured successfully (including integration tests)")
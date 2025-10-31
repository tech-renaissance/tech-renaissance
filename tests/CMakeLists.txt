# 测试目录的CMakeLists.txt

# 启用测试
enable_testing()

# 添加单元测试
add_subdirectory(unit_tests)

message(STATUS "All tests configured successfully")
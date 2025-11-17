# =============================================================================
# 技术觉醒框架 - Model模块构建配置
# 版本: 1.45.0
# 日期: 2025-11-17
# 作者: 技术觉醒团队
# =============================================================================

# 创建Model模块静态库
add_library(model STATIC
    module.cpp
    linear.cpp
    flatten.cpp
    tanh.cpp
)

# 设置包含目录
target_include_directories(model PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# 设置标准属性
set_target_properties(model PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "src"
)

# 链接依赖
target_link_libraries(model PUBLIC
    data
    backend
)

message(STATUS "Model module configured successfully")
# =============================================================================
# 技术觉醒框架 - 源文件模块构建配置
# 版本: 1.01.01
# 日期: 2025-10-24
# 作者: 技术觉醒团队
# =============================================================================

# 添加工具模块子目录
add_subdirectory(utils)

# 添加数据模块子目录
add_subdirectory(data)

# 添加后端模块子目录
add_subdirectory(backend)

# 添加Model模块子目录
add_subdirectory(model)

# 添加Trainer模块子目录
add_subdirectory(trainer)

# 创建主库目标（聚合所有模块）
add_library(tech_renaissance INTERFACE)

# 链接各个模块
target_link_libraries(tech_renaissance INTERFACE
    tech_renaissance_utils
    data
    backend
    model
    trainer
)

# 设置包含目录
target_include_directories(tech_renaissance INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

# 设置全局编译定义
target_compile_definitions(tech_renaissance INTERFACE
    TR_VERSION_MAJOR=1
    TR_VERSION_MINOR=01
    TR_VERSION_PATCH=01
    TR_VERSION="1.01.01"
)

message(STATUS "Source modules configured successfully (CPU Core Module only)")

# =============================================================================
# 技术觉醒框架 - 单元测试主配置
# 版本: 2.0.0
# 日期: 2025-11-23
# 作者: 技术觉醒团队
# 说明: 单元测试主入口，指向CPU Core模块测试
# =============================================================================

# 启用测试
enable_testing()

# 添加CPU Core模块测试
add_subdirectory(cpu_core)

# 添加CUDA Core模块测试（当启用CUDA时）
if(TR_USE_CUDA)
    add_subdirectory(cuda_core)
    message(STATUS "CUDA Core tests configured successfully")
endif()

message(STATUS "Unit tests configured successfully - CPU Core Module only")
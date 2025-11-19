# =============================================================================
# 技术觉醒框架 - Trainer模块构建配置
# 版本: 1.48.0
# 日期: 2025-11-17
# 作者: 技术觉醒团队
# =============================================================================

# 收集所有源文件
set(TRAINER_SOURCES
    loss.cpp
    cross_entropy_loss.cpp
    optimizer.cpp
    sgd.cpp
    adam.cpp
    state_manager.cpp
    scheduler.cpp
    step_lr.cpp
    multi_step_lr.cpp
    exponential_lr.cpp
    cosine_annealing_lr.cpp
    cosine_annealing_warm_restarts.cpp
    constant_lr.cpp
)

# 创建trainer库
add_library(trainer STATIC ${TRAINER_SOURCES})

# 设置目标属性
target_include_directories(trainer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(trainer
    tech_renaissance_utils
    data
    backend
    model
)

# 设置编译定义
target_compile_definitions(trainer PRIVATE
    TR_MODULE_TRAINER=1
)

message(STATUS "Trainer module configured successfully")
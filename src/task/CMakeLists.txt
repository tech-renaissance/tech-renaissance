# =============================================================================
# 技术觉醒框架 - Task模块构建配置
# 版本: 2.2.0
# 日期: 2025年11月24日
# 作者: 技术觉醒团队
# 说明: Task高级API模块，简化训练流程
# =============================================================================

# 创建Task模块库
add_library(task STATIC
    task.cpp
    mnist_dataset.cpp
)

# 设置目标属性
set_target_properties(task PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "src/task"
)

# 包含头文件目录
target_include_directories(task PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# 链接依赖模块
target_link_libraries(task PRIVATE
    tech_renaissance_base
    tech_renaissance_utils
    data
    backend
    model
    trainer
)

# OpenMP支持（如果可用）
if(OpenMP_CXX_FOUND)
    if(MSVC)
        target_compile_options(task PRIVATE /openmp)
    else()
        target_compile_options(task PRIVATE -fopenmp)
        target_link_libraries(task PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# Eigen优化（如果可用）
if(TR_USE_EIGEN AND EIGEN3_FOUND)
    target_include_directories(task PRIVATE ${EIGEN3_INCLUDE_DIRS})
    target_compile_definitions(task PRIVATE
        EIGEN_DONT_PARALLELIZE  # 禁用自动并行化，使用OpenMP手动控制
    )
endif()

# 编译器优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(task PRIVATE /O2 /arch:AVX2)
    else()
        target_compile_options(task PRIVATE -O3 -march=native)
    endif()
endif()

message(STATUS "Task module configured successfully")
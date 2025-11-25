# =============================================================================
# 技术觉醒框架 - 后端模块构建配置
# 版本: 1.00.00
# 日期: 2025-10-26
# 作者: 技术觉醒团队
# =============================================================================

# 基础后端源文件（始终编译）
set(BACKEND_SOURCES
    backend.cpp
    backend_manager.cpp
    cpu/cpu_backend.cpp
    cpu/cpu_basic_ops.cpp
    cpu/cpu_mm_fp32.cpp
    cpu/cpu_unary.cpp
    cpu/cpu_scalar.cpp
    cpu/cpu_broadcast.cpp
    cpu/cpu_dimension.cpp
    cpu/cpu_create.cpp
    cpu/cpu_cast.cpp
    cpu/cpu_slice.cpp
    cpu/cpu_pooling.cpp
    cpu/cpu_conv.cpp
    cpu/cpu_loss.cpp
    cpu/cpu_dropout.cpp
)

# 创建基础后端库
add_library(backend STATIC ${BACKEND_SOURCES})

# 包含头文件目录
target_include_directories(backend
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<INSTALL_INTERFACE:include>
)

# 检查是否启用Eigen（已在根CMakeLists.txt中全局配置）
if(TR_USE_EIGEN)
    # 添加Eigen包含目录（已由全局配置处理）
    target_include_directories(backend PUBLIC ${EIGEN_INCLUDE_DIR})

    # 添加Eigen源码目录（用于更优化的配置）
    target_include_directories(backend PUBLIC ${EIGEN_INCLUDE_DIR}/src)

    # 设置Eigen高性能配置
    target_compile_definitions(backend PRIVATE
        EIGEN_DONT_PARALLELIZE  # 禁用自动并行化，使用OpenMP手动控制
    )

    message(STATUS "Eigen optimizations: ENABLED for backend (global config)")
else()
    message(STATUS "Eigen optimizations: DISABLED for backend (global config)")
endif()

# 链接基础库和数据模块（backend模块需要Tensor类）
target_link_libraries(backend PUBLIC tech_renaissance_base data)

# 查找OpenMP支持
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(BACKEND_OPENMP_AVAILABLE TRUE)
    message(STATUS "Backend: OpenMP found for high-performance compilation")
else()
    set(BACKEND_OPENMP_AVAILABLE FALSE)
    message(WARNING "Backend: OpenMP not found, performance may be reduced")
endif()

# 启用高性能编译器优化
if(MSVC)
    # Visual Studio编译器优化 - 只在Release模式下启用高级优化
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(backend PRIVATE
            /O2              # 最高级优化
            /arch:AVX2        # 启用AVX2指令集
            /openmp          # 启用OpenMP支持
        )
    else()
        target_compile_options(backend PRIVATE
            /openmp          # 在Debug模式下只启用OpenMP
        )
    endif()
    # 链接OpenMP库
    if(BACKEND_OPENMP_AVAILABLE)
        target_link_libraries(backend PRIVATE OpenMP::OpenMP_CXX)
    endif()
else()
    # GCC/Clang编译器优化
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(backend PRIVATE
            -O3              # 最高级优化
            -march=native    # 针对当前CPU架构优化
            -fopenmp         # 启用OpenMP
        )
    else()
        target_compile_options(backend PRIVATE
            -fopenmp         # 在Debug模式下只启用OpenMP
        )
    endif()
    if(BACKEND_OPENMP_AVAILABLE)
        target_link_libraries(backend PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# 设置编译选项
target_compile_definitions(backend PRIVATE
    TR_BACKEND_VERSION_MAJOR=1
    TR_BACKEND_VERSION_MINOR=00
    TR_BACKEND_VERSION_PATCH=00
)

# 设置C++标准
set_target_properties(backend PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# 设置属性
set_target_properties(backend PROPERTIES
    FOLDER "src"
    POSITION_INDEPENDENT_CODE ON
)

# =============================================================================
# 构建需要backend依赖的utils组件
# =============================================================================

# 检查是否需要构建utils_backend
set(UTILS_BACKEND_SOURCES)
if(EXISTS "${CMAKE_SOURCE_DIR}/src/utils/mnist_loader.cpp")
    set(UTILS_BACKEND_SOURCES
        ${CMAKE_SOURCE_DIR}/src/utils/mnist_loader.cpp
    )
endif()

if(TR_BUILD_PYTHON_SESSION AND EXISTS "${CMAKE_SOURCE_DIR}/src/utils/python_session.cpp")
    list(APPEND UTILS_BACKEND_SOURCES
        ${CMAKE_SOURCE_DIR}/src/utils/python_session.cpp
    )
endif()

# 如果有需要backend依赖的utils文件，创建utils_backend库
if(UTILS_BACKEND_SOURCES)
    add_library(tech_renaissance_utils_backend STATIC ${UTILS_BACKEND_SOURCES})

    target_include_directories(tech_renaissance_utils_backend
        PUBLIC
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_BINARY_DIR}/include
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/utils
    )

    target_link_libraries(tech_renaissance_utils_backend
        PUBLIC
            tech_renaissance_base
            backend
            data
        PRIVATE
            OpenMP::OpenMP_CXX
    )

    set_target_properties(tech_renaissance_utils_backend PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "src"
        POSITION_INDEPENDENT_CODE ON
    )

    message(STATUS "Backend-dependent utils module configured successfully")
endif()

# =============================================================================
# CUDA后端配置已移除 - 模块化架构支持
# =============================================================================
message(STATUS "CUDA backend removed for modular architecture - CPU Core Module only")

message(STATUS "Backend module configured successfully")
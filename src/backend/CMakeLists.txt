# =============================================================================
# 技术觉醒框架 - 后端模块构建配置
# 版本: 1.00.00
# 日期: 2025-10-26
# 作者: 技术觉醒团队
# =============================================================================

# 基础后端源文件（始终编译）
set(BACKEND_SOURCES
    backend_manager.cpp
    cpu/cpu_backend.cpp
    cpu/cpu_mm_fp32.cpp
    cpu/cpu_unary.cpp
    cpu/cpu_scalar.cpp
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
    message(STATUS "Eigen optimizations: ENABLED for backend (global config)")
else()
    message(STATUS "Eigen optimizations: DISABLED for backend (global config)")
endif()

# 链接数据模块
target_link_libraries(backend PUBLIC data)

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
# CUDA后端配置（可选）
# =============================================================================
if(TR_ENABLE_CUDA AND CUDAToolkit_FOUND AND TARGET cudnn_static)
    message(STATUS "Configuring CUDA backend...")

    # 暂时注释掉CUDA语言启用，使用外部编译方法
    # enable_language(CUDA)

    # CUDA后端源文件（暂时改为.cpp以避免CUDA语言问题）
    set(CUDA_BACKEND_SOURCES
        cuda/cuda_backend.cpp
        cuda/cuda_mm_fp32.cpp
    )

    # 创建CUDA后端库
    add_library(cuda_backend STATIC ${CUDA_BACKEND_SOURCES})

    # CUDA库包含目录
    target_include_directories(cuda_backend
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
            $<INSTALL_INTERFACE:include>
    )

    # 链接CUDA库
    target_link_libraries(cuda_backend PUBLIC
        CUDA::cudart
        CUDA::cublas
        cudnn_static
        data
    )

    # 设置CUDA编译选项
    target_compile_options(cuda_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )

    # 设置属性
    set_target_properties(cuda_backend PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        FOLDER "src"
        POSITION_INDEPENDENT_CODE ON
    )

    # 将CUDA后端链接到主后端库
    target_link_libraries(backend PUBLIC cuda_backend)

    ### 确保三个层级（cuda_backend、backend、tech_renaissance）都定义该宏
    target_compile_definitions(cuda_backend PUBLIC TR_USE_CUDA)
    target_compile_definitions(backend PUBLIC TR_USE_CUDA)

    message(STATUS "CUDA backend configured successfully")
else()
    if(TR_ENABLE_CUDA)
        message(WARNING "CUDA backend disabled - missing CUDAToolkit or cuDNN")
    else()
        message(STATUS "CUDA backend disabled by user")
    endif()
endif()

message(STATUS "Backend module configured successfully")
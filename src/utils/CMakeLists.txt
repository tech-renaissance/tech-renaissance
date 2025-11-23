# 工具模块的CMakeLists.txt

# Python会话支持选项
option(TR_BUILD_PYTHON_SESSION "Enable Python session integration" ON)

# 核心工具源文件（无依赖）
set(CORE_UTILS_SOURCES
    tr_exception.cpp
    logger.cpp
)

# 收集源文件（依赖核心工具）
set(UTILS_SOURCES
    profiler.cpp
)

# mn_loader.cpp和python_session.cpp现在在backend模块中构建
# 因为它们需要backend和data模块的依赖

# 创建基础库目标（无依赖）
add_library(tech_renaissance_base STATIC ${CORE_UTILS_SOURCES})

# 创建工具库目标（依赖基础）
add_library(tech_renaissance_utils STATIC ${UTILS_SOURCES})

# utils_backend现在在backend模块中构建

# 设置基础库目标属性
target_include_directories(tech_renaissance_base
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# 设置工具库目标属性
target_include_directories(tech_renaissance_utils
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# utils_backend现在在backend模块中构建

# 基础库无依赖

# 链接依赖库
target_link_libraries(tech_renaissance_utils
    PUBLIC
        tech_renaissance_base
)

# utils_backend现在在backend模块中构建

if(TR_USE_PROTOBUF)
    target_link_libraries(tech_renaissance_utils PRIVATE protobuf::libprotobuf)
endif()

# 设置编译定义
target_compile_definitions(tech_renaissance_base
    PUBLIC
        $<$<BOOL:${TR_BUILD_PYTHON_SESSION}>:TR_BUILD_PYTHON_SESSION>
)

target_compile_definitions(tech_renaissance_utils
    PUBLIC
        $<$<BOOL:${TR_USE_PROTOBUF}>:TR_USE_PROTOBUF>
        $<$<BOOL:${TR_USE_ONNX}>:TR_USE_ONNX>
        $<$<BOOL:${TR_BUILD_PYTHON_SESSION}>:TR_BUILD_PYTHON_SESSION>
)

# 设置C++标准
set_property(TARGET tech_renaissance_base PROPERTY CXX_STANDARD 17)
set_property(TARGET tech_renaissance_base PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET tech_renaissance_base PROPERTY CXX_EXTENSIONS OFF)

set_property(TARGET tech_renaissance_utils PROPERTY CXX_STANDARD 17)
set_property(TARGET tech_renaissance_utils PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET tech_renaissance_utils PROPERTY CXX_EXTENSIONS OFF)

# 安装规则
install(TARGETS tech_renaissance_base tech_renaissance_utils
    EXPORT TechRenaissanceTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# 安装头文件
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tech_renaissance
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

message(STATUS "Export module configured successfully")
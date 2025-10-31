# 工具模块的CMakeLists.txt

# Python会话支持选项
option(TR_BUILD_PYTHON_SESSION "Enable Python session integration" ON)

# 收集源文件
set(UTILS_SOURCES
    tr_exception.cpp
    logger.cpp
)

# 添加PythonSession源文件（如果启用）
if(TR_BUILD_PYTHON_SESSION)
    list(APPEND UTILS_SOURCES python_session.cpp)
endif()

# 创建库目标
add_library(tech_renaissance_utils STATIC ${UTILS_SOURCES})

# 设置目标属性
target_include_directories(tech_renaissance_utils
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# 链接依赖库
if(TR_USE_PROTOBUF)
    target_link_libraries(tech_renaissance_utils PRIVATE protobuf::libprotobuf)
endif()

# 设置编译定义
target_compile_definitions(tech_renaissance_utils
    PUBLIC
        $<$<BOOL:${TR_USE_PROTOBUF}>:TR_USE_PROTOBUF>
        $<$<BOOL:${TR_USE_ONNX}>:TR_USE_ONNX>
        $<$<BOOL:${TR_BUILD_PYTHON_SESSION}>:TR_BUILD_PYTHON_SESSION>
)

# 设置C++标准
set_property(TARGET tech_renaissance_utils PROPERTY CXX_STANDARD 17)
set_property(TARGET tech_renaissance_utils PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET tech_renaissance_utils PROPERTY CXX_EXTENSIONS OFF)

# 安装规则
install(TARGETS tech_renaissance_utils
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
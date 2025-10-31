/**
 * @file dtype.h
 * @brief 数据类型枚举
 * @details 枚举类型，定义支持的数据类型（FP32、INT8等），用于张量类型标识
 * @version 1.01.01
 * @date 2025-10-24
 * @author 技术觉醒团队
 * @note 依赖项: 无
 * @note 所属系列: data
 */

#pragma once

#include <string>

namespace tr {

/**
 * @enum DType
 * @brief 数据类型枚举
 * @details 仅支持FP32和INT8两种类型，符合轻量级设计原则
 */
enum class DType {
    FP32 = 1,   ///< 32位浮点数
    INT8 = 2    ///< 8位有符号整数
};

/**
 * @brief 获取数据类型大小
 * @param dtype 数据类型
 * @return 字节数
 */
size_t dtype_size(DType dtype);

/**
 * @brief 数据类型转字符串
 * @param dtype 数据类型
 * @return 字符串表示
 */
std::string dtype_to_string(DType dtype);

/**
 * @brief 字符串转数据类型
 * @param str 字符串
 * @return 数据类型
 */
DType string_to_dtype(const std::string& str);

} // namespace tr
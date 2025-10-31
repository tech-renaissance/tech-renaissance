/**
 * @file dtype.cpp
 * @brief 数据类型枚举实现
 * @details 实现数据类型相关的工具函数
 * @version 1.00.00
 * @date 2025-10-25
 * @author 技术觉醒团队
 * @note 依赖项: dtype.h
 * @note 所属系列: data
 */

#include "tech_renaissance/data/dtype.h"
#include "tech_renaissance/utils/tr_exception.h"
#include <stdexcept>

namespace tr {

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FP32:
            return 4;
        case DType::INT8:
            return 1;
        default:
            return 0;
    }
}

std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::FP32:
            return "fp32";
        case DType::INT8:
            return "int8";
        default:
            return "unknown";
    }
}

DType string_to_dtype(const std::string& str) {
    if (str == "fp32" || str == "float32") {
        return DType::FP32;
    } else if (str == "int8") {
        return DType::INT8;
    } else {
        // Fail-Fast原则：不接受的字符串类型应抛出异常，而不是静默转换
        throw TRException("[string_to_dtype] Unsupported dtype string: '" + str +
                                   "'. Supported types are: 'fp32', 'float32', 'int8'");
    }
}

} // namespace tr
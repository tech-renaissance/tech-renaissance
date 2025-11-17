/**
 * @file tech_renaissance.h
 * @brief 技术觉醒框架主头文件
 * @details 包含框架所有核心接口，提供统一的入口点
 * @version 1.01.01
 * @date 2025-10-25
 * @author 技术觉醒团队
 * @note 依赖项: 无外部依赖
 * @note 所属系列: framework
 */

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <array>

// === 核心类头文件包含 ===
#include "tech_renaissance/data/dtype.h"
#include "tech_renaissance/data/device.h"
#include "tech_renaissance/data/shape.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/data/storage.h"
#include "tech_renaissance/backend/backend.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/backend/cuda/cuda_backend.h"
#include "tech_renaissance/model/module.h"
#include "tech_renaissance/model/linear.h"
#include "tech_renaissance/model/flatten.h"
#include "tech_renaissance/model/tanh.h"
#include "tech_renaissance/model/model.h"
#include "tech_renaissance/trainer/loss.h"
#include "tech_renaissance/trainer/cross_entropy_loss.h"
#include "tech_renaissance/utils/logger.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/utils/profiler.h"
#ifdef TR_BUILD_PYTHON_SESSION
#include "tech_renaissance/utils/python_session.h"
#endif

// 命名空间
namespace tr {

// === 张量工厂函数 ===
/**
 * @brief 创建全1张量
 * @param shape 张量形状
 * @param dtype 数据类型
 * @param device 设备
 * @return 张量对象
 */
Tensor full(const Shape& shape, float value, DType dtype = DType::FP32, const Device& device = CPU);

/**
 * @brief 创建空张量
 * @param shape 张量形状
 * @param dtype 数据类型
 * @param device 设备
 * @return 张量对象
 */
Tensor empty(const Shape& shape, DType dtype = DType::FP32, const Device& device = CPU);

// === 便捷工具函数 ===
namespace utils {
    /**
     * @brief 获取数据类型大小
     * @param dtype 数据类型
     * @return 字节数
     */
    inline size_t dtype_size(DType dtype) {
        return ::tr::dtype_size(dtype);
    }

    /**
     * @brief 数据类型转字符串
     * @param dtype 数据类型
     * @return 字符串表示
     */
    inline std::string dtype_to_string(DType dtype) {
        return ::tr::dtype_to_string(dtype);
    }

    /**
     * @brief 字符串转数据类型
     * @param str 字符串
     * @return 数据类型
     */
    inline DType string_to_dtype(const std::string& str) {
        return ::tr::string_to_dtype(str);
    }
}

} // namespace tr
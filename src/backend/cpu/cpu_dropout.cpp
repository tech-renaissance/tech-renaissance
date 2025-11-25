/**
 * @file cpu_dropout.cpp
 * @brief CPU后端Dropout操作实现
 * @details 实现Dropout正向运算，支持randbool mask生成和逐元素乘法
 * @version 1.00.00
 * @date 2025-11-25
 * @author 技术觉醒团队
 * @note 依赖项: cpu_backend.h, tensor.h
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/utils/logger.h"

#include <random>
#include <cstring>
#include <algorithm>

namespace tr {

void CpuBackend::dropout_into(const Tensor& input, Tensor& mask, Tensor& result, float p) {
    if (input.dtype() != DType::FP32) {
        throw TypeError("[CpuBackend::dropout_into] Input tensor must be FP32 type");
    }
    if (mask.dtype() != DType::FP32) {
        throw TypeError("[CpuBackend::dropout_into] Mask tensor must be FP32 type");
    }
    if (result.dtype() != DType::FP32) {
        throw TypeError("[CpuBackend::dropout_into] Result tensor must be FP32 type");
    }
    validate_same_device(input.device());
    validate_same_device(mask.device());
    validate_same_device(result.device());
    if (input.shape() != mask.shape() || input.shape() != result.shape()) {
        throw ShapeError("[CpuBackend::dropout_into] Input, mask and result tensors must have the same shape. "
                       "Input: " + input.shape().to_string() +
                       ", Mask: " + mask.shape().to_string() +
                       ", Result: " + result.shape().to_string());
    }
    if (input.is_empty()) {
        throw ShapeError("[CpuBackend::dropout_into] Input tensor is empty");
    }
    if (mask.is_empty()) {
        throw ShapeError("[CpuBackend::dropout_into] Mask tensor is empty");
    }
    if (result.is_empty()) {
        throw ShapeError("[CpuBackend::dropout_into] Result tensor is empty");
    }
    if (p < 0.0f || p > 1.0f) {
        throw ValueError("[CpuBackend::dropout_into] Dropout probability p must be between 0.0 and 1.0, got: " + std::to_string(p));
    }
    float factor = 1.0f / (1.0f - p);
    randbool_inplace(mask, p);
    mul_into(input, mask, result);
    mul_inplace(result, factor);
}

void CpuBackend::ddropout_into(const Tensor& input, const Tensor& mask, Tensor& result, float p) {
    float factor = 1.0f / (1.0f - p);
    mul_into(input, mask, result);
    mul_inplace(result, factor);
}

} // namespace tr
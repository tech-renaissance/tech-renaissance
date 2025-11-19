/**
 * @file cpu_basic_ops.cpp
 * @brief CPU后端基本运算实现
 * @details 实现张量间的基本运算（加法、乘法）
 * @version 1.51.0
 * @date 2025-11-19
 * @author 技术觉醒团队
 * @note 依赖项: cpu_backend.h, tensor.h, Eigen
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/utils/logger.h"

#ifdef TR_USE_EIGEN
#include "Core"
#endif

namespace tr {

// ===== 张量间加法运算 =====

Tensor CpuBackend::add(const Tensor& a, const Tensor& b) const {
    validate_same_device(a.device());
    validate_same_device(b.device());

    if (a.is_empty()) {
        throw TRException("[CpuBackend::add] First input tensor has no allocated Storage");
    }
    if (b.is_empty()) {
        throw TRException("[CpuBackend::add] Second input tensor has no allocated Storage");
    }

    if (a.shape() != b.shape()) {
        throw TRException("[CpuBackend::add] Shape mismatch: tensor_a shape " +
                        a.shape().to_string() + " vs tensor_b shape " +
                        b.shape().to_string());
    }

    if (a.dtype() != b.dtype()) {
        throw TRException("[CpuBackend::add] Data type mismatch: tensor_a dtype " +
                        dtype_to_string(a.dtype()) + " vs tensor_b dtype " +
                        dtype_to_string(b.dtype()));
    }

    if (a.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::add] Only FP32 tensors are supported for tensor addition. "
                        "TODO: Consider implementing INT8 tensor operations in future versions.");
    }

    // 创建输出张量
    Tensor result = this->empty(a.shape(), a.dtype());
    add_into(a, b, result);
    return result;
}

// ===== 张量间乘法运算 =====

Tensor CpuBackend::mul(const Tensor& a, const Tensor& b) const {
    validate_same_device(a.device());
    validate_same_device(b.device());

    if (a.is_empty()) {
        throw TRException("[CpuBackend::mul] First input tensor has no allocated Storage");
    }
    if (b.is_empty()) {
        throw TRException("[CpuBackend::mul] Second input tensor has no allocated Storage");
    }

    if (a.shape() != b.shape()) {
        throw TRException("[CpuBackend::mul] Shape mismatch: tensor_a shape " +
                        a.shape().to_string() + " vs tensor_b shape " +
                        b.shape().to_string());
    }

    if (a.dtype() != b.dtype()) {
        throw TRException("[CpuBackend::mul] Data type mismatch: tensor_a dtype " +
                        dtype_to_string(a.dtype()) + " vs tensor_b dtype " +
                        dtype_to_string(b.dtype()));
    }

    if (a.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::mul] Only FP32 tensors are supported for tensor multiplication. "
                        "TODO: Consider implementing INT8 tensor operations in future versions.");
    }

    // 创建输出张量
    Tensor result = this->empty(a.shape(), a.dtype());
    mul_into(a, b, result);
    return result;
}

} // namespace tr
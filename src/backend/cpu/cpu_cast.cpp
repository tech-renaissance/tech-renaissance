/**
 * @file cpu_cast.cpp
 * @brief CPU后端类型转换函数实现
 * @details 实现张量数据类型的转换功能，包括FP32、INT8、INT32之间的相互转换，支持非原地、原地和指定输出三种模式
 * @version 1.31.2
 * @date 2025-11-02
 * @author 技术觉醒团队
 * @note 依赖项: tensor.h, dtype.h, eigen/Eigen
 * @note 所属系列: backend
 */

#include <stdexcept>
#include <algorithm>

#ifdef TR_USE_EIGEN
#include "Core"
#endif

#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/data/dtype.h"
#include "tech_renaissance/utils/tr_exception.h"

namespace tr {

Tensor CpuBackend::cast(const Tensor& tensor_a, DType target_dtype) {
    // 验证输入张量和目标类型
    if (tensor_a.is_empty()) {
        throw TRException("[CpuBackend::cast] Input tensor cannot be empty");
    }

    if (tensor_a.dtype() == target_dtype) {
        throw TRException("[CpuBackend::cast] Input tensor already has the target dtype");
    }

    // 验证支持的类型转换
    if (tensor_a.dtype() != DType::FP32 && tensor_a.dtype() != DType::INT8 && tensor_a.dtype() != DType::INT32) {
        throw TRException("[CpuBackend::cast] Input tensor dtype must be FP32, INT8, or INT32");
    }

    if (target_dtype != DType::FP32 && target_dtype != DType::INT8 && target_dtype != DType::INT32) {
        throw TRException("[CpuBackend::cast] Target dtype must be FP32, INT8, or INT32");
    }

    // 创建结果张量
    Tensor result = this->empty(tensor_a.shape(), target_dtype);

    // 根据数据类型执行转换
    if (tensor_a.dtype() == DType::FP32 && target_dtype == DType::INT8) {
        // FP32 -> INT8
        const float* src_data = static_cast<const float*>(tensor_a.data_ptr());
        int8_t* dst_data = static_cast<int8_t*>(result.data_ptr());
        int64_t numel = tensor_a.numel();

      // Temporarily disable Eigen to avoid memory issues
        // Use naive implementation for safety
        for (int64_t i = 0; i < numel; ++i) {
            dst_data[i] = static_cast<int8_t>(std::clamp(src_data[i], -128.0f, 127.0f));
        }
    } else if (tensor_a.dtype() == DType::FP32 && target_dtype == DType::INT32) {
        // FP32 -> INT32
        const float* src_data = static_cast<const float*>(tensor_a.data_ptr());
        int32_t* dst_data = static_cast<int32_t*>(result.data_ptr());
        int64_t numel = tensor_a.numel();

    // Temporarily disable Eigen to avoid memory issues
        // Use naive implementation for safety
        for (int64_t i = 0; i < numel; ++i) {
            dst_data[i] = static_cast<int32_t>(src_data[i]);
        }

    } else if (tensor_a.dtype() == DType::INT8 && target_dtype == DType::FP32) {
        // INT8 -> FP32
        const int8_t* src_data = static_cast<const int8_t*>(tensor_a.data_ptr());
        float* dst_data = static_cast<float*>(result.data_ptr());
        int64_t numel = tensor_a.numel();

// Use naive implementation for safety
        for (int64_t i = 0; i < numel; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]);
        }

    } else if (tensor_a.dtype() == DType::INT8 && target_dtype == DType::INT32) {
        // INT8 -> INT32
        const int8_t* src_data = static_cast<const int8_t*>(tensor_a.data_ptr());
        int32_t* dst_data = static_cast<int32_t*>(result.data_ptr());
        int64_t numel = tensor_a.numel();

// Use naive implementation for safety
        for (int64_t i = 0; i < numel; ++i) {
            dst_data[i] = static_cast<int32_t>(src_data[i]);
        }

    } else if (tensor_a.dtype() == DType::INT32 && target_dtype == DType::FP32) {
        // INT32 -> FP32
        const int32_t* src_data = static_cast<const int32_t*>(tensor_a.data_ptr());
        float* dst_data = static_cast<float*>(result.data_ptr());
        int64_t numel = tensor_a.numel();

// Use naive implementation for safety
        for (int64_t i = 0; i < numel; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]);
        }

    } else if (tensor_a.dtype() == DType::INT32 && target_dtype == DType::INT8) {
        // INT32 -> INT8
        const int32_t* src_data = static_cast<const int32_t*>(tensor_a.data_ptr());
        int8_t* dst_data = static_cast<int8_t*>(result.data_ptr());
        int64_t numel = tensor_a.numel();

// Use naive implementation for safety
        for (int64_t i = 0; i < numel; ++i) {
            dst_data[i] = static_cast<int8_t>(std::clamp(src_data[i], -128, 127));
        }
    }

    return result;
}

void CpuBackend::cast_inplace(Tensor& tensor_a, DType target_dtype) {
    // 验证输入张量和目标类型
    if (tensor_a.is_empty()) {
        throw TRException("[CpuBackend::cast_inplace] Input tensor cannot be empty");
    }

    if (tensor_a.dtype() == target_dtype) {
        throw TRException("[CpuBackend::cast_inplace] Input tensor already has the target dtype");
    }

    // 验证支持的类型转换（不包括INT8）
    if (tensor_a.dtype() != DType::FP32 && tensor_a.dtype() != DType::INT32) {
        throw TRException("[CpuBackend::cast_inplace] Input tensor dtype must be FP32 or INT32");
    }

    if (target_dtype != DType::FP32 && target_dtype != DType::INT32) {
        throw TRException("[CpuBackend::cast_inplace] Target dtype must be FP32 or INT32");
    }

    if (tensor_a.dtype() == DType::INT8 || target_dtype == DType::INT8) {
        throw TRException("[CpuBackend::cast_inplace] Cannot cast to/from INT8 in-place");
    }

    // 创建临时张量进行转换
    Tensor temp_result = this->cast(tensor_a, target_dtype);

    // 将结果复制回原张量
    this->copy_into(temp_result, tensor_a);
}

void CpuBackend::cast_into(const Tensor& tensor_a, DType target_dtype, Tensor& result) {
    // 验证输入张量和目标类型
    if (tensor_a.is_empty()) {
        throw TRException("[CpuBackend::cast_into] Input tensor cannot be empty");
    }

    if (tensor_a.dtype() == target_dtype) {
        throw TRException("[CpuBackend::cast_into] Input tensor already has the target dtype");
    }

    if (result.is_empty()) {
        throw TRException("[CpuBackend::cast_into] Result tensor cannot be empty");
    }

    // 验证类型转换支持
    if (tensor_a.dtype() != DType::FP32 && tensor_a.dtype() != DType::INT8 && tensor_a.dtype() != DType::INT32) {
        throw TRException("[CpuBackend::cast_into] Input tensor dtype must be FP32, INT8, or INT32");
    }

    if (target_dtype != DType::FP32 && target_dtype != DType::INT8 && target_dtype != DType::INT32) {
        throw TRException("[CpuBackend::cast_into] Target dtype must be FP32, INT8, or INT32");
    }

    if (result.dtype() != target_dtype) {
        throw TRException("[CpuBackend::cast_into] Result tensor dtype must match target_dtype");
    }

    // 验证形状兼容性
    if (tensor_a.shape() != result.shape()) {
        throw TRException("[CpuBackend::cast_into] Input and result tensors must have the same shape");
    }

    // 验证设备兼容性
    if (tensor_a.device() != result.device()) {
        throw TRException("[CpuBackend::cast_into] Input and result tensors must be on the same device");
    }

    // 执行类型转换
    Tensor temp_result = this->cast(tensor_a, target_dtype);

    // 将结果复制到目标张量
    this->copy_into(temp_result, result);
}

} // namespace tr
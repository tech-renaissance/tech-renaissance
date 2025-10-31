/**
 * @file cpu_unary.cpp
 * @brief CPU后端单目运算实现
 * @details 实现9种单目运算，每种都包含非原地和原地两个版本
 * @version 1.00.00
 * @date 2025-10-31
 * @author 技术觉醒团队
 * @note 依赖项: cpu_backend.h, tensor.h
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/utils/logger.h"
#include <cmath>
#include <cstring>

// NaN检查宏配置
#ifndef TR_ENABLE_NAN_CHECK
#define TR_ENABLE_NAN_CHECK 1  // 默认检查并报错
#endif

// _into函数形状检查宏配置
#ifndef TR_ENABLE_INTO_FUNC_SHAPE_CHECK
#define TR_ENABLE_INTO_FUNC_SHAPE_CHECK 1  // 默认检查形状
#endif

// eps常量，用于处理除零等特殊情况
constexpr float TR_EPS = 1e-10f;

namespace tr {

// ===== 非原地运算函数 =====

Tensor CpuBackend::zeros_like(const Tensor& input) const {
    validate_same_device(input.device());

    // 创建与输入相同形状和类型的零张量
    Tensor result = Tensor::empty(input.shape(), input.dtype(), tr::CPU);

    // 使用memset高效填充零值
    void* data = result.data_ptr();
    size_t total_bytes = result.numel() * dtype_size(result.dtype());
    std::memset(data, 0, total_bytes);

    return result;
}

Tensor CpuBackend::ones_like(const Tensor& input) const {
    validate_same_device(input.device());

    // 创建与输入相同形状和类型的1张量
    Tensor result = Tensor::empty(input.shape(), input.dtype(), tr::CPU);

    // 手动填充1值
    if (result.dtype() == DType::FP32) {
        float* data = static_cast<float*>(result.data_ptr());
        size_t num_elements = result.numel();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = 1.0f;
        }
    } else if (result.dtype() == DType::INT8) {
        int8_t* data = static_cast<int8_t*>(result.data_ptr());
        size_t num_elements = result.numel();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = 1;
        }
    }

    return result;
}

Tensor CpuBackend::relu(const Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::relu] Only FP32 tensors are supported");
    }

    // 创建结果张量
    Tensor result = Tensor::empty(input.shape(), input.dtype(), tr::CPU);

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    size_t num_elements = input.numel();

    // ReLU: max(0, x)
    for (size_t i = 0; i < num_elements; ++i) {
        result_data[i] = (input_data[i] > 0.0f) ? input_data[i] : 0.0f;
    }

    return result;
}

Tensor CpuBackend::sign(const Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::sign] Only FP32 tensors are supported");
    }

    // 创建结果张量
    Tensor result = Tensor::empty(input.shape(), input.dtype(), tr::CPU);

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    size_t num_elements = input.numel();

    // Sign函数
    for (size_t i = 0; i < num_elements; ++i) {
        if (input_data[i] > 0.0f) {
            result_data[i] = 1.0f;
        } else if (input_data[i] < 0.0f) {
            result_data[i] = -1.0f;
        } else {
            result_data[i] = 0.0f;
        }
    }

    return result;
}

Tensor CpuBackend::square(const Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::square] Only FP32 tensors are supported");
    }

    // 创建结果张量
    Tensor result = Tensor::empty(input.shape(), input.dtype(), tr::CPU);

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    size_t num_elements = input.numel();

    // 平方运算
    for (size_t i = 0; i < num_elements; ++i) {
        result_data[i] = input_data[i] * input_data[i];
    }

    return result;
}

Tensor CpuBackend::sqrt(const Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::sqrt] Only FP32 tensors are supported");
    }

    // 创建结果张量
    Tensor result = Tensor::empty(input.shape(), input.dtype(), tr::CPU);

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    size_t num_elements = input.numel();

    // 平方根运算
    for (size_t i = 0; i < num_elements; ++i) {
        if (input_data[i] < 0.0f) {
#if TR_ENABLE_NAN_CHECK == 0
            // 不检查，直接计算（会产生NaN）
            result_data[i] = std::sqrt(input_data[i]);
#elif TR_ENABLE_NAN_CHECK == 1
            // 检查并报错
            throw TRException("[CpuBackend::sqrt] Negative input encountered: " + std::to_string(input_data[i]));
#else
            // 检查并替换为0
            result_data[i] = 0.0f;
#endif
        } else {
            result_data[i] = std::sqrt(input_data[i]);
        }
    }

    return result;
}

Tensor CpuBackend::abs(const Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::abs] Only FP32 tensors are supported");
    }

    // 创建结果张量
    Tensor result = Tensor::empty(input.shape(), input.dtype(), tr::CPU);

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    size_t num_elements = input.numel();

    // 绝对值运算
    for (size_t i = 0; i < num_elements; ++i) {
        result_data[i] = (input_data[i] < 0.0f) ? -input_data[i] : input_data[i];
    }

    return result;
}

Tensor CpuBackend::negative(const Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::negative] Only FP32 tensors are supported");
    }

    // 创建结果张量
    Tensor result = Tensor::empty(input.shape(), input.dtype(), tr::CPU);

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    size_t num_elements = input.numel();

    // 相反数运算
    for (size_t i = 0; i < num_elements; ++i) {
        result_data[i] = -input_data[i];
    }

    return result;
}

Tensor CpuBackend::reciprocal(const Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::reciprocal] Only FP32 tensors are supported");
    }

    // 创建结果张量
    Tensor result = Tensor::empty(input.shape(), input.dtype(), tr::CPU);

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    size_t num_elements = input.numel();

    // 倒数运算
    for (size_t i = 0; i < num_elements; ++i) {
        if (std::abs(input_data[i]) < TR_EPS) {
#if TR_ENABLE_NAN_CHECK == 0
            // 不检查，直接计算（会产生inf）
            result_data[i] = 1.0f / input_data[i];
#elif TR_ENABLE_NAN_CHECK == 1
            // 检查并报错
            throw TRException("[CpuBackend::reciprocal] Division by zero encountered: " + std::to_string(input_data[i]));
#else
            // 检查并替换为1/eps
            result_data[i] = 1.0f / TR_EPS;
#endif
        } else {
            result_data[i] = 1.0f / input_data[i];
        }
    }

    return result;
}

// ===== 原地运算函数 =====

void CpuBackend::zeros_inplace(Tensor& input) const {
    validate_same_device(input.device());

    // 使用memset高效填充零值
    void* data = input.data_ptr();
    size_t total_bytes = input.numel() * dtype_size(input.dtype());
    std::memset(data, 0, total_bytes);
}

void CpuBackend::ones_inplace(Tensor& input) const {
    validate_same_device(input.device());

    // 手动填充1值（memset不适用于设置数值1，因为字节全1在不同数据类型中表示的值不同）
    if (input.dtype() == DType::FP32) {
        float* data = static_cast<float*>(input.data_ptr());
        size_t num_elements = input.numel();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = 1.0f;
        }
    } else if (input.dtype() == DType::INT8) {
        int8_t* data = static_cast<int8_t*>(input.data_ptr());
        size_t num_elements = input.numel();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = 1;
        }
    }
}

void CpuBackend::relu_inplace(Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::relu_inplace] Only FP32 tensors are supported");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t num_elements = input.numel();

    // ReLU: max(0, x)
    for (size_t i = 0; i < num_elements; ++i) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

void CpuBackend::sign_inplace(Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::sign_inplace] Only FP32 tensors are supported");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t num_elements = input.numel();

    // Sign函数
    for (size_t i = 0; i < num_elements; ++i) {
        if (data[i] > 0.0f) {
            data[i] = 1.0f;
        } else if (data[i] < 0.0f) {
            data[i] = -1.0f;
        } else {
            data[i] = 0.0f;
        }
    }
}

void CpuBackend::square_inplace(Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::square_inplace] Only FP32 tensors are supported");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t num_elements = input.numel();

    // 平方运算
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = data[i] * data[i];
    }
}

void CpuBackend::sqrt_inplace(Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::sqrt_inplace] Only FP32 tensors are supported");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t num_elements = input.numel();

    // 平方根运算
    for (size_t i = 0; i < num_elements; ++i) {
        if (data[i] < 0.0f) {
#if TR_ENABLE_NAN_CHECK == 0
            // 不检查，直接计算（会产生NaN）
            data[i] = std::sqrt(data[i]);
#elif TR_ENABLE_NAN_CHECK == 1
            // 检查并报错
            throw TRException("[CpuBackend::sqrt_inplace] Negative input encountered: " + std::to_string(data[i]));
#else
            // 检查并替换为0
            data[i] = 0.0f;
#endif
        } else {
            data[i] = std::sqrt(data[i]);
        }
    }
}

void CpuBackend::abs_inplace(Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::abs_inplace] Only FP32 tensors are supported");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t num_elements = input.numel();

    // 绝对值运算
    for (size_t i = 0; i < num_elements; ++i) {
        if (data[i] < 0.0f) {
            data[i] = -data[i];
        }
    }
}

void CpuBackend::negative_inplace(Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::negative_inplace] Only FP32 tensors are supported");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t num_elements = input.numel();

    // 相反数运算
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = -data[i];
    }
}

void CpuBackend::reciprocal_inplace(Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::reciprocal_inplace] Only FP32 tensors are supported");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t num_elements = input.numel();

    // 倒数运算
    for (size_t i = 0; i < num_elements; ++i) {
        if (std::abs(data[i]) < TR_EPS) {
#if TR_ENABLE_NAN_CHECK == 0
            // 不检查，直接计算（会产生inf）
            data[i] = 1.0f / data[i];
#elif TR_ENABLE_NAN_CHECK == 1
            // 检查并报错
            throw TRException("[CpuBackend::reciprocal_inplace] Division by zero encountered: " + std::to_string(data[i]));
#else
            // 检查并替换为1/eps
            data[i] = 1.0f / TR_EPS;
#endif
        } else {
            data[i] = 1.0f / data[i];
        }
    }
}

// ===== 指定输出张量的运算函数 =====

void CpuBackend::zeros_into(const Tensor& input, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

#if TR_ENABLE_INTO_FUNC_SHAPE_CHECK == 1
    // 检查输出张量形状和类型是否与输入一致
    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::zeros_into] Shape mismatch: input shape " +
                         input.shape().to_string() + " != output shape " +
                         output.shape().to_string());
    }
    if (input.dtype() != output.dtype()) {
        throw TRException("[CpuBackend::zeros_into] Dtype mismatch: input dtype " +
                         std::to_string(static_cast<int>(input.dtype())) +
                         " != output dtype " + std::to_string(static_cast<int>(output.dtype())));
    }
#endif

    // 使用memset高效填充零值
    void* data = output.data_ptr();
    size_t total_bytes = output.numel() * dtype_size(output.dtype());
    std::memset(data, 0, total_bytes);
}

void CpuBackend::ones_into(const Tensor& input, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

#if TR_ENABLE_INTO_FUNC_SHAPE_CHECK == 1
    // 检查输出张量形状和类型是否与输入一致
    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::ones_into] Shape mismatch: input shape " +
                         input.shape().to_string() + " != output shape " +
                         output.shape().to_string());
    }
    if (input.dtype() != output.dtype()) {
        throw TRException("[CpuBackend::ones_into] Dtype mismatch: input dtype " +
                         std::to_string(static_cast<int>(input.dtype())) +
                         " != output dtype " + std::to_string(static_cast<int>(output.dtype())));
    }
#endif

    // 手动填充1值
    if (output.dtype() == DType::FP32) {
        float* data = static_cast<float*>(output.data_ptr());
        size_t num_elements = output.numel();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = 1.0f;
        }
    } else if (output.dtype() == DType::INT8) {
        int8_t* data = static_cast<int8_t*>(output.data_ptr());
        size_t num_elements = output.numel();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = 1;
        }
    }
}

void CpuBackend::relu_into(const Tensor& input, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::relu_into] Only FP32 tensors are supported");
    }

#if TR_ENABLE_INTO_FUNC_SHAPE_CHECK == 1
    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::relu_into] Shape mismatch: input shape " +
                         input.shape().to_string() + " != output shape " +
                         output.shape().to_string());
    }
#endif

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());

    size_t num_elements = input.numel();

    // ReLU: max(0, x)
    for (size_t i = 0; i < num_elements; ++i) {
        output_data[i] = (input_data[i] > 0.0f) ? input_data[i] : 0.0f;
    }
}

void CpuBackend::sign_into(const Tensor& input, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::sign_into] Only FP32 tensors are supported");
    }

#if TR_ENABLE_INTO_FUNC_SHAPE_CHECK == 1
    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::sign_into] Shape mismatch: input shape " +
                         input.shape().to_string() + " != output shape " +
                         output.shape().to_string());
    }
#endif

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());

    size_t num_elements = input.numel();

    // Sign函数
    for (size_t i = 0; i < num_elements; ++i) {
        if (input_data[i] > 0.0f) {
            output_data[i] = 1.0f;
        } else if (input_data[i] < 0.0f) {
            output_data[i] = -1.0f;
        } else {
            output_data[i] = 0.0f;
        }
    }
}

void CpuBackend::square_into(const Tensor& input, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::square_into] Only FP32 tensors are supported");
    }

#if TR_ENABLE_INTO_FUNC_SHAPE_CHECK == 1
    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::square_into] Shape mismatch: input shape " +
                         input.shape().to_string() + " != output shape " +
                         output.shape().to_string());
    }
#endif

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());

    size_t num_elements = input.numel();

    // 平方运算
    for (size_t i = 0; i < num_elements; ++i) {
        output_data[i] = input_data[i] * input_data[i];
    }
}

void CpuBackend::sqrt_into(const Tensor& input, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::sqrt_into] Only FP32 tensors are supported");
    }

#if TR_ENABLE_INTO_FUNC_SHAPE_CHECK == 1
    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::sqrt_into] Shape mismatch: input shape " +
                         input.shape().to_string() + " != output shape " +
                         output.shape().to_string());
    }
#endif

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());

    size_t num_elements = input.numel();

    // 平方根运算
    for (size_t i = 0; i < num_elements; ++i) {
        if (input_data[i] < 0.0f) {
#if TR_ENABLE_NAN_CHECK == 0
            // 不检查，直接计算（会产生NaN）
            output_data[i] = std::sqrt(input_data[i]);
#elif TR_ENABLE_NAN_CHECK == 1
            // 检查并报错
            throw TRException("[CpuBackend::sqrt_into] Negative input encountered: " + std::to_string(input_data[i]));
#else
            // 检查并替换为0
            output_data[i] = 0.0f;
#endif
        } else {
            output_data[i] = std::sqrt(input_data[i]);
        }
    }
}

void CpuBackend::abs_into(const Tensor& input, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::abs_into] Only FP32 tensors are supported");
    }

#if TR_ENABLE_INTO_FUNC_SHAPE_CHECK == 1
    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::abs_into] Shape mismatch: input shape " +
                         input.shape().to_string() + " != output shape " +
                         output.shape().to_string());
    }
#endif

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());

    size_t num_elements = input.numel();

    // 绝对值运算
    for (size_t i = 0; i < num_elements; ++i) {
        output_data[i] = (input_data[i] < 0.0f) ? -input_data[i] : input_data[i];
    }
}

void CpuBackend::negative_into(const Tensor& input, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::negative_into] Only FP32 tensors are supported");
    }

#if TR_ENABLE_INTO_FUNC_SHAPE_CHECK == 1
    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::negative_into] Shape mismatch: input shape " +
                         input.shape().to_string() + " != output shape " +
                         output.shape().to_string());
    }
#endif

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());

    size_t num_elements = input.numel();

    // 相反数运算
    for (size_t i = 0; i < num_elements; ++i) {
        output_data[i] = -input_data[i];
    }
}

void CpuBackend::reciprocal_into(const Tensor& input, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::reciprocal_into] Only FP32 tensors are supported");
    }

#if TR_ENABLE_INTO_FUNC_SHAPE_CHECK == 1
    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::reciprocal_into] Shape mismatch: input shape " +
                         input.shape().to_string() + " != output shape " +
                         output.shape().to_string());
    }
#endif

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());

    size_t num_elements = input.numel();

    // 倒数运算
    for (size_t i = 0; i < num_elements; ++i) {
        if (std::abs(input_data[i]) < TR_EPS) {
#if TR_ENABLE_NAN_CHECK == 0
            // 不检查，直接计算（会产生inf）
            output_data[i] = 1.0f / input_data[i];
#elif TR_ENABLE_NAN_CHECK == 1
            // 检查并报错
            throw TRException("[CpuBackend::reciprocal_into] Division by zero encountered: " + std::to_string(input_data[i]));
#else
            // 检查并替换为1/eps
            output_data[i] = 1.0f / TR_EPS;
#endif
        } else {
            output_data[i] = 1.0f / input_data[i];
        }
    }
}

} // namespace tr
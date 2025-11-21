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

#ifdef TR_USE_EIGEN
#include "Core"
#endif

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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    if (result.dtype() == DType::FP32) {
        float* data = static_cast<float*>(result.data_ptr());
        size_t num_elements = result.numel();
        using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
        Eigen::Map<MatrixType> eigen_vec(data, num_elements);
        eigen_vec.setConstant(1.0f);
    } else if (result.dtype() == DType::INT8) {
        int8_t* data = static_cast<int8_t*>(result.data_ptr());
        size_t num_elements = result.numel();
        using MatrixType = Eigen::Matrix<int8_t, Eigen::Dynamic, 1>;
        Eigen::Map<MatrixType> eigen_vec(data, num_elements);
        eigen_vec.setConstant(1);
    }
#else
    // 朴素实现
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
#endif

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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> result_vec(result_data, num_elements);
    result_vec = input_vec.cwiseMax(0.0f);
#else
    // 朴素实现
    // ReLU: max(0, x)
    for (size_t i = 0; i < num_elements; ++i) {
        result_data[i] = (input_data[i] > 0.0f) ? input_data[i] : 0.0f;
    }
#endif

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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> result_vec(result_data, num_elements);

    // 使用Eigen的signum函数
    result_vec = input_vec.array().sign().matrix();
#else
    // 朴素实现
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
#endif

    return result;
}

Tensor CpuBackend::square(const Tensor& input) {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::square] Only FP32 tensors are supported");
    }

    // 创建结果张量
    Tensor result = Tensor::empty(input.shape(), input.dtype(), tr::CPU);

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    size_t num_elements = input.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> result_vec(result_data, num_elements);
    result_vec = input_vec.array().square().matrix();
#else
    // 朴素实现
    // 平方运算
    for (size_t i = 0; i < num_elements; ++i) {
        result_data[i] = input_data[i] * input_data[i];
    }
#endif

    return result;
}

Tensor CpuBackend::sqrt(const Tensor& input) {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::sqrt] Only FP32 tensors are supported");
    }

    // 创建结果张量
    Tensor result = Tensor::empty(input.shape(), input.dtype(), tr::CPU);

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    size_t num_elements = input.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> result_vec(result_data, num_elements);

    // 检查负数并处理
    bool has_negative = false;
    for (size_t i = 0; i < num_elements; ++i) {
        if (input_data[i] < 0.0f) {
            has_negative = true;
            break;
        }
    }

    if (has_negative) {
        // 有负数，需要逐个处理
        for (size_t i = 0; i < num_elements; ++i) {
            if (input_data[i] < 0.0f) {
#if TR_ENABLE_NAN_CHECK == 0
                result_data[i] = std::sqrt(input_data[i]);
#elif TR_ENABLE_NAN_CHECK == 1
                throw TRException("[CpuBackend::sqrt] Negative input encountered: " + std::to_string(input_data[i]));
#else
                result_data[i] = 0.0f;
#endif
            } else {
                result_data[i] = std::sqrt(input_data[i]);
            }
        }
    } else {
        // 无负数，可以使用Eigen的sqrt
        result_vec = input_vec.array().sqrt().matrix();
    }
#else
    // 朴素实现
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
#endif

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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> result_vec(result_data, num_elements);
    result_vec = input_vec.array().abs().matrix();
#else
    // 朴素实现
    // 绝对值运算
    for (size_t i = 0; i < num_elements; ++i) {
        result_data[i] = (input_data[i] < 0.0f) ? -input_data[i] : input_data[i];
    }
#endif

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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> result_vec(result_data, num_elements);
    result_vec = -input_vec;
#else
    // 朴素实现
    // 相反数运算
    for (size_t i = 0; i < num_elements; ++i) {
        result_data[i] = -input_data[i];
    }
#endif

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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> result_vec(result_data, num_elements);

    // 检查零值并处理
    bool has_zero = false;
    for (size_t i = 0; i < num_elements; ++i) {
        if (std::abs(input_data[i]) < TR_EPS) {
            has_zero = true;
            break;
        }
    }

    if (has_zero) {
        // 有零值，需要逐个处理
        for (size_t i = 0; i < num_elements; ++i) {
            if (std::abs(input_data[i]) < TR_EPS) {
#if TR_ENABLE_NAN_CHECK == 0
                result_data[i] = 1.0f / input_data[i];
#elif TR_ENABLE_NAN_CHECK == 1
                throw TRException("[CpuBackend::reciprocal] Division by zero encountered: " + std::to_string(input_data[i]));
#else
                result_data[i] = 1.0f / TR_EPS;
#endif
            } else {
                result_data[i] = 1.0f / input_data[i];
            }
        }
    } else {
        // 无零值，可以使用Eigen的倒数运算
        result_vec = input_vec.array().inverse().matrix();
    }
#else
    // 朴素实现
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
#endif

    return result;
}

Tensor CpuBackend::round(const Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::round] Only FP32 tensors are supported");
    }

    // 创建结果张量
    Tensor result = Tensor::empty(input.shape(), input.dtype(), tr::CPU);

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    size_t num_elements = input.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> result_vec(result_data, num_elements);

    // 使用Eigen的round函数
    result_vec = input_vec.array().round().matrix();
#else
    // 朴素实现
    // 四舍五入运算
    for (size_t i = 0; i < num_elements; ++i) {
        result_data[i] = std::round(input_data[i]);
    }
#endif

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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    if (input.dtype() == DType::FP32) {
        float* data = static_cast<float*>(input.data_ptr());
        size_t num_elements = input.numel();
        using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
        Eigen::Map<MatrixType> eigen_vec(data, num_elements);
        eigen_vec.setConstant(1.0f);
    } else if (input.dtype() == DType::INT8) {
        int8_t* data = static_cast<int8_t*>(input.data_ptr());
        size_t num_elements = input.numel();
        using MatrixType = Eigen::Matrix<int8_t, Eigen::Dynamic, 1>;
        Eigen::Map<MatrixType> eigen_vec(data, num_elements);
        eigen_vec.setConstant(1);
    }
#else
    // 朴素实现
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
#endif
}

void CpuBackend::relu_inplace(Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::relu_inplace] Only FP32 tensors are supported");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t num_elements = input.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<MatrixType> eigen_vec(data, num_elements);
    eigen_vec = eigen_vec.cwiseMax(0.0f);
#else
    // 朴素实现
    // ReLU: max(0, x)
    for (size_t i = 0; i < num_elements; ++i) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
#endif
}

void CpuBackend::sign_inplace(Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::sign_inplace] Only FP32 tensors are supported");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t num_elements = input.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<MatrixType> eigen_vec(data, num_elements);
    eigen_vec = eigen_vec.array().sign().matrix();
#else
    // 朴素实现
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
#endif
}

void CpuBackend::square_inplace(Tensor& input) {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::square_inplace] Only FP32 tensors are supported");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t num_elements = input.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<MatrixType> eigen_vec(data, num_elements);
    eigen_vec = eigen_vec.array().square().matrix();
#else
    // 朴素实现
    // 平方运算
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = data[i] * data[i];
    }
#endif
}

void CpuBackend::sqrt_inplace(Tensor& input) {
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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<MatrixType> eigen_vec(data, num_elements);
    eigen_vec = eigen_vec.array().abs().matrix();
#else
    // 朴素实现
    // 绝对值运算
    for (size_t i = 0; i < num_elements; ++i) {
        if (data[i] < 0.0f) {
            data[i] = -data[i];
        }
    }
#endif
}

void CpuBackend::negative_inplace(Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::negative_inplace] Only FP32 tensors are supported");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t num_elements = input.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<MatrixType> eigen_vec(data, num_elements);
    eigen_vec = -eigen_vec;
#else
    // 朴素实现
    // 相反数运算
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = -data[i];
    }
#endif
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

void CpuBackend::round_inplace(Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::round_inplace] Only FP32 tensors are supported");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t num_elements = input.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<MatrixType> eigen_vec(data, num_elements);
    eigen_vec = eigen_vec.array().round().matrix();
#else
    // 朴素实现
    // 四舍五入运算
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = std::round(data[i]);
    }
#endif
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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    if (output.dtype() == DType::FP32) {
        float* data = static_cast<float*>(output.data_ptr());
        size_t num_elements = output.numel();
        using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
        Eigen::Map<MatrixType> eigen_vec(data, num_elements);
        eigen_vec.setConstant(1.0f);
    } else if (output.dtype() == DType::INT8) {
        int8_t* data = static_cast<int8_t*>(output.data_ptr());
        size_t num_elements = output.numel();
        using MatrixType = Eigen::Matrix<int8_t, Eigen::Dynamic, 1>;
        Eigen::Map<MatrixType> eigen_vec(data, num_elements);
        eigen_vec.setConstant(1);
    }
#else
    // 朴素实现
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
#endif
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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> output_vec(output_data, num_elements);
    output_vec = input_vec.cwiseMax(0.0f);
#else
    // 朴素实现
    // ReLU: max(0, x)
    for (size_t i = 0; i < num_elements; ++i) {
        output_data[i] = (input_data[i] > 0.0f) ? input_data[i] : 0.0f;
    }
#endif
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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> output_vec(output_data, num_elements);
    output_vec = input_vec.array().sign().matrix();
#else
    // 朴素实现
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
#endif
}

void CpuBackend::square_into(const Tensor& input, Tensor& output) {
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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> output_vec(output_data, num_elements);
    output_vec = input_vec.array().square().matrix();
#else
    // 朴素实现
    // 平方运算
    for (size_t i = 0; i < num_elements; ++i) {
        output_data[i] = input_data[i] * input_data[i];
    }
#endif
}

void CpuBackend::sqrt_into(const Tensor& input, Tensor& output) {
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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> output_vec(output_data, num_elements);

    // 检查负数并处理
    bool has_negative = false;
    for (size_t i = 0; i < num_elements; ++i) {
        if (input_data[i] < 0.0f) {
            has_negative = true;
            break;
        }
    }

    if (has_negative) {
        // 有负数，需要逐个处理
        for (size_t i = 0; i < num_elements; ++i) {
            if (input_data[i] < 0.0f) {
#if TR_ENABLE_NAN_CHECK == 0
                output_data[i] = std::sqrt(input_data[i]);
#elif TR_ENABLE_NAN_CHECK == 1
                throw TRException("[CpuBackend::sqrt_into] Negative input encountered: " + std::to_string(input_data[i]));
#else
                output_data[i] = 0.0f;
#endif
            } else {
                output_data[i] = std::sqrt(input_data[i]);
            }
        }
    } else {
        // 无负数，可以使用Eigen的sqrt
        output_vec = input_vec.array().sqrt().matrix();
    }
#else
    // 朴素实现
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
#endif
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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> output_vec(output_data, num_elements);
    output_vec = input_vec.array().abs().matrix();
#else
    // 朴素实现
    // 绝对值运算
    for (size_t i = 0; i < num_elements; ++i) {
        output_data[i] = (input_data[i] < 0.0f) ? -input_data[i] : input_data[i];
    }
#endif
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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> output_vec(output_data, num_elements);
    output_vec = -input_vec;
#else
    // 朴素实现
    // 相反数运算
    for (size_t i = 0; i < num_elements; ++i) {
        output_data[i] = -input_data[i];
    }
#endif
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

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> output_vec(output_data, num_elements);

    // 检查零值并处理
    bool has_zero = false;
    for (size_t i = 0; i < num_elements; ++i) {
        if (std::abs(input_data[i]) < TR_EPS) {
            has_zero = true;
            break;
        }
    }

    if (has_zero) {
        // 有零值，需要逐个处理
        for (size_t i = 0; i < num_elements; ++i) {
            if (std::abs(input_data[i]) < TR_EPS) {
#if TR_ENABLE_NAN_CHECK == 0
                output_data[i] = 1.0f / input_data[i];
#elif TR_ENABLE_NAN_CHECK == 1
                throw TRException("[CpuBackend::reciprocal_into] Division by zero encountered: " + std::to_string(input_data[i]));
#else
                output_data[i] = 1.0f / TR_EPS;
#endif
            } else {
                output_data[i] = 1.0f / input_data[i];
            }
        }
    } else {
        // 无零值，可以使用Eigen的倒数运算
        output_vec = input_vec.array().inverse().matrix();
    }
#else
    // 朴素实现
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
#endif
}

void CpuBackend::round_into(const Tensor& input, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::round_into] Only FP32 tensors are supported");
    }

#if TR_ENABLE_INTO_FUNC_SHAPE_CHECK == 1
    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::round_into] Shape mismatch: input shape " +
                         input.shape().to_string() + " != output shape " +
                         output.shape().to_string());
    }
#endif

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());

    size_t num_elements = input.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> output_vec(output_data, num_elements);

    // 使用Eigen的round函数
    output_vec = input_vec.array().round().matrix();
#else
    // 朴素实现
    // 四舍五入运算
    for (size_t i = 0; i < num_elements; ++i) {
        output_data[i] = std::round(input_data[i]);
    }
#endif
}

// ===== 矩阵转置操作 =====

Tensor CpuBackend::transpose(const Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::transpose] Only FP32 tensors are supported");
    }

    // 检查是否为2D张量
    if (input.shape().ndim() != 2) {
        throw TRException("[CpuBackend::transpose] Only 2D tensors are supported, but got " +
                         std::to_string(input.shape().ndim()) + "D tensor");
    }

    // 获取维度信息
    int32_t height = input.shape().height();
    int32_t width = input.shape().width();

    // 创建转置后的张量（交换维度）
    Tensor result = Tensor::empty(Shape(width, height), DType::FP32, tr::CPU);

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    // 将输入映射为Eigen矩阵（行主序）
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const MatrixType> input_matrix(input_data, height, width);
    Eigen::Map<MatrixType> result_matrix(result_data, width, height);

    // 使用Eigen的转置功能
    result_matrix = input_matrix.transpose();
#else
    // 朴素实现
    // 对于行主序存储的矩阵，转置操作如下：
    // 原始矩阵 A[H,W]，行主序内存布局：[a_00, a_01, ..., a_0(W-1), a_10, a_11, ..., a_(H-1)(W-1)]
    // 转置矩阵 B[W,H]，行主序内存布局：[a_00, a_10, ..., a_(H-1)0, a_01, a_11, ..., a_(W-1)(H-1)]

    for (int32_t i = 0; i < height; ++i) {        // 原矩阵的行
        for (int32_t j = 0; j < width; ++j) {     // 原矩阵的列
            // A[i,j] -> B[j,i]
            result_data[j * height + i] = input_data[i * width + j];
        }
    }
#endif

    return result;
}

void CpuBackend::transpose_inplace(Tensor& input) const {
    validate_same_device(input.device());

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::transpose] Only FP32 tensors are supported");
    }

    // 检查是否为2D张量
    if (input.shape().ndim() != 2) {
        throw TRException("[CpuBackend::transpose] Only 2D tensors are supported, but got " +
                         std::to_string(input.shape().ndim()) + "D tensor");
    }

    // 获取维度信息
    int32_t height = input.shape().height();
    int32_t width = input.shape().width();

    // 创建转置后的张量（交换维度）
    Tensor result = Tensor::empty(Shape(width, height), DType::FP32, tr::CPU);

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    // 将输入映射为Eigen矩阵（行主序）
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const MatrixType> input_matrix(input_data, height, width);
    Eigen::Map<MatrixType> result_matrix(result_data, width, height);

    // 使用Eigen的转置功能
    result_matrix = input_matrix.transpose();
#else
    // 朴素实现
    // 对于行主序存储的矩阵，转置操作如下：
    // 原始矩阵 A[H,W]，行主序内存布局：[a_00, a_01, ..., a_0(W-1), a_10, a_11, ..., a_(H-1)(W-1)]
    // 转置矩阵 B[W,H]，行主序内存布局：[a_00, a_10, ..., a_(H-1)0, a_01, a_11, ..., a_(W-1)(H-1)]

    for (int32_t i = 0; i < height; ++i) {        // 原矩阵的行
        for (int32_t j = 0; j < width; ++j) {     // 原矩阵的列
            // A[i,j] -> B[j,i]
            result_data[j * height + i] = input_data[i * width + j];
        }
    }
#endif

    input = result;
}

void CpuBackend::transpose_into(const Tensor& input, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::transpose_into] Only FP32 tensors are supported");
    }

    // 检查输入是否为2D张量
    if (input.shape().ndim() != 2) {
        throw TRException("[CpuBackend::transpose_into] Only 2D tensors are supported for input, but got " +
                         std::to_string(input.shape().ndim()) + "D tensor");
    }

    // 检查输出是否为2D张量
    if (output.shape().ndim() != 2) {
        throw TRException("[CpuBackend::transpose_into] Only 2D tensors are supported for output, but got " +
                         std::to_string(output.shape().ndim()) + "D tensor");
    }

    // 获取输入输出维度信息
    int32_t input_height = input.shape().height();
    int32_t input_width = input.shape().width();
    int32_t output_height = output.shape().height();
    int32_t output_width = output.shape().width();

    // 检查维度是否匹配（输入 H,W -> 输出 W,H）
    if (output_height != input_width || output_width != input_height) {
        throw TRException("[CpuBackend::transpose_into] Dimension mismatch: input shape " +
                         input.shape().to_string() + " cannot be transposed to output shape " +
                         output.shape().to_string() + ". Expected output shape " +
                         Shape(input_width, input_height).to_string());
    }

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    // 将输入映射为Eigen矩阵（行主序）
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const MatrixType> input_matrix(input_data, input_height, input_width);
    Eigen::Map<MatrixType> output_matrix(output_data, output_height, output_width);

    // 使用Eigen的转置功能
    output_matrix = input_matrix.transpose();
#else
    // 朴素实现
    // 对于行主序存储的矩阵，转置操作如下：
    // 原始矩阵 A[H,W]，行主序内存布局：[a_00, a_01, ..., a_0(W-1), a_10, a_11, ..., a_(H-1)(W-1)]
    // 转置矩阵 B[W,H]，行主序内存布局：[a_00, a_10, ..., a_(H-1)0, a_01, a_11, ..., a_(W-1)(H-1)]

    for (int32_t i = 0; i < input_height; ++i) {        // 原矩阵的行
        for (int32_t j = 0; j < input_width; ++j) {     // 原矩阵的列
            // A[i,j] -> B[j,i]
            output_data[j * input_height + i] = input_data[i * input_width + j];
        }
    }
#endif
}

// ===== 张量复制操作 =====

Tensor CpuBackend::copy(const Tensor& tensor) const {
    // 检查源张量是否属于CPU后端
    validate_same_device(tensor.device());

    // 创建结果张量
    Tensor result = Tensor::empty(tensor.shape(), tensor.dtype(), tr::CPU);

    // 执行深拷贝
    copy_data(result.data_ptr(), tensor.data_ptr(), tensor.numel() * dtype_size(tensor.dtype()),
              tr::CPU, tr::CPU);

    return result;
}

void CpuBackend::copy_into(const Tensor& src, Tensor& dst) const {
    validate_same_device(src.device());
    validate_same_device(dst.device());

    // ✅ 自我拷贝检测
    if (src.storage() == dst.storage() &&
        src.data_ptr() == dst.data_ptr()) {
        return; // 自我拷贝直接返回
    }

    // ✅ 形状和类型验证 - 使用具体异常类型
    if (src.shape() != dst.shape()) {
        throw ShapeError("[CpuBackend::copy_into] Shape mismatch: " +
            src.shape().to_string() + " vs " + dst.shape().to_string());
    }

    if (src.dtype() != dst.dtype()) {
        throw TypeError("[CpuBackend::copy_into] DType mismatch");
    }

    // 执行拷贝
    size_t size = src.memory_size();
    const void* src_ptr = src.data_ptr();
    void* dst_ptr = dst.data_ptr();
    std::memcpy(dst_ptr, src_ptr, size);
}

// ===== reshape方法实现 =====

Tensor CpuBackend::reshape(const Tensor& tensor_a, const Shape& shape) {
    validate_same_device(tensor_a.device());

    // 检查数据类型必须是FP32
    if (tensor_a.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::reshape] Only FP32 tensors are supported");
    }

    // 检查元素数量是否相等
    if (tensor_a.numel() != shape.numel()) {
        throw TRException("[CpuBackend::reshape] Shape mismatch: tensor has " +
                         std::to_string(tensor_a.numel()) + " elements, but target shape " +
                         shape.to_string() + " has " + std::to_string(shape.numel()) + " elements");
    }

    // 检查输入张量是否为空
    if (tensor_a.is_empty()) {
        throw TRException("[CpuBackend::reshape] Cannot reshape empty tensor");
    }

    // 创建结果张量
    Tensor result = Tensor::empty(shape, DType::FP32, tr::CPU);

    // 复制数据
    const void* src_data = tensor_a.data_ptr();
    void* dst_data = result.data_ptr();
    copy_data(dst_data, src_data, tensor_a.numel() * dtype_size(tensor_a.dtype()),
              tr::CPU, tr::CPU);

    return result;
}

void CpuBackend::reshape_inplace(Tensor& tensor_a, const Shape& shape) {
    validate_same_device(tensor_a.device());

    // 检查数据类型必须是FP32
    if (tensor_a.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::reshape_inplace] Only FP32 tensors are supported");
    }

    // 检查元素数量是否相等
    if (tensor_a.numel() != shape.numel()) {
        throw TRException("[CpuBackend::reshape_inplace] Shape mismatch: tensor has " +
                         std::to_string(tensor_a.numel()) + " elements, but target shape " +
                         shape.to_string() + " has " + std::to_string(shape.numel()) + " elements");
    }

    // 检查输入张量是否为空
    if (tensor_a.is_empty()) {
        throw TRException("[CpuBackend::reshape_inplace] Cannot reshape empty tensor");
    }

    // 保存原始数据到临时缓冲区
    std::vector<float> temp_data(tensor_a.numel());
    const float* src_data = static_cast<const float*>(tensor_a.data_ptr());
    std::memcpy(temp_data.data(), src_data, tensor_a.numel() * sizeof(float));

    // 创建新的张量（使用empty来分配内存）
    tensor_a = this->empty(shape, DType::FP32);

    // 恢复数据
    float* dst_data = static_cast<float*>(tensor_a.data_ptr());
    std::memcpy(dst_data, temp_data.data(), tensor_a.numel() * sizeof(float));
}

void CpuBackend::reshape_into(const Tensor& tensor_a, Tensor& result, const Shape& shape) {
    validate_same_device(tensor_a.device());
    validate_same_device(result.device());

    // 检查数据类型必须是FP32
    if (tensor_a.dtype() != DType::FP32 || result.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::reshape_into] Only FP32 tensors are supported");
    }

    // 检查元素数量是否相等
    if (tensor_a.numel() != shape.numel()) {
        throw TRException("[CpuBackend::reshape_into] Shape mismatch: tensor has " +
                         std::to_string(tensor_a.numel()) + " elements, but target shape " +
                         shape.to_string() + " has " + std::to_string(shape.numel()) + " elements");
    }

    // 检查输入张量是否为空
    if (tensor_a.is_empty()) {
        throw TRException("[CpuBackend::reshape_into] Cannot reshape empty tensor");
    }

    // 检查输出张量是否为空
    if (result.is_empty()) {
        throw TRException("[CpuBackend::reshape_into] Output tensor is empty");
    }

    // 检查输出张量的元素数量是否匹配
    if (result.numel() != shape.numel()) {
        throw TRException("[CpuBackend::reshape_into] Output tensor has " +
                         std::to_string(result.numel()) + " elements, but target shape " +
                         shape.to_string() + " has " + std::to_string(shape.numel()) + " elements");
    }

    // 复制数据到结果张量
    const void* src_data = tensor_a.data_ptr();
    void* dst_data = result.data_ptr();
    copy_data(dst_data, src_data, tensor_a.numel() * dtype_size(tensor_a.dtype()),
              tr::CPU, tr::CPU);

    // 注意：reshape_into不修改result的shape，只是复制数据
    // 这是reshape的特殊性质：需要重新构造张量来改变形状
    // 但由于result是const引用，我们无法重新构造它
    // 所以这里只能复制数据，调用者需要确保result有正确的形状
}

// ===== tanh方法实现 =====

Tensor CpuBackend::tanh(const Tensor& tensor_a) {
    validate_same_device(tensor_a.device());

    // 检查数据类型必须是FP32
    if (tensor_a.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::tanh] Only FP32 tensors are supported");
    }

    // 检查输入张量是否为空
    if (tensor_a.is_empty()) {
        throw TRException("[CpuBackend::tanh] Cannot compute tanh of empty tensor");
    }

    // 创建结果张量
    Tensor result = Tensor::empty(tensor_a.shape(), DType::FP32, tr::CPU);

    const float* input_data = static_cast<const float*>(tensor_a.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    size_t num_elements = tensor_a.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> result_vec(result_data, num_elements);
    result_vec = input_vec.array().tanh().matrix();
#else
    // 朴素实现：使用高效的计算方式避免大内存分配
    for (size_t i = 0; i < num_elements; ++i) {
        float x = input_data[i];
        // 双曲正切函数：tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        // 为了数值稳定性，使用以下等价形式：
        if (x > 0.0f) {
            float exp_neg_2x = std::exp(-2.0f * x);
            result_data[i] = (1.0f - exp_neg_2x) / (1.0f + exp_neg_2x);
        } else {
            float exp_2x = std::exp(2.0f * x);
            result_data[i] = (exp_2x - 1.0f) / (exp_2x + 1.0f);
        }
    }
#endif

    return result;
}

void CpuBackend::tanh_inplace(Tensor& tensor_a) {
    validate_same_device(tensor_a.device());

    // 检查数据类型必须是FP32
    if (tensor_a.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::tanh_inplace] Only FP32 tensors are supported");
    }

    // 检查输入张量是否为空
    if (tensor_a.is_empty()) {
        throw TRException("[CpuBackend::tanh_inplace] Cannot compute tanh of empty tensor");
    }

    float* data = static_cast<float*>(tensor_a.data_ptr());
    size_t num_elements = tensor_a.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<MatrixType> eigen_vec(data, num_elements);
    eigen_vec = eigen_vec.array().tanh().matrix();
#else
    // 朴素实现：原地计算
    for (size_t i = 0; i < num_elements; ++i) {
        float x = data[i];
        // 双曲正切函数：tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        // 为了数值稳定性，使用以下等价形式：
        if (x > 0.0f) {
            float exp_neg_2x = std::exp(-2.0f * x);
            data[i] = (1.0f - exp_neg_2x) / (1.0f + exp_neg_2x);
        } else {
            float exp_2x = std::exp(2.0f * x);
            data[i] = (exp_2x - 1.0f) / (exp_2x + 1.0f);
        }
    }
#endif
}

void CpuBackend::tanh_into(const Tensor& tensor_a, Tensor& result) {
    validate_same_device(tensor_a.device());
    validate_same_device(result.device());

    if (tensor_a.dtype() != DType::FP32 || result.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::tanh_into] Only FP32 tensors are supported");
    }

#if TR_ENABLE_INTO_FUNC_SHAPE_CHECK == 1
    if (tensor_a.shape() != result.shape()) {
        throw TRException("[CpuBackend::tanh_into] Shape mismatch: input shape " +
                         tensor_a.shape().to_string() + " != output shape " +
                         result.shape().to_string());
    }
#endif

    // 检查输入张量是否为空
    if (tensor_a.is_empty()) {
        throw TRException("[CpuBackend::tanh_into] Cannot compute tanh of empty tensor");
    }

    // 检查输出张量是否为空
    if (result.is_empty()) {
        throw TRException("[CpuBackend::tanh_into] Output tensor is empty");
    }

    const float* input_data = static_cast<const float*>(tensor_a.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    size_t num_elements = tensor_a.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> result_vec(result_data, num_elements);
    result_vec = input_vec.array().tanh().matrix();
#else
    // 朴素实现：使用高效的计算方式避免大内存分配
    for (size_t i = 0; i < num_elements; ++i) {
        float x = input_data[i];
        // 双曲正切函数：tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        // 为了数值稳定性，使用以下等价形式：
        if (x > 0.0f) {
            float exp_neg_2x = std::exp(-2.0f * x);
            result_data[i] = (1.0f - exp_neg_2x) / (1.0f + exp_neg_2x);
        } else {
            float exp_2x = std::exp(2.0f * x);
            result_data[i] = (exp_2x - 1.0f) / (exp_2x + 1.0f);
        }
    }
#endif
}

// ===== dtanh方法实现 =====

Tensor CpuBackend::dtanh(const Tensor& tensor_a) {
    validate_same_device(tensor_a.device());

    // 检查数据类型必须是FP32
    if (tensor_a.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::dtanh] Only FP32 tensors are supported");
    }

    // 检查输入张量是否为空
    if (tensor_a.is_empty()) {
        throw TRException("[CpuBackend::dtanh] Cannot compute dtanh of empty tensor");
    }

    // 创建结果张量
    Tensor result = Tensor::empty(tensor_a.shape(), DType::FP32, tr::CPU);

    const float* input_data = static_cast<const float*>(tensor_a.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    size_t num_elements = tensor_a.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现：使用公式 dtanh(x) = 1 - tanh(x)^2
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> result_vec(result_data, num_elements);

    // 首先计算tanh(x)，然后使用向量化操作计算1-tanh^2
    MatrixType tanh_vec = input_vec.array().tanh().matrix();
    result_vec = (MatrixType::Ones(num_elements) - tanh_vec.array().square().matrix());
#else
    // 朴素实现：使用高效的直接计算避免中间数组
    for (size_t i = 0; i < num_elements; ++i) {
        float x = input_data[i];

        // 直接计算dtanh，避免存储中间tanh结果
        // 使用公式：dtanh(x) = 4 / (exp(x) + exp(-x))^2
        // 这个公式等价于 1 - tanh(x)^2，但更直接
        if (x > 0.0f) {
            // 对于正数，使用更稳定的计算方式
            float exp_neg_x = std::exp(-x);
            float denominator = 1.0f + exp_neg_x * exp_neg_x;
            result_data[i] = 4.0f / (denominator * denominator);
        } else {
            // 对于负数，使用exp(x)避免数值溢出
            float exp_x = std::exp(x);
            float denominator = exp_x * exp_x + 1.0f;
            result_data[i] = 4.0f / (denominator * denominator);
        }
    }
#endif

    return result;
}

void CpuBackend::dtanh_inplace(Tensor& tensor_a) {
    validate_same_device(tensor_a.device());

    // 检查数据类型必须是FP32
    if (tensor_a.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::dtanh_inplace] Only FP32 tensors are supported");
    }

    // 检查输入张量是否为空
    if (tensor_a.is_empty()) {
        throw TRException("[CpuBackend::dtanh_inplace] Cannot compute dtanh of empty tensor");
    }

    float* data = static_cast<float*>(tensor_a.data_ptr());
    size_t num_elements = tensor_a.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现：使用公式 dtanh(x) = 1 - tanh(x)^2
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<MatrixType> eigen_vec(data, num_elements);

    // 首先计算tanh(x)，然后原地计算1-tanh^2
    MatrixType tanh_vec = eigen_vec.array().tanh().matrix();
    eigen_vec = (MatrixType::Ones(num_elements) - tanh_vec.array().square().matrix());
#else
    // 朴素实现：原地计算，避免中间数组
    for (size_t i = 0; i < num_elements; ++i) {
        float x = data[i];

        // 直接计算dtanh，避免存储中间tanh结果
        // 使用公式：dtanh(x) = 4 / (exp(x) + exp(-x))^2
        if (x > 0.0f) {
            float exp_neg_x = std::exp(-x);
            float denominator = 1.0f + exp_neg_x * exp_neg_x;
            data[i] = 4.0f / (denominator * denominator);
        } else {
            float exp_x = std::exp(x);
            float denominator = exp_x * exp_x + 1.0f;
            data[i] = 4.0f / (denominator * denominator);
        }
    }
#endif
}

void CpuBackend::dtanh_into(const Tensor& tensor_a, Tensor& result) {
    validate_same_device(tensor_a.device());
    validate_same_device(result.device());

    if (tensor_a.dtype() != DType::FP32 || result.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::dtanh_into] Only FP32 tensors are supported");
    }

#if TR_ENABLE_INTO_FUNC_SHAPE_CHECK == 1
    if (tensor_a.shape() != result.shape()) {
        throw TRException("[CpuBackend::dtanh_into] Shape mismatch: input shape " +
                         tensor_a.shape().to_string() + " != output shape " +
                         result.shape().to_string());
    }
#endif

    // 检查输入张量是否为空
    if (tensor_a.is_empty()) {
        throw TRException("[CpuBackend::dtanh_into] Cannot compute dtanh of empty tensor");
    }

    // 检查输出张量是否为空
    if (result.is_empty()) {
        throw TRException("[CpuBackend::dtanh_into] Output tensor is empty");
    }

    const float* input_data = static_cast<const float*>(tensor_a.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    size_t num_elements = tensor_a.numel();

#ifdef TR_USE_EIGEN
    // Eigen优化实现：使用公式 dtanh(x) = 1 - tanh(x)^2
    using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    Eigen::Map<const MatrixType> input_vec(input_data, num_elements);
    Eigen::Map<MatrixType> result_vec(result_data, num_elements);

    // 首先计算tanh(x)，然后使用向量化操作计算1-tanh^2
    MatrixType tanh_vec = input_vec.array().tanh().matrix();
    result_vec = (MatrixType::Ones(num_elements) - tanh_vec.array().square().matrix());
#else
    // 朴素实现：使用高效的直接计算避免中间数组
    for (size_t i = 0; i < num_elements; ++i) {
        float x = input_data[i];

        // 直接计算dtanh，避免存储中间tanh结果
        // 使用公式：dtanh(x) = 4 / (exp(x) + exp(-x))^2
        if (x > 0.0f) {
            float exp_neg_x = std::exp(-x);
            float denominator = 1.0f + exp_neg_x * exp_neg_x;
            result_data[i] = 4.0f / (denominator * denominator);
        } else {
            float exp_x = std::exp(x);
            float denominator = exp_x * exp_x + 1.0f;
            result_data[i] = 4.0f / (denominator * denominator);
        }
    }
#endif
}

} // namespace tr
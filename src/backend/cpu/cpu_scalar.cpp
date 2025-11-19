/**
 * @file cpu_scalar.cpp
 * @brief CPU后端标量运算实现
 * @details 实现张量与浮点数标量的运算，支持非原地、原地和指定输出张量三种模式
 * @version 1.00.00
 * @date 2025-11-01
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

#include <algorithm>

namespace tr {

// ===== 乘法运算：tensor * scalar =====

Tensor CpuBackend::mul(const Tensor& input, float scalar) const {
    validate_same_device(input.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::mul] Input tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::mul] Only FP32 tensors are supported for scalar multiplication. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    // 创建输出张量
    Tensor result = this->empty(input.shape(), input.dtype());
    mul_into(input, scalar, result);
    return result;
}

void CpuBackend::mul_inplace(Tensor& input, float scalar) const {
    validate_same_device(input.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::mul_inplace] Input tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::mul_inplace] Only FP32 tensors are supported for scalar multiplication. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t count = input.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<Eigen::VectorXf>(data, count) = Eigen::Map<Eigen::VectorXf>(data, count) * scalar;
#else
    std::transform(data, data + count, data, [scalar](float x) { return x * scalar; });
#endif
}

void CpuBackend::mul_into(const Tensor& input, float scalar, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::mul_into] Input tensor has no allocated Storage");
    }

    if (output.is_empty()) {
        throw TRException("[CpuBackend::mul_into] Output tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::mul_into] Only FP32 tensors are supported for scalar multiplication. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::mul_into] Shape mismatch: input shape " +
                        input.shape().to_string() + " != output shape " +
                        output.shape().to_string());
    }

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());
    size_t count = input.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> input_vec(input_data, count);
    Eigen::Map<Eigen::VectorXf> output_vec(output_data, count);
    output_vec = input_vec * scalar;
#else
    std::transform(input_data, input_data + count, output_data,
                   [scalar](float x) { return x * scalar; });
#endif
}

// ===== 加法运算：tensor + scalar =====

Tensor CpuBackend::add(const Tensor& input, float scalar) const {
    validate_same_device(input.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::add] Input tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::add] Only FP32 tensors are supported for scalar addition. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    // 创建输出张量
    Tensor result = this->empty(input.shape(), input.dtype());
    add_into(input, scalar, result);
    return result;
}

void CpuBackend::add_inplace(Tensor& input, float scalar) const {
    validate_same_device(input.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::add_inplace] Input tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::add_inplace] Only FP32 tensors are supported for scalar addition. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t count = input.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<Eigen::VectorXf>(data, count) = Eigen::Map<Eigen::VectorXf>(data, count) + Eigen::VectorXf::Constant(count, scalar);
#else
    std::transform(data, data + count, data, [scalar](float x) { return x + scalar; });
#endif
}

void CpuBackend::add_into(const Tensor& input, float scalar, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::add_into] Input tensor has no allocated Storage");
    }

    if (output.is_empty()) {
        throw TRException("[CpuBackend::add_into] Output tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::add_into] Only FP32 tensors are supported for scalar addition. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::add_into] Shape mismatch: input shape " +
                        input.shape().to_string() + " != output shape " +
                        output.shape().to_string());
    }

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());
    size_t count = input.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> input_vec(input_data, count);
    Eigen::Map<Eigen::VectorXf> output_vec(output_data, count);
    output_vec = input_vec + Eigen::VectorXf::Constant(count, scalar);
#else
    std::transform(input_data, input_data + count, output_data,
                   [scalar](float x) { return x + scalar; });
#endif
}

// ===== 减法运算：tensor - scalar =====

Tensor CpuBackend::minus(const Tensor& input, float scalar) const {
    validate_same_device(input.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::minus] Input tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::minus] Only FP32 tensors are supported for scalar subtraction. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    // 创建输出张量
    Tensor result = this->empty(input.shape(), input.dtype());
    minus_into(input, scalar, result);
    return result;
}

void CpuBackend::minus_inplace(Tensor& input, float scalar) const {
    validate_same_device(input.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::minus_inplace] Input tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::minus_inplace] Only FP32 tensors are supported for scalar subtraction. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t count = input.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<Eigen::VectorXf>(data, count) = Eigen::Map<Eigen::VectorXf>(data, count) - Eigen::VectorXf::Constant(count, scalar);
#else
    std::transform(data, data + count, data, [scalar](float x) { return x - scalar; });
#endif
}

void CpuBackend::minus_into(const Tensor& input, float scalar, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::minus_into] Input tensor has no allocated Storage");
    }

    if (output.is_empty()) {
        throw TRException("[CpuBackend::minus_into] Output tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::minus_into] Only FP32 tensors are supported for scalar subtraction. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::minus_into] Shape mismatch: input shape " +
                        input.shape().to_string() + " != output shape " +
                        output.shape().to_string());
    }

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());
    size_t count = input.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> input_vec(input_data, count);
    Eigen::Map<Eigen::VectorXf> output_vec(output_data, count);
    output_vec = input_vec - Eigen::VectorXf::Constant(count, scalar);
#else
    std::transform(input_data, input_data + count, output_data,
                   [scalar](float x) { return x - scalar; });
#endif
}

// ===== 减法运算：scalar - tensor =====

Tensor CpuBackend::minus(float scalar, const Tensor& input) const {
    validate_same_device(input.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::minus] Input tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::minus] Only FP32 tensors are supported for scalar subtraction. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    // 创建输出张量
    Tensor result = this->empty(input.shape(), input.dtype());
    minus_into(scalar, input, result);
    return result;
}

void CpuBackend::minus_inplace(float scalar, Tensor& input) const {
    validate_same_device(input.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::minus_inplace] Input tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::minus_inplace] Only FP32 tensors are supported for scalar subtraction. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t count = input.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<Eigen::VectorXf> data_vec(data, count);
    data_vec = Eigen::VectorXf::Constant(count, scalar) - data_vec;
#else
    std::transform(data, data + count, data, [scalar](float x) { return scalar - x; });
#endif
}

void CpuBackend::minus_into(float scalar, const Tensor& input, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::minus_into] Input tensor has no allocated Storage");
    }

    if (output.is_empty()) {
        throw TRException("[CpuBackend::minus_into] Output tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::minus_into] Only FP32 tensors are supported for scalar subtraction. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::minus_into] Shape mismatch: input shape " +
                        input.shape().to_string() + " != output shape " +
                        output.shape().to_string());
    }

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());
    size_t count = input.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> input_vec(input_data, count);
    Eigen::Map<Eigen::VectorXf> output_vec(output_data, count);
    output_vec = Eigen::VectorXf::Constant(count, scalar) - input_vec;
#else
    std::transform(input_data, input_data + count, output_data,
                   [scalar](float x) { return scalar - x; });
#endif
}

// ===== 乘加运算：tensor * scalar_x + scalar_y =====

Tensor CpuBackend::mac(const Tensor& input, float scalar_x, float scalar_y) const {
    validate_same_device(input.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::mac] Input tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::mac] Only FP32 tensors are supported for multiply-add operations. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    // 创建输出张量
    Tensor result = this->empty(input.shape(), input.dtype());
    mac_into(input, scalar_x, scalar_y, result);
    return result;
}

void CpuBackend::mac_inplace(Tensor& input, float scalar_x, float scalar_y) const {
    validate_same_device(input.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::mac_inplace] Input tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::mac_inplace] Only FP32 tensors are supported for multiply-add operations. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t count = input.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<Eigen::VectorXf> data_vec(data, count);
    data_vec = data_vec * scalar_x + Eigen::VectorXf::Constant(count, scalar_y);
#else
    std::transform(data, data + count, data,
                   [scalar_x, scalar_y](float x) { return x * scalar_x + scalar_y; });
#endif
}

void CpuBackend::mac_into(const Tensor& input, float scalar_x, float scalar_y, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::mac_into] Input tensor has no allocated Storage");
    }

    if (output.is_empty()) {
        throw TRException("[CpuBackend::mac_into] Output tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::mac_into] Only FP32 tensors are supported for multiply-add operations. "
                        "TODO: Consider implementing INT8 scalar operations in future versions.");
    }

    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::mac_into] Shape mismatch: input shape " +
                        input.shape().to_string() + " != output shape " +
                        output.shape().to_string());
    }

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());
    size_t count = input.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> input_vec(input_data, count);
    Eigen::Map<Eigen::VectorXf> output_vec(output_data, count);
    output_vec = input_vec * scalar_x + Eigen::VectorXf::Constant(count, scalar_y);
#else
    std::transform(input_data, input_data + count, output_data,
                   [scalar_x, scalar_y](float x) { return x * scalar_x + scalar_y; });
#endif
}

// ===== 裁剪运算：clamp(tensor, min_val, max_val) =====

Tensor CpuBackend::clamp(const Tensor& input, float min_val, float max_val) const {
    validate_same_device(input.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::clamp] Input tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::clamp] Only FP32 tensors are supported for clamp operation. "
                        "TODO: Consider implementing INT8 clamp operations in future versions.");
    }

    // 验证min_val <= max_val
    if (min_val > max_val) {
        throw TRException("[CpuBackend::clamp] min_val (" + std::to_string(min_val) +
                        ") cannot be greater than max_val (" + std::to_string(max_val) + ")");
    }

    // 创建输出张量
    Tensor result = this->empty(input.shape(), input.dtype());
    clamp_into(input, min_val, max_val, result);
    return result;
}

void CpuBackend::clamp_inplace(Tensor& input, float min_val, float max_val) const {
    validate_same_device(input.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::clamp_inplace] Input tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::clamp_inplace] Only FP32 tensors are supported for clamp operation. "
                        "TODO: Consider implementing INT8 clamp operations in future versions.");
    }

    // 验证min_val <= max_val
    if (min_val > max_val) {
        throw TRException("[CpuBackend::clamp_inplace] min_val (" + std::to_string(min_val) +
                        ") cannot be greater than max_val (" + std::to_string(max_val) + ")");
    }

    float* data = static_cast<float*>(input.data_ptr());
    size_t count = input.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<Eigen::VectorXf>(data, count) =
        Eigen::Map<Eigen::VectorXf>(data, count).cwiseMax(min_val).cwiseMin(max_val);
#else
    std::transform(data, data + count, data,
                   [min_val, max_val](float x) {
                       return (x < min_val) ? min_val : ((x > max_val) ? max_val : x);
                   });
#endif
}

void CpuBackend::clamp_into(const Tensor& input, float min_val, float max_val, Tensor& output) const {
    validate_same_device(input.device());
    validate_same_device(output.device());

    if (input.is_empty()) {
        throw TRException("[CpuBackend::clamp_into] Input tensor has no allocated Storage");
    }

    if (output.is_empty()) {
        throw TRException("[CpuBackend::clamp_into] Output tensor has no allocated Storage");
    }

    if (input.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::clamp_into] Only FP32 tensors are supported for clamp operation. "
                        "TODO: Consider implementing INT8 clamp operations in future versions.");
    }

    if (input.shape() != output.shape()) {
        throw TRException("[CpuBackend::clamp_into] Shape mismatch: input shape " +
                        input.shape().to_string() + " != output shape " +
                        output.shape().to_string());
    }

    // 验证min_val <= max_val
    if (min_val > max_val) {
        throw TRException("[CpuBackend::clamp_into] min_val (" + std::to_string(min_val) +
                        ") cannot be greater than max_val (" + std::to_string(max_val) + ")");
    }

    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* output_data = static_cast<float*>(output.data_ptr());
    size_t count = input.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> input_vec(input_data, count);
    Eigen::Map<Eigen::VectorXf> output_vec(output_data, count);
    output_vec = input_vec.cwiseMax(min_val).cwiseMin(max_val);
#else
    std::transform(input_data, input_data + count, output_data,
                   [min_val, max_val](float x) {
                       return (x < min_val) ? min_val : ((x > max_val) ? max_val : x);
                   });
#endif
}

} // namespace tr
/**
 * @file cpu_broadcast.cpp
 * @brief CPU后端可广播张量运算实现
 * @details 实现张量间的广播加法、减法、乘法运算，支持形状广播和逻辑广播优化
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
#include <stdexcept>

namespace tr {

// ===== 广播辅助函数 =====

/**
 * @brief 获取广播运算的输出形状
 * @param shape_a 张量a的形状
 * @param shape_b 张量b的形状
 * @return 输出张量的形状
 * @throws TRException 如果形状不兼容广播
 */
static Shape get_broadcast_output_shape(const Shape& shape_a, const Shape& shape_b) {
    // 1. 检查形状是否全等
    if (shape_a == shape_b) {
        return shape_a;  // 返回任意一个都可以，形状相同
    }

    // 2. 检查维度数是否相同：维度数不同不可以广播，但其中给一个是标量的情况除外
    if (shape_a.ndim() != shape_b.ndim() && (!shape_a.is_scalar()) && (!shape_b.is_scalar())) {
        throw TRException("[CPU Broadcast] Tensor shapes do not meet operation requirements: " +
                         shape_a.to_string() + " and " + shape_b.to_string());
    }

    // 3. 检查是否可以广播（只在相同维度数的情况下）
    if (shape_a.is_broadcastable_to(shape_b)) {
        return shape_b;  // b元素数更多，输出b的形状
    }

    if (shape_b.is_broadcastable_to(shape_a)) {
        return shape_a;  // a元素数更多，输出a的形状
    }

    // 4. 都不能广播，抛出异常
    throw TRException("[CPU Broadcast] Tensor shapes are incompatible and cannot be broadcast: " +
                     shape_a.to_string() + " and " + shape_b.to_string());
}

/**
 * @brief 验证张量是否为空
 * @param tensor 要检查的张量
 * @param name 张量名称（用于错误信息）
 * @throws TRException 如果张量为空
 */
static void validate_tensor_not_empty(const Tensor& tensor, const std::string& name) {
    if (tensor.is_empty()) {
        throw TRException("[CPU Broadcast] " + name + " is an empty tensor");
    }
}

/**
 * @brief 验证张量的数据类型和设备
 * @param tensor 要验证的张量
 * @param name 张量名称（用于错误信息）
 * @throws TRException 如果数据类型不是FP32或设备不是CPU
 */
static void validate_tensor_dtype_and_device(const Tensor& tensor, const std::string& name) {
    // 检查数据类型
    if (tensor.dtype() != DType::FP32) {
        throw TRException("[CPU Broadcast] Currently only supports FP32 tensor operations, " + name + " data type is: " +
                         dtype_to_string(tensor.dtype()));
    }

    // 检查设备
    if (tensor.device() != tr::CPU) {
        throw TRException("[CPU Broadcast] Device must be CPU, " + name + " device is: " +
                         tensor.device().to_string());
    }
}

/**
 * @brief 验证两个张量的数据类型和设备兼容性
 * @param tensor_a 张量a
 * @param tensor_b 张量b
 * @throws TRException 如果不兼容
 */
static void validate_tensors_compatibility(const Tensor& tensor_a, const Tensor& tensor_b) {
    // 验证数据类型
    validate_tensor_dtype_and_device(tensor_a, "tensor_a");
    validate_tensor_dtype_and_device(tensor_b, "tensor_b");
}

/**
 * @brief 处理两个空张量的特殊情况
 * @param tensor_a 张量a
 * @param tensor_b 张量b
 * @param result 输出张量
 * @return 如果两个都是空张量返回true
 */
static bool handle_empty_tensors(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result) {
    if (tensor_a.is_empty() && tensor_b.is_empty()) {
        // 两个都是空张量，返回空张量，不需要计算
        return true;
    }

    // 一个空张量一个非空张量的情况在validate_tensor_not_empty中处理
    return false;
}

/**
 * @brief 将线性索引转换为多维索引
 * @param linear_index 线性索引
 * @param shape 张量形状
 * @param indices 输出多维索引数组
 */
static void linear_to_multi_index(int64_t linear_index, const Shape& shape, int32_t* indices) {
    int32_t ndim = shape.ndim();

    if (ndim == 0) {
        // 标量张量
        return;
    }

    // 从右向左计算每个维度的索引
    int64_t remaining = linear_index;
    for (int32_t i = ndim - 1; i >= 0; --i) {
        int32_t dim_size = shape.dim(i);
        indices[i] = remaining % dim_size;
        remaining /= dim_size;
    }
}

/**
 * @brief 将多维索引转换为线性索引（考虑广播）
 * @param indices 多维索引数组
 * @param shape 张量形状
 * @param target_shape 目标形状
 * @return 线性索引
 */
static int64_t multi_index_to_linear_with_broadcast(
    const int32_t* indices,
    const Shape& shape,
    const Shape& target_shape) {

    int32_t target_ndim = target_shape.ndim();
    int32_t source_ndim = shape.ndim();

    if (target_ndim == 0) {
        // 目标是标量
        return 0;
    }

    int64_t linear_index = 0;
    int64_t stride = 1;

    // 从右向左计算，考虑广播
    for (int32_t i = target_ndim - 1; i >= 0; --i) {
        int32_t target_dim = target_shape.dim(i);
        int32_t target_idx = indices[i];

        // 计算在源张量中的索引
        int32_t source_idx = target_idx;
        if (source_ndim > 0) {
            int32_t source_dim_idx = i - (target_ndim - source_ndim);
            if (source_dim_idx >= 0) {
                int32_t source_dim = shape.dim(source_dim_idx);
                if (source_dim == 1) {
                    // 广播维度，索引为0
                    source_idx = 0;
                } else {
                    // 正常维度，索引为target_idx
                    source_idx = target_idx;
                }
            }
        }

        linear_index += source_idx * stride;
        stride *= (source_ndim > 0 && i >= target_ndim - source_ndim) ?
                  shape.dim(i - (target_ndim - source_ndim)) : 1;
    }

    return linear_index;
}

// ===== 广播运算核心实现 =====

/**
 * @brief 广播运算的通用实现（朴素版本）
 * @param tensor_a 输入张量a
 * @param tensor_b 输入张量b
 * @param result 输出张量
 * @param operation 运算函数（lambda）
 */
template<typename Operation>
static void broadcast_operation_naive(
    const Tensor& tensor_a,
    const Tensor& tensor_b,
    Tensor& result,
    Operation operation) {

    const Shape& shape_a = tensor_a.shape();
    const Shape& shape_b = tensor_b.shape();
    const Shape& shape_result = result.shape();

    int64_t total_elements = shape_result.numel();
    const float* data_a = static_cast<const float*>(tensor_a.data_ptr());
    const float* data_b = static_cast<const float*>(tensor_b.data_ptr());
    float* data_result = static_cast<float*>(result.data_ptr());

    // 标量广播优化
    if (shape_a.is_scalar()) {
        // a是标量，广播到b的形状
        float scalar_a = data_a[0];
        if (shape_b.is_scalar()) {
            // 两个标量
            for (int64_t i = 0; i < total_elements; ++i) {
                data_result[i] = operation(scalar_a, data_b[i]);
            }
        } else {
            // a标量广播到b
            for (int64_t i = 0; i < total_elements; ++i) {
                data_result[i] = operation(scalar_a, data_b[i]);
            }
        }
        return;
    }

    if (shape_b.is_scalar()) {
        // b是标量，广播到a的形状
        float scalar_b = data_b[0];
        for (int64_t i = 0; i < total_elements; ++i) {
            data_result[i] = operation(data_a[i], scalar_b);
        }
        return;
    }

    // 一般广播情况
    std::vector<int32_t> indices(shape_result.ndim());

    for (int64_t i = 0; i < total_elements; ++i) {
        // 将线性索引转换为多维索引
        linear_to_multi_index(i, shape_result, indices.data());

        // 计算在源张量中的线性索引（考虑广播）
        int64_t idx_a = multi_index_to_linear_with_broadcast(indices.data(), shape_a, shape_result);
        int64_t idx_b = multi_index_to_linear_with_broadcast(indices.data(), shape_b, shape_result);

        // 执行运算
        data_result[i] = operation(data_a[idx_a], data_b[idx_b]);
    }
}

#ifdef TR_USE_EIGEN
/**
 * @brief 广播运算的通用实现（Eigen优化版本）
 * @param tensor_a 输入张量a
 * @param tensor_b 输入张量b
 * @param result 输出张量
 * @param operation 运算函数（lambda）
 */
template<typename Operation>
static void broadcast_operation_eigen(
    const Tensor& tensor_a,
    const Tensor& tensor_b,
    Tensor& result,
    Operation operation) {

    const Shape& shape_a = tensor_a.shape();
    const Shape& shape_b = tensor_b.shape();
    const Shape& shape_result = result.shape();

    const float* data_a = static_cast<const float*>(tensor_a.data_ptr());
    const float* data_b = static_cast<const float*>(tensor_b.data_ptr());
    float* data_result = static_cast<float*>(result.data_ptr());

    // 标量广播优化
    if (shape_a.is_scalar()) {
        float scalar_a = data_a[0];
        Eigen::Map<const Eigen::VectorXf> vec_b(data_b, shape_result.numel());
        Eigen::Map<Eigen::VectorXf> vec_result(data_result, shape_result.numel());
        vec_result = vec_b.unaryExpr([&scalar_a, &operation](float x) { return operation(scalar_a, x); });
        return;
    }

    if (shape_b.is_scalar()) {
        float scalar_b = data_b[0];
        Eigen::Map<const Eigen::VectorXf> vec_a(data_a, shape_result.numel());
        Eigen::Map<Eigen::VectorXf> vec_result(data_result, shape_result.numel());
        vec_result = vec_a.unaryExpr([&scalar_b, &operation](float x) { return operation(x, scalar_b); });
        return;
    }

    // 对于简单情况（相同形状），使用Eigen直接操作
    if (shape_a == shape_b && shape_a == shape_result) {
        Eigen::Map<const Eigen::VectorXf> vec_a(data_a, shape_result.numel());
        Eigen::Map<const Eigen::VectorXf> vec_b(data_b, shape_result.numel());
        Eigen::Map<Eigen::VectorXf> vec_result(data_result, shape_result.numel());
        vec_result = vec_a.binaryExpr(vec_b, operation);
        return;
    }

    // 复杂广播情况回退到朴素实现
    broadcast_operation_naive(tensor_a, tensor_b, result, operation);
}
#endif

// ===== 加法运算 =====

Tensor CpuBackend::add_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const {
    // 验证数据类型和设备
    validate_tensors_compatibility(tensor_a, tensor_b);

    // 处理空张量特殊情况
    if (tensor_a.is_empty() && tensor_b.is_empty()) {
        return Tensor();  // 返回空张量
    }

    // 验证非空张量
    validate_tensor_not_empty(tensor_a, "tensor_a");
    validate_tensor_not_empty(tensor_b, "tensor_b");

    // 计算输出形状
    Shape output_shape = get_broadcast_output_shape(tensor_a.shape(), tensor_b.shape());

    // 创建输出张量
    Tensor result = Tensor::empty(output_shape, DType::FP32, tr::CPU);

    // 执行广播加法
    add_broadcast_into(tensor_a, tensor_b, result);

    return result;
}

void CpuBackend::add_broadcast_into(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result) const {
    // 验证数据类型和设备
    validate_tensors_compatibility(tensor_a, tensor_b);
    validate_tensor_dtype_and_device(result, "result");

    // 处理空张量特殊情况
    if (handle_empty_tensors(tensor_a, tensor_b, result)) {
        return;  // 两个都是空张量，直接返回
    }

    // 验证非空张量
    validate_tensor_not_empty(tensor_a, "tensor_a");
    validate_tensor_not_empty(tensor_b, "tensor_b");
    validate_tensor_not_empty(result, "result");

    // 验证输出形状
    Shape expected_shape = get_broadcast_output_shape(tensor_a.shape(), tensor_b.shape());
    if (result.shape() != expected_shape) {
        throw TRException("[CPU Broadcast] Output tensor shape mismatch, expected: " + expected_shape.to_string() +
                         ", actual: " + result.shape().to_string());
    }

    // 执行广播加法运算
    auto add_op = [](float a, float b) { return a + b; };

#ifdef TR_USE_EIGEN
    broadcast_operation_eigen(tensor_a, tensor_b, result, add_op);
#else
    broadcast_operation_naive(tensor_a, tensor_b, result, add_op);
#endif
}

// ===== 减法运算 =====

Tensor CpuBackend::minus_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const {
    // 验证数据类型和设备
    validate_tensors_compatibility(tensor_a, tensor_b);

    // 处理空张量特殊情况
    if (tensor_a.is_empty() && tensor_b.is_empty()) {
        return Tensor();  // 返回空张量
    }

    // 验证非空张量
    validate_tensor_not_empty(tensor_a, "tensor_a");
    validate_tensor_not_empty(tensor_b, "tensor_b");

    // 计算输出形状
    Shape output_shape = get_broadcast_output_shape(tensor_a.shape(), tensor_b.shape());

    // 创建输出张量
    Tensor result = Tensor::empty(output_shape, DType::FP32, tr::CPU);

    // 执行广播减法
    minus_broadcast_into(tensor_a, tensor_b, result);

    return result;
}

void CpuBackend::minus_broadcast_into(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result) const {
    // 验证数据类型和设备
    validate_tensors_compatibility(tensor_a, tensor_b);
    validate_tensor_dtype_and_device(result, "result");

    // 处理空张量特殊情况
    if (handle_empty_tensors(tensor_a, tensor_b, result)) {
        return;  // 两个都是空张量，直接返回
    }

    // 验证非空张量
    validate_tensor_not_empty(tensor_a, "tensor_a");
    validate_tensor_not_empty(tensor_b, "tensor_b");
    validate_tensor_not_empty(result, "result");

    // 验证输出形状
    Shape expected_shape = get_broadcast_output_shape(tensor_a.shape(), tensor_b.shape());
    if (result.shape() != expected_shape) {
        throw TRException("[CPU Broadcast] Output tensor shape mismatch, expected: " + expected_shape.to_string() +
                         ", actual: " + result.shape().to_string());
    }

    // 执行广播减法运算
    auto minus_op = [](float a, float b) { return a - b; };

#ifdef TR_USE_EIGEN
    broadcast_operation_eigen(tensor_a, tensor_b, result, minus_op);
#else
    broadcast_operation_naive(tensor_a, tensor_b, result, minus_op);
#endif
}

// ===== 乘法运算 =====

Tensor CpuBackend::mul_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const {
    // 验证数据类型和设备
    validate_tensors_compatibility(tensor_a, tensor_b);

    // 处理空张量特殊情况
    if (tensor_a.is_empty() && tensor_b.is_empty()) {
        return Tensor();  // 返回空张量
    }

    // 验证非空张量
    validate_tensor_not_empty(tensor_a, "tensor_a");
    validate_tensor_not_empty(tensor_b, "tensor_b");

    // 计算输出形状
    Shape output_shape = get_broadcast_output_shape(tensor_a.shape(), tensor_b.shape());

    // 创建输出张量
    Tensor result = Tensor::empty(output_shape, DType::FP32, tr::CPU);

    // 执行广播乘法
    mul_broadcast_into(tensor_a, tensor_b, result);

    return result;
}

void CpuBackend::mul_broadcast_into(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result) const {
    // 验证数据类型和设备
    validate_tensors_compatibility(tensor_a, tensor_b);
    validate_tensor_dtype_and_device(result, "result");

    // 处理空张量特殊情况
    if (handle_empty_tensors(tensor_a, tensor_b, result)) {
        return;  // 两个都是空张量，直接返回
    }

    // 验证非空张量
    validate_tensor_not_empty(tensor_a, "tensor_a");
    validate_tensor_not_empty(tensor_b, "tensor_b");
    validate_tensor_not_empty(result, "result");

    // 验证输出形状
    Shape expected_shape = get_broadcast_output_shape(tensor_a.shape(), tensor_b.shape());
    if (result.shape() != expected_shape) {
        throw TRException("[CPU Broadcast] Output tensor shape mismatch, expected: " + expected_shape.to_string() +
                         ", actual: " + result.shape().to_string());
    }

    // 执行广播乘法运算
    auto mul_op = [](float a, float b) { return a * b; };

#ifdef TR_USE_EIGEN
    broadcast_operation_eigen(tensor_a, tensor_b, result, mul_op);
#else
    broadcast_operation_naive(tensor_a, tensor_b, result, mul_op);
#endif
}

} // namespace tr
/**
 * @file cpu_dimension.cpp
 * @brief CPU后端张量维度操作实现
 * @details 实现张量的unsqueeze和squeeze操作，支持维度插入和删除
 * @version 1.00.00
 * @date 2025-11-01
 * @author 技术觉醒团队
 * @note 依赖项: cpu_backend.h, tensor.h
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/utils/logger.h"

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <string>

namespace tr {

// ===== Unsqueeze 操作核心实现 =====

/**
 * @brief 计算unsqueeze后的新形状
 * @param original_shape 原始形状
 * @param dim 要插入维度的位置
 * @return 新的形状
 * @throws TRException 如果维度索引超出范围
 */
static Shape calculate_unsqueeze_shape(const Shape& original_shape, int32_t dim) {
    int32_t original_ndim = original_shape.ndim();

    // 验证维度索引
    if (dim < 0 || dim > original_ndim) {
        throw TRException("[CPU Unsqueeze] Dimension index out of range");
    }

    // 构建新的形状数组
    std::vector<int32_t> new_dims;

    // 在指定位置插入1
    for (int32_t i = 0; i < original_ndim; ++i) {
        if (i == dim) {
            new_dims.push_back(1);
        }
        new_dims.push_back(original_shape.dim(i));
    }

    // 如果要在最后添加维度
    if (dim == original_ndim) {
        new_dims.push_back(1);
    }

    // 创建新形状
    Shape new_shape;
    if (new_dims.size() == 1) {
        new_shape = Shape(new_dims[0]);
    } else if (new_dims.size() == 2) {
        new_shape = Shape(new_dims[0], new_dims[1]);
    } else if (new_dims.size() == 3) {
        new_shape = Shape(new_dims[0], new_dims[1], new_dims[2]);
    } else if (new_dims.size() == 4) {
        new_shape = Shape(new_dims[0], new_dims[1], new_dims[2], new_dims[3]);
    }

    return new_shape;
}

/**
 * @brief 执行unsqueeze操作的核心实现
 * @param input 输入张量
 * @param result 输出张量（预分配了正确的形状）
 */
static void unsqueeze_operation_core(const Tensor& input, Tensor& result) {
    // 由于unsqueeze本质上是reshape操作，数据内容不需要改变
    // 只需要复制数据到新的张量中
    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    int64_t total_elements = input.numel();
    std::memcpy(result_data, input_data, total_elements * sizeof(float));
}

// ===== Squeeze 操作核心实现 =====

/**
 * @brief 计算squeeze后的新形状
 * @param original_shape 原始形状
 * @param dim 要删除的维度
 * @return 新的形状
 * @throws TRException 如果维度索引超出范围或维度大小不为1
 */
static Shape calculate_squeeze_shape(const Shape& original_shape, int32_t dim) {
    int32_t original_ndim = original_shape.ndim();

    // 验证维度索引
    if (dim < 0 || dim >= original_ndim) {
        throw TRException("[CPU Squeeze] Dimension index out of range");
    }

    // 检查指定维度是否为1
    if (original_shape.dim(dim) != 1) {
        throw TRException("[CPU Squeeze] Dimension size is not 1");
    }

    // 构建新的形状数组
    std::vector<int32_t> new_dims;

    for (int32_t i = 0; i < original_ndim; ++i) {
        if (i != dim) {
            new_dims.push_back(original_shape.dim(i));
        }
    }

    // 创建新形状
    Shape new_shape;
    if (new_dims.empty()) {
        // 标量
        new_shape = Shape();
    } else if (new_dims.size() == 1) {
        new_shape = Shape(new_dims[0]);
    } else if (new_dims.size() == 2) {
        new_shape = Shape(new_dims[0], new_dims[1]);
    } else if (new_dims.size() == 3) {
        new_shape = Shape(new_dims[0], new_dims[1], new_dims[2]);
    } else if (new_dims.size() == 4) {
        new_shape = Shape(new_dims[0], new_dims[1], new_dims[2], new_dims[3]);
    }

    return new_shape;
}

/**
 * @brief 执行squeeze操作的核心实现
 * @param input 输入张量
 * @param result 输出张量（预分配了正确的形状）
 */
static void squeeze_operation_core(const Tensor& input, Tensor& result) {
    // 由于squeeze本质上是reshape操作，数据内容不需要改变
    // 只需要复制数据到新的张量中
    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    int64_t total_elements = result.numel();  // 注意：这里是result的元素数量
    std::memcpy(result_data, input_data, total_elements * sizeof(float));
}

// ===== 辅助验证函数 =====

/**
 * @brief 验证张量的数据类型和设备
 * @param tensor 要验证的张量
 * @param name 张量名称（用于错误信息）
 * @throws TRException 如果数据类型不是FP32或设备不是CPU
 */
static void validate_tensor_dtype_and_device(const Tensor& tensor, const std::string& name) {
    // 检查数据类型
    if (tensor.dtype() != DType::FP32) {
        throw TRException("[CPU Dimension] Currently only supports FP32 tensor operations, " + name + " data type is: " +
                         dtype_to_string(tensor.dtype()));
    }

    // 检查设备
    if (tensor.device() != tr::CPU) {
        throw TRException("[CPU Dimension] Device must be CPU, " + name + " device is: " +
                         tensor.device().to_string());
    }
}

/**
 * @brief 验证张量是否为空
 * @param tensor 要检查的张量
 * @param name 张量名称（用于错误信息）
 * @throws TRException 如果张量为空
 */
static void validate_tensor_not_empty(const Tensor& tensor, const std::string& name) {
    if (tensor.is_empty()) {
        throw TRException("[CPU Dimension] " + name + " is an empty tensor");
    }
}

// ===== Unsqueeze 操作实现 =====

Tensor CpuBackend::unsqueeze(const Tensor& tensor_a, int32_t dim) const {
    // 验证数据类型和设备
    validate_tensor_dtype_and_device(tensor_a, "tensor_a");

    // 处理空张量特殊情况
    if (tensor_a.is_empty()) {
        return Tensor();  // 返回空张量
    }

    // 验证非空张量
    validate_tensor_not_empty(tensor_a, "tensor_a");

    // 计算新形状
    Shape new_shape = calculate_unsqueeze_shape(tensor_a.shape(), dim);

    // 创建输出张量
    Tensor result = Tensor::empty(new_shape, DType::FP32, tr::CPU);

    // 执行unsqueeze操作
    unsqueeze_operation_core(tensor_a, result);

    return result;
}

void CpuBackend::unsqueeze_inplace(Tensor& tensor_a, int32_t dim) const {
    // 验证数据类型和设备
    validate_tensor_dtype_and_device(tensor_a, "tensor_a");

    // 处理空张量特殊情况
    if (tensor_a.is_empty()) {
        return;  // 空张量不需要操作
    }

    // 验证非空张量
    validate_tensor_not_empty(tensor_a, "tensor_a");

    // 计算新形状
    Shape new_shape = calculate_unsqueeze_shape(tensor_a.shape(), dim);

    // 由于当前Tensor类不支持inplace reshape，我们需要创建新张量并替换数据
    Tensor temp = Tensor::empty(new_shape, DType::FP32, tr::CPU);
    unsqueeze_operation_core(tensor_a, temp);

    // 将temp的数据复制回原张量（注意：这是概念上的inplace，实际上创建了新对象）
    tensor_a = temp;
}

void CpuBackend::unsqueeze_into(const Tensor& tensor_a, Tensor& tensor_b) const {
    // 验证数据类型和设备
    validate_tensor_dtype_and_device(tensor_a, "tensor_a");
    validate_tensor_dtype_and_device(tensor_b, "tensor_b");

    // 处理空张量特殊情况
    if (tensor_a.is_empty()) {
        if (tensor_b.is_empty()) {
            return;  // 两个都是空张量，直接返回
        } else {
            throw TRException("[CPU Dimension] Cannot unsqueeze empty tensor into non-empty tensor");
        }
    }

    // 验证非空张量
    validate_tensor_not_empty(tensor_a, "tensor_a");
    validate_tensor_not_empty(tensor_b, "tensor_b");

    // 验证元素数是否相同
    if (tensor_a.numel() != tensor_b.numel()) {
        throw TRException("[CPU Dimension] Element count mismatch for unsqueeze_into: " +
                         std::to_string(tensor_a.numel()) + " vs " + std::to_string(tensor_b.numel()));
    }

    // 执行unsqueeze操作（本质上是数据复制）
    unsqueeze_operation_core(tensor_a, tensor_b);
}

// ===== Squeeze 操作实现 =====

Tensor CpuBackend::squeeze(const Tensor& tensor_a, int32_t dim) const {
    // 验证数据类型和设备
    validate_tensor_dtype_and_device(tensor_a, "tensor_a");

    // 处理空张量特殊情况
    if (tensor_a.is_empty()) {
        return Tensor();  // 返回空张量
    }

    // 验证非空张量
    validate_tensor_not_empty(tensor_a, "tensor_a");

    // 计算新形状
    Shape new_shape = calculate_squeeze_shape(tensor_a.shape(), dim);

    // 创建输出张量
    Tensor result = Tensor::empty(new_shape, DType::FP32, tr::CPU);

    // 执行squeeze操作
    squeeze_operation_core(tensor_a, result);

    return result;
}

void CpuBackend::squeeze_inplace(Tensor& tensor_a, int32_t dim) const {
    // 验证数据类型和设备
    validate_tensor_dtype_and_device(tensor_a, "tensor_a");

    // 处理空张量特殊情况
    if (tensor_a.is_empty()) {
        return;  // 空张量不需要操作
    }

    // 验证非空张量
    validate_tensor_not_empty(tensor_a, "tensor_a");

    // 计算新形状
    Shape new_shape = calculate_squeeze_shape(tensor_a.shape(), dim);

    // 由于当前Tensor类不支持inplace reshape，我们需要创建新张量并替换数据
    Tensor temp = Tensor::empty(new_shape, DType::FP32, tr::CPU);
    squeeze_operation_core(tensor_a, temp);

    // 将temp的数据复制回原张量（注意：这是概念上的inplace，实际上创建了新对象）
    tensor_a = temp;
}

void CpuBackend::squeeze_into(const Tensor& tensor_a, Tensor& tensor_b) const {
    // 验证数据类型和设备
    validate_tensor_dtype_and_device(tensor_a, "tensor_a");
    validate_tensor_dtype_and_device(tensor_b, "tensor_b");

    // 处理空张量特殊情况
    if (tensor_a.is_empty()) {
        if (tensor_b.is_empty()) {
            return;  // 两个都是空张量，直接返回
        } else {
            throw TRException("[CPU Dimension] Cannot squeeze empty tensor into non-empty tensor");
        }
    }

    // 验证非空张量
    validate_tensor_not_empty(tensor_a, "tensor_a");
    validate_tensor_not_empty(tensor_b, "tensor_b");

    // 验证元素数是否相同
    if (tensor_a.numel() != tensor_b.numel()) {
        throw TRException("[CPU Dimension] Element count mismatch for squeeze_into: " +
                         std::to_string(tensor_a.numel()) + " vs " + std::to_string(tensor_b.numel()));
    }

    // 执行squeeze操作（本质上是数据复制）
    squeeze_operation_core(tensor_a, tensor_b);
}

} // namespace tr
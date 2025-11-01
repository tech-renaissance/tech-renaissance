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
#include <cstring>

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
    // 检查数据类型 - 支持FP32和INT8
    if (tensor.dtype() != DType::FP32 && tensor.dtype() != DType::INT8) {
        throw TRException("[CPU Dimension] Currently only supports FP32 and INT8 tensor operations, " + name + " data type is: " +
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

// ===== Pad 操作核心实现 =====

/**
 * @brief 计算padding后的新形状
 * @param original_shape 原始形状
 * @param padding padding大小
 * @return 新的形状
 * @throws TRException 如果张量维度小于2维
 */
static Shape calculate_pad_shape(const Shape& original_shape, int32_t padding) {
    int32_t original_ndim = original_shape.ndim();

    // 验证维度至少为2
    if (original_ndim < 2) {
        throw TRException("[CPU Pad] Tensor must have at least 2 dimensions");
    }

    // 构建新的形状数组
    std::vector<int32_t> new_dims;

    if (original_ndim == 2) {
        // (H, W) -> (H+2*padding, W+2*padding)
        int32_t h = original_shape.dim(0);
        int32_t w = original_shape.dim(1);
        new_dims.push_back(h + 2 * padding);
        new_dims.push_back(w + 2 * padding);
    } else if (original_ndim == 3) {
        // (C, H, W) -> (C, H+2*padding, W+2*padding)
        int32_t c = original_shape.dim(0);
        int32_t h = original_shape.dim(1);
        int32_t w = original_shape.dim(2);
        new_dims.push_back(c);
        new_dims.push_back(h + 2 * padding);
        new_dims.push_back(w + 2 * padding);
    } else if (original_ndim == 4) {
        // (N, C, H, W) -> (N, C, H+2*padding, W+2*padding)
        int32_t n = original_shape.dim(0);
        int32_t c = original_shape.dim(1);
        int32_t h = original_shape.dim(2);
        int32_t w = original_shape.dim(3);
        new_dims.push_back(n);
        new_dims.push_back(c);
        new_dims.push_back(h + 2 * padding);
        new_dims.push_back(w + 2 * padding);
    }

    // 创建新形状
    Shape new_shape;
    if (new_dims.size() == 2) {
        new_shape = Shape(new_dims[0], new_dims[1]);
    } else if (new_dims.size() == 3) {
        new_shape = Shape(new_dims[0], new_dims[1], new_dims[2]);
    } else if (new_dims.size() == 4) {
        new_shape = Shape(new_dims[0], new_dims[1], new_dims[2], new_dims[3]);
    }

    return new_shape;
}

/**
 * @brief 执行padding操作的核心实现（FP32版本）
 * @param input 输入张量
 * @param result 输出张量（预分配了正确的形状）
 * @param padding padding大小
 */
static void pad_operation_core_fp32(const Tensor& input, Tensor& result, int32_t padding) {
    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    // 获取形状信息
    int32_t input_ndim = input.shape().ndim();

    // 首先将整个输出张量填充为0
    std::memset(result_data, 0, result.numel() * sizeof(float));

    if (input_ndim == 2) {
        // 2D情况：(H, W) -> (H+2p, W+2p)
        int32_t input_h = input.shape().dim(0);
        int32_t input_w = input.shape().dim(1);
        int32_t output_h = result.shape().dim(0);
        int32_t output_w = result.shape().dim(1);

        // 将输入数据复制到中心位置
        for (int32_t i = 0; i < input_h; ++i) {
            for (int32_t j = 0; j < input_w; ++j) {
                int32_t src_idx = i * input_w + j;
                int32_t dst_idx = (i + padding) * output_w + (j + padding);
                result_data[dst_idx] = input_data[src_idx];
            }
        }

    } else if (input_ndim == 3) {
        // 3D情况：(C, H, W) -> (C, H+2p, W+2p)
        int32_t input_c = input.shape().dim(0);
        int32_t input_h = input.shape().dim(1);
        int32_t input_w = input.shape().dim(2);
        int32_t output_h = result.shape().dim(1);
        int32_t output_w = result.shape().dim(2);

        for (int32_t c = 0; c < input_c; ++c) {
            for (int32_t i = 0; i < input_h; ++i) {
                for (int32_t j = 0; j < input_w; ++j) {
                    int32_t src_idx = c * input_h * input_w + i * input_w + j;
                    int32_t dst_idx = c * output_h * output_w +
                                   (i + padding) * output_w + (j + padding);
                    result_data[dst_idx] = input_data[src_idx];
                }
            }
        }

    } else if (input_ndim == 4) {
        // 4D情况：(N, C, H, W) -> (N, C, H+2p, W+2p)
        int32_t input_n = input.shape().dim(0);
        int32_t input_c = input.shape().dim(1);
        int32_t input_h = input.shape().dim(2);
        int32_t input_w = input.shape().dim(3);
        int32_t output_h = result.shape().dim(2);
        int32_t output_w = result.shape().dim(3);

        for (int32_t n = 0; n < input_n; ++n) {
            for (int32_t c = 0; c < input_c; ++c) {
                for (int32_t i = 0; i < input_h; ++i) {
                    for (int32_t j = 0; j < input_w; ++j) {
                        int32_t src_idx = n * input_c * input_h * input_w +
                                       c * input_h * input_w + i * input_w + j;
                        int32_t dst_idx = n * input_c * output_h * output_w +
                                       c * output_h * output_w +
                                       (i + padding) * output_w + (j + padding);
                        result_data[dst_idx] = input_data[src_idx];
                    }
                }
            }
        }
    }
}

/**
 * @brief 执行padding操作的核心实现（INT8版本）
 * @param input 输入张量
 * @param result 输出张量（预分配了正确的形状）
 * @param padding padding大小
 */
static void pad_operation_core_int8(const Tensor& input, Tensor& result, int32_t padding) {
    const int8_t* input_data = static_cast<const int8_t*>(input.data_ptr());
    int8_t* result_data = static_cast<int8_t*>(result.data_ptr());

    // 获取形状信息
    int32_t input_ndim = input.shape().ndim();

    // 首先将整个输出张量填充为0
    std::memset(result_data, 0, result.numel() * sizeof(int8_t));

    if (input_ndim == 2) {
        // 2D情况：(H, W) -> (H+2p, W+2p)
        int32_t input_h = input.shape().dim(0);
        int32_t input_w = input.shape().dim(1);
        int32_t output_h = result.shape().dim(0);
        int32_t output_w = result.shape().dim(1);

        // 将输入数据复制到中心位置
        for (int32_t i = 0; i < input_h; ++i) {
            for (int32_t j = 0; j < input_w; ++j) {
                int32_t src_idx = i * input_w + j;
                int32_t dst_idx = (i + padding) * output_w + (j + padding);
                result_data[dst_idx] = input_data[src_idx];
            }
        }

    } else if (input_ndim == 3) {
        // 3D情况：(C, H, W) -> (C, H+2p, W+2p)
        int32_t input_c = input.shape().dim(0);
        int32_t input_h = input.shape().dim(1);
        int32_t input_w = input.shape().dim(2);
        int32_t output_h = result.shape().dim(1);
        int32_t output_w = result.shape().dim(2);

        for (int32_t c = 0; c < input_c; ++c) {
            for (int32_t i = 0; i < input_h; ++i) {
                for (int32_t j = 0; j < input_w; ++j) {
                    int32_t src_idx = c * input_h * input_w + i * input_w + j;
                    int32_t dst_idx = c * output_h * output_w +
                                   (i + padding) * output_w + (j + padding);
                    result_data[dst_idx] = input_data[src_idx];
                }
            }
        }

    } else if (input_ndim == 4) {
        // 4D情况：(N, C, H, W) -> (N, C, H+2p, W+2p)
        int32_t input_n = input.shape().dim(0);
        int32_t input_c = input.shape().dim(1);
        int32_t input_h = input.shape().dim(2);
        int32_t input_w = input.shape().dim(3);
        int32_t output_h = result.shape().dim(2);
        int32_t output_w = result.shape().dim(3);

        for (int32_t n = 0; n < input_n; ++n) {
            for (int32_t c = 0; c < input_c; ++c) {
                for (int32_t i = 0; i < input_h; ++i) {
                    for (int32_t j = 0; j < input_w; ++j) {
                        int32_t src_idx = n * input_c * input_h * input_w +
                                       c * input_h * input_w + i * input_w + j;
                        int32_t dst_idx = n * input_c * output_h * output_w +
                                       c * output_h * output_w +
                                       (i + padding) * output_w + (j + padding);
                        result_data[dst_idx] = input_data[src_idx];
                    }
                }
            }
        }
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

// ===== Pad 操作实现 =====

Tensor CpuBackend::pad(const Tensor& tensor_a, int32_t padding) const {
    // 验证padding值
    if (padding < 0) {
        throw TRException("[CPU Pad] Padding value must be non-negative");
    }

    // 验证数据类型和设备
    validate_tensor_dtype_and_device(tensor_a, "tensor_a");

    // 处理空张量特殊情况
    if (tensor_a.is_empty()) {
        return Tensor();  // 返回空张量
    }

    // 验证非空张量
    validate_tensor_not_empty(tensor_a, "tensor_a");

    // 计算新形状
    Shape new_shape = calculate_pad_shape(tensor_a.shape(), padding);

    // 创建输出张量
    Tensor result = Tensor::empty(new_shape, tensor_a.dtype(), tr::CPU);

    // 执行padding操作
    if (tensor_a.dtype() == DType::FP32) {
        pad_operation_core_fp32(tensor_a, result, padding);
    } else if (tensor_a.dtype() == DType::INT8) {
        pad_operation_core_int8(tensor_a, result, padding);
    } else {
        throw TRException("[CPU Pad] Unsupported data type: " + dtype_to_string(tensor_a.dtype()));
    }

    return result;
}

void CpuBackend::pad_into(const Tensor& tensor_a, int32_t padding, Tensor& tensor_b) const {
    // 验证padding值
    if (padding < 0) {
        throw TRException("[CPU Pad] Padding value must be non-negative");
    }

    // 验证数据类型和设备
    validate_tensor_dtype_and_device(tensor_a, "tensor_a");
    validate_tensor_dtype_and_device(tensor_b, "tensor_b");

    // 处理空张量特殊情况
    if (tensor_a.is_empty()) {
        if (tensor_b.is_empty()) {
            return;  // 两个都是空张量，直接返回
        } else {
            throw TRException("[CPU Pad] Cannot pad empty tensor into non-empty tensor");
        }
    }

    // 验证非空张量
    validate_tensor_not_empty(tensor_a, "tensor_a");
    validate_tensor_not_empty(tensor_b, "tensor_b");

    // 验证数据类型一致
    if (tensor_a.dtype() != tensor_b.dtype()) {
        throw TRException("[CPU Pad] Data type mismatch for pad_into: " +
                         dtype_to_string(tensor_a.dtype()) + " vs " +
                         dtype_to_string(tensor_b.dtype()));
    }

    // 验证输出张量形状是否正确
    Shape expected_shape = calculate_pad_shape(tensor_a.shape(), padding);
    if (tensor_b.shape() != expected_shape) {
        throw TRException("[CPU Pad] Output tensor shape mismatch for pad_into. Expected: " +
                         expected_shape.to_string() + ", Actual: " + tensor_b.shape().to_string());
    }

    // 执行padding操作
    if (tensor_a.dtype() == DType::FP32) {
        pad_operation_core_fp32(tensor_a, tensor_b, padding);
    } else if (tensor_a.dtype() == DType::INT8) {
        pad_operation_core_int8(tensor_a, tensor_b, padding);
    } else {
        throw TRException("[CPU Pad] Unsupported data type: " + dtype_to_string(tensor_a.dtype()));
    }
}

} // namespace tr
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

// ===== 维度操作核心实现 =====

/**
 * @brief 规范化维度索引，支持负数索引
 * @param dim 原始维度索引
 * @param ndim 张量维度数量
 * @return 规范化后的维度索引
 * @throws TRException 如果维度索引超出范围
 */
static int32_t normalize_dim(int32_t dim, int32_t ndim) {
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw TRException("[CPU Dim] Dimension index out of range");
    }
    return dim;
}

/**
 * @brief 验证维度操作的基本参数
 * @param tensor 输入张量
 * @param dim 维度索引
 * @param operation_name 操作名称
 * @throws TRException 如果参数无效
 */
static void validate_dim_operation(const Tensor& tensor, int32_t dim, const std::string& operation_name) {
    // 验证数据类型和设备
    if (tensor.device() != tr::CPU) {
        throw TRException("[CPU " + operation_name + "] Device must be CPU");
    }

    // 验证内存是否已分配
    if (!tensor.storage_allocated()) {
        throw TRException("[CPU " + operation_name + "] Tensor storage not allocated");
    }

    // 验证不是标量
    if (tensor.shape().ndim() == 0) {
        throw TRException("[CPU " + operation_name + "] Cannot perform operation on scalar tensor");
    }

    // 规范化并验证维度索引
    int32_t ndim = tensor.shape().ndim();
    dim = normalize_dim(dim, ndim);

    // 验证指定维度的大小不为0
    if (tensor.shape().dim(dim) == 0) {
        throw TRException("[CPU " + operation_name + "] Cannot perform operation on dimension with size 0");
    }
}

/**
 * @brief 计算reduction操作后的输出形状
 * @param input_shape 输入形状
 * @param dim 规范化后的维度索引
 * @param keep_dim 是否保留维度
 * @return 输出形状
 */
static Shape calculate_reduction_shape(const Shape& input_shape, int32_t dim, bool keep_dim) {
    int32_t ndim = input_shape.ndim();
    std::vector<int32_t> output_dims;

    for (int32_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keep_dim) {
                output_dims.push_back(1);
            }
            // 如果keep_dim为false，跳过该维度
        } else {
            output_dims.push_back(input_shape.dim(i));
        }
    }

    // 处理特殊情况：如果所有维度都被移除了（不应该发生，因为我们已经排除了标量）
    if (output_dims.empty()) {
        return Shape();  // 返回标量
    }

    // 根据输出维度数量创建Shape
    Shape result;
    if (output_dims.size() == 1) {
        result = Shape(output_dims[0]);
    } else if (output_dims.size() == 2) {
        result = Shape(output_dims[0], output_dims[1]);
    } else if (output_dims.size() == 3) {
        result = Shape(output_dims[0], output_dims[1], output_dims[2]);
    } else if (output_dims.size() == 4) {
        result = Shape(output_dims[0], output_dims[1], output_dims[2], output_dims[3]);
    }

    return result;
}

// ===== Softmax 操作实现 =====

/**
 * @brief 执行softmax操作的核心实现（FP32版本）
 * @param input 输入张量
 * @param result 输出张量
 * @param dim 规范化后的维度索引
 */
static void softmax_operation_core_fp32(const Tensor& input, Tensor& result, int32_t dim) {
    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    const Shape& shape = input.shape();
    int32_t ndim = shape.ndim();

    // 计算每个切片的元素数量和总切片数量
    int32_t slice_size = shape.dim(dim);
    int32_t num_slices = input.numel() / slice_size;

    // 计算每个切片的步长
    std::vector<int32_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (int32_t i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape.dim(i + 1);
    }
    int32_t dim_stride = strides[dim];

    // 对每个切片执行softmax
    for (int32_t slice = 0; slice < num_slices; ++slice) {
        // 找到该切片的起始位置
        int32_t slice_start = (slice / dim_stride) * (dim_stride * slice_size) + (slice % dim_stride);

        // 找到最大值（数值稳定性）
        float max_val = input_data[slice_start];
        for (int32_t i = 1; i < slice_size; ++i) {
            int32_t idx = slice_start + i * dim_stride;
            max_val = std::max(max_val, input_data[idx]);
        }

        // 计算exp(x - max_val)的和
        float sum = 0.0f;
        std::vector<float> exp_values(slice_size);
        for (int32_t i = 0; i < slice_size; ++i) {
            int32_t idx = slice_start + i * dim_stride;
            exp_values[i] = std::exp(input_data[idx] - max_val);
            sum += exp_values[i];
        }

        // 计算softmax值
        for (int32_t i = 0; i < slice_size; ++i) {
            int32_t idx = slice_start + i * dim_stride;
            result_data[idx] = exp_values[i] / sum;
        }
    }
}

Tensor CpuBackend::softmax(const Tensor& tensor_a, int32_t dim) {
    // 验证输入
    validate_dim_operation(tensor_a, dim, "Softmax");
    dim = normalize_dim(dim, tensor_a.shape().ndim());

    // 验证数据类型
    if (tensor_a.dtype() != DType::FP32) {
        throw TRException("[CPU Softmax] Only supports FP32 tensor");
    }

    // 创建输出张量（与输入同形状）
    Tensor result = this->empty(tensor_a.shape(), DType::FP32);

    // 执行softmax操作
    softmax_operation_core_fp32(tensor_a, result, dim);

    return result;
}

void CpuBackend::softmax_inplace(Tensor& tensor_a, int32_t dim) {
    // 验证输入
    validate_dim_operation(tensor_a, dim, "Softmax");
    dim = normalize_dim(dim, tensor_a.shape().ndim());

    // 验证数据类型
    if (tensor_a.dtype() != DType::FP32) {
        throw TRException("[CPU Softmax] Only supports FP32 tensor");
    }

    // 创建临时张量存储结果
    Tensor temp = this->empty(tensor_a.shape(), DType::FP32);
    softmax_operation_core_fp32(tensor_a, temp, dim);

    // 将结果复制回原张量
    this->copy_into(temp, tensor_a);
}

void CpuBackend::softmax_into(const Tensor& tensor_a, Tensor& result, int32_t dim) {
    // 验证输入
    validate_dim_operation(tensor_a, dim, "Softmax");
    dim = normalize_dim(dim, tensor_a.shape().ndim());

    // 验证输出张量
    if (result.device() != tr::CPU) {
        throw TRException("[CPU Softmax] Result tensor must be on CPU device");
    }
    if (!result.storage_allocated()) {
        throw TRException("[CPU Softmax] Result tensor storage not allocated");
    }
    if (tensor_a.dtype() != DType::FP32 || result.dtype() != DType::FP32) {
        throw TRException("[CPU Softmax] Only supports FP32 tensor");
    }
    if (tensor_a.shape() != result.shape()) {
        throw TRException("[CPU Softmax] Input and result shapes must match");
    }

    // 执行softmax操作
    softmax_operation_core_fp32(tensor_a, result, dim);
}

// ===== Max 操作实现 =====

/**
 * @brief 执行max操作的核心实现（模板函数）
 * @tparam T 数据类型
 * @param input 输入张量
 * @param result 输出张量
 * @param dim 规范化后的维度索引
 */
template<typename T>
static void max_operation_core(const Tensor& input, Tensor& result, int32_t dim, bool keep_dim) {
    const T* input_data = static_cast<const T*>(input.data_ptr());
    T* result_data = static_cast<T*>(result.data_ptr());

    const Shape& input_shape = input.shape();
    const Shape& result_shape = result.shape();
    int32_t ndim = input_shape.ndim();

    // 计算步长
    std::vector<int32_t> input_strides(ndim);
    input_strides[ndim - 1] = 1;
    for (int32_t i = ndim - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_shape.dim(i + 1);
    }

    std::vector<int32_t> result_strides;
    if (result_shape.ndim() == 0) {
        // 标量结果
        result_strides.push_back(1);
    } else {
        result_strides.resize(result_shape.ndim());
        result_strides[result_shape.ndim() - 1] = 1;
        for (int32_t i = result_shape.ndim() - 2; i >= 0; --i) {
            result_strides[i] = result_strides[i + 1] * result_shape.dim(i + 1);
        }
    }

    int32_t dim_size = input_shape.dim(dim);
    int32_t dim_stride = input_strides[dim];

    // 计算总迭代次数
    int64_t total_iterations = result.numel();

    for (int64_t out_idx = 0; out_idx < total_iterations; ++out_idx) {
        // 计算输入张量中的对应位置
        int32_t input_idx = 0;
        int32_t temp_out_idx = out_idx;

        if (keep_dim) {
            // keep_dim=true的情形：输出维度数与输入相同
            for (int32_t i = 0; i < ndim; ++i) {
                int32_t coord;

                if (i == dim) {
                    // 这是reduction维度，坐标总是0
                    coord = 0;
                } else {
                    // 正常维度，从输出索引计算坐标
                    coord = temp_out_idx / result_strides[i];
                    temp_out_idx %= result_strides[i];
                }

                input_idx += coord * input_strides[i];
            }
        } else {
            // keep_dim=false的情形：输出维度数比输入少1
            for (int32_t i = 0; i < ndim; ++i) {
                if (i == dim) {
                    continue;  // 跳过reduction维度
                }

                int32_t result_dim = i;
                if (i > dim) {
                    result_dim = i - 1;  // 如果移除了前面的维度，需要调整
                }

                int32_t coord = temp_out_idx / result_strides[result_dim];
                temp_out_idx %= result_strides[result_dim];
                input_idx += coord * input_strides[i];
            }
        }

        // 在指定维度上找最大值
        T max_val = input_data[input_idx];
        for (int32_t i = 1; i < dim_size; ++i) {
            int32_t idx = input_idx + i * dim_stride;
            max_val = std::max(max_val, input_data[idx]);
        }

        result_data[out_idx] = max_val;
    }
}

Tensor CpuBackend::max(const Tensor& tensor_a, int32_t dim, bool keep_dim) {
    // 验证输入
    validate_dim_operation(tensor_a, dim, "Max");
    dim = normalize_dim(dim, tensor_a.shape().ndim());

    // 计算输出形状
    Shape output_shape = calculate_reduction_shape(tensor_a.shape(), dim, keep_dim);

    // 创建输出张量
    Tensor result = this->empty(output_shape, tensor_a.dtype());

    // 执行max操作
    if (tensor_a.dtype() == DType::FP32) {
        max_operation_core<float>(tensor_a, result, dim, keep_dim);
    } else if (tensor_a.dtype() == DType::INT8) {
        max_operation_core<int8_t>(tensor_a, result, dim, keep_dim);
    } else if (tensor_a.dtype() == DType::INT32) {
        max_operation_core<int32_t>(tensor_a, result, dim, keep_dim);
    } else {
        throw TRException("[CPU Max] Unsupported data type: " + dtype_to_string(tensor_a.dtype()));
    }

    return result;
}

void CpuBackend::max_into(const Tensor& tensor_a, Tensor& result, int32_t dim, bool keep_dim) {
    // 验证输入
    validate_dim_operation(tensor_a, dim, "Max");
    dim = normalize_dim(dim, tensor_a.shape().ndim());

    // 验证输出张量
    if (result.device() != tr::CPU) {
        throw TRException("[CPU Max] Result tensor must be on CPU device");
    }
    if (!result.storage_allocated()) {
        throw TRException("[CPU Max] Result tensor storage not allocated");
    }
    if (tensor_a.dtype() != result.dtype()) {
        throw TRException("[CPU Max] Data type mismatch");
    }

    // 验证输出形状
    Shape expected_shape = calculate_reduction_shape(tensor_a.shape(), dim, keep_dim);
    if (result.shape() != expected_shape) {
        throw TRException("[CPU Max] Result shape mismatch. expected: " + expected_shape.to_string() +
                        ", actual: " + result.shape().to_string());
    }

    // 执行max操作
    if (tensor_a.dtype() == DType::FP32) {
        max_operation_core<float>(tensor_a, result, dim, keep_dim);
    } else if (tensor_a.dtype() == DType::INT8) {
        max_operation_core<int8_t>(tensor_a, result, dim, keep_dim);
    } else if (tensor_a.dtype() == DType::INT32) {
        max_operation_core<int32_t>(tensor_a, result, dim, keep_dim);
    } else {
        throw TRException("[CPU Max] Unsupported data type: " + dtype_to_string(tensor_a.dtype()));
    }
}

// ===== Sum 操作实现 =====

/**
 * @brief 执行sum操作的核心实现（FP32版本）
 * @param input 输入张量
 * @param result 输出张量
 * @param dim 规范化后的维度索引
 */
static void sum_operation_core_fp32(const Tensor& input, Tensor& result, int32_t dim) {
    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    const Shape& input_shape = input.shape();
    const Shape& result_shape = result.shape();
    int32_t ndim = input_shape.ndim();

    // 计算步长
    std::vector<int32_t> input_strides(ndim);
    input_strides[ndim - 1] = 1;
    for (int32_t i = ndim - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_shape.dim(i + 1);
    }

    std::vector<int32_t> result_strides;
    if (result_shape.ndim() == 0) {
        result_strides.push_back(1);
    } else {
        result_strides.resize(result_shape.ndim());
        result_strides[result_shape.ndim() - 1] = 1;
        for (int32_t i = result_shape.ndim() - 2; i >= 0; --i) {
            result_strides[i] = result_strides[i + 1] * result_shape.dim(i + 1);
        }
    }

    int32_t dim_size = input_shape.dim(dim);
    int32_t dim_stride = input_strides[dim];

    // 初始化结果为0
    std::memset(result_data, 0, result.numel() * sizeof(float));

    // 计算总迭代次数
    int64_t total_iterations = result.numel();

    for (int64_t out_idx = 0; out_idx < total_iterations; ++out_idx) {
        // 计算输入张量中的对应位置
        int32_t input_idx = 0;
        int32_t temp_out_idx = out_idx;

        for (int32_t i = 0; i < ndim; ++i) {
            if (i == dim) {
                continue;
            }

            int32_t result_dim = i;
            if (i > dim) {
                result_dim = i - 1;
            }

            int32_t coord = temp_out_idx / result_strides[result_dim];
            temp_out_idx %= result_strides[result_dim];
            input_idx += coord * input_strides[i];
        }

        // 在指定维度上求和
        float sum_val = 0.0f;
        for (int32_t i = 0; i < dim_size; ++i) {
            int32_t idx = input_idx + i * dim_stride;
            sum_val += input_data[idx];
        }

        result_data[out_idx] = sum_val;
    }
}

Tensor CpuBackend::sum(const Tensor& tensor_a, int32_t dim, bool keep_dim) {
    // 验证输入
    validate_dim_operation(tensor_a, dim, "Sum");
    dim = normalize_dim(dim, tensor_a.shape().ndim());

    // 验证数据类型
    if (tensor_a.dtype() != DType::FP32) {
        throw TRException("[CPU Sum] Only supports FP32 tensor");
    }

    // 计算输出形状
    Shape output_shape = calculate_reduction_shape(tensor_a.shape(), dim, keep_dim);

    // 创建输出张量
    Tensor result = this->empty(output_shape, DType::FP32);

    // 执行sum操作
    sum_operation_core_fp32(tensor_a, result, dim);

    return result;
}

void CpuBackend::sum_into(const Tensor& tensor_a, Tensor& result, int32_t dim, bool keep_dim) {
    // 验证输入
    validate_dim_operation(tensor_a, dim, "Sum");
    dim = normalize_dim(dim, tensor_a.shape().ndim());

    // 验证输出张量
    if (result.device() != tr::CPU) {
        throw TRException("[CPU Sum] Result tensor must be on CPU device");
    }
    if (!result.storage_allocated()) {
        throw TRException("[CPU Sum] Result tensor storage not allocated");
    }
    if (tensor_a.dtype() != DType::FP32 || result.dtype() != DType::FP32) {
        throw TRException("[CPU Sum] Only supports FP32 tensor");
    }

    // 验证输出形状
    Shape expected_shape = calculate_reduction_shape(tensor_a.shape(), dim, keep_dim);
    if (result.shape() != expected_shape) {
        throw TRException("[CPU Sum] Result shape mismatch. expected: " + expected_shape.to_string() +
                        ", actual: " + result.shape().to_string());
    }

    // 执行sum操作
    sum_operation_core_fp32(tensor_a, result, dim);
}

// ===== ArgMax 操作实现 =====

/**
 * @brief 执行argmax操作的核心实现（模板函数）
 * @tparam T 数据类型
 * @param input 输入张量
 * @param result 输出张量
 * @param dim 规范化后的维度索引
 */
template<typename T>
static void argmax_operation_core(const Tensor& input, Tensor& result, int32_t dim, bool keep_dim) {
    const T* input_data = static_cast<const T*>(input.data_ptr());
    int32_t* result_data = static_cast<int32_t*>(result.data_ptr());

    const Shape& input_shape = input.shape();
    const Shape& result_shape = result.shape();
    int32_t ndim = input_shape.ndim();

    // 计算步长
    std::vector<int32_t> input_strides(ndim);
    input_strides[ndim - 1] = 1;
    for (int32_t i = ndim - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_shape.dim(i + 1);
    }

    std::vector<int32_t> result_strides;
    if (result_shape.ndim() == 0) {
        result_strides.push_back(1);
    } else {
        result_strides.resize(result_shape.ndim());
        result_strides[result_shape.ndim() - 1] = 1;
        for (int32_t i = result_shape.ndim() - 2; i >= 0; --i) {
            result_strides[i] = result_strides[i + 1] * result_shape.dim(i + 1);
        }
    }

    int32_t dim_size = input_shape.dim(dim);
    int32_t dim_stride = input_strides[dim];

    // 计算总迭代次数
    int64_t total_iterations = result.numel();

    for (int64_t out_idx = 0; out_idx < total_iterations; ++out_idx) {
        // 计算输入张量中的对应位置
        int32_t input_idx = 0;
        int32_t temp_out_idx = out_idx;

        if (keep_dim) {
            // keep_dim=true的情形：输出维度数与输入相同
            for (int32_t i = 0; i < ndim; ++i) {
                int32_t coord;

                if (i == dim) {
                    // 这是reduction维度，坐标总是0
                    coord = 0;
                } else {
                    // 正常维度，从输出索引计算坐标
                    coord = temp_out_idx / result_strides[i];
                    temp_out_idx %= result_strides[i];
                }

                input_idx += coord * input_strides[i];
            }
        } else {
            // keep_dim=false的情形：输出维度数比输入少1
            for (int32_t i = 0; i < ndim; ++i) {
                if (i == dim) {
                    continue;  // 跳过reduction维度
                }

                int32_t result_dim = i;
                if (i > dim) {
                    result_dim = i - 1;  // 如果移除了前面的维度，需要调整
                }

                int32_t coord = temp_out_idx / result_strides[result_dim];
                temp_out_idx %= result_strides[result_dim];
                input_idx += coord * input_strides[i];
            }
        }

        // 在指定维度上找最大值的索引
        T max_val = input_data[input_idx];
        int32_t max_idx = 0;
        for (int32_t i = 1; i < dim_size; ++i) {
            int32_t idx = input_idx + i * dim_stride;
            if (input_data[idx] > max_val) {
                max_val = input_data[idx];
                max_idx = i;
            }
        }

        result_data[out_idx] = max_idx;
    }
}

Tensor CpuBackend::argmax(const Tensor& tensor_a, int32_t dim, bool keep_dim) {
    // 验证输入
    validate_dim_operation(tensor_a, dim, "ArgMax");
    dim = normalize_dim(dim, tensor_a.shape().ndim());

    // 计算输出形状
    Shape output_shape = calculate_reduction_shape(tensor_a.shape(), dim, keep_dim);

    // 创建输出张量（总是INT32类型）
    Tensor result = this->empty(output_shape, DType::INT32);

    // 执行argmax操作
    if (tensor_a.dtype() == DType::FP32) {
        argmax_operation_core<float>(tensor_a, result, dim, keep_dim);
    } else if (tensor_a.dtype() == DType::INT8) {
        argmax_operation_core<int8_t>(tensor_a, result, dim, keep_dim);
    } else if (tensor_a.dtype() == DType::INT32) {
        argmax_operation_core<int32_t>(tensor_a, result, dim, keep_dim);
    } else {
        throw TRException("[CPU ArgMax] Unsupported data type: " + dtype_to_string(tensor_a.dtype()));
    }

    return result;
}

void CpuBackend::argmax_into(const Tensor& tensor_a, Tensor& result, int32_t dim, bool keep_dim) {
    // 验证输入
    validate_dim_operation(tensor_a, dim, "ArgMax");
    dim = normalize_dim(dim, tensor_a.shape().ndim());

    // 验证输出张量
    if (result.device() != tr::CPU) {
        throw TRException("[CPU ArgMax] Result tensor must be on CPU device");
    }
    if (!result.storage_allocated()) {
        throw TRException("[CPU ArgMax] Result tensor storage not allocated");
    }
    if (result.dtype() != DType::INT32) {
        throw TRException("[CPU ArgMax] Result tensor must be INT32 type");
    }

    // 验证输出形状
    Shape expected_shape = calculate_reduction_shape(tensor_a.shape(), dim, keep_dim);
    if (result.shape() != expected_shape) {
        throw TRException("[CPU ArgMax] Result shape mismatch. expected: " + expected_shape.to_string() +
                        ", actual: " + result.shape().to_string());
    }

    // 执行argmax操作
    if (tensor_a.dtype() == DType::FP32) {
        argmax_operation_core<float>(tensor_a, result, dim, keep_dim);
    } else if (tensor_a.dtype() == DType::INT8) {
        argmax_operation_core<int8_t>(tensor_a, result, dim, keep_dim);
    } else if (tensor_a.dtype() == DType::INT32) {
        argmax_operation_core<int32_t>(tensor_a, result, dim, keep_dim);
    } else {
        throw TRException("[CPU ArgMax] Unsupported data type: " + dtype_to_string(tensor_a.dtype()));
    }
}

} // namespace tr
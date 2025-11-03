/**
 * @file cpu_pooling.cpp
 * @brief CPU后端张量池化操作实现
 * @details 实现最大池化和全局平均池化操作，支持朴素版本和Eigen加速版本
 * @version 1.00.00
 * @date 2025-11-03
 * @author 技术觉醒团队
 * @note 依赖项: cpu_backend.h, tensor.h, eigen3/Eigen
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
#include <cmath>

namespace tr {

// ===== 辅助函数 =====

/**
 * @brief 验证池化操作的基本参数
 * @param tensor 输入张量
 * @param kernel_size 卷积核大小
 * @param stride 步长
 * @param operation_name 操作名称
 * @throws TRException 如果参数无效
 */
static void validate_pooling_parameters(const Tensor& tensor, int32_t kernel_size, int32_t stride, const std::string& operation_name) {
    // 验证设备
    if (tensor.device() != tr::CPU) {
        throw TRException("[CPU " + operation_name + "] Device must be CPU");
    }

    // 验证内存是否已分配
    if (!tensor.storage_allocated()) {
        throw TRException("[CPU " + operation_name + "] Tensor storage not allocated");
    }

    // 验证数据类型
    if (tensor.dtype() != DType::FP32) {
        throw TRException("[CPU " + operation_name + "] Only supports FP32 tensor");
    }

    // 验证张量维度（至少2D）
    if (tensor.shape().ndim() < 2) {
        throw TRException("[CPU " + operation_name + "] Tensor must have at least 2 dimensions");
    }

    // 验证kernel_size
    if (kernel_size <= 0) {
        throw TRException("[CPU " + operation_name + "] Kernel size must be positive");
    }

    // 验证stride
    if (stride <= 0) {
        throw TRException("[CPU " + operation_name + "] Stride must be positive");
    }

    // 验证stride范围（只支持1或2）
    if (stride != 1 && stride != 2) {
        throw TRException("[CPU " + operation_name + "] Only supports stride 1 or 2");
    }
}

/**
 * @brief 计算max pooling的输出形状
 * @param input_shape 输入形状
 * @param kernel_size 卷积核大小
 * @param stride 步长
 * @return 输出形状
 */
static Shape calculate_max_pool_shape(const Shape& input_shape, int32_t kernel_size, int32_t stride) {
    int32_t ndim = input_shape.ndim();
    std::vector<int32_t> output_dims;

    for (int32_t i = 0; i < ndim - 2; ++i) {
        output_dims.push_back(input_shape.dim(i));
    }

    int32_t input_h = input_shape.h();
    int32_t input_w = input_shape.w();

    // 计算输出高度和宽度: o = floor((i + 2p - k) / s) + 1
    // 这里padding=0，所以简化为: o = floor((i - k) / s) + 1
    int32_t output_h = (input_h - kernel_size) / stride + 1;
    int32_t output_w = (input_w - kernel_size) / stride + 1;

    if (output_h <= 0 || output_w <= 0) {
        throw TRException("[CPU MaxPool] Kernel size is too large for input tensor");
    }

    output_dims.push_back(output_h);
    output_dims.push_back(output_w);

    // 创建Shape对象
    Shape result;
    if (output_dims.size() == 2) {
        result = Shape(output_dims[0], output_dims[1]);
    } else if (output_dims.size() == 3) {
        result = Shape(output_dims[0], output_dims[1], output_dims[2]);
    } else if (output_dims.size() == 4) {
        result = Shape(output_dims[0], output_dims[1], output_dims[2], output_dims[3]);
    }

    return result;
}

/**
 * @brief 计算全局平均池化的输出形状
 * @param input_shape 输入形状
 * @return 输出形状
 */
static Shape calculate_global_avg_pool_shape(const Shape& input_shape) {
    int32_t ndim = input_shape.ndim();
    std::vector<int32_t> output_dims;

    for (int32_t i = 0; i < ndim - 2; ++i) {
        output_dims.push_back(input_shape.dim(i));
    }

    // H和W维度必定是1
    output_dims.push_back(1);
    output_dims.push_back(1);

    // 创建Shape对象
    Shape result;
    if (output_dims.size() == 2) {
        result = Shape(output_dims[0], output_dims[1]);
    } else if (output_dims.size() == 3) {
        result = Shape(output_dims[0], output_dims[1], output_dims[2]);
    } else if (output_dims.size() == 4) {
        result = Shape(output_dims[0], output_dims[1], output_dims[2], output_dims[3]);
    }

    return result;
}

// ===== Max Pooling 朴素实现 =====

/**
 * @brief 执行max pooling的核心实现（朴素版本）
 * @param input 输入张量
 * @param result 输出张量
 * @param kernel_size 卷积核大小
 * @param stride 步长
 */
static void max_pool_operation_core_naive(const Tensor& input, Tensor& result, int32_t kernel_size, int32_t stride) {
    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    const Shape& input_shape = input.shape();
    const Shape& result_shape = result.shape();
    int32_t ndim = input_shape.ndim();

    int32_t input_h = input_shape.h();
    int32_t input_w = input_shape.w();
    int32_t output_h = result_shape.h();
    int32_t output_w = result_shape.w();

    // 计算每个维度的步长
    std::vector<int32_t> input_strides(ndim);
    input_strides[ndim - 1] = 1;
    for (int32_t i = ndim - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_shape.dim(i + 1);
    }

    std::vector<int32_t> result_strides(ndim);
    result_strides[ndim - 1] = 1;
    for (int32_t i = ndim - 2; i >= 0; --i) {
        result_strides[i] = result_strides[i + 1] * result_shape.dim(i + 1);
    }

    // 计算batch和channel的维度
    int32_t batch_size = (ndim >= 4) ? input_shape.n() : 1;
    int32_t channel_size = (ndim >= 3) ? input_shape.c() : 1;

    // 对每个batch和channel执行max pooling
    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t c = 0; c < channel_size; ++c) {
            for (int32_t oh = 0; oh < output_h; ++oh) {
                for (int32_t ow = 0; ow < output_w; ++ow) {
                    // 计算输入中的起始位置
                    int32_t ih_start = oh * stride;
                    int32_t iw_start = ow * stride;

                    // 计算池化窗口内的最大值
                    float max_val = -std::numeric_limits<float>::infinity();

                    for (int32_t kh = 0; kh < kernel_size; ++kh) {
                        for (int32_t kw = 0; kw < kernel_size; ++kw) {
                            int32_t ih = ih_start + kh;
                            int32_t iw = iw_start + kw;

                            if (ih < input_h && iw < input_w) {
                                int32_t input_idx = 0;
                                if (ndim == 4) {
                                    input_idx = b * input_strides[0] + c * input_strides[1] + ih * input_strides[2] + iw;
                                } else if (ndim == 3) {
                                    input_idx = c * input_strides[0] + ih * input_strides[1] + iw;
                                } else { // ndim == 2
                                    input_idx = ih * input_strides[0] + iw;
                                }

                                max_val = std::max(max_val, input_data[input_idx]);
                            }
                        }
                    }

                    // 写入结果
                    int32_t result_idx = 0;
                    if (ndim == 4) {
                        result_idx = b * result_strides[0] + c * result_strides[1] + oh * result_strides[2] + ow;
                    } else if (ndim == 3) {
                        result_idx = c * result_strides[0] + oh * result_strides[1] + ow;
                    } else { // ndim == 2
                        result_idx = oh * result_strides[0] + ow;
                    }

                    result_data[result_idx] = max_val;
                }
            }
        }
    }
}

// ===== Max Pooling Eigen实现 =====

/**
 * @brief 执行max pooling的核心实现（Eigen版本）
 * @param input 输入张量
 * @param result 输出张量
 * @param kernel_size 卷积核大小
 * @param stride 步长
 */
static void max_pool_operation_core_eigen(const Tensor& input, Tensor& result, int32_t kernel_size, int32_t stride) {
    // 对于当前实现，Eigen版本使用相同的朴素算法
    // 未来可以用Eigen::Map和Eigen::Reducer进行优化
    max_pool_operation_core_naive(input, result, kernel_size, stride);
}

// ===== Global Average Pooling 朴素实现 =====

/**
 * @brief 执行全局平均池化的核心实现（朴素版本）
 * @param input 输入张量
 * @param result 输出张量
 */
static void global_avg_pool_operation_core_naive(const Tensor& input, Tensor& result) {
    const float* input_data = static_cast<const float*>(input.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    const Shape& input_shape = input.shape();
    int32_t ndim = input_shape.ndim();

    int32_t input_h = input_shape.h();
    int32_t input_w = input_shape.w();
    int32_t pool_size = input_h * input_w;

    // 计算每个维度的步长
    std::vector<int32_t> input_strides(ndim);
    input_strides[ndim - 1] = 1;
    for (int32_t i = ndim - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_shape.dim(i + 1);
    }

    // 计算batch和channel的维度
    int32_t batch_size = (ndim >= 4) ? input_shape.n() : 1;
    int32_t channel_size = (ndim >= 3) ? input_shape.c() : 1;

    // 对每个batch和channel执行全局平均池化
    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t c = 0; c < channel_size; ++c) {
            float sum_val = 0.0f;

            // 计算整个空间维度的总和
            for (int32_t ih = 0; ih < input_h; ++ih) {
                for (int32_t iw = 0; iw < input_w; ++iw) {
                    int32_t input_idx = 0;
                    if (ndim == 4) {
                        input_idx = b * input_strides[0] + c * input_strides[1] + ih * input_strides[2] + iw;
                    } else if (ndim == 3) {
                        input_idx = c * input_strides[0] + ih * input_strides[1] + iw;
                    } else { // ndim == 2
                        input_idx = ih * input_strides[0] + iw;
                    }

                    sum_val += input_data[input_idx];
                }
            }

            // 计算平均值并写入结果
            float avg_val = sum_val / pool_size;
            int32_t result_idx = 0;
            if (ndim == 4) {
                result_idx = b * 1 + c * 1;  // H和W都是1，所以步长为1
            } else if (ndim == 3) {
                result_idx = c * 1;  // W是1，所以步长为1
            } else { // ndim == 2
                result_idx = 0;  // 标量情况
            }

            result_data[result_idx] = avg_val;
        }
    }
}

// ===== Global Average Pooling Eigen实现 =====

/**
 * @brief 执行全局平均池化的核心实现（Eigen版本）
 * @param input 输入张量
 * @param result 输出张量
 */
static void global_avg_pool_operation_core_eigen(const Tensor& input, Tensor& result) {
    // 对于当前实现，Eigen版本使用相同的朴素算法
    // 未来可以用Eigen::Map和Eigen::Reducer进行优化
    global_avg_pool_operation_core_naive(input, result);
}

// ===== 公共API实现 =====

Tensor CpuBackend::max_pool(const Tensor& input, int32_t kernel_size, int32_t stride) {
    // 验证输入参数
    validate_pooling_parameters(input, kernel_size, stride, "MaxPool");

    // 计算输出形状
    Shape output_shape = calculate_max_pool_shape(input.shape(), kernel_size, stride);

    // 创建输出张量
    Tensor result = this->empty(output_shape, DType::FP32);

    // 执行max pooling操作
    max_pool_operation_core_naive(input, result, kernel_size, stride);

    return result;
}

void CpuBackend::max_pool_into(const Tensor& input, Tensor& result, int32_t kernel_size, int32_t stride) {
    // 验证输入参数
    validate_pooling_parameters(input, kernel_size, stride, "MaxPool");

    // 验证输出张量
    if (result.device() != tr::CPU) {
        throw TRException("[CPU MaxPool] Result tensor must be on CPU device");
    }
    if (!result.storage_allocated()) {
        throw TRException("[CPU MaxPool] Result tensor storage not allocated");
    }
    if (result.dtype() != DType::FP32) {
        throw TRException("[CPU MaxPool] Result tensor must be FP32 type");
    }

    // 验证输出形状
    Shape expected_shape = calculate_max_pool_shape(input.shape(), kernel_size, stride);
    if (result.shape() != expected_shape) {
        throw TRException("[CPU MaxPool] Result shape mismatch. expected: " + expected_shape.to_string() +
                        ", actual: " + result.shape().to_string());
    }

    // 执行max pooling操作
    max_pool_operation_core_naive(input, result, kernel_size, stride);
}

Tensor CpuBackend::global_avg_pool(const Tensor& input) {
    // 验证输入参数
    validate_pooling_parameters(input, 2, 1, "GlobalAvgPool");  // kernel_size=2, stride=1只是占位符

    // 计算输出形状
    Shape output_shape = calculate_global_avg_pool_shape(input.shape());

    // 创建输出张量
    Tensor result = this->empty(output_shape, DType::FP32);

    // 执行全局平均池化操作
    global_avg_pool_operation_core_naive(input, result);

    return result;
}

void CpuBackend::global_avg_pool_into(const Tensor& input, Tensor& result) {
    // 验证输入参数
    validate_pooling_parameters(input, 2, 1, "GlobalAvgPool");  // kernel_size=2, stride=1只是占位符

    // 验证输出张量
    if (result.device() != tr::CPU) {
        throw TRException("[CPU GlobalAvgPool] Result tensor must be on CPU device");
    }
    if (!result.storage_allocated()) {
        throw TRException("[CPU GlobalAvgPool] Result tensor storage not allocated");
    }
    if (result.dtype() != DType::FP32) {
        throw TRException("[CPU GlobalAvgPool] Result tensor must be FP32 type");
    }

    // 验证输出形状
    Shape expected_shape = calculate_global_avg_pool_shape(input.shape());
    if (result.shape() != expected_shape) {
        throw TRException("[CPU GlobalAvgPool] Result shape mismatch. expected: " + expected_shape.to_string() +
                        ", actual: " + result.shape().to_string());
    }

    // 执行全局平均池化操作
    global_avg_pool_operation_core_naive(input, result);
}

} // namespace tr
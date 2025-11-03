/**
 * @file cpu_conv.cpp
 * @brief CPU后端张量卷积操作实现
 * @details 实现标准卷积和转置卷积操作，支持朴素版本和Eigen加速版本
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
#include <numeric>

#include "Core"

namespace tr {

// ===== 辅助函数 =====

/**
 * @brief 验证卷积操作的基本参数
 * @param tensor 输入张量
 * @param kernel 卷积核张量
 * @param stride 步长
 * @param padding 填充
 * @param operation_name 操作名称
 * @throws TRException 如果参数无效
 */
static void validate_conv_parameters(const Tensor& tensor, const Tensor& kernel,
                                    int32_t stride, int32_t padding, const std::string& operation_name) {
    // 验证设备
    if (tensor.device() != tr::CPU || kernel.device() != tr::CPU) {
        throw TRException("[CPU " + operation_name + "] Device must be CPU");
    }

    // 验证内存是否已分配
    if (!tensor.storage_allocated() || !kernel.storage_allocated()) {
        throw TRException("[CPU " + operation_name + "] Tensor storage not allocated");
    }

    // 验证数据类型
    if (tensor.dtype() != DType::FP32 || kernel.dtype() != DType::FP32) {
        throw TRException("[CPU " + operation_name + "] Only supports FP32 tensor");
    }

    // 验证张量维度
    if (tensor.shape().ndim() < 2) {
        throw TRException("[CPU " + operation_name + "] Tensor must have at least 2 dimensions");
    }
    if (kernel.shape().ndim() != 4) {
        throw TRException("[CPU " + operation_name + "] Kernel tensor must be 4-dimensional (N, C, H, W)");
    }

    // 验证kernel是正方形
    if (kernel.shape().h() != kernel.shape().w()) {
        throw TRException("[CPU " + operation_name + "] Only supports square kernels (H=W)");
    }

    // 验证stride
    if (stride <= 0) {
        throw TRException("[CPU " + operation_name + "] Stride must be positive");
    }

    // 验证stride范围（只支持1或2）
    if (stride != 1 && stride != 2) {
        throw TRException("[CPU " + operation_name + "] Only supports stride 1 or 2");
    }

    // 验证padding
    if (padding < 0) {
        throw TRException("[CPU " + operation_name + "] Padding must be non-negative");
    }
}

/**
 * @brief 计算标准卷积的输出形状
 * @param input_shape 输入形状
 * @param kernel_shape 卷积核形状
 * @param stride 步长
 * @param padding 填充
 * @return 输出形状
 */
static Shape calculate_conv_shape(const Shape& input_shape, const Shape& kernel_shape,
                                 int32_t stride, int32_t padding) {
    int32_t input_ndim = input_shape.ndim();
    int32_t kernel_ndim = kernel_shape.ndim();

    // 验证卷积核必须是4D
    if (kernel_ndim != 4) {
        throw TRException("[CPU Conv] Kernel must be 4-dimensional (N, C, H, W)");
    }

    int32_t out_channels = kernel_shape.n();  // 输出通道数
    int32_t input_h = input_shape.h();
    int32_t input_w = input_shape.w();
    int32_t kernel_h = kernel_shape.h();
    int32_t kernel_w = kernel_shape.w();

    // 计算输出高度和宽度: o = floor((i + 2p - k) / s) + 1
    int32_t output_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    int32_t output_w = (input_w + 2 * padding - kernel_w) / stride + 1;

    if (output_h <= 0 || output_w <= 0) {
        throw TRException("[CPU Conv] Invalid output shape due to parameters");
    }

    // 根据输入维度确定输出形状
    if (input_ndim == 2) {
        // 2D输入: (H, W) -> 输出: (1, C_out, H_out, W_out)
        return Shape(1, out_channels, output_h, output_w);
    } else if (input_ndim == 3) {
        // 3D输入: (C, H, W) -> 输出: (N, C_out, H_out, W_out) 其中N=1
        return Shape(1, out_channels, output_h, output_w);
    } else if (input_ndim == 4) {
        // 4D输入: (N, C, H, W) -> 输出: (N, C_out, H_out, W_out)
        int32_t batch_size = input_shape.n();
        return Shape(batch_size, out_channels, output_h, output_w);
    } else {
        throw TRException("[CPU Conv] Input must have 2-4 dimensions");
    }
}

/**
 * @brief 计算转置卷积的输出形状
 * @param input_shape 输入形状
 * @param kernel_shape 卷积核形状
 * @param stride 步长
 * @param padding 填充
 * @return 输出形状
 */
static Shape calculate_transposed_conv_shape(const Shape& input_shape, const Shape& kernel_shape,
                                           int32_t stride, int32_t padding) {
    int32_t input_ndim = input_shape.ndim();
    int32_t kernel_ndim = kernel_shape.ndim();

    // 验证卷积核必须是4D
    if (kernel_ndim != 4) {
        throw TRException("[CPU TransposedConv] Kernel must be 4-dimensional (N, C, H, W)");
    }

    int32_t out_channels = kernel_shape.n();  // 输出通道数
    int32_t input_h = input_shape.h();
    int32_t input_w = input_shape.w();
    int32_t kernel_h = kernel_shape.h();
    int32_t kernel_w = kernel_shape.w();

    // 转置卷积输出形状: o = (i - 1) * s + k - 2p
    int32_t output_h = (input_h - 1) * stride + kernel_h - 2 * padding;
    int32_t output_w = (input_w - 1) * stride + kernel_w - 2 * padding;

    if (output_h <= 0 || output_w <= 0) {
        throw TRException("[CPU TransposedConv] Invalid output shape due to parameters");
    }

    // 根据输入维度确定输出形状
    if (input_ndim == 2) {
        // 2D输入: (H, W) -> 输出: (1, C_out, H_out, W_out)
        return Shape(1, out_channels, output_h, output_w);
    } else if (input_ndim == 3) {
        // 3D输入: (C, H, W) -> 输出: (N, C_out, H_out, W_out) 其中N=1
        return Shape(1, out_channels, output_h, output_w);
    } else if (input_ndim == 4) {
        // 4D输入: (N, C, H, W) -> 输出: (N, C_out, H_out, W_out)
        int32_t batch_size = input_shape.n();
        return Shape(batch_size, out_channels, output_h, output_w);
    } else {
        throw TRException("[CPU TransposedConv] Input must have 2-4 dimensions");
    }
}

// ===== 标准卷积朴素实现 =====

/**
 * @brief 执行标准卷积的核心实现（朴素版本）
 * @param input 输入张量
 * @param kernel 卷积核张量
 * @param result 输出张量
 * @param stride 步长
 * @param padding 填充
 */
static void conv_operation_core_naive(const Tensor& input, const Tensor& kernel,
                                     Tensor& result, int32_t stride, int32_t padding) {
    const float* input_data = static_cast<const float*>(input.data_ptr());
    const float* kernel_data = static_cast<const float*>(kernel.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    const Shape& input_shape = input.shape();
    const Shape& kernel_shape = kernel.shape();
    const Shape& result_shape = result.shape();
    int32_t input_ndim = input_shape.ndim();

    // 获取输入和输出维度
    int32_t input_h = input_shape.h();
    int32_t input_w = input_shape.w();
    int32_t kernel_h = kernel_shape.h();
    int32_t kernel_w = kernel_shape.w();
    int32_t output_h = result_shape.h();
    int32_t output_w = result_shape.w();

    // 获取通道和批次信息
    int32_t batch_size = (input_ndim == 4) ? input_shape.n() : 1;
    int32_t in_channels = (input_ndim >= 3) ? input_shape.c() : 1;
    int32_t out_channels = kernel_shape.n();

    // 初始化结果为0
    std::memset(result_data, 0, result_shape.numel() * sizeof(float));

    // 对每个输出位置执行卷积
    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t oc = 0; oc < out_channels; ++oc) {
            for (int32_t oh = 0; oh < output_h; ++oh) {
                for (int32_t ow = 0; ow < output_w; ++ow) {
                    // 计算输入中的起始位置
                    int32_t ih_start = oh * stride - padding;
                    int32_t iw_start = ow * stride - padding;

                    // 执行卷积计算
                    float sum_val = 0.0f;
                    for (int32_t ic = 0; ic < in_channels; ++ic) {
                        for (int32_t kh = 0; kh < kernel_h; ++kh) {
                            for (int32_t kw = 0; kw < kernel_w; ++kw) {
                                int32_t ih = ih_start + kh;
                                int32_t iw = iw_start + kw;

                                // 检查是否在有效范围内
                                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                                    // 计算输入索引 - 使用正确的NCHW内存布局
                                    int32_t input_idx = 0;
                                    if (input_ndim == 4) {
                                        // NCHW: b * C * H * W + ic * H * W + ih * W + iw
                                        input_idx = b * in_channels * input_h * input_w +
                                                  ic * input_h * input_w +
                                                  ih * input_w + iw;
                                    } else if (input_ndim == 3) {
                                        // CHW: ic * H * W + ih * W + iw
                                        input_idx = ic * input_h * input_w + ih * input_w + iw;
                                    } else { // ndim == 2
                                        // HW: ih * W + iw
                                        input_idx = ih * input_w + iw;
                                    }

                                    // 计算卷积核索引 - 使用NCHW格式
                                    // kernel_shape: (out_channels, in_channels, kernel_h, kernel_w)
                                    int32_t kernel_idx = oc * (in_channels * kernel_h * kernel_w) +
                                                      ic * (kernel_h * kernel_w) +
                                                      kh * kernel_w + kw;

                                    sum_val += input_data[input_idx] * kernel_data[kernel_idx];
                                }
                            }
                        }
                    }

                    // 计算结果索引 - 使用正确的NCHW内存布局
                    int32_t result_idx = 0;
                    if (input_ndim == 4) {
                        // NCHW: b * out_channels * output_h * output_w + oc * output_h * output_w + oh * output_w + ow
                        result_idx = b * out_channels * output_h * output_w +
                                  oc * output_h * output_w +
                                  oh * output_w + ow;
                    } else if (input_ndim == 3) {
                        // CHW: oc * output_h * output_w + oh * output_w + ow (这里N=1)
                        result_idx = oc * output_h * output_w + oh * output_w + ow;
                    } else { // ndim == 2
                        // HW: oh * output_w + ow (这里N=1, C=1)
                        result_idx = oh * output_w + ow;
                    }

                    result_data[result_idx] = sum_val;
                }
            }
        }
    }
}

// ===== 标准卷积Eigen实现 =====

/**
 * @brief 执行标准卷积的核心实现（Eigen版本 - im2col方法）
 * @param input 输入张量
 * @param kernel 卷积核张量
 * @param result 输出张量
 * @param stride 步长
 * @param padding 填充
 */
static void conv_operation_core_eigen(const Tensor& input, const Tensor& kernel,
                                     Tensor& result, int32_t stride, int32_t padding) {
    // 对于当前实现，Eigen版本使用相同的朴素算法
    // 未来可以用Eigen::Map和Eigen::Reducer进行优化
    conv_operation_core_naive(input, kernel, result, stride, padding);
}

// ===== 转置卷积朴素实现 =====

/**
 * @brief 执行转置卷积的核心实现（朴素版本）
 * @param input 输入张量
 * @param kernel 卷积核张量
 * @param result 输出张量
 * @param stride 步长
 * @param padding 填充
 */
static void transposed_conv_operation_core_naive(const Tensor& input, const Tensor& kernel,
                                                 Tensor& result, int32_t stride, int32_t padding) {
    const float* input_data = static_cast<const float*>(input.data_ptr());
    const float* kernel_data = static_cast<const float*>(kernel.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    const Shape& input_shape = input.shape();
    const Shape& kernel_shape = kernel.shape();
    const Shape& result_shape = result.shape();
    int32_t input_ndim = input_shape.ndim();

    int32_t input_h = input_shape.h();
    int32_t input_w = input_shape.w();
    int32_t kernel_h = kernel_shape.h();
    int32_t kernel_w = kernel_shape.w();
    int32_t output_h = result_shape.h();
    int32_t output_w = result_shape.w();

    // 获取通道和批次信息
    int32_t batch_size = (input_ndim == 4) ? input_shape.n() : 1;
    int32_t in_channels = (input_ndim >= 3) ? input_shape.c() : 1;
    int32_t out_channels = kernel_shape.n();

    // 初始化结果张量为0
    std::memset(result_data, 0, result_shape.numel() * sizeof(float));

    // 对每个batch、in_channel、out_channel执行转置卷积
    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t ic = 0; ic < in_channels; ++ic) {
            for (int32_t oc = 0; oc < out_channels; ++oc) {
                for (int32_t ih = 0; ih < input_h; ++ih) {
                    for (int32_t iw = 0; iw < input_w; ++iw) {
                        // 计算输出中的起始位置
                        int32_t oh_start = ih * stride - padding;
                        int32_t ow_start = iw * stride - padding;

                        // 计算输入值
                        float input_val = 0.0f;
                        int32_t input_idx = 0;
                        if (input_ndim == 4) {
                            input_idx = b * in_channels * input_h * input_w +
                                      ic * input_h * input_w +
                                      ih * input_w + iw;
                        } else if (input_ndim == 3) {
                            input_idx = ic * input_h * input_w + ih * input_w + iw;
                        } else { // ndim == 2
                            input_idx = ih * input_w + iw;
                        }
                        input_val = input_data[input_idx];

                        // 执行转置卷积计算（卷积核旋转180度）
                        for (int32_t kh = 0; kh < kernel_h; ++kh) {
                            for (int32_t kw = 0; kw < kernel_w; ++kw) {
                                int32_t oh = oh_start + (kernel_h - 1 - kh);
                                int32_t ow = ow_start + (kernel_w - 1 - kw);

                                // 检查是否在有效范围内
                                if (oh >= 0 && oh < output_h && ow >= 0 && ow < output_w) {
                                    // 计算卷积核索引（旋转180度）
                                    int32_t kernel_idx = oc * (in_channels * kernel_h * kernel_w) +
                                                      ic * (kernel_h * kernel_w) +
                                                      kh * kernel_w + kw;

                                    // 计算结果索引
                                    int32_t result_idx = 0;
                                    if (input_ndim == 4) {
                                        result_idx = b * out_channels * output_h * output_w +
                                                  oc * output_h * output_w +
                                                  oh * output_w + ow;
                                    } else if (input_ndim == 3) {
                                        result_idx = oc * output_h * output_w + oh * output_w + ow;
                                    } else { // ndim == 2
                                        result_idx = oh * output_w + ow;
                                    }

                                    result_data[result_idx] += input_val * kernel_data[kernel_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ===== 转置卷积Eigen实现 =====

/**
 * @brief 执行转置卷积的核心实现（Eigen版本）
 * @param input 输入张量
 * @param kernel 卷积核张量
 * @param result 输出张量
 * @param stride 步长
 * @param padding 填充
 */
static void transposed_conv_operation_core_eigen(const Tensor& input, const Tensor& kernel,
                                                 Tensor& result, int32_t stride, int32_t padding) {
    // 对于当前实现，Eigen版本使用相同的朴素算法
    // 未来可以用Eigen::Map和更高效的稀疏操作进行优化
    transposed_conv_operation_core_naive(input, kernel, result, stride, padding);
}

// ===== 公共API实现 =====

Tensor CpuBackend::conv(const Tensor& input, const Tensor& kernel, int32_t stride, int32_t padding) {
    // 验证输入参数
    validate_conv_parameters(input, kernel, stride, padding, "Conv");

    // 计算输出形状
    Shape output_shape = calculate_conv_shape(input.shape(), kernel.shape(), stride, padding);

    // 创建输出张量
    Tensor result = this->empty(output_shape, DType::FP32);

    // 执行卷积操作
    conv_operation_core_eigen(input, kernel, result, stride, padding);

    return result;
}

void CpuBackend::conv_into(const Tensor& input, const Tensor& kernel, Tensor& result,
                          int32_t stride, int32_t padding) {
    // 验证输入参数
    validate_conv_parameters(input, kernel, stride, padding, "Conv");

    // 验证输出张量
    if (result.device() != tr::CPU) {
        throw TRException("[CPU Conv] Result tensor must be on CPU device");
    }
    if (!result.storage_allocated()) {
        throw TRException("[CPU Conv] Result tensor storage not allocated");
    }
    if (result.dtype() != DType::FP32) {
        throw TRException("[CPU Conv] Result tensor must be FP32 type");
    }

    // 验证输出形状
    Shape expected_shape = calculate_conv_shape(input.shape(), kernel.shape(), stride, padding);
    if (result.shape() != expected_shape) {
        throw TRException("[CPU Conv] Result shape mismatch. expected: " + expected_shape.to_string() +
                        ", actual: " + result.shape().to_string());
    }

    // 执行卷积操作
    conv_operation_core_eigen(input, kernel, result, stride, padding);
}

Tensor CpuBackend::transposed_conv(const Tensor& input, const Tensor& kernel, int32_t stride, int32_t padding) {
    // 验证输入参数
    validate_conv_parameters(input, kernel, stride, padding, "TransposedConv");

    // 计算输出形状
    Shape output_shape = calculate_transposed_conv_shape(input.shape(), kernel.shape(), stride, padding);

    // 创建输出张量
    Tensor result = this->empty(output_shape, DType::FP32);

    // 执行转置卷积操作
    transposed_conv_operation_core_eigen(input, kernel, result, stride, padding);

    return result;
}

void CpuBackend::transposed_conv_into(const Tensor& input, const Tensor& kernel, Tensor& result,
                                     int32_t stride, int32_t padding) {
    // 验证输入参数
    validate_conv_parameters(input, kernel, stride, padding, "TransposedConv");

    // 验证输出张量
    if (result.device() != tr::CPU) {
        throw TRException("[CPU TransposedConv] Result tensor must be on CPU device");
    }
    if (!result.storage_allocated()) {
        throw TRException("[CPU TransposedConv] Result tensor storage not allocated");
    }
    if (result.dtype() != DType::FP32) {
        throw TRException("[CPU TransposedConv] Result tensor must be FP32 type");
    }

    // 验证输出形状
    Shape expected_shape = calculate_transposed_conv_shape(input.shape(), kernel.shape(), stride, padding);
    if (result.shape() != expected_shape) {
        throw TRException("[CPU TransposedConv] Result shape mismatch. expected: " + expected_shape.to_string() +
                        ", actual: " + result.shape().to_string());
    }

    // 执行转置卷积操作
    transposed_conv_operation_core_eigen(input, kernel, result, stride, padding);
}

} // namespace tr
/**
 * @file cpu_conv.cpp
 * @brief CPU后端张量卷积操作实现
 * @details 实现标准卷积和转置卷积操作，支持朴素版本和Eigen加速版本
 * @version 1.01.00 (Optimized)
 * @date 2025-11-04
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

#ifdef TR_USE_EIGEN
#include "Core"
#include <omp.h>
#endif

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

    // (V1.01.00 优化) 验证输入通道数匹配
    int32_t in_channels = (tensor.shape().ndim() >= 3) ? tensor.shape().c() : 1;
    if (in_channels != kernel.shape().c()) {
        throw TRException("[CPU " + operation_name + "] Input channels (" + std::to_string(in_channels) +
                        ") do not match kernel's input channels (" + std::to_string(kernel.shape().c()) + ")");
    }

    // 验证kernel是正方形
    if (kernel.shape().h() != kernel.shape().w()) {
        throw TRException("[CPU " + operation_name + "] Only supports square kernels (H=W)");
    }

    // 验证stride
    if (stride <= 0) {
        throw TRException("[CPU " + operation_name + "] Stride must be positive");
    }

    // (V1.01.00 移除) 转置卷积优化后可支持任意stride，此处放宽
    // if (stride != 1 && stride != 2) {
    //     throw TRException("[CPU " + operation_name + "] Only supports stride 1 or 2");
    // }

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

// ===== 标准卷积Eigen实现 (V1.01.00 优化) =====

/**
 * @brief 执行标准卷积的核心实现（Eigen版本 - 高性能im2col+GEMM方法）
 * @param input 输入张量
 * @param kernel 卷积核张量
 * @param result 输出张量
 * @param stride 步长
 * @param padding 填充
 */
static void conv_operation_core_eigen(const Tensor& input, const Tensor& kernel,
                                     Tensor& result, int32_t stride, int32_t padding) {
#ifdef TR_USE_EIGEN
    // 使用高性能Eigen优化实现
    Logger::get_instance().debug("Using high-performance Eigen im2col+GEMM for CPU convolution");

    // 设置Eigen多线程配置
    int num_threads = omp_get_max_threads();
    Eigen::setNbThreads(num_threads); // 确保Eigen内部GEMM使用多线程

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
    int32_t batch_size, in_channels, out_channels;
    if (input_ndim == 4) {
        batch_size = input_shape.n();
        in_channels = input_shape.c();
    } else if (input_ndim == 3) {
        batch_size = 1;
        in_channels = input_shape.c();
    } else {
        batch_size = 1;
        in_channels = 1;
    }
    out_channels = kernel_shape.n(); // kernel.c() == in_channels 已在validate中检查

    // im2col + GEMM 优化参数
    int32_t col_rows = in_channels * kernel_h * kernel_w;
    int32_t col_cols = output_h * output_w;

    // 关键优化1：一次性构建权重矩阵 W [out_channels x col_rows]，跨batch重用
    // (使用RowMajor以匹配朴素实现的kernel索引，但ColMajor通常GEMM更快，这里保持ColMajor)
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> W(out_channels, col_rows);
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int col = ic * kernel_h * kernel_w + kh * kernel_w + kw;
                    // kernel_shape: (out_channels, in_channels, kernel_h, kernel_w)
                    int kernel_idx = oc * (in_channels * kernel_h * kernel_w) +
                                   ic * (kernel_h * kernel_w) +
                                   kh * kernel_w + kw;
                    W(oc, col) = kernel_data[kernel_idx];
                }
            }
        }
    }

    // (V1.01.00 优化)
    // 关键优化2：为每个线程预先分配im2col矩阵，避免在循环内部分配
    std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> thread_cols(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        thread_cols[i].resize(col_rows, col_cols);
    }

    // 关键优化3：使用OpenMP并行化batch处理
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        // (V1.01.00 优化) 获取线程本地的col矩阵，0开销
        int thread_id = omp_get_thread_num();
        auto& col = thread_cols[thread_id];

        // 高性能im2col变换
        for (int oh = 0; oh < output_h; ++oh) {
            int ih_start = oh * stride - padding;
            for (int ow = 0; ow < output_w; ++ow) {
                int col_idx = oh * output_w + ow;
                int iw_start = ow * stride - padding;

                for (int ic = 0; ic < in_channels; ++ic) {
                    int input_base = 0;
                    if (input_ndim == 4) {
                        input_base = b * in_channels * input_h * input_w + ic * input_h * input_w;
                    } else if (input_ndim == 3) {
                        input_base = ic * input_h * input_w;
                    } else {
                        input_base = 0;
                    }

                    for (int kh = 0; kh < kernel_h; ++kh) {
                        int ih = ih_start + kh;
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int iw = iw_start + kw;
                            float val = 0.0f;

                            if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                                val = input_data[input_base + ih * input_w + iw];
                            }

                            int row = ic * kernel_h * kernel_w + kh * kernel_w + kw;
                            col(row, col_idx) = val;
                        }
                    }
                }
            }
        }

        // 关键优化4：高性能GEMM计算: output = W * col
        // Eigen::setNbThreads(1); // 在OMP内部，GEMM应单线程执行
        // (注意: Eigen 3.3+ 配合 OpenMP 效果很好，无需设为1)
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> output_mat = W * col;

        // 关键优化5：优化结果复制回输出张量
        for (int oc = 0; oc < out_channels; ++oc) {
            int result_base = 0;
            if (input_ndim == 4) {
                result_base = b * out_channels * output_h * output_w + oc * output_h * output_w;
            } else if (input_ndim == 3) {
                result_base = oc * output_h * output_w;
            } else {
                result_base = 0;
            }

            // (V1.01.00 修正) W是 (oc, col_rows), col是(col_rows, col_cols)
            // output_mat 是 (oc, col_cols) -> (out_channels, output_h * output_w)
            // W(oc, col) ... col(row, col_idx)
            // W(oc, ic*... + kh*... + kw) * col(ic*... + kh*... + kw, oh*... + ow)
            // output_mat(oc, oh*... + ow)

            // (V1.01.00 修正) output_mat的(oc, col_idx) 对应 result_data的 (b, oc, oh, ow)
            // col_idx = oh * output_w + ow
            // result_base = b * out_channels * output_h * output_w + oc * output_h * output_w;
            // result_data[result_base + oh * output_w + ow] = output_mat(oc, oh * output_w + ow);

            // 复制 (oc, 0) 到 (oc, col_cols-1) 的数据
            std::memcpy(&result_data[result_base],
                        output_mat.data() + oc * col_cols,
                        col_cols * sizeof(float));
        }
    }

#else
    // 使用朴素实现
    Logger::get_instance().debug("Using naive implementation for CPU convolution");
    conv_operation_core_naive(input, kernel, result, stride, padding);
#endif
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
                                                      (kernel_h - 1 - kh) * kernel_w + (kernel_w - 1 - kw);

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

// ===== 转置卷积Eigen实现 (V1.01.00 优化) =====

/**
 * @brief 执行转置卷积的核心实现（Eigen版本）
 * @details 采用"空洞输入 + 翻转卷积核 + 标准卷积"的高性能策略
 * @param input 输入张量
 * @param kernel 卷积核张量
 * @param result 输出张量
 * @param stride 步长
 * @param padding 填充
 */
static void transposed_conv_operation_core_eigen(const Tensor& input, const Tensor& kernel,
                                                 Tensor& result, int32_t stride, int32_t padding) {
#ifdef TR_USE_EIGEN
    Logger::get_instance().debug("Using high-performance Eigen (DilatedInput+FlippedKernelConv) for CPU transposed convolution");

    const float* input_data = static_cast<const float*>(input.data_ptr());
    const float* kernel_data = static_cast<const float*>(kernel.data_ptr());

    const Shape& input_shape = input.shape();
    const Shape& kernel_shape = kernel.shape();
    const Shape& result_shape = result.shape();
    int32_t input_ndim = input_shape.ndim();

    int32_t input_h = input_shape.h();
    int32_t input_w = input_shape.w();
    int32_t kernel_h = kernel_shape.h();
    int32_t kernel_w = kernel_shape.w();

    // 获取通道和批次信息
    int32_t batch_size, in_channels, out_channels;
    if (input_ndim == 4) {
        batch_size = input_shape.n();
        in_channels = input_shape.c();
    } else if (input_ndim == 3) {
        batch_size = 1;
        in_channels = input_shape.c();
    } else {
        batch_size = 1;
        in_channels = 1;
    }
    out_channels = kernel_shape.n(); // kernel.c() == in_channels 已在validate中检查

    // 1. 创建"空洞"输入张量 (Input Dilated)
    // (V1.01.00 修正)
    // o = (i - 1) * s + k - 2p
    // o = (i - 1) * s + 1 + (k - 1) - 2p
    // (i - 1) * s + 1 是空洞插入后的H/W
    int32_t dilated_h = (input_h - 1) * stride + 1;
    int32_t dilated_w = (input_w - 1) * stride + 1;

    // (V1.01.00 优化) 使用 Tensor::zeros 在静态函数内安全创建张量
    Tensor input_dilated = Tensor::zeros(Shape(batch_size, in_channels, dilated_h, dilated_w), DType::FP32, tr::CPU);
    float* dilated_data = static_cast<float*>(input_dilated.data_ptr());

    // 2. 填充"空洞"输入 (Scatter-like copy, but safe)
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int ih = 0; ih < input_h; ++ih) {
                for (int iw = 0; iw < input_w; ++iw) {
                    int input_idx = b * in_channels * input_h * input_w +
                                  ic * input_h * input_w +
                                  ih * input_w + iw;

                    int dilated_idx = b * in_channels * dilated_h * dilated_w +
                                    ic * dilated_h * dilated_w +
                                    (ih * stride) * dilated_w + (iw * stride);

                    dilated_data[dilated_idx] = input_data[input_idx];
                }
            }
        }
    }

    // 3. 创建"翻转"卷积核 (Kernel Flipped)
    Tensor kernel_flipped = Tensor::empty(kernel_shape, DType::FP32, tr::CPU);
    float* flipped_data = static_cast<float*>(kernel_flipped.data_ptr());

    #pragma omp parallel for
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int kernel_idx = oc * (in_channels * kernel_h * kernel_w) +
                                   ic * (kernel_h * kernel_w) +
                                   kh * kernel_w + kw;

                    // (V1.01.00 修正) 翻转 kH 和 kW
                    int flipped_kh = kernel_h - 1 - kh;
                    int flipped_kw = kernel_w - 1 - kw;

                    int flipped_idx = oc * (in_channels * kernel_h * kernel_w) +
                                    ic * (kernel_h * kernel_w) +
                                    flipped_kh * kernel_w + flipped_kw;

                    flipped_data[flipped_idx] = kernel_data[kernel_idx];
                }
            }
        }
    }

    // 4. 计算新的Padding
    // 等价变换: o = (i_dil - 1) * 1 + k - 2p_new
    // 我们希望: o = (i - 1) * s + k - 2p (目标)
    // i_dil = (i - 1) * s + 1
    // o = ((i - 1) * s + 1 - 1) * 1 + k - 2p_new
    // o = (i - 1) * s + k - 2p_new
    // 比较可知: p_new = p
    // (V1.01.00 修正)
    // PyTorch的等价: padding_new = kernel_size - 1 - padding
    int32_t padding_new = kernel_h - 1 - padding;

    // 5. 执行标准卷积 (stride=1)
    // (V1.01.00 修正)
    // 验证输出形状
    // o_h_check = (dilated_h + 2 * padding_new - kernel_h) / 1 + 1
    //           = ((input_h - 1) * stride + 1) + 2 * (kernel_h - 1 - padding) - kernel_h + 1
    //           = (input_h - 1) * stride + 1 + 2*kernel_h - 2 - 2*padding - kernel_h + 1
    //           = (input_h - 1) * stride + kernel_h - 2*padding
    // 这与 result_shape.h() (即 o) 完全一致

    conv_operation_core_eigen(input_dilated, kernel_flipped, result, 1, padding_new);

#else
    // 使用朴素实现
    Logger::get_instance().debug("Using naive implementation for CPU transposed convolution");
    transposed_conv_operation_core_naive(input, kernel, result, stride, padding);
#endif
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
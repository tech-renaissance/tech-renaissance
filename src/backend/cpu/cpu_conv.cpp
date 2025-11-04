/**
 * @file cpu_conv.cpp
 * @brief CPU后端张量卷积操作实现（性能优化版 - 方案D）
 * @details 实现标准卷积和转置卷积操作，使用高性能im2col/col2im + GEMM方法
 * @version 1.01.00 (Optimized - Plan D)
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
 */
static void validate_conv_parameters(const Tensor& tensor, const Tensor& kernel,
                                    int32_t stride, int32_t padding, const std::string& operation_name) {
    if (tensor.device() != tr::CPU || kernel.device() != tr::CPU) {
        throw TRException("[CPU " + operation_name + "] Device must be CPU");
    }

    if (!tensor.storage_allocated() || !kernel.storage_allocated()) {
        throw TRException("[CPU " + operation_name + "] Tensor storage not allocated");
    }

    if (tensor.dtype() != DType::FP32 || kernel.dtype() != DType::FP32) {
        throw TRException("[CPU " + operation_name + "] Only supports FP32 tensor");
    }

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

    if (kernel.shape().h() != kernel.shape().w()) {
        throw TRException("[CPU " + operation_name + "] Only supports square kernels (H=W)");
    }

    if (stride <= 0) {
        throw TRException("[CPU " + operation_name + "] Stride must be positive");
    }

    // (方案D) 保留stride限制
    if (stride != 1 && stride != 2) {
        throw TRException("[CPU " + operation_name + "] Only supports stride 1 or 2");
    }

    if (padding < 0) {
        throw TRException("[CPU " + operation_name + "] Padding must be non-negative");
    }
}

/**
 * @brief 计算标准卷积的输出形状
 */
static Shape calculate_conv_shape(const Shape& input_shape, const Shape& kernel_shape,
                                 int32_t stride, int32_t padding) {
    int32_t input_ndim = input_shape.ndim();
    int32_t kernel_ndim = kernel_shape.ndim();

    if (kernel_ndim != 4) {
        throw TRException("[CPU Conv] Kernel must be 4-dimensional (N, C, H, W)");
    }

    int32_t out_channels = kernel_shape.n();
    int32_t input_h = input_shape.h();
    int32_t input_w = input_shape.w();
    int32_t kernel_h = kernel_shape.h();
    int32_t kernel_w = kernel_shape.w();

    int32_t output_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    int32_t output_w = (input_w + 2 * padding - kernel_w) / stride + 1;

    if (output_h <= 0 || output_w <= 0) {
        throw TRException("[CPU Conv] Invalid output shape due to parameters");
    }

    if (input_ndim == 2) {
        return Shape(1, out_channels, output_h, output_w);
    } else if (input_ndim == 3) {
        return Shape(1, out_channels, output_h, output_w);
    } else if (input_ndim == 4) {
        int32_t batch_size = input_shape.n();
        return Shape(batch_size, out_channels, output_h, output_w);
    } else {
        throw TRException("[CPU Conv] Input must have 2-4 dimensions");
    }
}

/**
 * @brief 计算转置卷积的输出形状
 */
static Shape calculate_transposed_conv_shape(const Shape& input_shape, const Shape& kernel_shape,
                                           int32_t stride, int32_t padding) {
    int32_t input_ndim = input_shape.ndim();
    int32_t kernel_ndim = kernel_shape.ndim();

    if (kernel_ndim != 4) {
        throw TRException("[CPU TransposedConv] Kernel must be 4-dimensional (N, C, H, W)");
    }

    int32_t out_channels = kernel_shape.n();
    int32_t input_h = input_shape.h();
    int32_t input_w = input_shape.w();
    int32_t kernel_h = kernel_shape.h();
    int32_t kernel_w = kernel_shape.w();

    int32_t output_h = (input_h - 1) * stride + kernel_h - 2 * padding;
    int32_t output_w = (input_w - 1) * stride + kernel_w - 2 * padding;

    if (output_h <= 0 || output_w <= 0) {
        throw TRException("[CPU TransposedConv] Invalid output shape due to parameters");
    }

    if (input_ndim == 2) {
        return Shape(1, out_channels, output_h, output_w);
    } else if (input_ndim == 3) {
        return Shape(1, out_channels, output_h, output_w);
    } else if (input_ndim == 4) {
        int32_t batch_size = input_shape.n();
        return Shape(batch_size, out_channels, output_h, output_w);
    } else {
        throw TRException("[CPU TransposedConv] Input must have 2-4 dimensions");
    }
}

// ===== 标准卷积朴素实现（保持不变作为备用） =====

static void conv_operation_core_naive(const Tensor& input, const Tensor& kernel,
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

    int32_t batch_size = (input_ndim == 4) ? input_shape.n() : 1;
    int32_t in_channels = (input_ndim >= 3) ? input_shape.c() : 1;
    int32_t out_channels = kernel_shape.n();

    std::memset(result_data, 0, result_shape.numel() * sizeof(float));

    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t oc = 0; oc < out_channels; ++oc) {
            for (int32_t oh = 0; oh < output_h; ++oh) {
                for (int32_t ow = 0; ow < output_w; ++ow) {
                    int32_t ih_start = oh * stride - padding;
                    int32_t iw_start = ow * stride - padding;

                    float sum_val = 0.0f;
                    for (int32_t ic = 0; ic < in_channels; ++ic) {
                        for (int32_t kh = 0; kh < kernel_h; ++kh) {
                            for (int32_t kw = 0; kw < kernel_w; ++kw) {
                                int32_t ih = ih_start + kh;
                                int32_t iw = iw_start + kw;

                                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                                    int32_t input_idx = 0;
                                    if (input_ndim == 4) {
                                        input_idx = b * in_channels * input_h * input_w +
                                                  ic * input_h * input_w +
                                                  ih * input_w + iw;
                                    } else if (input_ndim == 3) {
                                        input_idx = ic * input_h * input_w + ih * input_w + iw;
                                    } else {
                                        input_idx = ih * input_w + iw;
                                    }

                                    int32_t kernel_idx = oc * (in_channels * kernel_h * kernel_w) +
                                                      ic * (kernel_h * kernel_w) +
                                                      kh * kernel_w + kw;

                                    sum_val += input_data[input_idx] * kernel_data[kernel_idx];
                                }
                            }
                        }
                    }

                    int32_t result_idx = 0;
                    if (input_ndim == 4) {
                        result_idx = b * out_channels * output_h * output_w +
                                  oc * output_h * output_w +
                                  oh * output_w + ow;
                    } else if (input_ndim == 3) {
                        result_idx = oc * output_h * output_w + oh * output_w + ow;
                    } else {
                        result_idx = oh * output_w + ow;
                    }

                    result_data[result_idx] = sum_val;
                }
            }
        }
    }
}

// ===== 优化后的标准卷积Eigen实现（方案D） =====

#ifdef TR_USE_EIGEN

/**
 * @brief 高性能im2col实现 - 优化内存访问模式
 * @param input_data 输入数据指针
 * @param col_data im2col矩阵数据指针
 * @param input_ndim 输入维度数
 * @param batch_idx 当前batch索引
 * @param in_channels 输入通道数
 * @param input_h 输入高度
 * @param input_w 输入宽度
 * @param kernel_h 卷积核高度
 * @param kernel_w 卷积核宽度
 * @param output_h 输出高度
 * @param output_w 输出宽度
 * @param stride 步长
 * @param padding 填充
 */
static inline void fast_im2col(const float* input_data, float* col_data,
                               int32_t input_ndim, int32_t batch_idx,
                               int32_t in_channels, int32_t input_h, int32_t input_w,
                               int32_t kernel_h, int32_t kernel_w,
                               int32_t output_h, int32_t output_w,
                               int32_t stride, int32_t padding) {
    const int32_t col_rows = in_channels * kernel_h * kernel_w;
    const int32_t col_cols = output_h * output_w;

    // 计算输入基地址
    int32_t input_base_offset = 0;
    if (input_ndim == 4) {
        input_base_offset = batch_idx * in_channels * input_h * input_w;
    }

    // 优化：按输出位置遍历（更好的缓存局部性）
    // 使用手动循环展开替代collapse指令，避免MSVC警告
    if (output_h * output_w > 1024) {
        #pragma omp parallel for
        for (int32_t oh = 0; oh < output_h; ++oh) {
            for (int32_t ow = 0; ow < output_w; ++ow) {
                const int32_t col_idx = oh * output_w + ow;
                const int32_t ih_start = oh * stride - padding;
                const int32_t iw_start = ow * stride - padding;

                float* col_ptr = col_data + col_idx;

                for (int32_t ic = 0; ic < in_channels; ++ic) {
                    int32_t channel_offset = 0;
                    if (input_ndim == 4) {
                        channel_offset = input_base_offset + ic * input_h * input_w;
                    } else if (input_ndim == 3) {
                        channel_offset = ic * input_h * input_w;
                    }

                    // 内层循环展开，减少边界检查
                    for (int32_t kh = 0; kh < kernel_h; ++kh) {
                        const int32_t ih = ih_start + kh;
                        const bool h_valid = (ih >= 0 && ih < input_h);

                        for (int32_t kw = 0; kw < kernel_w; ++kw) {
                            const int32_t iw = iw_start + kw;
                            float val = 0.0f;

                            if (h_valid && iw >= 0 && iw < input_w) {
                                val = input_data[channel_offset + ih * input_w + iw];
                            }

                            const int32_t row = ic * kernel_h * kernel_w + kh * kernel_w + kw;
                            col_ptr[row * col_cols] = val;
                        }
                    }
                }
            }
        }
    } else {
        // 小规模数据不使用并行化
        for (int32_t oh = 0; oh < output_h; ++oh) {
            for (int32_t ow = 0; ow < output_w; ++ow) {
                const int32_t col_idx = oh * output_w + ow;
                const int32_t ih_start = oh * stride - padding;
                const int32_t iw_start = ow * stride - padding;

                float* col_ptr = col_data + col_idx;

                for (int32_t ic = 0; ic < in_channels; ++ic) {
                    int32_t channel_offset = 0;
                    if (input_ndim == 4) {
                        channel_offset = input_base_offset + ic * input_h * input_w;
                    } else if (input_ndim == 3) {
                        channel_offset = ic * input_h * input_w;
                    }

                    // 内层循环展开，减少边界检查
                    for (int32_t kh = 0; kh < kernel_h; ++kh) {
                        const int32_t ih = ih_start + kh;
                        const bool h_valid = (ih >= 0 && ih < input_h);

                        for (int32_t kw = 0; kw < kernel_w; ++kw) {
                            const int32_t iw = iw_start + kw;
                            float val = 0.0f;

                            if (h_valid && iw >= 0 && iw < input_w) {
                                val = input_data[channel_offset + ih * input_w + iw];
                            }

                            const int32_t row = ic * kernel_h * kernel_w + kh * kernel_w + kw;
                            col_ptr[row * col_cols] = val;
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief 执行标准卷积的核心实现（优化的Eigen版本 - 方案D）
 */
static void conv_operation_core_eigen(const Tensor& input, const Tensor& kernel,
                                     Tensor& result, int32_t stride, int32_t padding) {
    Logger::get_instance().debug("Using optimized Eigen im2col+GEMM for CPU convolution (Plan D)");

    // 设置Eigen线程数
    const int max_threads = omp_get_max_threads();
    Eigen::setNbThreads(max_threads);

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
    out_channels = kernel_shape.n();

    const int32_t col_rows = in_channels * kernel_h * kernel_w;
    const int32_t col_cols = output_h * output_w;

    // 优化：预先构建权重矩阵（列主序，优化GEMM性能）
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> W(out_channels, col_rows);

    // 使用手动循环展开替代collapse指令，避免MSVC警告
    if (out_channels * col_rows > 4096) {
        #pragma omp parallel for
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        const int col = ic * kernel_h * kernel_w + kh * kernel_w + kw;
                        const int kernel_idx = oc * (in_channels * kernel_h * kernel_w) +
                                             ic * (kernel_h * kernel_w) +
                                             kh * kernel_w + kw;
                        W(oc, col) = kernel_data[kernel_idx];
                    }
                }
            }
        }
    } else {
        // 小规模数据不使用并行化
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        const int col = ic * kernel_h * kernel_w + kh * kernel_w + kw;
                        const int kernel_idx = oc * (in_channels * kernel_h * kernel_w) +
                                             ic * (kernel_h * kernel_w) +
                                             kh * kernel_w + kw;
                        W(oc, col) = kernel_data[kernel_idx];
                    }
                }
            }
        }
    }

    // 批次并行处理
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        // 每个线程独立的im2col矩阵
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> col(col_rows, col_cols);

        // 使用优化的im2col
        fast_im2col(input_data, col.data(), input_ndim, b,
                   in_channels, input_h, input_w,
                   kernel_h, kernel_w, output_h, output_w,
                   stride, padding);

        // 高性能GEMM: output = W * col
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> output_mat = W * col;

        // 优化的结果写回
        const int32_t result_base = (input_ndim == 4) ?
            (b * out_channels * output_h * output_w) : 0;

        for (int32_t oc = 0; oc < out_channels; ++oc) {
            const int32_t channel_offset = (input_ndim == 4 || input_ndim == 3) ?
                (result_base + oc * output_h * output_w) : result_base;

            // 连续内存复制
            std::memcpy(&result_data[channel_offset],
                       output_mat.data() + oc * col_cols,
                       col_cols * sizeof(float));
        }
    }
}

#endif // TR_USE_EIGEN

// ===== 转置卷积朴素实现（保持不变作为备用） =====

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

    int32_t batch_size = (input_ndim == 4) ? input_shape.n() : 1;
    int32_t in_channels = (input_ndim >= 3) ? input_shape.c() : 1;
    int32_t out_channels = kernel_shape.n();

    std::memset(result_data, 0, result_shape.numel() * sizeof(float));

    for (int32_t b = 0; b < batch_size; ++b) {
        for (int32_t ic = 0; ic < in_channels; ++ic) {
            for (int32_t oc = 0; oc < out_channels; ++oc) {
                for (int32_t ih = 0; ih < input_h; ++ih) {
                    for (int32_t iw = 0; iw < input_w; ++iw) {
                        int32_t oh_start = ih * stride - padding;
                        int32_t ow_start = iw * stride - padding;

                        float input_val = 0.0f;
                        int32_t input_idx = 0;
                        if (input_ndim == 4) {
                            input_idx = b * in_channels * input_h * input_w +
                                      ic * input_h * input_w +
                                      ih * input_w + iw;
                        } else if (input_ndim == 3) {
                            input_idx = ic * input_h * input_w + ih * input_w + iw;
                        } else {
                            input_idx = ih * input_w + iw;
                        }
                        input_val = input_data[input_idx];

                        for (int32_t kh = 0; kh < kernel_h; ++kh) {
                            for (int32_t kw = 0; kw < kernel_w; ++kw) {
                                int32_t oh = oh_start + (kernel_h - 1 - kh);
                                int32_t ow = ow_start + (kernel_w - 1 - kw);

                                if (oh >= 0 && oh < output_h && ow >= 0 && ow < output_w) {
                                    int32_t kernel_idx = oc * (in_channels * kernel_h * kernel_w) +
                                                      ic * (kernel_h * kernel_w) +
                                                      (kernel_h - 1 - kh) * kernel_w + (kernel_w - 1 - kw);

                                    int32_t result_idx = 0;
                                    if (input_ndim == 4) {
                                        result_idx = b * out_channels * output_h * output_w +
                                                  oc * output_h * output_w +
                                                  oh * output_w + ow;
                                    } else if (input_ndim == 3) {
                                        result_idx = oc * output_h * output_w + oh * output_w + ow;
                                    } else {
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

// ===== 优化后的转置卷积Eigen实现（方案D核心创新） =====

#ifdef TR_USE_EIGEN

/**
 * @brief 高性能col2im实现 - 转置卷积的关键操作（方案D核心创新）
 * @param col_data 列矩阵数据指针
 * @param output_data 输出数据指针
 * @param output_ndim 输出维度数
 * @param batch_idx 当前batch索引
 * @param out_channels 输出通道数
 * @param output_h 输出高度
 * @param output_w 输出宽度
 * @param kernel_h 卷积核高度
 * @param kernel_w 卷积核宽度
 * @param input_h 输入高度（转置卷积的输入）
 * @param input_w 输入宽度
 * @param stride 步长
 * @param padding 填充
 */
static inline void fast_col2im(const float* col_data, float* output_data,
                               int32_t output_ndim, int32_t batch_idx,
                               int32_t out_channels, int32_t output_h, int32_t output_w,
                               int32_t kernel_h, int32_t kernel_w,
                               int32_t input_h, int32_t input_w,
                               int32_t stride, int32_t padding) {
    const int32_t col_rows = out_channels * kernel_h * kernel_w;
    const int32_t col_cols = input_h * input_w;

    // 计算输出基地址
    int32_t output_base_offset = 0;
    if (output_ndim == 4) {
        output_base_offset = batch_idx * out_channels * output_h * output_w;
    }

    // 先清零输出区域（原子操作保证线程安全）
    const int32_t output_size = (output_ndim == 4) ?
        (out_channels * output_h * output_w) :
        ((output_ndim == 3) ? (out_channels * output_h * output_w) : (output_h * output_w));

    std::memset(output_data + output_base_offset, 0, output_size * sizeof(float));

    // 按输入位置遍历
    for (int32_t ih = 0; ih < input_h; ++ih) {
        for (int32_t iw = 0; iw < input_w; ++iw) {
            const int32_t col_idx = ih * input_w + iw;
            const int32_t oh_start = ih * stride - padding;
            const int32_t ow_start = iw * stride - padding;

            const float* col_ptr = col_data + col_idx;

            for (int32_t oc = 0; oc < out_channels; ++oc) {
                int32_t channel_offset = 0;
                if (output_ndim == 4) {
                    channel_offset = output_base_offset + oc * output_h * output_w;
                } else if (output_ndim == 3) {
                    channel_offset = oc * output_h * output_w;
                }

                // 展开卷积核循环
                for (int32_t kh = 0; kh < kernel_h; ++kh) {
                    const int32_t oh = oh_start + kh;
                    if (oh < 0 || oh >= output_h) continue;

                    for (int32_t kw = 0; kw < kernel_w; ++kw) {
                        const int32_t ow = ow_start + kw;
                        if (ow < 0 || ow >= output_w) continue;

                        const int32_t row = oc * kernel_h * kernel_w + kh * kernel_w + kw;
                        const float val = col_ptr[row * col_cols];

                        output_data[channel_offset + oh * output_w + ow] += val;
                    }
                }
            }
        }
    }
}

/**
 * @brief 执行转置卷积的核心实现（优化的Eigen col2im + GEMM版本 - 方案D）
 */
static void transposed_conv_operation_core_eigen(const Tensor& input, const Tensor& kernel,
                                                 Tensor& result, int32_t stride, int32_t padding) {
    Logger::get_instance().debug("Using optimized Eigen col2im+GEMM for CPU transposed convolution (Plan D)");

    // 设置Eigen线程数
    const int max_threads = omp_get_max_threads();
    Eigen::setNbThreads(max_threads);

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
    out_channels = kernel_shape.n();

    // 转置卷积: col_rows = out_channels * kernel_h * kernel_w
    //          col_cols = input_h * input_w
    const int32_t col_rows = out_channels * kernel_h * kernel_w;
    const int32_t col_cols = input_h * input_w;

    // 构建权重矩阵 W^T [col_rows x in_channels]
    // 注意：转置卷积需要转置权重并旋转180度
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> WT(col_rows, in_channels);

    // 使用手动循环展开替代collapse指令，避免MSVC警告
    if (col_rows * in_channels > 4096) {
        #pragma omp parallel for
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        // 旋转180度：读取位置为 (kernel_h-1-kh, kernel_w-1-kw)
                        const int kernel_idx = oc * (in_channels * kernel_h * kernel_w) +
                                             ic * (kernel_h * kernel_w) +
                                             (kernel_h - 1 - kh) * kernel_w + (kernel_w - 1 - kw);

                        const int row = oc * kernel_h * kernel_w + kh * kernel_w + kw;
                        WT(row, ic) = kernel_data[kernel_idx];
                    }
                }
            }
        }
    } else {
        // 小规模数据不使用并行化
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        // 旋转180度：读取位置为 (kernel_h-1-kh, kernel_w-1-kw)
                        const int kernel_idx = oc * (in_channels * kernel_h * kernel_w) +
                                             ic * (kernel_h * kernel_w) +
                                             (kernel_h - 1 - kh) * kernel_w + (kernel_w - 1 - kw);

                        const int row = oc * kernel_h * kernel_w + kh * kernel_w + kw;
                        WT(row, ic) = kernel_data[kernel_idx];
                    }
                }
            }
        }
    }

    // 批次并行处理
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        // 构建输入矩阵 [in_channels x (input_h * input_w)]
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
            input_mat(in_channels, col_cols);

        // 填充输入矩阵
        for (int32_t ic = 0; ic < in_channels; ++ic) {
            int32_t input_base = 0;
            if (input_ndim == 4) {
                input_base = b * in_channels * input_h * input_w + ic * input_h * input_w;
            } else if (input_ndim == 3) {
                input_base = ic * input_h * input_w;
            }

            std::memcpy(input_mat.data() + ic * col_cols,
                       input_data + input_base,
                       col_cols * sizeof(float));
        }

        // GEMM: col = WT * input_mat
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> col = WT * input_mat;

        // col2im：将col矩阵转换回输出张量
        fast_col2im(col.data(), result_data, input_ndim, b,
                   out_channels, output_h, output_w,
                   kernel_h, kernel_w, input_h, input_w,
                   stride, padding);
    }
}

#endif // TR_USE_EIGEN

// ===== 公共API实现 =====

Tensor CpuBackend::conv(const Tensor& input, const Tensor& kernel, int32_t stride, int32_t padding) {
    validate_conv_parameters(input, kernel, stride, padding, "Conv");
    Shape output_shape = calculate_conv_shape(input.shape(), kernel.shape(), stride, padding);
    Tensor result = this->empty(output_shape, DType::FP32);

#ifdef TR_USE_EIGEN
    conv_operation_core_eigen(input, kernel, result, stride, padding);
#else
    conv_operation_core_naive(input, kernel, result, stride, padding);
#endif

    return result;
}

void CpuBackend::conv_into(const Tensor& input, const Tensor& kernel, Tensor& result,
                          int32_t stride, int32_t padding) {
    validate_conv_parameters(input, kernel, stride, padding, "Conv");

    if (result.device() != tr::CPU) {
        throw TRException("[CPU Conv] Result tensor must be on CPU device");
    }
    if (!result.storage_allocated()) {
        throw TRException("[CPU Conv] Result tensor storage not allocated");
    }
    if (result.dtype() != DType::FP32) {
        throw TRException("[CPU Conv] Result tensor must be FP32 type");
    }

    Shape expected_shape = calculate_conv_shape(input.shape(), kernel.shape(), stride, padding);
    if (result.shape() != expected_shape) {
        throw TRException("[CPU Conv] Result shape mismatch. expected: " + expected_shape.to_string() +
                        ", actual: " + result.shape().to_string());
    }

#ifdef TR_USE_EIGEN
    conv_operation_core_eigen(input, kernel, result, stride, padding);
#else
    conv_operation_core_naive(input, kernel, result, stride, padding);
#endif
}

Tensor CpuBackend::transposed_conv(const Tensor& input, const Tensor& kernel, int32_t stride, int32_t padding) {
    validate_conv_parameters(input, kernel, stride, padding, "TransposedConv");
    Shape output_shape = calculate_transposed_conv_shape(input.shape(), kernel.shape(), stride, padding);
    Tensor result = this->empty(output_shape, DType::FP32);

#ifdef TR_USE_EIGEN
    transposed_conv_operation_core_eigen(input, kernel, result, stride, padding);
#else
    transposed_conv_operation_core_naive(input, kernel, result, stride, padding);
#endif

    return result;
}

void CpuBackend::transposed_conv_into(const Tensor& input, const Tensor& kernel, Tensor& result,
                                     int32_t stride, int32_t padding) {
    validate_conv_parameters(input, kernel, stride, padding, "TransposedConv");

    if (result.device() != tr::CPU) {
        throw TRException("[CPU TransposedConv] Result tensor must be on CPU device");
    }
    if (!result.storage_allocated()) {
        throw TRException("[CPU TransposedConv] Result tensor storage not allocated");
    }
    if (result.dtype() != DType::FP32) {
        throw TRException("[CPU TransposedConv] Result tensor must be FP32 type");
    }

    Shape expected_shape = calculate_transposed_conv_shape(input.shape(), kernel.shape(), stride, padding);
    if (result.shape() != expected_shape) {
        throw TRException("[CPU TransposedConv] Result shape mismatch. expected: " + expected_shape.to_string() +
                        ", actual: " + result.shape().to_string());
    }

#ifdef TR_USE_EIGEN
    transposed_conv_operation_core_eigen(input, kernel, result, stride, padding);
#else
    transposed_conv_operation_core_naive(input, kernel, result, stride, padding);
#endif
}

} // namespace tr
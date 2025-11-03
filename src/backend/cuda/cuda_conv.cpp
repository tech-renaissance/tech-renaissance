/**
 * @file cuda_conv.cpp
 * @brief CUDA后端卷积操作实现（已优化）
 * @details 基于cuDNN实现GPU加速卷积操作，实现描述符和算法缓存
 * @version 1.01.00
 * @date 2025-11-04
 * @author 技术觉醒团队
 * @note 依赖项: cuda_backend.h, tensor.h, cuDNN
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/cuda/cuda_backend.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/utils/logger.h"
#include <string>

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>
#include <cudnn.h>

// CUDA错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::string msg = "CUDA Error at " + std::string(__FILE__) + ":" + \
                         std::to_string(__LINE__) + ": " + \
                         cudaGetErrorString(err); \
        throw TRException(msg); \
    } \
} while (0)

// cuDNN错误检查宏
#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::string msg = "cuDNN Error at " + std::string(__FILE__) + ":" + \
                         std::to_string(__LINE__) + ": " + \
                         cudnnGetErrorString(status); \
        throw TRException(msg); \
    } \
} while (0)

namespace tr {

// ===== 辅助函数实现 =====

/**
 * @brief 验证卷积张量的基本参数
 */
void CudaBackend::validate_conv_tensors(const Tensor& input, const Tensor& kernel) const {
    // 验证设备
    if (input.device() != device() || kernel.device() != device()) {
        throw TRException("[CUDA Conv] Tensors must be on CUDA device");
    }

    // 验证存储已分配
    if (input.is_empty() || kernel.is_empty()) {
        throw TRException("[CUDA Conv] Tensor storage not allocated");
    }

    // 验证数据类型
    if (input.dtype() != DType::FP32 || kernel.dtype() != DType::FP32) {
        throw TRException("[CUDA Conv] Only supports FP32 tensors");
    }

    // 验证输入维度
    if (input.ndim() < 2) {
        throw TRException("[CUDA Conv] Input tensor must have at least 2 dimensions");
    }

    // 验证kernel维度
    if (kernel.ndim() != 4) {
        throw TRException("[CUDA Conv] Kernel tensor must be 4-dimensional (N, C, H, W)");
    }

    // 验证kernel是正方形
    if (kernel.shape().h() != kernel.shape().w()) {
        throw TRException("[CUDA Conv] Only supports square kernels (H=W)");
    }

    // 验证输入通道数匹配
    if (input.shape().c() != kernel.shape().c()) {
        throw TRException("[CUDA Conv] Input channels must match kernel input channels");
    }
}

/**
 * @brief 计算标准卷积的输出形状
 */
Shape CudaBackend::calculate_conv_output_shape(
    const Shape& input_shape, const Shape& kernel_shape, int32_t stride, int32_t padding) const {

    int32_t input_h = input_shape.h();
    int32_t input_w = input_shape.w();
    int32_t kernel_h = kernel_shape.h();
    int32_t kernel_w = kernel_shape.w();

    int32_t output_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    int32_t output_w = (input_w + 2 * padding - kernel_w) / stride + 1;

    if (input_shape.ndim() == 4) {
        return Shape(input_shape.n(), kernel_shape.n(), output_h, output_w);
    } else if (input_shape.ndim() == 3) {
        return Shape(kernel_shape.n(), output_h, output_w);
    } else {
        return Shape(output_h, output_w);
    }
}

/**
 * @brief 计算转置卷积的输出形状
 */
Shape CudaBackend::calculate_transposed_conv_output_shape(
    const Shape& input_shape, const Shape& kernel_shape, int32_t stride, int32_t padding) const {

    int32_t input_h = input_shape.h();
    int32_t input_w = input_shape.w();
    int32_t kernel_h = kernel_shape.h();
    int32_t kernel_w = kernel_shape.w();

    int32_t output_h = (input_h - 1) * stride + kernel_h - 2 * padding;
    int32_t output_w = (input_w - 1) * stride + kernel_w - 2 * padding;

    if (input_shape.ndim() == 4) {
        return Shape(input_shape.n(), kernel_shape.n(), output_h, output_w);
    } else if (input_shape.ndim() == 3) {
        return Shape(kernel_shape.n(), output_h, output_w);
    } else {
        return Shape(output_h, output_w);
    }
}

// ===== 标准卷积实现 =====

Tensor CudaBackend::conv(const Tensor& input, const Tensor& kernel, int32_t stride, int32_t padding) {
    validate_conv_tensors(input, kernel);

    // 验证输入张量的设备
    if (input.device() != device() || kernel.device() != device()) {
        throw TRException("[CUDA Conv] Input tensors must be on CUDA device");
    }

    // 检查stride和padding参数
    if (stride <= 0) {
        throw TRException("[CUDA Conv] Stride must be positive");
    }
    if (padding < 0) {
        throw TRException("[CUDA Conv] Padding must be non-negative");
    }

    // 计算输出形状
    Shape output_shape = calculate_conv_output_shape(input.shape(), kernel.shape(), stride, padding);
    auto result = this->empty(output_shape, DType::FP32);

    // 执行卷积
    conv_into(input, kernel, result, stride, padding);

    return result;
}

void CudaBackend::conv_into(const Tensor& input, const Tensor& kernel, Tensor& result,
                           int32_t stride, int32_t padding) {
    validate_conv_tensors(input, kernel);

    // 验证result张量
    if (result.device() != device()) {
        throw TRException("[CUDA Conv] Result tensor must be on CUDA device");
    }
    if (result.is_empty()) {
        throw TRException("[CUDA Conv] Result tensor storage not allocated");
    }
    if (result.dtype() != DType::FP32) {
        throw TRException("[CUDA Conv] Result tensor must be FP32 type");
    }

    // 验证stride和padding参数
    if (stride <= 0) {
        throw TRException("[CUDA Conv] Stride must be positive");
    }
    if (padding < 0) {
        throw TRException("[CUDA Conv] Padding must be non-negative");
    }

    // 验证输出形状
    Shape expected_shape = calculate_conv_output_shape(input.shape(), kernel.shape(), stride, padding);
    if (result.shape() != expected_shape) {
        throw TRException("[CUDA Conv] Result shape mismatch. expected: " + expected_shape.to_string() +
                        ", actual: " + result.shape().to_string());
    }

    // 设置设备
    set_device();

    // 1. [核心优化] 获取缓存的配置（描述符、算法、工作空间大小）
    auto config = get_conv_config(input, kernel, result, stride, padding);

    // 2. 获取缓存的工作空间内存（如果需要）
    std::shared_ptr<void> workspace = nullptr;
    if (config->workspace_size > 0) {
        workspace = get_workspace(config->workspace_size);
    }

    // 3. 设置卷积参数
    float alpha = 1.0f;
    float beta = 0.0f;

    // 4. [核心优化] 执行卷积（使用缓存的描述符）
    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn_handle(),
        &alpha,
        static_cast<cudnnTensorDescriptor_t>(config->input_desc),   // <-- 缓存的
        input.data_ptr(),
        static_cast<cudnnFilterDescriptor_t>(config->filter_desc),  // <-- 缓存的
        kernel.data_ptr(),
        static_cast<cudnnConvolutionDescriptor_t>(config->conv_desc),    // <-- 缓存的
        static_cast<cudnnConvolutionFwdAlgo_t>(config->algo),         // <-- 缓存的
        workspace.get(),      // 工作空间
        config->workspace_size, // <-- 缓存的
        &beta,
        static_cast<cudnnTensorDescriptor_t>(config->output_desc),  // <-- 缓存的
        result.data_ptr()));

    // 5. 无需清理描述符 - 由缓存管理
}

// ===== 转置卷积实现 =====

Tensor CudaBackend::transposed_conv(const Tensor& input, const Tensor& kernel, int32_t stride, int32_t padding) {
    validate_conv_tensors(input, kernel);

    // 检查stride和padding参数
    if (stride <= 0) {
        throw TRException("[CUDA TransposedConv] Stride must be positive");
    }
    if (padding < 0) {
        throw TRException("[CUDA TransposedConv] Padding must be non-negative");
    }

    // 计算输出形状
    Shape output_shape = calculate_transposed_conv_output_shape(input.shape(), kernel.shape(), stride, padding);
    Tensor result = empty(output_shape, DType::FP32);

    // 执行转置卷积
    transposed_conv_into(input, kernel, result, stride, padding);

    return result;
}

void CudaBackend::transposed_conv_into(const Tensor& input, const Tensor& kernel, Tensor& result,
                                       int32_t stride, int32_t padding) {
    validate_conv_tensors(input, kernel);

    // 验证result张量
    if (result.device() != device()) {
        throw TRException("[CUDA TransposedConv] Result tensor must be on CUDA device");
    }
    if (result.is_empty()) {
        throw TRException("[CUDA TransposedConv] Result tensor storage not allocated");
    }
    if (result.dtype() != DType::FP32) {
        throw TRException("[CUDA TransposedConv] Result tensor must be FP32 type");
    }

    // 验证stride和padding参数
    if (stride <= 0) {
        throw TRException("[CUDA TransposedConv] Stride must be positive");
    }
    if (padding < 0) {
        throw TRException("[CUDA TransposedConv] Padding must be non-negative");
    }

    // 验证输出形状
    Shape expected_shape = calculate_transposed_conv_output_shape(input.shape(), kernel.shape(), stride, padding);
    if (result.shape() != expected_shape) {
        throw TRException("[CUDA TransposedConv] Result shape mismatch. expected: " + expected_shape.to_string() +
                        ", actual: " + result.shape().to_string());
    }

    // 设置设备
    set_device();

    // 创建描述符（转置卷积暂时使用原有方式，后续可以优化）
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // RAII包装器确保清理
    struct DescriptorGuard {
        cudnnTensorDescriptor_t &i, &o;
        cudnnFilterDescriptor_t &f;
        cudnnConvolutionDescriptor_t &c;
        ~DescriptorGuard() {
            if (i) cudnnDestroyTensorDescriptor(i);
            if (o) cudnnDestroyTensorDescriptor(o);
            if (f) cudnnDestroyFilterDescriptor(f);
            if (c) cudnnDestroyConvolutionDescriptor(c);
        }
    } guard{input_desc, output_desc, filter_desc, conv_desc};

    // 设置描述符
    int batch_size = (input.ndim() == 4) ? input.shape().n() : 1;
    int in_channels = (input.ndim() >= 3) ? input.shape().c() : 1;
    int input_h = input.shape().h();
    int input_w = input.shape().w();
    int out_channels = kernel.shape().n();
    int kernel_h = kernel.shape().h();
    int kernel_w = kernel.shape().w();

    Shape output_shape = result.shape();
    int output_h = output_shape.h();
    int output_w = output_shape.w();

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch_size, in_channels, input_h, input_w));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                          out_channels, in_channels, kernel_h, kernel_w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch_size, out_channels, output_h, output_w));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, padding, padding,
                                                stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // 启用Tensor Core支持
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));

    // 设置转置卷积参数
    float alpha = 1.0f;
    float beta = 0.0f;

    // 执行转置卷积
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        cudnn_handle(),
        &alpha,
        filter_desc,
        kernel.data_ptr(),
        input_desc,
        input.data_ptr(),
        conv_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        nullptr, 0, // 工作空间（可选）
        &beta,
        output_desc,
        result.data_ptr()));
}

} // namespace tr

#endif // TR_USE_CUDA
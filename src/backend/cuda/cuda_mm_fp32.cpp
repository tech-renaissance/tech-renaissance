/**
 * @file cuda_mm_fp32.cpp
 * @brief CUDA后端FP32矩阵乘法实现
 * @details 实现CUDA后端的FP32矩阵乘法运算，基于cuBLAS优化
 * @version 1.00.00
 * @date 2025-10-31
 * @author 技术觉醒团队
 * @note 依赖项: cuda_backend.h, tensor.h, cuBLAS
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/cuda/cuda_backend.h"
#include "tech_renaissance/backend/cuda/cuda_common.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/utils/logger.h"

#ifdef TR_USE_CUDA

namespace tr {

Tensor CudaBackend::mm(const Tensor& a, const Tensor& b) {
    auto result = this->empty(Shape(a.height(), b.width()), DType::FP32);
    this->mm_into(a, b, result);
    return result;
}

void CudaBackend::mm_into(const Tensor& a, const Tensor& b, Tensor& result) {
    // 检查设备
    validate_same_device(a.device());
    validate_same_device(b.device());
    validate_same_device(result.device());
    set_device();

    // 检查数据类型
    if (a.dtype() != DType::FP32 || b.dtype() != DType::FP32 || result.dtype() != DType::FP32) {
        throw TRException("[CudaBackend::mm] All tensors must be FP32");
    }

    // 检查Storage是否已分配
    if (result.is_empty()) {
        throw TRException("[CudaBackend::mm] Result tensor has no allocated Storage");
    }
    if (a.is_empty()) {
        throw TRException("[CudaBackend::mm] Input tensor 'a' has no allocated Storage");
    }
    if (b.is_empty()) {
        throw TRException("[CudaBackend::mm] Input tensor 'b' has no allocated Storage");
    }

    // 从Tensor形状获取M, K, N
    // 支持二维和四维Shape
    // 对于二维Shape(M,K)存储为(0,0,M,K)，使用height()和width()获取
    // 对于四维Shape(M,K,1,1)，使用batch()和channel()获取
    int M, K, N;
    if (a.ndim() == 2) {
        // 二维Shape: M = height, K = width
        M = a.height();
        K = a.width();
    } else {
        // 四维Shape: M = batch, K = channel
        M = a.batch();
        K = a.channel();
    }

    if (b.ndim() == 2) {
        // 二维Shape: K = height, N = width
        K = b.height();
        N = b.width();
    } else {
        // 四维Shape: 对于filter格式，通常是(N,K,1,1)
        N = b.batch();
        // K已经在上面设置
    }

    // 对于行主序矩阵乘法，使用交换矩阵顺序的cuBLAS调用：
    // C(M,N) = A(M,K) × B(K,N) (row-major)
    // cuBLAS计算: C(N,M) = B(N,K) × A(K,M) (适配行主序数据)
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置
        N, M, K,                   // 结果维度
        &alpha,
        static_cast<const float*>(b.data_ptr()), N,  // B矩阵，leading dimension = N
        static_cast<const float*>(a.data_ptr()), K,  // A矩阵，leading dimension = K
        &beta,
        static_cast<float*>(result.data_ptr()), N    // 结果矩阵，leading dimension = N
    ));
}

// 算法查找与缓存的辅助函数
std::pair<int, size_t> CudaBackend::find_best_gemm_algorithm(int M, int K, int N) {
    std::tuple<int, int, int> key = {M, K, N};

    // 加锁以保护缓存
    std::lock_guard<std::mutex> lock(gemm_algo_cache_mutex_);

    // 检查缓存
    if (gemm_algo_cache_.count(key)) {
        return {gemm_algo_cache_.at(key), gemm_workspace_size_cache_.at(key)};
    }

    Logger::get_instance().info("Finding best GEMM algorithm for M=" + std::to_string(M) + ", K=" + std::to_string(K) + ", N=" + std::to_string(N));

    // RAII包装器创建临时描述符用于算法查找
    struct CudnnDescriptorDeleter {
        void operator()(cudnnTensorDescriptor_t desc) const {
            if (desc) cudnnDestroyTensorDescriptor(desc);
        }
        void operator()(cudnnFilterDescriptor_t desc) const {
            if (desc) cudnnDestroyFilterDescriptor(desc);
        }
        void operator()(cudnnConvolutionDescriptor_t desc) const {
            if (desc) cudnnDestroyConvolutionDescriptor(desc);
        }
    };

    using TensorDescriptor = std::unique_ptr<std::remove_pointer_t<cudnnTensorDescriptor_t>, CudnnDescriptorDeleter>;
    using FilterDescriptor = std::unique_ptr<std::remove_pointer_t<cudnnFilterDescriptor_t>, CudnnDescriptorDeleter>;
    using ConvDescriptor = std::unique_ptr<std::remove_pointer_t<cudnnConvolutionDescriptor_t>, CudnnDescriptorDeleter>;

    TensorDescriptor input_desc{[]() {
        cudnnTensorDescriptor_t desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
        return desc;
    }()};

    TensorDescriptor output_desc{[]() {
        cudnnTensorDescriptor_t desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
        return desc;
    }()};

    FilterDescriptor filter_desc{[]() {
        cudnnFilterDescriptor_t desc;
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc));
        return desc;
    }()};

    ConvDescriptor conv_desc{[]() {
        cudnnConvolutionDescriptor_t desc;
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc));
        return desc;
    }()};

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc.get(), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, M, K, 1, 1));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc.get(), CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N, K, 1, 1));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc.get(), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, M, N, 1, 1));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc.get(), 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgoPerf_t perf_result;
    int returned_algo_count = 0;

    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
        cudnn_handle_,
        input_desc.get(),
        filter_desc.get(),
        conv_desc.get(),
        output_desc.get(),
        1, &returned_algo_count, &perf_result));

    if (returned_algo_count == 0) {
        throw TRException("Failed to find any convolution algorithm for GEMM");
    }

    cudnnConvolutionFwdAlgo_t best_algo = perf_result.algo;
    size_t workspace_size = perf_result.memory;

    // 存入缓存
    gemm_algo_cache_[key] = static_cast<int>(best_algo);
    gemm_workspace_size_cache_[key] = workspace_size;

    Logger::get_instance().info("Best algorithm found: " + std::to_string(static_cast<int>(best_algo)) + ", workspace: " + std::to_string(workspace_size) + " bytes");

    return std::make_pair(static_cast<int>(best_algo), workspace_size);
}

} // namespace tr

#endif // TR_USE_CUDA
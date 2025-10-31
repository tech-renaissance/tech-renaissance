/**
 * @file cuda_backend.cu
 * @brief CUDA后端类实现
 * @details 基于cuDNN/cuBLAS实现GPU加速计算
 * @version 1.00.00
 * @date 2025-10-26
 * @author 技术觉醒团队
 * @note 依赖项: tensor.h, CUDA, cuDNN, cuBLAS
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/cuda/cuda_backend.h"
#include "tech_renaissance/backend/cuda/cuda_common.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/utils/logger.h"
#include <sstream>
#include <vector>

#ifdef TR_USE_CUDA

namespace tr {

CudaBackend::CudaBackend(int device_id)
    : device_id_(device_id), stream_(nullptr),
      cublas_handle_(nullptr), cudnn_handle_(nullptr) {

    if (device_id < 0 || device_id >= 8) {
        throw TRException("Invalid CUDA device ID: " + std::to_string(device_id));
    }

    init_cuda_context();
    Logger::get_instance().info("CUDA backend initialized for device " + std::to_string(device_id));
}

CudaBackend::~CudaBackend() {
    cleanup_cuda_context();
    std::cout << std::endl;
    Logger::get_instance().info("CUDA backend destroyed for device " + std::to_string(device_id_));
}

void CudaBackend::init_cuda_context() {
    try {
        set_device();

        // 创建流
        CUDA_CHECK(cudaStreamCreate(&stream_));

        // 创建cuBLAS句柄
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));

        // 创建cuDNN句柄
        CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
        CUDNN_CHECK(cudnnSetStream(cudnn_handle_, stream_));

    } catch (const std::exception& e) {
        cleanup_cuda_context();
        throw TRException("Failed to initialize CUDA context: " +
                         std::string(e.what()));
    }
}

void CudaBackend::cleanup_cuda_context() {
    if (cudnn_handle_) {
        cudnnDestroy(cudnn_handle_);
        cudnn_handle_ = nullptr;
    }
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void CudaBackend::set_device() const {
    CUDA_CHECK(cudaSetDevice(device_id_));
}

std::shared_ptr<void> CudaBackend::allocate(size_t size) {
    set_device();

    if (size == 0) {
        throw TRException("Cannot allocate zero bytes");
    }

    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));

    return std::shared_ptr<void>(ptr, [this](void* p) {
        if (p) {
            cudaSetDevice(device_id_);
            cudaFree(p);
        }
    });
}

void CudaBackend::deallocate(void* ptr) {
    if (ptr) {
        set_device();
        CUDA_CHECK(cudaFree(ptr));
    }
}

void* CudaBackend::get_data_ptr(const std::shared_ptr<void>& holder) {
    return holder.get();
}

void CudaBackend::copy(void* dst, const void* src, size_t size,
                      const Device& dst_device, const Device& src_device) const {
    set_device();

    cudaMemcpyKind kind;

    if (src_device.is_cpu() && dst_device.is_cuda()) {
        kind = cudaMemcpyHostToDevice;
    } else if (src_device.is_cuda() && dst_device.is_cpu()) {
        kind = cudaMemcpyDeviceToHost;
    } else if (src_device.is_cuda() && dst_device.is_cuda()) {
        kind = cudaMemcpyDeviceToDevice;
    } else {
        throw TRException("Unsupported copy direction");
    }

    CUDA_CHECK(cudaMemcpy(dst, src, size, kind));
}

void CudaBackend::fill(Tensor& dst, float value) {
    validate_same_device(dst.device());
    set_device();

    // 新增：检查Storage是否已分配
    if (dst.is_empty()) {
        throw TRException("[CudaBackend::fill] Target tensor has no allocated Storage");
    }

    if (dst.dtype() != DType::FP32) {
        throw TRException("[CudaBackend::fill] fill(float) requires FP32 tensor");
    }

    float* data = static_cast<float*>(dst.data_ptr());

    // 使用cudaMemset（仅适用于0值）
    if (value == 0.0f) {
        CUDA_CHECK(cudaMemset(data, 0, dst.memory_size()));
    } else {
        // 对于4D张量使用cuDNN，对于低维张量回退到CPU拷贝方法
        if (dst.ndim() == 4) {
            // 创建张量描述符
            cudnnTensorDescriptor_t tensor_desc;
            CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensor_desc));

            try {
                // 设置描述符（按NCHW格式）
                int n = dst.batch();
                int c = dst.channel();
                int h = dst.height();
                int w = dst.width();

                CUDNN_CHECK(cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW,
                                                      CUDNN_DATA_FLOAT, n, c, h, w));

                // 使用cudnnSetTensor高效填充
                CUDNN_CHECK(cudnnSetTensor(cudnn_handle_, tensor_desc, data, &value));
            } catch (...) {
                cudnnDestroyTensorDescriptor(tensor_desc);
                throw;
            }

            CUDNN_CHECK(cudnnDestroyTensorDescriptor(tensor_desc));
        } else {
            // 对于低维张量，回退到CPU创建数组+GPU拷贝的方法
            size_t count = dst.numel();
            std::vector<float> temp_data(count, value);
            CUDA_CHECK(cudaMemcpy(data, temp_data.data(),
                                 count * sizeof(float),
                                 cudaMemcpyHostToDevice));
        }
    }
}

void CudaBackend::fill(Tensor& dst, int8_t value) {
    validate_same_device(dst.device());
    set_device();

    // 新增：检查Storage是否已分配
    if (dst.is_empty()) {
        throw TRException("[CudaBackend::fill] Target tensor has no allocated Storage");
    }

    if (dst.dtype() != DType::INT8) {
        throw TRException("[CudaBackend::fill] fill(int8_t) requires INT8 tensor");
    }

    int8_t* data = static_cast<int8_t*>(dst.data_ptr());
    size_t count = dst.numel();

    CUDA_CHECK(cudaMemset(data, static_cast<int>(value), count));
}

void CudaBackend::add(Tensor& result, const Tensor& a, const Tensor& b) {
    validate_same_device(a.device());
    validate_same_device(b.device());
    validate_same_device(result.device());
    validate_tensor_shape(a, b);
    validate_tensor_shape(a, result);
    set_device();

    // 新增：检查Storage是否已分配
    if (result.is_empty()) {
        throw TRException("[CudaBackend::add] Result tensor has no allocated Storage");
    }
    if (a.is_empty()) {
        throw TRException("[CudaBackend::add] Input tensor 'a' has no allocated Storage");
    }
    if (b.is_empty()) {
        throw TRException("[CudaBackend::add] Input tensor 'b' has no allocated Storage");
    }

    if (a.dtype() != DType::FP32) {
        throw TRException("[CudaBackend::add] add only supports FP32 in first phase");
    }

    const float* a_data = static_cast<const float*>(a.data_ptr());
    const float* b_data = static_cast<const float*>(b.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());
    size_t count = a.numel();

    // 使用cuBLAS的axpy: result = a + b
    // 先拷贝a到result，再执行result += b
    CUDA_CHECK(cudaMemcpy(result_data, a_data,
                         count * sizeof(float), cudaMemcpyDeviceToDevice));

    float alpha = 1.0f;
    CUBLAS_CHECK(cublasSaxpy(cublas_handle_, count, &alpha,
                            b_data, 1, result_data, 1));
}

void CudaBackend::mul(Tensor& result, const Tensor& a, const Tensor& b) {
    validate_same_device(a.device());
    validate_same_device(b.device());
    validate_same_device(result.device());
    validate_tensor_shape(a, b);
    validate_tensor_shape(a, result);
    set_device();

    // 新增：检查Storage是否已分配
    if (result.is_empty()) {
        throw TRException("[CudaBackend::mul] Result tensor has no allocated Storage");
    }
    if (a.is_empty()) {
        throw TRException("[CudaBackend::mul] Input tensor 'a' has no allocated Storage");
    }
    if (b.is_empty()) {
        throw TRException("[CudaBackend::mul] Input tensor 'b' has no allocated Storage");
    }

    if (a.dtype() != DType::FP32) {
        throw TRException("[CudaBackend::mul] mul only supports FP32 in first phase");
    }

    // 使用cuDNN的OpTensor实现逐元素乘法
    const float* a_data = static_cast<const float*>(a.data_ptr());
    const float* b_data = static_cast<const float*>(b.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    // 创建张量描述符
    cudnnTensorDescriptor_t a_desc, b_desc, result_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&a_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&result_desc));

    try {
        // 设置张量描述符（假设为4D张量，NCHW格式）
        int n = a.batch();
        int c = a.channel();
        int h = a.height();
        int w = a.width();

        CUDNN_CHECK(cudnnSetTensor4dDescriptor(a_desc, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, n, c, h, w));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, n, c, h, w));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(result_desc, CUDNN_TENSOR_NCHW,
                                             CUDNN_DATA_FLOAT, n, c, h, w));

        // 创建OpTensor描述符
        cudnnOpTensorDescriptor_t op_desc;
        CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&op_desc));
        CUDNN_CHECK(cudnnSetOpTensorDescriptor(op_desc, CUDNN_OP_TENSOR_MUL,
                                              CUDNN_DATA_FLOAT,
                                              CUDNN_PROPAGATE_NAN));

        // 执行逐元素乘法: result = a * b
        const float alpha1 = 1.0f;
        const float alpha2 = 1.0f;
        const float beta = 0.0f;

        CUDNN_CHECK(cudnnOpTensor(cudnn_handle_, op_desc,
                                 &alpha1, a_desc, a_data,
                                 &alpha2, b_desc, b_data,
                                 &beta, result_desc, result_data));

        CUDNN_CHECK(cudnnDestroyOpTensorDescriptor(op_desc));
    } catch (...) {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(a_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(b_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(result_desc));
        throw;
    }

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(a_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(b_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(result_desc));
}

Device CudaBackend::device() const {
    return tr::CUDA[device_id_];
}

void CudaBackend::validate_same_device(const Device& device) const {
    if (!device.is_cuda() || device.index != device_id_) {
        throw TRException("CudaBackend: tensor device mismatch");
    }
}

void CudaBackend::validate_tensor_shape(const Tensor& a, const Tensor& b) const {
    if (a.shape() != b.shape()) {
        throw TRException("Tensor shape mismatch: " +
                         a.shape().to_string() + " vs " +
                         b.shape().to_string());
    }
}

// ===== 数据访问实现 =====

float CudaBackend::get_scalar_float(const Tensor& tensor) {
    if (!tensor.is_scalar()) {
        throw TRException("get_scalar_float: tensor must be scalar");
    }

    validate_same_device(tensor.device());
    set_device();

    const void* device_data = tensor.data_ptr();
    if (!device_data) {
        throw TRException("get_scalar_float: tensor data is null");
    }

    // 拷贝GPU数据到CPU
    float host_value;
    CUDA_CHECK(cudaMemcpy(&host_value, device_data, sizeof(float),
                         cudaMemcpyDeviceToHost));

    return host_value;
}

int32_t CudaBackend::get_scalar_int32(const Tensor& tensor) {
    if (!tensor.is_scalar()) {
        throw TRException("get_scalar_int32: tensor must be scalar");
    }

    validate_same_device(tensor.device());
    set_device();

    const void* device_data = tensor.data_ptr();
    if (!device_data) {
        throw TRException("get_scalar_int32: tensor data is null");
    }

    int32_t host_value;
    switch (tensor.dtype()) {
        case DType::FP32:
            {
                float float_value;
                CUDA_CHECK(cudaMemcpy(&float_value, device_data, sizeof(float),
                                     cudaMemcpyDeviceToHost));
                host_value = static_cast<int32_t>(float_value);
                break;
            }
        case DType::INT8:
            CUDA_CHECK(cudaMemcpy(&host_value, device_data, sizeof(int8_t),
                                 cudaMemcpyDeviceToHost));
            break;
        default:
            throw TRException("get_scalar_int32: unsupported dtype");
    }

    return host_value;
}

int8_t CudaBackend::get_scalar_int8(const Tensor& tensor) {
    if (!tensor.is_scalar()) {
        throw TRException("get_scalar_int8: tensor must be scalar");
    }

    validate_same_device(tensor.device());
    set_device();

    const void* device_data = tensor.data_ptr();
    if (!device_data) {
        throw TRException("get_scalar_int8: tensor data is null");
    }

    int8_t host_value;
    switch (tensor.dtype()) {
        case DType::FP32:
            {
                float float_value;
                CUDA_CHECK(cudaMemcpy(&float_value, device_data, sizeof(float),
                                     cudaMemcpyDeviceToHost));
                host_value = static_cast<int8_t>(float_value);
                break;
            }
        case DType::INT8:
            CUDA_CHECK(cudaMemcpy(&host_value, device_data, sizeof(int8_t),
                                 cudaMemcpyDeviceToHost));
            break;
        default:
            throw TRException("get_scalar_int8: unsupported dtype");
    }

    return host_value;
}

// ===== 矩阵乘法实现 =====
// 注意：CudaBackend::mm 方法的实现已分离到 cuda_mm_fp32.cpp 文件中

// 算法查找与缓存的辅助函数
// 注意：CudaBackend::find_best_gemm_algorithm 方法的实现已分离到 cuda_mm_fp32.cpp 文件中

// 设备转换方法实现
Tensor CudaBackend::to(const Tensor& tensor, const Device& device) const {
    if (tensor.device() == device) {
        return tensor; // 设备相同，直接返回
    }

    if (device.is_cpu()) {
        return to_cpu(tensor);
    } else if (device.is_cuda()) {
        // CUDA到CUDA的转换
        if (device.index != device_id_) {
            throw TRException("CUDA device transfer not yet implemented");
        }
        return tensor; // 同一设备，直接返回
    } else {
        throw TRException("Unsupported target device: " + device.to_string());
    }
}

Tensor CudaBackend::to_cpu(const Tensor& tensor) const {
    if (tensor.device().is_cpu()) {
        return tensor; // 已经在CPU上
    }

    // 创建CPU张量
    Tensor cpu_tensor = Tensor::empty(tensor.shape(), tensor.dtype(), tr::CPU);

    // 直接复制数据，保持行主序布局
    copy(cpu_tensor.data_ptr(), tensor.data_ptr(),
         tensor.memory_size(), tr::CPU, tensor.device());

    return cpu_tensor;
}

Tensor CudaBackend::from_cpu(const Tensor& tensor) const {
    if (tensor.device().is_cuda() &&
        tensor.device().is_cuda() && tensor.device().index == device_id_) {
        return tensor; // 已经在正确的CUDA设备上
    }

    if (!tensor.device().is_cpu()) {
        throw TRException("from_cpu expects input tensor to be on CPU device");
    }

    // 创建CUDA张量
    Device cuda_device = tr::CUDA[device_id_];
    Tensor cuda_tensor = Tensor::empty(tensor.shape(), tensor.dtype(), cuda_device);

    // 直接复制数据，保持原始的行主序布局
    copy(cuda_tensor.data_ptr(), tensor.data_ptr(),
         cuda_tensor.memory_size(), cuda_device, tr::CPU);

    return cuda_tensor;
}

void CudaBackend::synchronize() const {
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace tr

#endif // TR_USE_CUDA
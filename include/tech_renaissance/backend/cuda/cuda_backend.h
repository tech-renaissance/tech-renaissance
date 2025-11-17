/**
 * @file cuda_backend.h
 * @brief CUDA后端实现（已优化，移除CUDA头文件依赖）
 * @details 基于cuDNN/cuBLAS实现GPU加速计算
 * @version 1.01.00
 * @date 2025-10-26
 * @author 技术觉醒团队
 * @note 依赖项: backend.h
 * @note 所属系列: backend
 */

#pragma once

#include "tech_renaissance/backend/backend.h"
#include "tech_renaissance/data/strides.h"
#include <map>
#include <tuple>
#include <mutex>

#ifdef TR_USE_CUDA
// ########## 关键修改 ##########
// 不包含 <cuda_runtime.h>, <cudnn.h>, <cublas_v2.h>
// 仅前向声明（定义）CUDA句柄的不透明指针类型
// 这使得该头文件可以被非.cu文件安全地包含
struct cublasContext;
struct cudnnContext;
struct CUstream_st;
struct cudnnTensorStruct;
struct cudnnFilterStruct;
struct cudnnConvolutionStruct;

typedef struct cublasContext *cublasHandle_t;
typedef struct cudnnContext *cudnnHandle_t;
typedef struct CUstream_st *cudaStream_t;
typedef struct cudnnTensorStruct *cudnnTensorDescriptor_t;
typedef struct cudnnFilterStruct *cudnnFilterDescriptor_t;
typedef struct cudnnConvolutionStruct *cudnnConvolutionDescriptor_t;
// ########## 修改结束 ##########

namespace tr {

class CudaBackend : public Backend {
public:
    /**
     * @brief CUDA后端构造函数
     * @param device_id CUDA设备ID，默认为0
     */
    explicit CudaBackend(int device_id = 0);

    /**
     * @brief 析构函数
     */
    ~CudaBackend() override;

    // 内存管理
    std::shared_ptr<void> allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void* get_data_ptr(const std::shared_ptr<void>& holder) override;
    void copy_data(void* dst, const void* src, size_t size,
                   const Device& dst_device, const Device& src_device) const override;

    // 填充操作
    void fill(Tensor& dst, float value) override;
    void fill(Tensor& dst, int8_t value) override;
    void fill(Tensor& dst, int32_t value) override;

    // fill方法别名
    void fill_fp32(Tensor& dst, float value) override;
    void fill_int8(Tensor& dst, int8_t value) override;
    void fill_int32(Tensor& dst, int32_t value) override;

    // 基本运算
    void add(Tensor& result, const Tensor& a, const Tensor& b) override;
    void mul(Tensor& result, const Tensor& a, const Tensor& b) override;

    // 矩阵乘法
    Tensor mm(const Tensor& a, const Tensor& b) override;
    void mm_into(const Tensor& a, const Tensor& b, Tensor& result) override;

    // 设备转换方法
    Tensor to(const Tensor& tensor, const Device& device) const override;
    Tensor to_cpu(const Tensor& tensor) const override;
    Tensor from_cpu(const Tensor& tensor) const override;

    // 辅助方法
    std::string name() const override { return "CudaBackend"; }
    Device device() const override;

    // 数据访问
    float get_scalar_float(const Tensor& tensor) override;
    int32_t get_scalar_int32(const Tensor& tensor) override;
    int8_t get_scalar_int8(const Tensor& tensor) override;
    int64_t get_memory_size(const Tensor& tensor) override;

    // 张量创建操作
    Tensor empty(const Shape& shape, DType dtype) override;
    Tensor zeros(const Shape& shape, DType dtype) override;
    Tensor ones(const Shape& shape, DType dtype) override;
    Tensor randn(const Shape& shape, unsigned int seed = 0) override;

    // 视图操作
    Tensor view(const Tensor& input, const Shape& new_shape) override;

    // 张量复制操作（V1.26.5新增）
    Tensor copy(const Tensor& tensor) const override;
    void copy_into(const Tensor& src, Tensor& dst) const override;

    // 张量比较操作（V1.26.5新增）
    bool is_close(const Tensor& tensor_a, const Tensor& tensor_b, float eps = 5e-5f) const;

    // 卷积操作（V1.35.4新增）
    Tensor conv(const Tensor& input, const Tensor& kernel, int32_t stride = 1, int32_t padding = 0);
    void conv_into(const Tensor& input, const Tensor& kernel, Tensor& result, int32_t stride = 1, int32_t padding = 0);
    Tensor transposed_conv(const Tensor& input, const Tensor& kernel, int32_t stride = 1, int32_t padding = 0);
    void transposed_conv_into(const Tensor& input, const Tensor& kernel, Tensor& result, int32_t stride = 1, int32_t padding = 0);

    // CUDA特定接口（供高级模块如Conv层使用）
    cudaStream_t stream() const { return stream_; }
    cublasHandle_t cublas_handle() const { return cublas_handle_; }
    cudnnHandle_t cudnn_handle() const { return cudnn_handle_; }

    // 同步接口
    void synchronize() const;  // 同步设备

private:
    int device_id_;
    cudaStream_t stream_;
    cublasHandle_t cublas_handle_;
    cudnnHandle_t cudnn_handle_;

    // GEMM算法缓存
    mutable std::mutex gemm_algo_cache_mutex_;
    std::map<std::tuple<int, int, int>, int> gemm_algo_cache_;
    std::map<std::tuple<int, int, int>, size_t> gemm_workspace_size_cache_;

    void init_cuda_context();
    void cleanup_cuda_context();
    void set_device() const;
    void validate_same_device(const Device& device) const;
    void validate_tensor_shape(const Tensor& a, const Tensor& b) const;

    // GEMM算法查找辅助函数
    std::pair<int, size_t> find_best_gemm_algorithm(int M, int K, int N);

    // 卷积配置缓存键类型 (N, C, H, W, K, kH, s, p)
    using ConvConfigKey = std::tuple<int, int, int, int, int, int, int, int>;

    // 转置卷积配置缓存键类型 (N, C, H, W, K, kH, s, p)
    using TransposedConvConfigKey = std::tuple<int, int, int, int, int, int, int, int>;

    /**
     * @brief 缓存的卷积所需的所有对象
     */
    struct ConvConfigCacheEntry {
        void* input_desc;
        void* output_desc;
        void* filter_desc;
        void* conv_desc;
        int algo;
        size_t workspace_size;

        // 构造函数，确保所有句柄都已创建
        ConvConfigCacheEntry();

        // 析构函数，自动清理
        ~ConvConfigCacheEntry();

        // 禁用拷贝
        ConvConfigCacheEntry(const ConvConfigCacheEntry&) = delete;
        ConvConfigCacheEntry& operator=(const ConvConfigCacheEntry&) = delete;
    };

    /**
     * @brief 缓存的转置卷积所需的所有对象
     */
    struct TransposedConvConfigCacheEntry {
        void* input_desc;
        void* output_desc;
        void* filter_desc;
        void* conv_desc;
        int algo;
        size_t workspace_size;

        // 构造函数，确保所有句柄都已创建
        TransposedConvConfigCacheEntry();

        // 析构函数，自动清理
        ~TransposedConvConfigCacheEntry();

        // 禁用拷贝
        TransposedConvConfigCacheEntry(const TransposedConvConfigCacheEntry&) = delete;
        TransposedConvConfigCacheEntry& operator=(const TransposedConvConfigCacheEntry&) = delete;
    };

    // 卷积配置缓存（替代原来的分散缓存）
    mutable std::mutex conv_config_cache_mutex_;
    std::map<ConvConfigKey, std::shared_ptr<ConvConfigCacheEntry>> conv_config_cache_;

    // 转置卷积配置缓存
    mutable std::mutex transposed_conv_config_cache_mutex_;
    std::map<TransposedConvConfigKey, std::shared_ptr<TransposedConvConfigCacheEntry>> transposed_conv_config_cache_;

    // 工作空间缓存
    mutable std::mutex workspace_cache_mutex_;
    std::map<size_t, std::shared_ptr<void>> workspace_cache_; // 按大小缓存工作空间

    // 卷积辅助函数
    void validate_conv_tensors(const Tensor& input, const Tensor& kernel) const;
    std::shared_ptr<ConvConfigCacheEntry> get_conv_config(const Tensor& input, const Tensor& kernel, const Tensor& result, int32_t stride, int32_t padding);
    std::shared_ptr<TransposedConvConfigCacheEntry> get_transposed_conv_config(const Tensor& input, const Tensor& kernel, const Tensor& result, int32_t stride, int32_t padding);
    std::shared_ptr<void> get_workspace(size_t size); // 获取缓存的工作空间
    Shape calculate_conv_output_shape(const Shape& input_shape, const Shape& kernel_shape, int32_t stride, int32_t padding) const;
    Shape calculate_transposed_conv_output_shape(const Shape& input_shape, const Shape& kernel_shape, int32_t stride, int32_t padding) const;
};

} // namespace tr
#endif // TR_USE_CUDA

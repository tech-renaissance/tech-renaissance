/**
 * @file cuda_common.h
 * @brief CUDA后端通用宏和工具函数
 * @details 包含CUDA错误检查宏和其他通用CUDA工具
 * @version 1.00.00
 * @date 2025-10-31
 * @author 技术觉醒团队
 * @note 依赖项: 无外部依赖
 * @note 所属系列: backend
 */

#pragma once

#include <sstream>
#include "tech_renaissance/utils/tr_exception.h"

#ifdef TR_USE_CUDA

// CUDA头文件包含
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::ostringstream oss; \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                << " - " << cudaGetErrorString(error); \
            throw tr::TRException(oss.str()); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::ostringstream oss; \
            oss << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                << " - status: " << static_cast<int>(status); \
            throw tr::TRException(oss.str()); \
        } \
    } while(0)

#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            std::ostringstream oss; \
            oss << "cuDNN error at " << __FILE__ << ":" << __LINE__ \
                << " - " << cudnnGetErrorString(status); \
            throw tr::TRException(oss.str()); \
        } \
    } while(0)

#endif // TR_USE_CUDA
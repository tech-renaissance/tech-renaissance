/**
 * @file cpu_create.cpp
 * @brief CPU后端张量创建函数实现
 * @details 实现张量创建和填充功能，包括full、randn、uniform、randint、randbool等
 * @version 1.00.00
 * @date 2025-11-02
 * @author 技术觉醒团队
 * @note 依赖项: cpu_backend.h, tensor.h, backend_manager.h
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/data/storage.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/utils/tr_exception.h"

#include <random>
#include <cstring>
#include <stdexcept>

#ifdef TR_USE_EIGEN
#include "Core"
#endif

namespace tr {

// ===== 张量创建函数实现 =====

Tensor CpuBackend::full(const Shape& shape, float value, DType dtype) {
    // 检查INT8支持
    if (dtype == DType::INT8) {
        throw TRException("full: TODO: 未来会实现INT8类型张量的操作");
    }

    // 使用Tensor::empty创建张量并分配内存
    Tensor result = Tensor::empty(shape, dtype, tr::CPU);

#ifdef TR_USE_EIGEN
    // Eigen优化版本
    float* data = static_cast<float*>(result.data_ptr());
    int64_t numel = result.numel();
    Eigen::Map<Eigen::VectorXf> eigen_vec(data, numel);
    eigen_vec.setConstant(value);
#else
    // 朴素实现
    fill(result, value);
#endif

    return result;
}

void CpuBackend::full_inplace(Tensor& tensor_a, float value) {
    // 检查INT8支持
    if (tensor_a.dtype() == DType::INT8) {
        throw TRException("full_inplace: TODO: 未来会实现INT8类型张量的操作");
    }

    // 检查空张量
    if (tensor_a.is_empty()) {
        throw TRException("full_inplace: Cannot operate on empty tensor");
    }

    // 检查设备
    if (tensor_a.device() != tr::CPU) {
        throw TRException("full_inplace: Device must be CPU");
    }

#ifdef TR_USE_EIGEN
    // Eigen优化版本
    float* data = static_cast<float*>(tensor_a.data_ptr());
    int64_t numel = tensor_a.numel();
    Eigen::Map<Eigen::VectorXf> eigen_vec(data, numel);
    eigen_vec.setConstant(value);
#else
    // 朴素实现
    fill(tensor_a, value);
#endif
}

Tensor CpuBackend::randn(const Shape& shape, unsigned int seed) {
    // 创建张量并分配内存
    Tensor result = Tensor::empty(shape, DType::FP32, tr::CPU);

    // 使用C++11随机数生成器
    std::mt19937 engine(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float* data = static_cast<float*>(result.data_ptr());
    int64_t numel = result.numel();

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist(engine);
    }

    return result;
}

void CpuBackend::randn_inplace(Tensor& tensor_a, unsigned int seed) {
    // 检查INT8支持
    if (tensor_a.dtype() == DType::INT8) {
        throw TRException("randn_inplace: This function can only be used for FP32 tensors");
    }

    // 检查空张量
    if (tensor_a.is_empty()) {
        throw TRException("randn_inplace: Cannot operate on empty tensor");
    }

    // 检查设备
    if (tensor_a.device() != tr::CPU) {
        throw TRException("randn_inplace: Device must be CPU");
    }

    // 使用C++11随机数生成器
    std::mt19937 engine(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float* data = static_cast<float*>(tensor_a.data_ptr());
    int64_t numel = tensor_a.numel();

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist(engine);
    }
}

Tensor CpuBackend::uniform(const Shape& shape, float min_val, float max_val, unsigned int seed) {
    // 创建张量并分配内存
    Tensor result = Tensor::empty(shape, DType::FP32, tr::CPU);

    // 使用C++11随机数生成器
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);

    float* data = static_cast<float*>(result.data_ptr());
    int64_t numel = result.numel();

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist(engine);
    }

    return result;
}

void CpuBackend::uniform_inplace(Tensor& tensor_a, float min_val, float max_val, unsigned int seed) {
    // 检查INT8支持
    if (tensor_a.dtype() == DType::INT8) {
        throw TRException("uniform_inplace: This function can only be used for FP32 tensors");
    }

    // 检查空张量
    if (tensor_a.is_empty()) {
        throw TRException("uniform_inplace: Cannot operate on empty tensor");
    }

    // 检查设备
    if (tensor_a.device() != tr::CPU) {
        throw TRException("uniform_inplace: Device must be CPU");
    }

    // 使用C++11随机数生成器
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);

    float* data = static_cast<float*>(tensor_a.data_ptr());
    int64_t numel = tensor_a.numel();

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist(engine);
    }
}

Tensor CpuBackend::randint(const Shape& shape, int low, int high, unsigned int seed, DType dtype) {
    // 检查参数
    if (low >= high) {
        throw TRException("randint: low must be less than high");
    }

    // 检查INT8支持
    if (dtype == DType::INT8) {
        throw TRException("randint: TODO: 未来会实现INT8类型张量的操作");
    }

    // 创建张量并分配内存
    Tensor result = Tensor::empty(shape, DType::FP32, tr::CPU);

    // 使用C++11随机数生成器
    std::mt19937 engine(seed);
    std::uniform_int_distribution<int> dist(low, high - 1);

    float* data = static_cast<float*>(result.data_ptr());
    int64_t numel = result.numel();

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = static_cast<float>(dist(engine));
    }

    return result;
}

void CpuBackend::randint_inplace(Tensor& tensor_a, int low, int high, unsigned int seed) {
    // 检查参数
    if (low >= high) {
        throw TRException("randint_inplace: low must be less than high");
    }

    // 检查INT8支持
    if (tensor_a.dtype() == DType::INT8) {
        throw TRException("randint_inplace: TODO: 未来会实现INT8类型张量的操作");
    }

    // 检查空张量
    if (tensor_a.is_empty()) {
        throw TRException("randint_inplace: Cannot operate on empty tensor");
    }

    // 检查设备
    if (tensor_a.device() != tr::CPU) {
        throw TRException("randint_inplace: Device must be CPU");
    }

    // 使用C++11随机数生成器
    std::mt19937 engine(seed);
    std::uniform_int_distribution<int> dist(low, high - 1);

    float* data = static_cast<float*>(tensor_a.data_ptr());
    int64_t numel = tensor_a.numel();

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = static_cast<float>(dist(engine));
    }
}

Tensor CpuBackend::randbool(const Shape& shape, float rate_of_zeros, unsigned int seed, DType dtype) {
    // 检查参数
    if (rate_of_zeros < 0.0f || rate_of_zeros > 1.0f) {
        throw TRException("randbool: rate_of_zeros must be between 0.0 and 1.0");
    }

    // 检查INT8支持
    if (dtype == DType::INT8) {
        throw TRException("randbool: TODO: 未来会实现INT8类型张量的操作");
    }

    // 创建张量并分配内存
    Tensor result = Tensor::empty(shape, DType::FP32, tr::CPU);

#ifdef TR_USE_EIGEN
    // Eigen优化版本
    float* data = static_cast<float*>(result.data_ptr());
    int64_t numel = result.numel();
    Eigen::Map<Eigen::VectorXf> eigen_vec(data, numel);

    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    eigen_vec = eigen_vec.unaryExpr([&](float) {
        return (dist(engine) < rate_of_zeros) ? 0.0f : 1.0f;
    });
#else
    // 朴素实现
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float* data = static_cast<float*>(result.data_ptr());
    int64_t numel = result.numel();

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = (dist(engine) < rate_of_zeros) ? 0.0f : 1.0f;
    }
#endif

    return result;
}

void CpuBackend::randbool_inplace(Tensor& tensor_a, float rate_of_zeros, unsigned int seed) {
    // 检查参数
    if (rate_of_zeros < 0.0f || rate_of_zeros > 1.0f) {
        throw TRException("randbool_inplace: rate_of_zeros must be between 0.0 and 1.0");
    }

    // 检查INT8支持
    if (tensor_a.dtype() == DType::INT8) {
        throw TRException("randbool_inplace: TODO: 未来会实现INT8类型张量的操作");
    }

    // 检查空张量
    if (tensor_a.is_empty()) {
        throw TRException("randbool_inplace: Cannot operate on empty tensor");
    }

    // 检查设备
    if (tensor_a.device() != tr::CPU) {
        throw TRException("randbool_inplace: Device must be CPU");
    }

#ifdef TR_USE_EIGEN
    // Eigen优化版本
    float* data = static_cast<float*>(tensor_a.data_ptr());
    int64_t numel = tensor_a.numel();
    Eigen::Map<Eigen::VectorXf> eigen_vec(data, numel);

    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    eigen_vec = eigen_vec.unaryExpr([&](float) {
        return (dist(engine) < rate_of_zeros) ? 0.0f : 1.0f;
    });
#else
    // 朴素实现
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float* data = static_cast<float*>(tensor_a.data_ptr());
    int64_t numel = tensor_a.numel();

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = (dist(engine) < rate_of_zeros) ? 0.0f : 1.0f;
    }
#endif
}

} // namespace tr
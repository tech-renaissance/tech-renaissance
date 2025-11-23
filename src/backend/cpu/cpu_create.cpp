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
#include <Eigen/Core>
#endif

namespace tr {

Tensor CpuBackend::null_tensor() {
    return Tensor();
}

Tensor CpuBackend::empty(const Shape& shape, DType dtype) {
    Tensor result(shape, dtype, CPU);
    auto memory_holder = this->allocate(result.numel() * result.dtype_size());
    result.storage_ = std::make_shared<Storage>(result.numel() * result.dtype_size(), result.device());
    result.storage_->set_data_ptr(this->get_data_ptr(memory_holder), memory_holder);
    return result;
}

Tensor CpuBackend::empty(const Shape& shape, DType dtype) const {
    Tensor result(shape, dtype, CPU);
    auto memory_holder = const_cast<CpuBackend*>(this)->allocate(result.numel() * result.dtype_size());
    result.storage_ = std::make_shared<Storage>(result.numel() * result.dtype_size(), result.device());
    result.storage_->set_data_ptr(const_cast<CpuBackend*>(this)->get_data_ptr(memory_holder), memory_holder);
    return result;
}

Tensor CpuBackend::zeros(const Shape& shape, DType dtype) {
    Tensor result(shape, dtype, CPU);
    auto memory_holder = this->allocate(result.numel() * result.dtype_size());
    result.storage_ = std::make_shared<Storage>(result.numel() * result.dtype_size(), result.device());
    result.storage_->set_data_ptr(this->get_data_ptr(memory_holder), memory_holder);

    // 使用memset高效填充零值
    void* data = result.data_ptr();
    size_t total_bytes = result.numel() * dtype_size(result.dtype());
    std::memset(data, 0, total_bytes);

    return result;
}

Tensor CpuBackend::ones(const Shape& shape, DType dtype) {
    Tensor result(shape, dtype, CPU);
    auto memory_holder = this->allocate(result.numel() * result.dtype_size());
    result.storage_ = std::make_shared<Storage>(result.numel() * result.dtype_size(), result.device());
    result.storage_->set_data_ptr(this->get_data_ptr(memory_holder), memory_holder);

#ifdef TR_USE_EIGEN
    // Eigen优化实现
    if (result.dtype() == DType::FP32) {
        float* data = static_cast<float*>(result.data_ptr());
        size_t num_elements = result.numel();
        using MatrixType = Eigen::Matrix<float, Eigen::Dynamic, 1>;
        Eigen::Map<MatrixType> eigen_vec(data, num_elements);
        eigen_vec.setConstant(1.0f);
    } else if (result.dtype() == DType::INT8) {
        int8_t* data = static_cast<int8_t*>(result.data_ptr());
        size_t num_elements = result.numel();
        using MatrixType = Eigen::Matrix<int8_t, Eigen::Dynamic, 1>;
        Eigen::Map<MatrixType> eigen_vec(data, num_elements);
        eigen_vec.setConstant(1);
    } else if (result.dtype() == DType::INT32) {
        int32_t* data = static_cast<int32_t*>(result.data_ptr());
        size_t num_elements = result.numel();
        using MatrixType = Eigen::Matrix<int32_t, Eigen::Dynamic, 1>;
        Eigen::Map<MatrixType> eigen_vec(data, num_elements);
        eigen_vec.setConstant(1);
    }
#else
    // 朴素实现
    if (result.dtype() == DType::FP32) {
        float* data = static_cast<float*>(result.data_ptr());
        size_t num_elements = result.numel();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = 1.0f;
        }
    } else if (result.dtype() == DType::INT8) {
        int8_t* data = static_cast<int8_t*>(result.data_ptr());
        size_t num_elements = result.numel();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = 1;
        }
    } else if (result.dtype() == DType::INT32) {
        int32_t* data = static_cast<int32_t*>(result.data_ptr());
        size_t num_elements = result.numel();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = 1;
        }
    }
#endif

    return result;
}

// ===== 张量创建函数实现 =====

Tensor CpuBackend::full(const Shape& shape, float value, DType dtype) {
    // 检查INT8支持
    if (dtype != DType::FP32) {
        throw TRException("full: 暂时只支持FP32类型的操作");
    }
    Tensor result = this->empty(shape, dtype);

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
    if (tensor_a.dtype() != DType::FP32) {
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
    Tensor result = this->empty(shape, DType::FP32);

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
    if (tensor_a.dtype() != DType::FP32) {
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
    Tensor result = this->empty(shape, DType::FP32);

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
    if (tensor_a.dtype() != DType::FP32) {
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

Tensor CpuBackend::randint(const Shape& shape, int low, int high, DType dtype, unsigned int seed) {
    // 检查参数
    if (low >= high) {
        throw TRException("randint: low must be less than high");
    }

    // 检查数据类型支持
    if (dtype != DType::FP32 && dtype != DType::INT8 && dtype != DType::INT32) {
        throw TRException("randint: only supports FP32, INT8, and INT32 data types");
    }

    // 检查INT8范围
    if (dtype == DType::INT8) {
        if (low < -128 || high > 127) {
            throw TRException("randint: INT8 range must be within [-128, 127]");
        }
    }

    // 创建张量并分配内存
    Tensor result = this->empty(shape, dtype);

    // 使用C++11随机数生成器
    std::mt19937 engine(seed);
    std::uniform_int_distribution<int32_t> dist(low, high - 1);

    int64_t numel = result.numel();

    if (dtype == DType::FP32) {
        float* data = static_cast<float*>(result.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = static_cast<float>(dist(engine));
        }
    } else if (dtype == DType::INT8) {
        int8_t* data = static_cast<int8_t*>(result.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = static_cast<int8_t>(dist(engine));
        }
    } else if (dtype == DType::INT32) {
        int32_t* data = static_cast<int32_t*>(result.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = dist(engine);
        }
    }

    return result;
}

void CpuBackend::randint_inplace(Tensor& tensor_a, int low, int high, DType dtype, unsigned int seed) {
    // 检查参数
    if (low >= high) {
        throw TRException("randint_inplace: low must be less than high");
    }

    // 检查数据类型支持
    if (dtype != DType::FP32 && dtype != DType::INT8 && dtype != DType::INT32) {
        throw TRException("randint_inplace: only supports FP32, INT8, and INT32 data types");
    }

    // 检查张量数据类型与输入dtype是否一致
    if (tensor_a.dtype() != dtype) {
        throw TRException("randint_inplace: tensor dtype must match input dtype parameter");
    }

    // 检查INT8范围
    if (dtype == DType::INT8) {
        if (low < -128 || high > 127) {
            throw TRException("randint_inplace: INT8 range must be within [-128, 127]");
        }
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
    std::uniform_int_distribution<int32_t> dist(low, high - 1);

    int64_t numel = tensor_a.numel();

    if (dtype == DType::FP32) {
        float* data = static_cast<float*>(tensor_a.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = static_cast<float>(dist(engine));
        }
    } else if (dtype == DType::INT8) {
        int8_t* data = static_cast<int8_t*>(tensor_a.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = static_cast<int8_t>(dist(engine));
        }
    } else if (dtype == DType::INT32) {
        int32_t* data = static_cast<int32_t*>(tensor_a.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = dist(engine);
        }
    }
}

Tensor CpuBackend::randbool(const Shape& shape, float rate_of_zeros, unsigned int seed, DType dtype) {
    // 检查参数
    if (rate_of_zeros < 0.0f || rate_of_zeros > 1.0f) {
        throw TRException("randbool: rate_of_zeros must be between 0.0 and 1.0");
    }

    // 检查类型
    if (dtype != DType::FP32) {
        throw TRException("randbool: 暂时只支持FP32类型张量的操作");
    }

    // 创建张量并分配内存
    Tensor result = this->empty(shape, DType::FP32);

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
    if (tensor_a.dtype() != DType::FP32) {
        throw TRException("randbool_inplace: 暂时只支持FP32张量操作");
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
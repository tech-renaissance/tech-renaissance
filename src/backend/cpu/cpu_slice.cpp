/**
 * @file cpu_slice.cpp
 * @brief CPU后端张量切片操作实现
 * @details 实现基于Offset参数的张量切片功能，支持NCHW四维张量的灵活切片
 * @version 1.00.00
 * @date 2025-11-03
 * @author 技术觉醒团队
 * @note 依赖项: cpu_backend.h, offset.h, eigen3/Eigen
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/data/offset.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/utils/tr_exception.h"
#include <cstring>
#include <algorithm>
#include <string>

namespace tr {

// 计算切片后的实际结束位置
static int32_t calculate_end(int32_t start, int32_t end, int32_t dim_size, const std::string& dim_name) {
    if (end == -1) {
        return dim_size;  // 切到该维度的末尾
    }
    if (end > dim_size) {
        throw TRException("CpuBackend::slice: " + dim_name + " dimension end index " +
                        std::to_string(end) + " exceeds tensor size " + std::to_string(dim_size));
    }
    return end;
}

// 计算切片后的输出维度大小
static int32_t calculate_output_size(int32_t start, int32_t end, int32_t stride, const std::string& dim_name) {
    if (start >= end) {
        throw TRException("CpuBackend::slice: invalid " + dim_name + " range [" +
                        std::to_string(start) + ", " + std::to_string(end) + ")");
    }
    int32_t size = end - start;
    return (size + stride - 1) / stride;  // 向上取整
}

// 验证切片参数是否有效
static void validate_slice_parameters(const Tensor& tensor_a, const Offset& offset) {
    // 检查设备
    if (!tensor_a.device().is_cpu()) {
        throw TRException("CpuBackend::slice: tensor must be on CPU device");
    }

    // 检查内存是否已分配
    if (!tensor_a.storage_allocated()) {
        throw TRException("CpuBackend::slice: tensor storage not allocated");
    }

    const Shape& shape = tensor_a.shape();
    int32_t ndims = shape.ndim();

    // 验证W维度（所有维度都有）
    {
        int32_t w_start = offset.w_start();
        int32_t w_end = calculate_end(w_start, offset.w_end(), shape.w(), "W");
        if (w_start < 0 || w_start >= shape.w()) {
            throw TRException("CpuBackend::slice: W dimension start index " +
                            std::to_string(w_start) + " out of range [0, " +
                            std::to_string(shape.w()) + "]");
        }
    }

    // 验证H维度（2D及以上）
    if (ndims >= 2) {
        int32_t h_start = offset.h_start();
        int32_t h_end = calculate_end(h_start, offset.h_end(), shape.h(), "H");
        if (h_start < 0 || h_start >= shape.h()) {
            throw TRException("CpuBackend::slice: H dimension start index " +
                            std::to_string(h_start) + " out of range [0, " +
                            std::to_string(shape.h()) + "]");
        }
    }

    // 验证C维度（3D及以上）
    if (ndims >= 3) {
        int32_t c_start = offset.c_start();
        int32_t c_end = calculate_end(c_start, offset.c_end(), shape.c(), "C");
        if (c_start < 0 || c_start >= shape.c()) {
            throw TRException("CpuBackend::slice: C dimension start index " +
                            std::to_string(c_start) + " out of range [0, " +
                            std::to_string(shape.c()) + "]");
        }
    }

    // 验证N维度（4D）
    if (ndims >= 4) {
        int32_t n_start = offset.n_start();
        int32_t n_end = calculate_end(n_start, offset.n_end(), shape.n(), "N");
        if (n_start < 0 || n_start >= shape.n()) {
            throw TRException("CpuBackend::slice: N dimension start index " +
                            std::to_string(n_start) + " out of range [0, " +
                            std::to_string(shape.n()) + "]");
        }
    }
}

// 计算输出张量的形状
static Shape calculate_output_shape(const Tensor& tensor_a, const Offset& offset) {
    const Shape& input_shape = tensor_a.shape();
    int32_t ndims = input_shape.ndim();

    if (ndims == 0) {
        return Shape();
    } else if (ndims == 1) {
        // 1D张量，只使用W维度
        int32_t w_start = offset.w_start();
        int32_t w_end = calculate_end(w_start, offset.w_end(), input_shape.w(), "W");
        int32_t w_stride = offset.w_stride();
        int32_t w_size = calculate_output_size(w_start, w_end, w_stride, "W");
        return Shape(w_size);
    } else if (ndims == 2) {
        // 2D张量，使用H和W维度
        int32_t h_start = offset.h_start();
        int32_t h_end = calculate_end(h_start, offset.h_end(), input_shape.h(), "H");
        int32_t h_stride = offset.h_stride();
        int32_t h_size = calculate_output_size(h_start, h_end, h_stride, "H");

        int32_t w_start = offset.w_start();
        int32_t w_end = calculate_end(w_start, offset.w_end(), input_shape.w(), "W");
        int32_t w_stride = offset.w_stride();
        int32_t w_size = calculate_output_size(w_start, w_end, w_stride, "W");

        return Shape(h_size, w_size);
    } else if (ndims == 3) {
        // 3D张量，使用C、H和W维度
        int32_t c_start = offset.c_start();
        int32_t c_end = calculate_end(c_start, offset.c_end(), input_shape.c(), "C");
        int32_t c_stride = offset.c_stride();
        int32_t c_size = calculate_output_size(c_start, c_end, c_stride, "C");

        int32_t h_start = offset.h_start();
        int32_t h_end = calculate_end(h_start, offset.h_end(), input_shape.h(), "H");
        int32_t h_stride = offset.h_stride();
        int32_t h_size = calculate_output_size(h_start, h_end, h_stride, "H");

        int32_t w_start = offset.w_start();
        int32_t w_end = calculate_end(w_start, offset.w_end(), input_shape.w(), "W");
        int32_t w_stride = offset.w_stride();
        int32_t w_size = calculate_output_size(w_start, w_end, w_stride, "W");

        return Shape(c_size, h_size, w_size);
    } else {
        // 4D张量，使用N、C、H和W维度
        int32_t n_start = offset.n_start();
        int32_t n_end = calculate_end(n_start, offset.n_end(), input_shape.n(), "N");
        int32_t n_stride = offset.n_stride();
        int32_t n_size = calculate_output_size(n_start, n_end, n_stride, "N");

        int32_t c_start = offset.c_start();
        int32_t c_end = calculate_end(c_start, offset.c_end(), input_shape.c(), "C");
        int32_t c_stride = offset.c_stride();
        int32_t c_size = calculate_output_size(c_start, c_end, c_stride, "C");

        int32_t h_start = offset.h_start();
        int32_t h_end = calculate_end(h_start, offset.h_end(), input_shape.h(), "H");
        int32_t h_stride = offset.h_stride();
        int32_t h_size = calculate_output_size(h_start, h_end, h_stride, "H");

        int32_t w_start = offset.w_start();
        int32_t w_end = calculate_end(w_start, offset.w_end(), input_shape.w(), "W");
        int32_t w_stride = offset.w_stride();
        int32_t w_size = calculate_output_size(w_start, w_end, w_stride, "W");

        return Shape(n_size, c_size, h_size, w_size);
    }
}

// 执行实际的切片操作（内部函数）
template<typename T>
static void perform_slice_operation(const Tensor& tensor_a, Tensor& result, const Offset& offset) {
    const T* input_data = static_cast<const T*>(tensor_a.data_ptr());
    T* output_data = static_cast<T*>(result.data_ptr());

    const Shape& input_shape = tensor_a.shape();
    const Shape& output_shape = result.shape();
    int32_t ndims = input_shape.ndim();

    if (ndims == 0) {
        // 0D张量：直接复制单个元素
        if (input_shape.numel() > 0) {
            *output_data = *input_data;
        }
        return;
    }

    // 获取切片参数
    int32_t n_start = (ndims >= 4) ? offset.n_start() : 0;
    int32_t n_end = (ndims >= 4) ? calculate_end(n_start, offset.n_end(), input_shape.n(), "N") : 1;
    int32_t n_stride = (ndims >= 4) ? offset.n_stride() : 1;

    int32_t c_start = (ndims >= 3) ? offset.c_start() : 0;
    int32_t c_end = (ndims >= 3) ? calculate_end(c_start, offset.c_end(), input_shape.c(), "C") : 1;
    int32_t c_stride = (ndims >= 3) ? offset.c_stride() : 1;

    int32_t h_start = (ndims >= 2) ? offset.h_start() : 0;
    int32_t h_end = (ndims >= 2) ? calculate_end(h_start, offset.h_end(), input_shape.h(), "H") : 1;
    int32_t h_stride = (ndims >= 2) ? offset.h_stride() : 1;

    int32_t w_start = offset.w_start();
    int32_t w_end = calculate_end(w_start, offset.w_end(), input_shape.w(), "W");
    int32_t w_stride = offset.w_stride();

    size_t output_idx = 0;

    // 执行多维切片
    for (int32_t n = n_start; n < n_end; n += n_stride) {
        for (int32_t c = c_start; c < c_end; c += c_stride) {
            for (int32_t h = h_start; h < h_end; h += h_stride) {
                for (int32_t w = w_start; w < w_end; w += w_stride) {
                    // 计算输入张量中的线性索引
                    size_t input_idx = 0;

                    if (ndims == 1) {
                        input_idx = w;
                    } else if (ndims == 2) {
                        input_idx = h * input_shape.w() + w;
                    } else if (ndims == 3) {
                        input_idx = c * input_shape.h() * input_shape.w() +
                                   h * input_shape.w() + w;
                    } else {  // ndims == 4
                        input_idx = n * input_shape.c() * input_shape.h() * input_shape.w() +
                                   c * input_shape.h() * input_shape.w() +
                                   h * input_shape.w() + w;
                    }

                    output_data[output_idx++] = input_data[input_idx];
                }
            }
        }
    }
}

// 公共API实现
Tensor CpuBackend::slice(const Tensor& tensor_a, const Offset& offset) {
    // 验证输入参数
    validate_slice_parameters(tensor_a, offset);

    // 计算输出形状
    Shape output_shape = calculate_output_shape(tensor_a, offset);

    // 创建输出张量（使用后端方法）
    Tensor result = this->zeros(output_shape, tensor_a.dtype());

    // 执行切片操作
    if (tensor_a.dtype() == DType::FP32) {
        perform_slice_operation<float>(tensor_a, result, offset);
    } else if (tensor_a.dtype() == DType::INT8) {
        perform_slice_operation<int8_t>(tensor_a, result, offset);
    } else if (tensor_a.dtype() == DType::INT32) {
        perform_slice_operation<int32_t>(tensor_a, result, offset);
    } else {
        throw TRException("CpuBackend::slice: unsupported data type " + dtype_to_string(tensor_a.dtype()));
    }

    return result;
}

void CpuBackend::slice_into(const Tensor& tensor_a, Tensor& result, const Offset& offset) {
    // 验证输入参数
    validate_slice_parameters(tensor_a, offset);

    // 验证result的设备
    if (!result.device().is_cpu()) {
        throw TRException("CpuBackend::slice_into: result tensor must be on CPU device");
    }

    // 验证result的内存是否已分配
    if (!result.storage_allocated()) {
        throw TRException("CpuBackend::slice_into: result tensor storage not allocated");
    }

    // 验证数据类型一致性
    if (tensor_a.dtype() != result.dtype()) {
        throw TRException("CpuBackend::slice_into: data type mismatch. input: " +
                        dtype_to_string(tensor_a.dtype()) + ", result: " +
                        dtype_to_string(result.dtype()));
    }

    // 计算期望的输出形状
    Shape expected_shape = calculate_output_shape(tensor_a, offset);

    // 验证result形状是否匹配
    if (result.shape() != expected_shape) {
        throw TRException("CpuBackend::slice_into: result shape mismatch. expected: " +
                        expected_shape.to_string() + ", actual: " +
                        result.shape().to_string());
    }

    // 执行切片操作
    if (tensor_a.dtype() == DType::FP32) {
        perform_slice_operation<float>(tensor_a, result, offset);
    } else if (tensor_a.dtype() == DType::INT8) {
        perform_slice_operation<int8_t>(tensor_a, result, offset);
    } else if (tensor_a.dtype() == DType::INT32) {
        perform_slice_operation<int32_t>(tensor_a, result, offset);
    } else {
        throw TRException("CpuBackend::slice_into: unsupported data type " + dtype_to_string(tensor_a.dtype()));
    }
}

} // namespace tr
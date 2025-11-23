/**
 * @file cpu_loss.cpp
 * @brief CPU后端损失函数实现
 * @details 实现one-hot编码和交叉熵损失函数，支持标签平滑
 * @version 1.42.6
 * @date 2025-11-16
 * @author 技术觉醒团队
 * @note 依赖项: cpu_backend.h, tensor.h, Eigen
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/utils/logger.h"

#ifdef TR_USE_EIGEN
#include <Eigen/Core>
#endif

#include <cstring>
#include <algorithm>
#include <cmath>

namespace tr {

Tensor CpuBackend::one_hot(const Tensor& label, int32_t num_classes, float label_smoothing) {
    // 输入验证
    if (label.dtype() != DType::INT32) {
        TR_THROW_TYPE_ERROR("one_hot function requires INT32 input tensor");
    }

    if (label.ndim() != 1) {
        TR_THROW_SHAPE_ERROR("one_hot function requires 1D input tensor, got " +
                            std::to_string(label.ndim()) + "D tensor");
    }

    if (num_classes <= 0) {
        TR_THROW_VALUE_ERROR("num_classes must be positive, got " + std::to_string(num_classes));
    }

    if (label_smoothing < 0.0f || label_smoothing >= 1.0f) {
        TR_THROW_VALUE_ERROR("label_smoothing must be in [0, 1), got " + std::to_string(label_smoothing));
    }

    int32_t batch_size = label.dim_size(0);

    // 创建输出张量
    Shape result_shape(batch_size, num_classes);
    Tensor result = this->zeros(result_shape, DType::FP32);

    // 调用into版本实现具体逻辑
    this->one_hot_into(label, result, num_classes, label_smoothing);

    return result;
}

void CpuBackend::one_hot_into(const Tensor& label, Tensor& result, int32_t num_classes, float label_smoothing) {
    // 输入验证
    if (label.dtype() != DType::INT32) {
        TR_THROW_TYPE_ERROR("one_hot_into function requires INT32 input tensor");
    }

    if (result.dtype() != DType::FP32) {
        TR_THROW_TYPE_ERROR("one_hot_into function requires FP32 result tensor");
    }

    if (label.ndim() != 1 || result.ndim() != 2) {
        TR_THROW_SHAPE_ERROR("one_hot_into function requires 1D input tensor and 2D result tensor");
    }

    if (label.dim_size(0) != result.dim_size(0)) {
        TR_THROW_SHAPE_ERROR("Input tensor batch size (" + std::to_string(label.dim_size(0)) +
                            ") doesn't match result tensor batch size (" +
                            std::to_string(result.dim_size(0)) + ")");
    }

    if (result.dim_size(1) != num_classes) {
        TR_THROW_SHAPE_ERROR("Result tensor number of classes (" + std::to_string(result.dim_size(1)) +
                            ") doesn't match num_classes (" + std::to_string(num_classes) + ")");
    }

    if (num_classes <= 0) {
        TR_THROW_VALUE_ERROR("num_classes must be positive, got " + std::to_string(num_classes));
    }

    if (label_smoothing < 0.0f || label_smoothing >= 1.0f) {
        TR_THROW_VALUE_ERROR("label_smoothing must be in [0, 1), got " + std::to_string(label_smoothing));
    }

    int32_t batch_size = label.dim_size(0);

    // 获取数据指针
    const int32_t* label_data = static_cast<const int32_t*>(label.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    // 计算标签平滑参数
    float smooth_value = label_smoothing / static_cast<float>(num_classes);
    float one_minus_smooth = 1.0f - label_smoothing;

    // 验证标签值的有效性
    for (int32_t i = 0; i < batch_size; ++i) {
        int32_t class_index = label_data[i];
        if (class_index < 0 || class_index >= num_classes) {
            TR_THROW_INDEX_ERROR("Label value " + std::to_string(class_index) +
                                " at position " + std::to_string(i) +
                                " is out of range [0, " + std::to_string(num_classes-1) + "]");
        }
    }

    // 使用Eigen进行高效操作
#ifdef TR_USE_EIGEN
    // 创建Eigen映射
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        result_map(result_data, batch_size, num_classes);

    // 填充基础值（所有位置都设为平滑值）
    result_map.setConstant(smooth_value);

    // 在正确的类别位置加上(1 - label_smoothing)
    for (int32_t i = 0; i < batch_size; ++i) {
        int32_t class_index = label_data[i];
        result_map(i, class_index) += one_minus_smooth;  // smooth_value + one_minus_smooth = 1.0f
    }
#else
    // 纯C++实现
    size_t total_elements = static_cast<size_t>(batch_size * num_classes);

    // 首先填充所有元素为平滑值
    for (size_t idx = 0; idx < total_elements; ++idx) {
        result_data[idx] = smooth_value;
    }

    // 然后在正确的类别位置加上(1 - label_smoothing)
    for (int32_t i = 0; i < batch_size; ++i) {
        int32_t class_index = label_data[i];
        result_data[i * num_classes + class_index] += one_minus_smooth;
    }
#endif
}

float CpuBackend::crossentropy(const Tensor& pred, const Tensor& label, std::string reduction) {
    // 输入验证
    if (pred.dtype() != DType::FP32 || label.dtype() != DType::FP32) {
        TR_THROW_TYPE_ERROR("crossentropy function requires FP32 input tensors");
    }

    if (pred.ndim() != 2 || label.ndim() != 2) {
        TR_THROW_SHAPE_ERROR("crossentropy function requires 2D input tensors, got " +
                            std::to_string(pred.ndim()) + "D and " +
                            std::to_string(label.ndim()) + "D tensors");
    }

    if (pred.shape() != label.shape()) {
        TR_THROW_SHAPE_ERROR("Prediction tensor shape " + pred.shape().to_string() +
                            " doesn't match label tensor shape " + label.shape().to_string());
    }

    int32_t batch_size = pred.dim_size(0);
    int32_t num_classes = pred.dim_size(1);

    // 获取数据指针
    const float* pred_data = static_cast<const float*>(pred.data_ptr());
    const float* label_data = static_cast<const float*>(label.data_ptr());

    static constexpr float epsilon = 1e-12f;  // 数值稳定性参数

    // 计算交叉熵
    float total_loss = 0.0f;

#ifdef TR_USE_EIGEN
    // 使用Eigen进行高效计算
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        pred_map(pred_data, batch_size, num_classes);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        label_map(label_data, batch_size, num_classes);

    // 逐元素计算 -label * log(pred)
    for (int32_t i = 0; i < batch_size; ++i) {
        for (int32_t j = 0; j < num_classes; ++j) {
            float pred_val = pred_map(i, j);
            float label_val = label_map(i, j);

            // 数值稳定性：确保pred_val >= epsilon
            pred_val = std::max(pred_val, epsilon);

            // 只计算非零标签位置的损失
            if (label_val > 0.0f) {
                total_loss -= label_val * std::log(pred_val);
            }
        }
    }
#else
    // 纯C++实现
    size_t total_elements = static_cast<size_t>(batch_size * num_classes);

    for (size_t idx = 0; idx < total_elements; ++idx) {
        float pred_val = pred_data[idx];
        float label_val = label_data[idx];

        // 数值稳定性：确保pred_val >= epsilon
        pred_val = std::max(pred_val, epsilon);

        // 只计算非零标签位置的损失
        if (label_val > 0.0f) {
            total_loss -= label_val * std::log(pred_val);
        }
    }
#endif

    // 应用reduction
    if (reduction == "sum") {
        return total_loss;
    } else {  // 默认为"mean"
        return total_loss / static_cast<float>(batch_size);
    }
}

float CpuBackend::mse(const Tensor& pred, const Tensor& target, std::string reduction) {
    // 输入验证
    if (pred.dtype() != DType::FP32 || target.dtype() != DType::FP32) {
        TR_THROW_TYPE_ERROR("mse function requires FP32 input tensors");
    }

    if (pred.ndim() != 2 || target.ndim() != 2) {
        TR_THROW_SHAPE_ERROR("mse function requires 2D input tensors, got " +
                            std::to_string(pred.ndim()) + "D and " +
                            std::to_string(target.ndim()) + "D tensors");
    }

    if (pred.shape() != target.shape()) {
        TR_THROW_SHAPE_ERROR("Prediction tensor shape " + pred.shape().to_string() +
                            " doesn't match target tensor shape " + target.shape().to_string());
    }

    int32_t batch_size = pred.dim_size(0);
    int32_t num_elements = pred.dim_size(1);
    int32_t total_elements = batch_size * num_elements;

    // 获取数据指针
    const float* pred_data = static_cast<const float*>(pred.data_ptr());
    const float* target_data = static_cast<const float*>(target.data_ptr());

    float total_loss = 0.0f;

#ifdef TR_USE_EIGEN
    // 使用Eigen进行高效计算
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        pred_map(pred_data, batch_size, num_elements);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        target_map(target_data, batch_size, num_elements);

    // 逐元素计算(pred - target)^2并累加
    for (int32_t i = 0; i < batch_size; ++i) {
        for (int32_t j = 0; j < num_elements; ++j) {
            float diff = pred_map(i, j) - target_map(i, j);
            total_loss += diff * diff;  // (pred - target)^2
        }
    }
#else
    // 纯C++实现
    for (int32_t idx = 0; idx < total_elements; ++idx) {
        float diff = pred_data[idx] - target_data[idx];
        total_loss += diff * diff;  // (pred - target)^2
    }
#endif

    // 应用reduction
    if (reduction == "sum") {
        return total_loss / static_cast<float>(batch_size);
    } else {  // 默认为"mean"
        return total_loss / static_cast<float>(total_elements);
    }
}

} // namespace tr
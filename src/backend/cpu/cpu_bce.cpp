/**
 * @file cpu_bce.cpp
 * @brief CPU后端二元交叉熵运算实现
 * @details 实现逐元素二元交叉熵计算，支持非原地和指定输出张量两种模式
 * @version 1.00.00
 * @date 2025-11-02
 * @author 技术觉醒团队
 * @note 依赖项: cpu_backend.h, tensor.h, Eigen
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/utils/logger.h"

#ifdef TR_USE_EIGEN
#include "Core"
#endif

#include <algorithm>

namespace tr {

// ===== 二元交叉熵运算：bce(goal, pred) =====

Tensor CpuBackend::bce(const Tensor& goal, const Tensor& pred) const {
    validate_same_device(goal.device());
    validate_same_device(pred.device());

    if (goal.is_empty()) {
        throw TRException("[CpuBackend::bce] Goal tensor has no allocated Storage");
    }

    if (pred.is_empty()) {
        throw TRException("[CpuBackend::bce] Prediction tensor has no allocated Storage");
    }

    if (goal.dtype() != DType::FP32 || pred.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::bce] Only FP32 tensors are supported for binary cross entropy. "
                        "TODO: Consider implementing INT8 support in future versions.");
    }

    if (goal.shape() != pred.shape()) {
        throw TRException("[CpuBackend::bce] Shape mismatch: goal shape " +
                        goal.shape().to_string() + " != pred shape " +
                        pred.shape().to_string());
    }

    // 创建输出张量
    Tensor result = Tensor::empty(goal.shape(), goal.dtype(), goal.device());
    bce_into(goal, pred, result);
    return result;
}

void CpuBackend::bce_into(const Tensor& goal, const Tensor& pred, Tensor& result) const {
    validate_same_device(goal.device());
    validate_same_device(pred.device());
    validate_same_device(result.device());

    if (goal.is_empty()) {
        throw TRException("[CpuBackend::bce_into] Goal tensor has no allocated Storage");
    }

    if (pred.is_empty()) {
        throw TRException("[CpuBackend::bce_into] Prediction tensor has no allocated Storage");
    }

    if (result.is_empty()) {
        throw TRException("[CpuBackend::bce_into] Result tensor has no allocated Storage");
    }

    if (goal.dtype() != DType::FP32 || pred.dtype() != DType::FP32 || result.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::bce_into] Only FP32 tensors are supported for binary cross entropy. "
                        "TODO: Consider implementing INT8 support in future versions.");
    }

    if (goal.shape() != pred.shape()) {
        throw TRException("[CpuBackend::bce_into] Shape mismatch: goal shape " +
                        goal.shape().to_string() + " != pred shape " +
                        pred.shape().to_string());
    }

    if (goal.shape() != result.shape()) {
        throw TRException("[CpuBackend::bce_into] Shape mismatch: goal shape " +
                        goal.shape().to_string() + " != result shape " +
                        result.shape().to_string());
    }

    const float* goal_data = static_cast<const float*>(goal.data_ptr());
    const float* pred_data = static_cast<const float*>(pred.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());
    size_t count = goal.numel();

    const float eps = 1e-8f;
    const float one_minus_eps = 1.0f - eps;

#ifdef TR_USE_EIGEN
    // 使用Eigen进行向量化计算
    Eigen::Map<const Eigen::VectorXf> goal_vec(goal_data, count);
    Eigen::Map<const Eigen::VectorXf> pred_vec(pred_data, count);
    Eigen::Map<Eigen::VectorXf> result_vec(result_data, count);

    // 首先对pred进行裁剪：clamp(pred, eps, 1-eps)
    Eigen::VectorXf pred_clamped = pred_vec.cwiseMax(eps).cwiseMin(one_minus_eps);

    // 计算二元交叉熵：-goal*log(pred_clamped) - (1-goal)*log(1-pred_clamped)
    result_vec = (-goal_vec.array() * pred_clamped.array().log()
                  - (1.0f - goal_vec.array()) * (1.0f - pred_clamped.array()).log()).matrix();
#else
    // 朴素实现
    std::transform(goal_data, goal_data + count, pred_data, result_data,
                   [eps, one_minus_eps](float goal_val, float pred_val) {
                       // 对pred进行裁剪
                       float pred_clamped = (pred_val < eps) ? eps :
                                          ((pred_val > one_minus_eps) ? one_minus_eps : pred_val);

                       // 计算二元交叉熵
                       return -goal_val * std::log(pred_clamped)
                              - (1.0f - goal_val) * std::log(1.0f - pred_clamped);
                   });
#endif
}

} // namespace tr
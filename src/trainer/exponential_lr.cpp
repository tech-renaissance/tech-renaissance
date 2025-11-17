/**
 * @file exponential_lr.cpp
 * @brief ExponentialLR学习率调度器实现
 * @details 实现指数衰减调度器的功能
 * @version 1.48.0
 * @date 2025-11-17
 * @author Tech Renaissance Team
 * @note 依赖项: exponential_lr.h
 * @note 所属系列: trainer
 */

#include "tech_renaissance/trainer/exponential_lr.h"
#include "tech_renaissance/utils/tr_exception.h"

namespace tr {

ExponentialLR::ExponentialLR(float initial_lr, float gamma)
    : Scheduler(initial_lr), gamma_(gamma) {
    // 验证参数有效性
    if (gamma <= 0.0f || gamma >= 1.0f) {
        throw TRException("ExponentialLR: gamma must be in (0, 1), got " +
                         std::to_string(gamma));
    }
}

float ExponentialLR::get_lr(int epoch) {
    if (epoch < 0) {
        throw TRException("ExponentialLR: epoch must be non-negative, got " +
                         std::to_string(epoch));
    }

    // lr = lr0 * gamma^epoch
    return initial_lr_ * std::pow(gamma_, epoch);
}

std::string ExponentialLR::type_name() const {
    return "ExponentialLR";
}

} // namespace tr
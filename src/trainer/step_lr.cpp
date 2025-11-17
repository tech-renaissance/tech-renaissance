/**
 * @file step_lr.cpp
 * @brief StepLR学习率调度器实现
 * @details 实现固定步长衰减调度器的功能
 * @version 1.48.0
 * @date 2025-11-17
 * @author Tech Renaissance Team
 * @note 依赖项: step_lr.h
 * @note 所属系列: trainer
 */

#include "tech_renaissance/trainer/step_lr.h"
#include "tech_renaissance/utils/tr_exception.h"

namespace tr {

StepLR::StepLR(float initial_lr, int step_size, float gamma)
    : Scheduler(initial_lr), step_size_(step_size), gamma_(gamma) {
    // 验证参数有效性
    if (step_size <= 0) {
        throw TRException("StepLR: step_size must be positive, got " +
                         std::to_string(step_size));
    }
    if (gamma <= 0.0f || gamma >= 1.0f) {
        throw TRException("StepLR: gamma must be in (0, 1), got " +
                         std::to_string(gamma));
    }
}

float StepLR::get_lr(int epoch) {
    if (epoch < 0) {
        throw TRException("StepLR: epoch must be non-negative, got " +
                         std::to_string(epoch));
    }

    // lr = lr0 * gamma^(epoch / step_size)
    // 整数除法自动取floor
    int num_steps = epoch / step_size_;
    return initial_lr_ * std::pow(gamma_, num_steps);
}

std::string StepLR::type_name() const {
    return "StepLR";
}

} // namespace tr
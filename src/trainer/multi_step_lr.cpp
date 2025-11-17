/**
 * @file multi_step_lr.cpp
 * @brief MultiStepLR学习率调度器实现
 * @details 实现多个里程碑衰减调度器的功能
 * @version 1.48.0
 * @date 2025-11-17
 * @author Tech Renaissance Team
 * @note 依赖项: multi_step_lr.h
 * @note 所属系列: trainer
 */

#include "tech_renaissance/trainer/multi_step_lr.h"
#include "tech_renaissance/utils/tr_exception.h"
#include <algorithm>

namespace tr {

MultiStepLR::MultiStepLR(float initial_lr, const std::vector<int>& milestones, float gamma)
    : Scheduler(initial_lr), milestones_(milestones), gamma_(gamma) {
    // 验证参数有效性
    if (milestones.empty()) {
        throw TRException("MultiStepLR: milestones cannot be empty");
    }
    if (gamma <= 0.0f || gamma >= 1.0f) {
        throw TRException("MultiStepLR: gamma must be in (0, 1), got " +
                         std::to_string(gamma));
    }

    // 验证milestones是否为正数且升序排列
    for (size_t i = 0; i < milestones.size(); ++i) {
        if (milestones[i] <= 0) {
            throw TRException("MultiStepLR: all milestones must be positive, got " +
                             std::to_string(milestones[i]));
        }
        if (i > 0 && milestones[i] <= milestones[i-1]) {
            throw TRException("MultiStepLR: milestones must be in strictly increasing order");
        }
    }
}

float MultiStepLR::get_lr(int epoch) {
    if (epoch < 0) {
        throw TRException("MultiStepLR: epoch must be non-negative, got " +
                         std::to_string(epoch));
    }

    // 统计当前epoch超过的milestone数量
    int decay_count = 0;
    for (int milestone : milestones_) {
        if (epoch >= milestone) {
            decay_count++;
        } else {
            break;  // milestones是有序的，可以提前退出
        }
    }

    // lr = lr0 * gamma^decay_count
    return initial_lr_ * std::pow(gamma_, decay_count);
}

std::string MultiStepLR::type_name() const {
    return "MultiStepLR";
}

} // namespace tr
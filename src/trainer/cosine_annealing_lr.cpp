/**
 * @file cosine_annealing_lr.cpp
 * @brief CosineAnnealingLR学习率调度器实现
 * @details 实现余弦退火调度器的功能
 * @version 1.48.0
 * @date 2025-11-17
 * @author Tech Renaissance Team
 * @note 依赖项: cosine_annealing_lr.h
 * @note 所属系列: trainer
 */

#include "tech_renaissance/trainer/cosine_annealing_lr.h"
#include "tech_renaissance/utils/tr_exception.h"

namespace tr {

CosineAnnealingLR::CosineAnnealingLR(float initial_lr, int T_max, float eta_min)
    : Scheduler(initial_lr), T_max_(T_max), eta_min_(eta_min) {
    // 验证参数有效性
    if (T_max <= 0) {
        throw TRException("CosineAnnealingLR: T_max must be positive, got " +
                         std::to_string(T_max));
    }
    if (eta_min < 0.0f) {
        throw TRException("CosineAnnealingLR: eta_min must be non-negative, got " +
                         std::to_string(eta_min));
    }
    if (initial_lr <= eta_min) {
        throw TRException("CosineAnnealingLR: initial_lr must be greater than eta_min");
    }
}

float CosineAnnealingLR::get_lr(int epoch) {
    if (epoch < 0) {
        throw TRException("CosineAnnealingLR: epoch must be non-negative, got " +
                         std::to_string(epoch));
    }

    // lr = eta_min + (lr0 - eta_min) * (1 + cos(π * epoch / T_max)) / 2
    double progress = static_cast<double>(epoch) / T_max_;
    double cosine_value = std::cos(M_PI * progress);

    return eta_min_ + (initial_lr_ - eta_min_) * (1.0 + cosine_value) / 2.0;
}

std::string CosineAnnealingLR::type_name() const {
    return "CosineAnnealingLR";
}

} // namespace tr
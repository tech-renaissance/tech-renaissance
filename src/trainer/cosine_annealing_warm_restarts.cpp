/**
 * @file cosine_annealing_warm_restarts.cpp
 * @brief CosineAnnealingWarmRestarts学习率调度器实现
 * @details 实现带热重启的余弦退火调度器的功能
 * @version 1.48.0
 * @date 2025-11-17
 * @author Tech Renaissance Team
 * @note 依赖项: cosine_annealing_warm_restarts.h
 * @note 所属系列: trainer
 */

#include "tech_renaissance/trainer/cosine_annealing_warm_restarts.h"
#include "tech_renaissance/utils/tr_exception.h"

namespace tr {

CosineAnnealingWarmRestarts::CosineAnnealingWarmRestarts(float initial_lr, int T_0, int T_mult, float eta_min)
    : Scheduler(initial_lr), T_0_(T_0), T_mult_(T_mult), eta_min_(eta_min) {
    // 验证参数有效性
    if (T_0 <= 0) {
        throw TRException("CosineAnnealingWarmRestarts: T_0 must be positive, got " +
                         std::to_string(T_0));
    }
    if (T_mult <= 0) {
        throw TRException("CosineAnnealingWarmRestarts: T_mult must be positive, got " +
                         std::to_string(T_mult));
    }
    if (eta_min < 0.0f) {
        throw TRException("CosineAnnealingWarmRestarts: eta_min must be non-negative, got " +
                         std::to_string(eta_min));
    }
    if (initial_lr <= eta_min) {
        throw TRException("CosineAnnealingWarmRestarts: initial_lr must be greater than eta_min");
    }
}

float CosineAnnealingWarmRestarts::get_lr(int epoch) {
    if (epoch < 0) {
        throw TRException("CosineAnnealingWarmRestarts: epoch must be non-negative, got " +
                         std::to_string(epoch));
    }

    // 计算当前在哪个重启周期，以及周期内位置
    int T_cur = T_0_;          // 当前周期长度
    int epoch_in_period = epoch;

    // 找到当前周期
    while (epoch_in_period >= T_cur) {
        epoch_in_period -= T_cur;
        T_cur *= T_mult_;

        // 防止整数溢出
        if (T_cur <= 0) {
            throw TRException("CosineAnnealingWarmRestarts: Period length overflow");
        }
    }

    // 余弦退火公式
    if (T_cur == 0) {
        return initial_lr_;  // 边界情况保护
    }

    double progress = static_cast<double>(epoch_in_period) / static_cast<double>(T_cur);
    double cosine_value = std::cos(M_PI * progress);

    return eta_min_ + (initial_lr_ - eta_min_) * (1.0 + cosine_value) / 2.0;
}

std::string CosineAnnealingWarmRestarts::type_name() const {
    return "CosineAnnealingWarmRestarts";
}

} // namespace tr
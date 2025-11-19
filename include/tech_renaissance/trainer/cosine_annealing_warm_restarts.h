/**
 * @file cosine_annealing_warm_restarts.h
 * @brief CosineAnnealingWarmRestarts学习率调度器
 * @details 带热重启的余弦退火调度器，支持周期长度递增的余弦退火
 * @version 1.48.0
 * @date 2025-11-17
 * @author Tech Renaissance Team
 * @note 依赖项: scheduler.h
 * @note 所属系列: trainer
 */

#pragma once

#include "tech_renaissance/trainer/scheduler.h"
#include <cmath>

namespace tr {

/**
 * @brief CosineAnnealingWarmRestarts学习率调度器
 * @details 带热重启的余弦退火调度器，支持周期长度递增
 *
 * 公式：动态计算当前周期长度T_i和位置t_cur
 *       lr = eta_min + (lr0 - eta_min) * (1 + cos(π * t_cur / T_i)) / 2
 *
 * 使用示例：
 * ```cpp
 * CosineAnnealingWarmRestarts scheduler(0.1f, 10, 2, 0.0f);  // 初始周期10，每次倍增
 * float lr = scheduler.get_lr(epoch);
 * ```
 */
class CosineAnnealingWarmRestarts : public Scheduler {
public:
    /**
     * @brief 构造函数
     * @param initial_lr 初始学习率（最大值）
     * @param T_0 初始周期长度（epoch数）
     * @param T_mult 周期放大因子，默认为1
     * @param eta_min 最小学习率，默认为0
     */
    CosineAnnealingWarmRestarts(float initial_lr, int T_0, int T_mult = 1, float eta_min = 0.0f);

    /**
     * @brief 获取指定epoch的学习率
     * @param epoch 当前epoch数（从0开始）
     * @return 学习率
     */
    float get_lr(int epoch) override;

    /**
     * @brief 获取调度器类型名称
     * @return 调度器类型名称
     */
    std::string type_name() const override;

private:
    static constexpr double M_PI = 3.14159265358979323846;  ///< π常量

    int T_0_;       ///< 初始周期长度（epoch数）
    int T_mult_;     ///< 周期放大因子
    float eta_min_;  ///< 最小学习率
};

} // namespace tr
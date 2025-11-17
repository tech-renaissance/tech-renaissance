/**
 * @file cosine_annealing_lr.h
 * @brief CosineAnnealingLR学习率调度器
 * @details 余弦退火调度器，使用余弦函数从初始学习率平滑下降到最小学习率
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
 * @brief CosineAnnealingLR学习率调度器
 * @details 使用余弦函数从初始学习率平滑下降到最小学习率，周期长度为T_max
 *
 * 公式：lr = eta_min + (lr0 - eta_min) * (1 + cos(π * epoch / T_max)) / 2
 *
 * 使用示例：
 * ```cpp
 * CosineAnnealingLR scheduler(0.1f, 100, 0.0f);  // 初始0.1，周期100，最小0.0
 * float lr = scheduler.get_lr(epoch);
 * ```
 */
class CosineAnnealingLR : public Scheduler {
public:
    /**
     * @brief 构造函数
     * @param initial_lr 初始学习率（最大值）
     * @param T_max 半周期长度（epoch数）
     * @param eta_min 最小学习率，默认为0
     */
    CosineAnnealingLR(float initial_lr, int T_max, float eta_min = 0.0f);

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

    int T_max_;      ///< 半周期长度（epoch数）
    float eta_min_;   ///< 最小学习率
};

} // namespace tr
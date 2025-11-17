/**
 * @file step_lr.h
 * @brief StepLR学习率调度器
 * @details 固定步长衰减调度器，每隔指定epoch数按gamma因子衰减学习率
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
 * @brief StepLR学习率调度器
 * @details 每隔step_size个epoch将学习率乘以gamma因子衰减
 *
 * 公式：lr = lr0 * gamma^(epoch / step_size)
 *
 * 使用示例：
 * ```cpp
 * StepLR scheduler(0.1f, 30, 0.1f);  // 初始0.1，每30个epoch乘以0.1
 * float lr = scheduler.get_lr(epoch);
 * ```
 */
class StepLR : public Scheduler {
public:
    /**
     * @brief 构造函数
     * @param initial_lr 初始学习率
     * @param step_size 衰减间隔epoch数
     * @param gamma 衰减系数，范围(0, 1)
     */
    StepLR(float initial_lr, int step_size, float gamma);

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
    int step_size_;    ///< 衰减间隔epoch数
    float gamma_;      ///< 衰减系数
};

} // namespace tr
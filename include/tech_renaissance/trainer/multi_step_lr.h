/**
 * @file multi_step_lr.h
 * @brief MultiStepLR学习率调度器
 * @details 多个里程碑衰减调度器，在指定epoch时按gamma因子衰减学习率
 * @version 1.48.0
 * @date 2025-11-17
 * @author Tech Renaissance Team
 * @note 依赖项: scheduler.h
 * @note 所属系列: trainer
 */

#pragma once

#include "tech_renaissance/trainer/scheduler.h"
#include <vector>
#include <cmath>

namespace tr {

/**
 * @brief MultiStepLR学习率调度器
 * @details 在指定的milestones epoch时将学习率乘以gamma因子衰减
 *
 * 公式：lr = lr0 * gamma^(count of milestones that epoch >= milestone)
 *
 * 使用示例：
 * ```cpp
 * MultiStepLR scheduler(0.1f, {30, 80, 120}, 0.1f);  // 在epoch 30, 80, 120时衰减
 * float lr = scheduler.get_lr(epoch);
 * ```
 */
class MultiStepLR : public Scheduler {
public:
    /**
     * @brief 构造函数
     * @param initial_lr 初始学习率
     * @param milestones 衰减epoch的列表（需升序排列）
     * @param gamma 衰减系数，范围(0, 1)
     */
    MultiStepLR(float initial_lr, const std::vector<int>& milestones, float gamma);

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
    std::vector<int> milestones_;  ///< 衰减epoch的列表
    float gamma_;                 ///< 衰减系数
};

} // namespace tr
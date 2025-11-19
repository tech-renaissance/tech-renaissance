/**
 * @file exponential_lr.h
 * @brief ExponentialLR学习率调度器
 * @details 指数衰减调度器，每个epoch都将学习率乘以gamma因子衰减
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
 * @brief ExponentialLR学习率调度器
 * @details 每个epoch都将学习率乘以gamma因子衰减
 *
 * 公式：lr = lr0 * gamma^epoch
 *
 * 使用示例：
 * ```cpp
 * ExponentialLR scheduler(0.1f, 0.95f);  // 初始0.1，每个epoch乘以0.95
 * float lr = scheduler.get_lr(epoch);
 * ```
 */
class ExponentialLR : public Scheduler {
public:
    /**
     * @brief 构造函数
     * @param initial_lr 初始学习率
     * @param gamma 每个epoch的衰减系数，范围(0, 1)
     */
    ExponentialLR(float initial_lr, float gamma);

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
    float gamma_;  ///< 每个epoch的衰减系数
};

} // namespace tr
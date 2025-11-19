/**
 * @file constant_lr.h
 * @brief ConstantLR学习率调度器
 * @details 常数学习率调度器，始终返回固定的学习率，用于调试和基准测试
 * @version 1.48.0
 * @date 2025-11-17
 * @author Tech Renaissance Team
 * @note 依赖项: scheduler.h
 * @note 所属系列: trainer
 */

#pragma once

#include "tech_renaissance/trainer/scheduler.h"

namespace tr {

/**
 * @brief ConstantLR学习率调度器
 * @details 常数学习率调度器，始终返回初始学习率，用于调试和基准测试
 *
 * 公式：lr = lr0（恒定）
 *
 * 使用示例：
 * ```cpp
 * ConstantLR scheduler(0.01f);  // 始终返回0.01
 * float lr = scheduler.get_lr(epoch);  // 总是0.01
 * ```
 */
class ConstantLR : public Scheduler {
public:
    /**
     * @brief 构造函数
     * @param initial_lr 恒定的学习率
     */
    explicit ConstantLR(float initial_lr);

    /**
     * @brief 获取指定epoch的学习率
     * @param epoch 当前epoch数（从0开始）
     * @return 学习率（始终为初始学习率）
     */
    float get_lr(int epoch) override;

    /**
     * @brief 获取调度器类型名称
     * @return 调度器类型名称
     */
    std::string type_name() const override;
};

} // namespace tr
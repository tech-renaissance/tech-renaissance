/**
 * @file scheduler.h
 * @brief 学习率调度器基类
 * @details 定义学习率调度的抽象接口，支持基于epoch的学习率调整策略
 * @version 1.48.0
 * @date 2025-11-17
 * @author Tech Renaissance Team
 * @note 依赖项: 无
 * @note 所属系列: trainer
 */

#pragma once

#include <string>

namespace tr {

/**
 * @brief 学习率调度器基类
 * @details 定义学习率调度的抽象接口，支持基于epoch的学习率调整策略
 *
 * 设计特点：
 * - 抽象基类：所有具体调度策略都应继承此类
 * - 简洁接口：只提供必要的get_lr()方法
 * - 灵活扩展：子类可实现任意学习率调度策略
 * - 无状态依赖：子类可自行管理内部状态
 *
 * 使用示例：
 * ```cpp
 * class StepLR : public Scheduler {
 * public:
 *     StepLR(float initial_lr, int step_size, float gamma)
 *         : Scheduler(initial_lr), step_size_(step_size), gamma_(gamma) {}
 *
 *     float get_lr(int epoch) override {
 *         return initial_lr_ * std::pow(gamma_, epoch / step_size_);
 *     }
 * private:
 *     int step_size_;
 *     float gamma_;
 * };
 * ```
 */
class Scheduler {
public:
    /**
     * @brief 构造函数
     * @param initial_lr 初始学习率
     */
    explicit Scheduler(float initial_lr);

    /**
     * @brief 虚析构函数
     */
    virtual ~Scheduler() = default;

    /**
     * @brief 获取指定epoch的学习率
     * @param epoch 当前的epoch数，从0开始
     * @return 学习率值
     *
     * 子类必须实现此方法来定义具体的学习率调度策略
     */
    virtual float get_lr(int epoch) = 0;

    /**
     * @brief 获取初始学习率
     * @return 初始学习率
     */
    float get_initial_lr() const { return initial_lr_; }

    /**
     * @brief 获取调度器类型名称
     * @return 调度器的类型名称，子类应重写此方法
     */
    virtual std::string type_name() const = 0;

protected:
    float initial_lr_;  ///< 初始学习率
};

} // namespace tr
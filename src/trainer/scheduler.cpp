/**
 * @file scheduler.cpp
 * @brief 学习率调度器基类实现
 * @details 实现学习率调度器的基础功能
 * @version 1.48.0
 * @date 2025-11-17
 * @author Tech Renaissance Team
 * @note 依赖项: trainer/scheduler.h
 * @note 所属系列: trainer
 */

#include "tech_renaissance/trainer/scheduler.h"
#include "tech_renaissance/utils/tr_exception.h"

namespace tr {

// 构造函数
Scheduler::Scheduler(float initial_lr) : initial_lr_(initial_lr) {
    // 验证初始学习率的有效性
    if (initial_lr <= 0.0f) {
        throw TRException("Scheduler: initial_lr must be positive, got " +
                         std::to_string(initial_lr));
    }
}

} // namespace tr
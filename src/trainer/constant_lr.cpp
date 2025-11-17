/**
 * @file constant_lr.cpp
 * @brief ConstantLR学习率调度器实现
 * @details 实现常数学习率调度器的功能
 * @version 1.48.0
 * @date 2025-11-17
 * @author Tech Renaissance Team
 * @note 依赖项: constant_lr.h
 * @note 所属系列: trainer
 */

#include "tech_renaissance/trainer/constant_lr.h"
#include "tech_renaissance/utils/tr_exception.h"

namespace tr {

ConstantLR::ConstantLR(float initial_lr) : Scheduler(initial_lr) {
    // 基类构造函数已经验证了initial_lr > 0，这里无需额外验证
}

float ConstantLR::get_lr(int epoch) {
    if (epoch < 0) {
        throw TRException("ConstantLR: epoch must be non-negative, got " +
                         std::to_string(epoch));
    }

    // 始终返回初始学习率
    return initial_lr_;
}

std::string ConstantLR::type_name() const {
    return "ConstantLR";
}

} // namespace tr
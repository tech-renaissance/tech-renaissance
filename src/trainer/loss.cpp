/**
 * @file loss.cpp
 * @brief 损失函数基类实现
 * @details 所有损失函数的抽象基类，提供统一的损失计算接口和模式管理
 * @version 1.48.0
 * @date 2025年11月17日
 * @author 技术觉醒团队
 * @note 依赖项: loss.h
 * @note 所属系列: trainer
 */

#include "tech_renaissance/trainer/loss.h"

namespace tr {

Loss::Loss(bool training_mode)
    : training_mode_(training_mode), backend_(nullptr) {
}

} // namespace tr
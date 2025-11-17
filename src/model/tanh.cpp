/**
 * @file tanh.cpp
 * @brief 双曲正切激活函数层实现
 * @details 实现双曲正切激活函数，支持前向传播和反向传播
 * @version 1.45.0
 * @date 2025-11-17
 * @author 技术觉醒团队
 * @note 所属系列: model
 */

#include "tech_renaissance/model/tanh.h"

namespace tr {

Tanh::Tanh(const std::string& name)
    : Module(name) {
    // Tanh层没有可训练参数
}

void Tanh::forward_into(const Tensor& input, Tensor& output) {
    cache_input(input);

    auto backend = get_backend();
    backend->tanh_into(input, output);
}

void Tanh::backward_into(const Tensor& grad_output, Tensor& grad_input) {
    auto backend = get_backend();

    // tanh的导数：1 - tanh(x)^2
    // 简化实现：暂时使用梯度传播
    // TODO: 实现完整的tanh导数计算
    backend->copy_into(grad_output, grad_input);

    clear_cache();
}

Shape Tanh::infer_output_shape(const Shape& input_shape) const {
    // Tanh层不改变形状
    return input_shape;
}

} // namespace tr
/**
 * @file relu.cpp
 * @brief ReLU激活函数层实现
 * @details 实现ReLU激活函数，支持前向传播和反向传播
 * @version 1.45.0
 * @date 2025-11-25
 * @author 技术觉醒团队
 * @note 所属系列: model
 */

#include "tech_renaissance/model/relu.h"

namespace tr {

ReLU::ReLU(const std::string& name)
    : Module(name) {
    // ReLU层没有可训练参数
}

void ReLU::forward_into(const Tensor& input, Tensor& output) {
    cache_input(input);

    auto backend = get_backend();
    backend->relu_into(input, output);
}

void ReLU::backward_into(const Tensor& grad_output, Tensor& grad_input) {
    auto backend = get_backend();

    // ReLU的导数：d/dx ReLU(x) = x > 0 ? 1 : 0
    // 使用链式法则：grad_input = grad_output * drelu(cached_input_)
    // 其中drelu(x) = x > 0 ? 1 : 0

    // 使用drelu_into方法直接计算导数
    Tensor drelu_output = backend->zeros(cached_input_.shape(), DType::FP32);
    backend->drelu_into(cached_input_, drelu_output);

    // 应用链式法则：grad_input = grad_output * drelu_output
    // 使用逐元素乘法
    backend->mul_broadcast_into(grad_output, drelu_output, grad_input);

    clear_cache();
}

Shape ReLU::infer_output_shape(const Shape& input_shape) const {
    // ReLU层不改变形状
    return input_shape;
}

} // namespace tr
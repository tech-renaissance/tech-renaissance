/**
 * @file dropout.cpp
 * @brief Dropout层实现
 * @details 实现Dropout正则化层，支持训练和推理两种模式
 * @version 1.45.0
 * @date 2025-11-25
 * @author 技术觉醒团队
 * @note 所属系列: model
 */

#include "tech_renaissance/model/dropout.h"

namespace tr {

Dropout::Dropout(float p, const std::string& name)
    : Module(name), p_(p), training_(true) {
    if (p_ < 1e-8 || p_ > 1.0f) {
        throw ValueError("[Dropout] Dropout probability must be between 0.0 and 1.0, got: " + std::to_string(p_));
    }
}

void Dropout::forward_into(const Tensor& input, Tensor& output) {
    auto backend = get_backend();
    cache_input(input);
    if (training_) {
        if (mask_.shape() != input.shape()) {
            mask_ = get_backend()->zeros(input.shape(), DType::FP32);
        }
        backend->dropout_into(input, mask_, output, p_);
    } else {
        backend->copy_into(input, output);
    }
}

void Dropout::backward_into(const Tensor& grad_output, Tensor& grad_input) {
    auto backend = get_backend();
    backend->ddropout_into(grad_output, mask_, grad_input, p_);
    clear_cache();
}

void Dropout::set_dropout_probability(float p) {
    if (p < 0.0f || p > 1.0f) {
        throw ValueError("[Dropout::set_dropout_probability] Dropout probability must be between 0.0 and 1.0, got: " + std::to_string(p));
    }
    p_ = p;
}

Shape Dropout::infer_output_shape(const Shape& input_shape) const {
    // Dropout层不改变形状
    return input_shape;
}

} // namespace tr
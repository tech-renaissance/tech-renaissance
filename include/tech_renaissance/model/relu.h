/**
 * @file relu.h
 * @brief ReLU激活函数层
 * @details 实现ReLU激活函数，支持前向传播和反向传播
 * @version 1.45.0
 * @date 2025-11-25
 * @author 技术觉醒团队
 * @note 依赖项: module.h
 * @note 所属系列: model
 */

#pragma once

#include "tech_renaissance/model/module.h"

namespace tr {

class ReLU : public Module {
public:
    ReLU(const std::string& name = "ReLU");
    ~ReLU() = default;

    // === 核心计算（into型） ===
    void forward_into(const Tensor& input, Tensor& output) override;
    void backward_into(const Tensor& grad_output, Tensor& grad_input) override;

protected:
    Shape infer_output_shape(const Shape& input_shape) const override;
};

} // namespace tr
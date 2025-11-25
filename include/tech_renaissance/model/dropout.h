/**
 * @file dropout.h
 * @brief Dropout层
 * @details 实现Dropout正则化层，支持训练和推理两种模式
 * @version 1.45.0
 * @date 2025-11-25
 * @author 技术觉醒团队
 * @note 依赖项: module.h
 * @note 所属系列: model
 */

#pragma once

#include "tech_renaissance/model/module.h"

namespace tr {

class Dropout : public Module {
public:
    /**
     * @brief Dropout层构造函数
     * @param p dropout概率，默认为0.5（即50%的神经元被丢弃）
     * @param name 模块名称，默认为"Dropout"
     */
    Dropout(float p = 0.5f, const std::string& name = "Dropout");
    ~Dropout() = default;

    // === 核心计算（into型） ===
    void forward_into(const Tensor& input, Tensor& output) override;
    void backward_into(const Tensor& grad_output, Tensor& grad_input) override;

    /**
     * @brief 获取dropout概率
     * @return dropout概率值
     */
    float get_dropout_probability() const { return p_; }

    /**
     * @brief 设置dropout概率
     * @param p 新的dropout概率值（0.0到1.0之间）
     */
    void set_dropout_probability(float p);

protected:
    Shape infer_output_shape(const Shape& input_shape) const override;

private:
    float p_;                    // dropout概率
    bool training_;              // 是否为训练模式
    Tensor mask_;                // dropout mask，用于反向传播
    static constexpr float DEFAULT_SEED = 42;  // 默认随机种子
};

} // namespace tr
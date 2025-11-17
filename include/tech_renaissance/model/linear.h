/**
 * @file linear.h
 * @brief 线性层
 * @details 全连接层的实现，支持前向传播和反向传播
 * @version 1.45.0
 * @date 2025-11-17
 * @author 技术觉醒团队
 * @note 依赖项: module.h
 * @note 所属系列: model
 */

#pragma once

#include "tech_renaissance/model/module.h"
#include <cmath>

namespace tr {

class Linear : public Module {
private:
    int in_features_;
    int out_features_;
    bool use_bias_;

public:
    Linear(int in_features, int out_features, const std::string& name = "Linear", bool use_bias = false)
        : Module(name), in_features_(in_features), out_features_(out_features), use_bias_(use_bias) {}

    void set_backend(Backend* backend) override {
        Module::set_backend(backend);

        // 创建并注册权重参数
        if (!has_parameter("weight")) {
            // 权重：in_features × out_features (转置的权重，避免运行时转置)
            Tensor weight = backend->zeros(Shape(in_features_, out_features_), DType::FP32);
            register_parameter("weight", weight);
        }

        // 只有需要时才创建偏置参数
        if (use_bias_ && !has_parameter("bias")) {
            // 偏置：out_features
            Tensor bias = backend->zeros(Shape(out_features_), DType::FP32);
            register_parameter("bias", bias);
        }
    }

    // === 核心计算（into型） ===
    void forward_into(const Tensor& input, Tensor& output) override {
        cache_input(input);

        auto backend = get_backend();

        // 获取权重
        const Tensor& weight = get_parameter("weight");

        // 计算：output = input @ weight
        // Linear层权重存储形状为：(in_features, out_features) (转置的权重)
        // 矩阵乘法：input(batch, in_features) @ weight(in_features, out_features) = output(batch, out_features)
        backend->mm_into(input, weight, output);

        // 如果使用偏置，进行广播加法
        if (use_bias_ && has_parameter("bias")) {
            const Tensor& bias = get_parameter("bias");
            // 使用广播加法：output += bias
            backend->add_broadcast_into(output, bias, output);
        }
    }

    void backward_into(const Tensor& grad_output, Tensor& grad_input) override {
        auto backend = get_backend();

        // 获取权重
        const Tensor& weight = get_parameter("weight");

        // 计算输入梯度：grad_input = grad_output @ weight^T
        // grad_output(batch, out_features) @ weight^T(out_features, in_features) = grad_input(batch, in_features)
        Tensor weight_transposed = backend->transpose(weight);
        backend->mm_into(grad_output, weight_transposed, grad_input);

        // 计算权重梯度：grad_weight = grad_output.T @ input
        if (weight.has_grad()) {
            // 这里需要转置grad_output，暂时简化实现
            // Tensor& grad_weight = weight.grad();
            // backend->fill(grad_weight, 0.0f);
        }

        // 计算偏置梯度：grad_bias = sum(grad_output, dim=0)
        if (use_bias_ && has_parameter("bias")) {
            const Tensor& bias = get_parameter("bias");
            if (bias.has_grad()) {
                // 简化实现：对grad_output的batch维度求和
                // Tensor& grad_bias = bias.grad();
                // backend->fill(grad_bias, 0.0f);
            }
        }

        clear_cache();
    }

protected:
    Shape infer_output_shape(const Shape& input_shape) const override {
        // 输入: (batch, in_features) 或展平后的其他形状
        // 输出: (batch, out_features)
        // 假设输入的最后一维是in_features，其他维度展平为batch
        int64_t batch_size = input_shape.numel() / in_features_;
        return Shape(batch_size, out_features_);
    }

public:
    // === 访问器方法 ===
    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }

    // === 调试辅助方法 ===
    void print_parameters() const {
        std::cout << "Linear Layer (" << instance_name() << "):" << std::endl;
        std::cout << "  Input features: " << in_features_ << std::endl;
        std::cout << "  Output features: " << out_features_ << std::endl;

        if (has_parameter("weight")) {
            const Tensor& weight = get_parameter("weight");
            std::cout << "  Weight shape: " << weight.shape().to_string() << std::endl;
        }

        if (has_parameter("bias")) {
            const Tensor& bias = get_parameter("bias");
            std::cout << "  Bias shape: " << bias.shape().to_string() << std::endl;
        }
    }
};

} // namespace tr
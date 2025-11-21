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

    // 权重转置缓存
    mutable Tensor weight_transposed_;      // 缓存的转置权重
    mutable bool weight_transposed_valid_ = false;
    mutable bool weight_dirty_ = false;     // ✅ 新增：权重脏标记

public:
    Linear(int in_features, int out_features, const std::string& name = "Linear", bool use_bias = false)
        : Module(name), in_features_(in_features), out_features_(out_features), use_bias_(use_bias) {}

    void set_backend(std::shared_ptr<Backend> backend) override {
        Module::set_backend(backend);

        // 创建并注册权重参数
        if (!has_parameter("weight")) {
            // 权重：out_features × in_features (PyTorch标准格式)
            // 🔧 对齐PyTorch Linear层默认初始化：Kaiming Uniform初始化
            // PyTorch Linear层默认使用: uniform(-k, k) where k = sqrt(1/fan_in)
            float k = std::sqrt(1.0f / in_features_);  // Kaiming Uniform的边界值
            Tensor weight = backend->uniform(Shape(out_features_, in_features_), -k, k);
            register_parameter("weight", weight);

            // ✅ 启用梯度：为权重参数创建梯度张量
            Tensor weight_grad = backend->zeros(weight.shape(), DType::FP32);
            weight.set_grad(weight_grad);
        }

        // 只有需要时才创建偏置参数
        if (use_bias_ && !has_parameter("bias")) {
            // 偏置：(1, out_features) - 2D形状以便于广播
            // 🔧 对齐PyTorch偏置初始化：同样使用uniform(-k, k) where k = sqrt(1/fan_in)
            float k = std::sqrt(1.0f / in_features_);  // 与权重相同的边界值
            Tensor bias = backend->uniform(Shape(1, out_features_), -k, k);
            register_parameter("bias", bias);

            // ✅ 启用梯度：为偏置参数创建梯度张量
            Tensor bias_grad = backend->zeros(bias.shape(), DType::FP32);
            bias.set_grad(bias_grad);
        }

        // 初始化转置缓存（在权重创建之后）
        invalidate_weight_cache();
        weight_dirty_ = false;  // ✅ 确保初始状态正确
    }

    // === 核心计算（into型） ===
    void forward_into(const Tensor& input, Tensor& output) override {
        cache_input(input);

        auto backend = get_backend();

        // 获取权重
        const Tensor& weight = get_parameter("weight");

        // ✅ 只在权重被修改后才重新转置
        if (weight_dirty_) {
            invalidate_weight_cache();
            weight_dirty_ = false;
        }

        // 确保转置权重缓存有效
        if (!weight_transposed_valid_) {
            // 预计算并缓存转置权重：weight^T (in_features, out_features)
            weight_transposed_ = backend->transpose(weight);
            weight_transposed_valid_ = true;
        }

        // 计算：output = input @ weight^T
        // Linear层权重存储为：(out_features, in_features) (PyTorch标准格式)
        // 缓存的转置权重为：(in_features, out_features)
        // 矩阵乘法：input(batch, in_features) @ weight^T(in_features, out_features) = output(batch, out_features)
        // ⭐ 使用缓存的转置权重，避免运行时转置开销
        backend->mm_into(input, weight_transposed_, output);

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
        Tensor& weight = get_parameter("weight");

        // 计算输入梯度：grad_input = grad_output @ weight^T
        // grad_output(batch, out_features) @ weight^T(in_features, out_features)^T = grad_input(batch, in_features)
        backend->mm_into(grad_output, weight, grad_input);

        // 计算权重梯度：grad_weight = grad_output^T @ input
        if (weight.has_grad()) {
            // ⭐ 使用mm_into_transposed，避免临时转置张量
            // grad_output^T @ input = grad_weight (transpose_a=true)
            Shape grad_weight_shape(weight.shape());
            Tensor grad_weight = backend->zeros(grad_weight_shape, DType::FP32);
            backend->mm_into_transposed(grad_output, cached_input_, grad_weight, true, false);

            // 累积权重梯度
            if (!weight.grad().storage_allocated()) {
                weight.set_grad(grad_weight);
            } else {
                // 实现梯度累积：new_grad += old_grad
                Tensor& existing_grad = weight.grad();
                backend->add_into(grad_weight, existing_grad, existing_grad);
            }
        }

        // 计算偏置梯度：grad_bias = sum(grad_output, dim=0)
        if (use_bias_ && has_parameter("bias")) {
            Tensor& bias = get_parameter("bias");
            if (bias.has_grad()) {
                // 对grad_output的batch维度求和：grad_bias(out_features)
                Tensor grad_bias = backend->zeros(bias.shape(), DType::FP32);

                // 使用sum_into方法对dim=0进行求和
                backend->sum_into(grad_output, grad_bias, 0, false);

                // 累积偏置梯度
                if (!bias.grad().storage_allocated()) {
                    bias.set_grad(grad_bias);
                } else {
                    // 实现梯度累积：new_grad += old_grad
                    Tensor& existing_grad = bias.grad();
                    backend->add_into(grad_bias, existing_grad, existing_grad);
                }
            }
        }

        clear_cache();

    weight_dirty_ = true;  // ✅ 标记权重将被更新，而非立即失效缓存
    // 移除 invalidate_weight_cache();
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
    // === 设备转移 ===
    void to(const Device& device) override {
        // 调用基类方法
        Module::to(device);

        // 设备转移后，转置缓存失效
        invalidate_weight_cache();
    }

    // === 缓存管理 ===
    void invalidate_weight_cache() const {
        auto backend = get_backend();
        if (backend && has_parameter("weight")) {
            const Tensor& weight = get_parameter("weight");
            // 预分配转置权重缓存
            weight_transposed_ = backend->zeros(Shape(in_features_, out_features_), weight.dtype());
        }
        weight_transposed_valid_ = false;
        weight_dirty_ = false;  // ✅ 重置脏标记
    }

    // === 访问器方法 ===
    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }

    // === 调试辅助方法 ===
    void print_parameters() const {
        std::cout << "Linear Layer (" << instance_name() << "):" << std::endl;
        std::cout << "  Input features: " << in_features_ << std::endl;
        std::cout << "  Output features: " << out_features_ << std::endl;
        std::cout << "  Weight transposed cache: " << (weight_transposed_valid_ ? "VALID [OK]" : "INVALID [FAIL]") << std::endl;

        if (has_parameter("weight")) {
            const Tensor& weight = get_parameter("weight");
            std::cout << "  Weight shape: " << weight.shape().to_string() << " (PyTorch standard: out_features, in_features)" << std::endl;
        }

        if (has_parameter("bias")) {
            const Tensor& bias = get_parameter("bias");
            std::cout << "  Bias shape: " << bias.shape().to_string() << std::endl;
        }
    }
};

} // namespace tr
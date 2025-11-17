/**
 * @file flatten.h
 * @brief 展平层
 * @details 将多维张量展平为2D张量，保持batch维度不变
 * @version 1.45.0
 * @date 2025-11-17
 * @author 技术觉醒团队
 * @note 依赖项: module.h
 * @note 所属系列: model
 */

#pragma once

#include "tech_renaissance/model/module.h"

namespace tr {

class Flatten : public Module {
private:
    int start_dim_;    // 开始展平的维度
    int end_dim_;      // 结束展平的维度

public:
    // 简化的无参构造函数
    Flatten() : Module("Flatten"), start_dim_(1), end_dim_(-1) {}

    // 完整构造函数（保持向后兼容）
    Flatten(int start_dim, int end_dim, const std::string& name)
        : Module(name), start_dim_(start_dim), end_dim_(end_dim) {}

    // 兼容性构造函数（只有start_dim和end_dim）
    Flatten(int start_dim, int end_dim)
        : Module("Flatten"), start_dim_(start_dim), end_dim_(end_dim) {}

    void set_backend(std::shared_ptr<Backend> backend) override {
        Module::set_backend(backend);
        // Flatten层没有可训练参数，不需要初始化
    }

    // === 核心计算（into型） ===
    void forward_into(const Tensor& input, Tensor& output) override {
        cache_input(input);

        // 使用view创建展平的视图
        Shape flattened_shape = calculate_flattened_shape(input.shape());
        auto backend = get_backend();
        Tensor flattened_view = backend->view(input, flattened_shape);

        // 将视图数据复制到输出张量
        backend->copy_into(flattened_view, output);
    }

    void backward_into(const Tensor& grad_output, Tensor& grad_input) override {
        // 反向传播：将梯度重新整形为原始输入形状
        auto backend = get_backend();
        Tensor reshaped_grad = backend->view(grad_output, cached_input_.shape());
        backend->copy_into(reshaped_grad, grad_input);

        clear_cache();
    }

protected:
    Shape infer_output_shape(const Shape& input_shape) const override {
        return calculate_flattened_shape(input_shape);
    }

    Shape calculate_flattened_shape(const Shape& input_shape) const {
        int32_t ndim = input_shape.ndim();

        // 处理负数end_dim
        int actual_end_dim = (end_dim_ < 0) ? ndim + end_dim_ : end_dim_;

        // 确保维度范围有效
        if (start_dim_ < 0 || start_dim_ >= ndim ||
            actual_end_dim < 0 || actual_end_dim >= ndim ||
            start_dim_ > actual_end_dim) {
            throw TRException("[Flatten] Invalid flatten dimensions: start_dim=" +
                             std::to_string(start_dim_) + ", end_dim=" + std::to_string(end_dim_));
        }

        // 简化实现：计算展平维度的大小
        int64_t flattened_size = 1;
        for (int i = start_dim_; i <= actual_end_dim; ++i) {
            flattened_size *= input_shape.dim(i);
        }

        // 简化：返回2D形状（批次，展平大小）
        return Shape(input_shape.dim(0), static_cast<int32_t>(flattened_size));
    }

public:
    // === 访问器方法 ===
    int start_dim() const { return start_dim_; }
    int end_dim() const { return end_dim_; }

    // === 调试辅助方法 ===
    void print_info(const Shape& input_shape) const {
        std::cout << "Flatten Layer (" << instance_name() << "):" << std::endl;
        std::cout << "  Input shape: " << input_shape.to_string() << std::endl;
        std::cout << "  Start dim: " << start_dim_ << std::endl;
        std::cout << "  End dim: " << end_dim_ << std::endl;

        Shape output_shape = infer_output_shape(input_shape);
        std::cout << "  Output shape: " << output_shape.to_string() << std::endl;
    }
};

} // namespace tr
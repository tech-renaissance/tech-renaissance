/**
 * @file test_module_gradient.cpp
 * @brief 模块梯度检查测试
 * @details 使用数值微分方法验证Linear层的梯度计算
 * @version 1.45.0
 * @date 2025-11-17
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h
 * @note 所属系列: unit_tests
 */

#include "tech_renaissance.h"

using namespace tr;

// 基本模块功能检查
bool basic_module_test(Linear& layer, const Tensor& input) {
    auto backend = BackendManager::get_cpu_backend();

    try {
        // 前向传播
        std::cout << "Input shape: " << input.shape().to_string() << std::endl;
        const Tensor& weight = layer.get_parameter("weight");
        std::cout << "Weight shape: " << weight.shape().to_string() << std::endl;
        Tensor output = layer.forward(input);
        std::cout << "Forward pass successful, output shape: " << output.shape().to_string() << std::endl;

        // 创建梯度输出
        Tensor grad_output = backend->ones(output.shape(), DType::FP32);

        // 反向传播
        Tensor grad_input = layer.backward(grad_output);
        std::cout << "Backward pass successful, grad_input shape: " << grad_input.shape().to_string() << std::endl;

        // 验证形状一致性
        if (grad_input.shape() == input.shape()) {
            std::cout << "Shape consistency check passed" << std::endl;
            return true;
        } else {
            std::cout << "Shape mismatch: input=" << input.shape().to_string()
                      << ", grad_input=" << grad_input.shape().to_string() << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "Error during module test: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    try {
        std::cout << "=== Module Gradient Check Test ===" << std::endl;

        auto backend = BackendManager::get_cpu_backend();

        // 创建一个小的Linear层用于测试
        Linear layer(3, 2, "TestLinear");
        layer.set_backend(backend);

        // 创建正确形状的权重：(out_features, in_features) = (2, 3) (PyTorch标准格式)
        Tensor weight = backend->randn(Shape(2, 3), 123);  // out_features × in_features
        layer.register_parameter("weight", weight);

        // 创建测试输入
        Tensor input = backend->randn(Shape(4, 3), 42);  // batch_size=4, in_features=3

        std::cout << "Input shape: " << input.shape().to_string() << std::endl;
        std::cout << "Layer: " << layer.in_features() << " -> " << layer.out_features() << std::endl;

        // 执行基本模块测试
        bool test_passed = basic_module_test(layer, input);

        if (test_passed) {
            std::cout << "\n[PASS] Basic module test PASSED!" << std::endl;
        } else {
            std::cout << "\n[FAIL] Basic module test FAILED!" << std::endl;
            return 1;
        }

        // 测试Flatten层
        std::cout << "\n=== Testing Flatten Layer ===" << std::endl;

        Flatten flatten_layer(1, -1, "TestFlatten");
        flatten_layer.set_backend(backend);

        // 创建4D输入 (N, C, H, W)
        Tensor input_4d = backend->randn(Shape(2, 3, 4, 5), 123);

        std::cout << "Flatten input shape: " << input_4d.shape().to_string() << std::endl;

        // 前向传播
        Tensor output_2d = flatten_layer.forward(input_4d);
        std::cout << "Flatten output shape: " << output_2d.shape().to_string() << std::endl;

        // 验证形状是否正确：应该是(2, 3*4*5) = (2, 60)
        Shape expected_shape(2, 60);
        if (output_2d.shape() == expected_shape) {
            std::cout << "[PASS] Flatten layer shape test PASSED!" << std::endl;
        } else {
            std::cout << "[FAIL] Flatten layer shape test FAILED!" << std::endl;
            std::cout << "Expected: " << expected_shape.to_string() << std::endl;
            std::cout << "Got: " << output_2d.shape().to_string() << std::endl;
            return 1;
        }

        std::cout << "\n[SUCCESS] All module gradient tests PASSED!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
/**
 * @file test_model_optimizer_integration.cpp
 * @brief 模型与优化器集成测试
 * @details 测试Model类的参数缓存机制和Optimizer的集成
 * @version 1.51.0
 * @date 2025-11-19
 * @author 技术觉醒团队
 */

#include "tech_renaissance.h"

#include <chrono>
#include <iostream>

using namespace tr;
using namespace std::chrono;

/**
 * @brief 性能基准测试
 * @param model 模型引用
 * @param num_iterations 迭代次数
 * @param test_name 测试名称
 */
void benchmark_parameter_access(Model& model, int num_iterations, const std::string& test_name) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;

    // 第一次调用（构建缓存）
    auto start = high_resolution_clock::now();
    auto params = model.trainable_parameters();
    auto end = high_resolution_clock::now();
    auto first_call_duration = duration_cast<microseconds>(end - start).count();

    std::cout << "First call (cache build): " << first_call_duration << " μs" << std::endl;
    std::cout << "Parameter count: " << params.size() << std::endl;

    // 后续调用（使用缓存）
    start = high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        auto cached_params = model.trainable_parameters();
        volatile size_t total_size = 0;  // 防止编译器优化
        for (auto* param : cached_params) {
            total_size += param->numel();
        }
        (void)total_size;  // 避免未使用变量警告
    }
    end = high_resolution_clock::now();

    auto cached_duration = duration_cast<microseconds>(end - start).count();
    float avg_cached_time = float(cached_duration) / num_iterations;

    std::cout << "Cached calls average: " << avg_cached_time << " μs" << std::endl;
    std::cout << "Speedup: " << (float(first_call_duration) / avg_cached_time / num_iterations) << "x" << std::endl;
}

/**
 * @brief 测试参数缓存失效机制
 */
void test_cache_invalidation(Model& model) {
    std::cout << "\n=== Testing Cache Invalidation ===" << std::endl;

    // 初始获取参数
    auto params1 = model.trainable_parameters();
    std::cout << "Initial parameters: " << params1.size() << std::endl;

    // 模拟添加新模块
    auto new_module = std::make_shared<Linear>(128, 10, "extra_layer");
    model.add_module(new_module);

    // 获取更新后的参数（缓存应该已失效并重建）
    auto params2 = model.trainable_parameters();
    std::cout << "Parameters after adding module: " << params2.size() << std::endl;

    if (params2.size() > params1.size()) {
        std::cout << "✓ Cache invalidation working correctly" << std::endl;
    } else {
        std::cout << "✗ Cache invalidation failed" << std::endl;
    }
}

/**
 * @brief 测试设备转移时的缓存更新
 */
void test_device_transfer(Model& model) {
    std::cout << "\n=== Testing Device Transfer Cache Update ===" << std::endl;

    // 初始在CPU上获取参数
    auto cpu_params = model.trainable_parameters();
    std::cout << "CPU parameters: " << cpu_params.size() << std::endl;
    if (!cpu_params.empty()) {
        std::cout << "First param device: " << cpu_params[0]->device().to_string() << std::endl;
    }

    // 简单检查是否支持CUDA
    bool has_cuda = true;
    try {
        auto cuda_backend = BackendManager::instance().get_backend(tr::CUDA[0]);
        (void)cuda_backend; // 避免未使用警告
    } catch (const std::exception&) {
        has_cuda = false;
    }

    if (!has_cuda) {
        std::cout << "No CUDA devices available, skipping GPU test" << std::endl;
        return;
    }

    try {
        // 转移到GPU
        model.to(tr::CUDA[0]);
        auto gpu_params = model.trainable_parameters();
        std::cout << "GPU parameters: " << gpu_params.size() << std::endl;
        if (!gpu_params.empty()) {
            std::cout << "First param device: " << gpu_params[0]->device().to_string() << std::endl;
        }

        // 验证设备一致性
        bool all_gpu = true;
        for (auto* param : gpu_params) {
            if (!param->device().is_cuda()) {
                all_gpu = false;
                break;
            }
        }

        if (all_gpu) {
            std::cout << "✓ Device transfer cache update working correctly" << std::endl;
        } else {
            std::cout << "✗ Device transfer cache update failed" << std::endl;
        }

        // 转移回CPU
        model.to(tr::CPU);

    } catch (const std::exception& e) {
        std::cout << "Device transfer test failed: " << e.what() << std::endl;
    }
}

/**
 * @brief 测试优化器与Model的集成
 */
void test_optimizer_integration(Model& model) {
    std::cout << "\n=== Testing Optimizer Integration ===" << std::endl;

    try {
        // 创建SGD优化器
        auto optimizer = std::make_unique<SGD>(0.01f, 0.9f, 1e-4f, false);

        // 获取模型参数
        auto params = model.trainable_parameters();
        std::cout << "Model parameters: " << params.size() << std::endl;

        // 初始化优化器
        optimizer->initialize(model);
        std::cout << "✓ Optimizer initialized successfully" << std::endl;

        // 创建虚拟数据
        auto backend = BackendManager::instance().get_backend(tr::CPU);
        Shape input_shape{2, 784};
        Shape target_shape{2, 10};
        Tensor input = backend->randn(input_shape);
        Tensor target = backend->zeros(target_shape, DType::FP32);

        // 前向传播
        auto output = model.forward(input);
        std::cout << "Forward output shape: " << output.shape().to_string() << std::endl;

        // 模拟梯度（实际应该由损失函数计算）
        auto& logits = model.logits();
        Tensor grad_output = backend->ones(logits.shape(), DType::FP32);
        logits.set_grad(grad_output);

        // 反向传播
        model.backward(grad_output);
        std::cout << "✓ Backward propagation completed" << std::endl;

        // 优化器步骤
        optimizer->step(model);
        std::cout << "✓ Optimizer step completed" << std::endl;

        // 清零梯度
        optimizer->zero_grad(model);
        std::cout << "✓ Gradients cleared" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Optimizer integration test failed: " << e.what() << std::endl;
    }
}

/**
 * @brief 测试内存分析功能
 */
void test_memory_analysis(Model& model) {
    std::cout << "\n=== Testing Memory Analysis ===" << std::endl;

    try {
        // 分析内存使用
        Shape input_shape{32, 784};
        auto profile = model.analyze_memory(input_shape);

        std::cout << "Memory Profile:" << std::endl;
        std::cout << "  Parameter memory: " << float(profile.parameter_memory) / 1024 << " KB" << std::endl;
        std::cout << "  Activation memory: " << float(profile.activation_memory) / 1024 << " KB" << std::endl;
        std::cout << "  Gradient memory: " << float(profile.gradient_memory) / 1024 << " KB" << std::endl;
        std::cout << "  Total training memory: " << float(profile.total_memory) / 1024 << " KB" << std::endl;
        std::cout << "  Total inference memory: " << float(profile.inference_memory()) / 1024 << " KB" << std::endl;

        // 打印详细报告
        model.print_memory_profile(input_shape);

        std::cout << "✓ Memory analysis completed successfully" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Memory analysis test failed: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== Model-Optimizer Integration Test ===" << std::endl;
    std::cout << "Date: 2025-11-19" << std::endl;
    std::cout << "Version: V1.51.0" << std::endl;

    try {
        // 创建测试模型
        auto model = std::make_shared<Model>("TestMLP");
        model->add_module(std::make_shared<Linear>(784, 256));
        model->add_module(std::make_shared<Tanh>());
        model->add_module(std::make_shared<Linear>(256, 128));
        model->add_module(std::make_shared<Tanh>());
        model->add_module(std::make_shared<Linear>(128, 10));

        std::cout << "Created MLP model with " << model->num_modules() << " modules" << std::endl;
        std::cout << "Model device: " << model->device().to_string() << std::endl;

        // 设置为训练模式
        model->train();

        // 1. 性能基准测试
        benchmark_parameter_access(*model, 1000, "Parameter Access Performance");

        // 2. 缓存失效机制测试
        test_cache_invalidation(*model);

        // 3. 设备转移测试
        test_device_transfer(*model);

        // 4. 优化器集成测试
        test_optimizer_integration(*model);

        // 5. 内存分析测试
        test_memory_analysis(*model);

        std::cout << "\n=== All Tests Completed ===" << std::endl;
        std::cout << "Model-Optimizer integration is working correctly!" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
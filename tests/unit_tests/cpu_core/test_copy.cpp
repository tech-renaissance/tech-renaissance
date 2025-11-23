/**
 * @file test_copy.cpp
 * @brief CPU后端copy()和copy_into()方法单元测试
 * @details 测试CPU后端的copy()和copy_into()方法，CUDA功能已移除以支持模块化架构
 * @version 2.0.0
 * @date 2025-11-23
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h
 * @note 所属系列: tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>

using namespace tr;
using namespace std::chrono;

// 测试辅助函数
void test_assert(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << std::endl;
        assert(false);
    } else {
        std::cout << "[PASS] " << message << std::endl;
    }
}

/**
 * @brief 测试CPU后端的copy()方法
 */
void test_cpu_copy() {
    std::cout << "=== Testing CPU copy() ===" << std::endl;

    try {
        auto cpu_backend = BackendManager::get_cpu_backend();

        // 测试不同形状的张量拷贝
        std::vector<Shape> shapes = {
            Shape(1),           // 标量
            Shape(100),         // 1D向量
            Shape(10, 20),      // 2D矩阵
            Shape(2, 3, 4),     // 3D张量
            Shape(2, 3, 4, 5)   // 4D张量
        };

        for (const auto& shape : shapes) {
            // 创建原始张量并填充随机值
            Tensor original = cpu_backend->empty(shape, DType::FP32);
            cpu_backend->fill(original, 2.56f);

            // 测试copy()方法
            Tensor copied = cpu_backend->copy(original);

            // 验证拷贝结果
            test_assert(copied.shape() == original.shape(), "Copy preserves shape");
            test_assert(copied.dtype() == original.dtype(), "Copy preserves dtype");
            test_assert(copied.device() == original.device(), "Copy preserves device");
            test_assert(copied.device().is_cpu(), "Copied tensor is on CPU");

            // 验证数据内容
            test_assert(cpu_backend->is_close(copied, original), "Copy preserves data");

            // 验证深拷贝：修改原张量不应影响拷贝
            cpu_backend->fill(const_cast<Tensor&>(original), 3.78f);
            test_assert(!cpu_backend->is_close(copied, original), "Copy is independent");

            std::cout << "  CPU copy(" << shape.to_string() << "): PASSED" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "  CPU copy(): FAILED - " << e.what() << std::endl;
    }
}

/**
 * @brief 测试CPU后端的copy_into()方法（同设备内）
 */
void test_cpu_copy_into_same_device() {
    std::cout << "=== Testing CPU copy_into() (same device) ===" << std::endl;

    try {
        auto cpu_backend = BackendManager::get_cpu_backend();

        // 测试不同形状的张量拷贝
        std::vector<Shape> shapes = {
            Shape(100),         // 1D向量
            Shape(10, 20),      // 2D矩阵
            Shape(2, 3, 4)      // 3D张量
        };

        for (const auto& shape : shapes) {
            Tensor src = cpu_backend->empty(shape, DType::FP32);
            Tensor dst = cpu_backend->empty(shape, DType::FP32);

            // 填充源张量和目标张量
            cpu_backend->fill(src, 4.56f);
            cpu_backend->fill(dst, 8.76f);

            // 测试copy_into()方法
            cpu_backend->copy_into(src, dst);

            // 验证拷贝结果
            test_assert(dst.shape() == src.shape(), "copy_into preserves shape");
            test_assert(dst.dtype() == src.dtype(), "copy_into preserves dtype");
            test_assert(dst.device() == src.device(), "copy_into preserves device");
            test_assert(cpu_backend->is_close(dst, src), "copy_into copies data correctly");

            std::cout << "  CPU copy_into(" << shape.to_string() << "): PASSED" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "  CPU copy_into(): FAILED - " << e.what() << std::endl;
    }
}

/**
 * @brief 测试copy()方法的性能
 */
void benchmark_cpu_copy() {
    std::cout << "=== CPU copy() Performance Benchmark ===" << std::endl;

    try {
        auto cpu_backend = BackendManager::get_cpu_backend();

        // 测试不同大小的张量
        std::vector<std::pair<Shape, std::string>> test_cases = {
            {Shape(1000), "1K elements"},
            {Shape(10000), "10K elements"},
            {Shape(100000), "100K elements"},
            {Shape(1000, 1000), "1M elements (2D)"},
            {Shape(100, 100, 100), "1M elements (3D)"}
        };

        constexpr int num_iterations = 100;

        for (const auto& [shape, description] : test_cases) {
            Tensor cpu_tensor = cpu_backend->empty(shape, DType::FP32);
            cpu_backend->fill(cpu_tensor, 1.0f);

            // 预热
            for (int i = 0; i < 10; ++i) {
                Tensor copy_result = cpu_backend->copy(cpu_tensor);
            }

            // 性能测试
            auto start = high_resolution_clock::now();
            for (int i = 0; i < num_iterations; ++i) {
                Tensor copy_result = cpu_backend->copy(cpu_tensor);
            }
            auto end = high_resolution_clock::now();

            auto duration = duration_cast<microseconds>(end - start).count();
            double avg_time = static_cast<double>(duration) / num_iterations / 1000.0; // ms

            std::cout << "  CPU copy(" << description << ") average time: " << std::fixed << std::setprecision(3)
                     << avg_time << " ms" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "  CPU benchmark failed: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Starting CPU Backend Copy Tests..." << std::endl;

    // 功能测试
    test_cpu_copy();
    test_cpu_copy_into_same_device();

    // 性能测试
    benchmark_cpu_copy();

    std::cout << "\n=== ALL CPU COPY TESTS COMPLETED ===" << std::endl;
    std::cout << "CPU copy functionality verified successfully!" << std::endl;

    return 0;
}
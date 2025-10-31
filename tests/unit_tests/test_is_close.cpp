/**
 * @file test_is_close.cpp
 * @brief is_close方法测试
 * @details 测试CPU后端的is_close方法功能
 * @version 1.00.00
 * @date 2025-10-29
 * @author 技术觉醒团队
 * @note 依赖项: tensor.h, cpu_backend.h
 * @note 所属系列: tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace tr;

// 测试辅助函数
void print_test_header(const std::string& test_name) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
}

void print_test_result(const std::string& test_name, bool passed) {
    std::cout << "[TEST] " << test_name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

// 测试1: 完全相同的张量
bool test_identical_tensors() {
    print_test_header("Identical Tensors Test");

    try {
        auto& manager = BackendManager::instance();
        auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(manager.get_backend(tr::CPU));

        if (!cpu_backend) {
            std::cout << "[TEST] Failed to get CPU backend" << std::endl;
            return false;
        }

        // 创建两个相同的张量
        Shape shape(2, 3, 4, 5);
        Tensor tensor_a = Tensor::full(shape, 3.14159f, DType::FP32, tr::CPU);
        Tensor tensor_b = Tensor::full(shape, 3.14159f, DType::FP32, tr::CPU);

        std::cout << "[TEST] Created identical tensors with shape " << shape.to_string() << std::endl;

        // 测试is_close
        bool result = cpu_backend->is_close(tensor_a, tensor_b);
        std::cout << "[TEST] is_close result: " << (result ? "true" : "false") << std::endl;

        // 预期结果应该是true
        bool test_passed = (result == true);
        print_test_result("Identical Tensors Test", test_passed);
        return test_passed;

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        print_test_result("Identical Tensors Test", false);
        return false;
    }
}

// 测试2: 相近的张量
bool test_close_tensors() {
    print_test_header("Close Tensors Test");

    try {
        auto& manager = BackendManager::instance();
        auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(manager.get_backend(tr::CPU));

        // 创建两个相近的张量
        Shape shape(3, 4);
        Tensor tensor_a = Tensor::full(shape, 1.0f, DType::FP32, tr::CPU);
        Tensor tensor_b = Tensor::full(shape, 1.00001f, DType::FP32, tr::CPU);

        std::cout << "[TEST] Created close tensors with shape " << shape.to_string() << std::endl;
        std::cout << "[TEST] Tensor A values: 1.00000" << std::endl;
        std::cout << "[TEST] Tensor B values: 1.00001" << std::endl;

        // 计算理论平均绝对差
        float expected_avg_diff = std::abs(1.00001f - 1.0f);
        std::cout << "[TEST] Expected average absolute difference: " << std::scientific << expected_avg_diff << std::endl;

        // 测试is_close
        bool result = cpu_backend->is_close(tensor_a, tensor_b);
        std::cout << "[TEST] is_close result (default eps=5e-5): " << (result ? "true" : "false") << std::endl;

        // 测试自定义容差
        bool result_strict = cpu_backend->is_close(tensor_a, tensor_b, 1e-6f);
        std::cout << "[TEST] is_close result (eps=1e-6): " << (result_strict ? "true" : "false") << std::endl;

        // 预期：默认容差应该通过，严格容差应该失败
        bool test_passed = (result == true && result_strict == false);
        print_test_result("Close Tensors Test", test_passed);
        return test_passed;

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        print_test_result("Close Tensors Test", false);
        return false;
    }
}

// 测试3: 形状不匹配的张量
bool test_shape_mismatch() {
    print_test_header("Shape Mismatch Test");

    try {
        auto& manager = BackendManager::instance();
        auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(manager.get_backend(tr::CPU));

        // 创建不同形状的张量
        Tensor tensor_a = Tensor::full(Shape(2, 3), 1.0f, DType::FP32, tr::CPU);
        Tensor tensor_b = Tensor::full(Shape(3, 2), 1.0f, DType::FP32, tr::CPU);

        std::cout << "[TEST] Created tensors with shapes " << tensor_a.shape().to_string()
                  << " and " << tensor_b.shape().to_string() << std::endl;

        // 测试is_close
        bool result = cpu_backend->is_close(tensor_a, tensor_b);
        std::cout << "[TEST] is_close result: " << (result ? "true" : "false") << std::endl;

        // 预期结果应该是false
        bool test_passed = (result == false);
        print_test_result("Shape Mismatch Test", test_passed);
        return test_passed;

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        print_test_result("Shape Mismatch Test", false);
        return false;
    }
}

// 测试4: 空张量
bool test_empty_tensors() {
    print_test_header("Empty Tensors Test");

    try {
        auto& manager = BackendManager::instance();
        auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(manager.get_backend(tr::CPU));

        // 创建空张量
        Tensor empty_a = Tensor();  // 空张量
        Tensor empty_b = Tensor();  // 空张量
        Tensor non_empty = Tensor::full(Shape(2, 2), 1.0f, DType::FP32, tr::CPU);

        std::cout << "[TEST] Created empty and non-empty tensors" << std::endl;

        // 测试空张量vs空张量
        bool result1 = cpu_backend->is_close(empty_a, empty_b);
        std::cout << "[TEST] Empty vs Empty: " << (result1 ? "true" : "false") << std::endl;

        // 测试空张量vs非空张量
        bool result2 = cpu_backend->is_close(empty_a, non_empty);
        std::cout << "[TEST] Empty vs Non-empty: " << (result2 ? "true" : "false") << std::endl;

        // 预期：空张量vs空张量为true，空张量vs非空张量为false
        bool test_passed = (result1 == true && result2 == false);
        print_test_result("Empty Tensors Test", test_passed);
        return test_passed;

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        print_test_result("Empty Tensors Test", false);
        return false;
    }
}

// 测试5: 随机数据张量
bool test_random_tensors() {
    print_test_header("Random Tensors Test");

    try {
        auto& manager = BackendManager::instance();
        auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(manager.get_backend(tr::CPU));

        // 创建具有不同分布的随机张量
        Shape shape(100);  // 100个元素的1D张量
        Tensor tensor_a = Tensor::full(shape, 1.0f, DType::FP32, tr::CPU);
        Tensor tensor_b = Tensor::full(shape, 1.001f, DType::FP32, tr::CPU);

        std::cout << "[TEST] Created random-like tensors with shape " << shape.to_string() << std::endl;

        // 计算理论平均绝对差
        float expected_avg_diff = std::abs(1.001f - 1.0f);
        std::cout << "[TEST] Expected average absolute difference: " << std::scientific << expected_avg_diff << std::endl;

        // 测试不同的容差值
        bool result_default = cpu_backend->is_close(tensor_a, tensor_b);  // 5e-5
        bool result_loose = cpu_backend->is_close(tensor_a, tensor_b, 1e-2f);  // 1e-2
        bool result_tight = cpu_backend->is_close(tensor_a, tensor_b, 1e-4f);  // 1e-4

        std::cout << "[TEST] is_close (eps=5e-5):   " << (result_default ? "true" : "false") << std::endl;
        std::cout << "[TEST] is_close (eps=1e-2):   " << (result_loose ? "true" : "false") << std::endl;
        std::cout << "[TEST] is_close (eps=1e-4):   " << (result_tight ? "true" : "false") << std::endl;

        // 预期：默认和严格容差应该失败，宽松容差应该通过
        bool test_passed = (result_default == false && result_loose == true && result_tight == false);
        print_test_result("Random Tensors Test", test_passed);
        return test_passed;

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        print_test_result("Random Tensors Test", false);
        return false;
    }
}

// 测试6: 数据类型检查
bool test_dtype_check() {
    print_test_header("Data Type Check Test");

    try {
        auto& manager = BackendManager::instance();
        auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(manager.get_backend(tr::CPU));

        // 创建不同数据类型的张量
        Tensor fp32_tensor = Tensor::full(Shape(2, 2), 1.0f, DType::FP32, tr::CPU);
        Tensor int8_tensor = Tensor::full(Shape(2, 2), 1, DType::INT8, tr::CPU);

        std::cout << "[TEST] Created FP32 and INT8 tensors" << std::endl;

        // 测试FP32 vs INT8 - 应该抛出异常
        bool exception_thrown = false;
        try {
            bool result = cpu_backend->is_close(fp32_tensor, int8_tensor);
            std::cout << "[TEST] Unexpectedly succeeded with result: " << (result ? "true" : "false") << std::endl;
        } catch (const TRException& e) {
            exception_thrown = true;
            std::cout << "[TEST] Expected exception caught: " << e.what() << std::endl;
        }

        // 测试INT8 vs INT8 - 应该抛出异常
        bool exception_thrown2 = false;
        try {
            bool result = cpu_backend->is_close(int8_tensor, int8_tensor);
            std::cout << "[TEST] Unexpectedly succeeded with result: " << (result ? "true" : "false") << std::endl;
        } catch (const TRException& e) {
            exception_thrown2 = true;
            std::cout << "[TEST] Expected exception caught: " << e.what() << std::endl;
        }

        // 预期：两种情况都应该抛出异常
        bool test_passed = (exception_thrown && exception_thrown2);
        print_test_result("Data Type Check Test", test_passed);
        return test_passed;

    } catch (const TRException& e) {
        std::cout << "[TEST] Unexpected exception caught: " << e.what() << std::endl;
        print_test_result("Data Type Check Test", false);
        return false;
    }
}

int main() {
    std::cout << "=== CPU Backend is_close() Method Test Suite ===" << std::endl;

    bool all_passed = true;

    // 运行所有测试
    all_passed &= test_identical_tensors();
    all_passed &= test_close_tensors();
    all_passed &= test_shape_mismatch();
    all_passed &= test_empty_tensors();
    all_passed &= test_random_tensors();
    all_passed &= test_dtype_check();

    // 总结
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Overall Result: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;

    return all_passed ? 0 : 1;
}
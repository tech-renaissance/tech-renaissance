/**
 * @file test_cpu_conv_extra.cpp
 * @brief CPU后端卷积操作扩展测试
 * @details 测试不同stride配置下的卷积和转置卷积操作
 * @version 1.00.00
 * @date 2025-11-03
 * @author 技术觉醒团队
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/utils/logger.h"
#include "tech_renaissance/backend/backend_manager.h"

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <string>

using namespace tr;

// Simple test framework
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "ASSERTION FAILED: " << message << std::endl; \
            return false; \
        } \
    } while(0)

#define TEST_CASE(name, func) \
    do { \
        std::cout << "\n=== Running test: " << name << " ===" << std::endl; \
        if (func()) { \
            std::cout << "[PASS] " << name << " PASSED" << std::endl; \
            passed_tests++; \
        } else { \
            std::cout << "[FAIL] " << name << " FAILED" << std::endl; \
            failed_tests++; \
        } \
        total_tests++; \
    } while(0)

// Global variables for test framework
int total_tests = 0;
int passed_tests = 0;
int failed_tests = 0;

std::shared_ptr<CpuBackend> cpu_backend;

bool setup_conv_extra_tests() {
    cpu_backend = std::dynamic_pointer_cast<CpuBackend>(BackendManager::instance().get_backend(CPU));
    TEST_ASSERT(cpu_backend != nullptr, "Failed to get CPU backend");
    return true;
}

// ===== 标准卷积测试 =====

bool test_conv_stride1() {
    std::cout << "Testing: Standard Convolution with stride=1" << std::endl;

    // 创建5x5随机输入 (1,1,5,5)
    Tensor input = cpu_backend->randint(Shape(1, 1, 5, 5), 1, 6, DType::FP32); // 1~5的随机数
    std::cout << "Input tensor (1,1,5,5):" << std::endl;
    input.print();

    // 创建3x3全1卷积核 (1,1,3,3)
    Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 3), DType::FP32);
    std::cout << "\nKernel tensor (1,1,3,3) - all ones:" << std::endl;
    kernel.print();

    // 执行stride=1, padding=1的卷积
    Tensor result = cpu_backend->conv(input, kernel, 1, 1);

    std::cout << "\nOutput tensor - Convolution with stride=1, padding=1:" << std::endl;
    result.print();

    // 验证输出形状应该是(1,1,5,5)
    Shape expected_shape = Shape(1, 1, 5, 5);
    TEST_ASSERT(result.shape() == expected_shape, "Result shape mismatch for stride=1 conv");

    return true;
}

bool test_conv_stride2() {
    std::cout << "Testing: Standard Convolution with stride=2" << std::endl;

    // 创建5x5随机输入 (1,1,5,5)
    Tensor input = cpu_backend->randint(Shape(1, 1, 5, 5), 1, 6, DType::FP32); // 1~5的随机数
    std::cout << "Input tensor (1,1,5,5):" << std::endl;
    input.print();

    // 创建3x3全1卷积核 (1,1,3,3)
    Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 3), DType::FP32);
    std::cout << "\nKernel tensor (1,1,3,3) - all ones:" << std::endl;
    kernel.print();

    // 执行stride=2, padding=1的卷积
    Tensor result = cpu_backend->conv(input, kernel, 2, 1);

    std::cout << "\nOutput tensor - Convolution with stride=2, padding=1:" << std::endl;
    result.print();

    // 验证输出形状应该是(1,1,3,3)
    Shape expected_shape = Shape(1, 1, 3, 3);
    TEST_ASSERT(result.shape() == expected_shape, "Result shape mismatch for stride=2 conv");

    return true;
}

// ===== 转置卷积测试 =====

bool test_transposed_conv_stride1() {
    std::cout << "Testing: Transposed Convolution with stride=1" << std::endl;

    // 创建5x5随机输入 (1,1,5,5)
    Tensor input = cpu_backend->randint(Shape(1, 1, 5, 5), 1, 6, DType::FP32); // 1~5的随机数
    std::cout << "Input tensor (1,1,5,5):" << std::endl;
    input.print();

    // 创建3x3全1卷积核 (1,1,3,3)
    Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 3), DType::FP32);
    std::cout << "\nKernel tensor (1,1,3,3) - all ones:" << std::endl;
    kernel.print();

    // 执行stride=1, padding=1的转置卷积
    Tensor result = cpu_backend->transposed_conv(input, kernel, 1, 1);

    std::cout << "\nOutput tensor - Transposed Convolution with stride=1, padding=1:" << std::endl;
    result.print();

    // 验证输出形状应该是(1,1,5,5)
    Shape expected_shape = Shape(1, 1, 5, 5);
    TEST_ASSERT(result.shape() == expected_shape, "Result shape mismatch for stride=1 transposed conv");

    return true;
}

bool test_transposed_conv_stride2() {
    std::cout << "Testing: Transposed Convolution with stride=2" << std::endl;

    // 创建5x5随机输入 (1,1,5,5)
    Tensor input = cpu_backend->randint(Shape(1, 1, 5, 5), 1, 6, DType::FP32); // 1~5的随机数
    std::cout << "Input tensor (1,1,5,5):" << std::endl;
    input.print();

    // 创建3x3全1卷积核 (1,1,3,3)
    Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 3), DType::FP32);
    std::cout << "\nKernel tensor (1,1,3,3) - all ones:" << std::endl;
    kernel.print();

    // 执行stride=2, padding=1的转置卷积
    Tensor result = cpu_backend->transposed_conv(input, kernel, 2, 1);

    std::cout << "\nOutput tensor - Transposed Convolution with stride=2, padding=1:" << std::endl;
    result.print();

    // 验证输出形状应该是(1,1,9,9)
    // 公式: o = (i-1)*s + k - 2p = (5-1)*2 + 3 - 2*1 = 8 + 3 - 2 = 9
    Shape expected_shape = Shape(1, 1, 9, 9);
    TEST_ASSERT(result.shape() == expected_shape, "Result shape mismatch for stride=2 transposed conv");

    return true;
}

int main() {
    std::cout << "=== CPU Convolution Extra Tests ===" << std::endl;
    std::cout << "Testing different stride configurations for convolution and transposed convolution" << std::endl;
    std::cout << "All inputs: (1,1,5,5) with random values 1-5" << std::endl;
    std::cout << "All kernels: (1,1,3,3) with all ones" << std::endl;
    std::cout << "All tests: kernel_size=3, padding=1" << std::endl;

    // Setup
    TEST_CASE("Setup", setup_conv_extra_tests);

    // Standard Convolution Tests
    TEST_CASE("Convolution Stride 1", test_conv_stride1);
    TEST_CASE("Convolution Stride 2", test_conv_stride2);

    // Transposed Convolution Tests
    TEST_CASE("Transposed Convolution Stride 1", test_transposed_conv_stride1);
    TEST_CASE("Transposed Convolution Stride 2", test_transposed_conv_stride2);

    // Summary
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Total tests: " << total_tests << std::endl;
    std::cout << "Passed: " << passed_tests << std::endl;
    std::cout << "Failed: " << failed_tests << std::endl;

    if (failed_tests == 0) {
        std::cout << "ALL TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED!" << std::endl;
        return 1;
    }
}
/**
 * @file test_cpu_conv_new.cpp
 * @brief CPU后端卷积操作测试
 * @details 测试标准卷积和转置卷积功能
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
        std::cout << "Running test: " << name << "..." << std::endl; \
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

bool setup_conv_tests() {
    cpu_backend = std::dynamic_pointer_cast<CpuBackend>(BackendManager::instance().get_backend(CPU));
    TEST_ASSERT(cpu_backend != nullptr, "Failed to get CPU backend");
    return true;
}

// ===== 标准卷积测试 =====

bool test_conv_basic() {
    // 测试基本的2D卷积
    Tensor input = cpu_backend->ones(Shape(4, 4), DType::FP32);

    // 创建一个简单的3x3卷积核
    Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 3), DType::FP32);
    float* kernel_data = static_cast<float*>(kernel.data_ptr());
    for (int i = 0; i < 9; ++i) {
        kernel_data[i] = 1.0f;
    }

    Tensor result = cpu_backend->conv(input, kernel, 1, 0);

    // 对于2D输入(4,4)，输出应该是(1,1,2,2)
    Shape expected_shape = Shape(1, 1, 2, 2);
    TEST_ASSERT(result.shape() == expected_shape, "Result shape mismatch, got " + result.shape().to_string() + ", expected " + expected_shape.to_string());
    TEST_ASSERT(result.dtype() == DType::FP32, "Result dtype mismatch");

    float* result_data = static_cast<float*>(result.data_ptr());
    // 验证结果：每个位置应该是9个1的和
    for (int i = 0; i < 4; ++i) {
        TEST_ASSERT(std::abs(result_data[i] - 9.0f) < 1e-6, "Result should be 9.0f");
    }

    return true;
}

bool test_conv_stride2() {
    // 测试stride=2的卷积
    Tensor input = cpu_backend->ones(Shape(6, 6), DType::FP32);

    float* input_data = static_cast<float*>(input.data_ptr());
    for (int i = 0; i < 36; ++i) {
        input_data[i] = static_cast<float>(i + 1);
    }

    Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 3), DType::FP32);
    float* kernel_data = static_cast<float*>(kernel.data_ptr());
    for (int i = 0; i < 9; ++i) {
        kernel_data[i] = 1.0f;
    }

    Tensor result = cpu_backend->conv(input, kernel, 2, 0);

    Shape expected_shape = Shape(1, 1, 2, 2);
    TEST_ASSERT(result.shape() == expected_shape, "Result shape mismatch, got " + result.shape().to_string() + ", expected " + expected_shape.to_string());

    float* result_data = static_cast<float*>(result.data_ptr());
    // 验证第一个位置：(1+2+3+7+8+9+13+14+15)
    TEST_ASSERT(std::abs(result_data[0] - 72.0f) < 1e-6, "Result[0] should be 72.0f");

    return true;
}

bool test_conv_with_padding() {
    // 测试带padding的卷积
    Tensor input = cpu_backend->ones(Shape(4, 4), DType::FP32);

    float* input_data = static_cast<float*>(input.data_ptr());
    for (int i = 0; i < 16; ++i) {
        input_data[i] = static_cast<float>(i + 1);
    }

    Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 3), DType::FP32);
    float* kernel_data = static_cast<float*>(kernel.data_ptr());
    for (int i = 0; i < 9; ++i) {
        kernel_data[i] = 1.0f;
    }

    Tensor result = cpu_backend->conv(input, kernel, 1, 1);

    Shape expected_shape = Shape(1, 1, 4, 4);
    TEST_ASSERT(result.shape() == expected_shape, "Result shape mismatch, got " + result.shape().to_string() + ", expected " + expected_shape.to_string());

    float* result_data = static_cast<float*>(result.data_ptr());
    // 边界位置应该包含padding的贡献（padding值为0）
    TEST_ASSERT(std::abs(result_data[0] - 14.0f) < 1e-6, "Result[0] should be 14.0f");  // (0+0+0+0+1+2+0+5+6) = 14
    TEST_ASSERT(std::abs(result_data[5] - 54.0f) < 1e-6, "Result[5] should be 54.0f");  // (1+2+3+5+6+7+9+10+11) = 54

    return true;
}

bool test_conv_into() {
    // 测试conv_into函数
    Tensor input = cpu_backend->ones(Shape(4, 4), DType::FP32);

    float* input_data = static_cast<float*>(input.data_ptr());
    for (int i = 0; i < 16; ++i) {
        input_data[i] = static_cast<float>(i + 1);
    }

    Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 3), DType::FP32);
    float* kernel_data = static_cast<float*>(kernel.data_ptr());
    for (int i = 0; i < 9; ++i) {
        kernel_data[i] = 1.0f;
    }

    Shape expected_shape = Shape(1, 1, 2, 2);
    Tensor result = cpu_backend->empty(expected_shape, DType::FP32);

    cpu_backend->conv_into(input, kernel, result, 1, 0);

    TEST_ASSERT(result.shape() == expected_shape, "Result shape mismatch, got " + result.shape().to_string() + ", expected " + expected_shape.to_string());

    float* result_data = static_cast<float*>(result.data_ptr());
    TEST_ASSERT(std::abs(result_data[0] - 54.0f) < 1e-6, "Result[0] should be 54.0f");  // (1+2+3+5+6+7+9+10+11) = 54

    return true;
}

// ===== 转置卷积测试 =====

bool test_transposed_conv_basic() {
    // 测试基本的转置卷积
    Tensor input = cpu_backend->ones(Shape(2, 2), DType::FP32);

    float* input_data = static_cast<float*>(input.data_ptr());
    input_data[0] = 1.0f;
    input_data[1] = 2.0f;
    input_data[2] = 3.0f;
    input_data[3] = 4.0f;

    Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 3), DType::FP32);
    float* kernel_data = static_cast<float*>(kernel.data_ptr());
    for (int i = 0; i < 9; ++i) {
        kernel_data[i] = 1.0f;
    }

    Tensor result = cpu_backend->transposed_conv(input, kernel, 1, 0);

    Shape expected_shape = Shape(1, 1, 4, 4);
    TEST_ASSERT(result.shape() == expected_shape, "Result shape mismatch, got " + result.shape().to_string() + ", expected " + expected_shape.to_string());
    TEST_ASSERT(result.dtype() == DType::FP32, "Result dtype mismatch");

    float* result_data = static_cast<float*>(result.data_ptr());
    // 验证转置卷积结果
    TEST_ASSERT(std::abs(result_data[0] - 1.0f) < 1e-6, "Result[0] should be 1.0f");   // 只来自input[0]
    TEST_ASSERT(std::abs(result_data[1] - 3.0f) < 1e-6, "Result[1] should be 3.0f");   // input[0] + input[1]
    TEST_ASSERT(std::abs(result_data[5] - 10.0f) < 1e-6, "Result[5] should be 10.0f");  // input[0] + input[1] + input[2] + input[3]

    return true;
}

bool test_transposed_conv_stride2() {
    // 测试stride=2的转置卷积
    Tensor input = cpu_backend->ones(Shape(2, 2), DType::FP32);

    float* input_data = static_cast<float*>(input.data_ptr());
    for (int i = 0; i < 4; ++i) {
        input_data[i] = static_cast<float>(i + 1);
    }

    Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 3), DType::FP32);
    float* kernel_data = static_cast<float*>(kernel.data_ptr());
    for (int i = 0; i < 9; ++i) {
        kernel_data[i] = 1.0f;
    }

    Tensor result = cpu_backend->transposed_conv(input, kernel, 2, 0);

    Shape expected_shape = Shape(1, 1, 5, 5);
    TEST_ASSERT(result.shape() == expected_shape, "Result shape mismatch, got " + result.shape().to_string() + ", expected " + expected_shape.to_string());

    return true;
}

bool test_transposed_conv_into() {
    // 测试transposed_conv_into函数
    Tensor input = cpu_backend->ones(Shape(2, 2), DType::FP32);

    float* input_data = static_cast<float*>(input.data_ptr());
    input_data[0] = 1.0f;
    input_data[1] = 2.0f;
    input_data[2] = 3.0f;
    input_data[3] = 4.0f;

    Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 3), DType::FP32);
    float* kernel_data = static_cast<float*>(kernel.data_ptr());
    for (int i = 0; i < 9; ++i) {
        kernel_data[i] = 1.0f;
    }

    Shape expected_shape = Shape(1, 1, 4, 4);
    Tensor result = cpu_backend->empty(expected_shape, DType::FP32);

    cpu_backend->transposed_conv_into(input, kernel, result, 1, 0);

    TEST_ASSERT(result.shape() == expected_shape, "Result shape mismatch, got " + result.shape().to_string() + ", expected " + expected_shape.to_string());

    float* result_data = static_cast<float*>(result.data_ptr());
    TEST_ASSERT(std::abs(result_data[5] - 10.0f) < 1e-6, "Result[5] should be 10.0f");  // 中心位置的值

    return true;
}

// ===== 错误处理测试 =====

bool test_conv_invalid_kernel_dimensions() {
    // 测试无效的卷积核维度
    Tensor input = cpu_backend->ones(Shape(4, 4), DType::FP32);
    Tensor kernel2d = cpu_backend->ones(Shape(3, 3), DType::FP32);  // 2D kernel

    try {
        cpu_backend->conv(input, kernel2d, 1, 0);
        return false;  // Should throw exception
    } catch (const TRException&) {
        return true;  // Expected exception
    }
}

bool test_conv_non_square_kernel() {
    // 测试非正方形卷积核
    Tensor input = cpu_backend->ones(Shape(4, 4), DType::FP32);
    Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 5), DType::FP32);  // 3x5 kernel

    try {
        cpu_backend->conv(input, kernel, 1, 0);
        return false;  // Should throw exception
    } catch (const TRException&) {
        return true;  // Expected exception
    }
}

bool test_conv_invalid_stride() {
    // 测试无效的stride
    Tensor input = cpu_backend->ones(Shape(4, 4), DType::FP32);
    Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 3), DType::FP32);

    try {
        cpu_backend->conv(input, kernel, 3, 0);  // stride != 1 or 2
        return false;  // Should throw exception
    } catch (const TRException&) {
        return true;  // Expected exception
    }
}

int main() {
    std::cout << "=== CPU Convolution Tests ===" << std::endl;

    // Setup
    TEST_CASE("Setup", setup_conv_tests);

    // Standard Convolution Tests
    TEST_CASE("Conv Basic", test_conv_basic);
    TEST_CASE("Conv Stride 2", test_conv_stride2);
    TEST_CASE("Conv With Padding", test_conv_with_padding);
    TEST_CASE("Conv Into", test_conv_into);

    // Transposed Convolution Tests
    TEST_CASE("Transposed Conv Basic", test_transposed_conv_basic);
    TEST_CASE("Transposed Conv Stride 2", test_transposed_conv_stride2);
    TEST_CASE("Transposed Conv Into", test_transposed_conv_into);

    // Error Handling Tests
    TEST_CASE("Invalid Kernel Dimensions", test_conv_invalid_kernel_dimensions);
    TEST_CASE("Non Square Kernel", test_conv_non_square_kernel);
    TEST_CASE("Invalid Stride", test_conv_invalid_stride);

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
/**
 * @file test_cpu_pooling.cpp
 * @brief CPU后端池化操作测试
 * @details 测试max pooling和global average pooling功能
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

bool setup_pooling_tests() {
    cpu_backend = std::dynamic_pointer_cast<CpuBackend>(BackendManager::instance().get_backend(CPU));
    TEST_ASSERT(cpu_backend != nullptr, "Failed to get CPU backend");
    return true;
}

// ===== Max Pooling 测试 =====

bool test_max_pool_basic() {
    // 测试2D张量的max pooling
    Tensor input = cpu_backend->ones(Shape(4, 4), DType::FP32);

    // 手动设置输入值便于验证
    float* input_data = static_cast<float*>(input.data_ptr());
    for (int i = 0; i < 16; ++i) {
        input_data[i] = static_cast<float>(i);
    }

    // 2x2 max pooling, stride=2
    Tensor result = cpu_backend->max_pool(input, 2, 2);

    TEST_ASSERT(result.shape() == Shape(2, 2), "Result shape mismatch");
    TEST_ASSERT(result.dtype() == DType::FP32, "Result dtype mismatch");

    float* result_data = static_cast<float*>(result.data_ptr());
    // 验证结果：每个2x2块的最大值
    TEST_ASSERT(std::abs(result_data[0] - 5.0f) < 1e-6, "Result[0] should be 5.0f");  // max(0,1,4,5)
    TEST_ASSERT(std::abs(result_data[1] - 7.0f) < 1e-6, "Result[1] should be 7.0f");  // max(2,3,6,7)
    TEST_ASSERT(std::abs(result_data[2] - 13.0f) < 1e-6, "Result[2] should be 13.0f"); // max(8,9,12,13)
    TEST_ASSERT(std::abs(result_data[3] - 15.0f) < 1e-6, "Result[3] should be 15.0f"); // max(10,11,14,15)

    return true;
}

bool test_max_pool_stride1() {
    // 测试stride=1的max pooling
    Tensor input = cpu_backend->ones(Shape(3, 3), DType::FP32);

    float* input_data = static_cast<float*>(input.data_ptr());
    for (int i = 0; i < 9; ++i) {
        input_data[i] = static_cast<float>(i + 1);
    }

    Tensor result = cpu_backend->max_pool(input, 2, 1);

    TEST_ASSERT(result.shape() == Shape(2, 2), "Result shape mismatch");

    float* result_data = static_cast<float*>(result.data_ptr());
    TEST_ASSERT(std::abs(result_data[0] - 5.0f) < 1e-6, "Result[0] should be 5.0f");  // max(1,2,4,5)
    TEST_ASSERT(std::abs(result_data[1] - 6.0f) < 1e-6, "Result[1] should be 6.0f");  // max(2,3,5,6)
    TEST_ASSERT(std::abs(result_data[2] - 8.0f) < 1e-6, "Result[2] should be 8.0f");  // max(4,5,7,8)
    TEST_ASSERT(std::abs(result_data[3] - 9.0f) < 1e-6, "Result[3] should be 9.0f");  // max(5,6,8,9)

    return true;
}

bool test_max_pool_into() {
    // 测试max_pool_into函数
    Tensor input = cpu_backend->ones(Shape(4, 4), DType::FP32);

    float* input_data = static_cast<float*>(input.data_ptr());
    for (int i = 0; i < 16; ++i) {
        input_data[i] = static_cast<float>(i);
    }

    Shape expected_shape = Shape(2, 2);
    Tensor result = cpu_backend->empty(expected_shape, DType::FP32);

    cpu_backend->max_pool_into(input, result, 2, 2);

    TEST_ASSERT(result.shape() == expected_shape, "Result shape mismatch");

    float* result_data = static_cast<float*>(result.data_ptr());
    TEST_ASSERT(std::abs(result_data[0] - 5.0f) < 1e-6, "Result[0] should be 5.0f");
    TEST_ASSERT(std::abs(result_data[1] - 7.0f) < 1e-6, "Result[1] should be 7.0f");
    TEST_ASSERT(std::abs(result_data[2] - 13.0f) < 1e-6, "Result[2] should be 13.0f");
    TEST_ASSERT(std::abs(result_data[3] - 15.0f) < 1e-6, "Result[3] should be 15.0f");

    return true;
}

// ===== Global Average Pooling 测试 =====

bool test_global_avg_pool_2d() {
    // 测试2D张量的global average pooling
    Tensor input = cpu_backend->ones(Shape(2, 2), DType::FP32);

    float* input_data = static_cast<float*>(input.data_ptr());
    input_data[0] = 1.0f;
    input_data[1] = 3.0f;
    input_data[2] = 5.0f;
    input_data[3] = 7.0f;

    Tensor result = cpu_backend->global_avg_pool(input);

    TEST_ASSERT(result.shape() == Shape(1, 1), "Result shape mismatch");

    float* result_data = static_cast<float*>(result.data_ptr());
    TEST_ASSERT(std::abs(result_data[0] - 4.0f) < 1e-6, "Result should be 4.0f");  // (1+3+5+7)/4

    return true;
}

bool test_global_avg_pool_into() {
    // 测试global_avg_pool_into函数
    Tensor input = cpu_backend->ones(Shape(2, 3), DType::FP32);

    float* input_data = static_cast<float*>(input.data_ptr());
    for (int i = 0; i < 6; ++i) {
        input_data[i] = static_cast<float>(i + 1);
    }

    Shape expected_shape = Shape(1, 1);
    Tensor result = cpu_backend->empty(expected_shape, DType::FP32);

    cpu_backend->global_avg_pool_into(input, result);

    TEST_ASSERT(result.shape() == expected_shape, "Result shape mismatch");

    float* result_data = static_cast<float*>(result.data_ptr());
    TEST_ASSERT(std::abs(result_data[0] - 3.5f) < 1e-6, "Result should be 3.5f");  // (1+2+3+4+5+6)/6

    return true;
}

// ===== 错误处理测试 =====

bool test_max_pool_invalid_kernel_size() {
    // 测试无效的kernel size
    Tensor input = cpu_backend->ones(Shape(4, 4), DType::FP32);

    try {
        cpu_backend->max_pool(input, 0, 2);  // kernel_size <= 0
        return false;  // Should throw exception
    } catch (const TRException&) {
        return true;  // Expected exception
    }
}

bool test_max_pool_invalid_stride() {
    // 测试无效的stride
    Tensor input = cpu_backend->ones(Shape(4, 4), DType::FP32);

    try {
        cpu_backend->max_pool(input, 2, 3);  // stride != 1 or 2
        return false;  // Should throw exception
    } catch (const TRException&) {
        return true;  // Expected exception
    }
}

int main() {
    std::cout << "=== CPU Pooling Tests ===" << std::endl;

    // Setup
    TEST_CASE("Setup", setup_pooling_tests);

    // Max Pooling Tests
    TEST_CASE("Max Pool Basic", test_max_pool_basic);
    TEST_CASE("Max Pool Stride 1", test_max_pool_stride1);
    TEST_CASE("Max Pool Into", test_max_pool_into);

    // Global Average Pooling Tests
    TEST_CASE("Global Avg Pool 2D", test_global_avg_pool_2d);
    TEST_CASE("Global Avg Pool Into", test_global_avg_pool_into);

    // Error Handling Tests
    TEST_CASE("Invalid Kernel Size", test_max_pool_invalid_kernel_size);
    TEST_CASE("Invalid Stride", test_max_pool_invalid_stride);

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
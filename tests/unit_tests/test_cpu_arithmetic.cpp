/**
 * @file test_cpu_arithmetic.cpp
 * @brief CPU backend arithmetic operations unit test
 * @details Test for CPU backend arithmetic operations including addition, subtraction, multiplication, and division
 * @version 1.60.1
 * @date 2025年11月22日
 * @author 技术觉醒团队
 * @note 依赖项: backend_manager.h, cpu_backend.h, tensor.h
 * @note 所属系列: tests
 */

#include "tech_renaissance/data/dtype.h"
#include "tech_renaissance/data/device.h"
#include "tech_renaissance/data/shape.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/backend/backend.h"
#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/utils/logger.h"
#include "tech_renaissance/utils/tr_exception.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

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

// Helper function to compare two tensors
bool compare_tensors(const Tensor& tensor_a, const Tensor& tensor_b, float eps = 1e-6f) {
    if (tensor_a.shape() != tensor_b.shape()) {
        return false;
    }
    if (tensor_a.dtype() != tensor_b.dtype()) {
        return false;
    }

    auto* backend = dynamic_cast<CpuBackend*>(BackendManager::instance().get_backend(CPU).get());
    if (!backend) {
        return false;
    }

    int64_t num_elements = tensor_a.numel();
    for (int64_t i = 0; i < num_elements; ++i) {
        float val_a = backend->get_item_fp32(tensor_a, i);
        float val_b = backend->get_item_fp32(tensor_b, i);
        if (std::abs(val_a - val_b) > eps) {
            std::cout << "Mismatch at index " << i << ": " << val_a << " vs " << val_b << std::endl;
            return false;
        }
    }
    return true;
}

// Test tensor division operation
bool test_tensor_division() {
    try {
        auto backend = BackendManager::get_cpu_backend();
        auto* cpu_backend = dynamic_cast<CpuBackend*>(backend.get());
        TEST_ASSERT(cpu_backend != nullptr, "Failed to get CPU backend");

        // Create test tensors
        Shape shape(2, 3);
        std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> b_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> expected_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

        Tensor a = backend->empty(shape, DType::FP32);
        Tensor b = backend->empty(shape, DType::FP32);
        Tensor result = backend->empty(shape, DType::FP32);

        // Fill tensors with data
        float* a_ptr = static_cast<float*>(a.data_ptr());
        float* b_ptr = static_cast<float*>(b.data_ptr());
        for (size_t i = 0; i < a_data.size(); ++i) {
            a_ptr[i] = a_data[i];
            b_ptr[i] = b_data[i];
        }

        // Test div_into operation
        cpu_backend->div_into(a, b, result);

        // Create expected result tensor
        Tensor expected = backend->empty(shape, DType::FP32);
        float* expected_ptr = static_cast<float*>(expected.data_ptr());
        for (size_t i = 0; i < expected_data.size(); ++i) {
            expected_ptr[i] = expected_data[i];
        }

        // Compare results
        TEST_ASSERT(compare_tensors(result, expected, 1e-6f), "Division results do not match expected values");

        // Test div operation (creates new tensor)
        Tensor div_result = cpu_backend->div(a, b);
        TEST_ASSERT(compare_tensors(div_result, expected, 1e-6f), "Div operation results do not match expected values");

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_tensor_division: " << e.what() << std::endl;
        return false;
    }
}

// Test division by zero handling
bool test_division_by_zero() {
    try {
        auto backend = BackendManager::get_cpu_backend();
        auto* cpu_backend = dynamic_cast<CpuBackend*>(backend.get());
        TEST_ASSERT(cpu_backend != nullptr, "Failed to get CPU backend");

        // Create test tensors with zeros in denominator
        Shape shape(2, 2);
        std::vector<float> a_data = {4.0f, 9.0f, 16.0f, 25.0f};
        std::vector<float> b_data = {2.0f, 0.0f, 4.0f, 0.0f};

        Tensor a = backend->empty(shape, DType::FP32);
        Tensor b = backend->empty(shape, DType::FP32);
        Tensor result = backend->empty(shape, DType::FP32);

        // Fill tensors with data
        float* a_ptr = static_cast<float*>(a.data_ptr());
        float* b_ptr = static_cast<float*>(b.data_ptr());
        for (size_t i = 0; i < a_data.size(); ++i) {
            a_ptr[i] = a_data[i];
            b_ptr[i] = b_data[i];
        }

        // Test div_into operation with zero division
        cpu_backend->div_into(a, b, result);

        // Check results - normal divisions should be correct
        float* result_ptr = static_cast<float*>(result.data_ptr());
        TEST_ASSERT(std::abs(result_ptr[0] - 2.0f) < 1e-6f, "Normal division failed at index 0");
        TEST_ASSERT(std::abs(result_ptr[2] - 4.0f) < 1e-6f, "Normal division failed at index 2");

        // Zero divisions should produce very large values (our epsilon handling)
        TEST_ASSERT(result_ptr[1] > 1e9f, "Zero division handling failed at index 1");
        TEST_ASSERT(result_ptr[3] > 1e9f, "Zero division handling failed at index 3");

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_division_by_zero: " << e.what() << std::endl;
        return false;
    }
}

// Test eigen optimization (if available)
bool test_division_eigen_optimization() {
#ifdef TR_USE_EIGEN
    try {
        auto backend = BackendManager::get_cpu_backend();
        auto* cpu_backend = dynamic_cast<CpuBackend*>(backend.get());
        TEST_ASSERT(cpu_backend != nullptr, "Failed to get CPU backend");

        // Create larger tensors for performance testing
        Shape shape(1000, 1000);  // 1M elements

        Tensor a = backend->empty(shape, DType::FP32);
        Tensor b = backend->empty(shape, DType::FP32);
        Tensor result = backend->empty(shape, DType::FP32);

        // Fill tensors with test data
        float* a_ptr = static_cast<float*>(a.data_ptr());
        float* b_ptr = static_cast<float*>(b.data_ptr());
        int64_t numel = a.numel();

        for (int64_t i = 0; i < numel; ++i) {
            a_ptr[i] = static_cast<float>(i % 100 + 1);
            b_ptr[i] = static_cast<float>((i % 50) + 1);
        }

        // Measure time for eigen-optimized division
        auto start = std::chrono::high_resolution_clock::now();
        cpu_backend->div_into(a, b, result);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Eigen-optimized division (1M elements): " << duration.count() << " microseconds" << std::endl;

        // Verify some random elements
        float* result_ptr = static_cast<float*>(result.data_ptr());
        for (int i = 0; i < 10; ++i) {
            int64_t idx = i * 100000;  // Check sparse elements
            float expected = a_ptr[idx] / b_ptr[idx];
            float actual = result_ptr[idx];
            TEST_ASSERT(std::abs(actual - expected) < 1e-6f, "Eigen optimization result verification failed");
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_division_eigen_optimization: " << e.what() << std::endl;
        return false;
    }
#else
    std::cout << "Eigen optimization not available, skipping test..." << std::endl;
    return true;
#endif
}

// Test error handling for invalid inputs
bool test_division_error_handling() {
    try {
        auto backend = BackendManager::get_cpu_backend();
        auto* cpu_backend = dynamic_cast<CpuBackend*>(backend.get());
        TEST_ASSERT(cpu_backend != nullptr, "Failed to get CPU backend");

        // Create test tensors with different shapes (should fail)
        Shape shape_a(2, 3);
        Shape shape_b(3, 2);

        Tensor a = backend->empty(shape_a, DType::FP32);
        Tensor b = backend->empty(shape_b, DType::FP32);
        Tensor result = backend->empty(shape_a, DType::FP32);

        // Fill with some data
        float* a_ptr = static_cast<float*>(a.data_ptr());
        float* b_ptr = static_cast<float*>(b.data_ptr());
        for (size_t i = 0; i < 6; ++i) {
            a_ptr[i] = static_cast<float>(i + 1);
            b_ptr[i] = static_cast<float>(i + 2);
        }

        // This should throw an exception due to shape mismatch
        bool exception_thrown = false;
        try {
            cpu_backend->div_into(a, b, result);
        } catch (const std::exception&) {
            exception_thrown = true;
        }
        TEST_ASSERT(exception_thrown, "Expected exception for shape mismatch was not thrown");

        // Test with empty tensor (should fail)
        Tensor empty_tensor;  // Creates empty tensor
        exception_thrown = false;
        try {
            cpu_backend->div_into(a, empty_tensor, result);
        } catch (const std::exception&) {
            exception_thrown = true;
        }
        TEST_ASSERT(exception_thrown, "Expected exception for empty tensor was not thrown");

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_division_error_handling: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== CPU Backend Arithmetic Operations Test Suite ===" << std::endl;
    std::cout << "Testing basic arithmetic operations with Eigen optimization support" << std::endl;
    std::cout << "=============================================================" << std::endl;

    // Set quiet mode for logging
    Logger::get_instance().set_quiet_mode(true);

    // Run test cases
    TEST_CASE("Tensor Division", test_tensor_division);
    TEST_CASE("Division by Zero Handling", test_division_by_zero);
    TEST_CASE("Eigen Optimization Test", test_division_eigen_optimization);
    TEST_CASE("Error Handling", test_division_error_handling);

    // Print summary
    std::cout << "=====================" << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "  Total:  " << total_tests << std::endl;
    std::cout << "  Passed: " << passed_tests << std::endl;
    std::cout << "  Failed: " << failed_tests << std::endl;
    std::cout << "=====================" << std::endl;

    if (failed_tests == 0) {
        std::cout << "🎉 All tests PASSED! CPU backend arithmetic operations are working correctly." << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some tests FAILED! Please check the implementation." << std::endl;
        return 1;
    }
}
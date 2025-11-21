/**
 * @file test_cpu_div.cpp
 * @brief CPU backend division operations unit test
 * @details Test for CPU backend division operations (tensor / tensor)
 * @version 1.60.1
 * @date 2025年11月22日
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h
 * @note 所属系列: tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

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

// Test basic tensor division operation
bool test_basic_tensor_division() {
    try {
        auto backend = BackendManager::get_cpu_backend();
        auto* cpu_backend = dynamic_cast<CpuBackend*>(backend.get());
        TEST_ASSERT(cpu_backend != nullptr, "Failed to get CPU backend");

        // Create test tensors: [1,2,3] / [1,2,3] = [1,1,1]
        Shape shape(3);
        std::vector<float> a_data = {1.0f, 2.0f, 3.0f};
        std::vector<float> b_data = {1.0f, 2.0f, 3.0f};
        std::vector<float> expected_data = {1.0f, 1.0f, 1.0f};

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

        // Compare results manually
        float* result_ptr = static_cast<float*>(result.data_ptr());
        for (size_t i = 0; i < expected_data.size(); ++i) {
            if (std::abs(result_ptr[i] - expected_data[i]) > 1e-6f) {
                std::cout << "Mismatch at index " << i << ": " << result_ptr[i] << " vs " << expected_data[i] << std::endl;
                return false;
            }
        }

        // Test div operation (creates new tensor)
        Tensor div_result = cpu_backend->div(a, b);
        float* div_result_ptr = static_cast<float*>(div_result.data_ptr());
        for (size_t i = 0; i < expected_data.size(); ++i) {
            if (std::abs(div_result_ptr[i] - expected_data[i]) > 1e-6f) {
                std::cout << "Div mismatch at index " << i << ": " << div_result_ptr[i] << " vs " << expected_data[i] << std::endl;
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_basic_tensor_division: " << e.what() << std::endl;
        return false;
    }
}

// Test more complex tensor division
bool test_complex_tensor_division() {
    try {
        auto backend = BackendManager::get_cpu_backend();
        auto* cpu_backend = dynamic_cast<CpuBackend*>(backend.get());
        TEST_ASSERT(cpu_backend != nullptr, "Failed to get CPU backend");

        // Create test tensors: [6,8,10] / [2,4,2] = [3,2,5]
        Shape shape(3);
        std::vector<float> a_data = {6.0f, 8.0f, 10.0f};
        std::vector<float> b_data = {2.0f, 4.0f, 2.0f};
        std::vector<float> expected_data = {3.0f, 2.0f, 5.0f};

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

        // Compare results manually
        float* result_ptr = static_cast<float*>(result.data_ptr());
        for (size_t i = 0; i < expected_data.size(); ++i) {
            if (std::abs(result_ptr[i] - expected_data[i]) > 1e-6f) {
                std::cout << "Complex division mismatch at index " << i << ": " << result_ptr[i] << " vs " << expected_data[i] << std::endl;
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_complex_tensor_division: " << e.what() << std::endl;
        return false;
    }
}

// Test division by zero handling
bool test_division_by_zero() {
    try {
        auto backend = BackendManager::get_cpu_backend();
        auto* cpu_backend = dynamic_cast<CpuBackend*>(backend.get());
        TEST_ASSERT(cpu_backend != nullptr, "Failed to get CPU backend");

        // Create test tensors with zeros in denominator: [4,9,16] / [2,0,4] = [2,very_large,4]
        Shape shape(3);
        std::vector<float> a_data = {4.0f, 9.0f, 16.0f};
        std::vector<float> b_data = {2.0f, 0.0f, 4.0f};

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

        // Check results
        float* result_ptr = static_cast<float*>(result.data_ptr());
        TEST_ASSERT(std::abs(result_ptr[0] - 2.0f) < 1e-6f, "Normal division failed at index 0");
        TEST_ASSERT(std::abs(result_ptr[2] - 4.0f) < 1e-6f, "Normal division failed at index 2");

        // Zero division should produce very large values (our epsilon handling)
        TEST_ASSERT(result_ptr[1] > 1e9f, "Zero division handling failed at index 1");

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_division_by_zero: " << e.what() << std::endl;
        return false;
    }
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

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in test_division_error_handling: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== CPU Backend Division Operations Test Suite ===" << std::endl;
    std::cout << "Testing tensor division operations with Eigen optimization support" << std::endl;
    std::cout << "=========================================================" << std::endl;

    // Set quiet mode for logging
    Logger::get_instance().set_quiet_mode(true);

    // Run test cases
    TEST_CASE("Basic Tensor Division", test_basic_tensor_division);
    TEST_CASE("Complex Tensor Division", test_complex_tensor_division);
    TEST_CASE("Division by Zero Handling", test_division_by_zero);
    TEST_CASE("Error Handling", test_division_error_handling);

    // Print summary
    std::cout << "=====================" << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "  Total:  " << total_tests << std::endl;
    std::cout << "  Passed: " << passed_tests << std::endl;
    std::cout << "  Failed: " << failed_tests << std::endl;
    std::cout << "=====================" << std::endl;

    if (failed_tests == 0) {
        std::cout << "All tests PASSED! CPU backend division operations are working correctly." << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED! Please check the implementation." << std::endl;
        return 1;
    }
}
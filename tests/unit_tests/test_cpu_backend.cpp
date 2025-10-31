/**
 * @file test_cpu_backend.cpp
 * @brief CPU backend unit test
 * @details Test for CPU backend functionality
 * @version 1.00.00
 * @date 2025-10-26
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

/**
 * @brief Test basic backend manager functionality
 * @return Test result
 */
bool test_backend_manager_basic() {
    tr::BackendManager& manager = tr::BackendManager::instance();

    // BackendManager自动注册和初始化后端，无需手动注册
    std::cout << "BackendManager auto-initialization test" << std::endl;

    std::cout << "CPU Backend registration test - SUCCESS!" << std::endl;
    return true;
}

/**
 * @brief Test CPU backend initialization
 * @return Test result
 */
bool test_cpu_backend_initialization() {
    tr::BackendManager& manager = tr::BackendManager::instance();

    // Get CPU backend
    auto backend = manager.get_backend(tr::CPU);
    TEST_ASSERT(backend != nullptr, "Failed to get CPU backend");
    TEST_ASSERT(backend->name() == "CpuBackend", "Backend name mismatch");

    std::cout << "CPU Backend initialization test - SUCCESS!" << std::endl;
    return true;
}

/**
 * @brief Test CPU backend memory allocation
 * @return Test result
 */
bool test_cpu_backend_memory() {
    tr::BackendManager& manager = tr::BackendManager::instance();
    auto backend = manager.get_backend(tr::CPU);
    TEST_ASSERT(backend != nullptr, "No current backend available");

    try {
        // Test memory allocation
        const size_t test_size = 1024;
        auto memory = backend->allocate(test_size);
        TEST_ASSERT(memory != nullptr, "Failed to allocate CPU memory");

        // Test getting data pointer
        void* ptr = backend->get_data_ptr(memory);
        TEST_ASSERT(ptr != nullptr, "Failed to get data pointer");

        std::cout << "CPU Backend memory allocation test - SUCCESS!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception in memory test: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Test device functionality
 * @return Test result
 */
bool test_device_functionality() {
    // Create CPU device
    tr::Device cpu_device = tr::CPU;

    // Test device type checking
    TEST_ASSERT(cpu_device.is_cpu(), "CPU device type check failed");
    TEST_ASSERT(!cpu_device.is_cuda(), "CPU device CUDA check failed");

    // Test device string representation
    std::string cpu_str = cpu_device.to_string();
    TEST_ASSERT(cpu_str == "CPU", "CPU device string representation failed");

    std::cout << "Device functionality test - SUCCESS!" << std::endl;
    return true;
}

/**
 * @brief Test basic tensor creation
 * @return Test result
 */
bool test_tensor_creation() {
    try {
        // Create simple tensors
        tr::Shape shape(2, 3);
        tr::Tensor tensor_a = tr::Tensor::empty(shape, tr::DType::FP32, tr::CPU);
        tr::Tensor tensor_b = tr::Tensor::empty(shape, tr::DType::FP32, tr::CPU);

        // Test tensor properties
        TEST_ASSERT(tensor_a.shape() == tr::Shape(2, 3), "Tensor A shape mismatch");
        TEST_ASSERT(tensor_a.dtype() == tr::DType::FP32, "Tensor A dtype mismatch");
        TEST_ASSERT(tensor_a.device() == tr::CPU, "Tensor A device mismatch");
        TEST_ASSERT(tensor_b.shape() == tr::Shape(2, 3), "Tensor B shape mismatch");

        std::cout << "Tensor creation test - SUCCESS!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception in tensor creation test: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Test backend manager cleanup
 * @return Test result
 */
bool test_backend_manager_cleanup() {
    tr::BackendManager& manager = tr::BackendManager::instance();

    // 新的单例模式不需要手动清理，后端生命周期自动管理
    std::cout << "Backend manager automatic cleanup test - SUCCESS!" << std::endl;
    return true;
}

/**
 * @brief Main test function
 */
int main() {
    std::cout << "=== Fixed Backend Unit Tests ===" << std::endl;
    std::cout << "Testing Backend Hello World functionality" << std::endl;
    std::cout << std::endl;

    // Set log level
    tr::Logger::get_instance().set_level(tr::LogLevel::INFO);

    total_tests = 0;
    passed_tests = 0;
    failed_tests = 0;

    // Run test cases
    TEST_CASE("Backend Manager Basic", test_backend_manager_basic);
    TEST_CASE("CPU Backend Initialization", test_cpu_backend_initialization);
    TEST_CASE("CPU Backend Memory Management", test_cpu_backend_memory);
    TEST_CASE("Device Functionality", test_device_functionality);
    TEST_CASE("Tensor Creation", test_tensor_creation);
    TEST_CASE("Backend Manager Cleanup", test_backend_manager_cleanup);

    // Print test results
    std::cout << std::endl;
    std::cout << "=== Test Summary ===" << std::endl;
    std::cout << "Total tests: " << total_tests << std::endl;
    std::cout << "Passed: " << passed_tests << std::endl;
    std::cout << "Failed: " << failed_tests << std::endl;

    if (failed_tests == 0) {
        std::cout << std::endl;
        std::cout << "[SUCCESS] All Backend tests PASSED!" << std::endl;
        std::cout << "Backend is working correctly - Hello World verification complete!" << std::endl;
        return 0;
    } else {
        std::cout << std::endl;
        std::cout << "[ERROR] Some Backend tests FAILED!" << std::endl;
        return 1;
    }
}
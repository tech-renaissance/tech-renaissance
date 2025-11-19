/**
 * @file test_cuda_backend_simple.cpp
 * @brief 简单的CUDA后端测试 - 通过框架测试
 * @details 通过BackendManager测试CUDA后端，避免CUDA头文件污染
 * @version 1.00.00
 * @date 2025-10-26
 * @author 技术觉醒团队
 * @note 依赖项: backend_manager.h, 无CUDA头文件污染
 * @note 所属系列: tests
 */

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include "tech_renaissance/data/dtype.h"
#include "tech_renaissance/data/device.h"
#include "tech_renaissance/data/shape.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/backend/backend.h"
#include "tech_renaissance/utils/tr_exception.h"

// Simple test framework
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "ASSERTION FAILED: " << message << std::endl; \
            return false; \
        } \
    } while(0)

/**
 * @brief 测试CUDA后端可用性
 */
bool test_cuda_backend_availability() {
    try {
        tr::BackendManager& manager = tr::BackendManager::instance();

        // 尝试获取CUDA后端
        auto cuda_backend = manager.get_backend(tr::CUDA[0]);

        if (cuda_backend) {
            TEST_ASSERT(cuda_backend->name() == "CudaBackend", "CUDA backend name mismatch");
            std::cout << "CUDA backend available: " << cuda_backend->name() << std::endl;
            std::cout << "CUDA backend device: " << cuda_backend->device().to_string() << std::endl;
            return true;
        } else {
            std::cout << "CUDA backend not available - this is expected if CUDA is not properly configured" << std::endl;
            return false; // 这是测试失败的情况
        }

    } catch (const std::exception& e) {
        std::cout << "CUDA backend not available: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试CUDA后端内存分配
 */
bool test_cuda_memory_allocation() {
    try {
        tr::BackendManager& manager = tr::BackendManager::instance();
        auto backend = manager.get_backend(tr::CUDA[0]);

        if (!backend) {
            std::cout << "CUDA backend not available - skipping memory test" << std::endl;
            return false;
        }

        // 测试内存分配
        const size_t test_size = 1024 * 1024; // 1MB
        auto memory = backend->allocate(test_size);
        TEST_ASSERT(memory != nullptr, "Failed to allocate CUDA memory");

        // 测试获取数据指针
        void* ptr = backend->get_data_ptr(memory);
        TEST_ASSERT(ptr != nullptr, "Failed to get CUDA data pointer");

        std::cout << "CUDA memory allocation test - SUCCESS!" << std::endl;
        std::cout << "Allocated: " << test_size << " bytes" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception in CUDA memory test: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试CUDA张量创建
 */
bool test_cuda_tensor_creation() {
    try {
        tr::BackendManager& manager = tr::BackendManager::instance();
        auto backend = manager.get_backend(tr::CUDA[0]);

        if (!backend) {
            std::cout << "CUDA backend not available - skipping tensor test" << std::endl;
            return false;
        }

        // 创建CUDA张量
        tr::Shape shape(2, 3, 4, 5);
        tr::Tensor tensor_a = tr::Tensor::empty(shape, tr::DType::FP32, tr::CUDA[0]);
        tr::Tensor tensor_b = tr::Tensor::empty(shape, tr::DType::FP32, tr::CUDA[0]);

        // 测试张量属性
        TEST_ASSERT(tensor_a.shape() == tr::Shape(2, 3, 4, 5), "Tensor A shape mismatch");
        TEST_ASSERT(tensor_a.dtype() == tr::DType::FP32, "Tensor A dtype mismatch");
        TEST_ASSERT(tensor_a.device() == tr::CUDA[0], "Tensor A device mismatch");
        TEST_ASSERT(tensor_b.shape() == tr::Shape(2, 3, 4, 5), "Tensor B shape mismatch");

        std::cout << "CUDA tensor creation test - SUCCESS!" << std::endl;
        std::cout << "Tensor shape: " << tensor_a.shape().to_string() << std::endl;
        std::cout << "Total elements: " << tensor_a.numel() << std::endl;
        std::cout << "Memory size: " << tensor_a.memory_size() << " bytes" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception in CUDA tensor test: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试CUDA基本操作（fill, add）
 */
bool test_cuda_basic_operations() {
    try {
        tr::BackendManager& manager = tr::BackendManager::instance();
        auto backend = manager.get_backend(tr::CUDA[0]);

        if (!backend) {
            std::cout << "CUDA backend not available - skipping operations test" << std::endl;
            return false;
        }

        // 创建测试张量
        tr::Shape shape(2, 2);
        tr::Tensor tensor_a = tr::Tensor::empty(shape, tr::DType::FP32, tr::CUDA[0]);
        tr::Tensor tensor_b = tr::Tensor::empty(shape, tr::DType::FP32, tr::CUDA[0]);
        tr::Tensor result = tr::Tensor::empty(shape, tr::DType::FP32, tr::CUDA[0]);

        // 测试fill操作
        backend->fill(tensor_a, 2.0f);
        backend->fill(tensor_b, 3.0f);

        std::cout << "CUDA fill operations completed" << std::endl;

        // 测试add操作
        backend->add_into(tensor_a, tensor_b, result);

        std::cout << "CUDA add operation completed" << std::endl;
        std::cout << "CUDA basic operations test - SUCCESS!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception in CUDA operations test: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== Simple CUDA Backend Test ===" << std::endl;
    std::cout << "Testing CUDA backend through framework (no CUDA headers)" << std::endl;
    std::cout << std::endl;

    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;

    auto run_test = [&](const std::string& name, auto test_func) {
        total_tests++;
        std::cout << "Running test: " << name << "..." << std::endl;
        if (test_func()) {
            std::cout << "[PASS] " << name << " PASSED" << std::endl;
            passed_tests++;
        } else {
            std::cout << "[FAIL] " << name << " FAILED" << std::endl;
            failed_tests++;
        }
        std::cout << std::endl;
    };

    // 运行测试
    run_test("CUDA Backend Availability", test_cuda_backend_availability);
    run_test("CUDA Memory Allocation", test_cuda_memory_allocation);
    run_test("CUDA Tensor Creation", test_cuda_tensor_creation);
    run_test("CUDA Basic Operations", test_cuda_basic_operations);

    // 输出结果
    std::cout << "=== Test Summary ===" << std::endl;
    std::cout << "Total tests: " << total_tests << std::endl;
    std::cout << "Passed: " << passed_tests << std::endl;
    std::cout << "Failed: " << failed_tests << std::endl;

    if (failed_tests == 0) {
        std::cout << "[SUCCESS] All CUDA Backend tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "[FAILURE] Some CUDA Backend tests FAILED!" << std::endl;
        return 1;
    }
}
/**
 * @file test_backend_refactor.cpp
 * @brief 测试Backend基类重构
 * @details 验证Backend基类从抽象类改为可实例化类但抛出异常的功能
 * @version 1.43.0
 * @date 2025-11-16
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h
 * @note 所属系列: unit_tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <stdexcept>

using namespace tr;

int main() {
    std::cout << "=== Testing Backend Refactor ===" << std::endl;

    try {
        // 测试1: 验证Backend基类无法直接实例化
        std::cout << "Test 1: Testing Backend direct instantiation..." << std::endl;
        try {
            Backend backend;  // 这应该抛出异常
            std::cout << "ERROR: Backend instantiation should have failed!" << std::endl;
            return 1;
        } catch (const TRException& e) {
            std::cout << "PASS: Backend instantiation correctly failed with: " << e.what() << std::endl;
        }

        // 测试2: 验证CPU后端可以正常实例化
        std::cout << "\nTest 2: Testing CpuBackend instantiation..." << std::endl;
        try {
            auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(
                BackendManager::instance().get_backend(CPU));
            if (cpu_backend) {
                std::cout << "PASS: CpuBackend instantiation successful" << std::endl;
                std::cout << "Backend name: " << cpu_backend->name() << std::endl;
            } else {
                std::cout << "ERROR: Failed to get CpuBackend instance" << std::endl;
                return 1;
            }
        } catch (const std::exception& e) {
            std::cout << "ERROR: CpuBackend instantiation failed: " << e.what() << std::endl;
            return 1;
        }

        // 测试3: 验证CPU后端可以调用新增的方法
        std::cout << "\nTest 3: Testing new Backend methods..." << std::endl;
        try {
            auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(
                BackendManager::instance().get_backend(CPU));

            // 创建测试张量
            Shape shape(2, 3, 4, 5);
            Tensor test_tensor = cpu_backend->ones(shape, DType::FP32);

            // 测试reshape方法
            Shape new_shape(2, 3, 20);
            Tensor reshaped = cpu_backend->reshape(test_tensor, new_shape);
            std::cout << "PASS: reshape method works" << std::endl;

            // 测试tanh方法
            Tensor tanh_result = cpu_backend->tanh(test_tensor);
            std::cout << "PASS: tanh method works" << std::endl;

            // 测试标量运算
            Tensor scalar_result = cpu_backend->minus(test_tensor, 0.5f);
            std::cout << "PASS: scalar minus method works" << std::endl;

            // 测试广播运算
            Shape shape2(2, 3, 4, 5);
            Tensor test_tensor2 = cpu_backend->ones(shape2, DType::FP32);
            Tensor broadcast_result = cpu_backend->add_broadcast(test_tensor, test_tensor2);
            std::cout << "PASS: broadcast add method works" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "ERROR: New method test failed: " << e.what() << std::endl;
            return 1;
        }

        std::cout << "\n=== All Backend Refactor Tests Passed! ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cout << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
}
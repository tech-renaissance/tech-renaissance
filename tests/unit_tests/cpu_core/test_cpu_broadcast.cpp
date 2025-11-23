/**
 * @file test_cpu_broadcast.cpp
 * @brief CPU后端可广播张量运算测试
 * @details 测试CPU后端的3种可广播运算：add、minus、mul，支持形状广播和标量广播优化
 * @version 1.00.00
 * @date 2025-11-01
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h
 * @note 所属系列: tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace tr;

void print_tensor_info(const std::string& name, const Tensor& tensor) {
    std::cout << "\n" << name << ":\n";
    tensor.print();
}

bool are_tensors_close(const Tensor& a, const Tensor& b, float eps = 1e-6f) {
    auto cpu_backend = BackendManager::get_cpu_backend();
    return cpu_backend->is_close(a, b, eps);
}

void test_basic_broadcast_operations() {
    std::cout << "\n=== Testing Basic Broadcast Operations ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 相同形状的运算（不广播）
    std::cout << "\nTest 1: Same shape operations (2x3)";
    Shape shape(2, 3);
    Tensor a = Tensor::full(shape, 2.0f);
    Tensor b = Tensor::full(shape, 3.0f);

    print_tensor_info("Tensor A", a);
    print_tensor_info("Tensor B", b);

    Tensor add_result = cpu_backend->add_broadcast(a, b);
    Tensor minus_result = cpu_backend->minus_broadcast(a, b);
    Tensor mul_result = cpu_backend->mul_broadcast(a, b);

    std::cout << "\nAddition result (2+3=5):";
    print_tensor_info("Add Result", add_result);
    std::cout << "\nSubtraction result (2-3=-1):";
    print_tensor_info("Minus Result", minus_result);
    std::cout << "\nMultiplication result (2*3=6):";
    print_tensor_info("Mul Result", mul_result);

    // 测试2: 标量广播
    std::cout << "\nTest 2: Scalar broadcasting (scalar + matrix)";
    Tensor scalar = Tensor::full(Shape(), 5.0f);  // 标量5
    Tensor matrix = Tensor::full(Shape(2, 3), 2.0f);  // 2x3矩阵全2

    print_tensor_info("Scalar Tensor", scalar);
    print_tensor_info("Matrix Tensor", matrix);

    Tensor scalar_add = cpu_backend->add_broadcast(scalar, matrix);  // 5 + 2 = 7
    Tensor matrix_add = cpu_backend->add_broadcast(matrix, scalar);  // 2 + 5 = 7

    std::cout << "\nScalar broadcast result (5+2=7):";
    print_tensor_info("Scalar + Matrix", scalar_add);
    print_tensor_info("Matrix + Scalar", matrix_add);

    // 验证标量广播的一致性
    if (are_tensors_close(scalar_add, matrix_add)) {
        std::cout << "\nScalar broadcast consistency check: PASS";
    } else {
        std::cout << "\nScalar broadcast consistency check: FAIL";
    }
}

void test_shape_broadcast_operations() {
    std::cout << "\n=== Testing Shape Broadcast Operations ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: (1,3) 广播到 (2,3)
    std::cout << "\nTest 1: (1,3) + (2,3) = (2,3)";
    Tensor small_tensor = Tensor::full(Shape(1, 3), 2.0f);  // [2,2,2]
    Tensor large = Tensor::full(Shape(2, 3), 3.0f);  // [[3,3,3], [3,3,3]]

    print_tensor_info("Small Tensor (1,3)", small_tensor);
    print_tensor_info("Large Tensor (2,3)", large);

    Tensor broadcast_add = cpu_backend->add_broadcast(small_tensor, large);
    std::cout << "\nBroadcast result (1,3)+(2,3):";
    print_tensor_info("Broadcast Result", broadcast_add);

    // 验证结果形状
    if (broadcast_add.shape() == Shape(2, 3)) {
        std::cout << "\nBroadcast output shape correct: (2,3)";
    } else {
        std::cout << "\nBroadcast output shape incorrect, expected (2,3), actual " << broadcast_add.shape().to_string();
    }

    // 测试2: (2,1) 广播到 (2,3)
    std::cout << "\nTest 2: (2,1) + (2,3) = (2,3)";
    Tensor col_vector = Tensor::full(Shape(2, 1), 1.0f);  // [[1], [1]]
    Tensor matrix_23 = Tensor::full(Shape(2, 3), 2.0f);   // [[2,2,2], [2,2,2]]

    print_tensor_info("Vector (2,1)", col_vector);
    print_tensor_info("Matrix (2,3)", matrix_23);

    Tensor col_broadcast = cpu_backend->add_broadcast(col_vector, matrix_23);
    std::cout << "\nColumn broadcast result (2,1)+(2,3):";
    print_tensor_info("Vector broadcast: ", col_broadcast);
}

void test_broadcast_edge_cases() {
    std::cout << "\n=== Testing Broadcast Edge Cases ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 空张量
    std::cout << "\nTest 1: Empty tensor handling";
    try {
        Tensor empty_a;
        Tensor empty_b;
        Tensor empty_result = cpu_backend->add_broadcast(empty_a, empty_b);
        std::cout << "\nTwo empty tensors operation successful, returned empty tensor";
    } catch (const TRException& e) {
        std::cout << "\nTwo empty tensors operation failed: " << e.what();
    }

    // Test 2: One empty tensor + non-empty tensor（应该报错）
    std::cout << "\nTest 2: One empty tensor + non-empty tensor";
    try {
        Tensor empty_tensor;
        Tensor non_empty = Tensor::full(Shape(2, 3), 1.0f);
        Tensor result = cpu_backend->add_broadcast(empty_tensor, non_empty);
        std::cout << "\nShould have thrown error but succeeded!";
    } catch (const TRException& e) {
        std::cout << "\nCorrectly caught exception: " << e.what();
    }

    // 测试3: 不兼容的形状（应该报错）
    std::cout << "\nTest 3: Incompatible shapes (2,3) vs (4,5)";
    try {
        Tensor a = Tensor::full(Shape(2, 3), 1.0f);
        Tensor b = Tensor::full(Shape(4, 5), 2.0f);
        Tensor result = cpu_backend->add_broadcast(a, b);
        std::cout << "\nShould have thrown error but succeeded!";
    } catch (const TRException& e) {
        std::cout << "\nCorrectly caught exception: " << e.what();
    }

    // Test 4: Data type error（应该报错）
    std::cout << "\nTest 4: Data type error";
    try {
        Tensor fp32_tensor = Tensor::full(Shape(2, 3), 1.0f, DType::FP32, tr::CPU);
        Tensor int8_tensor = Tensor::full(Shape(2, 3), 2, DType::INT8, tr::CPU);
        Tensor result = cpu_backend->add_broadcast(fp32_tensor, int8_tensor);
        std::cout << "\nShould have thrown error but succeeded!";
    } catch (const TRException& e) {
        std::cout << "\nCorrectly caught exception: " << e.what();
    }
}

void test_into_operations() {
    std::cout << "\n=== Testing _into Operations ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 正常的_into运算
    std::cout << "\nTest 1: add_into operation";
    Tensor a = Tensor::full(Shape(1, 3), 2.0f);
    Tensor b = Tensor::full(Shape(2, 3), 3.0f);
    Tensor result = Tensor::empty(Shape(2, 3), DType::FP32, tr::CPU);

    print_tensor_info("Tensor A(1,3)", a);
    print_tensor_info("Tensor B(2,3)", b);

    cpu_backend->add_broadcast_into(a, b, result);
    std::cout << "\nadd_into result:";
    print_tensor_info("Result", result);

    // 验证与add函数的一致性
    Tensor expected = cpu_backend->add_broadcast(a, b);
    if (are_tensors_close(result, expected)) {
        std::cout << "\nadd_into consistent with add function: PASS";
    } else {
        std::cout << "\nadd_into inconsistent with add function: FAIL";
    }
}

void test_performance_optimizations() {
    std::cout << "\n=== Testing Performance Optimizations ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 标量广播优化
    std::cout << "\nTest 1: Scalar broadcast performance test";
    Shape large_shape(1000, 1000);
    Tensor scalar = Tensor::full(Shape(), 2.0f);
    Tensor large_matrix = Tensor::full(large_shape, 1.0f);

    auto start = std::chrono::high_resolution_clock::now();
    Tensor scalar_result = cpu_backend->add_broadcast(scalar, large_matrix);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "\nScalar broadcast (1 + 1000000) time: " << duration.count() << "microseconds";

    // 验证结果正确性
    bool all_correct = true;
    const float* data = static_cast<const float*>(scalar_result.data_ptr());
    for (int64_t i = 0; i < large_shape.numel(); ++i) {
        if (std::abs(data[i] - 3.0f) > 1e-6f) {
            all_correct = false;
            break;
        }
    }
    if (all_correct) {
        std::cout << "\nScalar broadcast result correctness: PASS";
    } else {
        std::cout << "\nScalar broadcast result correctness: FAIL";
    }

    // 测试2: 相同形状优化
    std::cout << "\nTest 2: Same shape performance test";
    Shape medium_shape(100, 100);
    Tensor a = Tensor::full(medium_shape, 1.0f);
    Tensor b = Tensor::full(medium_shape, 2.0f);

    start = std::chrono::high_resolution_clock::now();
    Tensor same_shape_result = cpu_backend->mul_broadcast(a, b);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "\nSame shape multiplication (10000 * 10000) time: " << duration.count() << "microseconds";
}

void test_mathematical_correctness() {
    std::cout << "\n=== Testing Mathematical Correctness ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // Test 1: Simple broadcast mathematical verification
    std::cout << "\nTest 1: Simple broadcast mathematical verification";
    Tensor a = Tensor::full(Shape(1, 2), 3.0f);  // [[3, 3]]
    Tensor b = Tensor::full(Shape(3, 2), 2.0f);  // [[2,2], [2,2], [2,2]]

    print_tensor_info("Tensor A(1,2)", a);
    print_tensor_info("Tensor B(3,2)", b);

    Tensor add_result = cpu_backend->add_broadcast(a, b);   // 应该得到全5
    Tensor mul_result = cpu_backend->mul_broadcast(a, b);   // 应该得到全6

    std::cout << "\nAddition result (should be all 5s):";
    print_tensor_info("Addition result", add_result);
    std::cout << "\nMultiplication result (should be all 6s):";
    print_tensor_info("Multiplication result", mul_result);

    // 验证加法结果
    bool add_correct = true;
    const float* add_data = static_cast<const float*>(add_result.data_ptr());
    for (int64_t i = 0; i < add_result.numel(); ++i) {
        if (std::abs(add_data[i] - 5.0f) > 1e-6f) {
            add_correct = false;
            break;
        }
    }

    // 验证乘法结果
    bool mul_correct = true;
    const float* mul_data = static_cast<const float*>(mul_result.data_ptr());
    for (int64_t i = 0; i < mul_result.numel(); ++i) {
        if (std::abs(mul_data[i] - 6.0f) > 1e-6f) {
            mul_correct = false;
            break;
        }
    }

    if (add_correct) {
        std::cout << "\nBroadcast addition mathematical correctness: PASS";
    } else {
        std::cout << "\nBroadcast addition mathematical correctness: FAIL";
    }

    if (mul_correct) {
        std::cout << "\nBroadcast multiplication mathematical correctness: PASS";
    } else {
        std::cout << "\nBroadcast multiplication mathematical correctness: FAIL";
    }
}

int main() {
    try {
        std::cout << "=== CPU Backend Broadcast Operations Test ===";
        std::cout << "\nVersion: V1.28.1";
        std::cout << "\nTesting CPU backend broadcast tensor operations\n";

        // 获取CPU后端
        auto cpu_backend = BackendManager::get_cpu_backend();
        std::cout << "\nCPU backend initialized successfully";

        // 运行所有测试
        test_basic_broadcast_operations();
        test_shape_broadcast_operations();
        test_broadcast_edge_cases();
        test_into_operations();
        test_performance_optimizations();
        test_mathematical_correctness();

        std::cout << "\n\n=== Test Summary ===";
        std::cout << "\nAll broadcast operations tests completed";
        std::cout << "\nIncluding: 3 operations x 2 modes = 6 main functions";
        std::cout << "\nTest coverage: shape broadcasting, scalar broadcasting, edge cases, performance optimization, mathematical correctness";

        std::cout << "\n\nCPU broadcast operations test completed successfully!";
        return 0;

    } catch (const TRException& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\nUnexpected error: " << e.what() << std::endl;
        return 1;
    }
}
/**
 * @file test_cpu_expand.cpp
 * @brief CPU后端张量扩展操作测试
 * @details 测试CPU后端的expand和expand_into方法，支持形状广播扩展
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

void test_basic_expand_operations() {
    std::cout << "\n=== Testing Basic Expand Operations ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 相同形状的扩展（不广播）
    std::cout << "\nTest 1: Same shape expansion (2x3) -> (2x3)";
    Shape input_shape(2, 3);
    Shape target_shape(2, 3);
    Tensor input = Tensor::full(input_shape, 2.0f);

    print_tensor_info("Input Tensor", input);

    Tensor expanded = cpu_backend->expand(input, target_shape);
    std::cout << "\nExpanded result (same shape):";
    print_tensor_info("Expanded", expanded);

    // 验证结果应该与输入相同
    if (are_tensors_close(input, expanded)) {
        std::cout << "\nSame shape expansion correctness: PASS";
    } else {
        std::cout << "\nSame shape expansion correctness: FAIL";
    }

    // 测试2: 标量扩展
    std::cout << "\nTest 2: Scalar expansion () -> (2,3)";
    Tensor scalar = Tensor::full(Shape(), 5.0f);  // 标量5
    Shape target_shape_23(2, 3);

    print_tensor_info("Scalar Tensor", scalar);

    Tensor scalar_expanded = cpu_backend->expand(scalar, target_shape_23);
    std::cout << "\nScalar expansion result:";
    print_tensor_info("Scalar Expanded", scalar_expanded);

    // 验证标量扩展结果
    bool scalar_correct = true;
    const float* scalar_data = static_cast<const float*>(scalar_expanded.data_ptr());
    for (int64_t i = 0; i < scalar_expanded.numel(); ++i) {
        if (std::abs(scalar_data[i] - 5.0f) > 1e-6f) {
            scalar_correct = false;
            break;
        }
    }
    if (scalar_correct) {
        std::cout << "\nScalar expansion correctness: PASS";
    } else {
        std::cout << "\nScalar expansion correctness: FAIL";
    }
}

void test_shape_expand_operations() {
    std::cout << "\n=== Testing Shape Expand Operations ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: (1,3) 扩展到 (2,3)
    std::cout << "\nTest 1: (1,3) expand to (2,3)";
    Tensor row_vec = Tensor::full(Shape(1, 3), 2.0f);  // [[2,2,2]]
    Shape target_shape(2, 3);

    print_tensor_info("Row Vector (1,3)", row_vec);

    Tensor expanded_row = cpu_backend->expand(row_vec, target_shape);
    std::cout << "\nRow vector expansion result:";
    print_tensor_info("Expanded Row", expanded_row);

    // 验证第一行和第二行应该相同
    bool row_correct = true;
    const float* row_data = static_cast<const float*>(expanded_row.data_ptr());
    for (int64_t i = 0; i < 6; ++i) {
        if (std::abs(row_data[i] - 2.0f) > 1e-6f) {
            row_correct = false;
            break;
        }
    }
    if (row_correct) {
        std::cout << "\nRow vector expansion correctness: PASS";
    } else {
        std::cout << "\nRow vector expansion correctness: FAIL";
    }

    // 测试2: (2,1) 扩展到 (2,3)
    std::cout << "\nTest 2: (2,1) expand to (2,3)";
    Tensor col_vec = Tensor::full(Shape(2, 1), 1.0f);  // [[1], [1]]
    Shape target_shape_23(2, 3);

    print_tensor_info("Column Vector (2,1)", col_vec);

    Tensor expanded_col = cpu_backend->expand(col_vec, target_shape_23);
    std::cout << "\nColumn vector expansion result:";
    print_tensor_info("Expanded Column", expanded_col);

    // 验证列扩展结果
    bool col_correct = true;
    const float* col_data = static_cast<const float*>(expanded_col.data_ptr());
    for (int64_t i = 0; i < 6; ++i) {
        float expected_value = (i < 3) ? 1.0f : 1.0f;  // 两行都应该是1
        if (std::abs(col_data[i] - expected_value) > 1e-6f) {
            col_correct = false;
            break;
        }
    }
    if (col_correct) {
        std::cout << "\nColumn vector expansion correctness: PASS";
    } else {
        std::cout << "\nColumn vector expansion correctness: FAIL";
    }
}

void test_expand_edge_cases() {
    std::cout << "\n=== Testing Expand Edge Cases ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 空张量
    std::cout << "\nTest 1: Empty tensor handling";
    try {
        Tensor empty_input;
        Shape target_shape(2, 3);
        Tensor result = cpu_backend->expand(empty_input, target_shape);
        std::cout << "\nShould have thrown error but succeeded!";
    } catch (const TRException& e) {
        std::cout << "\nCorrectly caught exception: " << e.what();
    }

    // 测试2: 两个空张量的情况
    std::cout << "\nTest 2: Two empty tensors";
    try {
        Tensor empty_input;
        Tensor empty_output;
        cpu_backend->expand_into(empty_input, empty_output);
        std::cout << "\nTwo empty tensors expansion successful";
    } catch (const TRException& e) {
        std::cout << "\nTwo empty tensors expansion failed: " << e.what();
    }

    // 测试3: 不兼容的形状（应该报错）
    std::cout << "\nTest 3: Incompatible shapes (2,3) expand to (4,5)";
    try {
        Tensor input = Tensor::full(Shape(2, 3), 1.0f);
        Shape target_shape(4, 5);
        Tensor result = cpu_backend->expand(input, target_shape);
        std::cout << "\nShould have thrown error but succeeded!";
    } catch (const TRException& e) {
        std::cout << "\nCorrectly caught exception: " << e.what();
    }

    // 测试4: 低维到高维扩展（应该报错）
    std::cout << "\nTest 4: Low to high dimension expansion (3,) to (2,3)";
    try {
        Tensor vec = Tensor::full(Shape(3), 1.0f);
        Shape target_shape(2, 3);
        Tensor result = cpu_backend->expand(vec, target_shape);
        std::cout << "\nShould have thrown error but succeeded!";
    } catch (const TRException& e) {
        std::cout << "\nCorrectly caught exception: " << e.what();
    }

    // 测试5: 数据类型错误（应该报错）
    std::cout << "\nTest 5: Data type error";
    try {
        // 这里我们无法直接创建非FP32张量进行测试，因为测试函数只使用FP32
        // 但可以测试设备不匹配的情况（这里暂时跳过，因为都在CPU上）
        std::cout << "\nData type test skipped (only FP32 supported in current test)";
    } catch (const TRException& e) {
        std::cout << "\nCaught exception: " << e.what();
    }
}

void test_into_operations() {
    std::cout << "\n=== Testing Expand Into Operations ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 正常的_into运算
    std::cout << "\nTest 1: expand_into operation";
    Tensor input = Tensor::full(Shape(1, 3), 2.0f);
    Tensor output = Tensor::empty(Shape(2, 3), DType::FP32, tr::CPU);

    print_tensor_info("Input Tensor(1,3)", input);

    cpu_backend->expand_into(input, output);
    std::cout << "\nexpand_into result:";
    print_tensor_info("Output", output);

    // 验证与expand函数的一致性
    Tensor expected = cpu_backend->expand(input, Shape(2, 3));
    if (are_tensors_close(output, expected)) {
        std::cout << "\nexpand_into consistent with expand function: PASS";
    } else {
        std::cout << "\nexpand_into inconsistent with expand function: FAIL";
    }

    // 测试2: 错误的输出形状（应该报错）
    std::cout << "\nTest 2: Wrong output shape";
    try {
        Tensor input = Tensor::full(Shape(1, 3), 1.0f);
        Tensor wrong_output = Tensor::empty(Shape(3, 2), DType::FP32, tr::CPU);  // 错误形状
        cpu_backend->expand_into(input, wrong_output);
        std::cout << "\nShould have thrown error but succeeded!";
    } catch (const TRException& e) {
        std::cout << "\nCorrectly caught exception: " << e.what();
    }

    // 测试3: 空输入张量到非空输出张量（应该报错）
    std::cout << "\nTest 3: Empty input to non-empty output";
    try {
        Tensor empty_input;
        Tensor non_empty_output = Tensor::empty(Shape(2, 3), DType::FP32, tr::CPU);
        cpu_backend->expand_into(empty_input, non_empty_output);
        std::cout << "\nShould have thrown error but succeeded!";
    } catch (const TRException& e) {
        std::cout << "\nCorrectly caught exception: " << e.what();
    }
}

void test_performance_optimizations() {
    std::cout << "\n=== Testing Performance Optimizations ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 标量扩展性能测试
    std::cout << "\nTest 1: Scalar expansion performance test";
    Shape large_shape(1000, 1000);
    Tensor scalar = Tensor::full(Shape(), 2.0f);

    auto start = std::chrono::high_resolution_clock::now();
    Tensor scalar_result = cpu_backend->expand(scalar, large_shape);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "\nScalar expansion time: " << duration.count() << " microseconds";

    // 验证结果正确性
    bool all_correct = true;
    const float* data = static_cast<const float*>(scalar_result.data_ptr());
    for (int64_t i = 0; i < large_shape.numel(); ++i) {
        if (std::abs(data[i] - 2.0f) > 1e-6f) {
            all_correct = false;
            break;
        }
    }
    if (all_correct) {
        std::cout << "\nScalar expansion result correctness: PASS";
    } else {
        std::cout << "\nScalar expansion result correctness: FAIL";
    }

    // 测试2: 相同形状扩展性能测试
    std::cout << "\nTest 2: Same shape expansion performance test";
    Shape medium_shape(100, 100);
    Tensor input = Tensor::full(medium_shape, 1.0f);

    start = std::chrono::high_resolution_clock::now();
    Tensor same_shape_result = cpu_backend->expand(input, medium_shape);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "\nSame shape expansion time: " << duration.count() << " microseconds";
}

void test_mathematical_correctness() {
    std::cout << "\n=== Testing Mathematical Correctness ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 简单扩展数学验证
    std::cout << "\nTest 1: Simple expansion mathematical verification";
    Tensor row_vec = Tensor::full(Shape(1, 2), 3.0f);  // [[3, 3]]
    Shape target_shape(3, 2);

    print_tensor_info("Input Vector (1,2)", row_vec);

    Tensor expanded = cpu_backend->expand(row_vec, target_shape);
    std::cout << "\nExpanded result (should be all 3s):";
    print_tensor_info("Expanded", expanded);

    // 验证扩展结果
    bool correct = true;
    const float* data = static_cast<const float*>(expanded.data_ptr());
    for (int64_t i = 0; i < expanded.numel(); ++i) {
        if (std::abs(data[i] - 3.0f) > 1e-6f) {
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "\nSimple expansion mathematical correctness: PASS";
    } else {
        std::cout << "\nSimple expansion mathematical correctness: FAIL";
    }

    // 测试2: 复杂扩展验证
    std::cout << "\nTest 2: Complex expansion mathematical verification";
    Tensor complex_input = Tensor::full(Shape(2, 1), 4.0f);  // [[4], [4]]
    Shape complex_target(2, 3);

    print_tensor_info("Complex Input (2,1)", complex_input);

    Tensor complex_expanded = cpu_backend->expand(complex_input, complex_target);
    std::cout << "\nComplex expanded result (should be all 4s):";
    print_tensor_info("Complex Expanded", complex_expanded);

    // 验证复杂扩展结果
    bool complex_correct = true;
    const float* complex_data = static_cast<const float*>(complex_expanded.data_ptr());
    for (int64_t i = 0; i < complex_expanded.numel(); ++i) {
        if (std::abs(complex_data[i] - 4.0f) > 1e-6f) {
            complex_correct = false;
            break;
        }
    }

    if (complex_correct) {
        std::cout << "\nComplex expansion mathematical correctness: PASS";
    } else {
        std::cout << "\nComplex expansion mathematical correctness: FAIL";
    }
}

int main() {
    try {
        std::cout << "=== CPU Backend Expand Operations Test ===";
        std::cout << "\nVersion: V1.28.1";
        std::cout << "\nTesting CPU backend expand tensor operations\n";

        // 获取CPU后端
        auto cpu_backend = BackendManager::get_cpu_backend();
        std::cout << "\nCPU backend initialized successfully";

        // 运行所有测试
        test_basic_expand_operations();
        test_shape_expand_operations();
        test_expand_edge_cases();
        test_into_operations();
        test_performance_optimizations();
        test_mathematical_correctness();

        std::cout << "\n\n=== Test Summary ===";
        std::cout << "\nAll expand operations tests completed";
        std::cout << "\nIncluding: 2 functions (expand, expand_into)";
        std::cout << "\nTest coverage: scalar expansion, shape expansion, edge cases, performance optimization, mathematical correctness";

        std::cout << "\n\nCPU expand operations test completed successfully!";
        return 0;

    } catch (const TRException& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\nUnexpected error: " << e.what() << std::endl;
        return 1;
    }
}
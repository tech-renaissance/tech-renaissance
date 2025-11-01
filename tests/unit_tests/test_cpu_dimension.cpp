/**
 * @file test_cpu_dimension.cpp
 * @brief CPU后端张量维度操作测试
 * @details 测试CPU后端的unsqueeze和squeeze操作，支持维度插入和删除
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

void test_basic_unsqueeze_operations() {
    std::cout << "\n=== Testing Basic Unsqueeze Operations ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 标量unsqueeze到0维
    std::cout << "\nTest 1: Scalar unsqueeze at dim 0";
    Tensor scalar = Tensor::full(Shape(), 3.14f);
    print_tensor_info("Original Scalar", scalar);

    Tensor unsqueezed = cpu_backend->unsqueeze(scalar, 0);
    std::cout << "\nAfter unsqueeze dim 0:";
    print_tensor_info("Unsqueezed", unsqueezed);

    // 验证形状和数据
    if (unsqueezed.shape() == Shape(1) &&
        std::abs(static_cast<const float*>(unsqueezed.data_ptr())[0] - 3.14f) < 1e-6f) {
        std::cout << "\nScalar unsqueeze correctness: PASS";
    } else {
        std::cout << "\nScalar unsqueeze correctness: FAIL";
    }

    // 测试2: 1D张量unsqueeze
    std::cout << "\nTest 2: 1D tensor unsqueeze at dim 0";
    Tensor vec1d = Tensor::full(Shape(3), 2.0f);
    print_tensor_info("Original 1D Tensor", vec1d);

    Tensor vec_unsqueezed = cpu_backend->unsqueeze(vec1d, 0);
    std::cout << "\nAfter unsqueeze dim 0:";
    print_tensor_info("1D Unsqueezed", vec_unsqueezed);

    // 验证形状和数据
    if (vec_unsqueezed.shape() == Shape(1, 3)) {
        bool data_correct = true;
        const float* data = static_cast<const float*>(vec_unsqueezed.data_ptr());
        for (int64_t i = 0; i < 3; ++i) {
            if (std::abs(data[i] - 2.0f) > 1e-6f) {
                data_correct = false;
                break;
            }
        }
        if (data_correct) {
            std::cout << "\n1D unsqueeze correctness: PASS";
        } else {
            std::cout << "\n1D unsqueeze correctness: FAIL";
        }
    } else {
        std::cout << "\n1D unsqueeze shape incorrect";
    }

    // 测试3: 2D张量unsqueeze在中间位置
    std::cout << "\nTest 3: 2D tensor unsqueeze at dim 1";
    Tensor mat2d = Tensor::full(Shape(2, 3), 1.0f);
    print_tensor_info("Original 2D Tensor", mat2d);

    Tensor mat_unsqueezed = cpu_backend->unsqueeze(mat2d, 1);
    std::cout << "\nAfter unsqueeze dim 1:";
    print_tensor_info("2D Unsqueezed", mat_unsqueezed);

    // 验证形状
    if (mat_unsqueezed.shape() == Shape(2, 1, 3)) {
        std::cout << "\n2D unsqueeze shape correctness: PASS";
    } else {
        std::cout << "\n2D unsqueeze shape correctness: FAIL";
    }

    // 测试4: unsqueeze在最后位置
    std::cout << "\nTest 4: 2D tensor unsqueeze at dim 2 (last position)";
    Tensor mat2d2 = Tensor::full(Shape(2, 3), 4.0f);
    print_tensor_info("Original 2D Tensor 2", mat2d2);

    Tensor mat_unsqueezed2 = cpu_backend->unsqueeze(mat2d2, 2);
    std::cout << "\nAfter unsqueeze dim 2:";
    print_tensor_info("2D Unsqueezed 2", mat_unsqueezed2);

    // 验证形状
    if (mat_unsqueezed2.shape() == Shape(2, 3, 1)) {
        std::cout << "\nLast position unsqueeze correctness: PASS";
    } else {
        std::cout << "\nLast position unsqueeze correctness: FAIL";
    }
}

void test_basic_squeeze_operations() {
    std::cout << "\n=== Testing Basic Squeeze Operations ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: squeeze 1D张量到标量
    std::cout << "\nTest 1: 1D tensor squeeze at dim 0";
    Tensor vec1d = Tensor::full(Shape(1), 5.0f);
    print_tensor_info("Original 1D Tensor", vec1d);

    Tensor squeezed = cpu_backend->squeeze(vec1d, 0);
    std::cout << "\nAfter squeeze dim 0:";
    print_tensor_info("Squeezed", squeezed);

    // 验证形状和数据
    if (squeezed.shape() == Shape() &&
        std::abs(static_cast<const float*>(squeezed.data_ptr())[0] - 5.0f) < 1e-6f) {
        std::cout << "\n1D squeeze correctness: PASS";
    } else {
        std::cout << "\n1D squeeze correctness: FAIL";
    }

    // 测试2: squeeze 2D张量的第一个维度
    std::cout << "\nTest 2: 2D tensor squeeze at dim 0";
    Tensor mat2d = Tensor::full(Shape(1, 3), 2.0f);
    print_tensor_info("Original 2D Tensor", mat2d);

    Tensor mat_squeezed = cpu_backend->squeeze(mat2d, 0);
    std::cout << "\nAfter squeeze dim 0:";
    print_tensor_info("2D Squeezed", mat_squeezed);

    // 验证形状和数据
    if (mat_squeezed.shape() == Shape(3)) {
        bool data_correct = true;
        const float* data = static_cast<const float*>(mat_squeezed.data_ptr());
        for (int64_t i = 0; i < 3; ++i) {
            if (std::abs(data[i] - 2.0f) > 1e-6f) {
                data_correct = false;
                break;
            }
        }
        if (data_correct) {
            std::cout << "\n2D squeeze correctness: PASS";
        } else {
            std::cout << "\n2D squeeze correctness: FAIL";
        }
    } else {
        std::cout << "\n2D squeeze shape incorrect";
    }

    // 测试3: squeeze 3D张量的中间维度
    std::cout << "\nTest 3: 3D tensor squeeze at dim 1";
    Tensor tensor3d = Tensor::full(Shape(2, 1, 4), 3.0f);
    print_tensor_info("Original 3D Tensor", tensor3d);

    Tensor tensor3d_squeezed = cpu_backend->squeeze(tensor3d, 1);
    std::cout << "\nAfter squeeze dim 1:";
    print_tensor_info("3D Squeezed", tensor3d_squeezed);

    // 验证形状
    if (tensor3d_squeezed.shape() == Shape(2, 4)) {
        std::cout << "\n3D squeeze shape correctness: PASS";
    } else {
        std::cout << "\n3D squeeze shape correctness: FAIL";
    }
}

void test_dimension_edge_cases() {
    std::cout << "\n=== Testing Dimension Edge Cases ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: unsqueeze维度超出范围
    std::cout << "\nTest 1: Unsqueeze dimension out of range";
    try {
        Tensor tensor = Tensor::full(Shape(2, 3), 1.0f);
        Tensor result = cpu_backend->unsqueeze(tensor, 4);  // 超出范围
        std::cout << "\nShould have thrown error but succeeded!";
    } catch (const TRException& e) {
        std::cout << "\nCorrectly caught exception: " << e.what();
    }

    // 测试2: squeeze维度超出范围
    std::cout << "\nTest 2: Squeeze dimension out of range";
    try {
        Tensor tensor = Tensor::full(Shape(2, 3), 1.0f);
        Tensor result = cpu_backend->squeeze(tensor, 2);  // 超出范围
        std::cout << "\nShould have thrown error but succeeded!";
    } catch (const TRException& e) {
        std::cout << "\nCorrectly caught exception: " << e.what();
    }

    // 测试3: squeeze非大小为1的维度
    std::cout << "\nTest 3: Squeeze dimension with size != 1";
    try {
        Tensor tensor = Tensor::full(Shape(2, 3), 1.0f);
        Tensor result = cpu_backend->squeeze(tensor, 0);  // 维度大小为2
        std::cout << "\nShould have thrown error but succeeded!";
    } catch (const TRException& e) {
        std::cout << "\nCorrectly caught exception: " << e.what();
    }

    // 测试4: 负维度索引
    std::cout << "\nTest 4: Negative dimension index";
    try {
        Tensor tensor = Tensor::full(Shape(2, 3), 1.0f);
        Tensor result = cpu_backend->unsqueeze(tensor, -1);  // 负索引
        std::cout << "\nShould have thrown error but succeeded!";
    } catch (const TRException& e) {
        std::cout << "\nCorrectly caught exception: " << e.what();
    }

    // 测试5: 空张量处理
    std::cout << "\nTest 5: Empty tensor handling";
    try {
        Tensor empty_tensor;
        Tensor result = cpu_backend->unsqueeze(empty_tensor, 0);
        std::cout << "\nEmpty tensor unsqueeze returned empty tensor";
    } catch (const TRException& e) {
        std::cout << "\nCaught exception: " << e.what();
    }
}

void test_into_operations() {
    std::cout << "\n=== Testing Into Operations ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: unsqueeze_into操作
    std::cout << "\nTest 1: unsqueeze_into operation";
    Tensor input = Tensor::full(Shape(2, 3), 2.0f);
    Tensor output = Tensor::empty(Shape(2, 1, 3), DType::FP32, tr::CPU);

    print_tensor_info("Input Tensor", input);
    print_tensor_info("Output Tensor (pre-allocated)", output);

    cpu_backend->unsqueeze_into(input, output);
    std::cout << "\nAfter unsqueeze_into:";
    print_tensor_info("Output", output);

    // 验证结果
    Tensor expected = cpu_backend->unsqueeze(input, 1);
    if (are_tensors_close(output, expected)) {
        std::cout << "\nunsqueeze_into correctness: PASS";
    } else {
        std::cout << "\nunsqueeze_into correctness: FAIL";
    }

    // 测试2: squeeze_into操作
    std::cout << "\nTest 2: squeeze_into operation";
    Tensor input2 = Tensor::full(Shape(1, 4), 3.0f);
    Tensor output2 = Tensor::empty(Shape(4), DType::FP32, tr::CPU);

    print_tensor_info("Input Tensor 2", input2);
    print_tensor_info("Output Tensor 2 (pre-allocated)", output2);

    cpu_backend->squeeze_into(input2, output2);
    std::cout << "\nAfter squeeze_into:";
    print_tensor_info("Output 2", output2);

    // 验证结果
    Tensor expected2 = cpu_backend->squeeze(input2, 0);
    if (are_tensors_close(output2, expected2)) {
        std::cout << "\nsqueeze_into correctness: PASS";
    } else {
        std::cout << "\nsqueeze_into correctness: FAIL";
    }

    // 测试3: 元素数不匹配的错误情况
    std::cout << "\nTest 3: Element count mismatch error";
    try {
        Tensor small_input = Tensor::full(Shape(2, 3), 1.0f);
        Tensor large_output = Tensor::empty(Shape(2, 1, 4), DType::FP32, tr::CPU);  // 元素数不匹配
        cpu_backend->unsqueeze_into(small_input, large_output);
        std::cout << "\nShould have thrown error but succeeded!";
    } catch (const TRException& e) {
        std::cout << "\nCorrectly caught exception: " << e.what();
    }
}

void test_inplace_operations() {
    std::cout << "\n=== Testing Inplace Operations ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: unsqueeze_inplace操作
    std::cout << "\nTest 1: unsqueeze_inplace operation";
    Tensor tensor = Tensor::full(Shape(2, 3), 1.0f);
    print_tensor_info("Original Tensor", tensor);

    cpu_backend->unsqueeze_inplace(tensor, 1);
    std::cout << "\nAfter unsqueeze_inplace dim 1:";
    print_tensor_info("Modified Tensor", tensor);

    // 验证形状
    if (tensor.shape() == Shape(2, 1, 3)) {
        std::cout << "\nunsqueeze_inplace shape correctness: PASS";
    } else {
        std::cout << "\nunsqueeze_inplace shape correctness: FAIL";
    }

    // 测试2: squeeze_inplace操作
    std::cout << "\nTest 2: squeeze_inplace operation";
    Tensor tensor2 = Tensor::full(Shape(1, 4), 2.0f);
    print_tensor_info("Original Tensor 2", tensor2);

    cpu_backend->squeeze_inplace(tensor2, 0);
    std::cout << "\nAfter squeeze_inplace dim 0:";
    print_tensor_info("Modified Tensor 2", tensor2);

    // 验证形状
    if (tensor2.shape() == Shape(4)) {
        std::cout << "\nsqueeze_inplace shape correctness: PASS";
    } else {
        std::cout << "\nsqueeze_inplace shape correctness: FAIL";
    }
}

void test_complex_operations() {
    std::cout << "\n=== Testing Complex Operations ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 多次unsqueeze操作
    std::cout << "\nTest 1: Multiple unsqueeze operations";
    Tensor original = Tensor::full(Shape(3), 1.0f);
    print_tensor_info("Original Tensor", original);

    Tensor step1 = cpu_backend->unsqueeze(original, 0);  // (1, 3)
    Tensor step2 = cpu_backend->unsqueeze(step1, 1);    // (1, 1, 3)
    Tensor step3 = cpu_backend->unsqueeze(step2, 2);    // (1, 1, 1, 3)

    print_tensor_info("After unsqueeze dim 0", step1);
    print_tensor_info("After unsqueeze dim 1", step2);
    print_tensor_info("After unsqueeze dim 2", step3);

    // 验证最终形状
    if (step3.shape() == Shape(1, 1, 1, 3)) {
        std::cout << "\nMultiple unsqueeze correctness: PASS";
    } else {
        std::cout << "\nMultiple unsqueeze correctness: FAIL";
    }

    // 测试2: unsqueeze和squeeze的逆操作
    std::cout << "\nTest 2: Unsqueeze and squeeze inverse operations";
    Tensor start = Tensor::full(Shape(2, 3), 4.0f);
    print_tensor_info("Start Tensor", start);

    Tensor unsqueezed = cpu_backend->unsqueeze(start, 1);  // (2, 1, 3)
    print_tensor_info("After unsqueeze dim 1", unsqueezed);

    Tensor squeezed = cpu_backend->squeeze(unsqueezed, 1);  // (2, 3)
    print_tensor_info("After squeeze dim 1", squeezed);

    // 验证逆操作
    if (are_tensors_close(start, squeezed)) {
        std::cout << "\nUnsqueeze-squeeze inverse correctness: PASS";
    } else {
        std::cout << "\nUnsqueeze-squeeze inverse correctness: FAIL";
    }

    // 测试3: 数据保持性验证
    std::cout << "\nTest 3: Data preservation verification";
    Tensor data_tensor = Tensor::full(Shape(1, 2, 1, 3), 7.0f);
    print_tensor_info("Original Complex Tensor", data_tensor);

    // 移除两个维度为1的维度
    Tensor step_a = cpu_backend->squeeze(data_tensor, 2);  // (1, 2, 3)
    Tensor step_b = cpu_backend->squeeze(step_a, 0);        // (2, 3)
    print_tensor_info("After double squeeze", step_b);

    // 验证数据保持
    bool data_preserved = true;
    const float* data = static_cast<const float*>(step_b.data_ptr());
    for (int64_t i = 0; i < 6; ++i) {
        if (std::abs(data[i] - 7.0f) > 1e-6f) {
            data_preserved = false;
            break;
        }
    }
    if (data_preserved) {
        std::cout << "\nData preservation correctness: PASS";
    } else {
        std::cout << "\nData preservation correctness: FAIL";
    }
}

void test_performance_characteristics() {
    std::cout << "\n=== Testing Performance Characteristics ===";

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 大张量的unsqueeze性能
    std::cout << "\nTest 1: Large tensor unsqueeze performance";
    Shape large_shape(1000, 1000);
    Tensor large_tensor = Tensor::full(large_shape, 1.0f);

    auto start = std::chrono::high_resolution_clock::now();
    Tensor large_unsqueezed = cpu_backend->unsqueeze(large_tensor, 0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "\nLarge tensor unsqueeze time: " << duration.count() << " microseconds";

    // 验证形状正确性
    if (large_unsqueezed.shape() == Shape(1, 1000, 1000)) {
        std::cout << "\nLarge unsqueeze shape correctness: PASS";
    } else {
        std::cout << "\nLarge unsqueeze shape correctness: FAIL";
    }

    // 测试2: 大张量的squeeze性能
    std::cout << "\nTest 2: Large tensor squeeze performance";
    Tensor tensor_to_squeeze = Tensor::full(Shape(1, 1000, 1000), 2.0f);

    start = std::chrono::high_resolution_clock::now();
    Tensor large_squeezed = cpu_backend->squeeze(tensor_to_squeeze, 0);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "\nLarge tensor squeeze time: " << duration.count() << " microseconds";

    // 验证形状正确性
    if (large_squeezed.shape() == Shape(1000, 1000)) {
        std::cout << "\nLarge squeeze shape correctness: PASS";
    } else {
        std::cout << "\nLarge squeeze shape correctness: FAIL";
    }
}

int main() {
    try {
        std::cout << "=== CPU Backend Dimension Operations Test ===";
        std::cout << "\nVersion: V1.29.2";
        std::cout << "\nTesting CPU backend dimension tensor operations\n";

        // 获取CPU后端
        auto cpu_backend = BackendManager::get_cpu_backend();
        std::cout << "\nCPU backend initialized successfully";

        // 运行所有测试
        test_basic_unsqueeze_operations();
        test_basic_squeeze_operations();
        test_dimension_edge_cases();
        test_into_operations();
        test_inplace_operations();
        test_complex_operations();
        test_performance_characteristics();

        std::cout << "\n\n=== Test Summary ===";
        std::cout << "\nAll dimension operations tests completed";
        std::cout << "\nIncluding: 6 functions (3 unsqueeze + 3 squeeze)";
        std::cout << "\nTest coverage: basic operations, edge cases, into operations, inplace operations, complex scenarios, performance";

        std::cout << "\n\nCPU dimension operations test completed successfully!";
        return 0;

    } catch (const TRException& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\nUnexpected error: " << e.what() << std::endl;
        return 1;
    }
}
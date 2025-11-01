/**
 * @file test_cpu_pad.cpp
 * @brief CPU后端pad函数测试
 * @details 测试CPU后端的pad和pad_into函数功能
 * @version 1.00.00
 * @date 2025-11-02
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h, pytorch通信模块
 * @note 所属系列: tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <vector>

using namespace tr;

// 简单的辅助函数来打印张量内容
void print_tensor_info(const Tensor& tensor, const std::string& name) {
    std::cout << name << " (shape: " << tensor.shape().to_string() << ", dtype: "
              << dtype_to_string(tensor.dtype()) << ")" << std::endl;
    tensor.print(name);
}

// 测试2D张量的padding
void test_pad_2d() {
    std::cout << "=== Testing 2D Tensor Padding ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 创建随机2D张量 (H=3, W=4)
    Shape input_shape(3, 4);
    Tensor input = Tensor::randn(input_shape, 42, DType::FP32, tr::CPU);
    print_tensor_info(input, "Input 2D tensor");

    // 测试pad函数，padding=2
    int32_t padding = 2;
    Tensor padded = cpu_backend->pad(input, padding);
    print_tensor_info(padded, "Result with padding=2");

    // 验证形状：3x4 -> 7x8
    Shape expected_shape(7, 8);
    if (padded.shape() != expected_shape) {
        std::cout << "Shape test failed! Expected: " << expected_shape.to_string()
                  << ", Actual: " << padded.shape().to_string() << std::endl;
        return;
    }

    // 验证边界区域为0
    const float* padded_data = static_cast<const float*>(padded.data_ptr());
    bool test_passed = true;

    // 检查边界应该为0（前2行和后2行，前2列和后2列）
    for (int32_t i = 0; i < 7; ++i) {
        for (int32_t j = 0; j < 8; ++j) {
            // 边界条件
            bool is_boundary = (i < padding) || (i >= 7 - padding) ||
                              (j < padding) || (j >= 8 - padding);

            if (is_boundary && std::abs(padded_data[i * 8 + j]) > 1e-6f) {
                test_passed = false;
                break;
            }
        }
        if (!test_passed) break;
    }

    if (test_passed) {
        std::cout << "PASS: 2D tensor padding test" << std::endl;
    } else {
        std::cout << "FAIL: 2D tensor padding test" << std::endl;
    }
}

// 测试3D张量的padding
void test_pad_3d() {
    std::cout << "\n=== Testing 3D Tensor Padding ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 创建随机3D张量 (C=2, H=3, W=4)
    Shape input_shape(2, 3, 4);
    Tensor input = Tensor::randn(input_shape, 123, DType::FP32, tr::CPU);
    print_tensor_info(input, "Input 3D tensor");

    // 测试pad函数，padding=1
    int32_t padding = 1;
    Tensor padded = cpu_backend->pad(input, padding);
    print_tensor_info(padded, "Result with padding=1");

    // 验证形状：(2, 3, 4) -> (2, 5, 6)
    Shape expected_shape(2, 5, 6);
    if (padded.shape() != expected_shape) {
        std::cout << "Shape test failed! Expected: " << expected_shape.to_string()
                  << ", Actual: " << padded.shape().to_string() << std::endl;
        return;
    }

    std::cout << "PASS: 3D tensor padding test" << std::endl;
}

// 测试4D张量的padding
void test_pad_4d() {
    std::cout << "\n=== Testing 4D Tensor Padding ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 创建随机4D张量 (N=2, C=3, H=4, W=5)
    Shape input_shape(2, 3, 4, 5);
    Tensor input = Tensor::randn(input_shape, 456, DType::FP32, tr::CPU);
    print_tensor_info(input, "Input 4D tensor");

    // 测试pad函数，padding=2
    int32_t padding = 2;
    Tensor padded = cpu_backend->pad(input, padding);
    print_tensor_info(padded, "Result with padding=2");

    // 验证形状：(2, 3, 4, 5) -> (2, 3, 8, 9)
    Shape expected_shape(2, 3, 8, 9);
    if (padded.shape() != expected_shape) {
        std::cout << "Shape test failed! Expected: " << expected_shape.to_string()
                  << ", Actual: " << padded.shape().to_string() << std::endl;
        return;
    }

    std::cout << "PASS: 4D tensor padding test" << std::endl;
}

// 测试INT8数据类型
void test_pad_int8() {
    std::cout << "\n=== Testing INT8 Tensor Padding ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 创建随机INT8张量 (H=3, W=4)
    Shape input_shape(3, 4);
    Tensor input = Tensor::randint(0, 100, input_shape, 789, tr::CPU);

    // 转换为INT8类型
    Tensor input_int8 = Tensor::empty(input_shape, DType::INT8, tr::CPU);
    const float* input_data = static_cast<const float*>(input.data_ptr());
    int8_t* int8_data = static_cast<int8_t*>(input_int8.data_ptr());
    for (int64_t i = 0; i < input.numel(); ++i) {
        int8_data[i] = static_cast<int8_t>(std::round(input_data[i]));
    }

    print_tensor_info(input_int8, "Input INT8 tensor");

    // 测试pad函数，padding=1
    Tensor padded = cpu_backend->pad(input_int8, 1);
    print_tensor_info(padded, "Result with padding=1");

    // 验证形状：3x4 -> 5x6
    Shape expected_shape(5, 6);
    if (padded.shape() != expected_shape) {
        std::cout << "INT8 shape test failed! Expected: " << expected_shape.to_string()
                  << ", Actual: " << padded.shape().to_string() << std::endl;
        return;
    }

    std::cout << "PASS: INT8 tensor padding test" << std::endl;
}

// 测试pad_into函数
void test_pad_into() {
    std::cout << "\n=== Testing pad_into Function ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 创建随机输入张量 (H=3, W=4)
    Shape input_shape(3, 4);
    Tensor input = Tensor::randn(input_shape, 321, DType::FP32, tr::CPU);
    print_tensor_info(input, "Input tensor");

    // 测试pad_into函数
    int32_t padding = 1;
    Shape expected_shape = Shape(input_shape.dim(0) + 2 * padding,
                                 input_shape.dim(1) + 2 * padding);
    Tensor output = Tensor::empty(expected_shape, DType::FP32, tr::CPU);

    cpu_backend->pad_into(input, padding, output);
    print_tensor_info(output, "Result of pad_into");

    // 验证形状
    if (output.shape() != expected_shape) {
        std::cout << "pad_into shape test failed! Expected: " << expected_shape.to_string()
                  << ", Actual: " << output.shape().to_string() << std::endl;
        return;
    }

    // 比较pad和pad_into的结果
    Tensor reference = cpu_backend->pad(input, padding);
    const float* ref_data = static_cast<const float*>(reference.data_ptr());
    const float* output_data = static_cast<const float*>(output.data_ptr());

    bool test_passed = true;
    for (int64_t i = 0; i < output.numel(); ++i) {
        if (std::abs(ref_data[i] - output_data[i]) > 1e-6f) {
            test_passed = false;
            break;
        }
    }

    if (test_passed) {
        std::cout << "PASS: pad_into function test" << std::endl;
    } else {
        std::cout << "FAIL: pad_into function test" << std::endl;
    }
}

// 测试错误情况
void test_error_cases() {
    std::cout << "\n=== Testing Error Cases ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    try {
        // 测试1D张量（应该失败）
        Tensor tensor_1d = Tensor::randn(Shape(5), 999, DType::FP32, tr::CPU);
        cpu_backend->pad(tensor_1d, 1);
        std::cout << "FAIL: 1D tensor should have failed" << std::endl;
    } catch (const TRException& e) {
        std::cout << "PASS: 1D tensor correctly threw exception: " << e.what() << std::endl;
    }

    try {
        // 测试负padding（应该失败）
        Tensor tensor_2d = Tensor::randn(Shape(3, 3), 888, DType::FP32, tr::CPU);
        cpu_backend->pad(tensor_2d, -1);
        std::cout << "FAIL: Negative padding should have failed" << std::endl;
    } catch (const TRException& e) {
        std::cout << "PASS: Negative padding correctly threw exception: " << e.what() << std::endl;
    }

    try {
        // 测试形状不匹配的pad_into（应该失败）
        Tensor input = Tensor::randn(Shape(2, 2), 777, DType::FP32, tr::CPU);
        Tensor output = Tensor::empty(Shape(4, 5), DType::FP32, tr::CPU);  // 应该是(4, 4)
        cpu_backend->pad_into(input, 1, output);
        std::cout << "FAIL: Shape mismatch should have failed" << std::endl;
    } catch (const TRException& e) {
        std::cout << "PASS: Shape mismatch correctly threw exception: " << e.what() << std::endl;
    }
}

int main() {
    Logger::get_instance().set_quiet_mode(true);
    std::cout << "Starting CPU backend pad function tests..." << std::endl;

    try {
        test_pad_2d();
        test_pad_3d();
        test_pad_4d();
        test_pad_int8();
        test_pad_into();
        test_error_cases();

        std::cout << "\n=== All Tests Completed ===" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Exception occurred during testing: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
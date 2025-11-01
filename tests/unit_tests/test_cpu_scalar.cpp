/**
 * @file test_cpu_scalar.cpp
 * @brief CPU后端标量运算测试
 * @details 测试CPU后端的5种标量运算：mul、add、minus（两种形式）、mac
 * @version 1.00.00
 * @date 2025-11-01
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h
 * @note 所属系列: tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <iomanip>

using namespace tr;

void print_tensor_info(const std::string& name, const Tensor& tensor) {
    std::cout << "\n" << name << ":\n";
    tensor.print();
}

int main() {
    try {
        std::cout << "=== CPU Backend Scalar Operations Test ===\n";

        // 获取CPU后端
        auto cpu_backend = BackendManager::get_cpu_backend();

        // 创建4x5的FP32张量，使用uniform随机值初始化
        Shape shape(4, 5);
        Tensor input = Tensor::uniform(shape, -1.0f, 1.0f, 42); // 固定种子42
        float scalar = 2.0f;

        std::cout << "\nTest Configuration:";
        std::cout << "\n  Input tensor shape: " << shape.to_string();
        std::cout << "\n  Scalar value: " << scalar;
        std::cout << "\n  Random seed: 42";

        // 打印原始输入张量
        print_tensor_info("Original Input Tensor", input);

        // ===== 测试乘法运算：tensor * scalar =====
        std::cout << "\n=== Testing Multiplication: tensor * scalar ===";

        Tensor mul_result = cpu_backend->mul(input, scalar);
        print_tensor_info("Multiplication Result (non-inplace)", mul_result);

        // 为每个测试创建独立的张量
        Tensor mul_inplace_input = Tensor::uniform(shape, -1.0f, 1.0f, 42);
        cpu_backend->mul_inplace(mul_inplace_input, scalar);
        print_tensor_info("Multiplication Result (inplace)", mul_inplace_input);

        Tensor mul_into_result = Tensor::empty(shape, DType::FP32, tr::CPU);
        cpu_backend->mul_into(input, scalar, mul_into_result);
        print_tensor_info("Multiplication Result (into)", mul_into_result);

        // ===== 测试加法运算：tensor + scalar =====
        std::cout << "\n=== Testing Addition: tensor + scalar ===";

        Tensor add_result = cpu_backend->add(input, scalar);
        print_tensor_info("Addition Result (non-inplace)", add_result);

        Tensor add_inplace_input = Tensor::uniform(shape, -1.0f, 1.0f, 42);
        cpu_backend->add_inplace(add_inplace_input, scalar);
        print_tensor_info("Addition Result (inplace)", add_inplace_input);

        Tensor add_into_result = Tensor::empty(shape, DType::FP32, tr::CPU);
        cpu_backend->add_into(input, scalar, add_into_result);
        print_tensor_info("Addition Result (into)", add_into_result);

        // ===== 测试减法运算：tensor - scalar =====
        std::cout << "\n=== Testing Subtraction: tensor - scalar ===";

        Tensor minus_result = cpu_backend->minus(input, scalar);
        print_tensor_info("Subtraction Result (tensor - scalar, non-inplace)", minus_result);

        Tensor minus_inplace_input = Tensor::uniform(shape, -1.0f, 1.0f, 42);
        cpu_backend->minus_inplace(minus_inplace_input, scalar);
        print_tensor_info("Subtraction Result (tensor - scalar, inplace)", minus_inplace_input);

        Tensor minus_into_result = Tensor::empty(shape, DType::FP32, tr::CPU);
        cpu_backend->minus_into(input, scalar, minus_into_result);
        print_tensor_info("Subtraction Result (tensor - scalar, into)", minus_into_result);

        // ===== 测试减法运算：scalar - tensor =====
        std::cout << "\n=== Testing Subtraction: scalar - tensor ===";

        Tensor scalar_minus_result = cpu_backend->minus(scalar, input);
        print_tensor_info("Subtraction Result (scalar - tensor, non-inplace)", scalar_minus_result);

        Tensor scalar_minus_inplace_input = Tensor::uniform(shape, -1.0f, 1.0f, 42);
        cpu_backend->minus_inplace(scalar, scalar_minus_inplace_input);
        print_tensor_info("Subtraction Result (scalar - tensor, inplace)", scalar_minus_inplace_input);

        Tensor scalar_minus_into_result = Tensor::empty(shape, DType::FP32, tr::CPU);
        cpu_backend->minus_into(scalar, input, scalar_minus_into_result);
        print_tensor_info("Subtraction Result (scalar - tensor, into)", scalar_minus_into_result);

        // ===== 测试乘加运算：tensor * scalar_x + scalar_y =====
        std::cout << "\n=== Testing Multiply-Add: tensor * scalar_x + scalar_y ===";

        float scalar_x = 2.0f;
        float scalar_y = 1.5f;
        std::cout << "\n  Using scalar_x = " << scalar_x << ", scalar_y = " << scalar_y;

        Tensor mac_result = cpu_backend->mac(input, scalar_x, scalar_y);
        print_tensor_info("Multiply-Add Result (non-inplace)", mac_result);

        Tensor mac_inplace_input = Tensor::uniform(shape, -1.0f, 1.0f, 42);
        cpu_backend->mac_inplace(mac_inplace_input, scalar_x, scalar_y);
        print_tensor_info("Multiply-Add Result (inplace)", mac_inplace_input);

        Tensor mac_into_result = Tensor::empty(shape, DType::FP32, tr::CPU);
        cpu_backend->mac_into(input, scalar_x, scalar_y, mac_into_result);
        print_tensor_info("Multiply-Add Result (into)", mac_into_result);

        // ===== 验证非原地和into方式的一致性 =====
        std::cout << "\n=== Verifying Consistency Between Implementations ===";

        bool mul_consistent = cpu_backend->is_close(mul_result, mul_into_result, 1e-6f);
        bool add_consistent = cpu_backend->is_close(add_result, add_into_result, 1e-6f);
        bool minus_consistent = cpu_backend->is_close(minus_result, minus_into_result, 1e-6f);
        bool scalar_minus_consistent = cpu_backend->is_close(scalar_minus_result, scalar_minus_into_result, 1e-6f);
        bool mac_consistent = cpu_backend->is_close(mac_result, mac_into_result, 1e-6f);

        std::cout << "\n  Multiplication consistency: " << (mul_consistent ? "PASS" : "FAIL");
        std::cout << "\n  Addition consistency: " << (add_consistent ? "PASS" : "FAIL");
        std::cout << "\n  Subtraction (tensor - scalar) consistency: " << (minus_consistent ? "PASS" : "FAIL");
        std::cout << "\n  Subtraction (scalar - tensor) consistency: " << (scalar_minus_consistent ? "PASS" : "FAIL");
        std::cout << "\n  Multiply-Add consistency: " << (mac_consistent ? "PASS" : "FAIL");

        bool all_consistent = mul_consistent && add_consistent && minus_consistent &&
                              scalar_minus_consistent && mac_consistent;

        std::cout << "\n\n=== Test Summary ===";
        std::cout << "\nAll scalar operations tested: 5 operations × 3 implementations = 15 tests";
        std::cout << "\nConsistency verification: " << (all_consistent ? "ALL PASS" : "SOME FAIL");

        if (all_consistent) {
            std::cout << "\n\n🎉 All CPU scalar operations completed successfully!";
            return 0;
        } else {
            std::cout << "\n\n❌ Some consistency checks failed!";
            return 1;
        }

    } catch (const TRException& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\nUnexpected error: " << e.what() << std::endl;
        return 1;
    }
}
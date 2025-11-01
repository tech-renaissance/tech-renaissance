/**
 * @file test_cpu_create.cpp
 * @brief CPU后端张量创建函数测试
 * @details 测试CPU后端的full、randn、uniform、randint、randbool等创建函数
 * @version 1.00.00
 * @date 2025-11-02
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h
 * @note 所属系列: tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <vector>

using namespace tr;

void print_tensor_info(const Tensor& tensor, const std::string& name) {
    std::cout << name << " (shape: " << tensor.shape().to_string() << ", dtype: "
              << dtype_to_string(tensor.dtype()) << ")" << std::endl;
    tensor.print(name);
}

// 测试full函数
void test_full() {
    std::cout << "=== Testing full function ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试基本full功能
    Shape shape(2, 3);
    Tensor result = cpu_backend->full(shape, 5.5f, DType::FP32);
    print_tensor_info(result, "full tensor with value 5.5");

    // 验证形状和值
    if (result.shape() != shape) {
        std::cout << "FAIL: Shape mismatch" << std::endl;
        return;
    }

    const float* data = static_cast<const float*>(result.data_ptr());
    bool test_passed = true;
    for (int64_t i = 0; i < result.numel(); ++i) {
        if (std::abs(data[i] - 5.5f) > 1e-6f) {
            test_passed = false;
            break;
        }
    }

    if (test_passed) {
        std::cout << "PASS: full function test" << std::endl;
    } else {
        std::cout << "FAIL: full function test - values incorrect" << std::endl;
    }

    // 测试INT8不支持的错误情况
    try {
        Tensor int8_result = cpu_backend->full(shape, 5.0f, DType::INT8);
        std::cout << "FAIL: INT8 should not be supported" << std::endl;
    } catch (const TRException& e) {
        std::cout << "PASS: INT8 correctly not supported: " << e.what() << std::endl;
    }
}

// 测试full_inplace函数
void test_full_inplace() {
    std::cout << "\n=== Testing full_inplace function ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试基本full_inplace功能
    Shape shape(2, 3);
    Tensor tensor = Tensor::empty(shape, DType::FP32, tr::CPU);
    cpu_backend->fill(tensor, 1.0f);  // 初始化
    print_tensor_info(tensor, "original tensor");

    cpu_backend->full_inplace(tensor, 9.9f);
    print_tensor_info(tensor, "after full_inplace with 9.9");

    // 验证值
    const float* data = static_cast<const float*>(tensor.data_ptr());
    bool test_passed = true;
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        if (std::abs(data[i] - 9.9f) > 1e-6f) {
            test_passed = false;
            break;
        }
    }

    if (test_passed) {
        std::cout << "PASS: full_inplace function test" << std::endl;
    } else {
        std::cout << "FAIL: full_inplace function test - values incorrect" << std::endl;
    }

    // 测试空张量错误
    try {
        Tensor empty_tensor;
        cpu_backend->full_inplace(empty_tensor, 1.0f);
        std::cout << "FAIL: Empty tensor should cause error" << std::endl;
    } catch (const TRException& e) {
        std::cout << "PASS: Empty tensor correctly throws exception" << std::endl;
    }
}

// 测试randn函数
void test_randn() {
    std::cout << "\n=== Testing randn function ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试基本randn功能
    Shape shape(3, 4);
    Tensor result = cpu_backend->randn(shape, 42);  // 固定种子
    print_tensor_info(result, "randn tensor with seed 42");

    // 验证形状
    if (result.shape() != shape) {
        std::cout << "FAIL: Shape mismatch" << std::endl;
        return;
    }

    // 验证可重现性（相同种子应该产生相同结果）
    Tensor result2 = cpu_backend->randn(shape, 42);  // 相同种子
    const float* data1 = static_cast<const float*>(result.data_ptr());
    const float* data2 = static_cast<const float*>(result2.data_ptr());

    bool test_passed = true;
    for (int64_t i = 0; i < result.numel(); ++i) {
        if (std::abs(data1[i] - data2[i]) > 1e-6f) {
            test_passed = false;
            break;
        }
    }

    if (test_passed) {
        std::cout << "PASS: randn function test" << std::endl;
    } else {
        std::cout << "FAIL: randn function test - not reproducible" << std::endl;
    }
}

// 测试randn_inplace函数
void test_randn_inplace() {
    std::cout << "\n=== Testing randn_inplace function ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试基本randn_inplace功能
    Shape shape(2, 3);
    Tensor tensor = Tensor::empty(shape, DType::FP32, tr::CPU);
    cpu_backend->fill(tensor, 0.0f);  // 初始化
    print_tensor_info(tensor, "original tensor");

    cpu_backend->randn_inplace(tensor, 123);
    print_tensor_info(tensor, "after randn_inplace with seed 123");

    // 验证形状保持不变
    if (tensor.shape() != shape) {
        std::cout << "FAIL: Shape changed" << std::endl;
        return;
    }

    // 验证可重现性
    Tensor tensor2 = Tensor::empty(shape, DType::FP32, tr::CPU);
    cpu_backend->randn_inplace(tensor2, 123);
    const float* data1 = static_cast<const float*>(tensor.data_ptr());
    const float* data2 = static_cast<const float*>(tensor2.data_ptr());

    bool test_passed = true;
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        if (std::abs(data1[i] - data2[i]) > 1e-6f) {
            test_passed = false;
            break;
        }
    }

    if (test_passed) {
        std::cout << "PASS: randn_inplace function test" << std::endl;
    } else {
        std::cout << "FAIL: randn_inplace function test - not reproducible" << std::endl;
    }
}

// 测试uniform函数
void test_uniform() {
    std::cout << "\n=== Testing uniform function ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试基本uniform功能
    Shape shape(3, 4);
    Tensor result = cpu_backend->uniform(shape, 10.0f, 20.0f, 456);  // 范围[10, 20)
    print_tensor_info(result, "uniform tensor in range [10, 20)");

    // 验证形状
    if (result.shape() != shape) {
        std::cout << "FAIL: Shape mismatch" << std::endl;
        return;
    }

    // 验证范围
    const float* data = static_cast<const float*>(result.data_ptr());
    bool test_passed = true;
    for (int64_t i = 0; i < result.numel(); ++i) {
        if (data[i] < 10.0f || data[i] >= 20.0f) {
            test_passed = false;
            std::cout << "Value out of range: " << data[i] << std::endl;
            break;
        }
    }

    if (test_passed) {
        std::cout << "PASS: uniform function test" << std::endl;
    } else {
        std::cout << "FAIL: uniform function test - values out of range" << std::endl;
    }
}

// 测试uniform_inplace函数
void test_uniform_inplace() {
    std::cout << "\n=== Testing uniform_inplace function ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试基本uniform_inplace功能
    Shape shape(2, 3);
    Tensor tensor = Tensor::empty(shape, DType::FP32, tr::CPU);
    cpu_backend->fill(tensor, 0.0f);  // 初始化

    cpu_backend->uniform_inplace(tensor, -5.0f, 5.0f, 789);  // 范围[-5, 5)
    print_tensor_info(tensor, "uniform_inplace in range [-5, 5)");

    // 验证范围
    const float* data = static_cast<const float*>(tensor.data_ptr());
    bool test_passed = true;
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        if (data[i] < -5.0f || data[i] >= 5.0f) {
            test_passed = false;
            std::cout << "Value out of range: " << data[i] << std::endl;
            break;
        }
    }

    if (test_passed) {
        std::cout << "PASS: uniform_inplace function test" << std::endl;
    } else {
        std::cout << "FAIL: uniform_inplace function test - values out of range" << std::endl;
    }
}

// 测试randint函数
void test_randint() {
    std::cout << "\n=== Testing randint function ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试基本randint功能
    Shape shape(3, 4);
    Tensor result = cpu_backend->randint(shape, 1, 5, 321);  // 范围[1, 5)
    print_tensor_info(result, "randint tensor in range [1, 5)");

    // 验证形状
    if (result.shape() != shape) {
        std::cout << "FAIL: Shape mismatch" << std::endl;
        return;
    }

    // 验证范围和整数性质
    const float* data = static_cast<const float*>(result.data_ptr());
    bool test_passed = true;
    for (int64_t i = 0; i < result.numel(); ++i) {
        float val = data[i];
        if (val < 1.0f || val >= 5.0f || std::abs(val - std::round(val)) > 1e-6f) {
            test_passed = false;
            std::cout << "Value not integer or out of range: " << val << std::endl;
            break;
        }
    }

    if (test_passed) {
        std::cout << "PASS: randint function test" << std::endl;
    } else {
        std::cout << "FAIL: randint function test - invalid values" << std::endl;
    }

    // 测试参数错误
    try {
        Tensor error_result = cpu_backend->randint(shape, 5, 1, 123);  // low >= high
        std::cout << "FAIL: Invalid parameters should cause error" << std::endl;
    } catch (const TRException& e) {
        std::cout << "PASS: Invalid parameters correctly throw exception" << std::endl;
    }
}

// 测试randint_inplace函数
void test_randint_inplace() {
    std::cout << "\n=== Testing randint_inplace function ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试基本randint_inplace功能
    Shape shape(2, 3);
    Tensor tensor = Tensor::empty(shape, DType::FP32, tr::CPU);
    cpu_backend->fill(tensor, 0.0f);  // 初始化

    cpu_backend->randint_inplace(tensor, 10, 15, 654);  // 范围[10, 15)
    print_tensor_info(tensor, "randint_inplace in range [10, 15)");

    // 验证范围和整数性质
    const float* data = static_cast<const float*>(tensor.data_ptr());
    bool test_passed = true;
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        float val = data[i];
        if (val < 10.0f || val >= 15.0f || std::abs(val - std::round(val)) > 1e-6f) {
            test_passed = false;
            std::cout << "Value not integer or out of range: " << val << std::endl;
            break;
        }
    }

    if (test_passed) {
        std::cout << "PASS: randint_inplace function test" << std::endl;
    } else {
        std::cout << "FAIL: randint_inplace function test - invalid values" << std::endl;
    }
}

// 测试randbool函数
void test_randbool() {
    std::cout << "\n=== Testing randbool function ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试基本randbool功能
    Shape shape(4, 5);
    float zero_rate = 0.3f;  // 30% zeros
    Tensor result = cpu_backend->randbool(shape, zero_rate, 987);
    print_tensor_info(result, "randbool tensor with 30% zeros");

    // 验证形状
    if (result.shape() != shape) {
        std::cout << "FAIL: Shape mismatch" << std::endl;
        return;
    }

    // 验证值只能是0或1
    const float* data = static_cast<const float*>(result.data_ptr());
    bool test_passed = true;
    int zero_count = 0;
    int one_count = 0;

    for (int64_t i = 0; i < result.numel(); ++i) {
        float val = data[i];
        if (std::abs(val - 0.0f) < 1e-6f) {
            zero_count++;
        } else if (std::abs(val - 1.0f) < 1e-6f) {
            one_count++;
        } else {
            test_passed = false;
            std::cout << "Invalid value: " << val << std::endl;
            break;
        }
    }

    if (test_passed) {
        float actual_zero_rate = static_cast<float>(zero_count) / result.numel();
        std::cout << "Zero rate: " << actual_zero_rate * 100 << "% (expected: " << zero_rate * 100 << "%)" << std::endl;

        // 对于小样本，允许一定的误差
        if (std::abs(actual_zero_rate - zero_rate) < 0.3f) {  // 允许30%的误差
            std::cout << "PASS: randbool function test" << std::endl;
        } else {
            std::cout << "FAIL: randbool function test - zero rate too far from expected" << std::endl;
        }
    } else {
        std::cout << "FAIL: randbool function test - invalid values" << std::endl;
    }

    // 测试参数错误
    try {
        Tensor error_result = cpu_backend->randbool(shape, -0.1f, 123);  // negative rate
        std::cout << "FAIL: Invalid rate should cause error" << std::endl;
    } catch (const TRException& e) {
        std::cout << "PASS: Invalid rate correctly throws exception" << std::endl;
    }
}

// 测试randbool_inplace函数
void test_randbool_inplace() {
    std::cout << "\n=== Testing randbool_inplace function ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试基本randbool_inplace功能
    Shape shape(3, 4);
    Tensor tensor = Tensor::empty(shape, DType::FP32, tr::CPU);
    cpu_backend->fill(tensor, 0.5f);  // 初始化

    cpu_backend->randbool_inplace(tensor, 0.8f, 246);  // 80% zeros
    print_tensor_info(tensor, "randbool_inplace with 80% zeros");

    // 验证值只能是0或1
    const float* data = static_cast<const float*>(tensor.data_ptr());
    bool test_passed = true;
    int zero_count = 0;
    int one_count = 0;

    for (int64_t i = 0; i < tensor.numel(); ++i) {
        float val = data[i];
        if (std::abs(val - 0.0f) < 1e-6f) {
            zero_count++;
        } else if (std::abs(val - 1.0f) < 1e-6f) {
            one_count++;
        } else {
            test_passed = false;
            std::cout << "Invalid value: " << val << std::endl;
            break;
        }
    }

    if (test_passed) {
        float actual_zero_rate = static_cast<float>(zero_count) / tensor.numel();
        std::cout << "Zero rate: " << actual_zero_rate * 100 << "% (expected: 80%)" << std::endl;
        std::cout << "PASS: randbool_inplace function test" << std::endl;
    } else {
        std::cout << "FAIL: randbool_inplace function test - invalid values" << std::endl;
    }
}

int main() {
    Logger::get_instance().set_quiet_mode(true);
    std::cout << "Starting CPU backend create functions tests..." << std::endl;

    try {
        test_full();
        test_full_inplace();
        test_randn();
        test_randn_inplace();
        test_uniform();
        test_uniform_inplace();
        test_randint();
        test_randint_inplace();
        test_randbool();
        test_randbool_inplace();

        std::cout << "\n=== All Create Functions Tests Completed ===" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Exception occurred during testing: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
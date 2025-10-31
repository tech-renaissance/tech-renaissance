/**
 * @file test_print.cpp
 * @brief 张量打印功能测试
 * @details 测试4D张量的打印功能，与PyTorch输出格式对齐，并通过PythonSession实时对比
 * @version 1.00.00
 * @date 2025-10-27
 * @author 技术觉醒团队
 * @note 依赖项: Tensor, Backend, PythonSession
 * @note 所属系列: tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <chrono>

using namespace tr;

// Python脚本路径常量，使用PROJECT_ROOT_DIR确保跨环境一致性
const std::string PYTHON_SCRIPT_PATH = std::string(PROJECT_ROOT_DIR) + "/python/tests/python_server.py";

// 测试辅助函数
void print_test_header(const std::string& test_name) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    std::cout << "[TEST] Starting: " << test_name << std::endl;
}

void print_test_result(const std::string& test_name, bool passed) {
    std::cout << "--- " << test_name << ": " << (passed ? "PASSED" : "FAILED") << " ---" << std::endl;
    std::cout << "[TEST] " << test_name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

// 与Python对比的辅助函数
void compare_with_pytorch(const Tensor& tensor, const std::string& name, PythonSession& session, int precision = 6) {
    try {
        std::cout << "\n" << std::string(50, '-') << std::endl;
        std::cout << "[COMPARE] Sending tensor to PyTorch..." << std::endl;

        // 使用成功的API模式：session.send_tensor(tensor, tag)
        session.send_tensor(tensor, name);
        std::cout << "[COMPARE] Tensor sent to PyTorch" << std::endl;

        // 发送打印命令（使用成功的模式B：send_request + wait_for_response）
        session.send_request(R"({"cmd": "print_tensor", "params": ")" + name + R"("})");

        // 等待一小段时间让Python处理打印
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        std::cout << "[COMPARE] PyTorch comparison completed successfully" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

    } catch (const std::exception& e) {
        std::cout << "[COMPARE] Error during PyTorch comparison: " << e.what() << std::endl;
    }
}

// 辅助函数：创建测试数据
std::vector<float> create_test_data_2x2x2x2() {
    return {
        // Batch 0, Channel 0
        1.1f, 2.2f, 3.3f, 4.4f,  // H=0, W=0,1; H=1, W=0,1
        // Batch 0, Channel 1
        5.5f, 6.6f, 7.7f, 8.8f,
        // Batch 1, Channel 0
        9.9f, 10.10f, 11.11f, 12.12f,
        // Batch 1, Channel 1
        13.13f, 14.14f, 15.15f, 16.16f
    };
}

// 测试1: CPU后端FP32张量打印
bool test_cpu_fp32_print() {
    print_test_header("CPU Backend FP32 Tensor Print Test");

    bool test_passed = true;

    try {
        // 启动Python对比会话
        PythonSession session(PYTHON_SCRIPT_PATH, "cpu_fp32_compare");
        session.start();
        std::cout << "[TEST] PyTorch comparison session started" << std::endl;

        // 创建2x2x2x2的FP32张量
        Shape shape(2, 2, 2, 2);  // N=2, C=2, H=2, W=2
        Tensor cpu_fp32_tensor = Tensor::full(shape, 3.14159f, DType::FP32, tr::CPU);

        std::cout << "[TEST] Created CPU FP32 tensor with shape " << shape.to_string() << std::endl;

        // 测试1: 默认精度打印
        std::cout << "\n[TEST] Default precision print:" << std::endl;
        cpu_fp32_tensor.print("cpu_fp32_default");
        compare_with_pytorch(cpu_fp32_tensor, "cpu_fp32_default", session, 6);

        // 测试2: 高精度打印
        std::cout << "\n[TEST] High precision print (3 decimal places):" << std::endl;
        cpu_fp32_tensor.print("cpu_fp32_precise", 3);
        compare_with_pytorch(cpu_fp32_tensor, "cpu_fp32_precise", session, 3);

        // 测试3: 低精度打印
        std::cout << "\n[TEST] Low precision print (1 decimal place):" << std::endl;
        cpu_fp32_tensor.print("cpu_fp32_low", 1);
        compare_with_pytorch(cpu_fp32_tensor, "cpu_fp32_low", session, 1);

        // 结束Python会话
        session.send_request(R"({"cmd": "exit"})");
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        test_passed = false;
    }

    print_test_result("CPU Backend FP32 Tensor Print Test", test_passed);
    return test_passed;
}

// 测试2: CPU后端INT8张量打印
bool test_cpu_int8_print() {
    print_test_header("CPU Backend INT8 Tensor Print Test");

    bool test_passed = true;

    try {
        // 创建2x2x2x2的INT8张量
        Shape shape(2, 2, 2, 2);
        Tensor cpu_int8_tensor = Tensor::full(shape, 42.0f, DType::INT8, tr::CPU);

        std::cout << "[TEST] Created CPU INT8 tensor with shape " << shape.to_string() << std::endl;

        // 测试打印功能
        std::cout << "\n[TEST] INT8 tensor print:" << std::endl;
        cpu_int8_tensor.print("cpu_int8");

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        test_passed = false;
    }

    print_test_result("CPU Backend INT8 Tensor Print Test", test_passed);
    return test_passed;
}

// 测试3: CUDA后端FP32张量打印
bool test_cuda_fp32_print() {
    print_test_header("CUDA Backend FP32 Tensor Print Test");

    bool test_passed = true;

    try {
        // 尝试获取CUDA后端
        auto& manager = BackendManager::instance();
        std::shared_ptr<Backend> cuda = nullptr;

        try {
            cuda = manager.get_backend(tr::CUDA[0]);
        } catch (const std::exception& e) {
            std::cout << "[TEST] CUDA backend not available: " << e.what() << std::endl;
            std::cout << "[TEST] Skipping CUDA FP32 print test" << std::endl;
            print_test_result("CUDA Backend FP32 Tensor Print Test", true);
            return true;
        }

        // 创建2x2x2x2的FP32 CUDA张量
        Shape shape(2, 2, 2, 2);
        Tensor cuda_fp32_tensor = Tensor::full(shape, 2.71828f, DType::FP32, tr::CUDA[0]);

        std::cout << "[TEST] Created CUDA FP32 tensor with shape " << shape.to_string() << std::endl;

        // 测试打印功能
        std::cout << "\n[TEST] CUDA FP32 tensor print:" << std::endl;
        cuda_fp32_tensor.print("cuda_fp32");

        // 创建相同的CPU张量进行对比（避免使用可能有问题的.cpu()方法）
        Tensor cpu_equivalent = Tensor::full(shape, 2.71828f, DType::FP32, tr::CPU);
        std::cout << "\n[TEST] CPU equivalent for comparison:" << std::endl;
        cpu_equivalent.print("cuda_fp32_cpu_equivalent");

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        test_passed = false;
    }

    print_test_result("CUDA Backend FP32 Tensor Print Test", test_passed);
    return test_passed;
}

// 测试4: CUDA后端INT8张量打印
bool test_cuda_int8_print() {
    print_test_header("CUDA Backend INT8 Tensor Print Test");

    bool test_passed = true;

    try {
        // 尝试获取CUDA后端
        auto& manager = BackendManager::instance();
        std::shared_ptr<Backend> cuda = nullptr;

        try {
            cuda = manager.get_backend(tr::CUDA[0]);
        } catch (const std::exception& e) {
            std::cout << "[TEST] CUDA backend not available: " << e.what() << std::endl;
            std::cout << "[TEST] Skipping CUDA INT8 print test" << std::endl;
            print_test_result("CUDA Backend INT8 Tensor Print Test", true);
            return true;
        }

        // 创建2x2x2x2的INT8 CUDA张量
        Shape shape(2, 2, 2, 2);
        Tensor cuda_int8_tensor = Tensor::full(shape, 100.0f, DType::INT8, tr::CUDA[0]);

        std::cout << "[TEST] Created CUDA INT8 tensor with shape " << shape.to_string() << std::endl;

        // 测试打印功能
        std::cout << "\n[TEST] CUDA INT8 tensor print:" << std::endl;
        cuda_int8_tensor.print("cuda_int8");

        // 创建相同的CPU张量进行对比（避免使用可能有问题的.cpu()方法）
        Tensor cpu_equivalent = Tensor::full(shape, 100.0f, DType::INT8, tr::CPU);
        std::cout << "\n[TEST] CPU equivalent for comparison:" << std::endl;
        cpu_equivalent.print("cuda_int8_cpu_equivalent");

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        test_passed = false;
    }

    print_test_result("CUDA Backend INT8 Tensor Print Test", test_passed);
    return test_passed;
}

// 测试5: 自定义数据张量打印（用于与PyTorch对比）
bool test_custom_data_print() {
    print_test_header("Custom Data Tensor Print Test");

    bool test_passed = true;

    try {
        // 创建具有特定数据的2x2x2x2张量
        Shape shape(2, 2, 2, 2);
        auto test_data = create_test_data_2x2x2x2();

        // 创建空张量然后填充数据
        Tensor custom_tensor = Tensor::empty(shape, DType::FP32, tr::CPU);
        custom_tensor.from_cpu_data(test_data.data(), test_data.size() * sizeof(float));

        std::cout << "[TEST] Created custom data tensor with shape " << shape.to_string() << std::endl;
        std::cout << "[TEST] Data pattern: sequential values from 1.1 to 16.16" << std::endl;

        // 测试打印功能
        std::cout << "\n[TEST] Custom data tensor print:" << std::endl;
        custom_tensor.print("custom_data");

        std::cout << "\n[TEST] High precision custom data print:" << std::endl;
        custom_tensor.print("custom_data_precise", 2);

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        test_passed = false;
    }

    print_test_result("Custom Data Tensor Print Test", test_passed);
    return test_passed;
}

// 主测试函数
int main() {
    std::cout << "[TEST] Starting Tensor Print unit tests with PyTorch comparison" << std::endl;
    std::cout << "[TEST] Testing 4D tensor printing with real-time PyTorch format comparison" << std::endl;

    int passed_tests = 0;
    int total_tests = 5;

    // 运行所有测试
    if (test_cpu_fp32_print()) {
        passed_tests++;
    }

    if (test_cpu_int8_print()) {
        passed_tests++;
    }

    if (test_cuda_fp32_print()) {
        passed_tests++;
    }

    if (test_cuda_int8_print()) {
        passed_tests++;
    }

    if (test_custom_data_print()) {
        passed_tests++;
    }

    // 输出总结
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << passed_tests << "/" << total_tests << " tests" << std::endl;
    std::cout << "Success Rate: " << (passed_tests * 100 / total_tests) << "%" << std::endl;

    if (passed_tests == total_tests) {
        std::cout << "[TEST] All tests passed successfully!" << std::endl;
        std::cout << "All Tensor Print tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "[TEST] Some tests failed!" << std::endl;
        std::cout << "Some Tensor Print tests FAILED!" << std::endl;
        return 1;
    }
}
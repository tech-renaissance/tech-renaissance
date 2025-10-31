/**
 * @file test_device.cpp
 * @brief 设备类单元测试
 * @details 测试Device类的所有功能，包括构造、验证和比较等
 * @version 1.00.00
 * @date 2025-10-26
 * @author 技术觉醒团队
 * @note 依赖项: device.h, iostream, string
 * @note 所属系列: tests
 */

#include <iostream>
#include <string>
#include <cassert>
#include "tech_renaissance.h"

using namespace tr;

// 测试辅助函数
void test_assert(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << std::endl;
        assert(false);
    } else {
        std::cout << "[PASS] " << message << std::endl;
    }
}

// 测试有效设备构造
void test_valid_device_construction() {
    std::cout << "\n=== Testing Valid Device Construction ===" << std::endl;

    // 测试CPU设备构造
    Device cpu_device("CPU", -1);
    test_assert(cpu_device.name == "CPU", "CPU device name");
    test_assert(cpu_device.index == -1, "CPU device index");
    test_assert(cpu_device.is_cpu(), "CPU device is_cpu()");
    test_assert(!cpu_device.is_cuda(), "CPU device is_cuda() is false");
    test_assert(cpu_device.str() == "CPU", "CPU device string representation");
    test_assert(cpu_device.to_string() == "CPU", "CPU device to_string()");

    // 测试CUDA设备构造
    Device cuda_device0("CUDA", 0);
    test_assert(cuda_device0.name == "CUDA", "CUDA device name");
    test_assert(cuda_device0.index == 0, "CUDA device index 0");
    test_assert(!cuda_device0.is_cpu(), "CUDA device is_cpu() is false");
    test_assert(cuda_device0.is_cuda(), "CUDA device is_cuda()");
    test_assert(cuda_device0.str() == "CUDA:0", "CUDA:0 device string representation");

    Device cuda_device1("CUDA", 1);
    test_assert(cuda_device1.str() == "CUDA:1", "CUDA:1 device string representation");

    // 测试静态全局对象
    test_assert(tr::CPU.name == "CPU", "Global CPU device name");
    test_assert(tr::CPU.index == -1, "Global CPU device index");
    test_assert(tr::CPU.is_cpu(), "Global CPU device is_cpu()");

    test_assert(tr::CUDA[0].name == "CUDA", "Global CUDA[0] device name");
    test_assert(tr::CUDA[0].index == 0, "Global CUDA[0] device index");
    test_assert(tr::CUDA[0].is_cuda(), "Global CUDA[0] device is_cuda()");
}

// 测试专家要求的设备验证
void test_device_validation() {
    std::cout << "\n=== Testing Device Validation ===" << std::endl;

    // 测试无效的设备类型
    try {
        Device invalid_device("INVALID", 0);
        test_assert(false, "Invalid device type should throw exception");
    } catch (const std::invalid_argument&) {
        test_assert(true, "Invalid device type throws invalid_argument exception");
    }

    try {
        Device invalid_device("OpenCL", 0);
        test_assert(false, "Unsupported device type should throw exception");
    } catch (const std::invalid_argument&) {
        test_assert(true, "Unsupported device type throws invalid_argument exception");
    }

    // 测试CPU设备的无效索引
    try {
        Device invalid_cpu("CPU", 0);  // CPU index must be -1
        test_assert(false, "CPU with index 0 should throw exception");
    } catch (const std::invalid_argument&) {
        test_assert(true, "CPU with invalid index throws invalid_argument exception");
    }

    try {
        Device invalid_cpu("CPU", 1);  // CPU index must be -1
        test_assert(false, "CPU with index 1 should throw exception");
    } catch (const std::invalid_argument&) {
        test_assert(true, "CPU with invalid index throws invalid_argument exception");
    }

    try {
        Device invalid_cpu("CPU", -2);  // CPU index must be -1
        test_assert(false, "CPU with index -2 should throw exception");
    } catch (const std::invalid_argument&) {
        test_assert(true, "CPU with invalid index throws invalid_argument exception");
    }

    // 测试CUDA设备的无效索引
    try {
        Device invalid_cuda("CUDA", -1);  // CUDA index must be >= 0
        test_assert(false, "CUDA with index -1 should throw exception");
    } catch (const std::invalid_argument&) {
        test_assert(true, "CUDA with invalid index throws invalid_argument exception");
    }

    try {
        Device invalid_cuda("CUDA", 8);  // CUDA index must be < 8
        test_assert(false, "CUDA with index 8 should throw exception");
    } catch (const std::invalid_argument&) {
        test_assert(true, "CUDA with invalid index throws invalid_argument exception");
    }

    try {
        Device invalid_cuda("CUDA", 100);  // CUDA index must be < 8
        test_assert(false, "CUDA with index 100 should throw exception");
    } catch (const std::invalid_argument&) {
        test_assert(true, "CUDA with invalid index throws invalid_argument exception");
    }
}

// 测试设备比较
void test_device_comparison() {
    std::cout << "\n=== Testing Device Comparison ===" << std::endl;

    Device cpu1("CPU", -1);
    Device cpu2("CPU", -1);
    Device cuda0("CUDA", 0);
    Device cuda1("CUDA", 1);
    Device another_cuda0("CUDA", 0);

    // 测试相等性比较
    test_assert(cpu1 == cpu2, "Identical CPU devices should be equal");
    test_assert(cuda0 == another_cuda0, "Identical CUDA devices should be equal");
    test_assert(!(cpu1 == cuda0), "CPU and CUDA devices should not be equal");
    test_assert(!(cuda0 == cuda1), "Different CUDA devices should not be equal");

    // 测试不等性比较
    test_assert(!(cpu1 != cpu2), "Identical CPU devices should not be unequal");
    test_assert(!(cuda0 != another_cuda0), "Identical CUDA devices should not be unequal");
    test_assert(cpu1 != cuda0, "CPU and CUDA devices should be unequal");
    test_assert(cuda0 != cuda1, "Different CUDA devices should be unequal");
}

// 测试设备类型检查方法
void test_device_type_methods() {
    std::cout << "\n=== Testing Device Type Methods ===" << std::endl;

    Device cpu_device("CPU", -1);
    Device cuda_device("CUDA", 0);

    // 测试is_cpu()方法
    test_assert(cpu_device.is_cpu(), "CPU device is_cpu() should return true");
    test_assert(!cuda_device.is_cpu(), "CUDA device is_cpu() should return false");

    // 测试is_cuda()方法
    test_assert(!cpu_device.is_cuda(), "CPU device is_cuda() should return false");
    test_assert(cuda_device.is_cuda(), "CUDA device is_cuda() should return true");
}

// 测试字符串表示
void test_string_representation() {
    std::cout << "\n=== Testing String Representation ===" << std::endl;

    Device cpu_device("CPU", -1);
    Device cuda_device0("CUDA", 0);
    Device cuda_device1("CUDA", 1);

    // 测试str()方法
    test_assert(cpu_device.str() == "CPU", "CPU device str()");
    test_assert(cuda_device0.str() == "CUDA:0", "CUDA:0 device str()");
    test_assert(cuda_device1.str() == "CUDA:1", "CUDA:1 device str()");

    // 测试to_string()方法
    test_assert(cpu_device.to_string() == "CPU", "CPU device to_string()");
    test_assert(cuda_device0.to_string() == "CUDA:0", "CUDA:0 device to_string()");
    test_assert(cuda_device1.to_string() == "CUDA:1", "CUDA:1 device to_string()");
}

// 测试全局静态对象
void test_global_devices() {
    std::cout << "\n=== Testing Global Device Objects ===" << std::endl;

    // 测试全局CPU设备
    test_assert(tr::CPU.name == "CPU", "Global CPU device name");
    test_assert(tr::CPU.index == -1, "Global CPU device index");
    test_assert(tr::CPU.is_cpu(), "Global CPU device type");
    test_assert(!tr::CPU.is_cuda(), "Global CPU device not CUDA");

    // 测试全局CUDA设备数组
    for (int i = 0; i < 8; ++i) {
        test_assert(tr::CUDA[i].name == "CUDA", "Global CUDA device name " + std::to_string(i));
        test_assert(tr::CUDA[i].index == i, "Global CUDA device index " + std::to_string(i));
        test_assert(tr::CUDA[i].is_cuda(), "Global CUDA device type " + std::to_string(i));
        test_assert(!tr::CUDA[i].is_cpu(), "Global CUDA device not CPU " + std::to_string(i));
        test_assert(tr::CUDA[i].str() == "CUDA:" + std::to_string(i),
                   "Global CUDA device string " + std::to_string(i));
    }
}

// 测试边界情况
void test_edge_cases() {
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;

    // 测试边界有效的CUDA设备索引
    Device cuda_first("CUDA", 0);
    Device cuda_last("CUDA", 7);

    test_assert(cuda_first.is_cuda(), "First CUDA device (index 0)");
    test_assert(cuda_last.is_cuda(), "Last CUDA device (index 7)");
    test_assert(cuda_first.str() == "CUDA:0", "First CUDA device string");
    test_assert(cuda_last.str() == "CUDA:7", "Last CUDA device string");

    // 测试字符串表示的一致性
    Device test_cuda("CUDA", 3);
    test_assert(test_cuda.str() == test_cuda.to_string(),
               "str() and to_string() consistency for CUDA:3");

    Device test_cpu("CPU", -1);
    test_assert(test_cpu.str() == test_cpu.to_string(),
               "str() and to_string() consistency for CPU");
}

int main() {
    std::cout << "=== Device Class Unit Tests ===" << std::endl;
    std::cout << "Testing comprehensive functionality of the Device class" << std::endl;

    try {
        test_valid_device_construction();
        test_device_validation();
        test_device_comparison();
        test_device_type_methods();
        test_string_representation();
        test_global_devices();
        test_edge_cases();

        std::cout << "\n=== All Device Tests PASSED! ===" << std::endl;
        std::cout << "Device class implementation is working correctly." << std::endl;
        std::cout << "All expert-identified validation requirements have been implemented." << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Device test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n[ERROR] Device test failed with unknown exception" << std::endl;
        return 1;
    }
}
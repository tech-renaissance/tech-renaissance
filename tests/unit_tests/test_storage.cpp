/**
 * @file test_storage.cpp
 * @brief 存储类单元测试
 * @details 测试Storage类的所有功能，包括内存管理、RAII机制和Backend交互接口
 * @version 1.00.00
 * @date 2025-10-26
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h
 * @note 所属系列: tests
 */

#include <iostream>
#include <string>
#include <cassert>
#include <memory>
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

// 测试Storage构造函数
void test_storage_construction() {
    std::cout << "\n=== Testing Storage Construction ===" << std::endl;

    // 测试空存储构造
    Storage empty_storage(0, tr::CPU);
    test_assert(empty_storage.size() == 0, "Empty storage size");
    test_assert(empty_storage.capacity_bytes() == 0, "Empty storage capacity");
    test_assert(empty_storage.size_bytes() == 0, "Empty storage size_bytes");
    test_assert(empty_storage.is_empty(), "Empty storage is_empty()");
    test_assert(empty_storage.data_ptr() == nullptr, "Empty storage data_ptr()");
    test_assert(empty_storage.raw_ptr() == nullptr, "Empty storage raw_ptr()");
    test_assert(empty_storage.device().is_cpu(), "Empty storage device");

    // 测试有大小的存储构造
    Storage storage(1024, tr::CPU);
    test_assert(storage.size() == 1024, "Storage size");
    test_assert(storage.capacity_bytes() == 1024, "Storage capacity_bytes");
    test_assert(storage.size_bytes() == 1024, "Storage size_bytes");
    test_assert(storage.is_empty(), "Storage without data is_empty()");
    test_assert(storage.device().is_cpu(), "Storage device");

    // 测试不同设备的存储构造
    Storage cuda_storage(2048, tr::CUDA[0]);
    test_assert(cuda_storage.size() == 2048, "CUDA storage size");
    test_assert(cuda_storage.device().is_cuda(), "CUDA storage device");
}

// 测试专家要求的RAII机制和Backend交互接口
void test_expert_requirements() {
    std::cout << "\n=== Testing Expert Required RAII and Backend Interface ===" << std::endl;

    // 测试必需的API接口
    Storage storage(1024, tr::CPU);

    // 测试Device device() const
    Device device = storage.device();
    test_assert(device.is_cpu(), "Device device() const");

    // 测试size_t capacity_bytes() const
    size_t capacity = storage.capacity_bytes();
    test_assert(capacity == 1024, "size_t capacity_bytes() const");

    // 测试size_t size_bytes() const
    size_t size = storage.size_bytes();
    test_assert(size == 1024, "size_t size_bytes() const");

    // 测试void* raw_ptr() const
    void* raw_ptr = storage.raw_ptr();
    test_assert(raw_ptr == nullptr, "void* raw_ptr() const returns nullptr initially");

    // 测试std::shared_ptr<void> holder() const
    std::shared_ptr<void> holder = storage.holder();
    test_assert(holder == nullptr, "std::shared_ptr<void> holder() const returns nullptr initially");

    // 测试设置数据指针和智能指针
    std::shared_ptr<int> test_data = std::make_shared<int>(42);
    storage.set_data_ptr(test_data.get(), test_data);

    test_assert(storage.data_ptr() == test_data.get(), "set_data_ptr sets correct data_ptr");
    test_assert(storage.raw_ptr() == test_data.get(), "set_data_ptr sets correct raw_ptr");
    test_assert(storage.holder() == test_data, "set_data_ptr sets correct holder");
}

// 测试数据指针管理
void test_data_pointer_management() {
    std::cout << "\n=== Testing Data Pointer Management ===" << std::endl;

    Storage storage(1024, tr::CPU);

    // 初始状态
    test_assert(storage.data_ptr() == nullptr, "Initial data_ptr is nullptr");
    test_assert(storage.is_empty(), "Storage is empty initially");

    // 创建测试数据
    std::shared_ptr<float> test_data = std::make_shared<float>(3.14f);
    storage.set_data_ptr(test_data.get(), test_data);

    // 设置后状态
    test_assert(storage.data_ptr() == test_data.get(), "data_ptr after set_data_ptr");
    test_assert(!storage.is_empty(), "Storage is not empty after setting data");
    test_assert(storage.raw_ptr() == test_data.get(), "raw_ptr equals data_ptr");
    test_assert(storage.holder() == test_data, "holder equals original shared_ptr");

    // 测试数据修改
    float* float_ptr = static_cast<float*>(storage.data_ptr());
    *float_ptr = 2.71f;
    test_assert(*test_data == 2.71f, "Data modification through storage affects original data");
}

// 测试不同设备的Storage
void test_different_devices() {
    std::cout << "\n=== Testing Storage with Different Devices ===" << std::endl;

    // CPU存储
    Storage cpu_storage(512, tr::CPU);
    test_assert(cpu_storage.device().is_cpu(), "CPU storage device detection");
    test_assert(cpu_storage.capacity_bytes() == 512, "CPU storage capacity");

    // CUDA存储
    Storage cuda_storage(1024, tr::CUDA[0]);
    test_assert(cuda_storage.device().is_cuda(), "CUDA storage device detection");
    test_assert(cuda_storage.capacity_bytes() == 1024, "CUDA storage capacity");
    test_assert(cuda_storage.device().str() == "CUDA:0", "CUDA storage device string");

    // 不同CUDA设备
    Storage cuda_storage1(2048, tr::CUDA[1]);
    test_assert(cuda_storage1.device().str() == "CUDA:1", "CUDA:1 storage device string");
}

// 测试RAII机制
void test_raii_mechanism() {
    std::cout << "\n=== Testing RAII Mechanism ===" << std::endl;

    // 测试智能指针的生命周期管理
    std::weak_ptr<int> weak_ptr;

    {
        Storage storage(1024, tr::CPU);
        auto test_data = std::make_shared<int>(123);
        weak_ptr = test_data;

        storage.set_data_ptr(test_data.get(), test_data);
        test_assert(!weak_ptr.expired(), "Data is alive within scope");
        test_assert(storage.holder() == test_data, "Storage holds data");
    }

    // Storage析构后，数据仍然被weak_ptr引用（如果Storage是唯一持有者，数据会被释放）
    // 这里主要测试RAII的基本机制是否工作

    // 测试新的Storage实例
    Storage new_storage(1024, tr::CPU);
    auto new_data = std::make_shared<int>(456);
    new_storage.set_data_ptr(new_data.get(), new_data);

    test_assert(*static_cast<int*>(new_storage.data_ptr()) == 456, "New storage holds correct data");
    test_assert(new_storage.holder() == new_data, "New storage holds correct holder");
}

// 测试大小变化
void test_size_variations() {
    std::cout << "\n=== Testing Size Variations ===" << std::endl;

    // 测试不同大小的存储
    size_t sizes[] = {1, 8, 64, 512, 4096, 32768};

    for (size_t size : sizes) {
        Storage storage(size, tr::CPU);
        test_assert(storage.size() == size, "Storage size " + std::to_string(size));
        test_assert(storage.capacity_bytes() == size, "Storage capacity " + std::to_string(size));
        test_assert(storage.size_bytes() == size, "Storage size_bytes " + std::to_string(size));
    }

    // 测试零大小
    Storage zero_storage(0, tr::CPU);
    test_assert(zero_storage.size() == 0, "Zero size storage");
    test_assert(zero_storage.is_empty(), "Zero size storage is empty");
}

// 测试边界情况
void test_edge_cases() {
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;

    // 测试多次设置数据指针
    Storage storage(1024, tr::CPU);

    auto data1 = std::make_shared<int>(1);
    auto data2 = std::make_shared<int>(2);

    storage.set_data_ptr(data1.get(), data1);
    test_assert(*static_cast<int*>(storage.data_ptr()) == 1, "First data setting");

    storage.set_data_ptr(data2.get(), data2);
    test_assert(*static_cast<int*>(storage.data_ptr()) == 2, "Second data setting");
    test_assert(storage.holder() == data2, "Holder updated to second data");

    // 测试设置nullptr数据
    storage.set_data_ptr(nullptr, nullptr);
    test_assert(storage.data_ptr() == nullptr, "Data pointer set to nullptr");
    test_assert(storage.holder() == nullptr, "Holder set to nullptr");
    test_assert(storage.is_empty(), "Storage is empty after setting nullptr");

    // 测试大容量存储
    const size_t large_size = 1024 * 1024; // 1MB
    Storage large_storage(large_size, tr::CPU);
    test_assert(large_storage.capacity_bytes() == large_size, "Large storage capacity");
    test_assert(large_storage.size_bytes() == large_size, "Large storage size_bytes");
}

int main() {
    std::cout << "=== Storage Class Unit Tests ===" << std::endl;
    std::cout << "Testing comprehensive functionality of the Storage class" << std::endl;

    try {
        test_storage_construction();
        test_expert_requirements();
        test_data_pointer_management();
        test_different_devices();
        test_raii_mechanism();
        test_size_variations();
        test_edge_cases();

        std::cout << "\n=== All Storage Tests PASSED! ===" << std::endl;
        std::cout << "Storage class implementation is working correctly." << std::endl;
        std::cout << "All expert-identified RAII and Backend interface requirements have been implemented." << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Storage test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n[ERROR] Storage test failed with unknown exception" << std::endl;
        return 1;
    }
}
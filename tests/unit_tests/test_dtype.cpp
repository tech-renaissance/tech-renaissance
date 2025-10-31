/**
 * @file test_dtype.cpp
 * @brief 数据类型单元测试
 * @details 测试DType枚举和相关函数的功能
 * @version 1.00.00
 * @date 2025-10-26
 * @author 技术觉醒团队
 * @note 依赖项: dtype.h, iostream, string
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

// 测试数据类型大小
void test_dtype_size() {
    std::cout << "\n=== Testing DType Size ===" << std::endl;

    test_assert(dtype_size(DType::FP32) == 4, "FP32 size should be 4 bytes");
    test_assert(dtype_size(DType::INT8) == 1, "INT8 size should be 1 byte");
}

// 测试数据类型转字符串
void test_dtype_to_string() {
    std::cout << "\n=== Testing DType to String ===" << std::endl;

    test_assert(dtype_to_string(DType::FP32) == "fp32", "FP32 to string");
    test_assert(dtype_to_string(DType::INT8) == "int8", "INT8 to string");
}

// 测试字符串转数据类型
void test_string_to_dtype() {
    std::cout << "\n=== Testing String to DType ===" << std::endl;

    // 测试有效的字符串转换
    test_assert(string_to_dtype("fp32") == DType::FP32, "String 'fp32' to FP32");
    test_assert(string_to_dtype("float32") == DType::FP32, "String 'float32' to FP32");
    test_assert(string_to_dtype("int8") == DType::INT8, "String 'int8' to INT8");

    // 测试大小写不敏感（当前实现是大小写敏感的，如果需要可以修改）
    // 这里按照当前实现测试，如果未来支持大小写不敏感，需要更新测试
}

// 测试专家提出的Fail-Fast原则
void test_fail_fast_principle() {
    std::cout << "\n=== Testing Fail-Fast Principle ===" << std::endl;

    // 测试无效的字符串应该抛出异常，而不是静默转换
    try {
        DType invalid1 = string_to_dtype("invalid_type");
        test_assert(false, "Invalid string 'invalid_type' should throw exception");
    } catch (const std::invalid_argument&) {
        test_assert(true, "Invalid string 'invalid_type' throws invalid_argument");
    }

    try {
        DType invalid2 = string_to_dtype("fp16");
        test_assert(false, "Unsupported type 'fp16' should throw exception");
    } catch (const std::invalid_argument&) {
        test_assert(true, "Unsupported type 'fp16' throws invalid_argument");
    }

    try {
        DType invalid3 = string_to_dtype("uint8");
        test_assert(false, "Unsupported type 'uint8' should throw exception");
    } catch (const std::invalid_argument&) {
        test_assert(true, "Unsupported type 'uint8' throws invalid_argument");
    }

    try {
        DType invalid4 = string_to_dtype("");
        test_assert(false, "Empty string should throw exception");
    } catch (const std::invalid_argument&) {
        test_assert(true, "Empty string throws invalid_argument");
    }
}

// 测试双向转换一致性
void test_conversion_consistency() {
    std::cout << "\n=== Testing Conversion Consistency ===" << std::endl;

    // 测试双向转换的一致性
    DType fp32 = DType::FP32;
    std::string fp32_str = dtype_to_string(fp32);
    DType fp32_back = string_to_dtype(fp32_str);
    test_assert(fp32 == fp32_back, "FP32 round-trip conversion consistency");

    DType int8 = DType::INT8;
    std::string int8_str = dtype_to_string(int8);
    DType int8_back = string_to_dtype(int8_str);
    test_assert(int8 == int8_back, "INT8 round-trip conversion consistency");
}

// 测试边界情况
void test_edge_cases() {
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;

    // 测试不同格式的有效字符串
    test_assert(string_to_dtype("fp32") == DType::FP32, "Lowercase 'fp32'");
    test_assert(string_to_dtype("float32") == DType::FP32, "Lowercase 'float32'");
    test_assert(string_to_dtype("int8") == DType::INT8, "Lowercase 'int8'");

    // 测试数值类型的正确性
    test_assert(static_cast<int>(DType::FP32) == 1, "FP32 enum value");
    test_assert(static_cast<int>(DType::INT8) == 2, "INT8 enum value");
}

int main() {
    std::cout << "=== DType Unit Tests ===" << std::endl;
    std::cout << "Testing comprehensive functionality of the DType enum and related functions" << std::endl;

    try {
        test_dtype_size();
        test_dtype_to_string();
        test_string_to_dtype();
        test_fail_fast_principle();
        test_conversion_consistency();
        test_edge_cases();

        std::cout << "\n=== All DType Tests PASSED! ===" << std::endl;
        std::cout << "DType implementation is working correctly." << std::endl;
        std::cout << "Fail-Fast principle has been properly implemented." << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] DType test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n[ERROR] DType test failed with unknown exception" << std::endl;
        return 1;
    }
}
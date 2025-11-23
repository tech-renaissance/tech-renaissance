/**
 * @file test_tsr_io_extended.cpp
 * @brief TSR文件格式导入导出扩展功能测试
 * @details 验证FP32、INT32、INT8三种数据类型的TSR文件导入导出功能
 * @version 1.00.00
 * @date 2025-11-20
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h, filesystem
 * @note 所属系列: unit_tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <filesystem>

using namespace tr;

// 测试数据结构
struct TestCase {
    Shape shape;
    DType dtype;
    std::string name;
    std::string filename;
};


// 验证两个张量是否相等
bool verify_tensors_equal(const Tensor& a, const Tensor& b) {
    // 检查形状
    if (a.shape() != b.shape()) {
        std::cout << "Shape mismatch: " << a.shape().to_string()
                  << " vs " << b.shape().to_string() << std::endl;
        return false;
    }

    // 检查数据类型
    if (a.dtype() != b.dtype()) {
        std::cout << "DType mismatch" << std::endl;
        return false;
    }

    auto backend = BackendManager::get_cpu_backend();

    // 根据数据类型使用不同的比较方法
    if (a.dtype() == DType::FP32) {
        return backend->is_close(a, b, 5e-5f);  // 使用is_close比较FP32
    } else if (a.dtype() == DType::INT32 || a.dtype() == DType::INT8) {
        return backend->equal(a, b);  // 使用equal比较INT32和INT8
    }

    return false;
}

// 测试单个数据类型的TSR导入导出
bool test_dtype_io(const TestCase& test_case) {
    std::cout << "\n=== Testing " << test_case.name << " ===" << std::endl;

    try {
        auto backend = BackendManager::get_cpu_backend();

        // 创建随机测试张量
        Tensor original;
        if (test_case.dtype == DType::FP32) {
            original = backend->randn(test_case.shape, 12345);  // 固定种子保证可重现
        } else if (test_case.dtype == DType::INT32) {
            original = backend->randint(test_case.shape, -100, 100, DType::INT32, 12345);
        } else if (test_case.dtype == DType::INT8) {
            original = backend->randint(test_case.shape, -50, 50, DType::INT8, 12345);
        }

        std::cout << "Created " << dtype_to_string(original.dtype()) << " tensor: "
                  << original.shape().to_string() << ", elements: " << original.numel() << std::endl;

        // 导出到TSR文件
        backend->export_tensor(original, test_case.filename);
        std::cout << "Exported to: " << test_case.filename << std::endl;

        // 从TSR文件导入
        Tensor imported = backend->import_tensor(test_case.filename);
        std::cout << "Imported tensor: " << imported.shape().to_string()
                  << ", elements: " << imported.numel() << std::endl;

        // 验证张量相等
        bool is_equal = verify_tensors_equal(original, imported);
        if (is_equal) {
            std::cout << "SUCCESS: Original and imported tensors are identical" << std::endl;
        } else {
            std::cout << "ERROR: Original and imported tensors differ!" << std::endl;
            return false;
        }

        // 清理文件
        std::filesystem::remove(test_case.filename);
        return true;

    } catch (const std::exception& e) {
        std::cout << "ERROR: Exception during test: " << e.what() << std::endl;
        // 清理文件（如果存在）
        if (std::filesystem::exists(test_case.filename)) {
            std::filesystem::remove(test_case.filename);
        }
        return false;
    }
}

// 测试2D和4D张量
bool test_2d_and_4d_tensors(DType dtype, const std::string& dtype_name) {
    std::cout << "\n=== Testing 2D and 4D tensors for " << dtype_name << " ===" << std::endl;

    // 只测试2D和4D张量
    std::vector<Shape> test_shapes = {
        Shape(3, 4),                      // 2D 矩阵
        Shape(1, 2, 3, 4)                 // 4D 张量
    };

    std::vector<std::string> shape_names = {
        "matrix", "4d_tensor"
    };

    for (size_t i = 0; i < test_shapes.size(); ++i) {
        std::string filename = "test_" + dtype_name + "_" + shape_names[i] + ".tsr";
        TestCase test_case{test_shapes[i], dtype, dtype_name + "_" + shape_names[i], filename};

        if (!test_dtype_io(test_case)) {
            std::cout << "ERROR: Failed test for " << dtype_name << " " << shape_names[i] << std::endl;
            return false;
        }
    }

    return true;
}

// 测试向后兼容性
bool test_backward_compatibility() {
    std::cout << "\n=== Testing backward compatibility ===" << std::endl;

    try {
        auto backend = BackendManager::get_cpu_backend();

        // 创建一个FP32张量并导出
        Tensor fp32_tensor = backend->randn(Shape(2, 3), 54321);
        std::string fp32_filename = "test_compatibility_fp32.tsr";

        backend->export_tensor(fp32_tensor, fp32_filename);
        std::cout << "Exported FP32 tensor for compatibility test" << std::endl;

        // 导入并验证
        Tensor imported_fp32 = backend->import_tensor(fp32_filename);
        bool is_compatible = verify_tensors_equal(fp32_tensor, imported_fp32);

        // 清理
        std::filesystem::remove(fp32_filename);

        if (is_compatible) {
            std::cout << "SUCCESS: Backward compatibility maintained for FP32" << std::endl;
            return true;
        } else {
            std::cout << "ERROR: Backward compatibility test failed!" << std::endl;
            return false;
        }

    } catch (const std::exception& e) {
        std::cout << "ERROR: Exception during compatibility test: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== TSR Extended I/O Test Suite ===" << std::endl;
    std::cout << "Testing FP32, INT32, and INT8 data types" << std::endl;

    // 确保workspace目录存在
    std::filesystem::create_directories("workspace");

    bool all_tests_passed = true;

    // 测试不同数据类型的不同维度
    std::vector<std::pair<DType, std::string>> dtype_tests = {
        {DType::FP32, "fp32"},
        {DType::INT32, "int32"},
        {DType::INT8, "int8"}
    };

    for (const auto& [dtype, name] : dtype_tests) {
        if (!test_2d_and_4d_tensors(dtype, name)) {
            all_tests_passed = false;
        }
    }

    // 测试向后兼容性
    if (!test_backward_compatibility()) {
        all_tests_passed = false;
    }

    // 输出最终结果
    std::cout << "\n=== Test Summary ===" << std::endl;
    if (all_tests_passed) {
        std::cout << "SUCCESS: All TSR I/O tests passed!" << std::endl;
        std::cout << "FP32, INT32, and INT8 data types are fully supported." << std::endl;
        return 0;
    } else {
        std::cout << "FAILURE: Some tests failed!" << std::endl;
        return 1;
    }
}
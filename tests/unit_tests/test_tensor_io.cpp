/**
 * @file test_tensor_io.cpp
 * @brief 张量导入导出功能测试样例
 * @details 测试CPU后端的.tsr文件格式导入导出功能，验证数据一致性
 * @version 1.00.00
 * @date 2025-10-27
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h
 * @note 所属系列: tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstring>


using namespace tr;

// 测试用的精度阈值
const float EPSILON = 1e-5f;

// 随机数生成器
class RandomGenerator {
private:
    std::mt19937 gen_;
    std::uniform_real_distribution<float> dist_float_;
    std::uniform_int_distribution<int16_t> dist_int16_; // 使用int16_t生成器

public:
    RandomGenerator() : gen_(42), dist_float_(-100.0f, 100.0f), dist_int16_(-128, 127) {}

    float random_float() {
        return dist_float_(gen_);
    }

    int8_t random_int8() {
        return static_cast<int8_t>(dist_int16_(gen_));
    }
};

static RandomGenerator rng;

// 比较张量是否相等（精确数据验证）
bool compare_tensors(const Tensor& a, const Tensor& b, float epsilon = EPSILON) {
    // 检查形状和数据类型
    if (a.shape() != b.shape()) {
        std::cout << "Shape mismatch: " << a.shape().to_string()
                  << " vs " << b.shape().to_string() << std::endl;
        return false;
    }

    if (a.dtype() != b.dtype()) {
        std::cout << "Dtype mismatch: " << static_cast<int>(a.dtype())
                  << " vs " << static_cast<int>(b.dtype()) << std::endl;
        return false;
    }

    if (a.is_empty() || b.is_empty()) {
        std::cout << "Empty tensor detected" << std::endl;
        return false;
    }

    // 进行精确的数据比较
    size_t element_count = a.numel();
    size_t element_size = a.dtype_size();
    size_t total_size = element_count * element_size;

    if (a.dtype() == DType::FP32) {
        const float* a_data = static_cast<const float*>(a.data_ptr());
        const float* b_data = static_cast<const float*>(b.data_ptr());

        for (size_t i = 0; i < element_count; i++) {
            float diff = std::abs(a_data[i] - b_data[i]);
            if (diff > epsilon) {
                std::cout << "Data mismatch at index " << i << ": "
                          << a_data[i] << " vs " << b_data[i]
                          << " (diff: " << diff << ")" << std::endl;
                return false;
            }
        }
    } else if (a.dtype() == DType::INT8) {
        const int8_t* a_data = static_cast<const int8_t*>(a.data_ptr());
        const int8_t* b_data = static_cast<const int8_t*>(b.data_ptr());

        for (size_t i = 0; i < element_count; i++) {
            if (a_data[i] != b_data[i]) {
                std::cout << "Data mismatch at index " << i << ": "
                          << static_cast<int>(a_data[i]) << " vs "
                          << static_cast<int>(b_data[i]) << std::endl;
                return false;
            }
        }
    } else {
        std::cout << "Unsupported dtype for comparison: " << static_cast<int>(a.dtype()) << std::endl;
        return false;
    }

    return true;
}

// 测试特定维度和数据类型的导入导出
bool test_tensor_io(const Shape& shape, DType dtype, const std::string& test_name) {
    std::cout << "\n=== Testing " << test_name << " ===" << std::endl;

    try {
        // 创建并填充原始张量
        Tensor original = Tensor::empty(shape, dtype, CPU);

        if (dtype == DType::FP32) {
            float* data = static_cast<float*>(original.data_ptr());
            for (size_t i = 0; i < original.numel(); i++) {
                data[i] = rng.random_float();
            }
        } else if (dtype == DType::INT8) {
            int8_t* data = static_cast<int8_t*>(original.data_ptr());
            for (size_t i = 0; i < original.numel(); i++) {
                data[i] = rng.random_int8();
            }
        }

        std::cout << "Original tensor: ";
        original.print();

        // 导出张量
        std::string filename = std::string(WORKSPACE_PATH) + "\\test_" + test_name + ".tsr";
        EXPORT_TENSOR(original, filename);
        std::cout << "Exported to: " << filename << std::endl;

        // 导入张量
        Tensor imported = IMPORT_TENSOR(filename);
        std::cout << "Imported tensor: ";
        imported.print();

        // 比较张量
        bool success = compare_tensors(original, imported);
        if (success) {
            std::cout << "[SUCCESS] " << test_name << " test passed!" << std::endl;
        } else {
            std::cout << "[FAILED] " << test_name << " test failed!" << std::endl;
        }

        // 清理测试文件
        std::remove(filename.c_str());

        return success;

    } catch (const std::exception& e) {
        std::cout << "[ERROR] " << test_name << " test error: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== Tensor Import/Export Test ===" << std::endl;
    std::cout << "Testing .tsr file format based on Scheme E" << std::endl;

    int total_tests = 0;
    int passed_tests = 0;

    // 测试不同维度和数据类型的张量
    std::vector<std::pair<Shape, DType>> test_cases = {
        {Shape(), DType::FP32},                              // 标量FP32
        {Shape(), DType::INT8},                              // 标量INT8
        {Shape(5), DType::FP32},                             // 1D张量FP32
        {Shape(3, 4), DType::FP32},                          // 2D张量FP32
        {Shape(2, 3, 4), DType::FP32},                       // 3D张量FP32
        {Shape(1, 2, 3, 4), DType::FP32},                   // 4D张量部分
        {Shape(2, 3, 4, 5), DType::FP32},                   // 完整4D张量FP32
        {Shape(2, 3), DType::INT8},                           // 2D张量INT8
        {Shape(1, 2, 3, 4), DType::INT8}                    // 4D张量INT8
    };

    std::vector<std::string> test_names = {
        "scalar_fp32",
        "scalar_int8",
        "1d_fp32",
        "2d_fp32",
        "3d_fp32",
        "4d_fp32_partial",
        "4d_fp32_full",
        "2d_int8",
        "4d_int8"
    };

    // 运行所有测试
    for (size_t i = 0; i < test_cases.size(); i++) {
        total_tests++;
        if (test_tensor_io(test_cases[i].first, test_cases[i].second, test_names[i])) {
            passed_tests++;
        }
    }

    // 输出测试结果
    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "Total tests: " << total_tests << std::endl;
    std::cout << "Passed tests: " << passed_tests << std::endl;
    std::cout << "Failed tests: " << (total_tests - passed_tests) << std::endl;

    if (passed_tests == total_tests) {
        std::cout << "[SUCCESS] All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "[FAILED] Some tests failed!" << std::endl;
        return 1;
    }
}
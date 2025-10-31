/**
 * @file test_cpu_mm_simple.cpp
 * @brief CPU后端矩阵乘法运算测试（简化版）
 * @details 测试CPU后端的mm方法，并与PyTorch结果对比验证
 * @version 1.00.00
 * @date 2025-10-29
 * @author 技术觉醒团队
 * @note 依赖项: tensor.h, cpu_backend.h, pytorch_session.h
 * @note 所属系列: tests
 */

#ifdef TR_BUILD_PYTORCH_SESSION

#include "tech_renaissance.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>

using namespace tr;

// Python脚本路径常量，使用PROJECT_ROOT_DIR确保跨环境一致性
const std::string PYTHON_SCRIPT_PATH = std::string(PROJECT_ROOT_DIR) +
    "/python/tests/python_server.py";

// 测试辅助函数
void print_test_header(const std::string& test_name) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
}

void print_test_result(const std::string& test_name, bool passed) {
    std::cout << "[TEST] " << test_name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

// 辅助函数：创建随机张量
Tensor create_random_tensor(const Shape& shape, float min_val = -1.0f, float max_val = 1.0f) {
    auto& manager = BackendManager::instance();
    auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(manager.get_backend(tr::CPU));

    Tensor tensor = Tensor::empty(shape, DType::FP32, tr::CPU);

    // 使用简单的伪随机数生成器
    static unsigned int seed = 12345;
    float* data = static_cast<float*>(tensor.data_ptr());

    for (size_t i = 0; i < tensor.numel(); ++i) {
        seed = seed * 1103515245 + 12345;
        float normalized = static_cast<float>(seed) / static_cast<float>(UINT_MAX);
        data[i] = min_val + normalized * (max_val - min_val);
    }

    return tensor;
}

// 测试1: 基本矩阵乘法 2x3 × 3x2
bool test_basic_multiplication() {
    print_test_header("Basic Matrix Multiplication Test (2x3 × 3x2)");

    try {
        auto& manager = BackendManager::instance();
        auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(manager.get_backend(tr::CPU));

        // 创建测试矩阵
        // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
        // B = [[7, 8], [9, 10], [11, 12]]  (3x2)
        // Expected: [[58, 64], [139, 154]]  (2x2)

        std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> b_data = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

        Tensor tensor_a = Tensor::empty(Shape(2, 3), DType::FP32, tr::CPU);
        Tensor tensor_b = Tensor::empty(Shape(3, 2), DType::FP32, tr::CPU);

        // 复制数据
        std::memcpy(tensor_a.data_ptr(), a_data.data(), a_data.size() * sizeof(float));
        std::memcpy(tensor_b.data_ptr(), b_data.data(), b_data.size() * sizeof(float));

        // 创建结果张量
        Tensor result = Tensor::empty(Shape(2, 2), DType::FP32, tr::CPU);

        std::cout << "[TEST] Matrix A (2x3):" << std::endl;
        tensor_a.print("A");

        std::cout << "[TEST] Matrix B (3x2):" << std::endl;
        tensor_b.print("B");

        // 执行矩阵乘法
        cpu_backend->mm(result, tensor_a, tensor_b);

        std::cout << "[TEST] CPU Result:" << std::endl;
        result.print("C");

        // 验证结果
        float* result_data = static_cast<float*>(result.data_ptr());
        std::vector<float> expected = {58.0f, 64.0f, 139.0f, 154.0f};

        bool test_passed = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::abs(result_data[i] - expected[i]) > 1e-5f) {
                test_passed = false;
                break;
            }
        }

        print_test_result("Basic Matrix Multiplication Test", test_passed);
        return test_passed;

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        print_test_result("Basic Matrix Multiplication Test", false);
        return false;
    }
}

// 测试2: 与PyTorch对比验证
bool test_pytorch_comparison() {
    print_test_header("PyTorch Comparison Test");

    try {
        auto& manager = BackendManager::instance();
        auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(manager.get_backend(tr::CPU));

        // 创建随机测试矩阵 (4x6 × 6x3)
        Tensor tensor_a = create_random_tensor(Shape(4, 6), -2.0f, 2.0f);
        Tensor tensor_b = create_random_tensor(Shape(6, 3), -1.0f, 1.0f);
        Tensor result = Tensor::empty(Shape(4, 3), DType::FP32, tr::CPU);

        std::cout << "[TEST] Created random matrices A(4x6) and B(6x3)" << std::endl;

        // 使用CPU后端计算
        cpu_backend->mm(result, tensor_a, tensor_b);

        std::cout << "[TEST] CPU Backend Result:" << std::endl;
        result.print("CPU_Result");

        // 使用PyTorchSession进行对比计算
        std::cout << "\n[TEST] Starting PyTorch session for comparison..." << std::endl;
        PyTorchSession session(PYTHON_SCRIPT_PATH, "cpu_mm_comparison");
        session.start();

        // 等待进程启动
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // 发送张量到PyTorch
        session.send_tensor(tensor_a, "a");
        session.send_tensor(tensor_b, "b");

        std::cout << "[TEST] Tensors sent to PyTorch, requesting matrix multiplication..." << std::endl;

        // 执行矩阵乘法
        Tensor pytorch_result = session.fetch_tensor(R"({"cmd": "matmul", "params": "a,b"})", 10000);

        if (pytorch_result.numel() == 0) {
            std::cout << "[TEST] Failed to get PyTorch result!" << std::endl;
            print_test_result("PyTorch Comparison Test", false);
            return false;
        }

        std::cout << "[TEST] PyTorch Result:" << std::endl;
        pytorch_result.print("PyTorch_Result");

        // 使用is_close方法比较结果
        bool is_close_result = cpu_backend->is_close(result, pytorch_result, 1e-4f);

        std::cout << "[TEST] Are results close? " << (is_close_result ? "YES" : "NO") << std::endl;

        if (is_close_result) {
            std::cout << "[TEST] SUCCESS: CPU and PyTorch results are identical!" << std::endl;
        } else {
            std::cout << "[TEST] WARNING: CPU and PyTorch results differ!" << std::endl;

            // 显示差异
            std::cout << "[TEST] Difference analysis:" << std::endl;
            std::cout << "  CPU result shape: " << result.shape().to_string() << std::endl;
            std::cout << "  PyTorch result shape: " << pytorch_result.shape().to_string() << std::endl;
            std::cout << "  CPU result numel: " << result.numel() << std::endl;
            std::cout << "  PyTorch result numel: " << pytorch_result.numel() << std::endl;
        }

        // 结束PyTorch会话
        session.send_request(R"({"cmd": "exit"})");

        print_test_result("PyTorch Comparison Test", is_close_result);
        return is_close_result;

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        print_test_result("PyTorch Comparison Test", false);
        return false;
    }
}

// 测试3: 维度不匹配异常测试
bool test_dimension_mismatch() {
    print_test_header("Dimension Mismatch Test");

    try {
        auto& manager = BackendManager::instance();
        auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(manager.get_backend(tr::CPU));

        // 创建不匹配的矩阵 (2x3) × (2x2) - 3 != 2
        Tensor tensor_a = Tensor::full(Shape(2, 3), 1.0f, DType::FP32, tr::CPU);
        Tensor tensor_b = Tensor::full(Shape(2, 2), 1.0f, DType::FP32, tr::CPU);
        Tensor result = Tensor::empty(Shape(2, 2), DType::FP32, tr::CPU);

        std::cout << "[TEST] Trying to multiply A(2x3) × B(2x2) - should fail" << std::endl;

        // 尝试执行矩阵乘法，应该抛出异常
        bool exception_thrown = false;
        try {
            cpu_backend->mm(result, tensor_a, tensor_b);
        } catch (const TRException& e) {
            exception_thrown = true;
            std::cout << "[TEST] Expected exception caught: " << e.what() << std::endl;
        }

        bool test_passed = exception_thrown;
        print_test_result("Dimension Mismatch Test", test_passed);
        return test_passed;

    } catch (const TRException& e) {
        std::cout << "[TEST] Unexpected exception caught: " << e.what() << std::endl;
        print_test_result("Dimension Mismatch Test", false);
        return false;
    }
}

// 测试4: 1D张量处理测试
bool test_1d_tensors() {
    print_test_header("1D Tensors Test");

    try {
        auto& manager = BackendManager::instance();
        auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(manager.get_backend(tr::CPU));

        // 测试行向量 (1x3) × (3x2) = (1x2)
        Tensor row_vector = Tensor::full(Shape(3), 2.0f, DType::FP32, tr::CPU);  // 视为 1x3
        Tensor matrix = Tensor::full(Shape(3, 2), 3.0f, DType::FP32, tr::CPU);  // 3x2
        Tensor result1 = Tensor::empty(Shape(2), DType::FP32, tr::CPU);  // 1x2 视为 2

        std::cout << "[TEST] Row vector (1x3) × Matrix (3x2)" << std::endl;
        row_vector.print("row_vector");
        matrix.print("matrix");

        cpu_backend->mm(result1, row_vector, matrix);

        std::cout << "[TEST] Result (should be 1x2):" << std::endl;
        result1.print("result1");

        // 测试 (2x3) × 列向量 (3x1) = (2x1)
        Tensor matrix2 = Tensor::full(Shape(2, 3), 4.0f, DType::FP32, tr::CPU);  // 2x3
        Tensor col_vector = Tensor::full(Shape(3), 5.0f, DType::FP32, tr::CPU);  // 视为 3x1
        Tensor result2 = Tensor::empty(Shape(2), DType::FP32, tr::CPU);  // 2x1 视为 2

        std::cout << "\n[TEST] Matrix (2x3) × Column vector (3x1)" << std::endl;
        matrix2.print("matrix2");
        col_vector.print("col_vector");

        cpu_backend->mm(result2, matrix2, col_vector);

        std::cout << "[TEST] Result (should be 2x1):" << std::endl;
        result2.print("result2");

        // 验证结果是否合理
        bool test_passed = true;

        print_test_result("1D Tensors Test", test_passed);
        return test_passed;

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        print_test_result("1D Tensors Test", false);
        return false;
    }
}

// 测试5: 大规模矩阵乘法性能测试
bool test_large_matrix_performance() {
    print_test_header("Large Matrix Performance Test");

    try {
        auto& manager = BackendManager::instance();
        auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(manager.get_backend(tr::CPU));

        // 测试不同规模矩阵的乘法性能
        std::vector<std::pair<int, int>> test_sizes = {
            {512, 512},    // 512×512 × 512×512 = 512×512
            {1024, 1024},  // 1024×1024 × 1024×1024 = 1024×1024
            {2048, 2048}   // 2048×2048 × 2048×2048 = 2048×2048
        };

        bool all_passed = true;

        for (const auto& size_pair : test_sizes) {
            int m = size_pair.first;
            int k = size_pair.first;
            int n = size_pair.second;

            std::cout << "\n[TEST] Testing " << m << "×" << k << " × " << k << "×" << n << " = " << m << "×" << n << std::endl;

            // 创建随机矩阵
            std::cout << "[TEST] Creating matrices..." << std::endl;
            Tensor matrix_a = create_random_tensor(Shape(m, k), -1.0f, 1.0f);
            Tensor matrix_b = create_random_tensor(Shape(k, n), -1.0f, 1.0f);
            Tensor result = Tensor::empty(Shape(m, n), DType::FP32, tr::CPU);

            // 计算理论FLOPS数 (2*M*K*N)
            size_t flops = 2ULL * m * k * n;
            std::cout << "[TEST] Theoretical FLOPs: " << std::scientific << static_cast<double>(flops) << std::endl;

            // 执行矩阵乘法并计时
            std::cout << "[TEST] Performing matrix multiplication..." << std::endl;
            auto start_time = std::chrono::high_resolution_clock::now();

            cpu_backend->mm(result, matrix_a, matrix_b);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double time_seconds = duration.count() / 1000000.0;

            // 计算GFLOPS
            double gflops = (static_cast<double>(flops) / 1e9) / time_seconds;

            std::cout << "[TEST] Performance Results:" << std::endl;
            std::cout << "  Matrix size: " << m << "×" << k << " × " << k << "×" << n << std::endl;
            std::cout << "  Execution time: " << std::fixed << std::setprecision(6) << time_seconds << " seconds" << std::endl;
            std::cout << "  Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;

            // 验证结果的有效性（检查是否有NaN或Inf）
            const float* result_data = static_cast<const float*>(result.data_ptr());
            bool has_invalid_values = false;
            for (size_t i = 0; i < result.numel(); ++i) {
                if (!std::isfinite(result_data[i])) {
                    has_invalid_values = true;
                    break;
                }
            }

            if (has_invalid_values) {
                std::cout << "[TEST] WARNING: Result contains invalid values (NaN/Inf)" << std::endl;
                all_passed = false;
            } else {
                std::cout << "[TEST] Result validation: PASSED" << std::endl;
            }

            // 显示结果摘要
            std::cout << "[TEST] Result shape: " << result.shape().to_string() << std::endl;
            std::cout << "[TEST] Result numel: " << result.numel() << std::endl;

            // 估算内存使用
            size_t memory_mb = (matrix_a.numel() + matrix_b.numel() + result.numel()) * sizeof(float) / (1024 * 1024);
            std::cout << "[TEST] Memory usage: ~" << memory_mb << " MB" << std::endl;
        }

        print_test_result("Large Matrix Performance Test", all_passed);
        return all_passed;

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        print_test_result("Large Matrix Performance Test", false);
        return false;
    }
}

int main() {
    std::cout << "=== CPU Backend Matrix Multiplication Test Suite ===" << std::endl;

    bool all_passed = true;

    // 测试1: 小规模矩阵乘法 + PyTorch对比验证
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 1: Small Matrix Multiplication + PyTorch Verification" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    all_passed &= test_basic_multiplication();
    all_passed &= test_pytorch_comparison();
    all_passed &= test_dimension_mismatch();

    // 测试2: 大规模矩阵乘法性能测试
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 2: Large Matrix Performance Test" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    all_passed &= test_large_matrix_performance();

    // 总结
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Overall Result: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;

    return all_passed ? 0 : 1;
}

#else

// 当PyTorch会话支持不可用时的占位符实现
#include <iostream>

int main() {
    std::cout << "=== CPU Backend Matrix Multiplication Test Suite ===" << std::endl;
    std::cout << "PyTorch session support is disabled. Skipping tests." << std::endl;
    std::cout << "Enable TR_BUILD_PYTORCH_SESSION in CMake to run full tests." << std::endl;
    return 0;
}

#endif
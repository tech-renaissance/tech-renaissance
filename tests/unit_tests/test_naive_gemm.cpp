/**
 * @file test_naive_gemm.cpp
 * @brief CPU后端朴素矩阵乘法测试实现
 * @details 基于C++标准库的朴素实现，不依赖Eigen库，用于对比性能
 * @version 1.00.00
 * @date 2025-10-25
 * @author 技术觉醒团队
 * @note 依赖项: C++标准库
 * @note 所属系列: tests
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstring>
#include <algorithm>

using namespace std;

/**
 * @brief 计算内存占用(MB)
 */
double calcMemMB(size_t bytes) {
    return bytes / (1024.0 * 1024.0);
}

/**
 * @brief CPU朴素矩阵乘法实现类（基于C++标准库）
 */
class NaiveGEMM {
private:
    // 矩阵维度
    const int M_;  // 矩阵A的行数
    const int K_;  // 矩阵A的列数 / 矩阵B的行数
    const int N_;  // 矩阵B的列数

    // C++标准库vector（使用列主布局模拟）
    vector<float> A_;
    vector<float> B_;
    vector<float> C_;

    // 内存大小（字节）
    size_t size_A_;
    size_t size_B_;
    size_t size_C_;

public:
    /**
     * @brief 构造函数
     */
    NaiveGEMM(int M, int K, int N)
        : M_(M), K_(K), N_(N), A_(M * K), B_(K * N), C_(M * N) {

        // 计算内存大小
        size_A_ = static_cast<size_t>(M) * K * sizeof(float);
        size_B_ = static_cast<size_t>(K) * N * sizeof(float);
        size_C_ = static_cast<size_t>(M) * N * sizeof(float);

        std::cout << "CPU Naive GEMM: Standard C++ implementation initialized" << std::endl;
        std::cout << "Matrix A: " << M << "x" << K << " (" << calcMemMB(size_A_) << " MB)" << std::endl;
        std::cout << "Matrix B: " << K << "x" << N << " (" << calcMemMB(size_B_) << " MB)" << std::endl;
        std::cout << "Matrix C: " << M << "x" << N << " (" << calcMemMB(size_C_) << " MB)" << std::endl;
        std::cout << "Total memory: " << getMemoryUsageMB() << " MB" << std::endl;
    }

    /**
     * @brief 初始化矩阵数据
     */
    void initializeData(float val_A, float val_B) {
        // 使用标准库的填充方法
        fill(A_.begin(), A_.end(), val_A);
        fill(B_.begin(), B_.end(), val_B);
        fill(C_.begin(), C_.end(), 0.0f);

        std::cout << "Data initialized: A=" << val_A << ", B=" << val_B << std::endl;
    }

    /**
     * @brief 执行朴素矩阵乘法（使用标准库实现）
     */
    void multiply() {
        // 朴素矩阵乘法：C = A * B
        // 使用列主布局计算以提高缓存效率
        for (int i = 0; i < M_; ++i) {
            for (int j = 0; j < N_; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K_; ++k) {
                    sum += A_[i * K_ + k] * B_[k * N_ + j];
                }
                C_[i * N_ + j] = sum;
            }
        }
    }

    /**
     * @brief 执行性能测试
     */
    double runPerformanceTest(int iterations = 1) {
        std::cout << "Running performance test with " << iterations << " iterations..." << std::endl;

        // 热身运行
        std::cout << "Warming up..." << std::endl;
        for (int i = 0; i < 3; i++) {
            multiply();
        }

        // 计时测试
        auto start = chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            multiply();
        }

        auto end = chrono::high_resolution_clock::now();
        double elapsed_time_ms = chrono::duration<double, std::milli>(end - start).count();

        double avg_time_ms = elapsed_time_ms / iterations;
        std::cout << "Average multiplication time: " << fixed << setprecision(3) << avg_time_ms << " ms" << std::endl;

        // 计算GFLOPS
        double flops = 2.0 * M_ * N_ * K_;  // GEMM浮点运算次数
        double gflops = flops / (avg_time_ms * 1e6);  // GFLOPS
        std::cout << "Performance: " << fixed << setprecision(1) << gflops << " GFLOPS" << std::endl;

        return avg_time_ms;
    }

    /**
     * @brief 验证结果并获取第一个元素
     */
    bool validateResult(float expected_value, float tolerance = 1e-4f) {
        // 检查第一个元素
        float first_element = C_[0];
        float rel_error = std::abs(first_element - expected_value) / expected_value;

        std::cout << "First element validation:" << std::endl;
        std::cout << "Expected: " << fixed << setprecision(6) << expected_value << std::endl;
        std::cout << "Actual: " << fixed << setprecision(6) << first_element << std::endl;
        std::cout << "Relative error: " << scientific << rel_error << std::endl;

        bool success = rel_error < tolerance;
        if (success) {
            std::cout << "Validation PASSED!" << std::endl;
        } else {
            std::cout << "Validation FAILED!" << std::endl;
        }

        return success;
    }

    /**
     * @brief 获取内存使用量（MB）
     */
    double getMemoryUsageMB() const {
        return calcMemMB(size_A_ + size_B_ + size_C_);
    }

    // Getter方法
    int getM() const { return M_; }
    int getK() const { return K_; }
    int getN() const { return N_; }

    /**
     * @brief 获取第一个元素值（用于验证）
     */
    float getFirstElement() const {
        return C_[0];
    }
};

/**
 * @brief 运行CPU朴素矩阵乘法测试
 */
bool runNaiveGEMMTest() {
    std::cout << "=== CPU Backend Naive GEMM Test ===" << std::endl;
    std::cout << "Matrix multiplication: C(512, 512) = A(512, 1024) * B(1024, 512)" << std::endl;
    std::cout << "Features: Pure C++ standard library without external dependencies" << std::endl;
    std::cout << std::endl;

    try {
        // 测试参数 (减半以提高性能)
        const int M = 512;   // 矩阵A的行数 (原1024)
        const int K = 1024;  // 矩阵A的列数 / 矩阵B的行数 (原2048)
        const int N = 512;   // 矩阵B的列数 (原1024)

        // 创建矩阵乘法对象
        NaiveGEMM gemm(M, K, N);

        std::cout << "Total memory usage: " << fixed << setprecision(3) << gemm.getMemoryUsageMB() << " MB" << std::endl;
        std::cout << std::endl;

        // 初始化数据
        const float val_A = 0.01f;
        const float val_B = 0.1f;
        gemm.initializeData(val_A, val_B);

        // 预期结果：C[i,j] = K * val_A * val_B = 2048 * 0.01 * 0.1 = 2.048
        float expected_value = static_cast<float>(K) * val_A * val_B;
        std::cout << "Expected value for each element: " << fixed << setprecision(6) << expected_value << std::endl;
        std::cout << std::endl;

        // 运行性能测试
        double avg_time_ms = gemm.runPerformanceTest();

        // 验证结果
        bool success = gemm.validateResult(expected_value);

        // 输出最终结果
        std::cout << "\n=== CPU Naive GEMM Final Results ===" << std::endl;
        std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N << " = " << M << "x" << N << std::endl;
        std::cout << "Average execution time: " << fixed << setprecision(3) << avg_time_ms << " ms" << std::endl;
        std::cout << "Memory usage: " << fixed << setprecision(3) << gemm.getMemoryUsageMB() << " MB" << std::endl;
        std::cout << "First element value: " << fixed << setprecision(6) << expected_value << std::endl;
        std::cout << "Status: " << (success ? "SUCCESS" : "FAILED") << std::endl;

        return success;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    // 设置输出精度
    std::cout << std::fixed << std::setprecision(6);
    std::cerr << std::fixed << std::setprecision(6);

    std::cout << "=== Matrix Multiplication Naive CPU Backend Test ===" << std::endl;
    std::cout << "Algorithm: Pure C++ standard library implementation" << std::endl;
    std::cout << "Implementation: Naive GEMM with direct matrix operations" << std::endl;
    std::cout << std::endl;

    bool success = runNaiveGEMMTest();

    if (success) {
        std::cout << "\n[SUCCESS] CPU Naive GEMM test completed successfully!" << std::endl;
        return 0;
    } else {
        std::cout << "\n[FAILURE] CPU Naive GEMM test failed!" << std::endl;
        return 1;
    }
}
/**
 * @file test_cpu_gemm.cpp
 * @brief CPU后端矩阵乘法测试实现
 * @details 基于Eigen库的高性能FP32矩阵乘法实现，测试1024x2048 * 2048x1024矩阵乘法
 * @version 1.00.00
 * @date 2025-10-25
 * @author 技术觉醒团队
 * @note 依赖项: Eigen3
 * @note 所属系列: tests
 */

#include "Core"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <omp.h>

int main() {
    using namespace std;
    using namespace std::chrono;
    using namespace Eigen;

    // 设置Eigen多线程配置
    Eigen::setNbThreads(omp_get_max_threads());

    cout << "=== Matrix Multiplication CPU Backend Test ===" << endl;
    cout << "Algorithm: Eigen library with automatic optimization" << endl;
    cout << "Implementation: High-performance GEMM with SIMD and multi-threading" << endl;
    cout << "Threads: " << omp_get_max_threads() << endl;
    cout << endl;
    cout << endl;

    // 配置矩阵维度
    const int M = 1024; // A的行数
    const int K = 2048; // A的列数 = B的行数
    const int N = 1024; // B的列数

    // 初始化矩阵
    cout << "Initializing matrices..." << endl;
    MatrixXf A = MatrixXf::Constant(M, K, 0.01f);
    MatrixXf B = MatrixXf::Constant(K, N, 0.1f);
    MatrixXf C = MatrixXf::Zero(M, N);

    cout << "Matrix A: " << M << "x" << K << " (8.000 MB)" << endl;
    cout << "Matrix B: " << K << "x" << N << " (8.000 MB)" << endl;
    cout << "Matrix C: " << M << "x" << N << " (4.000 MB)" << endl;
    cout << "Total memory: 20.000 MB" << endl;
    cout << endl;

    // 开始计时
    cout << "Starting matrix multiplication..." << endl;
    auto start = high_resolution_clock::now();

    // 矩阵乘法核心计算
    // Eigen 会自动调用高度优化的BLAS-like实现（SIMD + 多线程）
    C.noalias() = A * B;

    // 结束计时
    auto end = high_resolution_clock::now();
    double elapsed_sec = duration<double>(end - start).count();

    // 验证第一个元素
    // 理论计算：每个 C[i][j] = sum_{k=1}^{K} (0.01 * 0.1) = K * 0.001 = 2048 * 0.001 = 2.048
    float expected = 2048 * 0.001f;
    float actual = C(0, 0);
    float error = fabs(actual - expected);

    cout << fixed << setprecision(6);
    cout << "First element validation:" << endl;
    cout << "Expected: " << expected << endl;
    cout << "Actual: " << actual << endl;
    cout << "Absolute error: " << error << endl;

    bool success = error < 1e-4;
    if (success) {
        cout << "Validation PASSED!" << endl;
    } else {
        cout << "Validation FAILED!" << endl;
    }

    // 性能统计
    // 理论FLOPs = 2 × M × N × K
    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / elapsed_sec;

    cout << endl;
    cout << "Performance Results:" << endl;
    cout << "Execution time: " << elapsed_sec * 1000.0 << " ms" << endl;
    cout << "Performance: " << fixed << setprecision(1) << gflops << " GFLOPS" << endl;

    // 内存占用估算
    // 每个float 4字节，总内存 ≈ (M*K + K*N + M*N) * 4 Bytes
    size_t bytes = static_cast<size_t>(M) * K * sizeof(float)
                 + static_cast<size_t>(K) * N * sizeof(float)
                 + static_cast<size_t>(M) * N * sizeof(float);

    double mem_mib = bytes / (1024.0 * 1024.0);
    cout << "Memory usage: " << fixed << setprecision(3) << mem_mib << " MB" << endl;
    cout << endl;

    cout << "=== CPU GEMM Final Results ===" << endl;
    cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N << " = " << M << "x" << N << endl;
    cout << "Execution time: " << fixed << setprecision(3) << elapsed_sec * 1000.0 << " ms" << endl;
    cout << "Performance: " << fixed << setprecision(1) << gflops << " GFLOPS" << endl;
    cout << "First element value: " << fixed << setprecision(6) << expected << endl;
    cout << "Status: " << (success ? "SUCCESS" : "FAILED") << endl;

    if (success) {
        cout << endl << "[SUCCESS] CPU GEMM test completed successfully!" << endl;
        return 0;
    } else {
        cout << endl << "[FAILURE] CPU GEMM test failed!" << endl;
        return 1;
    }
}
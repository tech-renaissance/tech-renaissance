/**
 * @file test_cuda_gemm_e_unified.cpp
 * @brief Unified CUDA GEMM Test - Solution E (Simple cuBLAS)
 * @details Matrix multiplication: C(1024, 512) = A(1024, 2048) * B(2048, 512)
 *          Unified test standard with N(0,1) distribution and 3-line output
 * @version 1.00.00
 * @date 2025-10-30
 * @author 技术觉醒团队
 * @note 依赖项: CUDA, cuBLAS
 * @note 所属系列: tests
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS error code: " + std::to_string(stat)); \
    } \
} while (0)

class UnifiedSimpleCuBLAS {
private:
    cublasHandle_t handle_;
    int M_, K_, N_;
    float *d_A_, *d_B_, *d_C_;
    size_t size_A_, size_B_, size_C_;

public:
    UnifiedSimpleCuBLAS(int M, int K, int N)
        : M_(M), K_(K), N_(N), d_A_(nullptr), d_B_(nullptr), d_C_(nullptr) {
        size_A_ = static_cast<size_t>(M_) * K_ * sizeof(float);
        size_B_ = static_cast<size_t>(K_) * N_ * sizeof(float);
        size_C_ = static_cast<size_t>(M_) * N_ * sizeof(float);

        CUDA_CHECK(cudaSetDevice(0));
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));

        CUDA_CHECK(cudaMalloc(&d_A_, size_A_));
        CUDA_CHECK(cudaMalloc(&d_B_, size_B_));
        CUDA_CHECK(cudaMalloc(&d_C_, size_C_));
    }

    ~UnifiedSimpleCuBLAS() {
        if (d_A_) cudaFree(d_A_);
        if (d_B_) cudaFree(d_B_);
        if (d_C_) cudaFree(d_C_);
        if (handle_) cublasDestroy(handle_);
    }

    void uploadData(const std::vector<float>& h_A, const std::vector<float>& h_B) {
        CUDA_CHECK(cudaMemcpy(d_A_, h_A.data(), size_A_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_, h_B.data(), size_B_, cudaMemcpyHostToDevice));
    }

    void multiply() {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N_, M_, K_, &alpha, d_B_, N_, d_A_, K_,
                                 &beta, d_C_, N_));
    }

    double benchmark(int iterations = 100) {
        // 热身
        for (int i = 0; i < 10; i++) multiply();
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // 统一计时开始点
        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < iterations; i++) multiply();

        // 统一计时结束点
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return ms / iterations;
    }

    void downloadResult(std::vector<float>& h_C) {
        h_C.resize(M_ * N_);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C_, size_C_, cudaMemcpyDeviceToHost));
    }
};

void cpuMultiply(const std::vector<float>& A, const std::vector<float>& B,
                 std::vector<float>& C, int M, int K, int N) {
    C.assign(M * N, 0.0f);
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

void computeErrors(const std::vector<float>& gpu, const std::vector<float>& cpu,
                   double& mean_abs_err, double& mean_rel_err) {
    double sum_abs_err = 0, sum_cpu_abs = 0;
    size_t n = gpu.size();

    for (size_t i = 0; i < n; i++) {
        double err = std::abs(gpu[i] - cpu[i]);
        sum_abs_err += err;
        sum_cpu_abs += std::abs(cpu[i]);
    }

    mean_abs_err = sum_abs_err / n;
    mean_rel_err = mean_abs_err / (sum_cpu_abs / n);
}

int main() {
    try {
        // 统一测试参数
        const int M = 128, K = 1024, N = 256;
        const int iterations = 100;

        // 统一随机种子确保可复现性
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> h_A(M * K), h_B(K * N);
        std::generate(h_A.begin(), h_A.end(), [&](){ return dist(gen); });
        std::generate(h_B.begin(), h_B.end(), [&](){ return dist(gen); });

        // CPU参考计算
        std::vector<float> cpu_C;
        cpuMultiply(h_A, h_B, cpu_C, M, K, N);

        // GPU计算
        UnifiedSimpleCuBLAS solver(M, K, N);
        solver.uploadData(h_A, h_B);
        solver.multiply();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> gpu_C;
        solver.downloadResult(gpu_C);

        // 计算误差
        double mean_abs_err, mean_rel_err;
        computeErrors(gpu_C, cpu_C, mean_abs_err, mean_rel_err);

        // 性能测试
        double avg_time_ms = solver.benchmark(iterations);
        double flops = 2.0 * M * N * K;
        double gflops = flops / (avg_time_ms * 1e6);

        // 统一输出格式：3行
        std::cout << std::scientific << std::setprecision(6) << mean_abs_err << std::endl;
        std::cout << std::scientific << std::setprecision(6) << mean_rel_err << std::endl;
        std::cout << std::fixed << std::setprecision(2) << gflops << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
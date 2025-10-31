/**
 * @file test_solution_c.cpp
 * @brief Solution C: 3x3 Convolution Test - cuDNN Algorithm Finding
 * @details Using cudnnFindConvolutionForwardAlgorithm for optimal performance
 * @version 1.00.00
 * @date 2025-10-24
 * @author Tech Renaissance Team
 * @note Dependencies: CUDA, cuDNN
 * @note Series: tests
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cudnn.h>

/**
 * @brief 检查CUDA API调用的宏
 */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::string msg = "CUDA Error at " + std::string(__FILE__) + ":" + \
                         std::to_string(__LINE__) + ": " + \
                         cudaGetErrorString(err); \
        throw std::runtime_error(msg); \
    } \
} while (0)

/**
 * @brief 检查cuDNN API调用的宏
 */
#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::string msg = "cuDNN Error at " + std::string(__FILE__) + ":" + \
                         std::to_string(__LINE__) + ": " + \
                         cudnnGetErrorString(status); \
        throw std::runtime_error(msg); \
    } \
} while (0)

/**
 * @brief 方案C的3x3卷积实现类
 */
class SolutionCConv3x3 {
private:
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_, output_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;

    int batch_size_, in_channels_, out_channels_, height_, width_;
    int out_height_, out_width_;

    void* d_workspace_;
    size_t workspace_size_;
    cudnnConvolutionFwdAlgo_t best_algo_;

public:
    /**
     * @brief 构造函数
     */
    SolutionCConv3x3(int batch_size, int in_channels, int out_channels,
                     int height, int width)
        : batch_size_(batch_size), in_channels_(in_channels),
          out_channels_(out_channels), height_(height), width_(width),
          d_workspace_(nullptr), workspace_size_(0) {

        // 初始化CUDA
        CUDA_CHECK(cudaSetDevice(0));

        // 创建cuDNN句柄
        CUDNN_CHECK(cudnnCreate(&cudnn_handle_));

        // 创建描述符
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc_));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));

        // 设置输入描述符
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc_,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               batch_size_, in_channels_, height_, width_));

        // 设置卷积核描述符 (3x3, padding=1, stride=1)
        const int kernel_size = 3;
        const int padding = 1;
        const int stride = 1;

        CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc_,
                                             CUDNN_DATA_FLOAT,
                                             CUDNN_TENSOR_NCHW,
                                             out_channels_, in_channels_, kernel_size, kernel_size));

        // 设置卷积描述符
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_,
                                                   padding, padding,
                                                   stride, stride,
                                                   1, 1, // dilation
                                                   CUDNN_CROSS_CORRELATION,
                                                   CUDNN_DATA_FLOAT));

        // 计算输出尺寸
        out_height_ = (height + 2 * padding - kernel_size) / stride + 1;
        out_width_ = (width + 2 * padding - kernel_size) / stride + 1;

        // 设置输出描述符
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc_,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               batch_size_, out_channels_, out_height_, out_width_));

        // 查找最优算法 (方案C的核心特性)
        findBestAlgorithm();

        // 分配工作空间
        allocateWorkspace();
    }

    /**
     * @brief 析构函数
     */
    ~SolutionCConv3x3() {
        // 修复：避免析构函数中的虚函数调用
        if (d_workspace_) {
            cudaError_t err = cudaFree(d_workspace_);
            (void)err;
        }
        if (input_desc_) {
            cudnnStatus_t err = cudnnDestroyTensorDescriptor(input_desc_);
            (void)err;
        }
        if (output_desc_) {
            cudnnStatus_t err = cudnnDestroyTensorDescriptor(output_desc_);
            (void)err;
        }
        if (filter_desc_) {
            cudnnStatus_t err = cudnnDestroyFilterDescriptor(filter_desc_);
            (void)err;
        }
        if (conv_desc_) {
            cudnnStatus_t err = cudnnDestroyConvolutionDescriptor(conv_desc_);
            (void)err;
        }
        if (cudnn_handle_) {
            cudnnStatus_t err = cudnnDestroy(cudnn_handle_);
            (void)err;
        }
    }

    /**
     * @brief 查找最优卷积算法
     */
    void findBestAlgorithm() {
        cudnnConvolutionFwdAlgoPerf_t perf_result;
        int returned_algo_count = 0;

        CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
            cudnn_handle_, input_desc_, filter_desc_, conv_desc_, output_desc_,
            1, &returned_algo_count, &perf_result));

        if (returned_algo_count == 0) {
            throw std::runtime_error("Failed to find any convolution algorithm");
        }

        best_algo_ = perf_result.algo;
        std::cout << "Best algorithm: " << getAlgorithmName(best_algo_) << std::endl;
        std::cout << "Algorithm time: " << perf_result.time << " ms" << std::endl;
    }

    /**
     * @brief 分配工作空间
     */
    void allocateWorkspace() {
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn_handle_, input_desc_, filter_desc_, conv_desc_, output_desc_,
            best_algo_, &workspace_size_));

        if (workspace_size_ > 0) {
            CUDA_CHECK(cudaMalloc(&d_workspace_, workspace_size_));
        }
    }

    /**
     * @brief 执行前向卷积
     */
    void forward(const float* input, const float* filter, float* output) {
        float alpha = 1.0f;
        float beta = 0.0f;

        CUDNN_CHECK(cudnnConvolutionForward(
            cudnn_handle_, &alpha,
            input_desc_, input,
            filter_desc_, filter,
            conv_desc_, best_algo_, d_workspace_, workspace_size_,
            &beta, output_desc_, output));
    }

    /**
     * @brief 获取算法名称
     */
    const char* getAlgorithmName(cudnnConvolutionFwdAlgo_t algo) {
        switch (algo) {
            case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: return "IMPLICIT_GEMM";
            case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: return "IMPLICIT_PRECOMP_GEMM";
            case CUDNN_CONVOLUTION_FWD_ALGO_GEMM: return "GEMM";
            case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: return "DIRECT";
            case CUDNN_CONVOLUTION_FWD_ALGO_FFT: return "FFT";
            case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: return "FFT_TILING";
            case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: return "WINOGRAD";
            case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: return "WINOGRAD_NONFUSED";
            default: return "UNKNOWN";
        }
    }

    // Getter方法
    int getOutputHeight() const { return out_height_; }
    int getOutputWidth() const { return out_width_; }
    double getMemoryUsageMB() const {
        size_t input_size = batch_size_ * in_channels_ * height_ * width_ * sizeof(float);
        size_t output_size = batch_size_ * out_channels_ * out_height_ * out_width_ * sizeof(float);
        size_t filter_size = out_channels_ * in_channels_ * 3 * 3 * sizeof(float);
        return (input_size + output_size + filter_size + workspace_size_) / (1024.0 * 1024.0);
    }
};

/**
 * @brief 运行方案C的性能测试
 */
bool runSolutionCTest() {
    std::cout << "=== Solution C: cuDNN Algorithm Finding ===" << std::endl;
    std::cout << "Using cudnnFindConvolutionForwardAlgorithm for optimal performance" << std::endl;
    std::cout << std::endl;

    // 测试参数
    const int N = 32, C = 512, H = 7, W = 7;
    const int K = 512;

    std::cout << "Input: [" << N << "," << C << "," << H << "," << W << "] (all 0.01)" << std::endl;
    std::cout << "Filter: [" << K << "," << C << "," << 3 << "," << 3 << "] (all 0.1)" << std::endl;
    std::cout << "Expected output: ~2.048" << std::endl;
    std::cout << std::endl;

    try {
        // 创建卷积对象
        SolutionCConv3x3 conv(N, C, K, H, W);

        std::cout << "Created convolution object successfully" << std::endl;
        std::cout << "Output shape: [" << N << "," << K << "," << conv.getOutputHeight()
                  << "," << conv.getOutputWidth() << "]" << std::endl;
        std::cout << "Memory usage: " << conv.getMemoryUsageMB() << " MB" << std::endl;
        std::cout << std::endl;

        // 分配内存
        std::vector<float> h_input(N * C * H * W, 0.01f);
        std::vector<float> h_filter(K * C * 3 * 3, 0.1f);
        std::vector<float> h_output(N * K * conv.getOutputHeight() * conv.getOutputWidth(), 0.0f);

        // 分配GPU内存
        float *d_input, *d_filter, *d_output;
        size_t input_size = h_input.size() * sizeof(float);
        size_t filter_size = h_filter.size() * sizeof(float);
        size_t output_size = h_output.size() * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_input, input_size));
        CUDA_CHECK(cudaMalloc(&d_filter, filter_size));
        CUDA_CHECK(cudaMalloc(&d_output, output_size));

        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_filter, h_filter.data(), filter_size, cudaMemcpyHostToDevice));

        // 热身运行
        std::cout << "Warming up..." << std::endl;
        for (int i = 0; i < 3; i++) {
            conv.forward(d_input, d_filter, d_output);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 性能测试
        const int iterations = 20;
        std::cout << "Running " << iterations << " iterations for performance test..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            conv.forward(d_input, d_filter, d_output);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(end - start).count();

        // 获取结果
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size, cudaMemcpyDeviceToHost));

        // 输出性能结果
        std::cout << "\n=== Solution C Performance Results ===" << std::endl;
        std::cout << "Total convolution time: " << total_time << " ms" << std::endl;
        std::cout << "Average time per iteration: " << total_time / iterations << " ms" << std::endl;
        std::cout << "Memory usage: " << conv.getMemoryUsageMB() << " MB" << std::endl;

        // 计算吞吐量 (GFLOPS)
        double gflops = (2.0 * N * K * C * H * W * 3 * 3) / (total_time / iterations * 1e6);
        std::cout << "Throughput: " << gflops << " GFLOPS" << std::endl;

        // 输出第一个通道的结果
        std::cout << "\n=== First Channel (7x7) Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        for (int h = 0; h < conv.getOutputHeight(); h++) {
            for (int w = 0; w < conv.getOutputWidth(); w++) {
                int idx = 0 * K * conv.getOutputHeight() * conv.getOutputWidth() +
                         0 * conv.getOutputHeight() * conv.getOutputWidth() + h * conv.getOutputWidth() + w;
                std::cout << h_output[idx] << " ";
            }
            std::cout << std::endl;
        }

        // 验证结果 - 修正：检查角落、边缘、中心三个位置
        float expected_corner = 0.01f * 0.1f * 2 * 2 * C; // 4个点: 2.048
        float expected_edge = 0.01f * 0.1f * 2 * 3 * C;   // 6个点: 3.072
        float expected_center = 0.01f * 0.1f * 3 * 3 * C; // 9个点: 4.608

        float actual_corner = h_output[0];           // [0,0] 位置
        float actual_edge = h_output[1];             // [0,1] 位置
        float actual_center = h_output[conv.getOutputWidth() + 1]; // [1,1] 位置

        std::cout << "\n=== Validation ===" << std::endl;
        std::cout << "Corner position - Expected: " << expected_corner << ", Actual: " << actual_corner << std::endl;
        std::cout << "Edge position - Expected: " << expected_edge << ", Actual: " << actual_edge << std::endl;
        std::cout << "Center position - Expected: " << expected_center << ", Actual: " << actual_center << std::endl;

        // 计算相对误差
        float rel_error_corner = std::abs(actual_corner - expected_corner) / expected_corner;
        float rel_error_edge = std::abs(actual_edge - expected_edge) / expected_edge;
        float rel_error_center = std::abs(actual_center - expected_center) / expected_center;

        std::cout << "Relative errors - Corner: " << rel_error_corner
                  << ", Edge: " << rel_error_edge
                  << ", Center: " << rel_error_center << std::endl;

        // 所有三个位置的误差都要小于阈值
        bool success = (rel_error_corner < 1e-3 && rel_error_edge < 1e-3 && rel_error_center < 1e-3);
        if (success) {
            std::cout << "Validation PASSED! All corner, edge, and center values are correct." << std::endl;
        } else {
            std::cout << "Validation FAILED!" << std::endl;
        }

        // 清理资源
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_filter));
        CUDA_CHECK(cudaFree(d_output));

        return success;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== 3x3 Convolution Solution C Test ===" << std::endl;
    std::cout << "Algorithm: cuDNN Automatic Algorithm Finding" << std::endl;
    std::cout << "Features: Automatic selection of optimal convolution algorithm" << std::endl;
    std::cout << std::endl;

    bool success = runSolutionCTest();

    if (success) {
        std::cout << "\n[SUCCESS] Solution C test completed successfully!" << std::endl;
        return 0;
    } else {
        std::cout << "\n[FAILURE] Solution C test failed!" << std::endl;
        return 1;
    }
}
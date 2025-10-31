/**
 * @file test_cuda_gemm_framework.cpp
 * @brief 使用技术觉醒框架的CUDA后端进行高性能矩阵乘法测试
 * @details 验证CudaBackend::mm的正确性和性能，并与参考实现对齐
 * @version 1.00.00
 * @date 2025-10-29
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h（无CUDA头文件污染）
 * @note 所属系列: tests
 */

#include "tech_renaissance.h"


int main() {
    std::cout << std::fixed << std::setprecision(4);

    const int M = 4096;
    const int K = 8192;
    const int N = 4096;

    const float val_a = 0.01f;
    const float val_b = 0.1f;
    const float expected_value = static_cast<float>(K) * val_a * val_b;
    const int iterations = 20;

    std::cout << "=== Tech Renaissance Framework CUDA GEMM Test ===" << std::endl;
    std::cout << "Matrix Multiplication: C(" << M << "," << N << ") = A(" << M << "," << K << ") * B(" << K << "," << N << ")" << std::endl;
    std::cout << "Using CudaBackend::mm with 1x1 convolution optimization." << std::endl;
    std::cout << "Expected value for each element: " << expected_value << std::endl;
    std::cout << std::endl;

    try {
        // 1. 初始化后端
        auto& manager = tr::BackendManager::instance();
        // 获取CUDA后端（使用基类指针）
        auto backend = manager.get_backend(tr::CUDA[0]);
        auto cuda_backend = std::dynamic_pointer_cast<tr::CudaBackend>(backend);

        if (!backend) {
            throw std::runtime_error("Failed to get CUDA backend instance.");
        }

        // 2. 定义Tensor形状
        // A(M,K) -> Tensor(M, K, 1, 1)
        // B(K,N) -> Tensor(N, K, 1, 1)
        // C(M,N) -> Tensor(M, N, 1, 1)
        tr::Shape shape_a(M, K, 1, 1);
        tr::Shape shape_b(N, K, 1, 1);
        tr::Shape shape_c(M, N, 1, 1);

        // 3. 创建并初始化Tensor
        std::cout << "Creating and initializing tensors on CUDA:0..." << std::endl;
        tr::Tensor a = tr::Tensor::full(shape_a, val_a, tr::DType::FP32, tr::CUDA[0]);
        tr::Tensor b = tr::Tensor::full(shape_b, val_b, tr::DType::FP32, tr::CUDA[0]);
        tr::Tensor c = tr::Tensor::empty(shape_c, tr::DType::FP32, tr::CUDA[0]);
        std::cout << "Tensor initialization complete." << std::endl;

        // 4. 热身运行
        std::cout << "Warming up for 3 iterations..." << std::endl;
        for (int i = 0; i < 3; ++i) {
            backend->mm(c, a, b);
        }
        // 通过框架接口同步
        if (cuda_backend) {
            cuda_backend->synchronize();
        }
        std::cout << "Warm-up complete." << std::endl << std::endl;

        // 5. 性能测试 - 使用标准C++计时
        std::cout << "Running performance test with " << iterations << " iterations..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            backend->mm(c, a, b);
        }

        // 确保所有CUDA计算完成后再停止计时
        if (cuda_backend) {
            cuda_backend->synchronize();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double avg_time_ms = duration.count() / 1000.0 / iterations;

        // 计算GFLOPS
        double flops = 2.0 * M * N * K;
        double gflops = flops / (avg_time_ms * 1e6);

        std::cout << "Performance test complete." << std::endl;
        std::cout << "Average execution time: " << avg_time_ms << " ms" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl << std::endl;

        // 6. 验证结果
        std::cout << "Validating result..." << std::endl;
        std::vector<float> h_c(c.numel());
        c.to_cpu_data(h_c.data(), c.memory_size());

        float first_element = h_c[0];
        float tolerance = 1e-4f;
        float rel_error = std::abs((first_element - expected_value) / expected_value);

        std::cout << "First element validation:" << std::endl;
        std::cout << "  - Actual:   " << first_element << std::endl;
        std::cout << "  - Expected: " << expected_value << std::endl;
        std::cout << "  - Relative Error: " << rel_error << std::endl;

        if (rel_error < tolerance) {
            std::cout << "Validation PASSED!" << std::endl;
        } else {
            std::cout << "Validation FAILED!" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n[SUCCESS] Framework CUDA GEMM test completed successfully!" << std::endl;

    // ========== 测试2: 与PyTorch对比测试 ==========
    #ifdef TR_BUILD_PYTORCH_SESSION
    try {
        std::cout << "\n=== Test 2: PyTorch Comparison Test (4x8 x 8x5 = 4x5) ===" << std::endl;

        // 获取后端引用（从前面的测试中）
        auto& manager = tr::BackendManager::instance();
        auto backend = manager.get_backend(tr::CUDA[0]);
        auto cuda_backend = std::dynamic_pointer_cast<tr::CudaBackend>(backend);

        // 定义小规模矩阵尺寸：4×8 矩阵乘以 8×5 矩阵得到 4×5 矩阵
        const int M_small = 4;
        const int K_small = 8;
        const int N_small = 5;

        // Python脚本路径
        const std::string python_script_path = std::string(PROJECT_ROOT_DIR) + "/python/tests/python_server.py";

        // 1. 初始化Python会话
        tr::PythonSession python_session(python_script_path, "cuda_gemm_pytorch_test");
        python_session.start();

        // 等待PyTorch进程启动
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // 2. 创建随机浮点数矩阵
        std::cout << "Creating random matrices for comparison..." << std::endl;

        // 使用系统时间作为随机种子
        unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
        std::srand(seed);
        std::cout << "Random seed: " << seed << std::endl;

        // 创建输入矩阵A(4x8)和B(8x5)
        tr::Shape shape_a(M_small, K_small, 1, 1);
        tr::Shape shape_b(N_small, K_small, 1, 1);
        tr::Shape shape_c(M_small, N_small, 1, 1);

        // 在CPU上创建随机矩阵，然后复制到CUDA
        tr::Tensor a_cpu = tr::Tensor::empty(shape_a, tr::DType::FP32, tr::CPU);
        tr::Tensor b_cpu = tr::Tensor::empty(shape_b, tr::DType::FP32, tr::CPU);

        // 填充随机数据 (-1.0 到 1.0)
        std::vector<float> h_a(a_cpu.numel());
        std::vector<float> h_b(b_cpu.numel());

        for (int i = 0; i < a_cpu.numel(); ++i) {
            h_a[i] = -1.0f + 2.0f * (static_cast<float>(std::rand()) / RAND_MAX);
        }
        for (int i = 0; i < b_cpu.numel(); ++i) {
            h_b[i] = -1.0f + 2.0f * (static_cast<float>(std::rand()) / RAND_MAX);
        }

        a_cpu.from_cpu_data(h_a.data(), a_cpu.memory_size());
        b_cpu.from_cpu_data(h_b.data(), b_cpu.memory_size());

        // 复制到CUDA设备
        tr::Tensor a_cuda = tr::Tensor::empty(shape_a, tr::DType::FP32, tr::CUDA[0]);
        tr::Tensor b_cuda = tr::Tensor::empty(shape_b, tr::DType::FP32, tr::CUDA[0]);
        tr::Tensor c_cuda = tr::Tensor::empty(shape_c, tr::DType::FP32, tr::CUDA[0]);

        a_cuda.from_cpu_data(h_a.data(), a_cuda.memory_size());
        b_cuda.from_cpu_data(h_b.data(), b_cuda.memory_size());

        std::cout << "Matrix A (4x8) created on CUDA" << std::endl;
        std::cout << "Matrix B (8x5) created on CUDA" << std::endl;

        // 3. 使用CUDA后端计算矩阵乘法
        std::cout << "Computing matrix multiplication with CUDA backend..." << std::endl;
        backend->mm(c_cuda, a_cuda, b_cuda);
        if (cuda_backend) {
            cuda_backend->synchronize();
        }

        // 复制CUDA结果到CPU
        tr::Tensor c_cuda_cpu = tr::Tensor::empty(shape_c, tr::DType::FP32, tr::CPU);
        c_cuda.to_cpu_data(c_cuda_cpu.data_ptr(), c_cuda.memory_size());

        // 4. 发送矩阵到PyTorch进行计算
        std::cout << "Sending matrices to PyTorch for verification..." << std::endl;

        // 转换为PyTorch期望的格式：A(4,8), B(8,5)
        tr::Tensor a_pytorch = tr::Tensor::empty(tr::Shape(M_small, K_small), tr::DType::FP32, tr::CPU);
        tr::Tensor b_pytorch = tr::Tensor::empty(tr::Shape(K_small, N_small), tr::DType::FP32, tr::CPU);

        // 将数据转换为PyTorch格式（注意B需要转置）
        float* a_pytorch_data = static_cast<float*>(a_pytorch.data_ptr());
        float* b_pytorch_data = static_cast<float*>(b_pytorch.data_ptr());

        for (int i = 0; i < M_small; ++i) {
            for (int j = 0; j < K_small; ++j) {
                a_pytorch_data[i * K_small + j] = h_a[i * K_small + j];
            }
        }
        for (int i = 0; i < K_small; ++i) {
            for (int j = 0; j < N_small; ++j) {
                // B在CUDA实现中是(N,K)格式，PyTorch需要(K,N)格式
                b_pytorch_data[i * N_small + j] = h_b[j * K_small + i];
            }
        }

        python_session.send_tensor(a_pytorch, "a");
        python_session.send_tensor(b_pytorch, "b");

        // 获取PyTorch计算结果
        tr::Tensor c_pytorch = python_session.fetch_tensor(R"({"cmd": "matmul", "params": "a,b"})", 10000);

        if (c_pytorch.numel() == 0) {
            std::cout << "Failed to get PyTorch result!" << std::endl;
        } else {
            std::cout << "Successfully got result from PyTorch" << std::endl;

            // 5. 比较结果
            std::cout << "\n=== Result Comparison ===" << std::endl;

            // 打印CUDA结果
            std::cout << "CUDA Result (4x5):" << std::endl;
            const float* cuda_data = static_cast<const float*>(c_cuda_cpu.data_ptr());
            for (int i = 0; i < M_small; ++i) {
                std::cout << "  [";
                for (int j = 0; j < N_small; ++j) {
                    std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                             << cuda_data[i * N_small + j];
                    if (j < N_small - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }

            // 打印PyTorch结果
            std::cout << "\nPyTorch Result (4x5):" << std::endl;
            const float* pytorch_data = static_cast<const float*>(c_pytorch.data_ptr());
            for (int i = 0; i < M_small; ++i) {
                std::cout << "  [";
                for (int j = 0; j < N_small; ++j) {
                    std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                             << pytorch_data[i * N_small + j];
                    if (j < N_small - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }

            // 6. 使用CPU后端的is_close方法比较结果
            std::cout << "\n=== Accuracy Validation ===" << std::endl;

            // 获取CPU后端进行is_close比较
            auto cpu_backend = manager.get_backend(tr::CPU);
            if (cpu_backend) {
                auto cuda_cpu_backend = std::dynamic_pointer_cast<tr::CpuBackend>(cpu_backend);
                if (cuda_cpu_backend) {
                    // 创建CPU版本的CUDA结果用于比较
                    tr::Tensor c_cpu_for_comparison = tr::Tensor::empty(tr::Shape(M_small, N_small), tr::DType::FP32, tr::CPU);

                    // 转换数据格式
                    float* c_cpu_comp_data = static_cast<float*>(c_cpu_for_comparison.data_ptr());
                    for (int i = 0; i < M_small; ++i) {
                        for (int j = 0; j < N_small; ++j) {
                            c_cpu_comp_data[i * N_small + j] = cuda_data[i * N_small + j];
                        }
                    }

                    // 使用is_close方法比较
                    bool is_close_result = cuda_cpu_backend->is_close(c_cpu_for_comparison, c_pytorch, 1e-4f);

                    std::cout << "Using CPU backend is_close() method:" << std::endl;
                    std::cout << "  CUDA result vs PyTorch result: "
                             << (is_close_result ? "CLOSE" : "NOT CLOSE") << std::endl;

                    // 计算最大绝对误差
                    float max_error = 0.0f;
                    for (int i = 0; i < M_small * N_small; ++i) {
                        float error = std::abs(c_cpu_comp_data[i] - pytorch_data[i]);
                        max_error = std::max(max_error, error);
                    }

                    std::cout << "Maximum absolute error: " << std::scientific << std::setprecision(2) << max_error << std::endl;

                    if (is_close_result) {
                        std::cout << "[SUCCESS] PyTorch comparison test PASSED!" << std::endl;
                    } else {
                        std::cout << "[WARNING] PyTorch comparison test shows significant differences!" << std::endl;
                    }
                } else {
                    std::cout << "Failed to get CPU backend for is_close comparison" << std::endl;
                }
            } else {
                std::cout << "CPU backend not available for comparison" << std::endl;
            }
        }

        // 清理Python会话
        python_session.send_request(R"({"cmd": "exit"})");

    } catch (const std::exception& e) {
        std::cout << "[ERROR] PyTorch comparison test failed: " << e.what() << std::endl;
    }
    #else
    std::cout << "\n[INFO] PyTorch comparison test skipped - PyTorch support not enabled" << std::endl;
    std::cout << "Enable with TR_BUILD_PYTORCH_SESSION=ON" << std::endl;
    #endif

    return 0;
}
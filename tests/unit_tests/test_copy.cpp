/**
 * @file test_copy.cpp
 * @brief 张量复制功能测试
 * @details 测试CPU和CUDA后端的copy()和copy_into()方法
 * @version 1.00.00
 * @date 2025-10-31
 * @author 技术觉醒团队
 * @note 依赖项: CpuBackend, CudaBackend
 * @note 所属系列: backend tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <iomanip>

using namespace tr;

/**
 * @brief 测试CPU后端的copy()方法
 */
void test_cpu_copy() {
    std::cout << "=== Testing CPU copy() ===" << std::endl;

    try {
        auto cpu_backend = BackendManager::get_cpu_backend();

        // 创建测试张量
        Shape shape(2, 3, 4, 5);
        Tensor original = Tensor::empty(shape, DType::FP32, tr::CPU);
        cpu_backend->fill(original, 3.14f);

        // 测试copy()方法
        Tensor copied = cpu_backend->copy(original);

        // 验证形状和类型
        assert(copied.shape() == original.shape());
        assert(copied.dtype() == original.dtype());
        assert(copied.device().is_cpu());

        // 验证数据内容
        assert(cpu_backend->is_close(copied, original));

        // 验证数据独立性（修改原张量不影响复制的张量）
        cpu_backend->fill(const_cast<Tensor&>(original), 2.71f);
        assert(!cpu_backend->is_close(copied, original));

        std::cout << "  CPU copy(): PASSED" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "  CPU copy(): FAILED - " << e.what() << std::endl;
        assert(false);
    }
}

/**
 * @brief 测试CPU后端的copy_into()方法
 */
void test_cpu_copy_into() {
    std::cout << "=== Testing CPU copy_into() ===" << std::endl;

    try {
        auto cpu_backend = BackendManager::get_cpu_backend();

        // 创建测试张量
        Shape shape(2, 3, 4, 5);
        Tensor src = Tensor::empty(shape, DType::FP32, tr::CPU);
        Tensor dst = Tensor::empty(shape, DType::FP32, tr::CPU);

        cpu_backend->fill(src, 1.23f);
        cpu_backend->fill(dst, 9.87f);

        // 测试copy_into()方法
        cpu_backend->copy_into(src, dst);

        // 验证数据内容
        assert(cpu_backend->is_close(dst, src));

        std::cout << "  CPU copy_into(): PASSED" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "  CPU copy_into(): FAILED - " << e.what() << std::endl;
        assert(false);
    }
}

/**
 * @brief 测试CUDA后端的copy()方法
 */
void test_cuda_copy() {
    std::cout << "=== Testing CUDA copy() ===" << std::endl;

    try {
        auto cuda_backend = BackendManager::get_cuda_backend(0);

        // 创建测试张量
        Shape shape(64, 64);
        Tensor original = Tensor::empty(shape, DType::FP32, tr::CUDA[0]);
        cuda_backend->fill(original, 2.56f);

        // 测试copy()方法
        Tensor copied = cuda_backend->copy(original);

        // 验证形状和类型
        assert(copied.shape() == original.shape());
        assert(copied.dtype() == original.dtype());
        assert(copied.device().is_cuda());

        // 验证数据内容
        assert(cuda_backend->is_close(copied, original));

        // 验证数据独立性
        cuda_backend->fill(const_cast<Tensor&>(original), 3.78f);
        assert(!cuda_backend->is_close(copied, original));

        std::cout << "  CUDA copy(): PASSED" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "  CUDA copy(): FAILED - " << e.what() << std::endl;
        assert(false);
    }
}

/**
 * @brief 测试CUDA后端的copy_into()方法（同设备内）
 */
void test_cuda_copy_into_same_device() {
    std::cout << "=== Testing CUDA copy_into() (same device) ===" << std::endl;

    try {
        auto cuda_backend = BackendManager::get_cuda_backend(0);

        // 创建测试张量
        Shape shape(64, 64);
        Tensor src = Tensor::empty(shape, DType::FP32, tr::CUDA[0]);
        Tensor dst = Tensor::empty(shape, DType::FP32, tr::CUDA[0]);

        cuda_backend->fill(src, 4.56f);
        cuda_backend->fill(dst, 8.76f);

        // 测试copy_into()方法
        cuda_backend->copy_into(src, dst);

        // 验证数据内容
        assert(cuda_backend->is_close(dst, src));

        std::cout << "  CUDA copy_into() (same device): PASSED" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "  CUDA copy_into() (same device): FAILED - " << e.what() << std::endl;
        assert(false);
    }
}

/**
 * @brief 测试CUDA后端的copy_into()方法（CPU到CUDA）
 */
void test_cuda_copy_into_cpu_to_cuda() {
    std::cout << "=== Testing CUDA copy_into() (CPU to CUDA) ===" << std::endl;

    try {
        auto cpu_backend = BackendManager::get_cpu_backend();
        auto cuda_backend = BackendManager::get_cuda_backend(0);

        // 创建测试张量
        Shape shape(32, 32);
        Tensor cpu_src = Tensor::empty(shape, DType::FP32, tr::CPU);
        Tensor cuda_dst = Tensor::empty(shape, DType::FP32, tr::CUDA[0]);

        cpu_backend->fill(cpu_src, 5.67f);
        cuda_backend->fill(cuda_dst, 0.0f);

        // 测试copy_into()方法（CPU到CUDA）
        cuda_backend->copy_into(cpu_src, cuda_dst);

        // 将CUDA张量复制回CPU进行验证
        Tensor cpu_copy = cuda_backend->to_cpu(cuda_dst);

        // 验证数据内容
        assert(cpu_backend->is_close(cpu_copy, cpu_src));

        std::cout << "  CUDA copy_into() (CPU to CUDA): PASSED" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "  CUDA copy_into() (CPU to CUDA): FAILED - " << e.what() << std::endl;
        assert(false);
    }
}

/**
 * @brief 测试CUDA后端的copy_into()方法（CUDA到CPU）
 */
void test_cuda_copy_into_cuda_to_cpu() {
    std::cout << "=== Testing CUDA copy_into() (CUDA to CPU) ===" << std::endl;

    try {
        auto cpu_backend = BackendManager::get_cpu_backend();
        auto cuda_backend = BackendManager::get_cuda_backend(0);

        // 创建测试张量
        Shape shape(32, 32);
        Tensor cuda_src = Tensor::empty(shape, DType::FP32, tr::CUDA[0]);
        Tensor cpu_dst = Tensor::empty(shape, DType::FP32, tr::CPU);

        cuda_backend->fill(cuda_src, 6.78f);
        cpu_backend->fill(cpu_dst, 0.0f);

        // 测试copy_into()方法（CUDA到CPU）
        cuda_backend->copy_into(cuda_src, cpu_dst);

        // 验证数据内容
        // 将CUDA张量复制到CPU进行比较
        Tensor cpu_copy = cuda_backend->to_cpu(cuda_src);
        assert(cpu_backend->is_close(cpu_dst, cpu_copy));

        std::cout << "  CUDA copy_into() (CUDA to CPU): PASSED" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "  CUDA copy_into() (CUDA to CPU): FAILED - " << e.what() << std::endl;
        assert(false);
    }
}

/**
 * @brief 性能基准测试
 */
void benchmark_copy_performance() {
    std::cout << "=== Performance Benchmark ===" << std::endl;

    const int size = 1000;
    Shape shape(size, size);
    const int iterations = 100;

    // CPU性能测试
    try {
        auto cpu_backend = BackendManager::get_cpu_backend();
        Tensor cpu_tensor = Tensor::empty(shape, DType::FP32, tr::CPU);
        cpu_backend->fill(cpu_tensor, 1.0f);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            Tensor copy_result = cpu_backend->copy(cpu_tensor);
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time_ms = duration.count() / 1000.0 / iterations;

        std::cout << "  CPU copy() average time: " << std::fixed << std::setprecision(3)
                  << avg_time_ms << " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "  CPU benchmark failed: " << e.what() << std::endl;
    }

    // CUDA性能测试
    try {
        auto cuda_backend = BackendManager::get_cuda_backend(0);
        Tensor cuda_tensor = Tensor::empty(shape, DType::FP32, tr::CUDA[0]);
        cuda_backend->fill(cuda_tensor, 1.0f);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            Tensor copy_result = cuda_backend->copy(cuda_tensor);
        }
        cuda_backend->synchronize(); // 同步设备
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time_ms = duration.count() / 1000.0 / iterations;

        std::cout << "  CUDA copy() average time: " << std::fixed << std::setprecision(3)
                  << avg_time_ms << " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "  CUDA benchmark failed: " << e.what() << std::endl;
    }
}

/**
 * @brief 主测试函数
 */
int main() {
    Logger::get_instance().set_quiet_mode(true);
    std::cout << "Starting copy() and copy_into() tests..." << std::endl;
    std::cout << "=================================================" << std::endl;

    try {
        // CPU测试
        test_cpu_copy();
        test_cpu_copy_into();

        // CUDA测试（如果可用）
#ifdef TR_USE_CUDA
        test_cuda_copy();
        test_cuda_copy_into_same_device();
        test_cuda_copy_into_cpu_to_cuda();
        test_cuda_copy_into_cuda_to_cpu();
#else
        std::cout << "CUDA tests skipped (CUDA not available)" << std::endl;
#endif

        // 性能基准测试
        benchmark_copy_performance();

        std::cout << "=================================================" << std::endl;
        std::cout << "All copy tests PASSED!" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Test suite failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
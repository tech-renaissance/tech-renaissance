/**
 * @file test_performance.cpp
 * @brief CPU后端性能测试
 * @details 测试CPU后端的各种操作性能，CUDA功能已移除以支持模块化架构
 * @version 2.0.0
 * @date 2025-11-23
 * @author 技术觉醒团队
 * @note 所属系列: tests
 */

#include "tech_renaissance.h"

using namespace tr;

int main() {
    std::cout << std::fixed << std::setprecision(4);
    Logger::get_instance().set_quiet_mode(true);

    constexpr bool using_into_form = true;
    // constexpr bool using_into_form = false;

    constexpr int iterations_cpu = 25;
    constexpr int warmup_iterations = 10;

    auto cpu_backend = BackendManager::get_cpu_backend();

    Tensor cpu_a = cpu_backend->randn(Shape(1024, 2048), time(nullptr));
    Tensor cpu_b = cpu_backend->randn(Shape(2048, 512), time(nullptr));

    Tensor input = cpu_backend->randn(Shape(32, 512, 7, 7), time(nullptr));
    Tensor kernel_1 = cpu_backend->randn(Shape(512, 512, 1, 1), time(nullptr));
    Tensor kernel_3 = cpu_backend->randn(Shape(512, 512, 3, 3), time(nullptr));

    std::cout << "=== CPU Backend Performance Tests ===" << std::endl;

    // Matrix Multiplication Performance Test
    {
        auto cpu_result = cpu_backend->mm(cpu_a, cpu_b);
        for (int i = 0; i < warmup_iterations; ++i) {
            cpu_backend->mm(cpu_a, cpu_b);
        }

        Profiler profiler;
        profiler.set_iterations(iterations_cpu);
        profiler.describe_operation("mm", cpu_a.shape(), cpu_b.shape());

        profiler.start();

        if (using_into_form) {
            for (int i = 0; i < iterations_cpu; ++i) {
                cpu_backend->mm_into(cpu_a, cpu_b, cpu_result);  // FAST
            }
        }
        else {
            for (int i = 0; i < iterations_cpu; ++i) {
                cpu_result = cpu_backend->mm(cpu_a, cpu_b);  // SLOW
            }
        }

        profiler.stop();

        std::cout << "CPU MM Performance: " << std::fixed << std::setprecision(2)
        << profiler.get_performance() << " GFLOPS" << std::endl;
    }

    // 3x3 Convolution Performance Test
    {
        auto cpu_result = cpu_backend->conv(input, kernel_3, 1, 1);
        for (int i = 0; i < warmup_iterations; ++i) {
            cpu_backend->conv(input, kernel_3, 1, 1);
        }

        Profiler profiler;
        profiler.set_iterations(iterations_cpu);
        profiler.describe_operation("conv_k3_s1_p1", input.shape(), kernel_3.shape());

        profiler.start();

        if (using_into_form) {
            for (int i = 0; i < iterations_cpu; ++i) {
                cpu_backend->conv_into(input, kernel_3, cpu_result, 1, 1);  // FAST
            }
        }
        else {
            for (int i = 0; i < iterations_cpu; ++i) {
                cpu_result = cpu_backend->conv(input, kernel_3, 1, 1);  // SLOW
            }
        }

        profiler.stop();

        std::cout << "CPU 3x3 Conv Performance: " << std::fixed << std::setprecision(2)
        << profiler.get_performance() << " GFLOPS" << std::endl;
    }

    // 1x1 Convolution Performance Test
    {
        auto cpu_result = cpu_backend->conv(input, kernel_1, 1, 0);
        for (int i = 0; i < warmup_iterations; ++i) {
            cpu_backend->conv(input, kernel_1, 1, 0);
        }

        Profiler profiler;
        profiler.set_iterations(iterations_cpu);
        profiler.describe_operation("conv_k1_s1_p0", input.shape(), kernel_1.shape());

        profiler.start();

        if (using_into_form) {
            for (int i = 0; i < iterations_cpu; ++i) {
                cpu_backend->conv_into(input, kernel_1, cpu_result, 1, 0);  // FAST
            }
        }
        else {
            for (int i = 0; i < iterations_cpu; ++i) {
                cpu_result = cpu_backend->conv(input, kernel_1, 1, 0);  // SLOW
            }
        }

        profiler.stop();

        std::cout << "CPU 1x1 Conv Performance: " << std::fixed << std::setprecision(2)
        << profiler.get_performance() << " GFLOPS" << std::endl;
    }

    // 3x3 Transposed Convolution Performance Test
    {
        auto cpu_result = cpu_backend->transposed_conv(input, kernel_3, 1, 1);
        for (int i = 0; i < warmup_iterations; ++i) {
            cpu_backend->transposed_conv(input, kernel_3, 1, 1);
        }

        Profiler profiler;
        profiler.set_iterations(iterations_cpu);
        profiler.describe_operation("conv_k3_s1_p1", input.shape(), kernel_3.shape());

        profiler.start();

        if (using_into_form) {
            for (int i = 0; i < iterations_cpu; ++i) {
                cpu_backend->transposed_conv_into(input, kernel_3, cpu_result, 1, 1);  // FAST
            }
        }
        else {
            for (int i = 0; i < iterations_cpu; ++i) {
                cpu_result = cpu_backend->transposed_conv(input, kernel_3, 1, 1);  // SLOW
            }
        }

        profiler.stop();

        std::cout << "CPU 3x3 TConv Performance: " << std::fixed << std::setprecision(2)
        << profiler.get_performance() << " GFLOPS" << std::endl;
    }

    std::cout << "\n=== CPU Performance Tests Complete ===" << std::endl;
    std::cout << "All CPU backend operations tested successfully!" << std::endl;

    return 0;
}
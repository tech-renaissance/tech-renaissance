#include "tech_renaissance.h"

using namespace tr;

int main() {
    std::cout << std::fixed << std::setprecision(4);
    Logger::get_instance().set_quiet_mode(true);

    constexpr bool using_into_form = true;
    // constexpr bool using_into_form = false;

    constexpr int iterations_cuda = 100;
    constexpr int iterations_cpu = 25;
    constexpr int warmup_iterations = 10;

    auto cpu_backend = BackendManager::get_cpu_backend();
    auto cuda_backend = BackendManager::get_cuda_backend();

    Tensor cpu_a = cpu_backend->randn(Shape(1024, 2048), time(nullptr));
    Tensor cpu_b = cpu_backend->randn(Shape(2048, 512), time(nullptr));

    Tensor cuda_a = cuda_backend->from_cpu(cpu_a);
    Tensor cuda_b = cuda_backend->from_cpu(cpu_b);

    Tensor input = cpu_backend->randn(Shape(32, 512, 7, 7), time(nullptr));
    Tensor kernel_1 = cpu_backend->randn(Shape(512, 512, 1, 1), time(nullptr));
    Tensor kernel_3 = cpu_backend->randn(Shape(512, 512, 3, 3), time(nullptr));

    Tensor input_cuda = cuda_backend->from_cpu(input);
    Tensor kernel_1_cuda = cuda_backend->from_cpu(kernel_1);
    Tensor kernel_3_cuda = cuda_backend->from_cpu(kernel_3);

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
                cpu_result = cpu_backend->conv(input, kernel_1, 1, 1);  // SLOW
            }
        }

        profiler.stop();

        std::cout << "CPU 1x1 Conv Performance: " << std::fixed << std::setprecision(2)
                  << profiler.get_performance() << " GFLOPS" << std::endl;
    }

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

    {
        auto cuda_result = cuda_backend->mm(cuda_a, cuda_b);
        for (int i = 0; i < warmup_iterations; ++i) {
            cuda_backend->mm(cuda_a, cuda_b);
        }

        cuda_backend->synchronize();

        Profiler profiler;
        profiler.set_iterations(iterations_cuda);
        profiler.describe_operation("mm", cpu_a.shape(), cpu_b.shape());

        profiler.start();

        if (using_into_form) {
            for (int i = 0; i < iterations_cuda; ++i) {
                cuda_backend->mm_into(cuda_a, cuda_b, cuda_result);  // FAST
            }
        }
        else {
            for (int i = 0; i < iterations_cuda; ++i) {
                cuda_result = cuda_backend->mm(cuda_a, cuda_b);  // SLOW
            }
        }

        cuda_backend->synchronize();
        profiler.stop();

        std::cout << "CUDA MM Performance: " << std::fixed << std::setprecision(2)
                  << profiler.get_performance() << " GFLOPS" << std::endl;
    }

    {
        auto cuda_result = cuda_backend->conv(input_cuda, kernel_3_cuda, 1, 1);
        for (int i = 0; i < warmup_iterations; ++i) {
            cuda_backend->conv(input_cuda, kernel_3_cuda, 1, 1);
        }

        cuda_backend->synchronize();

        Profiler profiler;
        profiler.set_iterations(iterations_cuda);
        profiler.describe_operation("conv_k3_s1_p1", input.shape(), kernel_3.shape());

        profiler.start();

        if (using_into_form) {
            for (int i = 0; i < iterations_cuda; ++i) {
                cuda_backend->conv_into(input_cuda, kernel_3_cuda, cuda_result, 1, 1);  // FAST
            }
        }
        else {
            for (int i = 0; i < iterations_cuda; ++i) {
                cuda_result = cuda_backend->conv(input_cuda, kernel_3_cuda, 1, 1);  // SLOW
            }
        }

        cuda_backend->synchronize();
        profiler.stop();

        std::cout << "CUDA 3x3 Conv Performance: " << std::fixed << std::setprecision(2)
                  << profiler.get_performance() << " GFLOPS" << std::endl;
    }

    {
        auto cuda_result = cuda_backend->conv(input_cuda, kernel_1_cuda, 1, 0);
        for (int i = 0; i < warmup_iterations; ++i) {
            cuda_backend->conv(input_cuda, kernel_1_cuda, 1, 0);
        }

        cuda_backend->synchronize();

        Profiler profiler;
        profiler.set_iterations(iterations_cuda);
        profiler.describe_operation("conv_k1_s1_p0", input.shape(), kernel_1.shape());

        profiler.start();

        if (using_into_form) {
            for (int i = 0; i < iterations_cuda; ++i) {
                cuda_backend->conv_into(input_cuda, kernel_1_cuda, cuda_result, 1, 0);  // FAST (现在已优化)
            }
        }
        else {
            for (int i = 0; i < iterations_cuda; ++i) {
                cuda_result = cuda_backend->conv(input_cuda, kernel_1_cuda, 1, 1);  // SLOW (现在已优化)
            }
        }

        cuda_backend->synchronize();
        profiler.stop();

        std::cout << "CUDA 1x1 Conv Performance: " << std::fixed << std::setprecision(2)
                  << profiler.get_performance() << " GFLOPS" << std::endl;
    }

    {
        auto cuda_result = cuda_backend->transposed_conv(input_cuda, kernel_3_cuda, 1, 1);
        for (int i = 0; i < warmup_iterations; ++i) {
            cuda_backend->transposed_conv(input_cuda, kernel_3_cuda, 1, 1);
        }

        cuda_backend->synchronize();

        Profiler profiler;
        profiler.set_iterations(iterations_cuda);
        profiler.describe_operation("conv_k3_s1_p1", input.shape(), kernel_3.shape());

        profiler.start();

        if (using_into_form) {
            for (int i = 0; i < iterations_cuda; ++i) {
                cuda_backend->transposed_conv_into(input_cuda, kernel_3_cuda, cuda_result, 1, 1);  // FAST
            }
        }
        else {
            for (int i = 0; i < iterations_cuda; ++i) {
                cuda_result = cuda_backend->transposed_conv(input_cuda, kernel_3_cuda, 1, 1);  // SLOW
            }
        }

        cuda_backend->synchronize();
        profiler.stop();

        std::cout << "CUDA 3x3 TConv Performance: " << std::fixed << std::setprecision(2)
                  << profiler.get_performance() << " GFLOPS" << std::endl;
    }

    return 0;
}
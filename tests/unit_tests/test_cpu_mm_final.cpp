#include "tech_renaissance.h"

using namespace tr;

int main() {
    std::cout << std::fixed << std::setprecision(4);
    Logger::get_instance().set_quiet_mode(true);

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 使用二维Shape创建随机张量
    Tensor cpu_a = cpu_backend->randn(Shape(1024, 2048), time(nullptr));  // 1024 x 2048矩阵，种子22
    Tensor cpu_b = cpu_backend->randn(Shape(2048, 512), time(nullptr));  // 2048 x 512矩阵，种子22

    Tensor cuda_result = cpu_backend->empty(Shape(1024, 512), DType::FP32);
    cpu_backend->mm_into(cpu_a, cpu_b, cuda_result);
    auto tr_result = cuda_result;

#ifdef TR_BUILD_PYTHON_SESSION
    {
        std::cout << std::endl;
        std::cout << "****************************************" << std::endl;
        std::cout << "********* ACCURACY VERIFICATION ********" << std::endl;
        std::cout << "****************************************" << std::endl;

        int test_passed = 0;
        int total_tests = 1;

        double mean_abs_err_pytorch;
        double mean_rel_err_pytorch;

        // 增加PyTorch结果验证
        PythonSession ps("default", "verify", true);
        ps.start();

        std::cout << "\nTEST 1: mm" << std::endl;
        auto pytorch_result = ps.calculate("matmul", cpu_a, cpu_b);
        mean_abs_err_pytorch = cpu_backend->get_mean_abs_err(pytorch_result, tr_result);
        mean_rel_err_pytorch = cpu_backend->get_mean_rel_err(pytorch_result, tr_result);
        std::cout << "Mean absolute error (Vs PyTorch): " << std::scientific << std::setprecision(6) << mean_abs_err_pytorch << std::endl;
        std::cout << "Mean relative error (Vs PyTorch): " << std::scientific << std::setprecision(6) << mean_rel_err_pytorch << std::endl;
        if (mean_rel_err_pytorch < 1e-6) {
            std::cout << "TEST 1: [PASSED]" << std::endl;
            test_passed++;
        }
        else {
            std::cout << "TEST 1: [FAILED]" << std::endl;
        }

        ps.please_exit();

        std::cout << "TEST PASS: " << test_passed << "/" << total_tests << std::endl;
        if (test_passed == total_tests) {
            std::cout << "ALL TEST PASSED!!!" << std::endl;
        }
    }
#endif

{
    std::cout << std::endl;
    std::cout << "****************************************" << std::endl;
    std::cout << "******* PERFORMANCE VERIFICATION *******" << std::endl;
    std::cout << "****************************************" << std::endl;

    constexpr int iterations = 100;
    constexpr int warmup_iterations = 10;

    cpu_backend->mm_into(cpu_a, cpu_b, cuda_result);

    // 预热运行
    for (int i = 0; i < warmup_iterations; ++i) {
        cuda_result = cpu_backend->mm(cpu_a, cpu_b);
    }

    // 性能测试 - 使用简化的Profiler API
    Profiler profiler;
    profiler.set_iterations(iterations);
    profiler.describe_operation("mm", cpu_a.shape(), cpu_b.shape());

    profiler.start();
    for (int i = 0; i < iterations; ++i) {
        cpu_backend->mm_into(cpu_a, cpu_b, cuda_result);
    }

    profiler.stop();

    std::cout << "Performance: " << std::fixed << std::setprecision(2)
              << profiler.get_performance() << " GFLOPS" << std::endl;
}

    return 0;
}
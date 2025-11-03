#include "tech_renaissance.h"

using namespace tr;

int main() {
    std::cout << std::fixed << std::setprecision(4);
    Logger::get_instance().set_quiet_mode(true);

    auto cpu_backend = BackendManager::get_cpu_backend();

#ifdef TR_BUILD_PYTHON_SESSION
    {
        std::cout << std::endl;
        std::cout << "****************************************" << std::endl;
        std::cout << "********* ACCURACY VERIFICATION ********" << std::endl;
        std::cout << "****************************************" << std::endl;

        int test_passed = 0;
        int total_tests = 6;

        double mean_abs_err_pytorch;
        double mean_rel_err_pytorch;
        Tensor input = cpu_backend->randn(Shape(2, 3, 9, 9), time(nullptr));
        Tensor kernel_1 = cpu_backend->randn(Shape(1, 3, 1, 1), time(nullptr));
        Tensor kernel_3 = cpu_backend->randn(Shape(1, 3, 3, 3), time(nullptr));
        Tensor kernel_7 = cpu_backend->randn(Shape(1, 3, 7, 7), time(nullptr));

        // 增加PyTorch结果验证
        PythonSession ps("default", "verify", true);
        ps.start();

        std::cout << "\nTEST 1: conv_k3_s1_p0" << std::endl;
        auto tr_result = cpu_backend->conv(input, kernel_3, 1, 0);
        auto pytorch_result = ps.calculate("conv_k3_s1_p0", input, kernel_3);
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

        std::cout << "\nTEST 2: conv_k3_s1_p1" << std::endl;
        tr_result = cpu_backend->conv(input, kernel_3, 1, 1);
        pytorch_result = ps.calculate("conv_k3_s1_p1", input, kernel_3);
        mean_abs_err_pytorch = cpu_backend->get_mean_abs_err(pytorch_result, tr_result);
        mean_rel_err_pytorch = cpu_backend->get_mean_rel_err(pytorch_result, tr_result);
        std::cout << "Mean absolute error (Vs PyTorch): " << std::scientific << std::setprecision(6) << mean_abs_err_pytorch << std::endl;
        std::cout << "Mean relative error (Vs PyTorch): " << std::scientific << std::setprecision(6) << mean_rel_err_pytorch << std::endl;
        if (mean_rel_err_pytorch < 1e-6) {
            std::cout << "TEST 2: [PASSED]" << std::endl;
            test_passed++;
        }
        else {
            std::cout << "TEST 2: [FAILED]" << std::endl;
        }

        std::cout << "\nTEST 3: conv_k3_s2_p1" << std::endl;
        tr_result = cpu_backend->conv(input, kernel_3, 2, 1);
        pytorch_result = ps.calculate("conv_k3_s2_p1", input, kernel_3);
        mean_abs_err_pytorch = cpu_backend->get_mean_abs_err(pytorch_result, tr_result);
        mean_rel_err_pytorch = cpu_backend->get_mean_rel_err(pytorch_result, tr_result);
        std::cout << "Mean absolute error (Vs PyTorch): " << std::scientific << std::setprecision(6) << mean_abs_err_pytorch << std::endl;
        std::cout << "Mean relative error (Vs PyTorch): " << std::scientific << std::setprecision(6) << mean_rel_err_pytorch << std::endl;
        if (mean_rel_err_pytorch < 1e-6) {
            std::cout << "TEST 3: [PASSED]" << std::endl;
            test_passed++;
        }
        else {
            std::cout << "TEST 3: [FAILED]" << std::endl;
        }

        std::cout << "\nTEST 4: conv_k1_s1_p0" << std::endl;
        tr_result = cpu_backend->conv(input, kernel_1, 1, 0);
        pytorch_result = ps.calculate("conv_k1_s1_p0", input, kernel_1);
        mean_abs_err_pytorch = cpu_backend->get_mean_abs_err(pytorch_result, tr_result);
        mean_rel_err_pytorch = cpu_backend->get_mean_rel_err(pytorch_result, tr_result);
        std::cout << "Mean absolute error (Vs PyTorch): " << std::scientific << std::setprecision(6) << mean_abs_err_pytorch << std::endl;
        std::cout << "Mean relative error (Vs PyTorch): " << std::scientific << std::setprecision(6) << mean_rel_err_pytorch << std::endl;
        if (mean_rel_err_pytorch < 1e-6) {
            std::cout << "TEST 4: [PASSED]" << std::endl;
            test_passed++;
        }
        else {
            std::cout << "TEST 4: [FAILED]" << std::endl;
        }

        std::cout << "\nTEST 5: conv_k1_s2_p0" << std::endl;
        tr_result = cpu_backend->conv(input, kernel_1, 2, 0);
        pytorch_result = ps.calculate("conv_k1_s2_p0", input, kernel_1);
        mean_abs_err_pytorch = cpu_backend->get_mean_abs_err(pytorch_result, tr_result);
        mean_rel_err_pytorch = cpu_backend->get_mean_rel_err(pytorch_result, tr_result);
        std::cout << "Mean absolute error (Vs PyTorch): " << std::scientific << std::setprecision(6) << mean_abs_err_pytorch << std::endl;
        std::cout << "Mean relative error (Vs PyTorch): " << std::scientific << std::setprecision(6) << mean_rel_err_pytorch << std::endl;
        if (mean_rel_err_pytorch < 1e-6) {
            std::cout << "TEST 5: [PASSED]" << std::endl;
            test_passed++;
        }
        else {
            std::cout << "TEST 5: [FAILED]" << std::endl;
        }

        std::cout << "\nTEST 6: conv_k7_s2_p3" << std::endl;
        tr_result = cpu_backend->conv(input, kernel_7, 2, 3);
        pytorch_result = ps.calculate("conv_k7_s2_p3", input, kernel_7);
        mean_abs_err_pytorch = cpu_backend->get_mean_abs_err(pytorch_result, tr_result);
        mean_rel_err_pytorch = cpu_backend->get_mean_rel_err(pytorch_result, tr_result);
        std::cout << "Mean absolute error (Vs PyTorch): " << std::scientific << std::setprecision(6) << mean_abs_err_pytorch << std::endl;
        std::cout << "Mean relative error (Vs PyTorch): " << std::scientific << std::setprecision(6) << mean_rel_err_pytorch << std::endl;
        if (mean_rel_err_pytorch < 1e-6) {
            std::cout << "TEST 6: [PASSED]" << std::endl;
            test_passed++;
        }
        else {
            std::cout << "TEST 6: [FAILED]" << std::endl;
        }

        ps.please_exit();

        std::cout << "TEST PASS: " << test_passed << "/" << total_tests << std::endl;
        if (test_passed == total_tests) {
            std::cout << "ALL TEST PASSED!!!" << std::endl;
        }
    }
#endif

{
    Tensor input = cpu_backend->randn(Shape(256, 3, 9, 9), time(nullptr));
    Tensor kernel_3 = cpu_backend->randn(Shape(128, 3, 3, 3), time(nullptr));

    std::cout << std::endl;
    std::cout << "****************************************" << std::endl;
    std::cout << "******* PERFORMANCE VERIFICATION *******" << std::endl;
    std::cout << "****************************************" << std::endl;

    constexpr int iterations = 20;
    constexpr int warmup_iterations = 5;

    // 预热运行
    for (int i = 0; i < warmup_iterations; ++i) {
        cpu_backend->conv(input, kernel_3, 1, 1);
    }

    // 性能测试 - 使用简化的Profiler API
    Profiler profiler;
    profiler.set_iterations(iterations);
    profiler.describe_operation("conv_k3_s1_p1", input.shape(), kernel_3.shape());

    profiler.start();
    for (int i = 0; i < iterations; ++i) {
        cpu_backend->conv(input, kernel_3, 1, 1);
    }

    profiler.stop();

    std::cout << "Performance: " << std::fixed << std::setprecision(2)
              << profiler.get_performance() << " GFLOPS" << std::endl;
}

    return 0;
}
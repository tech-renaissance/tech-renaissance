#include "tech_renaissance.h"

using namespace tr;

int main() {
    std::cout << std::fixed << std::setprecision(4);
    Logger::get_instance().set_quiet_mode(true);

    auto cpu_backend = BackendManager::get_cpu_backend();
    auto cuda_backend = BackendManager::get_cuda_backend();

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
        Tensor input = cpu_backend->randn(Shape(4, 3, 9, 9), time(nullptr)); // 修复：C=1 (匹配 K_out)
        Tensor kernel_3 = cpu_backend->randn(Shape(3, 2, 3, 3), time(nullptr)); // (K_out=1, C_in=3)

        Tensor input_cuda = cuda_backend->from_cpu(input);
        Tensor kernel_3_cuda = cuda_backend->from_cpu(kernel_3);

        // 增加PyTorch结果验证
        PythonSession ps("default", "verify", true);
        ps.start();

        std::cout << "\nTEST 1: tconv_k3_s2_p1" << std::endl;
        auto tr_result_cuda = cuda_backend->transposed_conv(input_cuda, kernel_3_cuda, 2, 1);
        auto tr_result = cuda_backend->to_cpu(tr_result_cuda);
        auto pytorch_result = ps.calculate("tconv_k3_s2_p1", input, kernel_3);

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
    // 测试大样例（以前不能跑通）
    Tensor input = cpu_backend->randn(Shape(32, 512, 7, 7), time(nullptr));
    Tensor kernel_3 = cpu_backend->randn(Shape(512, 512, 3, 3), time(nullptr));

    Tensor input_cuda = cuda_backend->from_cpu(input);
    Tensor kernel_3_cuda = cuda_backend->from_cpu(kernel_3);

    std::cout << std::endl;
    std::cout << "****************************************" << std::endl;
    std::cout << "******* PERFORMANCE VERIFICATION *******" << std::endl;
    std::cout << "****************************************" << std::endl;

    constexpr int iterations = 100;
    constexpr int warmup_iterations = 10;

    // 预热运行
    for (int i = 0; i < warmup_iterations; ++i) {
        cuda_backend->transposed_conv(input_cuda, kernel_3_cuda, 1, 1);
    }

    // 同步CUDA设备确保预热完成
    cuda_backend->synchronize();

    // 性能测试 - 使用简化的Profiler API
    Profiler profiler;
    profiler.set_iterations(iterations);
    profiler.describe_operation("conv_k3_s1_p1", input.shape(), kernel_3.shape());

    profiler.start();
    for (int i = 0; i < iterations; ++i) {
        cuda_backend->transposed_conv(input_cuda, kernel_3_cuda, 1, 1);
    }

    // 确保所有CUDA计算完成后再停止计时
    cuda_backend->synchronize();
    profiler.stop();

    std::cout << "Performance: " << std::fixed << std::setprecision(2)
              << profiler.get_performance() << " GFLOPS" << std::endl;
}

    return 0;
}
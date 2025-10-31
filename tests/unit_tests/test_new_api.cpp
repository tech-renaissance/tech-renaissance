#include "tech_renaissance.h"

using namespace tr;

int main() {
    std::cout << std::fixed << std::setprecision(4);
    Logger::get_instance().set_quiet_mode(true);

    auto cuda_backend = BackendManager::get_cuda_backend();
    auto cpu_backend = BackendManager::get_cpu_backend();

    // 使用二维Shape创建随机张量
    Tensor cpu_a = Tensor::randn(Shape(1024, 2048), time(nullptr));  // 1024 x 2048矩阵，种子22
    Tensor cpu_b = Tensor::randn(Shape(2048, 512), time(nullptr));  // 2048 x 512矩阵，种子22

    // // CPU到CUDA转换
    Tensor cuda_a = cuda_backend->from_cpu(cpu_a);
    Tensor cuda_b = cuda_backend->from_cpu(cpu_b);

    // 矩阵乘法
    Tensor cpu_result = Tensor::empty(Shape(1024, 512)); // 默认是DType::FP32, tr::CPU
    cpu_backend->mm(cpu_result, cpu_a, cpu_b);

    // CUDA矩阵乘法
    Tensor cuda_result = Tensor::empty(Shape(1024, 512), DType::FP32, tr::CUDA[0]);
    cuda_backend->mm(cuda_result, cuda_a, cuda_b);

    // ===== 性能测试 =====

    const int iterations = 100;
    const int warmup_iterations = 10;

    // 预热运行
    for (int i = 0; i < warmup_iterations; ++i) {
        cuda_backend->mm(cuda_result, cuda_a, cuda_b);
    }

    // 同步CUDA设备确保预热完成
    cuda_backend->synchronize();

    // 性能测试 - 使用简化的Profiler API
    Profiler profiler;
    profiler.set_iterations(iterations);
    profiler.describe_operation("mm", cuda_a.shape(), cuda_b.shape());

    profiler.start();
    for (int i = 0; i < iterations; ++i) {
        cuda_backend->mm(cuda_result, cuda_a, cuda_b);
    }

    // 确保所有CUDA计算完成后再停止计时
    cuda_backend->synchronize();
    profiler.stop();

    // 直接获取性能结果
    std::cout << "Performance: " << std::fixed << std::setprecision(2)
              << profiler.get_performance() << " GFLOPS" << std::endl;

    // 结果一致性
    // 尝试使用copy_into来实现跨后端复制
    Tensor cuda_result_on_cpu = cpu_backend->zeros_like(cpu_result);
    cuda_backend->copy_into(cuda_result, cuda_result_on_cpu);

    // 以下是跨后端复制的另一种写法：
    // Tensor cuda_result_on_cpu = cuda_backend->to_cpu(cuda_result);

    // 使用CPU后端的is_close方法
    bool is_close = cpu_backend->is_close(cpu_result, cuda_result_on_cpu, 1e-4f);

    std::cout << "CPU vs CUDA results are close: " << (is_close ? "YES" : "NO") << std::endl;


    // 计算平均绝对误差和平均相对误差（使用新的API）
    double mean_abs_err = cpu_backend->get_mean_abs_err(cpu_result, cuda_result_on_cpu);
    double mean_rel_err = cpu_backend->get_mean_rel_err(cpu_result, cuda_result_on_cpu);

    std::cout << "Mean absolute error (Vs CPU): " << std::scientific << std::setprecision(6) << mean_abs_err << std::endl;
    std::cout << "Mean relative error (Vs CPU): " << std::scientific << std::setprecision(6) << mean_rel_err << std::endl;


#ifdef TR_BUILD_PYTHON_SESSION
    {
        // 增加PyTorch结果验证
        PythonSession ps("default", "verify", true);
        ps.start();

        Tensor pytorch_result = ps.calculate("matmul", cpu_a, cpu_b);

        ps.please_exit();

        double mean_abs_err_pytorch = cpu_backend->get_mean_abs_err(pytorch_result, cuda_result_on_cpu);
        double mean_rel_err_pytorch = cpu_backend->get_mean_rel_err(pytorch_result, cuda_result_on_cpu);

        std::cout << "Mean absolute error (Vs PyTorch): " << std::scientific << std::setprecision(6) << mean_abs_err_pytorch << std::endl;
        std::cout << "Mean relative error (Vs PyTorch): " << std::scientific << std::setprecision(6) << mean_rel_err_pytorch << std::endl;
    }
#endif
    return 0;
}
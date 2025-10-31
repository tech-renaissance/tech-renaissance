#include "tech_renaissance.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <iomanip>
#include <memory>

using namespace tr;

// 测试结构体
struct TestCase {
    std::string name;
    std::string pytorch_cmd;
    std::function<Tensor(CpuBackend&, const Tensor&)> cpu_func_non_inplace;
    std::function<void(CpuBackend&, Tensor&)> cpu_func_inplace;
};

int main() {
    std::cout << std::fixed << std::setprecision(6);
    Logger::get_instance().set_quiet_mode(true);
    auto cpu_backend = BackendManager::get_cpu_backend();

    PythonSession ps("default", "verify", true);
    ps.start();

#ifdef TR_BUILD_PYTHON_SESSION

    // 定义测试用例
    std::vector<TestCase> test_cases = {
        {
            "zeros_like", "zeros_like",
            [](CpuBackend& backend, const Tensor& input) { return backend.zeros_like(input); },
            [](CpuBackend& backend, Tensor& input) { backend.zeros_inplace(input); }
        },
        {
            "ones_like", "ones_like",
            [](CpuBackend& backend, const Tensor& input) { return backend.ones_like(input); },
            [](CpuBackend& backend, Tensor& input) { backend.ones_inplace(input); }
        },
        {
            "relu", "relu",
            [](CpuBackend& backend, const Tensor& input) { return backend.relu(input); },
            [](CpuBackend& backend, Tensor& input) { backend.relu_inplace(input); }
        },
        {
            "sign", "sign",
            [](CpuBackend& backend, const Tensor& input) { return backend.sign(input); },
            [](CpuBackend& backend, Tensor& input) { backend.sign_inplace(input); }
        },
        {
            "square", "square",
            [](CpuBackend& backend, const Tensor& input) { return backend.square(input); },
            [](CpuBackend& backend, Tensor& input) { backend.square_inplace(input); }
        },
        {
            "sqrt", "sqrt",
            [](CpuBackend& backend, const Tensor& input) { return backend.sqrt(input); },
            [](CpuBackend& backend, Tensor& input) { backend.sqrt_inplace(input); }
        },
        {
            "abs", "abs",
            [](CpuBackend& backend, const Tensor& input) { return backend.abs(input); },
            [](CpuBackend& backend, Tensor& input) { backend.abs_inplace(input); }
        },
        {
            "negative", "negative",
            [](CpuBackend& backend, const Tensor& input) { return backend.negative(input); },
            [](CpuBackend& backend, Tensor& input) { backend.negative_inplace(input); }
        },
        {
            "reciprocal", "reciprocal",
            [](CpuBackend& backend, const Tensor& input) { return backend.reciprocal(input); },
            [](CpuBackend& backend, Tensor& input) { backend.reciprocal_inplace(input); }
        }
    };

    int passed = 0;
    int total = 0;

    // 执行所有测试
    for (size_t i = 0; i < test_cases.size(); ++i) {
        const auto& test_case = test_cases[i];

        std::cout << "\n=== Testing " << test_case.name << " ===" << std::endl;

        // TEST 1: 非原地运算
        total++;
        try {
            // 为sqrt测试生成正数张量，其他测试用普通张量
            Tensor input;
            if (test_case.name == "sqrt") {
                // 生成4维正数张量，范围[0.1, 10.0]
                input = Tensor::uniform(Shape(2, 3, 4, 5), 0.1f, 10.0f, static_cast<int>(time(nullptr)) + i);
            } else {
                // 生成普通4维张量，范围[-1.0, 1.0]
                input = Tensor::uniform(Shape(2, 3, 4, 5), -1.0f, 1.0f, static_cast<int>(time(nullptr)) + i);
            }

            // PyTorch验算
            Tensor pytorch_result = ps.calculate(test_case.pytorch_cmd, input, 5000);

            // CPU计算
            Tensor cpu_result = test_case.cpu_func_non_inplace(*cpu_backend, input);

            // 验证结果
            bool is_close = cpu_backend->is_close(cpu_result, pytorch_result, 5e-5f);
            std::cout << "  Non-inplace: " << (is_close ? "PASSED" : "FAILED") << std::endl;

            if (is_close) passed++;
            else {
                std::cout << "  Error: CPU result differs from PyTorch!" << std::endl;
                // 打印一些调试信息
                if (cpu_result.numel() > 0) {
                    std::cout << "  CPU shape: " << cpu_result.shape().to_string() << ", PyTorch shape: " << pytorch_result.shape().to_string() << std::endl;
                    // 只对标量张量或1元素张量调用item()
                    if (cpu_result.numel() == 1) {
                        float cpu_val = cpu_result.item<float>();
                        float torch_val = pytorch_result.item<float>();
                        std::cout << "  CPU[0]=" << cpu_val << ", PyTorch[0]=" << torch_val << std::endl;
                    } else {
                        // 比较前几个元素
                        float* cpu_data = static_cast<float*>(cpu_result.data_ptr());
                        float* torch_data = static_cast<float*>(pytorch_result.data_ptr());
                        std::cout << "  CPU[0]=" << cpu_data[0] << ", PyTorch[0]=" << torch_data[0] << std::endl;
                        if (cpu_result.numel() > 1) {
                            std::cout << "  CPU[1]=" << cpu_data[1] << ", PyTorch[1]=" << torch_data[1] << std::endl;
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cout << "  Non-inplace: FAILED (Exception: " << e.what() << ")" << std::endl;
        }

        // TEST 2: 原地运算
        total++;
        try {
            // 为sqrt测试生成正数张量，其他测试用普通张量
            Tensor input;
            if (test_case.name == "sqrt") {
                // 生成4维正数张量，范围[0.1, 10.0]
                input = Tensor::uniform(Shape(2, 3, 4, 5), 0.1f, 10.0f, static_cast<int>(time(nullptr)) + i + 1000);
            } else {
                // 生成普通4维张量，范围[-1.0, 1.0]
                input = Tensor::uniform(Shape(2, 3, 4, 5), -1.0f, 1.0f, static_cast<int>(time(nullptr)) + i + 1000);
            }

            // 先发送到Python获取PyTorch结果
            Tensor pytorch_result = ps.calculate(test_case.pytorch_cmd, input, 5000);

            // 然后执行原地运算
            test_case.cpu_func_inplace(*cpu_backend, input);

            // 验证结果
            bool is_close = cpu_backend->is_close(input, pytorch_result, 5e-5f);
            std::cout << "  Inplace:     " << (is_close ? "PASSED" : "FAILED") << std::endl;

            if (is_close) passed++;
            else {
                std::cout << "  Error: Inplace result differs from PyTorch!" << std::endl;
                // 打印一些调试信息
                if (input.numel() > 0) {
                    std::cout << "  CPU shape: " << input.shape().to_string() << ", PyTorch shape: " << pytorch_result.shape().to_string() << std::endl;
                    // 只对标量张量或1元素张量调用item()
                    if (input.numel() == 1) {
                        float cpu_val = input.item<float>();
                        float torch_val = pytorch_result.item<float>();
                        std::cout << "  CPU[0]=" << cpu_val << ", PyTorch[0]=" << torch_val << std::endl;
                    } else {
                        // 比较前几个元素
                        float* cpu_data = static_cast<float*>(input.data_ptr());
                        float* torch_data = static_cast<float*>(pytorch_result.data_ptr());
                        std::cout << "  CPU[0]=" << cpu_data[0] << ", PyTorch[0]=" << torch_data[0] << std::endl;
                        if (input.numel() > 1) {
                            std::cout << "  CPU[1]=" << cpu_data[1] << ", PyTorch[1]=" << torch_data[1] << std::endl;
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cout << "  Inplace:     FAILED (Exception: " << e.what() << ")" << std::endl;
        }
    }

    // 输出总结
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "Test Summary: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "Success Rate: " << (100.0 * passed / total) << "%" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    if (passed == total) {
        std::cout << "*** All tests PASSED! ***" << std::endl;
    } else {
        std::cout << "*** Some tests FAILED! ***" << std::endl;
    }

    ps.please_exit();

#else
    std::cout << "Python session not enabled. Skipping tests." << std::endl;
    std::cout << "Build with -DTR_BUILD_PYTHON_SESSION=ON to enable Python tests." << std::endl;
#endif

    return (passed == total) ? 0 : 1;
}
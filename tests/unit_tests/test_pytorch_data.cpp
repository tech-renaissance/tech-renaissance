/**
 * @file test_pytorch_data.cpp
 * @brief 进一步测试PyTorchSession类
 * @details 借助test_pytorch_session.cpp的成功经验，进一步测试用PyTorch生成数据
 * @version 1.19.01
 * @date 2025-10-29
 * @author 技术觉醒团队
 * @note 依赖项: PyTorchSession, Tensor
 * @note 所属系列: tests
 */

#ifdef TR_BUILD_PYTORCH_SESSION

#include "tech_renaissance.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace tr;

// Python脚本路径常量，使用PROJECT_ROOT_DIR确保跨环境一致性
const std::string PYTHON_SCRIPT_PATH = std::string(PROJECT_ROOT_DIR) +
    "/python/tests/python_server.py";


// 测试1: Python程序正常结束的情况
bool test_python_natural_exit() {
    bool test_passed = true;

    try {
        PyTorchSession session(PYTHON_SCRIPT_PATH, "natural_exit_test");
        session.start();

        // 第一次请求 - 等待并显示响应
        session.send_request(R"({"cmd": "hello", "params": "world!"})");
        std::string response1 = session.wait_for_response(5000);
        std::cout << "Response 1: " << response1 << std::endl;

        // 第二次请求 - 使用fetch_response
        std::string response2 = session.fetch_response(R"({"cmd": "hi", "params": "world!"})", 5000);
        std::cout << "Response 2: " << response2 << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        session.send_request(R"({"cmd": "exit"})");

        bool exited_naturally = false;
        for (int i = 0; i < 100; i++) {
            if (!session.is_alive()) {
                exited_naturally = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (!exited_naturally) {
            std::cout << "[TEST] Python process did not exit naturally, terminating..." << std::endl;
            session.terminate();
            test_passed = false;
        }

        if (session.is_alive()) {
            std::cout << "[TEST] Python process is still alive after termination!" << std::endl;
            test_passed = false;
        }

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        test_passed = false;
    }
    return test_passed;
}

// 测试2: 在C++中手动控制Python程序结束
bool test_manual_termination() {
    bool test_passed = true;

    try {
        PyTorchSession session(PYTHON_SCRIPT_PATH, "manual_terminate_test");
        session.start();

        // 等待一小段时间确保进程完全启动
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        session.terminate();

        if (session.is_alive()) {
            std::cout << "[TEST] Python process is still alive after manual termination!" << std::endl;
            test_passed = false;
        }

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        test_passed = false;
    }
    return test_passed;
}


// 测试3: 期望的写法A - 2维矩阵乘法
bool test_matmul_style_a() {
    bool test_passed = true;

    try {
        std::cout << "\n=== Test Style A: 2D Matrix Multiplication ===" << std::endl;
        PyTorchSession session(PYTHON_SCRIPT_PATH, "matmul_test_a");
        session.start();

        // 等待进程启动
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // 创建测试张量：4x3 矩阵 × 3x5 矩阵 = 4x5 矩阵
        Tensor tensor_a = Tensor::full(Shape(4, 3), 1.5f, DType::FP32, tr::CPU);
        Tensor tensor_b = Tensor::full(Shape(3, 5), 2.0f, DType::FP32, tr::CPU);

        tensor_a.print("tensor_a (4x3)");
        tensor_b.print("tensor_b (3x5)");

        // 测试Shape类的<<运算符
        Shape shape_a = tensor_a.shape();
        Shape shape_b = tensor_b.shape();
        std::cout << "shape_a: " << shape_a << std::endl;
        std::cout << "shape_b: " << shape_b << std::endl;

        // 测试Tensor类的<<运算符（与print()的区别）
        std::cout << "tensor_a via <<: " << tensor_a << std::endl;
        std::cout << "tensor_b via <<: " << tensor_b << std::endl;

        std::cout << "[TEST] Style A: Send tensors and execute matrix multiplication..." << std::endl;

        // 期望的写法A：
        session.send_tensor(tensor_a, "a");
        session.send_tensor(tensor_b, "b");

        // 预期结果：4x5矩阵，每个元素 = 1.5 * 2.0 * 3 = 9.0
        Tensor result = session.fetch_tensor(R"({"cmd": "matmul", "params": "a,b"})", 10000);

        if (result.numel() == 0) {
            std::cout << "[TEST] Failed to get matmul result!" << std::endl;
            test_passed = false;
        } else {
            std::cout << "[TEST] Successfully got matrix multiplication result from PyTorch" << std::endl;
            result.print("result (4x5)");

            // 验证结果：应该是4x5矩阵，每个元素为9.0
            bool shape_correct = (result.shape().n() == 1 && result.shape().c() == 1 &&
                                result.shape().h() == 4 && result.shape().w() == 5);
            if (!shape_correct) {
                std::cout << "[TEST] Wrong result shape! Expected (1,1,4,5), got "
                         << result.shape().n() << "," << result.shape().c() << ","
                         << result.shape().h() << "," << result.shape().w() << std::endl;
                test_passed = false;
            }
        }

        session.send_request(R"({"cmd": "exit"})");

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        test_passed = false;
    }
    return test_passed;
}

// 测试4: 期望的写法B - 4维张量加法
bool test_add_style_b() {
    bool test_passed = true;

    try {
        std::cout << "\n=== Test Style B: 4D Tensor Addition ===" << std::endl;
        PyTorchSession session(PYTHON_SCRIPT_PATH, "add_test_b");
        session.start();

        // 等待进程启动
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // 创建测试张量：2x3x4x5 张量
        Tensor tensor_a = Tensor::full(Shape(2, 3, 4, 5), 3.0f, DType::FP32, tr::CPU);
        Tensor tensor_b = Tensor::full(Shape(2, 3, 4, 5), 4.0f, DType::FP32, tr::CPU);

        tensor_a.print("tensor_a (2x3x4x5)");
        tensor_b.print("tensor_b (2x3x4x5)");

        std::cout << "[TEST] Style B: Send tensors and execute addition..." << std::endl;

        // 期望的写法B：
        session.send_tensor(tensor_a, "a");
        session.send_tensor(tensor_b, "b");
        session.send_request(R"({"cmd": "add", "params": "a,b"})");

        // 模拟耗时任务
        std::cout << "[TEST] Simulating time-consuming task, waiting 2 seconds..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        // 获取结果
        Tensor result = session.wait_for_tensor(10000);

        if (result.numel() == 0) {
            std::cout << "[TEST] Failed to get add result!" << std::endl;
            test_passed = false;
        } else {
            std::cout << "[TEST] Successfully got addition result from PyTorch" << std::endl;
            result.print("result (2x3x4x5)");

            // 验证结果：应该是2x3x4x5张量，每个元素为7.0 (3.0 + 4.0)
            bool shape_correct = (result.shape().n() == 2 && result.shape().c() == 3 &&
                                result.shape().h() == 4 && result.shape().w() == 5);
            if (!shape_correct) {
                std::cout << "[TEST] Wrong result shape! Expected (2,3,4,5), got "
                         << result.shape().n() << "," << result.shape().c() << ","
                         << result.shape().h() << "," << result.shape().w() << std::endl;
                test_passed = false;
            }
        }

        session.send_request(R"({"cmd": "exit"})");

    } catch (const TRException& e) {
        std::cout << "[TEST] Exception caught: " << e.what() << std::endl;
        test_passed = false;
    }
    return test_passed;
}


int main() {
    std::cout << "=== Tech Renaissance PyTorch Tensor Communication Test ===" << std::endl;

    // 基础功能测试
    test_python_natural_exit();
    test_manual_termination();

    // 张量传递测试 - 写法A和B
    test_matmul_style_a();
    test_add_style_b();

    std::cout << "\n=== All tests completed ===" << std::endl;
    return 0;
}






#else // TR_BUILD_PYTORCH_SESSION

int main() {
    std::cout << "PyTorchSession support is not enabled in this build." << std::endl;
    std::cout << "Enable with TR_BUILD_PYTORCH_SESSION=ON" << std::endl;
    return 0;
}

#endif // TR_BUILD_PYTORCH_SESSION
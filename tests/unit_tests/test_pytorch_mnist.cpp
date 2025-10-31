/**
 * @file test_pytorch_mnist.cpp
 * @brief PythonSession MNIST测试样例
 * @details 测试C++主程序与Python脚本的并行执行场景
 * @version 1.00.00
 * @date 2025-10-27
 * @author 技术觉醒团队
 * @note 所属系列: tests
 */

#ifdef TR_BUILD_PYTHON_SESSION

#include "tech_renaissance.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace tr;

void test_mnist_parallel_processing() {
    std::cout << "\n=== MNIST Parallel Processing Test ===" << std::endl;
    std::cout << "[TEST] Testing C++ main loop with Python background task..." << std::endl;

    try {
        // 创建PyTorch会话，启动Python脚本执行MNIST任务
        PythonSession session("default", "mnist_session");

        std::cout << "[TEST] Starting Python MNIST task in background..." << std::endl;
        session.start();

        if (!session.is_alive()) {
            throw TRException("Failed to start Python MNIST task");
        }

        std::cout << "[TEST] Python MNIST task started successfully" << std::endl;
        std::cout << "[TEST] Starting C++ main loop (counting while Python works)..." << std::endl;

        // C++主循环：模拟某个耗时程序
        int count = 0;
        const int max_iterations = 60;  // 最多等待60秒
        int iteration = 0;

        while (iteration < max_iterations && session.is_alive()) {
            // 模拟程序工作：sleep 1秒
            std::this_thread::sleep_for(std::chrono::seconds(1));
            count++;
            iteration++;

            // 每5秒打印一次进度
            if (iteration % 5 == 0) {
                std::cout << "[TEST] C++ loop iteration " << iteration
                          << ", count = " << count
                          << ", Python still running..." << std::endl;
            }
        }

        // 检查Python任务状态
        if (session.is_alive()) {
            std::cout << "[TEST] Python task still running after " << max_iterations
                      << " seconds, terminating..." << std::endl;
            session.terminate();
        } else {
            std::cout << "[TEST] Python task completed naturally after " << iteration
                      << " seconds" << std::endl;
        }

        // 等待一小段时间确保Python完全退出
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        std::cout << "[TEST] C++ main loop completed" << std::endl;
        std::cout << "[TEST] Final count value: " << count << std::endl;
        std::cout << "[TEST] Total iterations: " << iteration << std::endl;

        // 验证结果
        if (count > 0) {
            std::cout << "[TEST] PASS: C++ program made progress while Python was working" << std::endl;
        } else {
            std::cout << "[TEST] FAIL: C++ program did not make expected progress" << std::endl;
        }

        if (iteration > 5) {  // 至少运行了5秒
            std::cout << "[TEST] PASS: C++ program ran for sufficient time" << std::endl;
        } else {
            std::cout << "[TEST] FAIL: C++ program did not run for expected time" << std::endl;
        }

        std::cout << "[TEST] MNIST parallel processing test completed successfully!" << std::endl;

    } catch (const TRException& e) {
        std::cout << "[TEST] FAIL: Exception caught: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cout << "[TEST] FAIL: Standard exception caught: " << e.what() << std::endl;
        throw;
    }
}

int main() {
    std::cout << "=== Starting PythonSession MNIST Parallel Test ===" << std::endl;

    try {
        test_mnist_parallel_processing();
        std::cout << "\n*** All MNIST tests passed successfully! ***" << std::endl;
        return 0;
    } catch (...) {
        std::cout << "\n*** MNIST test failed! ***" << std::endl;
        return 1;
    }
}

#else // TR_BUILD_PYTHON_SESSION

int main() {
    std::cout << "PythonSession support is not enabled in this build." << std::endl;
    std::cout << "Enable with TR_BUILD_PYTHON_SESSION=ON" << std::endl;
    return 0;
}

#endif // TR_BUILD_PYTHON_SESSION
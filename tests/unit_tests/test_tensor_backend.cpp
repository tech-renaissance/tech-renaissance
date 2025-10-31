/**
 * @file test_tensor_backend.cpp
 * @brief 张量-后端联合测试样例
 * @details 实现设计文档"【八、方案重要更新与补充】（十六）关于后端与Tensor类的联合测试样例"
 *          测试CPU后端和CUDA后端上的简单张量加法运算及显示
 * @version 1.01.01
 * @date 2025-10-25
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h
 * @note 所属系列: tests
 */

#include "tech_renaissance.h"
#include <iostream>

using namespace tr;

int main() {
    try {
        // 获取BackendManager实例（自动初始化）
        BackendManager& manager = BackendManager::instance();

        // 取得后端句柄
        auto cpu = manager.get_backend(CPU);
        std::shared_ptr<Backend> cuda = nullptr;

        try {
            cuda = manager.get_backend(CUDA[0]);
        } catch (const std::exception& e) {
            std::cout << "CUDA backend not available: " << e.what() << std::endl;
        }

        // ===== CPU后端 =====
        std::cout << "=== CPU Backend Test ===" << std::endl;
        Tensor cpu_a = Tensor::empty(Shape(3, 4), DType::FP32, CPU);
        Tensor cpu_b = Tensor::empty(Shape(3, 4), DType::FP32, CPU);
        Tensor cpu_result = Tensor::empty(Shape(3, 4), DType::FP32, CPU);

        // 填充张量
        cpu->fill(cpu_a, 1.5f);
        cpu->fill(cpu_b, 2.5f);

        // 执行加法
        cpu->add(cpu_result, cpu_a, cpu_b);

        // 打印结果
        cpu_a.print("cpu_a");
        cpu_b.print("cpu_b");
        cpu_result.print("cpu_result");

        // ===== CUDA后端（如果可用） =====
        if (cuda) {
            std::cout << "\n=== CUDA Backend Test ===" << std::endl;
            Tensor gpu_a = Tensor::empty(Shape(2, 4), DType::FP32, CUDA[0]);
            Tensor gpu_b = Tensor::empty(Shape(2, 4), DType::FP32, CUDA[0]);
            Tensor gpu_result = Tensor::empty(Shape(2, 4), DType::FP32, CUDA[0]);

            // 填充张量
            cuda->fill(gpu_a, 3.14f);
            cuda->fill(gpu_b, -3.14f);

            // 执行加法
            cuda->add(gpu_result, gpu_a, gpu_b);

            // 打印结果
            gpu_a.print("gpu_a");
            gpu_b.print("gpu_b");
            gpu_result.print("gpu_result");
        } else {
            std::cout << "\n=== CUDA Backend Skipped ===" << std::endl;
        }

        std::cout << "\n[SUCCESS] Tensor-Backend joint test completed!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

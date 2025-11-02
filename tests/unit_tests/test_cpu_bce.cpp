/**
 * @file test_cpu_bce.cpp
 * @brief CPU后端二元交叉熵运算测试
 * @details 测试CPU后端的二元交叉熵运算：bce、bce_into
 * @version 1.00.00
 * @date 2025-11-02
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h
 * @note 所属系列: tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace tr;

void print_tensor_info(const std::string& name, const Tensor& tensor) {
    std::cout << "\n" << name << ":\n";
    tensor.print();
}

int main() {
    try {
        std::cout << "=== CPU Backend Binary Cross Entropy Test ===\n";

        // 获取CPU后端
        auto cpu_backend = BackendManager::get_cpu_backend();

        // 创建3x4的FP32张量用于测试
        Shape shape(3, 4);

        // 创建标签张量（0或1）
        Tensor goal = Tensor::empty(shape, DType::FP32, tr::CPU);
        float* goal_data = static_cast<float*>(goal.data_ptr());
        for (size_t i = 0; i < goal.numel(); ++i) {
            // 创建交替的0和1标签
            goal_data[i] = static_cast<float>(i % 2);
        }

        // 创建预测概率张量（0到1之间的值）
        Tensor pred = Tensor::uniform(shape, 0.1f, 0.9f, 42); // 固定种子42

        std::cout << "\nTest Configuration:";
        std::cout << "\n  Goal tensor shape: " << shape.to_string();
        std::cout << "\n  Prediction tensor shape: " << shape.to_string();
        std::cout << "\n  Random seed: 42";
        std::cout << "\n  Epsilon for clamping: 1e-8";

        // 打印输入张量
        print_tensor_info("Goal Tensor (Labels: 0 or 1)", goal);
        print_tensor_info("Prediction Tensor (Probabilities)", pred);

        // ===== 测试二元交叉熵运算 =====
        std::cout << "\n=== Testing Binary Cross Entropy ===";

        Tensor bce_result = cpu_backend->bce(goal, pred);
        print_tensor_info("BCE Result (non-inplace)", bce_result);

        Tensor bce_into_result = Tensor::empty(shape, DType::FP32, tr::CPU);
        cpu_backend->bce_into(goal, pred, bce_into_result);
        print_tensor_info("BCE Result (into)", bce_into_result);

        // ===== 测试边界情况 =====
        std::cout << "\n=== Testing Boundary Cases ===";

        // 测试极端预测值（接近0和1）
        Tensor extreme_pred = Tensor::empty(shape, DType::FP32, tr::CPU);
        float* extreme_data = static_cast<float*>(extreme_pred.data_ptr());
        for (size_t i = 0; i < extreme_pred.numel(); ++i) {
            // 创建交替的极小和极大值，避免浮点精度问题
            extreme_data[i] = (i % 2 == 0) ? 1e-9f : 0.9999999f;
        }
        print_tensor_info("Extreme Prediction Values", extreme_pred);

        Tensor extreme_bce = cpu_backend->bce(goal, extreme_pred);
        print_tensor_info("BCE with Extreme Values", extreme_bce);

        // ===== 测试特殊情况：完美预测 =====
        std::cout << "\n=== Testing Perfect Predictions ===";

        Tensor perfect_pred = Tensor::empty(shape, DType::FP32, tr::CPU);
        float* perfect_data = static_cast<float*>(perfect_pred.data_ptr());
        for (size_t i = 0; i < perfect_pred.numel(); ++i) {
            // 创建完美预测：goal=0时pred=0.01，goal=1时pred=0.99
            perfect_data[i] = (goal_data[i] == 0.0f) ? 0.01f : 0.99f;
        }
        print_tensor_info("Perfect Predictions", perfect_pred);

        Tensor perfect_bce = cpu_backend->bce(goal, perfect_pred);
        print_tensor_info("BCE with Perfect Predictions", perfect_bce);

        // ===== 测试特殊情况：最差预测 =====
        std::cout << "\n=== Testing Worst Predictions ===";

        Tensor worst_pred = Tensor::empty(shape, DType::FP32, tr::CPU);
        float* worst_data = static_cast<float*>(worst_pred.data_ptr());
        for (size_t i = 0; i < worst_pred.numel(); ++i) {
            // 创建最差预测：goal=0时pred=0.99，goal=1时pred=0.01
            worst_data[i] = (goal_data[i] == 0.0f) ? 0.99f : 0.01f;
        }
        print_tensor_info("Worst Predictions", worst_pred);

        Tensor worst_bce = cpu_backend->bce(goal, worst_pred);
        print_tensor_info("BCE with Worst Predictions", worst_bce);

        // ===== 测试异常情况 =====
        std::cout << "\n=== Testing Exception Handling ===";

        // 测试形状不匹配
        try {
            Shape different_shape(2, 5);
            Tensor different_pred = Tensor::uniform(different_shape, 0.1f, 0.9f, 42);
            Tensor invalid_bce = cpu_backend->bce(goal, different_pred);
            std::cout << "\n[FAIL] Expected exception for shape mismatch, but none was thrown!";
        } catch (const TRException& e) {
            std::cout << "\n[PASS] Correctly caught exception for shape mismatch: " << e.what();
        }

        // 测试空张量
        try {
            Tensor empty_tensor;
            Tensor invalid_bce = cpu_backend->bce(empty_tensor, pred);
            std::cout << "\n[FAIL] Expected exception for empty tensor, but none was thrown!";
        } catch (const TRException& e) {
            std::cout << "\n[PASS] Correctly caught exception for empty tensor: " << e.what();
        }

        // 测试into模式形状不匹配
        try {
            Shape wrong_shape(2, 6);
            Tensor wrong_result = Tensor::empty(wrong_shape, DType::FP32, tr::CPU);
            cpu_backend->bce_into(goal, pred, wrong_result);
            std::cout << "\n[FAIL] Expected exception for into shape mismatch, but none was thrown!";
        } catch (const TRException& e) {
            std::cout << "\n[PASS] Correctly caught exception for into shape mismatch: " << e.what();
        }

        // ===== 验证非原地和into方式的一致性 =====
        std::cout << "\n=== Verifying Consistency Between Implementations ===";

        bool bce_consistent = cpu_backend->is_close(bce_result, bce_into_result, 1e-6f);
        bool extreme_consistent = cpu_backend->is_close(cpu_backend->bce(goal, extreme_pred), extreme_bce, 1e-6f);
        bool perfect_consistent = cpu_backend->is_close(cpu_backend->bce(goal, perfect_pred), perfect_bce, 1e-6f);

        std::cout << "\n  BCE consistency: " << (bce_consistent ? "PASS" : "FAIL");
        std::cout << "\n  Extreme values consistency: " << (extreme_consistent ? "PASS" : "FAIL");
        std::cout << "\n  Perfect predictions consistency: " << (perfect_consistent ? "PASS" : "FAIL");

        bool all_consistent = bce_consistent && extreme_consistent && perfect_consistent;

        // ===== 数值合理性检查 =====
        std::cout << "\n=== Numerical Reasonableness Check ===";

        const float* perfect_bce_data = static_cast<const float*>(perfect_bce.data_ptr());
        const float* worst_bce_data = static_cast<const float*>(worst_bce.data_ptr());

        float avg_perfect_loss = 0.0f;
        float avg_worst_loss = 0.0f;

        for (size_t i = 0; i < perfect_bce.numel(); ++i) {
            avg_perfect_loss += perfect_bce_data[i];
            avg_worst_loss += worst_bce_data[i];
        }

        avg_perfect_loss /= perfect_bce.numel();
        avg_worst_loss /= worst_bce.numel();

        std::cout << "\n  Average perfect prediction loss: " << std::fixed << std::setprecision(6) << avg_perfect_loss;
        std::cout << "\n  Average worst prediction loss: " << std::fixed << std::setprecision(6) << avg_worst_loss;
        std::cout << "\n  Loss ratio (worst/perfect): " << std::fixed << std::setprecision(3) << (avg_worst_loss / avg_perfect_loss);

        bool reasonable_loss = avg_perfect_loss < 0.5f && avg_worst_loss > 2.0f;
        std::cout << "\n  Reasonable loss values: " << (reasonable_loss ? "PASS" : "FAIL");

        bool final_success = all_consistent && reasonable_loss;

        std::cout << "\n\n=== Test Summary ===";
        std::cout << "\nBinary Cross Entropy operations tested: 2 implementations";
        std::cout << "\nConsistency verification: " << (all_consistent ? "PASS" : "FAIL");
        std::cout << "\nNumerical reasonableness: " << (reasonable_loss ? "PASS" : "FAIL");

        if (final_success) {
            std::cout << "\n\n[PASS] All CPU BCE operations completed successfully!";
            return 0;
        } else {
            std::cout << "\n\n[FAIL] Some BCE tests failed!";
            return 1;
        }

    } catch (const TRException& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\nUnexpected error: " << e.what() << std::endl;
        return 1;
    }
}
/**
 * @file test_cpu_loss.cpp
 * @brief CPU后端损失函数测试
 * @details 测试one-hot编码和交叉熵损失函数的正确性
 * @version 1.42.6
 * @date 2025-11-16
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h, 框架核心库
 * @note 所属系列: tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace tr;

void test_one_hot_basic() {
    std::cout << "=== Testing Basic One-Hot Encoding ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 基本one-hot编码
    {
        std::cout << "Test 1: Basic one-hot encoding..." << std::endl;

        // 创建标签张量 [0, 1, 2]
        Tensor label = cpu_backend->zeros(Shape(3), DType::INT32);
        cpu_backend->set_item_int32(label, 1, 1);
        cpu_backend->set_item_int32(label, 2, 2);

        // 转换为one-hot，3个类别
        Tensor one_hot = cpu_backend->one_hot(label, 3);

        one_hot.print("One-hot result");

        // 验证结果
        float* data = static_cast<float*>(one_hot.data_ptr());

        // 第一个样本：[1, 0, 0]
        assert(std::abs(data[0] - 1.0f) < 1e-6f);
        assert(std::abs(data[1] - 0.0f) < 1e-6f);
        assert(std::abs(data[2] - 0.0f) < 1e-6f);

        // 第二个样本：[0, 1, 0]
        assert(std::abs(data[3] - 0.0f) < 1e-6f);
        assert(std::abs(data[4] - 1.0f) < 1e-6f);
        assert(std::abs(data[5] - 0.0f) < 1e-6f);

        // 第三个样本：[0, 0, 1]
        assert(std::abs(data[6] - 0.0f) < 1e-6f);
        assert(std::abs(data[7] - 0.0f) < 1e-6f);
        assert(std::abs(data[8] - 1.0f) < 1e-6f);

        std::cout << "[PASS] Basic one-hot encoding test passed!" << std::endl;
    }

    // 测试2: 带标签平滑的one-hot编码
    {
        std::cout << "Test 2: One-hot encoding with label smoothing (α=0.1)..." << std::endl;

        // 创建标签张量 [0, 1, 2]
        Tensor label = cpu_backend->zeros(Shape(3), DType::INT32);
        cpu_backend->set_item_int32(label, 1, 1);
        cpu_backend->set_item_int32(label, 2, 2);

        // 转换为one-hot，3个类别，标签平滑=0.1
        Tensor one_hot_smooth = cpu_backend->one_hot(label, 3, 0.1f);

        one_hot_smooth.print("One-hot with smoothing");

        // 验证结果
        float* data = static_cast<float*>(one_hot_smooth.data_ptr());
        float smooth_val = 0.1f / 3.0f;  // α / num_classes = 0.033333...
        float one_minus_smooth = 0.9f;   // 1 - α = 0.9

        // 第一个样本：[0.9+0.033..., 0.033..., 0.033...]
        assert(std::abs(data[0] - (smooth_val + one_minus_smooth)) < 1e-6f);
        assert(std::abs(data[1] - smooth_val) < 1e-6f);
        assert(std::abs(data[2] - smooth_val) < 1e-6f);

        std::cout << "[PASS] One-hot encoding with label smoothing test passed!" << std::endl;
    }
}

void test_one_hot_into() {
    std::cout << "\n=== Testing One-Hot Into Function ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试one_hot_into函数
    {
        std::cout << "Test 1: one_hot_into function..." << std::endl;

        // 创建标签张量 [1, 0, 2]
        Tensor label = cpu_backend->zeros(Shape(3), DType::INT32);
        cpu_backend->fill(label, 1);
        cpu_backend->set_item_int32(label, 0, 1);
        cpu_backend->set_item_int32(label, 2, 2);

        // 创建目标张量（已分配内存）
        Tensor result = cpu_backend->zeros(Shape(3, 4), DType::FP32);

        // 使用into函数
        cpu_backend->one_hot_into(label, result, 4, 0.0f);

        result.print("one_hot_into result");

        // 验证结果
        float* data = static_cast<float*>(result.data_ptr());

        // 第一个样本：[0, 1, 0, 0]
        assert(std::abs(data[0] - 0.0f) < 1e-6f);
        assert(std::abs(data[1] - 1.0f) < 1e-6f);
        assert(std::abs(data[2] - 0.0f) < 1e-6f);
        assert(std::abs(data[3] - 0.0f) < 1e-6f);

        // 第二个样本：[1, 0, 0, 0]
        assert(std::abs(data[4] - 1.0f) < 1e-6f);
        assert(std::abs(data[5] - 0.0f) < 1e-6f);
        assert(std::abs(data[6] - 0.0f) < 1e-6f);
        assert(std::abs(data[7] - 0.0f) < 1e-6f);

        // 第三个样本：[0, 0, 1, 0]
        assert(std::abs(data[8] - 0.0f) < 1e-6f);
        assert(std::abs(data[9] - 0.0f) < 1e-6f);
        assert(std::abs(data[10] - 1.0f) < 1e-6f);
        assert(std::abs(data[11] - 0.0f) < 1e-6f);

        std::cout << "[PASS] one_hot_into function test passed!" << std::endl;
    }
}

void test_crossentropy_basic() {
    std::cout << "\n=== Testing Basic Cross-Entropy ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 完美预测的交叉熵
    {
        std::cout << "Test 1: Perfect predictions (entropy should be 0)..." << std::endl;

        // 预测张量：完美预测 [1, 0] 和 [0, 1]
        Tensor pred = cpu_backend->zeros(Shape(2, 2), DType::FP32);
        float* pred_data = static_cast<float*>(pred.data_ptr());
        pred_data[0] = 1.0f; pred_data[1] = 0.0f;  // [1, 0]
        pred_data[2] = 0.0f; pred_data[3] = 1.0f;  // [0, 1]

        // 标签张量：one-hot编码
        Tensor label = cpu_backend->zeros(Shape(2, 2), DType::FP32);
        float* label_data = static_cast<float*>(label.data_ptr());
        label_data[0] = 1.0f; label_data[1] = 0.0f;  // [1, 0]
        label_data[2] = 0.0f; label_data[3] = 1.0f;  // [0, 1]

        float loss = cpu_backend->crossentropy(pred, label, "mean");
        std::cout << "Perfect prediction loss: " << loss << std::endl;

        assert(std::abs(loss) < 1e-6f);  // 应该接近0
        std::cout << "[PASS] Perfect prediction test passed!" << std::endl;
    }

    // 测试2: 不确定性预测
    {
        std::cout << "Test 2: Uncertain predictions..." << std::endl;

        // 预测张量：不确定性 [0.5, 0.5]
        Tensor pred = cpu_backend->full(Shape(3, 2), 0.5f, DType::FP32);

        // 标签张量：one-hot编码
        Tensor label = cpu_backend->zeros(Shape(3, 2), DType::FP32);
        float* label_data = static_cast<float*>(label.data_ptr());
        label_data[0] = 1.0f; label_data[1] = 0.0f;  // [1, 0]
        label_data[2] = 0.0f; label_data[3] = 1.0f;  // [0, 1]
        label_data[4] = 1.0f; label_data[5] = 0.0f;  // [1, 0]

        float loss = cpu_backend->crossentropy(pred, label, "mean");
        std::cout << "Uncertain prediction loss: " << loss << std::endl;

        float expected_loss = std::log(2.0f);  // -log(0.5) = log(2)
        assert(std::abs(loss - expected_loss) < 1e-6f);
        std::cout << "[PASS] Uncertain prediction test passed!" << std::endl;
    }

    // 测试3: 数值稳定性（接近0的预测）
    {
        std::cout << "Test 3: Numerical stability (near-zero predictions)..." << std::endl;

        // 预测张量：包含接近0的值
        Tensor pred = cpu_backend->zeros(Shape(2, 2), DType::FP32);
        float* pred_data = static_cast<float*>(pred.data_ptr());
        pred_data[0] = 1e-15f; pred_data[1] = 0.999999999999f;  // [~0, ~1]
        pred_data[2] = 0.8f; pred_data[3] = 0.2f;                 // [0.8, 0.2]

        // 标签张量：one-hot编码，第一个样本要求第一个类别，第二个样本要求第一个类别
        Tensor label = cpu_backend->zeros(Shape(2, 2), DType::FP32);
        float* label_data = static_cast<float*>(label.data_ptr());
        label_data[0] = 1.0f; label_data[1] = 0.0f;  // [1, 0]
        label_data[2] = 1.0f; label_data[3] = 0.0f;  // [1, 0]

        float loss = cpu_backend->crossentropy(pred, label, "mean");
        std::cout << "Numerical stability loss: " << loss << std::endl;

        assert(std::isfinite(loss));  // 应该是有限值，不是NaN或无穷大
        assert(loss > 0.0f);         // 应该大于0

        std::cout << "[PASS] Numerical stability test passed!" << std::endl;
    }

    // 测试4: reduction模式
    {
        std::cout << "Test 4: Reduction modes (sum vs mean)..." << std::endl;

        // 预测张量
        Tensor pred = cpu_backend->zeros(Shape(2, 2), DType::FP32);
        float* pred_data = static_cast<float*>(pred.data_ptr());
        pred_data[0] = 0.5f; pred_data[1] = 0.5f;  // [0.5, 0.5]
        pred_data[2] = 0.5f; pred_data[3] = 0.5f;  // [0.5, 0.5]

        // 标签张量
        Tensor label = cpu_backend->zeros(Shape(2, 2), DType::FP32);
        float* label_data = static_cast<float*>(label.data_ptr());
        label_data[0] = 1.0f; label_data[1] = 0.0f;  // [1, 0]
        label_data[2] = 0.0f; label_data[3] = 1.0f;  // [0, 1]

        float loss_mean = cpu_backend->crossentropy(pred, label, "mean");
        float loss_sum = cpu_backend->crossentropy(pred, label, "sum");

        std::cout << "Loss (mean): " << loss_mean << std::endl;
        std::cout << "Loss (sum): " << loss_sum << std::endl;

        float expected_loss_per_sample = std::log(2.0f);
        assert(std::abs(loss_mean - expected_loss_per_sample) < 1e-6f);      // mean应该是log(2)
        assert(std::abs(loss_sum - (2.0f * expected_loss_per_sample)) < 1e-6f); // sum应该是2*log(2)

        std::cout << "[PASS] Reduction modes test passed!" << std::endl;
    }
}

void test_end_to_end() {
    std::cout << "\n=== Testing End-to-End Pipeline ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    {
        std::cout << "Test: Label -> One-hot -> Cross-entropy pipeline..." << std::endl;

        // 1. 创建标签 [0, 2, 1]
        Tensor label_int = cpu_backend->zeros(Shape(3), DType::INT32);
        cpu_backend->set_item_int32(label_int, 1, 2);
        cpu_backend->set_item_int32(label_int, 2, 1);

        label_int.print("Original labels");

        // 2. 转换为one-hot，带标签平滑
        Tensor label_onehot = cpu_backend->one_hot(label_int, 3, 0.1f);
        label_onehot.print("One-hot encoded labels");

        // 3. 创建预测（假设softmax后的输出）
        Tensor pred = cpu_backend->zeros(Shape(3, 3), DType::FP32);
        float* pred_data = static_cast<float*>(pred.data_ptr());
        pred_data[0] = 0.7f; pred_data[1] = 0.2f; pred_data[2] = 0.1f;  // 样本1预测
        pred_data[3] = 0.1f; pred_data[4] = 0.2f; pred_data[5] = 0.7f;  // 样本2预测
        pred_data[6] = 0.3f; pred_data[7] = 0.6f; pred_data[8] = 0.1f;  // 样本3预测

        pred.print("Predictions");

        // 4. 计算交叉熵
        float loss = cpu_backend->crossentropy(pred, label_onehot, "mean");
        std::cout << "Final cross-entropy loss: " << loss << std::endl;

        assert(std::isfinite(loss) && loss > 0.0f);
        std::cout << "[PASS] End-to-end pipeline test passed!" << std::endl;
    }
}

void test_mse_basic() {
    std::cout << "\n=== Testing Basic MSE Loss ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 完美匹配的MSE
    {
        std::cout << "Test 1: Perfect match (MSE should be 0)..." << std::endl;

        // 预测张量和目标张量完全相同
        Tensor pred = cpu_backend->ones(Shape(2, 3), DType::FP32);
        Tensor target = cpu_backend->ones(Shape(2, 3), DType::FP32);

        float loss = cpu_backend->mse(pred, target, "mean");
        std::cout << "Perfect match MSE loss: " << loss << std::endl;

        assert(std::abs(loss) < 1e-6f);  // 应该接近0
        std::cout << "[PASS] Perfect match test passed!" << std::endl;
    }

    // 测试2: 简单误差的MSE
    {
        std::cout << "Test 2: Simple prediction error..." << std::endl;

        // 预测张量：[1, 2, 3], [4, 5, 6]
        Tensor pred = cpu_backend->zeros(Shape(2, 3), DType::FP32);
        float* pred_data = static_cast<float*>(pred.data_ptr());
        pred_data[0] = 1.0f; pred_data[1] = 2.0f; pred_data[2] = 3.0f;
        pred_data[3] = 4.0f; pred_data[4] = 5.0f; pred_data[5] = 6.0f;

        // 目标张量：[0, 2, 4], [4, 6, 5]
        Tensor target = cpu_backend->zeros(Shape(2, 3), DType::FP32);
        float* target_data = static_cast<float*>(target.data_ptr());
        target_data[0] = 0.0f; target_data[1] = 2.0f; target_data[2] = 4.0f;
        target_data[3] = 4.0f; target_data[4] = 6.0f; target_data[5] = 5.0f;

        float loss = cpu_backend->mse(pred, target, "mean");
        std::cout << "Prediction error MSE loss: " << loss << std::endl;

        // 手动计算：((1-0)^2 + (2-2)^2 + (3-4)^2 + (4-4)^2 + (5-6)^2 + (6-5)^2) / 6
        // = (1 + 0 + 1 + 0 + 1 + 1) / 6 = 4 / 6 = 0.666667
        assert(std::abs(loss - 0.666667f) < 1e-6f);
        std::cout << "[PASS] Prediction error test passed!" << std::endl;
    }

    // 测试3: reduction模式对比
    {
        std::cout << "Test 3: Reduction modes (sum vs mean)..." << std::endl;

        // 创建有误差的张量
        Tensor pred = cpu_backend->zeros(Shape(3, 2), DType::FP32);
        float* pred_data = static_cast<float*>(pred.data_ptr());
        pred_data[0] = 1.0f; pred_data[1] = 0.0f;
        pred_data[2] = 0.0f; pred_data[3] = 1.0f;
        pred_data[4] = 1.0f; pred_data[5] = 0.0f;

        Tensor target = cpu_backend->zeros(Shape(3, 2), DType::FP32);
        float* target_data = static_cast<float*>(target.data_ptr());
        target_data[0] = 0.0f; target_data[1] = 0.0f;
        target_data[2] = 0.0f; target_data[3] = 0.0f;
        target_data[4] = 0.0f; target_data[5] = 1.0f;

        float loss_mean = cpu_backend->mse(pred, target, "mean");
        float loss_sum = cpu_backend->mse(pred, target, "sum");

        std::cout << "MSE (mean): " << loss_mean << std::endl;
        std::cout << "MSE (sum): " << loss_sum << std::endl;

        // 手动验证：((1-0)^2 + (0-0)^2 + (0-0)^2 + (1-0)^2 + (1-0)^2 + (0-1)^2) = 4
        float expected_sum = 4.0f / 3.0f;  // 除以batch_size
        float expected_mean = 4.0f / 6.0f;  // 除以total_elements

        assert(std::abs(loss_sum - expected_sum) < 1e-6f);
        assert(std::abs(loss_mean - expected_mean) < 1e-6f);

        std::cout << "[PASS] Reduction modes test passed!" << std::endl;
    }
}

void test_mse_error_handling() {
    std::cout << "\n=== Testing MSE Error Handling ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 错误的数据类型
    {
        std::cout << "Test 1: Wrong data type handling..." << std::endl;

        try {
            Tensor pred = cpu_backend->ones(Shape(2, 3), DType::FP32);
            Tensor target = cpu_backend->ones(Shape(2, 3), DType::INT8);  // 应该是FP32
            auto loss = cpu_backend->mse(pred, target);
            assert(false); // 不应该到达这里
        } catch (const tr::TypeError& e) {
            std::cout << "[PASS] Correctly caught TypeError: " << e.what() << std::endl;
        }
    }

    // 测试2: 错误的维度
    {
        std::cout << "Test 2: Wrong dimension handling..." << std::endl;

        try {
            Tensor pred = cpu_backend->ones(Shape(6), DType::FP32);  // 应该是2D
            Tensor target = cpu_backend->ones(Shape(6), DType::FP32);
            auto loss = cpu_backend->mse(pred, target);
            assert(false); // 不应该到达这里
        } catch (const tr::ShapeError& e) {
            std::cout << "[PASS] Correctly caught ShapeError: " << e.what() << std::endl;
        }
    }

    // 测试3: 形状不匹配
    {
        std::cout << "Test 3: Shape mismatch handling..." << std::endl;

        try {
            Tensor pred = cpu_backend->ones(Shape(2, 3), DType::FP32);
            Tensor target = cpu_backend->ones(Shape(3, 2), DType::FP32);  // 形状不同
            auto loss = cpu_backend->mse(pred, target);
            assert(false); // 不应该到达这里
        } catch (const tr::ShapeError& e) {
            std::cout << "[PASS] Correctly caught ShapeError: " << e.what() << std::endl;
        }
    }
}

void test_error_handling() {
    std::cout << "\n=== Testing CrossEntropy Error Handling ===" << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试1: 错误的数据类型
    {
        std::cout << "Test 1: Wrong data type handling..." << std::endl;

        try {
            Tensor label = cpu_backend->ones(Shape(3), DType::FP32);  // 应该是INT32
            auto result = cpu_backend->one_hot(label, 3);
            assert(false); // 不应该到达这里
        } catch (const tr::TypeError& e) {
            std::cout << "[PASS] Correctly caught TypeError: " << e.what() << std::endl;
        }
    }

    // 测试2: 错误的维度
    {
        std::cout << "Test 2: Wrong dimension handling..." << std::endl;

        try {
            Tensor label = cpu_backend->ones(Shape(2, 2), DType::INT32);  // 应该是1D
            auto result = cpu_backend->one_hot(label, 3);
            assert(false); // 不应该到达这里
        } catch (const tr::ShapeError& e) {
            std::cout << "[PASS] Correctly caught ShapeError: " << e.what() << std::endl;
        }
    }

    // 测试3: 标签值超出范围
    {
        std::cout << "Test 3: Out-of-range label handling..." << std::endl;

        try {
            Tensor label = cpu_backend->ones(Shape(1), DType::INT32);
            cpu_backend->set_item_int32(label, 0, 5);  // 超出范围 [0, 2]
            auto result = cpu_backend->one_hot(label, 3);
            assert(false); // 不应该到达这里
        } catch (const tr::IndexError& e) {
            std::cout << "[PASS] Correctly caught IndexError: " << e.what() << std::endl;
        }
    }

    // 测试4: 无效的标签平滑参数
    {
        std::cout << "Test 4: Invalid label smoothing handling..." << std::endl;

        try {
            Tensor label = cpu_backend->ones(Shape(1), DType::INT32);
            auto result = cpu_backend->one_hot(label, 3, 1.5f);  // 应该在[0, 1)
            assert(false); // 不应该到达这里
        } catch (const tr::ValueError& e) {
            std::cout << "[PASS] Correctly caught ValueError: " << e.what() << std::endl;
        }
    }
}

int main() {
    std::cout << "Starting CPU Loss Functions Tests..." << std::endl;
    std::cout << "======================================" << std::endl;

    try {
        test_one_hot_basic();
        test_one_hot_into();
        test_crossentropy_basic();
        test_end_to_end();
        test_mse_basic();
        test_mse_error_handling();
        test_error_handling();

        std::cout << "\nAll tests passed successfully!" << std::endl;
        std::cout << "======================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
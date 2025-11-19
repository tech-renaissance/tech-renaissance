/**
 * @file test_training.cpp
 * @brief 训练对齐测试 - 与PyTorch完全对齐
 * @details 使用TSR文件验证我们的框架与PyTorch的训练一致性
 * @version 1.00.00
 * @date 2025-11-19
 * @author 技术觉醒团队
 */

#include "tech_renaissance.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace tr;

// 全局CPU backend指针
static CpuBackend* cpu_backend = nullptr;

// 辅助函数：打印张量信息
void print_tensor_info(const std::string& name, const Tensor& tensor, bool show_values = false, int max_elements = 10) {
    std::cout << name << " - Shape: " << tensor.shape().to_string()
              << ", Dtype: " << static_cast<int>(tensor.dtype()) << std::endl;

    if (show_values && tensor.storage_allocated()) {
        std::cout << "  Values: [";
        auto* cpu_backend = dynamic_cast<CpuBackend*>(BackendManager::instance().get_backend(CPU).get());
        int64_t total_elements = tensor.numel();
        int elements_to_show = std::min(total_elements, static_cast<int64_t>(max_elements));

        for (int64_t i = 0; i < elements_to_show; ++i) {
            float value = cpu_backend->get_item_fp32(tensor, i);
            std::cout << std::fixed << std::setprecision(6) << value;
            if (i < elements_to_show - 1) std::cout << ", ";
        }

        if (total_elements > max_elements) {
            std::cout << ", ... (" << (total_elements - max_elements) << " more)";
        }
        std::cout << "]" << std::endl;
    }
}

// 辅助函数：比较两个张量是否接近
bool compare_tensors(const std::string& name, const Tensor& our_tensor, const Tensor& pytorch_tensor, float eps = 1e-5f) {
    auto* cpu_backend = dynamic_cast<CpuBackend*>(BackendManager::instance().get_backend(CPU).get());

    if (!cpu_backend->is_close(our_tensor, pytorch_tensor, eps)) {
        std::cout << "[FAIL] " << name << " mismatch!" << std::endl;
        std::cout << "  Our tensor:     ";
        print_tensor_info("", our_tensor, true);
        std::cout << "  PyTorch tensor: ";
        print_tensor_info("", pytorch_tensor, true);

        // 计算相对误差
        double rel_error = cpu_backend->get_mean_rel_err(our_tensor, pytorch_tensor);
        std::cout << "  Mean relative error: " << std::scientific << rel_error << std::endl;
        return false;
    } else {
        std::cout << "[PASS] " << name << " matches (eps=" << eps << ")" << std::endl;
        return true;
    }
}

int main() {
    std::cout << "=== PyTorch Training Alignment Test ===" << std::endl;
    std::cout << "Testing 2-layer NN (4-5-2) with SGD optimizer" << std::endl;
    std::cout << "========================================" << std::endl;

    // Set quiet mode
    Logger::get_instance().set_quiet_mode(true);

    try {
        auto backend = BackendManager::instance().get_cpu_backend();
        cpu_backend = dynamic_cast<CpuBackend*>(backend.get());

        // 1. 导入数据和标签
        std::cout << "\n1. Loading dataset..." << std::endl;
        Tensor data = IMPORT_TENSOR("R:/tech-renaissance/python/module/data.tsr");
        Tensor labels = IMPORT_TENSOR("R:/tech-renaissance/python/module/labels.tsr");
        print_tensor_info("Data", data, true);
        print_tensor_info("Labels", labels, true);

        // 2. 创建模型
        std::cout << "\n2. Creating model..." << std::endl;
        auto model = Model::create("AlignmentTest",
                                  std::make_shared<Linear>(4, 5, "Linear1", true),
                                  std::make_shared<Tanh>(),
                                  std::make_shared<Linear>(5, 2, "Linear2", true));

        model->set_backend(backend);
        model->train();

        // 3. 导入PyTorch初始化的权重并设置到我们的模型
        std::cout << "\n3. Loading PyTorch initial weights..." << std::endl;
        Tensor l1_weight = IMPORT_TENSOR("R:/tech-renaissance/python/module/init_l1_weight.tsr");
        Tensor l1_bias_1d = IMPORT_TENSOR("R:/tech-renaissance/python/module/init_l1_bias.tsr");
        Tensor l2_weight = IMPORT_TENSOR("R:/tech-renaissance/python/module/init_l2_weight.tsr");
        Tensor l2_bias_1d = IMPORT_TENSOR("R:/tech-renaissance/python/module/init_l2_bias.tsr");

        // 导入的PyTorch偏置是1D，需要reshape成2D以匹配我们的Linear层期望
        // PyTorch: (5,) -> Our framework: (1,5)
        // PyTorch: (2,) -> Our framework: (1,2)
        Tensor l1_bias_2d = backend->empty(Shape(1, 5), DType::FP32);
        Tensor l2_bias_2d = backend->empty(Shape(1, 2), DType::FP32);

        // 将PyTorch的1D偏置复制到2D张量
        for (int64_t i = 0; i < l1_bias_1d.numel(); ++i) {
            float bias_value = cpu_backend->get_item_fp32(l1_bias_1d, i);
            cpu_backend->set_item_fp32(l1_bias_2d, i, bias_value);  // 第0行第i列
        }

        for (int64_t i = 0; i < l2_bias_1d.numel(); ++i) {
            float bias_value = cpu_backend->get_item_fp32(l2_bias_1d, i);
            cpu_backend->set_item_fp32(l2_bias_2d, i, bias_value);  // 第0行第i列
        }

        // 设置权重到模型
        auto* linear1 = dynamic_cast<Linear*>(model->get_module(0).get());
        auto* linear2 = dynamic_cast<Linear*>(model->get_module(2).get());

        linear1->get_parameter("weight") = l1_weight;
        linear1->get_parameter("bias") = l1_bias_2d;
        linear2->get_parameter("weight") = l2_weight;
        linear2->get_parameter("bias") = l2_bias_2d;

        std::cout << "Initial weights loaded successfully" << std::endl;

        // 4. 创建SGD优化器（启用Nesterov动量）
        std::cout << "\n4. Creating SGD optimizer with Nesterov momentum..." << std::endl;
        auto optimizer = std::make_unique<SGD>(0.1f, 0.9f, 1e-4f, true, backend);
        optimizer->initialize(*model);

        // 5. 创建交叉熵损失函数
        std::cout << "\n5. Creating CrossEntropyLoss..." << std::endl;
        auto loss_fn = std::make_unique<CrossEntropyLoss>(backend);

        // 6. 训练循环 - 两个batch
        std::cout << "\n6. Starting training loop..." << std::endl;

        int total_tests = 0;
        int passed_tests = 0;

        for (int batch_idx = 0; batch_idx < 2; ++batch_idx) {
            std::cout << "\n=== Batch " << (batch_idx + 1) << " ===" << std::endl;

            // 获取当前batch数据
            Shape batch_shape({1, 4});  // batch_size=1, features=4
            Tensor batch_input = backend->empty(batch_shape, DType::FP32);

            // 从数据中提取当前batch
            auto* cpu_backend_ptr = dynamic_cast<CpuBackend*>(backend.get());
            for (int i = 0; i < 4; ++i) {
                float value = cpu_backend_ptr->get_item_fp32(data, batch_idx * 4 + i);
                cpu_backend_ptr->set_item_fp32(batch_input, i, value);
            }

            Shape label_shape({1, 2});  // batch_size=1, classes=2
            Tensor batch_label = backend->empty(label_shape, DType::FP32);
            for (int i = 0; i < 2; ++i) {
                float value = cpu_backend_ptr->get_item_fp32(labels, batch_idx * 2 + i);
                cpu_backend_ptr->set_item_fp32(batch_label, i, value);
            }

            std::cout << "Input: ";
            print_tensor_info("", batch_input, true);
            std::cout << "Target: ";
            print_tensor_info("", batch_label, true);

            // 清零梯度
            optimizer->zero_grad(*model);

            // 前向传播
            Tensor logits = model->forward(batch_input);

            // 比较logits
            Tensor pytorch_logits = IMPORT_TENSOR(("R:/tech-renaissance/python/module/batch" + std::to_string(batch_idx + 1) + "_logits.tsr").c_str());

            std::cout << "\n--- Comparing Logits ---" << std::endl;
            std::cout << "Our logits:" << std::endl;
            logits.print("Ours");
            std::cout << "PyTorch logits:" << std::endl;
            pytorch_logits.print("PyTorch");

            total_tests++;
            if (cpu_backend->is_close(logits, pytorch_logits, 1e-5f)) {
                std::cout << "[PASS] Batch " << (batch_idx + 1) << " logits match!" << std::endl;
                passed_tests++;
            } else {
                std::cout << "[FAIL] Batch " << (batch_idx + 1) << " logits mismatch!" << std::endl;
            }

            // 计算损失
            float loss = loss_fn->criterion(logits, batch_label);
            std::cout << "Our loss: " << std::fixed << std::setprecision(6) << loss << std::endl;

            // 检查logits是否有梯度
            if (logits.has_grad()) {
                std::cout << "Logits has gradient after loss calculation!" << std::endl;
                std::cout << "Logits grad: ";
                print_tensor_info("", logits.grad(), true, 2);
            } else {
                std::cout << "ERROR: Logits has no gradient after loss calculation!" << std::endl;
            }

            // 反向传播：将logits的梯度传播到模型参数
            try {
                model->backward(logits.grad());
                std::cout << "Backward propagation completed successfully!" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "[ERROR] Backward propagation failed: " << e.what() << std::endl;
                return 1;
            }

            // 比较loss
            Tensor pytorch_loss_tensor = IMPORT_TENSOR(("R:/tech-renaissance/python/module/batch" + std::to_string(batch_idx + 1) + "_loss.tsr").c_str());
            float pytorch_loss = cpu_backend_ptr->get_item_fp32(pytorch_loss_tensor, 0);
            total_tests++;
            if (std::abs(loss - pytorch_loss) < 1e-5f) {
                std::cout << "[PASS] Batch " << (batch_idx + 1) << " loss matches: " << loss << " vs " << pytorch_loss << std::endl;
                passed_tests++;
            } else {
                std::cout << "[FAIL] Batch " << (batch_idx + 1) << " loss mismatch: " << loss << " vs " << pytorch_loss
                          << ", diff: " << std::abs(loss - pytorch_loss) << std::endl;
            }

            // 比较梯度（如果有的话）
            if (linear1->get_parameter("weight").has_grad()) {
                std::cout << "\n--- Comparing Gradients ---" << std::endl;

                // Layer 1 weight gradient
                std::cout << "\nLayer 1 Weight Gradient:" << std::endl;
                Tensor our_l1_w_grad = linear1->get_parameter("weight").grad();
                Tensor pytorch_l1_w_grad = IMPORT_TENSOR(("R:/tech-renaissance/python/module/batch" + std::to_string(batch_idx + 1) + "_l1_weight_grad.tsr").c_str());

                our_l1_w_grad.print("Ours");
                pytorch_l1_w_grad.print("PyTorch");

                total_tests++;
                if (cpu_backend->is_close(our_l1_w_grad, pytorch_l1_w_grad, 1e-5f)) {
                    std::cout << "[PASS] Batch " << (batch_idx + 1) << " L1 weight gradients match!" << std::endl;
                    passed_tests++;
                } else {
                    std::cout << "[FAIL] Batch " << (batch_idx + 1) << " L1 weight gradients mismatch!" << std::endl;
                }

                // Layer 1 bias gradient
                std::cout << "\nLayer 1 Bias Gradient:" << std::endl;
                Tensor our_l1_b_grad = linear1->get_parameter("bias").grad();
                Tensor pytorch_l1_b_grad_1d = IMPORT_TENSOR(("R:/tech-renaissance/python/module/batch" + std::to_string(batch_idx + 1) + "_l1_bias_grad.tsr").c_str());
                Tensor pytorch_l1_b_grad_2d = backend->reshape(pytorch_l1_b_grad_1d, Shape(1, 5));

                our_l1_b_grad.print("Ours");
                pytorch_l1_b_grad_2d.print("PyTorch");

                total_tests++;
                if (cpu_backend->is_close(our_l1_b_grad, pytorch_l1_b_grad_2d, 1e-5f)) {
                    std::cout << "[PASS] Batch " << (batch_idx + 1) << " L1 bias gradients match!" << std::endl;
                    passed_tests++;
                } else {
                    std::cout << "[FAIL] Batch " << (batch_idx + 1) << " L1 bias gradients mismatch!" << std::endl;
                }

                // Layer 2 weight gradient
                std::cout << "\nLayer 2 Weight Gradient:" << std::endl;
                Tensor our_l2_w_grad = linear2->get_parameter("weight").grad();
                Tensor pytorch_l2_w_grad = IMPORT_TENSOR(("R:/tech-renaissance/python/module/batch" + std::to_string(batch_idx + 1) + "_l2_weight_grad.tsr").c_str());

                our_l2_w_grad.print("Ours");
                pytorch_l2_w_grad.print("PyTorch");

                total_tests++;
                if (cpu_backend->is_close(our_l2_w_grad, pytorch_l2_w_grad, 1e-5f)) {
                    std::cout << "[PASS] Batch " << (batch_idx + 1) << " L2 weight gradients match!" << std::endl;
                    passed_tests++;
                } else {
                    std::cout << "[FAIL] Batch " << (batch_idx + 1) << " L2 weight gradients mismatch!" << std::endl;
                }

                // Layer 2 bias gradient
                std::cout << "\nLayer 2 Bias Gradient:" << std::endl;
                Tensor our_l2_b_grad = linear2->get_parameter("bias").grad();
                Tensor pytorch_l2_b_grad_1d = IMPORT_TENSOR(("R:/tech-renaissance/python/module/batch" + std::to_string(batch_idx + 1) + "_l2_bias_grad.tsr").c_str());
                Tensor pytorch_l2_b_grad_2d = backend->reshape(pytorch_l2_b_grad_1d, Shape(1, 2));

                our_l2_b_grad.print("Ours");
                pytorch_l2_b_grad_2d.print("PyTorch");

                total_tests++;
                if (cpu_backend->is_close(our_l2_b_grad, pytorch_l2_b_grad_2d, 1e-5f)) {
                    std::cout << "[PASS] Batch " << (batch_idx + 1) << " L2 bias gradients match!" << std::endl;
                    passed_tests++;
                } else {
                    std::cout << "[FAIL] Batch " << (batch_idx + 1) << " L2 bias gradients mismatch!" << std::endl;
                }
            }

            // 更新参数
            std::cout << "\nUpdating parameters..." << std::endl;
            try {
                optimizer->step(*model);
                std::cout << "Parameter update completed successfully!" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "[ERROR] Parameter update failed: " << e.what() << std::endl;
                return 1;
            }

            // 比较更新后的权重
            std::cout << "\n--- Comparing Updated Weights ---" << std::endl;

            // Layer 1 updated weight
            std::cout << "\nLayer 1 Updated Weight:" << std::endl;
            Tensor pytorch_l1_weight_updated = IMPORT_TENSOR(("R:/tech-renaissance/python/module/batch" + std::to_string(batch_idx + 1) + "_l1_weight.tsr").c_str());

            linear1->get_parameter("weight").print("Ours");
            pytorch_l1_weight_updated.print("PyTorch");

            total_tests++;
            if (cpu_backend->is_close(linear1->get_parameter("weight"), pytorch_l1_weight_updated, 1e-5f)) {
                std::cout << "[PASS] Batch " << (batch_idx + 1) << " updated L1 weight matches!" << std::endl;
                passed_tests++;
            } else {
                std::cout << "[FAIL] Batch " << (batch_idx + 1) << " updated L1 weight mismatch!" << std::endl;
            }

            // Layer 1 updated bias
            std::cout << "\nLayer 1 Updated Bias:" << std::endl;
            Tensor pytorch_l1_bias_updated_1d = IMPORT_TENSOR(("R:/tech-renaissance/python/module/batch" + std::to_string(batch_idx + 1) + "_l1_bias.tsr").c_str());
            Tensor pytorch_l1_bias_updated_2d = backend->reshape(pytorch_l1_bias_updated_1d, Shape(1, 5));

            linear1->get_parameter("bias").print("Ours");
            pytorch_l1_bias_updated_2d.print("PyTorch");

            total_tests++;
            if (cpu_backend->is_close(linear1->get_parameter("bias"), pytorch_l1_bias_updated_2d, 1e-5f)) {
                std::cout << "[PASS] Batch " << (batch_idx + 1) << " updated L1 bias matches!" << std::endl;
                passed_tests++;
            } else {
                std::cout << "[FAIL] Batch " << (batch_idx + 1) << " updated L1 bias mismatch!" << std::endl;
            }

            // Layer 2 updated weight
            std::cout << "\nLayer 2 Updated Weight:" << std::endl;
            Tensor pytorch_l2_weight_updated = IMPORT_TENSOR(("R:/tech-renaissance/python/module/batch" + std::to_string(batch_idx + 1) + "_l2_weight.tsr").c_str());

            linear2->get_parameter("weight").print("Ours");
            pytorch_l2_weight_updated.print("PyTorch");

            total_tests++;
            if (cpu_backend->is_close(linear2->get_parameter("weight"), pytorch_l2_weight_updated, 1e-5f)) {
                std::cout << "[PASS] Batch " << (batch_idx + 1) << " updated L2 weight matches!" << std::endl;
                passed_tests++;
            } else {
                std::cout << "[FAIL] Batch " << (batch_idx + 1) << " updated L2 weight mismatch!" << std::endl;
            }

            // Layer 2 updated bias
            std::cout << "\nLayer 2 Updated Bias:" << std::endl;
            Tensor pytorch_l2_bias_updated_1d = IMPORT_TENSOR(("R:/tech-renaissance/python/module/batch" + std::to_string(batch_idx + 1) + "_l2_bias.tsr").c_str());
            Tensor pytorch_l2_bias_updated_2d = backend->reshape(pytorch_l2_bias_updated_1d, Shape(1, 2));

            linear2->get_parameter("bias").print("Ours");
            pytorch_l2_bias_updated_2d.print("PyTorch");

            total_tests++;
            if (cpu_backend->is_close(linear2->get_parameter("bias"), pytorch_l2_bias_updated_2d, 1e-5f)) {
                std::cout << "[PASS] Batch " << (batch_idx + 1) << " updated L2 bias matches!" << std::endl;
                passed_tests++;
            } else {
                std::cout << "[FAIL] Batch " << (batch_idx + 1) << " updated L2 bias mismatch!" << std::endl;
            }

            std::cout << "Batch " << (batch_idx + 1) << " completed" << std::endl;
        }

        // 7. 总结测试结果
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << total_tests << std::endl;
        std::cout << "Passed tests: " << passed_tests << std::endl;
        std::cout << "Failed tests: " << (total_tests - passed_tests) << std::endl;
        std::cout << "Success rate: " << std::fixed << std::setprecision(1)
                  << (100.0 * passed_tests / total_tests) << "%" << std::endl;

        if (passed_tests == total_tests) {
            std::cout << "\n[SUCCESS] All tests PASSED! Our framework is perfectly aligned with PyTorch!" << std::endl;
            return 0;
        } else {
            std::cout << "\n[FAILURE] Some tests FAILED! There are alignment issues with PyTorch." << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
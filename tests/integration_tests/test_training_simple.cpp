/**
 * @file test_training_simple.cpp
 * @brief 简化的训练对齐测试 - 逐步调试版本
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

int main() {
    std::cout << "=== Simple PyTorch Training Alignment Test ===" << std::endl;
    std::cout << "Testing basic tensor import and model creation" << std::endl;
    std::cout << "============================================" << std::endl;

    try {
        auto backend = BackendManager::instance().get_cpu_backend();
        cpu_backend = dynamic_cast<CpuBackend*>(backend.get());

        // 1. 导入数据和标签
        std::cout << "\n1. Loading dataset..." << std::endl;
        Tensor data = IMPORT_TENSOR("R:/tech-renaissance/python/module/data.tsr");
        Tensor labels = IMPORT_TENSOR("R:/tech-renaissance/python/module/labels.tsr");
        print_tensor_info("Data", data, true);
        print_tensor_info("Labels", labels, true);

        // 2. 导入PyTorch初始化的权重
        std::cout << "\n2. Loading PyTorch initial weights..." << std::endl;
        Tensor l1_weight = IMPORT_TENSOR("R:/tech-renaissance/python/module/init_l1_weight.tsr");
        Tensor l1_bias = IMPORT_TENSOR("R:/tech-renaissance/python/module/init_l1_bias.tsr");
        Tensor l2_weight = IMPORT_TENSOR("R:/tech-renaissance/python/module/init_l2_weight.tsr");
        Tensor l2_bias = IMPORT_TENSOR("R:/tech-renaissance/python/module/init_l2_bias.tsr");

        print_tensor_info("L1 weight", l1_weight);
        print_tensor_info("L1 bias", l1_bias);
        print_tensor_info("L2 weight", l2_weight);
        print_tensor_info("L2 bias", l2_bias);

        // 3. 创建Linear层并测试
        std::cout << "\n3. Testing Linear layers..." << std::endl;

        // 创建Linear层
        auto linear1 = std::make_shared<Linear>(4, 5, "Linear1", true);
        auto linear2 = std::make_shared<Linear>(5, 2, "Linear2", true);

        // 设置后端
        linear1->set_backend(backend);
        linear2->set_backend(backend);

        // 设置权重
        linear1->get_parameter("weight") = l1_weight;
        linear2->get_parameter("weight") = l2_weight;

        // 导入的PyTorch偏置是1D，需要reshape成2D以匹配我们的Linear层期望
        // PyTorch: (5,) -> Our framework: (1,5)
        // PyTorch: (2,) -> Our framework: (1,2)

        // 创建2D偏置张量
        Tensor l1_bias_2d = backend->empty(Shape(1, 5), DType::FP32);
        Tensor l2_bias_2d = backend->empty(Shape(1, 2), DType::FP32);

        // 将PyTorch的1D偏置复制到2D张量
        for (int64_t i = 0; i < l1_bias.numel(); ++i) {
            float bias_value = cpu_backend->get_item_fp32(l1_bias, i);
            cpu_backend->set_item_fp32(l1_bias_2d, i, bias_value);  // 第0行第i列
        }

        for (int64_t i = 0; i < l2_bias.numel(); ++i) {
            float bias_value = cpu_backend->get_item_fp32(l2_bias, i);
            cpu_backend->set_item_fp32(l2_bias_2d, i, bias_value);  // 第0行第i列
        }

        // 设置2D偏置
        linear1->get_parameter("bias") = l1_bias_2d;
        linear2->get_parameter("bias") = l2_bias_2d;

        print_tensor_info("L1 bias (reshaped to 2D)", l1_bias_2d);
        print_tensor_info("L2 bias (reshaped to 2D)", l2_bias_2d);

        std::cout << "Linear layers created and weights set successfully" << std::endl;

        // 4. 创建输入张量
        std::cout << "\n4. Creating input tensors..." << std::endl;
        Shape input_shape({1, 4});
        Tensor input = backend->ones(input_shape, DType::FP32);

        // 设置实际的数据值
        cpu_backend->set_item_fp32(input, 0, 0.5f);
        cpu_backend->set_item_fp32(input, 1, -0.3f);
        cpu_backend->set_item_fp32(input, 2, 0.8f);
        cpu_backend->set_item_fp32(input, 3, -0.1f);

        print_tensor_info("Input", input, true);

        // 5. 测试前向传播
        std::cout << "\n5. Testing forward pass..." << std::endl;

        // Linear 1
        Tensor l1_output = backend->empty(Shape(1, 5), DType::FP32);
        linear1->forward_into(input, l1_output);
        print_tensor_info("Linear1 output", l1_output, true);

        // Tanh
        Tensor tanh_output = backend->empty(Shape(1, 5), DType::FP32);
        backend->tanh_into(l1_output, tanh_output);
        print_tensor_info("Tanh output", tanh_output, true);

        // Linear 2
        Tensor l2_output = backend->empty(Shape(1, 2), DType::FP32);
        linear2->forward_into(tanh_output, l2_output);
        print_tensor_info("Linear2 output (logits)", l2_output, true);

        // 6. 比较与PyTorch的logits
        std::cout << "\n6. Comparing with PyTorch logits..." << std::endl;
        Tensor pytorch_logits = IMPORT_TENSOR("R:/tech-renaissance/python/module/batch1_logits.tsr");
        print_tensor_info("PyTorch logits", pytorch_logits, true);

        auto* cpu_backend = dynamic_cast<CpuBackend*>(BackendManager::instance().get_backend(CPU).get());
        if (cpu_backend->is_close(l2_output, pytorch_logits, 1e-5f)) {
            std::cout << "[PASS] Forward pass logits match PyTorch!" << std::endl;
        } else {
            std::cout << "[FAIL] Forward pass logits mismatch with PyTorch" << std::endl;

            // 计算相对误差
            double rel_error = cpu_backend->get_mean_rel_err(l2_output, pytorch_logits);
            std::cout << "  Mean relative error: " << std::scientific << rel_error << std::endl;

            std::cout << "  Our logits:     ";
            print_tensor_info("", l2_output, true, 2);
            std::cout << "  PyTorch logits: ";
            print_tensor_info("", pytorch_logits, true, 2);

            return 1;
        }

        std::cout << "\n[SUCCESS] Basic forward pass alignment verified!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
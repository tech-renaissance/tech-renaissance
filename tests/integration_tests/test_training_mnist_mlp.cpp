/**
 * @file test_training_mnist_mlp.cpp
 * @brief MNIST MLP完整训练测试
 * @details 使用TSR格式MNIST数据集训练3层MLP，输出训练过程的Accuracy和Loss变化
 * @version 1.57.0
 * @date 2025-11-21
 * @author 技术觉醒团队
 */

#include "tech_renaissance.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

using namespace tr;

// Training parameters (aligned with Python version)
const int BATCH_SIZE = 100;
const int NUM_EPOCHS = 5;  // 为了测试速度，减少到5个epoch
const float LEARNING_RATE = 0.1f;
const int PRINT_INTERVAL = 100;

// MNIST数据路径
const std::string MNIST_PATH = "R:/tech-renaissance/python/dataset/";

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

// 辅助函数：计算准确率
float calculate_accuracy(const Tensor& logits, const Tensor& labels) {
    auto cpu_backend = dynamic_cast<CpuBackend*>(BackendManager::instance().get_backend(CPU).get());

    int batch_size = logits.shape().dim(0);
    int num_classes = logits.shape().dim(1);

    int correct = 0;
    for (int i = 0; i < batch_size; ++i) {
        // 找到预测的最大值位置
        float max_logit = -1e9f;
        int predicted_class = 0;
        for (int j = 0; j < num_classes; ++j) {
            float logit_val = cpu_backend->get_item_fp32(logits, i * num_classes + j);
            if (logit_val > max_logit) {
                max_logit = logit_val;
                predicted_class = j;
            }
        }

        // 对于one-hot标签，找到值为1的位置
        int true_class = 0;
        for (int j = 0; j < num_classes; ++j) {
            float label_val = cpu_backend->get_item_fp32(labels, i * num_classes + j);
            if (label_val > 0.5f) {  // 找到one-hot编码中的1
                true_class = j;
                break;
            }
        }

        if (predicted_class == true_class) {
            correct++;
        }
    }

    return 100.0f * correct / batch_size;
}

// 创建MLP模型
std::shared_ptr<Model> create_mlp_model(std::shared_ptr<Backend> backend) {
    std::cout << "Creating MLP model..." << std::endl;

    auto model = Model::create("MNIST_MLP",
        std::make_shared<Flatten>(),              // flatten: (N,1,28,28) -> (N,784)
        std::make_shared<Linear>(784, 512),      // fc1: 784 -> 512
        std::make_shared<Tanh>(),                // tanh1
        std::make_shared<Linear>(512, 256),      // fc2: 512 -> 256
        std::make_shared<Tanh>(),                // tanh2
        std::make_shared<Linear>(256, 10)        // fc3: 256 -> 10
    );

    model->set_backend(backend);
    model->train();

    // 简单打印模型信息
    std::cout << "Model: MNIST_MLP (3-layer MLP with Tanh + Flatten)" << std::endl;
    std::cout << "Architecture: (N,1,28,28) -> Flatten -> (N,784) -> 512 -> 256 -> 10" << std::endl;

    return model;
}

// 加载MNIST数据集
std::pair<Tensor, Tensor> load_mnist_data(const std::string& split, std::shared_ptr<Backend> backend) {
    std::cout << "Loading MNIST " << split << " data..." << std::endl;

    std::string images_path = MNIST_PATH + split + "_images.tsr";
    std::string labels_path = MNIST_PATH + split + "_labels.tsr";

    // 使用IMPORT_TENSOR宏加载TSR文件
    Tensor images = IMPORT_TENSOR(images_path);
    Tensor labels = IMPORT_TENSOR(labels_path);

    auto cpu_backend = dynamic_cast<CpuBackend*>(backend.get());

    // 处理图像数据：从4D INT32转为4D FP32，然后归一化
    std::cout << "Processing image data..." << std::endl;
    Shape original_shape = images.shape();
    std::cout << "Original image shape: " << original_shape.to_string() << std::endl;

    // 创建4D FP32张量：(N, 1, 28, 28)
    Tensor fp32_images = backend->empty(original_shape, DType::FP32);

    // 转换数据：如果原数据是INT32才需要归一化，否则直接使用FP32值
    int64_t total_elements = original_shape.numel();
    for (int64_t i = 0; i < total_elements; ++i) {
        float pixel_val;
        if (images.dtype() == DType::INT32) {
            int32_t int_val = cpu_backend->get_item_int32(images, i);
            pixel_val = static_cast<float>(int_val) / 255.0f;  // 只有INT32数据才需要归一化
        } else {
            // 如果原数据已经是FP32，说明已经归一化过了，直接使用
            pixel_val = cpu_backend->get_item_fp32(images, i);
        }
        cpu_backend->set_item_fp32(fp32_images, i, pixel_val);
    }

    // 处理标签数据：转换为one-hot编码的2D FP32张量
    std::cout << "Processing label data..." << std::endl;
    Shape label_shape = labels.shape();
    Shape onehot_shape({label_shape.dim(0), 10});  // 10 classes
    Tensor onehot_labels = backend->empty(onehot_shape, DType::FP32);

    // 初始化为0
    cpu_backend->fill(onehot_labels, 0.0f);

    // 转换为one-hot编码
    for (int64_t i = 0; i < label_shape.dim(0); ++i) {
        int label_class;
        if (labels.dtype() == DType::INT32) {
            int32_t int_val = cpu_backend->get_item_int32(labels, i);
            label_class = static_cast<int>(int_val);
        } else {
            float float_val = cpu_backend->get_item_fp32(labels, i);
            label_class = static_cast<int>(float_val);
        }

        if (label_class >= 0 && label_class < 10) {
            cpu_backend->set_item_fp32(onehot_labels, i * 10 + label_class, 1.0f);
        }
    }

    print_tensor_info(split + " processed images", fp32_images);
    print_tensor_info(split + " processed labels (one-hot)", onehot_labels);

    return std::make_pair(fp32_images, onehot_labels);
}

// 数据批次生成器
class BatchGenerator {
private:
    Tensor images_;
    Tensor labels_;
    int batch_size_;
    int num_samples_;
    int current_idx_;
    std::shared_ptr<Backend> backend_;

public:
    BatchGenerator(const Tensor& images, const Tensor& labels, int batch_size, std::shared_ptr<Backend> backend)
        : images_(images), labels_(labels), batch_size_(batch_size), current_idx_(0), backend_(backend) {
        num_samples_ = images.shape().dim(0);
    }

    bool has_next() const {
        return current_idx_ < num_samples_;
    }

    std::pair<Tensor, Tensor> next_batch() {
        if (!has_next()) {
            throw TRException("[BatchGenerator] No more batches available");
        }

        int remaining = num_samples_ - current_idx_;
        int current_batch_size = std::min(batch_size_, remaining);

        // 创建批次张量 - 图像保持4D，标签保持2D
        // 注意：使用固定批次大小来避免形状不匹配问题
        Shape image_batch_shape({batch_size_, images_.shape().dim(1), images_.shape().dim(2), images_.shape().dim(3)});
        Shape label_batch_shape({batch_size_, labels_.shape().dim(1)});
        Tensor batch_images = backend_->empty(image_batch_shape, DType::FP32);
        Tensor batch_labels = backend_->empty(label_batch_shape, DType::FP32);

        // 将批次张量初始化为0
        auto cpu_backend = dynamic_cast<CpuBackend*>(backend_.get());
        cpu_backend->fill(batch_images, 0.0f);
        cpu_backend->fill(batch_labels, 0.0f);

        // 提取批次数据
        for (int i = 0; i < current_batch_size; ++i) {
            int src_idx = current_idx_ + i;
            auto cpu_backend = dynamic_cast<CpuBackend*>(backend_.get());

            // 复制4D图像数据
            int64_t image_size = images_.shape().dim(1) * images_.shape().dim(2) * images_.shape().dim(3);
            for (int64_t j = 0; j < image_size; ++j) {
                float val = cpu_backend->get_item_fp32(images_, src_idx * image_size + j);
                cpu_backend->set_item_fp32(batch_images, i * image_size + j, val);
            }

            // 复制2D标签数据（one-hot向量）
            for (int j = 0; j < labels_.shape().dim(1); ++j) {
                float label_val = cpu_backend->get_item_fp32(labels_, src_idx * labels_.shape().dim(1) + j);
                cpu_backend->set_item_fp32(batch_labels, i * labels_.shape().dim(1) + j, label_val);
            }
        }

        current_idx_ += current_batch_size;

        return std::make_pair(batch_images, batch_labels);
    }

    void reset() {
        current_idx_ = 0;
    }

    int get_num_batches() const {
        return (num_samples_ + batch_size_ - 1) / batch_size_;
    }
};

int main() {
    std::cout << "=== MNIST MLP Training Test ===" << std::endl;
    std::cout << "Training 3-layer MLP on MNIST dataset" << std::endl;
    std::cout << "Architecture: 784 -> 512 -> 256 -> 10 (with Tanh)" << std::endl;
    std::cout << "======================================" << std::endl;

    // 设置安静模式
    Logger::get_instance().set_quiet_mode(true);

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        // 1. 获取CPU后端
        auto backend = BackendManager::instance().get_cpu_backend();
        auto cpu_backend = dynamic_cast<CpuBackend*>(backend.get());

        // 2. 加载数据
        auto [train_images, train_labels] = load_mnist_data("train", backend);
        auto [test_images, test_labels] = load_mnist_data("test", backend);

        // 3. 创建模型
        auto model = create_mlp_model(backend);

        // 4. 创建优化器和损失函数
        auto optimizer = std::make_unique<SGD>(LEARNING_RATE, 0.0f, 0.0f);  // 简化SGD，无momentum和weight_decay
        optimizer->initialize(*model);

        auto loss_fn = std::make_unique<CrossEntropyLoss>(backend);

        // 调试：检查初始权重和梯度情况
        std::cout << "=== Debug Info: Initial Model Parameters ===" << std::endl;
        auto model_params = model->trainable_parameters();
        for (size_t i = 0; i < model_params.size(); ++i) {
            std::cout << "Parameter " << i << " - Shape: " << model_params[i]->shape().to_string()
                      << ", Has grad: " << (model_params[i]->has_grad() ? "true" : "false") << std::endl;

            // 检查权重是否都是0
            auto* cpu_backend_ptr = dynamic_cast<CpuBackend*>(backend.get());
            float sum = 0.0f;
            int64_t count = model_params[i]->numel();
            for (int64_t j = 0; j < std::min(count, static_cast<int64_t>(10)); ++j) {
                sum += cpu_backend_ptr->get_item_fp32(*model_params[i], j);
            }
            std::cout << "  First 10 values sum: " << sum << std::endl;
        }

        std::cout << "\nTraining started..." << std::endl;
        std::cout << "Batch size: " << BATCH_SIZE << std::endl;
        std::cout << "Training samples: " << train_images.shape().dim(0) << std::endl;
        std::cout << "Test samples: " << test_images.shape().dim(0) << std::endl;

        // 5. 创建数据生成器
        BatchGenerator train_loader(train_images, train_labels, BATCH_SIZE, backend);
        BatchGenerator test_loader(test_images, test_labels, BATCH_SIZE, backend);

        std::cout << "Training batches per epoch: " << train_loader.get_num_batches() << std::endl;
        std::cout << "======================================" << std::endl;

        // 6. 训练循环
        for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
            std::cout << "\n--- Epoch " << (epoch + 1) << "/" << NUM_EPOCHS << " ---" << std::endl;

            model->train();
            train_loader.reset();

            float epoch_loss = 0.0f;
            float epoch_accuracy = 0.0f;
            int num_batches = 0;

            int batch_idx = 0;
            while (train_loader.has_next()) {
                auto [batch_images, batch_labels] = train_loader.next_batch();

                // 训练步骤
                optimizer->zero_grad(*model);
                auto output = model->forward(batch_images);

                loss_fn->train();
                float batch_loss = loss_fn->criterion(output, batch_labels);

                // 计算准确率
                float batch_acc = calculate_accuracy(output, batch_labels);

                // 反向传播：损失函数会自动在output上创建梯度
                model->backward(output.grad());

                // 更新参数
                optimizer->step(*model);

                epoch_loss += batch_loss;
                epoch_accuracy += batch_acc;
                num_batches++;

                // 打印进度
                if (batch_idx % PRINT_INTERVAL == 0) {
                    std::cout << "Batch " << batch_idx << "/" << train_loader.get_num_batches()
                              << " - Loss: " << std::fixed << std::setprecision(4) << batch_loss
                              << ", Acc: " << std::setprecision(2) << batch_acc << "%" << std::endl;
                }

                batch_idx++;
            }

            // 计算epoch平均指标
            float avg_loss = epoch_loss / num_batches;
            float avg_accuracy = epoch_accuracy / num_batches;

            std::cout << "Epoch " << (epoch + 1) << " Summary:" << std::endl;
            std::cout << "  Average Loss: " << std::fixed << std::setprecision(4) << avg_loss << std::endl;
            std::cout << "  Average Accuracy: " << std::setprecision(2) << avg_accuracy << "%" << std::endl;

            // 评估
            std::cout << "Evaluating on test set..." << std::endl;
            model->eval();
            test_loader.reset();

            float test_loss = 0.0f;
            float test_accuracy = 0.0f;
            int test_num_batches = 0;

            while (test_loader.has_next()) {
                auto [batch_images, batch_labels] = test_loader.next_batch();

                // 评估步骤
                auto output = model->forward(batch_images);

                loss_fn->eval();
                float batch_loss = loss_fn->criterion(output, batch_labels);
                float batch_acc = calculate_accuracy(output, batch_labels);

                test_loss += batch_loss;
                test_accuracy += batch_acc;
                test_num_batches++;
            }

            float avg_test_loss = test_loss / test_num_batches;
            float avg_test_accuracy = test_accuracy / test_num_batches;

            std::cout << "Test Results:" << std::endl;
            std::cout << "  Test Loss: " << std::fixed << std::setprecision(4) << avg_test_loss << std::endl;
            std::cout << "  Test Accuracy: " << std::setprecision(2) << avg_test_accuracy << "%" << std::endl;
            std::cout << "======================================" << std::endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        std::cout << "\nTraining completed successfully!" << std::endl;
        std::cout << "Total training time: " << duration.count() << " seconds" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
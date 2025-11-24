/**
 * @file test_trainer_sgd.cpp
 * @brief Trainer类MNIST MLP训练测试 - 使用SGD优化器
 * @details 使用MnistLoader封装数据加载，使用SGD+余弦退火调度器
 * @version 1.60.0
 * @date 2025年11月21日
 * @author 技术觉醒团队
 */

#include "tech_renaissance.h"
#include <iostream>
#include <iomanip>
#include <utility>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

using namespace tr;

// 保存训练结果到日志文件的辅助函数
void save_training_results(const std::string& log_file, float best_accuracy, int best_epoch, long training_time) {
    std::ofstream log(log_file, std::ios::app);  // 以追加模式打开文件
    if (log.is_open()) {
        log << std::fixed << std::setprecision(2) << best_accuracy << "%" << std::endl;
        log << training_time << std::endl;
        log << "----------------------------------------" << std::endl;
        log.close();
    } else {
        std::cerr << "Error: Could not open " << log_file << " for writing" << std::endl;
    }
}

// SGD训练参数（匹配Python MLP配置）
const int BATCH_SIZE = 128;          // 匹配Python BATCH_SIZE = 128
const int NUM_EPOCHS = 20;           // 匹配Python NUM_EPOCHS = 20
const float LEARNING_RATE = 0.1f;    // 匹配Python LEARNING_RATE = 0.1
const float MOMENTUM = 0.0f;         // Python版本未使用动量
const float WEIGHT_DECAY = 0.0f;     // Python版本未使用权重衰减
const bool NESTEROV = false;         // Python版本未使用Nesterov动量
const float LABEL_SMOOTHING = 0.0f;
const int PRINT_INTERVAL = 100;

// MNIST数据路径
const std::string MNIST_PATH = std::string(WORKSPACE_PATH) + "/../../MNIST/tsr/";

// 辅助函数：计算准确率（与原始测试保持一致）
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
            if (label_val > 0.5f) {
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

// 创建MLP模型（与原始测试保持一致）
std::shared_ptr<Model> create_mlp_model(std::shared_ptr<Backend> backend) {
    std::cout << "Creating MLP model..." << std::endl;

    auto model = Model::create_ptr("MNIST_MLP",
        std::make_shared<Flatten>(),                    // flatten: (N,1,28,28) -> (N,784)
        std::make_shared<Linear>(784, 512, "fc1", false), // fc1: 784 -> 512 (bias=False)
        std::make_shared<Tanh>(),                       // tanh1
        std::make_shared<Linear>(512, 256, "fc2", false), // fc2: 512 -> 256 (bias=False)
        std::make_shared<Tanh>(),                       // tanh2
        std::make_shared<Linear>(256, 10, "fc3", false)   // fc3: 256 -> 10 (bias=False)
    );

    model->set_backend(std::move(backend));
    model->train();

    std::cout << "Model: MNIST_MLP (3-layer MLP with Tanh + Flatten, no bias)" << std::endl;
    std::cout << "Architecture: (N,1,28,28) -> Flatten -> (N,784) -> 512 -> 256 -> 10 (all Linear layers bias=False)" << std::endl;

    return model;
}


int main() {
    std::cout << "=== MNIST MLP Training Test (SGD Optimizer) ===" << std::endl;
    std::cout << "Using Trainer with SGD + ConstantLR scheduler (matching Python MLP)" << std::endl;
    std::cout << "Training 3-layer MLP on MNIST dataset for " << NUM_EPOCHS << " epochs" << std::endl;
    std::cout << "Architecture: 784 -> 512 -> 256 -> 10 (with Tanh, no bias)" << std::endl;
    std::cout << "Learning Rate: " << LEARNING_RATE << " (matching Python), Batch Size: " << BATCH_SIZE << std::endl;
    std::cout << "================================================================" << std::endl;

    // 设置安静模式
    Logger::get_instance().set_quiet_mode(true);

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        // 1. 获取CPU后端
        auto backend = BackendManager::get_cpu_backend();

        // 2. 创建MnistLoader并加载数据
        std::cout << "\n=== Data Loading with MnistLoader ===" << std::endl;
        MnistLoader mnist_loader(backend, MNIST_PATH);
        auto [train_data, test_data] = mnist_loader.load_data();

        // 3. 创建模型
        auto model = create_mlp_model(backend);

        // 4. 创建Trainer组件（使用新的统一API）
        std::cout << "\n=== Trainer Component Setup (New Unified API) ===" << std::endl;
        auto loss_fn = CrossEntropyLoss(LABEL_SMOOTHING);
        auto optimizer = SGD(LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, NESTEROV, backend);
        auto scheduler = ConstantLR(LEARNING_RATE);
        Trainer trainer(*model, loss_fn, optimizer, scheduler);

        std::cout << "[OK] Trainer created successfully" << std::endl;
        std::cout << "[OK] New unified API: All components use shared_ptr" << std::endl;
        std::cout << "[OK] Optimizer: SGD (lr=" << LEARNING_RATE << ", momentum=" << MOMENTUM
                  << ", weight_decay=" << WEIGHT_DECAY << ", nesterov=" << (NESTEROV ? "true" : "false") << ") - Python matching" << std::endl;
        std::cout << "[OK] Loss Function: CrossEntropyLoss (label_smoothing=" << LABEL_SMOOTHING << ")" << std::endl;
        std::cout << "[OK] Scheduler: ConstantLR (fixed lr=" << LEARNING_RATE << ") - Python matching" << std::endl;
        std::cout << "[OK] Data Normalization: MNIST (mean=0.1307, std=0.3081)" << std::endl;
        std::cout << "[OK] API consistency achieved: Model, Optimizer, Loss, Scheduler all treated uniformly!" << std::endl;

        std::cout << "[OK] Optimizer automatically initialized in Trainer constructor" << std::endl;

        // 5. 创建数据生成器
        std::cout << "\n=== Data Setup ===" << std::endl;
        auto train_loader = mnist_loader.get_train_loader(BATCH_SIZE);
        auto test_loader = mnist_loader.get_test_loader(BATCH_SIZE);

        std::cout << "Batch size: " << BATCH_SIZE << std::endl;
        std::cout << "Training batches per epoch: " << train_loader->get_num_batches() << std::endl;
        std::cout << "Test batches per epoch: " << test_loader->get_num_batches() << std::endl;
        std::cout << "======================================" << std::endl;

        // 6. 使用Trainer进行训练
        std::cout << "\n=== Training with Trainer ===" << std::endl;

        // 追踪最高测试准确率
        float best_test_accuracy = 0.0f;
        int best_epoch = -1;

        for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
            std::cout << "\n--- Epoch " << (epoch + 1) << "/" << NUM_EPOCHS << " ---" << std::endl;

            // 设置为训练模式
            trainer.train();
            train_loader->reset();

            float epoch_loss = 0.0f;
            float epoch_accuracy = 0.0f;
            int num_batches = 0;

            int batch_idx = 0;
            while (train_loader->has_next()) {
                auto [batch_images, batch_labels] = train_loader->next_batch();

                // 使用Trainer的训练步骤
                float batch_loss = trainer.train_step(batch_images, batch_labels);

                // 获取模型输出计算准确率 - 使用Model缓存的输出，避免重复forward
                // train_step执行后，Model的内部输出cached_output_已经被更新
                // 通过model->logits()方法可以直接零拷贝地访问它
                float batch_acc = calculate_accuracy(model->logits(), batch_labels);

                epoch_loss += batch_loss;
                epoch_accuracy += batch_acc;
                num_batches++;

                // 打印进度
                if (batch_idx % PRINT_INTERVAL == 0) {
                    std::cout << "Batch " << batch_idx << "/" << train_loader->get_num_batches()
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

            // 更新学习率
            float current_lr = trainer.step_lr_scheduler(epoch);
            std::cout << "  Learning Rate: " << std::setprecision(6) << current_lr << std::endl;

            // 评估
            std::cout << "Evaluating on test set..." << std::endl;
            trainer.eval();
            test_loader->reset();

            float test_loss = 0.0f;
            float test_accuracy = 0.0f;
            int test_num_batches = 0;

            while (test_loader->has_next()) {
                auto [batch_images, batch_labels] = test_loader->next_batch();

                // 使用Trainer的评估步骤
                float batch_loss = trainer.eval_step(batch_images, batch_labels);

                // 获取模型输出计算准确率 - 使用Model缓存的输出，避免重复forward
                float batch_acc = calculate_accuracy(model->logits(), batch_labels);

                test_loss += batch_loss;
                test_accuracy += batch_acc;
                test_num_batches++;
            }

            float avg_test_loss = test_loss / test_num_batches;
            float avg_test_accuracy = test_accuracy / test_num_batches;

            // 更新最高测试准确率
            if (avg_test_accuracy > best_test_accuracy) {
                best_test_accuracy = avg_test_accuracy;
                best_epoch = epoch + 1;
            }

            std::cout << "Test Results:" << std::endl;
            std::cout << "  Test Loss: " << std::fixed << std::setprecision(4) << avg_test_loss << std::endl;
            std::cout << "  Test Accuracy: " << std::setprecision(2) << avg_test_accuracy << "%" << std::endl;

            // 显示当前最佳成绩
            std::cout << "  Best Test Accuracy So Far: " << std::setprecision(2)
                      << best_test_accuracy << "% (Epoch " << best_epoch << ")" << std::endl;
            std::cout << "======================================" << std::endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        std::cout << "\n=== TRAINING COMPLETED ===" << std::endl;
        std::cout << "=== BEST PERFORMANCE ===" << std::endl;
        std::cout << "Best Test Accuracy: " << std::setprecision(2) << best_test_accuracy << "% (Epoch " << best_epoch << ")" << std::endl;
        std::cout << "Total training time: " << duration.count() << " seconds" << std::endl;

        // 保存训练结果到日志文件（续写模式）
        save_training_results("log_tr_sgd.txt", best_test_accuracy, best_epoch, duration.count());

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

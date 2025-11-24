#include "tech_renaissance/task/task.h"
#include "tech_renaissance/backend/backend.h"
#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/backend/backend_manager.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <utility>

namespace tr {

Task::Task(std::shared_ptr<Model> model,
           std::shared_ptr<Dataset> dataset,
           std::shared_ptr<Trainer> trainer,
           const std::shared_ptr<Backend> &backend)
    : model_(std::move(model)), dataset_(std::move(dataset)), trainer_(std::move(trainer)),
      use_shared_ptrs_(true) {
    if (!model_ || !dataset_ || !trainer_) {
        throw std::invalid_argument("Null pointer provided to Task constructor");
    }
    if (!backend) {
        set_backend(BackendManager::get_cpu_backend());
    }
    else {
        set_backend(backend);
    }
}

// 构造函数（使用引用）
Task::Task(Model& model,
           Dataset& dataset,
           Trainer& trainer,
           const std::shared_ptr<Backend> &backend)
    : model_(&model, [](Model*) {}),
      dataset_(&dataset, [](Dataset*) {}),
      trainer_(&trainer, [](Trainer*) {}),
    use_shared_ptrs_(false) {
    if (!backend) {
        set_backend(BackendManager::get_cpu_backend());
    }
    else {
        set_backend(backend);
    }
}

Task::~Task() = default;

void Task::config(const TaskConfig& config) {
    config_ = config;
}

const TaskConfig& Task::get_config() const {
    return config_;
}

float Task::calculate_accuracy(const Tensor& logits, const Tensor& labels) {
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

    return 100.0f * static_cast<float>(correct) / static_cast<float>(batch_size);
}

int Task::countDigits(int n) {
    if (n == 0) return 1;
    int digits = 0;
    while (n != 0) {
        n /= 10;
        ++digits;
    }
    return digits;
}

// 执行训练任务 - 这是核心方法
void Task::run() {
    if (config_.basic_info & MODEL_INFO) {
        output_model_info();
    }

    if (config_.basic_info & DATASET_INFO) {
        output_dataset_info();
    }

    if (config_.basic_info & TRAINING_INFO) {
        output_training_info();
    }

    auto train_loader = dataset_->get_train_loader(config_.batch_size);
    auto test_loader = dataset_->get_test_loader(config_.batch_size);

    task_start_timing();

    for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
        double train_start_time = get_time();
        if (config_.epoch_train_info & EPOCH_INFO)
            std::cout << "\n--- Epoch " << (epoch + 1) << "/" << config_.num_epochs << " ---" << std::endl;

        if (config_.epoch_train_info & EPOCH_LR) {
            std::cout << "Learning Rate: " << std::setprecision(6) << trainer_->get_current_lr() << std::endl;
        }
        trainer_->train();
        train_loader->reset();

        float epoch_loss = 0.0f;
        float epoch_accuracy = 0.0f;
        int num_batches = 0;

        int batch_idx = 0;
        while (train_loader->has_next()) {
            auto [batch_images, batch_labels] = train_loader->next_batch();

            float batch_loss = trainer_->train_step(batch_images, batch_labels);

            float batch_acc = calculate_accuracy(model_->logits(), batch_labels);

            epoch_loss += batch_loss;
            epoch_accuracy += batch_acc;
            num_batches++;

            if (batch_idx % config_.print_interval == 0) {
                auto total_batches = train_loader->get_num_batches();
                int totalWidth = countDigits(total_batches);
                if (config_.group_train_info & BATCH_INFO)
                    std::cout << "Batch [" << std::right << std::setw(totalWidth)
                << batch_idx << "/" << total_batches << "]\t";
                if (config_.group_train_info & TRAIN_LOSS)
                    std::cout << "Loss: " << std::fixed << std::setprecision(4) << batch_loss << "\t";
                if (config_.group_train_info & TRAIN_ACCURACY)
                    std::cout << "Accuracy: " << std::setprecision(2) << batch_acc << "%";
                if (config_.group_train_info & (BATCH_INFO | TRAIN_LOSS | TRAIN_ACCURACY))
                    std::cout << std::endl;
            }

            batch_idx++;
        }

        // 计算epoch平均指标
        float avg_loss = epoch_loss / static_cast<float>(num_batches);
        float avg_accuracy = epoch_accuracy / static_cast<float>(num_batches);

        if (config_.epoch_train_info & (TRAIN_LOSS | TRAIN_ACCURACY)) {
            std::cout << "Epoch " << (epoch + 1) << " - ";
            if (config_.epoch_train_info & TRAIN_LOSS) {
                std::cout << "Train Loss: " << std::fixed << std::setprecision(4) << avg_loss << "\t";
            }
            if (config_.epoch_train_info & TRAIN_ACCURACY) {
                std::cout << "Train Accuracy: " << std::setprecision(2) << avg_accuracy << "%";
            }
            std::cout << std::endl;
        }
        // 更新学习率
        stats_.current_lr = trainer_->step_lr_scheduler(epoch);
        double train_end_time = get_time();
        stats_.total_train_time += train_end_time - train_start_time;
        double test_start_time = get_time();

        trainer_->eval();
        test_loader->reset();

        float test_loss = 0.0f;
        float test_accuracy = 0.0f;
        int test_num_batches = 0;

        while (test_loader->has_next()) {
            auto [batch_images, batch_labels] = test_loader->next_batch();

            // 使用Trainer的评估步骤
            float batch_loss = trainer_->eval_step(batch_images, batch_labels);

            // 获取模型输出计算准确率 - 使用Model缓存的输出，避免重复forward
            float batch_acc = calculate_accuracy(model_->logits(), batch_labels);

            test_loss += batch_loss;
            test_accuracy += batch_acc;
            test_num_batches++;
        }
        double test_end_time = get_time();
        stats_.total_test_time += test_end_time - test_start_time;

        float avg_test_loss = test_loss / static_cast<float>(test_num_batches);
        float avg_test_accuracy = test_accuracy / static_cast<float>(test_num_batches);

        // 更新最终准确率统计
        stats_.final_accuracy = avg_test_accuracy;

        if (config_.epoch_test_info & (TEST_LOSS | TEST_ACCURACY | BEST_TEST_ACCURACY)) {
            std::cout << "Epoch " << (epoch + 1) << " - ";
            if (config_.epoch_test_info & TEST_LOSS) output_test_loss(avg_test_loss);
            if (config_.epoch_test_info & TEST_ACCURACY) output_test_accuracy(avg_test_accuracy);
            std::cout << std::endl;
        }

        if (avg_test_accuracy > stats_.best_accuracy) {
            stats_.best_accuracy = avg_test_accuracy;
            stats_.best_epoch = epoch + 1;
            if (config_.epoch_test_info & BEST_TEST_ACCURACY) output_best_accuracy();
            if (config_.save_best_model) {
                save_model_if_needed();
            }
        }
        if (config_.epoch_test_info & BEST_TEST_ACCURACY_SO_FAR) output_best_accuracy_so_far();

        stats_.total_epochs_completed++;

        // 调用进度回调
        if (progress_callback_) {
            progress_callback_(epoch + 1, config_.num_epochs, stats_.final_accuracy);
        }
    }

    std::cout << "\n\nTask Completed!" << std::endl;

    if (config_.final_info & (TOTAL_TRAIN_TIME | TOTAL_TEST_TIME)) {
        if (config_.final_info & TOTAL_TRAIN_TIME) {
            std::cout << "Total Train Time: " << std::fixed << std::setprecision(3) << stats_.total_train_time << " s" << "\t";
        }
        if (config_.final_info & TOTAL_TEST_TIME) {
            std::cout << "Total Test Time: " << std::fixed << std::setprecision(3) << stats_.total_test_time << " s";
        }
        std::cout << std::endl;
    }

    stats_.total_time = get_time();
    if (config_.final_info & TOTAL_TIME) {
        std::cout << "Total Time: " << std::fixed << std::setprecision(3) << stats_.total_time << " s" << std::endl;
    }
    if (config_.final_info & BEST_TEST_ACCURACY) {
        std::cout << "\nBest test accuracy: " << std::fixed << std::setprecision(2)
        << stats_.best_accuracy << "%" << std::endl;
    }
    std::cout << std::endl;

    if (config_.save_info & (BEST_TEST_ACCURACY | TOTAL_TIME)) {
        save_logs_if_needed();
    }
}

// 设置早停回调函数
void Task::set_early_stopping_callback(std::function<void(int, double)> callback) {
    early_stopping_callback_ = std::move(callback);
}

// 设置进度回调函数
void Task::set_progress_callback(std::function<void(int, int, double)> callback) {
    progress_callback_ = std::move(callback);
}

const Task::TrainingStats& Task::get_training_stats() const {
    return stats_;
}

void Task::output_test_loss(float test_loss) {
    std::cout << "Test Loss:  " << std::fixed << std::setprecision(4) << test_loss << "\t";
}

void Task::output_test_accuracy(float test_accuracy) {
    std::cout << "Test Accuracy:  " << std::setprecision(2) << test_accuracy << "%";
}

void Task::output_best_accuracy() const {
    std::cout << "\nNew best test accuracy achieved: " << std::setprecision(2)
              << stats_.best_accuracy << "%" << std::endl;
}

void Task::output_best_accuracy_so_far() const {
    std::cout << "Best test accuracy so far: " << std::setprecision(2)
              << stats_.best_accuracy << "% (Epoch " << stats_.best_epoch << ")" << std::endl;
}

void Task::output_model_info() const {
    std::cout << "[INFO] Model Information:" << std::endl;
    if (model_) {
        std::cout << "  - Model: [Provided]" << std::endl;
    } else {
        std::cout << "  - Model: [Demo Mode - Not Provided]" << std::endl;
    }
    std::cout << "  - Device: " << (config_.device.is_cpu() ? "CPU" : "GPU") << std::endl;
}

void Task::output_dataset_info() const {
    std::cout << "[INFO] Dataset Information:" << std::endl;
    if (dataset_) {
        std::cout << "  - Training samples: " << dataset_->get_train_size() << std::endl;
        std::cout << "  - Test samples: " << dataset_->get_test_size() << std::endl;
    } else {
        std::cout << "  - Dataset: [Demo Mode - Not Provided]" << std::endl;
    }
}

void Task::output_training_info() const {
    std::cout << "[INFO] Training Configuration:" << std::endl;
    std::cout << "  - Epochs: " << config_.num_epochs << std::endl;
    std::cout << "  - Batch size: " << config_.batch_size << std::endl;
}

void Task::save_model_if_needed() const {
    if (model_) {
        std::cout << "New best model has been saved to: " << config_.model_save_path << std::endl;
        // TODO: 实现模型保存功能
    } else {
        std::cout << "[WARNING] No model provided for saving" << std::endl;
    }
}

void Task::save_logs_if_needed() const {
    std::ofstream log_file(config_.log_save_path);
    if (log_file.is_open()) {
        if (config_.save_info & BEST_TEST_ACCURACY) {
            log_file << "Best test accuracy: " << std::fixed << std::setprecision(2) << stats_.best_accuracy << "%\n";
        }
        if (config_.save_info & TOTAL_TIME) {
            log_file << "Total Time: " << std::fixed << std::setprecision(3) << stats_.total_time << " s\n";
        }
        log_file.close();
        std::cout << "Logs saved to: " << config_.log_save_path << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to open log file: " << config_.log_save_path << std::endl;
    }
}

void Task::task_start_timing() {
    task_start_time_ = std::chrono::high_resolution_clock::now();
}

double Task::get_time() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - task_start_time_);
    return static_cast<double>(duration.count()) / 1000.0;
}

} // namespace tr
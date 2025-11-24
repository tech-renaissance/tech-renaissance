/**
 * @file test_task.cpp
 * @brief Task类MNIST MLP训练集成测试 - 使用全新的3行Task API
 * @details 验证Task类能够将复杂的175行训练代码简化为3行
 * @version 2.2.0
 * @date 2025年11月24日
 * @author 技术觉醒团队
 */

#include <iostream>
#include <memory>
#include <string>
#include "tech_renaissance.h"

using namespace tr;

constexpr int BATCH_SIZE = 128;
constexpr int NUM_EPOCHS = 20;
constexpr float LEARNING_RATE = 0.001f;
constexpr float WEIGHT_DECAY = 0.01f;
constexpr float BETA1 = 0.9f;
constexpr float BETA2 = 0.999f;
constexpr float EPS = 1e-8f;
constexpr float LABEL_SMOOTHING = 0.0f;
constexpr int PRINT_INTERVAL = 100;
const std::string MNIST_PATH = std::string(WORKSPACE_PATH) + "/../../MNIST/tsr/";

int main() {
    Logger::get_instance().set_quiet_mode(true);

    try {
        auto backend = BackendManager::get_cpu_backend();
        auto mnist = std::make_shared<MnistDataset>(backend, MNIST_PATH);
        auto [train_data, test_data] = mnist->load_data();

        auto model = Model::create("MNIST_MLP_Task",
            std::make_shared<Flatten>(),
            std::make_shared<Linear>(784, 512),
            std::make_shared<Tanh>(),
            std::make_shared<Linear>(512, 256),
            std::make_shared<Tanh>(),
            std::make_shared<Linear>(256, 10)
        );
        model->set_backend(backend);
        model->train();

        auto loss_fn = std::make_shared<CrossEntropyLoss>(backend, LABEL_SMOOTHING);
        auto optimizer = std::make_shared<Adam>(LEARNING_RATE, BETA1, BETA2, EPS, WEIGHT_DECAY);
        auto scheduler = std::make_shared<CosineAnnealingLR>(LEARNING_RATE, NUM_EPOCHS);
        auto trainer = std::make_shared<Trainer>(model, loss_fn, optimizer, scheduler);
        auto task = std::make_shared<Task>(model, mnist, trainer);

        TaskConfig cfg;
        cfg.num_epochs = NUM_EPOCHS;
        cfg.batch_size = BATCH_SIZE;
        cfg.print_interval = 100;
        cfg.model_save_path = "task_trained_model.mdl";
        cfg.log_save_path = "task_training.txt";
        cfg.save_best_model = true;
        cfg.basic_info = MODEL_INFO | DATASET_INFO | TRAINING_INFO;
        cfg.epoch_train_info = EPOCH_INFO | EPOCH_LR | TRAIN_LOSS | TRAIN_ACCURACY;
        cfg.group_train_info = BATCH_INFO | TRAIN_LOSS | TRAIN_ACCURACY;
        cfg.epoch_test_info = TEST_LOSS | TEST_ACCURACY | BEST_TEST_ACCURACY | BEST_TEST_ACCURACY_SO_FAR;
        cfg.final_info = BEST_TEST_ACCURACY | TOTAL_TRAIN_TIME | TOTAL_TEST_TIME | TOTAL_TIME;
        cfg.save_info = BEST_TEST_ACCURACY | TOTAL_TIME;
        task->config(cfg);

        task->run();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Task integration test failed: " << e.what() << std::endl;
        return -1;
    }
}
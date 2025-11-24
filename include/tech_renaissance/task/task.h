#pragma once

#include <memory>
#include <functional>
#include <string>
#include <chrono>
#include "tech_renaissance/data/device.h"
#include "tech_renaissance/model/model.h"
#include "tech_renaissance/trainer/trainer.h"
#include "tech_renaissance/task/dataset.h"

namespace tr {

enum TaskConfigFlags : uint32_t {
    NONE_INFO                   = 0,
    MODEL_INFO                  = 1 << 0,
    DATASET_INFO                = 1 << 1,
    TRAINING_INFO               = 1 << 2,
    EPOCH_INFO                  = 1 << 3,
    EPOCH_LR                    = 1 << 4,
    BATCH_INFO                  = 1 << 5,
    TRAIN_LOSS                  = 1 << 6,
    TRAIN_ACCURACY              = 1 << 7,
    TEST_ACCURACY               = 1 << 8,
    TEST_LOSS                   = 1 << 9,
    BEST_TEST_ACCURACY          = 1 << 10,
    BEST_TEST_ACCURACY_SO_FAR   = 1 << 11,
    TOTAL_TRAIN_TIME            = 1 << 12,
    TOTAL_TEST_TIME             = 1 << 13,
    TOTAL_TIME                  = 1 << 14,
    ALL_INFO                    = 0xFFFFFFFF
};

struct TaskConfig {
    Device device = CPU;
    int num_epochs = 10;
    int batch_size = 64;
    int print_interval = 100;
    std::string model_save_path = "trained_model.mdl";
    std::string log_save_path = "training.txt";
    bool save_best_model = true;
    uint32_t basic_info = MODEL_INFO | DATASET_INFO | TRAINING_INFO;
    uint32_t epoch_train_info = EPOCH_INFO | EPOCH_LR | TRAIN_LOSS | TRAIN_ACCURACY;
    uint32_t group_train_info = BATCH_INFO | TRAIN_LOSS | TRAIN_ACCURACY;
    uint32_t epoch_test_info = TEST_LOSS | TEST_ACCURACY | BEST_TEST_ACCURACY | BEST_TEST_ACCURACY_SO_FAR;
    uint32_t final_info = BEST_TEST_ACCURACY | TOTAL_TRAIN_TIME | TOTAL_TEST_TIME | TOTAL_TIME;
    uint32_t save_info = BEST_TEST_ACCURACY | TOTAL_TIME;
};

class Task {
public:
    Task(std::shared_ptr<Model> model,
         std::shared_ptr<Dataset> dataset,
         std::shared_ptr<Trainer> trainer);

    Task(Model& model,
         Dataset& dataset,
         Trainer& trainer);

    ~Task();

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    static int countDigits(int n);
    Task(Task&&) = default;
    Task& operator=(Task&&) = default;
    void config(const TaskConfig& config);
    const TaskConfig &get_config() const;
    void run();
    void set_early_stopping_callback(std::function<void(int, double)> callback);
    void set_progress_callback(std::function<void(int, int, double)> callback);
    struct TrainingStats {
        int best_epoch = 0;
        int total_epochs_completed = 0;
        int total_batches_processed = 0;
        double final_accuracy = 0.0;
        double best_accuracy = 0.0;
        double total_train_time = 0.0;
        double total_test_time = 0.0;
        double total_time = 0.0;
        bool early_stopped = false;
        int early_stopping_epoch = 0;
        float current_lr = 0.0;
    };
    const TrainingStats& get_training_stats() const;
protected:
    std::shared_ptr<Model> model_;
    std::shared_ptr<Dataset> dataset_;
    std::shared_ptr<Trainer> trainer_;
    bool use_shared_ptrs_;
    TaskConfig config_;
    TrainingStats stats_;
    std::function<void(int, double)> early_stopping_callback_;
    std::function<void(int, int, double)> progress_callback_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point task_start_time_;
    double best_accuracy_ = 0.0;

    static float calculate_accuracy(const Tensor& logits, const Tensor& labels);
    void output_model_info() const;
    void output_dataset_info() const;
    void output_training_info() const;
    void save_model_if_needed() const;
    void save_logs_if_needed() const;
    void task_start_timing();
    double get_time() const;
    static void output_test_loss(float test_loss);
    static void output_test_accuracy(float test_accuracy);
    void output_best_accuracy() const;
    void output_best_accuracy_so_far() const;
};

} // namespace tr
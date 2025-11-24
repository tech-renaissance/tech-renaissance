#include "tech_renaissance/task/mnist_dataset.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>

namespace tr {

// MnistDataset实现 - 独立实现的MNIST数据集访问类

MnistDataset::MnistDataset(std::shared_ptr<Backend> backend,
                           const std::string& data_path,
                           float mean,
                           float std_val)
    : backend_(backend), data_path_(data_path), mean_(mean), std_(std_val), data_loaded_(false) {
}

int MnistDataset::get_train_size() const {
    return 60000;  // MNIST训练集标准大小
}

int MnistDataset::get_test_size() const {
    return 10000;  // MNIST测试集标准大小
}

const char* MnistDataset::get_name() const {
    return "MNIST";
}

Shape MnistDataset::get_input_shape() const {
    return Shape(1, 28, 28);  // 1通道28x28图像
}

Shape MnistDataset::get_output_shape() const {
    return Shape(10);         // 10类分类
}

std::pair<std::pair<Tensor, Tensor>, std::pair<Tensor, Tensor>> MnistDataset::load_data() {
    if (data_loaded_) {
        return std::make_pair(
            std::make_pair(train_images_, train_labels_),
            std::make_pair(test_images_, test_labels_)
        );
    }

    std::cout << "Loading MNIST dataset..." << std::endl;

    // 加载训练和测试数据
    auto train_data = load_and_preprocess_split("train");
    auto test_data = load_and_preprocess_split("t10k");

    train_images_ = train_data.first;
    train_labels_ = train_data.second;
    test_images_ = test_data.first;
    test_labels_ = test_data.second;

    data_loaded_ = true;

    std::cout << "[OK] MNIST dataset loaded successfully!" << std::endl;
    print_dataset_info();

    return std::make_pair(train_data, test_data);
}

std::unique_ptr<SimpleBatchGenerator> MnistDataset::get_train_loader(int batch_size) {
    if (!data_loaded_) {
        load_data();
    }

    // 创建SimpleBatchGenerator对象
    auto generator = std::make_unique<SimpleBatchGenerator>(train_images_, train_labels_, batch_size, backend_, true);
    return generator;
}

std::unique_ptr<SimpleBatchGenerator> MnistDataset::get_test_loader(int batch_size) {
    if (!data_loaded_) {
        load_data();
    }

    // 创建SimpleBatchGenerator对象
    auto generator = std::make_unique<SimpleBatchGenerator>(test_images_, test_labels_, batch_size, backend_, false);
    return generator;
}

void MnistDataset::print_dataset_info() const {
    std::cout << "=== MNIST Dataset Information ===" << std::endl;
    std::cout << "Training samples: " << get_train_size() << std::endl;
    std::cout << "Test samples: " << get_test_size() << std::endl;
    std::cout << "Input shape: " << get_input_shape().to_string() << std::endl;
    std::cout << "Output shape: " << get_output_shape().to_string() << std::endl;
    std::cout << "Mean: " << mean_ << ", Std: " << std_ << std::endl;
    std::cout << "================================" << std::endl;
}

std::pair<Tensor, Tensor> MnistDataset::load_and_preprocess_split(const std::string& split) {
    std::cout << "Loading MNIST " << split << " data..." << std::endl;

    // 构建文件路径 - 兼容不同的命名约定
    std::string images_path, labels_path;
    if (split == "t10k") {
        images_path = data_path_ + "test_images.tsr";
        labels_path = data_path_ + "test_labels.tsr";
    } else {
        images_path = data_path_ + split + "_images.tsr";
        labels_path = data_path_ + split + "_labels.tsr";
    }

    // 加载TSR文件
    Tensor raw_images = IMPORT_TENSOR(images_path);
    Tensor raw_labels = IMPORT_TENSOR(labels_path);

    std::cout << "Original image shape: " << raw_images.shape().to_string() << std::endl;
    std::cout << "Original label shape: " << raw_labels.shape().to_string() << std::endl;

    // 预处理数据
    Tensor processed_images = preprocess_images(raw_images);
    Tensor processed_labels = preprocess_labels(raw_labels);

    return std::make_pair(processed_images, processed_labels);
}

Tensor MnistDataset::preprocess_images(const Tensor& raw_images) {
    Shape original_shape = raw_images.shape();

    // 创建4D FP32张量：(N, 1, 28, 28)
    Tensor fp32_images = backend_->empty(original_shape, DType::FP32);

    auto cpu_backend = dynamic_cast<CpuBackend*>(backend_.get());

    // 转换数据并进行MNIST标准化：output = (input - mean) / std
    int64_t total_elements = original_shape.numel();
    for (int64_t i = 0; i < total_elements; ++i) {
        float pixel_val;
        if (raw_images.dtype() == DType::INT32) {
            int32_t int_val = cpu_backend->get_item_int32(raw_images, i);
            pixel_val = static_cast<float>(int_val) / 255.0f;
        } else {
            pixel_val = cpu_backend->get_item_fp32(raw_images, i);
        }
        // MNIST标准化
        float normalized_val = (pixel_val - mean_) / std_;
        cpu_backend->set_item_fp32(fp32_images, i, normalized_val);
    }

    return fp32_images;
}

Tensor MnistDataset::preprocess_labels(const Tensor& raw_labels) {
    const int NUM_CLASSES = 10;
    Shape label_shape = raw_labels.shape();
    Shape onehot_shape({label_shape.dim(0), NUM_CLASSES});
    Tensor onehot_labels = backend_->empty(onehot_shape, DType::FP32);

    auto cpu_backend = dynamic_cast<CpuBackend*>(backend_.get());
    cpu_backend->fill(onehot_labels, 0.0f);

    // 转换为one-hot编码
    for (int64_t i = 0; i < label_shape.dim(0); ++i) {
        int label_class;
        if (raw_labels.dtype() == DType::INT32) {
            int32_t int_val = cpu_backend->get_item_int32(raw_labels, i);
            label_class = static_cast<int>(int_val);
        } else {
            float float_val = cpu_backend->get_item_fp32(raw_labels, i);
            label_class = static_cast<int>(float_val);
        }

        if (label_class >= 0 && label_class < NUM_CLASSES) {
            cpu_backend->set_item_fp32(onehot_labels, i * NUM_CLASSES + label_class, 1.0f);
        } else {
            std::cerr << "[WARNING] Invalid label value: " << label_class << " at index " << i << std::endl;
        }
    }

    return onehot_labels;
}

// ===== SimpleBatchGenerator 实现 =====

SimpleBatchGenerator::SimpleBatchGenerator(const Tensor& images, const Tensor& labels, int batch_size,
                                         std::shared_ptr<Backend> backend, bool is_training_data)
    : images_(images)
    , labels_(labels)
    , batch_size_(batch_size)
    , current_idx_(0)
    , backend_(backend)
    , is_training_data_(is_training_data) {

    num_samples_ = images.shape().dim(0);
    initialize_indices();
}

bool SimpleBatchGenerator::has_next() const {
    return current_idx_ < num_samples_;
}

std::pair<Tensor, Tensor> SimpleBatchGenerator::next_batch() {
    if (!has_next()) {
        throw std::runtime_error("[SimpleBatchGenerator] No more batches available");
    }

    int remaining = num_samples_ - current_idx_;
    int current_batch_size = std::min(batch_size_, remaining);

    // 创建批次张量
    Shape image_batch_shape({batch_size_, images_.shape().dim(1), images_.shape().dim(2), images_.shape().dim(3)});
    Shape label_batch_shape({batch_size_, labels_.shape().dim(1)});
    auto cpu_backend = dynamic_cast<CpuBackend*>(backend_.get());
    Tensor batch_images = cpu_backend->zeros(image_batch_shape, DType::FP32);
    Tensor batch_labels = cpu_backend->zeros(label_batch_shape, DType::FP32);

    // 提取批次数据（使用打乱的索引）
    for (int i = 0; i < current_batch_size; ++i) {
        int src_idx = shuffled_indices_[current_idx_ + i];

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

void SimpleBatchGenerator::reset() {
    current_idx_ = 0;

    // 如果是训练数据，每个epoch开始时重新打乱
    if (is_training_data_) {
        shuffle_indices();
    }
}

int SimpleBatchGenerator::get_num_batches() const {
    return (num_samples_ + batch_size_ - 1) / batch_size_;
}

void SimpleBatchGenerator::initialize_indices() {
    shuffled_indices_.resize(num_samples_);
    for (int i = 0; i < num_samples_; ++i) {
        shuffled_indices_[i] = i;
    }

    // 如果是训练数据，初始就打乱
    if (is_training_data_) {
        shuffle_indices();
    }
}

void SimpleBatchGenerator::shuffle_indices() {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shuffled_indices_.begin(), shuffled_indices_.end(), g);
}

} // namespace tr
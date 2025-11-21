/**
 * @file mnist_loader.cpp
 * @brief MNIST数据集加载器实现
 * @details 封装MNIST数据集的加载、预处理和批次生成功能
 * @version 1.57.0
 * @date 2025-11-21
 * @author 技术觉醒团队
 */

#include "tech_renaissance/utils/mnist_loader.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/utils/logger.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace tr {

// ===== BatchGenerator 实现 =====

BatchGenerator::BatchGenerator(const Tensor& images, const Tensor& labels, int batch_size,
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

bool BatchGenerator::has_next() const {
    return current_idx_ < num_samples_;
}

std::pair<Tensor, Tensor> BatchGenerator::next_batch() {
    if (!has_next()) {
        throw TRException("[BatchGenerator] No more batches available");
    }

    int remaining = num_samples_ - current_idx_;
    int current_batch_size = std::min(batch_size_, remaining);

    // 创建批次张量
    Shape image_batch_shape({batch_size_, images_.shape().dim(1), images_.shape().dim(2), images_.shape().dim(3)});
    Shape label_batch_shape({batch_size_, labels_.shape().dim(1)});
    Tensor batch_images = backend_->empty(image_batch_shape, DType::FP32);
    Tensor batch_labels = backend_->empty(label_batch_shape, DType::FP32);

    auto cpu_backend = dynamic_cast<CpuBackend*>(backend_.get());
    cpu_backend->fill(batch_images, 0.0f);
    cpu_backend->fill(batch_labels, 0.0f);

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

void BatchGenerator::reset() {
    current_idx_ = 0;

    // 如果是训练数据，每个epoch开始时重新打乱
    if (is_training_data_) {
        shuffle_indices();
    }
}

int BatchGenerator::get_num_batches() const {
    return (num_samples_ + batch_size_ - 1) / batch_size_;
}

void BatchGenerator::initialize_indices() {
    shuffled_indices_.resize(num_samples_);
    for (int i = 0; i < num_samples_; ++i) {
        shuffled_indices_[i] = i;
    }

    // 如果是训练数据，初始就打乱
    if (is_training_data_) {
        shuffle_indices();
    }
}

void BatchGenerator::shuffle_indices() {
    // 使用当前时间作为随机种子，确保每个epoch的打乱不同
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 g(seed);

    // Fisher-Yates打乱
    for (int i = num_samples_ - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(g);
        std::swap(shuffled_indices_[i], shuffled_indices_[j]);
    }
}

// ===== MnistLoader 实现 =====

MnistLoader::MnistLoader(std::shared_ptr<Backend> backend, const std::string& data_path, float mean, float std)
    : backend_(backend)
    , data_path_(data_path)
    , mean_(mean)
    , std_(std)
    , data_loaded_(false) {

    if (!backend_) {
        throw TRException("[MnistLoader] Backend cannot be null");
    }
}

std::pair<std::pair<Tensor, Tensor>, std::pair<Tensor, Tensor>> MnistLoader::load_data() {
    std::cout << "Loading MNIST dataset..." << std::endl;

    // 加载训练和测试数据
    auto train_data = load_and_preprocess_split("train");
    auto test_data = load_and_preprocess_split("test");

    data_loaded_ = true;

    // 缓存数据
    train_images_ = train_data.first;
    train_labels_ = train_data.second;
    test_images_ = test_data.first;
    test_labels_ = test_data.second;

    std::cout << "[OK] MNIST dataset loaded successfully!" << std::endl;
    print_dataset_info();

    return std::make_pair(train_data, test_data);
}

std::unique_ptr<BatchGenerator> MnistLoader::get_train_loader(int batch_size) {
    if (!data_loaded_) {
        load_data();
    }
    return std::make_unique<BatchGenerator>(train_images_, train_labels_, batch_size, backend_, true);
}

std::unique_ptr<BatchGenerator> MnistLoader::get_test_loader(int batch_size) {
    if (!data_loaded_) {
        load_data();
    }
    return std::make_unique<BatchGenerator>(test_images_, test_labels_, batch_size, backend_, false);
}

void MnistLoader::print_dataset_info() const {
    if (!data_loaded_) {
        std::cout << "Dataset not loaded yet." << std::endl;
        return;
    }

    std::cout << "\n=== MNIST Dataset Information ===" << std::endl;
    std::cout << "Training samples: " << train_images_.shape().dim(0) << std::endl;
    std::cout << "Test samples: " << test_images_.shape().dim(0) << std::endl;
    std::cout << "Image size: " << train_images_.shape().dim(1) << "x"
              << train_images_.shape().dim(2) << "x" << train_images_.shape().dim(3) << std::endl;
    std::cout << "Number of classes: " << train_labels_.shape().dim(1) << std::endl;
    std::cout << "Normalization: mean=" << mean_ << ", std=" << std_ << std::endl;
    std::cout << "================================" << std::endl;
}

std::pair<Tensor, Tensor> MnistLoader::load_and_preprocess_split(const std::string& split) {
    std::cout << "Loading MNIST " << split << " data..." << std::endl;

    // 构建文件路径
    std::string images_path = data_path_ + split + "_images.tsr";
    std::string labels_path = data_path_ + split + "_labels.tsr";

    // 加载TSR文件
    Tensor raw_images = IMPORT_TENSOR(images_path);
    Tensor raw_labels = IMPORT_TENSOR(labels_path);

    std::cout << "Original image shape: " << raw_images.shape().to_string() << std::endl;
    std::cout << "Original label shape: " << raw_labels.shape().to_string() << std::endl;

    // 预处理数据
    Tensor processed_images = preprocess_images(raw_images);
    Tensor processed_labels = preprocess_labels(raw_labels);

    print_tensor_info(split + " processed images", processed_images);
    print_tensor_info(split + " processed labels", processed_labels);

    return std::make_pair(processed_images, processed_labels);
}

Tensor MnistLoader::preprocess_images(const Tensor& raw_images) {
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

Tensor MnistLoader::preprocess_labels(const Tensor& raw_labels) {
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
        }
    }

    return onehot_labels;
}

void MnistLoader::print_tensor_info(const std::string& name, const Tensor& tensor) const {
    std::cout << name << " - Shape: " << tensor.shape().to_string()
              << ", Dtype: " << static_cast<int>(tensor.dtype()) << std::endl;
}

} // namespace tr
/**
 * @file mnist_loader.h
 * @brief MNIST数据集加载器
 * @details 封装MNIST数据集的加载、预处理和批次生成功能
 * @version 1.57.0
 * @date 2025-11-21
 * @author 技术觉醒团队
 */

#ifndef TECH_RENAISSANCE_UTILS_MNIST_LOADER_H
#define TECH_RENAISSANCE_UTILS_MNIST_LOADER_H

#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/backend/backend.h"
#include "tech_renaissance/data/shape.h"
#include "tech_renaissance/utils/tr_exception.h"
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <chrono>

namespace tr {

/**
 * @class BatchGenerator
 * @brief MNIST数据批次生成器，支持随机打乱
 */
class BatchGenerator {
private:
    Tensor images_;
    Tensor labels_;
    int batch_size_;
    int num_samples_;
    int current_idx_;
    std::shared_ptr<Backend> backend_;
    std::vector<int> shuffled_indices_;
    bool is_training_data_;

public:
    /**
     * @brief 构造函数
     * @param images 图像数据张量
     * @param labels 标签数据张量
     * @param batch_size 批次大小
     * @param backend 后端指针
     * @param is_training_data 是否为训练数据（需要打乱）
     */
    BatchGenerator(const Tensor& images, const Tensor& labels, int batch_size,
                   std::shared_ptr<Backend> backend, bool is_training_data = true);

    /**
     * @brief 检查是否还有批次
     * @return true 如果还有批次可用
     */
    bool has_next() const;

    /**
     * @brief 获取下一个批次
     * @return pair<图像批次, 标签批次>
     */
    std::pair<Tensor, Tensor> next_batch();

    /**
     * @brief 重置批次生成器
     */
    void reset();

    /**
     * @brief 获取总批次数
     * @return 批次数
     */
    int get_num_batches() const;

private:
    void initialize_indices();
    void shuffle_indices();
};

/**
 * @class MnistLoader
 * @brief MNIST数据集加载器，封装数据加载和预处理功能
 */
class MnistLoader {
private:
    std::shared_ptr<Backend> backend_;
    std::string data_path_;
    float mean_;
    float std_;

    // 缓存加载的数据
    Tensor train_images_, train_labels_;
    Tensor test_images_, test_labels_;
    bool data_loaded_;

public:
    // MNIST标准参数
    static constexpr float DEFAULT_MEAN = 0.1307f;
    static constexpr float DEFAULT_STD = 0.3081f;
    static constexpr int IMAGE_SIZE = 28;
    static constexpr int NUM_CLASSES = 10;

    /**
     * @brief 构造函数
     * @param backend 后端指针
     * @param data_path 数据集路径
     * @param mean 标准化均值
     * @param std 标准化标准差
     */
    explicit MnistLoader(std::shared_ptr<Backend> backend,
                        const std::string& data_path = "",
                        float mean = DEFAULT_MEAN,
                        float std = DEFAULT_STD);

    /**
     * @brief 加载MNIST数据集
     * @return pair<训练数据, 测试数据>，每个pair包含图像和标签
     */
    std::pair<std::pair<Tensor, Tensor>, std::pair<Tensor, Tensor>> load_data();

    /**
     * @brief 获取训练数据的批次生成器
     * @param batch_size 批次大小
     * @return BatchGenerator智能指针
     */
    std::unique_ptr<BatchGenerator> get_train_loader(int batch_size);

    /**
     * @brief 获取测试数据的批次生成器
     * @param batch_size 批次大小
     * @return BatchGenerator智能指针
     */
    std::unique_ptr<BatchGenerator> get_test_loader(int batch_size);

    /**
     * @brief 获取数据集信息
     */
    void print_dataset_info() const;

private:
    /**
     * @brief 加载并预处理单个数据集（训练或测试）
     * @param split 数据集名称 ("train" 或 "test")
     * @return pair<图像张量, 标签张量>
     */
    std::pair<Tensor, Tensor> load_and_preprocess_split(const std::string& split);

    /**
     * @brief 预处理图像数据：转换类型、标准化、重塑
     * @param raw_images 原始图像张量
     * @return 预处理后的图像张量
     */
    Tensor preprocess_images(const Tensor& raw_images);

    /**
     * @brief 预处理标签数据：转换为one-hot编码
     * @param raw_labels 原始标签张量
     * @return one-hot编码的标签张量
     */
    Tensor preprocess_labels(const Tensor& raw_labels);

    /**
     * @brief 打印张量信息（辅助函数）
     */
    void print_tensor_info(const std::string& name, const Tensor& tensor) const;
};

} // namespace tr

#endif // TECH_RENAISSANCE_UTILS_MNIST_LOADER_H
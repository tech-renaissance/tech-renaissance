#pragma once

#include "tech_renaissance/task/dataset.h"

namespace tr {

/**
 * @class MnistDataset
 * @brief MNIST数据集类，继承Dataset接口并独立实现MNIST数据访问功能
 * @details 独立实现的MNIST数据集访问类，提供Dataset接口
 */
class MnistDataset : public Dataset {
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
    explicit MnistDataset(std::shared_ptr<Backend> backend,
                          const std::string& data_path = "",
                          float mean = DEFAULT_MEAN,
                          float std = DEFAULT_STD);

    /**
     * @brief 析构函数
     */
    ~MnistDataset() override = default;

    // 实现Dataset接口
    int get_train_size() const override;
    int get_test_size() const override;
    const char* get_name() const override;
    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    /**
     * @brief 加载MNIST数据集
     * @return pair<训练数据, 测试数据>，每个pair包含图像和标签
     */
    std::pair<std::pair<Tensor, Tensor>, std::pair<Tensor, Tensor>> load_data();

    /**
     * @brief 获取训练数据的批次生成器
     * @param batch_size 批次大小
     * @return SimpleBatchGenerator智能指针
     */
    std::unique_ptr<SimpleBatchGenerator> get_train_loader(int batch_size);

    /**
     * @brief 获取测试数据的批次生成器
     * @param batch_size 批次大小
     * @return SimpleBatchGenerator智能指针
     */
    std::unique_ptr<SimpleBatchGenerator> get_test_loader(int batch_size);

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
     * @brief 预处理图像数据：类型转换、标准化、重塑
     * @param raw_images 原始图像数据
     * @return 处理后的图像张量
     */
    Tensor preprocess_images(const Tensor& raw_images);

    /**
     * @brief 预处理标签数据：转换为one-hot编码
     * @param raw_labels 原始标签数据
     * @return 处理后的标签张量
     */
    Tensor preprocess_labels(const Tensor& raw_labels);

};

} // namespace tr
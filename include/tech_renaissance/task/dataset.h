#pragma once

#include <memory>
#include <vector>
#include "tech_renaissance/data/shape.h"
#include "tech_renaissance/data/tensor.h"

namespace tr {

    /**
     * @brief 简单的BatchGenerator实现，避免依赖backend模块
     */
    class SimpleBatchGenerator {
    public:
        SimpleBatchGenerator(const Tensor& images, const Tensor& labels, int batch_size,
                            std::shared_ptr<Backend> backend, bool is_training_data);

        bool has_next() const;
        std::pair<Tensor, Tensor> next_batch();
        void reset();
        int get_num_batches() const;

    private:
        const Tensor& images_;
        const Tensor& labels_;
        int batch_size_;
        int current_idx_;
        int num_samples_;
        std::shared_ptr<Backend> backend_;
        bool is_training_data_;
        std::vector<int> shuffled_indices_;

        void initialize_indices();
        void shuffle_indices();
    };

/**
 * @brief 数据集抽象基类
 * @details 定义数据集的统一接口，支持训练和测试数据访问
 */
class Dataset {
public:
    virtual ~Dataset() = default;

    /**
     * @brief 获取训练数据集大小
     * @return 训练样本数量
     */
    virtual int get_train_size() const = 0;

    /**
     * @brief 获取测试数据集大小
     * @return 测试样本数量
     */
    virtual int get_test_size() const = 0;

    /**
     * @brief 获取数据集名称
     * @return 数据集名称
     */
    virtual const char* get_name() const = 0;

    /**
     * @brief 获取输入数据的形状
     * @return 输入数据的形状
     */
    virtual Shape get_input_shape() const = 0;

    /**
     * @brief 获取输出数据的形状
     * @return 输出数据的形状
     */
    virtual Shape get_output_shape() const = 0;
    /**
     * @brief 获取训练数据的批次生成器
     * @param batch_size 批次大小
     * @return SimpleBatchGenerator智能指针
     */
    virtual std::unique_ptr<SimpleBatchGenerator> get_train_loader(int batch_size) = 0;

    /**
     * @brief 获取测试数据的批次生成器
     * @param batch_size 批次大小
     * @return SimpleBatchGenerator智能指针
     */
    virtual std::unique_ptr<SimpleBatchGenerator> get_test_loader(int batch_size) = 0;
};

} // namespace tr
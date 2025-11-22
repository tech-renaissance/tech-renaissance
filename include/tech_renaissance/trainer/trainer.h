/**
 * @file trainer.h
 * @brief 训练器类
 * @details 集成Model、Optimizer、Loss和Learning Rate Scheduler的高级训练编排器
 *          利用Model的logits()缓存机制实现零拷贝训练流程
 * @version 1.00.00
 * @date 2025-11-19
 * @author 技术觉醒团队
 * @note 依赖项: model.h, optimizer.h, loss.h, scheduler.h
 * @note 所属系列: trainer
 */

#pragma once

#include "tech_renaissance/model/model.h"
#include "tech_renaissance/trainer/optimizer.h"
#include "tech_renaissance/trainer/loss.h"
#include "tech_renaissance/trainer/scheduler.h"
#include <memory>
#include <functional>
#include <iomanip>
#include <iostream>

namespace tr {

/**
 * @brief 训练器类
 * @details 集成Model、Optimizer、Loss和Learning Rate Scheduler的高级训练编排器
 *          利用Model的logits()缓存机制实现零拷贝训练流程
 */
class Trainer {
private:
    Model& model_;                                         // 模型引用
    std::unique_ptr<Optimizer> optimizer_;                 // 优化器
    std::unique_ptr<Loss> loss_fn_;                        // 损失函数
    std::unique_ptr<Scheduler> scheduler_;               // 学习率调度器

    // 缓存管理
    const std::vector<Tensor*>* cached_params_;             // 缓存参数
    std::function<void()> cache_invalidator_;               // 缓存失效器

    // 训练状态
    bool training_;                                         // 训练模式
    int current_epoch_;                                     // 当前epoch
    int current_step_;                                      // 当前step
    mutable bool grad_cleared_ = true;                       // ✅ 新增：梯度清零状态标记

public:
    /**
     * @brief 构造函数
     * @param model 模型引用
     * @param optimizer 优化器
     * @param loss_fn 损失函数
     * @param scheduler 学习率调度器（可选）
     */
    Trainer(Model& model,
            std::unique_ptr<Optimizer> optimizer,
            std::unique_ptr<Loss> loss_fn,
            std::unique_ptr<Scheduler> scheduler = nullptr);

    /**
     * @brief 析构函数
     */
    ~Trainer() = default;

    // === 训练步骤 ===

    /**
     * @brief 训练步骤
     * @param input 输入张量
     * @param target 目标张量
     * @return 损失值
     */
    float train_step(const Tensor& input, const Tensor& target);

    /**
     * @brief 评估步骤
     * @param input 输入张量
     * @param target 目标张量
     * @return 损失值
     */
    float eval_step(const Tensor& input, const Tensor& target);

    /**
     * @brief 训练一个epoch
     * @param train_loader 训练数据加载器
     */
    template<typename DataLoader>
    void train_epoch(const DataLoader& train_loader);

    /**
     * @brief 评估一个epoch
     * @param eval_loader 评估数据加载器
     * @return 平均损失
     */
    template<typename DataLoader>
    float eval_epoch(const DataLoader& eval_loader);

    // === 完整训练 ===

    /**
     * @brief 完整训练流程
     * @tparam DataLoader 数据加载器类型
     * @param num_epochs 训练轮数
     * @param train_loader 训练数据加载器
     * @param eval_loader 评估数据加载器（可选）
     * @param print_interval 打印间隔（可选）
     */
    template<typename DataLoader>
    void fit(int num_epochs,
             const DataLoader& train_loader,
             const DataLoader& eval_loader = nullptr,
             int print_interval = 100);

    // === 模式管理 ===

    /**
     * @brief 设置训练模式
     */
    void train();

    /**
     * @brief 设置评估模式
     */
    void eval();

    /**
     * @brief 检查是否为训练模式
     * @return true表示训练模式
     */
    bool is_training() const { return training_; }

    // === 访问器 ===

    /**
     * @brief 获取模型
     * @return 模型引用
     */
    Model& get_model() { return model_; }

    /**
     * @brief 获取优化器
     * @return 优化器指针
     */
    Optimizer* get_optimizer() const { return optimizer_.get(); }

    /**
     * @brief 获取损失函数
     * @return 损失函数指针
     */
    Loss* get_loss_function() const { return loss_fn_.get(); }

    /**
     * @brief 获取学习率调度器
     * @return 学习率调度器指针
     */
    Scheduler* get_scheduler() const { return scheduler_.get(); }

    /**
     * @brief 获取当前epoch
     * @return 当前epoch数
     */
    int get_current_epoch() const { return current_epoch_; }

    /**
     * @brief 获取当前step
     * @return 当前step数
     */
    int get_current_step() const { return current_step_; }

    // === 学习率调度 ===

    /**
     * @brief 执行一步学习率调度
     * @param epoch 当前epoch数
     * @return 当前学习率
     */
    float step_lr_scheduler(int epoch);

    /**
     * @brief 获取当前学习率
     * @return 学习率
     */
    float get_current_lr() const;

    // === 检查点 ===

    /**
     * @brief 保存检查点
     * @param filepath 文件路径
     */
    void save_checkpoint(const std::string& filepath) const;

    /**
     * @brief 加载检查点
     * @param filepath 文件路径
     */
    void load_checkpoint(const std::string& filepath);

    // === 设备管理 ===

    /**
     * @brief 设备转移
     * @param device 目标设备
     */
    void to(const Device& device);

    /**
     * @brief 获取当前设备
     * @return 当前设备
     */
    Device device() const;

private:
    /**
     * @brief 初始化缓存管理
     */
    void initialize_cache_management();

    /**
     * @brief 清理缓存管理
     */
    void cleanup_cache_management();

    /**
     * @brief 验证组件
     */
    void validate_components() const;

    /**
     * @brief 更新学习率
     */
    void update_learning_rate();

    /**
     * @brief 打印进度
     * @param epoch 当前epoch
     * @param step 当前step
     * @param loss 当前损失
     * @param total_steps 总步数
     */
    void print_progress(int epoch, int step, float loss, int total_steps) const;
};

// ===== 模板实现 =====

template<typename DataLoader>
void Trainer::train_epoch(const DataLoader& train_loader) {
    train();  // 设置训练模式

    int step = 0;
    float total_loss = 0.0f;

    for (auto& [batch_x, batch_y] : train_loader) {
        float loss = train_step(batch_x, batch_y);
        total_loss += loss;
        step++;
    }

    // epoch结束后更新学习率
    update_learning_rate();
    current_epoch_++;
}

template<typename DataLoader>
float Trainer::eval_epoch(const DataLoader& eval_loader) {
    eval();  // 设置评估模式

    float total_loss = 0.0f;
    int num_batches = 0;

    for (auto& [batch_x, batch_y] : eval_loader) {
        float loss = eval_step(batch_x, batch_y);
        total_loss += loss;
        num_batches++;
    }

    return total_loss / num_batches;
}

template<typename DataLoader>
void Trainer::fit(int num_epochs,
                 const DataLoader& train_loader,
                 const DataLoader& eval_loader,
                 int print_interval) {
    validate_components();

    // 初始化优化器状态（如果需要）
    if (!optimizer_->get_state_manager() ||
        !optimizer_->get_state_manager()->is_initialized()) {
        optimizer_->initialize(model_);
    }

    std::cout << "Starting training for " << num_epochs << " epochs..." << std::endl;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // 训练一个epoch
        train_epoch(train_loader);

        // 如果有评估数据，进行评估
        if (eval_loader) {
            float eval_loss = eval_epoch(eval_loader);
            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs
                      << " - Eval Loss: " << std::fixed << std::setprecision(4)
                      << eval_loss << std::endl;
        } else {
            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs
                      << " completed." << std::endl;
        }
    }

    std::cout << "Training completed!" << std::endl;
}

} // namespace tr
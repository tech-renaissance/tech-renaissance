/**
 * @file trainer.cpp
 * @brief 训练器实现
 * @details 集成Model、Optimizer、Loss和Learning Rate Scheduler的高级训练编排器
 * @version 1.57.0
 * @date 2025-11-21
 * @author 技术觉醒团队
 * @note 依赖项: trainer.h, backend_manager.h
 */

#include "tech_renaissance/trainer/trainer.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/utils/tr_exception.h"
#include <fstream>
#include <iostream>
#include <iomanip>

namespace tr {

// ===== Trainer 构造函数 =====

// 主要构造函数 - 推荐使用shared_ptr
Trainer::Trainer(std::shared_ptr<Model> model,
                 std::shared_ptr<Loss> loss_fn,
                 std::shared_ptr<Optimizer> optimizer,
                 std::shared_ptr<Scheduler> scheduler)
    : model_(model)
    , loss_fn_(loss_fn)
    , optimizer_(optimizer)
    , scheduler_(scheduler)
    , cached_params_(nullptr)
    , training_(true)
    , current_epoch_(0)
    , current_step_(0) {

    // 验证必要组件
    if (!model_) {
        throw TRException("[Trainer::Trainer] Model cannot be null");
    }
    if (!loss_fn_) {
        throw TRException("[Trainer::Trainer] Loss function cannot be null");
    }
    if (!optimizer_) {
        throw TRException("[Trainer::Trainer] Optimizer cannot be null");
    }

    // 初始化缓存管理
    initialize_cache_management();

    // 设置模型训练模式
    model_->train();
}

// 向后兼容构造函数 - 支持栈对象引用
Trainer::Trainer(Model& model,
                 Loss& loss_fn,
                 Optimizer& optimizer,
                 Scheduler& scheduler)
    : Trainer(
        std::shared_ptr<Model>(&model, [](Model*){}),
        std::shared_ptr<Loss>(&loss_fn, [](Loss*){}),
        std::shared_ptr<Optimizer>(&optimizer, [](Optimizer*){}),
        std::shared_ptr<Scheduler>(&scheduler, [](Scheduler*){})) {
}

// ===== Trainer 核心步骤 =====

float Trainer::train_step(const Tensor& input, const Tensor& target) {
    if (!training_) {
        train();  // 切换到训练模式
    }

    validate_components();

    // ✅ 智能清零：只在必要时执行
    if (!grad_cleared_) {
        optimizer_->zero_grad(*model_);
        grad_cleared_ = true;
    }

    // ✅ 确保参数有梯度（修复初始化问题）
    for (Tensor* param : model_->trainable_parameters()) {
        if (!param->has_grad()) {
            auto backend = BackendManager::instance().get_backend(model_->device());
            Tensor zero_grad = backend->zeros(param->shape(), DType::FP32);
            param->set_grad(zero_grad);
        }
    }

    // 2. 前向传播（参考成功的实现）
    auto output = model_->forward(input);

    // 3. 计算损失
    loss_fn_->train();
    float loss = loss_fn_->criterion(output, target);

    // 4. 反向传播：损失函数会自动在output上创建梯度
    model_->backward(output.grad());

    // 5. 参数更新
    optimizer_->step(*model_);

    grad_cleared_ = false;  // ✅ 标记需要清零
    current_step_++;
    return loss;
}

float Trainer::eval_step(const Tensor& input, const Tensor& target) {
    if (training_) {
        eval();  // 切换到评估模式
    }

    validate_components();

    // 1. 前向传播
    model_->forward(input);

    // 2. 计算损失，使用缓存的logits()结果
    loss_fn_->eval();
    float loss = loss_fn_->criterion(model_->logits(), target);

    return loss;
}

// ===== Trainer 模式管理 =====

void Trainer::train() {
    training_ = true;
    model_->train();
    if (loss_fn_) {
        loss_fn_->train();
    }
}

void Trainer::eval() {
    training_ = false;
    model_->eval();
    if (loss_fn_) {
        loss_fn_->eval();
    }
}

// ===== Trainer 检查点管理 =====

void Trainer::save_checkpoint(const std::string& filepath) const {
    // TODO: 实现检查点保存
    throw TRException("[Trainer::save_checkpoint] Checkpoint saving not implemented yet");
}

void Trainer::load_checkpoint(const std::string& filepath) {
    // TODO: 实现检查点加载
    throw TRException("[Trainer::load_checkpoint] Checkpoint loading not implemented yet");
}

// ===== Trainer 设备管理 =====

void Trainer::to(const Device& device) {
    // 转移模型
    model_->to(device);

    // 转移优化器
    if (optimizer_) {
        optimizer_->set_backend(BackendManager::instance().get_backend(device));
    }

    // 清理并重新初始化缓存管理
    cleanup_cache_management();
    initialize_cache_management();
}

Device Trainer::device() const {
    return model_->device();
}

// ===== Trainer 私有辅助方法 =====

void Trainer::initialize_cache_management() {
    // 设置缓存失效回调
    cache_invalidator_ = [this]() {
        cached_params_ = nullptr;
    };

    // TODO: 注册模型的缓存失效回调
    // model_.register_cache_invalidation_callback(cache_invalidator_);
}

void Trainer::cleanup_cache_management() {
    cached_params_ = nullptr;
    cache_invalidator_ = nullptr;
}

void Trainer::validate_components() const {
    if (!optimizer_) {
        throw TRException("[Trainer::validate_components] Optimizer is null");
    }
    if (!loss_fn_) {
        throw TRException("[Trainer::validate_components] Loss function is null");
    }

    // 检查优化器是否已初始化
    if (!optimizer_->get_state_manager() ||
        !optimizer_->get_state_manager()->is_initialized()) {
        throw TRException("[Trainer::validate_components] Optimizer not initialized. Call optimizer->initialize(model) first.");
    }
}

void Trainer::update_learning_rate() {
    if (scheduler_) {
        float new_lr = scheduler_->get_lr(current_epoch_);
        optimizer_->set_lr(new_lr);
    }
}

void Trainer::print_progress(int epoch, int step, float loss, int total_steps) const {
    std::cout << "Epoch " << (epoch + 1) << ", Step " << step << "/" << total_steps
              << " - Loss: " << std::fixed << std::setprecision(4) << loss << std::endl;
}

// ===== 学习率调度方法 =====

float Trainer::step_lr_scheduler(int epoch) {
    if (scheduler_) {
        float new_lr = scheduler_->get_lr(epoch);
        optimizer_->set_lr(new_lr);
        return new_lr;
    }
    return optimizer_->get_lr();
}

float Trainer::get_current_lr() const {
    return optimizer_ ? optimizer_->get_lr() : 0.0f;
}

} // namespace tr
/**
 * @file optimizer.cpp
 * @brief 优化器基类实现
 * @details 提供统一的优化器接口和状态管理框架
 * @version 1.50.0
 * @date 2025-11-19
 * @author 技术觉醒团队
 * @note 依赖项: optimizer.h, backend_manager.h
 */

#include "tech_renaissance/trainer/optimizer.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/utils/tr_exception.h"

namespace tr {

// ===== Optimizer 基础实现 =====

Optimizer::Optimizer(float lr, std::shared_ptr<Backend> backend)
    : learning_rate_(lr), backend_(backend) {

    // 参数验证
    if (lr <= 0.0f) {
        throw TRException("[Optimizer::Optimizer] Learning rate must be positive, got: " + std::to_string(lr));
    }
}

// ===== Optimizer 核心接口实现 =====

void Optimizer::initialize(const Model& model) {
    // 验证模型
    validate_model(model);

    // 确保设备一致性
    ensure_device_consistency(model);

    // 子类会覆盖此方法来执行具体初始化
}

void Optimizer::step(Model& model) {
    // 验证模型
    validate_model(model);

    // 确保设备一致性
    ensure_device_consistency(model);

    if (!state_manager_ || !state_manager_->is_initialized()) {
        throw TRException("[Optimizer::step] Optimizer not initialized. Call initialize() first.");
    }

    // 获取模型参数
    auto params = model.trainable_parameters();

    if (params.empty()) {
        return;  // 没有可训练参数
    }

    // 对每个参数执行更新
    for (size_t i = 0; i < params.size(); ++i) {
        Tensor& param = *params[i];

        // 检查梯度是否存在
        if (!param.has_grad()) {
            throw TRException("[Optimizer::step] Parameter " + std::to_string(i) +
                           " has no gradient. Make sure to call backward() before step().");
        }

        const Tensor& grad = param.grad();

        // 获取对应的状态
        OptimizerState& state = state_manager_->get_state(i);

        // 调用子类的更新方法
        update_parameter(param, grad, state);
    }

    // 递增时间步
    state_manager_->increment_time_step();
}

void Optimizer::zero_grad(Model& model) {
    // 验证模型
    validate_model(model);

    // 清零所有参数的梯度
    model.zero_grad();
}

// ===== Optimizer 后端管理实现 =====

void Optimizer::set_backend(std::shared_ptr<Backend> backend) {
    if (!backend) {
        throw TRException("[Optimizer::set_backend] Cannot set null backend");
    }
    backend_ = backend;

    // 如果状态管理器已存在，也更新其后端
    if (state_manager_) {
        state_manager_->set_backend(backend);
    }
}

// ===== Optimizer 辅助方法实现 =====

void Optimizer::validate_model(const Model& model) const {
    // 这里可以添加模型验证逻辑
    // 例如检查模型是否有参数、是否在训练模式等
    // 目前保持简单
    (void)model; // 避免未使用参数警告
}

void Optimizer::ensure_device_consistency(const Model& model) {
    if (!backend_) {
        // 如果没有设置后端，使用CPU后端
        backend_ = BackendManager::instance().get_cpu_backend();
    }

    // 如果状态管理器存在，确保其设备与模型一致
    if (state_manager_ && state_manager_->is_initialized()) {
        // 这里可以添加设备一致性检查逻辑
        // 目前保持简单，将来可以扩展
    }
}

} // namespace tr
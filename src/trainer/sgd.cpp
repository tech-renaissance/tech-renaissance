/**
 * @file sgd.cpp
 * @brief SGD（随机梯度下降）优化器实现
 * @details 支持动量、权重衰减和Nesterov动量
 * @version 1.50.0
 * @date 2025-11-19
 * @author 技术觉醒团队
 * @note 依赖项: sgd.h, backend_manager.h
 */

#include "tech_renaissance/trainer/sgd.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/utils/tr_exception.h"
#include <sstream>
#include <iomanip>

namespace tr {

// ===== SGD 构造函数实现 =====

SGD::SGD(float lr, float momentum, float weight_decay, bool nesterov, std::shared_ptr<Backend> backend)
    : Optimizer(lr, backend), momentum_(momentum), weight_decay_(weight_decay), use_nesterov_(nesterov) {

    // 参数验证
    if (lr <= 0.0f) {
        throw TRException("[SGD::SGD] Learning rate must be positive, got: " + std::to_string(lr));
    }
    if (momentum < 0.0f || momentum >= 1.0f) {
        throw TRException("[SGD::SGD] Momentum must be in [0, 1), got: " + std::to_string(momentum));
    }
    if (weight_decay < 0.0f) {
        throw TRException("[SGD::SGD] Weight decay must be non-negative, got: " + std::to_string(weight_decay));
    }
}

// ===== SGD 初始化实现 =====

void SGD::initialize(const Model& model) {
    // 设置后端
    if (!backend_) {
        backend_ = BackendManager::instance().get_cpu_backend();
    }

    // 初始化状态管理器
    state_manager_ = std::make_unique<StateManager>(backend_);
    state_manager_->set_backend(backend_);

    // 获取模型参数（需要const版本）
    Model& non_const_model = const_cast<Model&>(model);
    auto params = non_const_model.trainable_parameters();

    if (params.empty()) {
        return;  // 没有可训练参数
    }

    // 初始化SGD状态
    state_manager_->initialize_sgd_states(params, momentum_);

    // P1优化：预分配临时缓冲区
    temp_buffers_.resize(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        // 在参数设备上创建临时缓冲区
        temp_buffers_[i] = backend_->empty(params[i]->shape(), params[i]->dtype());
    }
}

// ===== SGD 参数更新实现 =====

void SGD::update_parameter(Tensor& param, const Tensor& grad, OptimizerState& state) {
    // 1. 应用权重衰减（如果启用）
    if (weight_decay_ > 0.0f) {
        apply_weight_decay(param);
    }

    // 2. 根据动量配置选择更新算法
    if (momentum_ > 0.0f) {
        if (use_nesterov_) {
            update_nesterov_sgd(param, grad, state);
        } else {
            update_classic_sgd(param, grad, state);
        }
    } else {
        // 纯SGD：param = param - lr * grad
        // 使用预分配的临时缓冲区（如果可用）
        if (!temp_buffers_.empty()) {
            backend_->mul_into(grad, learning_rate_, temp_buffers_[0]);
            backend_->minus_into(param, temp_buffers_[0], param);
        } else {
            Tensor lr_grad = backend_->mul(grad, learning_rate_);
            backend_->minus_into(param, lr_grad, param);
        }
    }
}

void SGD::update_classic_sgd(Tensor& param, const Tensor& grad, OptimizerState& state) {
    Tensor& velocity = state.momentum;

    // 1. 更新动量：velocity = momentum * velocity + grad
    backend_->mul_into(velocity, momentum_, velocity);
    backend_->add_into(velocity, grad, velocity);

    // 2. 更新参数：param = param - lr * velocity
    if (!temp_buffers_.empty()) {
        backend_->mul_into(velocity, learning_rate_, temp_buffers_[0]);
        backend_->minus_into(param, temp_buffers_[0], param);
    } else {
        Tensor lr_velocity = backend_->mul(velocity, learning_rate_);
        backend_->minus_into(param, lr_velocity, param);
    }
}

void SGD::update_nesterov_sgd(Tensor& param, const Tensor& grad, OptimizerState& state) {
    Tensor& velocity = state.momentum;

    // 1. 更新动量：velocity = momentum * velocity + grad
    backend_->mul_into(velocity, momentum_, velocity);
    backend_->add_into(velocity, grad, velocity);

    // 2. 计算Nesterov梯度：nesterov_grad = grad + momentum * velocity
    if (!temp_buffers_.empty()) {
        // 使用预分配缓冲区避免临时分配
        // temp = momentum * velocity
        backend_->mul_into(velocity, momentum_, temp_buffers_[0]);
        // temp = temp + grad (即 nesterov_grad)
        backend_->add_into(temp_buffers_[0], grad, temp_buffers_[0]);
        // temp = temp * lr
        backend_->mul_into(temp_buffers_[0], learning_rate_, temp_buffers_[0]);
        // param = param - temp
        backend_->minus_into(param, temp_buffers_[0], param);
    } else {
        // 创建临时张量（次优方案）
        Tensor momentum_term = backend_->mul(velocity, momentum_);
        Tensor nesterov_grad = backend_->add(momentum_term, grad);
        Tensor update = backend_->mul(nesterov_grad, learning_rate_);
        backend_->minus_into(param, update, param);
    }
}

void SGD::apply_weight_decay(Tensor& param) {
    // 权重衰减：param = param * (1 - lr * weight_decay)
    float decay_factor = 1.0f - learning_rate_ * weight_decay_;
    backend_->mul_inplace(param, decay_factor);
}

// ===== SGD 信息获取实现 =====

std::string SGD::get_info() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "SGD Optimizer:" << std::endl;
    oss << "  Learning rate: " << learning_rate_ << std::endl;
    oss << "  Momentum: " << momentum_ << std::endl;
    oss << "  Weight decay: " << weight_decay_ << std::endl;
    oss << "  Nesterov: " << (use_nesterov_ ? "true" : "false") << std::endl;

    if (state_manager_ && state_manager_->is_initialized()) {
        oss << "  Parameters: " << state_manager_->state_count() << std::endl;
        oss << "  Device: " << state_manager_->device().to_string() << std::endl;
    }

    return oss.str();
}

} // namespace tr
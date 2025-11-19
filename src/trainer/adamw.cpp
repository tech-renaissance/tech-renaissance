/**
 * @file adamw.cpp
 * @brief AdamW（解耦权重衰减的Adam）优化器实现
 * @details 实现AdamW优化算法，权重衰减与一阶矩二阶矩估计解耦
 * @version 1.54.0
 * @date 2025-11-19
 * @author 技术觉醒团队
 * @note 依赖项: adamw.h, backend_manager.h
 */

#include "tech_renaissance/trainer/adamw.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/utils/tr_exception.h"
#include <sstream>
#include <iomanip>
#include <cmath>

namespace tr {

// ===== AdamW 构造函数实现 =====

AdamW::AdamW(float lr, float beta1, float beta2, float eps, float weight_decay, std::shared_ptr<Backend> backend)
    : Optimizer(lr, backend), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay) {

    // 参数验证
    if (lr <= 0.0f) {
        throw TRException("[AdamW::AdamW] Learning rate must be positive, got: " + std::to_string(lr));
    }
    if (beta1 < 0.0f || beta1 >= 1.0f) {
        throw TRException("[AdamW::AdamW] Beta1 must be in [0, 1), got: " + std::to_string(beta1));
    }
    if (beta2 < 0.0f || beta2 >= 1.0f) {
        throw TRException("[AdamW::AdamW] Beta2 must be in [0, 1), got: " + std::to_string(beta2));
    }
    if (eps <= 0.0f) {
        throw TRException("[AdamW::AdamW] Epsilon must be positive, got: " + std::to_string(eps));
    }
    if (weight_decay < 0.0f) {
        throw TRException("[AdamW::AdamW] Weight decay must be non-negative, got: " + std::to_string(weight_decay));
    }
}

// ===== AdamW 初始化实现 =====

void AdamW::initialize(const Model& model) {
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

    // 初始化AdamW状态（m和v缓冲区）
    state_manager_->initialize_adam_states(params, beta1_, beta2_);

    // P1优化：预分配临时缓冲区
    size_t num_params = params.size();
    temp_m_hat_buffers_.resize(num_params);
    temp_v_hat_buffers_.resize(num_params);
    temp_update_buffers_.resize(num_params);

    for (size_t i = 0; i < num_params; ++i) {
        // 在参数设备上创建临时缓冲区
        temp_m_hat_buffers_[i] = backend_->empty(params[i]->shape(), params[i]->dtype());
        temp_v_hat_buffers_[i] = backend_->empty(params[i]->shape(), params[i]->dtype());
        temp_update_buffers_[i] = backend_->empty(params[i]->shape(), params[i]->dtype());
    }
}

// ===== AdamW 参数更新实现 =====

void AdamW::step(Model& model) {
    // 调用基类的step方法
    Optimizer::step(model);
}

void AdamW::update_parameter(Tensor& param, const Tensor& grad, OptimizerState& state, size_t param_index) {
    // 1. 确保AdamW状态存在
    if (!state.has_adam_state) {
        throw TRException("[AdamW::update_parameter] Adam state not initialized for parameter");
    }

    // 2. 获取当前时间步（在基类递增之前）
    int current_time_step = state_manager_->get_time_step() + 1;

    // 3. 更新一阶矩和二阶矩估计
    update_moments(state.adam_m, state.adam_v, grad, param_index);

    // 4. 计算偏置修正后的矩估计
    compute_bias_corrected_moments(temp_m_hat_buffers_[param_index], temp_v_hat_buffers_[param_index],
                                  state.adam_m, state.adam_v, current_time_step, param_index);

    // 5. 执行AdamW参数更新
    apply_adamw_update(param, temp_m_hat_buffers_[param_index], temp_v_hat_buffers_[param_index], param_index);

    // 6. 应用解耦权重衰减（如果启用）- 在AdamW更新后应用
    if (weight_decay_ > 0.0f) {
        apply_decoupled_weight_decay(param);
    }
}

void AdamW::apply_decoupled_weight_decay(Tensor& param) {
    // 解耦权重衰减：param = param - lr * weight_decay * param
    float decay_amount = learning_rate_ * weight_decay_;
    backend_->add_inplace(param, -decay_amount);  // param = param + (-lr * weight_decay)
}

void AdamW::update_moments(Tensor& m, Tensor& v, const Tensor& grad, size_t param_index) {
    // 更新一阶矩估计：m = beta1 * m + (1 - beta1) * grad

    // 计算beta1 * m，存入临时缓冲区
    backend_->mul_into(m, beta1_, temp_update_buffers_[param_index]);

    // 计算(1 - beta1) * grad，存入另一个临时缓冲区
    Tensor& temp_grad_buffer = temp_m_hat_buffers_[param_index];  // 使用m_hat缓冲区作为临时存储
    backend_->mul_into(grad, 1.0f - beta1_, temp_grad_buffer);

    // m = beta1 * m + (1 - beta1) * grad
    backend_->add_into(temp_update_buffers_[param_index], temp_grad_buffer, m);

    // 更新二阶矩估计：v = beta2 * v + (1 - beta2) * grad^2

    // 计算beta2 * v，存入临时缓冲区
    backend_->mul_into(v, beta2_, temp_update_buffers_[param_index]);

    // 计算grad^2，存入临时缓冲区
    backend_->square_into(grad, temp_grad_buffer);

    // 计算(1 - beta2) * grad^2
    backend_->mul_into(temp_grad_buffer, 1.0f - beta2_, temp_grad_buffer);

    // v = beta2 * v + (1 - beta2) * grad^2
    backend_->add_into(temp_update_buffers_[param_index], temp_grad_buffer, v);
}

void AdamW::compute_bias_corrected_moments(Tensor& m_hat, Tensor& v_hat,
                                          const Tensor& m, const Tensor& v,
                                          int time_step, size_t param_index) {
    // 计算偏置修正因子
    float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(time_step));
    float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(time_step));

    // 防止除零
    if (bias_correction1 <= 0.0f) bias_correction1 = 1e-8f;
    if (bias_correction2 <= 0.0f) bias_correction2 = 1e-8f;

    // 计算偏置修正后的矩估计
    // m_hat = m / (1 - beta1^t)
    backend_->mul_into(m, 1.0f / bias_correction1, m_hat);

    // v_hat = v / (1 - beta2^t)
    backend_->mul_into(v, 1.0f / bias_correction2, v_hat);
}

void AdamW::apply_adamw_update(Tensor& param, const Tensor& m_hat, const Tensor& v_hat, size_t param_index) {
    // 计算更新量：update = lr * m_hat / (sqrt(v_hat) + eps)

    // 1. 计算 sqrt(v_hat) - 使用CPU后端的sqrt_into
    auto* cpu_backend = dynamic_cast<CpuBackend*>(backend_.get());
    if (!cpu_backend) {
        throw TRException("[AdamW::apply_adamw_update] sqrt operation requires CpuBackend");
    }
    cpu_backend->sqrt_into(v_hat, temp_update_buffers_[param_index]);

    // 2. 计算 sqrt(v_hat) + eps
    backend_->add_inplace(temp_update_buffers_[param_index], eps_);

    // 3. 计算 m_hat / (sqrt(v_hat) + eps)
    // 使用临时缓冲区存储除法结果
    // 注意：这里需要逐元素除法，假设backend有逐元素除法操作，如果没有则用其他方式实现
    // 暂时使用mul_into和scalar除法来模拟
    for (int64_t i = 0; i < m_hat.numel(); ++i) {
        auto* cpu_backend = dynamic_cast<CpuBackend*>(backend_.get());
        if (cpu_backend) {
            float m_val = cpu_backend->get_item_fp32(m_hat, i);
            float denom_val = cpu_backend->get_item_fp32(temp_update_buffers_[param_index], i);
            cpu_backend->set_item_fp32(temp_update_buffers_[param_index], i, m_val / denom_val);
        }
    }

    // 4. 应用学习率：update = lr * result
    backend_->mul_inplace(temp_update_buffers_[param_index], learning_rate_);

    // 5. 更新参数：param = param - update
    backend_->minus_into(param, temp_update_buffers_[param_index], param);
}

// ===== AdamW 信息获取实现 =====

std::string AdamW::get_info() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "AdamW Optimizer:" << std::endl;
    oss << "  Learning rate: " << learning_rate_ << std::endl;
    oss << "  Beta1: " << beta1_ << std::endl;
    oss << "  Beta2: " << beta2_ << std::endl;
    oss << "  Epsilon: " << eps_ << std::endl;
    oss << "  Weight decay: " << weight_decay_ << std::endl;

    if (state_manager_ && state_manager_->is_initialized()) {
        oss << "  Parameters: " << state_manager_->state_count() << std::endl;
        oss << "  Device: " << state_manager_->device().to_string() << std::endl;
        oss << "  Time step: " << state_manager_->get_time_step() << std::endl;
    }

    return oss.str();
}

} // namespace tr
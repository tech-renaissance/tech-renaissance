/**
 * @file adam.cpp
 * @brief Adam（自适应矩估计）优化器实现
 * @details 支持权重衰减、偏置修正等完整Adam功能
 * @version 1.53.0
 * @date 2025-11-19
 * @author 技术觉醒团队
 * @note 依赖项: adam.h, backend_manager.h
 */

#include "tech_renaissance/trainer/adam.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/utils/tr_exception.h"
#include <sstream>
#include <iomanip>
#include <cmath>

namespace tr {

// ===== Adam 构造函数实现 =====

Adam::Adam(float lr, float beta1, float beta2, float eps, float weight_decay, std::shared_ptr<Backend> backend)
    : Optimizer(lr, backend), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay), last_time_step_(0) {

    // 参数验证
    if (lr <= 0.0f) {
        throw TRException("[Adam::Adam] Learning rate must be positive, got: " + std::to_string(lr));
    }
    if (beta1 < 0.0f || beta1 >= 1.0f) {
        throw TRException("[Adam::Adam] Beta1 must be in [0, 1), got: " + std::to_string(beta1));
    }
    if (beta2 < 0.0f || beta2 >= 1.0f) {
        throw TRException("[Adam::Adam] Beta2 must be in [0, 1), got: " + std::to_string(beta2));
    }
    if (eps <= 0.0f) {
        throw TRException("[Adam::Adam] Epsilon must be positive, got: " + std::to_string(eps));
    }
    if (weight_decay < 0.0f) {
        throw TRException("[Adam::Adam] Weight decay must be non-negative, got: " + std::to_string(weight_decay));
    }

    // 【P0级优化】初始化bias_correction缓存
    cached_bias_correction1_.clear();
    cached_bias_correction2_.clear();
}

// ===== Adam 初始化实现 =====

void Adam::initialize(const Model& model) {
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

    // 初始化Adam状态（m和v缓冲区）
    state_manager_->initialize_adam_states(params, beta1_, beta2_);

    // 【P0级优化】优化临时缓冲区分配，减少内存使用
    // 从4个缓冲区减少到2个，通过重用减少内存开销
    size_t num_params = params.size();
    temp_m_hat_buffers_.resize(num_params);
    temp_update_buffers_.resize(num_params);
    // 移除temp_v_hat_buffers_和temp_scratch_buffers_，通过重用temp_update_buffers_实现

    for (size_t i = 0; i < num_params; ++i) {
        // 在参数设备上创建临时缓冲区
        temp_m_hat_buffers_[i] = backend_->empty(params[i]->shape(), params[i]->dtype());
        temp_update_buffers_[i] = backend_->empty(params[i]->shape(), params[i]->dtype());
    }
}

// ===== Adam 参数更新实现 =====

void Adam::step(Model& model) {
    // 调用基类的step方法
    Optimizer::step(model);
}

void Adam::update_parameter(Tensor& param, const Tensor& grad, OptimizerState& state, size_t param_index) {
    // 1. 确保Adam状态存在
    if (!state.has_adam_state) {
        throw TRException("[Adam::update_parameter] Adam state not initialized for parameter");
    }

    // 2. 获取当前时间步（在基类递增之前）
    int current_time_step = state_manager_->get_time_step() + 1;  // 因为基类最后会递增

    // 3. 更新一阶矩和二阶矩估计
    update_moments(state.adam_m, state.adam_v, grad, param_index);

    // 4. 【P0级优化】计算偏置修正后的矩估计（重用temp_update_buffers_作为v_hat）
    compute_bias_corrected_moments(temp_m_hat_buffers_[param_index], temp_update_buffers_[param_index],
                                  state.adam_m, state.adam_v, current_time_step, param_index);

    // 5. 执行Adam参数更新
    apply_adam_update(param, temp_m_hat_buffers_[param_index], temp_update_buffers_[param_index], param_index);

    // 6. 应用权重衰减（如果启用）- 在Adam更新后应用
    if (weight_decay_ > 0.0f) {
        apply_weight_decay(param);
    }
}

void Adam::apply_weight_decay(Tensor& param) {
    // 权重衰减：param = param * (1 - lr * weight_decay)
    float decay_factor = 1.0f - learning_rate_ * weight_decay_;
    backend_->mul_inplace(param, decay_factor);
}

void Adam::update_moments(Tensor& m, Tensor& v, const Tensor& grad, size_t param_index) {
    // 【P0级优化】更新一阶矩和二阶矩估计，减少临时缓冲区使用

    // 更新一阶矩估计：m = beta1 * m + (1 - beta1) * grad
    // 使用temp_m_hat_buffers_作为临时缓冲区存储beta1 * m
    backend_->mul_into(m, beta1_, temp_m_hat_buffers_[param_index]);
    // 计算(1 - beta1) * grad，直接使用grad的平方操作空间
    backend_->square_into(grad, temp_update_buffers_[param_index]);
    // 重新计算(1 - beta1) * grad
    backend_->mul_into(grad, 1.0f - beta1_, temp_update_buffers_[param_index]);
    // m = beta1 * m + (1 - beta1) * grad
    backend_->add_into(temp_m_hat_buffers_[param_index], temp_update_buffers_[param_index], m);

    // 更新二阶矩估计：v = beta2 * v + (1 - beta2) * grad^2
    // 使用temp_m_hat_buffers_存储beta2 * v
    backend_->mul_into(v, beta2_, temp_m_hat_buffers_[param_index]);
    // 重新计算grad^2（之前覆盖了）
    backend_->square_into(grad, temp_update_buffers_[param_index]);
    // 计算(1 - beta2) * grad^2
    backend_->mul_into(temp_update_buffers_[param_index], 1.0f - beta2_, temp_update_buffers_[param_index]);
    // v = beta2 * v + (1 - beta2) * grad^2
    backend_->add_into(temp_m_hat_buffers_[param_index], temp_update_buffers_[param_index], v);
}

void Adam::compute_bias_corrected_moments(Tensor& m_hat, Tensor& v_hat,
                                         const Tensor& m, const Tensor& v,
                                         int time_step, size_t param_index) {
    // 【P0级优化】使用缓存的bias_correction避免重复pow运算
    // 只有时间步变化时才重新计算bias_correction
    if (time_step != last_time_step_) {
        // 计算偏置修正因子并缓存
        cached_bias_correction1_.resize(1);
        cached_bias_correction2_.resize(1);
        cached_bias_correction1_[0] = 1.0f - std::pow(beta1_, static_cast<float>(time_step));
        cached_bias_correction2_[0] = 1.0f - std::pow(beta2_, static_cast<float>(time_step));
        last_time_step_ = time_step;

        // 防止除零
        if (cached_bias_correction1_[0] <= 0.0f) cached_bias_correction1_[0] = 1e-8f;
        if (cached_bias_correction2_[0] <= 0.0f) cached_bias_correction2_[0] = 1e-8f;
    }

    // 计算偏置修正后的矩估计（使用缓存的值）
    // m_hat = m / (1 - beta1^t)
    backend_->mul_into(m, 1.0f / cached_bias_correction1_[0], m_hat);

    // v_hat = v / (1 - beta2^t)
    backend_->mul_into(v, 1.0f / cached_bias_correction2_[0], v_hat);
}

void Adam::apply_adam_update(Tensor& param, const Tensor& m_hat, const Tensor& v_hat, size_t param_index) {
    // 【P0级优化】计算更新量：update = lr * m_hat / (sqrt(v_hat) + eps)
    // 使用向量化操作替代逐元素循环，消除致命性能瓶颈

    // 1. 计算 sqrt(v_hat) -> temp_update_buffers_[param_index]
    backend_->sqrt_into(v_hat, temp_update_buffers_[param_index]);

    // 2. 计算 sqrt(v_hat) + eps -> temp_update_buffers_[param_index]
    backend_->add_inplace(temp_update_buffers_[param_index], eps_);

    // 3. 向量化除法：m_hat / (sqrt(v_hat) + eps) -> temp_update_buffers_[param_index]
    // 【关键优化】使用向量化div_into替代逐元素循环
    backend_->div_into(m_hat, temp_update_buffers_[param_index], temp_update_buffers_[param_index]);

    // 4. 应用学习率：update = lr * result
    backend_->mul_inplace(temp_update_buffers_[param_index], learning_rate_);

    // 5. 更新参数：param = param - update
    backend_->minus_into(param, temp_update_buffers_[param_index], param);
}

// ===== Adam 信息获取实现 =====

std::string Adam::get_info() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Adam Optimizer:" << std::endl;
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
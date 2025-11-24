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

    // 【P0级优化】按照Algorithm.md重新设计缓冲区，使用temp1和temp2命名
    // 每个参数使用2个临时缓冲区，通过重用减少内存开销
    size_t num_params = params.size();
    temp1_buffers_.resize(num_params);
    temp2_buffers_.resize(num_params);

    for (size_t i = 0; i < num_params; ++i) {
        // 在参数设备上创建临时缓冲区
        temp1_buffers_[i] = backend_->empty(params[i]->shape(), params[i]->dtype());
        temp2_buffers_[i] = backend_->empty(params[i]->shape(), params[i]->dtype());
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

    // 【按照Algorithm.md重新实现AdamW算法】

    // Step1: m = beta1 * m + (1 - beta1) * grad (使用原始梯度，不加权重衰减)
    // temp2 = (1 - beta1) * grad
    backend_->mul_into(grad, 1.0f - beta1_, temp2_buffers_[param_index]);
    // m *= beta1
    backend_->mul_inplace(state.adam_m, beta1_);
    // m = m + temp2
    backend_->add_into(state.adam_m, temp2_buffers_[param_index], state.adam_m);

    // Step2: m_hat = m / (1.0 - pow(beta1, time_step))，同时预乘学习率
    // coeff1 = lr / (1.0 - pow(beta1, time_step))
    float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(current_time_step));
    if (bias_correction1 <= 0.0f) bias_correction1 = 1e-8f;
    float coeff1 = learning_rate_ / bias_correction1;
    // temp2 = m * coeff1
    backend_->mul_into(state.adam_m, coeff1, temp2_buffers_[param_index]);

    // Step3: v = beta2 * v + (1 - beta2) * pow(grad, 2)
    // temp1 = square(grad)
    backend_->square_into(grad, temp1_buffers_[param_index]);
    // temp1 *= (1.0 - beta2)
    backend_->mul_inplace(temp1_buffers_[param_index], 1.0f - beta2_);
    // v *= beta2
    backend_->mul_inplace(state.adam_v, beta2_);
    // v = v + temp1
    backend_->add_into(state.adam_v, temp1_buffers_[param_index], state.adam_v);

    // Step4: v_hat = v / (1 - pow(beta2, time_step))
    // coeff2 = 1.0 / (1.0 - pow(beta2, time_step))
    float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(current_time_step));
    if (bias_correction2 <= 0.0f) bias_correction2 = 1e-8f;
    float coeff2 = 1.0f / bias_correction2;
    // temp1 = v * coeff2
    backend_->mul_into(state.adam_v, coeff2, temp1_buffers_[param_index]);

    // Step5: weight = (1 - lr * weight_decay) * weight - lr * m_hat / (sqrt(v_hat) + eps)
    // 先处理权重衰减部分：weight *= (1 - lr * weight_decay)
    if (weight_decay_ > 0.0f) {
        float coeff3 = 1.0f - learning_rate_ * weight_decay_;
        backend_->mul_inplace(param, coeff3);
    }

    // 然后处理Adam更新部分：weight = weight - temp2 / (sqrt(temp1) + eps)
    // temp1 = sqrt(temp1)
    backend_->sqrt_inplace(temp1_buffers_[param_index]);
    // temp1 += eps
    backend_->add_inplace(temp1_buffers_[param_index], eps_);
    // temp2 = temp2 / temp1
    backend_->div_into(temp2_buffers_[param_index], temp1_buffers_[param_index], temp2_buffers_[param_index]);
    // weight = weight - temp2
    backend_->minus_into(param, temp2_buffers_[param_index], param);

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
/**
 * @file state_manager.cpp
 * @brief 优化器状态管理器实现
 * @details 提供索引化状态管理，解决设备转移时的指针失效问题
 * @version 1.50.0
 * @date 2025-11-19
 * @author 技术觉醒团队
 * @note 依赖项: state_manager.h, backend_manager.h
 */

#include "tech_renaissance/trainer/state_manager.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/utils/tr_exception.h"
#include <iostream>
#include <stdexcept>

namespace tr {

// ===== StateManager 基础实现 =====

StateManager::StateManager(std::shared_ptr<Backend> backend)
    : backend_(backend), initialized_(false) {
}

void StateManager::set_backend(std::shared_ptr<Backend> backend) {
    if (!backend) {
        throw TRException("[StateManager::set_backend] Cannot set null backend");
    }
    backend_ = backend;
}

// ===== 初始化方法实现 =====

void StateManager::initialize_states(const std::vector<Tensor*>& params,
                                     const std::vector<std::string>& param_names) {
    if (!backend_) {
        throw TRException("[StateManager::initialize_states] Backend not set");
    }

    if (params.empty()) {
        return;  // 空参数列表，无需初始化
    }

    // 清空现有状态
    clear();

    // 预分配状态空间
    states_.resize(params.size());
    param_names_.resize(params.size());
    name_to_index_.clear();

    // 初始化状态（暂时为空，等待具体的优化器初始化）
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& param = *params[i];

        // 设置参数名称
        if (i < param_names.size()) {
            param_names_[i] = param_names[i];
        } else {
            param_names_[i] = "param_" + std::to_string(i);
        }

        // 建立名称映射
        name_to_index_[param_names_[i]] = i;

        // 状态保持为空，等待具体优化器填充
        states_[i] = OptimizerState{};
    }

    initialized_ = true;
}

void StateManager::initialize_sgd_states(const std::vector<Tensor*>& params, float momentum) {
    if (!backend_) {
        throw TRException("[StateManager::initialize_sgd_states] Backend not set");
    }

    // 首先初始化基础状态结构
    initialize_states(params);

    // 为每个参数创建SGD动量状态
    for (size_t i = 0; i < params.size(); ++i) {
        if (momentum > 0.0f) {
            const Tensor& param = *params[i];

            // 在参数设备上创建动量缓冲区
            Tensor momentum_buffer = backend_->zeros(param.shape(), param.dtype());

            states_[i].momentum = std::move(momentum_buffer);
            states_[i].has_momentum = true;
        }
    }
}

void StateManager::initialize_adam_states(const std::vector<Tensor*>& params,
                                         float beta1, float beta2) {
    if (!backend_) {
        throw TRException("[StateManager::initialize_adam_states] Backend not set");
    }

    // 首先初始化基础状态结构
    initialize_states(params);

    // 为每个参数创建Adam状态
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& param = *params[i];

        // 在参数设备上创建Adam缓冲区
        Tensor m_buffer = backend_->zeros(param.shape(), param.dtype());
        Tensor v_buffer = backend_->zeros(param.shape(), param.dtype());

        states_[i].adam_m = std::move(m_buffer);
        states_[i].adam_v = std::move(v_buffer);
        states_[i].has_adam_state = true;
    }
}

// ===== 状态访问接口实现 =====

OptimizerState& StateManager::get_state(size_t param_index) {
    if (!initialized_) {
        throw TRException("[StateManager::get_state] State manager not initialized");
    }

    if (param_index >= states_.size()) {
        throw TRException("[StateManager::get_state] Parameter index out of range: " + std::to_string(param_index));
    }

    return states_[param_index];
}

const OptimizerState& StateManager::get_state(size_t param_index) const {
    if (!initialized_) {
        throw TRException("[StateManager::get_state] State manager not initialized");
    }

    if (param_index >= states_.size()) {
        throw TRException("[StateManager::get_state] Parameter index out of range: " + std::to_string(param_index));
    }

    return states_[param_index];
}

OptimizerState& StateManager::get_state(const std::string& param_name) {
    auto it = name_to_index_.find(param_name);
    if (it == name_to_index_.end()) {
        throw TRException("[StateManager::get_state] Parameter not found: " + param_name);
    }

    return get_state(it->second);
}

// ===== 设备管理实现 =====

void StateManager::to(const Device& device) {
    if (!backend_) {
        throw TRException("[StateManager::to] Backend not set");
    }

    // 更新后端
    backend_ = BackendManager::instance().get_backend(device);

    // 转移所有状态张量到目标设备
    for (auto& state : states_) {
        if (state.has_momentum && state.momentum.storage_allocated()) {
            state.momentum = backend_->from_cpu(state.momentum);
        }

        if (state.has_adam_state) {
            if (state.adam_m.storage_allocated()) {
                state.adam_m = backend_->from_cpu(state.adam_m);
            }
            if (state.adam_v.storage_allocated()) {
                state.adam_v = backend_->from_cpu(state.adam_v);
            }
        }
    }
}

Device StateManager::device() const {
    if (!backend_) {
        return tr::CPU;  // 默认设备
    }
    return backend_->device();
}

// ===== 状态操作实现 =====

void StateManager::clear() {
    states_.clear();
    param_names_.clear();
    name_to_index_.clear();
    initialized_ = false;
}

void StateManager::increment_time_step() {
    for (auto& state : states_) {
        state.time_step++;
    }
}

int StateManager::get_time_step(size_t param_index) const {
    if (!initialized_ || param_index >= states_.size()) {
        return 0;
    }
    return states_[param_index].time_step;
}

// ===== 调试接口实现 =====

void StateManager::print_state_info() const {
    std::cout << "=== StateManager Info ===" << std::endl;
    std::cout << "Initialized: " << (initialized_ ? "YES" : "NO") << std::endl;
    std::cout << "State count: " << states_.size() << std::endl;
    std::cout << "Device: " << device().to_string() << std::endl;
    std::cout << "Time step: " << (states_.empty() ? 0 : states_[0].time_step) << std::endl;
    std::cout << std::endl;

    if (!initialized_) {
        return;
    }

    for (size_t i = 0; i < states_.size(); ++i) {
        const auto& state = states_[i];
        std::cout << "[" << i << "] " << param_names_[i] << ":" << std::endl;
        std::cout << "  SGD momentum: " << (state.has_momentum ? "YES" : "NO") << std::endl;
        std::cout << "  Adam state: " << (state.has_adam_state ? "YES" : "NO") << std::endl;
        std::cout << "  Time step: " << state.time_step << std::endl;

        if (state.has_momentum && state.momentum.storage_allocated()) {
            std::cout << "  Momentum shape: " << state.momentum.shape().to_string() << std::endl;
        }

        if (state.has_adam_state) {
            if (state.adam_m.storage_allocated()) {
                std::cout << "  Adam m shape: " << state.adam_m.shape().to_string() << std::endl;
            }
            if (state.adam_v.storage_allocated()) {
                std::cout << "  Adam v shape: " << state.adam_v.shape().to_string() << std::endl;
            }
        }
        std::cout << std::endl;
    }
}

const std::string& StateManager::get_param_name(size_t param_index) const {
    if (!initialized_ || param_index >= param_names_.size()) {
        static const std::string empty_name = "";
        return empty_name;
    }
    return param_names_[param_index];
}

size_t StateManager::get_param_index(const std::string& param_name) const {
    auto it = name_to_index_.find(param_name);
    if (it == name_to_index_.end()) {
        throw TRException("[StateManager::get_param_index] Parameter not found: " + param_name);
    }
    return it->second;
}

} // namespace tr
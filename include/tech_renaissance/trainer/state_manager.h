/**
 * @file state_manager.h
 * @brief 优化器状态管理器
 * @details 为Optimizer提供统一的状态管理接口，支持索引化访问和设备转移
 * @version 1.50.0
 * @date 2025-11-19
 * @author 技术觉醒团队
 * @note 依赖项: tensor.h, backend.h
 * @note 所属系列: optimizer
 */

#pragma once

#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/data/device.h"
#include "tech_renaissance/backend/backend.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace tr {

/**
 * @brief 优化器状态条目
 * @details 存储单个参数对应的所有优化器状态
 */
struct OptimizerState {
    // SGD状态
    Tensor momentum;
    bool has_momentum = false;

    // Adam状态（为未来扩展预留）
    Tensor adam_m;
    Tensor adam_v;
    bool has_adam_state = false;

    // 通用状态
    int time_step = 0;

    /**
     * @brief 清空所有状态
     */
    void clear() {
        momentum = Tensor();
        adam_m = Tensor();
        adam_v = Tensor();
        has_momentum = false;
        has_adam_state = false;
        time_step = 0;
    }

    /**
     * @brief 检查是否为空状态
     */
    bool is_empty() const {
        return !has_momentum && !has_adam_state;
    }
};

/**
 * @brief 优化器状态管理器
 * @details 提供索引化状态访问，解决设备转移时的指针失效问题
 */
class StateManager {
private:
    std::vector<OptimizerState> states_;                   // 按参数索引管理的状态
    std::shared_ptr<Backend> backend_;                     // 后端智能指针
    bool initialized_ = false;                             // 初始化标志

    // 参数映射（用于调试和状态访问）
    std::vector<std::string> param_names_;                 // 参数名称列表
    std::unordered_map<std::string, size_t> name_to_index_; // 名称到索引的映射

public:
    /**
     * @brief 构造函数
     * @param backend 后端智能指针
     */
    explicit StateManager(std::shared_ptr<Backend> backend = nullptr);

    /**
     * @brief 析构函数
     */
    ~StateManager() = default;

    /**
     * @brief 设置后端
     * @param backend 后端智能指针
     */
    void set_backend(std::shared_ptr<Backend> backend);

    /**
     * @brief 获取后端
     * @return 后端智能指针
     */
    std::shared_ptr<Backend> get_backend() const { return backend_; }

    // === 初始化方法 ===

    /**
     * @brief 初始化SGD状态
     * @param params 参数指针列表
     * @param momentum 动量系数
     */
    void initialize_sgd_states(const std::vector<Tensor*>& params, float momentum = 0.0f);

    /**
     * @brief 初始化Adam状态（为未来扩展预留）
     * @param params 参数指针列表
     * @param beta1 一阶矩衰减率
     * @param beta2 二阶矩衰减率
     */
    void initialize_adam_states(const std::vector<Tensor*>& params,
                               float beta1 = 0.9f, float beta2 = 0.999f);

    /**
     * @brief 初始化所有状态（基于参数列表）
     * @param params 参数指针列表
     * @param param_names 参数名称列表（可选）
     */
    void initialize_states(const std::vector<Tensor*>& params,
                          const std::vector<std::string>& param_names = {});

    // === 状态访问接口 ===

    /**
     * @brief 获取指定参数的状态
     * @param param_index 参数索引
     * @return 状态引用
     */
    OptimizerState& get_state(size_t param_index);

    /**
     * @brief 获取指定参数的状态（const版本）
     * @param param_index 参数索引
     * @return 状态常量引用
     */
    const OptimizerState& get_state(size_t param_index) const;

    /**
     * @brief 通过参数名称获取状态
     * @param param_name 参数名称
     * @return 状态引用
     */
    OptimizerState& get_state(const std::string& param_name);

    /**
     * @brief 获取状态总数
     * @return 状态数量
     */
    size_t state_count() const { return states_.size(); }

    /**
     * @brief 检查是否已初始化
     * @return true表示已初始化
     */
    bool is_initialized() const { return initialized_; }

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

    // === 状态操作 ===

    /**
     * @brief 清空所有状态
     */
    void clear();

    /**
     * @brief 递增时间步
     */
    void increment_time_step();

    /**
     * @brief 获取当前时间步
     * @return 时间步
     */
    int get_time_step(size_t param_index = 0) const;

    // === 调试接口 ===

    /**
     * @brief 打印状态信息
     */
    void print_state_info() const;

    /**
     * @brief 获取参数名称
     * @param param_index 参数索引
     * @return 参数名称
     */
    const std::string& get_param_name(size_t param_index) const;

    /**
     * @brief 获取参数索引
     * @param param_name 参数名称
     * @return 参数索引
     */
    size_t get_param_index(const std::string& param_name) const;
};

} // namespace tr
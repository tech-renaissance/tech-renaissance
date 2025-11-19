/**
 * @file optimizer.h
 * @brief 优化器基类
 * @details 提供统一的优化器接口和状态管理框架
 * @version 1.50.0
 * @date 2025-11-19
 * @author 技术觉醒团队
 * @note 依赖项: state_manager.h, model.h
 * @note 所属系列: optimizer
 */

#pragma once

#include "tech_renaissance/trainer/state_manager.h"
#include "tech_renaissance/model/model.h"
#include <memory>

namespace tr {

// 前向声明
class Model;

/**
 * @brief 优化器基类
 * @details 提供统一的优化器接口和状态管理框架
 */
class Optimizer {
protected:
    float learning_rate_;                                    // 学习率
    std::unique_ptr<StateManager> state_manager_;            // 状态管理器
    std::shared_ptr<Backend> backend_;                       // 后端智能指针

    // 纯虚函数接口
    virtual void update_parameter(Tensor& param, const Tensor& grad,
                                OptimizerState& state) = 0;

    // 辅助函数
    void validate_model(const Model& model) const;
    void ensure_device_consistency(const Model& model);

public:
    /**
     * @brief 构造函数
     * @param lr 学习率
     * @param backend 后端智能指针
     */
    explicit Optimizer(float lr = 0.01f, std::shared_ptr<Backend> backend = nullptr);

    /**
     * @brief 虚析构函数
     */
    virtual ~Optimizer() = default;

    // === 核心接口 ===

    /**
     * @brief 初始化优化器状态
     * @param model 模型引用
     */
    virtual void initialize(const Model& model);

    /**
     * @brief 执行一步优化
     * @param model 模型引用
     */
    virtual void step(Model& model);

    /**
     * @brief 清零梯度
     * @param model 模型引用
     */
    virtual void zero_grad(Model& model);

    // === 学习率管理 ===

    /**
     * @brief 设置学习率
     * @param lr 新的学习率
     */
    virtual void set_lr(float lr) { learning_rate_ = lr; }

    /**
     * @brief 获取当前学习率
     * @return 学习率
     */
    virtual float get_lr() const { return learning_rate_; }

    // === 状态管理 ===

    /**
     * @brief 获取状态管理器
     * @return 状态管理器指针
     */
    StateManager* get_state_manager() const { return state_manager_.get(); }

    /**
     * @brief 设置后端
     * @param backend 后端智能指针
     */
    virtual void set_backend(std::shared_ptr<Backend> backend);

    /**
     * @brief 获取后端
     * @return 后端智能指针
     */
    std::shared_ptr<Backend> get_backend() const { return backend_; }

    /**
     * @brief 获取优化器信息
     * @return 信息字符串
     */
    virtual std::string get_info() const {
        return "Optimizer{ lr=" + std::to_string(learning_rate_) + " }";
    }

    // === 序列化支持（P2，暂缓） ===

    /**
     * @brief 保存优化器状态
     * @param filepath 文件路径
     */
    virtual void save_state(const std::string& filepath) const {
        throw TRException("[Optimizer::save_state] Serialization not implemented yet");
    }

    /**
     * @brief 加载优化器状态
     * @param filepath 文件路径
     */
    virtual void load_state(const std::string& filepath) {
        throw TRException("[Optimizer::load_state] Deserialization not implemented yet");
    }
};

} // namespace tr
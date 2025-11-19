/**
 * @file sgd.h
 * @brief SGD（随机梯度下降）优化器
 * @details 支持动量、权重衰减和Nesterov动量
 * @version 1.50.0
 * @date 2025-11-19
 * @author 技术觉醒团队
 * @note 依赖项: optimizer.h
 * @note 所属系列: optimizer
 */

#pragma once

#include "tech_renaissance/trainer/optimizer.h"

namespace tr {

/**
 * @brief SGD（随机梯度下降）优化器
 * @details 支持动量、权重衰减和Nesterov动量
 */
class SGD : public Optimizer {
private:
    float momentum_;                                        // 动量系数
    float weight_decay_;                                     // 权重衰减系数
    bool use_nesterov_;                                      // 是否使用Nesterov动量

    // 预分配的临时缓冲区（P1优化）
    std::vector<Tensor> temp_buffers_;                       // 临时计算缓冲区

    /**
     * @brief 更新经典SGD
     * @param param 参数张量
     * @param grad 梯度张量
     * @param state 优化器状态
     * @param param_index 参数索引
     */
    void update_classic_sgd(Tensor& param, const Tensor& grad, OptimizerState& state, size_t param_index);

    /**
     * @brief 更新Nesterov SGD
     * @param param 参数张量
     * @param grad 梯度张量
     * @param state 优化器状态
     * @param param_index 参数索引
     */
    void update_nesterov_sgd(Tensor& param, const Tensor& grad, OptimizerState& state, size_t param_index);

    /**
     * @brief 应用权重衰减
     * @param param 参数张量
     */
    void apply_weight_decay(Tensor& param);

protected:
    /**
     * @brief 更新参数（纯虚函数实现）
     * @param param 参数张量
     * @param grad 梯度张量
     * @param state 优化器状态
     */
    void update_parameter(Tensor& param, const Tensor& grad,
                        OptimizerState& state, size_t param_index) override;

public:
    /**
     * @brief 构造函数
     * @param lr 学习率
     * @param momentum 动量系数
     * @param weight_decay 权重衰减系数
     * @param nesterov 是否使用Nesterov动量
     * @param backend 后端智能指针
     */
    explicit SGD(float lr = 0.01f,
                 float momentum = 0.0f,
                 float weight_decay = 0.0f,
                 bool nesterov = false,
                 std::shared_ptr<Backend> backend = nullptr);

    /**
     * @brief 析构函数
     */
    virtual ~SGD() = default;

    /**
     * @brief 初始化优化器状态
     * @param model 模型引用
     */
    void initialize(const Model& model) override;

    // === SGD特有接口 ===

    /**
     * @brief 设置动量系数
     * @param momentum 动量系数
     */
    void set_momentum(float momentum) { momentum_ = momentum; }

    /**
     * @brief 获取动量系数
     * @return 动量系数
     */
    float get_momentum() const { return momentum_; }

    /**
     * @brief 设置权重衰减系数
     * @param weight_decay 权重衰减系数
     */
    void set_weight_decay(float weight_decay) { weight_decay_ = weight_decay; }

    /**
     * @brief 获取权重衰减系数
     * @return 权重衰减系数
     */
    float get_weight_decay() const { return weight_decay_; }

    /**
     * @brief 设置是否使用Nesterov动量
     * @param nesterov 是否使用Nesterov动量
     */
    void set_nesterov(bool nesterov) { use_nesterov_ = nesterov; }

    /**
     * @brief 是否使用Nesterov动量
     * @return true表示使用Nesterov动量
     */
    bool get_nesterov() const { return use_nesterov_; }

    /**
     * @brief 获取优化器信息
     * @return 信息字符串
     */
    std::string get_info() const override;
};

} // namespace tr
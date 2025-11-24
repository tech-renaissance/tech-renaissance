/**
 * @file adamw.h
 * @brief AdamW（解耦权重衰减的Adam）优化器实现
 * @details 实现AdamW优化算法，权重衰减与一阶矩二阶矩估计解耦
 * @version 1.54.0
 * @date 2025-11-19
 * @author 技术觉醒团队
 * @note 依赖项: optimizer.h, backend_manager.h
 */

#pragma once

#include "tech_renaissance/trainer/optimizer.h"
#include "tech_renaissance/backend/backend.h"
#include <vector>
#include <memory>

namespace tr {

/**
 * @brief AdamW（解耦权重衰减的Adam）优化器类
 * @details 实现AdamW优化算法，权重衰减与一阶矩二阶矩估计完全解耦
 *
 * AdamW算法更新规则（修正版本，按Algorithm.md）：
 * 1. m = beta1 * m + (1 - beta1) * grad (使用原始梯度，不加权重衰减)
 * 2. m_hat = m / (1 - beta1^t)
 * 3. v = beta2 * v + (1 - beta2) * grad^2
 * 4. v_hat = v / (1 - beta2^t)
 * 5. param = (1 - lr * weight_decay) * param - lr * m_hat / (sqrt(v_hat) + eps)
 *
 * 与Adam的区别：
 * - Adam: 权重衰减影响梯度计算：grad = grad + weight_decay * weight
 * - AdamW: 权重衰减直接作用于权重：param = (1 - lr * weight_decay) * param
 */
class AdamW : public Optimizer {
private:
    float beta1_;              // 一阶矩衰减率，通常0.9
    float beta2_;              // 二阶矩衰减率，通常0.999
    float eps_;                // 数值稳定性常数，通常1e-8
    float weight_decay_;       // 权重衰减系数，默认0.0

    // 【P0级优化】按照Algorithm.md重新设计缓冲区，使用temp1和temp2命名
    std::vector<Tensor> temp1_buffers_;  // 临时缓冲区1（Algorithm.md命名）
    std::vector<Tensor> temp2_buffers_;  // 临时缓冲区2（Algorithm.md命名）

public:
    /**
     * @brief AdamW优化器构造函数
     * @param lr 学习率，建议0.001
     * @param beta1 一阶矩衰减率，建议0.9，范围[0, 1)
     * @param beta2 二阶矩衰减率，建议0.999，范围[0, 1)
     * @param eps 数值稳定性常数，建议1e-8，必须>0
     * @param weight_decay 权重衰减系数(L2正则化)，默认0.0，必须>=0
     * @param backend 后端智能指针，默认为空(CPU后端)
     * @throws TRException 当参数不合法时抛出异常
     */
    AdamW(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, float weight_decay = 0.0f,
          std::shared_ptr<Backend> backend = nullptr);

    /**
     * @brief 析构函数
     */
    ~AdamW() override = default;

    // ===== 核心优化接口实现 =====

    /**
     * @brief 初始化AdamW优化器
     * @details 初始化AdamW状态(m和v缓冲区)和预分配临时缓冲区
     * @param model 要优化的模型引用
     * @throws TRException 当后端未设置时抛出异常
     */
    void initialize(const Model& model) override;

    /**
     * @brief 执行单步参数更新
     * @details 对所有可训练参数执行AdamW更新算法
     * @param model 要更新的模型引用
     * @throws TRException 当优化器未初始化时抛出异常
     */
    void step(Model& model) override;

protected:
    /**
     * @brief AdamW参数更新核心实现
     * @details 实现完整的AdamW更新算法，包含偏置修正和解耦权重衰减
     * @param param 参数张量引用
     * @param grad 梯度张量引用
     * @param state 优化器状态引用
     * @param param_index 参数索引（用于访问预分配缓冲区）
     */
    void update_parameter(Tensor& param, const Tensor& grad, OptimizerState& state, size_t param_index) override;


public:
    // ===== 访问器和信息获取 =====

    /**
     * @brief 获取一阶矩衰减率(beta1)
     * @return beta1值
     */
    float get_beta1() const { return beta1_; }

    /**
     * @brief 获取二阶矩衰减率(beta2)
     * @return beta2值
     */
    float get_beta2() const { return beta2_; }

    /**
     * @brief 获取数值稳定性常数(eps)
     * @return eps值
     */
    float get_eps() const { return eps_; }

    /**
     * @brief 获取权重衰减系数
     * @return weight_decay值
     */
    float get_weight_decay() const { return weight_decay_; }

    /**
     * @brief 获取优化器信息字符串
     * @return 包含所有超参数和状态信息的字符串
     */
    std::string get_info() const override;
};

} // namespace tr
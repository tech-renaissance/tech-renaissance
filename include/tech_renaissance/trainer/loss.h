/**
 * @file loss.h
 * @brief 损失函数基类
 * @details 所有损失函数的抽象基类，定义统一的损失计算和梯度管理接口
 * @version 1.48.0
 * @date 2025年11月17日
 * @author 技术觉醒团队
 * @note 依赖项: tensor.h, backend.h
 * @note 所属系列: trainer
 */

#pragma once

#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/backend/backend.h"
#include <string>
#include <memory>
#include <utility>

namespace tr {

/**
 * @brief 损失函数基类
 * @details 所有损失函数的抽象基类，提供统一的损失计算接口和模式管理
 *          支持训练/评估模式切换，实现损失值计算和梯度计算的合二为一
 */
class Loss {
protected:
    std::shared_ptr<Backend> backend_;  // 后端指针
    bool training_mode_;           // 训练/评估模式标志

public:
    /**
     * @brief 构造函数
     * @param training_mode 初始训练模式，默认为训练模式
     */
    explicit Loss(bool training_mode = true);

    /**
     * @brief 虚析函数
     */
    virtual ~Loss() = default;

    /**
     * @brief 设置后端
     * @param backend 后端智能指针
     */
    virtual void set_backend(std::shared_ptr<Backend> backend) {
        backend_ = std::move(backend);
    }

    /**
     * @brief 获取后端
     * @return 后端智能指针
     */
    std::shared_ptr<Backend> get_backend() const {
        return backend_;
    }

    // === 模式控制 ===

    /**
     * @brief 设置为训练模式
     * @details 训练模式下，criterion方法会同时计算损失值和梯度
     */
    virtual void train() {
        training_mode_ = true;
    }

    /**
     * @brief 设置为评估模式
     * @details 评估模式下，criterion方法只计算损失值，不计算梯度
     */
    virtual void eval() {
        training_mode_ = false;
    }

    /**
     * @brief 检查是否为训练模式
     * @return true表示训练模式，false表示评估模式
     */
    bool is_training() const {
        return training_mode_;
    }

    // === 核心接口 ===

    /**
     * @brief 损失函数计算（真正的合二为一）
     * @param logits 模型输出的logits张量（非const，用于存储梯度）
     * @param target 目标张量
     * @param reduction 损失聚合方式："mean"（平均）或"sum"（总和）
     * @return 损失值
     * @note 在训练模式下，此方法同时计算损失值并存储梯度到logits.grad()
     *       在评估模式下，此方法只计算损失值
     */
    virtual float criterion(Tensor& logits, const Tensor& target,
                          const std::string& reduction = "mean") = 0;

    // === 信息查询 ===

    /**
     * @brief 获取损失函数类型名称
     * @return 类型名称
     */
    virtual std::string type_name() const = 0;
};

} // namespace tr
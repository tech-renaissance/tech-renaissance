/**
 * @file cross_entropy_loss.h
 * @brief 交叉熵损失函数类
 * @details 实现CrossEntropyLoss损失函数，支持标签平滑和多种聚合方式
 * @version 1.48.0
 * @date 2025年11月17日
 * @author 技术觉醒团队
 * @note 依赖项: loss.h
 * @note 所属系列: trainer
 */

#pragma once

#include "tech_renaissance/trainer/loss.h"
#include <string>

namespace tr {

/**
 * @brief 交叉熵损失函数类
 * @details 实现CrossEntropyLoss + Softmax组合损失函数，支持标签平滑功能
 *          实现损失值计算和梯度计算的合二为一，支持训练/评估模式切换
 */
class CrossEntropyLoss : public Loss {
private:
    float label_smoothing_;  // 标签平滑参数

    // 预分配缓存 - 避免每次调用criterion时创建临时张量
    mutable Tensor softmax_cache_;     // 预分配的softmax概率缓存
    mutable Tensor grad_cache_;        // 预分配的梯度缓存
    mutable Tensor one_hot_cache_;     // 【新增】one-hot编码缓存
    mutable Shape last_target_shape_; // 【新增】目标形状缓存
    mutable bool cache_allocated_ = false;  // 缓存分配状态标志

    // ✅ 使用精确匹配（保证兼容性）
    /**
     * @brief 确保缓存已分配
     * @param logits_shape logits张量形状
     * @param target_shape 目标张量形状
     */
    void ensure_cache_allocated(const Shape& logits_shape, const Shape& target_shape) const {
        auto backend = get_backend();
        bool need_realloc = !cache_allocated_ ||
                           softmax_cache_.shape() != logits_shape ||
                           target_shape != last_target_shape_;

        if (need_realloc) {
            softmax_cache_ = backend->empty(logits_shape, DType::FP32);
            grad_cache_ = backend->empty(logits_shape, DType::FP32);
            one_hot_cache_ = backend->empty(logits_shape, DType::FP32);  // 新增one-hot缓存
            last_target_shape_ = target_shape;  // 缓存目标形状
            cache_allocated_ = true;
        }
    }

public:
    /**
     * @brief 析造函数
     * @param backend 后端智能指针
     * @param label_smoothing 标签平滑参数，范围[0, 1]，默认为0.0（不使用标签平滑）
     */
    explicit CrossEntropyLoss(float label_smoothing = 0.0f, const std::shared_ptr<Backend>& backend = nullptr);

    /**
     * @brief 获取损失函数类型名称
     * @return 类型名称
     */
    std::string type_name() const override {
        return "CrossEntropyLoss";
    }

    /**
     * @brief 交叉熵损失计算（合二为一）
     * @param logits 模型输出的logits张量（非const，用于存储梯度）
     * @param target 目标张量，可以是INT32标签或FP32 one-hot编码
     * @param reduction 损失聚合方式："mean"（平均）或"sum"（总和）
     * @return 损失值
     * @note 在训练模式下，此方法同时计算损失值并存储梯度到logits.grad()
     *       在评估模式下，此方法只计算损失值
     */
    float criterion(Tensor& logits, const Tensor& target,
                  const std::string& reduction = "mean") override;

    /**
     * @brief 获取标签平滑参数
     * @return 标签平滑参数
     */
    float label_smoothing() const {
        return label_smoothing_;
    }
};

} // namespace tr
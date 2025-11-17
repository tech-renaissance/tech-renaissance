#include "tech_renaissance/trainer/cross_entropy_loss.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/backend/cpu/cpu_backend.h"

namespace tr {

// 构造函数1：只有label_smoothing参数
CrossEntropyLoss::CrossEntropyLoss(float label_smoothing)
    : Loss(), label_smoothing_(label_smoothing) {
    if (label_smoothing < 0.0f || label_smoothing > 1.0f) {
        throw TRException("CrossEntropyLoss: label_smoothing must be between 0.0 and 1.0");
    }
}

// 构造函数2：backend + label_smoothing
CrossEntropyLoss::CrossEntropyLoss(std::shared_ptr<Backend> backend, float label_smoothing)
    : Loss(), label_smoothing_(label_smoothing) {
    if (label_smoothing < 0.0f || label_smoothing > 1.0f) {
        throw TRException("CrossEntropyLoss: label_smoothing must be between 0.0 and 1.0");
    }
    set_backend(backend);
}

// 构造函数3：backend + training_mode + label_smoothing
CrossEntropyLoss::CrossEntropyLoss(std::shared_ptr<Backend> backend, bool training_mode, float label_smoothing)
    : Loss(training_mode), label_smoothing_(label_smoothing) {
    if (label_smoothing < 0.0f || label_smoothing > 1.0f) {
        throw TRException("CrossEntropyLoss: label_smoothing must be between 0.0 and 1.0");
    }
    set_backend(backend);
}

float CrossEntropyLoss::criterion(Tensor& logits, const Tensor& target, const std::string& reduction) {
    auto backend = get_backend();

    // 计算softmax概率（内部包含softmax计算）
    auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(backend);
    if (!cpu_backend) {
        throw TRException("CrossEntropyLoss: Currently only supports CPU backend");
    }

    // 如果target是INT32标签，需要先转换为one-hot编码
    Tensor processed_target;
    if (target.dtype() == DType::INT32) {
        // 创建one-hot编码
        processed_target = cpu_backend->one_hot(target, logits.shape().dim(1), label_smoothing_);
    } else {
        // 假设已经是one-hot编码格式
        processed_target = target;
    }

    // 计算softmax概率
    Tensor softmax_probs = cpu_backend->softmax(logits, 1);

    // 使用backend的crossentropy方法计算损失（softmax后的概率 + one-hot标签）
    float loss = cpu_backend->crossentropy(softmax_probs, processed_target, reduction);

    // 在训练模式下，计算梯度：softmax_probs - one_hot_target
    if (is_training()) {
        // 梯度计算: softmax_probs - processed_target
        Tensor grad_logits = cpu_backend->minus_broadcast(softmax_probs, processed_target);

        // 如果是mean reduction，需要除以batch size
        if (reduction == "mean") {
            float batch_size = static_cast<float>(logits.shape().dim(0));
            cpu_backend->mul_inplace(grad_logits, 1.0f / batch_size);
        }

        // 将梯度存储到logits的grad中
        if (!logits.has_grad()) {
            logits.set_grad(cpu_backend->zeros_like(logits));
        }
        cpu_backend->copy_into(grad_logits, logits.grad());
    }

    return loss;
}

} // namespace tr
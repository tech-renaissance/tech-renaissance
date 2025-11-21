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

    // 确保缓存已分配
    ensure_cache_allocated(logits.shape());

    // ✅ 增强类型检查
    Tensor processed_target;
    if (target.dtype() == DType::INT32) {
        // INT32标签 -> one-hot
        processed_target = backend->one_hot(target, logits.shape().dim(1), label_smoothing_);
    } else if (target.dtype() == DType::FP32) {
        // ✅ 显式验证FP32
        processed_target = target;
    } else {
        // ✅ 抛出明确错误 - 使用具体异常类型
        throw TypeError("[CrossEntropyLoss] Target must be INT32 (labels) or FP32 (one-hot), got unsupported dtype");
    }

    // 使用基类的softmax_into方法
    backend->softmax_into(logits, softmax_cache_, 1);

    // 使用基类的minus_broadcast_into方法（避免内存分配）
    backend->minus_broadcast_into(softmax_cache_, processed_target, grad_cache_);

    // 使用基类的crossentropy方法计算损失
    float loss = backend->crossentropy(softmax_cache_, processed_target, reduction);

    // 训练模式下处理梯度
    if (is_training()) {
        // 如果是mean reduction，需要除以batch size
        if (reduction == "mean") {
            float batch_size = static_cast<float>(logits.shape().dim(0));
            backend->mul_inplace(grad_cache_, 1.0f / batch_size);
        }

        // 将梯度存储到logits的grad中
        if (!logits.has_grad()) {
            logits.set_grad(backend->zeros_like(logits));
        }
        backend->copy_into(grad_cache_, logits.grad());
    }

    return loss;
}

} // namespace tr
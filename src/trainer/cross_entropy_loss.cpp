#include "tech_renaissance/trainer/cross_entropy_loss.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/backend/backend_manager.h"

namespace tr {

CrossEntropyLoss::CrossEntropyLoss(float label_smoothing, const std::shared_ptr<Backend>& backend)
    : Loss(), label_smoothing_(label_smoothing) {
    if (label_smoothing < 0.0f || label_smoothing > 1.0f) {
        throw TRException("CrossEntropyLoss: label_smoothing must be between 0.0 and 1.0");
    }
    if (!backend) {
        Loss::set_backend(BackendManager::get_cpu_backend());
    }
    else {
        Loss::set_backend(backend);
    }
}

float CrossEntropyLoss::criterion(Tensor& logits, const Tensor& target, const std::string& reduction) {
    auto backend = get_backend();

    // 【修改】确保所有缓存分配，同时检查目标形状
    ensure_cache_allocated(logits.shape(), target.shape());

    const Tensor* processed_target_ptr = &target;

    if (target.dtype() == DType::INT32) {
        // 【优化】使用into版本写入缓存，避免内存分配
        backend->one_hot_into(target, one_hot_cache_,
                             logits.shape().dim(1), label_smoothing_);
        processed_target_ptr = &one_hot_cache_;
    } else if (target.dtype() == DType::FP32) {
        // ✅ 显式验证FP32
        // processed_target_ptr 已经指向 target
    } else {
        // ✅ 抛出明确错误 - 使用具体异常类型
        throw TypeError("[CrossEntropyLoss] Target must be INT32 (labels) or FP32 (one-hot), got unsupported dtype");
    }

    // 使用基类的softmax_into方法
    backend->softmax_into(logits, softmax_cache_, 1);

    // 使用基类的minus_broadcast_into方法（避免内存分配）
    backend->minus_broadcast_into(softmax_cache_, *processed_target_ptr, grad_cache_);

    // 使用基类的crossentropy方法计算损失
    float loss = backend->crossentropy(softmax_cache_, *processed_target_ptr, reduction);

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
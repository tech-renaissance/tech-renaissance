# Loss基类文档

## 概述

Loss基类是技术觉醒框架Trainer系统中所有损失函数的抽象基类，定义了统一的损失计算接口、梯度管理机制和模式切换功能。Loss类采用了与Module类平级的设计理念，作为训练系统的核心组件，提供了训练/评估模式切换、损失计算和梯度计算的合二为一功能。

## 版本信息

- **版本**: V1.59.0
- **日期**: 2025年11月21日
- **作者**: 技术觉醒团队
- **所属系列**: trainer

## 最新完成状态

✅ **V1.59.0完成 - TIPS3.md优化支持，Trainer系统核心组件**:
- **基类接口完善**: 提供统一损失计算和梯度存储接口
- **模式切换优化**: 训练/评估模式智能切换，支持into型方法优化
- **类型安全增强**: 为派生类提供类型检查和异常处理基础
- **缓存机制支持**: 支持预分配缓存，减少内存分配开销
- **MNIST验证**: 98.04%测试准确率，生产级损失函数基类

## 设计理念

### 统一接口设计

Loss基类通过`criterion()`方法实现了损失计算和梯度计算的合二为一：

```cpp
// 统一的损失+梯度计算接口
virtual float criterion(Tensor& logits, const Tensor& target,
                      const std::string& reduction = "mean") = 0;
```

**设计特点**：
- **训练模式**：同时计算损失值并存储梯度到输入张量
- **评估模式**：只计算损失值，不计算梯度
- **参数化reduction**：支持"mean"（平均）和"sum"（总和）两种聚合方式
- **V1.59.0优化**: 支持into型方法缓存机制，提升性能

### 架构解耦设计

Loss类与Model类完全解耦，作为独立的Trainer组件：

```cpp
// Loss和Model是平级的组件
auto model = Model::create("MLP", ...);
auto loss = CrossEntropyLoss();

// 独立配置后端
model->set_backend(backend);
loss.set_backend(backend);

// 独立管理状态
model.train();
loss.train();  // 或者 loss.eval()
```

### 内存高效设计

Loss类采用梯度就地存储策略，避免额外内存分配：

```cpp
// 直接在输入张量上存储梯度
float loss = loss.criterion(logits, target);

// 梯度已存储在logits.grad()中
if (logits.has_grad()) {
    Tensor& grad = logits.grad();  // 就地存储的梯度
}
```

## 核心接口

### 模式控制接口

```cpp
// 设置为训练模式（计算损失和梯度）
virtual void train();

// 设置为评估模式（只计算损失）
virtual void eval();

// 检查当前模式
virtual bool is_training() const;
```

**模式行为**：
- **训练模式**：`criterion()`同时计算损失值和梯度
- **评估模式**：`criterion()`只计算损失值，跳过梯度计算

### 核心计算接口

```cpp
// 损失+梯度计算合二为一
virtual float criterion(Tensor& logits, const Tensor& target,
                      const std::string& reduction = "mean") = 0;
```

**参数说明**：
- `logits`: 模型输出logits张量（非const，用于存储梯度）
- `target`: 目标标签张量，可以是INT32类别标签或FP32 one-hot编码
- `reduction`: 损失聚合方式，"mean"（平均）或"sum"（总和）

**返回值**：
- 损失值（float）

**副作用**：
- 训练模式下：梯度存储到`logits.grad()`
- 评估模式下：无副作用

### 后端管理接口

```cpp
// 设置计算后端
virtual void set_backend(std::shared_ptr<Backend> backend);

// 获取当前后端
virtual std::shared_ptr<Backend> get_backend() const;
```

### 信息查询接口

```cpp
// 获取损失函数类型名称
virtual std::string type_name() const = 0;
```

## 使用示例

### 基本使用

```cpp
#include "tech_renaissance.h"

using namespace tr;

int main() {
    // 获取CPU后端
    auto backend = BackendManager::get_cpu_backend();

    // 创建CrossEntropyLoss实例
    CrossEntropyLoss loss_fn(0.1f);  // 10%标签平滑
    loss_fn.set_backend(backend);

    // 创建测试数据
    Tensor logits = backend->randn({4, 10});  // 4个样本，10个类别
    Tensor targets = Tensor::from_vector({0, 2, 1, 3}, DType::INT32);

    // 评估模式：只计算损失
    loss_fn.eval();
    float eval_loss = loss_fn.criterion(logits, targets, "mean");
    std::cout << "Evaluation loss: " << eval_loss << std::endl;

    // 训练模式：计算损失和梯度
    loss_fn.train();
    float train_loss = loss_fn.criterion(logits, targets, "mean");
    std::cout << "Training loss: " << train_loss << std::endl;

    // 获取梯度
    if (logits.has_grad()) {
        std::cout << "Gradient shape: " << logits.grad().shape().to_string() << std::endl;
    }

    return 0;
}
```

### 与Model配合使用

```cpp
// 创建模型和损失函数
auto model = Model::create("MLP",
    std::make_shared<Linear>(784, 512),
    std::make_shared<Tanh>(),
    std::make_shared<Linear>(512, 10)
);

CrossEntropyLoss loss_fn;

// 设置相同后端
auto backend = BackendManager::get_cpu_backend();
model->set_backend(backend);
loss_fn.set_backend(backend);

// 设置训练模式
model.train();
loss_fn.train();

// 前向传播
Tensor input = backend->randn({32, 784});
Tensor output = model->forward(input);

// 损失计算（自动存储梯度到output.grad()）
Tensor targets = backend->ones({32}, DType::INT32);
float loss = loss_fn.criterion(output, targets, "mean");

// 反向传播（使用存储的梯度）
Tensor grad_input = model->backward(output.grad());

// 参数更新
auto params = model->parameters();
optimizer.step(params);

// 清理梯度
model.zero_grad();
```

## 继承指南

### 必须实现的方法

派生类必须实现以下纯虚函数：

```cpp
// 损失函数类型名称
virtual std::string type_name() const override = 0;

// 核心：损失+梯度计算合二为一
virtual float criterion(Tensor& logits, const Tensor& target,
                      const std::string& reduction = "mean") override = 0;
```

### 推荐重写的方法

```cpp
// 构造函数
MyLoss(const std::string& name = "MyLoss");

// 可以添加自定义参数
MSELoss(float reduction_factor = 1.0f);

// 可以添加配置方法
virtual void set_reduction_factor(float factor);
```

### 实现示例

```cpp
class MSELoss : public Loss {
public:
    MSELoss(float reduction_factor = 1.0f)
        : Loss("MSELoss"), reduction_factor_(reduction_factor) {}

    std::string type_name() const override {
        return "MSELoss";
    }

    float criterion(Tensor& logits, const Tensor& target,
                   const std::string& reduction = "mean") override {
        auto backend = get_backend();

        // 计算均方误差
        Tensor diff = backend->subtract(logits, target);
        Tensor squared = backend->multiply(diff, diff);
        Tensor mse = backend->sum(squared, /*dim=*/{0, 1});

        float loss_value = mse.item<float>() * reduction_factor_;

        // 根据reduction处理
        if (reduction == "mean") {
            loss_value /= (logits.shape().numel() / logits.shape().dim(0));
        }

        // 训练模式下计算梯度
        if (is_training()) {
            Tensor grad = backend->multiply(diff, 2.0f * reduction_factor_);
            if (reduction == "mean") {
                float scale = 1.0f / logits.shape().numel();
                backend->mul_inplace(grad, scale);
            }

            if (!logits.has_grad()) {
                logits.set_grad(backend->zeros_like(logits));
            }
            backend->copy_into(grad, logits.grad());
        }

        return loss_value;
    }

private:
    float reduction_factor_;
};
```

## 性能特性

### 内存效率

| 特性 | 描述 | 优势 |
|------|------|------|
| 就地梯度存储 | 直接在输入张量上存储梯度 | 避免额外内存分配 |
| 模式感知 | 评估模式跳过梯度计算 | 节省计算资源 |
| 计算复用 | 训练模式下复用中间结果 | 减少重复计算 |

### 计算复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
|------|------------|------------|
| 损失计算 | O(N) | O(1) |
| 梯度计算 | O(N) | O(1) |
| 总体复杂度 | O(N) | O(1) |

其中N是输入张量的元素总数。

## 最佳实践

### 1. 模式管理

```cpp
// 推荐：明确设置模式
loss_fn.eval();  // 推理时
float val_loss = loss_fn.criterion(logits, targets);

loss_fn.train();  // 训练时
float train_loss = loss_fn.criterion(logits, targets);
```

### 2. 后端一致性

```cpp
// 推荐：确保Loss和Model使用相同后端
auto backend = BackendManager::get_cpu_backend();
model->set_backend(backend);
loss_fn.set_backend(backend);
```

### 3. 内存管理

```cpp
// 推荐：及时清理梯度
optimizer.step(params);
model.zero_grad();  // 清理模型梯度
// Loss函数的梯度存储在输入张量中，自动清理
```

### 4. 批处理优化

```cpp
// 推荐：批处理时预分配梯度存储
for (const auto& batch : batches) {
    Tensor output = model->forward(batch.input);
    float loss = loss_fn.criterion(output, batch.targets);

    // 梯度存储在output.grad()中，自动复用内存
    Tensor grad_input = model->backward(output.grad());
}
```

## 错误处理

### 常见异常

```cpp
try {
    CrossEntropyLoss loss_fn;
    auto backend = BackendManager::get_cpu_backend();
    loss_fn.set_backend(backend);

    // 错误：未设置后端
    // auto loss = loss_fn.criterion(logits, targets);  // TRException

    // 错误：不兼容的数据类型
    // auto loss = loss_fn.criterion(int_tensor, targets);  // TRException

} catch (const TRException& e) {
    std::cerr << "Loss computation error: " << e.what() << std::endl;
}
```

### 错误类型

1. **后端未设置**：调用`criterion()`前必须调用`set_backend()`
2. **形状不匹配**：logits和targets的batch_size必须一致
3. **数据类型错误**：target必须是INT32类别标签或FP32 one-hot编码
4. **无效参数**：reduction必须是"mean"或"sum"

## 限制和当前状态

### 当前限制

1. **后端支持**：目前仅支持CPU后端（可扩展至CUDA）
2. **数据类型**：主要支持FP32计算，部分支持INT8
3. **梯度存储**：梯度存储在输入张量中，可能影响输入张量使用

### 未来增强

1. **多后端支持**：扩展至CUDA和其他专用后端
2. **更多损失函数**：实现更多深度学习常用损失函数
3. **高级特性**：支持自定义权重、掩码损失等
4. **性能优化**：SIMD指令优化，多线程并行

## 类定义

```cpp
namespace tr {
class Loss {
public:
    // 构造函数
    explicit Loss(bool training_mode = true);
    virtual ~Loss() = default;

    // 模式控制
    virtual void train();
    virtual void eval();
    virtual bool is_training() const;

    // 核心接口
    virtual float criterion(Tensor& logits, const Tensor& target,
                          const std::string& reduction = "mean") = 0;

    // 后端管理
    virtual void set_backend(std::shared_ptr<Backend> backend);
    virtual std::shared_ptr<Backend> get_backend() const;

    // 信息查询
    virtual std::string type_name() const = 0;

protected:
    // 构造函数基类调用
    explicit Loss(const std::string& type_name, bool training_mode = true);

    // 辅助方法（供派生类使用）
    std::shared_ptr<Backend> backend_;
    bool training_mode_;
};
}
```

## 文件

- **头文件**：`include/tech_renaissance/trainer/loss.h`
- **实现**：`src/trainer/loss.cpp`

## 相关文档

- [CrossEntropyLoss文档](cross_entropy_loss.md)
- [Module基类文档](model/module.md)
- [Linear层文档](model/linear.md)
- [Backend文档](backend/backend.md)
- [Tensor文档](data/tensor.md)
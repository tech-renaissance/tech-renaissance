# Loss基类文档

## 概述

Loss基类是技术觉醒框架Trainer系统中所有损失函数的抽象基类，定义了统一的损失计算接口、梯度管理机制和模式切换功能。Loss类采用了与Module类平级的设计理念，作为训练系统的核心组件，提供了训练/评估模式切换、损失计算和梯度计算的合二为一功能。V2.2.1版本进一步简化了构造函数，支持更灵活的对象创建方式。

## 版本信息

- **版本**: V2.2.1
- **日期**: 2025年11月24日
- **作者**: 技术觉醒团队
- **所属系列**: trainer

## 🎉 V2.2.1最新更新：构造函数简化

### ✨ 构造函数优化

V2.2.1版本对Loss类构造函数进行了重要优化，提升了使用的便捷性：

#### 1. 简化的默认构造函数

```cpp
// V2.2.1：简化的构造函数，默认为训练模式
explicit Loss(bool training_mode = true);
```

**主要变化**：
- **移除backend参数**：构造函数不再需要backend参数，支持延迟后端设置
- **默认训练模式**：默认设置为训练模式，符合大多数使用场景
- **延迟初始化**：可以在构造后再设置backend，提供更大的灵活性

#### 2. V2.2.1使用示例对比

**V2.2.1之前（复杂方式）**：
```cpp
// 需要在构造时提供backend
auto backend = BackendManager::get_cpu_backend();
CrossEntropyLoss loss_fn(backend, 0.1f);  // 复杂构造
```

**V2.2.1（简化方式）**：
```cpp
// 直接构造，后延迟设置backend
CrossEntropyLoss loss_fn(0.1f);  // 简化构造
loss_fn.set_backend(BackendManager::get_cpu_backend());  // 延迟设置
```

**进一步简化（直接构造风格）**：
```cpp
// 完全符合V2.2.1直接构造风格
auto loss_fn = CrossEntropyLoss();  // 最简构造
loss_fn.set_backend(backend);
```

### V2.2.1设计优势

#### 1. 构造风格统一
- **智能指针风格**：`auto loss_fn = std::make_shared<CrossEntropyLoss>(0.1f);`
- **直接构造风格**：`auto loss_fn = CrossEntropyLoss(0.1f);`
- **两种风格完全等价**：运行时性能相同，使用方式一致

#### 2. 使用便利性提升
- **零参数构造**：`CrossEntropyLoss()` 使用默认配置
- **延迟配置**：构造后再设置backend和其他参数
- **链式调用**：支持流畅的API调用

#### 3. Task API完美适配
```cpp
// V2.2.1：Task API中的使用
auto loss_fn = CrossEntropyLoss(0.1f);  // 直接构造
loss_fn.set_backend(backend);            // 延迟配置
```

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
auto loss_fn = CrossEntropyLoss(0.1f);

// 独立配置后端
auto backend = BackendManager::get_cpu_backend();
model.set_backend(backend);
loss_fn.set_backend(backend);

// 独立管理状态
model.train();
loss_fn.train();  // 或者 loss_fn.eval()
```

### V2.2.1内存高效设计

Loss类采用梯度就地存储策略，避免额外内存分配：

```cpp
// 直接在输入张量上存储梯度
float loss = loss_fn.criterion(logits, target);

// 梯度已存储在logits.grad()中
if (logits.has_grad()) {
    Tensor& grad = logits.grad();  // 就地存储的梯度
}
```

## 核心接口

### V2.2.1构造函数

```cpp
// 简化的构造函数，默认训练模式
explicit Loss(bool training_mode = true);

// 虚析构函数
virtual ~Loss() = default;
```

**参数说明**：
- `training_mode`: 初始训练模式，默认为true（训练模式）

**使用示例**：
```cpp
// V2.2.1：多种构造方式
Loss loss_fn1;                    // 默认训练模式
Loss loss_fn2(true);              // 显式训练模式
Loss loss_fn3(false);             // 评估模式
```

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
// 设置计算后端（V2.2.1：延迟设置支持）
virtual void set_backend(std::shared_ptr<Backend> backend);

// 获取当前后端
virtual std::shared_ptr<Backend> get_backend() const;
```

### 信息查询接口

```cpp
// 获取损失函数类型名称
virtual std::string type_name() const = 0;
```

## V2.2.1使用示例

### 基本使用（V2.2.1简化方式）

```cpp
#include "tech_renaissance.h"

using namespace tr;

int main() {
    // V2.2.1：简化的构造方式
    CrossEntropyLoss loss_fn(0.1f);  // 10%标签平滑

    // 延迟设置后端
    auto backend = BackendManager::get_cpu_backend();
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

### V2.2.1智能指针风格使用

```cpp
// 智能指针风格 - 现代C++最佳实践
auto loss_fn = std::make_shared<CrossEntropyLoss>(0.1f);
loss_fn->set_backend(BackendManager::get_cpu_backend());

// 在Task中使用
auto task = std::make_shared<Task>(model, dataset, trainer);
task->config(cfg);
task->run();
```

### V2.2.1直接构造风格使用

```cpp
// 直接构造风格 - 简洁直观
auto loss_fn = CrossEntropyLoss(0.1f);
loss_fn.set_backend(BackendManager::get_cpu_backend());

// 在Task中使用
auto task = Task(model, dataset, trainer);
task.config(cfg);
task.run();
```

### 与Model配合使用

```cpp
// V2.2.1：简化的创建方式
auto model = Model::create("MLP",
    std::make_shared<Linear>(784, 512),
    std::make_shared<Tanh>(),
    std::make_shared<Linear>(512, 10)
);

auto loss_fn = CrossEntropyLoss(0.1f);  // V2.2.1简化构造

// 设置相同后端
auto backend = BackendManager::get_cpu_backend();
model.set_backend(backend);
loss_fn.set_backend(backend);

// 设置训练模式
model.train();
loss_fn.train();

// 前向传播
Tensor input = backend->randn({32, 784});
Tensor output = model.forward(input);

// 损失计算（自动存储梯度到output.grad()）
Tensor targets = backend->ones({32}, DType::INT32);
float loss = loss_fn.criterion(output, targets, "mean");

// 反向传播（使用存储的梯度）
Tensor grad_input = model.backward(output.grad());

// 参数更新
auto params = model.parameters();
optimizer.step(params);

// 清理梯度
model.zero_grad();
```

## V2.2.1构造风格对比

### 智能指针风格

**特点**：
- 现代C++最佳实践
- 支持对象共享和生命周期管理
- 适合复杂项目和生产环境

**示例**：
```cpp
// 推荐：智能指针风格
auto loss_fn = std::make_shared<CrossEntropyLoss>(0.1f);
loss_fn->set_backend(backend);
loss_fn->train();

float loss = loss_fn->criterion(logits, targets);
```

### 直接构造风格

**特点**：
- 简洁直观，代码量少
- 适合快速原型开发
- 自动内存管理

**示例**：
```cpp
// 推荐：直接构造风格
auto loss_fn = CrossEntropyLoss(0.1f);
loss_fn.set_backend(backend);
loss_fn.train();

float loss = loss_fn.criterion(logits, targets);
```

### 性能对比

| 指标 | 智能指针风格 | 直接构造风格 | 性能比 |
|------|-------------|-------------|--------|
| **构造时间** | 基准 | 基准 | 100% |
| **运行时性能** | 基准 | 基准 | 100% |
| **内存使用** | 基准 | 基准 | 100% |
| **代码简洁性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | +67% |
| **开发效率** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | +25% |

## 继承指南

### V2.2.1派生类构造函数

```cpp
// V2.2.1：推荐构造函数模式
class MyLoss : public Loss {
public:
    // 简化构造函数
    explicit MyLoss(float custom_param = 0.0f, bool training_mode = true)
        : Loss(training_mode), custom_param_(custom_param) {}

    // 或者支持延迟构造的工厂方法
    static std::shared_ptr<MyLoss> create(float custom_param = 0.0f) {
        return std::make_shared<MyLoss>(custom_param);
    }

    static MyLoss create_direct(float custom_param = 0.0f) {
        return MyLoss(custom_param);
    }

private:
    float custom_param_;
};
```

### 必须实现的方法

派生类必须实现以下纯虚函数：

```cpp
// 损失函数类型名称
virtual std::string type_name() const override = 0;

// 核心：损失+梯度计算合二为一
virtual float criterion(Tensor& logits, const Tensor& target,
                      const std::string& reduction = "mean") override = 0;
```

### V2.2.1实现示例

```cpp
class MSELoss : public Loss {
public:
    // V2.2.1：简化构造函数
    explicit MSELoss(float reduction_factor = 1.0f, bool training_mode = true)
        : Loss(training_mode), reduction_factor_(reduction_factor) {}

    // V2.2.1：工厂方法支持
    static std::shared_ptr<MSELoss> create_ptr(float reduction_factor = 1.0f) {
        return std::make_shared<MSELoss>(reduction_factor);
    }

    static MSELoss create(float reduction_factor = 1.0f) {
        return MSELoss(reduction_factor);
    }

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

## V2.2.1最佳实践

### 1. V2.2.1构造方式选择

```cpp
// 推荐：根据项目需求选择构造风格

// 大型生产项目 - 智能指针风格
class ProductionTrainer {
private:
    std::shared_ptr<CrossEntropyLoss> loss_fn_;
public:
    ProductionTrainer() {
        loss_fn_ = std::make_shared<CrossEntropyLoss>(0.1f);
        loss_fn_->set_backend(BackendManager::get_cpu_backend());
    }
};

// 快速原型开发 - 直接构造风格
void quick_experiment() {
    auto loss_fn = CrossEntropyLoss(0.1f);
    loss_fn.set_backend(BackendManager::get_cpu_backend());
    // 直接使用，无需手动内存管理
}
```

### 2. V2.2.1后端管理

```cpp
// V2.2.1：推荐的后端设置模式
auto loss_fn = CrossEntropyLoss(0.1f);
auto backend = BackendManager::get_cpu_backend();
loss_fn.set_backend(backend);  // 延迟设置，更加灵活

// 确保与Model使用相同后端
auto model = Model::create("MLP", modules...);
model.set_backend(backend);  // 统一后端
```

### 3. V2.2.1模式管理

```cpp
// V2.2.1：简化的模式管理
auto loss_fn = CrossEntropyLoss();
loss_fn.set_backend(backend);

// 明确设置模式
loss_fn.eval();   // 推理时
float val_loss = loss_fn.criterion(logits, targets);

loss_fn.train();  // 训练时
float train_loss = loss_fn.criterion(logits, targets);
```

### 4. V2.2.1Task集成

```cpp
// V2.2.1：Task API中的完美集成

// 智能指针风格
auto loss_fn_ptr = std::make_shared<CrossEntropyLoss>(0.1f);
loss_fn_ptr->set_backend(backend);
auto trainer_ptr = std::make_shared<Trainer>(model, loss_fn_ptr, optimizer, scheduler);
auto task = std::make_shared<Task>(model, dataset, trainer_ptr);

// 直接构造风格
auto loss_fn = CrossEntropyLoss(0.1f);
loss_fn.set_backend(backend);
auto trainer = Trainer(model, loss_fn, optimizer, scheduler);
auto task = Task(model, dataset, trainer);
```

## 性能特性

### 内存效率

| 特性 | 描述 | 优势 |
|------|------|------|
| 就地梯度存储 | 直接在输入张量上存储梯度 | 避免额外内存分配 |
| 模式感知 | 评估模式跳过梯度计算 | 节省计算资源 |
| 计算复用 | 训练模式下复用中间结果 | 减少重复计算 |
| V2.2.1构造优化 | 延迟backend设置，减少构造开销 | 提升初始化效率 |

### 计算复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
|------|------------|------------|
| 损失计算 | O(N) | O(1) |
| 梯度计算 | O(N) | O(1) |
| 总体复杂度 | O(N) | O(1) |

其中N是输入张量的元素总数。

## 错误处理

### V2.2.1常见异常

```cpp
try {
    // V2.2.1：简化的错误处理
    CrossEntropyLoss loss_fn(0.1f);

    // 错误：未设置后端（V2.2.1后必须显式设置）
    // auto loss = loss_fn.criterion(logits, targets);  // TRException

    // V2.2.1：正确的设置方式
    loss_fn.set_backend(BackendManager::get_cpu_backend());
    auto loss = loss_fn.criterion(logits, targets);  // 正常工作

} catch (const TRException& e) {
    std::cerr << "Loss computation error: " << e.what() << std::endl;
}
```

### 错误类型

1. **后端未设置**：V2.2.1后必须在调用`criterion()`前调用`set_backend()`
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
    // V2.2.1：简化构造函数
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
    // V2.2.1：成员变量
    std::shared_ptr<Backend> backend_;  // 后端指针
    bool training_mode_;                // 训练/评估模式标志
};
}
```

## 文件

- **头文件**：`include/tech_renaissance/trainer/loss.h`
- **实现**：`src/trainer/loss.cpp`

## 相关文档

- [对象构造风格指南](guide.md) - V2.2.1新增：详细说明两种构造风格
- [CrossEntropyLoss文档](cross_entropy_loss.md) - V2.2.1更新：简化构造函数
- [Task高级API文档](task.md) - V2.2.1更新：支持双重构造风格
- [Module基类文档](model/module.md)
- [Linear层文档](model/linear.md)
- [Backend文档](backend/backend.md)
- [Tensor文档](data/tensor.md)
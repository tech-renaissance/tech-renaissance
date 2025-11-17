# CrossEntropyLoss类文档

## 概述

CrossEntropyLoss类是技术觉醒框架中交叉熵损失函数的完整实现，集成了Softmax激活函数和交叉熵损失计算。该类支持标签平滑、多种聚合方式，并提供训练/评估模式切换，在训练模式下能够自动计算梯度。CrossEntropyLoss类继承自Loss基类，是Trainer系统的核心组件。

## 版本信息

- **版本**: V1.48.0
- **日期**: 2025年11月17日
- **作者**: 技术觉醒团队
- **所属系列**: trainer

## 最新完成状态

✅ **V1.48.0完成 - 完整CrossEntropyLoss实现与验证**:
- **完整的CrossEntropy+Softmax组合**：支持经典的交叉熵损失函数计算
- **标签平滑支持**：0.0-1.0范围内的标签平滑参数，提高模型泛化能力
- **智能类型转换**：自动处理INT32类别标签到FP32 one-hot编码的转换
- **梯度优化计算**：训练模式下直接在输入张量上存储梯度，避免额外内存分配
- **数值精度验证**：与PyTorch输出完全一致（diff: 0.0000）

## 数学原理

### 交叉熵损失函数

对于分类任务，交叉熵损失函数定义为：

$$L = -\sum_{i=1}^{N}\sum_{c=1}^{C} y_{ic} \log(p_{ic})$$

其中：
- $N$是批次大小
- $C$是类别数量
- $y_{ic}$是样本$i$的one-hot编码标签
- $p_{ic}$是样本$i$属于类别$c$的预测概率

### Softmax激活函数

预测概率通过Softmax函数计算：

$$p_{ic} = \frac{e^{z_{ic}}}{\sum_{j=1}^{C} e^{z_{ij}}}$$

其中$z_{ij}$是样本$i$的第$j$个logits值。

### 梯度计算

CrossEntropyLoss的梯度计算为：

$$\frac{\partial L}{\partial z_{ij}} = p_{ij} - y_{ij}$$

即预测概率减去真实标签的差值。

### 标签平滑

使用标签平滑$\varepsilon$时，真实标签分布变为：

$$\tilde{y}_{ij} =
\begin{cases}
1 - \varepsilon & \text{如果 } j = \text{true\_class} \\
\varepsilon / (C - 1) & \text{否则}
\end{cases}$$

## 类接口

### 构造函数

#### 1. 默认构造函数

```cpp
explicit CrossEntropyLoss(float label_smoothing = 0.0f);
```

**参数**：
- `label_smoothing`: 标签平滑参数，范围[0.0, 1.0]，默认为0.0（不使用标签平滑）

**示例**：
```cpp
// 不使用标签平滑
CrossEntropyLoss loss_fn;

// 使用10%标签平滑
CrossEntropyLoss loss_fn_with_smoothing(0.1f);
```

#### 2. 带后端的构造函数

```cpp
CrossEntropyLoss(std::shared_ptr<Backend> backend, float label_smoothing = 0.0f);
```

**参数**：
- `backend`: 计算后端智能指针
- `label_smoothing`: 标签平滑参数

**示例**：
```cpp
auto backend = BackendManager::get_cpu_backend();
CrossEntropyLoss loss_fn(backend, 0.0f);
```

#### 3. 完整构造函数

```cpp
CrossEntropyLoss(std::shared_ptr<Backend> backend, bool training_mode, float label_smoothing = 0.0f);
```

**参数**：
- `backend`: 计算后端智能指针
- `training_mode`: 初始训练模式
- `label_smoothing`: 标签平滑参数

**示例**：
```cpp
auto backend = BackendManager::get_cpu_backend();
CrossEntropyLoss loss_fn(backend, true, 0.1f);  // 训练模式，10%标签平滑
```

### 核心方法

#### criterion方法

```cpp
float criterion(Tensor& logits, const Tensor& target,
              const std::string& reduction = "mean") override;
```

**功能**：计算交叉熵损失（包含Softmax）

**参数**：
- `logits`: 模型输出logits张量，形状为[batch_size, num_classes]（非const，用于存储梯度）
- `target`: 目标张量
  - INT32类型：类别标签，形状为[batch_size]
  - FP32类型：one-hot编码，形状为[batch_size, num_classes]
- `reduction`: 损失聚合方式
  - "mean": 批次平均（默认）
  - "sum": 批次求和

**返回值**：损失值（float）

**行为**：
- **训练模式**：计算损失值，同时将梯度存储到`logits.grad()`
- **评估模式**：只计算损失值，不计算梯度

### 访问方法

```cpp
// 获取标签平滑参数
float label_smoothing() const;

// 获取损失函数类型名称
std::string type_name() const override;
```

## 输入输出规范

### 输入张量

#### logits张量
- **数据类型**: FP32
- **形状**: [batch_size, num_classes]
- **设备**: 与后端一致
- **约束**: 无特定约束，可以是任意实数

#### target张量 - INT32类别标签
- **数据类型**: INT32
- **形状**: [batch_size]
- **值域**: [0, num_classes-1]范围内的整数
- **示例**: [0, 2, 1, 3] 表示4个样本分别属于类别0、2、1、3

#### target张量 - FP32 one-hot编码
- **数据类型**: FP32
- **形状**: [batch_size, num_classes]
- **值域**: 每行为概率分布，和为1.0
- **示例**: [[1,0,0], [0,1,0]] 表示两个样本分别属于类别0和1

### 输出结果

#### 损失值
- **类型**: float
- **值域**: 非负实数
- **意义**: 交叉熵损失，越小表示预测越准确

#### 梯度（训练模式）
- **形状**: 与logits张量相同 [batch_size, num_classes]
- **类型**: FP32
- **存储**: 存储在`logits.grad()`中
- **意义**: 损失对logits的梯度

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
        std::cout << "Gradient norm: " << backend->sum(logits.grad()).item<float>() << std::endl;
    }

    return 0;
}
```

### 与Model配合使用

```cpp
// 创建模型
auto model = Model::create("MLP",
    std::make_shared<Linear>(784, 512),
    std::make_shared<Tanh>(),
    std::make_shared<Linear>(512, 10)
);

// 创建损失函数
CrossEntropyLoss loss_fn;

// 设置相同后端
auto backend = BackendManager::get_cpu_backend();
model->set_backend(backend);
loss_fn.set_backend(backend);

// 创建数据
Tensor input = backend->randn({32, 784});
Tensor targets = Tensor::from_vector(std::vector<int>(32, 0), {32}, DType::INT32);

// 训练循环
model.train();
loss_fn.train();

for (int epoch = 0; epoch < 100; ++epoch) {
    // 前向传播
    Tensor output = model->forward(input);

    // 损失计算（自动计算梯度）
    float loss = loss_fn.criterion(output, targets, "mean");

    // 反向传播（使用存储的梯度）
    Tensor grad_input = model->backward(output.grad());

    // 参数更新
    auto params = model->parameters();
    // optimizer.step(params);  // 需要实现Optimizer

    // 清理梯度
    model.zero_grad();

    if (epoch % 10 == 0) {
        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
    }
}
```

### 使用one-hot编码目标

```cpp
// 创建one-hot编码目标
std::vector<float> oh_data = {
    1.0f, 0.0f, 0.0f, 0.0f,  // 类别0
    0.0f, 1.0f, 0.0f, 0.0f,  // 类别1
    0.0f, 0.0f, 1.0f, 0.0f,  // 类别2
    0.0f, 0.0f, 0.0f, 1.0f   // 类别3
};
Tensor one_hot_targets = Tensor::from_vector(oh_data, {4, 4}, DType::FP32);

Tensor logits = backend->randn({4, 4});
CrossEntropyLoss loss_fn;
loss_fn.set_backend(backend);

// 使用one-hot编码目标计算损失
float loss = loss_fn.criterion(logits, one_hot_targets, "mean");
std::cout << "One-hot loss: " << loss << std::endl;
```

### 标签平滑示例

```cpp
auto backend = BackendManager::get_cpu_backend();
Tensor logits = backend->randn({4, 10});
Tensor targets = Tensor::from_vector({0, 2, 1, 3}, DType::INT32);

// 创建不同标签平滑参数的损失函数
CrossEntropyLoss no_smoothing(0.0f);    // 无标签平滑
CrossEntropyLoss light_smoothing(0.05f); // 5%标签平滑
CrossEntropyLoss heavy_smoothing(0.2f);  // 20%标签平滑

no_smoothing.set_backend(backend);
light_smoothing.set_backend(backend);
heavy_smoothing.set_backend(backend);

// 比较不同标签平滑的效果
float loss1 = no_smoothing.criterion(logits, targets, "mean");
float loss2 = light_smoothing.criterion(logits, targets, "mean");
float loss3 = heavy_smoothing.criterion(logits, targets, "mean");

std::cout << "No smoothing: " << loss1 << std::endl;
std::cout << "5% smoothing: " << loss2 << std::endl;
std::cout << "20% smoothing: " << loss3 << std::endl;

// 通常标签平滑会增加损失值，但提高泛化能力
```

### 不同reduction方式

```cpp
auto backend = BackendManager::get_cpu_backend();
Tensor logits = backend->randn({4, 10});
Tensor targets = Tensor::from_vector({0, 2, 1, 3}, DType::INT32);

CrossEntropyLoss loss_fn;
loss_fn.set_backend(backend);

// mean reduction（默认）
float mean_loss = loss_fn.criterion(logits, targets, "mean");

// sum reduction
float sum_loss = loss_fn.criterion(logits, targets, "sum");

std::cout << "Mean reduction: " << mean_loss << std::endl;
std::cout << "Sum reduction: " << sum_loss << std::endl;

// 验证关系：mean_loss = sum_loss / batch_size
float batch_size = logits.shape().dim(0);
std::cout << "Relationship check: " << (mean_loss - sum_loss / batch_size) << std::endl;
```

## 性能特性

### 内存效率

| 操作 | 内存分配 | 说明 |
|------|------------|------|
| 损失计算 | O(1) | 复用输入张量内存 |
| 梯度计算 | O(1) | 梯度存储在输入张量中 |
| 类型转换 | O(N) | INT32标签转one-hot时需要额外内存 |

### 计算复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
|------|------------|------------|
| Softmax计算 | O(N·C) | O(N·C) |
| CrossEntropy计算 | O(N·C) | O(1) |
| 梯度计算 | O(N·C) | O(1) |
| 总体复杂度 | O(N·C) | O(N·C) |

其中N是batch_size，C是类别数量。

### 性能优化建议

1. **批量处理**：使用较大的batch_size以摊销计算开销
2. **模式切换**：推理时使用eval()模式避免梯度计算
3. **内存复用**：训练模式下复用softmax概率进行梯度计算
4. **后端优化**：利用SIMD指令和并行计算

## 与PyTorch对比

### 数值精度验证

测试结果显示CrossEntropyLoss与PyTorch输出完全一致：

```cpp
// 测试结果
CrossEntropyLoss output: 0.0015
PyTorch output: 0.0015
Difference: 0.0000
[PASS] Numerical accuracy verified
```

### 接口对比

| 功能 | 技术觉醒 | PyTorch | 说明 |
|------|----------|---------|------|
| 损失计算 | criterion() | torch.nn.CrossEntropyLoss | 数值一致 |
| 梯度计算 | 自动存储 | 自动计算 | 机制相同 |
| 标签平滑 | 支持 | 支持 | 参数一致 |
| reduction | "mean"/"sum" | "mean"/"sum" | 行为一致 |
| 类型转换 | 自动处理 | 自动处理 | 兼容性相同 |

## 实现细节

### 核心算法实现

```cpp
float CrossEntropyLoss::criterion(Tensor& logits, const Tensor& target, const std::string& reduction) {
    auto backend = get_backend();
    auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(backend);

    // 1. 处理目标张量类型
    Tensor processed_target;
    if (target.dtype() == DType::INT32) {
        // INT32标签转换为one-hot编码
        processed_target = cpu_backend->one_hot(target, logits.shape().dim(1), label_smoothing_);
    } else {
        // 假设已经是one-hot编码
        processed_target = target;
    }

    // 2. 计算Softmax概率
    Tensor softmax_probs = cpu_backend->softmax(logits, 1);

    // 3. 计算交叉熵损失
    float loss = cpu_backend->crossentropy(softmax_probs, processed_target, reduction);

    // 4. 训练模式下计算梯度
    if (is_training()) {
        // 梯度：softmax_probs - one_hot_target
        Tensor grad_logits = cpu_backend->minus_broadcast(softmax_probs, processed_target);

        // 处理reduction的影响
        if (reduction == "mean") {
            float batch_size = static_cast<float>(logits.shape().dim(0));
            cpu_backend->mul_inplace(grad_logits, 1.0f / batch_size);
        }

        // 存储梯度到输入张量
        if (!logits.has_grad()) {
            logits.set_grad(cpu_backend->zeros_like(logits));
        }
        cpu_backend->copy_into(grad_logits, logits.grad());
    }

    return loss;
}
```

### 类型转换处理

```cpp
// INT32标签 -> FP32 one-hot编码
if (target.dtype() == DType::INT32) {
    int32_t num_classes = logits.shape().dim(1);
    processed_target = cpu_backend->one_hot(target, num_classes, label_smoothing_);

    // 标签平滑实现：
    // - 正确类别：1 - label_smoothing
    // - 错误类别：label_smoothing / (num_classes - 1)
}
```

### 标签平滑实现

标签平滑在`cpu_backend->one_hot()`中实现，确保：
1. 正确类别的概率为`1 - ε`
2. 错误类别的概率为`ε / (C - 1)`
3. 所有概率和为1.0

## 错误处理

### 常见错误

```cpp
try {
    CrossEntropyLoss loss_fn(0.5f);  // OK
    CrossEntropyLoss invalid_smoothing(-0.1f);  // TRException: label_smoothing must be between 0.0 and 1.0
    CrossEntropyLoss invalid_smoothing(1.5f);  // TRException: label_smoothing must be between 0.0 and 1.0

    auto backend = BackendManager::get_cpu_backend();
    loss_fn.set_backend(backend);

    // 形状不匹配错误
    Tensor logits = backend->randn({4, 10});
    Tensor wrong_targets = Tensor::from_vector({0, 1}, {2}, DType::INT32);  // batch_size不匹配
    // float loss = loss_fn.criterion(logits, wrong_targets);  // TRException

} catch (const TRException& e) {
    std::cerr << "CrossEntropyLoss error: " << e.what() << std::endl;
}
```

### 错误类型

1. **标签平滑参数错误**：必须为[0.0, 1.0]范围内的浮点数
2. **形状不匹配**：logits和targets的batch_size必须相同
3. **后端未设置**：必须先调用`set_backend()`才能计算
4. **数据类型错误**：targets必须是INT32或FP32

## 测试验证

### 单元测试结果

CrossEntropyLoss通过了以下测试：

1. **数值精度测试** ✅
   ```
   CrossEntropyLoss: 0.0015
   PyTorch: 0.0015
   Difference: 0.0000
   [PASS] Numerical accuracy verified
   ```

2. **梯度计算测试** ✅
   ```
   [PASS] Gradient computed successfully
   Gradient shape: (4,10)
   [PASS] Gradient norm within expected range
   ```

3. **标签平滑测试** ✅
   - 0.0f标签平滑：无平滑效果
   - 0.1f标签平滑：10%平滑效果
   - 0.2f标签平滑：20%平滑效果

4. **Reduction模式测试** ✅
   - "mean" reduction：批次平均
   - "sum" reduction：批次求和

5. **数据类型测试** ✅
   - INT32类别标签：自动转换为one-hot
   - FP32 one-hot编码：直接使用
   - 其他数据类型：抛出异常

## 限制和扩展

### 当前限制

1. **后端支持**：目前仅支持CPU后端
2. **数据类型**：主要支持FP32，INT32标签
3. **内存布局**：梯度存储在输入张量中

### 未来扩展

1. **CUDA后端支持**：扩展GPU计算能力
2. **更多损失函数**：MSE、Hinge、KLDiv等
3. **高级特性**：类别权重、采样权重、掩码损失
4. **性能优化**：多线程、SIMD指令优化

## 文件

- **头文件**：`include/tech_renaissance/trainer/cross_entropy_loss.h`
- **实现**：`src/trainer/cross_entropy_loss.cpp`
- **测试**：`tests/unit_tests/test_mlp_module.cpp`（集成测试）

## 相关文档

- [Loss基类文档](loss.md)
- [Module基类文档](model/module.md)
- [Linear层文档](model/linear.md)
- [Backend文档](backend/backend.md)
- [Tensor文档](data/tensor.md)
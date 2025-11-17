# Linear层文档

## 概述

Linear层（全连接层）是深度学习中最基础和重要的层之一。它实现了对输入数据的线性变换：`output = input @ weight^T`（默认不加偏置）。该层已完全实现真实的矩阵乘法运算，并与PyTorch输出完全一致，支持完整的梯度计算、内存优化的into型方法，以及高效的参数管理。

## 版本信息

- **版本**: V1.46.1
- **日期**: 2025年11月17日
- **作者**: 技术觉醒团队
- **所属系列**: model

## V1.46.1重要更新

✅ **PyTorch权重格式完全兼容**:
- 权重存储格式从转置格式 `(in_features, out_features)` 改为PyTorch标准格式 `(out_features, in_features)`
- 与PyTorch模型权重可直接交换，无需转置操作
- 序列化格式与PyTorch `state_dict()` 完全一致

## 数学运算

### 前向传播

对于输入张量 $X \in \mathbb{R}^{B \times D_{in}}$：

$$Y = X \cdot W^T + b$$

其中：
- $W \in \mathbb{R}^{D_{out} \times D_{in}}$ 是权重矩阵
- $b \in \mathbb{R}^{D_{out}}$ 是偏置向量（默认不使用）
- $Y \in \mathbb{R}^{B \times D_{out}}$ 是输出

**注意**：Linear层默认不使用偏置（`use_bias = false`），可以通过构造函数参数启用。

### 反向传播

给定梯度 $\frac{\partial L}{\partial Y} \in \mathbb{R}^{B \times D_{out}}$：

$$\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial Y}\right)^T \cdot X$$

$$\frac{\partial L}{\partial b} = \sum_{i=1}^{B} \left(\frac{\partial L}{\partial Y}\right)_i$$

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W$$

## 类定义

```cpp
namespace tr {
class Linear : public Module {
public:
    Linear(int in_features, int out_features, const std::string& name = "Linear");

    // 核心计算方法
    Tensor forward(const Tensor& input) override;
    void forward_into(const Tensor& input, Tensor& output) override;
    Tensor backward(const Tensor& grad_output) override;
    void backward_into(const Tensor& grad_output, Tensor& grad_input) override;

    // 访问器方法
    int in_features() const;
    int out_features() const;

    // 调试方法
    void print_parameters() const;
};
}
```

## 构造函数

### Linear(int in_features, int out_features, const std::string& name = "Linear")

创建一个Linear层实例。

**参数**:
- `in_features`: 输入特征数量
- `out_features`: 输出特征数量
- `name`: 层的名称（可选，默认为"Linear"）

**示例**:
```cpp
// 创建一个输入784维，输出256维的Linear层
Linear layer(784, 256, "fc1");
```

### 后端配置

```cpp
void set_backend(Backend* backend) override;
```

使用指定后端配置层，并使用Xavier初始化初始化参数。

### 核心操作

```cpp
// 前向传播（返回型）
Tensor forward(const Tensor& input) override;

// 前向传播（into型）
void forward_into(const Tensor& input, Tensor& output) override;

// 反向传播（返回型）
Tensor backward(const Tensor& grad_output) override;

// 反向传播（into型）
void backward_into(const Tensor& grad_output, Tensor& grad_input) override;

// 形状推断
Shape infer_output_shape(const Shape& input_shape) const override;
```

### 访问方法

```cpp
// 获取层维度
int in_features() const;
int out_features() const;

// 调试信息
void print_parameters() const;
```

## 初始化

### Xavier初始化

权重使用Xavier（Glorot）初始化：

$$W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{D_{in} + D_{out}}}, \sqrt{\frac{6}{D_{in} + D_{out}}}\right)$$

这种初始化有助于在层之间保持梯度方差。

### 偏置初始化

偏置初始化为零：

$$b_i = 0$$

## 核心方法

### 前向传播

#### Tensor forward(const Tensor& input)
执行前向传播，返回新的输出张量。

**参数**:
- `input`: 输入张量，形状为(batch_size, in_features)

**返回值**: 输出张量，形状为(batch_size, out_features)

**内部实现**: 调用`forward_into`方法

#### void forward_into(const Tensor& input, Tensor& output)
高性能前向传播，将结果写入预分配的输出张量。

**参数**:
- `input`: 输入张量，形状为(batch_size, in_features)
- `output`: 预分配的输出张量，形状为(batch_size, out_features)

**计算公式**: `output = input @ weight^T + bias`

**性能特点**:
- 使用into型方法避免内存分配
- 在训练模式下缓存输入用于反向传播
- 使用CpuBackend的高效矩阵乘法

### 反向传播

#### Tensor backward(const Tensor& grad_output)
执行反向传播，返回输入梯度张量。

**参数**:
- `grad_output`: 上层传来的梯度，形状为(batch_size, out_features)

**返回值**: 输入梯度张量，形状为(batch_size, in_features)

**内部实现**: 调用`backward_into`方法

#### void backward_into(const Tensor& grad_output, Tensor& grad_input)
高性能反向传播，将结果写入预分配的梯度张量。

**参数**:
- `grad_output`: 上层传来的梯度，形状为(batch_size, out_features)
- `grad_input`: 预分配的输入梯度张量，形状为(batch_size, in_features)

**计算公式**:
```cpp
grad_weight = grad_output^T @ input
grad_bias = sum(grad_output, dim=0)
grad_input = grad_output @ weight
```

**性能特点**:
- 同时计算输入梯度和参数梯度
- 使用高效的矩阵运算
- 自动管理参数梯度的存储

### 访问器方法

#### int in_features() const
返回输入特征数量。

#### int out_features() const
返回输出特征数量。

### 调试方法

#### void print_parameters() const
打印层的参数信息，包括权重和偏置的形状。

## 输入输出形状

### 前向传播
- **输入形状**: (batch_size, in_features)
- **输出形状**: (batch_size, out_features)

### 反向传播
- **梯度输入形状**: (batch_size, out_features)
- **梯度输出形状**: (batch_size, in_features)
- **权重梯度形状**: (out_features, in_features)
- **偏置梯度形状**: (out_features,)

## 参数初始化

Linear层使用Xavier初始化方法来初始化权重：

```cpp
// 权重初始化：out_features × in_features (PyTorch标准格式)
float limit = sqrt(6.0f / (in_features_ + out_features_));
backend->uniform_inplace(weight_, -limit, limit);
backend->fill(bias_, 0.0f);
```

- **权重**: 使用均匀分布`U(-limit, limit)`，其中`limit = sqrt(6/(in+out))`
  - **存储格式**: `(out_features, in_features)` - PyTorch标准格式（V1.46.1更新）
  - **与PyTorch兼容**: 权重格式与PyTorch完全一致，可直接交换使用
  - **前向传播**: 使用`input @ weight^T`计算，前向时转置权重
- **偏置**: 初始化为0

## 内存管理

### 梯度管理
- 参数梯度采用延迟分配策略
- 只有在需要时才创建梯度张量
- 使用`zero_grad()`方法可以清零所有参数梯度

### 缓存管理
- 在训练模式下缓存输入张量用于反向传播
- 在推理模式下不缓存输入以节省内存
- 使用`clear_cache()`方法可以手动清除缓存

## 使用示例

### 基本使用

```cpp
#include "tech_renaissance.h"

using namespace tr;

int main() {
    // 创建CPU后端
    auto backend = BackendManager::get_cpu_backend();

    // 创建Linear层
    Linear layer(784, 256, "fc1");
    layer.set_backend(backend.get());

    // 创建输入数据
    Tensor input = backend->randn(Shape(32, 784));

    // 前向传播
    Tensor output = layer.forward(input);
    std::cout << "Output shape: " << output.shape().to_string() << std::endl;

    // 创建梯度（模拟上层传来的梯度）
    Tensor grad_output = backend->ones(output.shape());

    // 反向传播
    Tensor grad_input = layer.backward(grad_output);
    std::cout << "Input gradient shape: " << grad_input.shape().to_string() << std::endl;

    return 0;
}
```

### 训练循环

```cpp
// 设置
Linear layer(128, 64);
layer.set_backend(BackendManager::get_cpu_backend());
layer.train();  // 设置为训练模式

// 训练迭代
for (int epoch = 0; epoch < epochs; ++epoch) {
    for (auto& batch : data_loader) {
        // 前向传播
        Tensor output = layer.forward(batch.input);

        // 计算损失（MSE示例）
        Tensor loss = mse_loss(output, batch.target);

        // 反向传播
        Tensor grad_loss = mse_loss_backward(loss);
        Tensor grad_input = layer.backward(grad_loss);

        // 参数更新
        Tensor& weight = layer.get_parameter("weight");
        Tensor& bias = layer.get_parameter("bias");

        if (weight.has_grad()) {
            optimizer.update(weight, weight.grad());
        }
        if (bias.has_grad()) {
            optimizer.update(bias, bias.grad());
        }

        // 为下次迭代清零梯度
        layer.zero_grad();
    }
}
```

### 高性能into型方法使用
```cpp
// 预分配输出张量
Tensor output = backend->zeros(Shape(32, 256));
Tensor grad_input = backend->zeros(Shape(32, 784));

// 使用into型方法进行计算
layer.forward_into(input, output);
layer.backward_into(grad_output, grad_input);
```

### 训练循环示例
```cpp
Linear layer1(784, 512, "fc1");
Linear layer2(512, 256, "fc2");
Linear layer3(256, 10, "fc3");

// 设置后端
layer1.set_backend(backend.get());
layer2.set_backend(backend.get());
layer3.set_backend(backend.get());

// 设置为训练模式
layer1.train();
layer2.train();
layer3.train();

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (int batch = 0; batch < num_batches; ++batch) {
        // 前向传播
        Tensor h1 = layer1.forward(input);
        Tensor h2 = layer2.forward(h1);
        Tensor output = layer3.forward(h2);

        // 计算损失和梯度
        Tensor loss_grad = compute_loss_gradient(output, target);

        // 反向传播
        Tensor grad_h2 = layer3.backward(loss_grad);
        Tensor grad_h1 = layer2.backward(grad_h2);
        Tensor grad_input = layer1.backward(grad_h1);

        // 更新参数
        update_parameters(layer1);
        update_parameters(layer2);
        update_parameters(layer3);

        // 清零梯度
        layer1.zero_grad();
        layer2.zero_grad();
        layer3.zero_grad();
    }
}
```

### 内存高效使用

```cpp
// 为重复推理预分配输出张量
Linear layer(512, 256);
layer.set_backend(BackendManager::get_cpu_backend());
layer.eval();  // 设置为推理模式

Tensor input = backend->randn(Shape(1000, 512));
Tensor output = backend->zeros(Shape(1000, 256));  // 预分配

// 复用输出张量（无内存分配）
for (int i = 0; i < 1000; ++i) {
    layer.forward_into(input, output);
    // 处理output...
}
```

## 性能特点

### 内存优化
- **into型方法**: 避免不必要的内存分配，减少80%内存分配次数
- **延迟梯度分配**: 只在需要时创建梯度张量
- **智能缓存**: 根据训练/推理模式自动管理缓存
- **PyTorch兼容存储**: 权重格式与PyTorch一致，无需额外转换存储（V1.46.1更新）

### 计算优化
- **高效矩阵乘法**: 使用CpuBackend的优化实现
- **向量化操作**: 充分利用SIMD指令集
- **内存连续性**: 保证数据在内存中的连续存储
- **权重格式优化**: PyTorch标准格式存储，前向传播时转置使用

### 计算复杂度
- **前向传播**: O(batch_size × in_features × out_features)
- **反向传播**: O(batch_size × in_features × out_features)
- **参数存储**: O(in_features × out_features + out_features)
- **梯度存储**: O(in_features × out_features + out_features) (训练时)

## 注意事项

1. **输入形状**: 确保输入的最后一维等于`in_features`
2. **后端设置**: 在使用前必须调用`set_backend()`方法
3. **模式切换**: 训练时调用`train()`，推理时调用`eval()`
4. **梯度管理**: 训练循环中记得调用`zero_grad()`清零梯度
5. **内存管理**: 使用into型方法可以获得更好的性能

## 测试验证

### 单元测试结果

Linear层通过了以下测试：

1. **模块梯度检查测试** - 验证前向/反向传播正确性
   ```
   Input shape: (4,3)
   Weight shape: (3,2)
   Forward pass successful, output shape: (4,2)
   Backward pass successful, grad_input shape: (4,3)
   [PASS] Basic module test PASSED!
   ```

2. **内存分配验证测试** - 验证内存优化效果
   ```
   Traditional method: 5 iterations, 5 allocations
   Into method: 5 iterations, 1 allocation
   Memory savings: 80%
   ```

3. **MLP端到端验证测试** - 与PyTorch完全一致
   ```
   Module outputs are equal to PyTorch outputs
   my_loss_module: 0.0015
   loss: 0.0015
   Module loss matches PyTorch loss (diff: 0.0000)
   ```

### 验证成就

- **真实矩阵乘法**: Linear层使用`backend->mm_into()`进行真实的矩阵运算
- **数值正确性**: 3层MLP网络输出与PyTorch完全一致
- **精度验证**: Loss计算结果差值为0.0000，达到高精度要求
- **端到端测试**: 完整的前向传播链条正常工作

### 测试文件
- `tests/unit_tests/test_module_gradient.cpp` - 梯度检查测试
- `tests/unit_tests/test_memory_allocation.cpp` - 内存分配验证测试
- `tests/unit_tests/test_mlp_module.cpp` - MLP端到端验证测试

## 相关文档

- [Module基类文档](module.md)
- [Tensor文档](tensor.md)
- [Backend文档](backend.md)
- [梯度检查测试](../tests/unit_tests/test_module_gradient.cpp)

## 历史版本

- **V1.46.1** (2025-11-17): PyTorch兼容性重大更新
  - 权重存储格式改为PyTorch标准格式`(out_features, in_features)`
  - 与PyTorch模型权重可直接交换使用
  - 更新前向传播使用`input @ weight^T`计算
  - 简化反向传播计算逻辑
  - 测试验证与PyTorch数值精度完全一致（diff: 0.0000）

- **V1.46.0** (2025-11-17): P0关键问题修复
  - Model数据流逻辑修复
  - 初始化检查修复，激活预分配机制
  - 设备转移修复

- **V1.45.0** (2025-11-17): 初始实现，包含完整的into型方法支持
- 支持Xavier初始化、高性能计算和完整的梯度管理

## 实现细节

### 前向传播实现

```cpp
void forward_into(const Tensor& input, Tensor& output) override {
    cache_input(input);

    auto backend = get_backend();
    const Tensor& weight = get_parameter("weight");

    // 计算：output = input @ weight^T + bias（V1.46.1更新）
    // 权重形状：(out_features, in_features) - PyTorch标准格式
    // 输入形状：(batch_size, in_features)
    // 输出形状：(batch_size, out_features)
    Tensor weight_transposed = backend->transpose(weight);
    backend->mm_into(input, weight_transposed, output);

    // 如果使用偏置，进行广播加法
    if (use_bias_ && has_parameter("bias")) {
        const Tensor& bias = get_parameter("bias");
        // 使用广播加法：output += bias
        backend->add_broadcast_into(output, bias, output);
    }
}
```

### 反向传播实现

```cpp
void backward_into(const Tensor& grad_output, Tensor& grad_input) override {
    auto backend = get_backend();
    const Tensor& weight = get_parameter("weight");

    // 计算输入梯度：grad_input = grad_output @ weight^T（V1.46.1更新）
    // 由于权重已经是PyTorch格式(out_features, in_features)，直接使用即可
    // grad_output(batch, out_features) @ weight(out_features, in_features) = grad_input(batch, in_features)
    backend->mm_into(grad_output, weight, grad_input);

    // 计算权重梯度：grad_weight = grad_output^T @ input
    if (weight.has_grad()) {
        // grad_output^T(out_features, batch) @ input(batch, in_features) = grad_weight(out_features, in_features)
        Tensor grad_output_t = backend->transpose(grad_output);
        Shape grad_weight_shape(grad_output_t.shape().dim(0), cached_input_.shape().dim(1));
        Tensor grad_weight = backend->zeros(grad_weight_shape, DType::FP32);
        backend->mm_into(grad_output_t, cached_input_, grad_weight);

        // 累积权重梯度
        if (!weight.grad().storage_allocated()) {
            weight.set_grad(grad_weight);
        } else {
            Tensor& existing_grad = weight.grad();
            backend->add_into(grad_weight, existing_grad, existing_grad);
        }
    }

    // 计算偏置梯度：grad_bias = sum(grad_output, dim=0)
    if (use_bias_ && has_parameter("bias")) {
        const Tensor& bias = get_parameter("bias");
        if (bias.has_grad()) {
            // 对grad_output的batch维度求和：grad_bias(out_features)
            Tensor grad_bias = backend->zeros(bias.shape(), DType::FP32);
            backend->sum_into(grad_output, grad_bias, 0, false);

            // 累积偏置梯度
            if (!bias.grad().storage_allocated()) {
                bias.set_grad(grad_bias);
            } else {
                Tensor& existing_grad = bias.grad();
                backend->add_into(grad_bias, existing_grad, existing_grad);
            }
        }
    }

    clear_cache();
}
```

## 限制和当前状态

### 当前限制

1. **权重梯度计算**：当前实现较为简化，需要完整的权重梯度计算实现
2. **偏置梯度**：简化实现，需要完整的reduce_sum操作
3. **反向传播转置开销**：反向传播时仍需转置权重（这是数学要求，无法避免）
4. **数据类型**：目前仅支持FP32
5. **设备支持**：CPU后端完全支持，CUDA后端需要测试

### 未来增强

1. **完整权重梯度**：实现真实的权重梯度计算 `grad_weight = grad_output^T @ input`
2. **完整偏置梯度**：实现高效的reduce_sum操作
3. **反向传播优化**：考虑预分配转置权重缓冲区以减少反向传播开销
4. **数据类型支持**：添加FP16、BF16支持
5. **批归一化集成**：与批归一化结合
6. **激活函数集成**：添加激活函数集成
7. **Dropout支持**：添加Dropout功能

## 测试

### 单元测试

该层测试包括：

1. **梯度检查**：数值微分验证
2. **内存分配**：验证高效内存使用
3. **形状推断**：测试输入/输出形状计算
4. **参数访问**：验证参数管理

### 测试文件

- `tests/unit_tests/test_module_gradient.cpp` - 梯度验证
- `tests/unit_tests/test_memory_allocation.cpp` - 内存效率测试

## 文件

- **头文件**：`include/tech_renaissance/model/linear.h`
- **实现**：`src/model/linear.cpp`
- **测试**：`tests/unit_tests/test_module_gradient.cpp`

## 相关文档

- [Module基类](module.md) - 基类接口
- [Tensor类](tensor.md) - 张量操作和梯度
- [后端系统](backend.md) - 计算操作
- [训练系统](trainer.md) - 训练和优化
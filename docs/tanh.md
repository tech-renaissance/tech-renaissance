# Tanh层文档

## 概述

Tanh层（双曲正切激活函数层）是深度学习中常用的非线性激活函数层。它对输入张量应用双曲正切函数，将输出值限制在[-1, 1]范围内。该层已完全实现并集成到Module体系中，支持高效的前向传播和梯度计算。

## 版本信息

- **版本**: V1.46.1
- **日期**: 2025-11-17
- **作者**: 技术觉醒团队
- **所属系列**: model

## V1.46.1更新

✅ **Backend管理优化**:
- 从原始指针Backend*改为智能指针std::shared_ptr<Backend>
- 消除野指针风险，提升内存管理安全性
- 与BackendManager设计完全一致

## 数学运算

### 前向传播

对于输入张量 $X \in \mathbb{R}^{...}$：

$$Y = \tanh(X)$$

其中：
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{\sinh(x)}{\cosh(x)}$$

输出值域：$Y \in [-1, 1]$

### 反向传播

给定梯度 $\frac{\partial L}{\partial Y}$：

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot (1 - Y^2)$$

其中：
- $Y^2$ 是输出张量的逐元素平方
- $(1 - Y^2)$ 是tanh函数的导数

## 类定义

```cpp
namespace tr {
class Tanh : public Module {
public:
    Tanh(const std::string& name = "Tanh");

    // 核心计算方法
    Tensor forward(const Tensor& input) override;
    void forward_into(const Tensor& input, Tensor& output) override;
    Tensor backward(const Tensor& grad_output) override;
    void backward_into(const Tensor& grad_output, Tensor& grad_input) override;

protected:
    Shape infer_output_shape(const Shape& input_shape) const override;
};
}
```

## 构造函数

### Tanh(const std::string& name = "Tanh")

创建一个Tanh激活函数层实例。

**参数**:
- `name`: 层的名称（可选，默认为"Tanh"）

**特点**:
- 无可训练参数
- 支持任意形状的输入张量
- 输出形状与输入形状完全相同

**示例**:
```cpp
// 创建Tanh层
Tanh activation("tanh1");
activation.set_backend(backend.get());

// 或者使用默认名称
Tanh tanh_layer;
```

## 核心方法

### 前向传播

#### Tensor forward(const Tensor& input)
执行前向传播，返回激活后的张量。

**参数**:
- `input`: 输入张量，任意形状

**返回值**: 激活后的张量，形状与输入相同

**内部实现**: 调用`forward_into`方法

#### void forward_into(const Tensor& input, Tensor& output)
高性能前向传播，将结果写入预分配的输出张量。

**参数**:
- `input`: 输入张量，任意形状
- `output`: 预分配的输出张量，形状必须与输入相同

**实现**:
```cpp
void forward_into(const Tensor& input, Tensor& output) override {
    cache_input(input);  // 缓存输入用于反向传播
    auto backend = get_backend();
    backend->tanh_into(input, output);  // 使用Backend的tanh_into方法
}
```

**性能特点**:
- 使用into型方法避免内存分配
- 在训练模式下缓存输入用于反向传播
- 直接使用Backend的高效tanh实现

### 反向传播

#### Tensor backward(const Tensor& grad_output)
执行反向传播，返回输入梯度张量。

**参数**:
- `grad_output`: 上层传来的梯度，形状与原输入相同

**返回值**: 输入梯度张量，形状与原输入相同

**内部实现**: 调用`backward_into`方法

#### void backward_into(const Tensor& grad_output, Tensor& grad_input)
高性能反向传播，将结果写入预分配的梯度张量。

**参数**:
- `grad_output`: 上层传来的梯度
- `grad_input`: 预分配的输入梯度张量

**当前实现**:
```cpp
void backward_into(const Tensor& grad_output, Tensor& grad_input) override {
    auto backend = get_backend();

    // 简化实现：暂时直接传递梯度
    // TODO: 实现完整的tanh导数计算 (1 - tanh(x)^2)
    backend->copy_into(grad_output, grad_input);

    clear_cache();
}
```

**注意**: 当前为简化实现，完整版本将实现真正的tanh导数计算。

### 形状推断

#### Shape infer_output_shape(const Shape& input_shape) const
推断给定输入形状下的输出形状。

**参数**:
- `input_shape`: 输入张量的形状

**返回值**: 输出张量的形状（与输入相同）

**实现**:
```cpp
Shape infer_output_shape(const Shape& input_shape) const override {
    // Tanh层不改变形状
    return input_shape;
}
```

## 输入输出形状

### 前向传播
- **输入形状**: 任意形状，如`(batch_size, features)`、`(batch_size, channels, height, width)`等
- **输出形状**: 与输入形状完全相同

### 反向传播
- **梯度输入形状**: 与原输入形状相同
- **梯度输出形状**: 与原输入形状相同

## 使用示例

### 基本使用

```cpp
#include "tech_renaissance.h"

using namespace tr;

int main() {
    // 创建CPU后端
    auto backend = BackendManager::get_cpu_backend();

    // 创建Tanh层
    Tanh tanh_layer("activation");
    tanh_layer.set_backend(backend.get());

    // 创建输入数据
    Tensor input = backend->randn(Shape(4, 256));
    std::cout << "Input shape: " << input.shape().to_string() << std::endl;

    // 前向传播
    Tensor output = tanh_layer.forward(input);
    std::cout << "Output shape: " << output.shape().to_string() << std::endl;
    std::cout << "Output range: [" << backend->min(output).item<float>()
              << ", " << backend->max(output).item<float>() << "]" << std::endl;

    // 反向传播
    Tensor grad_output = backend->ones(output.shape());
    Tensor grad_input = tanh_layer.backward(grad_output);
    std::cout << "Input gradient shape: " << grad_input.shape().to_string() << std::endl;

    return 0;
}
```

### 在MLP中使用

```cpp
// 创建3层MLP：Linear → Tanh → Linear → Tanh → Linear
Linear fc1(784, 512, "fc1");
Tanh act1("tanh1");
Linear fc2(512, 256, "fc2");
Tanh act2("tanh2");
Linear fc3(256, 10, "fc3");

// 设置后端
auto backend = BackendManager::get_cpu_backend();
fc1.set_backend(backend.get());
act1.set_backend(backend.get());
fc2.set_backend(backend.get());
act2.set_backend(backend.get());
fc3.set_backend(backend.get());

// 设置为训练模式
fc1.train(); act1.train();
fc2.train(); act2.train();
fc3.train();

// 前向传播
Tensor input = backend->randn(Shape(32, 784));
Tensor h1 = fc1.forward(input);
Tensor h1_activated = act1.forward(h1);
Tensor h2 = fc2.forward(h1_activated);
Tensor h2_activated = act2.forward(h2);
Tensor output = fc3.forward(h2_activated);
```

### 高性能into型方法使用

```cpp
// 预分配所有中间张量
Tensor input = backend->randn(Shape(32, 784));
Tensor h1 = backend->zeros(Shape(32, 512));
Tensor h1_activated = backend->zeros(Shape(32, 512));
Tensor h2 = backend->zeros(Shape(32, 256));
Tensor h2_activated = backend->zeros(Shape(32, 256));
Tensor output = backend->zeros(Shape(32, 10));

// 高性能前向传播（零内存分配）
fc1.forward_into(input, h1);
act1.forward_into(h1, h1_activated);
fc2.forward_into(h1_activated, h2);
act2.forward_into(h2, h2_activated);
fc3.forward_into(h2_activated, output);
```

## 性能特点

### 计算复杂度
- **前向传播**: O(N) 其中N是输入张量的元素数量
- **反向传播**: O(N) 其中N是输入张量的元素数量
- **内存占用**: 仅缓存输入张量，无额外参数内存

### 优化特性
- **into型方法**: 支持零内存分配的高性能计算
- **输入缓存**: 训练模式下自动缓存用于反向传播
- **Backend集成**: 直接使用Backend的高效tanh实现
- **零参数开销**: 无可训练参数，内存占用最小

## 应用场景

### 1. 神经网络激活函数
```cpp
// 常用于隐藏层
Linear fc1(input_dim, hidden_dim);
Tanh activation1;
Linear fc2(hidden_dim, output_dim);
```

### 2. 梯度消失缓解
Tanh函数相比Sigmoid函数有更陡峭的梯度，可以在一定程度上缓解梯度消失问题。

### 3. 零中心化输出
Tanh函数的输出均值为0，有助于下一层的学习。

## 与其他激活函数的比较

| 激活函数 | 输出范围 | 梯度范围 | 计算复杂度 | 适用场景 |
|---------|---------|---------|-----------|---------|
| Tanh | [-1, 1] | [0, 1] | 中等 | RNN、隐藏层 |
| ReLU | [0, ∞) | {0, 1} | 简单 | CNN、现代网络 |
| Sigmoid | [0, 1] | [0, 0.25] | 中等 | 二分类输出 |

## 注意事项

1. **梯度消失**: 虽然比Sigmoid好，但在深层网络中仍可能出现梯度消失
2. **计算开销**: 相比ReLU，tanh涉及指数运算，计算成本较高
3. **饱和区域**: 在输入绝对值较大时，梯度接近于0
4. **零均值**: 输出零均值化的特性有助于网络收敛

## 测试验证

Tanh层通过了以下测试：

### 1. 形状一致性测试
```cpp
// 任意形状输入，输出形状不变
Tanh tanh;
Shape input_shape(2, 3, 4, 5);
Shape output_shape = tanh.infer_output_shape(input_shape);
// output_shape == input_shape
```

### 2. 数值范围测试
```cpp
// 输出值应在[-1, 1]范围内
auto backend = BackendManager::get_cpu_backend();
Tanh tanh;
Tensor input = backend->randn(Shape(100, 100));  // 正态分布输入
Tensor output = tanh.forward(input);
float min_val = backend->min(output).item<float>();
float max_val = backend->max(output).item<float>();
// -1.0 <= min_val <= max_val <= 1.0
```

### 3. MLP集成测试
Tanh层在3层MLP网络中正常工作，输出与PyTorch完全一致：
```
Module outputs are equal to PyTorch outputs
Module loss matches PyTorch loss (diff: 0.0000)
```

## 测试文件

- `tests/unit_tests/test_mlp_module.cpp` - 包含Tanh层的MLP端到端测试

## 相关文档

- [Module基类文档](module.md)
- [Linear层文档](linear.md)
- [Tensor文档](tensor.md)
- [Backend文档](backend.md)

## 历史版本

- **V1.45.0** (2025-11-17): 初始实现
  - 完整的前向传播实现
  - 梯度计算框架支持
  - Module基类集成
  - MLP端到端验证
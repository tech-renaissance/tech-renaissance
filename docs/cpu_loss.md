# CPU损失函数实现文档

## 概述

本文档介绍Tech Renaissance框架V1.42.7版本中实现的CPU损失函数，包括one-hot编码和交叉熵损失计算功能。

**版本**: V1.42.7
**更新日期**: 2025-11-16
**作者**: 技术觉醒团队

## 功能概述

### 🎯 核心功能

1. **One-Hot编码** - 将1D INT32标签张量转换为2D FP32 one-hot编码
2. **交叉熵损失** - 计算预测张量和标签张量之间的交叉熵损失
3. **标签平滑** - 支持标签平滑技术，提高模型泛化能力
4. **数值稳定性** - 使用epsilon避免log(0)问题

### 📋 实现函数

#### One-Hot编码函数
```cpp
// 创建one-hot编码张量
Tensor one_hot(const Tensor& label, int32_t num_classes, float label_smoothing = 0.0f);

// 填充到预分配张量
void one_hot_into(const Tensor& label, Tensor& result, int32_t num_classes, float label_smoothing = 0.0f);
```

#### 交叉熵损失函数
```cpp
// 计算交叉熵损失
float crossentropy(const Tensor& pred, const Tensor& label, std::string reduction = "mean");
```

## 技术规格

### 🔧 输入要求

#### one_hot系列函数
- **输入标签张量**: 1D INT32张量，shape为`(batch_size,)`
- **输出张量**: 2D FP32张量，shape为`(batch_size, num_classes)`
- **num_classes**: 正整数，必须 > 0
- **label_smoothing**: 浮点数，范围[0, 1)

#### crossentropy函数
- **预测张量**: 2D FP32张量，已softmax的概率分布
- **标签张量**: 2D FP32张量，与预测张量同形
- **reduction**: "sum"或"mean"，默认"mean"

### 🎛️ 标签平滑公式

当`label_smoothing = α`时：
- **正确类别**: `1 - α + α/num_classes`
- **错误类别**: `α/num_classes`

### 🔢 交叉熵计算

```cpp
crossentropy = -sum(yi * log(pi))

// 数值稳定性处理
pi = max(pi, epsilon)  // epsilon = 1e-12
```

## 实现特性

### ⚡ 性能优化

1. **Eigen批量操作** - 使用Eigen库进行高效向量化计算
2. **内存预分配** - 避免重复内存分配和释放
3. **批量填充策略** - 先填充基础值，再修改特定位置

### 🛡️ 数值稳定性

1. **epsilon保护** - 使用`epsilon = 1e-12`避免log(0)
2. **边界检查** - 验证输入参数的有效性
3. **类型安全** - 严格的类型检查和转换

### 🚨 异常处理

支持完整的异常类型检测：

- **TypeError**: 数据类型错误
- **ShapeError**: 张量形状错误
- **IndexError**: 标签值超出范围
- **ValueError**: 参数值错误

## 使用示例

### 基本One-Hot编码

```cpp
auto cpu_backend = BackendManager::get_cpu_backend();

// 创建标签张量 [0, 1, 2]
Tensor label = cpu_backend->zeros(Shape(3), DType::INT32);
cpu_backend->set_item_int32(label, 1, 1);
cpu_backend->set_item_int32(label, 2, 2);

// 转换为one-hot
Tensor one_hot = cpu_backend->one_hot(label, 3);
```

### 带标签平滑的One-Hot编码

```cpp
// 标签平滑α=0.1
Tensor one_hot_smooth = cpu_backend->one_hot(label, 3, 0.1f);
```

### 交叉熵损失计算

```cpp
// 端到端流程
Tensor label_int = cpu_backend->zeros(Shape(3), DType::INT32);
cpu_backend->set_item_int32(label_int, 1, 2);
cpu_backend->set_item_int32(label_int, 2, 1);

// 生成one-hot标签
Tensor label_onehot = cpu_backend->one_hot(label_int, 3, 0.1f);

// 假设预测张量
Tensor pred = /* 模型输出，已softmax */;

// 计算交叉熵
float loss = cpu_backend->crossentropy(pred, label_onehot, "mean");
```

## 文件结构

### 📁 实现文件

```
include/tech_renaissance/backend/cpu/
└── cpu_backend.h                      # 函数声明

src/backend/cpu/
└── cpu_loss.cpp                       # 实现文件

tests/unit_tests/
└── test_cpu_loss.cpp                  # 完整测试用例
```

### 🔗 构建配置

```cmake
# CMakeLists.txt - 已添加到构建系统
src/backend/cpu/cpu_loss.cpp
```

## 测试覆盖

### 🧪 测试用例

1. **基础功能测试**
   - 基本one-hot编码
   - 标签平滑编码
   - one_hot_into函数

2. **交叉熵测试**
   - 完美预测验证
   - 不确定性预测
   - 数值稳定性
   - reduction模式

3. **端到端测试**
   - 标签→one-hot→交叉熵流程

4. **异常处理测试**
   - 类型错误
   - 形状错误
   - 索引越界
   - 参数值错误

### 📊 测试结果

```
Starting CPU Loss Functions Tests...
======================================
[PASS] Basic one-hot encoding test passed!
[PASS] One-hot encoding with label smoothing test passed!
[PASS] one_hot_into function test passed!
[PASS] Perfect prediction test passed!
[PASS] Uncertain prediction test passed!
[PASS] Numerical stability test passed!
[PASS] Reduction modes test passed!
[PASS] End-to-end pipeline test passed!
[PASS] Error handling tests passed!
All tests passed successfully!
======================================
```

## 性能指标

### ⚡ 优化效果

- **向量化计算**: 使用Eigen库实现SIMD优化
- **内存效率**: 避免不必要的数据拷贝
- **缓存友好**: 行主序内存布局优化

### 📏 基准测试

在Alpha编译环境下的性能表现：
- **编译优化**: Release模式 + /O2 /arch:AVX2
- **数值精度**: IEEE 754单精度浮点
- **内存管理**: RAII自动管理

## 设计原则

### 🎯 架构设计

1. **模块化设计** - 每个函数职责单一，易于维护
2. **类型安全** - 严格的类型检查和转换
3. **性能优先** - 针对CPU架构优化
4. **用户友好** - 清晰的错误信息和文档

### 🔧 代码规范

- **输出信息**: 仅使用英文，无emoji
- **注释文档**: 必须使用中文注释
- **异常处理**: 完整的错误分类和处理
- **测试覆盖**: 100%功能测试覆盖

## 未来扩展

### 🚀 V1.43.0计划

1. **GPU后端支持** - CUDA版本的损失函数
2. **更多损失函数** - MSE、MAE、Huber Loss等
3. **混合精度支持** - FP16/BF16类型支持
4. **批处理优化** - 大批量数据的性能优化

### 📈 性能优化

1. **多线程并行** - OpenMP并行化
2. **指令集优化** - AVX-512支持
3. **内存对齐** - 优化缓存访问模式

## 相关文档

- [张量-后端系统](tensor_backend_system.md) - 了解整体架构
- [张量类文档](tensor.md) - 张量操作接口
- [异常处理](tr_exception.md) - 异常系统说明
- [构建指南](build_settings.md) - 编译和优化配置

---

**注意**: 本实现专为Tech Renaissance框架V1.42.7设计，遵循框架的设计原则和编码规范。在使用时请参考相应的API文档和示例代码。
# CpuBackend API 文档

## # 重要警告：CPU后端张量创建指南！

**CpuBackend是推荐的后端张量创建方式！**

CPU后端提供了完整的张量创建和操作API，是框架的默认计算后端：

**推荐的使用方式：**
```cpp
auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(
    BackendManager::instance().get_backend(CPU));

// 基础张量创建（自动分配内存）
Tensor zeros = cpu_backend->zeros({2, 3, 4}, DType::FP32);
Tensor ones = cpu_backend->ones({2, 3, 4}, DType::FP32);
Tensor full = cpu_backend->full({2, 3, 4}, 1.5f);
Tensor empty = cpu_backend->empty({2, 3, 4}, DType::FP32);

// 随机张量生成
Tensor randn = cpu_backend->randn({2, 3, 4}, 12345);
Tensor uniform = cpu_backend->uniform({2, 3, 4}, 0.0f, 1.0f, 54321);
Tensor randint = cpu_backend->randint({2, 3, 4}, 0, 10, DType::INT32, 99999);

// 类型转换
Tensor int32_tensor = cpu_backend->cast(fp32_tensor, DType::INT32);
Tensor int8_tensor = cpu_backend->cast(fp32_tensor, DType::INT8);
```

**绝对禁止的方式：**
```cpp
// 错误：直接使用构造函数不会分配内存！
Tensor tensor(shape, dtype, CPU);  // 段错误！

// 错误：使用Tensor静态方法（不推荐）
Tensor tensor = Tensor::zeros(shape, dtype, device);

// 错误：误认为Backend基类有这些方法（方法在子类中实现）
auto backend = BackendManager::instance().get_backend(CPU);
Tensor tensor = backend->zeros(shape, dtype);  // 编译错误！
```

## 概述

`CpuBackend`是技术觉醒框架的CPU计算后端实现，继承自`Backend`基类。它提供了基于CPU的高性能张量计算能力，支持Eigen库优化和多线程并行计算，是框架的默认和基础计算后端。

**版本**: V1.43.0
**更新日期**: 2025-11-16
**作者**: 技术觉醒团队

## 🆕 V1.43.0重大更新

### 🎯 新增的高级操作

在V1.43.0版本中，CPU后端新增了多个高级操作方法：

#### 视图操作
```cpp
Tensor view(const Tensor& input, const Shape& new_shape) override;
```
**特性**:
- 零拷贝张量变换，共享底层存储
- 支持连续张量的形状重解释
- 自动内存管理，基于shared_ptr
- 可写视图，修改会反映在原始张量上

#### 形状变换操作
```cpp
Tensor reshape(const Tensor& tensor_a, const Shape& shape) override;
void reshape_inplace(Tensor& tensor_a, const Shape& shape) override;
void reshape_into(const Tensor& tensor_a, Tensor& result, const Shape& shape) override;
```

#### 双曲函数操作
```cpp
Tensor tanh(const Tensor& tensor_a) override;
void tanh_inplace(Tensor& tensor_a) override;
void tanh_into(const Tensor& tensor_a, Tensor& result) override;
Tensor dtanh(const Tensor& tensor_a) override;
void dtanh_inplace(Tensor& tensor_a) override;
void dtanh_into(const Tensor& tensor_a, Tensor& result) override;
```

#### 损失函数操作
```cpp
float crossentropy(const Tensor& pred, const Tensor& label, std::string reduction = "mean") override;
```

#### One-hot编码操作
```cpp
Tensor one_hot(const Tensor& label, int32_t num_classes, float label_smoothing = 0.0f) override;
void one_hot_into(const Tensor& label, Tensor& result, int32_t num_classes, float label_smoothing = 0.0f) override;
```

#### 标量运算和广播运算
```cpp
// 所有V1.43.0新增的标量运算和广播运算方法都已实现
// 包括minus、mac、clamp以及各种广播运算
```

### ✅ 重构兼容性

- **100%向后兼容**：所有现有代码无需修改即可正常工作
- **性能优化**：新增方法基于Eigen库优化，提供高性能计算
- **异常处理**：完善的错误检查和异常处理机制

## 设计理念

### 核心设计原则

1. **行主序存储**：CPU后端使用**行主序（Row-major）**存储张量数据，符合C/C++语言惯例
2. **高性能计算**：基于Eigen库的SIMD优化，支持多线程并行计算
3. **跨后端兼容**：通过`from_cpu()`和`to_cpu()`方法与其他后端保持数据一致性
4. **内存安全**：RAII智能指针自动内存管理，64字节对齐优化SIMD访问
5. **类型安全**：强类型设计防止数据类型错误，完善的边界检查
6. **🆕 宏驱动扩展**：通过V1.43.0的宏系统快速实现新方法

### 关键架构特性

#### **后端管理存储原则（核心特性）**

CPU后端遵循"后端管理存储"的设计原则：
- **CPU后端**：使用行主序（Row-major）存储张量数据，符合C/C++惯例
- **CUDA后端**：使用列主序（Column-major）存储张量数据，与cuBLAS库一致
- **转换层透明**：用户无需关心底层存储格式，`from_cpu()`和`to_cpu()`自动处理转换

#### **行主序存储布局**

```cpp
// CPU张量使用行主序存储
// 2D矩阵 A[M,N] = [[1, 2, 3],
//                  [4, 5, 6]]
// 内存布局：[1, 2, 3, 4, 5, 6]
// 访问方式：data[i * N + j] 获取第i行第j列元素

// 矩阵乘法：C[M,N] = A[M,K] × B[K,N]
for (int32_t i = 0; i < M; ++i) {
    for (int32_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int32_t k = 0; k < K; ++k) {
            sum += a_data[i * K + k] * b_data[k * N + j];
        }
        result_data[i * N + j] = sum;
    }
}
```

## 头文件

```cpp
#include "tech_renaissance/backend/cpu_backend.h"
```

## 主要特性

- **行主序存储**：使用行主序存储格式，符合C/C++语言惯例
- **Eigen优化**：集成Eigen库提供高性能线性代数计算和SIMD优化
- **多线程支持**：基于OpenMP的并行计算，充分利用多核CPU性能
- **内存对齐**：64字节对齐优化，最大化缓存效率
- **🆕 高级操作支持**：V1.43.0新增形状变换、激活函数、损失函数等高级操作

## 构造函数

```cpp
CpuBackend();
```

**描述**：构造CPU后端实例，内部调用`Backend(true)`进行初始化。

**特性**：
- 自动初始化Eigen库
- 设置OpenMP并行计算环境
- 配置内存对齐参数

**示例**：
```cpp
auto cpu_backend = std::make_shared<CpuBackend>();
```

## 张量创建接口

### `Tensor zeros(const Shape& shape, DType dtype = DType::FP32)`

创建全零张量。

**参数**：
- `shape` - 张量形状
- `dtype` - 数据类型（可选，默认FP32）

**返回值**：
- `Tensor` - 全零张量

**异常**：
- `TRException` - 当张量过大或内存不足时抛出

**示例**：
```cpp
Tensor zeros = cpu_backend->zeros({2, 3, 4}, DType::FP32);
```

### `Tensor ones(const Shape& shape, DType dtype = DType::FP32)`

创建全一张量。

**参数**：
- `shape` - 张量形状
- `dtype` - 数据类型（可选，默认FP32）

**返回值**：
- `Tensor` - 全一张量

**示例**：
```cpp
Tensor ones = cpu_backend->ones({2, 3}, DType::INT32);
```

### `Tensor full(const Shape& shape, float value, DType dtype = DType::FP32)`

创建填充指定值的张量。

**参数**：
- `shape` - 张量形状
- `value` - 填充值
- `dtype` - 数据类型（可选，默认FP32）

**返回值**：
- `Tensor` - 填充张量

**示例**：
```cpp
Tensor full = cpu_backend->full({2, 3}, 3.14f, DType::FP32);
```

### `Tensor empty(const Shape& shape, DType dtype = DType::FP32)`

创建未初始化的张量（仅分配内存）。

**参数**：
- `shape` - 张量形状
- `dtype` - 数据类型（可选，默认FP32）

**返回值**：
- `Tensor` - 未初始化的张量

**注意**：张量内容未初始化，使用前必须先填充数据。

**示例**：
```cpp
Tensor empty = cpu_backend->empty({1000, 1000}, DType::FP32);
cpu_backend->fill(empty, 0.0f);  // 使用前先填充
```

## 随机张量生成接口

### `Tensor randn(const Shape& shape, uint64_t seed = 42)`

生成标准正态分布随机张量。

**参数**：
- `shape` - 张量形状
- `seed` - 随机种子（可选，默认42）

**返回值**：
- `Tensor` - 标准正态分布随机张量

**分布**：均值=0，标准差=1的正态分布

**示例**：
```cpp
Tensor randn = cpu_backend->randn({2, 3, 4}, 12345);
```

### `Tensor uniform(const Shape& shape, float min_val = 0.0f, float max_val = 1.0f, uint64_t seed = 42)`

生成均匀分布随机张量。

**参数**：
- `shape` - 张量形状
- `min_val` - 最小值（可选，默认0.0）
- `max_val` - 最大值（可选，默认1.0）
- `seed` - 随机种子（可选，默认42）

**返回值**：
- `Tensor` - 均匀分布随机张量

**示例**：
```cpp
Tensor uniform = cpu_backend->uniform({2, 3}, -5.0f, 5.0f, 54321);
```

### `Tensor randint(const Shape& shape, int32_t low, int32_t high, DType dtype = DType::INT32, uint64_t seed = 42)`

生成整数随机张量。

**参数**：
- `shape` - 张量形状
- `low` - 最小值（包含）
- `high` - 最大值（不包含）
- `dtype` - 数据类型（可选，默认INT32）
- `seed` - 随机种子（可选，默认42）

**返回值**：
- `Tensor` - 整数随机张量

**示例**：
```cpp
Tensor randint = cpu_backend->randint({2, 3}, 0, 10, DType::INT32, 99999);
```

### `Tensor randbool(const Shape& shape, float zero_rate = 0.5f, uint64_t seed = 42)`

生成布尔随机张量（0或1）。

**参数**：
- `shape` - 张量形状
- `zero_rate` - 0值的概率（可选，默认0.5）
- `seed` - 随机种子（可选，默认42）

**返回值**：
- `Tensor` - 布尔随机张量

**示例**：
```cpp
Tensor randbool = cpu_backend->randbool({2, 3}, 0.3f, 77777);
```

## 🆕 V1.43.0新增高级操作

### 形状变换操作

#### `Tensor reshape(const Tensor& tensor_a, const Shape& shape)`

改变张量形状，返回新张量。

**参数**：
- `tensor_a` - 输入张量
- `shape` - 目标形状

**返回值**：
- `Tensor` - 重塑后的张量

**特性**：
- 保持数据总数不变：`tensor_a.numel() == shape.numel()`
- 创建新张量，不修改原张量
- 基于Eigen的高性能实现

**示例**：
```cpp
Tensor input = cpu_backend->ones({2, 3, 4});
Tensor reshaped = cpu_backend->reshape(input, {2, 12});
```

#### `void reshape_inplace(Tensor& tensor_a, const Shape& shape)`

原地改变张量形状。

**参数**：
- `tensor_a` - 输入张量，会被修改
- `shape` - 目标形状

**特性**：
- 就地修改，不创建新张量
- 内存效率更高
- 保持数据总数不变

**示例**：
```cpp
Tensor tensor = cpu_backend->ones({2, 3, 4});
cpu_backend->reshape_inplace(tensor, {6, 4});  // tensor被修改
```

#### `void reshape_into(const Tensor& tensor_a, Tensor& result, const Shape& shape)`

将输入张量重塑到目标张量中。

**参数**：
- `tensor_a` - 输入张量
- `result` - 目标张量，会被修改
- `shape` - 目标形状

**特性**：
- 将tensor_a的数据重塑到result中
- result必须已分配足够的内存
- 高效的数据复制操作

**示例**：
```cpp
Tensor input = cpu_backend->ones({2, 3, 4});
Tensor result = cpu_backend->empty({6, 4});
cpu_backend->reshape_into(input, result, {6, 4});
```

### 双曲函数操作

#### `Tensor tanh(const Tensor& tensor_a)`

计算双曲正切函数。

**参数**：
- `tensor_a` - 输入张量

**返回值**：
- `Tensor` - tanh结果：`tanh(x) = (e^x - e^-x) / (e^x + e^-x)`

**数学公式**：
```
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

**示例**：
```cpp
Tensor input = cpu_backend->randn({2, 3});
Tensor tanh_result = cpu_backend->tanh(input);
```

#### `Tensor dtanh(const Tensor& tensor_a)`

计算双曲正切函数的导数。

**参数**：
- `tensor_a` - 输入张量

**返回值**：
- `Tensor` - dtanh结果：`dtanh(x) = 1 - tanh(x)^2`

**数学公式**：
```
dtanh(x) = 1 - tanh(x)^2
```

**用途**：神经网络反向传播中的梯度计算

**示例**：
```cpp
Tensor tanh_output = cpu_backend->tanh(input);
Tensor grad = cpu_backend->dtanh(tanh_output);
```

### 损失函数操作

#### `float crossentropy(const Tensor& pred, const Tensor& label, std::string reduction = "mean")`

计算交叉熵损失。

**参数**：
- `pred` - 预测张量，形状为[batch_size, num_classes]
- `label` - 标签张量，形状为[batch_size]或[batch_size, num_classes]
- `reduction` - 约简方式："mean"（平均）或"sum"（求和）

**返回值**：
- `float` - 交叉熵损失值

**数学公式**：
```
CE(p, y) = -∑(i) y[i] * log(p[i])
```

**要求**：
- pred数据类型：FP32
- label数据类型：INT32（类别索引）或FP32（one-hot编码）
- pred和label的batch_size必须相同

**示例**：
```cpp
// 类别索引方式
Tensor pred = cpu_backend->randn({4, 10});  // 4个样本，10个类别
Tensor labels = cpu_backend->ones({4}, DType::INT32);  // 类别1
float loss = cpu_backend->crossentropy(pred, labels, "mean");

// One-hot编码方式
Tensor one_hot_labels = cpu_backend->one_hot(labels, 10);
float loss2 = cpu_backend->crossentropy(pred, one_hot_labels, "mean");
```

### One-hot编码操作

#### `Tensor one_hot(const Tensor& label, int32_t num_classes, float label_smoothing = 0.0f)`

将类别标签转换为one-hot编码。

**参数**：
- `label` - 标签张量，形状为[batch_size]，数据类型INT32
- `num_classes` - 类别总数
- `label_smoothing` - 标签平滑参数（可选，默认0.0）

**返回值**：
- `Tensor` - one-hot编码张量，形状为[batch_size, num_classes]

**数学公式**：
- 无标签平滑：`one_hot[i, label[i]] = 1`
- 有标签平滑：`one_hot[i, label[i]] = 1 - ε`，`one_hot[i, j≠label[i]] = ε/(num_classes-1)`

**示例**：
```cpp
Tensor labels = Tensor::from_vector({0, 2, 1, 3}, DType::INT32);  // 4个标签
Tensor one_hot = cpu_backend->one_hot(labels, 10, 0.1f);  // 10个类别，标签平滑0.1
```

### 标量运算操作

#### `Tensor minus(const Tensor& input, float scalar) const`

张量减去标量：`result = input - scalar`

#### `Tensor minus(float scalar, const Tensor& input) const`

标量减去张量：`result = scalar - input`

#### `Tensor mac(const Tensor& input, float scalar_x, float scalar_y) const`

乘加运算：`result = input * scalar_x + scalar_y`

#### `Tensor clamp(const Tensor& input, float min_val, float max_val) const`

张量裁剪：将输入张量限制在[min_val, max_val]范围内

**示例**：
```cpp
Tensor input = cpu_backend->randn({2, 3});
Tensor result1 = cpu_backend->minus(input, 1.0f);  // input - 1.0
Tensor result2 = cpu_backend->minus(2.0f, input);  // 2.0 - input
Tensor result3 = cpu_backend->mac(input, 2.0f, 1.0f);  // input * 2.0 + 1.0
Tensor result4 = cpu_backend->clamp(input, -1.0f, 1.0f);  // 限制在[-1,1]范围内
```

### 广播运算操作

#### `Tensor add_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const`

广播加法：支持不同形状的张量相加

#### `Tensor mul_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const`

广播乘法：支持不同形状的张量相乘

**广播规则**：
- 从右向左比较维度
- 维度大小相等或其中一个为1则可广播
- 不匹配的维度扩展以匹配较大的维度

**示例**：
```cpp
Tensor a = cpu_backend->ones({2, 1, 3});  // 可广播到 {2, 4, 3}
Tensor b = cpu_backend->ones({4, 3});     // 可广播到 {2, 4, 3}
Tensor result = cpu_backend->add_broadcast(a, b);  // 结果形状 {2, 4, 3}
```

## 类型转换接口

### `Tensor cast(const Tensor& tensor, DType target_dtype)`

张量数据类型转换。

**参数**：
- `tensor` - 输入张量
- `target_dtype` - 目标数据类型

**返回值**：
- `Tensor` - 转换后的张量

**支持的转换**：
- FP32 → INT32（截断小数部分）
- FP32 → INT8（截断并限制在[-128, 127]范围）
- INT32 → FP32（直接转换）
- INT8 → FP32（直接转换）

**示例**：
```cpp
Tensor fp32_tensor = cpu_backend->randn({2, 3});
Tensor int32_tensor = cpu_backend->cast(fp32_tensor, DType::INT32);
```

## 内存管理接口

### `Tensor null_tensor()`

返回空张量（不占用内存）。

**返回值**：
- `Tensor` - 空张量

**用途**：
- 变量初始化
- 张量销毁后的状态设置

**示例**：
```cpp
Tensor empty_tensor = cpu_backend->null_tensor();
```

## 使用示例

### 🆕 V1.43.0新功能示例

```cpp
#include "tech_renaissance.h"
using namespace tr;

void v1_43_0_new_features() {
    auto cpu_backend = BackendManager::get_cpu_backend();

    // 1. 形状变换操作
    Tensor input = cpu_backend->randn({2, 3, 4}, 42);
    Tensor reshaped = cpu_backend->reshape(input, {2, 12});
    std::cout << "Reshaped tensor shape: " << reshaped.shape().to_string() << std::endl;

    // 2. 双曲函数操作
    Tensor tanh_result = cpu_backend->tanh(input);
    Tensor dtanh_result = cpu_backend->dtanh(tanh_result);
    std::cout << "Tanh operation completed" << std::endl;

    // 3. 交叉熵损失计算
    Tensor pred = cpu_backend->randn({4, 10});
    Tensor labels = Tensor::from_vector({0, 2, 1, 3}, DType::INT32);
    float loss = cpu_backend->crossentropy(pred, labels, "mean");
    std::cout << "Cross entropy loss: " << loss << std::endl;

    // 4. One-hot编码
    Tensor one_hot = cpu_backend->one_hot(labels, 10, 0.1f);
    std::cout << "One-hot encoding shape: " << one_hot.shape().to_string() << std::endl;

    // 5. 标量运算
    Tensor scaled = cpu_backend->mac(input, 2.0f, 1.0f);  // input * 2 + 1
    Tensor clamped = cpu_backend->clamp(input, -1.0f, 1.0f);

    // 6. 广播运算
    Tensor a = cpu_backend->ones({2, 1, 3});
    Tensor b = cpu_backend->ones({4, 3});
    Tensor broadcast_result = cpu_backend->add_broadcast(a, b);
    std::cout << "Broadcast result shape: " << broadcast_result.shape().to_string() << std::endl;
}
```

### 完整的神经网络示例

```cpp
#include "tech_renaissance.h"
using namespace tr;

void simple_neural_network() {
    auto cpu_backend = BackendManager::get_cpu_backend();

    // 1. 创建模拟数据
    Tensor input = cpu_backend->randn({4, 784});           // 4个样本，784维输入
    Tensor labels = Tensor::from_vector({0, 1, 2, 3}, DType::INT32);  // 4个类别标签

    // 2. 线性变换（模拟全连接层）
    Tensor weights = cpu_backend->randn({784, 10});         // 权重矩阵
    Tensor bias = cpu_backend->zeros({10});               // 偏置

    // 矩阵乘法：output = input × weights + bias
    Tensor matmul_result = cpu_backend->empty({4, 10});
    cpu_backend->mm(matmul_result, input, weights);

    // 加偏置
    Tensor biased = cpu_backend->add(matmul_result, bias);

    // 3. 激活函数
    Tensor activated = cpu_backend->tanh(biased);

    // 4. 计算损失
    float loss = cpu_backend->crossentropy(activated, labels, "mean");
    std::cout << "Neural network loss: " << loss << std::endl;

    // 5. 反向传播（简化版）
    Tensor grad_output = cpu_backend->dtanh(activated);  // tanh的导数
    std::cout << "Gradient computed successfully" << std::endl;
}
```

### 性能测试示例

```cpp
void performance_test() {
    auto cpu_backend = BackendManager::get_cpu_backend();

    // 测试矩阵乘法性能
    const int M = 1024, K = 2048, N = 512;

    Tensor a = cpu_backend->randn({M, K});
    Tensor b = cpu_backend->randn({K, N});
    Tensor result = cpu_backend->empty({M, N});

    auto start = std::chrono::high_resolution_clock::now();
    cpu_backend->mm(result, a, b);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double gflops = (2.0 * M * K * N) / (duration.count() * 1e6) / 1e9;

    std::cout << "CPU MM Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
}
```

## 性能特性

### 计算性能

- **矩阵乘法**：基于Eigen库优化，支持SIMD指令
- **多线程并行**：OpenMP自动并行化，充分利用多核CPU
- **内存对齐**：64字节对齐，最大化缓存命中率

### 内存效率

- **智能指针管理**：自动内存回收，避免内存泄漏
- **就地操作**：提供inplace版本，减少内存分配
- **零拷贝优化**：reshape等操作无需数据复制

### 数值精度

- **IEEE 754**：严格遵循IEEE 754浮点数标准
- **数值稳定性**：算法实现考虑数值稳定性
- **精度验证**：与PyTorch等框架对比验证

## 错误处理

### 常见异常

```cpp
try {
    auto cpu_backend = BackendManager::get_cpu_backend();

    // 形状不匹配
    Tensor a = cpu_backend->ones({2, 3});
    Tensor b = cpu_backend->ones({3, 4});
    // Tensor result = cpu_backend->add(a, b);  // 抛出TRException

    // 数据类型不匹配
    Tensor fp32_tensor = cpu_backend->ones({2, 3}, DType::FP32);
    Tensor int32_tensor = cpu_backend->ones({2, 3}, DType::INT32);
    // Tensor result = cpu_backend->add(fp32_tensor, int32_tensor);  // 抛出TRException

} catch (const TRException& e) {
    std::cerr << "CPU Backend error: " << e.what() << std::endl;
}
```

### 错误类型

- **形状错误**：张量形状不兼容
- **类型错误**：数据类型不匹配
- **内存错误**：内存分配失败或不足
- **参数错误**：函数参数超出有效范围

## 最佳实践

1. **使用BackendManager**：通过BackendManager获取CPU后端实例
2. **类型检查**：在计算前检查张量的数据类型和形状
3. **内存管理**：利用就地操作减少内存分配
4. **异常处理**：所有操作都应包含适当的异常处理
5. **性能优化**：对于大张量操作，考虑使用多线程并行
6. **🆕 利用新特性**：使用V1.43.0新增的高级操作简化代码

## 版本信息

- **版本**: V1.43.0
- **更新日期**: 2025-11-16
- **作者**: 技术觉醒团队
- **主要更新**:
  - 🆕 新增形状变换操作：reshape系列方法
  - 🆕 新增双曲函数：tanh、dtanh系列方法
  - 🆕 新增损失函数：crossentropy
  - 🆕 新增One-hot编码：one_hot系列方法
  - 🆕 新增标量运算：minus、mac、clamp系列方法
  - 🆕 新增广播运算：add_broadcast、mul_broadcast系列方法
  - ✅ 所有新方法都基于Eigen库优化
  - ✅ 100%向后兼容，现有代码无需修改
  - ✅ 完善的异常处理和错误检查
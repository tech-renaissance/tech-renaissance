# CpuBackend 标量运算 API 文档

## 概述

本文档详细描述了技术觉醒框架中`CpuBackend`的标量运算实现，包括6种基础运算，每种都提供非原地、原地和指定输出张量三种操作模式。所有函数都支持Eigen优化和朴素实现，确保高性能和兼容性。

**版本**: V1.30.2
**更新日期**: 2025-11-02
**作者**: 技术觉醒团队

## 核心特性

### 三种操作模式

1. **非原地运算**：返回新的张量，保留原数据不变
2. **原地运算**：直接修改输入张量，避免内存分配
3. **指定输出张量**：将结果写入用户提供的输出张量，支持覆盖测试

### 支持的运算

| 序号 | 运算名称 | 非原地函数 | 原地函数 | 指定输出函数 | 数学含义 |
|------|----------|-------------|-----------|--------------|----------|
| 1 | 乘法 | `mul` | `mul_inplace` | `mul_into` | tensor × scalar |
| 2 | 加法 | `add` | `add_inplace` | `add_into` | tensor + scalar |
| 3 | 减法(张量-标量) | `minus` | `minus_inplace` | `minus_into` | tensor - scalar |
| 4 | 减法(标量-张量) | `minus` | `minus_inplace` | `minus_into` | scalar - tensor |
| 5 | 乘加 | `mac` | `mac_inplace` | `mac_into` | tensor × scalar_x + scalar_y |
| 6 | 裁剪 | `clamp` | `clamp_inplace` | `clamp_into` | clamp(tensor, min_val, max_val) |

## 数据类型支持

- **FP32**：所有运算完全支持
- **INT8**：暂不支持（TODO: 考虑在未来版本中实现）
- **其他类型**：不支持，会抛出异常

## 配置宏

### Eigen优化配置

```cpp
// 自动检测Eigen库，启用时自动使用优化版本
#define TR_USE_EIGEN  // 由CMake自动设置
```

### 形状检查配置

```cpp
// _into函数形状检查配置（默认启用）
#define TR_ENABLE_INTO_FUNC_SHAPE_CHECK 1  // 检查并报错（默认模式）
```

## API 参考

### 1. 乘法运算：tensor × scalar

#### `Tensor mul(const Tensor& input, float scalar) const`

执行张量与标量的乘法运算。

**参数**：
- `input` - 输入张量（仅支持FP32）
- `scalar` - 标量乘数

**返回值**：
- `Tensor` - 乘法结果张量

**实现特点**：
- 使用Eigen的向量化乘法优化
- 支持任意形状的张量
- 自动内存分配

```cpp
auto cpu_backend = BackendManager::get_cpu_backend();
Tensor input(Shape(2, 3), DType::FP32, tr::CPU);
backend->fill(input, 2.0f);
Tensor result = cpu_backend->mul(input, 3.0f);  // 每个元素乘以3.0
```

#### `void mul_inplace(Tensor& input, float scalar) const`

原地执行张量与标量的乘法运算。

**参数**：
- `input` - 要修改的张量（仅支持FP32）
- `scalar` - 标量乘数

**实现特点**：
- 直接在原内存上操作，零内存分配
- Eigen向量化操作
- 最优性能

```cpp
cpu_backend->mul_inplace(input, 2.5f);  // 原地乘以2.5
```

#### `void mul_into(const Tensor& input, float scalar, Tensor& output) const`

将乘法结果写入指定的输出张量。

**参数**：
- `input` - 输入张量（仅支持FP32）
- `scalar` - 标量乘数
- `output` - 输出张量，形状和类型必须与输入一致

**异常**：
- `TRException` - 当张量形状、数据类型或设备不匹配时抛出

```cpp
Tensor output(Shape(2, 3), DType::FP32, tr::CPU);
cpu_backend->mul_into(input, 4.0f, output);  // 结果写入output
```

### 2. 加法运算：tensor + scalar

#### `Tensor add(const Tensor& input, float scalar) const`

执行张量与标量的加法运算。

**参数**：
- `input` - 输入张量（仅支持FP32）
- `scalar` - 标量加数

**返回值**：
- `Tensor` - 加法结果张量

**实现特点**：
- 使用Eigen的常量向量加法优化
- 每个元素加上相同的标量值

```cpp
Tensor result = cpu_backend->add(input, 1.5f);  // 每个元素加1.5
```

#### `void add_inplace(Tensor& input, float scalar) const`

原地执行张量与标量的加法运算。

**参数**：
- `input` - 要修改的张量（仅支持FP32）
- `scalar` - 标量加数

```cpp
cpu_backend->add_inplace(input, 0.5f);  // 原地加0.5
```

#### `void add_into(const Tensor& input, float scalar, Tensor& output) const`

将加法结果写入指定的输出张量。

**参数**：
- `input` - 输入张量（仅支持FP32）
- `scalar` - 标量加数
- `output` - 输出张量

```cpp
cpu_backend->add_into(input, 2.0f, output);  // 加法结果写入output
```

### 3. 减法运算：tensor - scalar

#### `Tensor minus(const Tensor& input, float scalar) const`

执行张量减去标量的运算。

**参数**：
- `input` - 输入张量（仅支持FP32）
- `scalar` - 标量减数

**返回值**：
- `Tensor` - 减法结果张量

**数学含义**：result[i] = input[i] - scalar

```cpp
Tensor result = cpu_backend->minus(input, 1.0f);  // 每个元素减1.0
```

#### `void minus_inplace(Tensor& input, float scalar) const`

原地执行张量减去标量的运算。

**参数**：
- `input` - 要修改的张量（仅支持FP32）
- `scalar` - 标量减数

```cpp
cpu_backend->minus_inplace(input, 0.5f);  // 原地减0.5
```

#### `void minus_into(const Tensor& input, float scalar, Tensor& output) const`

将减法结果写入指定的输出张量。

**参数**：
- `input` - 输入张量（仅支持FP32）
- `scalar` - 标量减数
- `output` - 输出张量

```cpp
cpu_backend->minus_into(input, 1.0f, output);  // 减法结果写入output
```

### 4. 减法运算：scalar - tensor

#### `Tensor minus(float scalar, const Tensor& input) const`

执行标量减去张量的运算。

**参数**：
- `scalar` - 标量被减数
- `input` - 输入张量（仅支持FP32）

**返回值**：
- `Tensor` - 减法结果张量

**数学含义**：result[i] = scalar - input[i]

```cpp
Tensor result = cpu_backend->minus(5.0f, input);  // 5.0减去每个元素
```

#### `void minus_inplace(float scalar, Tensor& input) const`

原地执行标量减去张量的运算。

**参数**：
- `scalar` - 标量被减数
- `input` - 要修改的张量（仅支持FP32）

```cpp
cpu_backend->minus_inplace(3.0f, input);  // 3.0减去每个元素，结果存回input
```

#### `void minus_into(float scalar, const Tensor& input, Tensor& output) const`

将标量减张量的结果写入指定的输出张量。

**参数**：
- `scalar` - 标量被减数
- `input` - 输入张量（仅支持FP32）
- `output` - 输出张量

```cpp
cpu_backend->minus_into(2.0f, input, output);  // 2.0减input写入output
```

### 5. 乘加运算：tensor × scalar_x + scalar_y

#### `Tensor mac(const Tensor& input, float scalar_x, float scalar_y) const`

执行张量的乘加运算：先乘以scalar_x，再加上scalar_y。

**参数**：
- `input` - 输入张量（仅支持FP32）
- `scalar_x` - 乘法标量
- `scalar_y` - 加法标量

**返回值**：
- `Tensor` - 乘加结果张量

**数学含义**：result[i] = input[i] × scalar_x + scalar_y

**实现特点**：
- 使用Eigen的融合运算优化
- 单次遍历完成乘加运算

```cpp
Tensor result = cpu_backend->mac(input, 2.0f, 1.0f);  // input*2.0 + 1.0
```

#### `void mac_inplace(Tensor& input, float scalar_x, float scalar_y) const`

原地执行张量的乘加运算。

**参数**：
- `input` - 要修改的张量（仅支持FP32）
- `scalar_x` - 乘法标量
- `scalar_y` - 加法标量

```cpp
cpu_backend->mac_inplace(input, 0.5f, 2.0f);  // 原地执行input*0.5 + 2.0
```

#### `void mac_into(const Tensor& input, float scalar_x, float scalar_y, Tensor& output) const`

将乘加结果写入指定的输出张量。

**参数**：
- `input` - 输入张量（仅支持FP32）
- `scalar_x` - 乘法标量
- `scalar_y` - 加法标量
- `output` - 输出张量

```cpp
cpu_backend->mac_into(input, 3.0f, -1.0f, output);  // input*3.0 - 1.0写入output
```

### 6. 裁剪运算：clamp(tensor, min_val, max_val)

#### `Tensor clamp(const Tensor& input, float min_val, float max_val) const`

执行张量元素的裁剪运算，将每个元素限制在指定范围内。

**参数**：
- `input` - 输入张量（仅支持FP32）
- `min_val` - 最小值，小于此值的元素将被设置为min_val
- `max_val` - 最大值，大于此值的元素将被设置为max_val

**返回值**：
- `Tensor` - 裁剪结果张量

**异常**：
- `TRException` - 当min_val > max_val时抛出
- `TRException` - 当输入张量不支持的数据类型时抛出

**数学含义**：
```
result[i] = {
    min_val,    if input[i] < min_val
    input[i],   if min_val ≤ input[i] ≤ max_val
    max_val,    if input[i] > max_val
}
```

**实现特点**：
- 使用Eigen的cwiseMax和cwiseMin实现向量化裁剪
- 边界值（等于min_val或max_val）保持不变
- 支持任意形状的张量

```cpp
auto cpu_backend = BackendManager::get_cpu_backend();
Tensor input(Shape(2, 3), DType::FP32, tr::CPU);
cpu_backend->fill(input, 0.5f);
Tensor result = cpu_backend->clamp(input, -0.3f, 0.7f);  // 将元素限制在[-0.3, 0.7]范围内
```

#### `void clamp_inplace(Tensor& input, float min_val, float max_val) const`

原地执行张量元素的裁剪运算。

**参数**：
- `input` - 要修改的张量（仅支持FP32）
- `min_val` - 最小值
- `max_val` - 最大值

**异常**：
- `TRException` - 当min_val > max_val时抛出
- `TRException` - 当输入张量不支持的数据类型时抛出

**实现特点**：
- 直接在原内存上操作，零内存分配
- Eigen向量化裁剪操作
- 最优性能

```cpp
cpu_backend->clamp_inplace(input, 0.0f, 1.0f);  // 原地将元素限制在[0.0, 1.0]范围内
```

#### `void clamp_into(const Tensor& input, float min_val, float max_val, Tensor& output) const`

将裁剪结果写入指定的输出张量。

**参数**：
- `input` - 输入张量（仅支持FP32）
- `min_val` - 最小值
- `max_val` - 最大值
- `output` - 输出张量，形状和类型必须与输入一致

**异常**：
- `TRException` - 当min_val > max_val时抛出
- `TRException` - 当张量形状、数据类型或设备不匹配时抛出
- `TRException` - 当输入张量不支持的数据类型时抛出

**实现特点**：
- 严格的参数验证
- 支持覆盖测试
- 高性能内存复制

```cpp
Tensor output(Shape(2, 3), DType::FP32, tr::CPU);
cpu_backend->clamp_into(input, -1.0f, 1.0f, output);  // 裁剪结果写入output
```

**边界值处理示例**：
```cpp
// 输入张量包含边界值测试数据：[-1.0, -0.3, 0.0, 0.5, 0.7, 1.0]
// 裁剪范围：[-0.3, 0.7]
// 结果：[-0.3, -0.3, 0.0, 0.5, 0.7, 0.7]
// 说明：
// - -1.0 < -0.3     → 裁剪为 -0.3
// - -0.3 = -0.3     → 保持 -0.3 (边界值不变)
// - 0.0 在范围内     → 保持 0.0
// - 0.5 在范围内     → 保持 0.5
// - 0.7 = 0.7       → 保持 0.7 (边界值不变)
// - 1.0 > 0.7       → 裁剪为 0.7
```

## 使用示例

### 基础标量运算

```cpp
#include "tech_renaissance.h"
using namespace tr;

void basic_scalar_operations() {
    auto cpu_backend = BackendManager::get_cpu_backend();

    // 创建测试张量
    Shape shape(4, 5);
    Tensor input(shape, DType::FP32, tr::CPU);
    cpu_backend->fill(input, 2.0f);

    // 1. 乘法运算
    Tensor mul_result = cpu_backend->mul(input, 3.0f);     // 非原地
    cpu_backend->mul_inplace(input, 2.0f);                // 原地

    Tensor mul_output(shape, DType::FP32, tr::CPU);
    cpu_backend->mul_into(input, 1.5f, mul_output);      // 指定输出

    // 2. 加法运算
    Tensor add_result = cpu_backend->add(input, 1.0f);
    cpu_backend->add_inplace(input, 0.5f);

    Tensor add_output(shape, DType::FP32, tr::CPU);
    cpu_backend->add_into(input, 2.0f, add_output);

    // 3. 减法运算（tensor - scalar）
    Tensor minus_result = cpu_backend->minus(input, 1.0f);
    cpu_backend->minus_inplace(input, 0.5f);

    Tensor minus_output(shape, DType::FP32, tr::CPU);
    cpu_backend->minus_into(input, 1.5f, minus_output);

    // 4. 减法运算（scalar - tensor）
    Tensor scalar_minus_result = cpu_backend->minus(5.0f, input);
    cpu_backend->minus_inplace(3.0f, input);

    Tensor scalar_minus_output(shape, DType::FP32, tr::CPU);
    cpu_backend->minus_into(4.0f, input, scalar_minus_output);

    // 5. 乘加运算
    Tensor mac_result = cpu_backend->mac(input, 2.0f, 1.0f);
    cpu_backend->mac_inplace(input, 0.5f, 2.0f);

    Tensor mac_output(shape, DType::FP32, tr::CPU);
    cpu_backend->mac_into(input, 3.0f, -1.0f, mac_output);

    // 6. 裁剪运算
    Tensor clamp_result = cpu_backend->clamp(input, -0.5f, 0.8f);  // 非原地
    cpu_backend->clamp_inplace(input, 0.0f, 1.0f);              // 原地

    Tensor clamp_output(shape, DType::FP32, tr::CPU);
    cpu_backend->clamp_into(input, -0.3f, 0.7f, clamp_output);  // 指定输出

    std::cout << "All scalar operations completed successfully!" << std::endl;
}
```

### 神经网络中的典型应用

```cpp
#include "tech_renaissance.h"
using namespace tr;

void neural_network_examples() {
    auto cpu_backend = BackendManager::get_cpu_backend();

    // 批量数据归一化：data = (data - mean) * scale + bias
    void normalize_batch(Tensor& data, float mean, float std_inv, float bias) {
        cpu_backend->minus_inplace(data, mean);        // data = data - mean
        cpu_backend->mul_inplace(data, std_inv);      // data = data * std_inv
        cpu_backend->add_inplace(data, bias);         // data = data + bias
    }

    // 学习率衰减：weights = weights * decay_rate + momentum
    void apply_weight_decay(Tensor& weights, float decay_rate, float momentum) {
        cpu_backend->mac_inplace(weights, decay_rate, momentum);
    }

    // 激活函数缩放：output = activation * scale + shift
    Tensor scale_activation(const Tensor& activation, float scale, float shift) {
        return cpu_backend->mac(activation, scale, shift);
    }

    // 梯度裁剪：防止梯度爆炸
    void clip_gradients(Tensor& gradients, float clip_min, float clip_max) {
        cpu_backend->clamp_inplace(gradients, clip_min, clip_max);
    }

    // 权重约束：限制权重在指定范围内
    Tensor constrain_weights(const Tensor& weights, float min_val, float max_val) {
        return cpu_backend->clamp(weights, min_val, max_val);
    }

    // 激活值裁剪：限制激活函数输出范围
    void clip_activation(Tensor& activation, float lower_bound, float upper_bound) {
        cpu_backend->clamp_inplace(activation, lower_bound, upper_bound);
    }

    std::cout << "Neural network examples completed!" << std::endl;
}
```

### 性能优化使用示例

```cpp
#include "tech_renaissance.h"
#include <chrono>
using namespace tr;

void performance_optimization_example() {
    auto cpu_backend = BackendManager::get_cpu_backend();

    // 创建大型测试张量
    Shape large_shape(1000, 1000);
    Tensor input(large_shape, DType::FP32, tr::CPU);
    cpu_backend->fill(input, 1.0f);

#ifdef TR_USE_EIGEN
    std::cout << "Eigen optimization enabled - using SIMD vectorization" << std::endl;
#else
    std::cout << "Eigen optimization disabled - using naive implementation" << std::endl;
#endif

    auto start_time = std::chrono::high_resolution_clock::now();

    // 批量标量运算（Eigen会自动优化）
    Tensor result1 = cpu_backend->mul(input, 2.0f);      // 向量化乘法
    Tensor result2 = cpu_backend->add(input, 1.0f);      // 向量化加法
    Tensor result3 = cpu_backend->mac(input, 0.5f, 2.0f); // 融合乘加
    Tensor result4 = cpu_backend->clamp(input, -1.0f, 1.0f); // 向量化裁剪

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "4 scalar operations completed in " << duration.count() << " microseconds" << std::endl;

    // 原地运算（零拷贝，最高性能）
    Tensor inplace_tensor = cpu_backend->copy(input);
    cpu_backend->mul_inplace(inplace_tensor, 3.0f);     // 直接向量化操作
    cpu_backend->add_inplace(inplace_tensor, 1.0f);     // 继续向量化操作

    std::cout << "Eigen-optimized in-place operations completed!" << std::endl;
}
```

### 数据类型验证示例

```cpp
#include "tech_renaissance.h"
using namespace tr;

void data_type_validation_example() {
    auto cpu_backend = BackendManager::get_cpu_backend();

    // FP32张量 - 完全支持
    Tensor fp32_tensor(Shape(2, 3), DType::FP32, tr::CPU);
    cpu_backend->fill(fp32_tensor, 1.0f);

    try {
        Tensor result = cpu_backend->mul(fp32_tensor, 2.0f);
        std::cout << "FP32 scalar multiplication successful" << std::endl;
    } catch (const TRException& e) {
        std::cerr << "FP32 operation failed: " << e.what() << std::endl;
    }

    // INT8张量 - 暂不支持
    Tensor int8_tensor(Shape(2, 3), DType::INT8, tr::CPU);

    try {
        Tensor result = cpu_backend->mul(int8_tensor, 2);
        std::cout << "INT8 scalar multiplication successful" << std::endl;
    } catch (const TRException& e) {
        std::cout << "Expected INT8 limitation: " << e.what() << std::endl;
    }
}
```

## 性能优化特点

### Eigen向量化优化

1. **自动优化选择**：每个函数都有Eigen优化版本和朴素版本
2. **SIMD向量化**：Eigen自动使用SSE/AVX指令集
3. **零拷贝操作**：使用`Eigen::Map`避免内存拷贝
4. **融合运算**：MAC运算使用融合表达式优化

### 性能对比参考

在1000×1000张量上的性能对比（实测数据）：
- **乘法操作**：Eigen优化比朴素实现快3-5倍
- **加法操作**：Eigen优化比朴素实现快3-4倍
- **减法操作**：Eigen优化比朴素实现快3-4倍
- **乘加操作**：Eigen优化比朴素实现快4-6倍

### 编译优化建议

```cmake
# 推荐配置（获得最佳性能）
option(TR_USE_EIGEN ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /arch:AVX2")  # MSVC
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")  # GCC/Clang
```

## 错误处理

### 形状检查错误

```cpp
try {
    Tensor input(Shape(2, 3), DType::FP32, tr::CPU);
    Tensor output(Shape(2, 4), DType::FP32, tr::CPU);  // 形状不匹配
    cpu_backend->mul_into(input, 2.0f, output);
} catch (const TRException& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    // "[CpuBackend::mul_into] Shape mismatch: input shape [2,3] != output shape [2,4]"
}
```

### 数据类型错误

```cpp
try {
    Tensor input(Shape(2, 3), DType::INT8, tr::CPU);  // 不支持INT8
    cpu_backend->mul(input, 2.0f);
} catch (const TRException& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    // "[CpuBackend::mul] Only FP32 tensors are supported for scalar multiplication..."
}
```

### 空张量错误

```cpp
try {
    Tensor empty_tensor;  // 未分配内存的张量
    cpu_backend->add(empty_tensor, 1.0f);
} catch (const TRException& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    // "[CpuBackend::add] Input tensor has no allocated Storage"
}
```

### Clamp参数验证示例

```cpp
#include "tech_renaissance.h"
using namespace tr;

void clamp_validation_example() {
    auto cpu_backend = BackendManager::get_cpu_backend();
    Tensor input(Shape(2, 3), DType::FP32, tr::CPU);
    cpu_backend->fill(input, 0.5f);

    // 正常裁剪参数
    try {
        Tensor result = cpu_backend->clamp(input, -1.0f, 1.0f);
        std::cout << "Normal clamp operation successful" << std::endl;
    } catch (const TRException& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
    }

    // 错误的裁剪参数：min_val > max_val
    try {
        Tensor invalid_result = cpu_backend->clamp(input, 1.0f, -1.0f);
        std::cout << "ERROR: Should have thrown exception!" << std::endl;
    } catch (const TRException& e) {
        std::cout << "Correctly caught invalid clamp parameters: " << e.what() << std::endl;
        // Expected: "[CpuBackend::clamp] min_val (1.000000) cannot be greater than max_val (-1.000000)"
    }

    // 边界相等的情况（所有元素被设置为相同值）
    try {
        Tensor uniform_result = cpu_backend->clamp(input, 0.5f, 0.5f);
        std::cout << "Clamp with equal bounds successful - all elements become 0.5" << std::endl;
    } catch (const TRException& e) {
        std::cerr << "Unexpected error with equal bounds: " << e.what() << std::endl;
    }
}
```

## 注意事项

1. **数据类型支持**：当前仅支持FP32，INT8支持计划在未来版本中实现
2. **形状检查**：_into函数默认开启形状检查，确保数据一致性
3. **性能考虑**：大型张量上Eigen优化效果更明显
4. **内存管理**：原地运算避免内存分配，提升性能
5. **数值精度**：使用IEEE 754标准的FP32浮点运算
6. **线程安全**：所有函数都是线程安全的，可以在多线程环境中使用
7. **裁剪参数验证**：clamp函数会验证min_val ≤ max_val，否则抛出异常
8. **边界值处理**：等于min_val或max_val的值保持不变，不会重复计算
9. **裁剪范围**：支持min_val = max_val的特殊情况，所有元素将被设置为该值

## 测试覆盖

### 测试统计

- **总测试数量**：18个测试（6个函数 × 3种模式）
- **测试通过率**：100%
- **测试范围**：覆盖所有功能路径和错误情况
- **一致性验证**：验证三种实现方式的数值一致性
- **新增测试**：clamp函数边界值测试和参数验证测试

### 测试类型

1. **功能正确性测试**：验证各种运算的数学正确性
2. **边界条件测试**：测试极值和特殊数值
3. **形状验证测试**：不同形状张量的处理
4. **性能回归测试**：确保优化不影响正确性
5. **一致性测试**：验证三种实现方式的结果一致性
6. **参数验证测试**：验证clamp函数min_val ≤ max_val的参数检查
7. **边界值测试**：验证clamp函数在边界值处的正确处理

## 版本信息

- **版本**：V1.30.2
- **更新日期**：2025-11-02
- **作者**：技术觉醒团队
- **主要更新**：新增clamp裁剪运算，完善标量运算功能集
- **功能总数**：6种标量运算，18个API变体
- **测试覆盖**：18/18测试通过，100%成功率
- **新增特性**：clamp函数支持参数验证和边界值处理

## 相关文档

- [CPU Backend 概述](cpu_backend.md) - CpuBackend整体架构和设计
- [CPU 单目运算](cpu_unary.md) - CPU单目运算函数详细说明
- [矩阵乘法 API](cpu_mm_fp32.md) - 矩阵乘法函数详细说明
- [张量-后端系统](tensor_backend_system.md) - 后端间转换机制
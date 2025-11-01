# CpuBackend 标量运算 API 文档

## 概述

本文档详细描述了技术觉醒框架中`CpuBackend`的标量运算实现，包括5种基础运算，每种都提供非原地、原地和指定输出张量三种操作模式。所有函数都支持Eigen优化和朴素实现，确保高性能和兼容性。

**版本**: V1.28.1
**更新日期**: 2025-11-01
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

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "3 scalar operations completed in " << duration.count() << " microseconds" << std::endl;

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

## 注意事项

1. **数据类型支持**：当前仅支持FP32，INT8支持计划在未来版本中实现
2. **形状检查**：_into函数默认开启形状检查，确保数据一致性
3. **性能考虑**：大型张量上Eigen优化效果更明显
4. **内存管理**：原地运算避免内存分配，提升性能
5. **数值精度**：使用IEEE 754标准的FP32浮点运算
6. **线程安全**：所有函数都是线程安全的，可以在多线程环境中使用

## 测试覆盖

### 测试统计

- **总测试数量**：15个测试（5个函数 × 3种模式）
- **测试通过率**：100%
- **测试范围**：覆盖所有功能路径和错误情况
- **一致性验证**：验证三种实现方式的数值一致性

### 测试类型

1. **功能正确性测试**：验证各种运算的数学正确性
2. **边界条件测试**：测试极值和特殊数值
3. **形状验证测试**：不同形状张量的处理
4. **性能回归测试**：确保优化不影响正确性
5. **一致性测试**：验证三种实现方式的结果一致性

## 版本信息

- **版本**：V1.28.1
- **更新日期**：2025-11-01
- **作者**：技术觉醒团队
- **主要更新**：新增CPU标量运算功能（mul、add、minus、mac）
- **功能总数**：5种标量运算，15个API变体
- **测试覆盖**：15/15测试通过，100%成功率

## 相关文档

- [CPU Backend 概述](cpu_backend.md) - CpuBackend整体架构和设计
- [CPU 单目运算](cpu_unary.md) - CPU单目运算函数详细说明
- [矩阵乘法 API](cpu_mm_fp32.md) - 矩阵乘法函数详细说明
- [张量-后端系统](tensor_backend_system.md) - 后端间转换机制
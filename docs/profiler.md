# Profiler性能分析器API文档

**版本**: V1.25.1
**日期**: 2025-10-31
**作者**: 技术觉醒团队
**所属系列**: utils

## 概述

Profiler类是技术觉醒框架的核心性能分析工具，专门用于深度学习计算的性能测试和分析。它提供了高精度的计时功能、自动化的FLOPS计算以及简洁的性能报告接口。

## 特性

- **高精度计时**: 基于C++ `std::chrono::steady_clock`，提供微秒级精度
- **自动FLOPS计算**: 支持矩阵乘法等常见操作的浮点运算次数自动计算
- **多格式支持**: 同时支持2D和4D张量形状描述
- **线程安全**: 支持多线程环境下的性能测试
- **异常安全**: 完整的错误检查和异常处理机制

## 快速开始

```cpp
#include "tech_renaissance/utils/profiler.h"

using namespace tr;

// 创建Profiler实例
Profiler profiler;

// 设置测试参数
profiler.set_iterations(100);                    // 设置迭代次数
profiler.describe_operation("mm", shape_a, shape_b); // 描述操作类型

// 开始性能测试
profiler.start();
for (int i = 0; i < 100; ++i) {
    // 执行要测试的操作，例如矩阵乘法
    backend->mm(result, tensor_a, tensor_b);
}
profiler.stop();

// 获取性能结果
double performance = profiler.get_performance();  // GFLOPS
double avg_time = profiler.avg_time();           // 平均时间(ms)
double total_time = profiler.total_time();       // 总时间(ms)
```

## API参考

### 构造函数

#### `Profiler()`

创建一个新的Profiler实例，初始化所有内部状态。

```cpp
Profiler profiler;
```

**初始状态**:
- `timer_started_`: `false`
- `iterations_`: `1`
- `total_`: `-1.0`
- `flops_`: `-1`

---

### 基础计时方法

#### `void start()`

开始计时。如果计时器已经在运行，将抛出异常。

```cpp
profiler.start();
```

**异常**:
- `TRException`: 如果计时器已经启动

#### `void stop()`

停止计时并计算总耗时。如果计时器未启动，将抛出异常。

```cpp
profiler.stop();
```

**功能**:
- 记录结束时间点
- 计算总耗时（毫秒）并存储在 `total_` 中

**异常**:
- `TRException`: 如果计时器未启动

---

### 配置方法

#### `void set_iterations(int iterations)`

设置性能测试的迭代次数。

```cpp
profiler.set_iterations(100);
```

**参数**:
- `iterations` (int): 迭代次数，必须为正数

**异常**:
- `TRException`: 如果迭代次数小于等于0

#### `void describe_operation(const std::string& operation_type, Shape shape_a, Shape shape_b)`

描述要测试的操作类型和输入张量形状，用于自动计算FLOPS。

```cpp
profiler.describe_operation("mm", shape_a, shape_b);
```

**参数**:
- `operation_type` (const std::string&): 操作类型，目前支持 "mm"（矩阵乘法）
- `shape_a` (Shape): 第一个输入张量的形状
- `shape_b` (Shape): 第二个输入张量的形状

**支持的操作类型**:

| 类型 | 描述 | FLOPS计算公式 |
|------|------|---------------|
| "mm" | 矩阵乘法 | 2 × M × K × N |

**形状处理**:
- **2D Shape**: 使用 `h()` 和 `w()` 方法获取维度
- **4D Shape**: 使用 `n()` 和 `c()` 方法获取维度

**异常**:
- `TRException`: 如果不支持的操作类型

---

### 结果获取方法

#### `double avg_time() const`

获取平均每次操作的执行时间（毫秒）。

```cpp
double avg_ms = profiler.avg_time();
```

**返回值**:
- `double`: 平均时间（毫秒）

**异常**:
- `TRException`: 如果计时器仍在运行
- `TRException`: 如果迭代次数无效
- `TRException`: 如果计时未完成

#### `double total_time() const`

获取所有操作的总执行时间（毫秒）。

```cpp
double total_ms = profiler.total_time();
```

**返回值**:
- `double`: 总时间（毫秒）

**异常**:
- `TRException`: 如果计时器仍在运行

#### `double get_performance()`

获取计算性能，以GFLOPS（十亿次浮点运算每秒）为单位。

```cpp
double gflops = profiler.get_performance();
```

**返回值**:
- `double`: 性能值（GFLOPS）

**计算公式**:
```
GFLOPS = FLOPS / (平均时间 × 1e6)
```

**异常**:
- `TRException`: 如果操作类型未指定
- `TRException`: 如果迭代次数无效
- `TRException`: 如果计时未完成

---

## 使用示例

### 基础性能测试

```cpp
#include "tech_renaissance.h"
#include <iostream>

using namespace tr;

int main() {
    // 创建测试数据
    auto cuda_backend = BackendManager::get_cuda_backend();
    Tensor a = Tensor::randn(Shape(1024, 2048), 42);
    Tensor b = Tensor::randn(Shape(2048, 512), 42);
    Tensor result = Tensor::empty(Shape(1024, 512), DType::FP32, CUDA[0]);

    // 转换到CUDA
    Tensor cuda_a = cuda_backend->from_cpu(a);
    Tensor cuda_b = cuda_backend->from_cpu(b);

    // 创建Profiler并配置
    Profiler profiler;
    profiler.set_iterations(100);
    profiler.describe_operation("mm", cuda_a.shape(), cuda_b.shape());

    // 预热
    for (int i = 0; i < 10; ++i) {
        cuda_backend->mm(result, cuda_a, cuda_b);
    }
    cuda_backend->synchronize();

    // 性能测试
    profiler.start();
    for (int i = 0; i < 100; ++i) {
        cuda_backend->mm(result, cuda_a, cuda_b);
    }
    cuda_backend->synchronize();
    profiler.stop();

    // 输出结果
    std::cout << "Performance: " << profiler.get_performance() << " GFLOPS" << std::endl;
    std::cout << "Average time: " << profiler.avg_time() << " ms" << std::endl;
    std::cout << "Total time: " << profiler.total_time() << " ms" << std::endl;

    return 0;
}
```

### 多操作性能比较

```cpp
void compare_operations() {
    auto backend = BackendManager::get_cuda_backend();

    // 创建不同大小的测试数据
    std::vector<std::pair<Shape, Shape>> test_shapes = {
        {Shape(512, 1024), Shape(1024, 512)},
        {Shape(1024, 2048), Shape(2048, 1024)},
        {Shape(2048, 4096), Shape(4096, 2048)}
    };

    for (const auto& shapes : test_shapes) {
        Profiler profiler;
        profiler.set_iterations(50);
        profiler.describe_operation("mm", shapes.first, shapes.second);

        // 创建测试张量
        Tensor a = Tensor::randn(shapes.first, 42);
        Tensor b = Tensor::randn(shapes.second, 42);
        Tensor c = Tensor::empty(Shape(a.shape()[0], b.shape()[1]), DType::FP32, CUDA[0]);

        profiler.start();
        for (int i = 0; i < 50; ++i) {
            backend->mm(c, a, b);
        }
        backend->synchronize();
        profiler.stop();

        std::cout << "Shape " << shapes.first.to_string() << " × "
                  << shapes.second.to_string() << ": "
                  << profiler.get_performance() << " GFLOPS" << std::endl;
    }
}
```

---

## 最佳实践

### 1. 预热运行

```cpp
// 推荐：始终进行预热
for (int i = 0; i < 10; ++i) {
    backend->mm(result, a, b);
}
backend->synchronize(); // 确保预热完成
```

### 2. 合理设置迭代次数

```cpp
// 小规模操作：增加迭代次数
profiler.set_iterations(1000);

// 大规模操作：适当减少迭代次数
profiler.set_iterations(20);
```

### 3. 错误处理

```cpp
try {
    profiler.start();
    // 执行测试
    profiler.stop();
    double performance = profiler.get_performance();
} catch (const TRException& e) {
    std::cerr << "Profiler error: " << e.what() << std::endl;
}
```

### 4. 多次测试求平均

```cpp
const int test_runs = 5;
double total_performance = 0.0;

for (int run = 0; run < test_runs; ++run) {
    Profiler profiler;
    profiler.set_iterations(100);
    profiler.describe_operation("mm", shape_a, shape_b);

    // 执行测试...

    total_performance += profiler.get_performance();
}

double avg_performance = total_performance / test_runs;
```

---

## 注意事项

1. **内存管理**: Profiler不会管理测试数据的内存，请确保张量在测试期间有效
2. **设备同步**: 对于CUDA操作，请确保在调用 `stop()` 之前同步设备
3. **线程安全**: 每个Profiler实例是线程安全的，但不要在多个线程间共享同一个实例
4. **精度限制**: 时间精度受限于系统时钟，通常为微秒级
5. **操作类型**: 目前仅支持矩阵乘法，后续版本将支持更多操作类型

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| V1.25.1 | 2025-10-31 | 初始版本，支持矩阵乘法性能分析，简化API设计 |

---

## 相关文档

- [数据类型文档](dtype.md)
- [张量操作文档](tensor.md)
- [后端接口文档](backend.md)
- [CUDA后端文档](cuda_backend.md)
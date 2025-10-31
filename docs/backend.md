# Backend API 文档

## 概述

`Backend`是技术觉醒框架的抽象后端基类，定义了所有计算后端（CPU、CUDA等）必须实现的统一接口。它采用纯虚函数设计，确保不同计算设备的后端实现具有一致的API接口，从而实现设备无关的张量计算。

**版本**: V1.23.1
**更新日期**: 2025-10-30
**作者**: 技术觉醒团队

## 设计理念

### 核心设计原则

1. **抽象接口**：纯虚基类，定义统一的后端操作接口
2. **设备无关**：提供与具体硬件无关的计算抽象
3. **后端管理存储**：每个后端负责管理自己的张量存储格式
4. **跨后端转换**：通过`from_cpu()`、`to_cpu()`和`to()`方法实现跨设备数据转换
5. **RAII设计**：使用智能指针管理内存，避免资源泄漏
6. **前向声明**：避免循环依赖，不包含具体实现的头文件
7. **异常安全**：完善的错误处理机制

### 关键架构特性

#### **跨后端转换接口（V1.23.1核心特性）**

Backend类定义了标准的跨后端转换接口：
- **`from_cpu()`**: 从CPU转换到当前后端设备，自动处理内存布局转换
- **`to_cpu()`**: 从当前后端设备转换到CPU，自动处理内存布局转换
- **`to()`**: 通用的设备转换接口

#### **后端存储管理原则**

每个后端实现负责管理自己的张量存储格式：
- **CPU后端**：使用行主序（Row-major）存储，符合C/C++惯例
- **CUDA后端**：使用列主序（Column-major）存储，与cuBLAS库接口一致
- **转换层透明**：用户无需关心底层存储格式，转换层自动处理

## 头文件

```cpp
#include "tech_renaissance/backend/backend.h"
```

## 纯虚接口

### 内存管理接口

#### `virtual std::shared_ptr<void> allocate(size_t size) = 0`

分配指定大小的内存。

**参数：**
- `size` - 要分配的内存大小（字节）

**返回值：**
- `std::shared_ptr<void>` - 内存块的智能指针

**异常：**
- `TRException` - 当分配失败时抛出

**示例：**
```cpp
auto backend = manager.get_backend(tr::CPU);
auto memory = backend->allocate(1024);  // 分配1KB内存
```

#### `virtual void deallocate(void* ptr) = 0`

释放已分配的内存。

**参数：**
- `ptr` - 要释放的内存指针

**异常：**
- `TRException` - 当释放失败时抛出

**示例：**
```cpp
// 注意：通常通过智能指针自动管理，很少直接调用
backend->deallocate(raw_ptr);
```

#### `virtual void* get_data_ptr(const std::shared_ptr<void>& holder) = 0`

从内存持有者中获取原始数据指针。

**参数：**
- `holder` - 内存智能指针

**返回值：**
- `void*` - 原始数据指针

**示例：**
```cpp
auto memory = backend->allocate(1024);
void* data_ptr = backend->get_data_ptr(memory);
```

#### `virtual void copy_data(void* dst, const void* src, size_t size, const Device& dst_device, const Device& src_device) = 0`

在设备间复制数据。

**参数：**
- `dst` - 目标内存指针
- `src` - 源内存指针
- `size` - 复制大小（字节）
- `dst_device` - 目标设备
- `src_device` - 源设备

**异常：**
- `TRException` - 当复制失败时抛出

**支持的复制方向：**
- CPU → CPU
- CPU → CUDA
- CUDA → CPU
- CUDA → CUDA

**示例：**
```cpp
// 从CPU复制到CUDA
backend->copy_data(gpu_ptr, cpu_ptr, size, tr::CUDA[0], tr::CPU);
```

### 张量操作接口

#### `virtual void fill(Tensor& dst, float value) = 0`

用浮点数值填充张量。

**参数：**
- `dst` - 目标张量
- `value` - 填充值

**异常：**
- `TRException` - 当张量数据类型不是FP32时抛出

**示例：**
```cpp
tr::Tensor tensor(shape, tr::DType::FP32, tr::CPU);
backend->fill(tensor, 3.14f);  // 将所有元素填充为3.14
```

#### `virtual void fill(Tensor& dst, int8_t value) = 0`

用8位整数值填充张量。

**参数：**
- `dst` - 目标张量
- `value` - 填充值

**异常：**
- `TRException` - 当张量数据类型不是INT8时抛出

**示例：**
```cpp
tr::Tensor tensor(shape, tr::DType::INT8, tr::CPU);
backend->fill(tensor, 42);  // 将所有元素填充为42
```

#### `virtual void add(Tensor& result, const Tensor& a, const Tensor& b) = 0`

执行逐元素加法运算：result = a + b。

**参数：**
- `result` - 结果张量
- `a` - 第一个操作数张量
- `b` - 第二个操作数张量

**异常：**
- `TRException` - 当张量形状、数据类型或设备不匹配时抛出

**前提条件：**
- 所有张量必须具有相同的形状
- 所有张量必须具有相同的数据类型
- 所有张量必须位于同一设备上

**示例：**
```cpp
tr::Tensor a(shape, tr::DType::FP32, tr::CPU);
tr::Tensor b(shape, tr::DType::FP32, tr::CPU);
tr::Tensor result(shape, tr::DType::FP32, tr::CPU);

backend->fill(a, 2.0f);
backend->fill(b, 3.0f);
backend->add(result, a, b);  // result = [5.0, 5.0, ...]
```

#### `virtual void mul(Tensor& result, const Tensor& a, const Tensor& b) = 0`

执行逐元素乘法运算：result = a * b。

**参数**：
- `result` - 结果张量
- `a` - 第一个操作数张量
- `b` - 第二个操作数张量

**异常**：
- `TRException` - 当张量形状、数据类型或设备不匹配时抛出

**前提条件**：
- 所有张量必须具有相同的形状
- 所有张量必须具有相同的数据类型
- 所有张量必须位于同一设备上

**示例**：
```cpp
tr::Tensor a(shape, tr::DType::FP32, tr::CPU);
tr::Tensor b(shape, tr::DType::FP32, tr::CPU);
tr::Tensor result(shape, tr::DType::FP32, tr::CPU);

backend->fill(a, 2.0f);
backend->fill(b, 3.0f);
backend->mul(result, a, b);  // result = [6.0, 6.0, ...]
```

#### `virtual void mm(Tensor& result, const Tensor& a, const Tensor& b) = 0`

执行矩阵乘法运算：result = a × b。

**参数**：
- `result` - 结果张量，形状应为(M,N)
- `a` - 输入张量A，形状应为(M,K)
- `b` - 输入张量B，形状应为(K,N)

**异常**：
- `TRException` - 当张量形状、数据类型或设备不匹配时抛出

**前提条件**：
- 仅支持FP32数据类型
- A的列数必须等于B的行数：a.width() == b.height()
- 所有张量必须位于同一设备上

**矩阵乘法规则**：C(M,N) = A(M,K) × B(K,N)

**示例**：
```cpp
// A: 1024×2048, B: 2048×512, C: 1024×512
tr::Tensor a(tr::Shape(1024, 2048), tr::DType::FP32, device);
tr::Tensor b(tr::Shape(2048, 512), tr::DType::FP32, device);
tr::Tensor result(tr::Shape(1024, 512), tr::DType::FP32, device);

backend->mm(result, a, b);  // 执行矩阵乘法
```

### 跨后端转换接口（V1.23.1核心特性）

#### `virtual Tensor from_cpu(const Tensor& tensor) const = 0`

从CPU转换张量到当前后端设备，自动处理内存布局转换。

**参数**：
- `tensor` - CPU设备上的张量（行主序存储）

**返回值**：
- `Tensor` - 当前后端设备上的张量（后端特定存储格式）

**关键特性**：
- **CPU → CUDA**：行主序转换为列主序（对于2D矩阵）
- **CPU → CPU**：恒等变换（对于CPU后端）
- **内存分配**：自动在目标设备上分配内存
- **格式转换**：2D矩阵执行转置操作，非2D张量直接复制

**示例**：
```cpp
auto cuda_backend = BackendManager::get_cuda_backend();
Tensor cpu_tensor = Tensor::randn(Shape(2, 3), 42);
Tensor cuda_tensor = cuda_backend->from_cpu(cpu_tensor);  // 行主序 → 列主序
```

#### `virtual Tensor to_cpu(const Tensor& tensor) const = 0`

从当前后端设备转换张量到CPU，自动处理内存布局转换。

**参数**：
- `tensor` - 当前后端设备上的张量（后端特定存储格式）

**返回值**：
- `Tensor` - CPU设备上的张量（行主序存储）

**关键特性**：
- **CUDA → CPU**：列主序转换为行主序（对于2D矩阵）
- **CPU → CPU**：恒等变换（对于CPU后端）
- **内存分配**：自动在CPU上分配内存
- **格式转换**：2D矩阵执行转置操作，非2D张量直接复制

**示例**：
```cpp
auto cuda_backend = BackendManager::get_cuda_backend();
Tensor cuda_tensor = /* CUDA张量 */;
Tensor cpu_tensor = cuda_backend->to_cpu(cuda_tensor);  // 列主序 → 行主序
```

#### `virtual Tensor to(const Tensor& tensor, const Device& target_device) const = 0`

通用的设备转换接口，支持在任意设备间转换张量。

**参数**：
- `tensor` - 源张量
- `target_device` - 目标设备

**返回值**：
- `Tensor` - 目标设备上的张量

**使用场景**：
- 当源设备和目标设备相同时，直接返回原张量
- 当转换到CPU时，调用`to_cpu()`
- 当转换到其他设备时，委托给目标设备的后端处理

**示例**：
```cpp
auto backend = BackendManager::get_backend(tr::CPU);
Tensor tensor = /* 某个张量 */;
Tensor cpu_tensor = backend->to(tensor, tr::CPU);  // 转换到CPU
```

### 设备信息接口

#### `virtual Device device() const = 0`

获取后端对应的设备信息。

**返回值：**
- `Device` - 后端设备对象

**示例：**
```cpp
auto device = backend->device();
std::cout << "Backend device: " << device.to_string() << std::endl;
```

#### `virtual std::string name() const = 0`

获取后端的名称。

**返回值：**
- `std::string` - 后端名称

**示例：**
```cpp
std::cout << "Backend name: " << backend->name() << std::endl;
// 输出："CpuBackend" 或 "CudaBackend"
```

## 使用示例

### 跨后端矩阵乘法（V1.23.1推荐用法）

```cpp
#include "tech_renaissance.h"
using namespace tr;

void cross_backend_matrix_multiplication() {
    try {
        // 获取后端实例（V1.23.1新API）
        auto cpu_backend = BackendManager::get_cpu_backend();
        auto cuda_backend = BackendManager::get_cuda_backend();

        // 创建CPU随机张量（行主序存储）
        Tensor cpu_a = Tensor::randn(Shape(1024, 2048), 42);
        Tensor cpu_b = Tensor::randn(Shape(2048, 512), 42);

        // CPU矩阵乘法（行主序计算）
        Tensor cpu_result = Tensor::empty(Shape(1024, 512), DType::FP32, tr::CPU);
        cpu_backend->mm(cpu_result, cpu_a, cpu_b);

        // 转换到CUDA（自动转换为列主序）
        Tensor cuda_a = cuda_backend->from_cpu(cpu_a);  // 行主序 → 列主序
        Tensor cuda_b = cuda_backend->from_cpu(cpu_b);
        Tensor cuda_result = Tensor::empty(Shape(1024, 512), DType::FP32, tr::CUDA[0]);

        // CUDA矩阵乘法（列主序计算）
        cuda_backend->mm(cuda_result, cuda_a, cuda_b);

        // 转换回CPU（自动转换回行主序）
        Tensor cuda_result_cpu = cuda_backend->to_cpu(cuda_result);  // 列主序 → 行主序

        // 结果验证：CPU和CUDA结果在行主序下应该一致
        bool is_close = cpu_backend->is_close(cpu_result, cuda_result_cpu, 1e-4f);
        std::cout << "Results are close: " << (is_close ? "YES" : "NO") << std::endl;

    } catch (const TRException& e) {
        std::cerr << "Backend error: " << e.what() << std::endl;
    }
}
```

### 基本张量操作

```cpp
#include "tech_renaissance.h"
using namespace tr;

void basic_operations() {
    try {
        auto backend = BackendManager::get_cpu_backend();

        // 创建张量
        Shape shape(2, 3);
        Tensor a(shape, DType::FP32, tr::CPU);
        Tensor b(shape, DType::FP32, tr::CPU);
        Tensor result(shape, DType::FP32, tr::CPU);

        // 填充张量
        backend->fill(a, 1.5f);
        backend->fill(b, 2.5f);

        // 执行加法
        backend->add(result, a, b);

        std::cout << "Basic operations completed successfully!" << std::endl;

    } catch (const TRException& e) {
        std::cerr << "Backend error: " << e.what() << std::endl;
    }
}
```

### 通用设备转换

```cpp
void universal_device_conversion() {
    try {
        auto backend = BackendManager::get_backend(tr::CPU);

        // 创建CPU张量
        Tensor cpu_tensor = Tensor::randn(Shape(3, 4), 42);

        // 使用通用to()方法转换到CPU（恒等操作）
        Tensor same_cpu_tensor = backend->to(cpu_tensor, tr::CPU);

        // 转换到CUDA（通过CUDA后端）
        auto cuda_backend = BackendManager::get_cuda_backend();
        Tensor cuda_tensor = cuda_backend->to(cpu_tensor, tr::CUDA[0]);

        std::cout << "Universal conversion completed!" << std::endl;

    } catch (const TRException& e) {
        std::cerr << "Conversion error: " << e.what() << std::endl;
    }
}
```

### 内存管理示例

```cpp
void memory_management() {
    auto backend = tr::BackendManager::instance().get_backend(tr::CPU);

    // 分配内存
    const size_t size = 1024 * sizeof(float);
    auto memory = backend->allocate(size);
    float* data = static_cast<float*>(backend->get_data_ptr(memory));

    // 使用内存
    for (size_t i = 0; i < 1024; ++i) {
        data[i] = static_cast<float>(i);
    }

    // 内存会通过智能指针自动释放，无需手动释放
    std::cout << "Memory allocated and will be automatically freed" << std::endl;
}
```

## 实现后端

要创建自定义后端，需要继承Backend类并实现所有纯虚函数：

```cpp
class CustomBackend : public tr::Backend {
public:
    // 实现所有纯虚函数
    std::shared_ptr<void> allocate(size_t size) override {
        // 自定义内存分配逻辑
    }

    void deallocate(void* ptr) override {
        // 自定义内存释放逻辑
    }

    // ... 其他方法的实现
};
```

## 注意事项

1. **数据类型检查**：每个操作前都会检查张量的数据类型，确保操作兼容
2. **形状验证**：二元操作前会验证操作数张量的形状是否匹配
3. **设备一致性**：操作涉及的所有张量必须位于同一设备上
4. **内存安全**：所有内存操作都通过智能指针管理，避免内存泄漏
5. **异常处理**：所有方法都可能抛出`TRException`，调用时应妥善处理

## 性能考虑

- 后端实例创建后可重复使用，避免频繁创建销毁
- 大数据量的跨设备复制操作开销较大，应谨慎使用
- 不同后端的性能特征可能差异很大，应根据计算需求选择合适的后端

### 数据访问接口

#### `virtual float get_scalar_float(const Tensor& tensor) = 0`

获取标量张量的数据（float版本）。

**参数：**
- `tensor` - 标量张量

**返回值：**
- `float` - 标量值

**异常：**
- `TRException` - 当不是标量张量或数据类型不匹配时抛出

**示例：**
```cpp
tr::Tensor scalar(tr::Shape(), tr::DType::FP32, tr::CPU);
backend->fill(scalar, 3.14f);
float value = backend->get_scalar_float(scalar);  // 返回 3.14f
```

#### `virtual int32_t get_scalar_int32(const Tensor& tensor) = 0`

获取标量张量的数据（int32_t版本）。

**参数：**
- `tensor` - 标量张量

**返回值：**
- `int32_t` - 标量值

**异常：**
- `TRException` - 当不是标量张量或数据类型不匹配时抛出

**示例：**
```cpp
tr::Tensor scalar(tr::Shape(), tr::DType::INT32, tr::CPU);
int32_t value = backend->get_scalar_int32(scalar);
```

#### `virtual int8_t get_scalar_int8(const Tensor& tensor) = 0`

获取标量张量的数据（int8_t版本）。

**参数：**
- `tensor` - 标量张量

**返回值：**
- `int8_t` - 标量值

**异常：**
- `TRException` - 当不是标量张量或数据类型不匹配时抛出

**示例：**
```cpp
tr::Tensor scalar(tr::Shape(), tr::DType::INT8, tr::CPU);
int8_t value = backend->get_scalar_int8(scalar);
```

## 关键设计原则总结（V1.23.1）

### 后端管理存储
- **CPU后端**：使用行主序存储，符合C/C++语言惯例
- **CUDA后端**：使用列主序存储，与cuBLAS/cuDNN库接口一致
- **转换透明**：`from_cpu()`和`to_cpu()`自动处理格式转换

### 统一接口设计
- **抽象基类**：纯虚函数确保所有后端实现一致的API
- **设备无关**：用户代码无需关心底层硬件差异
- **跨后端转换**：标准化的转换接口支持无缝设备切换

### 内存管理
- **RAII设计**：智能指针自动管理内存生命周期
- **前向声明**：避免循环依赖，保持代码清洁
- **异常安全**：完善的错误处理和资源清理

## 最佳实践

1. **使用BackendManager**：通过BackendManager获取后端实例，而不是直接创建
2. **跨后端操作**：使用`from_cpu()`和`to_cpu()`进行设备间转换
3. **内存管理**：依赖智能指针自动管理内存，避免手动释放
4. **异常处理**：所有后端操作都应包含适当的异常处理
5. **设备一致性**：确保计算操作中的所有张量都在同一设备上

## 实现指导

要创建自定义后端，需要继承Backend类并实现所有纯虚函数：

```cpp
class CustomBackend : public tr::Backend {
public:
    // 内存管理接口
    std::shared_ptr<void> allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void* get_data_ptr(const std::shared_ptr<void>& holder) override;
    void copy_data(void* dst, const void* src, size_t size,
                   const Device& dst_device, const Device& src_device) const override;

    // 张量操作接口
    void fill(Tensor& dst, float value) override;
    void fill(Tensor& dst, int8_t value) override;
    void add(Tensor& result, const Tensor& a, const Tensor& b) override;
    void mul(Tensor& result, const Tensor& a, const Tensor& b) override;
    void mm(Tensor& result, const Tensor& a, const Tensor& b) override;

    // 跨后端转换接口
    Tensor to(const Tensor& tensor, const Device& device) const override;
    Tensor to_cpu(const Tensor& tensor) const override;
    Tensor from_cpu(const Tensor& tensor) const override;

    // 辅助接口
    std::string name() const override;
    Device device() const override;

    // 数据访问接口
    float get_scalar_float(const Tensor& tensor) override;
    int32_t get_scalar_int32(const Tensor& tensor) override;
    int8_t get_scalar_int8(const Tensor& tensor) override;
};
```

## 版本信息

- **版本**：V1.23.1
- **更新日期**：2025-10-30
- **作者**：技术觉醒团队
- **主要特性**：跨后端转换接口、矩阵乘法支持、统一抽象设计
# Storage API 文档

## 概述

`Storage`类是技术觉醒框架的内存管理核心，负责张量数据的存储、设备内存分配和RAII管理。它通过智能指针提供自动内存管理，同时为Backend类提供底层内存访问接口。

**版本**: V1.23.1
**更新日期**: 2025-10-30
**作者**: 技术觉醒团队

## 设计理念

### 核心设计原则

1. **RAII内存管理**：使用智能指针自动管理内存生命周期，防止内存泄漏，即使在异常情况下也能确保正确释放。
2. **设备抽象**：Storage类不关心具体的设备类型，只负责管理内存。设备相关的操作委托给Backend类。
3. **智能指针封装**：通过`std::shared_ptr<void>`提供引用计数，支持多个Tensor共享同一块内存，实现高效的零拷贝操作。
4. **Backend接口**：提供专门的方法供Backend类访问原始内存，同时保持对用户的安全封装。
5. **容量管理**：区分已分配容量和实际使用大小，支持内存预留和灵活扩展。

## 类结构

```cpp
class Storage {
    std::shared_ptr<void> data_ptr_;  // 智能指针管理的内存块
    size_t size_;                     // 实际使用大小（字节）
    size_t capacity_;                 // 已分配容量（字节）
    Device device_;                   // 内存所在设备
    DType dtype_;                     // 数据类型
};
```

## 核心API

### 构造函数

#### `Storage()`
创建空的Storage对象。

#### `Storage(size_t size, const Device& device, DType dtype)`
创建指定大小、设备和数据类型的Storage。

**参数**：
- `size`：要分配的内存大小（字节）
- `device`：内存所在的设备
- `dtype`：存储的数据类型

### 内存访问

#### `void* data_ptr() noexcept`
返回原始内存指针。

#### `const void* data_ptr() const noexcept`
返回常量原始内存指针。

### 设备和数据类型访问

#### `const Device& device() const noexcept`
返回Storage所在的设备。

#### `DType dtype() const noexcept`
返回Storage存储的数据类型。

### 容量管理

#### `size_t size() const noexcept`
返回实际使用的内存大小。

#### `size_t capacity() const noexcept`
返回已分配的内存容量。

#### `size_t size_bytes() const noexcept`
返回以字节为单位的内存大小。

#### `size_t capacity_bytes() const noexcept`
返回以字节为单位的内存容量。

### 空状态检查

#### `bool empty() const noexcept`
检查Storage是否为空。

#### `operator bool() const noexcept`
布尔转换操作符。

### Backend专用接口

#### `std::shared_ptr<void> holder() const noexcept`
返回智能指针持有者。

### 内存操作

#### `void resize(size_t new_size)`
调整Storage大小。

#### `void reserve(size_t new_capacity)`
预留内存容量。

#### `void clear()`
清空Storage内容。

### 工具方法

#### `std::string to_string() const`
返回Storage的字符串表示。

## 内存布局和后端管理（V1.23.1关键特性）

### **多后端存储原则**

Storage类作为后端无关的内存抽象层，负责：
- **内存分配**：根据设备类型分配相应内存
- **生命周期管理**：通过智能指针自动管理内存
- **元数据管理**：维护设备、数据类型、大小等信息
- **设备无关性**：不关心具体的内存布局格式

### **后端特定的内存布局**

虽然Storage类本身是设备无关的，但不同的Backend会以不同的格式存储数据：

```cpp
// CPU Storage：行主序存储
Storage cpu_storage = Storage::allocate(1024 * sizeof(float), tr::CPU, DType::FP32);
float* cpu_data = static_cast<float*>(cpu_storage.data_ptr());
// cpu_data[i] 访问第i个元素（行主序）

// CUDA Storage：列主序存储
Storage cuda_storage = Storage::allocate(1024 * sizeof(float), tr::CUDA[0], DType::FP32);
float* cuda_data = static_cast<float*>(cuda_storage.data_ptr());
// cuda_data[i] 访问第i个元素（列主序）
```

### **Storage在跨后端转换中的作用**

Storage在后端转换中保持数据完整性，而转换层处理格式变化：

```cpp
// CPU -> CUDA 转换过程
Tensor CudaBackend::from_cpu(const Tensor& cpu_tensor) {
    // 1. 创建CUDA Storage（设备特定）
    Storage cuda_storage = Storage::allocate(
        cpu_tensor.memory_size(),
        tr::CUDA[device_id_],
        cpu_tensor.dtype()
    );

    // 2. 执行格式转换（行主序 → 列主序）
    convert_layout_cpu_to_cuda(cpu_tensor.storage(), cuda_storage);

    // 3. 返回新的Tensor（包含转换后的Storage）
    return Tensor(cpu_tensor.shape(), cuda_storage);
}

// CUDA -> CPU 转换过程
Tensor CudaBackend::to_cpu(const Tensor& cuda_tensor) {
    // 1. 创建CPU Storage（设备特定）
    Storage cpu_storage = Storage::allocate(
        cuda_tensor.memory_size(),
        tr::CPU,
        cuda_tensor.dtype()
    );

    // 2. 执行格式转换（列主序 → 行主序）
    convert_layout_cuda_to_cpu(cuda_tensor.storage(), cpu_storage);

    // 3. 返回新的Tensor（包含转换后的Storage）
    return Tensor(cuda_tensor.shape(), cpu_storage);
}
```

### **内存布局转换的底层实现**

转换层负责处理不同内存布局之间的数据转换：

```cpp
// 2D矩阵的内存布局转换
void convert_layout_cpu_to_cuda(const Storage& cpu_storage, Storage& cuda_storage) {
    if (/* 是2D矩阵 */) {
        int32_t M = /* 行数 */;
        int32_t N = /* 列数 */;

        const float* cpu_data = static_cast<const float*>(cpu_storage.data_ptr());
        float* cuda_data = static_cast<float*>(cuda_storage.data_ptr());

        // 行主序 → 列主序转换
        for (int32_t i = 0; i < M; ++i) {
            for (int32_t j = 0; j < N; ++j) {
                cuda_data[j * M + i] = cpu_data[i * N + j];
            }
        }
    }
}

void convert_layout_cuda_to_cpu(const Storage& cuda_storage, Storage& cpu_storage) {
    if (/* 是2D矩阵 */) {
        int32_t M = /* 行数 */;
        int32_t N = /* 列数 */;

        const float* cuda_data = static_cast<const float*>(cuda_storage.data_ptr());
        float* cpu_data = static_cast<float*>(cpu_storage.data_ptr());

        // 列主序 → 行主序转换
        for (int32_t i = 0; i < M; ++i) {
            for (int32_t j = 0; j < N; ++j) {
                cpu_data[i * N + j] = cuda_data[j * M + i];
            }
        }
    }
}
```

## 使用示例

### 基础Storage创建

```cpp
#include "tech_renaissance/data/storage.h"
using namespace tr;

// 创建CPU上的FP32 Storage（行主序）
Device cpu_device = tr::CPU;
Storage cpu_storage(1024, cpu_device, DType::FP32);

// 创建CUDA上的FP32 Storage（列主序）
Device gpu_device = tr::CUDA[0];
Storage gpu_storage(2048, gpu_device, DType::FP32);
```

### 跨后端内存操作

```cpp
// 创建CPU张量（行主序存储）
Tensor cpu_a = Tensor::randn(Shape(1024, 2048), 42);

// 转换到CUDA（自动转换为列主序）
Tensor cuda_a = cuda_backend->from_cpu(cpu_a);
// 此时cuda_a.storage()包含列主序数据

// 直接访问CUDA Storage（列主序）
Storage& cuda_storage = cuda_a.storage();
float* cuda_data = static_cast<float*>(cuda_storage.data_ptr());
// cuda_data[j * M + i] 访问第i行第j列的元素

// 转换回CPU（自动转换回行主序）
Tensor cpu_result = cuda_backend->to_cpu(cuda_result);
// 此时cpu_result.storage()包含行主序数据
```

### Backend接口使用

```cpp
// Backend类访问Storage的示例
class MatrixMultiplier {
public:
    void multiply_cpu(const Storage& a_storage, const Storage& b_storage, Storage& result_storage) {
        // CPU存储：行主序访问
        const float* a_data = static_cast<const float*>(a_storage.data_ptr());
        const float* b_data = static_cast<const float*>(b_storage.data_ptr());
        float* result_data = static_cast<float*>(result_storage.data_ptr());

        // 行主序矩阵乘法
        // result[i][j] = sum(a[i][k] * b[k][j])
        // result_data[i * N + j] = sum(a_data[i * K + k] * b_data[k * N + j])
    }

    void multiply_cuda(const Storage& a_storage, const Storage& b_storage, Storage& result_storage) {
        // CUDA存储：列主序访问
        const float* a_data = static_cast<const float*>(a_storage.data_ptr());
        const float* b_data = static_cast<const float*>(b_storage.data_ptr());
        float* result_data = static_cast<float*>(result_storage.data_ptr());

        // 列主序矩阵乘法（cuBLAS接口）
        // 调用cublasSgemm处理列主序数据
        cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                   &alpha, b_data, N, a_data, K, &beta, result_data, N);
    }
};
```

## 性能特征

### 内存对齐

- FP32数据：4字节对齐
- INT8数据：1字节对齐
- 设备特定的对齐要求

### 缓存友好性

- 紧凑的对象布局（通常32-48字节）
- 连续内存分配
- 智能指针的引用计数局部性

### 零拷贝操作

```cpp
// 多个Tensor共享同一Storage
Storage shared_storage(1024, tr::CPU, DType::FP32);
std::shared_ptr<void> holder = shared_storage.holder();

// Backend可以使用shared_ptr实现零拷贝
// 多个Tensor引用相同的内存块
```

## 线程安全

### 读操作
- 所有const方法都是线程安全的
- 只读操作不会修改共享状态

### 写操作
- 非const方法需要外部同步
- 智能指针的引用计数操作是原子的

## 错误处理

### 内存分配失败

```cpp
try {
    Storage large_storage(SIZE_MAX, tr::CPU, DType::FP32);
} catch (const std::bad_alloc& e) {
    std::cerr << "Memory allocation failed: " << e.what() << std::endl;
}
```

## 与框架的集成

### Tensor集成

```cpp
// Tensor使用Storage管理数据
class Tensor {
private:
    Shape shape_;
    std::shared_ptr<Storage> storage_;

public:
    Tensor(const Shape& shape, DType dtype, const Device& device) {
        size_t bytes = shape.numel() * dtype_size(dtype);
        storage_ = std::make_shared<Storage>(bytes, device, dtype);
    }

    // 跨后端转换方法
    Tensor to(const Device& target_device) const {
        if (device() == target_device) return *this;

        // 调用Backend的转换方法
        return BackendManager::get_backend(target_device)->to(*this, target_device);
    }
};
```

## 最佳实践

### 内存预分配

```cpp
// 好的做法：预估内存需求
Storage storage;
storage.reserve(expected_max_size);
```

### 跨后端操作

```cpp
// 好的做法：使用Backend的转换方法
Tensor cpu_tensor = /* ... */;
Tensor cuda_tensor = cuda_backend->from_cpu(cpu_tensor);  // 自动处理格式转换

// 避免：手动操作不同格式的数据
```

## 未来扩展

### 内存池支持
支持预分配内存池以提高分配性能。

### 异步内存管理
支持异步内存操作以提高并发性能。

### 跨设备内存共享
支持统一虚拟地址以简化多设备编程。

---

## 版本信息

- **版本**: V1.23.1
- **更新日期**: 2025-10-30
- **作者**: 技术觉醒团队
- **主要更新**: 强化了跨后端存储管理、内存布局转换等关键特性
# BackendManager API 文档

## 概述

`BackendManager`是技术觉醒框架的后端管理器类，采用单例模式设计，负责管理所有计算后端（CPU、CUDA等）的注册、初始化和访问。它提供了统一的接口来获取不同设备的计算后端，确保多线程环境下的安全访问。

**版本**: V1.23.1
**更新日期**: 2025-10-30
**作者**: 技术觉醒团队

## 设计理念

### 核心设计原则

1. **单例模式**：采用Meyers单例模式，确保全局唯一实例，C++11保证线程安全
2. **后端管理存储支持**：管理与不同存储格式的后端（CPU行主序、CUDA列主序）
3. **类型安全访问**：提供类型安全的静态便利方法，简化常用后端访问
4. **自动初始化**：程序启动时自动检测和注册可用的计算后端
5. **线程安全**：使用互斥锁保证多线程环境下的安全访问
6. **异常安全**：完善的错误处理和日志记录机制

### 关键架构特性

#### **V1.23.1静态便利方法**

为简化跨后端操作，BackendManager提供了类型安全的静态便利方法：
- **`get_cpu_backend()`**: 直接返回CpuBackend智能指针
- **`get_cuda_backend(device_id)`**: 直接返回指定设备的CudaBackend智能指针

#### **跨后端管理支持**

BackendManager支持管理具有不同存储格式的后端：
- **CPU后端**: 行主序存储，符合C/C++惯例
- **CUDA后端**: 列主序存储，与cuBLAS库接口一致
- **统一接口**: 通过Backend基类提供一致的访问接口

## 头文件

```cpp
#include "tech_renaissance/backend/backend_manager.h"
```

## 主要特性

- **单例模式**：采用Meyers单例模式，确保全局唯一实例，C++11保证线程安全
- **静态便利方法**：V1.23.1新增类型安全的后端访问方法
- **跨后端支持**：管理不同存储格式的后端（CPU行主序、CUDA列主序）
- **自动检测**：启动时自动检测CUDA设备并注册相应后端
- **线程安全**：使用互斥锁保证多线程环境下的安全访问
- **异常处理**：完善的错误处理和日志记录

## 公共接口

### 静态方法

#### `static BackendManager& instance()`

获取BackendManager的唯一实例。

**返回值**：
- `BackendManager&` - 后端管理器的引用

**异常**：
- 无（保证线程安全）

**示例**：
```cpp
tr::BackendManager& manager = tr::BackendManager::instance();
```

### 静态便利方法（V1.23.1新增）

#### `static std::shared_ptr<CpuBackend> get_cpu_backend()`

获取CPU后端的类型安全智能指针。

**返回值**：
- `std::shared_ptr<CpuBackend>` - CPU后端智能指针

**异常**：
- `TRException` - 当CPU后端未注册时抛出（几乎不可能发生）

**特点**：
- **类型安全**：直接返回CpuBackend类型，无需类型转换
- **线程安全**：内部使用互斥锁保护
- **便利性**：简化CPU后端访问，推荐在V1.23.1中使用

**示例**：
```cpp
// V1.23.1推荐用法
auto cpu_backend = BackendManager::get_cpu_backend();

// 创建CPU张量（行主序存储）
Tensor cpu_tensor = Tensor::randn(Shape(2, 3), 42);
cpu_backend->fill(cpu_tensor, 3.14f);
```

#### `static std::shared_ptr<CudaBackend> get_cuda_backend(int device_id = 0)`

获取指定CUDA设备的类型安全智能指针。

**参数**：
- `device_id` - CUDA设备ID，默认为0

**返回值**：
- `std::shared_ptr<CudaBackend>` - CUDA后端智能指针

**异常**：
- `TRException` - 当指定CUDA后端未注册时抛出

**特点**：
- **类型安全**：直接返回CudaBackend类型，无需类型转换
- **设备指定**：支持指定具体的CUDA设备ID
- **跨后端支持**：返回的后端支持与CPU后端的格式转换

**示例**：
```cpp
// V1.23.1推荐用法
auto cuda_backend = BackendManager::get_cuda_backend(0);

// 从CPU转换到CUDA（自动转换为列主序）
Tensor cpu_tensor = Tensor::randn(Shape(2, 3), 42);
Tensor cuda_tensor = cuda_backend->from_cpu(cpu_tensor);

// CUDA矩阵乘法（列主序计算）
Tensor result = Tensor::empty(Shape(1024, 512), DType::FP32, tr::CUDA[0]);
cuda_backend->mm(result, cuda_a, cuda_b);
```


### 实例方法

#### `std::shared_ptr<Backend> get_backend(const Device& device)`

根据设备类型获取对应的后端实例。

**参数：**
- `device` - 目标设备（如CPU、CUDA等）

**返回值：**
- `std::shared_ptr<Backend>` - 后端实例的智能指针

**异常：**
- `TRException` - 当找不到对应设备的后端时抛出

**示例：**
```cpp
// 获取CPU后端
auto cpu_backend = manager.get_backend(tr::CPU);

// 获取CUDA后端
auto cuda_backend = manager.get_backend(tr::CUDA[0]);
```

#### `void register_backend(const Device& device, std::shared_ptr<Backend> backend)`

注册后端实例（内部使用）。

**参数：**
- `device` - 设备标识
- `backend` - 后端实例

**示例：**
```cpp
// 注册自定义后端
auto custom_backend = std::make_shared<CustomBackend>();
manager.register_backend(tr::Device(tr::DeviceType::CUSTOM, 0), custom_backend);
```

#### `bool is_registered(const Device& device) const`

检查后端是否已注册。

**参数：**
- `device` - 设备标识

**返回值：**
- `bool` - 如果已注册返回true，否则返回false

**示例：**
```cpp
if (manager.is_registered(tr::CUDA[0])) {
    std::cout << "CUDA backend is available" << std::endl;
} else {
    std::cout << "CUDA backend is not available" << std::endl;
}
```

## 自动注册的后端

### CPU后端
- **设备标识**：`tr::CPU`
- **设备键值**：`"CPU"`
- **注册状态**：始终注册

### CUDA后端
- **设备标识**：`tr::CUDA[0]`, `tr::CUDA[1]`, ..., `tr::CUDA[7]`
- **设备键值**：`"CUDA:0"`, `"CUDA:1"`, ..., `"CUDA:7"`
- **注册状态**：
  - 仅在`TR_USE_CUDA`宏定义时注册
  - 只注册实际存在的CUDA设备
  - 最多注册8个设备

## 使用示例

### 跨后端矩阵乘法（V1.23.1推荐用法）

```cpp
#include "tech_renaissance.h"
using namespace tr;

void cross_backend_example() {
    try {
        // V1.23.1新API：直接获取类型安全的后端实例
        auto cpu_backend = BackendManager::get_cpu_backend();
        auto cuda_backend = BackendManager::get_cuda_backend(0);

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
        std::cout << "Cross-backend results are close: " << (is_close ? "YES" : "NO") << std::endl;

    } catch (const TRException& e) {
        std::cerr << "Backend error: " << e.what() << std::endl;
    }
}
```

### 多GPU设备管理

```cpp
void multi_gpu_example() {
    try {
        auto cpu_backend = BackendManager::get_cpu_backend();

        // 检查可用的CUDA设备
        std::vector<std::shared_ptr<CudaBackend>> cuda_backends;

        for (int device_id = 0; device_id < 8; ++device_id) {
            try {
                auto cuda_backend = BackendManager::get_cuda_backend(device_id);
                cuda_backends.push_back(cuda_backend);
                std::cout << "CUDA device " << device_id << " is available" << std::endl;
            } catch (const TRException& e) {
                std::cout << "CUDA device " << device_id << " not available: "
                          << e.what() << std::endl;
                break;  // 通常设备是连续的，可以提前退出
            }
        }

        // 在可用设备上并行计算
        for (size_t i = 0; i < cuda_backends.size(); ++i) {
            Tensor cpu_data = Tensor::randn(Shape(100, 100), 42 + i);
            Tensor cuda_data = cuda_backends[i]->from_cpu(cpu_data);

            // 执行GPU计算...

            std::cout << "Computation completed on GPU " << i << std::endl;
        }

    } catch (const TRException& e) {
        std::cerr << "Multi-GPU error: " << e.what() << std::endl;
    }
}
```

### 传统接口使用（向后兼容）

```cpp
#include "tech_renaissance.h"
using namespace tr;

void traditional_api_example() {
    try {
        // 传统方式：通过实例获取后端
        BackendManager& manager = BackendManager::instance();

        // 获取CPU后端
        auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(
            manager.get_backend(tr::CPU));

        // 获取CUDA后端
        auto cuda_backend = std::dynamic_pointer_cast<CudaBackend>(
            manager.get_backend(tr::CUDA[0]));

        // 检查CUDA后端是否可用
        if (manager.is_registered(tr::CUDA[0])) {
            std::cout << "CUDA backend is available" << std::endl;

            // 使用CUDA后端进行计算...
            Tensor cpu_tensor = Tensor::randn(Shape(2, 3), 42);
            Tensor cuda_tensor = cuda_backend->from_cpu(cpu_tensor);

        } else {
            std::cout << "CUDA backend is not available, using CPU only" << std::endl;
        }

    } catch (const TRException& e) {
        std::cerr << "Backend error: " << e.what() << std::endl;
    }
}
```

### 多线程使用

```cpp
#include "tech_renaissance/backend/backend_manager.h"
#include <thread>
#include <vector>

void worker_thread(int thread_id) {
    try {
        tr::BackendManager& manager = tr::BackendManager::instance();
        auto backend = manager.get_backend(tr::CPU);

        // 使用后端进行计算...
        std::cout << "Thread " << thread_id << " got backend: "
                  << backend->name() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Thread " << thread_id << " error: " << e.what() << std::endl;
    }
}

int main() {
    std::vector<std::thread> threads;

    // 创建多个工作线程
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(worker_thread, i);
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    return 0;
}
```

## 注意事项

1. **单例访问**：始终通过`BackendManager::instance()`获取实例，不要尝试创建新实例
2. **线程安全**：所有公共方法都是线程安全的，可以在多线程环境中使用
3. **CUDA可用性**：CUDA后端的可用性取决于编译时设置和运行时硬件
4. **异常处理**：调用`get_backend()`时应妥善处理可能的异常
5. **生命周期**：后端实例的生命周期由BackendManager自动管理，无需手动释放

## 错误处理

BackendManager会自动记录以下情况的日志：

- CUDA设备检测失败
- CUDA后端初始化失败
- 后端注册失败

常见的异常情况：

```cpp
try {
    auto backend = manager.get_backend(tr::CUDA[99]);  // 不存在的设备
} catch (const tr::TRException& e) {
    // "Backend not found for device: CUDA:99"
    std::cerr << e.what() << std::endl;
}
```

## 性能考虑

- 后端实例在初始化时创建，后续访问为O(1)复杂度
- 互斥锁仅在后端注册和获取时使用，对计算性能无影响
- 建议在程序启动时获取所需的后端引用，避免频繁调用`get_backend()`

## 最佳实践（V1.23.1）

### 1. 使用静态便利方法
推荐使用V1.23.1新增的静态便利方法：
```cpp
// 推荐：类型安全的后端访问
auto cpu_backend = BackendManager::get_cpu_backend();
auto cuda_backend = BackendManager::get_cuda_backend(0);

// 避免：传统方式需要类型转换
auto backend = std::dynamic_pointer_cast<CpuBackend>(
    BackendManager::instance().get_backend(tr::CPU));
```

### 2. 跨后端操作
充分利用后端的跨设备转换能力：
```cpp
// CPU → CUDA → CPU 的工作流
auto cpu_backend = BackendManager::get_cpu_backend();
auto cuda_backend = BackendManager::get_cuda_backend();

Tensor cpu_data = Tensor::randn(Shape(2, 3), 42);
Tensor cuda_data = cuda_backend->from_cpu(cpu_data);  // 自动格式转换
// CUDA计算...
Tensor result_cpu = cuda_backend->to_cpu(cuda_result);  // 自动格式转换
```

### 3. 异常处理
妥善处理后端获取异常：
```cpp
try {
    auto cuda_backend = BackendManager::get_cuda_backend(0);
    // 使用CUDA后端
} catch (const TRException& e) {
    std::cout << "CUDA not available, falling back to CPU: " << e.what() << std::endl;
    auto cpu_backend = BackendManager::get_cpu_backend();
    // 使用CPU后端
}
```

### 4. 性能优化
- 在程序启动时获取后端引用，避免频繁调用
- 对于长时间运行的应用，缓存后端引用
- 利用多GPU进行并行计算

## 关键设计原则总结

### 跨后端管理
- **统一接口**：通过Backend基类提供一致的API
- **格式透明**：用户无需关心行主序vs列主序的差异
- **自动转换**：`from_cpu()`和`to_cpu()`自动处理格式转换

### 类型安全
- **静态方法**：V1.23.1新增类型安全的便利方法
- **智能指针**：自动内存管理，防止泄漏
- **异常安全**：完善的错误处理机制

### 线程安全
- **Meyers单例**：C++11保证初始化线程安全
- **互斥锁保护**：所有公共方法都是线程安全的
- **无共享状态**：后端实例之间独立工作

## 版本信息

- **版本**：V1.23.1
- **更新日期**：2025-10-30
- **作者**：技术觉醒团队
- **主要特性**：静态便利方法、跨后端管理、类型安全访问
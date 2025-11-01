# Tensor API 文档

## 概述

`Tensor`类是技术觉醒框架的核心数据结构，表示多维数组，支持多种数据类型和计算设备。它遵循轻量级元数据容器的设计理念，计算委托给Backend类，内存管理委托给Storage类。

**版本**: V1.28.1
**更新日期**: 2025-11-01
**作者**: 技术觉醒团队

## 设计理念

### 核心设计原则

1. **关注点分离**：Tensor类专注于元数据管理（形状、数据类型、设备），将计算委托给Backend，内存管理委托给Storage。
2. **后端管理存储**：每个后端负责管理自己的张量存储格式，转换层处理格式变化。
3. **设备抽象**：支持CPU、CUDA等多种计算设备，透明的设备间数据传输。
4. **RAII管理**：智能指针自动内存管理，防止内存泄漏。
5. **类型安全**：强类型设计防止数据类型错误，编译时错误检测。

### 系统架构

```
┌─────────────────────────────────────┐
│           用户代码/算法              │
├─────────────────────────────────────┤
│            Tensor 类                │  ← 元数据和设备管理
├─────────────────────────────────────┤
│       转换层 (to_cpu/from_cpu)      │  ← 内存布局格式转换
├─────────────────────────────────────┤
│            Storage 类                │  ← 设备无关的内存抽象
├─────────────────────────────────────┤
│            Backend 类                │  ← 具体计算和存储实现
└─────────────────────────────────────┘
```

## 类结构

```cpp
class Tensor {
    // 轻量级元数据容器
    Shape shape_;                    // 形状信息
    DType dtype_;                    // 数据类型
    Device device_;                  // 设备信息
    std::shared_ptr<Storage> storage_; // 内存管理（委托）
    size_t offset_;                  // Storage中的偏移量（未来视图支持）
};
```

## 核心API

### 构造函数

#### `Tensor()`
创建空的标量张量，数据类型为FP32，设备为CPU。

#### `Tensor(const Shape& shape, DType dtype = DType::FP32, const Device& device = CPU)`
用指定的形状、数据类型和设备创建张量。

**设计特点**：
- **轻量级构造**：只存储元数据，不分配内存
- **内存委托**：内存分配委托给Backend或工厂方法
- **设备指定**：显式设备参数确保设备一致性

### 元数据访问

#### `const Shape& shape() const noexcept`
返回张量的形状。

#### `DType dtype() const noexcept`
返回张量的数据类型。

#### `Device device() const noexcept`
返回张量数据所在的设备。

### 维度访问（V1.23.1增强）

#### `int32_t dim_size(int32_t dim) const`
返回指定维度的大小，带边界检查。

#### 4D便利方法
- `int32_t batch() const noexcept` - 批次维度（N维度）
- `int32_t channel() const noexcept` - 通道维度（C维度）
- `int32_t height() const noexcept` - 高度维度（H维度）
- `int32_t width() const noexcept` - 宽度维度（W维度）

#### **新增别名方法（V1.23.1）**
- `int32_t number() const` - batch()的别名，提高代码可读性
- `int32_t height() const` - 矩阵行数（2D张量专用）
- `int32_t width() const` - 矩阵列数（2D张量专用）

**设计优势**：这些别名方法使矩阵操作代码更加直观易懂。

### 跨后端转换（V1.23.1核心特性）

技术觉醒框架采用"后端管理存储"的设计理念，设备间的数据转移通过后端接口实现：

#### `Tensor from_cpu(const Tensor& tensor)` （CUDA后端专用）
将CPU张量转换为CUDA张量，自动处理行主序到列主序的转换。

**关键特性**：
```cpp
// 对于2D矩阵，执行内存布局转换
if (tensor.shape().ndim() == 2) {
    int32_t M = tensor.shape().height();  // 行数
    int32_t N = tensor.shape().width();   // 列数

    // 行主序 → 列主序转换
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t j = 0; j < N; ++j) {
            cuda_data[j * M + i] = cpu_data[i * N + j];
        }
    }
}
```

#### `Tensor to_cpu(const Tensor& tensor)` （CUDA后端专用）
将CUDA张量转换为CPU张量，自动处理列主序到行主序的转换。

**关键特性**：
```cpp
// 对于2D矩阵，执行内存布局转换
if (tensor.shape().ndim() == 2) {
    int32_t M = tensor.shape().height();  // 行数
    int32_t N = tensor.shape().width();   // 列数

    // 列主序 → 行主序转换
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t j = 0; j < N; ++j) {
            cpu_data[i * N + j] = cuda_data[j * M + i];
        }
    }
}
```

### 数据访问

#### `void* data_ptr() noexcept`和`const void* data_ptr() const noexcept`
原始数据指针访问。

**设计特点**：
- **直接访问**：提供对底层内存的直接访问
- **设备感知**：访问格式与后端存储格式一致
- **安全考虑**：用户需了解对应的内存布局格式

### 标量访问

#### `template<typename T> T item() const`
获取标量张量的值。

**设计特点**：
```cpp
template<typename T>
T item() const {
    auto backend = get_backend();
    if constexpr (std::is_same_v<T, float>) {
        return backend->get_scalar_float(*this);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return backend->get_scalar_int32(*this);
    }
    // 编译时类型检查
}
```

### 形状操作

#### `bool is_scalar() const noexcept`
检查张量是否为标量。

#### `bool is_matmul_compatible(const Shape& other) const`
检查形状是否与另一个形状兼容矩阵乘法。

### 工厂方法

#### `static Tensor empty(const Shape& shape, DType dtype = DType::FP32, const Device& device = tr::CPU)`
创建未初始化的张量。

#### `static Tensor randn(const Shape& shape, unsigned int seed = 42, DType dtype = DType::FP32, const Device& device = tr::CPU)`
创建正态分布随机张量。

#### `static Tensor zeros(const Shape& shape, DType dtype = DType::FP32, const Device& device = tr::CPU)`
创建零填充张量。

#### `static Tensor ones(const Shape& shape, DType dtype = DType::FP32, const Device& device = tr::CPU)`
创建单位填充张量。

#### `static Tensor full(const Shape& shape, float value, DType dtype = DType::FP32, const Device& device = tr::CPU)`
创建常数值填充张量。

### 比较运算符

#### `bool operator==(const Tensor& other) const noexcept`
#### `bool operator!=(const Tensor& other) const noexcept`

### 工具方法

#### `std::string to_string() const`
返回人类可读的字符串表示。

#### `size_t memory_size() const noexcept`
返回张量占用的内存大小（字节）。

#### `size_t numel() const noexcept`
返回张量的元素总数。

## 内存布局和后端管理（V1.23.1核心特性）

### **多后端存储原则**

Tensor类本身是设备无关的，但通过BackendManager管理不同后端的存储格式：

- **CPU后端**：使用**行主序（Row-major）**存储
- **CUDA后端**：使用**列主序（Column-major）**存储

### **数据访问一致性**

- **用户视角**：所有张量都是行主序访问，无论在哪个后端
- **后端内部**：各自选择最优的内存布局进行计算
- **转换透明**：`to_cpu()`、`from_cpu()` 自动处理格式转换

### **跨后端一致性保证**

```cpp
// 用户代码示例：数据访问一致性
Tensor cpu_a = Tensor::randn(Shape(1024, 2048), 42);
Tensor cuda_a = cuda_backend->from_cpu(cpu_a);  // 自动转换为列主序

// 无论在哪个后端，用户访问都是行主序语义
float cpu_value = cpu_a.data_ptr<float>()[i * 2048 + j];  // 行主序访问
float cuda_value = cuda_a.data_ptr<float>()[j * 1024 + i];  // 列主序访问

// 但经过转换层后，数学结果在行主序下是一致的
```

## 使用示例

### 基础张量创建
```cpp
// 创建空张量
Tensor t1;

// 创建CPU张量（行主序存储）
Tensor cpu_tensor(Shape(2, 3), DType::FP32, tr::CPU);

// 创建CUDA张量（列主序存储）
Tensor cuda_tensor(Shape(2, 3), DType::FP32, tr::CUDA[0]);
```

### 跨后端操作
```cpp
// 获取后端实例
auto cuda_backend = BackendManager::get_cuda_backend();
auto cpu_backend = BackendManager::get_cpu_backend();

// 创建CPU张量
Tensor cpu_a = Tensor::randn(Shape(1024, 2048), 42);

// 转换到CUDA（自动转换为列主序）
Tensor cuda_a = cuda_backend->from_cpu(cpu_a);

// CUDA矩阵乘法（列主序计算）
Tensor cuda_result = Tensor::empty(Shape(1024, 512), DType::FP32, tr::CUDA[0]);
cuda_backend->mm(cuda_result, cuda_a, cuda_b);

// 转换回CPU（自动转换回行主序）
Tensor cpu_result = cuda_backend->to_cpu(cuda_result);

// 结果验证：CPU和CUDA结果在行主序下应该一致
bool is_close = cpu_backend->is_close(cpu_a_result, cpu_b_result, 1e-4f);
```

### 新API使用（V1.23.1）
```cpp
// 使用新的静态便利方法
auto cuda_backend = BackendManager::get_cuda_backend();
auto cpu_backend = BackendManager::get_cpu_backend();

// 使用新的矩阵维度别名方法
int32_t M = cpu_a.height();  // 1024
int32_t K = cpu_a.width();   // 2048
int32_t N = cpu_b.width();   // 512

// 形状兼容性检查
if (cpu_a.shape().is_matmul_compatible(cpu_b.shape())) {
    std::cout << "Matrices are compatible for multiplication" << std::endl;
}
```

### 标量操作
```cpp
// 创建标量张量
Tensor scalar = Tensor::full(Shape(), 3.14f);

// 获取标量值
float value = scalar.item<float>();

// 设置标量值
scalar = Tensor::full(Shape(), 2.718f);
```

### 工厂方法使用
```cpp
// 创建零张量
Tensor zeros = Tensor::zeros(Shape(3, 4), DType::FP32, tr::CUDA[0]);

// 创建单位张量
Tensor ones = Tensor::ones(Shape(3, 4), DType::FP32, tr::CPU);

// 创建常量张量
Tensor constant = Tensor::full(Shape(3, 4), 2.5f, DType::FP32, tr::CUDA[1]);

// 创建随机张量
Tensor random = Tensor::randn(Shape(3, 4), 12345, DType::FP32, tr::CPU);
```

## 内存布局转换详解

### 行主序到列主序转换
```cpp
// 原始行主序数据（CPU）
// A[M,K] = [[1, 2, 3],
//           [4, 5, 6]]
// 内存布局：[1, 2, 3, 4, 5, 6]

// 转换为列主序数据（CUDA）
// A^T[K,M] = [[1, 4],
//            [2, 5],
//            [3, 6]]
// 内存布局：[1, 4, 2, 5, 3, 6]

for (int32_t i = 0; i < M; ++i) {        // i = 0,1
    for (int32_t j = 0; j < K; ++j) {    // j = 0,1,2
        cuda_data[j * M + i] = cpu_data[i * K + j];
        // cuda_data[0*2+0] = cpu_data[0*3+0] = 1
        // cuda_data[0*2+1] = cpu_data[1*3+0] = 4
        // cuda_data[1*2+0] = cpu_data[0*3+1] = 2
        // cuda_data[1*2+1] = cpu_data[1*3+1] = 5
        // cuda_data[2*2+0] = cpu_data[0*3+2] = 3
        // cuda_data[2*2+1] = cpu_data[1*3+2] = 6
    }
}
```

## 设计决策和原理

### 后端管理存储原则
**决策**：每个后端负责管理自己的张量存储格式。
**原理**：
- CPU后端使用行主序，符合C/C++惯例
- CUDA后端使用列主序，匹配cuBLAS/cuDNN接口
- 转换层透明处理格式转换，用户无需关心底层实现

### 转换层设计
**决策**：在`from_cpu()`和`to_cpu()`中自动处理格式转换。
**原理**：
- 2D矩阵专门优化转置转换
- 非2D张量直接复制
- 临时缓冲区避免多次内存分配

### 数据访问一致性
**决策**：用户始终看到行主序的数据访问方式。
**原理**：
- 转换层确保跨后端数据一致性
- 后端内部使用最优存储格式
- API统一提供一致的访问体验

### RAII内存管理
**决策**：通过Storage使用shared_ptr进行内存管理。
**原理**：
- 自动内存管理防止泄漏
- 支持多个Tensor共享同一Storage
- 异常安全保证资源正确释放

## 性能特征

### 内存效率
- **轻量级对象**：Tensor对象仅存储元数据
- **零拷贝视图**：多个Tensor可共享同一Storage
- **缓存友好**：紧凑的元数据布局

### 转换开销
- **2D矩阵**：O(M×N)转置开销
- **非2D张量**：直接复制，O(N)开销
- **内存优化**：临时缓冲区减少重复分配

### 计算性能
- **CPU后端**：基于Eigen的SIMD优化
- **CUDA后端**：基于cuBLAS的高性能矩阵运算
- **自动优化**：各后端选择最优的计算方法

## 线程安全

### 读操作
- 所有const方法都是线程安全的
- 元数据访问不修改共享状态

### 写操作
- 跨后端转换需要外部同步
- 智础设施层确保内存安全
- BackendManager提供线程安全的后端访问

## 错误处理

### 异常安全
```cpp
try {
    Tensor tensor = Tensor::randn(Shape(1024, 2048), 42);
    Tensor cuda_tensor = cuda_backend->from_cpu(tensor);
    // 使用tensor...
} catch (const TRException& e) {
    std::cerr << "Tensor operation failed: " << e.what() << std::endl;
}
```

### 边界检查
- Shape维度访问边界检查
- 内存分配大小验证
- 设备有效性检查

## 扩展性设计

### 新后端支持
1. 继承Backend基类
2. 定义存储格式（行主序、列主序或其他）
3. 实现转换方法（`from_cpu`、`to_cpu`、`to`）
4. 注册到BackendManager

### 新存储格式
- 稀疏张量存储格式
- 压缩存储格式
- 特定硬件优化格式

## 未来扩展

### 高级视图支持
offset_成员预留用于未来切片和跨步视图的实现。

### 异步操作
支持异步设备传输，提高并发性能。

### 内存池集成
与全局内存池集成，提高分配性能。

## 总结

技术觉醒框架的Tensor类通过创新的"后端管理存储"设计，实现了：

1. **用户友好**：转换层透明处理格式转换，用户无需关心内存布局
2. **高性能**：各后端选择最优存储格式和计算方法
3. **类型安全**：强类型设计和完善的错误检查
4. **设备无关**：统一API支持多设备和跨设备数据传输
5. **可扩展性**：模块化设计支持新后端和新存储格式

**关键创新**：
- **后端管理存储原则**：每个后端选择最优的内存布局
- **透明转换层**：自动处理不同存储格式之间的转换
- **一致的访问接口**：用户始终看到行主序的数据访问方式

---

## 版本信息

- **版本**: V1.23.1
- **更新日期**: 2025-10-30
- **作者**: 技术觉醒团队
- **主要更新**: 完善了跨后端存储管理、内存布局转换等核心特性
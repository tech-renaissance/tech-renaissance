# Tensor类文档

## 概述

Tensor类是技术觉醒框架的核心数据结构，表示带有相关元数据的多维数组。它作为张量元数据和存储句柄的轻量级容器，同时将所有计算操作委托给后端实现。Tensor类支持完整的梯度管理，为深度学习模型的训练和推理提供了基础支撑。

## 版本信息

- **版本**: V1.45.0
- **日期**: 2025-11-17
- **作者**: 技术觉醒团队
- **所属系列**: data

## 重要警告：不要直接使用Tensor构造函数！

### 警告：Tensor类的构造函数不会分配内存！

执行Tensor构造函数只会创建对象的元数据（形状、类型、设备等），但不会为张量数据分配实际的内存空间。

**重要区别**：
- **Tensor构造函数**：创建Tensor对象但**不分配内存**（段错误！）
- **Backend::empty() / Backend::zeros()**：**分配内存**
- **Backend::null_tensor()**：真正的空张量，**不占用内存**

**正确的张量创建方式：**
```cpp
// 正确：使用Backend方法
auto backend = BackendManager::get_cpu_backend();
Tensor tensor1 = backend->zeros(shape, dtype);
Tensor tensor2 = backend->ones(shape, dtype);
Tensor tensor3 = backend->full(shape, value, dtype);
Tensor tensor4 = backend->empty(shape, dtype);
```

**错误的方式：**
```cpp
// 错误：不要直接使用构造函数！
Tensor tensor(shape, dtype, device);  // 没有分配内存！
```

## 设计特点

Tensor类遵循轻量级设计理念：

- **元数据容器**: Tensor存储形状、数据类型、设备信息和存储句柄
- **后端委托**: 所有计算操作都委托给后端实现
- **设备无关**: 支持跨不同设备的操作（CPU、CUDA）
- **内存高效**: 使用共享存储和引用计数进行内存管理
- **维度限制**: 出于性能考虑支持最多4维张量
- **梯度管理**: 内置完整的梯度管理系统（V1.45.0新增）

## 类定义

```cpp
class Tensor {
public:
    // Constructors and assignment operators
    Tensor();
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) = default;

    // Metadata access methods
    const Shape& shape() const noexcept;
    DType dtype() const noexcept;
    Device device() const noexcept;
    int32_t ndim() const noexcept;
    int64_t numel() const noexcept;
    int32_t dim_size(int32_t dim) const;

    // Dimension-specific accessors
    int32_t batch() const noexcept;
    int32_t channel() const noexcept;
    int32_t height() const noexcept;
    int32_t width() const noexcept;

    // Storage and memory information
    std::shared_ptr<Storage> storage() const noexcept;
    size_t dtype_size() const noexcept;
    size_t memory_size() const noexcept;
    bool is_empty() const noexcept;
    bool is_scalar() const noexcept;
    bool is_contiguous() const noexcept;

    // ===== V1.45.0新增：梯度管理方法 =====

    // 梯度访问和管理
    Tensor& grad();                           // 获取梯度（延迟分配）
    const Tensor& grad() const;               // 获取梯度（const版本）
    void set_grad(const Tensor& grad);        // 设置梯度
    void set_grad(Tensor&& grad);             // 设置梯度（移动版本）
    bool has_grad() const;                    // 检查是否有梯度
    void zero_grad();                         // 清零梯度

    // ===== 视图和步长方法 =====

    /**
     * @brief 获取张量的步长
     * @return Strides对象
     */
    const Strides& strides() const noexcept;

    /**
     * @brief 检查张量是否为视图
     * @return 如果是视图则返回true
     */
    bool is_view() const noexcept;

    // Data access
    void* data_ptr() noexcept;
    const void* data_ptr() const noexcept;

    // Static factory methods
    static Tensor zeros(const Shape& shape, DType dtype = DType::FP32, const Device& device = tr::CPU);
    static Tensor ones(const Shape& shape, DType dtype = DType::FP32, const Device& device = tr::CPU);
    static Tensor full(const Shape& shape, float value, DType dtype = DType::FP32, const Device& device = tr::CPU);
    static Tensor empty(const Shape& shape, DType dtype = DType::FP32, const Device& device = tr::CPU);

    // Random number generation
    static Tensor randn(const Shape& shape, unsigned int seed = 0, DType dtype = DType::FP32, const Device& device = tr::CPU);
    static Tensor uniform(const Shape& shape, float min_val = 0.0f, float max_val = 1.0f, unsigned int seed = 0, DType dtype = DType::FP32, const Device& device = tr::CPU);
    static Tensor randint(int low, int high, const Shape& shape, DType dtype, unsigned int seed = 0, const Device& device = tr::CPU);

    // View operations
    Tensor view() const;

    // Data movement
    void from_cpu_data(const void* data, size_t size);
    void to_cpu_data(void* data, size_t size) const;

    // Utility methods
    std::string to_string() const;
    void print(const std::string& name = "") const;
    void print(const std::string& name, int precision) const;
    void summary(const std::string& name = "") const;

    // Scalar access
    template<typename T>
    T item() const;

    // Comparison operators
    bool operator==(const Tensor& other) const noexcept;
    bool operator!=(const Tensor& other) const noexcept;
};
```

## Core Features

### 1. 元数据管理

Tensor类维护关于张量的全面元数据：

- **形状**: 多维维度（最多4D）
- **数据类型**: 支持FP32或INT8
- **设备**: CPU或CUDA设备规格
- **存储**: 带引用计数的共享内存句柄
- **偏移**: 存储内的内存偏移（当前始终为0）
- **步长**: 每个维度的步长用于计算内存偏移
- **视图标志**: 指示张量是否为另一个张量的视图

### 2. V1.45.0新增：梯度管理系统

Tensor类现在支持完整的梯度管理，为深度学习模型训练提供支撑：

#### 梯度访问方法
```cpp
// 获取梯度（延迟分配）
Tensor& grad();

// 获取梯度（const版本）
const Tensor& grad() const;

// 检查是否有梯度
bool has_grad() const;

// 清零梯度
void zero_grad();
```

#### 梯度设置方法
```cpp
// 设置梯度（复制版本）
void set_grad(const Tensor& grad);

// 设置梯度（移动版本）
void set_grad(Tensor&& grad);
```

#### 梯度管理特点

1. **延迟分配**: 梯度张量只有在首次访问时才分配内存
2. **自动管理**: 梯度张量的生命周期由智能指针自动管理
3. **形状一致**: 梯度张量与原张量具有相同的形状和数据类型
4. **设备一致性**: 梯度张量与原张量位于同一设备

#### 梯度使用示例
```cpp
#include "tech_renaissance.h"

using namespace tr;

// 创建一个参数张量
auto backend = BackendManager::get_cpu_backend();
Tensor weight = backend->randn(Shape(256, 128));

std::cout << "Has grad: " << weight.has_grad() << std::endl;  // false

// 首次访问梯度时自动分配
Tensor& weight_grad = weight.grad();
std::cout << "Has grad: " << weight.has_grad() << std::endl;  // true
std::cout << "Grad shape: " << weight_grad.shape().to_string() << std::endl;  // (256,128)

// 清零梯度
weight.zero_grad();

// 手动设置梯度
Tensor new_grad = backend->ones(weight.shape());
weight.set_grad(new_grad);
```

#### 内存优化效果

梯度管理系统显著优化了内存使用：

```cpp
// 传统方法：默认为所有参数分配梯度
Tensor param1 = backend->randn(Shape(1000, 1000));  // 4MB
Tensor param1_grad = backend->zeros(Shape(1000, 1000));  // 额外4MB
// 总内存：8MB

// 新方法：延迟分配梯度
Tensor param2 = backend->randn(Shape(1000, 1000));  // 4MB
// 梯度尚未分配，总内存：4MB

Tensor& param2_grad = param2.grad();  // 现在分配梯度
// 总内存：8MB（但分配是按需的）
```

### 3. 静态工厂方法

张量创建通过静态工厂方法完成：

#### Basic Creation Methods
```cpp
// Create tensors filled with specific values
Tensor zeros = Tensor::zeros(Shape(2, 3));           // All zeros
Tensor ones = Tensor::ones(Shape(2, 3));            // All ones
Tensor full = Tensor::full(Shape(2, 3), 5.0f);       // All 5.0
Tensor empty = Tensor::empty(Shape(2, 3));          // Uninitialized

// All methods support dtype and device parameters
Tensor zeros_int8 = Tensor::zeros(Shape(2, 3), DType::INT8);
Tensor ones_int32 = Tensor::ones(Shape(2, 3), DType::INT32);
Tensor empty_cuda = Tensor::empty(Shape(2, 3), DType::FP32, tr::CUDA[0]);
```

#### Random Number Generation
```cpp
// Different random distributions
Tensor normal = Tensor::randn(Shape(2, 3));         // Normal distribution N(0,1)
Tensor uniform = Tensor::uniform(Shape(2, 3));      // Uniform distribution U(0,1)
Tensor randint = Tensor::randint(0, 10, Shape(2, 3), DType::FP32); // Integer uniform U[0,10]
Tensor randint_int8 = Tensor::randint(0, 100, Shape(2, 3), DType::INT8); // INT8 integers
Tensor randint_int32 = Tensor::randint(0, 1000, Shape(2, 3), DType::INT32); // INT32 integers
```

### 3. View and Strides Operations

The Tensor class now supports view operations for zero-copy tensor transformations:

#### Creating Views
Views are created through backend methods, providing zero-copy tensor reshaping:

```cpp
auto cpu_backend = BackendManager::get_cpu_backend();

// Create original tensor
Tensor original = cpu_backend->zeros(Shape(2, 3, 4), DType::FP32);

// Create a view with different shape (same element count)
Tensor view_2d = cpu_backend->view(original, Shape(6, 4));
Tensor view_4d = cpu_backend->view(view_2d, Shape(2, 2, 3, 2));
```

#### View Properties
```cpp
// Check if tensor is a view
bool is_view_tensor = tensor.is_view();

// Get tensor strides
const Strides& strides = tensor.strides();

// Check if memory layout is contiguous
bool is_contiguous = tensor.is_contiguous();

// Verify memory sharing
bool shares_memory = (original.storage() == view_tensor.storage());
```

#### Strides Information
The strides object describes how to compute memory offsets for multi-dimensional access:

```cpp
// Access individual stride values
int64_t stride_n = strides.n();  // Batch dimension stride
int64_t stride_c = strides.c();  // Channel dimension stride
int64_t stride_h = strides.h();  // Height dimension stride
int64_t stride_w = strides.w();  // Width dimension stride

// Calculate linear offset
int64_t offset = strides.get_offset(n, c, h, w);
```

**Key Characteristics of Views:**
- **Zero-Copy**: Views share the same underlying storage
- **Memory Efficient**: No data duplication, only metadata changes
- **Automatic Lifetime**: Managed by shared_ptr reference counting
- **Writable**: Modifying a view affects the original tensor
- **Contiguous Only**: Current version supports only contiguous tensors

### 4. Data Access and Manipulation

#### Direct Data Access
```cpp
// Get raw data pointers
void* ptr = tensor.data_ptr();
const void* const_ptr = tensor.data_ptr();

// CPU data transfer
tensor.from_cpu_data(cpu_data, size);  // Copy from CPU
tensor.to_cpu_data(cpu_data, size);    // Copy to CPU
```

#### Scalar Access
```cpp
// Extract scalar values (template method)
float value = tensor.item<float>();
int32_t int_value = tensor.item<int32_t>();
int8_t byte_value = tensor.item<int8_t>();
```

### 4. Device Support

The Tensor class supports multiple devices:

```cpp
// Create tensors on different devices
Tensor cpu_tensor = Tensor::ones(Shape(2, 3), DType::FP32, tr::CPU);
Tensor cuda_tensor = Tensor::ones(Shape(2, 3), DType::FP32, tr::CUDA(0));

// Check device
Device device = tensor.device();
bool is_cpu = device.is_cpu();
int32_t device_index = device.index();
```

### 5. View Operations

```cpp
// Create shallow copies sharing storage
Tensor view_tensor = original_tensor.view();
// Both tensors share the same underlying storage
```

## Data Types

The Tensor class supports the following data types:

- **FP32**: 32-bit floating point numbers (default)
- **INT8**: 8-bit signed integers

Type conversions and operations are handled by backend implementations.

## Dimension Support

Tensors support up to 4 dimensions with the following conventions:

- **0D**: Scalar tensors (single value)
- **1D**: Vectors
- **2D**: Matrices
- **3D**: 3D tensors (often used for feature maps)
- **4D**: 4D tensors (NCHW format for batches)

### Dimension Accessors

```cpp
Tensor tensor = Tensor::ones(Shape(2, 3, 4, 5));

// General dimension access
int32_t ndims = tensor.ndim();           // 4
int64_t elements = tensor.numel();       // 120
int32_t dim0 = tensor.dim_size(0);       // 2
int32_t dim1 = tensor.dim_size(1);       // 3

// NCHW-specific accessors
int32_t batch = tensor.batch();          // 2 (N dimension)
int32_t channel = tensor.channel();      // 3 (C dimension)
int32_t height = tensor.height();        // 4 (H dimension)
int32_t width = tensor.width();          // 5 (W dimension)
```

### 5. Gradient Management

#### Overview (V1.45.0)

Starting from V1.45.0, Tensor class includes comprehensive gradient management functionality for neural network training. This feature enables automatic gradient tracking and storage during backward propagation.

#### Gradient Interface

Tensor provides the following gradient management methods:

```cpp
// Get gradient tensor (creates one if doesn't exist)
Tensor& grad();

// Get gradient tensor (const version)
const Tensor& grad() const;

// Set gradient tensor
void set_grad(const Tensor& grad);
void set_grad(Tensor&& grad);

// Check if gradient exists
bool has_grad() const;

// Clear gradient (set to zero)
void zero_grad();
```

#### Lazy Allocation

Gradients are allocated lazily to avoid unnecessary memory usage:

```cpp
Tensor param = backend->zeros(Shape(100, 100));  // No gradient allocated yet
bool has_grad = param.has_grad();               // Returns false

Tensor& grad = param.grad();                    // Gradient allocated now
has_grad = param.has_grad();                    // Returns true
```

#### Memory Efficiency

- **Default behavior**: Tensors don't have gradients by default
- **Lazy allocation**: Gradients only allocated when accessed via `grad()`
- **Automatic cleanup**: Gradients follow RAII pattern with smart pointers
- **Shape matching**: Gradient tensors automatically match parent tensor shape and dtype

#### Usage in Neural Networks

Gradient management is designed for Module classes during training:

```cpp
// During forward pass (no gradients yet)
Tensor output = module.forward(input);

// During backward pass (gradients automatically managed)
module.backward(grad_output);

// Access gradients for parameter updates
for (auto& [name, param] : module.parameters()) {
    if (param.has_grad()) {
        optimizer.update(param, param.grad());
    }
}

// Clear gradients for next iteration
module.zero_grad();
```

### 6. Memory Management

### Storage Model

- **Shared Storage**: Multiple tensors can share the same underlying storage
- **Reference Counting**: Automatic memory management through shared pointers
- **Offset Support**: Currently implemented but always set to 0
- **Contiguity**: All tensors are contiguous in the current implementation

### Memory Efficiency

```cpp
// Memory size calculation
size_t bytes = tensor.memory_size();
size_t element_size = tensor.dtype_size();
int64_t num_elements = tensor.numel();

// Empty tensor checking
if (tensor.is_empty()) {
    // Handle empty tensor case
}
```

## Utility and Display Methods

### String Representation

```cpp
// Basic string representation
std::string str = tensor.to_string();

// Formatted printing
tensor.print("My Tensor");                    // Default precision
tensor.print("My Tensor", 6);                 // Custom precision
tensor.summary("Tensor Info");                // Summary information
```

### Output Format

The tensor printing follows PyTorch-style formatting:

```
My Tensor:
tensor([
  [1.0000, 2.0000, 3.0000],
  [4.0000, 5.0000, 6.0000]
])
```

## Comparison and Equality

```cpp
Tensor a = Tensor::ones(Shape(2, 3));
Tensor b = Tensor::ones(Shape(2, 3));

// Equality comparison (metadata and storage)
if (a == b) {
    // Tensors are identical
}

// Inequality comparison
if (a != b) {
    // Tensors differ
}
```

## Error Handling

The Tensor class provides comprehensive error handling:

- **Shape Validation**: Validates tensor shapes and dimensions
- **Type Checking**: Ensures compatible data types
- **Memory Safety**: Checks for proper memory allocation
- **Device Validation**: Validates device specifications
- **Bounds Checking**: Prevents out-of-bounds access

Common error scenarios:

```cpp
try {
    int32_t dim = tensor.dim_size(10);  // Out of bounds
} catch (const std::out_of_range& e) {
    // Handle dimension error
}

try {
    float value = tensor.item<float>();  // On non-scalar tensor
} catch (const TRException& e) {
    // Handle scalar access error
}
```

## Performance Considerations

### Memory Efficiency

- **Zero-Copy Views**: The `view()` method creates zero-copy shallow copies
- **Shared Storage**: Multiple tensors can share the same memory
- **Lazy Allocation**: Memory is allocated only when needed

### Computational Efficiency

- **Backend Delegation**: Operations are performed by optimized backend implementations
- **Device Optimized**: Each backend (CPU/CUDA) provides optimized implementations
- **Type Specialization**: Different optimizations for FP32 and INT8 data types

## Usage Examples

### Basic Tensor Operations

```cpp
#include "tech_renaissance.h"

// Create tensors
Tensor a = Tensor::ones(Shape(2, 3));
Tensor b = Tensor::full(Shape(2, 3), 2.0f);

// Access metadata
std::cout << "Shape: " << a.shape().to_string() << std::endl;
std::cout << "Elements: " << a.numel() << std::endl;
std::cout << "Data type: " << dtype_to_string(a.dtype()) << std::endl;

// Print tensor
a.print("Tensor A");

// Scalar operations
Tensor scalar = Tensor::full(Shape(), 3.14f);
float value = scalar.item<float>();
std::cout << "Scalar value: " << value << std::endl;
```

### Random Tensor Generation

```cpp
// Different random distributions with fixed seeds
Tensor normal_dist = Tensor::randn(Shape(3, 3), 42);        // Reproducible normal
Tensor uniform_dist = Tensor::uniform(Shape(3, 3), 0.0f, 1.0f, 123);
Tensor int_dist = Tensor::randint(0, 100, Shape(3, 3), DType::FP32, 456);

normal_dist.print("Normal Distribution");
uniform_dist.print("Uniform Distribution");
int_dist.print("Integer Distribution");
```

### Device Operations

```cpp
// Create tensors on different devices
Tensor cpu_tensor = Tensor::ones(Shape(100, 100));
Tensor cuda_tensor = Tensor::ones(Shape(100, 100), DType::FP32, tr::CUDA(0));

// Data transfer between devices
std::vector<float> cpu_data(cpu_tensor.numel());
cpu_tensor.to_cpu_data(cpu_data.data(), cpu_data.size() * sizeof(float));

// Create GPU tensor from CPU data
Tensor gpu_tensor = Tensor::empty(Shape(100, 100), DType::FP32, tr::CUDA(0));
gpu_tensor.from_cpu_data(cpu_data.data(), cpu_data.size() * sizeof(float));
```

## Integration with Backend System

The Tensor class integrates seamlessly with the backend system:

```cpp
// Tensor operations are delegated to backends
auto cpu_backend = BackendManager::get_cpu_backend();

// Backend provides tensor operations
Tensor result = cpu_backend->add(a, b);
Tensor expanded = cpu_backend->expand(a, Shape(2, 3, 1));
Tensor unsqueezed = cpu_backend->unsqueeze(a, 1);
```

## Removed Methods

In the current implementation (V1.29.2), the following methods have been removed from the Tensor class and are now provided by backend implementations:

- **reshape()**: Shape changing operations are now provided by CPU/CUDA backends
- **squeeze_dim()**: Dimension removal operations are now provided by CPU/CUDA backends
- **unsqueeze_dim()**: Dimension insertion operations are now provided by CPU/CUDA backends

These operations are now available through the backend API:

```cpp
auto cpu_backend = BackendManager::get_cpu_backend();

// Instead of: tensor.squeeze_dim(0)
Tensor squeezed = cpu_backend->squeeze(tensor, 0);

// Instead of: tensor.unsqueeze_dim(1)
Tensor unsqueezed = cpu_backend->unsqueeze(tensor, 1);
```

## 张量销毁最佳实践

### # 推荐的张量销毁方法

在Tech Renaissance框架中，对于大型张量的销毁，我们强烈建议结合以下两种方法：

#### 方法1：RAII作用域管理（推荐用于局部张量）

```cpp
{
    // 在大括号内创建大型张量
    Tensor temp_tensor = cpu_backend->zeros(Shape(1000, 1000, 1000), DType::FP32);

    // 使用temp_tensor进行计算
    // ...

}  // temp_tensor在这里自动析构，内存立即释放
```

**优点：**
- 自动内存管理，符合RAII原则
- 作用域清晰，内存释放时机明确
- 代码简洁，无需手动管理

#### 方法2：显式后端null_tensor()方法（推荐用于需要灵活控制的场景）

```cpp
// 创建大型张量
Tensor large_tensor = cpu_backend->zeros(Shape(1000, 1000, 1000), DType::FP32);

// 使用large_tensor进行计算
// ...

// 显式销毁，立即释放内存
large_tensor = cpu_backend->null_tensor();  // 明确告知：这是一个null张量
```

**优点：**
- 显式操作，代码意图清晰
- 灵活控制释放时机
- 符合"后端管理存储"的设计原则

#### View张量的特殊考虑

当处理view张量时，内存管理具有一些特殊考虑：

```cpp
// 原始张量和view的内存管理
Tensor original = cpu_backend->zeros(Shape(1000, 1000));
Tensor view_tensor = cpu_backend->view(original, Shape(500, 2000));

// 内存会一直保持，直到所有引用都被释放
{
    Tensor additional_view = cpu_backend->view(original, Shape(2000, 500));
    // 此时storage引用计数为3，内存不会被释放
} // additional_view离开作用域，引用计数降为2

// 手动释放原始张量会影响所有view
original = cpu_backend->null_tensor();  // 所有view都失效
// 此时view_tensor成为悬空引用，不应再使用
```

**View张量管理要点：**
- **共享生命周期**: view与原始张量共享底层存储的生命周期
- **引用计数跟踪**: 所有引用的张量都必须被释放，内存才会真正释放
- **显式清理**: 如果需要提前释放内存，显式设置原始张量为null_tensor
- **避免悬空引用**: 原始张量释放后，所有view都会失效

### 为什么推荐这两种方法？

1. **避免构造函数误用**：防止用户直接调用`Tensor()`构造函数
   ```cpp
   // ❌ 危险：违反后端解耦原则
   tensor = Tensor();  // 用户不清楚这个操作的含义

   // ✅ 安全：显式使用后端方法
   tensor = cpu_backend->null_tensor();  // 明确：通过CPU后端设为null
   ```

2. **API明确性**：`null_tensor()`比`empty()`更无歧义
   - `empty()`：**分配内存但未初始化数据**（用户可能误以为没分配内存）
   - `null_tensor()`：真正的空张量，**不占用内存**
   - `Tensor构造函数`：只创建元数据，**不分配内存**（段错误！）

3. **符合框架设计**：所有操作都通过后端，保持一致性

### 实际案例参考

参见 `tests/unit_tests/test_memory_occupation.cpp` 中的完整测试案例，该测试验证了：
- RAII作用域管理的有效性
- `null_tensor()`方法的正确性
- 不同销毁方式的内存释放效果

### 组合使用建议

```cpp
void process_large_data() {
    auto cpu_backend = BackendManager::get_cpu_backend();

    // 方法1：对于临时使用的张量，使用RAII
    {
        Tensor temp_data = cpu_backend->zeros(Shape(1000, 1000));
        // 处理temp_data
    }  // 自动释放

    // 方法2：对于需要长期存在但可能提前释放的张量
    Tensor persistent_tensor = cpu_backend->zeros(Shape(2000, 2000));

    // 在某些条件下提前释放
    if (some_condition) {
        persistent_tensor = cpu_backend->null_tensor();
    }

    // 继续使用persistent_tensor（如果没被释放）
}
```

**核心原则**：无论使用哪种方法，都要避免直接调用Tensor类的构造函数进行销毁操作。

## Limitations and Constraints

### Current Limitations

1. **Dimension Limit**: Maximum 4 dimensions
2. **Data Types**: Only FP32 and INT8 supported
3. **No In-place Operations**: All operations create new tensors
4. **No Advanced Indexing**: No slicing, advanced indexing, or masking
5. **No Automatic Broadcasting**: Shape compatibility must be manually ensured

### Design Constraints

1. **Lightweight Design**: Tensor is primarily a metadata container
2. **Backend Delegation**: All computation delegated to backend implementations
3. **Memory Model**: Shared storage with reference counting
4. **Device Model**: Explicit device management required

## Future Enhancements

Planned improvements for the Tensor class:

1. **Extended Dimension Support**: Support for more than 4 dimensions
2. **Additional Data Types**: Support for more numeric types (FP16, FP64, INT32, etc.)
3. **Advanced Indexing**: Slicing, masking, and fancy indexing
4. **In-place Operations**: Native support for in-place modifications
5. **Automatic Broadcasting**: Shape broadcasting compatibility
6. **Memory Layout Options**: Support for different memory layouts (row-major, column-major)

## Related Documentation

- [Shape Class](shape.md) - Tensor shape management
- [Device Class](device.md) - Device management
- [Storage Class](storage.md) - Memory management
- [Strides Class](strides.md) - Tensor strides and memory layout
- [Backend System](backend.md) - Computational operations
- [CPU Backend](cpu_backend.md) - CPU-specific operations
- [CUDA Backend](cuda_backend.md) - GPU-specific operations
- [CPU Dimension Operations](cpu_dimension.md) - Unsqueeze and squeeze operations
- [View Operations](about_view.md) - View creation and usage guide

## Files

- **Header**: `include/tech_renaissance/data/tensor.h`
- **Implementation**: `src/data/tensor.cpp`
- **Tests**: `tests/unit_tests/test_tensor.cpp`
- **Related**: Shape, Device, Storage, and Backend classes
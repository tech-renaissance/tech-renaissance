# Tensor Class Documentation

## 重要警告：不要直接使用Tensor构造函数！

### # 不要直接使用Tensor构造函数创建张量！

**警告：Tensor类的构造函数不会分配内存！**

执行Tensor构造函数只会创建对象的元数据（形状、类型、设备等），但不会为张量数据分配实际的内存空间。

**后果：**
- 使用构造函数创建的张量无法进行任何数据操作
- 会导致段错误或内存访问错误
- 违反了框架的设计原则

**正确的张量创建方式：**
```cpp
// 正确：使用Backend子类的方法
auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(
    BackendManager::instance().get_backend(CPU));
Tensor tensor1 = cpu_backend->zeros(shape, dtype);
Tensor tensor2 = cpu_backend->ones(shape, dtype);
Tensor tensor3 = cpu_backend->full(shape, value, dtype);
Tensor tensor4 = cpu_backend->empty(shape, dtype);

// 或者使用其他Backend子类的方法
auto cuda_backend = std::dynamic_pointer_cast<CudaBackend>(
    BackendManager::instance().get_backend(CUDA));
Tensor tensor5 = cuda_backend->zeros(shape, dtype);
```

**错误的方式：**
```cpp
// 错误：不要直接使用构造函数！
Tensor tensor(shape, dtype, device);  // 没有分配内存！
```

## Overview

The Tensor class is the core data structure in Tech Renaissance framework, representing multi-dimensional arrays with associated metadata. It serves as a lightweight container for tensor metadata and a handle to storage, while delegating all computational operations to backend implementations.

## Version Information

- **Version**: V1.31.2
- **Date**: 2025-11-03
- **Author**: 技术觉醒团队

## Design Philosophy

The Tensor class follows a lightweight design philosophy:

- **Metadata Container**: Tensor stores shape, data type, device information, and storage handle
- **Backend Delegation**: All computational operations are delegated to backend implementations
- **Device Agnostic**: Supports operations across different devices (CPU, CUDA)
- **Memory Efficient**: Uses shared storage and reference counting for memory management
- **Dimension Limited**: Supports tensors up to 4 dimensions for performance considerations

## Class Structure

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

### 1. Metadata Management

The Tensor class maintains comprehensive metadata about the tensor:

- **Shape**: Multi-dimensional dimensions (up to 4D)
- **Data Type**: FP32 or INT8 support
- **Device**: CPU or CUDA device specification
- **Storage**: Shared memory handle with reference counting
- **Offset**: Memory offset within storage (currently always 0)

### 2. Static Factory Methods

Tensor creation is done through static factory methods:

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

### 3. Data Access and Manipulation

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

## Memory Management

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
- [Backend System](backend.md) - Computational operations
- [CPU Backend](cpu_backend.md) - CPU-specific operations
- [CUDA Backend](cuda_backend.md) - GPU-specific operations
- [CPU Dimension Operations](cpu_dimension.md) - Unsqueeze and squeeze operations

## Files

- **Header**: `include/tech_renaissance/data/tensor.h`
- **Implementation**: `src/data/tensor.cpp`
- **Tests**: `tests/unit_tests/test_tensor.cpp`
- **Related**: Shape, Device, Storage, and Backend classes
# Tensor-Backend System Architecture Documentation

## # 重要警告：不要直接使用Tensor构造函数！

**警告：Tensor类的构造函数不会分配内存！**

在Tech Renaissance框架中，Tensor构造函数只创建元数据，不分配实际内存。所有张量必须通过Backend类的方法来创建，因为Backend会在创建后立即分配内存。

**正确的张量创建流程：**
1. 获取Backend实例：`BackendManager::instance().get_backend(CPU)`
2. 使用Backend方法创建：`backend->zeros(shape, dtype)`
3. Backend自动分配内存并返回可用张量

**错误的操作（会导致段错误）：**
- 直接调用`Tensor(shape, dtype, device)`构造函数
- 使用Tensor类的静态工厂方法（不推荐）
- 试图访问未分配内存的张量

## Overview

The Tensor-Backend system in Tech Renaissance framework adopts a layered decoupled design, implementing efficient and safe tensor data management through five core classes. The system follows the "backend manages storage" principle, providing a unified data abstraction layer for deep learning computations.

## Version Information

- **Version**: V1.31.2
- **Date**: 2025-11-03
- **Author**: 技术觉醒团队

## Design Philosophy

### Core Design Principles

1. **Separation of Concerns**: Tensor manages metadata, Storage manages memory, Backend handles computation and storage formats
2. **Backend-Managed Storage**: Each backend manages its own tensor storage format, with conversion layers handling format changes
3. **Type Safety**: Strong typing prevents data type errors with compile-time error detection
4. **Device Agnostic**: Supports CPU, CUDA and other devices with transparent device-to-device data transfer
5. **RAII Management**: Smart pointer automatic memory management prevents memory leaks

### System Architecture Diagram

```
┌─────────────────────────────────────┐
│           User Code/Algorithms        │
├─────────────────────────────────────┤
│            Tensor Class                │  ← Metadata and device management
├─────────────────────────────────────┤
│       Conversion Layer (Backend Ops)   │  ← Computation and shape manipulation
├─────────────────────────────────────┤
│            Storage Class                │  ← Device-agnostic memory abstraction
├─────────────────────────────────────┤
│            Backend Classes              │  ← Specific computation implementations
└─────────────────────────────────────┘
```

## Key Design: Backend-Managed Storage

### Multi-Backend Storage Principle

The core design philosophy of Tech Renaissance framework is **"Backend-Managed Storage"**:

1. **CPU Backend**: Uses **row-major (Row-major)** storage for tensor data
2. **CUDA Backend**: Uses **column-major (Column-major)** storage for tensor data
3. **Transparent Conversion**: Users don't need to care about underlying storage format; conversion layers handle it automatically

### Operation Delegation

The framework delegates computational operations to backend implementations:

- **Arithmetic Operations**: `add`, `subtract`, `multiply`, etc.
- **Shape Operations**: `expand`, `unsqueeze`, `squeeze`, etc.
- **Memory Operations**: `copy`, `fill`, etc.
- **Device Transfers**: `to_cpu`, `from_cpu`, etc.

## Core Components Details

### 1. Tensor Class - Metadata and Device Management

**Design Position**: Tensor class is the core user interface, responsible for metadata management and device coordination.

**Core Data Structure**:

```cpp
class Tensor {
    Shape shape_;                          // Shape information
    DType dtype_;                          // Data type
    Device device_;                        // Device information
    std::shared_ptr<Storage> storage_;     // Memory handle (delegated management)
    size_t offset_;                        // Offset (reserved for future view support)
};
```

**Key Features**:

#### a) Multi-Type Support
- **FP32**: 32-bit floating point for training and inference
- **INT8**: 8-bit signed integers for quantized inference
- **INT32**: 32-bit signed integers for labels and index operations
- All tensor operations support the three data types

#### b) Cross-Backend Conversion Interface

```cpp
// CPU to CUDA conversion (row-major → column-major)
Tensor CudaBackend::from_cpu(const Tensor& tensor);

// CUDA to CPU conversion (column-major → row-major)
Tensor CudaBackend::to_cpu(const Tensor& tensor);
```

**Design Philosophy**: Device-to-device data transfer is implemented entirely through backend interfaces. The Tensor class itself contains no device transfer logic, maintaining lightweight design.

#### b) Type-Safe Scalar Access

```cpp
template<typename T>
T item() const {
    auto backend = get_backend();
    if constexpr (std::is_same_v<T, float>) {
        return backend->get_scalar_float(*this);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return backend->get_scalar_int32(*this);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return backend->get_scalar_int8(*this);
    }
    // Compile-time type checking
}
```

#### c) Metadata Access Interface

```cpp
// Shape information
const Shape& shape() const noexcept;
int32_t ndim() const noexcept;
int64_t numel() const noexcept;
int32_t dim_size(int32_t dim) const;

// Matrix dimension aliases
int32_t batch() const noexcept;    // N dimension
int32_t channel() const noexcept;  // C dimension
int32_t height() const noexcept;    // H dimension
int32_t width() const noexcept;     // W dimension

// Raw data access
void* data_ptr() noexcept;
const void* data_ptr() const noexcept;
```

#### d) Removed Methods (V1.29.2)

The following methods have been removed from the Tensor class and are now provided by backend implementations:

- `reshape()`: Shape changing operations
- `squeeze_dim()`: Dimension removal operations
- `unsqueeze_dim()`: Dimension insertion operations

These operations are now accessed through backend APIs:

```cpp
auto cpu_backend = BackendManager::get_cpu_backend();

// Instead of: tensor.squeeze_dim(0)
Tensor squeezed = cpu_backend->squeeze(tensor, 0);

// Instead of: tensor.unsqueeze_dim(1)
Tensor unsqueezed = cpu_backend->unsqueeze(tensor, 1);

// Instead of: tensor.reshape(Shape(2, 3, 4))
Tensor reshaped = cpu_backend->reshape(tensor, Shape(2, 3, 4));
```

### 2. Storage Class - Device-Agnostic Memory Abstraction

**Design Position**: Encapsulates raw memory, provides RAII management, and serves as a bridge between Tensor and Backend.

**Core Data Structure**:

```cpp
class Storage {
    std::shared_ptr<void> data_ptr_;  // Smart-pointer managed memory block
    size_t size_;                     // Actual used size
    size_t capacity_;                 // Allocated capacity
    Device device_;                   // Memory location device
    DType dtype_;                     // Data type
};
```

**Key Features**:

#### a) Device-Agnostic Memory Management

```cpp
// Storage itself doesn't care about memory layout format
Storage(size_t size, const Device& device, DType dtype)
    : size_(size), capacity_(size), device_(device), dtype_(dtype) {
    // Delegate to Backend for device-specific memory allocation
    auto backend = BackendManager::get_backend(device);
    // Memory format is determined by Backend
}
```

#### b) Backend Interface Support

```cpp
// Provide raw memory access for Backend use
void* data_ptr() noexcept { return data_ptr_.get(); }
const void* data_ptr() const noexcept { return data_ptr_.get(); }
```

### 3. Backend Abstract Base Class - Computation and Storage Implementation

**Design Position**: Defines unified computation interfaces, with specific implementations handled by each backend.

**Core Interface**:

```cpp
class Backend {
public:
    // Memory management interfaces
    virtual std::shared_ptr<void> allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void copy_data(void* dst, const void* src, size_t size,
                       const Device& dst_device, const Device& src_device) = 0;

    // Cross-backend conversion interfaces
    virtual Tensor from_cpu(const Tensor& tensor) = 0;
    virtual Tensor to_cpu(const Tensor& tensor) = 0;
    virtual Tensor to(const Tensor& tensor, const Device& device) = 0;

    // Computation operation interfaces
    virtual void mm(Tensor& result, const Tensor& a, const Tensor& b) = 0;
    virtual void fill(Tensor& dst, float value) = 0;
    virtual void fill(Tensor& dst, int8_t value) = 0;
    virtual void add(Tensor& result, const Tensor& a, const Tensor& b) = 0;
    virtual void mul(Tensor& result, const Tensor& a, const Tensor& b) = 0;

    // Advanced operation interfaces (V1.29.2)
    // Scalar operations
    virtual Tensor mul(const Tensor& input, float scalar) const = 0;
    virtual Tensor add(const Tensor& input, float scalar) const = 0;
    virtual Tensor minus(const Tensor& input, float scalar) const = 0;
    virtual Tensor minus(float scalar, const Tensor& input) const = 0;
    virtual Tensor mac(const Tensor& input, float scalar_x, float scalar_y) const = 0;

    // Broadcast operations
    virtual Tensor add_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const = 0;
    virtual Tensor minus_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const = 0;
    virtual Tensor mul_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const = 0;

    // Expansion operations
    virtual Tensor expand(const Tensor& tensor_a, const Shape& shape_b) const = 0;

    // Dimension operations
    virtual Tensor unsqueeze(const Tensor& tensor_a, int32_t dim) const = 0;
    virtual Tensor squeeze(const Tensor& tensor_a, int32_t dim) const = 0;

    // Data access interfaces
    virtual float get_scalar_float(const Tensor& tensor) = 0;
    virtual int32_t get_scalar_int32(const Tensor& tensor) = 0;
    virtual int8_t get_scalar_int8(const Tensor& tensor) = 0;

    // Tensor comparison
    virtual bool is_close(const Tensor& tensor_a, const Tensor& tensor_b, float eps = 5e-5f) const = 0;
};
```

### 4. BackendManager Backend Manager

**Design Features**:

- **Meyers Singleton**: Thread-safe singleton implementation
- **Static Convenience Methods**: Provide type-safe backend access
- **Auto-Registration**: Support compile-time configuration and runtime discovery

**Core Implementation**:

```cpp
class BackendManager {
public:
    // Meyers singleton, C++11 thread-safe
    static BackendManager& instance() {
        static BackendManager instance;
        return instance;
    }

    // Static convenience methods
    static std::shared_ptr<CudaBackend> get_cuda_backend(int device_id = 0) {
        return std::dynamic_pointer_cast<CudaBackend>(
            instance().get_backend(tr::CUDA(device_id))
        );
    }

    static std::shared_ptr<CpuBackend> get_cpu_backend() {
        return std::dynamic_pointer_cast<CpuBackend>(
            instance().get_backend(tr::CPU)
        );
    }

    std::shared_ptr<Backend> get_backend(const Device& device);
    void register_backend(const Device& device, std::shared_ptr<Backend> backend);
};
```

### 5. Specific Backend Implementations

#### CpuBackend - Row-Major Storage Implementation

**Storage Characteristics**:
- **Memory Layout**: Row-major (Row-major) storage
- **Memory Alignment**: 64-byte aligned, optimized for SIMD access
- **Computation Optimization**: Integrated Eigen3 library for vectorized computation

**Matrix Multiplication Implementation**:

```cpp
void CpuBackend::mm(Tensor& result, const Tensor& a, const Tensor& b) {
    // CPU tensors use row-major storage
    const float* a_data = static_cast<const float*>(a.data_ptr());
    const float* b_data = static_cast<const float*>(b.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    int32_t M = a.height();  // Row count
    int32_t K = a.width();   // Column count
    int32_t N = b.width();   // B's column count

    // Row-major matrix multiplication: C[M,N] = A[M,K] × B[K,N]
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int32_t k = 0; k < K; ++k) {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            result_data[i * N + j] = sum;
        }
    }
}
```

#### CudaBackend - Column-Major Storage Implementation

**Storage Characteristics**:
- **Memory Layout**: Column-major (Column-major) storage
- **Computation Libraries**: Based on cuBLAS and cuDNN
- **Performance Optimization**: Automatic algorithm selection, GPU performance near hardware limits

**Matrix Multiplication Implementation**:

```cpp
void CudaBackend::mm(Tensor& result, const Tensor& a, const Tensor& b) {
    // CUDA tensors use column-major storage
    const float* a_data = static_cast<const float*>(a.data_ptr());
    const float* b_data = static_cast<const float*>(b.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    int32_t M = a.height();  // Row count
    int32_t K = a.width();   // Column count
    int32_t N = b.width();   // B's column count

    // cuBLAS standard column-major matrix multiplication: C[M,N] = A[M,K] × B[K,N]
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
        N, M, K,                   // Result dimensions
        &alpha,
        b_data, N,                 // B matrix, leading dimension = N
        a_data, K,                 // A matrix, leading dimension = K
        &beta,
        result_data, N             // Result matrix, leading dimension = N
    ));
}
```

## Data Flow and Interaction Mechanisms

### Backend-Based Tensor Creation Flow (V1.31.1)

```cpp
// Backend-based tensor creation with type support
auto cpu_backend = BackendManager::get_cpu_backend();

// Create tensors with different data types
Tensor fp32_tensor = cpu_backend->randint(Shape(2, 3), 0, 10, DType::FP32, 42);
Tensor int8_tensor = cpu_backend->randint(Shape(2, 3), 0, 100, DType::INT8, 123);
Tensor int32_tensor = cpu_backend->randint(Shape(2, 3), 0, 1000, DType::INT32, 456);

// Cross-backend conversion preserves data types
auto cuda_backend = BackendManager::get_cuda_backend();
Tensor cuda_fp32 = cuda_backend->from_cpu(fp32_tensor);
Tensor cuda_int8 = cuda_backend->from_cpu(int8_tensor);
```

### Cross-Backend Computation Flow

```cpp
// 1. Create CPU tensor (row-major storage)
Tensor cpu_a = Tensor::randn(Shape(1024, 2048), 42, DType::FP32, tr::CPU);
Tensor cpu_b = Tensor::randn(Shape(2048, 512), 42, DType::FP32, tr::CPU);

// 2. Convert to CUDA (automatically converted to column-major)
auto cuda_backend = BackendManager::get_cuda_backend();
Tensor cuda_a = cuda_backend->from_cpu(cpu_a);  // Row-major → Column-major
Tensor cuda_b = cuda_backend->from_cpu(cpu_b);

// 3. CUDA matrix multiplication (column-major computation)
Tensor cuda_result = Tensor::empty(Shape(1024, 512), DType::FP32, tr::CUDA(0));
cuda_backend->mm(cuda_result, cuda_a, cuda_b);

// 4. Convert back to CPU (automatically converted back to row-major)
Tensor cpu_result = cuda_backend->to_cpu(cuda_result);  // Column-major → Row-major

// 5. Result verification: CPU and CUDA results should be consistent in row-major view
bool is_close = BackendManager::get_cpu_backend()->is_close(
    cpu_result, cpu_result, 1e-4f);
```

### Memory Layout Conversion Example

**Row-major to Column-major Conversion**:

```cpp
// Original row-major data (CPU)
// A[M,K] = [[1, 2, 3],
//           [4, 5, 6]]
// Memory layout: [1, 2, 3, 4, 5, 6]

// Convert to column-major data (CUDA)
// A^T[K,M] = [[1, 4],
//            [2, 5],
//            [3, 6]]
// Memory layout: [1, 4, 2, 5, 3, 6]

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

## Backend Operations (V1.29.2)

### Available Operation Categories

The backend system provides comprehensive tensor operations:

#### 1. Basic Arithmetic Operations
```cpp
// Element-wise operations
Tensor add_result = backend->add(tensor_a, tensor_b);
Tensor mul_result = backend->mul(tensor_a, tensor_b);
```

#### 2. Scalar Operations (New in V1.29.2)
```cpp
// Scalar arithmetic
Tensor scalar_mul = backend->mul(tensor, 2.0f);
Tensor scalar_add = backend->add(tensor, 1.0f);
Tensor scalar_mac = backend->mac(tensor, 2.0f, 1.0f);  // tensor * 2 + 1
```

#### 3. Broadcast Operations (New in V1.29.2)
```cpp
// Broadcasting tensor operations
Tensor broadcast_add = backend->add_broadcast(tensor_a, tensor_b);
Tensor broadcast_mul = backend->mul_broadcast(tensor_a, tensor_b);
```

#### 4. Shape Manipulation Operations
```cpp
// Shape expansion
Tensor expanded = backend->expand(tensor, Shape(2, 1, 3));

// Dimension manipulation (New in V1.29.2)
Tensor unsqueezed = backend->unsqueeze(tensor, 1);  // Insert dimension at position 1
Tensor squeezed = backend->squeeze(tensor, 0);     // Remove dimension at position 0
```

#### 5. Device Transfer Operations
```cpp
// Device conversions
Tensor cpu_tensor = backend->to_cpu(cuda_tensor);
Tensor cuda_tensor = backend->from_cpu(cpu_tensor);
```

## Performance Characteristics and Benchmarks

### Measured Performance (V1.29.2)

**CUDA Backend Performance**:
- **Matrix Multiplication**: 6673.76 GFLOPS (1024×2048 × 2048×512)
- **Average Execution Time**: 0.3218 ms
- **Efficiency**: 52.4% of theoretical peak (RTX 3060-class)
- **Precision**: Average relative error 4.590400e-07

**CPU Backend Performance**:
- **Matrix Multiplication**: Eigen-based SIMD optimization
- **Memory Alignment**: 64-byte alignment for cache access optimization
- **Multi-threading**: OpenMP parallel computation support

### Performance Optimization Strategies

1. **Memory Layout Optimization**: Each backend selects optimal memory layout format
2. **Zero-Copy Design**: Conversion layers execute format transformation only when necessary
3. **Cache-Friendly**: Contiguous memory layout and alignment optimization
4. **Algorithm Selection**: CUDA automatically selects optimal cuBLAS algorithms
5. **Vectorization**: CPU backend uses Eigen for SIMD optimization

## Usage Examples

### Basic Cross-Backend Operations

```cpp
#include "tech_renaissance.h"
using namespace tr;

int main() {
    // Get backend instances
    auto cuda_backend = BackendManager::get_cuda_backend();
    auto cpu_backend = BackendManager::get_cpu_backend();

    // Create random tensors (CPU, row-major)
    Tensor cpu_a = Tensor::randn(Shape(1024, 2048), 42);
    Tensor cpu_b = Tensor::randn(Shape(2048, 512), 42);

    // Convert to CUDA (automatically converted to column-major)
    Tensor cuda_a = cuda_backend->from_cpu(cpu_a);
    Tensor cuda_b = cuda_backend->from_cpu(cpu_b);

    // CUDA matrix multiplication (column-major computation)
    Tensor cuda_result = Tensor::empty(Shape(1024, 512), DType::FP32, tr::CUDA(0));
    cuda_backend->mm(cuda_result, cuda_a, cuda_b);

    // Convert back to CPU (automatically converted back to row-major)
    Tensor cpu_result = cuda_backend->to_cpu(cuda_result);

    // Result verification
    bool is_close = cpu_backend->is_close(cpu_result, cpu_result, 1e-4f);
    std::cout << "Results are close: " << (is_close ? "YES" : "NO") << std::endl;

    return 0;
}
```

### New Backend Operations (V1.29.2)

```cpp
// Scalar operations
Tensor scalar_result = cpu_backend->mul(input_tensor, 2.0f);
Tensor mac_result = cpu_backend->mac(input_tensor, 2.0f, 1.0f);

// Broadcast operations
Tensor broadcast_result = cpu_backend->add_broadcast(tensor_a, tensor_b);

// Shape operations
Tensor expanded_result = cpu_backend->expand(input_tensor, Shape(2, 1, 3));

// Dimension operations
Tensor unsqueezed_result = cpu_backend->unsqueeze(input_tensor, 1);
Tensor squeezed_result = cpu_backend->squeeze(unsqueezed_result, 1);
```

### Advanced API Usage (V1.29.2)

```cpp
// Use static convenience methods
auto cuda_backend = BackendManager::get_cuda_backend();
auto cpu_backend = BackendManager::get_cpu_backend();

// Use new matrix dimension alias methods
int32_t M = cpu_a.height();  // 1024
int32_t K = cpu_a.width();   // 2048
int32_t N = cpu_b.width();   // 512

// Shape compatibility checking
if (cpu_a.shape().is_matmul_compatible(cpu_b.shape())) {
    std::cout << "Matrices are compatible for multiplication" << std::endl;
}

// Chain operations with backend delegation
Tensor result = cpu_backend->add(
    cpu_backend->expand(tensor_a, Shape(2, 1, 3)),
    cpu_backend->squeeze(tensor_b, 1)
);
```

## Error Handling and Safety Guarantees

### Exception Safety Design

```cpp
// Unified exception class
class TRException : public std::exception {
public:
    TRException(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }
private:
    std::string message_;
};
```

### Memory Safety Guarantees

- **RAII Management**: Smart pointer automatic memory deallocation
- **Exception Safety**: Strong exception safety guarantees
- **Bounds Checking**: Shape dimension access bounds checking
- **Type Safety**: Compile-time and runtime type checking

## Extensibility Design

### Adding New Backends

1. **Inherit Backend Base Class** and implement all virtual functions
2. **Define Storage Format** (row-major, column-major, or other)
3. **Implement Conversion Methods** (`from_cpu`, `to_cpu`, `to`)
4. **Register with BackendManager**

### New Memory Format Support

The framework supports future memory format extensions:
- Sparse tensor storage formats
- Compressed storage formats
- Hardware-specific optimization formats

## Summary

The Tech Renaissance framework's Tensor-Backend system through the innovative "Backend-Managed Storage" design achieves:

1. **High Performance**: Each backend selects optimal memory layout, GPU performance reaches hardware limits
2. **User-Friendly**: Conversion layers transparently handle format conversions, users don't need to care about underlying implementation
3. **Type Safety**: Strong typing and comprehensive error checking mechanisms
4. **Device-Agnostic**: Unified API supports multiple devices and cross-device data transfer
5. **Extensibility**: Modular design supports new backends and new storage formats

**Key Innovations**:
- **Backend-Managed Storage Principle**: Each backend selects optimal memory layout
- **Transparent Conversion Layers**: Automatically handle conversions between different storage formats
- **Consistent Access Interface**: Users always see row-major data access
- **Operation Delegation**: Computational and shape operations delegated to specialized backend implementations

---

## Version Information

- **Version**: V1.31.1
- **Date**: 2025-11-02
- **Author**: 技术觉醒团队
- **Major Updates**: Removed Tensor shape operations, expanded backend operation capabilities, added scalar and broadcast operations, enhanced dimension manipulation
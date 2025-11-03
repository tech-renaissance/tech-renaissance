# CPU Backend Tensor Creation Operations

## # 重要警告：CPU后端是唯一推荐张量创建方式！

**所有张量必须通过CPU后端方法创建！**

在Tech Renaissance框架中，CPU后端提供了完整的张量创建API，这些方法会自动分配内存并返回可用的张量对象：

**推荐的张量创建方式：**
```cpp
auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(
    BackendManager::instance().get_backend(CPU));

// 基础张量创建（自动分配内存）
Tensor tensor_zeros = cpu_backend->zeros({2, 3, 4}, DType::FP32);
Tensor tensor_ones = cpu_backend->ones({2, 3, 4}, DType::FP32);
Tensor tensor_full = cpu_backend->full({2, 3, 4}, 1.5f);
Tensor tensor_empty = cpu_backend->empty({2, 3, 4}, DType::FP32);

// 随机张量生成
Tensor tensor_randn = cpu_backend->randn({2, 3, 4}, 12345);
Tensor tensor_uniform = cpu_backend->uniform({2, 3, 4}, 0.0f, 1.0f, 54321);
Tensor tensor_randint = cpu_backend->randint({2, 3, 4}, 0, 10, DType::INT32, 99999);
```

**绝对禁止的方式：**
```cpp
// 错误：直接使用Tensor构造函数不会分配内存！
Tensor tensor(shape, dtype, CPU);  // 段错误！

// 错误：使用Tensor静态方法（不推荐）
Tensor tensor = Tensor::zeros(shape, dtype, device);

// 错误：误认为其他Backend子类的方法和CPU后端一样
auto cuda_backend = std::dynamic_pointer_cast<CudaBackend>(
    BackendManager::instance().get_backend(CUDA));
// 注意：CUDA后端可能有不同的方法或参数！
```

**重要提醒：**
- 每个Backend子类（如CpuBackend、CudaBackend）可能有不同的方法实现
- 使用前请检查具体Backend子类的API文档
- 如果某个Backend子类没有提供特定方法，可以考虑先在CPU后端创建，然后转换到目标设备

## Overview

This document describes the implementation of tensor creation functions in the CPU backend of Tech Renaissance. The creation operations support various ways to generate tensors with specific values, random distributions, and patterns.

## Version Information

- **Version**: V1.32.3
- **Date**: 2025-11-03
- **Author**: 技术觉醒团队

## Supported Operations

The CPU backend implements 12 creation functions across 5 operation types:

### Value-based Creation
1. **Full**: `full`, `full_inplace`
   - Creates tensors filled with a specified value
   - Supports FP32 only (INT8 planned for future)
2. **Ones**: `ones`
   - Creates tensors filled with 1 values
   - Supports FP32, INT8, INT32

### Random Number Creation
2. **Normal Distribution**: `randn`, `randn_inplace`
   - Creates tensors with standard normal distribution (mean=0, std=1)
   - FP32 only

3. **Uniform Distribution**: `uniform`, `uniform_inplace`
   - Creates tensors with uniform distribution in specified range
   - FP32 only

4. **Integer Distribution**: `randint`, `randint_inplace`
   - Creates tensors with random integers in specified range
   - Supports FP32, INT8, INT32 data types
   - INT8 includes range validation [-128, 127]

5. **Boolean Distribution**: `randbool`, `randbool_inplace`
   - Creates tensors with random 0.0f and 1.0f values
   - Configurable zero occurrence probability
   - FP32 only, INT8 planned

### Function Modes
- **Regular Mode**: Returns a new tensor with generated data
- **Inplace Mode**: Modifies an existing tensor in place

## Creation Rules and Behavior

### Full Operations

#### `Tensor empty(const Shape& shape, DType dtype)`
- **Purpose**: Creates an uninitialized tensor with specified shape and data type
- **Parameters**:
  - `shape`: Shape of the tensor
  - `dtype`: Data type (FP32, INT8, INT32)
- **Returns**: New empty tensor
- **Device**: CPU
- **Important**: **This method allocates memory but leaves data uninitialized**
- **Example**: `Tensor t = cpu_backend->empty(Shape(3, 4), DType::FP32);`

**Note**: Unlike Tensor constructors which don't allocate memory, `empty()` actually allocates memory but doesn't initialize the data values.

#### `Tensor full(const Shape& shape, float value, DType dtype = DType::FP32)`
- **Purpose**: Creates a tensor filled with the specified value
- **Parameters**:
  - `shape`: Target tensor shape
  - `value`: Fill value for all elements
  - `dtype`: Data type (FP32 supported, INT8 throws TODO exception)
- **Returns**: New tensor with all elements equal to `value`
- **Optimization**: Uses Eigen `setConstant()` when available

#### `void full_inplace(Tensor& tensor_a, float value)`
- **Purpose**: Fills an existing tensor with the specified value
- **Parameters**:
  - `tensor_a`: Target tensor (must be allocated, non-empty, on CPU)
  - `value`: Fill value for all elements
- **Optimization**: Uses Eigen `setConstant()` when available

#### `Tensor ones(const Shape& shape, DType dtype = DType::FP32)`
- **Purpose**: Creates a tensor filled with 1 values
- **Parameters**:
  - `shape`: Target tensor shape
  - `dtype`: Data type (FP32, INT8, INT32 supported)
- **Returns**: New tensor with all elements equal to 1
- **Device**: CPU
- **Data Type Values**:
  - FP32: Fills with 1.0f
  - INT8: Fills with int8_t(1)
  - INT32: Fills with int32_t(1)
- **Optimization**: Uses Eigen `setConstant()` when available
- **Implementation**: Dual support for Eigen optimization and naive implementation
- **Example**:
  ```cpp
  Tensor fp32_ones = cpu_backend->ones(Shape(2, 3), DType::FP32);   // 1.0f
  Tensor int8_ones = cpu_backend->ones(Shape(2, 3), DType::INT8);    // int8_t(1)
  Tensor int32_ones = cpu_backend->ones(Shape(2, 3), DType::INT32);  // int32_t(1)
  ```

### Normal Distribution Operations

#### `Tensor randn(const Shape& shape, unsigned int seed = 0)`
- **Purpose**: Creates tensor with standard normal distribution (μ=0, σ=1)
- **Parameters**:
  - `shape`: Target tensor shape
  - `seed`: Random seed for reproducibility (default: 0)
- **Returns**: New tensor with normally distributed random values
- **Distribution**: Standard normal distribution N(0,1)

#### `void randn_inplace(Tensor& tensor_a, unsigned int seed = 0)`
- **Purpose**: Fills existing tensor with standard normal distribution
- **Parameters**:
  - `tensor_a`: Target tensor (FP32 only, non-empty, on CPU)
  - `seed`: Random seed for reproducibility (default: 0)

### Uniform Distribution Operations

#### `Tensor uniform(const Shape& shape, float min_val = 0.0f, float max_val = 1.0f, unsigned int seed = 0)`
- **Purpose**: Creates tensor with uniform distribution in specified range
- **Parameters**:
  - `shape`: Target tensor shape
  - `min_val`: Minimum value (inclusive)
  - `max_val`: Maximum value (exclusive)
  - `seed`: Random seed for reproducibility (default: 0)
- **Distribution**: Uniform distribution U(min_val, max_val)

#### `void uniform_inplace(Tensor& tensor_a, float min_val = 0.0f, float max_val = 1.0f, unsigned int seed = 0)`
- **Purpose**: Fills existing tensor with uniform distribution
- **Parameters**:
  - `tensor_a`: Target tensor (FP32 only, non-empty, on CPU)
  - `min_val`: Minimum value (inclusive)
  - `max_val`: Maximum value (exclusive)
  - `seed`: Random seed for reproducibility (default: 0)

### Integer Distribution Operations

#### `Tensor randint(const Shape& shape, int low, int high, DType dtype, unsigned int seed = 0)`
- **Purpose**: Creates tensor with random integers in specified range
- **Parameters**:
  - `shape`: Target tensor shape
  - `low`: Minimum integer (inclusive)
  - `high`: Maximum integer (exclusive)
  - `dtype`: Data type (FP32, INT8, INT32 supported)
  - `seed`: Random seed for reproducibility (default: 0)
- **Returns**: New tensor with integer values
- **Range**: Values in [low, high)
- **Validation**: Throws exception if low >= high or dtype unsupported
- **INT8 Range Check**: Validates range [-128, 127] for INT8 dtype

#### `void randint_inplace(Tensor& tensor_a, int low, int high, DType dtype, unsigned int seed = 0)`
- **Purpose**: Fills existing tensor with random integers
- **Parameters**:
  - `tensor_a`: Target tensor (must match dtype)
  - `low`: Minimum integer (inclusive)
  - `high`: Maximum integer (exclusive)
  - `dtype`: Data type (must match tensor_a.dtype())
  - `seed`: Random seed for reproducibility (default: 0)
- **Validation**: Throws exception if tensor dtype != input dtype or range invalid
- **Range**: Values in [low, high)

### Boolean Distribution Operations

#### `Tensor randbool(const Shape& shape, float rate_of_zeros, unsigned int seed = 0, DType dtype = DType::FP32)`
- **Purpose**: Creates tensor with random boolean values (0.0f and 1.0f)
- **Parameters**:
  - `shape`: Target tensor shape
  - `rate_of_zeros`: Probability of zero values (0.0 to 1.0)
  - `seed`: Random seed for reproducibility (default: 0)
  - `dtype`: Data type (FP32 supported, INT8 throws TODO exception)
- **Returns**: New tensor with 0.0f and 1.0f values
- **Distribution**: Bernoulli distribution with specified zero probability
- **Optimization**: Uses Eigen `unaryExpr()` when available
- **Validation**: Throws exception if rate_of_zeros not in [0,1]

#### `void randbool_inplace(Tensor& tensor_a, float rate_of_zeros, unsigned int seed = 0)`
- **Purpose**: Fills existing tensor with random boolean values
- **Parameters**:
  - `tensor_a`: Target tensor
  - `rate_of_zeros`: Probability of zero values (0.0 to 1.0)
  - `seed`: Random seed for reproducibility (default: 0)
- **Optimization**: Uses Eigen `unaryExpr()` when available
- **Validation**: Throws exception if rate_of_zeros not in [0,1]

## Implementation Details

### Performance Optimization

#### Eigen-Accelerated Functions
The following functions have Eigen optimizations when `TR_USE_EIGEN` is enabled:

1. **Full Operations**:
   - Uses `Eigen::VectorXf::setConstant()` for highly optimized memory filling
   - Significant performance improvement for large tensors

2. **Boolean Random Operations**:
   - Uses `Eigen::VectorXf::unaryExpr()` for vectorized random generation
   - Better cache utilization and SIMD optimization

#### Memory Management
- Uses `Tensor::empty()` for memory allocation
- Leverages existing backend memory management
- Proper RAII and smart pointer usage

### Random Number Generation
- Uses C++11 `<random>` library for high-quality random numbers
- `std::mt19937` for Mersenne Twister engine
- `std::normal_distribution<float>` for normal distribution
- `std::uniform_real_distribution<float>` for uniform distribution
- `std::uniform_int_distribution<int>` for integer distribution

### Error Handling
All functions throw `TRException` with descriptive error messages for:
- **Data Type Mismatches**: INT8 operations (TODO for future implementation)
- **Empty Tensors**: Inplace operations on unallocated tensors
- **Device Mismatches**: Non-CPU tensors
- **Invalid Parameters**: Out-of-range values, invalid distributions
- **Parameter Validation**: Invalid ranges, negative values where not allowed

## API Reference

### Functions

#### Value-based Creation
```cpp
Tensor full(const Shape& shape, float value, DType dtype = DType::FP32)
void full_inplace(Tensor& tensor_a, float value)
```

#### Normal Distribution
```cpp
Tensor randn(const Shape& shape, unsigned int seed = 0)
void randn_inplace(Tensor& tensor_a, unsigned int seed = 0)
```

#### Uniform Distribution
```cpp
Tensor uniform(const Shape& shape, float min_val = 0.0f, float max_val = 1.0f, unsigned int seed = 0)
void uniform_inplace(Tensor& tensor_a, float min_val = 0.0f, float max_val = 1.0f, unsigned int seed = 0)
```

#### Integer Distribution
```cpp
Tensor randint(const Shape& shape, int low, int high, unsigned int seed = 0, DType dtype = DType::FP32)
void randint_inplace(Tensor& tensor_a, int low, int high, unsigned int seed = 0)
```

#### Boolean Distribution
```cpp
Tensor randbool(const Shape& shape, float rate_of_zeros, unsigned int seed = 0, DType dtype = DType::FP32)
void randbool_inplace(Tensor& tensor_a, float rate_of_zeros, unsigned int seed = 0)
```

## Usage Examples

### Basic Value Creation
```cpp
auto cpu_backend = BackendManager::get_cpu_backend();

// Create tensors filled with specific values
Tensor ones = cpu_backend->full(Shape(3, 4), 1.0f);  // 3x4 tensor filled with 1.0
Tensor zeros = cpu_backend->full(Shape(2, 2, 2), 0.0f);  // 2x2x2 tensor filled with 0.0

// Fill existing tensor
Tensor existing = Tensor::empty(Shape(2, 3), DType::FP32, tr::CPU);
cpu_backend->full_inplace(existing, 5.5f);  // Fill with 5.5
```

### Random Number Generation
```cpp
// Normal distribution
Tensor normal = cpu_backend->randn(Shape(1000), 42);  // 1000 elements with N(0,1)
cpu_backend->randn_inplace(existing, 123);  // Fill existing with N(0,1)

// Uniform distribution
Tensor uniform = cpu_backend->uniform(Shape(3, 4), 10.0f, 20.0f, 456);  // [10,20)
cpu_backend->uniform_inplace(existing, -5.0f, 5.0f, 789);  // [-5,5)

// Integer distribution
Tensor integers = cpu_backend->randint(Shape(2, 3), 1, 10, 321);  // [1,10)
cpu_backend->randint_inplace(existing, 0, 100, 654);  // [0,100)

// Boolean distribution
Tensor booleans = cpu_backend->randbool(Shape(4, 5), 0.3f, 987);  // 30% zeros
cpu_backend->randbool_inplace(existing, 0.8f, 246);  // 80% zeros
```

### Reproducible Results
```cpp
// Same seed produces same results
Tensor a = cpu_backend->randn(Shape(3, 3), 42);
Tensor b = cpu_backend->randn(Shape(3, 3), 42);
// a and b contain identical values
```

### Complex Scenarios
```cpp
// Create different types of tensors for testing
auto cpu_backend = BackendManager::get_cpu_backend();

// Neural network weight initialization
Tensor weights = cpu_backend->randn(Shape(784, 256), 42) * 0.01f;

// Bias initialization
Tensor bias = cpu_backend->full(Shape(256), 0.0f);

// Binary mask
Tensor mask = cpu_backend->randbool(Shape(256), 0.5f, 123);

// Random labels (0-9)
Tensor labels = cpu_backend->randint(Shape(100), 0, 10, 456);
```

## Performance Characteristics

Based on implementation analysis:

### Optimized Operations (Eigen)
- **full/full_inplace**: Highly optimized memory filling with vectorization
- **randbool/randbool_inplace**: Vectorized random generation with better cache performance

### Standard Operations
- **randn**: Standard C++11 normal distribution (already well-optimized)
- **uniform**: Standard C++11 uniform distribution
- **randint**: Standard C++11 integer distribution

### Memory Efficiency
- Direct memory access without intermediate copies
- Efficient use of Eigen's memory mapping
- Proper alignment for SIMD operations

## Testing

The implementation includes comprehensive test coverage:

### Test Categories
1. **Functionality Tests**: Verify correct value generation
2. **Reproducibility Tests**: Ensure same seeds produce same results
3. **Range Tests**: Verify random numbers stay within specified ranges
4. **Type Tests**: Verify integer values and boolean values
5. **Error Handling Tests**: Verify proper exception handling
6. **Parameter Validation**: Test boundary conditions and invalid inputs

### Running Tests
```bash
# Build creation tests
cmake --build . --target test_cpu_create

# Run creation tests
bin/tests/Release/test_cpu_create.exe
```

### Expected Test Output
```
Starting CPU backend create functions tests...
=== Testing full function ===
PASS: full function test
=== Testing randn function ===
PASS: randn function test
...
=== All Create Functions Tests Completed ===
```

## Design Considerations

### Performance Optimization
- **Eigen Integration**: Leverages Eigen's optimized linear algebra operations
- **Memory Mapping**: Uses Eigen::Map for zero-copy operations
- **Vectorization**: SIMD optimizations for compatible operations

### API Consistency
- **Uniform Interface**: Consistent parameter ordering and naming
- **Default Parameters**: Sensible defaults for optional parameters
- **Error Handling**: Standardized exception handling across all functions

### Extensibility
- **Modular Design**: Easy to add new distribution types
- **Eigen Integration**: Ready for additional optimized operations
- **Type Support**: Framework ready for INT8 implementation

## Dependencies

- **Standard Library**: `<random>`, `<stdexcept>`, `<cstring>`
- **Eigen Library**: Eigen/Dense (when TR_USE_EIGEN is enabled)
- **Internal Modules**: cpu_backend.h, tensor.h, backend_manager.h, tr_exception.h
- **Backend System**: CPU backend memory management and data access

## Future Improvements

1. **INT8 Support**: Complete INT8 implementation for all functions
2. **Advanced Distributions**: Support for Poisson, exponential, etc.
3. **Performance Optimization**: Additional Eigen optimizations for other functions
4. **GPU Backend**: CUDA implementations for tensor creation
5. **Parallel Generation**: Multi-threaded random number generation
6. **Seeding Strategies**: Better seed management for reproducibility
7. **Memory Pool Integration**: Integration with backend memory pools

## Files

- **Implementation**: `src/backend/cpu/cpu_create.cpp`
- **Header**: `include/tech_renaissance/backend/cpu/cpu_backend.h`
- **Tests**: `tests/unit_tests/test_cpu_create.cpp`
- **Build Configuration**: `src/backend/CMakeLists.txt`, `tests/unit_tests/CMakeLists.txt`

## 张量销毁最佳实践

### # 推荐的张量销毁方法

在Tech Renaissance框架中，对于大型张量的销毁，我们强烈建议结合以下两种方法：

#### 方法1：RAII作用域管理（推荐用于局部张量）

```cpp
auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(
    BackendManager::instance().get_backend(CPU));

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
auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(
    BackendManager::instance().get_backend(CPU));

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

### 内存分配的重要区别

**关键理解不同方法的内存行为：**

1. **Tensor构造函数**：只创建元数据，**不分配内存**（段错误！）
   ```cpp
   Tensor tensor(shape, dtype, device);  // 危险！没有内存
   ```

2. **Backend::empty()**：**分配内存但未初始化数据**
   ```cpp
   Tensor tensor = cpu_backend->empty(shape, dtype);  // 有内存，数据未初始化
   ```

3. **Backend::null_tensor()**：真正的空张量，**不占用内存**
   ```cpp
   tensor = cpu_backend->null_tensor();  // 真正的空张量
   ```

### 为什么推荐这两种方法？

1. **避免构造函数误用**：防止用户直接调用`Tensor()`构造函数
2. **API明确性**：`null_tensor()`比`empty()`更无歧义
3. **符合框架设计**：所有操作都通过后端，保持一致性

### 实际案例参考

参见 `tests/unit_tests/test_memory_occupation.cpp` 中的完整测试案例，该测试验证了：
- RAII作用域管理的有效性
- `null_tensor()`方法的正确性
- 不同销毁方式的内存释放效果

### 内存管理最佳实践总结

```cpp
void optimal_memory_usage_example() {
    auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(
        BackendManager::instance().get_backend(CPU));

    // 方法1：临时张量使用RAII
    {
        Tensor temp_data = cpu_backend->zeros(Shape(1000, 1000));
        // 处理temp_data
    }  // 自动释放

    // 方法2：需要长期存在但可能提前释放的张量
    Tensor persistent_tensor = cpu_backend->zeros(Shape(2000, 2000));

    // 在某些条件下提前释放
    if (some_condition) {
        persistent_tensor = cpu_backend->null_tensor();
    }

    // 继续使用persistent_tensor（如果没被释放）
}
```

**核心原则**：无论使用哪种方法，都要避免直接调用Tensor类的构造函数进行销毁操作。

## Related Documentation

- [CPU Backend Dimension Operations](cpu_dimension.md) - Tensor dimension manipulation
- [CPU Backend Unary Operations](cpu_unary.md) - Single-tensor operations
- [CPU Backend Broadcast Operations](cpu_broadcast.md) - Broadcasting operations
- [CPU Backend Scalar Operations](cpu_scalar.md) - Scalar-tensor operations
- [CPU Backend Matrix Operations](cpu_mm.md) - Matrix multiplication operations
- [Tensor Class Documentation](tensor.md) - Core tensor functionality
- [Backend Architecture](backend.md) - Overall backend system design
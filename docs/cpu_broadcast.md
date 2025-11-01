# CPU Backend Broadcast Operations

## Overview

This document describes the implementation of broadcast tensor operations in the CPU backend of Tech Renaissance. The broadcast operations support element-wise arithmetic operations (addition, subtraction, multiplication) between tensors of compatible shapes, following NumPy-style broadcasting rules.

## Version Information

- **Version**: V1.29.2
- **Date**: 2025-11-01
- **Author**: 技术觉醒团队

## Supported Operations

The CPU backend implements 8 broadcast functions across 4 operation types:

### Arithmetic Operations
1. **Addition**: `add_broadcast`, `add_broadcast_into`
2. **Subtraction**: `minus_broadcast`, `minus_broadcast_into`
3. **Multiplication**: `mul_broadcast`, `mul_broadcast_into`

### Expansion Operations
4. **Tensor Expansion**: `expand`, `expand_into`

### Function Modes
- **Regular Mode**: Returns a new tensor with the result
- **Into Mode**: Writes the result to a pre-allocated tensor

## Broadcasting Rules

### Supported Broadcasting Patterns

1. **Same Shape**: Direct element-wise operation
   - Example: `(2,3) + (2,3) = (2,3)`

2. **Scalar Broadcasting**: Scalar can broadcast to any tensor shape
   - Example: `scalar(5) + tensor(2,3) = tensor(2,3)`

3. **Dimension Broadcasting**: Size-1 dimensions can expand to match larger dimensions
   - Example: `(1,3) + (2,3) = (2,3)`
   - Example: `(2,1) + (2,3) = (2,3)`

### Unsupported Patterns

1. **Dimension Mismatch**: Tensors with different numbers of dimensions cannot broadcast (except for scalar)
   - Example: `(3,) + (2,3)` → **Error**

2. **Incompatible Shapes**: Dimensions that are neither equal nor 1 cannot broadcast
   - Example: `(2,3) + (4,5)` → **Error**

## Implementation Details

### Core Components

#### 1. Shape Validation (`get_broadcast_output_shape`)
```cpp
static Shape get_broadcast_output_shape(const Shape& shape_a, const Shape& shape_b)
```
- Validates tensor shape compatibility
- Determines output shape for broadcast operations
- Throws exceptions for incompatible shapes

#### 2. Tensor Validation Functions
- `validate_tensors_compatibility`: Checks data types and devices
- `validate_tensor_not_empty`: Ensures tensors are not empty
- `validate_tensor_dtype_and_device`: Validates FP32 and CPU device

#### 3. Index Conversion Functions
- `linear_to_multi_index`: Converts linear index to multi-dimensional indices
- `multi_index_to_linear_with_broadcast`: Handles broadcast-aware index mapping

### Execution Modes

#### Naive Implementation (`broadcast_operation_naive`)
- Universal implementation for all cases
- Handles complex broadcasting through index conversion
- Supports any broadcast pattern

#### Eigen Optimization (`broadcast_operation_eigen`)
- Optimized for simple cases:
  - Scalar broadcasting
  - Same-shape operations
- Uses Eigen's vectorized operations
- Falls back to naive implementation for complex cases

#### Expand Implementation
- **Naive Expand (`expand_operation_naive`)**: Handles tensor expansion through index conversion
- **Eigen Expand (`expand_operation_eigen`)**: Optimized for scalar expansion and same-shape copying
- **Shape Validation**: Validates compatibility between input tensor and target shape

### Performance Optimizations

1. **Scalar Broadcasting**: Direct scalar arithmetic without indexing
2. **Same Shape**: Direct vectorized operations using Eigen
3. **Eigen Vectorization**: Leverages SIMD instructions when available
4. **Scalar Expansion**: Efficient scalar filling using Eigen setConstant
5. **Same Shape Expansion**: Direct memory copy for identical shapes

## API Reference

### Functions

#### `Tensor add_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const`
Performs element-wise addition with broadcasting, returning a new tensor.

#### `void add_broadcast_into(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result) const`
Performs element-wise addition with broadcasting, writing to pre-allocated result tensor.

#### `Tensor minus_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const`
Performs element-wise subtraction with broadcasting, returning a new tensor.

#### `void minus_broadcast_into(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result) const`
Performs element-wise subtraction with broadcasting, writing to pre-allocated result tensor.

#### `Tensor mul_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const`
Performs element-wise multiplication with broadcasting, returning a new tensor.

#### `void mul_broadcast_into(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result) const`
Performs element-wise multiplication with broadcasting, writing to pre-allocated result tensor.

#### `Tensor expand(const Tensor& tensor_a, const Shape& shape_b) const`
Expands a tensor to the specified shape, returning a new tensor with broadcasted values.

#### `void expand_into(const Tensor& tensor_a, Tensor& tensor_b) const`
Expands tensor_a to the shape of tensor_b, writing the result directly to tensor_b.

### Error Handling

All functions throw `TRException` with descriptive error messages for:
- Empty tensors
- Data type mismatches (only FP32 supported)
- Device mismatches (only CPU supported)
- Incompatible shapes
- Output shape mismatches

## Usage Examples

### Basic Broadcasting
```cpp
auto cpu_backend = BackendManager::get_cpu_backend();

// Same shape operation
Tensor a = Tensor::full(Shape(2, 3), 2.0f);
Tensor b = Tensor::full(Shape(2, 3), 3.0f);
Tensor result = cpu_backend->add_broadcast(a, b);  // Result: all 5s

// Scalar broadcasting
Tensor scalar = Tensor::full(Shape(), 5.0f);
Tensor matrix = Tensor::full(Shape(2, 3), 2.0f);
Tensor result2 = cpu_backend->add_broadcast(scalar, matrix);  // Result: all 7s

// Dimension broadcasting
Tensor row_vec = Tensor::full(Shape(1, 3), 1.0f);
Tensor matrix = Tensor::full(Shape(2, 3), 4.0f);
Tensor result3 = cpu_backend->add_broadcast(row_vec, matrix);  // Result: all 5s
```

### Into Operations
```cpp
Tensor a = Tensor::full(Shape(1, 3), 2.0f);
Tensor b = Tensor::full(Shape(2, 3), 3.0f);
Tensor result = Tensor::empty(Shape(2, 3), DType::FP32, tr::CPU);

cpu_backend->add_broadcast_into(a, b, result);  // Result written to pre-allocated tensor
```

### Expand Operations
```cpp
// Scalar expansion
Tensor scalar = Tensor::full(Shape(), 3.0f);
Shape target_shape(2, 3);
Tensor expanded = cpu_backend->expand(scalar, target_shape);  // 2x3 tensor of 3s

// Shape expansion
Tensor row_vec = Tensor::full(Shape(1, 4), 2.0f);
Tensor result = cpu_backend->expand(row_vec, Shape(3, 4));  // 3x4 tensor, each row [2,2,2,2]

// Into expansion
Tensor input = Tensor::full(Shape(2, 1), 5.0f);
Tensor output = Tensor::empty(Shape(2, 4), DType::FP32, tr::CPU);
cpu_backend->expand_into(input, output);  // Output becomes [[5,5,5,5], [5,5,5,5]]
```

## Performance Characteristics

Based on benchmark tests:
- **Scalar Broadcasting**: ~511 microseconds for 1M elements
- **Same Shape Operations**: ~8 microseconds for 10K elements
- **Scalar Expansion**: ~486 microseconds for 1M elements
- **Same Shape Expansion**: ~9 microseconds for 10K elements
- **Eigen Optimization**: Significant speedup for simple cases
- **Complex Broadcasting**: Moderate overhead due to index conversion

## Testing

The implementation includes comprehensive test coverage:

### Test Categories
1. **Basic Operations**: Same shape and scalar broadcasting
2. **Shape Broadcasting**: Dimension broadcasting patterns
3. **Edge Cases**: Empty tensors, incompatible shapes, type errors
4. **Into Operations**: Output tensor validation
5. **Performance**: Large tensor operations
6. **Mathematical Correctness**: Result verification

### Running Tests
```bash
# Build broadcast tests
cmake --build . --target test_cpu_broadcast

# Run broadcast tests
bin/tests/Release/test_cpu_broadcast.exe

# Build expand tests
cmake --build . --target test_cpu_expand

# Run expand tests
bin/tests/Release/test_cpu_expand.exe
```

## Dependencies

- **Eigen Library**: For vectorized operations
- **C++17**: For modern C++ features
- **OpenMP**: For parallel processing (if enabled)

## Future Improvements

1. **Extended Data Type Support**: Add support for other numeric types
2. **GPU Backend**: Implement CUDA-based broadcasting
3. **Advanced Broadcasting**: Support more complex NumPy-style patterns
4. **Memory Optimization**: In-place operations where possible
5. **Parallel Processing**: Enhanced multi-threading for large tensors

## Files

- **Implementation**: `src/backend/cpu/cpu_broadcast.cpp`
- **Header**: `include/tech_renaissance/backend/cpu/cpu_backend.h`
- **Broadcast Tests**: `tests/unit_tests/test_cpu_broadcast.cpp`
- **Expand Tests**: `tests/unit_tests/test_cpu_expand.cpp`
- **Build Configuration**: `src/backend/CMakeLists.txt`
- **Test Configuration**: `tests/unit_tests/CMakeLists.txt`
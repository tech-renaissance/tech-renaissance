# CPU Backend Dimension Operations

## Overview

This document describes the implementation of tensor dimension operations in the CPU backend of Tech Renaissance. The dimension operations support adding and removing size-1 dimensions from tensors, following NumPy-style dimension manipulation rules.

## Version Information

- **Version**: V1.29.2
- **Date**: 2025-11-01
- **Author**: 技术觉醒团队

## Supported Operations

The CPU backend implements 6 dimension functions across 2 operation types:

### Unsqueeze Operations (Adding Dimensions)
1. **Unsqueeze**: `unsqueeze`, `unsqueeze_inplace`, `unsqueeze_into`
   - Inserts a size-1 dimension at the specified position
   - Increases tensor dimensionality by 1

### Squeeze Operations (Removing Dimensions)
2. **Squeeze**: `squeeze`, `squeeze_inplace`, `squeeze_into`
   - Removes a size-1 dimension at the specified position
   - Decreases tensor dimensionality by 1

### Function Modes
- **Regular Mode**: Returns a new tensor with modified dimensions
- **Inplace Mode**: Modifies the tensor in place (creates new object internally)
- **Into Mode**: Writes the result to a pre-allocated tensor

## Dimension Manipulation Rules

### Unsqueeze Rules

1. **Valid Range**: `dim` must be in `[0, ndim]`
   - `dim = 0`: Insert at the beginning
   - `dim = ndim`: Insert at the end
   - `dim < 0` or `dim > ndim`: Error

2. **Shape Transformation**: `(d1, d2, ..., dn)` → `(d1, ..., dim, 1, dim+1, ..., dn)`
   - Inserts a size-1 dimension at the specified position
   - Total number of elements remains unchanged

3. **Examples**:
   - `scalar() unsqueeze(0)` → `(1,)`
   - `(3,) unsqueeze(0)` → `(1, 3)`
   - `(2, 3) unsqueeze(1)` → `(2, 1, 3)`
   - `(2, 3) unsqueeze(2)` → `(2, 3, 1)`

### Squeeze Rules

1. **Valid Range**: `dim` must be in `[0, ndim-1]`
   - `dim < 0` or `dim >= ndim`: Error

2. **Size Requirement**: The dimension at `dim` must have size 1
   - `tensor.shape()[dim] == 1`: Required
   - Otherwise: Error

3. **Shape Transformation**: `(d1, ..., dim, 1, dim+1, ..., dn)` → `(d1, ..., dim-1, dim+1, ..., dn)`
   - Removes the size-1 dimension
   - Total number of elements remains unchanged

4. **Examples**:
   - `(1,) squeeze(0)` → `scalar()`
   - `(1, 3) squeeze(0)` → `(3,)`
   - `(2, 1, 3) squeeze(1)` → `(2, 3)`
   - `(2, 3, 1) squeeze(2)` → `(2, 3)`

## Implementation Details

### Core Components

#### 1. Shape Calculation Functions
```cpp
static Shape calculate_unsqueeze_shape(const Shape& original_shape, int32_t dim)
static Shape calculate_squeeze_shape(const Shape& original_shape, int32_t dim)
```
- Calculate the resulting shape after dimension operations
- Validate dimension indices and sizes
- Construct new Shape objects with appropriate dimensions

#### 2. Core Operation Functions
```cpp
static void unsqueeze_operation_core(const Tensor& input, Tensor& result)
static void squeeze_operation_core(const Tensor& input, Tensor& result)
```
- Perform the actual data transfer between tensors
- Since these are reshape operations, only memory copying is needed
- Preserve all data values without modification

#### 3. Validation Functions
- `validate_tensor_dtype_and_device`: Ensures FP32 and CPU compatibility
- `validate_tensor_not_empty`: Prevents operations on empty tensors
- Element count validation for `into` operations

### Execution Modes

#### Memory Copy Implementation
- **Data Preservation**: All operations preserve data values through memory copying
- **Efficient Transfer**: Uses `memcpy` for optimal performance
- **Shape Independence**: Data layout remains compatible across dimension changes

#### Shape Reconstruction
- **Dynamic Shape Building**: Constructs new shapes using vector operations
- **Multi-dimensional Support**: Handles up to 4 dimensions (current framework limit)
- **Automatic Detection**: Creates appropriate Shape objects based on dimension count

## API Reference

### Functions

#### `Tensor unsqueeze(const Tensor& tensor_a, int32_t dim) const`
Inserts a size-1 dimension at the specified position, returning a new tensor.

#### `void unsqueeze_inplace(Tensor& tensor_a, int32_t dim) const`
Modifies the tensor by inserting a size-1 dimension at the specified position.

#### `void unsqueeze_into(const Tensor& tensor_a, Tensor& tensor_b) const`
Copies tensor_a into tensor_b, assuming tensor_b has the target unsqueezed shape.

#### `Tensor squeeze(const Tensor& tensor_a, int32_t dim) const`
Removes a size-1 dimension at the specified position, returning a new tensor.

#### `void squeeze_inplace(Tensor& tensor_a, int32_t dim) const`
Modifies the tensor by removing a size-1 dimension at the specified position.

#### `void squeeze_into(const Tensor& tensor_a, Tensor& tensor_b) const`
Copies tensor_a into tensor_b, assuming tensor_b has the target squeezed shape.

### Error Handling

All functions throw `TRException` with descriptive error messages for:
- Empty tensors
- Data type mismatches (only FP32 supported)
- Device mismatches (only CPU supported)
- Dimension index out of range
- Invalid squeeze targets (dimension size ≠ 1)
- Element count mismatches in `into` operations

## Usage Examples

### Basic Unsqueeze Operations
```cpp
auto cpu_backend = BackendManager::get_cpu_backend();

// Scalar to 1D
Tensor scalar = Tensor::full(Shape(), 3.14f);
Tensor vec = cpu_backend->unsqueeze(scalar, 0);  // Shape: (1,)

// 1D to 2D
Tensor vec1d = Tensor::full(Shape(3), 2.0f);
Tensor mat = cpu_backend->unsqueeze(vec1d, 0);  // Shape: (1, 3)

// 2D to 3D (insert at position 1)
Tensor mat2d = Tensor::full(Shape(2, 3), 1.0f);
Tensor tensor3d = cpu_backend->unsqueeze(mat2d, 1);  // Shape: (2, 1, 3)
```

### Basic Squeeze Operations
```cpp
// 1D to scalar
Tensor vec = Tensor::full(Shape(1), 5.0f);
Tensor scalar = cpu_backend->squeeze(vec, 0);  // Shape: ()

// 2D to 1D
Tensor mat = Tensor::full(Shape(1, 4), 2.0f);
Tensor squeezed = cpu_backend->squeeze(mat, 0);  // Shape: (4,)

// 3D to 2D
Tensor tensor3d = Tensor::full(Shape(2, 1, 3), 3.0f);
Tensor mat2d = cpu_backend->squeeze(tensor3d, 1);  // Shape: (2, 3)
```

### Into Operations
```cpp
// Unsqueeze into pre-allocated tensor
Tensor input = Tensor::full(Shape(2, 3), 1.0f);
Tensor output = Tensor::empty(Shape(2, 1, 3), DType::FP32, tr::CPU);
cpu_backend->unsqueeze_into(input, output);  // Output gets the unsqueezed data

// Squeeze into pre-allocated tensor
Tensor input2 = Tensor::full(Shape(1, 4), 2.0f);
Tensor output2 = Tensor::empty(Shape(4), DType::FP32, tr::CPU);
cpu_backend->squeeze_into(input2, output2);  // Output2 gets the squeezed data
```

### Inplace Operations
```cpp
// Unsqueeze inplace
Tensor tensor = Tensor::full(Shape(2, 3), 1.0f);
cpu_backend->unsqueeze_inplace(tensor, 1);  // tensor.shape() becomes (2, 1, 3)

// Squeeze inplace
Tensor tensor2 = Tensor::full(Shape(1, 4), 2.0f);
cpu_backend->squeeze_inplace(tensor2, 0);  // tensor2.shape() becomes (4,)
```

### Complex Operations
```cpp
// Multiple unsqueeze operations
Tensor original = Tensor::full(Shape(3), 1.0f);
Tensor step1 = cpu_backend->unsqueeze(original, 0);    // (1, 3)
Tensor step2 = cpu_backend->unsqueeze(step1, 1);      // (1, 1, 3)
Tensor step3 = cpu_backend->unsqueeze(step2, 2);      // (1, 1, 1, 3)

// Inverse operations
Tensor start = Tensor::full(Shape(2, 3), 4.0f);
Tensor unsqueezed = cpu_backend->unsqueeze(start, 1);   // (2, 1, 3)
Tensor squeezed = cpu_backend->squeeze(unsqueezed, 1);  // (2, 3) - back to original
```

## Performance Characteristics

Based on benchmark tests:
- **Large Tensor Unsqueeze**: ~441 microseconds for 1M elements
- **Large Tensor Squeeze**: ~436 microseconds for 1M elements
- **Memory Copy Operations**: Efficient data transfer using `memcpy`
- **Shape Calculation**: Minimal overhead for dimension manipulation
- **Data Preservation**: No data modification, only reorganization

## Testing

The implementation includes comprehensive test coverage:

### Test Categories
1. **Basic Operations**: Core unsqueeze and squeeze functionality
2. **Dimension Variations**: Different tensor dimensions and positions
3. **Edge Cases**: Error conditions and boundary cases
4. **Into Operations**: Pre-allocated tensor writing
5. **Inplace Operations**: Direct tensor modification
6. **Complex Scenarios**: Multiple operations and inverse operations
7. **Performance**: Large tensor operations

### Running Tests
```bash
# Build dimension tests
cmake --build . --target test_cpu_dimension

# Run dimension tests
bin/tests/Release/test_cpu_dimension.exe
```

## Design Considerations

### Data Integrity
- **No Data Loss**: All operations preserve data values completely
- **Memory Safety**: Proper bounds checking and validation
- **Shape Consistency**: Accurate shape calculation and verification

### Performance Optimization
- **Efficient Memory Operations**: Direct memory copying for data transfer
- **Minimal Computation**: Focus on shape manipulation rather than data processing
- **Scalable Design**: Handles tensors of various sizes efficiently

### API Consistency
- **Uniform Interface**: Consistent with other CPU backend operations
- **Error Handling**: Standardized exception handling across all functions
- **Documentation**: Clear function signatures and behavior descriptions

## Dependencies

- **Standard Library**: `<vector>`, `<string>`, `<algorithm>`, `<stdexcept>`
- **Internal Modules**: cpu_backend.h, tensor.h, tr_exception.h
- **Memory Operations**: `memcpy` for efficient data transfer

## Future Improvements

1. **Extended Dimension Support**: Support for more than 4 dimensions
2. **Advanced Shape Manipulation**: Support for multiple dimension operations
3. **GPU Backend**: Implement CUDA-based dimension operations
4. **Broadcast Integration**: Better integration with broadcasting operations
5. **Memory Optimization**: In-place memory operations where possible
6. **Validation Enhancement**: More comprehensive shape compatibility checking

## Files

- **Implementation**: `src/backend/cpu/cpu_dimension.cpp`
- **Header**: `include/tech_renaissance/backend/cpu/cpu_backend.h`
- **Tests**: `tests/unit_tests/test_cpu_dimension.cpp`
- **Build Configuration**: `src/backend/CMakeLists.txt`, `tests/unit_tests/CMakeLists.txt`

## Related Documentation

- [CPU Backend Broadcast Operations](cpu_broadcast.md) - Broadcasting operations that work well with dimension manipulation
- [Tensor Class Documentation](tensor.md) - Core tensor functionality
- [Backend Architecture](backend.md) - Overall backend system design
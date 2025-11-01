# CPU Backend Dimension Operations

## Overview

This document describes the implementation of tensor dimension operations in the CPU backend of Tech Renaissance. The dimension operations support adding and removing size-1 dimensions, as well as padding tensors with zeros around their spatial dimensions.

## Version Information

- **Version**: V1.29.4
- **Date**: 2025-11-02
- **Author**: 技术觉醒团队

## Supported Operations

The CPU backend implements 9 dimension functions across 3 operation types:

### Unsqueeze Operations (Adding Dimensions)
1. **Unsqueeze**: `unsqueeze`, `unsqueeze_inplace`, `unsqueeze_into`
   - Inserts a size-1 dimension at the specified position
   - Increases tensor dimensionality by 1

### Squeeze Operations (Removing Dimensions)
2. **Squeeze**: `squeeze`, `squeeze_inplace`, `squeeze_into`
   - Removes a size-1 dimension at the specified position
   - Decreases tensor dimensionality by 1

### Pad Operations (Zero Padding)
3. **Pad**: `pad`, `pad_into`
   - Adds zero padding around the H and W dimensions of tensors
   - Expands spatial dimensions while preserving original data in the center

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

### Pad Rules

1. **Dimension Requirement**: Tensors must have at least 2 dimensions
   - Supports 2D, 3D, and 4D tensors
   - 1D tensors are not supported for padding

2. **Shape Transformation**:
   - **2D**: `(H, W)` → `(H+2p, W+2p)`
   - **3D**: `(C, H, W)` → `(C, H+2p, W+2p)`
   - **4D**: `(N, C, H, W)` → `(N, C, H+2p, W+2p)`
   where `p` is the padding size

3. **Padding Behavior**:
   - Adds `p` rows/columns of zeros around each spatial dimension
   - Original data is placed in the center of the expanded tensor
   - All boundary elements are filled with zeros

4. **Examples**:
   - `(3, 4) pad(1)` → `(5, 6)`
   - `(2, 3, 4) pad(2)` → `(2, 7, 8)`
   - `(2, 3, 4, 5) pad(1)` → `(2, 3, 6, 7)`

## Implementation Details

### Core Components

#### 1. Shape Calculation Functions
```cpp
static Shape calculate_unsqueeze_shape(const Shape& original_shape, int32_t dim)
static Shape calculate_squeeze_shape(const Shape& original_shape, int32_t dim)
static Shape calculate_pad_shape(const Shape& original_shape, int32_t padding)
```
- Calculate the resulting shape after dimension operations
- Validate dimension indices and sizes
- Construct new Shape objects with appropriate dimensions

#### 2. Core Operation Functions
```cpp
static void unsqueeze_operation_core(const Tensor& input, Tensor& result)
static void squeeze_operation_core(const Tensor& input, Tensor& result)
static void pad_operation_core_fp32(const Tensor& input, Tensor& result, int32_t padding)
static void pad_operation_core_int8(const Tensor& input, Tensor& result, int32_t padding)
```
- Perform the actual data transfer between tensors
- Unsqueeze/squeeze: Memory copying for reshape operations
- Pad: Zero filling and centering of original data

#### 3. Validation Functions
- `validate_tensor_dtype_and_device`: Ensures FP32/INT8 and CPU compatibility
- `validate_tensor_not_empty`: Prevents operations on empty tensors
- Element count validation for `into` operations
- Padding value validation for pad operations

### Execution Modes

#### Memory Copy Implementation
- **Data Preservation**: Unsqueeze/squeeze operations preserve data values through memory copying
- **Efficient Transfer**: Uses `memcpy` for optimal performance
- **Shape Independence**: Data layout remains compatible across dimension changes

#### Padding Implementation
- **Zero Initialization**: Complete output tensor filled with zeros using `memset`
- **Centered Copy**: Original data copied to center position with proper offset calculations
- **Type Support**: Separate optimized implementations for FP32 and INT8
- **Memory Alignment**: Optimized for aligned memory access patterns

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

#### `Tensor pad(const Tensor& tensor_a, int32_t padding) const`
Adds zero padding around H and W dimensions, returning a new tensor.

#### `void pad_into(const Tensor& tensor_a, int32_t padding, Tensor& tensor_b) const`
Writes padded tensor to pre-allocated tensor_b with correct expanded shape.

### Error Handling

All functions throw `TRException` with descriptive error messages for:
- Empty tensors
- Data type mismatches (only FP32 and INT8 supported)
- Device mismatches (only CPU supported)
- Dimension index out of range
- Invalid squeeze targets (dimension size ≠ 1)
- Invalid padding values (must be non-negative)
- Tensor dimensions < 2 for padding operations
- Element count mismatches in `into` operations
- Shape mismatches in `pad_into` operations

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

### Basic Pad Operations
```cpp
auto cpu_backend = BackendManager::get_cpu_backend();

// 2D padding
Tensor input2d = Tensor::randn(Shape(3, 4), 42, DType::FP32, tr::CPU);
Tensor padded2d = cpu_backend->pad(input2d, 1);  // Shape: (5, 6)

// 3D padding (channel-first format)
Tensor input3d = Tensor::randn(Shape(2, 3, 4), 123, DType::FP32, tr::CPU);
Tensor padded3d = cpu_backend->pad(input3d, 2);  // Shape: (2, 7, 8)

// 4D padding (batch, channel, height, width)
Tensor input4d = Tensor::randn(Shape(2, 3, 4, 5), 456, DType::FP32, tr::CPU);
Tensor padded4d = cpu_backend->pad(input4d, 1);  // Shape: (2, 3, 6, 7)

// INT8 padding
Tensor input_int8 = Tensor::randint(0, 100, Shape(3, 4), 789, tr::CPU);
// Convert to INT8 and pad...
```

### Into Operations
```cpp
// Unsqueeze into pre-allocated tensor
Tensor input = Tensor::full(Shape(2, 3), 1.0f);
Tensor output = Tensor::empty(Shape(2, 1, 3), DType::FP32, tr::CPU);
cpu_backend->unsqueeze_into(input, output);  // Output gets the unsqueezed data

// Pad into pre-allocated tensor
Tensor pad_input = Tensor::randn(Shape(3, 4), 321, DType::FP32, tr::CPU);
Tensor pad_output = Tensor::empty(Shape(5, 6), DType::FP32, tr::CPU);
cpu_backend->pad_into(pad_input, 1, pad_output);  // Output gets padded data
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
// Multiple dimension operations
Tensor original = Tensor::randn(Shape(3, 4), 42, DType::FP32, tr::CPU);
Tensor step1 = cpu_backend->unsqueeze(original, 0);    // (1, 3, 4)
Tensor step2 = cpu_backend->squeeze(step1, 0);       // (3, 4) - back to original
Tensor step3 = cpu_backend->pad(original, 1);         // (5, 6)

// Combine operations for complex transformations
Tensor img_2d = Tensor::randn(Shape(224, 224), 100, DType::FP32, tr::CPU);
Tensor img_3d = cpu_backend->unsqueeze(img_2d, 0);    // (1, 224, 224) - add channel dim
Tensor img_4d = cpu_backend->unsqueeze(img_3d, 0);    // (1, 1, 224, 224) - add batch dim
Tensor padded_img = cpu_backend->pad(img_4d, 2);      // (1, 1, 228, 228) - add zero padding
```

## Performance Characteristics

Based on benchmark tests:

### Unsqueeze/Squeeze Operations
- **Large Tensor Unsqueeze**: ~441 microseconds for 1M elements
- **Large Tensor Squeeze**: ~436 microseconds for 1M elements
- **Memory Copy Operations**: Efficient data transfer using `memcpy`
- **Shape Calculation**: Minimal overhead for dimension manipulation

### Pad Operations
- **Memory Initialization**: Fast zero filling using `memset`
- **Centered Copy**: Efficient memory copy with proper indexing
- **Type Optimization**: Separate optimized paths for FP32 and INT8
- **Cache Efficiency**: Sequential memory access patterns for better performance

## Testing

The implementation includes comprehensive test coverage:

### Test Categories
1. **Basic Operations**: Core unsqueeze, squeeze, and padding functionality
2. **Dimension Variations**: Different tensor dimensions and positions
3. **Edge Cases**: Error conditions and boundary cases
4. **Into Operations**: Pre-allocated tensor writing
5. **Inplace Operations**: Direct tensor modification
6. **Complex Scenarios**: Multiple operations and inverse operations
7. **Data Types**: Testing both FP32 and INT8 support
8. **Padding Verification**: Boundary checking and centering validation

### Running Tests
```bash
# Build dimension tests
cmake --build . --target test_cpu_dimension

# Build padding tests
cmake --build . --target test_cpu_pad

# Run dimension tests
bin/tests/Release/test_cpu_dimension.exe

# Run padding tests
bin/tests/Release/test_cpu_pad.exe
```

## Design Considerations

### Data Integrity
- **No Data Loss**: All operations preserve data values completely
- **Memory Safety**: Proper bounds checking and validation
- **Shape Consistency**: Accurate shape calculation and verification
- **Padding Precision**: Exact boundary handling for padding operations

### Performance Optimization
- **Efficient Memory Operations**: Direct memory copying for data transfer
- **Zero Initialization**: Optimized `memset` for padding operations
- **Type Specialization**: Separate implementations for different data types
- **Minimal Computation**: Focus on shape manipulation rather than data processing

### API Consistency
- **Uniform Interface**: Consistent with other CPU backend operations
- **Error Handling**: Standardized exception handling across all functions
- **Documentation**: Clear function signatures and behavior descriptions
- **English Output**: All user-facing messages in English, comments in Chinese

## Dependencies

- **Standard Library**: `<vector>`, `<string>`, `<algorithm>`, `<stdexcept>`, `<cstring>`
- **Internal Modules**: cpu_backend.h, tensor.h, tr_exception.h, logger.h
- **Memory Operations**: `memcpy`, `memset` for efficient data transfer

## Future Improvements

1. **Extended Dimension Support**: Support for more than 4 dimensions
2. **Advanced Shape Manipulation**: Support for multiple dimension operations
3. **GPU Backend**: Implement CUDA-based dimension operations
4. **Broadcast Integration**: Better integration with broadcasting operations
5. **Memory Optimization**: In-place memory operations where possible
6. **Validation Enhancement**: More comprehensive shape compatibility checking
7. **Padding Variants**: Support for different padding modes (reflect, replicate, etc.)
8. **Performance Optimization**: SIMD optimizations for large tensor operations

## Files

- **Implementation**: `src/backend/cpu/cpu_dimension.cpp`
- **Header**: `include/tech_renaissance/backend/cpu/cpu_backend.h`
- **Dimension Tests**: `tests/unit_tests/test_cpu_dimension.cpp`
- **Padding Tests**: `tests/unit_tests/test_cpu_pad.cpp`
- **Build Configuration**: `src/backend/CMakeLists.txt`, `tests/unit_tests/CMakeLists.txt`

## Related Documentation

- [CPU Backend Broadcast Operations](cpu_broadcast.md) - Broadcasting operations that work well with dimension manipulation
- [CPU Backend Unary Operations](cpu_unary.md) - Single-tensor operations
- [Tensor Class Documentation](tensor.md) - Core tensor functionality
- [Backend Architecture](backend.md) - Overall backend system design
# CPU卷积操作实现文档

## 概述

本文档详细介绍了技术觉醒框架中CPU后端卷积操作的实现，包括标准卷积和转置卷积。实现支持多种stride、padding配置，并提供了高效的数值计算算法。

**版本**: V1.35.0
**更新日期**: 2025-11-03
**作者**: 技术觉醒团队
**文件位置**: `src/backend/cpu/cpu_conv.cpp`

## 功能特性

### 核心功能
- ✅ **标准卷积** (`conv`, `conv_into`)
- ✅ **转置卷积** (`transposed_conv`, `transposed_conv_into`)
- ✅ **多种stride支持**: 1, 2
- ✅ **灵活padding**: 0及任意非负值
- ✅ **张量维度支持**: 2D, 3D, 4D输入
- ✅ **内存布局**: NCHW格式，右对齐存储

### 约束条件
- 仅支持FP32数据类型
- 卷积核必须为正方形（kernel_h = kernel_w）
- 输入张量维度必须≥2
- 卷积核维度必须为4D (N, C, H, W)

## API接口

### 标准卷积

```cpp
Tensor conv(const Tensor& input, const Tensor& kernel,
           int32_t stride = 1, int32_t padding = 0);

void conv_into(const Tensor& input, const Tensor& kernel, Tensor& result,
              int32_t stride = 1, int32_t padding = 0);
```

### 转置卷积

```cpp
Tensor transposed_conv(const Tensor& input, const Tensor& kernel,
                      int32_t stride = 1, int32_t padding = 0);

void transposed_conv_into(const Tensor& input, const Tensor& kernel, Tensor& result,
                         int32_t stride = 1, int32_t padding = 0);
```

## 实现架构

### 1. 参数验证

所有卷积操作都通过 `validate_conv_parameters()` 函数进行严格的参数验证：

```cpp
static void validate_conv_parameters(const Tensor& tensor, const Tensor& kernel,
    int32_t stride, int32_t padding, const std::string& operation_name)
```

**验证项目**:
- 设备类型必须是CPU
- 张量存储已分配
- 数据类型必须是FP32
- 输入维度≥2，卷积核维度=4
- 卷积核为正方形
- stride = 1或2
- padding ≥ 0

### 2. 形状计算

#### 标准卷积形状公式
```cpp
static Shape calculate_conv_shape(const Shape& input_shape, const Shape& kernel_shape,
                                 int32_t stride, int32_t padding)
```

**计算公式**:
```
output_h = floor((input_h + 2 * padding - kernel_h) / stride) + 1
output_w = floor((input_w + 2 * padding - kernel_w) / stride) + 1
```

#### 转置卷积形状公式
```cpp
static Shape calculate_transposed_conv_shape(const Shape& input_shape, const Shape& kernel_shape,
                                           int32_t stride, int32_t padding)
```

**计算公式**:
```
output_h = (input_h - 1) * stride + kernel_h - 2 * padding
output_w = (input_w - 1) * stride + kernel_w - 2 * padding
```

### 3. 内存布局处理

实现支持多种输入维度的NCHW右对齐存储：

```cpp
// 2D输入: (H, W) -> 存储: (0, 0, H, W)
if (input_ndim == 2) {
    input_idx = ih * input_w + iw;
}
// 3D输入: (C, H, W) -> 存储: (0, C, H, W)
else if (input_ndim == 3) {
    input_idx = ic * input_h * input_w + ih * input_w + iw;
}
// 4D输入: (N, C, H, W) -> 存储: (N, C, H, W)
else if (input_ndim == 4) {
    input_idx = b * in_channels * input_h * input_w +
              ic * input_h * input_w +
              ih * input_w + iw;
}
```

### 4. 卷积算法实现

#### 标准卷积核心算法

```cpp
static void conv_operation_core_naive(const Tensor& input, const Tensor& kernel,
                                     Tensor& result, int32_t stride, int32_t padding)
```

**算法步骤**:

1. **计算起始位置**:
```cpp
int32_t ih_start = oh * stride - padding;
int32_t iw_start = ow * stride - padding;
```

2. **遍历卷积窗口**:
```cpp
for (int32_t kh = 0; kh < kernel_h; ++kh) {
    for (int32_t kw = 0; kw < kernel_w; ++kw) {
        int32_t ih = ih_start + kh;
        int32_t iw = iw_start + kw;

        // 边界检查，实现padding=0的效果
        if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
            // 计算输入和卷积核索引
            // 累加卷积结果
            sum_val += input_data[input_idx] * kernel_data[kernel_idx];
        }
    }
}
```

3. **Padding处理**:
   - 通过边界检查实现zero-padding
   - 超出输入边界的位置视为0值

#### 转置卷积核心算法

```cpp
static void transposed_conv_operation_core_naive(const Tensor& input, const Tensor& kernel,
                                                 Tensor& result, int32_t stride, int32_t padding)
```

**算法特点**:

1. **卷积核旋转**: 转置卷积本质上是卷积核旋转180度的卷积
2. **输出映射**: 每个输入元素影响输出中的一个区域
3. **累加操作**: 多个输入元素可能对同一输出位置有贡献

```cpp
// 计算输出中的起始位置
int32_t oh_start = ih * stride - padding;
int32_t ow_start = iw * stride - padding;

// 应用旋转180度的卷积核
int32_t oh = oh_start + (kernel_h - 1 - kh);
int32_t ow = ow_start + (kernel_w - 1 - kw);

// 累加到结果张量
result_data[result_idx] += input_val * kernel_data[kernel_idx];
```

### 5. 性能优化

#### Eigen优化支持
虽然当前实现使用朴素算法，但预留了Eigen优化接口：

```cpp
static void conv_operation_core_eigen(const Tensor& input, const Tensor& kernel,
                                     Tensor& result, int32_t stride, int32_t padding)
{
    // 未来可使用im2col + GEMM优化
    conv_operation_core_naive(input, kernel, result, stride, padding);
}
```

#### 优化方向
- **im2col变换**: 将卷积转换为矩阵乘法
- **Winograd算法**: 针对3x3卷积的快速算法
- **并行化**: 利用OpenMP进行多线程优化
- **缓存友好**: 优化内存访问模式

## 使用示例

### 基本卷积操作

```cpp
#include "tech_renaissance/backend/cpu/cpu_backend.h"

auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(
    BackendManager::instance().get_backend(CPU));

// 创建输入张量 (1, 1, 4, 4)
Tensor input = cpu_backend->ones(Shape(1, 1, 4, 4), DType::FP32);

// 创建3x3卷积核 (1, 1, 3, 3)
Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 3), DType::FP32);

// 执行卷积，stride=1, padding=1
Tensor result = cpu_backend->conv(input, kernel, 1, 1);
```

### 转置卷积操作

```cpp
// 2x2输入，stride=2上采样到5x5
Tensor input = cpu_backend->ones(Shape(2, 2), DType::FP32);
Tensor kernel = cpu_backend->ones(Shape(1, 1, 3, 3), DType::FP32);

// 转置卷积，stride=2, padding=0
Tensor result = cpu_backend->transposed_conv(input, kernel, 2, 0);
// 输出形状: (1, 1, 5, 5)
```

### In-place操作

```cpp
// 预分配输出张量
Shape output_shape = Shape(1, 1, 2, 2);
Tensor result = cpu_backend->empty(output_shape, DType::FP32);

// 直接写入预分配的张量
cpu_backend->conv_into(input, kernel, result, 1, 0);
```

## 数学原理

### 标准卷积

对于输出位置 `(oh, ow)`，卷积计算为：

```
output[oh, ow] = Σ(ic=0 to C-1) Σ(kh=0 to K-1) Σ(kw=0 to K-1)
                 input[ic, oh*stride+kh-padding, ow*stride+kw-padding] *
                 kernel[oc, ic, kh, kw]
```

其中超出输入边界的 `input` 值视为0（zero padding）。

### 转置卷积

转置卷积是标准卷积的梯度操作，对于输入位置 `(ih, iw)`：

```
output[oh, ow] += input[ih, iw] * kernel[oc, ic, K-1-kh, K-1-kw]
```

其中：
```
oh = ih * stride - padding + kh
ow = iw * stride - padding + kw
```

## 错误处理

实现提供了全面的错误处理机制：

### 常见错误类型

1. **设备类型错误**: 张量不在CPU设备上
2. **内存未分配**: 张量存储未正确初始化
3. **数据类型错误**: 不支持FP32以外的数据类型
4. **维度错误**: 输入维度<2或卷积核维度≠4
5. **形状错误**: 卷积核不是正方形
6. **参数错误**: stride不支持或padding为负数

### 异常示例

```cpp
try {
    Tensor result = cpu_backend->conv(input, kernel, 3, 0); // stride=3不支持
} catch (const TRException& e) {
    std::cout << "卷积错误: " << e.what() << std::endl;
    // 输出: [CPU Conv] Only supports stride 1 or 2
}
```

## 测试验证

实现通过了全面的单元测试：

- **基础功能测试**: 验证卷积计算的正确性
- **形状测试**: 验证不同参数组合下的输出形状
- **边界测试**: 验证padding和stride的边界情况
- **错误处理测试**: 验证异常情况的处理
- **性能测试**: 验证算法的时间和空间复杂度

测试文件: `tests/unit_tests/test_cpu_conv_new.cpp`

## 版本历史

- **V1.35.0** (2025-11-03): 初始实现，支持标准卷积和转置卷积
- 支持多种stride和padding配置
- 完整的参数验证和错误处理
- 详细的文档和测试覆盖

## 相关文件

- **实现文件**: `src/backend/cpu/cpu_conv.cpp`
- **头文件**: `include/tech_renaissance/backend/cpu/cpu_backend.h`
- **测试文件**: `tests/unit_tests/test_cpu_conv_new.cpp`
- **扩展测试**: `tests/unit_tests/test_cpu_conv_extra.cpp`
- **形状文档**: `docs/shape.md`
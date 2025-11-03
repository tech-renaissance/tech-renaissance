# CPU池化操作实现文档

## 概述

本文档详细介绍了技术觉醒框架中CPU后端池化操作的实现，包括最大池化（Max Pooling）和全局平均池化（Global Average Pooling）。实现支持多种kernel size和stride配置，并提供了高效的数值计算算法。

**版本**: V1.35.0
**更新日期**: 2025-11-03
**作者**: 技术觉醒团队
**文件位置**: `src/backend/cpu/cpu_pooling.cpp`

## 功能特性

### 核心功能
- ✅ **最大池化** (`max_pool`, `max_pool_into`)
- ✅ **全局平均池化** (`global_avg_pool`, `global_avg_pool_into`)
- ✅ **多种kernel size支持**: 2, 3, 4, 5等
- ✅ **灵活stride支持**: 1, 2
- ✅ **张量维度支持**: 2D, 3D, 4D输入
- ✅ **内存布局**: NCHW格式，右对齐存储

### 约束条件
- 仅支持FP32数据类型
- 输入张量维度必须≥2
- kernel_size必须为正数
- stride必须为1或2

## API接口

### 最大池化

```cpp
Tensor max_pool(const Tensor& input, int32_t kernel_size = 2, int32_t stride = 2);

void max_pool_into(const Tensor& input, Tensor& result,
                   int32_t kernel_size = 2, int32_t stride = 2);
```

### 全局平均池化

```cpp
Tensor global_avg_pool(const Tensor& input);

void global_avg_pool_into(const Tensor& input, Tensor& result);
```

## 实现架构

### 1. 参数验证

所有池化操作都通过严格的参数验证：

#### 最大池化验证
```cpp
static void validate_pooling_parameters(const Tensor& tensor, int32_t kernel_size,
                                       int32_t stride, const std::string& operation_name)
```

**验证项目**:
- 设备类型必须是CPU
- 张量存储已分配
- 数据类型必须是FP32
- 输入维度≥2
- kernel_size > 0
- stride = 1或2

#### 全局平均池化验证
```cpp
static void validate_global_avg_pool_parameters(const Tensor& tensor, const std::string& operation_name)
```

**验证项目**:
- 设备类型必须是CPU
- 张量存储已分配
- 数据类型必须是FP32
- 输入维度≥2

### 2. 形状计算

#### 最大池化形状公式
```cpp
static Shape calculate_pool_shape(const Shape& input_shape, int32_t kernel_size, int32_t stride)
```

**计算公式**:
```
output_h = floor((input_h - kernel_size) / stride) + 1
output_w = floor((input_w - kernel_size) / stride) + 1
```

**维度处理**:
```cpp
// 根据输入维度确定输出形状
if (input_ndim == 2) {
    // 2D输入: (H, W) -> 输出: (H_out, W_out)
    return Shape(output_h, output_w);
} else if (input_ndim == 3) {
    // 3D输入: (C, H, W) -> 输出: (C, H_out, W_out)
    return Shape(input_shape.c(), output_h, output_w);
} else if (input_ndim == 4) {
    // 4D输入: (N, C, H, W) -> 输出: (N, C, H_out, W_out)
    return Shape(input_shape.n(), input_shape.c(), output_h, output_w);
}
```

#### 全局平均池化形状
```cpp
static Shape calculate_global_avg_pool_shape(const Shape& input_shape)
```

**固定输出**: 全局平均池化总是输出 `(1, 1)` 形状，将所有空间维度压缩为单个值。

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

### 4. 池化算法实现

#### 最大池化核心算法

```cpp
static void max_pool_operation_core_naive(const Tensor& input, Tensor& result,
                                         int32_t kernel_size, int32_t stride)
```

**算法步骤**:

1. **遍历输出位置**:
```cpp
for (int32_t b = 0; b < batch_size; ++b) {
    for (int32_t c = 0; c < in_channels; ++c) {
        for (int32_t oh = 0; oh < output_h; ++oh) {
            for (int32_t ow = 0; ow < output_w; ++ow) {
```

2. **计算池化窗口**:
```cpp
// 计算池化窗口的起始位置
int32_t ih_start = oh * stride;
int32_t iw_start = ow * stride;

// 初始化最大值为最小浮点数
float max_val = -std::numeric_limits<float>::infinity();
```

3. **寻找最大值**:
```cpp
for (int32_t kh = 0; kh < kernel_size; ++kh) {
    for (int32_t kw = 0; kw < kernel_size; ++kw) {
        int32_t ih = ih_start + kh;
        int32_t iw = iw_start + kw;

        // 边界检查
        if (ih < input_h && iw < input_w) {
            int32_t input_idx = /* 计算输入索引 */;
            max_val = std::max(max_val, input_data[input_idx]);
        }
    }
}
```

4. **存储结果**:
```cpp
int32_t result_idx = /* 计算结果索引 */;
result_data[result_idx] = max_val;
```

#### 全局平均池化核心算法

```cpp
static void global_avg_pool_operation_core_naive(const Tensor& input, Tensor& result)
```

**算法特点**:

1. **全局统计**: 计算整个张量空间的平均值
2. **维度压缩**: 将空间维度压缩为单个值
3. **保持通道**: 保持批次和通道维度不变

```cpp
// 初始化累加器
float sum_val = 0.0f;

// 遍历所有空间位置
for (int32_t ih = 0; ih < input_h; ++ih) {
    for (int32_t iw = 0; iw < input_w; ++iw) {
        int32_t input_idx = /* 计算输入索引 */;
        sum_val += input_data[input_idx];
    }
}

// 计算平均值
int32_t total_elements = input_h * input_w;
result_data[result_idx] = sum_val / total_elements;
```

### 5. 性能优化

#### 算法复杂度

- **最大池化**: O(N × C × H_out × W_out × K²)
- **全局平均池化**: O(N × C × H × W)

#### 优化方向
- **并行化**: 利用OpenMP对批次和通道维度进行并行处理
- **向量化**: 使用SIMD指令加速最大值查找
- **缓存优化**: 优化内存访问模式，提高缓存命中率
- **分支预测**: 优化边界检查的分支结构

#### Eigen优化支持
虽然当前实现使用朴素算法，但预留了Eigen优化接口：

```cpp
static void max_pool_operation_core_eigen(const Tensor& input, Tensor& result,
                                          int32_t kernel_size, int32_t stride)
{
    // 未来可使用Eigen的reduce操作优化
    max_pool_operation_core_naive(input, result, kernel_size, stride);
}
```

## 使用示例

### 最大池化操作

```cpp
#include "tech_renaissance/backend/cpu/cpu_backend.h"

auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(
    BackendManager::instance().get_backend(CPU));

// 创建4x4输入张量
Tensor input = cpu_backend->ones(Shape(4, 4), DType::FP32);

// 2x2最大池化，stride=2
Tensor result = cpu_backend->max_pool(input, 2, 2);
// 输出形状: (2, 2)
```

### 不同stride配置

```cpp
// 3x3最大池化，stride=1 (重叠池化)
Tensor result1 = cpu_backend->max_pool(input, 3, 1);

// 3x3最大池化，stride=2 (非重叠池化)
Tensor result2 = cpu_backend->max_pool(input, 3, 2);
```

### 全局平均池化

```cpp
// 4x4输入的全局平均池化
Tensor input = cpu_backend->ones(Shape(4, 4), DType::FP32);
Tensor result = cpu_backend->global_avg_pool(input);
// 输出形状: (1, 1)，值为所有元素的平均值
```

### 多维张量处理

```cpp
// 4D张量: (2, 3, 4, 4) - 2个批次，3个通道
Tensor input = cpu_backend->randint(Shape(2, 3, 4, 4), 1, 11, DType::FP32);
Tensor result = cpu_backend->max_pool(input, 2, 2);
// 输出形状: (2, 3, 2, 2) - 保持批次和通道维度
```

### In-place操作

```cpp
// 预分配输出张量
Shape output_shape = Shape(2, 2);
Tensor result = cpu_backend->empty(output_shape, DType::FP32);

// 直接写入预分配的张量
cpu_backend->max_pool_into(input, result, 2, 2);
```

## 数学原理

### 最大池化

对于输出位置 `(oh, ow)`，最大池化计算为：

```
output[oh, ow] = max(ih=oh*stride to oh*stride+K-1,
                     iw=ow*stride to ow*stride+K-1)
                 input[ih, iw]
```

其中超出输入边界的值被忽略。

### 全局平均池化

全局平均池化计算为：

```
output = (1/(H×W)) × Σ(ih=0 to H-1, iw=0 to W-1) input[ih, iw]
```

将整个空间维度压缩为单个平均值。

## 边界处理

### 最大池化边界情况
- **部分窗口**: 当池化窗口超出输入边界时，只计算有效区域内的元素
- **空窗口**: 理论上不会出现，因为形状计算确保了至少有一个有效位置

### 全局平均池化边界情况
- **空张量**: 通过参数验证防止空张量输入
- **零除错误**: 通过形状验证确保元素总数>0

## 错误处理

实现提供了全面的错误处理机制：

### 常见错误类型

1. **设备类型错误**: 张量不在CPU设备上
2. **内存未分配**: 张量存储未正确初始化
3. **数据类型错误**: 不支持FP32以外的数据类型
4. **维度错误**: 输入维度<2
5. **参数错误**: kernel_size≤0或stride不支持
6. **形状错误**: 输出形状无效

### 异常示例

```cpp
try {
    Tensor result = cpu_backend->max_pool(input, 0, 2); // kernel_size=0无效
} catch (const TRException& e) {
    std::cout << "池化错误: " << e.what() << std::endl;
    // 输出: [CPU MaxPool] Kernel size must be positive
}

try {
    Tensor result = cpu_backend->max_pool(input, 2, 3); // stride=3不支持
} catch (const TRException& e) {
    std::cout << "池化错误: " << e.what() << std::endl;
    // 输出: [CPU MaxPool] Only supports stride 1 or 2
}
```

## 应用场景

### 计算机视觉

1. **特征下采样**: 最大池化用于减少特征图的空间维度
2. **平移不变性**: 最大池化提供一定程度的平移不变性
3. **特征选择**: 选择局部区域中最显著的特征

### 深度学习架构

1. **CNN骨干网络**: ResNet、VGG等架构中的标准组件
2. **特征金字塔**: 多尺度特征提取中的池化操作
3. **全局特征提取**: 分类网络中的全局平均池化

### 性能优化

1. **计算效率**: 池化操作显著减少后续层的计算量
2. **内存效率**: 减少特征图的内存占用
3. **过拟合控制**: 提供正则化效果

## 测试验证

实现通过了全面的单元测试：

- **基础功能测试**: 验证池化计算的正确性
- **形状测试**: 验证不同参数组合下的输出形状
- **边界测试**: 验证kernel size和stride的边界情况
- **多维测试**: 验证2D、3D、4D张量的处理
- **错误处理测试**: 验证异常情况的处理
- **数值精度测试**: 验证FP32计算的精度

测试文件: `tests/unit_tests/test_cpu_pooling.cpp`

## 版本历史

- **V1.35.0** (2025-11-03): 初始实现，支持最大池化和全局平均池化
- 支持多种kernel size和stride配置
- 完整的参数验证和错误处理
- 详细的文档和测试覆盖

## 相关文件

- **实现文件**: `src/backend/cpu/cpu_pooling.cpp`
- **头文件**: `include/tech_renaissance/backend/cpu/cpu_backend.h`
- **测试文件**: `tests/unit_tests/test_cpu_pooling.cpp`
- **形状文档**: `docs/shape.md`
- **卷积文档**: `docs/cpu_conv.md`
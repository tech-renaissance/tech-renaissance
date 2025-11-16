# Strides Class Documentation

## 概述

`Strides`类是技术觉醒框架中用于管理张量内存布局的核心组件，描述了在多维数组中如何计算每个维度的内存偏移量。Strides是实现view操作、transpose、slice等高级张量操作的基础。

**版本**: V1.44.1
**更新日期**: 2025-11-16
**作者**: 技术觉醒团队

## 设计理念

Strides类遵循以下设计原则：

- **内存布局描述**: 精确描述张量数据在内存中的排列方式
- **高效计算**: 提供快速的线性偏移量计算
- **连续性检查**: 验证内存布局是否符合连续存储规范
- **维度支持**: 专为大4维张量（NCHW）设计，与框架保持一致

## 类结构

```cpp
class Strides {
public:
    // 构造函数
    Strides() noexcept;
    explicit Strides(const Shape& shape);
    Strides(int64_t n, int64_t c, int64_t h, int64_t w) noexcept;

    // 访问方法
    int64_t stride(int32_t dim) const;
    int64_t n() const noexcept;  // N维度步长
    int64_t c() const noexcept;  // C维度步长
    int64_t h() const noexcept;  // H维度步长
    int64_t w() const noexcept;  // W维度步长

    // 计算方法
    int64_t get_offset(int64_t n, int64_t c, int64_t h, int64_t w) const noexcept;

    // 比较操作
    bool operator==(const Strides& other) const noexcept;
    bool operator!=(const Strides& other) const noexcept;

    // 实用方法
    std::string to_string() const;
    bool is_contiguous(const Shape& shape) const;

private:
    std::array<int64_t, 4> strides_;
    void validate_dim(int32_t dim) const;
};
```

## 核心概念

### 1. 步长（Stride）的定义

步长表示在内存中从一个元素移动到下一个维度元素需要跳过的元素数量：

- **W维度步长**: 总是1（最内层维度）
- **H维度步长**: W维度的大小
- **C维度步长**: H维度大小 × W维度大小
- **N维度步长**: C维度大小 × H维度大小 × W维度大小

### 2. 连续存储（Contiguous Storage）

连续存储意味着张量数据在内存中是按照行主序（Row-major）连续排列的。对于连续张量，步长有固定的计算模式。

### 3. 线性偏移量计算

给定4D索引 `(n, c, h, w)`，线性偏移量计算公式：
```
offset = n * stride_n + c * stride_c + h * stride_h + w * stride_w
```

## 使用方法

### 基本创建

```cpp
// 从Shape自动计算连续存储的步长
Shape shape(2, 3, 4, 5);
Strides strides(shape);

// 直接指定步长值
Strides custom_strides(120, 40, 10, 2);

// 默认构造（全零步长）
Strides zero_strides;
```

### 访问步长信息

```cpp
// 通过索引访问
int64_t stride_0 = strides.stride(0);  // N维度
int64_t stride_1 = strides.stride(1);  // C维度
int64_t stride_2 = strides.stride(2);  // H维度
int64_t stride_3 = strides.stride(3);  // W维度

// 通过命名方法访问
int64_t n_stride = strides.n();
int64_t c_stride = strides.c();
int64_t h_stride = strides.h();
int64_t w_stride = strides.w();

// 获取原始数组指针
const int64_t* data = strides.data();
```

### 偏移量计算

```cpp
// 计算4D索引的线性偏移量
int64_t offset = strides.get_offset(1, 2, 3, 4);

// 手动计算验证
int64_t manual_offset = 1 * strides.n() + 2 * strides.c() +
                       3 * strides.h() + 4 * strides.w();
assert(offset == manual_offset);
```

### 连续性检查

```cpp
Shape shape(2, 3, 4, 5);
Strides contiguous_strides(shape);

bool is_contiguous = contiguous_strides.is_contiguous(shape);  // true

// 非连续stride示例
Strides non_contiguous_strides(120, 1, 30, 5);
bool is_contiguous_2 = non_contiguous_strides.is_contiguous(shape);  // false
```

## 在View操作中的应用

Strides是view操作的核心组件：

```cpp
// 创建原始张量（连续存储）
Tensor original = cpu_backend->zeros(Shape(2, 3, 4), DType::FP32);

// 创建view - 自动计算新步长
Tensor view_tensor = cpu_backend->view(original, Shape(6, 4));

// 检查步长
const Strides& original_strides = original.strides();
const Strides& view_strides = view_tensor.strides();

std::cout << "Original strides: " << original_strides.to_string() << std::endl;
std::cout << "View strides: " << view_strides.to_string() << std::endl;
```

输出示例：
```
Original strides: Strides(12, 4, 1, 1)
View strides: Strides(4, 1, 0, 0)
```

## 内存布局示例

### 2D张量示例

```cpp
// 创建2x3的张量
Shape shape(1, 1, 2, 3);  // N=1, C=1, H=2, W=3
Strides strides(shape);

// 步长计算
// stride_w = 1
// stride_h = 3 (W维度大小)
// stride_c = 6 (H × W = 2 × 3)
// stride_n = 6 (C × H × W = 1 × 2 × 3)
```

内存布局（索引→线性偏移）：
```
(0,0,0,0) → 0    (0,0,0,1) → 1    (0,0,0,2) → 2
(0,0,1,0) → 3    (0,0,1,1) → 4    (0,0,1,2) → 5
```

### 4D张量示例

```cpp
// 创建2x3x4x5的张量
Shape shape(2, 3, 4, 5);
Strides strides(shape);

// 步长计算
// stride_w = 1
// stride_h = 5
// stride_c = 20 (5 × 4)
// stride_n = 60 (20 × 3)
```

## 性能特性

### 1. 内存效率

- **固定大小**: Strides对象只包含4个int64_t值，内存占用极小
- **栈分配**: 通常在栈上分配，无堆内存开销
- **快速访问**: 所有操作都是O(1)时间复杂度

### 2. 计算效率

```cpp
// 高效的偏移量计算
inline int64_t get_offset(int64_t n, int64_t c, int64_t h, int64_t w) const noexcept {
    return n * strides_[0] + c * strides_[1] + h * strides_[2] + w * strides_[3];
}
```

## 错误处理

### 维度索引检查

```cpp
try {
    int64_t stride = strides.stride(5);  // 超出范围
} catch (const std::out_of_range& e) {
    std::cout << "Error: " << e.what() << std::endl;
    // 输出: Error: [Strides::stride] Dimension index out of range: 5
}
```

### 边界条件

```cpp
// 空张量的stride计算
Shape empty_shape;
Strides empty_strides(empty_shape);
// 所有stride都为0

// 标量张量的stride计算
Shape scalar_shape;
Strides scalar_strides(scalar_shape);
// 所有stride都为1（但实践中会被特殊处理）
```

## 与其他框架的对比

| 特性 | Tech Renaissance | PyTorch | NumPy |
|------|-----------------|---------|-------|
| 维度限制 | 最大4维 | 无限制 | 无限制 |
| 步长计算 | 固定NCHW顺序 | 用户指定 | 自动计算 |
| 内存布局 | 行主序 | 行主序 | 行主序 |
| 视图支持 | 基于Strides | 原生支持 | 原生支持 |
| 性能 | 高度优化 | 优化 | 优化 |

## 最佳实践

### 1. 使用自动计算

```cpp
// 推荐：让框架自动计算步长
Strides strides(shape);

// 不推荐：手动计算步长（容易出错）
Strides manual_strides(shape.c() * shape.h() * shape.w(),
                        shape.h() * shape.w(),
                        shape.w(),
                        1);
```

### 2. 验证连续性

```cpp
// 在需要连续内存时验证
Strides strides(shape);
if (!strides.is_contiguous(shape)) {
    // 处理非连续情况
    Tensor contiguous_tensor = cpu_backend->contiguous(tensor);
}
```

### 3. 调试和诊断

```cpp
// 使用字符串表示进行调试
std::cout << "Tensor strides: " << tensor.strides().to_string() << std::endl;
std::cout << "Tensor is contiguous: " << tensor.is_contiguous() << std::endl;
```

## 扩展性

Strides类设计为可扩展的，为未来的功能预留了空间：

### 1. 支持更多维度

```cpp
// 未来可扩展支持更多维度
template<size_t N>
class StridesN {
    std::array<int64_t, N> strides_;
    // ...
};
```

### 2. 支持不同内存布局

```cpp
enum class MemoryLayout {
    RowMajor,    // 当前行主序
    ColumnMajor, // 未来可能的列主序
    Custom       // 自定义布局
};
```

## 相关文档

- [Tensor Class](tensor.md) - 张量类文档
- [View Operations](about_view.md) - 视图操作指南
- [CPU Backend](cpu_backend.md) - CPU后端实现
- [CUDA Backend](cuda_backend.md) - CUDA后端实现

## 实现文件

- **头文件**: `include/tech_renaissance/data/strides.h`
- **实现文件**: `src/data/strides.cpp`
- **测试文件**: `tests/unit_tests/test_view.cpp` (包含strides测试)

---

**注意**: Strides类是张量系统的重要组成部分，正确理解和使用strides对于开发高效的深度学习应用至关重要。
# Shape API 文档

## 概述

`Shape`类表示技术觉醒框架中张量的维度结构。它提供了类型安全、不可变的方式来处理多维数组，支持0D到4D张量，遵循深度学习框架常用的右对齐存储约定。

**版本**: V1.23.1
**更新日期**: 2025-10-30
**作者**: 技术觉醒团队

## 设计理念

### 核心设计原则

1. **类型安全**：使用专门的类而不是原始数组防止维度计数错误，并提供边界检查。
2. **不变性**：一旦创建，形状就不能修改。这防止了计算过程中形状变化的细微错误，并确保线程安全。
3. **右对齐存储**：遵循深度学习约定，维度右对齐（[N,C,H,W]），这简化了广播和维度操作。
4. **全面验证**：所有构造函数都立即验证输入参数，在开发周期的早期捕获错误。
5. **性能优化**：紧凑的内存布局和高效的操作，适用于高频张量操作。

## 类结构

```cpp
class Shape {
    int32_t dims_[4] = {0, 0, 0, 0};  // 右对齐维度存储，标量为(0,0,0,0)
    int32_t ndim_ = 0;                // 实际维度数量（非零维度计数）
    int64_t numel_ = 0;               // 元素总数
};
```

**重要设计说明**：Shape内部使用右对齐存储，标量张量存储为(0,0,0,0)而不是(1,1,1,1)，这确保了ndim和numel的正确计算。

## 核心API

### 构造函数

#### `Shape()`
创建标量形状（0D张量），内部存储为(0,0,0,0)。

#### `explicit Shape(int32_t dim0)`
创建指定大小的1D形状，内部存储为(0,0,0,dim0)。

#### `Shape(int32_t dim0, int32_t dim1)`
创建2D形状（矩阵），内部存储为(0,0,dim0,dim1)。

#### `Shape(int32_t dim0, int32_t dim1, int32_t dim2)`
创建3D形状，内部存储为(0,dim0,dim1,dim2)。

#### `Shape(int32_t n, int32_t c, int32_t h, int32_t w)`
创建遵循NCHW约定的4D形状，内部存储为(n,c,h,w)。

#### `explicit Shape(std::initializer_list<int32_t> dims)`
从初始化列表创建形状，支持1-4个参数。

### 维度访问

#### `int32_t ndim() const noexcept`
返回维度数量（非零维度的数量，0-4）。

#### `int64_t numel() const noexcept`
返回元素总数（标量返回1）。

#### `int32_t dim(int32_t dim) const`
返回指定维度的大小，dim为0-based相对于实际维度。

#### NCHW便利方法
- `int32_t n() const noexcept` - 批次维度
- `int32_t c() const noexcept` - 通道维度
- `int32_t h() const noexcept` - 高度维度（矩阵行数）
- `int32_t w() const noexcept` - 宽度维度（矩阵列数）

#### 新增别名方法（V1.23.1）
为了提高代码可读性，特别是矩阵操作：
- `int32_t number() const` - n()的别名
- `int32_t channel() const` - c()的别名
- `int32_t height() const` - h()的别名，用于2D矩阵的行数
- `int32_t width() const` - w()的别名，用于2D矩阵的列数

**设计理由**：这些别名方法使代码意图更加明确，特别是在矩阵乘法等操作中。

### 比较操作

#### `bool operator==(const Shape& other) const noexcept`
#### `bool operator!=(const Shape& other) const noexcept`

### 特殊方法

#### `bool is_scalar() const noexcept`
检查形状是否表示标量（所有维度为0）。

#### `bool is_matmul_compatible(const Shape& other) const`
检查两个形状是否兼容矩阵乘法：`A[M,K] × B[K,N] = C[M,N]`。

#### `bool is_broadcastable_to(const Shape& target) const`
检查当前形状是否可以通过广播规则扩展到目标形状。

### 字符串表示

#### `std::string to_string() const`
返回紧凑的字符串表示，如"(4,3,2)"或"()"。

## 内存布局和后端管理

### 关键设计原则（V1.23.1）

#### **后端管理存储原则**
每个后端负责管理自己的张量存储格式：
- **CPU后端**：使用行主序（Row-major）存储
- **CUDA后端**：使用列主序（Column-major）存储

#### **转换层处理格式转换**
`to_cpu()` 和 `from_cpu()` 方法负责处理不同后端之间的内存布局转换：

```cpp
// CPU -> CUDA：行主序 → 列主序转换
Tensor CudaBackend::from_cpu(const Tensor& tensor) {
    // 对于2D矩阵，执行行主序到列主序的转置
    if (tensor.shape().ndim() == 2) {
        int32_t M = tensor.shape().height();  // 行数
        int32_t N = tensor.shape().width();   // 列数

        // 创建列主序存储的CUDA张量
        Tensor cuda_tensor = Tensor::empty(tensor.shape(), tensor.dtype, tr::CUDA[device_id_]);

        // 转置：行主序 → 列主序
        for (int32_t i = 0; i < M; ++i) {
            for (int32_t j = 0; j < N; ++j) {
                cuda_data[j * M + i] = cpu_data[i * N + j];
            }
        }
        return cuda_tensor;
    }
}

// CUDA -> CPU：列主序 → 行主序转换
Tensor CudaBackend::to_cpu(const Tensor& tensor) {
    // 对于2D矩阵，执行列主序到行主序的转置
    if (tensor.shape().ndim() == 2) {
        int32_t M = tensor.shape().height();  // 行数
        int32_t N = tensor.shape().width();   // 列数

        // 创建行主序存储的CPU张量
        Tensor cpu_tensor = Tensor::empty(tensor.shape(), tensor.dtype, tr::CPU);

        // 转置：列主序 → 行主序
        for (int32_t i = 0; i < M; ++i) {
            for (int32_t j = 0; j < N; ++j) {
                cpu_data[i * N + j] = cuda_data[j * M + i];
            }
        }
        return cpu_tensor;
    }
}
```

#### **数据访问一致性**
- 用户视角：所有张量都是行主序访问，无论在哪个后端
- 后端内部：各自选择最优的内存布局
- 转换透明：转换层自动处理格式转换，用户无需关心

## 使用示例

### 基础形状创建
```cpp
// 标量
Shape scalar;                          // ()

// 1D向量
Shape vector(100);                     // (100)

// 2D矩阵
Shape matrix(50, 100);                 // (50, 100)

// 3D张量
Shape volume(10, 3, 32);               // (10, 3, 32)

// 4D图像张量（NCHW）
Shape image(1, 3, 224, 224);           // (1, 3, 224, 224)

// 使用初始化列表
Shape flexible({2, 3, 4, 5});          // (2, 3, 4, 5)
```

### 矩阵操作（特别强调V1.23.1改进）
```cpp
// 矩阵乘法设置
Shape weights(256, 128);     // [256, 128]
Shape input(128, 64);        // [128, 64]

// 使用新的别名方法提高可读性
int32_t weight_rows = weights.height();    // 256
int32_t weight_cols = weights.width();     // 128
int32_t input_rows = input.height();       // 128
int32_t input_cols = input.width();        // 64

if (weights.is_matmul_compatible(input)) {
    // 可以相乘：weights * input -> [256, 64]
    std::cout << "形状兼容矩阵乘法" << std::endl;
}
```

### 跨后端张量操作（V1.23.1关键特性）
```cpp
// 创建CPU张量（行主序）
Tensor cpu_a = Tensor::randn(Shape(1024, 2048), 42);

// 转换到CUDA（自动转换为列主序）
Tensor cuda_a = cuda_backend->from_cpu(cpu_a);  // 行主序 → 列主序

// CUDA矩阵乘法（列主序存储）
Tensor cuda_result = cuda_backend->mm(cuda_a, cuda_b);

// 转换回CPU（自动转换回行主序）
Tensor cpu_result = cuda_backend->to_cpu(cuda_result);  // 列主序 → 行主序

// 结果一致性验证：CPU和CUDA结果在行主序下应该一致
bool is_close = cpu_backend->is_close(cpu_a_result, cpu_b_result, 1e-4f);
```

## 性能特征

- **内存高效**：每个Shape对象仅24字节（4个int32_t + 2个元数据）
- **缓存友好**：紧凑布局支持高效的复制和比较
- **无堆分配**：栈分配，无动态内存开销
- **快速操作**：简单算术操作，适用于高频调用

## 线程安全

Shape类由于其不变设计而完全线程安全。所有操作要么是只读的，要么创建新的Shape对象，无需同步开销即可消除竞争条件。

## 框架集成

### 与后端系统的集成（V1.23.1）

Shape类在后端转换中发挥关键作用：
- **维度信息传递**：Shape在CPU和CUDA之间传递维度信息
- **内存布局指导**：Shape指导转换层如何执行格式转换
- **操作兼容性检查**：确保后端操作在正确的维度上执行

### 多后端一致性
- 用户代码始终看到一致的行主序数据访问
- 各后端内部使用最优的内存布局
- 转换层确保数据格式转换的正确性

## 未来扩展

### 形状推断
未来方法可以支持计算图的自动形状推断。

### 动态形状
支持框架未来版本中的符号或动态形状，用于可变长度输入。

### 更高维度支持
内部4元素数组可以扩展以支持更高维度，同时保持当前API以实现向后兼容性。

---

## 版本信息

- **版本**: V1.23.1
- **更新日期**: 2025-10-30
- **作者**: 技术觉醒团队
- **主要更新**: 添加了内存布局管理、后端转换、别名方法等关键特性
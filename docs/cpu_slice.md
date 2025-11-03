# CPU后端张量切片功能文档

## 概述

CPU后端张量切片功能是Tech Renaissance框架V1.33.2版本新增的重要特性，允许用户通过Offset参数对NCHW格式的张量进行灵活的切片操作。该功能支持多维张量的精确切片、步长采样以及数据类型兼容性。

## 版本信息

- **版本号**: V1.33.2
- **发布日期**: 2025-11-03
- **作者**: 技术觉醒团队
- **所属系列**: backend

## 功能特性

### 核心方法

#### 1. `slice(const Tensor& tensor_a, const Offset& offset)`

创建一个新的张量，包含从输入张量中切片出的数据。

```cpp
Tensor CpuBackend::slice(const Tensor& tensor_a, const Offset& offset);
```

**特性**:
- 返回新的张量，不修改原始张量
- 支持任意维度的张量切片（1D到4D）
- 自动计算输出张量形状
- 支持步长采样
- 完整的参数验证

#### 2. `slice_into(const Tensor& tensor_a, Tensor& result, const Offset& offset)`

将切片结果直接写入预分配的目标张量。

```cpp
void CpuBackend::slice_into(const Tensor& tensor_a, Tensor& result, const Offset& offset);
```

**特性**:
- 原地操作，避免额外的内存分配
- 目标张量必须预分配且形状匹配
- 支持批量操作和内存优化
- 严格的数据类型一致性检查

### Offset类

Offset类定义了切片操作的参数，包含NCHW四个维度的起始位置、结束位置和步长：

```cpp
class Offset {
private:
    int32_t w_start_, w_end_, w_stride_;    // 宽度维度
    int32_t h_start_, h_end_, h_stride_;    // 高度维度
    int32_t c_start_, c_end_, c_stride_;    // 通道维度
    int32_t n_start_, n_end_, n_stride_;    // 批次维度
public:
    // 4种构造函数，支持不同维度组合
    Offset(int32_t w_start, int32_t w_end);
    Offset(int32_t h_start, int32_t h_end, int32_t w_start, int32_t w_end);
    Offset(int32_t c_start, int32_t c_end, int32_t h_start, int32_t h_end,
           int32_t w_start, int32_t w_end);
    Offset(int32_t n_start, int32_t n_end, int32_t c_start, int32_t c_end,
           int32_t h_start, int32_t h_end, int32_t w_start, int32_t w_end);
};
```

## 使用方法

### 基本切片操作

```cpp
// 获取CPU后端
auto cpu = BackendManager::get_cpu_backend();

// 创建4D张量 (N=4, C=3, H=5, W=6)
Tensor tensor4d = cpu->ones(Shape(4, 3, 5, 6), DType::FP32);

// 定义切片参数：N[1,3), C[0,2), H[1,4), W[2,5)
Offset offset4d(1, 3, 0, 2, 1, 4, 2, 5);

// 执行切片
Tensor slice4d = cpu->slice(tensor4d, offset4d);
// 结果形状：(2, 2, 3, 3)
```

### 使用-1表示到末尾

```cpp
// 切片到维度末尾
Offset offset_to_end(1, -1, 1, -1, 2, -1);  // N[1,end), C[1,end), H[2,end)
Tensor slice_to_end = cpu->slice(tensor4d, offset_to_end);
```

### 步长采样

```cpp
// 设置步长
Offset offset_stride(0, -1, 0, -1, 0, -1, 0, -1);
offset_stride.set_w_stride(2);  // W方向每隔2个采样
offset_stride.set_h_stride(2);  // H方向每隔2个采样

Tensor slice_stride = cpu->slice(tensor4d, offset_stride);
```

### slice_into操作

```cpp
// 预分配目标张量
auto result = cpu->empty(Shape(2, 2, 3, 3), DType::FP32);
Offset offset_into(0, 2, 0, 2, 1, 4, 1, 4);

// 直接写入目标张量
cpu->slice_into(tensor4d, result, offset_into);
```

### 不同维度支持

```cpp
// 1D张量切片
Tensor tensor1d = cpu->ones(Shape(10), DType::FP32);
Offset offset1d(2, 8);  // W[2,8)
Tensor slice1d = cpu->slice(tensor1d, offset1d);

// 2D张量切片
Tensor tensor2d = cpu->ones(Shape(4, 8), DType::FP32);
Offset offset2d(1, 3, 2, 6);  // H[1,3), W[2,6)
Tensor slice2d = cpu->slice(tensor2d, offset2d);

// 3D张量切片
Tensor tensor3d = cpu->ones(Shape(2, 4, 6), DType::FP32);
Offset offset3d(0, 2, 1, 3, 2, 5);  // C[0,2), H[1,3), W[2,5)
Tensor slice3d = cpu->slice(tensor3d, offset3d);
```

## 数据类型支持

切片功能支持以下数据类型：

- **FP32** (float) - 单精度浮点数
- **INT8** (int8_t) - 8位有符号整数
- **INT32** (int32_t) - 32位有符号整数

```cpp
// INT8张量切片
Tensor int8_tensor = cpu->ones(Shape(2, 3, 4), DType::INT8);
Offset offset_int8(0, 2, 1, 2, 2, 4);
Tensor slice_int8 = cpu->slice(int8_tensor, offset_int8);
```

## 错误处理

### 参数验证

```cpp
try {
    // 超出范围的切片
    Offset bad_offset(10, 15, 0, 1, 0, 1, 0, 1);  // N维度超出范围
    Tensor bad_slice = cpu->slice(tensor4d, bad_offset);
} catch (const TRException& e) {
    std::cout << "Error: " << e.what() << std::endl;
    // 输出: TRException: CpuBackend::slice: N dimension end index 15 exceeds tensor size 4
}
```

### slice_into验证

```cpp
try {
    // 形状不匹配
    auto wrong_result = cpu->empty(Shape(2, 2, 2, 2), DType::FP32);
    Offset offset_correct(0, 2, 0, 2, 1, 4, 1, 4);  // 期望输出：(2,2,3,3)
    cpu->slice_into(tensor4d, wrong_result, offset_correct);
} catch (const TRException& e) {
    std::cout << "Shape mismatch: " << e.what() << std::endl;
    // 输出: TRException: CpuBackend::slice_into: result shape mismatch. expected: (2,2,3,3), actual: (2,2,2,2)
}

try {
    // 数据类型不匹配
    auto fp32_result = cpu->empty(Shape(2, 2, 3, 3), DType::FP32);
    cpu->slice_into(int8_tensor, fp32_result, offset_int8);
} catch (const TRException& e) {
    std::cout << "Type mismatch: " << e.what() << std::endl;
    // 输出: TRException: CpuBackend::slice_into: data type mismatch. input: INT8, result: FP32
}
```

## 实现细节

### 算法复杂度

- **时间复杂度**: O(output_numel) - 线性于输出张量的元素数量
- **空间复杂度**: O(1) - slice()方法需要分配输出张量，slice_into()方法原地操作

### 内存优化

1. **模板特化**: 使用模板函数实现不同数据类型的优化处理
2. **线性索引计算**: 高效的多维到线性索引转换
3. **RAII内存管理**: 自动内存管理，避免内存泄漏
4. **原地操作**: slice_into()方法支持直接写入现有张量

### 索引计算

```cpp
// 4D张量线性索引计算示例
size_t input_idx = n * input_shape.c() * input_shape.h() * input_shape.w() +
                   c * input_shape.h() * input_shape.w() +
                   h * input_shape.w() + w;
```

## 性能建议

### 使用slice而不是多次get_item

```cpp
// 推荐：使用slice
Offset offset(0, 1, 0, 1, 0, 1, 0, 10);
Tensor slice_data = cpu->slice(tensor, offset);

// 不推荐：多次get_item
for (int i = 0; i < 10; i++) {
    float value = cpu->get_item_fp32(tensor, i);
    // 处理单个元素
}
```

### 使用slice_into进行批量操作

```cpp
// 推荐：预分配张量并使用slice_into
auto result = cpu->empty(expected_shape, DType::FP32);
cpu->slice_into(tensor, result, offset);

// 不推荐：多次slice创建新张量
for (int batch = 0; batch < num_batches; batch++) {
    Offset batch_offset(batch, batch+1, 0, -1, 0, -1, 0, -1);
    Tensor batch_slice = cpu->slice(tensor, batch_offset);
    // 处理每个批次
}
```

## 兼容性

- **后端**: 仅支持CPU后端
- **数据类型**: FP32, INT8, INT32
- **维度**: 1D到4D张量
- **框架版本**: V1.33.2+

## 示例代码

完整的使用示例请参考：
- `tests/unit_tests/test_cpu_slice.cpp` - 包含7个测试用例
- 涵盖基本切片、步长采样、错误处理等所有功能

## 相关文档

- [Shape类文档](shape.md) - 了解张量形状操作
- [Offset类文档](offset.md) - 了解切片参数定义
- [CPU后端文档](cpu_backend.md) - 了解其他CPU操作
- [张量操作文档](tensor_operations.md) - 了解其他张量操作

---

*本文档最后更新时间: 2025-11-03*
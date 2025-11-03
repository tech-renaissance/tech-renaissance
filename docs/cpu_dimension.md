# CPU后端张量维度操作功能文档

## 概述

CPU后端张量维度操作功能是Tech Renaissance框架V1.34.1版本新增的重要特性，提供了丰富的张量维度变换和reduction操作。该功能支持多种数据类型、负数维度索引，以及与PyTorch兼容的API设计。

## 版本信息

- **版本号**: V1.34.1
- **发布日期**: 2025-11-03
- **作者**: 技术觉醒团队
- **所属系列**: backend

## 功能特性

### 维度变换操作

#### 1. Unsqueeze操作
在指定位置插入大小为1的维度。

```cpp
Tensor unsqueeze(const Tensor& tensor_a, int32_t dim) const;
void unsqueeze_inplace(Tensor& tensor_a, int32_t dim) const;
void unsqueeze_into(const Tensor& tensor_a, Tensor& tensor_b) const;
```

**特性**:
- 支持负数维度索引
- 自动处理维度范围验证
- 保持原有数据不变，仅改变形状

#### 2. Squeeze操作
移除大小为1的指定维度。

```cpp
Tensor squeeze(const Tensor& tensor_a, int32_t dim) const;
void squeeze_inplace(Tensor& tensor_a, int32_t dim) const;
void squeeze_into(const Tensor& tensor_a, Tensor& tensor_b) const;
```

**特性**:
- 验证指定维度大小必须为1
- 支持多维度张量压缩
- 自动调整数据布局

#### 3. Pad操作
在张量的H和W维度周围补零。

```cpp
Tensor pad(const Tensor& tensor_a, int32_t padding) const;
void pad_into(const Tensor& tensor_a, int32_t padding, Tensor& tensor_b) const;
```

**特性**:
- 支持2D、3D、4D张量
- 只在高度和宽度维度补零
- 支持多种数据类型

### Reduction操作（V1.34.1新增）

#### 1. Softmax操作
沿指定维度计算softmax值，提供数值稳定性保证。

```cpp
Tensor softmax(const Tensor& tensor_a, int32_t dim);
void softmax_inplace(Tensor& tensor_a, int32_t dim);
void softmax_into(const Tensor& tensor_a, Tensor& result, int32_t dim);
```

**特性**:
- **数据类型**: 仅支持FP32
- **数值稳定性**: 使用减最大值的方法避免指数溢出
- **API兼容**: 与PyTorch的softmax行为一致
- **维度支持**: 支持1D到4D张量
- **负数索引**: 支持`dim=-1`等负数维度索引

**数值稳定性实现**:
```cpp
// 对每个切片先减去最大值，再计算softmax
float max_val = input_data[slice_start];
for (int32_t i = 1; i < slice_size; ++i) {
    max_val = std::max(max_val, input_data[slice_start + i * dim_stride]);
}

// 计算 exp(x - max_val) 并归一化
float sum = 0.0f;
for (int32_t i = 0; i < slice_size; ++i) {
    float exp_val = std::exp(input_data[idx] - max_val);
    sum += exp_val;
    result_data[idx] = exp_val / sum;
}
```

#### 2. Max操作
沿指定维度找最大值，支持keep_dim参数控制输出形状。

```cpp
Tensor max(const Tensor& tensor_a, int32_t dim, bool keep_dim = false);
void max_into(const Tensor& tensor_a, Tensor& result, int32_t dim, bool keep_dim = false);
```

**特性**:
- **数据类型**: 支持FP32、INT8、INT32
- **keep_dim参数**:
  - `true`: 保留reduction维度，大小设为1
  - `false`: 移除reduction维度（默认）
- **输出类型**: 与输入张量相同的数据类型

**形状变换示例**:
```cpp
// 输入: (2, 3, 4), dim=1
Tensor result = cpu->max(input, 1, false);  // 输出: (2, 4)
Tensor result = cpu->max(input, 1, true);   // 输出: (2, 1, 4)
```

#### 3. Sum操作
沿指定维度求和，支持keep_dim参数控制输出形状。

```cpp
Tensor sum(const Tensor& tensor_a, int32_t dim, bool keep_dim = false);
void sum_into(const Tensor& tensor_a, Tensor& result, int32_t dim, bool keep_dim = false);
```

**特性**:
- **数据类型**: 仅支持FP32
- **keep_dim参数**: 同max操作
- **输出类型**: 总是FP32类型
- **累加精度**: 不考虑溢出问题

#### 4. ArgMax操作
沿指定维度寻找最大值所在的索引，返回第一个出现的最大值位置。

```cpp
Tensor argmax(const Tensor& tensor_a, int32_t dim, bool keep_dim = false);
void argmax_into(const Tensor& tensor_a, Tensor& result, int32_t dim, bool keep_dim = false);
```

**特性**:
- **数据类型**: 输入支持FP32、INT8、INT32
- **输出类型**: 总是INT32类型
- **索引规则**: 返回指定维度内的相对索引（0-based）
- **重复值处理**: 返回第一个出现的最大值索引
- **keep_dim参数**: 同max操作

## 核心算法实现

### 维度索引规范化

支持负数维度索引，与PyTorch完全兼容：

```cpp
static int32_t normalize_dim(int32_t dim, int32_t ndim) {
    if (dim < 0) {
        dim += ndim;  // 负数索引转换为正数
    }
    if (dim < 0 || dim >= ndim) {
        throw TRException("[CPU Dim] Dimension index out of range");
    }
    return dim;
}
```

### 输出形状计算

统一的reduction形状计算逻辑：

```cpp
static Shape calculate_reduction_shape(const Shape& input_shape, int32_t dim, bool keep_dim) {
    int32_t ndim = input_shape.ndim();
    std::vector<int32_t> output_dims;

    for (int32_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keep_dim) {
                output_dims.push_back(1);  // 保留维度，大小为1
            }
            // keep_dim=false时跳过该维度
        } else {
            output_dims.push_back(input_shape.dim(i));
        }
    }

    return create_shape_from_dims(output_dims);
}
```

### 索引映射算法

针对keep_dim参数的优化索引映射：

```cpp
if (keep_dim) {
    // keep_dim=true: 输出维度数与输入相同
    for (int32_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            coord = 0;  // reduction维度坐标总是0
        } else {
            coord = temp_out_idx / result_strides[i];
            temp_out_idx %= result_strides[i];
        }
        input_idx += coord * input_strides[i];
    }
} else {
    // keep_dim=false: 输出维度数比输入少1
    for (int32_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            continue;  // 跳过reduction维度
        }
        int32_t result_dim = i;
        if (i > dim) {
            result_dim = i - 1;  // 调整输出维度索引
        }
        coord = temp_out_idx / result_strides[result_dim];
        temp_out_idx %= result_strides[result_dim];
        input_idx += coord * input_strides[i];
    }
}
```

## 参数验证机制

### 统一验证框架

```cpp
static void validate_dim_operation(const Tensor& tensor, int32_t dim, const std::string& operation_name) {
    // 设备验证
    if (tensor.device() != tr::CPU) {
        throw TRException("[CPU " + operation_name + "] Device must be CPU");
    }

    // 内存验证
    if (!tensor.storage_allocated()) {
        throw TRException("[CPU " + operation_name + "] Tensor storage not allocated");
    }

    // 标量验证
    if (tensor.shape().ndim() == 0) {
        throw TRException("[CPU " + operation_name + "] Cannot perform operation on scalar tensor");
    }

    // 维度索引验证
    dim = normalize_dim(dim, tensor.shape().ndim());

    // 零维度验证
    if (tensor.shape().dim(dim) == 0) {
        throw TRException("[CPU " + operation_name + "] Cannot perform operation on dimension with size 0");
    }
}
```

### 数据类型验证

每个操作都有相应的数据类型限制：

```cpp
// Softmax示例
if (tensor_a.dtype() != DType::FP32) {
    throw TRException("[CPU Softmax] Only supports FP32 tensor");
}

// Max/ArgMax示例 - 支持多种类型
if (tensor_a.dtype() == DType::FP32) {
    max_operation_core<float>(tensor_a, result, dim, keep_dim);
} else if (tensor_a.dtype() == DType::INT8) {
    max_operation_core<int8_t>(tensor_a, result, dim, keep_dim);
} else if (tensor_a.dtype() == DType::INT32) {
    max_operation_core<int32_t>(tensor_a, result, dim, keep_dim);
}
```

## 使用示例

### 基本Reduction操作

```cpp
auto cpu = BackendManager::get_cpu_backend();

// 创建测试张量
Tensor tensor = cpu->randint(Shape(2, 3, 4), 0, 10, DType::FP32);

// Softmax操作
Tensor softmax_result = cpu->softmax(tensor, 1);  // 沿维度1
Tensor softmax_neg = cpu->softmax(tensor, -1); // 负数索引

// Max操作
Tensor max_false = cpu->max(tensor, 1, false);  // (2, 4)
Tensor max_true = cpu->max(tensor, 1, true);   // (2, 1, 4)

// Sum操作
Tensor sum_false = cpu->sum(tensor, 2, false);  // (2, 3)
Tensor sum_true = cpu->sum(tensor, 2, true);   // (2, 3, 1)

// ArgMax操作
Tensor argmax_false = cpu->argmax(tensor, 1, false);  // (2, 4) - INT32类型
Tensor argmax_true = cpu->argmax(tensor, 1, true);   // (2, 1, 4) - INT32类型
```

### _into操作示例

```cpp
// 预分配输出张量
Tensor result = cpu->empty(expected_shape, DType::FP32);

// 使用_into操作避免额外内存分配
cpu->softmax_into(tensor, result, 1);
cpu->max_into(tensor, result, 1, true);
cpu->sum_into(tensor, result, 2, false);
cpu->argmax_into(tensor, result, 1, false);  // result必须为INT32类型
```

### 多数据类型支持

```cpp
// FP32张量
Tensor fp32_tensor = cpu->randint(Shape(2, 3), 0, 10, DType::FP32);
Tensor fp32_max = cpu->max(fp32_tensor, 1);  // FP32输出

// INT8张量
Tensor int8_tensor = cpu->randint(Shape(2, 3), -10, 10, DType::INT8);
Tensor int8_max = cpu->max(int8_tensor, 1);  // INT8输出
Tensor int8_argmax = cpu->argmax(int8_tensor, 1);  // INT32输出
```

## 性能特性

### 时间复杂度

- **Softmax**: O(N × D) - N为元素总数，D为reduction维度大小
- **Max/ArgMax**: O(N × D) - 需要遍历每个reduction组
- **Sum**: O(N × D) - 线性累加操作

### 空间复杂度

- **非原地操作**: O(N) - 需要分配输出张量
- **原地操作**: O(1) - 在原张量上操作
- **_into操作**: O(1) - 写入预分配张量

### 内存优化

1. **模板特化**: 针对不同数据类型优化实现
2. **原地操作**: 减少内存分配开销
3. **步长计算**: 高效的多维索引映射
4. **数值稳定性**: Softmax避免指数溢出

## 错误处理

### 常见错误类型

1. **维度索引超出范围**
   ```cpp
   // 错误示例
   Tensor result = cpu->max(tensor, 10);  // 超出张量维度范围

   // 异常信息
   // TRException: [CPU Dim] Dimension index out of range
   ```

2. **标量张量操作**
   ```cpp
   // 错误示例
   Tensor scalar = cpu->ones(Shape(), DType::FP32);
   Tensor result = cpu->max(scalar, 0);

   // 异常信息
   // TRException: [CPU Max] Cannot perform operation on scalar tensor
   ```

3. **零维度操作**
   ```cpp
   // 错误示例
   Tensor zero_dim = cpu->ones(Shape(2, 0, 3), DType::FP32);
   Tensor result = cpu->max(zero_dim, 1);

   // 异常信息
   // TRException: [CPU Max] Cannot perform operation on dimension with size 0
   ```

4. **数据类型不匹配**
   ```cpp
   // 错误示例
   Tensor fp32_tensor = cpu->randint(Shape(2, 3), 0, 10, DType::FP32);
   cpu->softmax(fp32_tensor, 1);  // OK
   Tensor int8_tensor = cpu->randint(Shape(2, 3), 0, 10, DType::INT8);
   cpu->softmax(int8_tensor, 1);  // 错误！

   // 异常信息
   // TRException: [CPU Softmax] Only supports FP32 tensor
   ```

## 兼容性

### PyTorch兼容性

所有reduction操作都与PyTorch的对应函数保持行为一致：

| PyTorch函数 | CPU后端函数 | 行为一致性 |
|-------------|-------------|-----------|
| `torch.softmax(x, dim)` | `cpu->softmax(x, dim)` | ✅ 完全一致 |
| `torch.max(x, dim)` | `cpu->max(x, dim)` | ✅ 完全一致 |
| `torch.sum(x, dim)` | `cpu->sum(x, dim)` | ✅ 完全一致 |
| `torch.argmax(x, dim)` | `cpu->argmax(x, dim)` | ✅ 完全一致 |

### 数据类型支持

| 操作 | FP32 | INT8 | INT32 |
|------|------|------|-------|
| Softmax | ✅ | ❌ | ❌ |
| Max | ✅ | ✅ | ✅ |
| Sum | ✅ | ❌ | ❌ |
| ArgMax | ✅ | ✅ | ✅ |
| Unsqueeze | ✅ | ✅ | ✅ |
| Squeeze | ✅ | ✅ | ✅ |
| Pad | ✅ | ✅ | ✅ |

## 最佳实践

### 1. 选择合适的操作类型

```cpp
// 推荐：使用_into操作避免额外内存分配
auto result = cpu->empty(expected_shape, DType::FP32);
cpu->max_into(tensor, result, 1, keep_dim);

// 避免重复内存分配
// Tensor result = cpu->max(tensor, 1, keep_dim);  // 每次都分配新内存
```

### 2. 利用负数索引提高可读性

```cpp
// 推荐：使用负数索引
Tensor result = cpu->softmax(tensor, -1);  // 最后一维
Tensor result = cpu->max(tensor, -2);     // 倒数第二维

// 不推荐：硬编码正数索引
Tensor result = cpu->softmax(tensor, tensor.shape().ndim() - 1);
```

### 3. 注意数据类型转换

```cpp
// INT8张量进行ArgMax时输出为INT32
Tensor int8_tensor = cpu->randint(Shape(2, 3), 0, 10, DType::INT8);
Tensor indices = cpu->argmax(int8_tensor, 1);  // 输出为INT32类型

// Softmax只能用于FP32张量
Tensor fp32_tensor = cpu->cast(int8_tensor, DType::FP32);
Tensor softmax_result = cpu->softmax(fp32_tensor, 1);
```

### 4. 批量操作优化

```cpp
// 对多个张量进行相同操作时，重用内存
Tensor buffer = cpu->empty(max_shape, DType::FP32);
for (auto& tensor : tensor_list) {
    cpu->softmax_into(tensor, buffer, 1);
    // 处理buffer中的结果
}
```

## 测试验证

框架提供了完整的测试套件：

```bash
# 构建测试
cd build && cmake --build . --config Release --target test_cpu_dim_ops

# 运行测试
./bin/tests/Release/test_cpu_dim_ops.exe
```

测试覆盖：
- ✅ 基本功能测试
- ✅ keep_dim参数验证
- ✅ 负数维度索引
- ✅ 多数据类型支持
- ✅ _into和_inplace操作
- ✅ 错误处理验证
- ✅ 数值稳定性验证

## 相关文档

- [CPU后端文档](cpu_backend.md) - 了解其他CPU操作
- [张量操作文档](tensor_operations.md) - 了解张量基础操作
- [形状类文档](shape.md) - 了解张量形状操作
- [数据类型文档](dtype.md) - 了解支持的数据类型

---

*本文档最后更新时间: 2025-11-03*
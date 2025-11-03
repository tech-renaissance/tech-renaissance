# Offset 类文档

## 概述

`Offset` 类是技术觉醒框架中用于定义张量在各个维度上的偏移量范围和步长的数据结构。该类提供了灵活的张量切片和索引功能，支持 NCHW 四个维度的独立配置。

## 版本信息
- **版本**: 1.00.00
- **日期**: 2025-11-03
- **作者**: 技术觉醒团队
- **所属系列**: data

## 类结构

### 私有成员变量

`Offset` 类包含 12 个私有成员变量，分别对应 NCHW 四个维度：

```cpp
// W维度参数
int32_t w_start_;  // W维度起始位置
int32_t w_end_;    // W维度结束位置 (-1表示最后一个元素)
int32_t w_stride_; // W维度步长

// H维度参数
int32_t h_start_;  // H维度起始位置
int32_t h_end_;    // H维度结束位置 (-1表示最后一个元素)
int32_t h_stride_; // H维度步长

// C维度参数
int32_t c_start_;  // C维度起始位置
int32_t c_end_;    // C维度结束位置 (-1表示最后一个元素)
int32_t c_stride_; // C维度步长

// N维度参数
int32_t n_start_;  // N维度起始位置
int32_t n_end_;    // N维度结束位置 (-1表示最后一个元素)
int32_t n_stride_; // N维度步长
```

### 构造函数

`Offset` 类提供 4 个重载构造函数，支持不同维度的初始化：

```cpp
// 1. 仅W维度
Offset(int32_t w_start, int32_t w_end);

// 2. H和W维度
Offset(int32_t h_start, int32_t h_end, int32_t w_start, int32_t w_end);

// 3. C、H和W维度
Offset(int32_t c_start, int32_t c_end, int32_t h_start, int32_t h_end,
       int32_t w_start, int32_t w_end);

// 4. 完整NCHW维度
Offset(int32_t n_start, int32_t n_end, int32_t c_start, int32_t c_end,
       int32_t h_start, int32_t h_end, int32_t w_start, int32_t w_end);
```

**默认初始化规则**：
- 未指定的维度：`start = 0`, `end = -1`
- 所有维度的步长：`stride = 1`

### 访问器方法

每个维度都提供完整的访问器：

```cpp
// W维度
int32_t w_start() const;    // 获取W起始位置
int32_t w_end() const;      // 获取W结束位置
int32_t w_stride() const;   // 获取W步长
void set_w_stride(int32_t w_stride); // 设置W步长

// H维度
int32_t h_start() const;    // 获取H起始位置
int32_t h_end() const;      // 获取H结束位置
int32_t h_stride() const;   // 获取H步长
void set_h_stride(int32_t h_stride); // 设置H步长

// C维度
int32_t c_start() const;    // 获取C起始位置
int32_t c_end() const;      // 获取C结束位置
int32_t c_stride() const;   // 获取C步长
void set_c_stride(int32_t c_stride); // 设置C步长

// N维度
int32_t n_start() const;    // 获取N起始位置
int32_t n_end() const;      // 获取N结束位置
int32_t n_stride() const;   // 获取N步长
void set_n_stride(int32_t n_stride); // 设置N步长
```

## 特殊语义

### end = -1 的含义

当 `end` 设置为 `-1` 时，表示该维度的"最后一个元素"。这是一个特殊的标记值，在实际使用时需要结合具体张量的维度信息来解析为实际的结束位置。

### 步长的作用

步长（stride）决定了在遍历维度时的间隔：
- `stride = 1`：访问每个元素
- `stride = 2`：每隔一个元素访问一次
- 以此类推

## 安全机制

### 范围验证

构造函数会验证每个维度的范围：
```cpp
bool valid_range = (end == -1 || start < end);
```

如果范围无效，会抛出 `TRException` 异常：
```
Offset: invalid range for W dimension, start=5, end=3. Expected: end == -1 OR start < end
```

### 步长验证

设置步长时会验证步长必须为正数：
```cpp
bool valid_stride = (stride > 0);
```

如果步长无效，会抛出 `TRException` 异常：
```
Offset: invalid stride for W dimension, stride=0. Expected: stride > 0
```

## 使用示例

### 基本用法

```cpp
// 1. 仅定义W维度的偏移
Offset w_offset(0, 10);           // w: [0, 10), stride=1

// 2. 定义H和W维度的偏移
Offset hw_offset(0, 5, 0, 10);    // h: [0, 5), w: [0, 10), stride=1

// 3. 定义完整张量偏移
Offset full_offset(0, 2, 0, 3, 0, 4, 0, 5);  // n:[0,2), c:[0,3), h:[0,4), w:[0,5)
```

### 使用 -1 表示到最后一个元素

```cpp
// 定义到W维度末尾的偏移
Offset to_end_w(0, -1);           // w: [0, 最后一个元素)

// 定义完整的张量切片（到各个维度的末尾）
Offset full_slice(0, -1, 0, -1, 0, -1, 0, -1);
```

### 设置步长

```cpp
Offset offset(0, 10);

// 设置每隔一个元素访问
offset.set_w_stride(2);           // w: [0, 10), stride=2

// 设置每隔三个元素访问
offset.set_h_stride(3);           // h: [0, 默认), stride=3
```

## 应用场景

### 1. 张量切片

```cpp
// 切片张量的部分区域
Offset slice(0, 1, 1, 3, 10, 20, 5, 15);  // 取第0个batch，第1-2个channel，第10-19行，第5-14列
```

### 2. 降采样

```cpp
// 使用步长实现降采样
Offset downsample(0, -1, 0, -1, 0, -1, 0, -1);
downsample.set_h_stride(2);  // 行方向降采样
downsample.set_w_stride(2);  // 列方向降采样
```

### 3. 间隔采样

```cpp
// 在特定维度进行间隔采样
Offset interval_sample(0, -1, 0, -1, 0, -1, 0, 10);
interval_sample.set_w_stride(3);  // 每隔3个宽度位置采样一次
```

## 注意事项

1. **end = -1 需要额外处理**：这个值在实际使用时需要结合具体张量的维度信息进行转换
2. **范围检查**：所有构造函数都会进行范围验证，确保 `start < end` 或 `end = -1`
3. **步长限制**：步长必须为正数，负数或零步长会抛出异常
4. **维度顺序**：构造函数参数按照 N-C-H-W 的顺序排列
5. **线程安全**：该类是纯数据结构，所有方法都是线程安全的

## 依赖关系

- **依赖**：`tech_renaissance/exception/tr_exception.h`
- **标准库**：`<cstdint>`, `<string>`
- **被依赖**：其他需要张量偏移量功能的模块

该类设计为轻量级的数据结构，不包含复杂的计算逻辑，主要用于传递张量切片和索引的参数信息。
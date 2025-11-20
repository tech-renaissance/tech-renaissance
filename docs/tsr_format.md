# TSR文件格式规范文档

**格式版本**: 1.0
**实现版本**: V1.56.0
**日期**: 2025年11月20日
**作者**: 技术觉醒团队
**文档版本**: V1.56.00

## 概述

TSR(Technical Renaissance)文件格式是技术觉醒框架的专用张量二进制存储格式，旨在提供高效、自描述、跨平台的张量数据交换机制。**V1.56.0版本已完整支持FP32、INT32、INT8三种数据类型，并提供C++和Python双重接口**。

## 🎉 V1.56.0最新更新：INT32数据类型支持

### ✨ 核心功能扩展

- **🎯 INT32数据类型**: 新增32位有符号整数张量的完整TSR支持
- **📊 存储效率**: INT32与FP32相同大小，INT8仅为1/4大小
- **🔍 完全兼容**: 100%向后兼容现有FP32和INT8 TSR文件
- **⚡ 双重接口**: C++ CpuBackend和Python tech_renaissance.py同时支持

### 实现状态
| 功能 | FP32 | INT32 | INT8 | 说明 |
|------|------|-------|------|------|
| C++ 导出 | ✅ | ✅ | ✅ | CpuBackend::export_tensor |
| C++ 导入 | ✅ | ✅ | ✅ | CpuBackend::import_tensor |
| Python 导出 | ✅ | ✅ | ✅ | export_tsr(tensor, filename) |
| Python 导入 | ✅ | ✅ | ✅ | import_tsr(filename) |
| 精确比较 | is_close | equal | equal | 数据类型适配的比较方法 |

### 设计目标

1. **高效性**: 直接内存映射，零拷贝加载
2. **自描述**: 文件头包含完整元数据，无需外部信息
3. **扩展性**: 预留版本控制和扩展字段
4. **健壮性**: 多重验证机制确保数据完整性
5. **跨平台**: 使用标准字节序和小端序存储

## 文件结构

TSR文件采用二进制格式，由固定头部和变长数据块组成：

```
TSR文件整体结构:
┌─────────────────────────────────────────────┐
│                文件头 (64字节)           │
├─────────────────────────────────────────────┤
│              张量数据 (变长)               │
└─────────────────────────────────────────────┘
```

### 1. 文件头详细结构

文件头占固定64字节，包含魔数、版本信息和完整的张量元数据：

```cpp
struct TSRHeader {
    // === 魔数与版本控制 (16字节) ===
    char magic[4];          // 魔数标识 'TSR!'
    int32_t version;        // 格式版本，当前为1 (小端序)
    int32_t header_size;    // 头部大小，固定为64 (小端序)
    int32_t reserved_1;     // 保留字段，设置为0

    // === 元数据块 (48字节) ===
    int32_t dtype;         // 数据类型枚举 (小端序)
    int32_t ndim;          // 维度数量 (0-4) (小端序)
    int32_t dims[4];       // 各维度尺寸，按NCHW顺序 (小端序)
    int64_t total_elements; // 元素总数 (小端序)
    int64_t reserved_2;     // 保留字段，设置为0
    int64_t reserved_3;     // 保留字段，设置为0
};
```

#### 魔数与版本控制 (16字节)

| 偏移 | 大小 | 字段 | 值 | 描述 |
|-------|------|------|-----|------|
| 0 | 4 | magic | 'TSR!' | 魔数标识，固定值0x54,0x53,0x52,0x21 |
| 4 | 4 | version | 1 | 文件格式版本，当前为1 |
| 8 | 4 | header_size | 64 | 头部大小，固定为64字节 |
| 12 | 4 | reserved_1 | 0 | 保留字段，必须设置为0 |

#### 元数据块 (48字节)

| 偏移 | 大小 | 字段 | 描述 |
|-------|------|------|------|
| 16 | 4 | dtype | 数据类型：1=FP32, 2=INT8, 3=INT32 |
| 20 | 4 | ndim | 维度数量：0-4 |
| 24 | 16 | dims[4] | 维度尺寸数组，按N,C,H,W顺序 |
| 40 | 8 | total_elements | 元素总数 |
| 48 | 8 | reserved_2 | 保留字段，必须设置为0 |
| 56 | 8 | reserved_3 | 保留字段，必须设置为0 |

### 2. 张量数据块

张量数据紧随文件头之后，采用连续存储格式：

```
数据块结构:
┌─────────────────────────────────────────────┐
│            张量数据 (变长)               │
│           总大小 = total_elements × element_size │
└─────────────────────────────────────────────┘
```

#### 数据存储格式

**FP32数据类型**:
- 格式: IEEE 754单精度浮点数
- 大小: 4字节/元素
- 顺序: 按NCHW连续存储

**INT8数据类型**:
- 格式: 有符号8位整数
- 大小: 1字节/元素
- 顺序: 按NCHW连续存储

**INT32数据类型**:
- 格式: 有符号32位整数
- 大小: 4字节/元素
- 顺序: 按NCHW连续存储

## 与Tensor类的属性映射

TSR文件头字段与Tensor类属性的对应关系：

```cpp
// Tensor类属性
class Tensor {
private:
    Shape shape_;           // 形状信息
    DType dtype_;          // 数据类型
    Device device_;        // 设备信息
    std::shared_ptr<Storage> storage_;  // 存储句柄
};

// 属性映射关系
TSR Header字段    →  Tensor类属性        →  描述
dtype           →  dtype_              →  数据类型枚举
ndim           →  shape_.ndim()       →  实际维度数量
dims[0]        →  shape_.batch()      →  N维度(批次)
dims[1]        →  shape_.channel()    →  C维度(通道)
dims[2]        →  shape_.height()     →  H维度(高度)
dims[3]        →  shape_.width()      →  W维度(宽度)
total_elements  →  shape_.numel()      →  元素总数
```

### 维度存储规则

TSR文件采用**右对齐存储**策略，将张量维度信息按NCHW顺序存储：

```cpp
// 不同维度张量的dims数组填充规则
int32_t dims[4];  // NCHW顺序存储

// 标量 (0维)
dims = {1, 1, 1, 1};  // 全部填充为1

// 1D张量 (向量)
Shape shape = Shape(5);  // 形状[5]
dims = {1, 1, 1, 5};  // W=5, 其余为1

// 2D张量 (矩阵)
Shape shape = Shape(3, 4);  // 形状[3,4]
dims = {1, 1, 3, 4};  // H=3, W=4, 其余为1

// 3D张量
Shape shape = Shape(2, 3, 4);  // 形状[2,3,4]
dims = {1, 2, 3, 4};  // C=2, H=3, W=4, N=1

// 4D张量 (完整NCHW)
Shape shape = Shape(2, 3, 4, 5);  // 形状[2,3,4,5]
dims = {2, 3, 4, 5};  // N=2, C=3, H=4, W=5
```

## 数据类型定义

### DType枚举对应

```cpp
enum class DType : int32_t {
    UNKNOWN = 0,  // 无效类型
    FP32 = 1,     // 32位浮点数 (IEEE 754)
    INT8 = 2,     // 8位有符号整数
    INT32 = 3     // 32位有符号整数 (V1.56.0新增)
};
```

### 数据类型特性

| 类型 | 值 | 大小 | 范围 | 用途 |
|------|-----|------|------|------|
| FP32 | 1 | 4字节 | ±3.4E±38 | 深度学习训练和推理 |
| INT8 | 2 | 1字节 | -128 到 +127 | 量化推理 |
| INT32 | 3 | 4字节 | -2³¹ 到 2³¹-1 | 标签存储和整数运算 |

## 实现指南

### 读取TSR文件

```cpp
// 基本读取流程
Tensor load_tsr_file(const std::string& filename) {
    // 1. 打开文件并验证大小
    std::ifstream file(filename, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();

    // 2. 读取头部
    TSRHeader header;
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(&header), sizeof(TSRHeader));

    // 3. 验证魔数和版本
    if (std::memcmp(header.magic, "TSR!", 4) != 0) {
        throw std::runtime_error("Invalid TSR file magic number");
    }
    if (header.version != 1) {
        throw std::runtime_error("Unsupported TSR version");
    }

    // 4. 重建Tensor形状
    Shape shape;
    if (header.ndim == 0) {
        shape = Shape();  // 标量
    } else if (header.ndim == 1) {
        shape = Shape(header.dims[3]);  // 1D
    } else if (header.ndim == 2) {
        shape = Shape(header.dims[2], header.dims[3]);  // 2D
    } else if (header.ndim == 3) {
        shape = Shape(header.dims[1], header.dims[2], header.dims[3]);  // 3D
    } else {  // 4D
        shape = Shape(header.dims[0], header.dims[1],
                    header.dims[2], header.dims[3]);
    }

    // 5. 创建Tensor并读取数据
    DType dtype = static_cast<DType>(header.dtype);
    Tensor tensor = Tensor::empty(shape, dtype, CPU);

    size_t data_size = tensor.memory_size();
    file.read(reinterpret_cast<char*>(tensor.data_ptr()), data_size);

    return tensor;
}
```

### 写入TSR文件

```cpp
// 基本写入流程
void save_tsr_file(const Tensor& tensor, const std::string& filename) {
    // 1. 准备文件头
    TSRHeader header = {};
    std::memcpy(header.magic, "TSR!", 4);
    header.version = 1;
    header.header_size = 64;
    header.dtype = static_cast<int32_t>(tensor.dtype());
    header.ndim = tensor.ndim();

    // 2. 填充维度数组
    int32_t dims[4] = {1, 1, 1, 1};
    for (int i = 0; i < 4; i++) {
        if (i < 4 - header.ndim) {
            dims[i] = 1;  // 前导维度为1
        } else {
            dims[i] = tensor.dim_size(i - (4 - header.ndim));
        }
    }
    std::memcpy(header.dims, dims, sizeof(dims));

    header.total_elements = static_cast<int64_t>(tensor.numel());

    // 3. 写入文件
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&header), sizeof(TSRHeader));
    file.write(reinterpret_cast<const char*>(tensor.data_ptr()),
             tensor.memory_size());
}
```

## 验证与错误处理

### 文件完整性验证

1. **魔数验证**: 确保文件为TSR格式
2. **版本检查**: 验证版本兼容性
3. **大小验证**: 检查文件大小与预期数据大小一致
4. **维度验证**: 确保维度数量在0-4范围内
5. **数据类型验证**: 验证数据类型为支持的值
6. **元数据一致性**: 验证total_elements与dims数组的一致性

### 常见错误处理

```cpp
try {
    Tensor tensor = load_tsr_file("data.tsr");
} catch (const std::exception& e) {
    std::cerr << "TSR文件加载失败: " << e.what() << std::endl;
    // 处理具体错误类型
}
```

## 扩展性设计

### 版本兼容性

TSR格式设计支持向前兼容：

- **版本1**: 当前版本，支持FP32、INT32、INT8，0-4维张量
- **未来版本**: 可通过version字段识别，添加新数据类型或功能

### 保留字段

文件头中的保留字段为未来扩展预留：

- `reserved_1`: 预留魔数扩展
- `reserved_2`, `reserved_3`: 预留元数据扩展

### 扩展数据类型

当前和未来可支持的数据类型：

```cpp
enum class DType : int32_t {
    UNKNOWN = 0,  // 无效类型
    FP32 = 1,     // ✅ 当前支持：32位浮点数
    INT8 = 2,     // ✅ 当前支持：8位有符号整数
    INT32 = 3,    // ✅ 当前支持：32位有符号整数 (V1.56.0新增)
    FP16 = 4,     // 未来：16位浮点数
    BF16 = 5,     // 未来：bfloat16
    FP64 = 6,     // 未来：64位浮点数
    INT16 = 7,    // 未来：16位整数
    UINT8 = 8     // 未来：8位无符号整数
};
```

## 性能特性

### 存储效率

- **紧凑存储**: 无额外格式开销，直接二进制存储
- **内存映射**: 支持零拷贝内存映射加载
- **对齐友好**: 数据自然对齐，优化内存访问

### 加载性能

```cpp
// 内存映射加载示例 (高性能)
Tensor mmap_load_tsr(const std::string& filename) {
    // 使用内存映射避免数据拷贝
    // 实现细节取决于平台 (Windows: CreateFileMapping, Linux: mmap)
}
```

## 工具支持

### 文件信息查看

```bash
# 命令行工具查看TSR文件信息
tsr_info data.tsr
# 输出:
# File: data.tsr
# Format: TSR v1
# Type: FP32/INT32/INT8
# Shape: [2, 3, 4, 5] (N=2, C=3, H=4, W=5)
# Elements: 120
# Size: 480 bytes (FP32/INT32) / 120 bytes (INT8)
```

### 数据转换

```bash
# TSR到其他格式转换
tsr_to_npy data.tsr data.npy  # 转换为NumPy格式
tsr_to_hdf5 data.tsr data.h5  # 转换为HDF5格式
```

## 总结

TSR文件格式为技术觉醒框架提供了：

1. **完整的张量描述**: 文件头包含重建Tensor所需的所有信息
2. **高效的存储机制**: 紧凑的二进制格式，支持零拷贝加载
3. **健壮的错误处理**: 多层验证确保数据完整性
4. **良好的扩展性**: 版本控制和保留字段支持未来功能
5. **简单的实现**: 清晰的格式定义便于第三方实现

该格式特别适合深度学习应用场景，支持训练过程中检查点的保存、模型权重的存储、以及与其他框架的数据交换。

## 实际实现参考

### C++实现 (CPU后端)
```cpp
// 源文件: src/backend/cpu/cpu_backend.cpp
// 函数: CpuBackend::export_tensor() 和 CpuBackend::import_tensor()
// V1.56.0更新：新增INT32数据类型支持

// 关键实现细节
constexpr char MAGIC_NUMBER[4] = {'T', 'S', 'R', '!'};
constexpr int32_t FORMAT_VERSION = 1;
constexpr int32_t HEADER_SIZE = 64;

// 数据类型验证 (V1.56.0更新)
if (tensor.dtype() != DType::FP32 && tensor.dtype() != DType::INT8 && tensor.dtype() != DType::INT32) {
    throw TRException("Tensor export only supports FP32, INT8 and INT32 data types");
}

// 验证魔数
if (std::memcmp(header.magic, MAGIC_NUMBER, 4) != 0) {
    throw TRException("Invalid TSR file magic number. Expected 'TSR!', got: " +
                     std::string(header.magic, 4));
}

// 维度重建逻辑
Shape shape;
if (header.ndim == 0) shape = Shape();                           // 标量
else if (header.ndim == 1) shape = Shape(header.dims[3]);        // 1D: [W]
else if (header.ndim == 2) shape = Shape(header.dims[2], header.dims[3]);  // 2D: [H,W]
else if (header.ndim == 3) shape = Shape(header.dims[1], header.dims[2], header.dims[3]);  // 3D: [C,H,W]
else shape = Shape(header.dims[0], header.dims[1], header.dims[2], header.dims[3]);  // 4D: [N,C,H,W]
```

### Python实现 (PyTorch集成)
```python
# 源文件: python/module/tech_renaissance.py
# 函数: export_tsr() 和 import_tsr()
# V1.56.0更新：新增INT32数据类型支持

# 关键实现细节
def export_tsr(tensor: torch.Tensor, filename: str) -> None:
    # 数据类型检查 (V1.56.0更新)
    if tensor.dtype not in [torch.float32, torch.int8, torch.int32]:
        raise TSRError(f"Unsupported data type {tensor.dtype}, only support float32, int8 and int32")

    # 头部打包（64字节）
    header = struct.pack(
        '<4s i i i i i i i i i q q q',  # 64字节格式
        b'TSR!',      # 魔数
        1,            # 版本
        64,           # 头部大小
        0,            # reserved
        dtype_enum,   # 数据类型 (1=FP32, 2=INT8, 3=INT32)
        ndim,         # 维度数量
        nchw[0], nchw[1], nchw[2], nchw[3],  # NCHW维度
        tensor.numel(),  # 元素总数
        0, 0          # 保留字段
    )

def import_tsr(filename: str) -> torch.Tensor:
    # 头部解析
    magic, version, header_size, reserved_1, dtype_enum, ndim, \
    dim0, dim1, dim2, dim3, total_elements, reserved_2, reserved_3 = \
        struct.unpack('<4s i i i i i i i i i q q q', header_data)

    # 数据类型转换 (V1.56.0更新)
    if dtype_enum == 1: dtype = torch.float32
    elif dtype_enum == 2: dtype = torch.int8
    elif dtype_enum == 3: dtype = torch.int32
    else: raise TSRError(f"Unknown data type enum: {dtype_enum}")
```

### 测试验证

#### C++测试 (V1.56.0更新)
```cpp
// 源文件: tests/unit_tests/test_tsr_io_extended.cpp
// V1.56.0新增：完整的INT32数据类型测试

// 测试内容
- FP32 2D/4D张量导入导出
- INT32 2D/4D张量导入导出
- INT8 2D/4D张量导入导出
- 向后兼容性验证
```

#### Python测试 (V1.56.0新增)
```python
# 源文件: python/module/test_tsr_extended.py
# 测试FP32、INT32、INT8三种数据类型的完整TSR支持

# 运行测试
python test_tsr_extended.py
# 预期结果：所有测试通过
```

#### 兼容性测试 (V1.56.0新增)
```python
# 源文件: python/module/test_tsr_compatibility.py
# 验证V1.56.0与旧版本的完全兼容性

# 运行测试
python test_tsr_compatibility.py
# 预期结果：向后兼容性100%验证通过
```
// 测试用例覆盖所有维度和数据类型的组合

std::vector<std::pair<Shape, DType>> test_cases = {
    {Shape(), DType::FP32},                              // 标量FP32
    {Shape(), DType::INT8},                              // 标量INT8
    {Shape(5), DType::FP32},                             // 1D张量FP32
    {Shape(3, 4), DType::FP32},                          // 2D张量FP32
    {Shape(2, 3, 4), DType::FP32},                       // 3D张量FP32
    {Shape(1, 2, 3, 4), DType::FP32},                   // 4D张量部分
    {Shape(2, 3, 4, 5), DType::FP32},                   // 完整4D张量FP32
    {Shape(2, 3), DType::INT8},                           // 2D张量INT8
    {Shape(1, 2, 3, 4), DType::INT8}                    // 4D张量INT8
};
```

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| V1.18.01 | 2025-10-28 | 更新版本号至V1.18.01，同步最新实现状态 |
| V1.16.04 | 2025-10-27 | 初始TSR格式规范，定义64字节标准头部 |

---

**注意**: 本文档对应TSR格式版本1.0，与技术觉醒框架V1.18.01保持同步。后续版本更新请参考对应版本的文档。
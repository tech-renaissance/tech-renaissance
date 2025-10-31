# CpuBackend API 文档

## 概述

`CpuBackend`是技术觉醒框架的CPU计算后端实现，继承自`Backend`基类。它提供了基于CPU的高性能张量计算能力，支持Eigen库优化和多线程并行计算，是框架的默认和基础计算后端。

**版本**: V1.27.1
**更新日期**: 2025-10-31
**作者**: 技术觉醒团队

## 设计理念

### 核心设计原则

1. **行主序存储**：CPU后端使用**行主序（Row-major）**存储张量数据，符合C/C++语言惯例
2. **高性能计算**：基于Eigen库的SIMD优化，支持多线程并行计算
3. **跨后端兼容**：通过`from_cpu()`和`to_cpu()`方法与其他后端保持数据一致性
4. **内存安全**：RAII智能指针自动内存管理，64字节对齐优化SIMD访问
5. **类型安全**：强类型设计防止数据类型错误，完善的边界检查

### 关键架构特性

#### **后端管理存储原则（V1.23.1核心特性）**

CPU后端遵循"后端管理存储"的设计原则：
- **CPU后端**：使用行主序（Row-major）存储张量数据，符合C/C++惯例
- **CUDA后端**：使用列主序（Column-major）存储张量数据，与cuBLAS库一致
- **转换层透明**：用户无需关心底层存储格式，`from_cpu()`和`to_cpu()`自动处理转换

#### **行主序存储布局**

```cpp
// CPU张量使用行主序存储
// 2D矩阵 A[M,N] = [[1, 2, 3],
//                  [4, 5, 6]]
// 内存布局：[1, 2, 3, 4, 5, 6]
// 访问方式：data[i * N + j] 获取第i行第j列元素

// 矩阵乘法：C[M,N] = A[M,K] × B[K,N]
for (int32_t i = 0; i < M; ++i) {
    for (int32_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int32_t k = 0; k < K; ++k) {
            sum += a_data[i * K + k] * b_data[k * N + j];
        }
        result_data[i * N + j] = sum;
    }
}
```

## 头文件

```cpp
#include "tech_renaissance/backend/cpu_backend.h"
```

## 主要特性

- **行主序存储**：使用行主序存储格式，符合C/C++语言惯例
- **Eigen优化**：集成Eigen库提供高性能线性代数计算和SIMD优化
- **多线程支持**：支持OpenMP并行计算，充分利用多核CPU
- **内存对齐**：64字节对齐内存分配，优化SIMD访问性能
- **跨后端转换**：提供`from_cpu()`和`to_cpu()`方法支持跨后端数据转换
- **张量IO**：独有的张量导入导出功能，支持TSR格式文件

## 核心API

### 构造函数

#### `CpuBackend()`

默认构造函数，初始化CPU后端。

**功能**：
- 初始化CPU计算环境
- 自动创建workspace目录（如果不存在）
- 记录初始化日志信息

**异常**：
- 无（workspace目录创建失败不会阻止初始化）

**示例**：
```cpp
auto cpu_backend = BackendManager::get_cpu_backend();  // V1.23.1推荐用法
```

### 跨后端转换接口（V1.23.1核心特性）

#### `Tensor from_cpu(const Tensor& tensor) const override`

将CPU张量转换为CPU张量（对于CPU后端，此操作为恒等变换）。

**参数**：
- `tensor` - CPU张量（行主序存储）

**返回值**：
- `Tensor` - 相同的CPU张量（行主序存储）

**特点**：
- CPU后端的`from_cpu()`是恒等操作
- 与CUDA后端的`from_cpu()`保持接口一致性
- 用于跨后端代码的统一处理

#### `Tensor to_cpu(const Tensor& tensor) const override`

将CPU张量转换为CPU张量（对于CPU后端，此操作为恒等变换）。

**参数**：
- `tensor` - CPU张量（行主序存储）

**返回值**：
- `Tensor` - 相同的CPU张量（行主序存储）

**特点**：
- CPU后端的`to_cpu()`是恒等操作
- 与CUDA后端的`to_cpu()`保持接口一致性
- 用于跨后端代码的统一处理

#### `Tensor to(const Tensor& tensor, const Device& target_device) const override`

通用设备转换接口，支持CPU到其他设备的转换。

**参数**：
- `tensor` - 源张量
- `target_device` - 目标设备

**返回值**：
- `Tensor` - 目标设备上的张量

**注意**：CPU后端不能直接转换到其他设备，需要通过BackendManager获取目标后端。

## 继承的接口实现

### 内存管理

#### `std::shared_ptr<void> allocate(size_t size) override`

在CPU内存中分配指定大小的内存块。

**参数：**
- `size` - 要分配的内存大小（字节）

**返回值：**
- `std::shared_ptr<void>` - CPU内存块的智能指针，自动管理生命周期

**异常：**
- `TRException` - 当size为0或分配失败时抛出

**实现特点：**
- 使用64字节对齐内存分配，优化SIMD访问
- 跨平台内存分配（Windows: `_aligned_malloc`, Linux: `posix_memalign`）
- 智能指针自动管理，自定义删除器确保正确的释放方式
- 内存分配失败时抛出详细异常信息

**示例：**
```cpp
auto backend = std::make_shared<tr::CpuBackend>();
auto memory = backend->allocate(1024 * 1024);  // 分配1MB内存
float* data = static_cast<float*>(backend->get_data_ptr(memory));
```

#### `void deallocate(void* ptr) override`

释放CPU内存（通常通过智能指针自动调用）。

**参数：**
- `ptr` - 要释放的内存指针

**异常：**
- `TRException` - 当ptr为null时抛出

**注意：**
- 建议使用智能指针自动管理内存，很少需要直接调用此方法

#### `void* get_data_ptr(const std::shared_ptr<void>& holder) override`

从内存智能指针中获取原始数据指针。

**参数：**
- `holder` - 内存智能指针

**返回值：**
- `void*` - 原始数据指针

**示例：**
```cpp
auto memory = backend->allocate(1024);
float* ptr = static_cast<float*>(backend->get_data_ptr(memory));
```

#### `void copy_data(void* dst, const void* src, size_t size, const Device& dst_device, const Device& src_device) override`

在CPU内存或CPU与其他设备间复制数据。

**参数：**
- `dst` - 目标内存指针
- `src` - 源内存指针
- `size` - 复制大小（字节）
- `dst_device` - 目标设备
- `src_device` - 源设备

**异常：**
- `TRException` - 当设备类型不支持时抛出

**支持的复制方向：**
- CPU → CPU：使用`std::memcpy`
- CUDA → CPU：通过CUDA运行时API
- CPU → CUDA：通过CUDA运行时API

**示例：**
```cpp
// CPU内部复制
backend->copy_data(dst_ptr, src_ptr, size, tr::CPU, tr::CPU);
```

### 张量操作

#### `void fill(Tensor& dst, float value) override`

用浮点数值填充张量（FP32类型）。

**参数：**
- `dst` - 目标张量，必须是FP32类型
- `value` - 填充值

**异常：**
- `TRException` - 当张量数据类型不是FP32时抛出

**实现特点：**
- 使用Eigen的Map实现高效填充
- 支持任意形状的张量

**示例：**
```cpp
tr::Tensor tensor(tr::Shape(2, 3), tr::DType::FP32, tr::CPU);
backend->fill(tensor, 3.14f);  // 所有元素设为3.14
```

#### `void fill(Tensor& dst, int8_t value) override`

用8位整数值填充张量（INT8类型）。

**参数：**
- `dst` - 目标张量，必须是INT8类型
- `value` - 填充值

**异常：**
- `TRException` - 当张量数据类型不是INT8时抛出

**示例：**
```cpp
tr::Tensor tensor(tr::Shape(2, 3), tr::DType::INT8, tr::CPU);
backend->fill(tensor, 42);  // 所有元素设为42
```

#### `void add(Tensor& result, const Tensor& a, const Tensor& b) override`

执行逐元素加法运算：result = a + b。

**参数：**
- `result` - 结果张量
- `a` - 第一个操作数张量
- `b` - 第二个操作数张量

**异常：**
- `TRException` - 当张量形状、数据类型或设备不匹配时抛出

**实现特点：**
- 使用Eigen的向量化操作，性能优异
- 自动处理多维张量的扁平化

**示例：**
```cpp
tr::Tensor a(tr::Shape(2, 3), tr::DType::FP32, tr::CPU);
tr::Tensor b(tr::Shape(2, 3), tr::DType::FP32, tr::CPU);
tr::Tensor result(tr::Shape(2, 3), tr::DType::FP32, tr::CPU);

backend->fill(a, 2.0f);
backend->fill(b, 3.0f);
backend->add(result, a, b);  // 结果: [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
```

#### `void mul(Tensor& result, const Tensor& a, const Tensor& b) override`

执行逐元素乘法运算：result = a * b。

**参数**：
- `result` - 结果张量
- `a` - 第一个操作数张量
- `b` - 第二个操作数张量

**异常**：
- `TRException` - 当张量形状、数据类型或设备不匹配时抛出

**实现特点**：
- 使用Eigen的向量化乘法操作
- 支持SIMD指令集优化

**示例**：
```cpp
tr::Tensor a(tr::Shape(2, 3), tr::DType::FP32, tr::CPU);
tr::Tensor b(tr::Shape(2, 3), tr::DType::FP32, tr::CPU);
tr::Tensor result(tr::Shape(2, 3), tr::DType::FP32, tr::CPU);

backend->fill(a, 2.0f);
backend->fill(b, 3.0f);
backend->mul(result, a, b);  // 结果: [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
```

### 矩阵乘法操作

CPU后端提供高性能的矩阵乘法功能，支持行主序存储格式和Eigen优化。详细的矩阵乘法API、性能基准测试和跨后端示例请参考 [矩阵乘法 API 文档](cpu_mm_fp32.md)。

**主要功能**：
- 高性能行主序矩阵乘法：`mm(result, a, b)`
- Eigen优化和朴素实现双重支持
- 自动SIMD向量化和多线程并行计算
- 跨后端一致性保证

### 设备信息

#### `Device device() const override`

获取CPU后端的设备信息。

**返回值：**
- `Device` - 返回`tr::CPU`设备对象

**示例：**
```cpp
auto device = backend->device();
std::cout << "Device: " << device.to_string() << std::endl;  // "CPU"
```

#### `std::string name() const override`

获取CPU后端的名称。

**返回值：**
- `std::string` - 返回"CpuBackend"

**示例：**
```cpp
std::cout << "Backend name: " << backend->name() << std::endl;  // "CpuBackend"
```

## 使用示例

### 跨后端操作示例

```cpp
#include "tech_renaissance.h"
using namespace tr;

void cross_backend_example() {
    try {
        // 获取后端实例
        auto cpu_backend = BackendManager::get_cpu_backend();
        auto cuda_backend = BackendManager::get_cuda_backend();

        // 创建CPU张量
        Tensor cpu_a = Tensor::randn(Shape(1024, 2048), 42);
        Tensor cpu_b = Tensor::randn(Shape(2048, 512), 42);
        Tensor cpu_result = Tensor::empty(Shape(1024, 512), DType::FP32, tr::CPU);

        // 执行基本运算
        cpu_backend->add(cpu_result, cpu_a, cpu_b);

        // 转换到其他后端进行计算
        Tensor cuda_a = cuda_backend->from_cpu(cpu_a);  // 行主序 → 列主序
        Tensor cuda_result_cpu = cuda_backend->to_cpu(cuda_a);  // 列主序 → 行主序

        std::cout << "Cross-backend operation completed!" << std::endl;

    } catch (const TRException& e) {
        std::cerr << "CPU error: " << e.what() << std::endl;
    }
}
```

### 基本CPU张量操作

```cpp
#include "tech_renaissance.h"
using namespace tr;

void basic_cpu_operations() {
    try {
        auto cpu_backend = BackendManager::get_cpu_backend();

        // 创建CPU张量（行主序存储）
        Shape shape(3, 4);
        Tensor a(shape, DType::FP32, tr::CPU);
        Tensor b(shape, DType::FP32, tr::CPU);
        Tensor result(shape, DType::FP32, tr::CPU);

        // 填充张量
        cpu_backend->fill(a, 1.5f);
        cpu_backend->fill(b, 2.5f);

        // 执行加法
        cpu_backend->add(result, a, b);

        std::cout << "CPU computation completed successfully!" << std::endl;

    } catch (const TRException& e) {
        std::cerr << "CPU error: " << e.what() << std::endl;
    }
}
```

### 性能测试示例

```cpp
void cpu_performance_example() {
    auto cpu_backend = BackendManager::get_cpu_backend();

    // 创建测试张量
    const int32_t size = 1000;
    Shape shape(size, size);
    Tensor a(shape, DType::FP32, tr::CPU);
    Tensor b(shape, DType::FP32, tr::CPU);
    Tensor result(shape, DType::FP32, tr::CPU);

    // 生成随机数据
    a = Tensor::randn(shape, 42);
    b = Tensor::randn(shape, 123);

    // 性能测试
    const int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        cpu_backend->add(result, a, b);  // 基本运算性能测试
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_time_ms = duration.count() / 1000.0 / iterations;
    std::cout << "CPU Performance Test:" << std::endl;
    std::cout << "  Tensor size: " << size << "x" << size << std::endl;
    std::cout << "  Average time: " << avg_time_ms << " ms" << std::endl;
}
```

### 张量复制操作（V1.27.1新增）

CPU后端提供了高效的张量复制功能，支持同设备内的深拷贝操作。

#### `Tensor copy(const Tensor& tensor) const`

复制张量，返回新的张量副本。

**参数**：
- `tensor` - 源张量，必须属于CPU设备

**返回值**：
- `Tensor` - 复制后的新张量，属于CPU设备

**特性**：
- **深拷贝**：生成独立的数据副本
- **同设备**：仅在CPU设备内操作
- **高效复制**：使用std::memcpy进行内存拷贝

**示例**：
```cpp
auto cpu_backend = BackendManager::get_cpu_backend();
Tensor original = Tensor::empty(Shape(2, 3), DType::FP32, tr::CPU);
cpu_backend->fill(original, 1.23f);
Tensor copied = cpu_backend->copy(original);  // 深拷贝
```

#### `void copy_into(const Tensor& src, Tensor& dst) const`

将源张量复制到指定目标张量。

**参数**：
- `src` - 源张量，必须属于CPU设备
- `dst` - 目标张量，必须属于CPU设备

**特性**：
- **深拷贝**：将src完整复制到dst的内存中
- **CPU专用**：仅支持CPU↔CPU操作，跨设备会报错
- **参数检查**：验证数据类型、形状和设备一致性

**示例**：
```cpp
auto cpu_backend = BackendManager::get_cpu_backend();
Tensor src = Tensor::empty(Shape(2, 3), DType::FP32, tr::CPU);
Tensor dst = Tensor::empty(Shape(2, 3), DType::FP32, tr::CPU);
cpu_backend->copy_into(src, dst);  // 深拷贝到dst
```

### 单目运算操作

CPU后端提供了完整的单目运算功能，包括10种运算的30个API变体。详细的单目运算API、使用示例和性能优化请参考 [单目运算 API 文档](cpu_unary.md)。

**主要功能**：
- **10种运算**：zeros, ones, relu, sign, square, sqrt, abs, negative, reciprocal, round
- **3种模式**：非原地运算（返回新张量）、原地运算（修改输入张量）、指定输出张量运算
- **Eigen优化**：SIMD向量化加速，支持智能降级到朴素实现
- **内存安全**：64字节对齐，完善的异常处理机制
- **灵活配置**：NaN检查、形状验证等编译时配置选项

**数据类型支持**：
- FP32：全部10种运算完全支持
- INT8：仅支持zeros和ones运算

**配置宏**：
- `TR_USE_EIGEN`：启用Eigen优化
- `TR_ENABLE_NAN_CHECK`：NaN检查模式配置
- `TR_ENABLE_INTO_FUNC_SHAPE_CHECK`：形状检查配置

### 大规模数据处理示例

```cpp
void large_scale_computation() {
    auto backend = BackendManager::get_cpu_backend();

    // 创建大型张量（1000x1000）
    Shape shape(1000, 1000);
    Tensor a(shape, DType::FP32, tr::CPU);
    Tensor b(shape, DType::FP32, tr::CPU);
    Tensor result(shape, DType::FP32, tr::CPU);

    // 填充数据
    backend->fill(a, 1.0f);
    backend->fill(b, 2.0f);

    // 执行计算（Eigen会自动使用多线程优化）
    backend->add(result, a, b);

    std::cout << "Large-scale computation completed!" << std::endl;
}
```

### 内存管理示例

```cpp
void memory_management_example() {
    auto backend = BackendManager::get_cpu_backend();

    // 分配内存
    const size_t num_elements = 1000;
    const size_t memory_size = num_elements * sizeof(float);
    auto memory = backend->allocate(memory_size);

    // 获取指针并操作
    float* data = static_cast<float*>(backend->get_data_ptr(memory));
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = static_cast<float>(i);
    }

    // 内存会在智能指针析构时自动释放
    std::cout << "Memory allocated and will be automatically freed" << std::endl;
}
```

## 性能优化建议

### Eigen库优化
- **自动优化选择**：CpuBackend会自动选择Eigen优化版本或朴素实现
- **零拷贝操作**：使用`Eigen::Map`直接操作张量数据，避免内存拷贝
- **SIMD向量化**：Eigen自动使用SSE/AVX指令集进行向量化计算
- **自动启用**：`TR_USE_EIGEN`宏会在找到Eigen库时自动设置

### 性能特点
- **矩阵乘法**：Eigen优化比朴素实现快3-5倍（详见[矩阵乘法文档](cpu_mm_fp32.md)）
- **单目运算**：SIMD向量化加速，支持智能降级（详见[单目运算文档](cpu_unary.md)）
- **多线程支持**：Eigen自动使用OpenMP进行并行计算
- **内存优化**：64字节对齐优化SIMD访问性能

### 编译优化配置
```cmake
# 推荐配置（获得最佳性能）
option(TR_USE_EIGEN ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /arch:AVX2")  # MSVC
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")  # GCC/Clang
```

### 多线程控制
可以通过环境变量控制Eigen的OpenMP线程数：
```bash
export OMP_NUM_THREADS=4  # Linux/macOS
set OMP_NUM_THREADS=4     # Windows
```

## 编译配置

### CMake选项
```cmake
option(TR_USE_EIGEN "Enable Eigen for CPU optimizations" ON)
```

### Eigen库检测和配置
CMake会自动检测系统中的Eigen库：

1. **自动检测**：CMake会在以下位置查找Eigen库
   - `third_party/Eigen`（项目内置推荐）
   - 系统安装的Eigen库（通过`find_package(Eigen)`）

2. **自动启用**：如果找到Eigen库，会自动设置：
   - `TR_USE_EIGEN`编译宏
   - Eigen头文件路径
   - 编译器优化标志

3. **手动配置**：如果需要手动指定Eigen路径：
   ```cmake
   set(EIGEN3_INCLUDE_DIR "/path/to/eigen/include")
   find_package(Eigen3 REQUIRED)
   ```

### 编译时优化建议
- **Release模式**：使用`-O3`或`/O2`优化级别
- **向量化支持**：启用相应的指令集
  - MSVC：`/arch:AVX2`（支持AVX2）
  - GCC/Clang：`-march=native`（自动检测最优指令集）
- **OpenMP支持**：Eigen会自动使用OpenMP进行并行计算

### 编译器优化
- Release模式：启用`/O2`或`-O3`优化
- Debug模式：启用调试符号，便于问题排查
- 启用适当的指令集：`/arch:AVX2`（MSVC）或`-march=native`（GCC/Clang）

## 注意事项

1. **数据类型检查**：操作前会检查张量数据类型，确保操作兼容
2. **形状匹配**：二元操作要求操作数张量形状完全匹配
3. **内存对齐**：Eigen库要求数据内存对齐，CpuBackend会自动处理
4. **线程安全**：单个CpuBackend实例不是线程安全的，多线程环境下应分别创建实例
5. **异常处理**：所有操作都可能抛出异常，建议使用try-catch块处理

## 错误处理

常见异常情况：

```cpp
try {
    // 错误的数据类型
    tr::Tensor tensor(shape, tr::DType::INT8, tr::CPU);
    backend->fill(tensor, 3.14f);  // 抛出异常
} catch (const tr::TRException& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    // "fill(float) requires FP32 tensor"
}
```

## 张量IO算子（CPU后端独有功能）

### `void export_tensor(const Tensor& tensor, const std::string& filename) const`

将张量导出为TSR格式文件。

**参数：**
- `tensor` - 要导出的张量
- `filename` - 输出文件路径（.tsr格式）

**异常：**
- `TRException` - 当张量为空、类型不支持或文件写入失败时抛出

**支持的数据类型：**
- FP32（4字节浮点数）
- INT8（1字节整数）

**文件格式：**
- TSR（Tech Renaissance）格式，包含完整的张量元数据和二进制数据

**示例：**
```cpp
// 创建张量
tr::Shape shape(2, 3, 4, 5);
tr::Tensor tensor(shape, tr::DType::FP32, tr::CPU);
backend->fill(tensor, 3.14f);

// 导出到文件
backend->export_tensor(tensor, "workspace/test_tensor.tsr");
```

### `Tensor import_tensor(const std::string& filename) const`

从TSR格式文件导入张量。

**参数：**
- `filename` - 输入文件路径（.tsr格式）

**返回值：**
- `Tensor` - 导入的张量对象

**异常：**
- `TRException` - 当文件不存在、格式错误或读取失败时抛出

**支持的功能：**
- 自动检测张量形状、数据类型和设备
- 验证TSR文件格式完整性
- 创建对应的数据存储并加载数据

**示例：**
```cpp
// 从文件导入张量
tr::Tensor tensor = backend->import_tensor("workspace/test_tensor.tsr");

// 验证导入的张量
std::cout << "Shape: " << tensor.shape().to_string() << std::endl;
std::cout << "DType: " << static_cast<int>(tensor.dtype()) << std::endl;
```

### 便捷宏定义

CPU后端提供了便捷的宏来简化张量导入导出操作：

```cpp
#define EXPORT_TENSOR dynamic_cast<CpuBackend*>(BackendManager::instance().get_backend(CPU).get())->export_tensor
#define IMPORT_TENSOR dynamic_cast<CpuBackend*>(BackendManager::instance().get_backend(CPU).get())->import_tensor
```

**使用示例：**
```cpp
// 使用便捷宏导出张量
tr::Tensor tensor(shape, tr::DType::FP32, tr::CPU);
EXPORT_TENSOR(tensor, "output.tsr");

// 使用便捷宏导入张量
tr::Tensor imported_tensor = IMPORT_TENSOR("input.tsr");
```

## 关键设计原则总结（V1.23.1）

### 后端管理存储
- **CPU后端**：使用行主序存储，符合C/C++语言惯例
- **CUDA后端**：使用列主序存储，与cuBLAS/cuDNN库接口一致
- **转换透明**：`from_cpu()`和`to_cpu()`自动处理格式转换

### 数据访问一致性
- **用户视角**：所有张量都是行主序访问，无论在哪个后端
- **后端内部**：各自选择最优的内存布局进行计算
- **转换保证**：跨后端转换保证数学结果的正确性

### 性能优化
- **内存对齐**：64字节对齐优化SIMD访问
- **Eigen集成**：自动SIMD向量化和多线程并行
- **跨后端一致性**：与CUDA后端结果完全一致

## 最佳实践

1. **使用转换方法**：在跨后端操作时使用目标后端的`from_cpu()`和`to_cpu()`方法
2. **内存布局理解**：理解CPU使用行主序，CUDA使用列主序的差异
3. **性能监控**：监控CPU和CUDA计算的性能差异
4. **精度验证**：使用`is_close()`方法验证跨后端计算结果的一致性

## 注意事项

1. **设备一致性**：所有操作的张量必须位于CPU设备上
2. **内存管理**：使用智能指针自动管理64字节对齐的内存
3. **多线程**：Eigen自动配置OpenMP，可通过环境变量控制
4. **异常处理**：完善的异常检查和错误处理机制
5. **跨后端操作**：需要通过BackendManager获取目标后端进行转换

## 版本信息

- **版本**：V1.27.1
- **更新日期**：2025-10-31
- **作者**：技术觉醒团队
- **主要特性**：行主序存储、Eigen向量化优化、跨后端转换、高性能矩阵乘法、完整单目运算支持、张量复制功能
- **架构特性**：
  - **后端管理存储原则**：CPU使用行主序，与C/C++语言惯例一致
  - **跨后端透明转换**：`from_cpu()`和`to_cpu()`自动处理格式转换
  - **高性能计算**：Eigen库SIMD优化和多线程并行计算
  - **内存安全**：64字节对齐优化，RAII智能指针管理
- **核心功能**：
  - **基础运算**：张量填充、逐元素运算、内存管理
  - **矩阵乘法**：高性能GEMM实现（详见[矩阵乘法文档](cpu_mm_fp32.md)）
  - **单目运算**：10种运算的30个API变体（详见[单目运算文档](cpu_unary.md)）
  - **张量复制**：同设备内深拷贝操作（V1.27.1新增）
  - **张量IO**：独有TSR格式导入导出功能
  - **跨后端转换**：与其他后端的无缝数据转换
- **性能优化**：
  - **双重实现策略**：Eigen优化版本和朴素实现
  - **SIMD向量化**：Eigen自动使用SSE/AVX指令集
  - **零拷贝操作**：`Eigen::Map`避免内存拷贝
  - **智能优化选择**：根据数据特性选择最优实现
  - **全局Eigen配置**：默认开启Eigen优化，支持手动禁用
- **复制功能特性**：
  - **语义明确**：copy()返回新张量，copy_into()写入指定目标
  - **深拷贝保证**：所有复制操作生成独立数据副本
  - **类型安全**：严格的数据类型和形状检查
  - **性能测试**：CPU复制平均0.494ms，CUDA复制平均0.318ms
- **依赖库**：Eigen3（默认启用以获得最佳性能）、标准C++库
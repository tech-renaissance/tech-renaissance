# CpuBackend API 文档

## 概述

`CpuBackend`是技术觉醒框架的CPU计算后端实现，继承自`Backend`基类。它提供了基于CPU的高性能张量计算能力，支持Eigen库优化和多线程并行计算，是框架的默认和基础计算后端。

**版本**: V1.25.1
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

#### `void copy(void* dst, const void* src, size_t size, const Device& dst_device, const Device& src_device) override`

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
backend->copy(dst_ptr, src_ptr, size, tr::CPU, tr::CPU);
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

### 矩阵乘法操作（V1.23.1核心实现）

#### `void mm(Tensor& result, const Tensor& tensor_a, const Tensor& tensor_b) override`

执行CPU张量的矩阵乘法运算：result = a × b。使用行主序存储格式，支持Eigen优化。

**参数**：
- `result` - 结果CPU张量（行主序存储）
- `tensor_a` - 第一个操作数CPU张量（行主序存储）
- `tensor_b` - 第二个操作数CPU张量（行主序存储）

**异常**：
- `TRException` - 当张量形状、数据类型或设备不匹配时抛出

**关键实现**：
```cpp
void CpuBackend::mm(Tensor& result, const Tensor& tensor_a, const Tensor& tensor_b) {
    // CPU张量使用行主序存储
    const float* a_data = static_cast<const float*>(tensor_a.data_ptr());
    const float* b_data = static_cast<const float*>(tensor_b.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());

    int32_t M = tensor_a.height();  // 行数
    int32_t K = tensor_a.width();   // 列数
    int32_t N = tensor_b.width();   // B的列数

#ifdef TR_USE_EIGEN
    // 使用Eigen优化的实现（行主序）
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eigen_a(a_data, M, K);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eigen_b(b_data, K, N);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eigen_result(result_data, M, N);

    eigen_result.noalias() = eigen_a * eigen_b;
#else
    // 朴素实现（行主序矩阵乘法）
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int32_t k = 0; k < K; ++k) {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            result_data[i * N + j] = sum;
        }
    }
#endif
}
```

**性能特征**：
- **Eigen优化**：自动SIMD向量化，多线程并行计算
- **内存布局**：行主序存储，与C/C++数组访问方式一致
- **线程安全**：Eigen自动配置OpenMP多线程
- **精度保证**：与CUDA后端结果在行主序下完全一致

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

### 跨后端矩阵乘法（V1.23.1推荐用法）

```cpp
#include "tech_renaissance.h"
using namespace tr;

void cross_backend_matrix_multiplication() {
    try {
        // 获取后端实例（V1.23.1新API）
        auto cpu_backend = BackendManager::get_cpu_backend();
        auto cuda_backend = BackendManager::get_cuda_backend();

        // 创建CPU随机张量（行主序存储）
        Tensor cpu_a = Tensor::randn(Shape(1024, 2048), 42);
        Tensor cpu_b = Tensor::randn(Shape(2048, 512), 42);

        // CPU矩阵乘法（行主序计算）
        Tensor cpu_result = Tensor::empty(Shape(1024, 512), DType::FP32, tr::CPU);
        cpu_backend->mm(cpu_result, cpu_a, cpu_b);

        // 转换到CUDA（自动转换为列主序）
        Tensor cuda_a = cuda_backend->from_cpu(cpu_a);  // 行主序 → 列主序
        Tensor cuda_b = cuda_backend->from_cpu(cpu_b);
        Tensor cuda_result = Tensor::empty(Shape(1024, 512), DType::FP32, tr::CUDA[0]);

        // CUDA矩阵乘法（列主序计算）
        cuda_backend->mm(cuda_result, cuda_a, cuda_b);

        // 转换回CPU（自动转换回行主序）
        Tensor cuda_result_cpu = cuda_backend->to_cpu(cuda_result);  // 列主序 → 行主序

        // 结果验证：CPU和CUDA结果在行主序下应该一致
        bool is_close = cpu_backend->is_close(cpu_result, cuda_result_cpu, 1e-4f);
        std::cout << "Results are close: " << (is_close ? "YES" : "NO") << std::endl;

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

### 高性能矩阵乘法基准测试

```cpp
void cpu_performance_benchmark() {
    auto cpu_backend = BackendManager::get_cpu_backend();

    // 创建测试矩阵
    const int32_t M = 1024, K = 2048, N = 512;
    Tensor cpu_a = Tensor::randn(Shape(M, K), 42);
    Tensor cpu_b = Tensor::randn(Shape(K, N), 42);
    Tensor cpu_result = Tensor::empty(Shape(M, N), DType::FP32, tr::CPU);

    // 预热
    for (int i = 0; i < 10; ++i) {
        cpu_backend->mm(cpu_result, cpu_a, cpu_b);
    }

    // 性能测试
    const int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        cpu_backend->mm(cpu_result, cpu_a, cpu_b);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 计算GFLOPS
    double flops = 2.0 * M * K * N;  // 矩阵乘法的浮点运算次数
    double avg_time_ms = duration.count() / 1000.0 / iterations;
    double gflops = flops / (avg_time_ms * 1e6);

    std::cout << "CPU GEMM Performance:" << std::endl;
    std::cout << "  Matrix size: " << M << "x" << K << " x " << K << "x" << N << std::endl;
    std::cout << "  Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << gflops << std::endl;
}
```

### 单目运算操作（V1.25.1新增）

CPU后端提供了9种单目运算，每种都包含非原地和原地两个版本，支持FP32和INT8数据类型。

#### `Tensor zeros_like(const Tensor& input) const`

创建与输入张量相同形状的全零张量。

**参数**：
- `input` - 输入张量

**返回值**：
- `Tensor` - 全零张量，形状和类型与输入相同

**实现特点**：
- 使用`std::memset`高效填充，性能优异
- 支持FP32和INT8数据类型
- 内存对齐优化

**示例**：
```cpp
Tensor input(Shape(2, 3), DType::FP32, tr::CPU);
backend->fill(input, 5.0f);
Tensor zeros = backend->zeros_like(input);  // 所有元素为0.0f
```

#### `Tensor ones_like(const Tensor& input) const`

创建与输入张量相同形状的全1张量。

**参数**：
- `input` - 输入张量

**返回值**：
- `Tensor` - 全1张量，形状和类型与输入相同

**示例**：
```cpp
Tensor input(Shape(2, 3), DType::FP32, tr::CPU);
Tensor ones = backend->ones_like(input);  // 所有元素为1.0f
```

#### `Tensor relu(const Tensor& input) const`

执行ReLU激活函数：max(0, x)。

**参数**：
- `input` - 输入张量（仅支持FP32）

**返回值**：
- `Tensor` - ReLU激活后的张量

**示例**：
```cpp
Tensor input(Shape(2, 3), DType::FP32, tr::CPU);
backend->fill(input, -2.0f);
Tensor result = backend->relu(input);  // 负数变为0，正数保持不变
```

#### `Tensor sign(const Tensor& input) const`

执行符号函数：x>0返回1，x<0返回-1，x=0返回0。

**参数**：
- `input` - 输入张量（仅支持FP32）

**返回值**：
- `Tensor` - 符号函数结果张量

**示例**：
```cpp
Tensor result = backend->sign(input);  // 每个元素为-1, 0, 或1
```

#### `Tensor square(const Tensor& input) const`

执行平方运算：x²。

**参数**：
- `input` - 输入张量（仅支持FP32）

**返回值**：
- `Tensor` - 平方结果张量

**示例**：
```cpp
Tensor result = backend->square(input);  // 每个元素平方
```

#### `Tensor sqrt(const Tensor& input) const`

执行平方根运算：√x。

**参数**：
- `input` - 输入张量（仅支持FP32，必须非负）

**返回值**：
- `Tensor` - 平方根结果张量

**异常**：
- `TRException` - 当输入包含负数时抛出（可配置）

**示例**：
```cpp
Tensor result = backend->sqrt(input);  // 每个元素平方根
```

#### `Tensor abs(const Tensor& input) const`

执行绝对值运算：|x|。

**参数**：
- `input` - 输入张量（仅支持FP32）

**返回值**：
- `Tensor` - 绝对值结果张量

**示例**：
```cpp
Tensor result = backend->abs(input);  // 每个元素绝对值
```

#### `Tensor negative(const Tensor& input) const`

执行相反数运算：-x。

**参数**：
- `input` - 输入张量（仅支持FP32）

**返回值**：
- `Tensor` - 相反数结果张量

**示例**：
```cpp
Tensor result = backend->negative(input);  // 每个元素取负
```

#### `Tensor reciprocal(const Tensor& input) const`

执行倒数运算：1/x。

**参数**：
- `input` - 输入张量（仅支持FP32）

**返回值**：
- `Tensor` - 倒数结果张量

**异常**：
- `TRException` - 当输入包含0时抛出（可配置）

**示例**：
```cpp
Tensor result = backend->reciprocal(input);  // 每个元素倒数
```

### 原地单目运算操作（V1.25.1新增）

原地运算直接修改输入张量，避免内存分配，性能更高。

#### `void zeros_inplace(Tensor& input) const`

原地将张量所有元素设置为0。

**参数**：
- `input` - 要修改的张量

**实现特点**：
- 使用`std::memset`高效填充，性能优异
- 无额外内存分配

**示例**：
```cpp
backend->zeros_inplace(input);  // input直接变为全0
```

#### `void ones_inplace(Tensor& input) const`

原地将张量所有元素设置为1。

**参数**：
- `input` - 要修改的张量

**示例**：
```cpp
backend->ones_inplace(input);  // input直接变为全1
```

#### `void relu_inplace(Tensor& input) const`

原地执行ReLU激活函数。

**参数**：
- `input` - 要修改的张量（仅支持FP32）

**示例**：
```cpp
backend->relu_inplace(input);  // 负数原地变为0
```

#### `void sign_inplace(Tensor& input) const`

原地执行符号函数。

**参数**：
- `input` - 要修改的张量（仅支持FP32）

**示例**：
```cpp
backend->sign_inplace(input);  // 原地计算符号
```

#### `void square_inplace(Tensor& input) const`

原地执行平方运算。

**参数**：
- `input` - 要修改的张量（仅支持FP32）

**示例**：
```cpp
backend->square_inplace(input);  // 原地平方
```

#### `void sqrt_inplace(Tensor& input) const`

原地执行平方根运算。

**参数**：
- `input` - 要修改的张量（仅支持FP32，必须非负）

**示例**：
```cpp
backend->sqrt_inplace(input);  // 原地平方根
```

#### `void abs_inplace(Tensor& input) const`

原地执行绝对值运算。

**参数**：
- `input` - 要修改的张量（仅支持FP32）

**示例**：
```cpp
backend->abs_inplace(input);  // 原地绝对值
```

#### `void negative_inplace(Tensor& input) const`

原地执行相反数运算。

**参数**：
- `input` - 要修改的张量（仅支持FP32）

**示例**：
```cpp
backend->negative_inplace(input);  // 原地取负
```

#### `void reciprocal_inplace(Tensor& input) const`

原地执行倒数运算。

**参数**：
- `input` - 要修改的张量（仅支持FP32）

**示例**：
```cpp
backend->reciprocal_inplace(input);  // 原地倒数
```

### NaN检查配置（V1.25.1新增）

单目运算支持3种NaN检查模式，通过编译时宏`TR_ENABLE_NAN_CHECK`配置：

```cpp
// 检查模式配置
#define TR_ENABLE_NAN_CHECK 0  // 不检查，直接计算（产生NaN/inf）
#define TR_ENABLE_NAN_CHECK 1  // 检查并报错（默认模式）
#define TR_ENABLE_NAN_CHECK 2  // 检查并替换（sqrt负数→0，倒数零→1/eps）
```

**eps常量**：
```cpp
constexpr float TR_EPS = 1e-10f;  // 用于处理除零等特殊情况
```

### 性能优化特点（V1.25.1）

1. **memset优化**：`zeros_like`和`zeros_inplace`使用`std::memset`，比循环快数倍
2. **内存对齐**：所有操作都基于64字节对齐的内存，优化SIMD访问
3. **原地运算**：避免额外内存分配，提升性能
4. **编译时优化**：支持编译器自动向量化

### 大规模数据处理

```cpp
void large_scale_computation() {
    auto backend = std::make_shared<tr::CpuBackend>();

    // 创建大型张量（1000x1000）
    tr::Shape shape(1000, 1000);
    tr::Tensor a(shape, tr::DType::FP32, tr::CPU);
    tr::Tensor b(shape, tr::DType::FP32, tr::CPU);
    tr::Tensor result(shape, tr::DType::FP32, tr::CPU);

    // 填充数据
    backend->fill(a, 1.0f);
    backend->fill(b, 2.0f);

    // 执行计算（Eigen会自动使用多线程优化）
    backend->add(result, a, b);

    std::cout << "Large-scale computation completed!" << std::endl;
}
```

### 单目运算示例（V1.25.1新增）

```cpp
#include "tech_renaissance.h"
using namespace tr;

void unary_operations_example() {
    auto cpu_backend = BackendManager::get_cpu_backend();

    // 创建测试张量
    Shape shape(2, 3, 4, 5);
    Tensor input(shape, DType::FP32, tr::CPU);
    cpu_backend->fill(input, 2.5f);

    // 非原地运算
    Tensor zeros = cpu_backend->zeros_like(input);
    Tensor ones = cpu_backend->ones_like(input);
    Tensor relu_result = cpu_backend->relu(input);
    Tensor square_result = cpu_backend->square(input);
    Tensor sqrt_result = cpu_backend->sqrt(input);  // 输入必须非负
    Tensor abs_result = cpu_backend->abs(input);
    Tensor negative_result = cpu_backend->negative(input);
    Tensor sign_result = cpu_backend->sign(input);
    Tensor reciprocal_result = cpu_backend->reciprocal(input);

    // 原地运算（性能更高）
    Tensor inplace_tensor = Tensor::randn(shape, 42);
    cpu_backend->relu_inplace(inplace_tensor);      // 直接修改原张量
    cpu_backend->square_inplace(inplace_tensor);    // 链式原地运算

    std::cout << "Unary operations completed successfully!" << std::endl;
}
```

### 内存管理示例

```cpp
void memory_management_example() {
    auto backend = std::make_shared<tr::CpuBackend>();

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

### Eigen库使用
- CpuBackend会自动使用Eigen库进行向量化计算
- 确保编译时启用适当的优化标志（如-O2, -O3）
- 在支持的平台启用AVX/SSE指令集优化

### 多线程优化
- Eigen库会自动使用OpenMP进行并行计算
- 可以通过环境变量控制线程数：
  ```bash
  export OMP_NUM_THREADS=4  # Linux/macOS
  set OMP_NUM_THREADS=4     # Windows
  ```

### 内存访问模式
- 尽量使用连续内存布局，提高缓存命中率
- 避免频繁的小块内存分配，可以预分配大块内存

## 编译配置

### CMake选项
```cmake
option(TR_USE_EIGEN "Enable Eigen for CPU optimizations" ON)
```

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

- **版本**：V1.25.1
- **更新日期**：2025-10-31
- **作者**：技术觉醒团队
- **主要特性**：行主序存储、Eigen优化、跨后端转换、高性能矩阵乘法、9种单目运算（18个API）、静默模式支持
- **性能优化**：memset零填充优化、原地运算支持、NaN检查配置
- **新增功能**：
  - 单目运算：zeros_like, ones_like, relu, sign, square, sqrt, abs, negative, reciprocal
  - 原地版本：zeros_inplace, ones_inplace, relu_inplace, sign_inplace, square_inplace, sqrt_inplace, abs_inplace, negative_inplace, reciprocal_inplace
  - 静默模式：Python服务器DEBUG_MODE配置
  - NaN检查：3种可配置的NaN处理模式
- **依赖库**：Eigen3（可选）、标准C++库
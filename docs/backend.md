# Backend API 文档

## 版本信息

- **版本**: V1.46.3
- **日期**: 2025年11月17日
- **作者**: 技术觉醒团队
- **所属系列**: backend

## 最新完成状态

✅ **V1.46.3完成 - 代码规范优化和构造函数设计统一化**:
- Backend构造函数设计统一 - 使用protected explicit构造函数，防止直接实例化
- CpuBackend构造函数优化 - 使用explicit关键字，统一接口设计
- CudaBackend构造函数优化 - 使用explicit关键字，统一接口设计
- 代码规范统一化 - 移除重复定义，统一注释风格
- Alpha编译验证通过 - backend库编译无错误

✅ **V1.44.1完成 - 后端基类重构**:
- 从抽象类改为可实例化类 - Backend不再是抽象类，但直接实例化会抛出异常
- 宏定义系统引入 - 统一的宏系统来声明和实现新方法
- 默认NotImplementedError - 未实现的方法自动抛出统一格式的异常

## # 重要警告：张量创建的正确方式！

**Backend是唯一推荐的张量创建方式！**

在Tech Renaissance框架中，所有张量都必须通过Backend类的方法来创建：

**推荐的张量创建方法：**
```cpp
// 获取Backend基类实例
auto backend = BackendManager::instance().get_backend(CPU);

// 转换为具体的Backend子类（如CpuBackend）
auto cpu_backend = std::dynamic_pointer_cast<CpuBackend>(backend);

// 使用Backend子类的方法（这些方法在子类中实现）
Tensor zeros = cpu_backend->zeros(shape, dtype);
Tensor ones = cpu_backend->ones(shape, dtype);
Tensor full = cpu_backend->full(shape, value, dtype);
Tensor empty = cpu_backend->empty(shape, dtype);

// 随机生成方法（如果子类支持）
Tensor randn = cpu_backend->randn(shape, seed);
Tensor uniform = cpu_backend->uniform(shape, min_val, max_val, seed);
Tensor randint = cpu_backend->randint(shape, low, high, dtype, seed);
```

**绝对禁止的方式：**
- 直接使用`Tensor(shape, dtype, device)`构造函数（不分配内存！）
- 使用Tensor类的静态工厂方法（不推荐）
- 试图手动分配张量内存
- 误认为Backend基类直接包含创建方法（这些方法在子类中实现）

## 概述

`Backend`是技术觉醒框架的后端基类，定义了所有计算后端（CPU、CUDA等）必须实现的统一接口。在V1.43.0版本中进行了重大重构，从抽象类改为可实例化但抛出异常的类，并引入了宏定义系统来简化新方法的添加。

**版本**: V1.44.1
**更新日期**: 2025-11-16
**作者**: 技术觉醒团队

## 🆕 V1.43.0重大更新：后端基类重构

### 🎯 重构核心变化

在V1.43.0版本中，Backend基类经历了重大重构：

1. **从抽象类改为可实例化类**：Backend不再是抽象类，但直接实例化会抛出异常
2. **宏定义系统**：引入统一的宏系统来声明和实现新方法
3. **默认NotImplementedError**：未实现的方法自动抛出统一格式的异常
4. **100%向后兼容**：所有现有代码无需修改即可正常工作

### 🔧 新的构造函数机制

```cpp
class Backend {
public:
    /**
     * @brief 公共构造函数 - 防止直接实例化
     * @throws TRException 直接实例化时抛出异常
     */
    Backend() {
        throw TRException("Backend class cannot be instantiated directly! Use specific backend implementations instead.");
    }

protected:
    /**
     * @brief 受保护的构造函数 - 允许派生类构造
     * @param allow_construction 是否允许构造（派生类传true）
     */
    Backend(bool allow_construction);
};
```

### 📝 宏定义系统

```cpp
// 定义未实现方法的宏
#define DEFINE_NOT_IMPLEMENTED_METHOD(method_name, return_type, params, const_qualifier) \
    return_type Backend::method_name params const_qualifier { \
        throw NotImplementedError("[" + name() + " " #method_name "] Operation NOT implemented!"); \
    }

// 定义void返回类型未实现方法的宏
#define DEFINE_NOT_IMPLEMENTED_VOID_METHOD(method_name, params, const_qualifier) \
    void Backend::method_name params const_qualifier { \
        throw NotImplementedError("[" + name() + " " #method_name "] Operation NOT implemented!"); \
    }
```

### ✅ 重构优势

1. **扩展性极强**：新增方法只需要在Backend基类添加一行宏定义
2. **维护成本低**：无需修改所有后端类的头文件
3. **异常信息统一**：所有未实现方法都有清晰的错误提示
4. **类型安全**：编译时检查，避免运行时错误
5. **向后兼容**：现有代码无需任何修改

## 设计理念

### 核心设计原则

1. **统一接口**：定义统一的后端操作接口，支持多设备计算
2. **设备无关**：提供与具体硬件无关的计算抽象
3. **后端管理存储**：每个后端负责管理自己的张量存储格式
4. **跨后端转换**：通过`from_cpu()`、`to_cpu()`和`to()`方法实现跨设备数据转换
5. **RAII设计**：使用智能指针管理内存，避免资源泄漏
6. **前向声明**：避免循环依赖，不包含具体实现的头文件
7. **异常安全**：完善的错误处理机制
8. **🆕 宏驱动扩展**：通过宏系统简化新方法的添加和维护

### 关键架构特性

#### **跨后端转换接口（核心特性）**

Backend类定义了标准的跨后端转换接口：
- **`from_cpu()`**: 从CPU转换到当前后端设备，自动处理内存布局转换
- **`to_cpu()`**: 从当前后端设备转换到CPU，自动处理内存布局转换
- **`to()`**: 通用的设备转换接口

#### **后端存储管理原则**

每个后端实现负责管理自己的张量存储格式：
- **CPU后端**：使用行主序（Row-major）存储，符合C/C++惯例
- **CUDA后端**：使用列主序（Column-major）存储，与cuBLAS库接口一致
- **转换层透明**：用户无需关心底层存储格式，转换层自动处理

## 头文件

```cpp
#include "tech_renaissance/backend/backend.h"
```

## 接口详情

### 内存管理接口

#### `virtual std::shared_ptr<void> allocate(size_t size) = 0`

分配指定大小的内存。

**参数：**
- `size` - 要分配的内存大小（字节）

**返回值：**
- `std::shared_ptr<void>` - 内存块的智能指针

**异常：**
- `TRException` - 当分配失败时抛出

#### `virtual void deallocate(void* ptr) = 0`

释放已分配的内存。

**参数：**
- `ptr` - 要释放的内存指针

**异常：**
- `TRException` - 当释放失败时抛出

#### `virtual void* get_data_ptr(const std::shared_ptr<void>& holder) = 0`

从内存持有者中获取原始数据指针。

**参数：**
- `holder` - 内存智能指针

**返回值：**
- `void*` - 原始数据指针

#### `virtual void copy_data(void* dst, const void* src, size_t size, const Device& dst_device, const Device& src_device) = 0`

在设备间复制数据。

**参数：**
- `dst` - 目标内存指针
- `src` - 源内存指针
- `size` - 复制大小（字节）
- `dst_device` - 目标设备
- `src_device` - 源设备

**支持的复制方向：**
- CPU → CPU
- CPU → CUDA
- CUDA → CPU
- CUDA → CUDA

### 张量操作接口

#### `virtual void fill(Tensor& dst, float value) = 0`

用浮点数值填充张量。

**参数：**
- `dst` - 目标张量
- `value` - 填充值

**异常：**
- `TRException` - 当张量数据类型不是FP32时抛出

#### `virtual void fill(Tensor& dst, int8_t value) = 0`

用8位整数值填充张量。

**参数：**
- `dst` - 目标张量
- `value` - 填充值

**异常：**
- `TRException` - 当张量数据类型不是INT8时抛出

#### `virtual void add(Tensor& result, const Tensor& a, const Tensor& b) = 0`

执行逐元素加法运算：result = a + b。

**参数：**
- `result` - 结果张量
- `a` - 第一个操作数张量
- `b` - 第二个操作数张量

**异常：**
- `TRException` - 当张量形状、数据类型或设备不匹配时抛出

#### `virtual void mul(Tensor& result, const Tensor& a, const Tensor& b) = 0`

执行逐元素乘法运算：result = a * b。

**参数**：
- `result` - 结果张量
- `a` - 第一个操作数张量
- `b` - 第二个操作数张量

**异常**：
- `TRException` - 当张量形状、数据类型或设备不匹配时抛出

#### `virtual void mm(Tensor& result, const Tensor& a, const Tensor& b) = 0`

执行矩阵乘法运算：result = a × b。

**参数**：
- `result` - 结果张量，形状应为(M,N)
- `a` - 输入张量A，形状应为(M,K)
- `b` - 输入张量B，形状应为(K,N)

**异常**：
- `TRException` - 当张量形状、数据类型或设备不匹配时抛出

**矩阵乘法规则**：C(M,N) = A(M,K) × B(K,N)

### 🆕 V1.43.0新增接口

以下方法通过宏定义系统实现，默认抛出`NotImplementedError`异常：

#### 视图操作

```cpp
virtual Tensor view(const Tensor& input, const Shape& new_shape);
```

**说明**: 创建张量视图，提供零拷贝的形状变换。视图与原始张量共享存储空间，支持高效的数据重解释。

#### 形状变换操作

```cpp
virtual Tensor reshape(const Tensor& tensor_a, const Shape& shape);
virtual void reshape_inplace(Tensor& tensor_a, const Shape& shape);
virtual void reshape_into(const Tensor& tensor_a, Tensor& result, const Shape& shape);
```

#### 双曲函数操作

```cpp
virtual Tensor tanh(const Tensor& tensor_a);
virtual void tanh_inplace(Tensor& tensor_a);
virtual void tanh_into(const Tensor& tensor_a, Tensor& result);
virtual Tensor dtanh(const Tensor& tensor_a);
virtual void dtanh_inplace(Tensor& tensor_a);
virtual void dtanh_into(const Tensor& tensor_a, Tensor& result);
```

#### 损失函数操作

```cpp
virtual float crossentropy(const Tensor& pred, const Tensor& label, std::string reduction);
```

#### One-hot编码操作

```cpp
virtual Tensor one_hot(const Tensor& label, int32_t num_classes, float label_smoothing);
virtual void one_hot_into(const Tensor& label, Tensor& result, int32_t num_classes, float label_smoothing);
```

#### 标量运算操作

```cpp
virtual Tensor minus(const Tensor& input, float scalar) const;
virtual void minus_inplace(Tensor& input, float scalar) const;
virtual void minus_into(const Tensor& input, float scalar, Tensor& output) const;
virtual Tensor minus(float scalar, const Tensor& input) const;
virtual void minus_inplace(float scalar, Tensor& input) const;
virtual void minus_into(float scalar, const Tensor& input, Tensor& output) const;
virtual Tensor mac(const Tensor& input, float scalar_x, float scalar_y) const;
virtual void mac_inplace(Tensor& input, float scalar_x, float scalar_y) const;
virtual void mac_into(const Tensor& input, float scalar_x, float scalar_y, Tensor& output) const;
virtual Tensor clamp(const Tensor& input, float min_val, float max_val) const;
virtual void clamp_inplace(Tensor& input, float min_val, float max_val) const;
virtual void clamp_into(const Tensor& input, float min_val, float max_val, Tensor& output) const;
```

#### 广播运算操作

```cpp
virtual Tensor add_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const;
virtual void add_broadcast_into(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result) const;
virtual Tensor minus_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const;
virtual void minus_broadcast_into(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result) const;
virtual Tensor mul_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const;
virtual void mul_broadcast_into(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result) const;
```

### 跨后端转换接口

#### `virtual Tensor from_cpu(const Tensor& tensor) const = 0`

从CPU转换张量到当前后端设备，自动处理内存布局转换。

**参数**：
- `tensor` - CPU设备上的张量（行主序存储）

**返回值**：
- `Tensor` - 当前后端设备上的张量（后端特定存储格式）

#### `virtual Tensor to_cpu(const Tensor& tensor) const = 0`

从当前后端设备转换张量到CPU，自动处理内存布局转换。

**参数**：
- `tensor` - 当前后端设备上的张量（后端特定存储格式）

**返回值**：
- `Tensor` - CPU设备上的张量（行主序存储）

#### `virtual Tensor to(const Tensor& tensor, const Device& target_device) const = 0`

通用的设备转换接口，支持在任意设备间转换张量。

**参数**：
- `tensor` - 源张量
- `target_device` - 目标设备

**返回值**：
- `Tensor` - 目标设备上的张量

### 张量复制操作接口

#### `virtual Tensor copy(const Tensor& tensor) const = 0`

复制张量，返回新的张量副本（同后端内操作）。

**参数**：
- `tensor` - 源张量，必须属于当前后端

**返回值**：
- `Tensor` - 复制后的新张量，属于同一后端

#### `virtual void copy_into(const Tensor& src, Tensor& dst) const = 0`

将源张量复制到指定目标张量，支持跨设备操作。

**参数**：
- `src` - 源张量
- `dst` - 目标张量，用于接收复制结果

### 设备信息接口

#### `virtual Device device() const = 0`

获取后端对应的设备信息。

**返回值：**
- `Device` - 后端设备对象

#### `virtual std::string name() const = 0`

获取后端的名称。

**返回值：**
- `std::string` - 后端名称

## 🚀 新方法添加流程

### V1.43.0简化的扩展流程

使用新的宏系统，添加新方法变得极其简单：

#### 步骤1：在Backend基类中声明方法
```cpp
// 在backend.h中
class Backend {
    // ... 现有方法
    virtual Tensor new_advanced_op(const Tensor& input, float param) const;
};
```

#### 步骤2：在backend.cpp中使用宏实现
```cpp
// 在backend.cpp中使用宏
DEFINE_NOT_IMPLEMENTED_METHOD(new_advanced_op, Tensor, (const Tensor& input, float param), const)
```

#### 步骤3：在需要的后端中重写
```cpp
// 在cpu_backend.h中重写
class CpuBackend : public Backend {
    Tensor new_advanced_op(const Tensor& input, float param) const override;
};

// 在cpu_backend.cpp中实现
Tensor CpuBackend::new_advanced_op(const Tensor& input, float param) const {
    // CPU后端具体实现
    // 例如：基于Eigen的高性能实现
}
```

### 异常信息格式

所有未实现的方法都会抛出统一格式的异常：
```
[BackendName method_name] Operation NOT implemented!
```

示例：
```
[CudaBackend new_advanced_op] Operation NOT implemented!
[CPUBackend new_advanced_op] Operation NOT implemented!
```

## 使用示例

### 🆕 V1.43.0新增方法使用

```cpp
#include "tech_renaissance.h"
using namespace tr;

void v1_43_0_features() {
    try {
        auto cpu_backend = BackendManager::get_cpu_backend();

        // 创建测试张量
        Tensor input = cpu_backend->randn(Shape(2, 3, 4), 42);

        // 使用新增的方法
        Tensor reshaped = cpu_backend->reshape(input, Shape(2, 12));
        Tensor tanh_result = cpu_backend->tanh(input);

        // One-hot编码
        Tensor label = cpu_backend->ones(Shape(4), DType::INT32);
        Tensor one_hot = cpu_backend->one_hot(label, 10, 0.1f);

        // 交叉熵损失
        float loss = cpu_backend->crossentropy(pred, label, "mean");

        std::cout << "V1.43.0 features working correctly!" << std::endl;

    } catch (const NotImplementedError& e) {
        std::cout << "Method not implemented: " << e.what() << std::endl;
    } catch (const TRException& e) {
        std::cerr << "Backend error: " << e.what() << std::endl;
    }
}
```

### 跨后端矩阵乘法（推荐用法）

```cpp
#include "tech_renaissance.h"
using namespace tr;

void cross_backend_matrix_multiplication() {
    try {
        // 获取后端实例
        auto cpu_backend = BackendManager::get_cpu_backend();
        auto cuda_backend = BackendManager::get_cuda_backend();

        // 创建CPU随机张量（行主序存储）
        Tensor cpu_a = Tensor::randn(Shape(1024, 2048), 42);
        Tensor cpu_b = Tensor::randn(Shape(2048, 512), 42);

        // 转换到CUDA（自动转换为列主序）
        Tensor cuda_a = cuda_backend->from_cpu(cpu_a);
        Tensor cuda_b = cuda_backend->from_cpu(cpu_b);
        Tensor cuda_result = Tensor::empty(Shape(1024, 512), DType::FP32, tr::CUDA[0]);

        // CUDA矩阵乘法（列主序计算）
        cuda_backend->mm(cuda_result, cuda_a, cuda_b);

        // 转换回CPU（自动转换回行主序）
        Tensor cuda_result_cpu = cuda_backend->to_cpu(cuda_result);

        // 结果验证
        bool is_close = cpu_backend->is_close(cpu_result, cuda_result_cpu, 1e-4f);
        std::cout << "Results are close: " << (is_close ? "YES" : "NO") << std::endl;

    } catch (const TRException& e) {
        std::cerr << "Backend error: " << e.what() << std::endl;
    }
}
```

### 基本张量操作

```cpp
#include "tech_renaissance.h"
using namespace tr;

void basic_operations() {
    try {
        auto backend = BackendManager::get_cpu_backend();

        // 创建张量
        Shape shape(2, 3);
        Tensor a(shape, DType::FP32, tr::CPU);
        Tensor b(shape, DType::FP32, tr::CPU);
        Tensor result(shape, DType::FP32, tr::CPU);

        // 填充张量
        backend->fill(a, 1.5f);
        backend->fill(b, 2.5f);

        // 执行加法
        backend->add(result, a, b);

        std::cout << "Basic operations completed successfully!" << std::endl;

    } catch (const TRException& e) {
        std::cerr << "Backend error: " << e.what() << std::endl;
    }
}
```

## 实现指导

### 创建自定义后端

要创建自定义后端，需要继承Backend类并实现所有纯虚函数：

```cpp
class CustomBackend : public tr::Backend {
public:
    // 构造函数 - 必须调用Backend(true)
    CustomBackend() : Backend(true) {
        // 初始化自定义后端
    }

    // 内存管理接口
    std::shared_ptr<void> allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void* get_data_ptr(const std::shared_ptr<void>& holder) override;
    void copy_data(void* dst, const void* src, size_t size,
                   const Device& dst_device, const Device& src_device) const override;

    // 张量操作接口
    void fill(Tensor& dst, float value) override;
    void fill(Tensor& dst, int8_t value) override;
    void add(Tensor& result, const Tensor& a, const Tensor& b) override;
    void mul(Tensor& result, const Tensor& a, const Tensor& b) override;
    void mm(Tensor& result, const Tensor& a, const Tensor& b) override;

    // 跨后端转换接口
    Tensor to(const Tensor& tensor, const Device& device) const override;
    Tensor to_cpu(const Tensor& tensor) const override;
    Tensor from_cpu(const Tensor& tensor) const override;

    // 🆕 V1.43.0新增方法的重写示例
    Tensor reshape(const Tensor& tensor_a, const Shape& shape) override {
        // 自定义reshape实现
        // 例如：基于特定的内存布局优化
    }

    // 辅助接口
    std::string name() const override { return "CustomBackend"; }
    Device device() const override { return /* 自定义设备 */; }

    // 数据访问接口
    float get_scalar_float(const Tensor& tensor) override;
    int32_t get_scalar_int32(const Tensor& tensor) override;
    int8_t get_scalar_int8(const Tensor& tensor) override;
};
```

## 错误处理

### 异常类型

```cpp
// 基础异常类
class TRException : public std::exception {
public:
    TRException(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }
private:
    std::string message_;
};

// 未实现方法的异常类
class NotImplementedError : public TRException {
public:
    NotImplementedError(const std::string& message) : TRException(message) {}
};
```

### 错误处理示例

```cpp
try {
    auto backend = BackendManager::get_cuda_backend();
    Tensor result = backend->some_new_method(input);  // 如果未实现会抛出NotImplementedError
} catch (const NotImplementedError& e) {
    std::cout << "Method not available: " << e.what() << std::endl;
    // 处理未实现方法的逻辑
} catch (const TRException& e) {
    std::cerr << "Backend operation failed: " << e.what() << std::endl;
    // 处理其他错误
}
```

## 性能考虑

- **后端实例复用**：通过BackendManager获取后端实例，避免频繁创建销毁
- **跨设备复制**：大数据量的跨设备复制操作开销较大，应谨慎使用
- **方法选择**：根据具体需求选择支持该方法的后端
- **内存对齐**：CPU后端使用64字节对齐，CUDA后端遵循GPU内存对齐要求

## 最佳实践

1. **使用BackendManager**：通过BackendManager获取后端实例，而不是直接创建
2. **跨后端操作**：使用`from_cpu()`和`to_cpu()`进行设备间转换
3. **异常处理**：所有后端操作都应包含适当的异常处理
4. **设备一致性**：确保计算操作中的所有张量都在同一设备上
5. **🆕 方法扩展**：使用宏系统快速添加新方法，提高开发效率
6. **向后兼容**：新方法的添加不会破坏现有代码

## 版本信息

- **版本**: V1.43.0
- **更新日期**: 2025-11-16
- **作者**: 技术觉醒团队
- **主要更新**:
  - 🆕 Backend基类重构：从抽象类改为可实例化类
  - 🆕 宏定义系统：统一方法声明和默认实现机制
  - 🆕 新增高级操作：reshape、tanh、crossentropy、one_hot等
  - 🆕 扩展性大幅提升：新增方法只需一行宏定义
  - ✅ 100%向后兼容：现有代码无需修改
  - ✅ 完整的异常处理机制
  - ✅ 统一的错误信息格式
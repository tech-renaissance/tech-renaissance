# TRException 异常系统文档

## 概述

TRException是技术觉醒框架的统一异常处理系统，支持类型化异常分类和向后兼容的宏接口。系统从V1.10.00版本开始支持分层异常架构，提供精确的错误类型识别和调试友好的错误信息。

**版本**: V1.10.00
**更新日期**: 2025-11-09
**作者**: 技术觉醒团队

## 核心特性

### 🎯 类型化异常分类
- **6种专门异常类型**：覆盖常见错误场景
- **精确类型捕获**：支持按异常类型精确处理
- **统一继承体系**：所有异常继承自TRException基类

### 🔄 100%向后兼容
- **原有宏完全保留**：现有代码无需修改
- **渐进式升级**：可选择性使用新特性
- **编译时类型安全**：避免运行时类型错误

### 🐛 调试友好设计
- **详细错误信息**：包含文件名和行号
- **类型标识**：错误消息前缀显示异常类型
- **清晰格式化**：统一的信息输出格式

## 异常类型层次结构

```
std::exception
    └── tr::TRException (基类)
        ├── tr::FileNotFoundError
        ├── tr::NotImplementedError
        ├── tr::ZeroDivisionError
        ├── tr::TypeError
        ├── tr::ValueError
        └── tr::IndexError
```

## 异常类型详细说明

### 1. TRException (基类)
框架统一异常基类，所有异常的父类。

**构造函数**:
```cpp
TRException(const std::string& message,
           const std::string& file = "",
           int line = 0);
```

**方法**:
```cpp
const char* what() const noexcept;     // 获取完整错误信息
const char* type() const noexcept;     // 获取异常类型名称
const std::string& file() const;      // 获取文件名
int line() const;                     // 获取行号
```

### 2. FileNotFoundError
文件未找到异常，用于处理文件IO相关错误。

**使用场景**:
```cpp
if (!std::ifstream(filename).good()) {
    TR_THROW_FILE_NOT_FOUND("Checkpoint file not found: " + filename);
}
```

### 3. NotImplementedError
功能未实现异常，用于标记尚未实现的功能。

**使用场景**:
```cpp
Tensor CpuBackend::conv(const Tensor& input, const Tensor& kernel) {
    TR_THROW_NOT_IMPLEMENTED("CPU backend convolution not implemented yet");
}
```

### 4. ValueError
数值或参数取值错误异常，用于验证参数范围和有效性。

**使用场景**:
```cpp
if (kernel_size != 1 && kernel_size != 3 && kernel_size != 5 && kernel_size != 7) {
    TR_THROW_VALUE_ERROR("Invalid kernel size: must be 1, 3, 5, or 7");
}
```

### 5. IndexError
索引越界异常，用于处理数组、张量等索引访问错误。

**使用场景**:
```cpp
if (index >= tensor.numel()) {
    TR_THROW_INDEX_ERROR("Tensor index out of bounds");
}
```

### 6. TypeError
类型错误异常，用于处理数据类型不匹配问题。

**使用场景**:
```cpp
if (input.dtype() != DType::FP32) {
    TR_THROW_TYPE_ERROR("Expected FP32 input, got " + dtype_to_string(input.dtype()));
}
```

### 7. ZeroDivisionError
除零错误异常，用于处理数学运算中的除零问题。

**使用场景**:
```cpp
if (denominator == 0.0f) {
    TR_THROW_ZERO_DIVISION("Division by zero in normalization");
}
```

## 宏接口

### 向后兼容宏 (原有宏)

#### TR_THROW
抛出通用TRException异常。

```cpp
TR_THROW("Generic error message");
// 输出: TRException: Generic error message (File: file.cpp, Line: 42)
```

#### TR_THROW_IF
条件抛出异常。

```cpp
TR_THROW_IF(condition, "Error message when condition is true");
```

### 新增类型化宏

#### 专用异常宏
```cpp
TR_THROW_FILE_NOT_FOUND("File not found");      // FileNotFoundError
TR_THROW_NOT_IMPLEMENTED("Feature missing");    // NotImplementedError
TR_THROW_VALUE_ERROR("Invalid value");          // ValueError
TR_THROW_INDEX_ERROR("Index out of bounds");    // IndexError
TR_THROW_TYPE_ERROR("Wrong type");             // TypeError
TR_THROW_ZERO_DIVISION("Division by zero");    // ZeroDivisionError
```

#### 通用类型宏
```cpp
TR_THROW_TYPE(ValueError, "Custom message");   // 创建指定类型的异常
```

## 使用模式

### 1. 基本错误处理

```cpp
#include "tech_renaissance/utils/tr_exception.h"

void load_model(const std::string& path) {
    std::ifstream file(path);
    if (!file.good()) {
        TR_THROW_FILE_NOT_FOUND("Cannot open model file: " + path);
    }

    // 继续处理...
}
```

### 2. 精确异常捕获

```cpp
try {
    auto result = model.forward(input);
} catch (const tr::FileNotFoundError& e) {
    std::cerr << "文件问题: " << e.what() << std::endl;
    // 处理文件相关问题
} catch (const tr::ValueError& e) {
    std::cerr << "参数问题: " << e.what() << std::endl;
    // 处理参数相关问题
} catch (const tr::NotImplementedError& e) {
    std::cerr << "功能未实现: " << e.what() << std::endl;
    // 处理未实现功能
} catch (const tr::TRException& e) {
    std::cerr << "框架异常: " << e.what() << std::endl;
    // 处理其他框架异常
}
```

### 3. 向后兼容升级

**原有代码** (无需修改):
```cpp
// 旧代码 - 继续正常工作
TR_THROW("Something went wrong");
```

**升级代码** (可选升级):
```cpp
// 新代码 - 使用类型化异常
TR_THROW_VALUE_ERROR("Invalid parameter: " + std::to_string(value));
```

### 4. 在框架核心代码中的使用

```cpp
// 后端实现示例
Tensor CpuBackend::conv(const Tensor& input, const Tensor& kernel) {
    // 参数验证
    TR_THROW_IF(input.device().type() != DeviceType::CPU,
                "CpuBackend::conv requires CPU tensors");

    // 功能检查
    TR_THROW_NOT_IMPLEMENTED("CPU backend convolution not yet implemented");

    // 类型检查
    if (input.dtype() != DType::FP32) {
        TR_THROW_TYPE_ERROR("CPU backend only supports FP32");
    }

    // 继续实现...
}
```

## 错误信息格式

### 标准格式
```
{ExceptionType}: {error_message} (File: {file_path}, Line: {line_number})
```

### 示例输出
```
FileNotFoundError: Cannot open model file: model.pth (File: R:\project\src\model.cpp, Line: 127)
ValueError: Invalid kernel size: 4, must be 1, 3, 5, or 7 (File: R:\project\src\conv.cpp, Line: 89)
NotImplementedError: CUDA backend INT8 support not yet implemented (File: R:\project\src\cuda_backend.cpp, Line: 234)
```

## 最佳实践

### 1. 异常类型选择指南

- **FileNotFoundError**: 文件、路径、目录相关问题
- **NotImplementedError**: 临时标记未实现功能
- **ValueError**: 参数值验证、范围检查、格式验证
- **IndexError**: 数组、张量、容器的索引访问
- **TypeError**: 数据类型、模板参数类型检查
- **ZeroDivisionError**: 数学运算中的除零检查
- **TRException**: 通用错误、向后兼容

### 2. 错误消息编写规范

**好的错误消息**:
```cpp
TR_THROW_VALUE_ERROR("Invalid kernel size: " + std::to_string(size) +
                    ", must be 1, 3, 5, or 7");
```

**包含信息的错误消息**:
- 描述问题本身
- 提供相关的值或上下文
- 给出解决方案或期望值范围

### 3. 异常处理策略

```cpp
// 推荐的处理顺序：具体 -> 抽象
try {
    operation();
} catch (const tr::FileNotFoundError& e) {
    // 最具体的处理
    handle_file_error(e);
} catch (const tr::TRException& e) {
    // 兜底处理
    handle_general_error(e);
}
```

## 迁移指南

### 从V1.00.00升级到V1.10.00

**无需修改的代码**:
```cpp
// 这些代码继续正常工作
TR_THROW("Error message");
TR_THROW_IF(condition, "Error message");
```

**可选升级**:
```cpp
// 旧代码
TR_THROW("Invalid parameter value");

// 新代码 (更精确)
TR_THROW_VALUE_ERROR("Invalid parameter value: " + std::to_string(value));
```

### 渐进式采用策略

1. **第一阶段**: 继续使用现有宏，验证兼容性
2. **第二阶段**: 新代码使用类型化宏
3. **第三阶段**: 逐步重构关键路径使用精确异常类型

## 性能考虑

- **内存开销**: 每个异常对象约100-200字节
- **构建开销**: 延迟构建，只在`what()`被调用时才格式化消息
- **类型安全**: 编译时检查，零运行时类型检查开销
- **兼容性**: 无性能回归，旧代码保持原有性能

## 示例代码

### 完整示例程序

```cpp
#include "tech_renaissance/utils/tr_exception.h"
#include <iostream>

void demonstrate_exceptions() {
    // 1. 文件错误
    try {
        std::ifstream file("nonexistent.txt");
        if (!file.good()) {
            TR_THROW_FILE_NOT_FOUND("Configuration file not found");
        }
    } catch (const tr::FileNotFoundError& e) {
        std::cout << "捕获文件错误: " << e.what() << std::endl;
    }

    // 2. 参数验证
    try {
        int kernel_size = 4;  // 无效值
        if (kernel_size != 1 && kernel_size != 3 && kernel_size != 5 && kernel_size != 7) {
            TR_THROW_VALUE_ERROR("Invalid kernel size: " + std::to_string(kernel_size));
        }
    } catch (const tr::ValueError& e) {
        std::cout << "捕获参数错误: " << e.what() << std::endl;
    }

    // 3. 向后兼容
    try {
        TR_THROW("传统异常消息");
    } catch (const tr::TRException& e) {
        std::cout << "捕获通用异常: " << e.what() << std::endl;
    }
}

int main() {
    demonstrate_exceptions();
    return 0;
}
```

## 相关文件

- **头文件**: `include/tech_renaissance/utils/tr_exception.h`
- **实现文件**: `src/utils/tr_exception.cpp`
- **测试文件**: `tests/unit_tests/test_tr_exception.cpp`
- **使用示例**: 参见框架各模块中的错误处理代码

---

**注意**: 本异常系统专为技术觉醒框架设计，强调类型安全、调试友好和向后兼容性。在使用时请遵循框架的错误处理规范。
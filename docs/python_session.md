# PythonSession Python会话管理器API文档

**版本**: V1.25.1
**日期**: 2025-10-31
**作者**: 技术觉醒团队
**所属系列**: utils

## 概述

PythonSession类是技术觉醒框架的Python进程管理工具，实现了C++与Python的实时交互。它通过临时文件通信机制，支持张量数据的传输、PyTorch计算任务的执行以及结果的回传，为框架提供了强大的Python生态集成能力。

## 特性

- **进程管理**: 完整的Python进程生命周期管理
- **张量传输**: 支持TSR格式的张量数据双向传输
- **简化API**: 提供直观的`calculate`方法执行PyTorch计算
- **异步通信**: 基于临时文件的异步消息传递
- **错误处理**: 完整的异常处理和超时机制
- **会话隔离**: 每个会话拥有独立的临时目录
- **单目运算支持**: 9种单目运算验证，支持CPU后端结果对比（V1.25.1新增）
- **静默模式**: Python服务器调试输出可控，支持清洁的测试环境（V1.25.1新增）

## 快速开始

### 简化的计算API（推荐）

```cpp
#include "tech_renaissance/utils/python_session.h"

using namespace tr;

// 创建并启动Python会话
PythonSession session("default", "compute", true);
session.start();

// 创建测试张量
Tensor a = Tensor::randn(Shape(1024, 2048), 42);
Tensor b = Tensor::randn(Shape(2048, 512), 42);

// 使用简化API执行矩阵乘法
Tensor result = session.calculate("matmul", a, b);

// 清理会话
session.please_exit();
```

### 传统的通信API

```cpp
// 发送张量到Python
session.send_tensor(a, "a");
session.send_tensor(b, "b");

// 执行计算命令
Tensor result = session.fetch_tensor(R"({"cmd": "matmul", "params": "a,b"})", 10000);
```

## API参考

### 构造函数

#### `PythonSession(const std::string& script_path = "default", const std::string& session_id = "default", bool quiet_mode = false)`

创建Python会话管理器实例。

```cpp
PythonSession session("default", "my_session", true);
```

**参数**:
- `script_path` (const std::string&): Python脚本路径，"default"使用内置脚本
- `session_id` (const std::string&): 会话唯一标识符
- `quiet_mode` (bool): 是否启用静默模式，禁用INFO级日志

---

### 会话管理方法

#### `void start(int warmup_time = 300)`

启动Python脚本进程。

```cpp
session.start();  // 使用默认300ms预热时间
session.start(500);  // 使用500ms预热时间
```

**参数**:
- `warmup_time` (int): 预热等待时间（毫秒）

**异常**:
- `TRException`: 如果Python进程启动失败

#### `bool is_alive()`

检查Python进程是否仍在运行。

```cpp
if (session.is_alive()) {
    std::cout << "Python session is running" << std::endl;
}
```

**返回值**:
- `bool`: 如果进程存活返回true，否则返回false

#### `void terminate()`

强制终止Python进程。

```cpp
session.terminate();
```

#### `void please_exit(uint32_t timeout_ms = 10000, bool ensure = true)`

优雅地请求Python进程退出。

```cpp
session.please_exit(10000, true);  // 等待10秒并确保退出
```

**参数**:
- `timeout_ms` (uint32_t): 超时时间（毫秒）
- `ensure` (bool): 是否确保进程退出

---

### 状态查询方法

#### `bool is_ready() const`

检查Python是否已就绪可以接收请求。

#### `bool is_busy() const`

检查Python是否正在处理请求。

#### `bool new_response() const`

检查是否有新的响应可读。

#### `bool wait_until_ready(uint32_t timeout_ms = 10000) const`

等待Python就绪。

#### `bool wait_until_ok(uint32_t timeout_ms = 10000) const`

等待Python返回OK状态。

---

### 消息通信方法

#### `void send_request(const std::string& msg) const`

发送请求消息到Python。

```cpp
session.send_request(R"({"cmd": "status"})");
```

#### `std::string read_response() const`

直接读取响应，不检查状态。

```cpp
std::string response = session.read_response();
```

#### `std::string wait_for_response(uint32_t timeout_ms = 10000) const`

等待并读取响应。

```cpp
std::string response = session.wait_for_response(5000);
```

#### `std::string fetch_response(const std::string& msg, uint32_t timeout_ms = 10000) const`

发送请求并等待响应。

```cpp
std::string response = session.fetch_response(R"({"cmd": "ping"})", 3000);
```

---

### 张量操作方法

#### `void send_tensor(const Tensor& tensor, const std::string& tag) const`

发送张量到Python。

```cpp
Tensor a = Tensor::randn(Shape(1024, 1024), 42);
session.send_tensor(a, "input_a");
```

**参数**:
- `tensor` (const Tensor&): 要发送的张量
- `tag` (const std::string&): 张量标签，用于Python端识别

#### `Tensor fetch_tensor(const std::string& msg, uint32_t timeout_ms = 10000) const`

发送命令并获取张量结果。

```cpp
Tensor result = session.fetch_tensor(R"({"cmd": "matmul", "params": "a,b"})", 10000);
```

#### `Tensor wait_for_tensor(uint32_t timeout_ms = 10000) const`

等待张量响应。

```cpp
Tensor result = session.wait_for_tensor(5000);
```

---

### 简化计算API

#### `Tensor calculate(const std::string& msg, const Tensor& tensor_a, uint32_t timeout_ms = 10000) const`

执行单张量计算。

```cpp
Tensor result = session.calculate("relu", input_tensor);
```

**参数**:
- `msg` (const std::string&): 计算命令
- `tensor_a` (const Tensor&): 输入张量
- `timeout_ms` (uint32_t): 超时时间（毫秒）

**返回值**:
- `Tensor`: 计算结果张量

#### `Tensor calculate(const std::string& msg, const Tensor& tensor_a, const Tensor& tensor_b, uint32_t timeout_ms = 10000) const`

执行双张量计算。

```cpp
Tensor result = session.calculate("matmul", tensor_a, tensor_b);
```

**参数**:
- `msg` (const std::string&): 计算命令
- `tensor_a` (const Tensor&): 第一个输入张量
- `tensor_b` (const Tensor&): 第二个输入张量
- `timeout_ms` (uint32_t): 超时时间（毫秒）

**返回值**:
- `Tensor`: 计算结果张量

#### `Tensor calculate(const std::string& msg, const Tensor& tensor_a, const Tensor& tensor_b, const Tensor& tensor_c, uint32_t timeout_ms = 10000) const`

执行三张量计算。

```cpp
Tensor result = session.calculate("linear", input_tensor, weight_tensor, bias_tensor);
```

**参数**:
- `msg` (const std::string&): 计算命令
- `tensor_a` (const Tensor&): 第一个输入张量
- `tensor_b` (const Tensor&): 第二个输入张量
- `tensor_c` (const Tensor&): 第三个输入张量
- `timeout_ms` (uint32_t): 超时时间（毫秒）

**返回值**:
- `Tensor`: 计算结果张量

---

### 辅助方法

#### `std::string session_dir() const`

获取会话临时目录路径。

```cpp
std::string dir = session.session_dir();
```

#### `const std::string& session_id() const`

获取会话ID。

```cpp
std::string id = session.session_id();
```

#### `void set_quiet_mode(bool quiet_mode)`

设置静默模式。

```cpp
session.set_quiet_mode(true);  // 启用静默模式
```

---

## 使用示例

### 矩阵乘法计算

```cpp
#include "tech_renaissance.h"
#include <iostream>

using namespace tr;

int main() {
    try {
        // 创建并启动会话
        PythonSession session("default", "matmul_test", true);
        session.start();

        // 创建输入张量
        Tensor a = Tensor::randn(Shape(1024, 2048), 42);
        Tensor b = Tensor::randn(Shape(2048, 512), 42);

        // 执行矩阵乘法
        std::cout << "Executing matrix multiplication..." << std::endl;
        Tensor result = session.calculate("matmul", a, b);

        std::cout << "Result shape: " << result.shape().to_string() << std::endl;
        std::cout << "Result device: " << result.device().to_string() << std::endl;

        // 清理会话
        session.please_exit();

        std::cout << "Computation completed successfully!" << std::endl;

    } catch (const TRException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

### 批量计算验证

```cpp
void batch_computation_verification() {
    PythonSession session("default", "batch_verify", true);
    session.start();

    // 创建测试数据
    std::vector<std::pair<Tensor, Tensor>> test_pairs;
    for (int i = 0; i < 5; ++i) {
        Tensor a = Tensor::randn(Shape(512, 1024), i + 1000);
        Tensor b = Tensor::randn(Shape(1024, 256), i + 2000);
        test_pairs.emplace_back(a, b);
    }

    // 批量计算
    for (size_t i = 0; i < test_pairs.size(); ++i) {
        std::cout << "Test case " << (i + 1) << ":" << std::endl;

        auto& [a, b] = test_pairs[i];
        Tensor pytorch_result = session.calculate("matmul", a, b);

        // 本地计算验证
        auto cuda_backend = BackendManager::get_cuda_backend();
        Tensor local_result = Tensor::empty(pytorch_result.shape(), pytorch_result.dtype(), CUDA[0]);

        // 转换并计算
        Tensor cuda_a = cuda_backend->from_cpu(a);
        Tensor cuda_b = cuda_backend->from_cpu(b);
        cuda_backend->mm(local_result, cuda_a, cuda_b);

        // 比较结果
        Tensor pytorch_on_cpu = cuda_backend->to_cpu(pytorch_result);
        bool is_close = cuda_backend->is_close(local_result, pytorch_on_cpu, 1e-4f);

        std::cout << "  Result shape: " << pytorch_result.shape().to_string() << std::endl;
        std::cout << "  Results match: " << (is_close ? "YES" : "NO") << std::endl;
    }

    session.please_exit();
}
```

### 单目运算验证示例（V1.25.1新增）

```cpp
#include "tech_renaissance.h"
#include <iostream>
#include <vector>

using namespace tr;

void comprehensive_unary_verification() {
    try {
        // 初始化后端和会话
        auto cpu_backend = BackendManager::get_cpu_backend();
        PythonSession session("default", "unary_verify", true);  // 静默模式
        session.start();

        // 定义所有单目运算
        std::vector<std::string> operations = {
            "zeros_like", "ones_like", "relu", "sign",
            "square", "sqrt", "abs", "negative", "reciprocal"
        };

        int passed = 0;
        int total = operations.size() * 2;  // 每个运算测试非原地和原地版本

        std::cout << "=== 单目运算验证测试 ===" << std::endl;

        for (const auto& op : operations) {
            std::cout << "\n--- Testing " << op << " ---" << std::endl;

            // 为sqrt生成正数张量，其他运算用普通张量
            Tensor input;
            if (op == "sqrt") {
                input = Tensor::uniform(Shape(2, 3, 4, 5), 0.1f, 10.0f, 42);
            } else {
                input = Tensor::uniform(Shape(2, 3, 4, 5), -1.0f, 1.0f, 42);
            }

            // TEST 1: 非原地运算验证
            try {
                // C++计算
                Tensor cpu_result;
                if (op == "zeros_like") {
                    cpu_result = cpu_backend->zeros_like(input);
                } else if (op == "ones_like") {
                    cpu_result = cpu_backend->ones_like(input);
                } else if (op == "relu") {
                    cpu_result = cpu_backend->relu(input);
                } else if (op == "sign") {
                    cpu_result = cpu_backend->sign(input);
                } else if (op == "square") {
                    cpu_result = cpu_backend->square(input);
                } else if (op == "sqrt") {
                    cpu_result = cpu_backend->sqrt(input);
                } else if (op == "abs") {
                    cpu_result = cpu_backend->abs(input);
                } else if (op == "negative") {
                    cpu_result = cpu_backend->negative(input);
                } else if (op == "reciprocal") {
                    cpu_result = cpu_backend->reciprocal(input);
                }

                // PyTorch验证
                Tensor pytorch_result = session.calculate(op, input);

                // 结果对比
                bool is_close = cpu_backend->is_close(cpu_result, pytorch_result, 5e-5f);
                std::cout << "  Non-inplace: " << (is_close ? "PASSED" : "FAILED") << std::endl;

                if (is_close) passed++;

            } catch (const std::exception& e) {
                std::cout << "  Non-inplace: FAILED (Exception: " << e.what() << ")" << std::endl;
            }

            // TEST 2: 原地运算验证
            try {
                // 创建输入副本用于原地测试
                Tensor inplace_input = cpu_backend->copy(input);

                // C++原地计算
                if (op == "zeros_like") {
                    cpu_backend->zeros_inplace(inplace_input);
                } else if (op == "ones_like") {
                    cpu_backend->ones_inplace(inplace_input);
                } else if (op == "relu") {
                    cpu_backend->relu_inplace(inplace_input);
                } else if (op == "sign") {
                    cpu_backend->sign_inplace(inplace_input);
                } else if (op == "square") {
                    cpu_backend->square_inplace(inplace_input);
                } else if (op == "sqrt") {
                    cpu_backend->sqrt_inplace(inplace_input);
                } else if (op == "abs") {
                    cpu_backend->abs_inplace(inplace_input);
                } else if (op == "negative") {
                    cpu_backend->negative_inplace(inplace_input);
                } else if (op == "reciprocal") {
                    cpu_backend->reciprocal_inplace(inplace_input);
                }

                // PyTorch验证
                Tensor pytorch_result = session.calculate(op, input);

                // 结果对比
                bool is_close = cpu_backend->is_close(inplace_input, pytorch_result, 5e-5f);
                std::cout << "  Inplace:     " << (is_close ? "PASSED" : "FAILED") << std::endl;

                if (is_close) passed++;

            } catch (const std::exception& e) {
                std::cout << "  Inplace:     FAILED (Exception: " << e.what() << ")" << std::endl;
            }
        }

        // 输出总结
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "验证总结: " << passed << "/" << total << " 测试通过" << std::endl;
        std::cout << "成功率: " << (100.0 * passed / total) << "%" << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        session.please_exit();

    } catch (const TRException& e) {
        std::cerr << "验证失败: " << e.what() << std::endl;
    }
}
```

### 错误处理示例

```cpp
void robust_computation() {
    PythonSession session("default", "robust_test", true);

    try {
        session.start();

        // 检查会话状态
        if (!session.wait_until_ready(5000)) {
            throw TRException("Python session failed to start");
        }

        Tensor a = Tensor::randn(Shape(256, 512), 42);
        Tensor b = Tensor::randn(Shape(512, 128), 42);

        // 执行计算，设置较短超时
        Tensor result = session.calculate("matmul", a, b, 5000);

        std::cout << "Computation successful!" << std::endl;

    } catch (const TRException& e) {
        std::cerr << "Computation failed: " << e.what() << std::endl;

        // 确保清理
        if (session.is_alive()) {
            session.terminate();
        }
    }
}
```

---

## C++与Python集成架构（V1.25.1）

### 整体架构设计

PythonSession实现了C++后端与Python环境的深度集成，为技术觉醒框架提供了PyTorch生态系统的无缝接入能力。

```
┌─────────────────┐    TSR文件传输    ┌──────────────────┐
│   C++ 后端      │ ←────────────→  │   Python 服务器   │
│                 │                 │                  │
│ • CPU Backend   │                 │ • PyTorch 计算   │
│ • CUDA Backend  │                 │ • 张量操作       │
│ • 张量管理      │                 │ • 验证与对比     │
└─────────────────┘                 └──────────────────┘
```

### C++ 端特性

#### 1. 会话管理
- **进程生命周期**: 完整管理Python进程的启动、运行、退出
- **资源隔离**: 每个会话独立临时目录，避免冲突
- **异常安全**: RAII模式确保资源正确清理

#### 2. 张量传输
- **TSR格式**: 专有二进制格式，高效存储张量数据
- **类型转换**: 自动处理C++与PyTorch数据类型映射
- **内存优化**: 64字节对齐，支持SIMD访问

#### 3. 计算验证
- **单目运算验证**: 9种CPU单目运算与PyTorch结果对比
- **精度控制**: 可配置的数值比较容差
- **性能测试**: 自动化基准测试支持

### Python 端特性

#### 1. 服务器架构
```python
# 基于TechRenaissanceServer的继承架构
class SimpleHelloServer(TechRenaissanceServer):
    def main_logic(self, command: str, parameters: str) -> bool:
        # 处理各种计算命令
        if command.lower() == 'relu':
            tensor = self.get_tensors(parameters, 1)
            result = torch.relu(tensor)
            self.send_tensors(result)
```

#### 2. 静默模式（V1.25.1新增）
```python
# 调试输出可控
DEBUG_MODE = False  # 关闭[PYTHON_DEBUG]输出

if DEBUG_MODE:
    print(f"[PYTHON_DEBUG] Processing relu command")
```

#### 3. 智能轮询
- **自适应频率**: 根据等待时间动态调整轮询间隔
- **性能优化**: 减少不必要的CPU占用
- **响应速度**: 保持快速响应能力

### 通信协议

#### 请求格式
```json
{
    "cmd": "relu",
    "params": "input_tensor"
}
```

#### 响应格式
```json
{
    "cmd": "relu",
    "params": "result"
}
```

### 使用模式

#### 1. 验证模式
```cpp
// C++实现 vs PyTorch验证
Tensor cpu_result = cpu_backend->relu(input);
Tensor pytorch_result = session.calculate("relu", input);
bool is_close = cpu_backend->is_close(cpu_result, pytorch_result, 5e-5f);
```

#### 2. 计算模式
```cpp
// 直接使用PyTorch计算能力
Tensor complex_result = session.calculate("custom_operation", input1, input2);
```

### 性能特征

- **启动延迟**: ~300ms（Python进程初始化）
- **传输开销**: ~1-5ms（取决于张量大小）
- **计算延迟**: 取决于PyTorch操作复杂度
- **内存占用**: 临时TSR文件 + Python进程内存

---

## 支持的计算命令

### 矩阵运算

| 命令 | 描述 | 参数 | 示例 |
|------|------|------|------|
| "matmul" | 矩阵乘法 | 2个张量 | `calculate("matmul", A, B)` |
| "add" | 张量加法 | 2个张量 | `calculate("add", A, B)` |

### 单目运算（V1.25.1新增）

| 命令 | 描述 | 参数 | 示例 | PyTorch对应 |
|------|------|------|------|------------|
| "zeros_like" | 全零张量 | 1个张量 | `calculate("zeros_like", X)` | `torch.zeros_like(X)` |
| "ones_like" | 全1张量 | 1个张量 | `calculate("ones_like", X)` | `torch.ones_like(X)` |
| "relu" | ReLU激活 | 1个张量 | `calculate("relu", X)` | `torch.relu(X)` |
| "sign" | 符号函数 | 1个张量 | `calculate("sign", X)` | `torch.sign(X)` |
| "square" | 平方运算 | 1个张量 | `calculate("square", X)` | `torch.square(X)` |
| "sqrt" | 平方根 | 1个张量 | `calculate("sqrt", X)` | `torch.sqrt(X)` |
| "abs" | 绝对值 | 1个张量 | `calculate("abs", X)` | `torch.abs(X)` |
| "negative" | 相反数 | 1个张量 | `calculate("negative", X)` | `torch.negative(X)` |
| "reciprocal" | 倒数 | 1个张量 | `calculate("reciprocal", X)` | `torch.reciprocal(X)` |

### 传统运算（向后兼容）

| 命令 | 描述 | 参数 | 示例 |
|------|------|------|------|
| "mul" | 张量乘法 | 2个张量 | `calculate("mul", A, B)` |
| "sigmoid" | Sigmoid激活 | 1个张量 | `calculate("sigmoid", X)` |
| "linear" | 线性变换 | 3个张量 | `calculate("linear", X, W, b)` |

---

## 最佳实践

### 1. 会话生命周期管理

```cpp
// 使用RAII模式管理会话
{
    PythonSession session("default", "auto_cleanup", true);
    session.start();

    // 执行计算...
    Tensor result = session.calculate("matmul", a, b);

    // 自动调用析构函数清理会话
    // 或者显式清理
    session.please_exit();
}
```

### 2. 错误处理

```cpp
try {
    PythonSession session("default", "safe_test", true);
    session.start();

    if (!session.wait_until_ready(10000)) {
        throw TRException("Session initialization timeout");
    }

    // 计算逻辑...

} catch (const TRException& e) {
    std::cerr << "Session error: " << e.what() << std::endl;
    // 处理错误...
}
```

### 3. 超时设置

```cpp
// 根据计算复杂度设置合适的超时
Tensor small_result = session.calculate("add", a, b, 1000);        // 1秒
Tensor medium_result = session.calculate("matmul", a, b, 10000);    // 10秒
Tensor large_result = session.calculate("matmul", big_a, big_b, 60000); // 60秒
```

### 4. 内存管理

```cpp
// 及时清理不需要的张量
{
    Tensor temp = session.calculate("matmul", a, b);
    // 使用temp...

    // temp自动析构，释放内存
}
```

---

## 注意事项

1. **进程管理**: 每个PythonSession都会创建独立的Python进程，使用完毕后务必调用`please_exit()`或`terminate()`清理
2. **临时文件**: 会话会在系统临时目录创建临时文件，正常退出时会自动清理
3. **内存使用**: 大张量的传输会占用临时存储空间，注意系统磁盘空间
4. **超时设置**: 根据计算复杂度合理设置超时时间，避免无限等待
5. **Python环境**: 确保系统安装了支持的Python环境和必要的库（PyTorch等）
6. **线程安全**: 每个PythonSession实例不是线程安全的，多线程使用时需要同步

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| V1.25.1 | 2025-10-31 | **重大更新**: 新增9种单目运算支持，静默模式，C++与Python集成架构文档 |
| | | - 新增单目运算：zeros_like, ones_like, relu, sign, square, sqrt, abs, negative, reciprocal |
| | | - 静默模式：DEBUG_MODE配置控制调试输出 |
| | | - 集成架构：完整的C++与Python协同工作说明 |
| | | - 验证示例：综合单目运算验证代码示例 |
| V1.25.0 | 2025-10-27 | 初始版本，基础Python会话管理功能 |

---

## 相关文档

- [张量操作文档](tensor.md)
- [后端接口文档](backend.md)
- [CUDA后端文档](cuda_backend.md)
- [TSR格式文档](tsr_format.md)
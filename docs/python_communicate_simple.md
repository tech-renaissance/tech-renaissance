# 技术觉醒框架 - Python通信机制（简版）

## 📋 **快速开始**

技术觉醒框架提供升级版C++与Python跨进程通信机制，V1.20.01版本实现了**完整张量数据交换**的高效通信系统。

### **核心特性（V1.20.01革命性升级）**
- 🆕 **TSR张量传输**：真正的二进制张量数据交换
- 🆕 **完整API体系**：fetch_tensor, wait_for_tensor, send_tensor等
- 🆕 **多种通信模式**：支持同步/异步张量运算
- 🆕 **真实矩阵运算**：支持matmul, add等复杂运算
- ✅ **V1.19.02特性**：原子操作机制，避免读写冲突
- ✅ **V1.19.02特性**：智能轮询频率，32ms→1024ms自适应
- ✅ **V1.19.02特性**：标准JSON协议，零解析错误
- ✅ 进程隔离：Python崩溃不影响C++
- ✅ 跨平台兼容：Windows/Linux统一接口
- ✅ 零依赖：仅使用C++标准库

---

## 🚀 **5分钟快速上手**

### **1. 张量矩阵乘法示例（V1.20.01核心功能）**

```cpp
#include "tech_renaissance.h"
using namespace tr;

int main() {
    // 创建会话
    PythonSession session("../python/module/python_server.py", "matmul_demo");
    session.start();
    session.wait_until_ready(5000);

    // 创建测试张量：4x3 矩阵 × 3x5 矩阵 = 4x5 矩阵
    Tensor tensor_a = Tensor::full(Shape(4, 3), 1.5f, DType::FP32, tr::CPU);
    Tensor tensor_b = Tensor::full(Shape(3, 5), 2.0f, DType::FP32, tr::CPU);

    // V1.20.01核心API：发送张量到Python
    session.send_tensor(tensor_a, "input_a");
    session.send_tensor(tensor_b, "input_b");

    // 触发矩阵乘法计算
    session.send_request("matmul input_a input_b");

    // 等待并获取计算结果
    Tensor result = session.wait_for_tensor(10000);

    // 验证结果
    std::cout << "矩阵乘法结果形状: " << result.shape().to_string() << std::endl;
    result.print("result");

    // 优雅退出
    session.please_exit(5000, true);

    return 0;
}
```

**对应的Python服务器：**

```python
# python_server.py
from tech_renaissance import TechRenaissanceServer

class MatrixMathServer(TechRenaissanceServer):
    def main_logic(self, command: str, parameters: str) -> bool:
        if command.lower() == 'matmul':
            tensors = self.get_tensors(parameters, 2)
            if tensors is None:
                self.write_response('', 'invalid')
            else:
                result = torch.mm(tensors[0], tensors[1])
                self.send_tensors(result)
                self.write_response('', 'ok')
            return True
        return False

def main():
    import sys
    server = MatrixMathServer(debug=False)
    server.run(sys.argv[1])

if __name__ == "__main__":
    main()
```

### **2. Hello World示例**

```cpp
#include "tech_renaissance.h"
using namespace tr;

int main() {
    PythonSession session("default", "hello_test");
    session.start();
    session.wait_until_ready(5000);

    // 发送请求并等待响应
    std::string response = session.fetch_response("hello world", 3000);
    std::cout << "Python响应: " << response << std::endl;

    session.please_exit(5000, true);
    return 0;
}
```

**Python服务器：**

```python
from tech_renaissance import TechRenaissanceServer

class HelloServer(TechRenaissanceServer):
    def main_logic(self, command: str, parameters: str) -> bool:
        if command.lower() == 'hello':
            self.write_response('', f'Hello {parameters.title()}')
            return True
        return False

def main():
    import sys
    server = HelloServer(debug=False)
    server.run(sys.argv[1])

if __name__ == "__main__":
    main()
```

---

## 🔧 **核心API速查**

### **PythonSession类（C++端）**

#### **构造函数**
```cpp
PythonSession(script_path="default", session_id="default", quiet_mode=false)
```

#### **进程管理**
```cpp
session.start()                          // 启动Python进程
session.is_alive()                       // 检查进程状态
session.please_exit(timeout_ms, ensure)  // 优雅退出
session.terminate()                      // 强制终止
```

#### **状态查询**
```cpp
session.is_ready()                       // 检查是否就绪
session.is_busy()                        // 检查是否繁忙
session.wait_until_ready(timeout_ms)     // 等待就绪
session.new_response()                   // 检查是否有新响应
```

#### **文本通信**
```cpp
session.send_request(message)            // 发送文本请求
session.wait_for_response(timeout_ms)     // 等待文本响应
session.fetch_response(message, timeout_ms) // 发送并等待响应
```

#### **张量通信（V1.20.01新增）**
```cpp
session.send_tensor(tensor, tag)         // 发送张量
session.wait_for_tensor(timeout_ms)      // 等待张量结果
session.fetch_tensor(message, timeout_ms) // 发送并等待张量
```

### **TechRenaissanceServer类（Python端）**

#### **核心方法**
```python
def main_logic(self, command: str, parameters: str) -> bool:
    """业务逻辑方法（需要重写）"""
    pass

def send_tensors(self, *tensors) -> None:
    """发送张量到C++"""
    pass

def get_tensors(self, params: str, count: int) -> List[torch.Tensor]:
    """从C++接收张量"""
    pass

def write_response(self, command: str, result: str) -> None:
    """写入响应"""
    pass
```

---

## 📁 **文件协议**

### **会话目录结构**
```
workspace/python_session/
└── tr_session_{session_id}/
    ├── request.json     # C++→Python请求
    ├── response.json    # Python→C++响应
    ├── status.txt       # 状态文件
    ├── input_a.tsr      # 输入张量A
    ├── input_b.tsr      # 输入张量B
    └── result_0.tsr     # 结果张量
```

### **JSON协议格式**

#### **请求格式（request.json）**
```json
{
    "command": "matmul",
    "parameters": "input_a input_b",
    "timestamp": 1699123456789
}
```

#### **响应格式（response.json）**
```json
{
    "command": "matmul",
    "result": "ok",
    "message": "Matrix multiplication completed",
    "timestamp": 1699123456790
}
```

#### **状态文件（status.txt）**
```
ready      # 就绪，等待请求
busy       # 正在处理请求
ok         # 处理成功
error: ... # 处理失败
exiting    # 正在退出
```

---

## 🛠️ **编译配置**

### **启用Python会话支持**
```bash
cmake .. -DTR_BUILD_PYTHON_SESSION=ON
cmake --build . --config Release
```

### **禁用Python会话支持**
```bash
cmake .. -DTR_BUILD_PYTHON_SESSION=OFF
```

### **条件编译**
```cpp
#ifdef TR_BUILD_PYTHON_SESSION
#include "tech_renaissance/utils/python_session.h"
// 使用PythonSession
#endif
```

---

## 🐛 **常见问题**

### **1. Python进程启动失败**
```cpp
// 检查脚本路径
if (!std::filesystem::exists(script_path)) {
    std::cerr << "Python脚本不存在: " << script_path << std::endl;
    return 1;
}
```

### **2. 通信超时**
```cpp
// 增加超时时间
if (!session.wait_until_ready(15000)) {  // 15秒
    std::cerr << "Python启动超时" << std::endl;
    return 1;
}
```

### **3. 张量格式错误**
```python
# 确保张量在CPU上
tensor = tensor.cpu()
# 确保数据类型支持FP32或INT8
if tensor.dtype not in [torch.float32, torch.int8]:
    tensor = tensor.float()
```

---

## 📈 **性能参考**

| 操作类型 | 平均延迟 | 吞吐量 | 内存使用 |
|----------|----------|--------|----------|
| 文本通信 | 15ms | 67 msg/s | <1MB |
| 张量发送 | 50ms | 20 tensor/s | 2×张量大小 |
| 张量接收 | 45ms | 22 tensor/s | 2×张量大小 |
| 矩阵乘法 | 200ms | 5 op/s | 3×张量大小 |

### **优化建议**
- **批量传输**：一次发送多个小张量
- **合理超时**：文本5秒，张量10秒
- **错误处理**：捕获异常并优雅退出
- **资源管理**：使用RAII自动清理

---

## 🔮 **进阶功能**

### **异步通信**
```cpp
// 异步等待结果
auto future = std::async(std::launch::async, [&session]() {
    return session.wait_for_tensor(10000);
});

// 做其他事情...
Tensor result = future.get();
```

### **多张量处理**
```cpp
// 批量发送
std::vector<Tensor> tensors = {a, b, c};
std::vector<std::string> tags = {"input_a", "input_b", "input_c"};
for (size_t i = 0; i < tensors.size(); ++i) {
    session.send_tensor(tensors[i], tags[i]);
}

// 触发批量处理
session.send_request("batch_process input_a input_b input_c");
```

### **错误恢复**
```cpp
try {
    Tensor result = session.wait_for_tensor(5000);
} catch (const TRException& e) {
    std::cerr << "张量获取失败: " << e.what() << std::endl;

    // 重试机制
    session.terminate();
    session.start();
    session.wait_until_ready(5000);

    // 重新发送请求
    session.send_request("retry_command");
    result = session.wait_for_tensor(5000);
}
```

---

## 📚 **API完整参考**

### **PythonSession类方法**

| 方法 | 描述 | 参数 | 返回值 |
|------|------|------|--------|
| `start()` | 启动Python进程 | 无 | void |
| `is_alive()` | 检查进程状态 | 无 | bool |
| `wait_until_ready(ms)` | 等待就绪 | 超时(ms) | bool |
| `send_request(msg)` | 发送文本请求 | 消息字符串 | void |
| `fetch_response(msg, ms)` | 发送并等待响应 | 消息,超时 | string |
| `send_tensor(tensor, tag)` | 发送张量 | 张量,标签 | void |
| `wait_for_tensor(ms)` | 等待张量结果 | 超时(ms) | Tensor |
| `please_exit(ms, ensure)` | 优雅退出 | 超时,强制退出 | void |

### **TechRenaissanceServer类方法**

| 方法 | 描述 | 参数 | 返回值 |
|------|------|------|--------|
| `run(session_id)` | 运行服务器主循环 | 会话ID | void |
| `main_logic(cmd, params)` | 业务逻辑（重写） | 命令,参数 | bool |
| `send_tensors(*tensors)` | 发送张量到C++ | 张量列表 | void |
| `get_tensors(params, count)` | 从C++接收张量 | 参数字符串,数量 | 张量列表 |
| `write_response(cmd, result)` | 写入响应 | 命令,结果 | void |

---

**版本信息**：V1.20.01
**最后更新**：2025-10-31
**兼容性**：C++17, Python 3.8+, PyTorch 1.12+

*更多详细信息请参考 [python_communicate.md](python_communicate.md)*
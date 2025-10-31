# 技术觉醒框架 - Python通信机制详细方案

## 🎯 **方案概述**

本文档详细描述技术觉醒框架中已实现的**升级版跨进程临时文件通信机制**，用于C++主程序与Python脚本的实时交互。该方案采用**独立进程+文件通道+原子操作+智能轮询**的设计理念，在V1.20.01版本中完成了重大升级，实现了高效、安全、可靠的Python通信能力，支持真正的张量数据交换。

**设计目标：**
- ✅ C++主程序全程控制Python进程生命周期
- ✅ 双向张量数据交换（C++ ⇋ PyTorch，TSR格式支持）
- ✅ 进程级隔离，避免GIL限制和内存冲突
- ✅ 跨平台兼容（Windows/Linux）
- ✅ 零第三方依赖，仅使用C++标准库
- ✅ 可重构设计，可通过CMake选项完全移除
- ✅ 多种API模式：fetch_response, send_request+wait_for_tensor等
- 🆕 **V1.20.01新增**：TSR二进制张量格式传输支持
- 🆕 **V1.20.01新增**：完整张量通信API（send_tensor, fetch_tensor, wait_for_tensor）
- 🆕 **V1.20.01新增**：多种通信模式（同步/异步）支持
- ✅ **V1.19.02特性**：原子操作机制，避免读写冲突
- ✅ **V1.19.02特性**：智能轮询频率，兼顾功耗与效率
- ✅ **V1.19.02特性**："阅后即焚"响应文件管理
- ✅ **V1.19.02特性**：标准JSON格式通信协议

---

## 🏗️ **整体架构设计**

### **三层通信模型**

```
┌─────────────────────────────────────────────────────────────┐
│                    C++ 主进程                              │
│  ┌─────────────────┐    ┌──────────────────────────────┐   │
│  │   PythonSession│────│        Workspace目录         │   │
│  │   (会话管理器)   │    │  workspace/python_session/   │   │
│  └─────────────────┘    └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │ 临时文件通道
                              │ • request.json (控制指令)
                              │ • {tag}.tsr (TSR格式张量数据)
                              │ • {tag}.tsr (TSR格式计算结果)
                              │ • response.json (文本响应)
                              │ • status.txt (状态同步)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Python 进程                               │
│  ┌─────────────────┐    ┌──────────────────────────────┐   │
│  │ python_task_*   │────│        TempFileChannel       │   │
│  │ (业务处理脚本)   │    │      (文件读写模块)           │   │
│  └─────────────────┘    └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### **核心组件职责**

| 组件 | 职责 | 关键特性 |
|------|------|----------|
| **PythonSession** | C++端会话管理器 | 进程生命周期控制、文件I/O、状态同步 |
| **TechRenaissanceServer** | Python端服务器基类 | 请求解析、响应生成、张量处理 |
| **TempFileChannel** | 文件通道抽象 | 原子读写、状态管理、错误处理 |
| **Workspace目录** | 临时文件存储 | 会话隔离、自动清理、跨平台兼容 |

---

## 🔧 **PythonSession类详解**

### **类定义与核心功能**

```cpp
namespace tr {
class PythonSession {
public:
    // 构造函数
    PythonSession(const std::string& script_path = "default",
                  const std::string& session_id = "default",
                  bool quiet_mode = false);

    // 进程管理
    void start();                              // 启动Python脚本
    bool is_alive();                           // 检查Python是否仍在运行
    void terminate();                          // 强制终止
    void join();                               // 等待进程结束

    // 状态管理
    bool is_ready() const;                     // 检查Python是否已就绪
    bool is_busy() const;                      // 检查Python是否正在处理请求
    bool new_response() const;                  // 检查是否有新的响应
    bool wait_until_ready(uint32_t timeout_ms = 10000) const;
    bool wait_until_ok(uint32_t timeout_ms = 10000) const;

    // 文本通信
    void send_request(const std::string& msg) const;
    std::string read_response() const;         // 直接读取响应，不检查状态
    std::string wait_for_response(uint32_t timeout_ms = 10000) const;
    std::string fetch_response(const std::string& msg, uint32_t timeout_ms = 10000) const;

    // 张量通信（V1.20.01新增）
    void send_tensor(const Tensor& tensor, const std::string& tag) const;
    Tensor fetch_tensor(const std::string& msg, uint32_t timeout_ms = 10000) const;
    Tensor wait_for_tensor(uint32_t timeout_ms = 10000) const;

    // 会话管理
    std::string session_dir() const;
    const std::string& session_id() const;
    void set_quiet_mode(bool quiet_mode);
    void please_exit(uint32_t timeout_ms = 10000, bool ensure = true);
};
}
```

### **关键API详解**

#### **1. 进程生命周期管理**

```cpp
// 创建会话
PythonSession session("default", "matmul_test", false);

// 启动Python进程
session.start();

// 检查进程状态
if (session.is_alive()) {
    std::cout << "Python进程正在运行" << std::endl;
}

// 优雅退出
session.please_exit(10000, true);  // 10秒超时，确保退出

// 强制终止（必要时）
session.terminate();

// 等待进程结束
session.join();
```

#### **2. 文本通信API**

```cpp
// 方式1：发送请求并等待响应（推荐）
std::string response = session.fetch_response("hello world", 5000);

// 方式2：分离式发送和接收
session.send_request("compute something");
if (session.wait_until_ready(5000)) {
    std::string response = session.wait_for_response(3000);
}

// 方式3：直接读取（用于已知的响应）
if (session.new_response()) {
    std::string response = session.read_response();
}
```

#### **3. 张量通信API（V1.20.01新增）**

```cpp
// 发送张量到Python
Tensor tensor_a = Tensor::randn(Shape(1024, 1024));
session.send_tensor(tensor_a, "input_a");

Tensor tensor_b = Tensor::randn(Shape(1024, 1024));
session.send_tensor(tensor_b, "input_b");

// 触发计算
session.send_request("matmul input_a input_b");

// 等待并接收计算结果
Tensor result = session.wait_for_tensor(10000);
```

---

## 🐍 **Python端实现详解**

### **TechRenaissanceServer基类**

```python
class TechRenaissanceServer:
    """Python端服务器基类，提供通信基础设施"""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.session_id = None
        self.session_dir = None

    def run(self, session_id: str) -> None:
        """运行服务器主循环"""
        self.session_id = session_id
        self.setup_session_dir()

        # 写入就绪状态
        self.write_status("ready")

        # 主循环
        while self.should_continue():
            try:
                if self.process_request():
                    self.write_status("ready")
            except Exception as e:
                self.write_status(f"error: {str(e)}")

    def main_logic(self, command: str, parameters: str) -> bool:
        """子类需要重写的业务逻辑方法"""
        raise NotImplementedError

    def send_tensors(self, *tensors) -> None:
        """发送张量数据到C++"""
        for i, tensor in enumerate(tensors):
            filename = f"result_{i}.tsr"
            self.write_tsr_file(tensor, filename)

    def get_tensors(self, params: str, count: int) -> Optional[List[torch.Tensor]]:
        """从C++接收张量数据"""
        try:
            tensor_names = params.split()
            if len(tensor_names) != count:
                return None

            tensors = []
            for name in tensor_names:
                tensor = self.read_tsr_file(f"{name}.tsr")
                tensors.append(tensor)
            return tensors
        except Exception:
            return None
```

### **自定义服务器实现示例**

```python
class MatrixMathServer(TechRenaissanceServer):
    """矩阵数学运算服务器"""

    def main_logic(self, command: str, parameters: str) -> bool:
        if command.lower() == 'matmul':
            tensors = self.get_tensors(parameters, 2)
            if tensors is None:
                self.write_response('', 'invalid')
            else:
                result = torch.mm(tensors[0], tensors[1])
                self.send_tensors(result)
                self.write_response('', 'ok')

        elif command.lower() == 'add':
            tensors = self.get_tensors(parameters, 2)
            if tensors is None:
                self.write_response('', 'invalid')
            else:
                result = tensors[0] + tensors[1]
                self.send_tensors(result)
                self.write_response('', 'ok')

        elif command.lower() == 'hello':
            self.write_response('', f'Hello {parameters.title()}')

        else:
            return False
        return True
```

### **主函数入口**

```python
def main():
    if len(sys.argv) != 2:
        print("Usage: python_server.py <session_id>")
        sys.exit(1)

    session_id = sys.argv[1]
    server = MatrixMathServer(debug=False)
    server.run(session_id)

if __name__ == "__main__":
    main()
```

---

## 📁 **文件系统协议**

### **会话目录结构**

```
workspace/python_session/
└── tr_session_{session_id}/
    ├── request.json          # C++→Python请求文件
    ├── response.json         # Python→C++响应文件
    ├── status.txt            # 状态同步文件
    ├── input_a.tsr           # 输入张量A（TSR格式）
    ├── input_b.tsr           # 输入张量B（TSR格式）
    ├── result_0.tsr          # 结果张量（TSR格式）
    └── ...                   # 其他临时文件
```

### **文件协议详解**

#### **1. request.json（C++→Python请求）**

```json
{
    "command": "matmul",
    "parameters": "input_a input_b",
    "timestamp": 1699123456789
}
```

#### **2. response.json（Python→C++响应）**

```json
{
    "command": "matmul",
    "result": "ok",
    "message": "Matrix multiplication completed",
    "timestamp": 1699123456790
}
```

#### **3. status.txt（状态同步）**

```
ready      # Python就绪，等待请求
busy       # Python正在处理请求
ok         # 请求处理成功
error: ... # 处理出错
exiting    # Python正在退出
```

#### **4. TSR张量文件格式**

```
TSR文件格式（二进制）：
┌─────────────┬──────────────┬──────────────┐
│  Header     │   Shape      │    Data      │
├─────────────┼──────────────┼──────────────┤
│ 魔数+版本   │  N, C, H, W  │  张量数据    │
│ (12字节)    │  (16字节)     │  (变长)      │
└─────────────┴──────────────┴──────────────┘
```

**TSR格式详细结构：**
- **魔数** (4字节): "TSR\0"
- **版本** (4字节): 1
- **数据类型** (4字节): 1=FP32, 2=INT8
- **形状** (16字节): N, C, H, W (int32)
- **数据** (变长): 按行主序排列的张量数据

---

## 🔄 **通信流程详解**

### **标准通信时序图**

```
C++ PythonSession                    Python Server
     │                                   │
     │ 1. start()                        │
     ├─────────────────────────────────→│
     │                                   │ 2. 启动并初始化
     │                                   │
     │ 3. wait_until_ready()            │
     ├─────────────────────────────────→│
     │                                   │ 4. 检查status.txt="ready"
     │                                   │
     │ 5. send_request("matmul a b")    │
     ├─────────────────────────────────→│
     │                                   │ 6. 读取request.json
     │                                   │ 7. 解析命令
     │                                   │ 8. 读取a.tsr, b.tsr
     │                                   │ 9. 执行计算
     │                                   │10. 写入result_0.tsr
     │                                   │11. 写入response.json
     │                                   │12. 更新status.txt="ok"
     │13. wait_for_tensor()             │
     ├─────────────────────────────────→│
     │                                   │14. 检查status.txt="ok"
     │                                   │15. 读取result_0.tsr
     │16. ←─────────────────────────────│
     │                                   │17. 返回Tensor对象
```

### **错误处理流程**

```
C++ PythonSession                    Python Server
     │                                   │
     │ 1. send_request("invalid_cmd")  │
     ├─────────────────────────────────→│
     │                                   │ 2. 解析命令失败
     │                                   │ 3. 写入response.json
     │                                   │ 4. 更新status.txt="error"
     │5. wait_for_response()            │
     ├─────────────────────────────────→│
     │                                   │ 6. 检测到错误状态
     │7. ←─────────────────────────────│
     │8. 抛出异常或返回错误信息          │
```

---

## ⚡ **性能优化特性**

### **1. 智能轮询机制**

```cpp
bool PythonSession::wait_until_ready(uint32_t timeout_ms) const {
    auto start = std::chrono::steady_clock::now();

    // 初始快速轮询
    for (int i = 0; i < 10; ++i) {
        if (is_ready()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // 中速轮询
    for (int i = 0; i < 20; ++i) {
        if (is_ready()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // 慢速轮询
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::steady_clock::now() - start).count() < timeout_ms) {
        if (is_ready()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return false;
}
```

### **2. 原子文件操作**

```python
def write_response(self, command: str, result: str) -> None:
    """原子写入响应文件"""
    temp_file = f"{self.session_dir}/response.json.tmp"
    final_file = f"{self.session_dir}/response.json"

    # 写入临时文件
    with open(temp_file, 'w') as f:
        json.dump({
            "command": command,
            "result": result,
            "timestamp": time.time() * 1000
        }, f)

    # 原子重命名
    os.rename(temp_file, final_file)
```

### **3. "阅后即焚"机制**

```cpp
std::string PythonSession::read_response() const {
    std::string response_file = session_dir_ + "/response.json";

    // 读取响应
    std::string content = read_file_content(response_file);

    // 立即删除响应文件，避免重复读取
    std::remove(response_file.c_str());

    return content;
}
```

---

## 🛠️ **使用示例**

### **基础示例：Hello World**

```cpp
#include "tech_renaissance.h"
using namespace tr;

int main() {
    // 创建会话
    PythonSession session("../python/module/python_server.py", "hello_test", false);

    try {
        // 启动Python进程
        session.start();

        // 等待Python就绪
        if (!session.wait_until_ready(5000)) {
            std::cerr << "Python进程启动超时" << std::endl;
            return 1;
        }

        // 发送请求并等待响应
        std::string response = session.fetch_response("hello world", 3000);
        std::cout << "Python响应: " << response << std::endl;

        // 优雅退出
        session.please_exit(5000, true);

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        session.terminate();
        return 1;
    }

    return 0;
}
```

**对应的Python服务器：**

```python
class HelloServer(TechRenaissanceServer):
    def main_logic(self, command: str, parameters: str) -> bool:
        if command.lower() == 'hello':
            self.write_response('', f'Hello {parameters.title()}')
            return True
        return False
```

### **进阶示例：矩阵乘法**

```cpp
#include "tech_renaissance.h"
using namespace tr;

int main() {
    PythonSession session("default", "matmul_test", false);

    try {
        session.start();
        session.wait_until_ready(5000);

        // 创建测试张量
        auto cpu_backend = BackendManager::get_cpu_backend();
        Tensor a = Tensor::randn(Shape(1024, 1024));
        Tensor b = Tensor::randn(Shape(1024, 1024));

        // 发送张量到Python
        session.send_tensor(a, "input_a");
        session.send_tensor(b, "input_b");

        // 触发矩阵乘法计算
        session.send_request("matmul input_a input_b");

        // 等待计算结果
        Tensor result = session.wait_for_tensor(10000);

        std::cout << "矩阵乘法完成，结果形状: " << result.shape().to_string() << std::endl;

        session.please_exit(5000, true);

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        session.terminate();
        return 1;
    }

    return 0;
}
```

**对应的Python服务器：**

```python
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
```

---

## 🔧 **配置选项**

### **CMake配置**

```cmake
# 启用Python会话支持
option(TR_BUILD_PYTORCH_SESSION "Enable Python session integration" ON)

# 条件编译
#ifdef TR_BUILD_PYTORCH_SESSION
#include "tech_renaissance/utils/python_session.h"
#endif
```

### **编译配置**

```bash
# 启用Python会话支持
cmake .. -DTR_BUILD_PYTORCH_SESSION=ON

# 禁用Python会话支持（减少依赖）
cmake .. -DTR_BUILD_PYTORCH_SESSION=OFF
```

---

## 🐛 **错误处理与调试**

### **常见错误类型**

1. **进程启动失败**
   ```cpp
   // 检查Python脚本是否存在
   if (!std::filesystem::exists(script_path_)) {
       throw TRException("Python script not found: " + script_path_);
   }
   ```

2. **通信超时**
   ```cpp
   // 设置合理的超时时间
   if (!session.wait_until_ready(10000)) {
       throw TRException("Python进程启动超时");
   }
   ```

3. **文件I/O错误**
   ```cpp
   // 检查会话目录权限
   if (!std::filesystem::exists(session_dir_)) {
       throw TRException("会话目录创建失败: " + session_dir_);
   }
   ```

### **调试技巧**

```cpp
// 启用调试模式
PythonSession session("default", "debug_test", false);  // quiet_mode=false

// 检查会话目录
std::cout << "会话目录: " << session.session_dir() << std::endl;

// 监控状态文件
while (session.is_alive()) {
    std::cout << "状态: " << session.is_ready() << ", "
              << session.is_busy() << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}
```

### **Python端调试**

```python
# 启用调试模式
server = MatrixMathServer(debug=True)

# 添加调试日志
def debug_message(self, message: str) -> None:
    if self.debug:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
```

---

## 📈 **性能基准测试**

### **通信延迟测试**

| 操作类型 | 平均延迟 | 95%分位延迟 | 吞吐量 |
|----------|----------|-------------|--------|
| 文本通信 | 15ms | 25ms | 67 msg/s |
| 张量发送 | 50ms | 80ms | 20 tensor/s |
| 张量接收 | 45ms | 75ms | 22 tensor/s |
| 矩阵乘法 | 200ms | 300ms | 5 op/s |

### **内存使用测试**

| 张量大小 | C++内存 | Python内存 | 文件大小 | 传输时间 |
|----------|---------|------------|----------|----------|
| 1024×1024 | 4MB | 8MB | 4MB | 45ms |
| 2048×2048 | 16MB | 32MB | 16MB | 180ms |
| 4096×4096 | 64MB | 128MB | 64MB | 720ms |

---

## 🚀 **最佳实践**

### **1. 会话管理**

```cpp
// 使用RAII管理会话生命周期
class ScopedPythonSession {
private:
    PythonSession session_;
public:
    ScopedPythonSession(const std::string& script, const std::string& id)
        : session_(script, id, false) {
        session_.start();
        session_.wait_until_ready(10000);
    }

    ~ScopedPythonSession() {
        session_.please_exit(5000, true);
    }

    PythonSession& get() { return session_; }
};
```

### **2. 错误处理**

```cpp
try {
    ScopedPythonSession python("../python/module/python_server.py", "safe_test");

    // 业务逻辑
    auto result = python.get().fetch_response("hello world", 5000);

} catch (const TRException& e) {
    // 记录错误日志
    Logger::get_instance().error("Python会话错误: ", e.what());

} catch (const std::exception& e) {
    // 处理标准异常
    Logger::get_instance().error("标准异常: ", e.what());

} catch (...) {
    // 处理未知异常
    Logger::get_instance().error("未知异常发生");
}
```

### **3. 性能优化**

```cpp
// 批量发送张量
void batch_send_tensors(PythonSession& session,
                       const std::vector<Tensor>& tensors,
                       const std::vector<std::string>& tags) {
    for (size_t i = 0; i < tensors.size(); ++i) {
        session.send_tensor(tensors[i], tags[i]);
    }
}

// 异步等待结果
std::future<Tensor> async_fetch_tensor(PythonSession& session,
                                     uint32_t timeout_ms) {
    return std::async(std::launch::async, [&session, timeout_ms]() {
        return session.wait_for_tensor(timeout_ms);
    });
}
```

---

## 📚 **API参考手册**

### **PythonSession类完整API**

#### **构造函数与析构函数**

```cpp
PythonSession(const std::string& script_path = "default",
              const std::string& session_id = "default",
              bool quiet_mode = false);
~PythonSession();
```

#### **进程管理API**

```cpp
void start()                              // 启动Python脚本
bool is_alive()                           // 检查进程是否存活
void terminate()                          // 强制终止进程
void join()                               // 等待进程结束
void please_exit(uint32_t timeout_ms = 10000, bool ensure = true)
```

#### **状态管理API**

```cpp
bool is_ready() const                     // 检查是否就绪
bool is_busy() const                      // 检查是否繁忙
bool new_response() const                 // 检查是否有新响应
bool wait_until_ready(uint32_t timeout_ms = 10000) const
bool wait_until_ok(uint32_t timeout_ms = 10000) const
```

#### **文本通信API**

```cpp
void send_request(const std::string& msg) const
std::string read_response() const
std::string wait_for_response(uint32_t timeout_ms = 10000) const
std::string fetch_response(const std::string& msg, uint32_t timeout_ms = 10000) const
```

#### **张量通信API**

```cpp
void send_tensor(const Tensor& tensor, const std::string& tag) const
Tensor fetch_tensor(const std::string& msg, uint32_t timeout_ms = 10000) const
Tensor wait_for_tensor(uint32_t timeout_ms = 10000) const
```

#### **会话管理API**

```cpp
std::string session_dir() const
const std::string& session_id() const
void set_quiet_mode(bool quiet_mode)
```

### **TechRenaissanceServer基类API**

#### **核心方法**

```python
def run(session_id: str) -> None           # 运行服务器主循环
def main_logic(self, command: str, parameters: str) -> bool  # 业务逻辑（需重写）
def should_continue(self) -> bool          # 检查是否应该继续运行
def process_request(self) -> bool          # 处理单个请求
```

#### **通信方法**

```python
def write_response(self, command: str, result: str) -> None
def write_status(self, status: str) -> None
def send_tensors(self, *tensors) -> None
def get_tensors(self, params: str, count: int) -> Optional[List[torch.Tensor]]
```

#### **工具方法**

```python
def debug_message(self, message: str) -> None
def setup_session_dir(self) -> None
def cleanup_session_dir(self) -> None
```

---

## 🔮 **未来扩展计划**

### **V1.21.00计划功能**

1. **异步通信支持**
   - 基于std::future的异步API
   - 并发请求处理
   - 回调机制支持

2. **网络通信扩展**
   - TCP/IP套接字支持
   - 远程Python服务器
   - 负载均衡机制

3. **性能优化**
   - 内存映射文件
   - 零拷贝传输
   - 压缩传输支持

4. **安全增强**
   - 进程间认证
   - 数据加密传输
   - 访问控制机制

### **长期规划**

1. **多语言支持**
   - Python/NumPy扩展
   - Julia语言支持
   - R语言集成

2. **分布式计算**
   - 多节点Python集群
   - 任务分发机制
   - 结果聚合

3. **可视化支持**
   - 实时性能监控
   - 通信状态可视化
   - 调试界面

---

**版本信息**：
- **当前版本**: V1.20.01
- **最后更新**: 2025-10-31
- **兼容性**: C++17, Python 3.8+, PyTorch 1.12+
- **平台支持**: Windows 10+, Linux (Ubuntu 18.04+)

---

*本文档涵盖了技术觉醒框架Python通信机制的完整技术细节。如有疑问或需要进一步的示例代码，请参考tests/unit_tests目录下的测试用例。*
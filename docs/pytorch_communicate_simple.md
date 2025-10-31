# 技术觉醒框架 - PyTorch通信机制（简版）

## 📋 **快速开始**

技术觉醒框架提供升级版C++与PyTorch跨进程通信机制，V1.20.01版本实现了**完整张量数据交换**的高效通信系统。

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
#include "tech_renaissance/utils/pytorch_session.h"

int main() {
    // 创建会话
    PyTorchSession session("python/tests/python_server.py", "matmul_demo");
    session.start();

    // 创建测试张量：4x3 矩阵 × 3x5 矩阵 = 4x5 矩阵
    Tensor tensor_a = Tensor::full(Shape(4, 3), 1.5f, DType::FP32, tr::CPU);
    Tensor tensor_b = Tensor::full(Shape(3, 5), 2.0f, DType::FP32, tr::CPU);

    // V1.20.01核心API：发送张量到PyTorch
    session.send_tensor(tensor_a, "a");
    session.send_tensor(tensor_b, "b");

    // 执行矩阵乘法并获取结果
    Tensor result = session.fetch_tensor(R"({"cmd": "matmul", "params": "a,b"})", 10000);

    // 验证结果
    std::cout << "矩阵乘法结果形状: " << result.shape() << std::endl;
    result.print("result");

    // 结束会话
    session.send_request(R"({"cmd": "exit"})");
    return 0;
}
```

### **2. 张量加法示例（V1.20.01多种API模式）**

```cpp
#include "tech_renaissance.h"
#include "tech_renaissance/utils/pytorch_session.h"

int main() {
    PyTorchSession session("python/tests/python_server.py", "add_demo");
    session.start();

    // 创建测试张量：2x3x4x5 张量
    Tensor tensor_a = Tensor::full(Shape(2, 3, 4, 5), 3.0f, DType::FP32, tr::CPU);
    Tensor tensor_b = Tensor::full(Shape(2, 3, 4, 5), 4.0f, DType::FP32, tr::CPU);

    // 发送张量
    session.send_tensor(tensor_a, "a");
    session.send_tensor(tensor_b, "b");

    // V1.20.01模式B：发送请求 + 等待张量
    session.send_request(R"({"cmd": "add", "params": "a,b"})");

    // 模拟耗时任务
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // 获取结果张量
    Tensor result = session.wait_for_tensor(10000);
    result.print("add_result");

    session.send_request(R"({"cmd": "exit"})");
    return 0;
}
```

---

## 📁 **文件协议**

### **V1.20.01通信文件格式**

| 文件 | 方向 | 格式 | 示例 |
|------|------|------|------|
| `request.json` | C++→Python | JSON | `{"cmd": "matmul", "params": "a,b"}` |
| `response.json` | Python→C++ | JSON | `{"cmd": "matmul", "params": "result_tag"}` |
| `{tag}.tsr` | C++→Python | TSR | 二进制张量数据（完整信息） |
| `{tag}.tsr` | Python→C++ | TSR | 二进制张量数据（计算结果） |
| `status.txt` | 双向 | TXT | `ready`/`running`/`terminated` |

### **V1.20.01目录结构**
```
workspace/pytorch_session/tr_session_{session_id}/
├── request.json          # C++发送的请求
├── response.json         # Python的响应（阅后即焚）
├── a.tsr                  # 输入张量数据
├── b.tsr                  # 输入张量数据
├── result_tag.tsr         # 输出张量数据
└── status.txt             # 状态同步
```

---

## ⚡ **核心接口**

### **PyTorchSession类V1.20.01完整API**

```cpp
class PyTorchSession {
public:
    // 进程管理
    void start();                                    // 启动Python进程
    void terminate();                               // 终止进程
    bool is_alive();                                // 检查进程状态

    // 文本通信
    void send_request(const std::string& msg);      // 发送JSON请求
    std::string wait_response(uint32_t timeout_ms = 10000);     // 等待文本响应
    std::string fetch_response(const std::string& msg, uint32_t timeout_ms = 10000);  // 发送+等待响应

    // V1.20.01核心：TSR张量传输
    void send_tensor(const Tensor& tensor, const std::string& tag);   // 发送张量
    Tensor fetch_tensor(const std::string& msg, uint32_t timeout_ms = 10000);   // 发送请求+等待张量
    Tensor wait_for_tensor(uint32_t timeout_ms = 10000);                 // 等待张量结果

    // 状态检查
    bool is_ready() const;                          // 检查是否可发送请求
    bool is_busy() const;                          // 检查是否正在处理
};
```

---

## 🔧 **配置选项**

### **CMake编译配置**

```cmake
# 启用PyTorch通信支持
option(TR_BUILD_PYTORCH_SESSION "Enable PyTorch session integration" ON)

# 构建目标
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE tech_renaissance)
```

### **关键宏定义**

- `TR_BUILD_PYTORCH_SESSION`：启用PyTorch通信功能
- `WORKSPACE_PATH`：临时文件存储路径
- `AUTO_QUEST_FREQUENCY`：启用智能轮询频率

---

## 🧪 **V1.20.01测试验证**

### **运行真实张量测试**

```bash
# 构建项目
cmake --build . --target test_pytorch_data

# 运行V1.20.01张量测试
./bin/tests/test_pytorch_data.exe
```

### **V1.20.01预期输出**

```
=== Test Style A: 2D Matrix Multiplication ===
[TEST] Style A: Send tensors and execute matrix multiplication...
[TEST] Successfully got matrix multiplication result from PyTorch
result (4x5):
tensor([[[9., 9., 9., 9., 9.],
         [9., 9., 9., 9., 9.],
         [9., 9., 9., 9., 9.],
         [9., 9., 9., 9., 9.]]])

=== Test Style B: 4D Tensor Addition ===
[TEST] Style B: Send tensors and execute addition...
[TEST] Successfully got addition result from PyTorch
result (2x3x4x5):
tensor([[[[7., 7., 7., 7., 7.],
          [7., 7., 7., 7., 7.],
          [7., 7., 7., 7., 7.],
          [7., 7., 7., 7., 7.]],
         ...]])

=== Test Summary ===
Passed: 5/5 tests
Success Rate: 100%
All PyTorch Data tests PASSED!
```

---

## 💡 **最佳实践**

### **1. V1.20.01张量通信最佳实践**

```cpp
try {
    PyTorchSession session("python/tests/python_server.py", "tensor_test");
    session.start();

    // V1.20.01推荐模式：fetch_tensor
    Tensor input = Tensor::full(Shape(2, 2), 3.14f, DType::FP32, tr::CPU);
    session.send_tensor(input, "input");

    Tensor result = session.fetch_tensor(R"({"cmd": "square", "params": "input"})", 5000);
    result.print("squared_result");

} catch (const TRException& e) {
    std::cout << "张量通信失败: " << e.what() << std::endl;
}
```

### **2. 资源管理**

```cpp
{
    PyTorchSession session("script.py", "test");
    session.start();
    // 使用session...
    // 析构函数自动清理资源
} // session作用域结束，自动清理
```

### **3. 超时控制**

```cpp
// 设置合适的超时时间
std::string response = session.wait_response(10000);  // 10秒超时
if (response.empty()) {
    std::cout << "请求超时" << std::endl;
}
```

---

## 🆚 **V1.20.01 vs V1.19.02 vs 旧版本对比**

| 特性 | V1.18.x | V1.19.02 | V1.20.01 |
|------|---------|----------|----------|
| 张量传输 | TXT格式，仅元数据 | TXT格式，仅元数据 | **TSR格式，完整数据** |
| 真实运算 | ❌ 不支持 | ❌ 不支持 | **✅ matmul, add等** |
| API模式 | 基础文本通信 | JSON标准协议 | **多种API模式** |
| JSON格式 | 非标准 `( {..} )` | 标准 `{...}` | 标准 `{...}` |
| 读写冲突 | 可能发生 | 原子操作避免 | 原子操作避免 |
| 传输效率 | 低（文本格式） | 中（优化文本） | **高（二进制格式）** |
| 通信延迟 | ~100ms | 32-1024ms智能 | **32ms智能** |
| 扩展性 | 有限 | 良好 | **优秀** |

---

## 🐛 **常见问题**

### **Q1: Python进程启动失败**
**A**: 检查Python脚本路径和Python环境，确保在系统PATH中。

### **Q2: 响应超时**
**A**: 检查Python脚本是否正确处理请求，增加超时时间或启用调试模式。

### **Q3: JSON解析错误**
**A**: V1.19.02已修复，确保使用标准JSON格式。

### **Q4: 临时文件残留**
**A**: V1.19.02的"阅后即焚"机制自动清理，无需手动处理。

---

## 📈 **V1.20.01性能指标**

| 指标 | V1.19.02 | V1.20.01 |
|------|---------|----------|
| 张量传输速度 | ~1MB/s（文本） | **~10MB/s+（二进制）** |
| 通信延迟 | 32-100ms智能 | **32ms智能优化** |
| CPU占用 | 降低60%+ | **降低65%+** |
| 内存开销 | <1MB会话数据 | **<2MB张量缓存** |
| 成功率 | 100%（文本通信） | **100%（张量通信）** |
| 解析错误率 | 0%（标准JSON） | **0%（标准JSON）** |
| 支持运算类型 | ❌ 仅元数据 | **✅ 真实矩阵运算** |

---

## 🔮 **未来规划**

**V1.21.00计划特性：**
- 多会话并行管理
- 流式大数据传输
- 网络远程通信支持
- GPU直接内存访问优化

**V1.20.01已实现特性：**
- ✅ TSR二进制张量传输
- ✅ 真实矩阵运算支持
- ✅ 多种API通信模式
- ✅ 完整的张量数据交换

---

## 🏗️ **技术架构**

### **核心设计原理**

技术觉醒框架采用**多进程+临时文件+原子操作**的通信架构：

```
C++主进程                     Python进程
    |                              |
    | 1. 启动Python进程              |
    |------------------------------->|
    | 2. 写入request.json           |
    |<------------------------------| 3. 读取请求
    | 4. 智能轮询等待               |    |
    |<------------------------------| 5. 原子写入response.json
    | 6. "阅后即焚"读取              |    |
    |                              | 7. 继续监听
```

### **V1.20.01关键创新**

1. **TSR张量传输**：二进制格式实现完整张量数据交换
2. **多种API模式**：fetch_tensor, wait_for_tensor等灵活接口
3. **真实矩阵运算**：支持matmul, add等复杂PyTorch运算
4. **原子操作机制**：临时文件+重命名确保写入安全
5. **智能轮询频率**：自适应调整节省CPU资源
6. **标准JSON协议**：统一格式避免解析错误
7. **进程隔离设计**：Python崩溃不影响C++主程序

---

**📅 文档版本：V1.20.01**
**👥 维护团队：技术觉醒团队**
**📅 最后更新：2025-10-29**
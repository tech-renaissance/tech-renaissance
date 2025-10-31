![Logo](docs/logo.png)

# 技术觉醒（Tech Renaissance）



一个轻量级、可重构、多后端的开源深度学习框架



## 项目概述

技术觉醒框架是一个从零开始构建的深度学习框架，专注于教学、科研和轻量级应用场景。项目采用C++17开发，支持CPU、GPU、FPGA等多后端的高性能计算，易于适应自研AI芯片，具有清晰的架构设计和良好的可扩展性



## 核心特性

### 🏗️ 架构设计
- **关注点分离**：Tensor类负责元数据管理，Backend类负责计算，Storage类负责内存管理
- **多后端支持**：CPU后端（Eigen优化）、CUDA后端（cuBLAS/cuDNN集成）
- **静态图框架**：先定义计算图，再执行计算，适合硬件优化
- **内存安全**：智能指针管理内存，防止内存泄漏和数据解耦

### 🚀 技术特色
- **轻量级设计**：最小依赖，避免庞大的第三方库
- **可重构架构**：支持编译时组件裁剪，按需定制功能
- **设备抽象**：统一的Backend接口，支持多平台扩展
- **中文注释**：完整的技术文档使用中文编写



## 快速开始

### 环境要求
- **编译器**：Visual Studio 2019/2022（Windows）或GCC/Clang（Linux）
- **CMake**：3.20 或更高版本
- **CUDA**：12.8 或更高版本（可选，用于GPU支持）
- **cuDNN**：8.9.7 或更高版本
- **Python**：3.8 或更高版本

### 编译安装

```bash
# 克隆项目
git clone https://github.com/tech-renaissance/tech_renaissance.git
cd tech_renaissance

# 配置构建（启用CUDA支持）
cmake -S . -B build -DTR_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug

# 编译项目
cmake --build build

# 运行测试
./build/bin/tests/Debug/test_tensor_backend.exe
```

### 基本使用

```cpp
#include "tech_renaissance.h"

using namespace tr;

int main() {
    // 创建后端管理器
    BackendManager& manager = BackendManager::instance();

    // 创建张量
    Shape shape(2, 3);
    Tensor a(shape, DType::FP32, CPU);
    Tensor b(shape, DType::FP32, CPU);
    Tensor result(shape, DType::FP32, CPU);

    // 填充和计算
    auto backend = manager.get_backend(CPU);
    backend->fill(a, 1.5f);
    backend->fill(b, 2.5f);
    backend->add(result, a, b);  // result = a + b

    // 输出结果
    result.print("result");
    return 0;
}
```



## 项目文档

- **[设计文档](tech_renaissance_prompt.md)**：完整的项目设计和架构说明
- **[API文档](docs/api/)**：详细的类和接口文档
- **[开发日志](LOG.md)**：开发过程中的记录和经验总结
- **[问题追踪](QUEST.md)**：技术问题和解决方案



## 贡献指南

欢迎提交Issue和Pull Request！请遵循以下原则：

1. **代码风格**：遵循Google C++ Style Guide
2. **注释要求**：所有注释使用中文
3. **测试覆盖**：新功能需要包含相应的单元测试
4. **文档更新**：重要修改需要更新相关文档



## 技术交流

- **GitHub Issues**：[项目问题反馈](https://github.com/tech-renaissance/tech_renaissance/issues)
- **技术讨论**：欢迎在Issues中提出技术问题和建议



## 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。



## 版本信息

**当前版本**: V1.11.05 (2025-10-26)

---

*技术觉醒框架正在积极开发中，欢迎关注项目进展和参与技术讨论。*
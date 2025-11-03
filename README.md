![Logo](docs/logo.png)

# 技术觉醒（Tech Renaissance）

一个轻量级、可重构、多后端的开源深度学习框架。



## 🚀 近期突破

**部分核心运算在CUDA上性能超越PyTorch！** 在最新版本V1.40.1中，我们的自研深度学习框架在核心运算上实现了历史性突破：

| 运算类型 | PyTorch性能 | TR性能 | 相对性能 |
|:--------:|:-----------:|:------:|:--------:|
| **3×3卷积** | 8394.59 GFLOPS | **11896.71 GFLOPS** | **+41.72%** |
| **1×1卷积** | 5781.71 GFLOPS | **6602.31 GFLOPS** | **+14.19%** |
| **转置卷积** | 8420.02 GFLOPS | **13418.89 GFLOPS** | **+59.37%** |
| **矩阵乘法** | 6604.40 GFLOPS | **6678.33 GFLOPS** | **+1.12%** |

*测试环境：NVIDIA RTX 4060, CUDA 12.8, cuDNN 8.9.7*



## 🏗️ 架构特色

### 后端解耦设计
框架采用创新的**Tensor-Backend分层解耦架构**：
- **Tensor**：仅存储元数据（形状、类型、设备），不负责计算
- **Backend**：负责所有计算和存储管理，支持CPU、CUDA等多种后端
- **Storage**：封装内存管理，通过智能指针实现RAII

这种设计让张量在硬件上的行为更具有透明度，更加方便开发者进行灵活的开发和高效的优化。

### 性能支柱：`into`型方法
我们提出了`into`型方法设计，实现了极致的内存复用：

```cpp
// 传统方式：每次都分配新内存
Tensor result = backend->conv(input, kernel);  // 内存分配 + 计算

// into方式：复用已有内存，零分配开销
backend->conv_into(input, kernel, result);     // 仅计算，无内存分配
```

`into`型方法通过预分配内存缓冲区，避免了深度学习中最耗时的内存分配操作，实现了接近硬件理论极限的性能。这种方法可以最大限度地利用静态图的优势。



## 🎯 核心特性

### 算法优化突破
- **1×1卷积算法查找修复**：V1.40.1核心技术突破，性能比上一版本提升119%
- **描述符智能缓存**：cuDNN描述符完整缓存，初始化开销降至接近零
- **工作空间内存池化**：智能工作空间管理，减少15-20%内存管理开销
- **Tensor Core加速**：全面启用现代GPU的Tensor Core加速能力

### 轻量级设计
- **最小依赖**：核心功能仅依赖标准C++库，可选Eigen和CUDA
- **可重构架构**：通过CMake选项支持编译时组件裁剪
- **跨平台支持**：Windows/Linux，CPU/GPU/FPGA多后端

### 开发友好
- **静态图设计**：先定义计算图，再执行计算，便于硬件优化
- **中文技术文档**：完整的设计文档和API文档使用中文编写
- **精度对齐**：与PyTorch数学结果100%一致，可直接替换



## 🚀 快速开始

### 环境要求
- **编译器**：Visual Studio 2022（Windows）或GCC/Clang（Linux）
- **CMake**：3.20 或更高版本
- **CUDA**：12.8 或更高版本（可选，用于GPU支持）
- **cuDNN**：8.9.7 或更高版本

### 编译选项（推荐，最高性能）
```bash
# 配置高性能构建
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE="path/to/vcpkg/scripts/buildsystems/vcpkg.cmake" \
      -S . -B build/cmake-build-release

# 编译
cmake --build build/cmake-build-release -j 30
```

### 基本使用
```cpp
#include "tech_renaissance.h"

using namespace tr;

int main() {
    // 获取CUDA后端
    auto backend = BackendManager::get_cuda_backend();

    // 创建张量（Backend自动分配内存）
    Shape input_shape(32, 512, 7, 7);
    Shape kernel_shape(512, 512, 3, 3);
    Tensor input = backend->randn(input_shape);
    Tensor kernel = backend->randn(kernel_shape);
    Tensor output = backend->empty(Shape(32, 512, 7, 7));

    // 高性能卷积运算
    backend->conv_into(input, kernel, output);  // 零内存分配

    return 0;
}
```



## 📚 技术文档

### 核心设计
- **[设计文档](tech_renaissance_prompt.md)**：完整的项目设计和架构说明
- **[张量-后端系统](docs/tensor_backend_system.md)**：核心架构设计详解
- **[性能基准](docs/performance.md)**：详细的性能测试报告

### API文档
- **[后端API](docs/backend.md)**：Backend抽象接口设计
- **[CUDA后端](docs/cuda_backend.md)**：GPU高性能计算实现
- **[卷积操作](docs/cuda_conv.md)**：CUDA卷积算法详解

### 构建部署
- **[构建设置](docs/build_settings.md)**：编译配置和优化指南
- **[API索引](API.md)**：完整的类和接口文档



## 🏆 技术成就

### 算法创新
- **根性问题解决方案**：采用"修复根源而非添加补丁"的技术路线
- **智能缓存体系**：描述符缓存、工作空间池化、算法选择的完整优化链
- **内存布局优化**：列主序存储与cuBLAS/cuDNN完美对齐

### 工程价值
- **生产就绪**：性能稳定，具备工业级应用能力
- **架构优雅**：清晰的分层设计，易于维护和扩展
- **技术示范**：验证了自研深度学习框架达到国际领先水平的可行性



## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. **代码风格**：遵循Google C++ Style Guide
2. **注释要求**：所有注释使用中文
3. **测试覆盖**：新功能需要包含相应的单元测试
4. **性能验证**：核心运算需要与PyTorch进行性能对比



## 📞 技术交流

- **GitHub Issues**：[项目问题反馈](https://github.com/tech-renaissance/tech_renaissance/issues)
- **技术讨论**：欢迎在Issues中提出技术问题和建议



## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---



**当前版本**: V1.40.1 (2025-11-04)


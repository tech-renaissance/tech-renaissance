![Logo](docs/logo.png)

# 技术觉醒（Tech Renaissance）

一个轻量级、可重构、跨平台的开源深度学习框架。



## 🚀 核心特性

### 跨平台支持
- **Windows/Linux**：双平台完全支持
- **MSVC/GCC**：双编译器兼容
- **自动配置**：智能依赖检测和环境配置
- **一键编译**：编译实现真正的"一键编译"

### 性能优势
**训练性能对比**：

#### （1）Intel Core i9 + Windows

| 优化器 | PyTorch  | Tech Renaissance | Speed Up |
| :----: | :------: | :--------------: | :------: |
|  SGD   | 108.40 s |     60.85 s      |  1.78×   |
|  Adam  | 112.00 s |     67.90 s      |  1.65×   |
| AdamW  | 114.30 s |     67.95 s      |  1.68×   |

测试条件：Intel Core i9-14900HX，内存32.0 GB，Windows 11专业版，三层MLP（784-512-256-10）训练，数据集为MNIST。PyTorch版本为2.9.0。所有数据都是20次独立测试的结果取平均值。测试样例详见：**[PyTorch](python/tests/pytorch_sgd.py)** Vs **[TR](tests/integration_tests/test_trainer_sgd)**

#### （2）Intel Xeon + Ubuntu

| 优化器 | PyTorch  | Tech Renaissance | Speed Up |
| :----: | :------: | :--------------: | :------: |
|  SGD   | 177.30 s |     79.85 s      |  2.22×   |
|  Adam  | 180.60 s |     97.15 s      |  1.86×   |
| AdamW  | 181.50 s |     97.10 s      |  1.87×   |

测试条件：Xeon Platinum 8369B，内存60.0 GB，Ubuntu 24.04 LTS，三层MLP（784-512-256-10）训练，数据集为MNIST。PyTorch版本为2.9.0。所有数据都是20次独立测试的结果取平均值。测试样例详见：**[PyTorch](python/tests/pytorch_sgd.py)** Vs **[TR](tests/integration_tests/test_trainer_sgd)**

**测试准确率对比**：

#### （1）Intel Core i9 + Windows

| 优化器 | PyTorch | Tech Renaissance |  Diff  |
| :----: | :-----: | :--------------: | :----: |
|  SGD   | 98.29%  |      98.34%      | 0.06%  |
|  Adam  | 98.07%  |      98.09%      | 0.02%  |
| AdamW  | 98.07%  |      98.04%      | -0.03% |

测试条件：Intel Core i9-14900HX，内存32.0 GB，Windows 11专业版，三层MLP（784-512-256-10）训练，数据集为MNIST。PyTorch版本为2.9.0。所有数据都是20次独立测试的结果取平均值。测试样例详见：**[PyTorch](python/tests/pytorch_sgd.py)** Vs **[TR](tests/integration_tests/test_trainer_sgd)**

#### （2）Intel Xeon + Ubuntu

| 优化器 | PyTorch | Tech Renaissance | Diff  |
| :----: | :-----: | :--------------: | :---: |
|  SGD   | 98.26%  |      98.36%      | 0.09% |
|  Adam  | 98.06%  |      98.07%      | 0.01% |
| AdamW  | 98.05%  |      98.07%      | 0.02% |

测试条件：Xeon Platinum 8369B，内存60.0 GB，Ubuntu 24.04 LTS，三层MLP（784-512-256-10）训练，数据集为MNIST。PyTorch版本为2.9.0。所有数据都是20次独立测试的结果取平均值。测试样例详见：**[PyTorch](python/tests/pytorch_sgd.py)** Vs **[TR](tests/integration_tests/test_trainer_sgd)**



## 🛠️ 构建系统

### 自动配置
```bash
# 智能依赖检测和配置
python configure.py
```

### 一键编译
```bash
# Windows MSVC (推荐)
powershell.exe -Command "& { .\build_msvc.bat }"

# Windows GCC
powershell.exe -Command "& { .\build_msys2.bat }"

# Linux GCC
python configure.py
chmod +x build.sh && ./build.sh
```

### 环境要求
- **编译器**: Visual Studio 2022 或 GCC 13.0+
- **CMake**: 3.24+
- **CUDA**: 12.8+ (GPU支持，可选)
- **Python**: 3.10+ (配置工具)



## 🏗️ 架构设计

### 后端解耦架构
采用创新的**Tensor-Backend分层解耦架构**：
- **Tensor**：仅存储元数据，不负责计算
- **Backend**：负责所有计算和存储管理
- **Storage**：封装内存管理，支持RAII

### 核心技术
- **into型方法**：预分配内存，避免运行时分配开销
- **智能缓存机制**：权重转置缓存、one-hot编码缓存
- **动态批处理**：完美处理不完整批次
- **零拷贝优化**：减少内存拷贝，提升性能

### CUDA加速性能
| 运算类型 | PyTorch GFLOPS | Tech Renaissance GFLOPS | 性能提升 |
|:--------:|:--------------:|:----------------------:|:--------:|
| **3×3卷积** | 8394.59 | **11896.71** | **+41.72%** |
| **转置卷积** | 8420.02 | **13418.89** | **+59.37%** |
| **1×1卷积** | 5781.71 | **6602.31** | **+14.19%** |
| **矩阵乘法** | 6604.40 | **6678.33** | **+1.12%** |

*测试环境：NVIDIA RTX 4060, CUDA 12.8, cuDNN 8.9.7*



## 🚀 快速开始

### 基本使用
```cpp
#include "tech_renaissance.h"

using namespace tr;

int main() {
    // 获取后端
    auto backend = BackendManager::get_cuda_backend();

    // 创建张量
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

### 训练示例
```cpp
// 创建模型
auto model = Model::create("MLP",
    std::make_shared<Linear>(784, 512),
    std::make_shared<Tanh>(),
    std::make_shared<Linear>(512, 10)
);

// 创建训练器
Trainer trainer(model,
    std::make_unique<Adam>(0.001f),
    std::make_unique<CrossEntropyLoss>());

// 训练循环
for (auto [batch_x, batch_y] : train_loader) {
    float loss = trainer.train_step(batch_x, batch_y);
    std::cout << "Loss: " << loss << std::endl;
}
```



## 📚 技术文档

### 核心设计
- **[设计文档](tech_renaissance_prompt.md)**：完整的项目设计和架构说明
- **[张量-后端系统](docs/tensor_backend_system.md)**：核心架构设计详解
- **[Model-Trainer系统](docs/model_trainer_system.md)**：完整训练系统设计详解
- **[跨平台构建](docs/toward_2.0.0.md)**：V2.0.0架构重构和迁移方案

### 构建配置
- **[编译指南](docs/gamma_build.md)**：一键编译配置和使用方法
- **[CLion配置](docs/clion_build_settings.md)**：IDE集成开发环境配置

### API文档
- **[后端API](docs/backend.md)**：Backend抽象接口设计
- **[CUDA后端](docs/cuda_backend.md)**：GPU高性能计算实现
- **[性能基准](docs/performance.md)**：详细的性能测试报告



## 🤝 开发指南

欢迎提交Issue和Pull Request！

1. **代码风格**：遵循Google C++ Style Guide
2. **注释要求**：所有注释使用中文
3. **测试覆盖**：新功能需要包含相应的单元测试
4. **性能验证**：核心运算需要与PyTorch进行性能对比

### 技术交流
- **GitHub Issues**：[项目问题反馈](https://github.com/tech-renaissance/tech_renaissance/issues)
- **技术讨论**：欢迎在Issues中提出技术问题和建议



## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---

**当前版本**: V2.1.1 (2025-11-23)

**主要特性**:
- ✅ 跨平台支持（Windows/Linux）
- ✅ 自动配置（智能依赖检测）
- ✅ 一键编译（Gamma编译脚本）
- ✅ 性能优越（训练速度超越PyTorch）
- ✅ 完整生态（训练、推理、多后端）
# 跨平台C++项目总结文档

## 项目概述

这是一个跨平台C++项目，展示了如何集成CUDA、cuDNN、Eigen3、OpenMP和Python。项目特色是智能配置向导（`configure.py`），能够自动检测开发工具并生成平台特定的构建脚本。

## 项目目录结构

```
best_project/
├── configure.py                           # 智能配置向导 ⭐
├── CMakeLists.txt                         # 主CMake配置文件
├── CMakePresets.json                      # CMake预设配置（6种构建模式）
├── vcpkg.json                            # vcpkg包依赖管理
├── PROJECT_SUMMARY.md                     # 本文档
├── cmake/
│   └── modules/
│       └── FindCuDNN.cmake               # 自定义cuDNN查找模块
├── config/
│   └── user_paths.cmake                   # 自动生成的配置文件
├── tests/
│   └── unit_tests/
│       ├── CMakeLists.txt                # 单元测试CMake配置
│       ├── eigen_openmp_hello_world.cpp  # Eigen3 + OpenMP测试
│       └── test_cuda_gemm.cpp            # CUDA + cuDNN + cuBLAS测试
├── python/
│   └── tests/
│       └── pytorch_hello_world.py        # PyTorch + NumPy测试
└── build/                                # 生成的构建目录
    ├── windows-msvc-release/             # Windows MSVC构建
    ├── windows-msys2-release/            # Windows MSYS2构建
    └── linux-gcc-release/                # Linux GCC构建
```

## configure.py：智能配置向导

### 设计目的与理念

`configure.py` 是本项目跨平台构建系统的核心。它旨在解决不同平台（Windows MSVC、Windows MSYS2、Linux）上手动配置环境的常见问题，尤其是针对不同工具链配置的复杂性。

### 核心特性

1. **7步智能检测**：自动发现开发工具
2. **统一配置**：通过 `config/user_paths.cmake` 实现单一真实来源
3. **平台无关性**：在Windows和Linux上无缝工作
4. **版本验证**：确保所有依赖满足最低版本要求
5. **优雅回退**：每个依赖都有多种搜索策略
6. **智能路径生成**：平台特定的库路径处理

### 检测逻辑详解

#### 第1步：基础构建工具
- **CMake**：要求 ≥3.24.0
- **Ninja**：用于快速并行构建
- **vcpkg**：C++包管理器

#### 第2步：编译器设置
- **Windows MSVC**：检测Visual Studio 2022，验证cl.exe版本 ≥14.44
- **Windows MSYS2**：在MSYS2环境中检测GCC
- **Linux**：检测GCC ≥13.x并验证版本

#### 第3步：CUDA和cuDNN检测
- **CUDA**：搜索标准位置，验证版本 ≥12.8
- **cuDNN**：严格要求8.x版本，4种验证方法：
  - 头文件版本解析（`cudnn.h` 和 `cudnn_version.h`）
  - 目录名模式（`v8*`, `8.*`）
  - 库文件存在性
  - DLL版本检查（Windows）

#### 第4步：Python环境
- **Python**：要求 ≥3.11，自动PATH检测
- **NumPy**：验证 ≥2.3.4可用性

#### 第5步：Eigen3检测（优先级顺序）
1. **项目特定路径**：`./third_party/Eigen`, `./vendor/Eigen`, `./external/Eigen`
2. **系统PATH搜索**：扫描PATH环境变量中的所有目录
3. **vcpkg安装**：检查vcpkg包注册表
4. **系统位置**：`/usr/local/include/Eigen`, `/opt/Eigen`, `C:/Eigen`

#### 第6步：其他依赖
- **OpenMP**：自动检测和配置

#### 第7步：配置生成
- **user_paths.cmake**：统一的CMake配置，路径标准化
- **构建脚本**：平台特定的批处理/Shell脚本

### 平台特定路径处理

**Windows MSVC**：
```python
# CUDA: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8
# cuDNN: C:/Program Files/NVIDIA/CUDNN/v8.9.7/lib/x64
# MSVC: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207
```

**Windows MSYS2**：
```python
# GCC: T:/Softwares/msys64/mingw64/bin/gcc.exe
# CUDA: 禁用（与GCC+nvcc不兼容）
# vcpkg: T:/Softwares/vcpkg
```

**Linux**：
```python
# CUDA: /usr/local/cuda
# cuDNN: /usr/local/cuda/lib64（自动检测）
# GCC: /usr/bin/gcc-13
# Eigen3: /usr/local/include/Eigen 或 /opt/Eigen
```

## 完整使用指南

### 1. 初始设置

**前置要求**：
- Python 3.11+
- CMake 3.24+
- Windows：Visual Studio 2022 或 MSYS2
- Linux：GCC 13.x

**安装命令**：

*Windows*：
```cmd
# 安装Visual Studio 2022（包含C++开发工具）
# 安装vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# 安装Eigen3
.\vcpkg install eigen3
```

*Linux*：
```bash
# 安装开发工具
sudo apt update
sudo apt install build-essential cmake ninja-build

# 从NVIDIA官网安装CUDA 12.8和cuDNN 8.x
# 安装Eigen3
sudo apt install libeigen3-dev

# 安装vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
```

### 2. 配置流程

**步骤1：运行配置向导**
```bash
python configure.py
```

**输出示例**：
```
Starting simple configuration wizard...
=== Smart Project Configuration Wizard ===
System: Windows

[Step 1/7] Checking basic build tools...
  [OK] CMake 3.28.3
  [OK] Ninja 1.11.1 (PATH)
  [OK] vcpkg (VCPKG_ROOT): T:/Softwares/vcpkg

[Step 2/7] Setting up compiler...
  [INFO] Setting up Windows MSVC...
  [OK] Found MSVC via vswhere: Visual Studio Community 2022

[Step 3/7] Checking CUDA and cuDNN...
  [OK] CUDA: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8
  [OK] cuDNN 8.x: C:/Program Files/NVIDIA/CUDNN/v8.9.7

[Step 4/7] Checking Python and NumPy...
  [OK] Python: C:/Python314/python.exe (3.14.0)
  [OK] NumPy 2.3.4

[Step 5/7] Checking other dependencies...
  [INFO] Checking Eigen...
  [OK] Found Eigen in project: ./third_party/Eigen

[OK] Configuration completed successfully!
Next steps:
  - Run build_msvc.bat (Windows MSVC)
  - Run build_msys2.bat (Windows MSYS2)
```

**步骤2：生成的文件**

配置成功后，会生成以下文件：

1. **`config/user_paths.cmake`** - 统一配置：
```cmake
# 自动生成的配置文件
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
set(CUDNN_ROOT "C:/Program Files/NVIDIA/CUDNN/v8.9.7")
set(Python3_EXECUTABLE "C:/Python314/python.exe")
set(CMAKE_TOOLCHAIN_FILE "T:/Softwares/vcpkg/scripts/buildsystems/vcpkg.cmake")
```

2. **构建脚本**（平台特定）：
   - Windows：`build_msvc.bat`, `build_msys2.bat`
   - Linux：`build.sh`

### 3. 构建过程

**自动构建（推荐）**：

*Windows MSVC*：
```cmd
.\build_msvc.bat
```

*Windows MSYS2*：
```cmd
.\build_msys2.bat
```

*Linux*：
```bash
chmod +x build.sh
./build.sh
```

**手动构建命令**：

*Windows MSVC*：
```cmd
cmake --preset windows-msvc-release
cmake --build build/windows-msvc-release --parallel
```

*Linux*：
```bash
cmake --preset linux-gcc-release
cmake --build build/linux-gcc-release --parallel
```

### 4. 测试执行

**构建输出位置**：
- Windows：`build/windows-msvc-release/tests/unit_tests/`
- Linux：`build/linux-gcc-release/tests/unit_tests/`

**运行测试**：
```bash
# Windows
build\windows-msvc-release\tests\unit_tests\eigen_openmp_hello_world.exe
build\windows-msvc-release\tests\unit_tests\test_cuda_gemm.exe

# Linux
./build/linux-gcc-release/tests/unit_tests/eigen_openmp_hello_world
./build/linux-gcc-release/tests/unit_tests/test_cuda_gemm
```

## 命令行CMake编译（推荐写法）

### 高级CMake用法

对于喜欢直接使用CMake命令的开发者：

**1. 配置**：
```bash
# Windows MSVC
cmake -B build/windows-msvc-release `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CXX_COMPILER=cl `
    -DCMAKE_TOOLCHAIN_FILE=T:/Softwares/vcpkg/scripts/buildsystems/vcpkg.cmake `
    -DENABLE_CUDA=ON

# Linux
cmake -B build/linux-gcc-release \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++-13 \
    -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DENABLE_CUDA=ON
```

**2. 构建**：
```bash
# 并行构建（推荐）
cmake --build build/windows-msvc-release --parallel 4
cmake --build build/linux-gcc-release --parallel $(nproc)

# 构建特定目标
cmake --build build/linux-gcc-release --target eigen_openmp_hello_world
```

**3. 测试**：
```bash
# 使用CTest运行测试
cd build/linux-gcc-release
ctest --output-on-failure

# 或直接运行可执行文件
./tests/unit_tests/eigen_openmp_hello_world
```

### 自定义配置选项

```bash
# 禁用CUDA（仅CPU构建）
cmake -B build -DENABLE_CUDA=OFF

# 自定义vcpkg三元组
cmake -B build -DVCPKG_TARGET_TRIPLET=x64-linux

# 调试构建
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# 详细构建输出
cmake --build build --verbose
```

### 推荐的CMake命令行工作流

**首次配置**：
```bash
# 1. 创建构建目录
mkdir -p build/my-project
cd build/my-project

# 2. 配置项目（选择适合的预设）
cmake --preset linux-gcc-release ../..
# 或手动配置
cmake ../.. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
```

**开发迭代**：
```bash
# 增量构建（只编译修改的部分）
cmake --build .

# 并行构建（利用多核）
cmake --build . --parallel $(nproc)

# 仅构建特定目标
cmake --build . --target eigen_openmp_hello_world
```

**清理重建**：
```bash
# 清理构建文件
cmake --build . --target clean

# 完全重新配置
rm -rf *
cmake --preset linux-gcc-release ../..
```

## 设计优势

1. **统一体验**：所有平台使用相同的命令
2. **零配置**：开箱即用，支持常见安装方式
3. **可扩展**：易于添加新的依赖检测逻辑
4. **可维护**：通过单一文件进行集中配置
5. **用户友好**：清晰的错误信息和有用建议
6. **跨平台**：自动处理Windows/Linux路径差异

## 故障排除

**常见问题**：

1. **找不到cuDNN**：安装cuDNN 8.x，不要安装9.x
2. **Eigen3检测失败**：检查 `./third_party/Eigen` 或通过包管理器安装
3. **找不到MSVC**：确保安装了Visual Studio 2022和C++工具
4. **CUDA编译失败**：验证CUDA 12.8+和兼容的编译器版本

**调试模式**：
```bash
python configure.py --debug  # 启用详细日志记录
```

这个配置系统为使用CUDA和cuDNN等高级依赖的跨平台C++开发提供了强大的基础。无论是初学者还是经验丰富的开发者，都能从中受益，实现快速、可靠的项目配置和构建。
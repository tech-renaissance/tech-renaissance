# Tensor::print() 方法详解

## 概述

`print()` 方法是技术觉醒框架中Tensor类的核心显示功能，提供与PyTorch风格一致的张量内容打印输出。该方法支持多设备、多数据类型的张量显示，并具有跨设备数据访问能力。

**版本**: V1.00.00
**日期**: 2025年10月29日
**作者**: 技术觉醒团队

## 方法签名

```cpp
// 默认精度打印（4位小数，与PyTorch默认一致）
void Tensor::print(const std::string& name = "") const;

// 自定义精度打印
void Tensor::print(const std::string& name, int precision) const;
```

### 参数说明

- **name** (可选): 张量显示名称，用于标识输出内容
- **precision**: 浮点数显示精度（小数位数），默认为4位

## 核心行为机制

### 1. 设备无关的数据访问

`print()` 方法通过 `to_cpu_data()` 函数实现跨设备数据访问：

```cpp
// 关键代码 (tensor.cpp:535-543)
std::vector<float> data(numel());
std::vector<int8_t> int8_data(numel());

if (dtype_ == DType::FP32) {
    to_cpu_data(data.data(), data.size() * sizeof(float));
} else if (dtype_ == DType::INT8) {
    to_cpu_data(int8_data.data(), int8_data.size() * sizeof(int8_t));
}
```

### 2. 跨设备数据传输

```cpp
// 核心传输机制 (tensor.cpp:393)
backend->copy(data, src_ptr, size, tr::CPU, this->device());
```

该方法通过Backend的统一接口，实现从任意设备到CPU的数据拷贝。

## CUDA后端特殊行为详解

### 关键问题解答

**Q: 当一个位于CUDA后端的Tensor调用print()时，它的行为是怎么样的？**

**A: 采用"临时拷贝，原张量不变"的策略：**

1. **数据临时传输**: 从CUDA显存拷贝数据到CPU内存
2. **创建临时缓冲区**: 在CPU上创建临时存储用于格式化显示
3. **原张量保持不变**: CUDA上的原始Tensor对象完全不受影响
4. **缓冲区自动销毁**: 打印完成后临时CPU缓冲区自动释放

### 详细执行流程

```cpp
// 1. 创建临时CPU缓冲区
std::vector<float> cpu_data(numel());

// 2. 跨设备数据拷贝
auto backend = get_backend();  // 获取CUDA后端
const void* cuda_ptr = data_ptr();  // CUDA设备指针
backend->copy(cpu_data.data(), cuda_ptr, size, tr::CPU, this->device());

// 3. CPU端格式化输出
format_tensor_content(oss, precision);  // 使用cpu_data进行显示

// 4. 临时缓冲区自动销毁（函数结束时）
```

### 变量指向关系

**Q: 变量名称指向的到底是CUDA上的Tensor还是CPU上的Tensor？**

**A: 变量名称始终指向原始的CUDA Tensor。**

```cpp
// 示例代码
Tensor cuda_tensor = Tensor::full(shape, 3.14f, DType::FP32, tr::CUDA[0]);
cuda_tensor.print("gpu_tensor");  // 打印后，cuda_tensor仍指向CUDA上的原始张量

// 验证：cuda_tensor.device() 仍然返回 CUDA[0]
assert(cuda_tensor.device().is_cuda());  // ✓ 通过
```

### 设备信息显示

非CPU设备的张量在打印时会显示设备信息：

```cpp
// 输出示例
gpu_tensor:
tensor([[[[3.1400, 3.1400],
           [3.1400, 3.1400]]]], device='cuda:0')
```

而CPU张量不显示设备信息：

```cpp
// 输出示例
cpu_tensor:
tensor([[[[3.1400, 3.1400],
           [3.1400, 3.1400]]]])
```

## 输出格式规范

### 1. 标量张量 (0D)
```cpp
Tensor scalar = Tensor::scalar(3.14159f);
scalar.print("pi");
// 输出: pi:
//       tensor(3.1416)
```

### 2. 1D张量
```cpp
Tensor vec = Tensor::full({4}, 1.0f);
vec.print("vector");
// 输出: vector:
//       tensor([1.0000, 1.0000, 1.0000, 1.0000])
```

### 3. 2D张量
```cpp
Tensor mat = Tensor::full({2, 3}, 2.5f);
mat.print("matrix");
// 输出: matrix:
//       tensor([[2.5000, 2.5000, 2.5000],
//               [2.5000, 2.5000, 2.5000]])
```

### 4. 3D/4D张量
```cpp
Tensor tensor4d = Tensor::full({2, 2, 2, 2}, 1.0f);
tensor4d.print("tensor4d");
// 输出: tensor4d:
//       tensor([[[[1.0000, 1.0000],
//                 [1.0000, 1.0000]],
//                [[1.0000, 1.0000],
//                 [1.0000, 1.0000]]],
//               [[[1.0000, 1.0000],
//                 [1.0000, 1.0000]],
//                [[1.0000, 1.0000],
//                 [1.0000, 1.0000]]]], device='cuda:0')
```

## 数据类型支持

### FP32浮点数
```cpp
Tensor fp32_tensor = Tensor::full({2, 2}, 3.14159f);
fp32_tensor.print("fp32", 2);  // 2位精度
// 输出: fp32:
//       tensor([[3.14, 3.14],
//               [3.14, 3.14]])
```

### INT8整数
```cpp
Tensor int8_tensor = Tensor::full({2, 2}, 42, DType::INT8);
int8_tensor.print("int8");
// 输出: int8:
//       tensor([[42, 42],
//               [42, 42]])
```

## 错误处理机制

### 1. 空张量处理
```cpp
Tensor empty;
empty.print("empty");
// 输出: empty:
//       tensor([])
```

### 2. 数据拷贝失败
```cpp
// 如果跨设备拷贝失败
try {
    cuda_tensor.print("cuda_data");
} catch (const TRException& e) {
    // 输出: cuda_data:
    //       tensor([...data unavailable...])
}
```

### 3. 设备不可用
```cpp
// 如果CUDA设备不可用
try {
    auto cuda_tensor = Tensor::full(shape, 1.0f, DType::FP32, tr::CUDA[0]);
    cuda_tensor.print("cuda");
} catch (const TRException& e) {
    // 异常信息: "CUDA backend not available"
}
```

## 性能考虑

### 1. 内存开销
- **临时缓冲区**: 创建与原张量大小相同的CPU缓冲区
- **内存峰值**: 打印期间内存使用翻倍（原始GPU内存 + 临时CPU内存）
- **自动释放**: 函数结束时临时缓冲区自动销毁

### 2. 传输开销
- **PCIe带宽**: 受GPU到CPU的PCIe带宽限制
- **同步开销**: CUDA内核同步等待计算完成
- **建议**: 大型张量避免频繁调用print()

### 3. 优化建议
```cpp
// 对于大型张量，考虑以下优化策略：

// 1. 打印前检查张量大小
if (tensor.numel() > 1000000) {
    std::cout << "Large tensor (" << tensor.numel() << " elements), skipping detailed print" << std::endl;
} else {
    tensor.print("large_tensor");
}

// 2. 使用部分切片打印
if (tensor.ndim() >= 2 && tensor.dim_size(0) > 10) {
    // 只打印前几行
    auto slice = tensor.slice({0, 0, 0, 0}, {std::min(5, tensor.dim_size(0)), -1, -1, -1});
    slice.print("tensor_preview");
    std::cout << "... (showing first 5 rows of " << tensor.dim_size(0) << ")" << std::endl;
}
```

## 使用示例

### 基础使用
```cpp
#include "tech_renaissance.h"

int main() {
    try {
        // 创建不同设备和类型的张量
        Tensor cpu_tensor = Tensor::full({2, 3}, 1.5f);
        Tensor cuda_tensor = Tensor::full({2, 3}, 2.5f, DType::FP32, tr::CUDA[0]);
        Tensor int8_tensor = Tensor::full({2, 2}, 100, DType::INT8);

        // 打印输出
        cpu_tensor.print("CPU Tensor");
        cuda_tensor.print("CUDA Tensor");
        int8_tensor.print("INT8 Tensor");

        // 自定义精度
        cpu_tensor.print("High Precision", 6);

    } catch (const TRException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
```

### 调试场景
```cpp
// 在模型训练中调试中间结果
void debug_layer_output(const Tensor& layer_output, const std::string& layer_name) {
    std::cout << "\n=== " << layer_name << " Debug Info ===" << std::endl;
    std::cout << "Shape: " << layer_output.shape().to_string() << std::endl;
    std::cout << "Device: " << layer_output.device().str() << std::endl;
    std::cout << "Data Type: " << dtype_to_string(layer_output.dtype()) << std::endl;

    // 打印张量内容（如果不太大）
    if (layer_output.numel() <= 100) {
        layer_output.print(layer_name + "_output");
    } else {
        std::cout << "Large tensor (" << layer_output.numel() << " elements), showing stats:" << std::endl;
        std::cout << "Min: " << layer_output.min().item<float>() << std::endl;
        std::cout << "Max: " << layer_output.max().item<float>() << std::endl;
        std::cout << "Mean: " << layer_output.mean().item<float>() << std::endl;
    }
}
```

## 与PyTorch的兼容性

### 相同特性
- 输出格式与PyTorch完全一致
- 默认4位小数精度
- 设备信息显示格式相同
- 多维张量的缩进格式相同

### 差异说明
- **跨设备访问**: 技术觉醒框架通过Backend抽象实现统一接口
- **性能优化**: 针对静态图特性优化了内存访问模式
- **错误处理**: 使用自定义TRException统一错误处理

## 常见问题

### Q: print()会改变张量的设备位置吗？
**A: 不会。** print()是只读操作，原张量保持原有的设备和位置不变。

### Q: 打印大型GPU张量会很慢吗？
**A: 是的。** 因为需要将数据从GPU拷贝到CPU，受PCIe带宽限制。建议对大型张量谨慎使用print()。

### Q: 可以打印空张量吗？
**A: 可以。** 空张量会显示为`tensor([])`格式。

### Q: 如果CUDA设备不可用会怎样？
**A: 会抛出TRException异常，建议使用try-catch处理。

### Q: print()是线程安全的吗？
**A: 是的。** print()函数是const成员函数，不修改张量状态，支持并发调用。

---

*文档版本: V1.00.00*
*最后更新: 2025年10月29日*
*维护团队: 技术觉醒团队*
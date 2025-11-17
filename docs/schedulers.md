# 学习率调度器系统

Tech Renaissance框架提供了7个学习率调度器，用于动态调整训练过程中的学习率。

## 基础类

### Scheduler（抽象基类）

所有学习率调度器的基类，定义了统一接口。

```cpp
class Scheduler {
public:
    explicit Scheduler(float initial_lr);
    virtual float get_lr(int epoch) = 0;
    virtual std::string type_name() const = 0;
    float get_initial_lr() const;

protected:
    float initial_lr_;  // 初始学习率
};
```

## 具体调度器

### 1. StepLR（阶梯式衰减）

按照固定步长衰减学习率。

```cpp
StepLR scheduler(0.1f, 30, 0.1f);  // 初始学习率0.1，每30个epoch衰减到0.1倍
```

**特点：**
- 每`step_size`个epoch学习率乘以`gamma`
- 适用于需要阶段性降低学习率的场景

### 2. MultiStepLR（多阶梯衰减）

在指定的里程碑epoch衰减学习率。

```cpp
std::vector<int> milestones = {30, 80, 120};
MultiStepLR scheduler(0.1f, milestones, 0.1f);  // 在30、80、120 epoch衰减
```

**特点：**
- 灵活设置衰减节点
- 支持任意间隔的milestone

### 3. ExponentialLR（指数衰减）

每个epoch按指数规律衰减学习率。

```cpp
ExponentialLR scheduler(0.1f, 0.95f);  // 初始学习率0.1，每epoch乘以0.95
```

**特点：**
- 平滑的学习率下降
- 适用于需要稳定衰减的训练

### 4. CosineAnnealingLR（余弦退火）

使用余弦函数周期性调整学习率。

```cpp
CosineAnnealingLR scheduler(0.1f, 100, 0.0f);  // 周期长度100，最小学习率0.0
```

**特点：**
- 学习率在最大值和最小值之间平滑变化
- 适用于需要周期性调整学习率的场景

### 5. CosineAnnealingWarmRestarts（带热重启的余弦退火）

余弦退火加上周期性重启机制。

```cpp
CosineAnnealingWarmRestarts scheduler(0.1f, 10, 2, 0.0f);  // 初始周期10，周期倍增因子2
```

**特点：**
- 支持固定周期（T_mult=1）或增长周期（T_mult>1）
- 每个周期结束后学习率重启到最大值
- 有助于跳出局部最优

### 6. ConstantLR（常数学习率）

保持学习率不变，用于调试和基准测试。

```cpp
ConstantLR scheduler(0.005f);  // 固定学习率0.005
```

**特点：**
- 学习率始终不变
- 适用于调试验证

## 使用示例

```cpp
#include "tech_renaissance.h"

using namespace tr;

int main() {
    // 创建调度器
    StepLR scheduler(0.1f, 30, 0.1f);

    // 获取不同epoch的学习率
    float lr0 = scheduler.get_lr(0);    // 0.1
    float lr30 = scheduler.get_lr(30);  // 0.01
    float lr60 = scheduler.get_lr(60);  // 0.001

    return 0;
}
```

## 错误处理

所有调度器都包含参数验证，无效参数会抛出`TRException`：

- `initial_lr`必须大于0
- `gamma`必须在(0, 1)范围内
- `step_size`必须为正数
- `T_max`必须为正数

## 测试

完整的单元测试位于`tests/unit_tests/test_lr_schedulers.cpp`，验证所有调度器的正确性。

```bash
./build/cmake-build-release-alpha/bin/tests/test_lr_schedulers.exe
```
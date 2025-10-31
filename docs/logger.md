# Logger API 文档

## 概述

`Logger`是技术觉醒框架的轻量级线程安全日志系统，采用单例模式设计，支持多级别日志输出、文件记录和静默模式。Logger提供结构化的日志格式，包含精确时间戳、日志等级标识和框架标记，便于调试和监控应用程序运行状态。

**版本**: V1.23.1
**更新日期**: 2025-10-30
**作者**: 技术觉醒团队

## 设计理念

### 核心设计原则

1. **单例模式**: 确保全局唯一的日志实例，提供统一的日志输出管理
2. **线程安全**: 使用互斥锁保护所有操作，支持多线程环境下的并发日志写入
3. **分级管理**: 支持DEBUG/INFO/WARN/ERROR四个日志等级，便于调试和生产环境使用
4. **静默控制**: V1.23.1新增静默模式，可选择性禁用INFO级别日志输出
5. **格式统一**: 标准化的日志格式，包含时间戳、等级标识和框架标记
6. **轻量级实现**: 基于C++标准库实现，无第三方依赖

### 关键架构特性

#### **V1.23.1静默模式功能**

新增的静默模式允许在运行时动态禁用INFO级别日志输出：
- **选择性抑制**: 只影响INFO级别，DEBUG/WARN/ERROR仍正常输出
- **全局生效**: 一次调用影响所有后续的INFO日志
- **线程安全**: 静默状态受互斥锁保护，确保多线程环境下的正确性
- **持久性**: 一旦激活，静默模式持续到程序结束

#### **单例模式实现**

使用Meyers单例模式确保线程安全的初始化：
```cpp
static Logger& get_instance() {
    static Logger instance;  // C++11保证线程安全的初始化
    return instance;
}
```

## 头文件

```cpp
#include "tech_renaissance/utils/logger.h"
```

## 日志等级

### LogLevel 枚举

| 等级 | 数值 | 描述 | 典型用途 |
|------|------|------|----------|
| `DEBUG` | 0 | 调试信息 | 开发调试、详细执行流程 |
| `INFO` | 1 | 一般信息 | 正常运行状态、重要操作记录 |
| `WARN` | 2 | 警告信息 | 潜在问题、不影响运行的异常 |
| `ERROR` | 3 | 错误信息 | 错误情况、异常处理 |

### 等级过滤机制

Logger支持基于等级的日志过滤：
- 设置日志等级为`L`时，只有等级数值 ≥ `L`的日志会被输出
- 例如：设置为`WARN`时，只输出`WARN`和`ERROR`级别日志
- `INFO`为默认等级，适合大多数应用程序使用

## 核心API

### 单例访问

#### `static Logger& get_instance()`

获取Logger的全局唯一实例。

**返回值**：
- `Logger&` - Logger实例的引用

**异常**：
- 无（保证线程安全）

**示例**：
```cpp
tr::Logger& logger = tr::Logger::get_instance();
```

### 配置管理

#### `void set_level(LogLevel level)`

设置日志输出等级。

**参数**：
- `level` - 最低输出日志等级

**行为**：
- 只有等级数值 ≥ `level`的日志会被输出
- 线程安全，立即生效

**示例**：
```cpp
// 只输出警告和错误
logger.set_level(tr::LogLevel::WARN);

// 输出所有日志
logger.set_level(tr::LogLevel::DEBUG);
```

#### `void set_output_file(const std::string& filename)`

设置日志输出文件。

**参数**：
- `filename` - 输出文件路径，空字符串表示输出到控制台

**行为**：
- 非空字符串：日志追加写入指定文件
- 空字符串：日志输出到标准控制台
- 线程安全，立即生效

**示例**：
```cpp
// 输出到文件
logger.set_output_file("application.log");

// 切换回控制台
logger.set_output_file("");
```

#### `void be_quiet()`

**V1.23.1新增**：激活静默模式，禁用所有INFO级别日志输出。

**行为**：
- 立即激活静默模式
- 只影响INFO级别日志，DEBUG/WARN/ERROR仍正常输出
- 静默模式持续到程序结束
- 线程安全，可从任何线程调用

**使用场景**：
- 生产环境减少日志噪音
- 性能测试时避免INFO干扰
- 用户应用中隐藏框架内部信息

**示例**：
```cpp
// 在main函数开始时调用
tr::Logger::get_instance().be_quiet();

// 后续所有INFO日志将被静默
logger.info("这条消息不会显示");  // 被静默
logger.warn("这条警告仍会显示");  // 正常显示
```

### 日志输出方法

#### `void debug(const std::string& message)`

输出DEBUG级别日志。

**参数**：
- `message` - 日志消息内容

**输出条件**：
- 当前日志等级 ≤ DEBUG
- DEBUG级别不受静默模式影响

**示例**：
```cpp
logger.debug("变量值: " + std::to_string(variable));
logger.debug("进入函数: " + __FUNCTION__);
```

#### `void info(const std::string& message)`

输出INFO级别日志。

**参数**：
- `message` - 日志消息内容

**输出条件**：
- 当前日志等级 ≤ INFO
- 静默模式未激活

**示例**：
```cpp
logger.info("应用程序启动");
logger.info("操作完成，处理了 " + std::to_string(count) + " 个项目");
```

#### `void warn(const std::string& message)`

输出WARN级别日志。

**参数**：
- `message` - 日志消息内容

**输出条件**：
- 当前日志等级 ≤ WARN
- WARN级别不受静默模式影响

**示例**：
```cpp
logger.warn("配置文件未找到，使用默认配置");
logger.warn("内存使用率较高: " + std::to_string(usage) + "%");
```

#### `void error(const std::string& message)`

输出ERROR级别日志。

**参数**：
- `message` - 日志消息内容

**输出条件**：
- 当前日志等级 ≤ ERROR
- ERROR级别不受静默模式影响

**示例**：
```cpp
logger.error("文件打开失败: " + filename);
logger.error("网络连接超时");
```

## 日志格式

### 标准格式

所有日志输出采用统一的格式：

```
[YYYY-MM-DD HH:MM:SS.mmm] [LEVEL] [TR] message
```

**格式说明**：
- `YYYY-MM-DD HH:MM:SS.mmm`: 精确到毫秒的时间戳
- `LEVEL`: 日志等级（DEBUG/INFO/WARN/ERROR）
- `[TR]`: 技术觉醒框架标识
- `message`: 用户提供的日志消息

### 格式示例

```
[2025-10-30 14:37:54.524] [INFO] [TR] Initializing backend manager...
[2025-10-30 14:37:54.524] [INFO] [TR] CPU backend initialized
[2025-10-30 14:37:56.157] [WARN] [TR] CUDA device memory usage high
[2025-10-30 14:37:56.209] [ERROR] [TR] Failed to allocate GPU memory
```

## 便捷宏定义

### 日志宏

为简化日志调用，框架提供以下便捷宏：

```cpp
#define TR_LOG_DEBUG(message) tr::Logger::get_instance().debug(message)
#define TR_LOG_INFO(message)  tr::Logger::get_instance().info(message)
#define TR_LOG_WARN(message)  tr::Logger::get_instance().warn(message)
#define TR_LOG_ERROR(message) tr::Logger::get_instance().error(message)
```

**使用示例**：
```cpp
TR_LOG_INFO("应用程序启动完成");
TR_LOG_WARN("使用默认配置");
TR_LOG_ERROR("初始化失败");
```

**优点**：
- 简化代码编写
- 统一调用方式
- 便于日志管理工具识别

## 使用示例

### 基础使用

```cpp
#include "tech_renaissance/utils/logger.h"

using namespace tr;

int main() {
    // 获取Logger实例
    Logger& logger = Logger::get_instance();

    // 输出各级别日志
    logger.debug("调试信息：变量初始化");
    logger.info("应用程序启动");
    logger.warn("配置文件未找到，使用默认设置");
    logger.error("无法连接到数据库");

    return 0;
}
```

### 配置管理

```cpp
#include "tech_renaissance/utils/logger.h"

int main() {
    Logger& logger = Logger::get_instance();

    // 设置日志级别为WARNING
    logger.set_level(LogLevel::WARN);

    // 设置输出文件
    logger.set_output_file("app.log");

    // 只有WARNING和ERROR会被记录到文件
    logger.debug("这条调试信息不会显示");  // 被过滤
    logger.info("这条信息不会显示");        // 被过滤
    logger.warn("这条警告会显示");          // 输出到文件
    logger.error("这个错误会显示");         // 输出到文件

    return 0;
}
```

### V1.23.1静默模式使用

```cpp
#include "tech_renaissance/utils/logger.h"

int main() {
    // 激活静默模式，禁用INFO级别日志
    Logger::get_instance().be_quiet();

    // 设置日志级别为DEBUG（包含INFO）
    Logger::get_instance().set_level(LogLevel::DEBUG);

    // INFO级别被静音，但其他级别正常
    Logger::get_instance().debug("调试信息仍显示");  // 显示
    Logger::get_instance().info("信息被静音");      // 不显示
    Logger::get_instance().warn("警告仍显示");      // 显示
    Logger::get_instance().error("错误仍显示");     // 显示

    return 0;
}
```

### 多线程环境使用

```cpp
#include "tech_renaissance/utils/logger.h"
#include <thread>
#include <vector>

void worker_thread(int thread_id) {
    Logger& logger = Logger::get_instance();

    logger.info("线程 " + std::to_string(thread_id) + " 开始工作");

    // 模拟工作
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    logger.info("线程 " + std::to_string(thread_id) + " 完成工作");
}

int main() {
    // 激活静默模式减少输出噪音
    Logger::get_instance().be_quiet();

    std::vector<std::thread> threads;

    // 创建多个工作线程
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back(worker_thread, i);
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    Logger::get_instance().warn("所有线程工作完成");

    return 0;
}
```

### 生产环境配置

```cpp
#include "tech_renaissance/utils/logger.h"

void configure_production_logging() {
    Logger& logger = Logger::get_instance();

    // 生产环境推荐配置
    logger.be_quiet();                    // 静默INFO，减少噪音
    logger.set_level(LogLevel::WARN);     // 只记录警告和错误
    logger.set_output_file("production.log");  // 输出到文件

    logger.warn("生产环境日志配置完成");
}

int main() {
    configure_production_logging();

    // 应用程序逻辑...

    return 0;
}
```

## 性能特征

### 时间复杂度

- **日志输出**: O(1) - 常数时间复杂度
- **等级检查**: O(1) - 简单整数比较
- **静默检查**: O(1) - 布尔值检查
- **时间戳生成**: O(1) - 系统调用优化

### 内存使用

- **实例大小**: 约64-96字节（包含互斥锁）
- **线程栈**: 最小额外内存开销
- **文件输出**: 按需打开，无内存缓存

### 线程安全性能

- **互斥锁粒度**: 细粒度锁定，最小化竞争
- **锁持有时间**: 微秒级别，快速释放
- **并发支持**: 支持高并发日志写入

## 最佳实践

### 1. 等级使用建议

```cpp
// 推荐：合理使用不同等级
logger.debug("进入函数，参数: " + param);     // 开发调试
logger.info("操作成功完成");                  // 重要状态
logger.warn("使用默认值: " + default_value);  // 潜在问题
logger.error("操作失败: " + error_msg);       // 错误情况
```

### 2. V1.23.1静默模式使用

```cpp
// 推荐：生产环境激活静默模式
#ifdef NDEBUG
    Logger::get_instance().be_quiet();
    Logger::get_instance().set_level(LogLevel::WARN);
#endif

// 推荐：性能测试时静音INFO
if (performance_test_mode) {
    Logger::get_instance().be_quiet();
}
```

### 3. 消息格式化

```cpp
// 推荐：使用字符串拼接提供上下文
int processed = 100;
int total = 150;
logger.info("进度: " + std::to_string(processed) + "/" + std::to_string(total));

// 避免：过于简略的消息
logger.info("处理完成");  // 缺少具体信息
```

### 4. 文件输出管理

```cpp
// 推荐：应用程序启动时设置输出文件
Logger::get_instance().set_output_file("app_" + get_date_string() + ".log");

// 推荐：重要错误前切换到文件输出
if (critical_error) {
    Logger::get_instance().set_output_file("critical_errors.log");
    Logger::get_instance().error("严重错误: " + error_details);
}
```

### 5. 多线程环境

```cpp
// 推荐：在多线程环境中直接使用Logger实例
void thread_function() {
    // 线程安全，无需额外同步
    Logger::get_instance().info("线程开始工作");
}
```

## 错误处理

### 文件输出错误

当设置文件输出失败时，Logger会：
- 继续使用控制台输出
- 不抛出异常，保证程序稳定运行
- 在下次日志写入时重试文件打开

### 时间戳生成错误

时间戳生成异常时的处理：
- 使用"1970-01-01 00:00:00.000"作为默认时间戳
- 保证日志格式完整性
- 不影响日志消息输出

## 集成示例

### 与BackendManager集成

```cpp
// BackendManager中的典型使用
void BackendManager::initialize() {
    Logger::get_instance().info("Initializing backend manager...");

    try {
        // 初始化逻辑...
        Logger::get_instance().info("Backend manager initialized successfully");
    } catch (const std::exception& e) {
        Logger::get_instance().error("Backend manager initialization failed: " + std::string(e.what()));
        throw;
    }
}
```

### 与CUDA后端集成

```cpp
// CUDA后端中的日志使用
CudaBackend::CudaBackend(int device_id) : device_id_(device_id) {
    Logger::get_instance().info("CUDA backend initialized for device " + std::to_string(device_id));

    if (!check_device_availability()) {
        Logger::get_instance().warn("CUDA device " + std::to_string(device_id) + " not available");
    }
}
```

## 故障排除

### 常见问题

1. **日志不显示**
   - 检查日志等级设置是否正确
   - 确认是否激活了静默模式
   - 验证输出目标（控制台vs文件）

2. **文件输出失败**
   - 检查文件路径是否存在
   - 验证写入权限
   - 确认磁盘空间充足

3. **性能问题**
   - 考虑在性能关键路径使用静默模式
   - 适当提高日志等级减少输出
   - 使用异步日志缓冲（未来版本）

### 调试技巧

```cpp
// 临时启用所有日志进行调试
Logger::get_instance().set_level(LogLevel::DEBUG);
Logger::get_instance().set_output_file("debug.log");

// 检查当前配置
// 注意：Logger不提供状态查询方法，需要自己跟踪配置状态
```

## 版本信息

- **版本**: V1.23.1
- **更新日期**: 2025-10-30
- **主要特性**:
  - 静默模式支持（`be_quiet()`方法）
  - 线程安全的单例模式
  - 标准化的日志格式
  - 多等级日志过滤
  - 文件和控制台输出支持

## 未来扩展

### 计划中的功能

1. **异步日志**: 后台线程处理日志写入，提高性能
2. **日志轮转**: 自动分割和压缩历史日志文件
3. **格式化器**: 支持自定义日志格式和字段
4. **网络日志**: 支持远程日志服务器
5. **性能监控**: 内置日志性能统计和分析

### 扩展API预览

```cpp
// 未来可能的功能
Logger::get_instance().set_async_mode(true);
Logger::get_instance().enable_log_rotation("app.log", 10 * 1024 * 1024);  // 10MB
Logger::get_instance().set_custom_formatter("[%H:%M:%S] [%l] %v");
Logger::get_instance().add_remote_server("logserver.example.com:514");
```

---

## 关键设计原则总结

### 单例与线程安全
- **Meyers单例**: C++11保证线程安全的初始化
- **互斥锁保护**: 所有公共方法都是线程安全的
- **无竞争条件**: 静默状态和配置变更都有适当保护

### V1.23.1静默模式
- **选择性抑制**: 只影响INFO级别，保持重要错误信息可见
- **全局生效**: 一次调用影响整个应用程序的日志行为
- **即时响应**: 配置变更立即生效，无需重启

### 轻量级设计
- **标准库实现**: 无第三方依赖，易于集成
- **最小开销**: 快速的等级检查和格式化
- **内存效率**: 单实例设计，避免重复分配

### 格式标准化
- **统一格式**: 时间戳 + 等级 + 标识 + 消息
- **精确时间**: 毫秒级时间戳，便于问题定位
- **框架标识**: [TR]标记便于日志分析和过滤
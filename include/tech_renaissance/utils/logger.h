/**
 * @file logger.h
 * @brief 日志器类声明
 * @details 高性能、线程安全的单例日志器，支持格式化输出、多级日志、文件/控制台切换。
 * @version 1.19.01
 * @date 2025-10-31
 * @author 技术觉醒团队
 * @note 依赖项: string, memory, mutex, fstream, sstream
 * @note 所属系列: utils
 */

#pragma once

#include <string>
#include <memory>
#include <mutex>
#include <fstream>
#include <sstream>
#include <chrono>

namespace tr {

/**
 * @brief 日志等级枚举
 */
enum class LogLevel {
    DEBUG = 0,  ///< 调试信息，用于开发阶段的详细追踪
    INFO = 1,   ///< 一般信息，用于关键流程和状态变化的报告
    WARN = 2,   ///< 警告信息，表示潜在问题，但不影响程序运行
    ERROR = 3   ///< 错误信息，表示严重问题，可能导致程序失败
};

class Logger {
public:
    /**
     * @brief 获取日志器单例实例
     * @details 采用'永不销毁'的单例模式，避免静态对象销毁顺序问题。
     * @return Logger& 日志器实例引用
     */
    static Logger& get_instance();

    /**
     * @brief 禁用拷贝构造函数和赋值操作
     */
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    /**
     * @brief 设置全局日志等级
     * @param level 新的日志等级
     */
    void set_level(LogLevel level);

    /**
     * @brief 设置日志输出文件。如果文件名为空，则输出到控制台。
     * @param filename 日志文件名。如果文件已打开，会先关闭旧文件。
     */
    void set_output_file(const std::string& filename);

    /**
     * @brief 设置静默模式
     * @param quiet 如果为true，所有INFO级别的日志将被禁用
     */
    void set_quiet_mode(bool quiet);

    /**
     * @brief 格式化输出DEBUG等级日志
     * @tparam Args 参数包类型
     * @param args 日志消息的各个部分，将被自动拼接
     */
    template<typename... Args>
    void debug(const Args&... args);

    /**
     * @brief 格式化输出INFO等级日志
     * @tparam Args 参数包类型
     * @param args 日志消息的各个部分，将被自动拼接
     */
    template<typename... Args>
    void info(const Args&... args);

    /**
     * @brief 格式化输出WARN等级日志
     * @tparam Args 参数包类型
     * @param args 日志消息的各个部分，将被自动拼接
     */
    template<typename... Args>
    void warn(const Args&... args);

    /**
     * @brief 格式化输出ERROR等级日志
     * @tparam Args 参数包类型
     * @param args 日志消息的各个部分，将被自动拼接
     */
    template<typename... Args>
    void error(const Args&... args);

private:
    /**
     * @brief 构造函数（私有）
     */
    Logger();

    /**
     * @brief 析构函数（私有），确保文件流被正确关闭
     */
    ~Logger();

    /**
     * @brief 内部日志输出实现
     * @param level 日志等级
     * @param message 最终格式化好的日志消息
     */
    void log_internal(LogLevel level, const std::string& message);

    /**
     * @brief 可变参数模板的消息格式化函数
     * @tparam Args 参数包类型
     * @param args 待格式化的参数
     * @return std::string 格式化后的字符串
     */
    template<typename... Args>
    std::string format_message(const Args&... args);

    // 成员变量
    mutable std::mutex _mutex;           ///< 保证线程安全
    LogLevel _current_level;             ///< 当前日志等级
    std::ofstream _file_stream;          ///< 持久化的文件输出流
    bool _quiet_mode;                    ///< 静默模式开关
};

/**
 * @brief 全局初始化函数，建议在main函数开始时调用
 * @param filename 日志文件名，默认为空（输出到控制台）
 * @param level 日志等级，默认为INFO
 * @param quiet 是否开启静默模式，默认为false
 */
void InitLogger(const std::string& filename = "", LogLevel level = LogLevel::INFO, bool quiet = false);

// --- Template Implementations ---

template<typename... Args>
void Logger::debug(const Args&... args) {
    if (_current_level <= LogLevel::DEBUG) {
        log_internal(LogLevel::DEBUG, format_message(args...));
    }
}

template<typename... Args>
void Logger::info(const Args&... args) {
    // 检查静默模式和日志等级
    if (!_quiet_mode && _current_level <= LogLevel::INFO) {
        log_internal(LogLevel::INFO, format_message(args...));
    }
}

template<typename... Args>
void Logger::warn(const Args&... args) {
     if (_current_level <= LogLevel::WARN) {
        log_internal(LogLevel::WARN, format_message(args...));
    }
}

template<typename... Args>
void Logger::error(const Args&... args) {
     if (_current_level <= LogLevel::ERROR) {
        log_internal(LogLevel::ERROR, format_message(args...));
    }
}

template<typename... Args>
std::string Logger::format_message(const Args&... args) {
    std::ostringstream oss;
    // 使用C++17的折叠表达式实现优雅的参数拼接
    (oss << ... << args);
    return oss.str();
}

} // namespace tr

/**
 * @brief 便捷的日志输出宏，支持可变参数
 */
#define TR_LOG_DEBUG(...) tr::Logger::get_instance().debug(__VA_ARGS__)
#define TR_LOG_INFO(...)  tr::Logger::get_instance().info(__VA_ARGS__)
#define TR_LOG_WARN(...)  tr::Logger::get_instance().warn(__VA_ARGS__)
#define TR_LOG_ERROR(...) tr::Logger::get_instance().error(__VA_ARGS__)
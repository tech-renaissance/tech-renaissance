/**
 * @file logger.h
 * @brief 日志器类声明
 * @details 轻量级线程安全日志器，支持 DEBUG/INFO/WARN/ERROR 等级及文件输出，通过 C++ 标准库实现。
 * @version 1.01.00
 * @date 2025-10-24
 * @author 技术觉醒团队
 * @note 依赖项: iostream, fstream, string, mutex, chrono
 * @note 所属系列: export
 */

#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <mutex>
#include <chrono>

namespace tr {

/**
 * @brief 日志等级枚举
 */
enum class LogLevel {
    DEBUG = 0,  ///< 调试信息
    INFO = 1,   ///< 一般信息
    WARN = 2,   ///< 警告信息
    ERROR = 3   ///< 错误信息
};

/**
 * @brief 日志器类
 * @details 轻量级日志输出，支持不同等级和文件输出
 */
class Logger {
public:
    /**
     * @brief 获取日志器单例实例
     * @return Logger& 日志器实例引用
     */
    static Logger& get_instance();

    /**
     * @brief 析构函数
     */
    ~Logger();

    /**
     * @brief 禁用拷贝构造函数和赋值操作
     */
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    /**
     * @brief 设置日志等级
     * @param level 日志等级
     */
    void set_level(LogLevel level);

    /**
     * @brief 设置日志输出文件
     * @param filename 文件名
     */
    void set_output_file(const std::string& filename);

    /**
     * @brief 输出DEBUG等级日志
     * @param message 日志消息
     */
    void debug(const std::string& message);

    /**
     * @brief 输出INFO等级日志
     * @param message 日志消息
     */
    void info(const std::string& message);

    /**
     * @brief 输出WARN等级日志
     * @param message 日志消息
     */
    void warn(const std::string& message);

    /**
     * @brief 输出ERROR等级日志
     * @param message 日志消息
     */
    void error(const std::string& message);

    /**
     * @brief 静默模式：禁用所有INFO级别的日志输出
     * @details 一旦调用，所有info()方法将被禁用，直到程序结束
     */
    void be_quiet();

private:
    /**
     * @brief 构造函数（私有）
     */
    Logger();

    mutable std::mutex mutex_;   // 线程安全
    LogLevel current_level_;     // 当前日志等级
    std::string output_file_;    // 输出文件名
    bool use_file_output_;       // 是否使用文件输出
    bool quiet_mode_;            // 静默模式：禁用INFO日志

    /**
     * @brief 获取当前时间戳字符串
     * @return std::string 时间戳字符串
     */
    std::string get_timestamp() const;

    /**
     * @brief 获取日志等级字符串
     * @param level 日志等级
     * @return std::string 等级字符串
     */
    std::string get_level_string(LogLevel level) const;

    /**
     * @brief 内部日志输出实现
     * @param level 日志等级
     * @param message 日志消息
     */
    void log_internal(LogLevel level, const std::string& message);
};

} // namespace tr

/**
 * @brief 便捷的日志输出宏
 */
#define TR_LOG_DEBUG(message) tr::Logger::get_instance().debug(message)
#define TR_LOG_INFO(message)  tr::Logger::get_instance().info(message)
#define TR_LOG_WARN(message)  tr::Logger::get_instance().warn(message)
#define TR_LOG_ERROR(message) tr::Logger::get_instance().error(message)
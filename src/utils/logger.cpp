/**
 * @file logger.cpp
 * @brief 日志器实现
 * @details 实现高性能、线程安全的单例日志系统，支持格式化输出。
 * @version 1.19.01
 * @date 2025-10-31
 * @author 技术觉醒团队
 * @note 依赖项: logger.h, iostream, chrono, iomanip
 * @note 所属系列: utils
 */
#include "tech_renaissance/utils/logger.h"
#include <iostream>
#include <chrono>
#include <iomanip>

namespace tr {

// 将获取时间戳和等级字符串的辅助函数移到匿名命名空间中，作为文件的内部实现
namespace {
    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &time_t_now);
#else
        localtime_r(&time_t_now, &tm_buf); // 使用线程安全的localtime_r
#endif

        std::ostringstream oss;
        oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
        oss << "." << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

    const char* get_level_string(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO";
            case LogLevel::WARN:  return "WARN";
            case LogLevel::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }
} // namespace

Logger& Logger::get_instance() {
    // 采用"泄漏的单例"(Leaky Singleton)模式。
    // 指针在首次调用时被创建，且永不销毁。
    // 这可以完美避免静态对象在程序退出时的销毁顺序问题。
    // 对于全局日志器，这是一个标准且可靠的实践。
    static Logger* instance = new Logger();
    return *instance;
}

Logger::Logger() : _current_level(LogLevel::INFO), _quiet_mode(false) {
    // 构造函数中不做任何复杂操作
}

Logger::~Logger() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_file_stream.is_open()) {
        _file_stream.close();
    }
}

void Logger::set_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(_mutex);
    _current_level = level;
}

void Logger::set_output_file(const std::string& filename) {
    std::lock_guard<std::mutex> lock(_mutex);
    // 如果已有文件流打开，先关闭
    if (_file_stream.is_open()) {
        _file_stream.close();
    }

    if (!filename.empty()) {
        _file_stream.open(filename, std::ios::app);
        if (!_file_stream.is_open()) {
            // 如果文件打开失败，输出一个错误到控制台
            std::cerr << "[" << get_timestamp() << "] "
                      << "[ERROR] [TR] Failed to open log file: " << filename << std::endl;
        }
    }
}

void Logger::set_quiet_mode(bool quiet) {
    std::lock_guard<std::mutex> lock(_mutex);
    _quiet_mode = quiet;
}

void Logger::log_internal(LogLevel level, const std::string& message) {
    // 仅在最底层加锁，修复了死锁问题
    std::lock_guard<std::mutex> lock(_mutex);

    std::ostringstream oss;
    oss << "[" << get_timestamp() << "] "
        << "[" << get_level_string(level) << "] "
        << "[TR] " << message;

    std::string log_line = oss.str();

    if (_file_stream.is_open()) {
        _file_stream << log_line << std::endl;
    } else {
        // 如果等级是ERROR，输出到std::cerr，否则输出到std::cout
        if (level >= LogLevel::ERROR) {
            std::cerr << log_line << std::endl;
        } else {
            std::cout << log_line << std::endl;
        }
    }
}

void InitLogger(const std::string& filename, LogLevel level, bool quiet) {
    auto& logger = Logger::get_instance();
    logger.set_level(level);
    logger.set_output_file(filename);
    logger.set_quiet_mode(quiet);
}

} // namespace tr
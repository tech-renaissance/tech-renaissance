/**
 * @file logger.cpp
 * @brief 日志器实现
 * @details 实现轻量级线程安全日志系统，可输出控制台或文件
 * @version 1.01.00
 * @date 2025-10-24
 * @author 技术觉醒团队
 * @note 依赖项: logger.h, chrono, iostream, fstream, sstream
 * @note 所属系列: export
 */

#include "tech_renaissance/utils/logger.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

namespace tr {

Logger& Logger::get_instance() {
    static Logger instance;
    return instance;
}

Logger::Logger() : current_level_(LogLevel::INFO), use_file_output_(false), output_file_(""), quiet_mode_(false) {
}

Logger::~Logger() {
}

void Logger::set_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_level_ = level;
}

void Logger::set_output_file(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    output_file_ = filename;
    use_file_output_ = !filename.empty();
}

void Logger::debug(const std::string& message) {
    if (LogLevel::DEBUG >= current_level_) {
        log_internal(LogLevel::DEBUG, message);
    }
}

void Logger::info(const std::string& message) {
    // 检查静默模式和日志等级
    std::lock_guard<std::mutex> lock(mutex_);
    if (!quiet_mode_ && LogLevel::INFO >= current_level_) {
        log_internal(LogLevel::INFO, message);
    }
}

void Logger::warn(const std::string& message) {
    if (LogLevel::WARN >= current_level_) {
        log_internal(LogLevel::WARN, message);
    }
}

void Logger::error(const std::string& message) {
    if (LogLevel::ERROR >= current_level_) {
        log_internal(LogLevel::ERROR, message);
    }
}

void Logger::be_quiet() {
    std::lock_guard<std::mutex> lock(mutex_);
    quiet_mode_ = true;
}

std::string Logger::get_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

std::string Logger::get_level_string(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

void Logger::log_internal(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream oss;
    oss << "[" << get_timestamp() << "] "
        << "[" << get_level_string(level) << "] "
        << "[TR] " << message;

    std::string log_line = oss.str();

    if (use_file_output_) {
        std::ofstream file(output_file_, std::ios::app);
        if (file.is_open()) {
            file << log_line << std::endl;
        }
    } else {
        std::cout << log_line << std::endl;
    }
}

} // namespace tr
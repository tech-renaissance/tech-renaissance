/**
 * @file tr_exception.h
 * @brief 框架异常类声明
 * @details 技术觉醒框架统一异常类，封装错误信息和堆栈跟踪，继承自std::exception
 * @version 1.00.00
 * @date 2025-10-23
 * @author 技术觉醒团队
 * @note 依赖项: std::exception
 * @note 所属系列: export
 */

#pragma once

#include <exception>
#include <string>
#include <sstream>

namespace tr {

/**
 * @brief 技术觉醒框架统一异常类
 * @details 继承自std::exception，封装错误信息、文件名和行号
 */
class TRException : public std::exception {
public:
    /**
     * @brief 构造函数
     * @param message 错误消息
     * @param file 源文件名（可选）
     * @param line 行号（可选）
     */
    explicit TRException(const std::string& message,
                        const std::string& file = "",
                        int line = 0);

    /**
     * @brief 析构函数
     */
    ~TRException() noexcept override = default;

    /**
     * @brief 获取错误消息
     * @return const char* C风格字符串错误消息
     */
    const char* what() const noexcept override;

    /**
     * @brief 获取文件名
     * @return const std::string& 源文件名
     */
    const std::string& file() const noexcept { return file_; }

    /**
     * @brief 获取行号
     * @return int 行号
     */
    int line() const noexcept { return line_; }

private:
    std::string message_;      // 错误消息
    std::string file_;         // 源文件名
    int line_;                 // 行号
    mutable std::string what_; // 缓存的what()结果

    /**
     * @brief 构建完整的错误消息
     */
    void build_what() const;
};

} // namespace tr

/**
 * @brief 抛出技术觉醒异常的便捷宏
 * @param message 错误消息
 */
#define TR_THROW(message) \
    throw tr::TRException(message, __FILE__, __LINE__)

/**
 * @brief 条件抛出异常的便捷宏
 * @param condition 条件
 * @param message 错误消息
 */
#define TR_THROW_IF(condition, message) \
    do { \
        if (condition) { \
            TR_THROW(message); \
        } \
    } while(0)
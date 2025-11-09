/**
 * @file tr_exception.h
 * @brief 框架异常类声明（含子类）
 * @details 技术觉醒框架统一异常体系，支持错误类型分类，完全向后兼容
 * @version 1.10.00
 * @date 2025-11-09
 * @author 技术觉醒团队
 * @note 依赖项: std::exception
 * @note 所属系列: utils
 */

#pragma once

#include <exception>
#include <string>
#include <sstream>

namespace tr {

/**
 * @brief 框架统一异常基类
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

    /** @brief 返回异常类型（默认：TRException） */
    virtual const char* type() const noexcept { return "TRException"; }

protected:
    std::string message_;      // 错误消息
    std::string file_;         // 源文件名
    int line_;                 // 行号
    mutable std::string what_; // 缓存的what()结果

    /**
     * @brief 构建完整的错误消息
     */
    void build_what() const;
};

/**
 * @brief 文件未找到异常
 */
class FileNotFoundError : public TRException {
public:
    using TRException::TRException;
    const char* type() const noexcept override { return "FileNotFoundError"; }
};

/**
 * @brief 未实现功能异常
 */
class NotImplementedError : public TRException {
public:
    using TRException::TRException;
    const char* type() const noexcept override { return "NotImplementedError"; }
};

/**
 * @brief 除零异常
 */
class ZeroDivisionError : public TRException {
public:
    using TRException::TRException;
    const char* type() const noexcept override { return "ZeroDivisionError"; }
};

/**
 * @brief 类型错误异常
 */
class TypeError : public TRException {
public:
    using TRException::TRException;
    const char* type() const noexcept override { return "TypeError"; }
};

/**
 * @brief 数值/参数取值错误异常
 */
class ValueError : public TRException {
public:
    using TRException::TRException;
    const char* type() const noexcept override { return "ValueError"; }
};

/**
 * @brief 索引越界异常
 */
class IndexError : public TRException {
public:
    using TRException::TRException;
    const char* type() const noexcept override { return "IndexError"; }
};

} // namespace tr

// ========================= 宏定义 =========================

/** 旧版兼容宏：抛出通用异常 */
#define TR_THROW(message) \
    throw tr::TRException(message, __FILE__, __LINE__)

/** 条件抛出通用异常 */
#define TR_THROW_IF(condition, message) \
    do { if (condition) { TR_THROW(message); } } while (0)

/** 新版宏：按类型抛出特定异常 */
#define TR_THROW_TYPE(ExceptionType, message) \
    throw tr::ExceptionType(message, __FILE__, __LINE__)

#define TR_THROW_NOT_IMPLEMENTED(msg) TR_THROW_TYPE(NotImplementedError, msg)
#define TR_THROW_FILE_NOT_FOUND(msg)  TR_THROW_TYPE(FileNotFoundError, msg)
#define TR_THROW_VALUE_ERROR(msg)     TR_THROW_TYPE(ValueError, msg)
#define TR_THROW_INDEX_ERROR(msg)     TR_THROW_TYPE(IndexError, msg)
#define TR_THROW_TYPE_ERROR(msg)      TR_THROW_TYPE(TypeError, msg)
#define TR_THROW_ZERO_DIVISION(msg)   TR_THROW_TYPE(ZeroDivisionError, msg)
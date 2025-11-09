/**
 * @file tr_exception.cpp
 * @brief 框架异常类实现（含子类）
 * @details 实现统一异常体系的错误信息封装和格式化输出，支持类型分类
 * @version 1.10.00
 * @date 2025-11-09
 * @author 技术觉醒团队
 * @note 依赖项: tr_exception.h
 * @note 所属系列: utils
 */

#include "tech_renaissance/utils/tr_exception.h"
#include <iomanip>

namespace tr {

TRException::TRException(const std::string& message,
                         const std::string& file,
                         int line)
    : message_(message), file_(file), line_(line) {
    // Don't build what() in constructor to allow virtual type() to work correctly
    // build_what() will be called on first what() access
}

const char* TRException::what() const noexcept {
    if (what_.empty()) {
        build_what();  // Build on first access
    }
    return what_.c_str();
}

void TRException::build_what() const {
    std::ostringstream oss;
    oss << type() << ": " << message_;

    if (!file_.empty()) {
        oss << " (File: " << file_;
        if (line_ > 0) {
            oss << ", Line: " << line_;
        }
        oss << ")";
    }

    what_ = oss.str();
}

} // namespace tr
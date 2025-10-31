/**
 * @file tr_exception.cpp
 * @brief 框架异常类实现
 * @details 实现统一异常类的错误信息封装和格式化输出
 * @version 1.00.00
 * @date 2025-10-23
 * @author 技术觉醒团队
 * @note 依赖项: tr_exception.h
 * @note 所属系列: export
 */

#include "tech_renaissance/utils/tr_exception.h"
#include <iomanip>

namespace tr {

TRException::TRException(const std::string& message,
                         const std::string& file,
                         int line)
    : message_(message), file_(file), line_(line) {
    build_what();
}

const char* TRException::what() const noexcept {
    return what_.c_str();
}

void TRException::build_what() const {
    std::ostringstream oss;
    oss << "TRException: " << message_;

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
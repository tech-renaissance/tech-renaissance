/**
 * @file strides.cpp
 * @brief 步长类实现
 * @details 实现4维张量的步长管理和连续存储检查
 * @version 1.00.00
 * @date 2025-11-16
 * @author 技术觉醒团队
 * @note 依赖项: shape.h
 * @note 所属系列: data
 */

#include "tech_renaissance/data/strides.h"
#include "tech_renaissance/data/shape.h"
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace tr {

Strides::Strides() noexcept
    : strides_{0, 0, 0, 0} {
}

Strides::Strides(const Shape& shape) {
    // 根据连续存储布局计算步长（从右到左累乘）
    // 对于连续存储，最后一个维度的步长为1

    // W维度步长总是1（最内层）
    strides_[3] = 1;

    // H维度步长 = W维度大小
    strides_[2] = shape.w();

    // C维度步长 = H维度大小 * W维度大小
    strides_[1] = shape.h() * shape.w();

    // N维度步长 = C维度大小 * H维度大小 * W维度大小
    strides_[0] = shape.c() * shape.h() * shape.w();
}

Strides::Strides(int64_t n, int64_t c, int64_t h, int64_t w) noexcept
    : strides_{n, c, h, w} {
}

int64_t Strides::stride(int32_t dim) const {
    validate_dim(dim);
    return strides_[dim];
}

bool Strides::operator==(const Strides& other) const noexcept {
    return strides_ == other.strides_;
}

bool Strides::operator!=(const Strides& other) const noexcept {
    return !(*this == other);
}

int64_t Strides::get_offset(int64_t n, int64_t c, int64_t h, int64_t w) const noexcept {
    return n * strides_[0] + c * strides_[1] + h * strides_[2] + w * strides_[3];
}

std::string Strides::to_string() const {
    std::ostringstream oss;
    oss << "Strides(" << strides_[0] << ", " << strides_[1] << ", "
        << strides_[2] << ", " << strides_[3] << ")";
    return oss.str();
}

bool Strides::is_contiguous(const Shape& shape) const {
    // 检查当前步长是否与连续存储一致
    Strides contiguous_strides(shape);
    return *this == contiguous_strides;
}

void Strides::validate_dim(int32_t dim) const {
    if (dim < 0 || dim >= 4) {
        throw std::out_of_range("[Strides::stride] Dimension index out of range: " + std::to_string(dim));
    }
}

} // namespace tr
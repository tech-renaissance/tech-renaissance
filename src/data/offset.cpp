/**
 * @file offset.cpp
 * @brief 张量偏移量类实现
 * @details 用于定义张量在各个维度上的起始、结束位置和步长
 * @version 1.00.00
 * @date 2025-11-03
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "tech_renaissance/data/offset.h"
#include <string>

namespace tr {

// 验证函数实现
void Offset::validate_range(int32_t start, int32_t end, const std::string& dim_name) const {
    if (!(end == -1 || start < end)) {
        throw TRException("Offset: invalid range for " + dim_name + " dimension, start=" +
                         std::to_string(start) + ", end=" + std::to_string(end) +
                         ". Expected: end == -1 OR start < end");
    }
}

void Offset::validate_stride(int32_t stride, const std::string& dim_name) const {
    if (stride <= 0) {
        throw TRException("Offset: invalid stride for " + dim_name + " dimension, stride=" +
                         std::to_string(stride) + ". Expected: stride > 0");
    }
}

// 仅W维度的构造函数
Offset::Offset(int32_t w_start, int32_t w_end)
    : w_start_(w_start), w_end_(w_end), w_stride_(1),
      h_start_(0), h_end_(-1), h_stride_(1),
      c_start_(0), c_end_(-1), c_stride_(1),
      n_start_(0), n_end_(-1), n_stride_(1) {

    validate_range(w_start, w_end, "W");
}

// H和W维度的构造函数
Offset::Offset(int32_t h_start, int32_t h_end, int32_t w_start, int32_t w_end)
    : w_start_(w_start), w_end_(w_end), w_stride_(1),
      h_start_(h_start), h_end_(h_end), h_stride_(1),
      c_start_(0), c_end_(-1), c_stride_(1),
      n_start_(0), n_end_(-1), n_stride_(1) {

    validate_range(h_start, h_end, "H");
    validate_range(w_start, w_end, "W");
}

// C、H和W维度的构造函数
Offset::Offset(int32_t c_start, int32_t c_end, int32_t h_start, int32_t h_end,
               int32_t w_start, int32_t w_end)
    : w_start_(w_start), w_end_(w_end), w_stride_(1),
      h_start_(h_start), h_end_(h_end), h_stride_(1),
      c_start_(c_start), c_end_(c_end), c_stride_(1),
      n_start_(0), n_end_(-1), n_stride_(1) {

    validate_range(c_start, c_end, "C");
    validate_range(h_start, h_end, "H");
    validate_range(w_start, w_end, "W");
}

// N、C、H和W维度的完整构造函数
Offset::Offset(int32_t n_start, int32_t n_end, int32_t c_start, int32_t c_end,
               int32_t h_start, int32_t h_end, int32_t w_start, int32_t w_end)
    : w_start_(w_start), w_end_(w_end), w_stride_(1),
      h_start_(h_start), h_end_(h_end), h_stride_(1),
      c_start_(c_start), c_end_(c_end), c_stride_(1),
      n_start_(n_start), n_end_(n_end), n_stride_(1) {

    validate_range(n_start, n_end, "N");
    validate_range(c_start, c_end, "C");
    validate_range(h_start, h_end, "H");
    validate_range(w_start, w_end, "W");
}

// W维度步长设置器
void Offset::set_w_stride(int32_t w_stride) {
    validate_stride(w_stride, "W");
    w_stride_ = w_stride;
}

// H维度步长设置器
void Offset::set_h_stride(int32_t h_stride) {
    validate_stride(h_stride, "H");
    h_stride_ = h_stride;
}

// C维度步长设置器
void Offset::set_c_stride(int32_t c_stride) {
    validate_stride(c_stride, "C");
    c_stride_ = c_stride;
}

// N维度步长设置器
void Offset::set_n_stride(int32_t n_stride) {
    validate_stride(n_stride, "N");
    n_stride_ = n_stride;
}

} // namespace tr
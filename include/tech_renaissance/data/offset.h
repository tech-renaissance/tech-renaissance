/**
 * @file offset.h
 * @brief 张量偏移量类
 * @details 用于定义张量在各个维度上的起始、结束位置和步长
 * @version 1.00.00
 * @date 2025-11-03
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include <cstdint>
#include "tech_renaissance/utils/tr_exception.h"

namespace tr {

/**
 * @class Offset
 * @brief 张量偏移量类
 * @details 定义张量在NCHW四个维度上的起始、结束位置和步长
 */
class Offset {
private:
    // W维度参数
    int32_t w_start_;
    int32_t w_end_;
    int32_t w_stride_;

    // H维度参数
    int32_t h_start_;
    int32_t h_end_;
    int32_t h_stride_;

    // C维度参数
    int32_t c_start_;
    int32_t c_end_;
    int32_t c_stride_;

    // N维度参数
    int32_t n_start_;
    int32_t n_end_;
    int32_t n_stride_;

    // 验证函数
    void validate_range(int32_t start, int32_t end, const std::string& dim_name) const;
    void validate_stride(int32_t stride, const std::string& dim_name) const;

public:
    // 仅W维度的构造函数
    Offset(int32_t w_start, int32_t w_end);

    // H和W维度的构造函数
    Offset(int32_t h_start, int32_t h_end, int32_t w_start, int32_t w_end);

    // C、H和W维度的构造函数
    Offset(int32_t c_start, int32_t c_end, int32_t h_start, int32_t h_end, int32_t w_start, int32_t w_end);

    // N、C、H和W维度的完整构造函数
    Offset(int32_t n_start, int32_t n_end, int32_t c_start, int32_t c_end,
           int32_t h_start, int32_t h_end, int32_t w_start, int32_t w_end);

    // W维度访问器
    int32_t w_start() const { return w_start_; }
    int32_t w_end() const { return w_end_; }
    int32_t w_stride() const { return w_stride_; }
    void set_w_stride(int32_t w_stride);

    // H维度访问器
    int32_t h_start() const { return h_start_; }
    int32_t h_end() const { return h_end_; }
    int32_t h_stride() const { return h_stride_; }
    void set_h_stride(int32_t h_stride);

    // C维度访问器
    int32_t c_start() const { return c_start_; }
    int32_t c_end() const { return c_end_; }
    int32_t c_stride() const { return c_stride_; }
    void set_c_stride(int32_t c_stride);

    // N维度访问器
    int32_t n_start() const { return n_start_; }
    int32_t n_end() const { return n_end_; }
    int32_t n_stride() const { return n_stride_; }
    void set_n_stride(int32_t n_stride);
};

} // namespace tr
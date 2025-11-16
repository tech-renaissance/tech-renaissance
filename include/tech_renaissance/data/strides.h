/**
 * @file strides.h
 * @brief 步长类声明
 * @details 管理4维张量的步长信息，用于支持view等操作
 * @version 1.00.00
 * @date 2025-11-16
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include <cstdint>
#include <string>
#include <array>

namespace tr {

/**
 * @brief 形状类的前向声明
 */
class Shape;

/**
 * @brief 步长类
 * @details 管理4维张量（NCHW）的步长信息，支持view等零拷贝操作
 */
class Strides {
public:
    /**
     * @brief 默认构造函数
     * @details 创建全零步长
     */
    Strides() noexcept;

    /**
     * @brief 通过形状构造连续存储的步长
     * @param shape 张量形状
     * @details 根据连续存储布局计算步长（从右到左累乘）
     */
    explicit Strides(const Shape& shape);

    /**
     * @brief 通过四个值构造步长
     * @param n N维度步长
     * @param c C维度步长
     * @param h H维度步长
     * @param w W维度步长
     */
    Strides(int64_t n, int64_t c, int64_t h, int64_t w) noexcept;

    /**
     * @brief 拷贝构造函数
     */
    Strides(const Strides& other) = default;

    /**
     * @brief 移动构造函数
     */
    Strides(Strides&& other) noexcept = default;

    /**
     * @brief 拷贝赋值操作符
     */
    Strides& operator=(const Strides& other) = default;

    /**
     * @brief 移动赋值操作符
     */
    Strides& operator=(Strides&& other) noexcept = default;

    /**
     * @brief 析构函数
     */
    ~Strides() = default;

    // 访问方法

    /**
     * @brief 获取指定维度的步长
     * @param dim 维度索引（0=N, 1=C, 2=H, 3=W）
     * @return 步长值
     * @throws std::out_of_range 如果维度索引越界
     */
    int64_t stride(int32_t dim) const;

    /**
     * @brief 获取N维度步长
     */
    int64_t n() const noexcept { return strides_[0]; }

    /**
     * @brief 获取C维度步长
     */
    int64_t c() const noexcept { return strides_[1]; }

    /**
     * @brief 获取H维度步长
     */
    int64_t h() const noexcept { return strides_[2]; }

    /**
     * @brief 获取W维度步长
     */
    int64_t w() const noexcept { return strides_[3]; }

    /**
     * @brief 获取步长数组的原始指针
     */
    const int64_t* data() const noexcept { return strides_.data(); }

    /**
     * @brief 获取步长数组的原始指针（可修改）
     */
    int64_t* data() noexcept { return strides_.data(); }

    // 比较操作符

    /**
     * @brief 相等比较操作符
     */
    bool operator==(const Strides& other) const noexcept;

    /**
     * @brief 不等比较操作符
     */
    bool operator!=(const Strides& other) const noexcept;

    // 实用方法

    /**
     * @brief 计算线性偏移量
     * @param n N维度索引
     * @param c C维度索引
     * @param h H维度索引
     * @param w W维度索引
     * @return 线性内存偏移量
     */
    int64_t get_offset(int64_t n, int64_t c, int64_t h, int64_t w) const noexcept;

    /**
     * @brief 转换为字符串表示
     * @return 字符串格式的步长信息
     */
    std::string to_string() const;

    /**
     * @brief 检查是否为连续存储
     * @param shape 对应的形状
     * @return 如果是连续存储则返回true
     */
    bool is_contiguous(const Shape& shape) const;

private:
    std::array<int64_t, 4> strides_;  ///< 步长数组 [N, C, H, W]

    /**
     * @brief 验证维度索引的有效性
     * @param dim 维度索引
     * @throws std::out_of_range 如果维度索引越界
     */
    void validate_dim(int32_t dim) const;
};

} // namespace tr
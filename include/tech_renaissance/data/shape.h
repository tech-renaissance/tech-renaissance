/**
 * @file shape.h
 * @brief 形状类声明
 * @details 封装张量的形状信息(N,C,H,W)，提供维度查询和验证功能，支持0维到4维张量
 * @version 1.00.00
 * @date 2025-10-25
 * @author 技术觉醒团队
 * @note 依赖项:
 * @note 所属系列: data
 */

#pragma once

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

namespace tr {

/**
 * @class Shape
 * @brief 张量形状类
 * @details 维护4个非负整数值表示NCHW维度，采用右对齐规则，支持0维到4维张量
 *
 * 设计要点：
 * - 使用int32_t类型以便检测负数输入
 * - 右对齐规则：非零值必须在零值的右边
 * - 不变性：一旦创建，内容不可修改
 * - 标量(0,0,0,0)：ndim=0, numel=1, 打印为()
 */
class Shape {
public:
    /**
     * @brief 默认构造函数 - 创建标量
     * @details 创建0维标量张量，内部存储为(0,0,0,0)
     */
    Shape();

    /**
     * @brief 单参数构造函数 - 创建1维张量
     * @param dim0 第一个维度尺寸（必须为正数）
     * @throws std::invalid_argument 如果参数为0或负数
     */
    explicit Shape(int32_t dim0);

    /**
     * @brief 双参数构造函数 - 创建2维张量（矩阵）
     * @param dim0 第一个维度尺寸（必须为正数），对应行数M
     * @param dim1 第二个维度尺寸（必须为正数），对应列数N
     * @details 内部存储为(0,0,dim0,dim1)，与PyTorch的张量语义兼容
     * @throws std::invalid_argument 如果任何参数为0或负数
     */
    Shape(int32_t dim0, int32_t dim1);

    /**
     * @brief 三参数构造函数 - 创建3维张量
     * @param dim0 第一个维度尺寸（必须为正数）
     * @param dim1 第二个维度尺寸（必须为正数）
     * @param dim2 第三个维度尺寸（必须为正数）
     * @throws std::invalid_argument 如果任何参数为0或负数
     */
    Shape(int32_t dim0, int32_t dim1, int32_t dim2);

    /**
     * @brief 四参数构造函数 - 创建4维张量
     * @param n 批次大小N
     * @param c 通道数C
     * @param h 高度H
     * @param w 宽度W
     * @throws std::invalid_argument 如果任何参数为负数
     */
    Shape(int32_t n, int32_t c, int32_t h, int32_t w);

    /**
     * @brief 初始化列表构造函数 - 支持1到4个参数
     * @param dims 维度列表
     * @throws std::invalid_argument 如果参数数量超过4个或包含0/负数
     */
    explicit Shape(std::initializer_list<int32_t> dims);

    /**
     * @brief 拷贝构造函数
     * @param other 另一个Shape对象
     */
    Shape(const Shape& other) = default;

    /**
     * @brief 赋值运算符
     * @param other 另一个Shape对象
     * @return 当前对象的引用
     */
    Shape& operator=(const Shape& other) = default;

    /**
     * @brief 析构函数
     */
    ~Shape() = default;

    /**
     * @brief 获取维度数
     * @return 非零维度的数量（0到4）
     */
    int32_t ndim() const;

    /**
     * @brief 获取元素总数
     * @return 所有非零维度的乘积（标量返回1）
     */
    int64_t numel() const;

    /**
     * @brief 获取指定维度的尺寸
     * @param dim 维度索引（0-based，相对于实际维度）
     * @return 维度尺寸
     * @throws std::out_of_range 如果索引超出范围
     */
    int32_t dim(int32_t dim) const;

    /**
     * @brief 获取N维度尺寸（批次大小）
     * @return N维度尺寸，如果为0维张量则返回1
     */
    int32_t n() const { return dims_[0]; }

    /**
     * @brief 获取C维度尺寸（通道数）
     * @return C维度尺寸
     */
    int32_t c() const { return dims_[1]; }

    /**
     * @brief 获取H维度尺寸（高度）
     * @return H维度尺寸
     */
    int32_t h() const { return dims_[2]; }

    /**
     * @brief 获取W维度尺寸（宽度）
     * @return W维度尺寸
     */
    int32_t w() const { return dims_[3]; }

    // ===== 维度别名方法（便于用户理解） =====

    /**
     * @brief 获取批次数量（N维度的别名）
     * @return 批次数量
     */
    int32_t number() const { return n(); }

    /**
     * @brief 获取通道数量（C维度的别名）
     * @return 通道数量
     */
    int32_t channel() const { return c(); }

    /**
     * @brief 获取高度（H维度的别名，用于二维矩阵的行数）
     * @return 高度
     */
    int32_t height() const { return h(); }

    /**
     * @brief 获取宽度（W维度的别名，用于二维矩阵的列数）
     * @return 宽度
     */
    int32_t width() const { return w(); }

    /**
     * @brief 相等性比较运算符
     * @param other 另一个Shape对象
     * @return 如果所有维度都相等则返回true
     */
    bool operator==(const Shape& other) const;

    /**
     * @brief 不等性比较运算符
     * @param other 另一个Shape对象
     * @return 如果任何维度不相等则返回true
     */
    bool operator!=(const Shape& other) const;

    /**
     * @brief 获取字符串表示
     * @return 形状的字符串表示，如"(4,3,2)"或"()"
     */
    std::string to_string() const;

    /**
     * @brief 检查是否为标量
     * @return 如果是标量（所有维度为0）则返回true
     */
    bool is_scalar() const;

    /**
     * @brief 检查是否为有效的矩阵乘法形状
     * @details 当前形状作为左矩阵[m,k]，检查能否与右矩阵[k,n]相乘
     * @param other 右矩阵形状[k,n]
     * @return 如果可以相乘则返回true
     */
    bool is_matmul_compatible(const Shape& other) const;

    /**
     * @brief 检查是否可以广播到目标形状
     * @details 检查当前形状是否可以通过广播规则扩展到目标形状
     * @param target 目标形状
     * @return 如果可以广播则返回true
     */
    bool is_broadcastable_to(const Shape& target) const;

private:
    int32_t dims_[4];  ///< 维度尺寸数组 [N, C, H, W]
    int32_t ndim_;     ///< 维度数（非零维度的数量）
    int64_t numel_;    ///< 元素总数（非零维度的乘积）

    /**
     * @brief 计算维度数和元素数
     * @details 在构造函数中调用，根据dims_计算ndim_和numel_
     */
    void compute_metadata();

    /**
     * @brief 验证维度参数
     * @param dims 维度数组
     * @param count 维度数量
     * @throws std::invalid_argument 如果参数无效
     */
    void validate_dims(const int32_t* dims, int32_t count);

    /**
     * @brief 获取第一个非零维度的索引
     * @return 第一个非零维度的索引（0到3）
     */
    int32_t first_nonzero_dim() const;

    /**
     * @brief 检查右对齐规则
     * @details 确保非零值都在零值的右边
     * @return 如果符合右对齐规则则返回true
     */
    bool is_right_aligned() const;
};

/**
 * @brief 输出流运算符重载
 * @param os 输出流
 * @param shape Shape对象
 * @return 输出流的引用
 */
std::ostream& operator<<(std::ostream& os, const Shape& shape);

} // namespace tr
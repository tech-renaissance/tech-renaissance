/**
 * @file backend.h
 * @brief 后端抽象基类
 * @details 定义所有后端必须实现的接口（已优化，避免循环依赖）
 * @version 1.01.00
 * @date 2025-10-26
 * @author 技术觉醒团队
 * @note 依赖项: device.h, dtype.h
 * @note 所属系列: backend
 */

#pragma once

#include "tech_renaissance/data/device.h"
#include "tech_renaissance/data/dtype.h"
#include <memory>
#include <cstddef>

namespace tr {

// 前向声明
class Tensor;
class Shape;

// ########## 关键修改 ##########
// 不包含 "tech_renaissance/data/tensor.h"
// 仅使用前向声明，彻底解耦
class Tensor;

/**
 * @brief 后端抽象基类
 */
class Backend {
public:
    virtual ~Backend() = default;

    // ===== 内存管理接口 =====

    /**
     * @brief 分配内存
     * @param size 字节数
     * @return 数据持有者（智能指针）
     * @note 删除器会自动调用deallocate
     */
    virtual std::shared_ptr<void> allocate(size_t size) = 0;

    /**
     * @brief 释放内存
     * @param ptr 内存指针
     */
    virtual void deallocate(void* ptr) = 0;

    /**
     * @brief 从智能指针获取原始指针
     * @param holder 数据持有者
     * @return 原始指针
     */
    virtual void* get_data_ptr(const std::shared_ptr<void>& holder) = 0;

    /**
     * @brief 内存拷贝
     * @param dst 目标指针
     * @param src 源指针
     * @param size 字节数
     * @param dst_device 目标设备
     * @param src_device 源设备
     */
    virtual void copy_data(void* dst, const void* src, size_t size,
                          const Device& dst_device, const Device& src_device) const = 0;

    // ===== 填充操作 =====

    /**
     * @brief 填充张量（FP32）
     * @param dst 目标张量
     * @param value 填充值
     */
    virtual void fill(Tensor& dst, float value) = 0;

    /**
     * @brief 填充张量（INT8）
     * @param dst 目标张量
     * @param value 填充值
     */
    virtual void fill(Tensor& dst, int8_t value) = 0;

    /**
     * @brief 填充张量（INT32）
     * @param dst 目标张量
     * @param value 填充值
     */
    virtual void fill(Tensor& dst, int32_t value) = 0;

    /**
     * @brief 填充张量（FP32，别名方法）
     * @param dst 目标张量
     * @param value 填充值
     */
    virtual void fill_fp32(Tensor& dst, float value) = 0;

    /**
     * @brief 填充张量（INT8，别名方法）
     * @param dst 目标张量
     * @param value 填充值
     */
    virtual void fill_int8(Tensor& dst, int8_t value) = 0;

    /**
     * @brief 填充张量（INT32，别名方法）
     * @param dst 目标张量
     * @param value 填充值
     */
    virtual void fill_int32(Tensor& dst, int32_t value) = 0;

    // ===== 基本运算 =====

    /**
     * @brief 张量加法
     * @param result 结果张量
     * @param a 输入张量A
     * @param b 输入张量B
     */
    virtual void add(Tensor& result, const Tensor& a, const Tensor& b) = 0;

    /**
     * @brief 张量乘法
     * @param result 结果张量
     * @param a 输入张量A
     * @param b 输入张量B
     */
    virtual void mul(Tensor& result, const Tensor& a, const Tensor& b) = 0;

    /**
     * @brief 矩阵乘法 C(M,N) = A(M,K) * B(K,N)
     * @param a 输入张量A
     * @param b 输入张量B
     * @return 结果张量
     */
    virtual Tensor mm(const Tensor& a, const Tensor& b) = 0;

    /**
     * @brief 矩阵乘法 C(M,N) = A(M,K) * B(K,N) (指定输出张量)
     * @param a 输入张量A
     * @param b 输入张量B
     * @param result 结果张量
     */
    virtual void mm_into(const Tensor& a, const Tensor& b, Tensor& result) = 0;

    // ===== 设备转换方法 =====

    /**
     * @brief 通用设备转换方法
     * @param tensor 输入张量
     * @param device 目标设备
     * @return 转换后的张量
     */
    virtual Tensor to(const Tensor& tensor, const Device& device) const = 0;

    /**
     * @brief 将张量转换到CPU设备
     * @param tensor 输入张量
     * @return CPU设备上的张量
     */
    virtual Tensor to_cpu(const Tensor& tensor) const = 0;

    /**
     * @brief 从CPU设备转换张量到当前后端设备
     * @param tensor CPU设备上的张量
     * @return 当前后端设备上的张量
     */
    virtual Tensor from_cpu(const Tensor& tensor) const = 0;

    // ===== 辅助方法 =====

    /**
     * @brief 获取后端名称
     * @return 后端名称
     */
    virtual std::string name() const = 0;

    /**
     * @brief 获取后端设备
     * @return 设备标识
     */
    virtual Device device() const = 0;

    // ===== 张量创建操作 =====

    /**
     * @brief 创建空张量
     * @param shape 张量形状
     * @param dtype 数据类型
     * @return 空张量
     */
    virtual Tensor empty(const Shape& shape, DType dtype) = 0;

    /**
     * @brief 创建零张量
     * @param shape 张量形状
     * @param dtype 数据类型
     * @return 零张量
     */
    virtual Tensor zeros(const Shape& shape, DType dtype) = 0;

    /**
     * @brief 创建一张量
     * @param shape 张量形状
     * @param dtype 数据类型
     * @return 一张量
     */
    virtual Tensor ones(const Shape& shape, DType dtype) = 0;

    /**
     * @brief 创建正态分布随机张量
     * @param shape 张量形状
     * @param seed 随机种子
     * @return 随机张量
     */
    virtual Tensor randn(const Shape& shape, unsigned int seed = 0) = 0;

    // ===== 张量复制操作 =====

    /**
     * @brief 复制张量（同后端内操作）
     * @param tensor 源张量，必须属于当前后端
     * @return 复制后的新张量，属于同一后端
     * @throws TRException 当张量不属于当前后端时抛出
     * @note 深拷贝操作，生成独立的数据副本
     */
    virtual Tensor copy(const Tensor& tensor) const = 0;

    /**
     * @brief 复制张量到指定目标（支持跨设备，但语义明确）
     * @param src 源张量
     * @param dst 目标张量，用于接收复制结果
     * @throws TRException 当操作不被支持时抛出
     * @note 深拷贝操作，将src完全复制到dst的内存中
     * @note CPU后端：仅支持CPU↔CPU操作
     * @note 其他后端：支持本后端↔CPU操作
     */
    virtual void copy_into(const Tensor& src, Tensor& dst) const = 0;

    // ===== 数据访问 =====

    /**
     * @brief 获取标量张量的数据（float版本）
     * @param tensor 标量张量
     * @return 标量值
     * @throws std::runtime_error 如果不是标量张量或读取失败
     */
    virtual float get_scalar_float(const Tensor& tensor) = 0;

    /**
     * @brief 获取标量张量的数据（int32_t版本）
     * @param tensor 标量张量
     * @return 标量值
     * @throws std::runtime_error 如果不是标量张量或读取失败
     */
    virtual int32_t get_scalar_int32(const Tensor& tensor) = 0;

    /**
     * @brief 获取标量张量的数据（int8_t版本）
     * @param tensor 标量张量
     * @return 标量值
     * @throws std::runtime_error 如果不是标量张量或读取失败
     */
    virtual int8_t get_scalar_int8(const Tensor& tensor) = 0;

    /**
     * @brief 获取张量占用的内存空间字节数
     * @param tensor 输入张量
     * @return 张量占用的内存字节数
     * @throws std::runtime_error 如果张量为空或无效
     */
    virtual int64_t get_memory_size(const Tensor& tensor) = 0;

protected:
    Backend() = default;
};

} // namespace tr
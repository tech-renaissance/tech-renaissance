/**
 * @file storage.h
 * @brief 存储类声明
 * @details 封装一块原始的、连续的内存（如 CPU RAM 或 GPU VRAM），持有数据指针、大小、设备信息和智能指针管理生命周期
 * @version 1.01.01
 * @date 2025-10-24
 * @author 技术觉醒团队
 * @note 依赖项: device.h
 * @note 所属系列: data
 */

#pragma once

#include <memory>
#include "tech_renaissance/data/device.h"

namespace tr {

/**
 * @class Storage
 * @brief 存储类
 * @details 封装一块原始的、连续的内存，管理生命周期
 */
class Storage {
public:
    /**
     * @brief 构造函数
     * @param size 内存大小（字节）
     * @param device 设备类型
     */
    Storage(size_t size, const Device& device = tr::CPU);

    /**
     * @brief 析构函数
     */
    ~Storage() = default;

    /**
     * @brief 获取数据指针
     * @return 数据指针
     */
    void* data_ptr();

    /**
     * @brief 获取数据指针（const版本）
     * @return 数据指针
     */
    const void* data_ptr() const;

    /**
     * @brief 获取内存大小
     * @return 内存大小（字节）
     */
    size_t size() const;

    /**
     * @brief 获取设备
     * @return 设备
     */
    const Device& device() const;

    /**
     * @brief 检查是否为空
     * @return 是否为空
     */
    bool is_empty() const;

    /**
     * @brief 设置数据指针和智能指针
     * @param ptr 数据指针
     * @param holder 智能指针管理器
     */
    void set_data_ptr(void* ptr, std::shared_ptr<void> holder);

    /**
     * @brief 获取内存容量（字节）
     * @return 内存容量
     */
    size_t capacity_bytes() const { return size_; }

    /**
     * @brief 获取内存使用大小（字节）
     * @return 内存使用大小
     */
    size_t size_bytes() const { return size_; }

    /**
     * @brief 获取原始指针（仅供Backend内部使用）
     * @return 原始数据指针
     */
    void* raw_ptr() const { return data_ptr_; }

    /**
     * @brief 获取智能指针holder
     * @return 智能指针holder
     */
    std::shared_ptr<void> holder() const { return holder_; }

private:
    void* data_ptr_;                           ///< 原始数据指针
    size_t size_;                              ///< 内存大小（字节）
    Device device_;                           ///< 设备
    std::shared_ptr<void> holder_;            ///< 智能指针，管理内存生命周期
};

} // namespace tr
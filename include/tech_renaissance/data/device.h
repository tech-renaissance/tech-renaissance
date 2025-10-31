/**
 * @file device.h
 * @brief 设备描述类声明
 * @details 去掉DeviceType::CPU，只保留Device实体与静态全局对象。用于标识Tensor和Storage所处的位置，支持设备间数据传输
 * @version 1.01.01
 * @date 2025-10-25
 * @author 技术觉醒团队
 * @note 依赖项: 无
 * @note 所属系列: data
 */

#pragma once

#include <string>
#include <array>

namespace tr {

/**
 * @struct Device
 * @brief 设备实体类
 * @details 去掉DeviceType::CPU，只保留Device实体与静态全局对象
 */
struct Device {
    std::string name;  ///< 设备名称
    int index;         ///< 设备索引

    /**
     * @brief 构造函数
     * @param device_name 设备名称
     * @param device_index 设备索引
     */
    Device(const std::string& device_name, int device_index = 0);

    /**
     * @brief 获取设备字符串表示
     * @return 设备字符串
     */
    std::string str() const;

    /**
     * @brief 获取设备完整描述字符串
     * @return 设备描述字符串
     */
    std::string to_string() const;

    /**
     * @brief 检查设备是否相等
     * @param other 其他设备
     * @return 是否相等
     */
    bool operator==(const Device& other) const;

    /**
     * @brief 检查设备是否不等
     * @param other 其他设备
     * @return 是否不等
     */
    bool operator!=(const Device& other) const;

    /**
     * @brief 检查是否为CPU设备
     * @return 是否为CPU设备
     */
    bool is_cpu() const;

    /**
     * @brief 检查是否为CUDA设备
     * @return 是否为CUDA设备
     */
    bool is_cuda() const;
};

// 静态全局设备对象
inline const Device CPU{"CPU", -1};
inline std::array<Device, 8> CUDA = {
    Device{"CUDA", 0}, Device{"CUDA", 1}, Device{"CUDA", 2}, Device{"CUDA", 3},
    Device{"CUDA", 4}, Device{"CUDA", 5}, Device{"CUDA", 6}, Device{"CUDA", 7}
};

// Device的hash支持，用于unordered_map
struct DeviceHash {
    std::size_t operator()(const Device& device) const noexcept {
        // 使用字符串hash和索引的简单组合
        return std::hash<std::string>{}(device.name) ^
               std::hash<int>{}(device.index);
    }
};

} // namespace tr

// 为std命名空间添加Device的hash特化
namespace std {
    template<>
    struct hash<tr::Device> {
        std::size_t operator()(const tr::Device& device) const noexcept {
            return tr::DeviceHash{}(device);
        }
    };
}
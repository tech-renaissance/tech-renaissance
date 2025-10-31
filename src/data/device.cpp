/**
 * @file device.cpp
 * @brief 设备描述类实现
 * @details 去掉DeviceType::CPU，只保留Device实体与静态全局对象。用于标识Tensor和Storage所处的位置，支持设备间数据传输
 * @version 1.01.01
 * @date 2025-10-25
 * @author 技术觉醒团队
 * @note 依赖项: device.h
 * @note 所属系列: data
 */

#include "tech_renaissance/data/device.h"
#include "tech_renaissance/utils/tr_exception.h"
#include <sstream>

namespace tr {

Device::Device(const std::string& device_name, int device_index)
    : name(device_name), index(device_index) {

    // 设备有效性验证
    if (name != "CPU" && name != "CUDA") {
        throw TRException("[Device::Device] Unsupported device type: '" + name +
                                   "'. Supported types are: 'CPU', 'CUDA'");
    }

    // CPU设备索引必须为-1，CUDA设备索引必须非负
    if (name == "CPU" && index != -1) {
        throw TRException("[Device::Device] CPU device index must be -1, got: " + std::to_string(index));
    }
    if (name == "CUDA" && (index < 0 || index >= 8)) {
        throw TRException("[Device::Device] CUDA device index must be between 0 and 7, got: " + std::to_string(index));
    }
}

std::string Device::str() const {
    if (index >= 0) {
        return name + ":" + std::to_string(index);
    }
    return name;
}

std::string Device::to_string() const {
    return str();
}

bool Device::operator==(const Device& other) const {
    return name == other.name && index == other.index;
}

bool Device::operator!=(const Device& other) const {
    return !(*this == other);
}

bool Device::is_cpu() const {
    return name == "CPU";
}

bool Device::is_cuda() const {
    return name == "CUDA";
}

} // namespace tr
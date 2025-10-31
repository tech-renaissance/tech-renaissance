/**
 * @file backend_manager.h
 * @brief 后端管理器类
 * @details 单例模式管理所有后端实例，提供统一访问接口
 * @version 1.01.00
 * @date 2025-10-26
 * @author 技术觉醒团队
 * @note 依赖项: device.h
 * @note 所属系列: backend
 */

#pragma once

#include "tech_renaissance/data/device.h"
#include <memory>
#include <unordered_map>
#include <mutex>
#include <string>

namespace tr {

class Backend; // 前向声明，避免循环包含

/**
 * @brief 后端管理器（Meyers单例）
 */
class BackendManager {
public:
    /**
     * @brief 获取单例实例
     * @return 单例引用
     */
    static BackendManager& instance();

    /**
     * @brief 获取指定设备的后端
     * @param device 设备标识
     * @return 后端智能指针
     * @throws std::runtime_error 如果后端未注册
     */
    std::shared_ptr<Backend> get_backend(const Device& device);

    /**
     * @brief 获取指定设备的后端（静态便利方法）
     * @param device 设备标识
     * @return 后端智能指针
     * @throws std::runtime_error 如果后端未注册
     */
    static std::shared_ptr<Backend> get_backend_static(const Device& device);

    /**
     * @brief 注册后端（内部使用）
     * @param device 设备标识
     * @param backend 后端实例
     */
    void register_backend(const Device& device, std::shared_ptr<Backend> backend);

    /**
     * @brief 检查后端是否已注册
     * @param device 设备标识
     * @return 如果已注册返回true
     */
    bool is_registered(const Device& device) const;

    // API优化：静态便利方法
    /**
     * @brief 获取CUDA后端（便利方法）
     * @param device_id CUDA设备ID，默认为0
     * @return CUDA后端智能指针
     * @throws std::runtime_error 如果CUDA后端未注册
     */
    static std::shared_ptr<class CudaBackend> get_cuda_backend(int device_id = 0);

    /**
     * @brief 获取CPU后端（便利方法）
     * @return CPU后端智能指针
     * @throws std::runtime_error 如果CPU后端未注册
     */
    static std::shared_ptr<class CpuBackend> get_cpu_backend();

    
    // 禁止拷贝和移动
    BackendManager(const BackendManager&) = delete;
    BackendManager& operator=(const BackendManager&) = delete;
    BackendManager(BackendManager&&) = delete;
    BackendManager& operator=(BackendManager&&) = delete;

private:
    BackendManager(); // 构造函数中初始化
    ~BackendManager() = default;

    /**
     * @brief 初始化所有后端（自动调用一次）
     */
    void init_backends();

    std::unordered_map<std::string, std::shared_ptr<Backend>> backends_;
    mutable std::mutex mutex_;

    // 辅助函数
    std::string device_key(const Device& device) const;
};

} // namespace tr
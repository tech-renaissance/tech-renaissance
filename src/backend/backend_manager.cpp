/**
 * @file backend_manager.cpp
 * @brief 后端管理器类实现
 * @details 实现单例模式管理所有后端实例，提供统一访问接口
 * @version 1.01.00
 * @date 2025-10-26
 * @author 技术觉醒团队
 * @note 依赖项: cpu_backend.h, cuda_backend.h
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/utils/logger.h"

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef TR_USE_CUDA
#include "tech_renaissance/backend/cuda/cuda_backend.h"
#endif

namespace tr {

BackendManager::BackendManager() {
    init_backends();
}

BackendManager& BackendManager::instance() {
    static BackendManager instance; // C++11保证这里的初始化是线程安全的
    return instance;
}

void BackendManager::init_backends() {
    Logger::get_instance().info("Initializing backend manager...");

    // 1. CPU后端始终注册
    register_backend(tr::CPU, std::make_shared<CpuBackend>());
    Logger::get_instance().info("CPU backend registered successfully");

    // 2. CUDA后端按编译选项注册
#ifdef TR_USE_CUDA
    try {
        // 首先检查有多少个CUDA设备可用
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);

        if (error != cudaSuccess) {
            Logger::get_instance().warn("CUDA runtime not available: " + std::string(cudaGetErrorString(error)));
        } else if (device_count == 0) {
            Logger::get_instance().info("No CUDA devices found - CUDA backend disabled");
        } else {
            Logger::get_instance().info("Found " + std::to_string(device_count) + " CUDA devices");
            // 只注册实际存在的设备
            for (int i = 0; i < device_count && i < 8; ++i) {
                try {
                    register_backend(tr::CUDA[i], std::make_shared<CudaBackend>(i));
                    Logger::get_instance().info("CUDA backend registered for device " + std::to_string(i));
                } catch (const std::exception& e) {
                    Logger::get_instance().warn("Failed to register CUDA device " + std::to_string(i) + ": " + std::string(e.what()));
                    // 继续尝试其他设备
                }
            }
        }
    } catch (const std::exception& e) {
        Logger::get_instance().warn("CUDA backend initialization failed: " + std::string(e.what()));
    }
#else
    Logger::get_instance().info("CUDA backend not compiled (TR_USE_CUDA=OFF)");
#endif

    Logger::get_instance().info("Backend manager initialization completed\n\n");
}

std::shared_ptr<Backend> BackendManager::get_backend(const Device& device) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key = device_key(device);
    auto it = backends_.find(key);

    if (it == backends_.end()) {
        throw TRException("Backend not found for device: " + device.to_string());
    }

    return it->second;
}

void BackendManager::register_backend(const Device& device,
                                      std::shared_ptr<Backend> backend) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = device_key(device);

    // 允许覆盖注册（支持用户自定义后端）
    backends_[key] = std::move(backend);
}

bool BackendManager::is_registered(const Device& device) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return backends_.find(device_key(device)) != backends_.end();
}

std::string BackendManager::device_key(const Device& device) const {
    return device.to_string(); // 例如 "CPU" 或 "CUDA:0"
}

// API优化：静态便利方法实现
#ifdef TR_USE_CUDA
std::shared_ptr<CudaBackend> BackendManager::get_cuda_backend(int device_id) {
    return std::dynamic_pointer_cast<CudaBackend>(
        instance().get_backend(tr::CUDA[device_id])
    );
}
#endif

std::shared_ptr<CpuBackend> BackendManager::get_cpu_backend() {
    return std::dynamic_pointer_cast<CpuBackend>(
        instance().get_backend(tr::CPU)
    );
}

std::shared_ptr<Backend> BackendManager::get_backend_static(const Device& device) {
    return instance().get_backend(device);
}

} // namespace tr
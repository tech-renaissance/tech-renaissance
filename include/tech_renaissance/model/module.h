/**
 * @file module.h
 * @brief 模块基类
 * @details 所有神经网络层的抽象基类，定义计算和参数管理接口
 * @version 1.45.0
 * @date 2025-11-17
 * @author 技术觉醒团队
 * @note 依赖项: tensor.h, shape.h, backend.h, backend_manager.h
 * @note 所属系列: model
 */

#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/data/shape.h"
#include "tech_renaissance/backend/backend.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/backend/cpu/cpu_backend.h"

namespace tr {

class Module {
public:
    // === 构造与析构 ===
    explicit Module(const std::string& type_name)
        : name_(type_name), instance_name_(""), backend_(nullptr), training_(true) {}

    virtual ~Module() = default;

    // === 核心计算接口 ===

    /**
     * @brief 前向传播（返回型，便于用户使用）
     * @param input 输入张量
     * @return 输出张量
     * @note 内部调用forward_into，避免重复实现
     */
    virtual Tensor forward(const Tensor& input) {
        Tensor output = create_output_tensor(input);
        forward_into(input, output);
        return output;
    }

    /**
     * @brief 前向传播（into型，性能关键路径）
     * @param input 输入张量
     * @param output 预分配的输出张量
     * @note 子类必须实现此方法
     */
    virtual void forward_into(const Tensor& input, Tensor& output) = 0;

    /**
     * @brief 反向传播（返回型）
     * @param grad_output 上层梯度
     * @return 当前层输入的梯度
     */
    virtual Tensor backward(const Tensor& grad_output) {
        if (!cached_input_.storage_allocated()) {
            throw TRException("[Module::backward] No cached input. Did you call forward in training mode?");
        }
        Tensor grad_input = create_input_gradient_tensor();
        backward_into(grad_output, grad_input);
        return grad_input;
    }

    /**
     * @brief 反向传播（into型）
     * @param grad_output 上层梯度
     * @param grad_input 预分配的输入梯度张量
     */
    virtual void backward_into(const Tensor& grad_output, Tensor& grad_input) = 0;

    // === 形状推断（内存分析用） ===

    /**
     * @brief 推断输出形状
     * @param input_shape 输入形状
     * @return 输出形状
     * @note 子类必须实现
     */
    virtual Shape infer_output_shape(const Shape& input_shape) const = 0;

    // === 参数管理 ===

    void register_parameter(const std::string& key, Tensor tensor) {
        parameters_[key] = std::move(tensor);
    }

    void register_buffer(const std::string& key, Tensor tensor) {
        buffers_[key] = std::move(tensor);
    }

    bool has_parameter(const std::string& key) const {
        return parameters_.count(key) > 0;
    }

    Tensor& get_parameter(const std::string& key) {
        auto it = parameters_.find(key);
        if (it == parameters_.end()) {
            throw TRException("[Module] Parameter '" + key + "' not found in " + instance_name());
        }
        return it->second;
    }

    const Tensor& get_parameter(const std::string& key) const {
        auto it = parameters_.find(key);
        if (it == parameters_.end()) {
            throw TRException("[Module] Parameter '" + key + "' not found in " + instance_name());
        }
        return it->second;
    }

    const std::unordered_map<std::string, Tensor>& parameters() const {
        return parameters_;
    }

    std::unordered_map<std::string, Tensor>& parameters() {
        return parameters_;
    }

    // === 内存占用分析 ===

    size_t parameter_memory() const {
        size_t total = 0;
        for (const auto& [key, param] : parameters_) {
            total += param.memory_size();
        }
        for (const auto& [key, buffer] : buffers_) {
            total += buffer.memory_size();
        }
        return total;
    }

    // === 后端管理 ===

    virtual void set_backend(Backend* backend) {
        backend_ = backend;
    }

    Backend* get_backend() const {
        if (!backend_) {
            throw TRException("[Module] Backend not set for " + instance_name());
        }
        return backend_;
    }

    // === 设备转移 ===

    virtual void to(const Device& device) {
        auto cpu_backend = BackendManager::get_cpu_backend().get();
        backend_ = reinterpret_cast<Backend*>(cpu_backend);

        // 转移所有参数
        for (auto& [key, param] : parameters_) {
            if (param.device() != device) {
                Tensor new_param = backend_->empty(param.shape(), param.dtype());
                backend_->copy_into(param, new_param);
                param = std::move(new_param);
            }
        }

        // 转移所有缓冲区
        for (auto& [key, buffer] : buffers_) {
            if (buffer.device() != device) {
                Tensor new_buffer = backend_->empty(buffer.shape(), buffer.dtype());
                backend_->copy_into(buffer, new_buffer);
                buffer = std::move(new_buffer);
            }
        }
    }

    // === 模式切换 ===

    virtual void train() {
        training_ = true;
    }

    virtual void eval() {
        training_ = false;
        clear_cache();  // 推理模式不需要缓存
    }

    bool is_training() const {
        return training_;
    }

    // === 梯度管理 ===

    void zero_grad() {
        for (auto& [key, param] : parameters_) {
            if (param.grad().storage_allocated()) {
                backend_->fill(param.grad(), 0.0f);
            }
        }
    }

    // === 命名管理 ===

    const std::string& name() const {
        return name_;
    }

    const std::string& instance_name() const {
        return instance_name_.empty() ? name_ : instance_name_;
    }

    void set_instance_name(const std::string& name) {
        instance_name_ = name;
    }

    // === 序列化 ===

    virtual void save(std::ostream& os) const {
        // 保存类型名
        save_string(os, name_);

        // 保存参数数量
        uint32_t num_params = parameters_.size();
        os.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

        // 保存每个参数
        for (const auto& [key, param] : parameters_) {
            save_string(os, key);
            save_tensor(os, param);
        }

        // 保存缓冲区数量
        uint32_t num_buffers = buffers_.size();
        os.write(reinterpret_cast<const char*>(&num_buffers), sizeof(num_buffers));

        // 保存每个缓冲区
        for (const auto& [key, buffer] : buffers_) {
            save_string(os, key);
            save_tensor(os, buffer);
        }
    }

    virtual void load(std::istream& is) {
        // 验证类型名
        std::string loaded_name = load_string(is);
        if (loaded_name != name_) {
            throw TRException("[Module::load] Type mismatch: expected " +
                             name_ + ", got " + loaded_name);
        }

        // 加载参数
        uint32_t num_params;
        is.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

        for (uint32_t i = 0; i < num_params; ++i) {
            std::string key = load_string(is);
            Tensor param = load_tensor(is);

            if (!has_parameter(key)) {
                throw TRException("[Module::load] Unknown parameter: " + key);
            }

            parameters_[key] = std::move(param);
        }

        // 加载缓冲区
        uint32_t num_buffers;
        is.read(reinterpret_cast<char*>(&num_buffers), sizeof(num_buffers));

        for (uint32_t i = 0; i < num_buffers; ++i) {
            std::string key = load_string(is);
            Tensor buffer = load_tensor(is);
            buffers_[key] = std::move(buffer);
        }
    }

protected:
    // === 输入缓存管理（子类使用） ===

    void cache_input(const Tensor& input) {
        if (training_) {
            cached_input_ = input;  // 浅拷贝（共享Storage）
        }
    }

    void clear_cache() {
        cached_input_ = Tensor();
    }

    // === 辅助方法（子类实现） ===

    virtual Tensor create_output_tensor(const Tensor& input) const {
        Shape output_shape = infer_output_shape(input.shape());
        return backend_->empty(output_shape, input.dtype());
    }

    virtual Tensor create_input_gradient_tensor() const {
        return backend_->empty(cached_input_.shape(), cached_input_.dtype());
    }

    // === 序列化辅助 ===

    static void save_string(std::ostream& os, const std::string& str) {
        uint32_t len = str.length();
        os.write(reinterpret_cast<const char*>(&len), sizeof(len));
        os.write(str.c_str(), len);
    }

    static std::string load_string(std::istream& is) {
        uint32_t len;
        is.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string str(len, '\0');
        is.read(&str[0], len);
        return str;
    }

    void save_tensor(std::ostream& os, const Tensor& tensor) const {
        // 简化实现：暂时保存基本元数据
        save_string(os, tensor.shape().to_string());
        save_string(os, std::to_string(static_cast<int>(tensor.dtype())));
        save_string(os, tensor.device().to_string());

        // TODO: 实现完整的数据序列化
    }

    Tensor load_tensor(std::istream& is) const {
        // 简化实现：暂时只加载元数据
        std::string shape_str = load_string(is);
        std::string dtype_str = load_string(is);
        std::string device_str = load_string(is);

        // TODO: 实现完整的数据反序列化
        return backend_->empty(Shape({1}), static_cast<DType>(std::stoi(dtype_str)));
    }

protected:
    // === 成员变量 ===
    std::string name_;                                    // 类型名（如"Linear"）
    std::string instance_name_;                          // 实例名（如"Linear1"）
    Backend* backend_;                                   // 后端指针
    std::unordered_map<std::string, Tensor> parameters_; // 可训练参数
    std::unordered_map<std::string, Tensor> buffers_;    // 非训练状态
    bool training_;                                      // 训练/推理标志
    Tensor cached_input_;                                // 输入缓存（训练用）
};

} // namespace tr
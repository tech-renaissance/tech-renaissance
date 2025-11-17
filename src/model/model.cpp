/**
 * @file model.cpp
 * @brief 模型类实现
 * @details Module的容器和编排器，管理Module的生命周期和执行顺序，提供预分配内存优化
 * @version 1.45.0
 * @date 2025-11-17
 * @author 技术觉醒团队
 * @note 依赖项: model.h
 * @note 所属系列: model
 */

#include "tech_renaissance/model/model.h"
#include "tech_renaissance/backend/backend_manager.h"
#include <fstream>
#include <sstream>

namespace tr {

// ===== InternalContext 实现 =====

void Model::InternalContext::allocate(const std::vector<std::shared_ptr<Module>>& modules,
                                     const Shape& input_shape,
                                     std::shared_ptr<Backend> backend) {
    if (allocated_) {
        return;
    }

    if (!backend) {
        throw TRException("[Model::InternalContext::allocate] Backend is null");
    }

    forward_cache_.clear();
    backward_cache_.clear();

    // 预分配所有层的输出空间
    Shape current_shape = input_shape;
    for (auto& module : modules) {
        if (!module) {
            throw TRException("[Model::InternalContext::allocate] Null module detected");
        }

        // 推断输出形状
        Shape output_shape = module->infer_output_shape(current_shape);

        // 预分配前向传播缓存（输出空间）
        forward_cache_.emplace_back(backend->empty(output_shape, DType::FP32));

        // 预分配反向传播缓存（输入梯度空间）
        backward_cache_.emplace_back(backend->empty(current_shape, DType::FP32));

        current_shape = output_shape;
    }

    // 预分配最终输出的反向传播缓存
    backward_cache_.emplace_back(backend->empty(current_shape, DType::FP32));

    allocated_ = true;
}

void Model::InternalContext::clear() {
    forward_cache_.clear();
    backward_cache_.clear();
    allocated_ = false;
}

Tensor& Model::InternalContext::get_forward_cache(size_t index) {
    if (!allocated_ || index >= forward_cache_.size()) {
        throw TRException("[Model::InternalContext::get_forward_cache] Cache not allocated or index out of range");
    }
    return forward_cache_[index];
}

Tensor& Model::InternalContext::get_backward_cache(size_t index) {
    if (!allocated_ || index >= backward_cache_.size()) {
        throw TRException("[Model::InternalContext::get_backward_cache] Cache not allocated or index out of range");
    }
    return backward_cache_[index];
}

// ===== Model 构造函数实现 =====

Model::Model(const std::string& name)
    : model_name_(name) {
}

Model::Model(const std::string& name,
             const std::vector<std::shared_ptr<Module>>& modules)
    : model_name_(name) {
    for (auto& module : modules) {
        add_module(module);
    }
}


// ===== 模块管理实现 =====

void Model::add_module(std::shared_ptr<Module> module) {
    if (!module) {
        throw TRException("[Model::add_module] Cannot add null module");
    }

    if (frozen_) {
        throw TRException("[Model::add_module] Cannot add module after model is frozen");
    }

    // 自动命名
    auto_name_module(module);

    // 添加到模块列表
    modules_.push_back(module);

    // 如果已设置后端，立即同步
    if (backend_) {
        module->set_backend(backend_);
    }
}

void Model::add_module(const std::string& custom_name,
                      std::shared_ptr<Module> module) {
    if (!module) {
        throw TRException("[Model::add_module] Cannot add null module");
    }

    if (frozen_) {
        throw TRException("[Model::add_module] Cannot add module after model is frozen");
    }

    // 手动命名
    module->set_instance_name(custom_name);

    // 添加到模块列表
    modules_.push_back(module);

    // 如果已设置后端，立即同步
    if (backend_) {
        module->set_backend(backend_);
    }
}

std::shared_ptr<Module> Model::get_module(size_t index) const {
    if (index >= modules_.size()) {
        throw TRException("[Model::get_module] Module index out of range: " + std::to_string(index));
    }
    return modules_[index];
}

// ===== 核心计算实现 =====

Tensor Model::forward(const Tensor& input) {
    if (modules_.empty()) {
        return input;  // 空模型直接返回输入
    }

    // 确保预分配缓存已初始化
    if (!ctx_.is_allocated()) {
        ctx_.allocate(modules_, input.shape(), backend_);
    }

    // 使用预分配缓存的性能路径
    Tensor output = backend_->empty(
        modules_.back()->infer_output_shape(input.shape()),
        DType::FP32
    );
    forward_into(input, output);
    return output;
}

void Model::forward_into(const Tensor& input, Tensor& output) {
    if (modules_.empty()) {
        // 空模型直接复制输入到输出
        backend_->copy_into(input, output);
        return;
    }

    // 确保预分配缓存已初始化
    if (!ctx_.is_allocated()) {
        ctx_.allocate(modules_, input.shape(), backend_);
    }

    // 第一层：输入到缓存0
    modules_[0]->forward_into(input, ctx_.get_forward_cache(0));

    // 中间层：缓存i-1 到 缓存i
    for (size_t i = 1; i < modules_.size(); ++i) {
        modules_[i]->forward_into(ctx_.get_forward_cache(i-1), ctx_.get_forward_cache(i));
    }

    // 最后一层：缓存到输出
    backend_->copy_into(ctx_.get_forward_cache(modules_.size() - 1), output);
}

Tensor Model::backward(const Tensor& grad_output) {
    if (modules_.empty()) {
        return grad_output;  // 空模型直接返回梯度
    }

    // 确保预分配缓存已初始化
    if (!ctx_.is_allocated()) {
        throw TRException("[Model::backward] InternalContext not allocated. Call forward first.");
    }

    // 使用预分配缓存的性能路径
    // 输入梯度存储在backward_cache[0]中
    backward_into(grad_output, ctx_.get_backward_cache(0));

    // 返回输入梯度的副本
    return ctx_.get_backward_cache(0);
}

void Model::backward_into(const Tensor& grad_output, Tensor& grad_input) {
    if (modules_.empty()) {
        // 空模型直接复制梯度输出到梯度输入
        backend_->copy_into(grad_output, grad_input);
        return;
    }

    // 确保预分配缓存已初始化
    if (!ctx_.is_allocated()) {
        throw TRException("[Model::backward_into] InternalContext not allocated. Call forward first.");
    }

    // 将最终输出的梯度复制到最后一个反向缓存
    size_t last_idx = modules_.size();
    backend_->copy_into(grad_output, ctx_.get_backward_cache(last_idx));

    // 逐层反向传播（逆序，全部使用into型）
    for (int i = modules_.size() - 1; i >= 0; --i) {
        modules_[i]->backward_into(
            ctx_.get_backward_cache(i + 1),  // 当前层的输出梯度
            ctx_.get_backward_cache(i)      // 当前层的输入梯度
        );
    }

    // 将对模型输入的梯度复制到输出
    backend_->copy_into(ctx_.get_backward_cache(0), grad_input);
}

// ===== 设备管理实现 =====

void Model::to(const Device& device) {
    backend_ = BackendManager::instance().get_backend(device);

    // 递归设置所有模块的后端
    initialize_modules_backend();

    // 清空预分配缓存（需要重新分配）
    ctx_.clear();
}

Device Model::device() const {
    if (!backend_) {
        return tr::CPU;  // 默认设备
    }
    return backend_->device();
}

// ===== 后端管理实现 =====

void Model::set_backend(std::shared_ptr<Backend> backend) {
    if (!backend) {
        throw TRException("[Model::set_backend] Cannot set null backend");
    }

    backend_ = backend;

    // 递归设置所有模块的后端
    initialize_modules_backend();

    // 清空预分配缓存（需要重新分配）
    ctx_.clear();
}

// ===== 训练模式管理实现 =====

void Model::train() {
    training_ = true;
    for (auto& module : modules_) {
        module->train();
    }
}

void Model::eval() {
    training_ = false;
    for (auto& module : modules_) {
        module->eval();
    }
}

// ===== 参数管理实现 =====

std::unordered_map<std::string, Tensor> Model::parameters() const {
    std::unordered_map<std::string, Tensor> all_params;

    for (auto& module : modules_) {
        const auto& module_params = module->parameters();
        std::string prefix = module->instance_name() + ".";

        for (const auto& [key, param] : module_params) {
            all_params[prefix + key] = param;
        }
    }

    return all_params;
}

std::unordered_map<std::string, Tensor> Model::gradients() const {
    std::unordered_map<std::string, Tensor> all_grads;

    for (auto& module : modules_) {
        const auto& module_params = module->parameters();
        std::string prefix = module->instance_name() + ".";

        for (const auto& [key, param] : module_params) {
            if (param.has_grad()) {
                all_grads[prefix + key] = param.grad();
            }
        }
    }

    return all_grads;
}

void Model::zero_grad() {
    for (auto& module : modules_) {
        module->zero_grad();
    }
}

size_t Model::parameter_memory() const {
    size_t total_memory = 0;

    for (auto& module : modules_) {
        total_memory += module->parameter_memory();
    }

    return total_memory;
}

// ===== 自动命名机制实现 =====

void Model::auto_name_module(std::shared_ptr<Module> module) {
    if (!module) {
        return;
    }

    std::string type = module->name();
    int& counter = type_counters_[type];
    counter++;

    module->set_instance_name(type + std::to_string(counter));
}

// ===== 工厂方法实现 =====
// template函数在头文件中定义

// ===== 内存分析实现 =====

void Model::initialize(const Shape& input_shape) {
    if (!backend_) {
        throw TRException("[Model::initialize] Backend not set");
    }

    ctx_.allocate(modules_, input_shape, backend_);
}

std::string Model::analyze_memory() const {
    std::stringstream ss;
    ss << "=== Model Memory Analysis ===" << std::endl;
    ss << "Model name: " << model_name_ << std::endl;
    ss << "Number of modules: " << modules_.size() << std::endl;
    ss << "Parameter memory: " << parameter_memory() << " bytes" << std::endl;

    if (ctx_.is_allocated()) {
        ss << "Internal context: ALLOCATED" << std::endl;
        ss << "Forward cache size: " << ctx_.forward_cache_.size() << " tensors" << std::endl;
        ss << "Backward cache size: " << ctx_.backward_cache_.size() << " tensors" << std::endl;
    } else {
        ss << "Internal context: NOT ALLOCATED" << std::endl;
    }

    ss << "=============================" << std::endl;
    return ss.str();
}

// ===== 序列化实现 =====

void Model::save(const std::string& filename) const {
    // TODO: 实现模型序列化
    throw TRException("[Model::save] Serialization not implemented yet");
}

std::shared_ptr<Model> Model::load(const std::string& filename) {
    // TODO: 实现模型反序列化
    throw TRException("[Model::load] Deserialization not implemented yet");
}

// ===== 调试辅助实现 =====

void Model::print_model() const {
    std::cout << "=== Model: " << model_name_ << " ===" << std::endl;
    std::cout << "Modules: " << modules_.size() << std::endl;
    std::cout << "Training mode: " << (training_ ? "true" : "false") << std::endl;
    std::cout << "Backend: " << (backend_ ? backend_->name() : "none") << std::endl;

    for (size_t i = 0; i < modules_.size(); ++i) {
        std::cout << "  [" << i << "] " << modules_[i]->instance_name()
                  << " (" << modules_[i]->name() << ")" << std::endl;
    }
    std::cout << "Parameter memory: " << parameter_memory() << " bytes" << std::endl;
    std::cout << "=========================" << std::endl;
}

// ===== 私有辅助方法实现 =====

void Model::initialize_modules_backend() {
    for (auto& module : modules_) {
        if (backend_) {
            module->set_backend(backend_);
        }
    }
}

void Model::validate_model() const {
    for (size_t i = 0; i < modules_.size(); ++i) {
        if (!modules_[i]) {
            throw TRException("[Model::validate_model] Null module at index " + std::to_string(i));
        }

        if (!modules_[i]->get_backend()) {
            throw TRException("[Model::validate_model] Module " + modules_[i]->instance_name() +
                             " has no backend set");
        }
    }
}


} // namespace tr
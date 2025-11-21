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
#include <iomanip>  // 用于std::setprecision和std::fixed

namespace tr {

// ===== InternalContext 实现 =====

void Model::InternalContext::allocate(const std::vector<std::shared_ptr<Module>>& modules,
                                     const Shape& input_shape,
                                     std::shared_ptr<Backend> backend) {

    // ✅ 智能重用检测
    if (allocated_) {
        bool shape_same = (last_input_shape_ == input_shape);
        bool backend_same = (last_backend_ == backend.get());

        if (shape_same && backend_same) {
            return; // 缓存仍然有效，直接复用
        }
    }

    if (!backend) {
        throw TRException("[Model::InternalContext::allocate] Backend is null");
    }

    // 需要重新分配
    clear();

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

    last_input_shape_ = input_shape;
    last_backend_ = backend.get();
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
    : model_name_(name), last_cached_device_(tr::CPU) {
}

Model::Model(const std::string& name,
             const std::vector<std::shared_ptr<Module>>& modules)
    : model_name_(name), last_cached_device_(tr::CPU) {
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
        cached_output_ = input;  // 空模型直接缓存输入
        return input;  // 空模型直接返回输入
    }

    // 确保预分配缓存已初始化
    if (!ctx_.is_allocated()) {
        ctx_.allocate(modules_, input.shape(), backend_);
    }

    // ⭐ 零拷贝优化：直接使用预分配缓存，避免最后一次内存拷贝
    // 原来的实现：forward_into(input, output) → backend_->copy_into(cache, output)
    // 优化后：直接返回缓存张量，零拷贝访问
    modules_[0]->forward_into(input, ctx_.get_forward_cache(0));

    // 中间层：缓存i-1 到 缓存i
    for (size_t i = 1; i < modules_.size(); ++i) {
        modules_[i]->forward_into(ctx_.get_forward_cache(i-1), ctx_.get_forward_cache(i));
    }

    // 直接返回缓存张量，零拷贝！
    cached_output_ = ctx_.get_forward_cache(modules_.size() - 1);
    return cached_output_;
}

Tensor& Model::logits() {
    return cached_output_;
}

void Model::forward_into(const Tensor& input, Tensor& output) {
    if (modules_.empty()) {
        // 空模型直接复制输入到输出
        backend_->copy_into(input, output);
        // 缓存输出（用于logits访问）
        cached_output_ = output;
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
    // 注意：forward_into仍然需要拷贝到用户指定的output张量
    // 而forward()现在直接返回缓存张量，实现零拷贝
    backend_->copy_into(ctx_.get_forward_cache(modules_.size() - 1), output);

    // 缓存输出（用于logits访问）
    cached_output_ = output;
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

    // ⭐ 设备转移后，使参数缓存失效
    invalidate_all_param_caches();
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

    // ⭐ 后端设置后，使参数缓存失效
    invalidate_all_param_caches();
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

// ⭐ 智能缓存零拷贝参数指针接口实现
std::vector<Tensor*> Model::trainable_parameters() {
    // 检查缓存是否有效：设备变化或缓存未构建
    Device current_device = backend_ ? backend_->device() : tr::CPU;
    if (!param_cache_valid_ || last_cached_device_ != current_device) {
        rebuild_param_cache();
        param_cache_valid_ = true;
        last_cached_device_ = current_device;
    }

    return cached_param_ptrs_;
}

std::vector<Tensor*> Model::all_parameters() {
    // 检查缓存是否有效：设备变化或缓存未构建
    Device current_device = backend_ ? backend_->device() : tr::CPU;
    if (!all_cache_valid_ || last_cached_device_ != current_device) {
        rebuild_all_cache();
        all_cache_valid_ = true;
        last_cached_device_ = current_device;
    }

    return cached_all_ptrs_;
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

Model::MemoryProfile Model::analyze_memory(const Shape& input_shape) const {
    MemoryProfile profile;

    // 初始化分析结果
    profile.parameter_memory = 0;
    profile.activation_memory = 0;
    profile.gradient_memory = 0;
    profile.total_memory = 0;
    profile.layer_activations.clear();
    profile.layer_parameters.clear();

    // 预分配空间以提高性能
    profile.layer_activations.reserve(modules_.size());
    profile.layer_parameters.reserve(modules_.size());

    Shape current_shape = input_shape;

    for (const auto& module : modules_) {
        if (!module) {
            continue;
        }

        // 计算参数内存（仅基于数学计算，不分配实际内存）
        size_t param_mem = module->parameter_memory();
        profile.parameter_memory += param_mem;
        profile.layer_parameters.push_back(param_mem);

        // 使用infer_output_shape推断输出形状
        Shape output_shape = module->infer_output_shape(current_shape);

        // 计算激活值内存（假设FP32数据类型）
        size_t activation_mem = output_shape.numel() * sizeof(float);
        profile.activation_memory += activation_mem;
        profile.layer_activations.push_back(activation_mem);

        current_shape = output_shape;
    }

    // 梯度内存 = 参数内存（每个参数都有对应的梯度）
    profile.gradient_memory = profile.parameter_memory;

    // 总内存（训练模式）
    profile.total_memory = profile.parameter_memory +
                          profile.activation_memory +
                          profile.gradient_memory;

    return profile;
}

void Model::print_memory_profile(const Shape& input_shape) const {
    auto profile = analyze_memory(input_shape);

    std::cout << "=== Memory Profile ===" << std::endl;
    std::cout << "Model: " << model_name_ << std::endl;
    std::cout << "Input Shape: " << input_shape.to_string() << std::endl;
    std::cout << std::endl;

    std::cout << "Layer-wise Breakdown:" << std::endl;
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::cout << "  [" << i << "] " << modules_[i]->instance_name() << std::endl;
        std::cout << "    Parameters: " << format_bytes(profile.layer_parameters[i]) << std::endl;
        std::cout << "    Activations: " << format_bytes(profile.layer_activations[i]) << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Total Summary:" << std::endl;
    std::cout << "  Parameters: " << format_bytes(profile.parameter_memory) << std::endl;
    std::cout << "  Activations: " << format_bytes(profile.activation_memory) << std::endl;
    std::cout << "  Gradients: " << format_bytes(profile.gradient_memory) << std::endl;
    std::cout << "  Total (Training): " << format_bytes(profile.total_memory) << std::endl;
    std::cout << "  Total (Inference): "
              << format_bytes(profile.inference_memory()) << std::endl;
}

// ===== 私有辅助方法实现 =====

std::string Model::format_bytes(size_t bytes) const {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
    return oss.str();
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

// ===== ⭐ 参数缓存机制实现 =====

void Model::rebuild_param_cache() const {
    // 清空现有缓存
    cached_param_ptrs_.clear();

    // 预分配空间以提高性能
    size_t total_params = 0;
    for (auto& module : modules_) {
        total_params += module->parameters().size();
    }
    cached_param_ptrs_.reserve(total_params);

    // 直接收集参数指针，无内存拷贝
    for (auto& module : modules_) {
        auto& module_params = module->parameters();  // 使用非const版本
        for (auto& [key, param] : module_params) {
            cached_param_ptrs_.push_back(&param);  // 直接返回指针，零拷贝！
        }
    }
}

void Model::rebuild_all_cache() const {
    // 清空现有缓存
    cached_all_ptrs_.clear();

    // 预分配空间（目前只包含parameters，因为buffers没有公共接口）
    size_t total_params = 0;
    for (auto& module : modules_) {
        total_params += module->parameters().size();
    }
    cached_all_ptrs_.reserve(total_params);

    // 收集训练参数
    for (auto& module : modules_) {
        auto& module_params = module->parameters();  // 使用非const版本
        for (auto& [key, param] : module_params) {
            cached_all_ptrs_.push_back(&param);
        }
    }

    // TODO: 当buffers()接口可用时，添加buffer收集
    // 目前buffers没有公共访问接口，所以暂时只返回parameters
}

void Model::invalidate_all_param_caches() const {
    param_cache_valid_ = false;
    all_cache_valid_ = false;
    // 注意：不清空vector本身，只标记为无效，下次访问时重新构建
}


} // namespace tr
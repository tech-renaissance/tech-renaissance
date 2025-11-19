/**
 * @file model.h
 * @brief 模型类
 * @details Module的容器和编排器，管理Module的生命周期和执行顺序，提供预分配内存优化
 * @version 1.45.0
 * @date 2025-11-17
 * @author 技术觉醒团队
 * @note 依赖项: module.h
 * @note 所属系列: model
 */

#pragma once

#include "tech_renaissance/model/module.h"
#include <memory>
#include <vector>
#include <unordered_map>

namespace tr {

/**
 * @brief 模型类
 * @details Module的容器和编排器，提供三种构造方式和预分配内存优化
 */
class Model {
public:
    /**
     * @brief 内存分析结果结构体
     * @details 提供详细的内存使用分析数据
     */
    struct MemoryProfile {
        size_t parameter_memory;                     // 参数占用内存（字节）
        size_t activation_memory;                    // 激活值占用内存（字节）
        size_t gradient_memory;                      // 梯度占用内存（字节）
        size_t total_memory;                         // 总占用内存（训练模式）

        std::vector<size_t> layer_activations;       // 各层激活值内存
        std::vector<size_t> layer_parameters;        // 各层参数内存

        /**
         * @brief 获取推理模式总内存（不包含梯度）
         * @return 推理模式内存大小
         */
        size_t inference_memory() const {
            return parameter_memory + activation_memory;
        }

        /**
         * @brief 获取训练模式总内存
         * @return 训练模式内存大小
         */
        size_t training_memory() const {
            return total_memory;
        }
    };

private:
    /**
     * @brief 内部上下文（私有实现细节）
     * @details 负责预分配内存管理，用户完全无感知
     */
    struct InternalContext {
        std::vector<Tensor> forward_cache_;   // 前向传播缓存
        std::vector<Tensor> backward_cache_;  // 反向传播缓存
        bool allocated_ = false;              // 分配状态标志

        /**
         * @brief 预分配所有层的缓存空间
         * @param modules 模块列表
         * @param input_shape 输入形状
         * @param backend 后端智能指针
         */
        void allocate(const std::vector<std::shared_ptr<Module>>& modules,
                     const Shape& input_shape,
                     std::shared_ptr<Backend> backend);

        /**
         * @brief 清空所有缓存
         */
        void clear();

        /**
         * @brief 检查是否已分配
         * @return true表示已分配，false表示未分配
         */
        bool is_allocated() const { return allocated_; }

        /**
         * @brief 获取指定层的前向缓存
         * @param index 层索引
         * @return 前向缓存张量引用
         */
        Tensor& get_forward_cache(size_t index);

        /**
         * @brief 获取指定层的反向缓存
         * @param index 层索引
         * @return 反向缓存张量引用
         */
        Tensor& get_backward_cache(size_t index);
    };

    // === 成员变量 ===
    std::string model_name_;                                    // 模型名称
    std::vector<std::shared_ptr<Module>> modules_;              // 有序模块列表
    std::shared_ptr<Backend> backend_;                           // 全局后端智能指针
    InternalContext ctx_;                                       // 内部上下文（预分配管理）
    std::unordered_map<std::string, int> type_counters_;        // 类型计数器（用于自动命名）
    bool training_ = true;                                      // 训练/推理模式
    bool frozen_ = false;                                       // 结构冻结标志
    Tensor cached_output_;                                      // 缓存的最后输出（用于logits访问）

public:
    /**
     * @brief 构造函数1：默认构造
     * @param name 模型名称，默认为"Model"
     */
    explicit Model(const std::string& name = "Model");

    /**
     * @brief 构造函数2：初始化列表构造
     * @param name 模型名称
     * @param modules 模块初始化列表
     */
    explicit Model(const std::string& name,
                   const std::vector<std::shared_ptr<Module>>& modules);

    
    /**
     * @brief 析构函数
     */
    ~Model() = default;

    // === 模块管理 ===

    /**
     * @brief 添加模块（自动命名）
     * @param module 模块智能指针
     */
    void add_module(std::shared_ptr<Module> module);

    /**
     * @brief 添加模块（手动命名）
     * @param custom_name 自定义名称
     * @param module 模块智能指针
     */
    void add_module(const std::string& custom_name,
                   std::shared_ptr<Module> module);

    /**
     * @brief 获取模块数量
     * @return 模块数量
     */
    size_t num_modules() const { return modules_.size(); }

    /**
     * @brief 获取指定模块
     * @param index 模块索引
     * @return 模块智能指针
     */
    std::shared_ptr<Module> get_module(size_t index) const;

    // === 核心计算 ===

    /**
     * @brief 前向传播
     * @param input 输入张量
     * @return 输出张量
     */
    Tensor forward(const Tensor& input);

    /**
     * @brief 前向传播（into型，使用预分配缓存）
     * @param input 输入张量
     * @param output 输出张量
     */
    void forward_into(const Tensor& input, Tensor& output);

    /**
     * @brief 获取模型最后输出的logits（非const引用，用于Loss类）
     * @return 缓存的输出张量引用
     * @note 必须在forward()或forward_into()调用后使用
     */
    Tensor& logits();

    /**
     * @brief 反向传播
     * @param grad_output 输出梯度
     * @return 输入梯度
     */
    Tensor backward(const Tensor& grad_output);

    /**
     * @brief 反向传播（into型，使用预分配缓存）
     * @param grad_output 输出梯度
     * @param grad_input 输入梯度
     */
    void backward_into(const Tensor& grad_output, Tensor& grad_input);

    // === 设备管理 ===

    /**
     * @brief 设备转移
     * @param device 目标设备
     */
    void to(const Device& device);

    /**
     * @brief 获取当前设备
     * @return 当前设备
     */
    Device device() const;

    // === 后端管理 ===

    /**
     * @brief 设置后端
     * @param backend 后端智能指针
     */
    void set_backend(std::shared_ptr<Backend> backend);

    /**
     * @brief 获取后端
     * @return 后端智能指针
     */
    std::shared_ptr<Backend> get_backend() const { return backend_; }

    // === 训练模式管理 ===

    /**
     * @brief 设置为训练模式
     */
    void train();

    /**
     * @brief 设置为推理模式
     */
    void eval();

    /**
     * @brief 检查是否为训练模式
     * @return true表示训练模式，false表示推理模式
     */
    bool is_training() const { return training_; }

    // === 参数管理 ===

    /**
     * @brief 获取所有参数（递归聚合）
     * @return 参数映射表
     */
    std::unordered_map<std::string, Tensor> parameters() const;

    /**
     * @brief 获取所有可训练参数的指针（零拷贝）
     * @return 参数指针向量，用于高效优化器更新
     * @details 返回所有模块的可训练参数指针，无内存拷贝
     */
    std::vector<Tensor*> trainable_parameters();

    /**
     * @brief 获取所有参数的指针（包括buffers，零拷贝）
     * @return 所有参数指针向量，用于完整性检查
     * @details 包含buffers等非训练参数，无内存拷贝
     */
    std::vector<Tensor*> all_parameters();

    /**
     * @brief 获取所有参数的梯度（递归聚合）
     * @return 参数梯度映射表
     */
    std::unordered_map<std::string, Tensor> gradients() const;

    /**
     * @brief 清零所有参数的梯度
     */
    void zero_grad();

    /**
     * @brief 计算参数内存占用
     * @return 内存占用大小（字节）
     */
    size_t parameter_memory() const;

    // === 自动命名机制 ===

    /**
     * @brief 自动为模块命名
     * @param module 模块智能指针
     */
    void auto_name_module(std::shared_ptr<Module> module);

    // === 工厂方法（构造方式3） ===

    /**
     * @brief 工厂方法：创建模型（推荐使用）
     * @tparam Args 模块参数类型
     * @param name 模型名称
     * @param args 模块参数列表
     * @return 模型智能指针
     */
    template<typename... Args>
    static std::shared_ptr<Model> create(const std::string& name, Args&&... args);

    // === 内存分析 ===

    /**
     * @brief 初始化预分配缓存
     * @param input_shape 输入形状
     */
    void initialize(const Shape& input_shape);

    /**
     * @brief 分析模型内存使用情况
     * @param input_shape 输入张量形状
     * @return 内存分析结果
     * @note 轻量级方法，仅基于形状进行数学计算，不分配实际内存
     */
    MemoryProfile analyze_memory(const Shape& input_shape) const;

    /**
     * @brief 打印详细的内存使用报告
     * @param input_shape 输入张量形状
     * @note 提供美观的层级内存分布展示
     */
    void print_memory_profile(const Shape& input_shape) const;

private:
    /**
     * @brief 格式化字节数为可读字符串
     * @param bytes 字节数
     * @return 格式化后的字符串（如 "1.23 MB"）
     */
    std::string format_bytes(size_t bytes) const;

    // === 序列化 ===

    /**
     * @brief 保存模型
     * @param filename 文件名
     */
    void save(const std::string& filename) const;

    /**
     * @brief 加载模型
     * @param filename 文件名
     * @return 模型智能指针
     */
    static std::shared_ptr<Model> load(const std::string& filename);

    // === 调试辅助 ===

    /**
     * @brief 打印模型结构
     */
    void print_model() const;

    /**
     * @brief 获取模型名称
     * @return 模型名称
     */
    const std::string& name() const { return model_name_; }

private:
    /**
     * @brief 初始化所有模块的后端
     */
    void initialize_modules_backend();

    /**
     * @brief 验证模型结构
     */
    void validate_model() const;
};

// ===== 模板实现 =====

template<typename... Args>
std::shared_ptr<Model> Model::create(const std::string& name, Args&&... args) {
    auto model = std::make_shared<Model>(name);
    (model->add_module(std::forward<Args>(args)), ...);
    // validate_model() will be called after backend is set
    return model;
}

} // namespace tr
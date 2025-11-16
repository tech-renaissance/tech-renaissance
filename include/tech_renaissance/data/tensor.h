/**
 * @file tensor.h
 * @brief 张量类声明
 * @details 张量类，持有Storage句柄以及张量的形状、类型、偏移等信息，支持设备无关操作，维度不得超过4维
 * @version 1.00.00
 * @date 2025-10-25
 * @author 技术觉醒团队
 * @note 依赖项: shape.h, dtype.h, device.h, storage.h
 * @note 所属系列: data
 */

#pragma once

#include <memory>
#include <string>
#include <ostream>
#include <type_traits>
#include "tech_renaissance/data/shape.h"
#include "tech_renaissance/data/dtype.h"
#include "tech_renaissance/data/device.h"
#include "tech_renaissance/data/strides.h"

namespace tr {

// 前向声明
class Storage;
class Backend;

/**
 * @class Tensor
 * @brief 张量类
 * @details 持有Storage句柄以及张量的形状、类型、偏移等信息，支持设备无关操作
 *
 * 设计理念：
 * - Tensor是轻量级的元数据容器 + Storage句柄，不负责计算
 * - 所有运算都由Backend完成，Tensor只提供基本信息访问
 * - 支持设备无关操作，设备转移通过Backend的拷贝接口实现
 * - 第一期不支持原地计算（inplace操作）
 */
class Tensor {
public:
    /**
     * @brief 默认构造函数
     * @details 创建一个空的Tensor（无Storage）
     *
     * 警告：此构造函数创建的是空张量，没有分配内存。
     * 不能直接使用空张量进行任何数据操作。
     * 空张量通常用作占位符或后续赋值的目标。
     */
    Tensor();

    /**
     * @brief 拷贝构造函数
     * @param other 另一个Tensor对象
     */
    Tensor(const Tensor& other) = default;

    /**
     * @brief 移动构造函数
     * @param other 另一个Tensor对象
     */
    Tensor(Tensor&& other) = default;

    /**
     * @brief 析构函数
     */
    ~Tensor() = default;

    /**
     * @brief 拷贝赋值运算符
     * @param other 另一个Tensor对象
     * @return 当前对象的引用
     */
    Tensor& operator=(const Tensor& other) = default;

    /**
     * @brief 移动赋值运算符
     * @param other 另一个Tensor对象
     * @return 当前对象的引用
     */
    Tensor& operator=(Tensor&& other) = default;

    // ===== 元数据访问方法 =====

    /**
     * @brief 获取张量形状
     * @return Shape对象
     */
    const Shape& shape() const noexcept;

    /**
     * @brief 获取数据类型
     * @return 数据类型
     */
    DType dtype() const noexcept;

    /**
     * @brief 获取设备
     * @return 设备对象
     */
    Device device() const noexcept;

    /**
     * @brief 获取维度数
     * @return 维度数（0-4）
     */
    int32_t ndim() const noexcept;

    /**
     * @brief 获取元素总数
     * @return 元素总数
     */
    int64_t numel() const noexcept;

    /**
     * @brief 获取指定维度的尺寸
     * @param dim 维度索引（0-based，相对于实际维度）
     * @return 维度尺寸
     * @throws std::out_of_range 如果索引超出范围
     */
    int32_t dim_size(int32_t dim) const;

    /**
     * @brief 获取批次大小N
     * @return N维度尺寸
     */
    int32_t batch() const noexcept;

    /**
     * @brief 获取通道数C
     * @return C维度尺寸
     */
    int32_t channel() const noexcept;

    /**
     * @brief 获取高度H
     * @return H维度尺寸
     */
    int32_t height() const noexcept;

    /**
     * @brief 获取宽度W
     * @return W维度尺寸
     */
    int32_t width() const noexcept;

    /**
     * @brief 获取数据类型大小（字节）
     * @return 数据类型大小
     */
    size_t dtype_size() const noexcept;

    /**
     * @brief 获取Storage句柄
     * @return Storage智能指针
     */
    std::shared_ptr<Storage> storage() const noexcept;

    /**
     * @brief 检查是否为空Tensor
     * @return 如果没有Storage则为true
     */
    bool is_empty() const noexcept;

    /**
     * @brief 检查是否为标量
     * @return 如果是标量则为true
     */
    bool is_scalar() const noexcept;

    /**
     * @brief 检查是否连续存储
     * @return 如果内存布局是连续的则返回true
     */
    bool is_contiguous() const noexcept;

    /**
     * @brief 获取张量的步长
     * @return Strides对象
     */
    const Strides& strides() const noexcept;

    /**
     * @brief 检查张量是否为视图
     * @return 如果是视图则返回true
     */
    bool is_view() const noexcept;

    /**
     * @brief 获取所需内存大小
     * @return 内存大小（字节）
     */
    size_t memory_size() const noexcept;

    // ===== 静态工厂方法 =====

    // ========== 重要：张量创建方法警告 ==========
    // 注意：虽然以下静态方法会自动分配内存并返回可用张量，
    // 但强烈建议不要使用Tensor类的工厂函数创建新张量！
    // 推荐使用相应Backend子类的方法：
    // - Tensor.zeros() -> CpuBackend::zeros() 或其他Backend子类的zeros()
    // - Tensor.ones() -> CpuBackend::ones() 或其他Backend子类的ones()
    // - Tensor.full() -> CpuBackend::full() 或其他Backend子类的full()
    // - Tensor.empty() -> CpuBackend::empty() 或其他Backend子类的empty()
    // Backend子类方法提供更好的设备管理和性能优化！

    /**
     * @brief 创建全零张量
     * @warning 不建议使用！请使用Backend子类的zeros()替代
     * @param shape 张量形状
     * @param dtype 数据类型
     * @param device 设备
     * @return 新的Tensor对象（已分配内存）
     */
    static Tensor zeros(const Shape& shape, DType dtype = DType::FP32, const Device& device = tr::CPU);

    /**
     * @brief 创建全一张量
     * @warning 不建议使用！请使用Backend子类的ones()替代
     * @param shape 张量形状
     * @param dtype 数据类型
     * @param device 设备
     * @return 新的Tensor对象
     */
    static Tensor ones(const Shape& shape, DType dtype = DType::FP32, const Device& device = tr::CPU);

    /**
     * @brief 创建填充指定值的张量
     * @warning 不建议使用！请使用Backend子类的full()替代
     * @param shape 张量形状
     * @param value 填充值
     * @param dtype 数据类型
     * @param device 设备
     * @return 新的Tensor对象
     */
    static Tensor full(const Shape& shape, float value, DType dtype = DType::FP32, const Device& device = tr::CPU);

    /**
     * @brief 创建未初始化的张量
     * @warning 不建议使用！请使用Backend子类的empty()替代
     * @param shape 张量形状
     * @param dtype 数据类型
     * @param device 设备
     * @return 新的Tensor对象
     */
    static Tensor empty(const Shape& shape, DType dtype = DType::FP32, const Device& device = tr::CPU);

    /**
     * @brief 获取原始数据指针
     * @return 数据指针
     */
    void* data_ptr() noexcept;

    /**
     * @brief 获取原始数据指针（const版本）
     * @return 数据指针
     */
    const void* data_ptr() const noexcept;

    // ===== 数据移动操作 =====

    /**
     * @brief 创建视图（浅拷贝）
     * @details 创建共享Storage的新Tensor对象
     * @return 新的Tensor对象
     */
    Tensor view() const;

    // ===== 实用方法 =====

    /**
     * @brief 获取张量的字符串表示
     * @return 字符串表示
     */
    std::string to_string() const;

    /**
     * @brief 张量是否已分配内存
     * @return 是否已分配内存
     */
    bool storage_allocated() const {return storage_ != nullptr;}

    /**
     * @brief 打印张量内容和元数据
     * @param name 张量名称
     */
    void print(const std::string& name = "") const;

    /**
     * @brief 打印张量内容和元数据（带精度控制）
     * @param name 张量名称
     * @param precision 浮点数小数位数（仅对FP32有效）
     */
    void print(const std::string& name, int precision) const;

    /**
     * @brief 打印张量摘要信息（不包含具体数据）
     * @param name 张量名称
     */
    void summary(const std::string& name = "") const;

    /**
     * @brief 从CPU数据初始化
     * @param data CPU数据指针
     * @param size 数据大小（字节）
     * @throws std::invalid_argument 如果数据大小不匹配
     */
    void from_cpu_data(const void* data, size_t size);

    /**
     * @brief 复制到CPU数据
     * @param data CPU数据指针
     * @param size 数据大小（字节）
     * @throws std::invalid_argument 如果数据大小不匹配
     */
    void to_cpu_data(void* data, size_t size) const;

    /**
     * @brief 获取标量张量的值
     * @return 标量值
     * @throws std::runtime_error 如果不是标量张量
     */
    template<typename T>
    T item() const {
        if (!is_scalar()) {
            throw TRException("item() only for scalar tensors");
        }

        // 新增：检查Storage是否已分配
        if (is_empty()) {
            throw TRException("Cannot get item: Tensor is empty (no Storage allocated)");
        }

        auto backend = get_backend();
        if (!backend) {
            throw TRException("Backend not available for item() operation");
        }

        // 根据类型调用相应的Backend方法
        if constexpr (std::is_same_v<T, float>) {
            return backend->get_scalar_float(*this);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return backend->get_scalar_int32(*this);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return backend->get_scalar_int8(*this);
        } else {
            static_assert(!sizeof(T), "Unsupported type for item()");
        }
    }

    // ===== 比较运算符 =====

    /**
     * @brief 相等比较运算符
     * @param other 另一个Tensor对象
     * @return 如果相等返回true
     */
    bool operator==(const Tensor& other) const noexcept;

    /**
     * @brief 不等比较运算符
     * @param other 另一个Tensor对象
     * @return 如果不等返回true
     */
    bool operator!=(const Tensor& other) const noexcept;

protected:
    /**
     * @brief 构造函数（受保护，强制使用工厂方法）
     *
     * 警告：此构造函数不分配内存！
     * 执行此构造函数只会创建Tensor对象的元数据（形状、类型、设备等），
     * 但不会为张量数据分配实际的内存空间。
     *
     * 不要直接使用构造函数来创建张量！
     * 构造函数只能被Backend类及其子类使用，因为它们会在构造后立即分配内存。
     *
     * 正确的张量创建方式：
     * - 使用Backend的子类的empty(), zeros(), ones(), full()等方法（如果有）
     * - 使用Backend的子类的randn(), uniform(), randint()等随机生成方法（如果有）
     * - 如果Backend的子类没有提供上述方法，则可考虑先在CPU后端上创建张量，再拷贝到目标设备
     *
     * @param shape 张量形状
     * @param dtype 数据类型
     * @param device 设备
     * @throws std::invalid_argument 如果shape为标量且dtype不支持
     */
    Tensor(const Shape& shape, DType dtype, const Device& device = tr::CPU);

    /**
     * @brief 创建视图的构造函数（仅Backend使用）
     * @details 由Backend调用，用于零拷贝创建视图
     * @param storage Storage句柄（共享）
     * @param shape 张量形状
     * @param strides 步长信息
     * @param dtype 数据类型
     * @param device 设备信息
     * @param offset 在Storage中的偏移量
     */
    Tensor(std::shared_ptr<Storage> storage, const Shape& shape, const Strides& strides,
           DType dtype, const Device& device, size_t offset);

private:
    Shape shape_;                          ///< 张量形状
    DType dtype_;                          ///< 数据类型
    Device device_;                        ///< 设备信息
    std::shared_ptr<Storage> storage_;     ///< Storage句柄
    size_t offset_;                        ///< 在Storage中的偏移量（目前为0）
    Strides strides_;                      ///< 步长信息
    bool is_view_;                         ///< 视图标识

    // 友元声明：Backend类可以访问protected成员
    friend class Backend;
    friend class CpuBackend;
    friend class CudaBackend;
    friend class PythonSession;

  
    /**
     * @brief 验证形状和数据类型的兼容性
     * @throws std::invalid_argument 如果不兼容
     */
    void validate_shape_dtype() const;

    /**
     * @brief 获取当前Backend实例
     * @return Backend智能指针
     * @throws std::runtime_error 如果Backend不可用
     */
    std::shared_ptr<Backend> get_backend() const;

    /**
     * @brief 格式化张量内容输出（PyTorch风格）
     * @param oss 输出流
     * @param precision 浮点数精度
     */
    void format_tensor_content(std::ostringstream& oss, int precision) const;

    /**
     * @brief 创建并分配存储的私有辅助函数
     * @warning 仅限内部使用！外部请使用Backend方法
     * @param shape 张量形状
     * @param dtype 数据类型
     * @param device 设备
     * @return 已分配内存的Tensor
     */
    static Tensor create_and_allocate(const Shape& shape, DType dtype, const Device& device);
};

/**
 * @brief 输出流运算符重载
 * @param os 输出流
 * @param tensor Tensor对象
 * @return 输出流的引用
 */
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

} // namespace tr
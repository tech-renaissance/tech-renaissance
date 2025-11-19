/**
 * @file cpu_backend.h
 * @brief CPU后端实现
 * @details 基于Eigen库实现高性能CPU计算
 * @version 1.00.00
 * @date 2025-10-26
 * @author 技术觉醒团队
 * @note 依赖项: backend.h, Eigen
 * @note 所属系列: backend
 */

#pragma once

#include "tech_renaissance/backend/backend.h"
#include "tech_renaissance/data/shape.h"
#include "tech_renaissance/data/offset.h"
#include "tech_renaissance/data/strides.h"
#define EXPORT_TENSOR dynamic_cast<CpuBackend*>(BackendManager::instance().get_backend(CPU).get())->export_tensor
#define IMPORT_TENSOR dynamic_cast<CpuBackend*>(BackendManager::instance().get_backend(CPU).get())->import_tensor


namespace tr {

class CpuBackend : public Backend {
public:
    /**
     * @brief CPU后端构造函数
     */
    explicit CpuBackend();

    /**
     * @brief 析构函数
     */
    ~CpuBackend() override = default;

    // 内存管理
    std::shared_ptr<void> allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void* get_data_ptr(const std::shared_ptr<void>& holder) override;
    void copy_data(void* dst, const void* src, size_t size,
                   const Device& dst_device, const Device& src_device) const override;

    // 填充操作
    void fill(Tensor& dst, float value) override;
    void fill(Tensor& dst, int8_t value) override;
    void fill(Tensor& dst, int32_t value) override;

    // fill方法别名
    void fill_fp32(Tensor& dst, float value) override;
    void fill_int8(Tensor& dst, int8_t value) override;
    void fill_int32(Tensor& dst, int32_t value) override;

    // 基本运算
    Tensor add(const Tensor& a, const Tensor& b) const;
    void add_into(const Tensor& a, const Tensor& b, Tensor& result) const override;
    void minus_into(const Tensor& a, const Tensor& b, Tensor& result) const override;
    void sum_into(const Tensor& tensor_a, Tensor& result, int32_t dim, bool keep_dim = false) const override;
    Tensor mul(const Tensor& a, const Tensor& b) const;
    Tensor mm(const Tensor& a, const Tensor& b) override;
    void mm_into(const Tensor& a, const Tensor& b, Tensor& result) override;

    // 设备转换方法
    Tensor to(const Tensor& tensor, const Device& device) const override;
    Tensor to_cpu(const Tensor& tensor) const override;
    Tensor from_cpu(const Tensor& tensor) const override;

    // 辅助方法
    std::string name() const override { return "CpuBackend"; }
    Device device() const override { return tr::CPU; }

    // 数据访问
    float get_scalar_float(const Tensor& tensor) override;
    int32_t get_scalar_int32(const Tensor& tensor) override;
    int8_t get_scalar_int8(const Tensor& tensor) override;
    int64_t get_memory_size(const Tensor& tensor) override;

    // 元素级数据访问（线性索引）
    float get_item_fp32(const Tensor& tensor_a, int64_t element_index);
    void set_item_fp32(Tensor& tensor_a, int64_t element_index, float value);
    int8_t get_item_int8(const Tensor& tensor_a, int64_t element_index);
    void set_item_int8(Tensor& tensor_a, int64_t element_index, int8_t value);
    int32_t get_item_int32(const Tensor& tensor_a, int64_t element_index);
    void set_item_int32(Tensor& tensor_a, int64_t element_index, int32_t value);

    // 这个方法会创建一个大小为(0, 0, 0, 0)的空张量，不占用内存，是本框架建议的销毁张量的方式
    static Tensor null_tensor();

    Tensor empty(const Shape& shape, DType dtype) override;
    Tensor empty(const Shape& shape, DType dtype) const;
    Tensor zeros(const Shape& shape, DType dtype) override;
    Tensor ones(const Shape& shape, DType dtype) override;
    // 张量创建函数（V1.29.4新增）
    Tensor full(const Shape& shape, float value, DType dtype = DType::FP32);
    static void full_inplace(Tensor& tensor_a, float value);
    Tensor randn(const Shape& shape, unsigned int seed = 0) override;
    static void randn_inplace(Tensor& tensor_a, unsigned int seed = 0);
    Tensor uniform(const Shape& shape, float min_val = 0.0f, float max_val = 1.0f, unsigned int seed = 0);
    static void uniform_inplace(Tensor& tensor_a, float min_val = 0.0f, float max_val = 1.0f, unsigned int seed = 0);
    Tensor randint(const Shape& shape, int low, int high, DType dtype, unsigned int seed = 0);
    static void randint_inplace(Tensor& tensor_a, int low, int high, DType dtype, unsigned int seed = 0);
    Tensor randbool(const Shape& shape, float rate_of_zeros, unsigned int seed = 0, DType dtype = DType::FP32);
    static void randbool_inplace(Tensor& tensor_a, float rate_of_zeros, unsigned int seed = 0);

    // 类型转换函数（V1.31.2新增）
    Tensor cast(const Tensor& tensor_a, DType target_dtype);
    void cast_inplace(Tensor& tensor_a, DType target_dtype);
    void cast_into(const Tensor& tensor_a, DType target_dtype, Tensor& result);

    // 张量IO算子（CPU后端独有功能）
    void export_tensor(const Tensor& tensor, const std::string& filename) const;
    Tensor import_tensor(const std::string& filename) const;

    // 张量比较算子
    bool is_close(const Tensor& tensor_a, const Tensor& tensor_b, float eps = 5e-5f) const;

    // 误差计算方法（V1.23.1新增）
    double get_mean_abs_err(const Tensor& tensor_a, const Tensor& tensor_b) const;
    double get_mean_rel_err(const Tensor& tensor_a, const Tensor& tensor_b) const;

    // 单目运算（V1.25.1新增）
    // 非原地运算
    Tensor zeros_like(const Tensor& input) const;
    Tensor ones_like(const Tensor& input) const;
    Tensor relu(const Tensor& input) const;
    Tensor sign(const Tensor& input) const;
    Tensor square(const Tensor& input) override;
    Tensor sqrt(const Tensor& input) override;
    Tensor abs(const Tensor& input) const;
    Tensor negative(const Tensor& input) const;
    Tensor reciprocal(const Tensor& input) const;
    Tensor round(const Tensor& input) const;
    Tensor transpose(const Tensor& input) const override;

    // 原地运算
    void zeros_inplace(Tensor& input) const;
    void ones_inplace(Tensor& input) const;
    void relu_inplace(Tensor& input) const;
    void sign_inplace(Tensor& input) const;
    void square_inplace(Tensor& input) override;
    void sqrt_inplace(Tensor& input) override;
    void abs_inplace(Tensor& input) const;
    void negative_inplace(Tensor& input) const;
    void reciprocal_inplace(Tensor& input) const;
    void round_inplace(Tensor& input) const;
    void transpose_inplace(Tensor& input) const;

    // 指定输出张量的运算（V1.26.3新增）
    void zeros_into(const Tensor& input, Tensor& output) const;
    void ones_into(const Tensor& input, Tensor& output) const;
    void relu_into(const Tensor& input, Tensor& output) const;
    void sign_into(const Tensor& input, Tensor& output) const;
    void square_into(const Tensor& input, Tensor& output) override;
    void sqrt_into(const Tensor& input, Tensor& output) override;
    void abs_into(const Tensor& input, Tensor& output) const;
    void negative_into(const Tensor& input, Tensor& output) const;
    void reciprocal_into(const Tensor& input, Tensor& output) const;
    void round_into(const Tensor& input, Tensor& output) const;
    void transpose_into(const Tensor& input, Tensor& output) const override;

    // 张量复制操作（V1.26.5新增）
    Tensor copy(const Tensor& tensor) const override;
    void copy_into(const Tensor& src, Tensor& dst) const override;

    // 张量切片操作（V1.33.2新增）
    Tensor slice(const Tensor& tensor_a, const Offset& offset);
    void slice_into(const Tensor& tensor_a, Tensor& result, const Offset& offset);

    // 标量运算（V1.28.1新增）
    // 乘法：tensor * scalar
    Tensor mul(const Tensor& input, float scalar) const;
    void mul_inplace(Tensor& input, float scalar) const;
    void mul_into(const Tensor& input, float scalar, Tensor& output) const;
    void mul_into(const Tensor& a, const Tensor& b, Tensor& result) const;

    // 加法：tensor + scalar
    Tensor add(const Tensor& input, float scalar) const;
    void add_inplace(Tensor& input, float scalar) const;
    void add_into(const Tensor& input, float scalar, Tensor& output) const;

    // 减法：tensor - scalar
    Tensor minus(const Tensor& input, float scalar) const;
    void minus_inplace(Tensor& input, float scalar) const;
    void minus_into(const Tensor& input, float scalar, Tensor& output) const;

    // 减法：scalar - tensor
    Tensor minus(float scalar, const Tensor& input) const;
    void minus_inplace(float scalar, Tensor& input) const;
    void minus_into(float scalar, const Tensor& input, Tensor& output) const;

    // 乘加：tensor * scalar_x + scalar_y
    Tensor mac(const Tensor& input, float scalar_x, float scalar_y) const;
    void mac_inplace(Tensor& input, float scalar_x, float scalar_y) const;
    void mac_into(const Tensor& input, float scalar_x, float scalar_y, Tensor& output) const;

    // 裁剪：clamp(tensor, min_val, max_val)
    Tensor clamp(const Tensor& input, float min_val, float max_val) const;
    void clamp_inplace(Tensor& input, float min_val, float max_val) const;
    void clamp_into(const Tensor& input, float min_val, float max_val, Tensor& output) const;

  
    // 可广播张量运算（V1.28.1新增）
    // 张量加法：支持广播
    Tensor add_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const;
    void add_broadcast_into(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result) const;

    // 张量减法：支持广播
    Tensor minus_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const;
    void minus_broadcast_into(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result) const;

    // 张量乘法：支持广播
    Tensor mul_broadcast(const Tensor& tensor_a, const Tensor& tensor_b) const;
    void mul_broadcast_into(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result) const;

    // 张量扩展广播（V1.28.1新增）
    // 将张量广播到指定形状
    Tensor expand(const Tensor& tensor_a, const Shape& shape_b) const;
    void expand_into(const Tensor& tensor_a, Tensor& tensor_b) const;

    // 张量维度操作（V1.29.2新增）
    // Unsqueeze：在指定位置插入大小为1的维度
    Tensor unsqueeze(const Tensor& tensor_a, int32_t dim) const;
    void unsqueeze_inplace(Tensor& tensor_a, int32_t dim) const;
    void unsqueeze_into(const Tensor& tensor_a, Tensor& tensor_b) const;

    // Squeeze：移除大小为1的指定维度
    Tensor squeeze(const Tensor& tensor_a, int32_t dim) const;
    void squeeze_inplace(Tensor& tensor_a, int32_t dim) const;
    void squeeze_into(const Tensor& tensor_a, Tensor& tensor_b) const;

    // Pad：在张量的H和W维度周围补零
    Tensor pad(const Tensor& tensor_a, int32_t padding) const;
    void pad_into(const Tensor& tensor_a, int32_t padding, Tensor& tensor_b) const;

    // Softmax操作（V1.34.0新增）
    Tensor softmax(const Tensor& tensor_a, int32_t dim);
    void softmax_inplace(Tensor& tensor_a, int32_t dim);
    void softmax_into(const Tensor& tensor_a, Tensor& result, int32_t dim);

    // Max操作（V1.34.0新增）
    Tensor max(const Tensor& tensor_a, int32_t dim, bool keep_dim = false);
    void max_into(const Tensor& tensor_a, Tensor& result, int32_t dim, bool keep_dim = false);

    // Sum操作（V1.34.0新增）
    Tensor sum(const Tensor& tensor_a, int32_t dim, bool keep_dim = false);
    void sum_into(const Tensor& tensor_a, Tensor& result, int32_t dim, bool keep_dim = false);

    // ArgMax操作（V1.34.0新增）
    Tensor argmax(const Tensor& tensor_a, int32_t dim, bool keep_dim = false);
    void argmax_into(const Tensor& tensor_a, Tensor& result, int32_t dim, bool keep_dim = false);

    // Pooling操作（V1.35.0新增）
    Tensor max_pool(const Tensor& input, int32_t kernel_size = 2, int32_t stride = 2);
    void max_pool_into(const Tensor& input, Tensor& result, int32_t kernel_size = 2, int32_t stride = 2);
    Tensor global_avg_pool(const Tensor& input);
    void global_avg_pool_into(const Tensor& input, Tensor& result);

    // Convolution操作（V1.35.0新增）
    Tensor conv(const Tensor& input, const Tensor& kernel, int32_t stride = 1, int32_t padding = 0);
    void conv_into(const Tensor& input, const Tensor& kernel, Tensor& result, int32_t stride = 1, int32_t padding = 0);
    Tensor transposed_conv(const Tensor& input, const Tensor& kernel, int32_t stride = 1, int32_t padding = 0);
    void transposed_conv_into(const Tensor& input, const Tensor& kernel, Tensor& result, int32_t stride = 1, int32_t padding = 0);

    // 形状变换和双曲函数操作（V1.42.1新增）
    // 视图操作
    Tensor view(const Tensor& input, const Shape& new_shape) override;

    // reshape操作
    Tensor reshape(const Tensor& tensor_a, const Shape& shape) override;
    void reshape_inplace(Tensor& tensor_a, const Shape& shape) override;
    void reshape_into(const Tensor& tensor_a, Tensor& result, const Shape& shape) override;

    // tanh双曲正切函数
    Tensor tanh(const Tensor& tensor_a) override;
    void tanh_inplace(Tensor& tensor_a) override;
    void tanh_into(const Tensor& tensor_a, Tensor& result) override;

    // dtanh双曲正切导函数
    Tensor dtanh(const Tensor& tensor_a) override;
    void dtanh_inplace(Tensor& tensor_a) override;
    void dtanh_into(const Tensor& tensor_a, Tensor& result) override;

    // INT32张量比较操作（V1.42.4新增）
    // 比较两个INT32张量的每个元素是否相等
    void eq_into(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result) const;
    Tensor eq(const Tensor& tensor_a, const Tensor& tensor_b) const;
    bool equal(const Tensor& tensor_a, const Tensor& tensor_b) const;

    // One-hot编码操作（V1.42.6新增）
    // 将1D INT32标签张量转换为2D FP32 one-hot编码
    Tensor one_hot(const Tensor& label, int32_t num_classes, float label_smoothing = 0.0f) override;
    void one_hot_into(const Tensor& label, Tensor& result, int32_t num_classes, float label_smoothing = 0.0f) override;

    // 交叉熵损失函数（V1.42.6新增）
    // 计算预测张量和标签张量之间的交叉熵损失
    float crossentropy(const Tensor& pred, const Tensor& label, std::string reduction = "mean") override;

    // MSE损失函数（V1.42.7新增）
    // 计算预测张量和目标张量之间的均方误差损失
    float mse(const Tensor& pred, const Tensor& target, std::string reduction = "mean");

    // ===== Backend基类新增方法的override声明 =====

    // 注意：以下方法在CPU后端中已经存在，只需要添加override关键字
    // 形状变换操作
    // Tensor reshape(const Tensor& tensor_a, const Shape& shape) override;  // 已存在，需加override
    // void reshape_inplace(Tensor& tensor_a, const Shape& shape) override;   // 已存在，需加override
    // void reshape_into(const Tensor& tensor_a, Tensor& result, const Shape& shape) override;  // 已存在，需加override

    // 双曲函数操作
    // Tensor tanh(const Tensor& tensor_a) override;  // 已存在，需加override
    // void tanh_inplace(Tensor& tensor_a) override;  // 已存在，需加override
    // void tanh_into(const Tensor& tensor_a, Tensor& result) override;  // 已存在，需加override
    // Tensor dtanh(const Tensor& tensor_a) override;  // 已存在，需加override
    // void dtanh_inplace(Tensor& tensor_a) override;  // 已存在，需加override
    // void dtanh_into(const Tensor& tensor_a, Tensor& result) override;  // 已存在，需加override

    // 交叉熵损失函数
    // float crossentropy(const Tensor& pred, const Tensor& label, std::string reduction = "mean") override;  // 已存在，需加override

    // One-hot编码操作
    // Tensor one_hot(const Tensor& label, int32_t num_classes, float label_smoothing = 0.0f) override;  // 已存在，需加override
    // void one_hot_into(const Tensor& label, Tensor& result, int32_t num_classes, float label_smoothing = 0.0f) override;  // 已存在，需加override

private:
    void validate_same_device(const Device& device) const;
    void validate_tensor_shape(const Tensor& a, const Tensor& b) const;
};

} // namespace tr
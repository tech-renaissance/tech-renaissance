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
#define EXPORT_TENSOR dynamic_cast<CpuBackend*>(BackendManager::instance().get_backend(CPU).get())->export_tensor
#define IMPORT_TENSOR dynamic_cast<CpuBackend*>(BackendManager::instance().get_backend(CPU).get())->import_tensor


namespace tr {

class CpuBackend : public Backend {
public:
    CpuBackend();
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

    // 基本运算
    void add(Tensor& result, const Tensor& a, const Tensor& b) override;
    void mul(Tensor& result, const Tensor& a, const Tensor& b) override;
    void mm(Tensor& result, const Tensor& tensor_a, const Tensor& tensor_b) override;

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
    Tensor square(const Tensor& input) const;
    Tensor sqrt(const Tensor& input) const;
    Tensor abs(const Tensor& input) const;
    Tensor negative(const Tensor& input) const;
    Tensor reciprocal(const Tensor& input) const;
    Tensor round(const Tensor& input) const;
    Tensor transpose(const Tensor& input) const;

    // 原地运算
    void zeros_inplace(Tensor& input) const;
    void ones_inplace(Tensor& input) const;
    void relu_inplace(Tensor& input) const;
    void sign_inplace(Tensor& input) const;
    void square_inplace(Tensor& input) const;
    void sqrt_inplace(Tensor& input) const;
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
    void square_into(const Tensor& input, Tensor& output) const;
    void sqrt_into(const Tensor& input, Tensor& output) const;
    void abs_into(const Tensor& input, Tensor& output) const;
    void negative_into(const Tensor& input, Tensor& output) const;
    void reciprocal_into(const Tensor& input, Tensor& output) const;
    void round_into(const Tensor& input, Tensor& output) const;
    void transpose_into(const Tensor& input, Tensor& output) const;

    // 张量复制操作（V1.26.5新增）
    Tensor copy(const Tensor& tensor) const override;
    void copy_into(const Tensor& src, Tensor& dst) const override;

    // 标量运算（V1.28.1新增）
    // 乘法：tensor * scalar
    Tensor mul(const Tensor& input, float scalar) const;
    void mul_inplace(Tensor& input, float scalar) const;
    void mul_into(const Tensor& input, float scalar, Tensor& output) const;

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

private:
    void validate_same_device(const Device& device) const;
    void validate_tensor_shape(const Tensor& a, const Tensor& b) const;
};

} // namespace tr
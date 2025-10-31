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
    void copy(void* dst, const void* src, size_t size,
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

private:
    void validate_same_device(const Device& device) const;
    void validate_tensor_shape(const Tensor& a, const Tensor& b) const;
};

} // namespace tr
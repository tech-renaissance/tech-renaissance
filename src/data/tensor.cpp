/**
 * @file tensor.cpp
 * @brief 张量类实现
 * @details 实现张量类的核心逻辑，包括元数据管理、数据移动和视图操作等
 * @version 1.00.00
 * @date 2025-10-25
 * @author 技术觉醒团队
 * @note 依赖项: tensor.h, storage.h, backend_manager.h, dtype.h
 * @note 所属系列: data
 */

#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/data/storage.h"
#include "tech_renaissance/backend/backend.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/utils/tr_exception.h"
#include <sstream>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <vector>
#include <mutex>

namespace tr {

Tensor::Tensor()
    : shape_(), dtype_(DType::FP32), device_(tr::CPU),
      storage_(nullptr), offset_(0) {
}

Tensor::Tensor(const Shape& shape, DType dtype, const Device& device)
    : shape_(shape), dtype_(dtype), device_(device), storage_(nullptr), offset_(0) {

    // 验证形状和数据类型
    validate_shape_dtype();

    // Tensor构造函数不再直接分配内存
    // 这是轻量级构造函数，仅创建元数据容器
    // 内存分配应通过静态工厂方法或Backend的显式接口完成

    // 构造函数仅创建元数据容器，不分配内存
    // 内存分配通过静态工厂方法完成
}

const Shape& Tensor::shape() const noexcept {
    return shape_;
}

DType Tensor::dtype() const noexcept {
    return dtype_;
}

Device Tensor::device() const noexcept {
    return device_;
}

int32_t Tensor::ndim() const noexcept {
    return shape_.ndim();
}

int64_t Tensor::numel() const noexcept {
    return shape_.numel();
}

int32_t Tensor::dim_size(int32_t dim) const {
    return shape_.dim(dim);
}

int32_t Tensor::batch() const noexcept {
    return shape_.n();
}

int32_t Tensor::channel() const noexcept {
    return shape_.c();
}

int32_t Tensor::height() const noexcept {
    return shape_.h();
}

int32_t Tensor::width() const noexcept {
    return shape_.w();
}

size_t Tensor::dtype_size() const noexcept {
    return ::tr::dtype_size(dtype_);
}

std::shared_ptr<Storage> Tensor::storage() const noexcept {
    return storage_;
}

bool Tensor::is_empty() const noexcept {
    return storage_ == nullptr || storage_->is_empty();
}

bool Tensor::is_scalar() const noexcept {
    return shape_.is_scalar();
}

bool Tensor::is_contiguous() const noexcept {
    // 第一期实现中，所有Tensor都是连续存储的
    return true;
}

Tensor Tensor::to(const Device& device) const {
    Device current_device = this->device();
    if (current_device == device) {
        return clone(); // 同设备直接克隆
    }

    auto src_backend = BackendManager::instance().get_backend(current_device);
    auto dst_backend = BackendManager::instance().get_backend(device);

    Tensor result(shape_, dtype_, device);
    size_t size_bytes = memory_size();

    if (!is_empty()) {
        try {
            // 由目标backend分配内存
            auto holder = dst_backend->allocate(size_bytes);
            result.storage_ = std::make_shared<Storage>(size_bytes, device);
            result.storage_->set_data_ptr(dst_backend->get_data_ptr(holder), holder);

            // 由目标backend执行跨设备拷贝
            dst_backend->copy(result.data_ptr(), data_ptr(), size_bytes, device, current_device);
        } catch (const std::exception& e) {
            throw TRException("[Tensor::to] Failed to copy Tensor to device: " + std::string(e.what()));
        }
    }

    return result;
}

Tensor Tensor::cpu() const {
    return to(tr::CPU);
}

Tensor Tensor::cuda(int device_id) const {
    if (device_id < 0 || device_id >= 8) {
        throw TRException("[Tensor::cuda] CUDA device ID must be between 0 and 7");
    }
    return to(tr::CUDA[device_id]);
}

Tensor Tensor::clone() const {
    Device current_device = this->device();
    Tensor result(shape_, dtype_, current_device);

    if (!is_empty()) {
        try {
            auto backend = get_backend();
            void* dst_ptr = result.data_ptr();
            const void* src_ptr = data_ptr();
            size_t size = memory_size();

            // 在相同设备间拷贝数据
            backend->copy(dst_ptr, src_ptr, size, current_device, current_device);
        } catch (const std::exception& e) {
            throw TRException("Failed to clone Tensor: " + std::string(e.what()));
        }
    }

    return result;
}

Tensor Tensor::view() const {
    // 创建共享Storage的新Tensor
    Tensor result;
    result.shape_ = shape_;
    result.dtype_ = dtype_;
    result.storage_ = storage_;
    result.offset_ = offset_;

    return result;
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    // 检查元素总数是否匹配
    if (shape_.numel() != new_shape.numel()) {
        throw TRException("[Tensor::reshape] Element count mismatch (" +
                                   std::to_string(shape_.numel()) + " vs " +
                                   std::to_string(new_shape.numel()) + ")");
    }

    // 创建新的Tensor（共享Storage）
    Tensor result;
    result.shape_ = new_shape;
    result.dtype_ = dtype_;
    result.storage_ = storage_;
    result.offset_ = offset_;

    return result;
}

Tensor Tensor::expand_dim(int32_t dim) const {
    if (dim < 0 || dim > ndim()) {
        throw TRException("[Tensor::expand_dim] Dimension index out of range");
    }

    // 构建新的形状数组
    std::vector<int32_t> new_dims;

    // 在指定位置插入1
    for (int32_t i = 0; i < ndim(); ++i) {
        if (i == dim) {
            new_dims.push_back(1);
        }
        new_dims.push_back(dim_size(i));
    }

    // 如果要在最后添加维度
    if (dim == ndim()) {
        new_dims.push_back(1);
    }

    // 创建新形状
    Shape new_shape;
    if (new_dims.size() == 1) {
        new_shape = Shape(new_dims[0]);
    } else if (new_dims.size() == 2) {
        new_shape = Shape(new_dims[0], new_dims[1]);
    } else if (new_dims.size() == 3) {
        new_shape = Shape(new_dims[0], new_dims[1], new_dims[2]);
    } else if (new_dims.size() == 4) {
        new_shape = Shape(new_dims[0], new_dims[1], new_dims[2], new_dims[3]);
    }

    return reshape(new_shape);
}

Tensor Tensor::squeeze_dim(int32_t dim) const {
    if (dim < 0 || dim >= ndim()) {
        throw TRException("[Tensor::squeeze_dim] Dimension index out of range");
    }

    // 检查指定维度是否为1
    if (dim_size(dim) != 1) {
        throw TRException("[Tensor::squeeze_dim] Dimension size is not 1");
    }

    // 构建新的形状数组
    std::vector<int32_t> new_dims;

    for (int32_t i = 0; i < ndim(); ++i) {
        if (i != dim) {
            new_dims.push_back(dim_size(i));
        }
    }

    // 创建新形状
    Shape new_shape;
    if (new_dims.empty()) {
        // 标量
        new_shape = Shape();
    } else if (new_dims.size() == 1) {
        new_shape = Shape(new_dims[0]);
    } else if (new_dims.size() == 2) {
        new_shape = Shape(new_dims[0], new_dims[1]);
    } else if (new_dims.size() == 3) {
        new_shape = Shape(new_dims[0], new_dims[1], new_dims[2]);
    } else if (new_dims.size() == 4) {
        new_shape = Shape(new_dims[0], new_dims[1], new_dims[2], new_dims[3]);
    }

    return reshape(new_shape);
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    Device current_device = this->device();
    oss << "Tensor(shape=" << shape_.to_string()
        << ", dtype=" << (dtype_ == DType::FP32 ? "FP32" : "INT8")
        << ", device=" << current_device.to_string();

    if (is_empty()) {
        oss << ", empty";
    } else {
        oss << ", numel=" << numel();

        // 显示张量数据内容（类似print()但不显示名称）
        if (numel() <= 16) {  // 只对小张量显示完整内容
            oss << ", data=";
            format_tensor_content(oss, 4);  // 使用默认4位精度
        } else {  // 大张量只显示摘要信息，不读取具体数据
            oss << ", data=[...large tensor, use print() for details...]";
        }
    }

    oss << ")";
    return oss.str();
}

void Tensor::print(const std::string& name) const {
    // 默认精度为4位小数（与PyTorch默认一致）
    print(name, 4);
}

void Tensor::print(const std::string& name, int precision) const {
    std::ostringstream oss;

    // 打印张量名称（如果提供）
    if (!name.empty()) {
        oss << name << ":\n";
    }

    // 打印张量内容（PyTorch风格）
    oss << "tensor(";

    if (is_empty()) {
        oss << "[]";
    } else {
        // 张量格式化打印
        format_tensor_content(oss, precision);
    }

    // 打印设备和数据类型信息（仅在需要时显示）
    Device current_device = this->device();

    // 只有非CPU设备才显示device信息
    if (!current_device.is_cpu()) {
        oss << ", device='cuda:" << current_device.index << "'";
    }

    // 只有INT8类型才显示dtype信息（FP32是默认的）
    if (dtype_ == DType::INT8) {
        oss << ", dtype=INT8";
    }

    oss << ")";

    std::cout << oss.str() << std::endl;
}

void Tensor::summary(const std::string& name) const {
    std::ostringstream oss;

    if (!name.empty()) {
        oss << name << ": ";
    }

    oss << to_string();
    oss << ", memory_size=" << memory_size() << " bytes";

    std::cout << oss.str() << std::endl;
}

void Tensor::from_cpu_data(const void* data, size_t size) {
    if (!data) {
        throw TRException("[Tensor::from_cpu_data] Input data pointer is null");
    }

    size_t required_size = memory_size();
    if (size != required_size) {
        throw TRException("[Tensor::from_cpu_data] Data size mismatch: expected " +
                                   std::to_string(required_size) +
                                   " bytes, got " + std::to_string(size) + " bytes");
    }

    if (is_empty()) {
        throw TRException("[Tensor::from_cpu_data] Cannot copy data to empty Tensor");
    }

    try {
        auto backend = get_backend();
        void* dst_ptr = data_ptr();

        // 从CPU拷贝到目标设备
        backend->copy(dst_ptr, data, size, this->device(), tr::CPU);
    } catch (const std::exception& e) {
        throw TRException("Failed to copy data from CPU: " + std::string(e.what()));
    }
}

void Tensor::to_cpu_data(void* data, size_t size) const {
    if (!data) {
        throw TRException("[Tensor::to_cpu_data] Output data pointer is null");
    }

    size_t required_size = memory_size();
    if (size != required_size) {
        throw TRException("[Tensor::to_cpu_data] Data size mismatch: expected " +
                                   std::to_string(required_size) +
                                   " bytes, got " + std::to_string(size) + " bytes");
    }

    if (is_empty()) {
        throw TRException("[Tensor::to_cpu_data] Cannot copy data from empty Tensor");
    }

    try {
        auto backend = get_backend();
        const void* src_ptr = data_ptr();

        // 从目标设备拷贝到CPU
        backend->copy(data, src_ptr, size, tr::CPU, this->device());
    } catch (const std::exception& e) {
        throw TRException("Failed to copy data to CPU: " + std::string(e.what()));
    }
}

void* Tensor::data_ptr() noexcept {
    return storage_ ? storage_->data_ptr() : nullptr;
}

const void* Tensor::data_ptr() const noexcept {
    return storage_ ? storage_->data_ptr() : nullptr;
}

void Tensor::validate_shape_dtype() const {
    // 检查数据类型是否有效
    if (dtype_ != DType::FP32 && dtype_ != DType::INT8) {
        throw TRException("[Tensor::validate_shape_dtype] Tensor dtype must be FP32 or INT8");
    }

    // 检查元素总数是否合理
    if (numel() < 0) {
        throw TRException("[Tensor::validate_shape_dtype] Invalid Tensor: negative element count");
    }

    // 检查内存大小是否溢出
    size_t total_size = static_cast<size_t>(numel()) * dtype_size();
    if (total_size > SIZE_MAX / 2) { // 留一些安全边界
        throw TRException("[Tensor::validate_shape_dtype] Tensor memory size too large");
    }
}

size_t Tensor::memory_size() const noexcept {
    int64_t elements = numel();
    size_t type_size = dtype_size();

    // 溢出检测：确保numel() * dtype_size()不会溢出size_t
    if (elements < 0) {
        return 0; // 无效元素数量，返回0
    }

    // 检查是否会导致size_t溢出
    if (static_cast<uint64_t>(elements) > SIZE_MAX / type_size) {
        return SIZE_MAX; // 返回最大值表示溢出
    }

    return static_cast<size_t>(elements) * type_size;
}


std::shared_ptr<Backend> Tensor::get_backend() const {
    // 直接获取管理器实例，并请求对应设备的后端
    // 这不会再触发任何初始化，只是简单的查找操作
    auto& manager = BackendManager::instance();
    return manager.get_backend(this->device());
}


// ===== 私有辅助函数实现 =====

Tensor Tensor::create_and_allocate(const Shape& shape, DType dtype, const Device& device) {
    Tensor result(shape, dtype, device);

    // 获取后端并分配内存
    auto& manager = BackendManager::instance();
    auto backend = manager.get_backend(device);

    // 分配内存
    auto memory_holder = backend->allocate(result.numel() * result.dtype_size());
    result.storage_ = std::make_shared<Storage>(result.numel() * result.dtype_size(), result.device());
    result.storage_->set_data_ptr(backend->get_data_ptr(memory_holder), memory_holder);

    return result;
}

// ===== 静态工厂方法实现 =====

Tensor Tensor::zeros(const Shape& shape, DType dtype, const Device& device) {
    Tensor result = create_and_allocate(shape, dtype, device);

    // 填充为0
    auto backend = result.get_backend();
    if (dtype == DType::FP32) {
        backend->fill(result, 0.0f);
    } else if (dtype == DType::INT8) {
        backend->fill(result, static_cast<int8_t>(0));
    }

    return result;
}

Tensor Tensor::ones(const Shape& shape, DType dtype, const Device& device) {
    Tensor result = create_and_allocate(shape, dtype, device);

    // 填充为1
    auto backend = result.get_backend();
    if (dtype == DType::FP32) {
        backend->fill(result, 1.0f);
    } else if (dtype == DType::INT8) {
        backend->fill(result, static_cast<int8_t>(1));
    }

    return result;
}

Tensor Tensor::full(const Shape& shape, float value, DType dtype, const Device& device) {
    Tensor result = create_and_allocate(shape, dtype, device);

    // 填充指定值
    auto backend = result.get_backend();
    if (dtype == DType::FP32) {
        backend->fill(result, value);
    } else if (dtype == DType::INT8) {
        backend->fill(result, static_cast<int8_t>(value));
    }

    return result;
}

Tensor Tensor::empty(const Shape& shape, DType dtype, const Device& device) {
    return create_and_allocate(shape, dtype, device);
}

// ===== 随机数生成静态工厂方法实现 =====

Tensor Tensor::randn(const Shape& shape, unsigned int seed,
                     DType dtype, const Device& device) {
    if (dtype != DType::FP32) {
        throw TRException("randn only supports FP32 data type");
    }

    Tensor result = empty(shape, dtype, device);

    // 在CPU上生成随机数据，然后如果需要再移动到目标设备
    Tensor cpu_tensor = result.device().is_cpu() ? result : empty(shape, dtype, tr::CPU);

    // 使用C++11随机数生成器
    std::mt19937 engine(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float* data = static_cast<float*>(cpu_tensor.data_ptr());
    int64_t numel = cpu_tensor.numel();

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist(engine);
    }

    // 如果目标设备不是CPU，转移数据
    if (!result.device().is_cpu()) {
        auto backend = BackendManager::get_backend_static(device);
        result = backend->from_cpu(cpu_tensor);
    }

    return result;
}

Tensor Tensor::uniform(const Shape& shape, float min_val, float max_val,
                      unsigned int seed, DType dtype, const Device& device) {
    if (dtype != DType::FP32) {
        throw TRException("uniform only supports FP32 data type");
    }

    Tensor result = empty(shape, dtype, device);

    // 在CPU上生成随机数据，然后如果需要再移动到目标设备
    Tensor cpu_tensor = result.device().is_cpu() ? result : empty(shape, dtype, tr::CPU);

    // 使用C++11随机数生成器
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);

    float* data = static_cast<float*>(cpu_tensor.data_ptr());
    int64_t numel = cpu_tensor.numel();

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist(engine);
    }

    // 如果目标设备不是CPU，转移数据
    if (!result.device().is_cpu()) {
        auto backend = BackendManager::get_backend_static(device);
        result = backend->from_cpu(cpu_tensor);
    }

    return result;
}

Tensor Tensor::randint(int low, int high, const Shape& shape,
                      unsigned int seed, const Device& device) {
    if (low >= high) {
        throw TRException("randint: low must be less than high");
    }

    Tensor result = empty(shape, DType::FP32, device);

    // 在CPU上生成随机数据，然后如果需要再移动到目标设备
    Tensor cpu_tensor = result.device().is_cpu() ? result : empty(shape, DType::FP32, tr::CPU);

    // 使用C++11随机数生成器
    std::mt19937 engine(seed);
    std::uniform_int_distribution<int> dist(low, high - 1);

    float* data = static_cast<float*>(cpu_tensor.data_ptr());
    int64_t numel = cpu_tensor.numel();

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = static_cast<float>(dist(engine));
    }

    // 如果目标设备不是CPU，转移数据
    if (!result.device().is_cpu()) {
        auto backend = BackendManager::get_backend_static(device);
        result = backend->from_cpu(cpu_tensor);
    }

    return result;
}

bool Tensor::operator==(const Tensor& other) const noexcept {
    return shape_ == other.shape_ &&
           dtype_ == other.dtype_ &&
           device() == other.device() &&
           storage_ == other.storage_ &&
           offset_ == other.offset_;
}

bool Tensor::operator!=(const Tensor& other) const noexcept {
    return !(*this == other);
}

void Tensor::format_tensor_content(std::ostringstream& oss, int precision) const {
    if (is_empty()) {
        oss << "[]";
        return;
    }

    // 获取数据到CPU
    std::vector<float> data(numel());
    std::vector<int8_t> int8_data(numel());

    try {
        if (dtype_ == DType::FP32) {
            to_cpu_data(data.data(), data.size() * sizeof(float));
        } else if (dtype_ == DType::INT8) {
            to_cpu_data(int8_data.data(), int8_data.size() * sizeof(int8_t));
        }
    } catch (...) {
        oss << "[...data unavailable...]";
        return;
    }

    // 根据维度格式化输出
    if (ndim() == 0) {
        // 标量
        if (dtype_ == DType::FP32) {
            oss << std::fixed << std::setprecision(precision) << data[0];
        } else {
            oss << static_cast<int>(int8_data[0]);
        }
    } else if (ndim() == 1) {
        // 1D张量
        oss << "[";
        for (int32_t i = 0; i < dim_size(0); ++i) {
            if (i > 0) oss << ", ";
            if (dtype_ == DType::FP32) {
                oss << std::fixed << std::setprecision(precision) << data[i];
            } else {
                oss << static_cast<int>(int8_data[i]);
            }
        }
        oss << "]";
    } else if (ndim() == 2) {
        // 2D张量
        oss << "[" << std::endl;
        for (int32_t i = 0; i < dim_size(0); ++i) {
            oss << "  [";
            for (int32_t j = 0; j < dim_size(1); ++j) {
                if (j > 0) oss << ", ";
                int32_t idx = i * dim_size(1) + j;
                if (dtype_ == DType::FP32) {
                    oss << std::fixed << std::setprecision(precision) << data[idx];
                } else {
                    oss << static_cast<int>(int8_data[idx]);
                }
            }
            oss << "]";
            if (i < dim_size(0) - 1) oss << ",";
            oss << std::endl;
        }
        oss << "]";
    } else if (ndim() == 3) {
        // 3D张量
        oss << "[" << std::endl;
        for (int32_t i = 0; i < dim_size(0); ++i) {
            oss << "  [";
            for (int32_t j = 0; j < dim_size(1); ++j) {
                if (j > 0) oss << std::endl << "   ";
                oss << "[";
                for (int32_t k = 0; k < dim_size(2); ++k) {
                    if (k > 0) oss << ", ";
                    int32_t idx = i * dim_size(1) * dim_size(2) + j * dim_size(2) + k;
                    if (dtype_ == DType::FP32) {
                        oss << std::fixed << std::setprecision(precision) << data[idx];
                    } else {
                        oss << static_cast<int>(int8_data[idx]);
                    }
                }
                oss << "]";
            }
            oss << "]";
            if (i < dim_size(0) - 1) oss << ",";
            oss << std::endl;
        }
        oss << "]";
    } else if (ndim() == 4) {
        // 4D张量 (NCHW格式) - 完全匹配PyTorch风格
        int32_t N = dim_size(0);  // Batch
        int32_t C = dim_size(1);  // Channel
        int32_t H = dim_size(2);  // Height
        int32_t W = dim_size(3);  // Width

        oss << "[" << std::endl;
        for (int32_t n = 0; n < N; ++n) {
            oss << " [";
            for (int32_t c = 0; c < C; ++c) {
                if (c > 0) oss << std::endl << " ";
                oss << "[";
                for (int32_t h = 0; h < H; ++h) {
                    if (h > 0) oss << std::endl << "  ";
                    oss << "[";
                    for (int32_t w = 0; w < W; ++w) {
                        if (w > 0) oss << ", ";
                        // NCHW到线性索引的转换
                        int32_t idx = n * C * H * W + c * H * W + h * W + w;
                        if (dtype_ == DType::FP32) {
                            oss << std::fixed << std::setprecision(precision) << data[idx];
                        } else {
                            oss << static_cast<int>(int8_data[idx]);
                        }
                    }
                    oss << "]";
                    if (h < H - 1) oss << ",";
                }
                oss << "]";
            }
            oss << "]";
            if (n < N - 1) {
                oss << "," << std::endl << std::endl;
            } else {
                oss << std::endl;
            }
        }
        oss << "]";
    } else {
        // 更高维度暂时不支持
        oss << "[...unsupported dimensions...]";
    }
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.to_string();
    return os;
}

} // namespace tr
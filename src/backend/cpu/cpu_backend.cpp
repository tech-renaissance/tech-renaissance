/**
 * @file cpu_backend.cpp
 * @brief CPU后端类实现
 * @details 基于Eigen库实现高性能CPU计算
 * @version 1.00.00
 * @date 2025-10-26
 * @author 技术觉醒团队
 * @note 依赖项: tensor.h, Eigen
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/utils/logger.h"

#ifdef TR_USE_EIGEN
#include "Core"
#endif

#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <filesystem>

namespace tr {

CpuBackend::CpuBackend() {
    // 确保workspace目录存在
    namespace fs = std::filesystem;
    std::string workspace_path = WORKSPACE_PATH;

    try {
        if (!fs::exists(workspace_path)) {
            fs::create_directories(workspace_path);
            Logger::get_instance().info("Created workspace directory: " + workspace_path);
        } else {
            Logger::get_instance().debug("Workspace directory already exists: " + workspace_path);
        }
    } catch (const std::exception& e) {
        Logger::get_instance().warn("Failed to create workspace directory: " + std::string(e.what()));
        // 不抛出异常，继续初始化CPU后端
    }

    Logger::get_instance().info("CPU backend initialized");
}

std::shared_ptr<void> CpuBackend::allocate(size_t size) {
    if (size == 0) {
        throw TRException("Cannot allocate zero bytes");
    }

    // 分配对齐内存（64字节对齐，优化SIMD访问）
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, 64);
#else
    if (posix_memalign(&ptr, 64, size) != 0) {
        ptr = nullptr;
    }
#endif

    if (!ptr) {
        throw TRException("CPU memory allocation failed: " + std::to_string(size) + " bytes");
    }

    // 创建智能指针，自定义删除器
    return std::shared_ptr<void>(ptr, [](void* p) {
#ifdef _WIN32
        _aligned_free(p);
#else
        free(p);
#endif
    });
}

void CpuBackend::deallocate(void* ptr) {
    if (ptr) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

void* CpuBackend::get_data_ptr(const std::shared_ptr<void>& holder) {
    return holder.get();
}

void CpuBackend::copy_data(void* dst, const void* src, size_t size,
                           const Device& dst_device, const Device& src_device) const {
    if (!dst || !src) {
        throw TRException("Null pointer in copy operation");
    }

    // CPU只支持CPU->CPU拷贝
    if (!dst_device.is_cpu() || !src_device.is_cpu()) {
        throw TRException("CpuBackend only supports CPU<->CPU copy");
    }

    std::memcpy(dst, src, size);
}

void CpuBackend::fill(Tensor& dst, float value) {
    validate_same_device(dst.device());

    // 新增：检查Storage是否已分配
    if (dst.is_empty()) {
        throw TRException("[CpuBackend::fill] Target tensor has no allocated Storage");
    }

    if (dst.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::fill] fill(float) requires FP32 tensor");
    }

    float* data = static_cast<float*>(dst.data_ptr());
    size_t count = dst.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<Eigen::VectorXf>(data, count).setConstant(value);
#else
    std::fill_n(data, count, value);
#endif
}

void CpuBackend::fill(Tensor& dst, int8_t value) {
    validate_same_device(dst.device());

    // 新增：检查Storage是否已分配
    if (dst.is_empty()) {
        throw TRException("[CpuBackend::fill] Target tensor has no allocated Storage");
    }

    if (dst.dtype() != DType::INT8) {
        throw TRException("[CpuBackend::fill] fill(int8_t) requires INT8 tensor");
    }

    int8_t* data = static_cast<int8_t*>(dst.data_ptr());
    size_t count = dst.numel();
    std::fill_n(data, count, value);
}

void CpuBackend::add(Tensor& result, const Tensor& a, const Tensor& b) {
    validate_same_device(a.device());
    validate_same_device(b.device());
    validate_same_device(result.device());
    validate_tensor_shape(a, b);
    validate_tensor_shape(a, result);

    // 新增：检查Storage是否已分配
    if (result.is_empty()) {
        throw TRException("[CpuBackend::add] Result tensor has no allocated Storage");
    }
    if (a.is_empty()) {
        throw TRException("[CpuBackend::add] Input tensor 'a' has no allocated Storage");
    }
    if (b.is_empty()) {
        throw TRException("[CpuBackend::add] Input tensor 'b' has no allocated Storage");
    }

    if (a.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::add] add only supports FP32 in first phase");
    }

    const float* a_data = static_cast<const float*>(a.data_ptr());
    const float* b_data = static_cast<const float*>(b.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());
    size_t count = a.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> a_vec(a_data, count);
    Eigen::Map<const Eigen::VectorXf> b_vec(b_data, count);
    Eigen::Map<Eigen::VectorXf> result_vec(result_data, count);
    result_vec = a_vec + b_vec;
#else
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = a_data[i] + b_data[i];
    }
#endif
}

void CpuBackend::mul(Tensor& result, const Tensor& a, const Tensor& b) {
    validate_same_device(a.device());
    validate_same_device(b.device());
    validate_same_device(result.device());
    validate_tensor_shape(a, b);
    validate_tensor_shape(a, result);

    // 新增：检查Storage是否已分配
    if (result.is_empty()) {
        throw TRException("[CpuBackend::mul] Result tensor has no allocated Storage");
    }
    if (a.is_empty()) {
        throw TRException("[CpuBackend::mul] Input tensor 'a' has no allocated Storage");
    }
    if (b.is_empty()) {
        throw TRException("[CpuBackend::mul] Input tensor 'b' has no allocated Storage");
    }

    if (a.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::mul] mul only supports FP32 in first phase");
    }

    const float* a_data = static_cast<const float*>(a.data_ptr());
    const float* b_data = static_cast<const float*>(b.data_ptr());
    float* result_data = static_cast<float*>(result.data_ptr());
    size_t count = a.numel();

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> a_vec(a_data, count);
    Eigen::Map<const Eigen::VectorXf> b_vec(b_data, count);
    Eigen::Map<Eigen::VectorXf> result_vec(result_data, count);
    result_vec = a_vec.cwiseProduct(b_vec);
#else
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = a_data[i] * b_data[i];
    }
#endif
}

void CpuBackend::validate_same_device(const Device& device) const {
    if (!device.is_cpu()) {
        throw TRException("CpuBackend: tensor must be on CPU device");
    }
}

void CpuBackend::validate_tensor_shape(const Tensor& a, const Tensor& b) const {
    if (a.shape() != b.shape()) {
        throw TRException("Tensor shape mismatch: " +
                         a.shape().to_string() + " vs " +
                         b.shape().to_string());
    }
}

// ===== 数据访问实现 =====

float CpuBackend::get_scalar_float(const Tensor& tensor) {
    if (!tensor.is_scalar()) {
        throw TRException("get_scalar_float: tensor must be scalar");
    }

    const void* data = tensor.data_ptr();
    if (!data) {
        throw TRException("get_scalar_float: tensor data is null");
    }

    return *static_cast<const float*>(data);
}

int32_t CpuBackend::get_scalar_int32(const Tensor& tensor) {
    if (!tensor.is_scalar()) {
        throw TRException("get_scalar_int32: tensor must be scalar");
    }

    const void* data = tensor.data_ptr();
    if (!data) {
        throw TRException("get_scalar_int32: tensor data is null");
    }

    switch (tensor.dtype()) {
        case DType::FP32:
            return static_cast<int32_t>(*static_cast<const float*>(data));
        case DType::INT8:
            return static_cast<int32_t>(*static_cast<const int8_t*>(data));
        default:
            throw TRException("get_scalar_int32: unsupported dtype");
    }
}

int8_t CpuBackend::get_scalar_int8(const Tensor& tensor) {
    if (!tensor.is_scalar()) {
        throw TRException("get_scalar_int8: tensor must be scalar");
    }

    const void* data = tensor.data_ptr();
    if (!data) {
        throw TRException("get_scalar_int8: tensor data is null");
    }

    switch (tensor.dtype()) {
        case DType::FP32:
            return static_cast<int8_t>(*static_cast<const float*>(data));
        case DType::INT8:
            return *static_cast<const int8_t*>(data);
        default:
            throw TRException("get_scalar_int8: unsupported dtype");
    }
}

// ===== 张量IO算子实现（CPU后端独有功能） =====

void CpuBackend::export_tensor(const Tensor& tensor, const std::string& filename) const {
    try {
        // 验证设备类型
        validate_same_device(tensor.device());

        // 验证张量
        if (tensor.dtype() != DType::FP32 && tensor.dtype() != DType::INT8) {
            throw TRException("Tensor export only supports FP32 and INT8 data types. "
                             "Current dtype: " + std::to_string(static_cast<int>(tensor.dtype())));
        }

        if (tensor.ndim() > 4) {
            throw TRException("Tensor export only supports up to 4D tensors. "
                             "Current ndim: " + std::to_string(tensor.ndim()));
        }

        if (tensor.is_empty()) {
            throw TRException("Cannot export empty tensor (no allocated storage)");
        }

        // 打开文件
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw TRException("Failed to open file for writing: " + filename);
        }

        Logger::get_instance().info("Exporting tensor to " + filename);

        // 写入文件头
        constexpr char MAGIC_NUMBER[4] = {'T', 'S', 'R', '!'};
        constexpr int32_t FORMAT_VERSION = 1;
        constexpr int32_t HEADER_SIZE = 64;

        file.write(MAGIC_NUMBER, 4);
        int32_t version = FORMAT_VERSION;
        file.write(reinterpret_cast<const char*>(&version), sizeof(int32_t));
        int32_t header_size = HEADER_SIZE;
        file.write(reinterpret_cast<const char*>(&header_size), sizeof(int32_t));
        int32_t reserved = 0;
        file.write(reinterpret_cast<const char*>(&reserved), sizeof(int32_t));

        // 写入元数据
        int32_t dtype_value = static_cast<int32_t>(tensor.dtype());
        file.write(reinterpret_cast<const char*>(&dtype_value), sizeof(int32_t));

        int32_t ndim = tensor.ndim();
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(int32_t));

        // 各维度大小 (16字节，按NCHW顺序)
        int32_t dims[4];
        for (int i = 0; i < 4; i++) {
            if (i < 4 - ndim) {
                dims[i] = 1;  // 前导维度为1
            } else {
                dims[i] = tensor.dim_size(i - (4 - ndim));
            }
        }
        file.write(reinterpret_cast<const char*>(dims), sizeof(dims));

        // 元素总数 (8字节)
        int64_t total_elements = static_cast<int64_t>(tensor.numel());
        file.write(reinterpret_cast<const char*>(&total_elements), sizeof(int64_t));

        // 保留字段 (8字节)
        int64_t reserved_0 = 0;
        int64_t reserved_1 = 0;
        file.write(reinterpret_cast<const char*>(&reserved_0), sizeof(int64_t));
        file.write(reinterpret_cast<const char*>(&reserved_1), sizeof(int64_t));

        // 写入数据
        size_t data_size = tensor.memory_size();
        const void* data_ptr = tensor.data_ptr();  // CpuBackend是友元，可以访问

        
        if (!data_ptr) {
            throw TRException("Tensor data pointer is null");
        }

        file.write(static_cast<const char*>(data_ptr), data_size);

        file.close();

        Logger::get_instance().info("Tensor exported successfully to " + filename);

    } catch (const std::exception& e) {
        throw TRException("Tensor export failed: " + std::string(e.what()));
    }
}

Tensor CpuBackend::import_tensor(const std::string& filename) const {
    try {
        // 打开文件
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw TRException("Failed to open file for reading: " + filename);
        }

        Logger::get_instance().info("Importing tensor from " + filename);

        // 读取文件头（64字节）
        struct TSRHeader {
            char magic[4];        // 魔数 'TSR!'
            int32_t version;     // 格式版本
            int32_t header_size; // 头部大小
            int32_t reserved;    // 保留字段

            // 元数据块
            int32_t dtype;       // 数据类型
            int32_t ndim;        // 维度数量
            int32_t dims[4];     // 各维度大小 (NCHW)
            int64_t total_elements; // 元素总数
            int64_t reserved_0;  // 保留字段
            int64_t reserved_1;  // 保留字段
        } header;

        file.read(reinterpret_cast<char*>(&header), sizeof(TSRHeader));

        if (file.gcount() != sizeof(TSRHeader)) {
            throw TRException("File too small to contain valid TSR header: " + filename);
        }

        // 验证魔数
        constexpr char MAGIC_NUMBER[4] = {'T', 'S', 'R', '!'};
        if (std::memcmp(header.magic, MAGIC_NUMBER, 4) != 0) {
            throw TRException("Invalid TSR file magic number. Expected 'TSR!', got: " +
                             std::string(header.magic, 4));
        }

        // 验证版本
        if (header.version != 1) {
            throw TRException("Unsupported TSR format version: " + std::to_string(header.version) +
                             ". Supported version: 1");
        }

        // 验证数据类型
        if (header.dtype != static_cast<int32_t>(DType::FP32) &&
            header.dtype != static_cast<int32_t>(DType::INT8)) {
            throw TRException("Unsupported dtype value: " + std::to_string(header.dtype) +
                             ". Supported values: " + std::to_string(static_cast<int>(DType::FP32)) +
                             " (FP32), " + std::to_string(static_cast<int>(DType::INT8)) + " (INT8)");
        }

        // 验证维度数量
        if (header.ndim < 0 || header.ndim > 4) {
            throw TRException("Invalid ndim value: " + std::to_string(header.ndim) +
                             ". Must be between 0 and 4");
        }

        // 验证维度一致性
        int64_t calculated_elements = static_cast<int64_t>(header.dims[0]) *
                                   static_cast<int64_t>(header.dims[1]) *
                                   static_cast<int64_t>(header.dims[2]) *
                                   static_cast<int64_t>(header.dims[3]);

        if (header.total_elements != calculated_elements) {
            throw TRException("Element count mismatch. Header claims " +
                             std::to_string(header.total_elements) +
                             " but dimensions suggest " + std::to_string(calculated_elements));
        }

        // 验证各维度非负
        for (int i = 0; i < 4; i++) {
            if (header.dims[i] < 0) {
                throw TRException("Invalid dimension value at index " + std::to_string(i) +
                                 ": " + std::to_string(header.dims[i]));
            }
        }

        // 验证文件大小
        file.seekg(0, std::ios::end);
        std::streampos file_size = file.tellg();
        file.seekg(sizeof(TSRHeader), std::ios::beg);

        size_t element_size = (header.dtype == static_cast<int32_t>(DType::FP32)) ? 4 : 1;
        size_t expected_data_size = header.total_elements * element_size;
        size_t expected_file_size = sizeof(TSRHeader) + expected_data_size;

        if (static_cast<size_t>(file_size) != expected_file_size) {
            throw TRException("File size mismatch. Expected " +
                             std::to_string(expected_file_size) + " bytes, got " +
                             std::to_string(static_cast<size_t>(file_size)) + " bytes");
        }

        // 创建张量
        Shape shape;
        if (header.ndim == 0) {
            shape = Shape(); // 标量
        } else if (header.ndim == 1) {
            shape = Shape(header.dims[3]);
        } else if (header.ndim == 2) {
            shape = Shape(header.dims[2], header.dims[3]);
        } else if (header.ndim == 3) {
            shape = Shape(header.dims[1], header.dims[2], header.dims[3]);
        } else { // header.ndim == 4
            shape = Shape(header.dims[0], header.dims[1], header.dims[2], header.dims[3]);
        }

        DType dtype = static_cast<DType>(header.dtype);
        Tensor tensor = Tensor::empty(shape, dtype, tr::CPU);

        // 读取张量数据
        size_t data_size = tensor.memory_size();
        void* data_ptr = tensor.data_ptr();  // CpuBackend是友元，可以访问

        if (!data_ptr) {
            throw TRException("Tensor data pointer is null");
        }

        file.read(static_cast<char*>(data_ptr), data_size);

        if (file.gcount() != static_cast<std::streamsize>(data_size)) {
            throw TRException("Failed to read complete tensor data. Expected " +
                             std::to_string(data_size) + " bytes, got " +
                             std::to_string(file.gcount()) + " bytes");
        }

        file.close();

        Logger::get_instance().info("Tensor imported successfully from " + filename);
        return tensor;

    } catch (const std::exception& e) {
        throw TRException("Tensor import failed: " + std::string(e.what()));
    }
}

bool CpuBackend::is_close(const Tensor& tensor_a, const Tensor& tensor_b, float eps) const {
    // 1. 检查设备一致性
    validate_same_device(tensor_a.device());
    validate_same_device(tensor_b.device());

    // 2. 检查数据类型必须是FP32
    if (tensor_a.dtype() != DType::FP32 || tensor_b.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::is_close] Only FP32 tensors are supported");
    }

    // 3. 检查形状一致性
    if (tensor_a.shape() != tensor_b.shape()) {
        return false;
    }

    // 4. 处理空张量情况
    if (tensor_a.is_empty() && tensor_b.is_empty()) {
        return true;
    }
    if (tensor_a.is_empty() || tensor_b.is_empty()) {
        return false;
    }

    // 5. 获取数据指针
    const float* data_a = static_cast<const float*>(tensor_a.data_ptr());
    const float* data_b = static_cast<const float*>(tensor_b.data_ptr());

    if (!data_a || !data_b) {
        throw TRException("[CpuBackend::is_close] Invalid tensor data pointers");
    }

    // 6. 计算绝对差的平均值
    size_t num_elements = tensor_a.numel();
    float sum_abs_diff = 0.0f;

#ifdef TR_USE_EIGEN
    // 使用Eigen加速计算
    Eigen::Map<const Eigen::VectorXf> eigen_a(data_a, num_elements);
    Eigen::Map<const Eigen::VectorXf> eigen_b(data_b, num_elements);
    Eigen::VectorXf diff = (eigen_a - eigen_b).cwiseAbs();
    sum_abs_diff = diff.sum();
#else
    // 朴素实现
    for (size_t i = 0; i < num_elements; ++i) {
        sum_abs_diff += std::abs(data_a[i] - data_b[i]);
    }
#endif

    float avg_abs_diff = sum_abs_diff / static_cast<float>(num_elements);

    // 7. 比较平均绝对差是否在容差范围内
    return avg_abs_diff <= eps;
}

// ===== 误差计算方法实现（V1.23.1新增） =====

double CpuBackend::get_mean_abs_err(const Tensor& tensor_a, const Tensor& tensor_b) const {
    // 1. 检查设备一致性
    validate_same_device(tensor_a.device());
    validate_same_device(tensor_b.device());

    // 2. 检查数据类型必须是FP32
    if (tensor_a.dtype() != DType::FP32 || tensor_b.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::get_mean_abs_err] Only FP32 tensors are supported");
    }

    // 3. 检查形状一致性
    if (tensor_a.shape() != tensor_b.shape()) {
        throw TRException("[CpuBackend::get_mean_abs_err] Tensor shapes must be identical");
    }

    // 4. 处理空张量情况
    if (tensor_a.is_empty() && tensor_b.is_empty()) {
        return 0.0;  // 两个空张量的平均绝对误差为0
    }
    if (tensor_a.is_empty() || tensor_b.is_empty()) {
        throw TRException("[CpuBackend::get_mean_abs_err] Cannot compute error with empty tensor");
    }

    // 5. 获取数据指针
    const float* data_a = static_cast<const float*>(tensor_a.data_ptr());
    const float* data_b = static_cast<const float*>(tensor_b.data_ptr());

    if (!data_a || !data_b) {
        throw TRException("[CpuBackend::get_mean_abs_err] Invalid tensor data pointers");
    }

    // 6. 计算平均绝对误差
    size_t num_elements = tensor_a.numel();
    double sum_abs_error = 0.0;

#ifdef TR_USE_EIGEN
    // 使用Eigen加速计算
    Eigen::Map<const Eigen::VectorXf> eigen_a(data_a, num_elements);
    Eigen::Map<const Eigen::VectorXf> eigen_b(data_b, num_elements);
    Eigen::VectorXf diff = (eigen_a - eigen_b).cwiseAbs();
    sum_abs_error = diff.sum();
#else
    // 朴素实现
    for (size_t i = 0; i < num_elements; ++i) {
        sum_abs_error += std::abs(data_a[i] - data_b[i]);
    }
#endif

    return sum_abs_error / static_cast<double>(num_elements);
}

double CpuBackend::get_mean_rel_err(const Tensor& tensor_a, const Tensor& tensor_b) const {
    // 1. 检查设备一致性
    validate_same_device(tensor_a.device());
    validate_same_device(tensor_b.device());

    // 2. 检查数据类型必须是FP32
    if (tensor_a.dtype() != DType::FP32 || tensor_b.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::get_mean_rel_err] Only FP32 tensors are supported");
    }

    // 3. 检查形状一致性
    if (tensor_a.shape() != tensor_b.shape()) {
        throw TRException("[CpuBackend::get_mean_rel_err] Tensor shapes must be identical");
    }

    // 4. 处理空张量情况
    if (tensor_a.is_empty() && tensor_b.is_empty()) {
        return 0.0;  // 两个空张量的平均相对误差为0
    }
    if (tensor_a.is_empty() || tensor_b.is_empty()) {
        throw TRException("[CpuBackend::get_mean_rel_err] Cannot compute error with empty tensor");
    }

    // 5. 获取数据指针
    const float* data_a = static_cast<const float*>(tensor_a.data_ptr());
    const float* data_b = static_cast<const float*>(tensor_b.data_ptr());

    if (!data_a || !data_b) {
        throw TRException("[CpuBackend::get_mean_rel_err] Invalid tensor data pointers");
    }

    // 6. 计算平均绝对误差
    double mean_abs_err = get_mean_abs_err(tensor_a, tensor_b);

    // 7. 计算平均相对误差的分母（tensor_a的绝对值平均值）
    size_t num_elements = tensor_a.numel();
    double sum_abs_a = 0.0;

#ifdef TR_USE_EIGEN
    // 使用Eigen加速计算
    Eigen::Map<const Eigen::VectorXf> eigen_a(data_a, num_elements);
    sum_abs_a = eigen_a.cwiseAbs().sum();
#else
    // 朴素实现
    for (size_t i = 0; i < num_elements; ++i) {
        sum_abs_a += std::abs(data_a[i]);
    }
#endif

    // 8. 处理除数为0的情况
    double mean_abs_a = sum_abs_a / static_cast<double>(num_elements);
    if (mean_abs_a < std::numeric_limits<double>::epsilon()) {
        return 0.0;  // 当tensor_a的绝对值平均值接近0时，返回0
    }

    return mean_abs_err / mean_abs_a;
}

// 设备转换方法实现
Tensor CpuBackend::to(const Tensor& tensor, const Device& device) const {
    if (tensor.device() == device) {
        return tensor; // 设备相同，直接返回
    }

    if (device.is_cpu()) {
        return to_cpu(tensor);
    } else {
        // CPU后端不支持直接转换到其他设备
        throw TRException("CPU backend cannot directly convert to device: " + device.to_string() +
                         ". Please use BackendManager to get the appropriate backend first.");
    }
}

Tensor CpuBackend::to_cpu(const Tensor& tensor) const {
    if (!tensor.device().is_cpu()) {
        throw TRException("CpuBackend::to_cpu expects input tensor to be on CPU device");
    }

    // 对于CPU后端，to_cpu就是返回自身（或副本，根据需求）
    // 这里直接返回自身，因为已经是CPU张量
    return tensor;
}

Tensor CpuBackend::from_cpu(const Tensor& tensor) const {
    if (!tensor.device().is_cpu()) {
        throw TRException("CpuBackend::from_cpu expects input tensor to be on CPU device");
    }

    // 对于CPU后端，from_cpu也是返回自身（或副本）
    // 这里直接返回自身，因为已经是CPU张量
    return tensor;
}

} // namespace tr
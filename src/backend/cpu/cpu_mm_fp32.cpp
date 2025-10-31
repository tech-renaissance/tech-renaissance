/**
 * @file cpu_mm_fp32.cpp
 * @brief CPU后端FP32矩阵乘法实现
 * @details 实现CPU后端的FP32矩阵乘法运算，支持Eigen优化和朴素实现
 * @version 1.00.00
 * @date 2025-10-29
 * @author 技术觉醒团队
 * @note 依赖项: tensor.h, backend.h, Eigen
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include "tech_renaissance/data/tensor.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/utils/logger.h"

#ifdef TR_USE_EIGEN
#include "Core"
#include <omp.h>
#endif

namespace tr {

void CpuBackend::mm(Tensor& result, const Tensor& tensor_a, const Tensor& tensor_b) {
    // 1. 验证输入张量都在CPU设备上
    validate_same_device(tensor_a.device());
    validate_same_device(tensor_b.device());
    validate_same_device(result.device());

    // 2. 验证数据类型都是FP32
    if (tensor_a.dtype() != DType::FP32 || tensor_b.dtype() != DType::FP32 || result.dtype() != DType::FP32) {
        throw TRException("[CpuBackend::mm] Only FP32 tensors are supported for matrix multiplication");
    }

    // 3. 验证张量都不为空
    if (tensor_a.is_empty() || tensor_b.is_empty() || result.is_empty()) {
        throw TRException("[CpuBackend::mm] Cannot perform matrix multiplication on empty tensors");
    }

    // 4. 验证维度符合矩阵乘法要求
    // 矩阵乘法要求: A(M,K) × B(K,N) = C(M,N)
    // 在我们的框架中，张量可以是多维的，但我们将其视为2D矩阵进行计算
    int32_t a_rows, a_cols, b_rows, b_cols;

    // 获取实际的矩阵维度（处理多维张量的情况）
    if (tensor_a.ndim() == 1) {
        // 1D张量视为行向量 (1, K)
        a_rows = 1;
        a_cols = tensor_a.dim_size(0);
    } else if (tensor_a.ndim() >= 2) {
        // 多维张量取最后两个维度
        a_rows = tensor_a.dim_size(tensor_a.ndim() - 2);
        a_cols = tensor_a.dim_size(tensor_a.ndim() - 1);
    } else {
        throw TRException("[CpuBackend::mm] Tensor A must have at least 1 dimension");
    }

    if (tensor_b.ndim() == 1) {
        // 1D张量视为列向量 (K, 1)
        b_rows = tensor_b.dim_size(0);
        b_cols = 1;
    } else if (tensor_b.ndim() >= 2) {
        // 多维张量取最后两个维度
        b_rows = tensor_b.dim_size(tensor_b.ndim() - 2);
        b_cols = tensor_b.dim_size(tensor_b.ndim() - 1);
    } else {
        throw TRException("[CpuBackend::mm] Tensor B must have at least 1 dimension");
    }

    // 验证矩阵乘法的维度匹配：A的列数必须等于B的行数
    if (a_cols != b_rows) {
        throw TRException("[CpuBackend::mm] Matrix multiplication dimension mismatch: "
                         "A.cols(" + std::to_string(a_cols) + ") != B.rows(" + std::to_string(b_rows) + ")");
    }

    // 验证结果张量的维度是否正确
    int32_t expected_result_rows = a_rows;
    int32_t expected_result_cols = b_cols;

    if (result.ndim() == 1) {
        if (expected_result_cols == 1) {
            // 结果为列向量，应该是1D
            if (result.dim_size(0) != expected_result_rows) {
                throw TRException("[CpuBackend::mm] Result tensor dimension mismatch");
            }
        } else {
            // 结果为行向量，应该是1D
            if (result.dim_size(0) != expected_result_cols) {
                throw TRException("[CpuBackend::mm] Result tensor dimension mismatch");
            }
        }
    } else if (result.ndim() >= 2) {
        if (result.dim_size(result.ndim() - 2) != expected_result_rows ||
            result.dim_size(result.ndim() - 1) != expected_result_cols) {
            throw TRException("[CpuBackend::mm] Result tensor dimension mismatch");
        }
    }

    // 5. 获取数据指针
    const float* a_data = static_cast<const float*>(tensor_a.data_ptr());
    const float* b_data = static_cast<const float*>(tensor_b.data_ptr());
    float* c_data = static_cast<float*>(result.data_ptr());

    if (!a_data || !b_data || !c_data) {
        throw TRException("[CpuBackend::mm] Invalid tensor data pointers");
    }

    // 6. 执行矩阵乘法计算
    try {
#ifdef TR_USE_EIGEN
        // 使用Eigen优化的实现
        Logger::get_instance().debug("Using Eigen for CPU matrix multiplication");

        // 设置Eigen多线程配置，与test_cpu_gemm.cpp保持一致
        Eigen::setNbThreads(omp_get_max_threads());

        // 创建Eigen矩阵映射（指定行主序）
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_a(a_data, a_rows, a_cols);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_b(b_data, b_rows, b_cols);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_c(c_data, expected_result_rows, expected_result_cols);

        // 执行矩阵乘法（使用noalias避免别名问题，与test_cpu_gemm.cpp保持一致）
        eigen_c.noalias() = eigen_a * eigen_b;

#else
        // 使用朴素实现
        Logger::get_instance().debug("Using naive implementation for CPU matrix multiplication");

        // 朴素矩阵乘法：C(M,N) = A(M,K) × B(K,N)
        for (int32_t i = 0; i < expected_result_rows; ++i) {
            for (int32_t j = 0; j < expected_result_cols; ++j) {
                float sum = 0.0f;
                for (int32_t k = 0; k < a_cols; ++k) {
                    // 计算索引：对于多维张量，需要展平访问
                    int32_t a_idx = i * a_cols + k;  // A[i,k]
                    int32_t b_idx = k * b_cols + j;  // B[k,j]
                    sum += a_data[a_idx] * b_data[b_idx];
                }
                c_data[i * expected_result_cols + j] = sum;  // C[i,j]
            }
        }

#endif
        // Logger::get_instance().info("CPU matrix multiplication completed successfully: " +
        //                              std::to_string(a_rows) + "*" + std::to_string(a_cols) + " * " +
        //                              std::to_string(b_rows) + "*" + std::to_string(b_cols) + " = " +
        //                              std::to_string(expected_result_rows) + "*" + std::to_string(expected_result_cols));

    } catch (const std::exception& e) {
        throw TRException("[CpuBackend::mm] Matrix multiplication failed: " + std::string(e.what()));
    }
}

} // namespace tr
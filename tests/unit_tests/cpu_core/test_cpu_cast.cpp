/**
 * @file test_cpu_cast.cpp
 * @brief CPU backend cast function test
 * @details Test tensor data type conversion functions, including conversions between FP32, INT8, INT32
 * @version 1.31.2
 * @date 2025-11-02
 * @author Tech Renaissance Team
 * @note Dependencies: cpu_backend.h, tensor.h, backend_manager.h
 * @note Series: backend
 */

#include <iostream>
#include "tech_renaissance.h"

using namespace tr;


int main() {
    try {
        std::cout << "===== CPU Backend Cast Function Tests =====\n";

        // Get CPU backend instance
        auto cpu_backend = BackendManager::get_cpu_backend();

        if (!cpu_backend) {
            throw TRException("Failed to get CPU backend instance");
        }

        // ===== Test Case 1: 2x4x5 tensor, test FP32->INT32->INT8->INT32->FP32 conversion chain =====
        std::cout << "\n=== Test Case 1: 2x4x5 tensor conversion chain FP32->INT32->INT8->INT32->FP32 ===\n";

        // Generate (0,1) normal distribution tensor (shape 2x4x5)
        Shape shape1({2, 4, 5});
        Tensor tensor_fp32_1 = cpu_backend->randn(shape1, 12345);
        std::cout << "\n1. Generate initial FP32 tensor (0,1 normal distribution):";
        tensor_fp32_1.print("FP32_Original");

        // Multiply by 100.0f for scaling
        Tensor tensor_scaled_1 = cpu_backend->mul(tensor_fp32_1, 100.0f);
        std::cout << "\n2. After multiplying by 100.0f:";
        tensor_scaled_1.print("FP32_Scaled");
        // Test FP32->INT32
        Tensor tensor_int32_1 = cpu_backend->cast(tensor_scaled_1, DType::INT32);
        std::cout << "\n3. FP32->INT32 conversion:";
        tensor_int32_1.print("INT32_From_FP32");

        // Test INT32->INT8
        Tensor tensor_int8_1 = cpu_backend->cast(tensor_int32_1, DType::INT8);
        std::cout << "\n4. INT32->INT8 conversion:";
        tensor_int8_1.print("INT8_From_INT32");

        // Test INT8->INT32
        Tensor tensor_int32_2 = cpu_backend->cast(tensor_int8_1, DType::INT32);
        std::cout << "\n5. INT8->INT32 conversion:";
        tensor_int32_2.print("INT32_From_INT8");

        // Test INT32->FP32
        Tensor tensor_fp32_2 = cpu_backend->cast(tensor_int32_2, DType::FP32);
        std::cout << "\n6. INT32->FP32 conversion:";
        tensor_fp32_2.print("FP32_From_INT32");

        // ===== Test Case 2: 2x3x4 tensor, test FP32->INT8->FP32 conversion =====
        std::cout << "\n\n=== Test Case 2: 2x3x4 tensor conversion FP32->INT8->FP32 ===\n";

        // Generate (0,1) normal distribution tensor (shape 2x3x4)
        Shape shape2({2, 3, 4});
        Tensor tensor_fp32_orig2 = cpu_backend->randn(shape2, 67890);
        std::cout << "\n1. Generate initial FP32 tensor (0,1 normal distribution):";
        tensor_fp32_orig2.print("FP32_Original_2");

        // Multiply by 50.0f for scaling
        Tensor tensor_scaled_2 = cpu_backend->mul(tensor_fp32_orig2, 50.0f);
        std::cout << "\n2. After multiplying by 50.0f:";
        tensor_scaled_2.print("FP32_Scaled_2");

        // Test FP32->INT8
        Tensor tensor_int8_2 = cpu_backend->cast(tensor_scaled_2, DType::INT8);
        std::cout << "\n3. FP32->INT8 conversion:";
        tensor_int8_2.print("INT8_From_FP32_2");

        // Test INT8->FP32
        Tensor tensor_fp32_final2 = cpu_backend->cast(tensor_int8_2, DType::FP32);
        std::cout << "\n4. INT8->FP32 conversion:";
        tensor_fp32_final2.print("FP32_From_INT8_2");

        // ===== Verify conversion accuracy =====
        std::cout << "\n\n=== Conversion Accuracy Verification ===\n";

        // Calculate difference between INT8->INT32->FP32 chain and original scaled tensor
        double mean_abs_err_1 = cpu_backend->get_mean_abs_err(tensor_scaled_1, tensor_fp32_2);
        double mean_rel_err_1 = cpu_backend->get_mean_rel_err(tensor_scaled_1, tensor_fp32_2);

        std::cout << "Test Case 1 conversion chain accuracy (FP32->INT32->INT8->INT32->FP32):\n";
        std::cout << "  Mean absolute error: " << std::scientific << mean_abs_err_1 << "\n";
        std::cout << "  Mean relative error: " << std::scientific << mean_rel_err_1 << "\n";

        // Calculate difference between INT8->FP32 and original scaled tensor
        double mean_abs_err_2 = cpu_backend->get_mean_abs_err(tensor_scaled_2, tensor_fp32_final2);
        double mean_rel_err_2 = cpu_backend->get_mean_rel_err(tensor_scaled_2, tensor_fp32_final2);

        std::cout << "Test Case 2 conversion accuracy (FP32->INT8->FP32):\n";
        std::cout << "  Mean absolute error: " << std::scientific << mean_abs_err_2 << "\n";
        std::cout << "  Mean relative error: " << std::scientific << mean_rel_err_2 << "\n";

        std::cout << "\n===== All cast function tests passed! =====\n";

    } catch (const TRException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
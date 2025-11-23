#include "tech_renaissance.h"
#include <iostream>

using namespace tr;

int main() {
    Logger::get_instance().set_quiet_mode(false);
    auto cpu = BackendManager::get_cpu_backend();

    std::cout << "=== Testing CPU Slice Operations ===" << std::endl;

    try {
        // 测试1: 4D张量切片
        std::cout << "\n1. Testing 4D tensor slice:" << std::endl;
        Tensor tensor4d = cpu->ones(Shape(4, 3, 5, 6), DType::FP32);

        // 填充一些测试数据
        float* data = static_cast<float*>(tensor4d.data_ptr());
        for (int64_t i = 0; i < tensor4d.numel(); ++i) {
            data[i] = static_cast<float>(i % 100);
        }

        // 创建Offset：N[1,3), C[0,2), H[1,4), W[2,5)
        Offset offset4d(1, 3, 0, 2, 1, 4, 2, 5);
        Tensor slice4d = cpu->slice(tensor4d, offset4d);

        std::cout << "   Original tensor shape: " << tensor4d.shape().to_string() << std::endl;
        std::cout << "   Slice tensor shape: " << slice4d.shape().to_string() << std::endl;
        std::cout << "   First few values: [";
        for (int i = 0; i < std::min(10, static_cast<int>(slice4d.numel())); ++i) {
            std::cout << cpu->get_item_fp32(slice4d, i);
            if (i < std::min(9, static_cast<int>(slice4d.numel()) - 1)) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // 测试2: 使用-1表示到末尾
        std::cout << "\n2. Testing slice with -1 (to end):" << std::endl;
        Offset offset_to_end(1, -1, 1, -1, 2, -1);
        Tensor slice_to_end = cpu->slice(tensor4d, offset_to_end);

        std::cout << "   Slice to end shape: " << slice_to_end.shape().to_string() << std::endl;

        // 测试3: 2D张量切片
        std::cout << "\n3. Testing 2D tensor slice:" << std::endl;
        Tensor tensor2d = cpu->randint(Shape(4, 8), 0, 10, DType::FP32);

        std::cout << "   Original 2D tensor: " << std::endl;
        tensor2d.print();
        std::cout << "Offset: (1, 3, 2, 6)" << std::endl;

        Offset offset2d(1, 3, 2, 6);  // H[1,3), W[2,6)

        Tensor slice2d = cpu->slice(tensor2d, offset2d);
        std::cout << "   Sliced 2D tensor: " << std::endl;
        slice2d.print();

        // 验证数据正确性
        std::cout << "   Slice values: [";
        for (int i = 0; i < static_cast<int>(slice2d.numel()); ++i) {
            std::cout << cpu->get_item_fp32(slice2d, i);
            if (i < static_cast<int>(slice2d.numel()) - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // 测试4: stride采样
        std::cout << "\n4. Testing slice with stride:" << std::endl;
        Offset offset_stride(0, -1, 0, -1, 0, -1, 0, -1);
        offset_stride.set_w_stride(2);  // W方向每隔2个采样
        offset_stride.set_h_stride(2);  // H方向每隔2个采样

        Tensor slice_stride = cpu->slice(tensor4d, offset_stride);
        std::cout << "   Original shape: " << tensor4d.shape().to_string() << std::endl;
        std::cout << "   Stride slice shape: " << slice_stride.shape().to_string() << std::endl;

        // 测试5: slice_into功能
        std::cout << "\n5. Testing slice_into functionality:" << std::endl;
        auto result = cpu->empty(Shape(2, 2, 3, 3), DType::FP32);
        Offset offset_into(0, 2, 0, 2, 1, 4, 1, 4);

        cpu->slice_into(tensor4d, result, offset_into);
        std::cout << "   Target shape: " << result.shape().to_string() << std::endl;
        std::cout << "   Expected shape: [2, 2, 3, 3]" << std::endl;
        std::cout << "   First value: " << cpu->get_item_fp32(result, 0) << std::endl;

        // 测试6: INT8数据类型
        std::cout << "\n6. Testing INT8 tensor slice:" << std::endl;
        Tensor int8_tensor = cpu->ones(Shape(2, 3, 4), DType::INT8);

        // 填充测试数据
        int8_t* int8_data = static_cast<int8_t*>(int8_tensor.data_ptr());
        for (int i = 0; i < 24; ++i) {
            int8_data[i] = static_cast<int8_t>(i % 50 - 25);
        }

        Offset offset_int8(0, 2, 1, 2, 2, 4);  // N[0,2), C[1,2), H[2,4)
        Tensor slice_int8 = cpu->slice(int8_tensor, offset_int8);

        std::cout << "   INT8 tensor shape: " << int8_tensor.shape().to_string() << std::endl;
        std::cout << "   INT8 slice shape: " << slice_int8.shape().to_string() << std::endl;
        std::cout << "   First INT8 value: " << static_cast<int>(cpu->get_item_int8(slice_int8, 0)) << std::endl;

        // 测试7: 错误情况测试
        std::cout << "\n7. Testing error cases:" << std::endl;

        try {
            // 超出范围的切片
            Offset bad_offset(10, 15, 0, 1, 0, 1, 0, 1);  // N维度超出范围
            Tensor bad_slice = cpu->slice(tensor4d, bad_offset);
            std::cout << "   ERROR: Should have thrown exception for out of range!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "   Out of range error caught correctly: " << e.what() << std::endl;
        }

        try {
            // slice_into形状不匹配
            auto wrong_result = cpu->empty(Shape(2, 2, 2, 2), DType::FP32);
            Offset offset_correct(0, 2, 0, 2, 1, 4, 1, 4);  // 实际输出应该是[2,2,3,3]
            cpu->slice_into(tensor4d, wrong_result, offset_correct);
            std::cout << "   ERROR: Should have thrown exception for shape mismatch!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "   Shape mismatch error caught correctly: " << e.what() << std::endl;
        }

        try {
            // 数据类型不匹配
            auto fp32_result = cpu->empty(Shape(2, 2, 3, 3), DType::FP32);
            cpu->slice_into(int8_tensor, fp32_result, offset_int8);
            std::cout << "   ERROR: Should have thrown exception for type mismatch!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "   Type mismatch error caught correctly: " << e.what() << std::endl;
        }

        std::cout << "\n=== All slice tests completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
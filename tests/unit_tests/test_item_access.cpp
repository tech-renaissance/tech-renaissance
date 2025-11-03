#include "tech_renaissance.h"
#include <iostream>

using namespace tr;

int main() {
    Logger::get_instance().set_quiet_mode(false);
    auto cpu = BackendManager::get_cpu_backend();

    std::cout << "=== Testing Item Access Methods ===" << std::endl;

    try {
        // 测试1: FP32张量的元素访问
        std::cout << "\n1. Testing FP32 tensor item access:" << std::endl;
        Tensor fp32_tensor = cpu->zeros(Shape(2, 3, 2, 2), DType::FP32);

        // 设置一些值
        cpu->set_item_fp32(fp32_tensor, 0, 1.5f);    // 第一个元素
        cpu->set_item_fp32(fp32_tensor, 5, 2.7f);    // 第六个元素
        cpu->set_item_fp32(fp32_tensor, 23, 8.9f);   // 最后一个元素

        // 读取值
        float val1 = cpu->get_item_fp32(fp32_tensor, 0);
        float val2 = cpu->get_item_fp32(fp32_tensor, 5);
        float val3 = cpu->get_item_fp32(fp32_tensor, 23);

        std::cout << "   Set values: 0->1.5, 5->2.7, 23->8.9" << std::endl;
        std::cout << "   Get values: 0->" << val1 << ", 5->" << val2 << ", 23->" << val3 << std::endl;

        // 测试2: INT8张量的元素访问
        std::cout << "\n2. Testing INT8 tensor item access:" << std::endl;
        Tensor int8_tensor = cpu->zeros(Shape(3, 2, 2, 2), DType::INT8);

        // 设置一些值
        cpu->set_item_int8(int8_tensor, 1, 42);    // 第二个元素
        cpu->set_item_int8(int8_tensor, 10, -13);   // 第十一个元素

        // 读取值
        int8_t ival1 = cpu->get_item_int8(int8_tensor, 1);
        int8_t ival2 = cpu->get_item_int8(int8_tensor, 10);

        std::cout << "   Set values: 1->42, 10->-13" << std::endl;
        std::cout << "   Get values: 1->" << static_cast<int>(ival1) << ", 10->" << static_cast<int>(ival2) << std::endl;

        // 测试3: INT32张量的元素访问
        std::cout << "\n3. Testing INT32 tensor item access:" << std::endl;
        Tensor int32_tensor = cpu->zeros(Shape(2, 2, 2, 2), DType::INT32);

        // 设置一些值
        cpu->set_item_int32(int32_tensor, 3, 1000);   // 第四个元素
        cpu->set_item_int32(int32_tensor, 15, -5000); // 第十六个元素

        // 读取值
        int32_t i32val1 = cpu->get_item_int32(int32_tensor, 3);
        int32_t i32val2 = cpu->get_item_int32(int32_tensor, 15);

        std::cout << "   Set values: 3->1000, 15->-5000" << std::endl;
        std::cout << "   Get values: 3->" << i32val1 << ", 15->" << i32val2 << std::endl;

        // 测试4: 错误情况
        std::cout << "\n4. Testing error cases:" << std::endl;

        try {
            cpu->get_item_fp32(fp32_tensor, -1);
        } catch (const std::exception& e) {
            std::cout << "   Negative index error caught: " << e.what() << std::endl;
        }

        try {
            cpu->get_item_fp32(fp32_tensor, 24); // 超出范围 (0-23)
        } catch (const std::exception& e) {
            std::cout << "   Out of range error caught: " << e.what() << std::endl;
        }

        try {
            cpu->get_item_int8(fp32_tensor, 0); // 类型不匹配
        } catch (const std::exception& e) {
            std::cout << "   Type mismatch error caught: " << e.what() << std::endl;
        }

        std::cout << "\n=== All item access tests completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
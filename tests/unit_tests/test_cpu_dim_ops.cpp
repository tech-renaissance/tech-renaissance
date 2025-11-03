#include "tech_renaissance.h"
#include <iostream>
#include <iomanip>

using namespace tr;

int main() {
    Logger::get_instance().set_quiet_mode(false);
    auto cpu = BackendManager::get_cpu_backend();

    std::cout << "=== Testing CPU Dimension Operations ===" << std::endl;

    try {
        // 测试1: Softmax操作
        std::cout << "\n1. Testing Softmax operation:" << std::endl;
        Tensor tensor2d = cpu->randint(Shape(2, 3), 0, 5, DType::FP32);

        std::cout << "   Input tensor:" << std::endl;
        tensor2d.print();

        Tensor softmax_result = cpu->softmax(tensor2d, 1);  // 沿维度1求softmax
        std::cout << "   Softmax along dim=1:" << std::endl;
        softmax_result.print();

        // 验证softmax和为1
        std::cout << "   Row sums (should be close to 1.0):" << std::endl;
        for (int i = 0; i < tensor2d.shape().dim(0); ++i) {
            float row_sum = 0.0f;
            for (int j = 0; j < softmax_result.shape().dim(1); ++j) {
                row_sum += cpu->get_item_fp32(softmax_result, i * softmax_result.shape().dim(1) + j);
            }
            std::cout << "   Row " << i << ": " << std::fixed << std::setprecision(6) << row_sum << std::endl;
        }

        // 测试2: Max操作
        std::cout << "\n2. Testing Max operation:" << std::endl;
        Tensor tensor3d = cpu->randint(Shape(2, 3, 4), 0, 10, DType::FP32, time(nullptr));

        std::cout << "   Input tensor shape: " << tensor3d.shape().to_string() << std::endl;
        std::cout << "   Input tensor:" << std::endl;
        tensor3d.print();

        Tensor max_dim1 = cpu->max(tensor3d, 1, false);  // 沿维度1求最大值，不保留维度
        std::cout << "   Max along dim=1 (keep_dim=false): " << max_dim1.shape().to_string() << std::endl;
        max_dim1.print();

        Tensor max_dim1_keep = cpu->max(tensor3d, 1, true);  // 沿维度1求最大值，保留维度
        std::cout << "   Max along dim=1 (keep_dim=true): " << max_dim1_keep.shape().to_string() << std::endl;
        max_dim1_keep.print();

        // 测试3: Sum操作
        std::cout << "\n3. Testing Sum operation:" << std::endl;
        Tensor sum_dim2 = cpu->sum(tensor3d, 2, false);  // 沿维度2求和，不保留维度
        std::cout << "   Sum along dim=2 (keep_dim=false): " << sum_dim2.shape().to_string() << std::endl;
        sum_dim2.print();

        Tensor sum_dim2_keep = cpu->sum(tensor3d, 2, true);  // 沿维度2求和，保留维度
        std::cout << "   Sum along dim=2 (keep_dim=true): " << sum_dim2_keep.shape().to_string() << std::endl;
        sum_dim2_keep.print();

        // 测试4: ArgMax操作
        std::cout << "\n4. Testing ArgMax operation:" << std::endl;
        Tensor argmax_dim1 = cpu->argmax(tensor3d, 1, false);  // 沿维度1找最大值索引，不保留维度
        std::cout << "   ArgMax along dim=1 (keep_dim=false): " << argmax_dim1.shape().to_string() << std::endl;
        argmax_dim1.print();

        Tensor argmax_dim1_keep = cpu->argmax(tensor3d, 1, true);  // 沿维度1找最大值索引，保留维度
        std::cout << "   ArgMax along dim=1 (keep_dim=true): " << argmax_dim1_keep.shape().to_string() << std::endl;
        argmax_dim1_keep.print();

        // 测试5: 负维度索引
        std::cout << "\n5. Testing negative dimension indices:" << std::endl;
        Tensor softmax_neg = cpu->softmax(tensor2d, -1);  // 等价于dim=1
        std::cout << "   Softmax along dim=-1 (equivalent to dim=1):" << std::endl;
        softmax_neg.print();

        Tensor max_neg = cpu->max(tensor2d, -1, false);  // 等价于dim=1
        std::cout << "   Max along dim=-1 (equivalent to dim=1): " << max_neg.shape().to_string() << std::endl;
        max_neg.print();

        // 测试6: 不同数据类型
        std::cout << "\n6. Testing different data types:" << std::endl;
        Tensor int8_tensor = cpu->randint(Shape(2, 3), -10, 10, DType::INT8);
        std::cout << "   INT8 tensor:" << std::endl;
        int8_tensor.print();

        Tensor int8_max = cpu->max(int8_tensor, 1, false);
        std::cout << "   INT8 max along dim=1:" << std::endl;
        int8_max.print();

        Tensor int8_argmax = cpu->argmax(int8_tensor, 1, false);
        std::cout << "   INT8 argmax along dim=1:" << std::endl;
        int8_argmax.print();

        // 测试7: _into操作
        std::cout << "\n7. Testing _into operations:" << std::endl;
        auto softmax_target = cpu->empty(tensor2d.shape(), DType::FP32);
        cpu->softmax_into(tensor2d, softmax_target, 1);
        std::cout << "   Softmax_into result:" << std::endl;
        softmax_target.print();

        auto max_target = cpu->empty(max_dim1.shape(), DType::FP32);
        cpu->max_into(tensor3d, max_target, 1, false);
        std::cout << "   Max_into result:" << std::endl;
        max_target.print();

        auto sum_target = cpu->empty(sum_dim2.shape(), DType::FP32);
        cpu->sum_into(tensor3d, sum_target, 2, false);
        std::cout << "   Sum_into result:" << std::endl;
        sum_target.print();

        auto argmax_target = cpu->empty(argmax_dim1.shape(), DType::INT32);
        cpu->argmax_into(tensor3d, argmax_target, 1, false);
        std::cout << "   ArgMax_into result:" << std::endl;
        argmax_target.print();

        // 测试8: _inplace操作
        std::cout << "\n8. Testing _inplace operations:" << std::endl;
        Tensor softmax_inplace_tensor = cpu->copy(tensor2d);
        cpu->softmax_inplace(softmax_inplace_tensor, 1);
        std::cout << "   Softmax_inplace result:" << std::endl;
        softmax_inplace_tensor.print();

        std::cout << "\n=== All dimension operations tests completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
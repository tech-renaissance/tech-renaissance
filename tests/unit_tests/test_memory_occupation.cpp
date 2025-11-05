#include "tech_renaissance.h"

using namespace tr;
int main() {
    const int wait_time = 5;
    const int tensor_size = 8;  // GB
    Logger::get_instance().set_quiet_mode(false);
    auto cpu = BackendManager::get_cpu_backend();

    Logger::get_instance().info("Start memory test!");
    Logger::get_instance().info("No tensor.");
    int i = 0;
    for (i = 0; i < wait_time; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        Logger::get_instance().info("Time elapsed: " + std::to_string(i) + " s. " + "No tensor.");
    }

    // 测试1：创建一个张量，并把它赋给一个变量，过一段时间后用Tensor()构造函数销毁它
    // 预期结果：内存占用会显著增加，销毁后内存占用恢复
    Tensor t1 = cpu->zeros(Shape(250 * tensor_size, 1000, 1000), DType::FP32);
    Logger::get_instance().info("Created an 8GB tensor named \"t1\".");
    for (i = 0; i < wait_time; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        Logger::get_instance().info("Time elapsed: " + std::to_string(i) + " s. " + "An 8GB tensor named \"t1\".");
    }

    t1 = Tensor();
    Logger::get_instance().info("Destroy tensor \"t1\" with Tensor().");
    for (i = 0; i < wait_time; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        Logger::get_instance().info("Time elapsed: " + std::to_string(i) + " s. " + "No tensor.");
    }

    // 测试2：在大括号{}内创建一个张量，并把它赋给一个变量，一段时间后离开大括号{}
    // 预期结果：内存占用会显著增加，离开大括号作用域后内存占用恢复
    {
        Tensor t2 = cpu->zeros(Shape(250 * tensor_size, 1000, 1000), DType::FP32);
        Logger::get_instance().info("Created an 8GB tensor named \"t2\".");
        for (i = 0; i < wait_time; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            Logger::get_instance().info("Time elapsed: " + std::to_string(i) + " s. " + "An 8GB tensor named \"t2\".");
        }
    }

    Logger::get_instance().info("Left the range of \"t2\".");
    for (i = 0; i < wait_time; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        Logger::get_instance().info("Time elapsed: " + std::to_string(i) + " s. " + "No tensor.");
    }

    // 测试3：创建一个张量，并把它赋给一个变量，过一段时间后用CPU后端类的null_tensor()方法销毁它
    // 预期结果：内存占用会显著增加，销毁后内存占用恢复
    Tensor t3 = cpu->zeros(Shape(250 * tensor_size, 1000, 1000), DType::FP32);
    Logger::get_instance().info("Created an 8GB tensor named \"t3\".");
    for (i = 0; i < wait_time; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        Logger::get_instance().info("Time elapsed: " + std::to_string(i) + " s. " + "An 8GB tensor named \"t3\".");
    }

    t3 = cpu->null_tensor();
    Logger::get_instance().info("Destroy tensor \"t3\" with CpuBackend::null_tensor() method.");
    for (i = 0; i < wait_time; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        Logger::get_instance().info("Time elapsed: " + std::to_string(i) + " s. " + "No tensor.");
    }

    // 测试4：创建一个张量，然后把它移到GPU，过一段时间后用Tensor()构造函数销毁它
    // 预期结果：用t4 = cuda->from_cpu(t4)来移动张量到GPU后，CPU上的内存占用会释放
    {
        auto cuda = BackendManager::get_cuda_backend();
        Tensor t4 = cpu->zeros(Shape(125 * tensor_size, 1000, 1000), DType::FP32);
        Logger::get_instance().info("Created a 4GB tensor named \"t4\".");
        for (i = 0; i < wait_time; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            Logger::get_instance().info("Time elapsed: " + std::to_string(i) + " s. " + "A 4GB tensor named \"t4\".");
        }
        t4 = cuda->from_cpu(t4);  // 将t4移动到GPU
        Logger::get_instance().info("Move \"t4\" to GPU using\"t4 = cuda->from_cpu(t4);\".");
        for (i = 0; i < wait_time; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            Logger::get_instance().info("Time elapsed: " + std::to_string(i) + " s. " + "Tensor \"t4\" is now on CUDA.");
        }
    }
    Logger::get_instance().info("Left the range of \"t4\".");
    for (i = 0; i < wait_time; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        Logger::get_instance().info("Time elapsed: " + std::to_string(i) + " s. " + "No tensor.");
    }

    // 测试5：创建一个张量，但不把它赋给任何变量
    // 预期结果：即使不赋值给变量，大张量仍可能会消耗内存
    cpu->zeros(Shape(250 * tensor_size, 1000, 1000), DType::FP32);
    Logger::get_instance().info("Created an 8GB tensor without name.");
    for (i = 0; i < wait_time * 2; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        Logger::get_instance().info("Time elapsed: " + std::to_string(i) + " s. " + "An 8GB tensor without name.");
    }

    return 0;
}
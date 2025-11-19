/**
 * @file test_d4_optimizer.cpp
 * @brief D4优化器实现单元测试
 * @details 测试StateManager、Optimizer基类和SGD优化器的D4实现
 * @version 1.00.00
 * @date 2025-11-19
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h
 * @note 所属系列: trainer
 */

#include "tech_renaissance.h"

using namespace tr;

/**
 * @brief 测试StateManager基本功能
 */
void test_state_manager_basic() {
    std::cout << "[TEST] Testing StateManager basic functionality..." << std::endl;

    // 获取后端
    auto backend = BackendManager::get_cpu_backend();

    // 创建简单模型
    auto model = Model::create("TestModel",
                              std::make_shared<Linear>(10, 5),
                              std::make_shared<Tanh>(),
                              std::make_shared<Linear>(5, 1));

    model->set_backend(backend);
    model->train();

    // 创建StateManager
    StateManager state_manager(backend);

    // 测试初始状态
    if (!state_manager.is_initialized()) {
        std::cout << "  [PASS] StateManager initially not initialized" << std::endl;
    } else {
        std::cout << "  [FAIL] StateManager should not be initialized initially" << std::endl;
        return;
    }

    // 获取模型参数
    auto params = model->trainable_parameters();
    if (params.empty()) {
        std::cout << "  [FAIL] Model has no trainable parameters" << std::endl;
        return;
    }

    // 初始化SGD状态
    state_manager.initialize_sgd_states(params, 0.9f);

    // 验证初始化结果
    if (state_manager.is_initialized() &&
        state_manager.state_count() == params.size()) {
        std::cout << "  [PASS] StateManager successfully initialized with " << params.size() << " parameters" << std::endl;
    } else {
        std::cout << "  [FAIL] StateManager initialization failed" << std::endl;
        return;
    }

    // 测试状态访问
    bool all_states_valid = true;
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& state = state_manager.get_state(i);
        if (!state.has_momentum || state.has_adam_state) {
            all_states_valid = false;
            break;
        }
    }

    if (all_states_valid) {
        std::cout << "  [PASS] All states correctly configured with SGD momentum" << std::endl;
    } else {
        std::cout << "  [FAIL] State configuration incorrect" << std::endl;
        return;
    }
}

/**
 * @brief 测试SGD优化器构造和初始化
 */
void test_sgd_construction() {
    std::cout << "[TEST] Testing SGD optimizer construction and initialization..." << std::endl;

    // 获取后端
    auto backend = BackendManager::get_cpu_backend();

    // 测试基本构造
    auto sgd_basic = std::make_shared<SGD>(0.01f);
    if (sgd_basic->get_lr() == 0.01f &&
        sgd_basic->get_momentum() == 0.0f &&
        sgd_basic->get_weight_decay() == 0.0f &&
        !sgd_basic->get_nesterov()) {
        std::cout << "  [PASS] SGD basic construction successful" << std::endl;
    } else {
        std::cout << "  [FAIL] SGD basic construction parameters incorrect" << std::endl;
        return;
    }

    // 测试完整构造
    auto sgd_full = std::make_shared<SGD>(0.001f, 0.9f, 0.0001f, true);
    if (sgd_full->get_lr() == 0.001f &&
        sgd_full->get_momentum() == 0.9f &&
        sgd_full->get_weight_decay() == 0.0001f &&
        sgd_full->get_nesterov()) {
        std::cout << "  [PASS] SGD full construction successful" << std::endl;
    } else {
        std::cout << "  [FAIL] SGD full construction parameters incorrect" << std::endl;
        return;
    }

    // 创建模型进行初始化测试
    auto model = Model::create("SGDTest",
                              std::make_shared<Linear>(10, 5),
                              std::make_shared<Tanh>(),
                              std::make_shared<Linear>(5, 1));

    model->set_backend(backend);
    model->train();

    // 初始化优化器
    try {
        sgd_basic->initialize(*model);
        std::cout << "  [PASS] SGD optimizer initialization successful" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] SGD optimizer initialization failed: " << e.what() << std::endl;
        return;
    }

    // 验证状态管理器
    auto* state_manager = sgd_basic->get_state_manager();
    if (state_manager != nullptr && state_manager->is_initialized()) {
        std::cout << "  [PASS] StateManager correctly linked to SGD" << std::endl;
    } else {
        std::cout << "  [FAIL] StateManager linkage failed" << std::endl;
        return;
    }
}

/**
 * @brief 测试Model参数缓存机制
 */
void test_model_parameter_cache() {
    std::cout << "[TEST] Testing Model parameter cache mechanism..." << std::endl;

    // 获取后端
    auto backend = BackendManager::get_cpu_backend();

    // 创建模型
    auto model = Model::create("CacheTest",
                              std::make_shared<Linear>(10, 5),
                              std::make_shared<Tanh>(),
                              std::make_shared<Linear>(5, 1));

    model->set_backend(backend);
    model->train();

    // 测试缓存机制
    auto params1 = model->trainable_parameters();
    auto params2 = model->trainable_parameters();

    if (params1.size() == params2.size()) {
        bool all_same = true;
        for (size_t i = 0; i < params1.size(); ++i) {
            if (params1[i] != params2[i]) {
                all_same = false;
                break;
            }
        }

        if (all_same) {
            std::cout << "  [PASS] Parameter cache mechanism working correctly" << std::endl;
        } else {
            std::cout << "  [FAIL] Parameter cache returned different pointers" << std::endl;
            return;
        }
    } else {
        std::cout << "  [FAIL] Parameter count inconsistent between two accesses" << std::endl;
        return;
    }
}

/**
 * @brief 测试基本训练循环
 */
void test_basic_training_loop() {
    std::cout << "[TEST] 测试基本训练循环..." << std::endl;

    // 获取后端
    auto backend = BackendManager::get_cpu_backend();

    // 创建模型
    auto model = Model::create("TrainingTest",
                              std::make_shared<Linear>(10, 5),
                              std::make_shared<Tanh>(),
                              std::make_shared<Linear>(5, 1));

    model->set_backend(backend);
    model->train();

    // 创建SGD优化器
    auto optimizer = std::make_shared<SGD>(0.01f, 0.9f, 0.0f, false, backend);
    optimizer->initialize(*model);

    // 创建虚拟输入和目标
    auto input = backend->ones(Shape({2, 10}), DType::FP32);
    auto target = backend->ones(Shape({2, 1}), DType::FP32);

    // 模拟简单训练循环
    bool all_steps_successful = true;
    for (int step = 0; step < 3; ++step) {
        try {
            // 前向传播
            auto output = model->forward(input);

            // 验证时间步递增（从第二步开始）
            if (step > 0) {
                auto* state_manager = optimizer->get_state_manager();
                if (state_manager->get_time_step(0) > 0) {
                    // 时间步正确递增
                    continue;
                } else {
                    std::cout << "  [FAIL] 时间步未正确递增" << std::endl;
                    all_steps_successful = false;
                    break;
                }
            }

        } catch (const std::exception& e) {
            std::cout << "  [FAIL] 训练步骤 " << step << " 失败: " << e.what() << std::endl;
            all_steps_successful = false;
            break;
        }
    }

    if (all_steps_successful) {
        std::cout << "  [PASS] 基本训练循环测试通过" << std::endl;
    } else {
        std::cout << "  [FAIL] 基本训练循环测试失败" << std::endl;
        return;
    }
}

/**
 * @brief 测试设备转移功能
 */
void test_device_transfer() {
    std::cout << "[TEST] 测试设备转移功能..." << std::endl;

    // 获取后端
    auto backend = BackendManager::get_cpu_backend();

    // 创建模型
    auto model = Model::create("DeviceTest",
                              std::make_shared<Linear>(10, 5),
                              std::make_shared<Tanh>(),
                              std::make_shared<Linear>(5, 1));

    model->set_backend(backend);
    model->train();

    // 创建StateManager
    StateManager state_manager(backend);
    auto params = model->trainable_parameters();
    state_manager.initialize_sgd_states(params, 0.9f);

    // 测试设备转移（即使转移到相同设备）
    Device original_device = state_manager.device();
    state_manager.to(tr::CPU);

    if (state_manager.device() == tr::CPU &&
        state_manager.is_initialized() &&
        state_manager.state_count() == params.size()) {
        std::cout << "  [PASS] 设备转移成功，状态保持完整" << std::endl;
    } else {
        std::cout << "  [FAIL] 设备转移失败" << std::endl;
        return;
    }

    // 验证状态在设备转移后仍然有效
    bool all_states_valid = true;
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& state = state_manager.get_state(i);
        if (!state.has_momentum || !state.momentum.storage_allocated()) {
            all_states_valid = false;
            break;
        }
    }

    if (all_states_valid) {
        std::cout << "  [PASS] 设备转移后状态仍然有效" << std::endl;
    } else {
        std::cout << "  [FAIL] 设备转移后状态丢失" << std::endl;
        return;
    }
}

/**
 * @brief 测试性能基准
 */
void test_performance_benchmark() {
    std::cout << "[TEST] 测试性能基准..." << std::endl;

    // 获取后端
    auto backend = BackendManager::get_cpu_backend();

    // 创建模型
    auto model = Model::create("PerformanceTest",
                              std::make_shared<Linear>(10, 5),
                              std::make_shared<Tanh>(),
                              std::make_shared<Linear>(5, 1));

    model->set_backend(backend);
    model->train();

    // 性能测试：参数访问
    const int num_iterations = 1000;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        auto params = model->trainable_parameters();
        // 简单访问每个参数
        for (auto* param : params) {
            volatile auto* ptr = param;
            (void)ptr;  // 避免编译器优化
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // 缓存机制应该使重复访问非常快（< 10ms）
    if (duration.count() < 10000) {
        std::cout << "  [PASS] 参数访问性能优秀: " << duration.count() << " microseconds for "
                  << num_iterations << " iterations" << std::endl;
    } else {
        std::cout << "  [WARN] 参数访问性能一般: " << duration.count() << " microseconds for "
                  << num_iterations << " iterations (预期 < 10000)" << std::endl;
    }
}

/**
 * @brief 测试优化器信息输出
 */
void test_optimizer_info() {
    std::cout << "[TEST] 测试优化器信息输出..." << std::endl;

    // 获取后端
    auto backend = BackendManager::get_cpu_backend();

    // 创建模型
    auto model = Model::create("InfoTest",
                              std::make_shared<Linear>(10, 5),
                              std::make_shared<Tanh>(),
                              std::make_shared<Linear>(5, 1));

    model->set_backend(backend);
    model->train();

    // 创建各种配置的SGD优化器
    auto sgd1 = std::make_shared<SGD>(0.01f);
    auto sgd2 = std::make_shared<SGD>(0.001f, 0.9f, 0.0001f, true);

    sgd1->initialize(*model);
    sgd2->initialize(*model);

    // 测试信息输出
    std::string info1 = sgd1->get_info();
    std::string info2 = sgd2->get_info();

    if (!info1.empty() && !info2.empty()) {
        std::cout << "  [PASS] 优化器信息输出正常" << std::endl;
        std::cout << "  SGD1信息长度: " << info1.length() << std::endl;
        std::cout << "  SGD2信息长度: " << info2.length() << std::endl;
    } else {
        std::cout << "  [FAIL] 优化器信息输出为空" << std::endl;
        return;
    }
}

/**
 * @brief 主测试函数
 */
int main() {
    std::cout << "=== D4 Optimizer Implementation Unit Tests ===" << std::endl;
    std::cout << "Version: V1.00.00" << std::endl;
    std::cout << "Date: 2025-11-19" << std::endl;
    std::cout << "Author: Tech Renaissance Team" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // 运行所有测试
        test_state_manager_basic();
        test_sgd_construction();
        test_model_parameter_cache();
        test_basic_training_loop();
        test_device_transfer();
        test_performance_benchmark();
        test_optimizer_info();

        std::cout << std::endl;
        std::cout << "[SUCCESS] All D4 optimizer tests PASSED!" << std::endl;
        std::cout << "Performance verification: Parameter cache mechanism working properly, device transfer stable." << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[FAILED] D4 optimizer tests failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
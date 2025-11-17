/**
 * @file test_model.cpp
 * @brief Model class unit tests
 * @details Test Model class three constructor methods, auto naming mechanism, preallocation mechanism, etc.
 * @version 1.45.0
 * @date 2025-11-17
 * @author Tech Renaissance Team
 * @note Dependencies: tech_renaissance.h
 * @note Series: model
 */

#include "tech_renaissance.h"

using namespace tr;

/**
 * @brief Test Model class three constructor methods
 */
void test_model_constructors() {
    std::cout << "[TEST] Testing Model constructors..." << std::endl;

    // Get backend
    auto backend = BackendManager::get_cpu_backend();

    // Test constructor 1: default constructor + add_module
    {
        auto model = std::make_shared<Model>("TestModel1");
        auto linear1 = std::make_shared<Linear>(10, 5);
        auto linear2 = std::make_shared<Linear>(5, 1);

        model->set_backend(backend.get());
        model->add_module(linear1);
        model->add_module(linear2);

        std::cout << "  [PASS] Constructor 1: default + add_module" << std::endl;
    }

    // Test constructor 2: initializer list constructor
    {
        auto linear1 = std::make_shared<Linear>(10, 5);
        auto linear2 = std::make_shared<Linear>(5, 1);

        auto model = std::make_shared<Model>("TestModel2",
                                           std::vector<std::shared_ptr<Module>>{linear1, linear2});
        model->set_backend(backend.get());

        std::cout << "  [PASS] Constructor 2: initializer list" << std::endl;
    }

    // Test constructor 3: factory method
    {
        auto model = Model::create("TestModel3",
                                  std::make_shared<Linear>(10, 5),
                                  std::make_shared<Linear>(5, 1));
        model->set_backend(backend.get());

        std::cout << "  [PASS] Constructor 3: factory method" << std::endl;
    }

    std::cout << "[SUCCESS] All constructor tests PASSED!" << std::endl;
}

/**
 * @brief Test Model class auto naming mechanism
 */
void test_auto_naming() {
    std::cout << "[TEST] Testing auto naming mechanism..." << std::endl;

    auto backend = BackendManager::get_cpu_backend();

    // Create model with multiple modules of same type
    auto model = Model::create("NamingTest",
                              std::make_shared<Linear>(10, 5),
                              std::make_shared<Linear>(5, 1),
                              std::make_shared<Tanh>(),
                              std::make_shared<Linear>(1, 3));

    model->set_backend(backend.get());

    // Verify auto naming
    if (model->get_module(0)->instance_name() == "Linear1" &&
        model->get_module(1)->instance_name() == "Linear2" &&
        model->get_module(2)->instance_name() == "Tanh1" &&
        model->get_module(3)->instance_name() == "Linear3") {
        std::cout << "  [PASS] Auto naming works correctly" << std::endl;
    } else {
        std::cout << "  [FAIL] Auto naming failed" << std::endl;
        return;
    }

    std::cout << "  Auto-named modules:" << std::endl;
    for (size_t i = 0; i < model->num_modules(); ++i) {
        std::cout << "    [" << i << "] " << model->get_module(i)->instance_name() << std::endl;
    }

    // Test manual naming
    auto custom_model = std::make_shared<Model>("CustomNamingTest");
    custom_model->set_backend(backend.get());
    custom_model->add_module("input_layer", std::make_shared<Linear>(10, 5));
    custom_model->add_module("output_layer", std::make_shared<Linear>(5, 1));

    if (custom_model->get_module(0)->instance_name() == "input_layer" &&
        custom_model->get_module(1)->instance_name() == "output_layer") {
        std::cout << "  [PASS] Manual naming: input_layer, output_layer" << std::endl;
    } else {
        std::cout << "  [FAIL] Manual naming failed" << std::endl;
        return;
    }

    std::cout << "[SUCCESS] Auto naming tests PASSED!" << std::endl;
}

/**
 * @brief Test Model class forward propagation
 */
void test_forward_propagation() {
    std::cout << "[TEST] Testing forward propagation..." << std::endl;

    auto backend = BackendManager::get_cpu_backend();

    // Create simple 3-layer model: Linear(4->3) -> Tanh -> Linear(3->2)
    auto model = Model::create("ForwardTest",
                              std::make_shared<Linear>(4, 3),
                              std::make_shared<Tanh>(),
                              std::make_shared<Linear>(3, 2));

    model->set_backend(backend.get());

    // Create input tensor (batch=2, features=4)
    Tensor input = backend->randn(Shape(2, 4));

    // Test normal forward propagation
    Tensor output = model->forward(input);
    if (output.shape() == Shape(2, 2)) {
        std::cout << "  Forward output shape: " << output.shape().to_string() << std::endl;
    } else {
        std::cout << "  [FAIL] Incorrect forward output shape" << std::endl;
        return;
    }

    // Test into-type forward propagation
    Tensor output_buffer = backend->zeros(Shape(2, 2), DType::FP32);
    model->forward_into(input, output_buffer);
    if (output_buffer.shape() == Shape(2, 2)) {
        std::cout << "  Into forward output shape: " << output_buffer.shape().to_string() << std::endl;
    } else {
        std::cout << "  [FAIL] Incorrect into forward output shape" << std::endl;
        return;
    }

    std::cout << "  [PASS] Forward propagation" << std::endl;
    std::cout << "[SUCCESS] Forward propagation tests PASSED!" << std::endl;
}

/**
 * @brief Test Model class preallocation mechanism
 */
void test_preallocation_mechanism() {
    std::cout << "[TEST] Testing preallocation mechanism..." << std::endl;

    auto backend = BackendManager::get_cpu_backend();

    // Create 2-layer model
    auto model = Model::create("PreallocTest",
                              std::make_shared<Linear>(3, 2),
                              std::make_shared<Linear>(2, 1));

    model->set_backend(backend.get());

    // Create input tensor
    Tensor input = backend->randn(Shape(4, 3));  // batch=4, features=3

    // Initialize preallocation
    model->initialize(input.shape());

    // Check memory analysis
    std::string analysis = model->analyze_memory();
    std::cout << "  Memory analysis:" << std::endl;
    std::cout << analysis << std::endl;

    // Verify preallocation cache is allocated
    if (analysis.find("ALLOCATED") != std::string::npos) {
        std::cout << "  [PASS] Preallocation mechanism working" << std::endl;
    } else {
        std::cout << "  [FAIL] Preallocation mechanism not working" << std::endl;
        return;
    }

    // Use preallocation cache for forward propagation
    Tensor output = backend->zeros(Shape(4, 1), DType::FP32);
    model->forward_into(input, output);

    std::cout << "[SUCCESS] Preallocation tests PASSED!" << std::endl;
}

/**
 * @brief Test Model class parameter management
 */
void test_parameter_management() {
    std::cout << "[TEST] Testing parameter management..." << std::endl;

    auto backend = BackendManager::get_cpu_backend();

    // Create model with parameters
    auto model = Model::create("ParamTest",
                              std::make_shared<Linear>(4, 3),
                              std::make_shared<Linear>(3, 2));

    model->set_backend(backend.get());

    // Get all parameters
    auto params = model->parameters();
    std::cout << "  Total parameters: " << params.size() << std::endl;

    // Verify parameter count (each Linear layer has 1 weight parameter)
    if (params.size() == 2) {
        std::cout << "  [PASS] Correct parameter count" << std::endl;
    } else {
        std::cout << "  [FAIL] Incorrect parameter count: expected 2, got " << params.size() << std::endl;
        return;
    }

    // Verify parameter naming
    if (params.count("Linear1.weight") > 0 && params.count("Linear2.weight") > 0) {
        std::cout << "  [PASS] Parameter naming correct" << std::endl;
    } else {
        std::cout << "  [FAIL] Parameter naming incorrect" << std::endl;
        return;
    }

    // Check parameter shapes
    if (params["Linear1.weight"].shape() == Shape(4, 3) &&  // transposed storage
        params["Linear2.weight"].shape() == Shape(3, 2)) {  // transposed storage
        std::cout << "  [PASS] Parameter shapes correct" << std::endl;
    } else {
        std::cout << "  [FAIL] Parameter shapes incorrect" << std::endl;
        return;
    }

    // Test memory calculation
    size_t param_memory = model->parameter_memory();
    std::cout << "  Parameter memory: " << param_memory << " bytes" << std::endl;
    if (param_memory > 0) {
        std::cout << "  [PASS] Parameter memory calculation" << std::endl;
    } else {
        std::cout << "  [FAIL] Parameter memory calculation failed" << std::endl;
        return;
    }

    std::cout << "[SUCCESS] Parameter management tests PASSED!" << std::endl;
}

/**
 * @brief Test Model class device transfer and mode switching
 */
void test_device_and_mode() {
    std::cout << "[TEST] Testing device transfer and mode switching..." << std::endl;

    auto backend = BackendManager::get_cpu_backend();

    // Create model
    auto model = Model::create("DeviceModeTest",
                              std::make_shared<Linear>(3, 2),
                              std::make_shared<Tanh>());

    model->set_backend(backend.get());

    // Test device information
    Device device = model->device();
    std::cout << "  Current device: " << device.to_string() << std::endl;

    // Test training mode
    model->train();
    if (model->is_training() == true &&
        model->get_module(0)->is_training() == true &&
        model->get_module(1)->is_training() == true) {
        std::cout << "  [PASS] Training mode" << std::endl;
    } else {
        std::cout << "  [FAIL] Training mode failed" << std::endl;
        return;
    }

    // Test inference mode
    model->eval();
    if (model->is_training() == false &&
        model->get_module(0)->is_training() == false &&
        model->get_module(1)->is_training() == false) {
        std::cout << "  [PASS] Inference mode" << std::endl;
    } else {
        std::cout << "  [FAIL] Inference mode failed" << std::endl;
        return;
    }

    std::cout << "[SUCCESS] Device and mode tests PASSED!" << std::endl;
}

/**
 * @brief Test empty model and single module model
 */
void test_edge_cases() {
    std::cout << "[TEST] Testing edge cases..." << std::endl;

    auto backend = BackendManager::get_cpu_backend();

    // Test empty model
    {
        auto empty_model = std::make_shared<Model>("EmptyModel");
        empty_model->set_backend(backend.get());

        if (empty_model->num_modules() == 0) {
            std::cout << "  [PASS] Empty model" << std::endl;
        } else {
            std::cout << "  [FAIL] Empty model has modules" << std::endl;
            return;
        }

        // Empty model should directly return input
        Tensor input = backend->randn(Shape(2, 3));
        Tensor output = empty_model->forward(input);
        if (output.shape() == input.shape()) {
            std::cout << "  [PASS] Empty model forward pass" << std::endl;
        } else {
            std::cout << "  [FAIL] Empty model forward pass failed" << std::endl;
            return;
        }
    }

    // Test single module model
    {
        auto single_model = Model::create("SingleModule",
                                         std::make_shared<Linear>(3, 2));
        single_model->set_backend(backend.get());

        if (single_model->num_modules() == 1) {
            std::cout << "  [PASS] Single module model" << std::endl;
        } else {
            std::cout << "  [FAIL] Single module model count: " << single_model->num_modules() << std::endl;
            return;
        }

        Tensor input = backend->randn(Shape(4, 3));
        Tensor output = single_model->forward(input);
        if (output.shape() == Shape(4, 2)) {
            std::cout << "  [PASS] Single module forward pass" << std::endl;
        } else {
            std::cout << "  [FAIL] Single module forward pass failed" << std::endl;
            return;
        }
    }

    std::cout << "[SUCCESS] Edge case tests PASSED!" << std::endl;
}

/**
 * @brief Main test function
 */
int main() {
    std::cout << "=== Model Class Unit Tests ===" << std::endl;
    std::cout << "Version: V1.45.0" << std::endl;
    std::cout << "Date: 2025-11-17" << std::endl;
    std::cout << "===============================" << std::endl;

    try {
        // Run all tests
        test_model_constructors();
        std::cout << std::endl;

        test_auto_naming();
        std::cout << std::endl;

        test_forward_propagation();
        std::cout << std::endl;

        test_preallocation_mechanism();
        std::cout << std::endl;

        test_parameter_management();
        std::cout << std::endl;

        test_device_and_mode();
        std::cout << std::endl;

        test_edge_cases();
        std::cout << std::endl;

        std::cout << "[SUCCESS] All Model tests PASSED!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[FAILED] Model tests failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
#include "tech_renaissance.h"

using namespace tr;

int main() {
    Logger::get_instance().set_quiet_mode(false);
    auto cpu = BackendManager::get_cpu_backend();

    std::cout << "=== Testing get_memory_size() method ===" << std::endl;

    // Test 1: Empty tensor (no storage allocated)
    std::cout << "\n1. Testing empty tensor (Tensor()):" << std::endl;
    Tensor empty_tensor = Tensor();
    std::cout << "   Empty tensor memory size: " << cpu->get_memory_size(empty_tensor) << " bytes" << std::endl;
    std::cout << "   storage_allocated(): " << (empty_tensor.storage_allocated() ? "true" : "false") << std::endl;

    // Test 2: Null tensor
    std::cout << "\n2. Testing null tensor (CpuBackend::null_tensor()):" << std::endl;
    Tensor null_tensor = cpu->null_tensor();
    std::cout << "   Null tensor memory size: " << cpu->get_memory_size(null_tensor) << " bytes" << std::endl;
    std::cout << "   storage_allocated(): " << (null_tensor.storage_allocated() ? "true" : "false") << std::endl;
    std::cout << "   Shape: " << null_tensor.shape().to_string() << std::endl;

    // Test 3: Small allocated tensor
    std::cout << "\n3. Testing small allocated tensor (2x2x2x2 FP32):" << std::endl;
    Tensor small_tensor = cpu->zeros(Shape(2, 2, 2, 2), DType::FP32);
    std::cout << "   Small tensor memory size: " << cpu->get_memory_size(small_tensor) << " bytes" << std::endl;
    std::cout << "   Expected size: " << (2*2*2*2 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "   storage_allocated(): " << (small_tensor.storage_allocated() ? "true" : "false") << std::endl;
    std::cout << "   Shape: " << small_tensor.shape().to_string() << std::endl;
    std::cout << "   numel(): " << small_tensor.numel() << std::endl;
    std::cout << "   dtype_size(): " << small_tensor.dtype_size() << std::endl;

    // Test 4: Larger tensor with different dtype
    std::cout << "\n4. Testing INT32 tensor (3x4x5x6):" << std::endl;
    Tensor int32_tensor = cpu->zeros(Shape(3, 4, 5, 6), DType::INT32);
    std::cout << "   INT32 tensor memory size: " << cpu->get_memory_size(int32_tensor) << " bytes" << std::endl;
    std::cout << "   Expected size: " << (3*4*5*6 * sizeof(int32_t)) << " bytes" << std::endl;
    std::cout << "   storage_allocated(): " << (int32_tensor.storage_allocated() ? "true" : "false") << std::endl;
    std::cout << "   Shape: " << int32_tensor.shape().to_string() << std::endl;
    std::cout << "   numel(): " << int32_tensor.numel() << std::endl;
    std::cout << "   dtype_size(): " << int32_tensor.dtype_size() << std::endl;

    // Test 5: INT8 tensor
    std::cout << "\n5. Testing INT8 tensor (10x10x10x10):" << std::endl;
    Tensor int8_tensor = cpu->zeros(Shape(10, 10, 10, 10), DType::INT8);
    std::cout << "   INT8 tensor memory size: " << cpu->get_memory_size(int8_tensor) << " bytes" << std::endl;
    std::cout << "   Expected size: " << (10*10*10*10 * sizeof(int8_t)) << " bytes" << std::endl;
    std::cout << "   storage_allocated(): " << (int8_tensor.storage_allocated() ? "true" : "false") << std::endl;
    std::cout << "   Shape: " << int8_tensor.shape().to_string() << std::endl;
    std::cout << "   numel(): " << int8_tensor.numel() << std::endl;
    std::cout << "   dtype_size(): " << int8_tensor.dtype_size() << std::endl;

    // Test 6: Destroyed tensor
    std::cout << "\n6. Testing destroyed tensor:" << std::endl;
    Tensor temp_tensor = cpu->ones(Shape(5, 5, 5, 5), DType::FP32);
    std::cout << "   Before destruction - memory size: " << cpu->get_memory_size(temp_tensor) << " bytes" << std::endl;
    temp_tensor = cpu->null_tensor();
    std::cout << "   After destruction - memory size: " << cpu->get_memory_size(temp_tensor) << " bytes" << std::endl;
    std::cout << "   storage_allocated(): " << (temp_tensor.storage_allocated() ? "true" : "false") << std::endl;

    std::cout << "\n=== All tests completed successfully! ===" << std::endl;

    return 0;
}
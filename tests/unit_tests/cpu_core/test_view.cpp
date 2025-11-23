/**
 * @file test_view.cpp
 * @brief Tensor视图功能测试
 * @details 测试Tensor的view方法，包括内存共享、形状变换、连续性检查等功能
 * @version 1.00.00
 * @date 2025-11-16
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance.h
 * @note 所属系列: unit tests
 */

#include "tech_renaissance.h"
#include <iostream>
#include <cassert>
#include <stdexcept>

using namespace tr;

void test_basic_view() {
    std::cout << "Testing basic view functionality..." << std::endl;

    // 获取CPU后端
    auto cpu_backend = BackendManager::get_cpu_backend();

    // 创建一个2x3x4的张量
    Shape original_shape(2, 3, 4);
    Tensor original = cpu_backend->zeros(original_shape, DType::FP32);

    // 填充一些数据用于测试
    cpu_backend->fill(original, 1.0f);

    std::cout << "Original tensor shape: " << original.shape().to_string() << std::endl;
    std::cout << "Original tensor is_view: " << original.is_view() << std::endl;
    std::cout << "Original tensor is_contiguous: " << original.is_contiguous() << std::endl;
    std::cout << "Original tensor numel: " << original.numel() << std::endl;

    // 创建一个view，改变形状为6x4
    Shape new_shape(6, 4);
    Tensor view_tensor = cpu_backend->view(original, new_shape);

    std::cout << "View tensor shape: " << view_tensor.shape().to_string() << std::endl;
    std::cout << "View tensor is_view: " << view_tensor.is_view() << std::endl;
    std::cout << "View tensor is_contiguous: " << view_tensor.is_contiguous() << std::endl;
    std::cout << "View tensor numel: " << view_tensor.numel() << std::endl;

    // 验证元素数量相同
    assert(original.numel() == view_tensor.numel());

    // 验证view张量确实是视图
    assert(view_tensor.is_view());

    // 验证内存共享（Storage的引用计数）
    std::cout << "Original storage use count: " << original.storage().use_count() << std::endl;
    std::cout << "View storage use count: " << view_tensor.storage().use_count() << std::endl;

    // 由于有view，storage的引用计数应该至少为2
    assert(original.storage().use_count() >= 2);
    assert(view_tensor.storage().use_count() >= 2);

    // 验证storage是同一个对象
    assert(original.storage() == view_tensor.storage());

    std::cout << "[PASS] Basic view test passed!" << std::endl << std::endl;
}

void test_view_memory_sharing() {
    std::cout << "Testing view memory sharing..." << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 创建原始张量
    Tensor original = cpu_backend->zeros(Shape(2, 6), DType::FP32);
    cpu_backend->fill(original, 2.0f);

    // 创建view
    Tensor view_tensor = cpu_backend->view(original, Shape(3, 4));

    std::cout << "Before modification:" << std::endl;
    std::cout << "Original storage address: " << original.storage().get() << std::endl;
    std::cout << "View storage address: " << view_tensor.storage().get() << std::endl;
    std::cout << "Storage objects are the same: " << (original.storage() == view_tensor.storage()) << std::endl;

    // 修改view张量
    cpu_backend->fill(view_tensor, 5.0f);

    std::cout << "After modifying view:" << std::endl;
    std::cout << "Original storage address: " << original.storage().get() << std::endl;
    std::cout << "View storage address: " << view_tensor.storage().get() << std::endl;
    std::cout << "Storage objects are still the same: " << (original.storage() == view_tensor.storage()) << std::endl;

    // 验证内存共享
    assert(original.storage() == view_tensor.storage());
    assert(original.storage().use_count() >= 2);
    assert(view_tensor.storage().use_count() >= 2);

    std::cout << "[PASS] View memory sharing test passed!" << std::endl << std::endl;
}

void test_view_shape_validation() {
    std::cout << "Testing view shape validation..." << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 创建一个2x3x4的张量（24个元素）
    Tensor original = cpu_backend->zeros(Shape(2, 3, 4), DType::FP32);

    try {
        // 尝试用不匹配的元素数量创建view（24 != 25）
        Tensor invalid_view = cpu_backend->view(original, Shape(5, 5));
        // 如果能执行到这里，说明没有抛出异常，这是错误的
        assert(false && "Expected exception for shape mismatch was not thrown");
    } catch (const TRException& e) {
        std::cout << "[PASS] Correctly caught exception: " << e.what() << std::endl;
    }

    try {
        // 尝试用匹配的元素数量创建view（24 == 24）
        Tensor valid_view = cpu_backend->view(original, Shape(4, 6));
        std::cout << "[PASS] Valid shape (4,6) accepted, new shape: " << valid_view.shape().to_string() << std::endl;
    } catch (const TRException& e) {
        // 这个不应该抛出异常
        assert(false && "Unexpected exception for valid shape");
    }

    std::cout << "[PASS] View shape validation test passed!" << std::endl << std::endl;
}

void test_view_4d_tensor() {
    std::cout << "Testing view with 4D tensors..." << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    // 创建一个4D张量 (N=2, C=3, H=4, W=5)
    Tensor original_4d = cpu_backend->zeros(Shape(2, 3, 4, 5), DType::FP32);
    cpu_backend->fill(original_4d, 3.0f);

    std::cout << "Original 4D tensor shape: " << original_4d.shape().to_string() << std::endl;
    std::cout << "Original 4D tensor numel: " << original_4d.numel() << std::endl;

    // 将4D张量reshape为2D（展平前两个维度）
    Tensor view_2d = cpu_backend->view(original_4d, Shape(6, 20));

    std::cout << "View 2D tensor shape: " << view_2d.shape().to_string() << std::endl;
    std::cout << "View 2D tensor numel: " << view_2d.numel() << std::endl;
    std::cout << "View 2D tensor is_view: " << view_2d.is_view() << std::endl;

    // 验证元素数量相同
    assert(original_4d.numel() == view_2d.numel());
    assert(view_2d.is_view());

    // 再将2D张量reshape为3D
    Tensor view_3d = cpu_backend->view(view_2d, Shape(2, 5, 12));

    std::cout << "View 3D tensor shape: " << view_3d.shape().to_string() << std::endl;
    std::cout << "View 3D tensor numel: " << view_3d.numel() << std::endl;
    std::cout << "View 3D tensor is_view: " << view_3d.is_view() << std::endl;

    // 验证元素数量相同
    assert(view_2d.numel() == view_3d.numel());
    assert(view_3d.is_view());

    std::cout << "[PASS] 4D tensor view test passed!" << std::endl << std::endl;
}

void test_strides_calculation() {
    std::cout << "Testing strides calculation..." << std::endl;

    // 测试Strides类的构造函数
    Shape shape(2, 3, 4);
    Strides strides(shape);

    std::cout << "Shape: " << shape.to_string() << std::endl;
    std::cout << "Strides: " << strides.to_string() << std::endl;
    std::cout << "Stride N: " << strides.n() << std::endl;
    std::cout << "Stride C: " << strides.c() << std::endl;
    std::cout << "Stride H: " << strides.h() << std::endl;
    std::cout << "Stride W: " << strides.w() << std::endl;

    // 验证步长计算的正确性
    // 对于连续存储: stride[W] = 1, stride[H] = W, stride[C] = H*W, stride[N] = C*H*W
    assert(strides.w() == 1);
    assert(strides.h() == 4);      // W
    assert(strides.c() == 12);     // H * W = 4 * 4
    assert(strides.n() == 36);     // C * H * W = 3 * 4 * 4

    // 测试偏移量计算
    int64_t offset = strides.get_offset(1, 2, 3, 2); // 1*36 + 2*12 + 3*4 + 2*1 = 36 + 24 + 12 + 2 = 74
    std::cout << "Offset for (1,2,3,2): " << offset << std::endl;
    assert(offset == 74);

    // 测试连续性检查
    assert(strides.is_contiguous(shape));

    std::cout << "[PASS] Strides calculation test passed!" << std::endl << std::endl;
}

void test_view_lifetime() {
    std::cout << "Testing view lifetime management..." << std::endl;

    auto cpu_backend = BackendManager::get_cpu_backend();

    Tensor original = cpu_backend->zeros(Shape(2, 4), DType::FP32);
    cpu_backend->fill(original, 7.0f);

    std::cout << "Original storage use count: " << original.storage().use_count() << std::endl;

    {
        // 创建作用域，在其中创建view
        Tensor view_tensor = cpu_backend->view(original, Shape(4, 2));
        std::cout << "Inside scope - Original storage use count: " << original.storage().use_count() << std::endl;
        std::cout << "Inside scope - View storage use count: " << view_tensor.storage().use_count() << std::endl;

        // 验证内存共享
        assert(original.storage().use_count() >= 2);
        assert(view_tensor.storage().use_count() >= 2);
        assert(original.storage() == view_tensor.storage());
        assert(view_tensor.is_view());

    } // view_tensor在这里离开作用域，被析构

    std::cout << "After scope - Original storage use count: " << original.storage().use_count() << std::endl;

    // 验证原始张量仍然有效
    assert(!original.is_empty());

    std::cout << "[PASS] View lifetime management test passed!" << std::endl << std::endl;
}

int main() {
    std::cout << "=== Tensor View Functionality Tests ===" << std::endl;
    std::cout << "Testing view implementation with shared memory management" << std::endl << std::endl;

    try {
        test_basic_view();
        test_view_memory_sharing();
        test_view_shape_validation();
        test_view_4d_tensor();
        test_strides_calculation();
        test_view_lifetime();

        std::cout << "[SUCCESS] All view tests passed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
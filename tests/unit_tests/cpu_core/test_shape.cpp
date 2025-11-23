/**
 * @file test_shape.cpp
 * @brief 形状类单元测试
 * @details 测试Shape类的所有功能，包括构造、维度查询、比较运算和特殊方法
 * @version 1.00.00
 * @date 2025-10-25
 * @author 技术觉醒团队
 * @note 依赖项: shape.h, iostream, string
 * @note 所属系列: tests
 */

#include <iostream>
#include <string>
#include <cassert>
#include <sstream>
#include "tech_renaissance.h"

using namespace tr;

// 测试辅助函数
void test_assert(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << std::endl;
        assert(false);
    } else {
        std::cout << "[PASS] " << message << std::endl;
    }
}

// 测试构造函数
void test_constructors() {
    std::cout << "\n=== Testing Constructors ===" << std::endl;

    // 测试默认构造函数（标量）
    Shape scalar;
    test_assert(scalar.ndim() == 0, "Default constructor creates scalar");
    test_assert(scalar.numel() == 1, "Scalar has 1 element");
    test_assert(scalar.to_string() == "()", "Scalar string representation");

    // 测试单参数构造函数
    Shape shape1(5);
    test_assert(shape1.ndim() == 1, "Single parameter creates 1D tensor");
    test_assert(shape1.numel() == 5, "1D tensor numel calculation");
    test_assert(shape1.dim(0) == 5, "1D tensor dim access");
    test_assert(shape1.to_string() == "(5)", "1D tensor string representation");

    // 测试双参数构造函数
    Shape shape2(3, 4);
    test_assert(shape2.ndim() == 2, "Two parameters create 2D tensor");
    test_assert(shape2.numel() == 12, "2D tensor numel calculation");
    test_assert(shape2.dim(0) == 3, "2D tensor first dim access");
    test_assert(shape2.dim(1) == 4, "2D tensor second dim access");
    test_assert(shape2.to_string() == "(3,4)", "2D tensor string representation");

    // 测试三参数构造函数
    Shape shape3(2, 3, 4);
    test_assert(shape3.ndim() == 3, "Three parameters create 3D tensor");
    test_assert(shape3.numel() == 24, "3D tensor numel calculation");
    test_assert(shape3.dim(0) == 2, "3D tensor first dim access");
    test_assert(shape3.dim(1) == 3, "3D tensor second dim access");
    test_assert(shape3.dim(2) == 4, "3D tensor third dim access");
    test_assert(shape3.to_string() == "(2,3,4)", "3D tensor string representation");

    // 测试四参数构造函数
    Shape shape4(1, 2, 3, 4);
    test_assert(shape4.ndim() == 4, "Four parameters create 4D tensor");
    test_assert(shape4.numel() == 24, "4D tensor numel calculation");
    test_assert(shape4.n() == 1, "4D tensor N dimension access");
    test_assert(shape4.c() == 2, "4D tensor C dimension access");
    test_assert(shape4.h() == 3, "4D tensor H dimension access");
    test_assert(shape4.w() == 4, "4D tensor W dimension access");
    test_assert(shape4.to_string() == "(1,2,3,4)", "4D tensor string representation");

    // 测试初始化列表构造函数
    Shape list1({5});
    test_assert(list1.ndim() == 1 && list1.numel() == 5, "Initializer list with 1 element");

    Shape list2({3, 4});
    test_assert(list2.ndim() == 2 && list2.numel() == 12, "Initializer list with 2 elements");

    Shape list3({2, 3, 4});
    test_assert(list3.ndim() == 3 && list3.numel() == 24, "Initializer list with 3 elements");

    Shape list4({1, 2, 3, 4});
    test_assert(list4.ndim() == 4 && list4.numel() == 24, "Initializer list with 4 elements");
}

// 测试异常情况
void test_exceptions() {
    std::cout << "\n=== Testing Exception Handling ===" << std::endl;

    // 测试负数维度
    try {
        Shape negative(-5);
        test_assert(false, "Negative dimension should throw exception");
    } catch (const TRException&) {
        test_assert(true, "Negative dimension throws TRException");
    }

    // 测试零维度（多参数）
    try {
        Shape zero_dim(3, 0, 4);
        test_assert(false, "Zero dimension should throw exception");
    } catch (const TRException&) {
        test_assert(true, "Zero dimension throws TRException");
    }

    // 测试过多维度
    try {
        Shape too_many({1, 2, 3, 4, 5});
        test_assert(false, "Too many dimensions should throw exception");
    } catch (const TRException&) {
        test_assert(true, "Too many dimensions throws TRException");
    }

    // 测试超出范围的维度访问
    Shape shape(3, 4);
    try {
        int32_t dim = shape.dim(5);
        (void)dim; // 避免未使用变量警告
        test_assert(false, "Out of range dim access should throw exception");
    } catch (const TRException&) {
        test_assert(true, "Out of range dim access throws TRException");
    }
}

// 测试比较运算符
void test_comparison() {
    std::cout << "\n=== Testing Comparison Operators ===" << std::endl;

    Shape shape1(3, 4);
    Shape shape2(3, 4);
    Shape shape3(3, 5);
    Shape shape4(4, 3);

    test_assert(shape1 == shape2, "Equal shapes comparison");
    test_assert(!(shape1 == shape3), "Different shapes comparison");
    test_assert(shape1 != shape3, "Not equal comparison");
    test_assert(shape3 != shape4, "Different shapes not equal");

    // 测试标量比较
    Shape scalar1;
    Shape scalar2;
    test_assert(scalar1 == scalar2, "Scalar equality");
    test_assert(!(scalar1 != scalar2), "Scalar inequality");
}

// 测试特殊方法
void test_special_methods() {
    std::cout << "\n=== Testing Special Methods ===" << std::endl;

    // 测试标量检查
    Shape scalar;
    Shape tensor(3, 4);
    test_assert(scalar.is_scalar(), "Scalar detection");
    test_assert(!tensor.is_scalar(), "Non-scalar detection");

    // 测试矩阵乘法兼容性
    Shape mat1(4, 6);     // [4,6]
    Shape mat2(6, 8);     // [6,8] - 兼容
    Shape mat3(7, 8);     // [7,8] - 不兼容
    Shape mat4(4, 6, 1);  // [4,6,1] - 不兼容

    test_assert(mat1.is_matmul_compatible(mat2), "Compatible matrix multiplication");
    test_assert(!mat1.is_matmul_compatible(mat3), "Incompatible matrix multiplication (wrong inner dimension)");
    test_assert(!mat1.is_matmul_compatible(mat4), "Incompatible matrix multiplication (wrong dimensions)");

    // 测试专家指出的具体修复案例
    Shape shape_a(3, 4);  // 3行4列 [m,k]
    Shape shape_b(4, 5);  // 4行5列 [k,n]
    Shape shape_c(5, 4);  // 5行4列 [k',n] (不兼容)

    test_assert(shape_a.is_matmul_compatible(shape_b), "Expert fix: 3x4 should be compatible with 4x5");
    test_assert(!shape_a.is_matmul_compatible(shape_c), "Expert fix: 3x4 should not be compatible with 5x4");

    // 测试广播兼容性
    Shape broadcast_src(1, 3);
    Shape broadcast_target(4, 3);
    Shape broadcast_target2(4, 5);

    test_assert(broadcast_src.is_broadcastable_to(broadcast_target), "Broadcastable to target");
    test_assert(!broadcast_src.is_broadcastable_to(broadcast_target2), "Not broadcastable to target");

    // 测试低维到高维的广播
    Shape low_dim(3);
    Shape high_dim(4, 3);
    test_assert(low_dim.is_broadcastable_to(high_dim), "Low dimension broadcast to high dimension");

    // 测试标量广播
    Shape scalar_broadcast;
    Shape any_target(2, 3, 4);
    test_assert(scalar_broadcast.is_broadcastable_to(any_target), "Scalar broadcast to any shape");
}

// 测试输出流运算符
void test_output_stream() {
    std::cout << "\n=== Testing Output Stream Operator ===" << std::endl;

    Shape scalar;
    Shape shape1(5);
    Shape shape2(3, 4);
    Shape shape3(2, 3, 4);
    Shape shape4(1, 2, 3, 4);

    std::ostringstream oss1, oss2, oss3, oss4, oss5;
    oss1 << scalar;
    oss2 << shape1;
    oss3 << shape2;
    oss4 << shape3;
    oss5 << shape4;

    test_assert(oss1.str() == "()", "Scalar output stream");
    test_assert(oss2.str() == "(5)", "1D tensor output stream");
    test_assert(oss3.str() == "(3,4)", "2D tensor output stream");
    test_assert(oss4.str() == "(2,3,4)", "3D tensor output stream");
    test_assert(oss5.str() == "(1,2,3,4)", "4D tensor output stream");
}

// 测试拷贝和赋值
void test_copy_assignment() {
    std::cout << "\n=== Testing Copy and Assignment ===" << std::endl;

    Shape original(2, 3, 4);
    Shape copy(original);
    Shape assigned;
    assigned = original;

    test_assert(copy == original, "Copy constructor");
    test_assert(assigned == original, "Assignment operator");
    test_assert(copy.ndim() == original.ndim(), "Copied object ndim");
    test_assert(copy.numel() == original.numel(), "Copied object numel");
    test_assert(assigned.ndim() == original.ndim(), "Assigned object ndim");
    test_assert(assigned.numel() == original.numel(), "Assigned object numel");
}

// 测试边界情况
void test_edge_cases() {
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;

    // 测试大数
    Shape large_shape(1000, 2000);
    test_assert(large_shape.numel() == 2000000, "Large number handling");

    // 测试维度为1的情况
    Shape ones_shape(1, 1, 1, 1);
    test_assert(ones_shape.ndim() == 4, "All ones shape ndim");
    test_assert(ones_shape.numel() == 1, "All ones shape numel");

    // 测试只有一个维度为1的广播
    Shape broadcast_one(1, 100);
    Shape broadcast_target(50, 100);
    test_assert(broadcast_one.is_broadcastable_to(broadcast_target), "Single dimension broadcasting");
}

int main() {
    std::cout << "=== Shape Class Unit Tests ===" << std::endl;
    std::cout << "Testing comprehensive functionality of the Shape class" << std::endl;

    try {
        test_constructors();
        test_exceptions();
        test_comparison();
        test_special_methods();
        test_output_stream();
        test_copy_assignment();
        test_edge_cases();

        std::cout << "\n=== All Tests PASSED! ===" << std::endl;
        std::cout << "Shape class implementation is working correctly." << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n[ERROR] Test failed with unknown exception" << std::endl;
        return 1;
    }
}
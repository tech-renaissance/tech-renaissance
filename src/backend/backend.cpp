/**
 * @file backend.cpp
 * @brief Backend基类实现
 * @details 实现Backend基类的默认方法（抛出NotImplementedError）
 * @version 1.43.0
 * @date 2025-11-16
 * @author 技术觉醒团队
 * @note 依赖项: backend.h
 * @note 所属系列: backend
 */

#include "tech_renaissance/backend/backend.h"
#include "tech_renaissance/data/tensor.h"

namespace tr {

/**
 * @brief 定义未实现方法的宏
 * @param method_name 方法名
 * @param return_type 返回类型
 * @param params 参数列表（带括号）
 * @param const_qualifier const限定符（如果方法不是const则为空）
 * @details 生成默认抛出NotImplementedError异常的方法实现
 */
#define DEFINE_NOT_IMPLEMENTED_METHOD(method_name, return_type, params, const_qualifier) \
    return_type Backend::method_name params const_qualifier { \
        throw NotImplementedError("[" + name() + " " #method_name "] Operation NOT implemented!"); \
    }

/**
 * @brief 定义void返回类型未实现方法的宏（重载版本）
 * @param method_name 方法名
 * @param params 参数列表（带括号）
 * @param const_qualifier const限定符（如果方法不是const则为空）
 * @details 为void返回类型的方法提供便利宏
 */
#define DEFINE_NOT_IMPLEMENTED_VOID_METHOD(method_name, params, const_qualifier) \
    void Backend::method_name params const_qualifier { \
        throw NotImplementedError("[" + name() + " " #method_name "] Operation NOT implemented!"); \
    }

// 视图操作
DEFINE_NOT_IMPLEMENTED_METHOD(view, Tensor, (const Tensor& input, const Shape& new_shape), )

// 转置操作
DEFINE_NOT_IMPLEMENTED_METHOD(transpose, Tensor, (const Tensor& input), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(transpose_into, (const Tensor& input, Tensor& output), const)

// 形状变换操作
DEFINE_NOT_IMPLEMENTED_METHOD(reshape, Tensor, (const Tensor& tensor_a, const Shape& shape), )
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(reshape_inplace, (Tensor& tensor_a, const Shape& shape), )
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(reshape_into, (const Tensor& tensor_a, Tensor& result, const Shape& shape), )

// 双曲函数操作
DEFINE_NOT_IMPLEMENTED_METHOD(tanh, Tensor, (const Tensor& tensor_a), )
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(tanh_inplace, (Tensor& tensor_a), )
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(tanh_into, (const Tensor& tensor_a, Tensor& result), )
DEFINE_NOT_IMPLEMENTED_METHOD(dtanh, Tensor, (const Tensor& tensor_a), )
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(dtanh_inplace, (Tensor& tensor_a), )
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(dtanh_into, (const Tensor& tensor_a, Tensor& result), )

DEFINE_NOT_IMPLEMENTED_METHOD(sqrt, Tensor, (const Tensor& input), )
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(sqrt_inplace, (Tensor& input), )
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(sqrt_into, (const Tensor& input, Tensor& output), )
DEFINE_NOT_IMPLEMENTED_METHOD(square, Tensor, (const Tensor& input), )
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(square_inplace, (Tensor& input), )
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(square_into, (const Tensor& input, Tensor& output), )

// 交叉熵损失函数
DEFINE_NOT_IMPLEMENTED_METHOD(crossentropy, float, (const Tensor& pred, const Tensor& label, std::string reduction), )

// softmax操作
DEFINE_NOT_IMPLEMENTED_METHOD(softmax, Tensor, (const Tensor& input, int dim), )
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(softmax_into, (const Tensor& input, Tensor& output, int dim), )

// One-hot编码操作
DEFINE_NOT_IMPLEMENTED_METHOD(one_hot, Tensor, (const Tensor& label, int32_t num_classes, float label_smoothing), )
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(one_hot_into, (const Tensor& label, Tensor& result, int32_t num_classes, float label_smoothing), )

// 张量运算
DEFINE_NOT_IMPLEMENTED_METHOD(add, Tensor, (const Tensor& a, const Tensor& b), const)
DEFINE_NOT_IMPLEMENTED_METHOD(mul, Tensor, (const Tensor& a, const Tensor& b), const)

// 张量运算（into版本）
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(add_into, (const Tensor& a, const Tensor& b, Tensor& result), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(minus_into, (const Tensor& a, const Tensor& b, Tensor& result), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(div_into, (const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(sum_into, (const Tensor& tensor_a, Tensor& result, int32_t dim, bool keep_dim), const)

// 标量运算（tensor + scalar）
DEFINE_NOT_IMPLEMENTED_METHOD(add, Tensor, (const Tensor& input, float scalar), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(add_inplace, (Tensor& input, float scalar), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(add_into, (const Tensor& input, float scalar, Tensor& output), const)

// 标量运算（tensor * scalar）
DEFINE_NOT_IMPLEMENTED_METHOD(mul, Tensor, (const Tensor& input, float scalar), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(mul_inplace, (Tensor& input, float scalar), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(mul_into, (const Tensor& input, float scalar, Tensor& output), const)

// 标量运算（tensor - scalar）
DEFINE_NOT_IMPLEMENTED_METHOD(minus, Tensor, (const Tensor& input, float scalar), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(minus_inplace, (Tensor& input, float scalar), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(minus_into, (const Tensor& input, float scalar, Tensor& output), const)

// 标量运算（scalar - tensor）
DEFINE_NOT_IMPLEMENTED_METHOD(minus, Tensor, (float scalar, const Tensor& input), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(minus_inplace, (float scalar, Tensor& input), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(minus_into, (float scalar, const Tensor& input, Tensor& output), const)

// 标量乘加运算
DEFINE_NOT_IMPLEMENTED_METHOD(mac, Tensor, (const Tensor& input, float scalar_x, float scalar_y), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(mac_inplace, (Tensor& input, float scalar_x, float scalar_y), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(mac_into, (const Tensor& input, float scalar_x, float scalar_y, Tensor& output), const)

// 标量裁剪运算
DEFINE_NOT_IMPLEMENTED_METHOD(clamp, Tensor, (const Tensor& input, float min_val, float max_val), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(clamp_inplace, (Tensor& input, float min_val, float max_val), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(clamp_into, (const Tensor& input, float min_val, float max_val, Tensor& output), const)

// 广播运算
DEFINE_NOT_IMPLEMENTED_METHOD(add_broadcast, Tensor, (const Tensor& tensor_a, const Tensor& tensor_b), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(add_broadcast_into, (const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result), const)
DEFINE_NOT_IMPLEMENTED_METHOD(minus_broadcast, Tensor, (const Tensor& tensor_a, const Tensor& tensor_b), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(minus_broadcast_into, (const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result), const)
DEFINE_NOT_IMPLEMENTED_METHOD(mul_broadcast, Tensor, (const Tensor& tensor_a, const Tensor& tensor_b), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(mul_broadcast_into, (const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result), const)

// 张量创建方法（const重载）
DEFINE_NOT_IMPLEMENTED_METHOD(empty, Tensor, (const Shape& shape, DType dtype), const)

// 均匀分布随机张量创建方法
DEFINE_NOT_IMPLEMENTED_METHOD(uniform, Tensor, (const Shape& shape, float min_val, float max_val, unsigned int seed), )

// 张量复制操作
DEFINE_NOT_IMPLEMENTED_METHOD(copy, Tensor, (const Tensor& tensor), const)
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(copy_into, (const Tensor& src, Tensor& dst), const)
DEFINE_NOT_IMPLEMENTED_METHOD(zeros_like, Tensor, (const Tensor& input), const)

// 支持转置的矩阵乘法
DEFINE_NOT_IMPLEMENTED_VOID_METHOD(mm_into_transposed, (const Tensor& a, const Tensor& b, Tensor& result, bool transpose_a, bool transpose_b), )

} // namespace tr
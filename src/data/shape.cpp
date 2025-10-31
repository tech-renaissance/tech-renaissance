/**
 * @file shape.cpp
 * @brief 形状类实现
 * @details 实现形状类的核心逻辑，包括维度管理、元数据计算和验证功能
 * @version 1.00.00
 * @date 2025-10-25
 * @author 技术觉醒团队
 * @note 依赖项: shape.h
 * @note 所属系列: data
 */

#include "tech_renaissance/data/shape.h"
#include "tech_renaissance/utils/tr_exception.h"
#include <algorithm>
#include <cassert>

namespace tr {

Shape::Shape() : dims_{0, 0, 0, 0} {
    compute_metadata();
}

Shape::Shape(int32_t dim0) : dims_{0, 0, 0, 0} {
    validate_dims(&dim0, 1);
    dims_[3] = dim0;  // 存储到最后一个位置
    compute_metadata();
}

Shape::Shape(int32_t dim0, int32_t dim1) : dims_{0, 0, 0, 0} {
    int32_t dims[2] = {dim0, dim1};
    validate_dims(dims, 2);
    dims_[2] = dim0;
    dims_[3] = dim1;
    compute_metadata();
}

Shape::Shape(int32_t dim0, int32_t dim1, int32_t dim2) : dims_{0, 0, 0, 0} {
    int32_t dims[3] = {dim0, dim1, dim2};
    validate_dims(dims, 3);
    dims_[1] = dim0;
    dims_[2] = dim1;
    dims_[3] = dim2;
    compute_metadata();
}

Shape::Shape(int32_t n, int32_t c, int32_t h, int32_t w) : dims_{n, c, h, w} {
    int32_t dims[4] = {n, c, h, w};
    validate_dims(dims, 4);
    compute_metadata();
}

Shape::Shape(std::initializer_list<int32_t> dims) : dims_{0, 0, 0, 0} {
    if (dims.size() > 4) {
        throw TRException("[Shape::Shape] Shape cannot have more than 4 dimensions");
    }
    if (dims.size() == 0) {
        // 默认为标量
        compute_metadata();
        return;
    }

    validate_dims(dims.begin(), static_cast<int32_t>(dims.size()));

    // 根据维度数量确定存储位置（右对齐）
    int32_t start_idx = 4 - static_cast<int32_t>(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        dims_[start_idx + i] = dims.begin()[i];
    }

    compute_metadata();
}

int32_t Shape::ndim() const {
    return ndim_;
}

int64_t Shape::numel() const {
    return numel_;
}

int32_t Shape::dim(int32_t dim) const {
    if (dim < 0 || dim >= ndim_) {
        throw TRException("[Shape::dim] Dimension index out of range");
    }

    int32_t first_idx = first_nonzero_dim();
    return dims_[first_idx + dim];
}

bool Shape::operator==(const Shape& other) const {
    for (int32_t i = 0; i < 4; ++i) {
        if (dims_[i] != other.dims_[i]) {
            return false;
        }
    }
    return true;
}

bool Shape::operator!=(const Shape& other) const {
    return !(*this == other);
}

std::string Shape::to_string() const {
    std::ostringstream oss;
    if (ndim_ == 0) {
        oss << "()";
    } else {
        oss << "(";
        int32_t first_idx = first_nonzero_dim();
        for (int32_t i = first_idx; i < 4; ++i) {
            if (i != first_idx) {
                oss << ",";
            }
            oss << dims_[i];
        }
        oss << ")";
    }
    return oss.str();
}

bool Shape::is_scalar() const {
    return ndim_ == 0;
}

bool Shape::is_matmul_compatible(const Shape& other) const {
    // 矩阵乘法兼容性检查：[m,k] × [k,n] → [m,n]
    if (ndim_ != 2 || other.ndim_ != 2) {
        return false;
    }

    // 当前矩阵 [m, k]，另一个矩阵 [k', n]
    // 矩阵乘法要求：当前矩阵的列数 == 另一个矩阵的行数
    // 当前矩阵的列数（右数第二个）vs 另一个矩阵的行数（左数第一个）
    int32_t my_cols = dim(1);      // 当前矩阵的列数 k
    int32_t other_rows = other.dim(0);  // 另一个矩阵的行数 k'

    return my_cols == other_rows;  // k == k'
}

bool Shape::is_broadcastable_to(const Shape& target) const {
    // 广播规则检查
    if (ndim_ > target.ndim_) {
        return false;
    }

    // 从右向左检查每个维度
    for (int32_t i = 1; i <= ndim_; ++i) {
        int32_t my_dim = dim(ndim_ - i);
        int32_t target_dim = target.dim(target.ndim_ - i);

        // 当前维度必须等于目标维度，或者当前维度为1
        if (my_dim != target_dim && my_dim != 1) {
            return false;
        }
    }

    return true;
}

void Shape::compute_metadata() {
    // 计算维度数（非零维度的数量）
    ndim_ = 0;
    for (int32_t i = 0; i < 4; ++i) {
        if (dims_[i] != 0) {
            ndim_++;
        }
    }

    // 计算元素总数
    numel_ = 1;
    if (ndim_ == 0) {
        // 标量：元素数为1
        return;
    }

    int32_t first_idx = first_nonzero_dim();
    for (int32_t i = first_idx; i < 4; ++i) {
        numel_ *= dims_[i];
    }
}

void Shape::validate_dims(const int32_t* dims, int32_t count) {
    if (count <= 0) {
        return;  // 标量情况，由调用者处理
    }

    for (int32_t i = 0; i < count; ++i) {
        if (dims[i] < 0) {
            throw TRException("[Shape::validate_dims] Shape dimensions must be non-negative");
        }
        if (dims[i] == 0 && count > 1) {
            throw TRException("[Shape::validate_dims] Shape dimensions cannot be zero (except for scalar)");
        }
    }

    // 检查数值范围，避免溢出
    int64_t test_numel = 1;
    for (int32_t i = 0; i < count; ++i) {
        test_numel *= dims[i];
        if (test_numel > INT64_MAX / 1000) {  // 留一些安全边界
            throw TRException("[Shape::validate_dims] Shape dimensions are too large");
        }
    }
}

int32_t Shape::first_nonzero_dim() const {
    for (int32_t i = 0; i < 4; ++i) {
        if (dims_[i] != 0) {
            return i;
        }
    }
    return 0;  // 标量情况
}

bool Shape::is_right_aligned() const {
    bool found_zero = false;
    for (int32_t i = 0; i < 4; ++i) {
        if (dims_[i] == 0) {
            found_zero = true;
        } else if (found_zero && dims_[i] != 0) {
            return false;  // 在零之后发现了非零值
        }
    }
    return true;
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << shape.to_string();
    return os;
}

} // namespace tr
/**
 * @file storage.cpp
 * @brief 存储类实现
 * @details 实现存储类的核心逻辑，包括内存管理和数据访问
 * @version 1.00.00
 * @date 2025-10-25
 * @author 技术觉醒团队
 * @note 依赖项: storage.h
 * @note 所属系列: data
 */

#include "tech_renaissance/data/storage.h"

namespace tr {

Storage::Storage(size_t size, const Device& device)
    : data_ptr_(nullptr), size_(size), device_(device), holder_(nullptr) {
}

void* Storage::data_ptr() {
    return data_ptr_;
}

const void* Storage::data_ptr() const {
    return data_ptr_;
}

size_t Storage::size() const {
    return size_;
}

const Device& Storage::device() const {
    return device_;
}

bool Storage::is_empty() const {
    return data_ptr_ == nullptr || size_ == 0;
}

void Storage::set_data_ptr(void* ptr, std::shared_ptr<void> holder) {
    data_ptr_ = ptr;
    holder_ = holder;
}

} // namespace tr
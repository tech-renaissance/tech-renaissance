/**
* @file profiler.h
 * @brief 性能分析器类声明
 * @details 性能分析器，用于方便计时和运算性能分析。
 * @version 1.25.1
 * @date 2025-10-31
 * @author 技术觉醒团队
 * @note 依赖项: 标准库
 * @note 所属系列: utils
 */

#pragma once

#include <chrono>
#include "tech_renaissance/utils/logger.h"
#include "tech_renaissance/data/shape.h"
#include "tech_renaissance/utils/tr_exception.h"

namespace tr {
class Profiler {
public:
    Profiler();
    virtual ~Profiler();
    void start();
    void stop();
    double avg_time() const;
    void set_iterations(int iterations);
    double total_time() const;
    void describe_operation(const std::string& operation_type, Shape shape_a, Shape shape_b);
    double get_performance();
private:
    bool timer_started_;
    int iterations_;
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    std::chrono::time_point<std::chrono::steady_clock> end_time_;
    double total_;
    long long flops_;
};
} // namespace tr

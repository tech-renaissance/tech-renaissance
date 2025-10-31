#include "tech_renaissance/utils/profiler.h"


namespace tr {
    Profiler::Profiler() : timer_started_(false), iterations_(1), start_time_(), end_time_(), total_(-1.0), flops_(-1) {
    }

    Profiler::~Profiler() {
    }
    void Profiler::start() {
        if (timer_started_) {
            throw TRException("[Profiler::start] Timer has already started!");
        }
        else {
            timer_started_ = true;
            start_time_ = std::chrono::steady_clock::now();
        }
    }
    void Profiler::stop() {
        if (timer_started_) {
            timer_started_ = false;
            end_time_ = std::chrono::steady_clock::now();
        }
        else {
            throw TRException("[Profiler::stop] Timer has not yet started!");
        }
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_);
        total_ = duration.count() / 1000.0;
    }
    double Profiler::avg_time() const {
        if (timer_started_) {
            throw TRException("[Profiler::avg_time] Timer is still running!");
        }
        if (iterations_ <= 0) {
            throw TRException("[Profiler::avg_time] Invalid iteration count: " + std::to_string(iterations_));
        }
        double elapsed = total_ / iterations_;
        return elapsed;
    }
    void Profiler::set_iterations(int iterations) {
        if (iterations <= 0) {
            throw TRException("[Profiler::set_iterations] Invalid iteration count: " + std::to_string(iterations));
        }
        iterations_ = iterations;
    }
    double Profiler::total_time() const {
        if (timer_started_) {
            throw TRException("[Profiler::total_time] Timer is still running!");
        }
        return total_;
    }
    void Profiler::describe_operation(const std::string& operation_type, Shape shape_a, Shape shape_b) {
        if (operation_type == "mm") {
            // 对于矩阵乘法: A(M,K) * B(K,N) = C(M,N)
            // 对于二维Shape使用h()和w()，对于四维Shape使用n()和c()
            int M, K, N;
            if (shape_a.ndim() == 2) {
                M = shape_a.h();
                K = shape_a.w();
            } else {
                M = shape_a.n();
                K = shape_a.c();
            }

            if (shape_b.ndim() == 2) {
                N = shape_b.w();
                // K = shape_b.h(); // K已经从shape_a获取
            } else {
                N = shape_b.n();
                // K = shape_b.c(); // K已经从shape_a获取
            }

            flops_ = 2LL * M * K * N;
        }
        else {
            throw TRException("[Profiler::describe_operation] Unsupported operation type!");
        }
    }
    double Profiler::get_performance() {
        if (flops_ <= 0) {
            throw TRException("[Profiler::get_performance] Operation type not specified!");
        }
        else if (iterations_ <= 0) {
            throw TRException("[Profiler::get_performance] Invalid iteration count: " + std::to_string(iterations_));
        }
        else if (total_ < 0) {
            throw TRException("[Profiler::get_performance] Timer has not yet started!");
        }
        else {
            double gflops = flops_ / (total_ / iterations_ * 1e6);
            return gflops;
        }
    }
} // namespace tr
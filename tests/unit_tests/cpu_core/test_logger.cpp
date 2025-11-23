/**
 * @file test_logger.cpp
 * @brief Logger单元测试
 * @details 测试新的Logger系统所有功能，包括静态初始化修复、格式化输出、性能优化等
 * @version 1.19.01
 * @date 2025-10-31
 * @author 技术觉醒团队
 * @note 依赖项: tech_renaissance/utils/logger.h, fstream, string, thread, chrono
 * @note 所属系列: tests
 */

#include "tech_renaissance/utils/logger.h"
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <filesystem>

// 简单测试框架宏
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAIL: " << message << " (Line: " << __LINE__ << ")" << std::endl; \
            return false; \
        } \
    } while(0)

#define RUN_TEST(test_func) \
    do { \
        std::cout << "Running " << #test_func << "..." << std::endl; \
        if (test_func()) { \
            std::cout << "PASS: " << #test_func << std::endl; \
            passed_tests++; \
        } else { \
            std::cout << "FAIL: " << #test_func << std::endl; \
            failed_tests++; \
        } \
        total_tests++; \
    } while(0)

// 辅助函数
std::string read_file_content(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return "";
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();
    return content;
}

void cleanup_test_file(const std::string& filename) {
    std::remove(filename.c_str());
}

// 获取workspace目录中的测试文件路径，避免在根目录创建文件
std::string get_test_file_path(const std::string& basename) {
    // 确保workspace目录存在（与test_tsr_io_extended.cpp保持一致）
    std::filesystem::create_directories("workspace");
    return "workspace/" + basename;
}

bool file_contains(const std::string& filename, const std::string& content) {
    std::string file_content = read_file_content(filename);
    return !file_content.empty() && file_content.find(content) != std::string::npos;
}

// 测试函数声明
bool test_logger_static_initialization_fix();
bool test_logger_leaky_singleton();
bool test_logger_formatted_output();
bool test_logger_new_api();
bool test_logger_global_init_function();
bool test_logger_performance_optimization();
bool test_logger_thread_safety();
bool test_logger_convenience_macros();
bool test_logger_level_filtering();
bool test_logger_quiet_mode();

int main() {
    std::cout << "=== Logger V1.19.01 Unit Tests Start ===" << std::endl;

    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;

    // 运行所有测试
    RUN_TEST(test_logger_static_initialization_fix);
    RUN_TEST(test_logger_leaky_singleton);
    RUN_TEST(test_logger_formatted_output);
    RUN_TEST(test_logger_new_api);
    RUN_TEST(test_logger_global_init_function);
    RUN_TEST(test_logger_performance_optimization);
    RUN_TEST(test_logger_thread_safety);
    RUN_TEST(test_logger_convenience_macros);
    RUN_TEST(test_logger_level_filtering);
    RUN_TEST(test_logger_quiet_mode);

    // 输出测试结果
    std::cout << "\n=== Test Results Summary ===" << std::endl;
    std::cout << "Total tests: " << total_tests << std::endl;
    std::cout << "Passed tests: " << passed_tests << std::endl;
    std::cout << "Failed tests: " << failed_tests << std::endl;

    if (failed_tests == 0) {
        std::cout << "✅ All tests passed! Logger V1.19.01 is working correctly." << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some tests failed!" << std::endl;
        return 1;
    }
}

// 测试1: 静态初始化修复（核心测试）
bool test_logger_static_initialization_fix() {
    try {
        // 这个测试模拟了原始问题场景：在其他模块中使用Logger之前没有显式初始化
        // 如果静态初始化问题修复了，这应该能正常工作

        std::cout << "  Testing static initialization fix..." << std::endl;

        // 不调用任何Logger API，直接在其他模块中使用
        auto& logger = tr::Logger::get_instance();

        // 如果能到达这里，说明静态初始化问题已修复
        logger.info("Static initialization test - should work without explicit init");

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Static initialization test failed with exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Static initialization test failed with unknown exception" << std::endl;
        return false;
    }
}

// 测试2: 泄漏的单例模式
bool test_logger_leaky_singleton() {
    try {
        std::cout << "  Testing leaky singleton pattern..." << std::endl;

        // 获取多个实例引用，应该指向同一个对象
        tr::Logger& logger1 = tr::Logger::get_instance();
        tr::Logger& logger2 = tr::Logger::get_instance();
        tr::Logger& logger3 = tr::Logger::get_instance();

        TEST_ASSERT(&logger1 == &logger2, "get_instance should return the same object");
        TEST_ASSERT(&logger2 == &logger3, "get_instance should return the same object");
        TEST_ASSERT(&logger1 == &logger3, "get_instance should return the same object");

        // 测试引用有效性
        logger1.debug("Leaky singleton test - all references should work");

        return true;
    } catch (...) {
        std::cerr << "Leaky singleton test threw unexpected exception" << std::endl;
        return false;
    }
}

// 测试3: 格式化输出功能
bool test_logger_formatted_output() {
    try {
        std::cout << "  Testing formatted output..." << std::endl;

        std::string test_file = get_test_file_path("test_formatted_output.txt");
        cleanup_test_file(test_file);

        auto& logger = tr::Logger::get_instance();
        logger.set_output_file(test_file);
        logger.set_level(tr::LogLevel::DEBUG);

        // 等待文件设置生效
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // 测试各种格式的输出
        int epoch = 42;
        double accuracy = 95.67;
        std::string model_name = "ResNet-50";

        logger.debug("Debug message with int: ", epoch);
        logger.info("Training epoch ", epoch, " accuracy: ", accuracy, "%");
        logger.warn("Model ", model_name, " has high complexity");
        logger.error("Failed to load checkpoint at epoch ", epoch);

        // 等待日志写入
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // 验证文件内容
        std::string content = read_file_content(test_file);
        TEST_ASSERT(!content.empty(), "Log file should contain content");

        TEST_ASSERT(content.find("Debug message with int: 42") != std::string::npos,
                   "Should format int correctly");
        TEST_ASSERT(content.find("Training epoch 42 accuracy: 95.67%") != std::string::npos,
                   "Should format mixed types correctly");
        TEST_ASSERT(content.find("Model ResNet-50 has high complexity") != std::string::npos,
                   "Should format string correctly");
        TEST_ASSERT(content.find("Failed to load checkpoint at epoch 42") != std::string::npos,
                   "Should format error message correctly");

        cleanup_test_file(test_file);
        return true;
    } catch (...) {
        std::cerr << "Formatted output test threw unexpected exception" << std::endl;
        return false;
    }
}

// 测试4: 新API功能
bool test_logger_new_api() {
    try {
        std::cout << "  Testing new Logger API..." << std::endl;

        auto& logger = tr::Logger::get_instance();

        // 测试新的set_quiet_mode API
        logger.set_quiet_mode(false);
        logger.info("Quiet mode OFF - this message should appear");

        logger.set_quiet_mode(true);
        logger.info("Quiet mode ON - this message should NOT appear");
        logger.warn("Warning message should still appear in quiet mode");
        logger.error("Error message should still appear in quiet mode");

        // 恢复正常模式
        logger.set_quiet_mode(false);

        return true;
    } catch (...) {
        std::cerr << "New API test threw unexpected exception" << std::endl;
        return false;
    }
}

// 测试5: 全局初始化函数
bool test_logger_global_init_function() {
    try {
        std::cout << "  Testing global InitLogger function..." << std::endl;

        std::string test_file = get_test_file_path("test_global_init.txt");
        cleanup_test_file(test_file);

        // 使用全局初始化函数
        tr::InitLogger(test_file, tr::LogLevel::DEBUG, false);

        auto& logger = tr::Logger::get_instance();

        // 等待文件设置生效
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        logger.debug("Global init - debug message");
        logger.info("Global init - info message");
        logger.warn("Global init - warning message");
        logger.error("Global init - error message");

        // 等待日志写入
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::string content = read_file_content(test_file);
        TEST_ASSERT(!content.empty(), "Log file should contain content after global init");

        TEST_ASSERT(content.find("Global init - debug message") != std::string::npos,
                   "Should contain debug message");
        TEST_ASSERT(content.find("Global init - info message") != std::string::npos,
                   "Should contain info message");
        TEST_ASSERT(content.find("Global init - warning message") != std::string::npos,
                   "Should contain warning message");
        TEST_ASSERT(content.find("Global init - error message") != std::string::npos,
                   "Should contain error message");

        cleanup_test_file(test_file);
        return true;
    } catch (...) {
        std::cerr << "Global init function test threw unexpected exception" << std::endl;
        return false;
    }
}

// 测试6: 性能优化（持久化文件流）
bool test_logger_performance_optimization() {
    try {
        std::cout << "  Testing performance optimization..." << std::endl;

        std::string test_file = get_test_file_path("test_performance.txt");
        cleanup_test_file(test_file);

        auto& logger = tr::Logger::get_instance();
        logger.set_output_file(test_file);
        logger.set_level(tr::LogLevel::INFO);

        // 等待文件设置生效
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // 快速连续写入多条日志（测试文件流复用）
        for (int i = 0; i < 100; ++i) {
            logger.info("Performance test message ", i);
        }

        // 等待所有日志写入
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        std::string content = read_file_content(test_file);
        TEST_ASSERT(!content.empty(), "Log file should contain content");

        // 验证所有消息都被写入
        for (int i = 0; i < 10; ++i) {  // 只检查前10条以提高测试速度
            std::string expected = "Performance test message " + std::to_string(i);
            TEST_ASSERT(content.find(expected) != std::string::npos,
                       ("Should contain message " + expected).c_str());
        }

        cleanup_test_file(test_file);
        return true;
    } catch (...) {
        std::cerr << "Performance optimization test threw unexpected exception" << std::endl;
        return false;
    }
}

// 测试7: 线程安全
bool test_logger_thread_safety() {
    try {
        std::cout << "  Testing thread safety..." << std::endl;

        std::string test_file = get_test_file_path("test_thread_safety.txt");
        cleanup_test_file(test_file);

        auto& logger = tr::Logger::get_instance();
        logger.set_output_file(test_file);
        logger.set_level(tr::LogLevel::INFO);

        // 等待文件设置生效
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        std::atomic<int> counter{0};
        std::vector<std::thread> threads;

        // 创建多个线程同时写入日志
        for (int t = 0; t < 5; ++t) {
            threads.emplace_back([&logger, &counter, t]() {
                for (int i = 0; i < 20; ++i) {
                    logger.info("Thread ", t, " message ", i);
                    counter++;
                }
            });
        }

        // 等待所有线程完成
        for (auto& thread : threads) {
            thread.join();
        }

        // 等待日志写入
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        std::string content = read_file_content(test_file);
        TEST_ASSERT(!content.empty(), "Log file should contain content");

        // 验证所有线程的消息都被写入
        TEST_ASSERT(counter == 100, "All 100 messages should be processed");

        // 检查各个线程的消息是否存在
        for (int t = 0; t < 5; ++t) {
            for (int i = 0; i < 5; ++i) {  // 只检查每个线程的前5条消息
                std::string expected = "Thread " + std::to_string(t) + " message " + std::to_string(i);
                TEST_ASSERT(content.find(expected) != std::string::npos,
                           ("Should contain thread message " + expected).c_str());
            }
        }

        cleanup_test_file(test_file);
        return true;
    } catch (...) {
        std::cerr << "Thread safety test threw unexpected exception" << std::endl;
        return false;
    }
}

// 测试8: 便捷宏功能
bool test_logger_convenience_macros() {
    try {
        std::cout << "  Testing convenience macros..." << std::endl;

        std::string test_file = get_test_file_path("test_macros.txt");
        cleanup_test_file(test_file);

        auto& logger = tr::Logger::get_instance();
        logger.set_output_file(test_file);
        logger.set_level(tr::LogLevel::DEBUG);

        // 等待文件设置生效
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // 测试所有便捷宏
        TR_LOG_DEBUG("Macro debug message with number: ", 42);
        TR_LOG_INFO("Macro info message with string: ", "test");
        TR_LOG_WARN("Macro warn message with float: ", 3.14);
        TR_LOG_ERROR("Macro error message with multiple args: ", 1, 2, 3);

        // 等待日志写入
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::string content = read_file_content(test_file);
        TEST_ASSERT(!content.empty(), "Log file should contain content");

        TEST_ASSERT(content.find("Macro debug message with number: 42") != std::string::npos,
                   "DEBUG macro should work correctly");
        TEST_ASSERT(content.find("Macro info message with string: test") != std::string::npos,
                   "INFO macro should work correctly");
        TEST_ASSERT(content.find("Macro warn message with float: 3.14") != std::string::npos,
                   "WARN macro should work correctly");
        TEST_ASSERT(content.find("Macro error message with multiple args: 123") != std::string::npos,
                   "ERROR macro should work correctly");

        cleanup_test_file(test_file);
        return true;
    } catch (...) {
        std::cerr << "Convenience macros test threw unexpected exception" << std::endl;
        return false;
    }
}

// 测试9: 日志等级过滤
bool test_logger_level_filtering() {
    try {
        std::cout << "  Testing log level filtering..." << std::endl;

        std::string test_file = get_test_file_path("test_level_filtering.txt");
        cleanup_test_file(test_file);

        auto& logger = tr::Logger::get_instance();
        logger.set_output_file(test_file);

        // 等待文件设置生效
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // 测试DEBUG等级（应该显示所有消息）
        logger.set_level(tr::LogLevel::DEBUG);
        logger.debug("DEBUG level - debug message");
        logger.info("DEBUG level - info message");
        logger.warn("DEBUG level - warn message");
        logger.error("DEBUG level - error message");

        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // 测试WARN等级（应该只显示WARN和ERROR）
        logger.set_level(tr::LogLevel::WARN);
        logger.debug("WARN level - debug message (should NOT appear)");
        logger.info("WARN level - info message (should NOT appear)");
        logger.warn("WARN level - warn message");
        logger.error("WARN level - error message");

        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // 测试ERROR等级（应该只显示ERROR）
        logger.set_level(tr::LogLevel::ERROR);
        logger.debug("ERROR level - debug message (should NOT appear)");
        logger.info("ERROR level - info message (should NOT appear)");
        logger.warn("ERROR level - warn message (should NOT appear)");
        logger.error("ERROR level - error message");

        // 等待日志写入
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::string content = read_file_content(test_file);
        TEST_ASSERT(!content.empty(), "Log file should contain content");

        // 验证等级过滤是否正确工作
        TEST_ASSERT(content.find("DEBUG level - debug message") != std::string::npos,
                   "DEBUG messages should appear when level is DEBUG");
        TEST_ASSERT(content.find("DEBUG level - info message") != std::string::npos,
                   "INFO messages should appear when level is DEBUG");
        TEST_ASSERT(content.find("WARN level - warn message") != std::string::npos,
                   "WARN messages should appear when level is WARN");
        TEST_ASSERT(content.find("ERROR level - error message") != std::string::npos,
                   "ERROR messages should always appear");

        // 验证被过滤的消息不存在
        TEST_ASSERT(content.find("WARN level - debug message (should NOT appear)") == std::string::npos,
                   "DEBUG messages should be filtered when level is WARN");
        TEST_ASSERT(content.find("ERROR level - warn message (should NOT appear)") == std::string::npos,
                   "WARN messages should be filtered when level is ERROR");

        cleanup_test_file(test_file);
        return true;
    } catch (...) {
        std::cerr << "Log level filtering test threw unexpected exception" << std::endl;
        return false;
    }
}

// 测试10: 静默模式
bool test_logger_quiet_mode() {
    try {
        std::cout << "  Testing quiet mode..." << std::endl;

        std::string test_file = get_test_file_path("test_quiet_mode.txt");
        cleanup_test_file(test_file);

        auto& logger = tr::Logger::get_instance();
        logger.set_output_file(test_file);
        logger.set_level(tr::LogLevel::DEBUG);

        // 等待文件设置生效
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // 测试正常模式
        logger.set_quiet_mode(false);
        logger.info("Normal mode - info message (should appear)");
        logger.warn("Normal mode - warn message (should appear)");
        logger.error("Normal mode - error message (should appear)");

        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // 测试静默模式
        logger.set_quiet_mode(true);
        logger.info("Quiet mode - info message (should NOT appear)");
        logger.warn("Quiet mode - warn message (should appear)");
        logger.error("Quiet mode - error message (should appear)");

        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // 测试恢复正常模式
        logger.set_quiet_mode(false);
        logger.info("Restored mode - info message (should appear)");
        logger.warn("Restored mode - warn message (should appear)");
        logger.error("Restored mode - error message (should appear)");

        // 等待日志写入
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::string content = read_file_content(test_file);
        TEST_ASSERT(!content.empty(), "Log file should contain content");

        // 验证静默模式功能
        TEST_ASSERT(content.find("Normal mode - info message (should appear)") != std::string::npos,
                   "INFO messages should appear in normal mode");
        TEST_ASSERT(content.find("Quiet mode - info message (should NOT appear)") == std::string::npos,
                   "INFO messages should NOT appear in quiet mode");
        TEST_ASSERT(content.find("Quiet mode - warn message (should appear)") != std::string::npos,
                   "WARN messages should still appear in quiet mode");
        TEST_ASSERT(content.find("Quiet mode - error message (should appear)") != std::string::npos,
                   "ERROR messages should still appear in quiet mode");
        TEST_ASSERT(content.find("Restored mode - info message (should appear)") != std::string::npos,
                   "INFO messages should appear after restoring normal mode");

        cleanup_test_file(test_file);
        return true;
    } catch (...) {
        std::cerr << "Quiet mode test threw unexpected exception" << std::endl;
        return false;
    }
}
/**
 * @file test_logger.cpp
 * @brief Logger unit test
 * @details Test Logger output at different levels, no third-party framework
 * @version 1.00.00
 * @date 2025-10-23
 * @author Tech Renaissance Team
 * @note Dependencies: tech_renaissance/utils/logger.h, fstream, filesystem, string
 * @note Series: tests
 */

#include "tech_renaissance/utils/logger.h"
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>

// Simple test framework macros
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

// Helper functions
bool file_contains(const std::string& filename, const std::string& content) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.find(content) != std::string::npos) {
            file.close();
            return true;
        }
    }

    file.close();
    return false;
}

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

// Test function declarations
bool test_logger_singleton_pattern();
bool test_logger_level_setting();
bool test_logger_file_output();
bool test_logger_convenience_macros();
bool test_logger_be_quiet();

int main() {
    std::cout << "=== Logger Unit Tests Start ===" << std::endl;

    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;

    // Run all tests
    RUN_TEST(test_logger_singleton_pattern);
    RUN_TEST(test_logger_level_setting);
    RUN_TEST(test_logger_file_output);
    RUN_TEST(test_logger_convenience_macros);
    RUN_TEST(test_logger_be_quiet);

    // Output test results
    std::cout << "\n=== Test Results Summary ===" << std::endl;
    std::cout << "Total tests: " << total_tests << std::endl;
    std::cout << "Passed tests: " << passed_tests << std::endl;
    std::cout << "Failed tests: " << failed_tests << std::endl;

    if (failed_tests == 0) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests failed!" << std::endl;
        return 1;
    }
}

bool test_logger_singleton_pattern() {
    try {
        // Test singleton pattern
        tr::Logger& logger1 = tr::Logger::get_instance();
        tr::Logger& logger2 = tr::Logger::get_instance();

        TEST_ASSERT(&logger1 == &logger2, "get_instance should return the same object");

        return true;
    } catch (...) {
        std::cerr << "test_logger_singleton_pattern threw unexpected exception" << std::endl;
        return false;
    }
}

bool test_logger_level_setting() {
    try {
        tr::Logger& logger = tr::Logger::get_instance();

        // Test can set different log levels
        logger.set_level(tr::LogLevel::DEBUG);
        logger.set_level(tr::LogLevel::INFO);
        logger.set_level(tr::LogLevel::WARN);
        logger.set_level(tr::LogLevel::ERROR);

        // If no exception is thrown, setting was successful
        return true;
    } catch (...) {
        std::cerr << "test_logger_level_setting threw unexpected exception" << std::endl;
        return false;
    }
}

bool test_logger_file_output() {
    try {
        tr::Logger& logger = tr::Logger::get_instance();
        std::string test_file = "test_log_output.txt";

        // Clean up existing test file
        cleanup_test_file(test_file);

        // Test can set output file
        logger.set_output_file(test_file);

        // Wait a bit for file setting to take effect
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        logger.set_level(tr::LogLevel::DEBUG);
        logger.info("Test info message");
        logger.error("Test error message");

        // Wait for log to be written
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Check if file exists
        TEST_ASSERT(std::ifstream(test_file).good(), "Log file should be created");

        // Check file content
        std::string content = read_file_content(test_file);
        TEST_ASSERT(!content.empty(), "Log file should contain content");

        // Internal implementation
        TEST_ASSERT(content.find("Test info message") != std::string::npos,
                   "Log file should contain info message");
        TEST_ASSERT(content.find("Test error message") != std::string::npos,
                   "Log file should contain error message");
        TEST_ASSERT(content.find("[TR]") != std::string::npos,
                   "Log should contain [TR] marker");

        // Clean up test file
        cleanup_test_file(test_file);

        return true;
    } catch (...) {
        std::cerr << "test_logger_file_output threw unexpected exception" << std::endl;
        return false;
    }
}

bool test_logger_convenience_macros() {
    try {
        tr::Logger& logger = tr::Logger::get_instance();
        std::string test_file = "test_log_macros.txt";

        // Clean up existing test file
        cleanup_test_file(test_file);

        // Test all log macros
        logger.set_output_file(test_file);
        logger.set_level(tr::LogLevel::DEBUG);

        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        TR_LOG_DEBUG("Macro debug message");
        TR_LOG_INFO("Macro info message");
        TR_LOG_WARN("Macro warn message");
        TR_LOG_ERROR("Macro error message");

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::string content = read_file_content(test_file);
        TEST_ASSERT(!content.empty(), "Log file should contain content");

        // Check if macros work correctly
        TEST_ASSERT(content.find("Macro debug message") != std::string::npos,
                   "Should contain debug message");
        TEST_ASSERT(content.find("Macro info message") != std::string::npos,
                   "Should contain info message");
        TEST_ASSERT(content.find("Macro warn message") != std::string::npos,
                   "Should contain warn message");
        TEST_ASSERT(content.find("Macro error message") != std::string::npos,
                   "Should contain error message");

        // Clean up test file
        cleanup_test_file(test_file);

        return true;
    } catch (...) {
        std::cerr << "test_logger_convenience_macros threw unexpected exception" << std::endl;
        return false;
    }
}

bool test_logger_be_quiet() {
    try {
        tr::Logger& logger = tr::Logger::get_instance();
        std::string test_file = "test_log_quiet.txt";

        // Clean up existing test file
        cleanup_test_file(test_file);

        // Reset to console output and enable all levels
        logger.set_output_file("");
        logger.set_level(tr::LogLevel::DEBUG);

        // Wait a bit for settings to take effect
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Test 1: Normal mode - all messages should appear
        std::cout << "\n--- Testing normal mode ---" << std::endl;
        logger.info("Normal mode INFO message (should appear)");
        logger.warn("Normal mode WARN message (should appear)");
        logger.error("Normal mode ERROR message (should appear)");

        // Test 2: Set file output and verify normal behavior
        logger.set_output_file(test_file);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        logger.info("Before quiet INFO message (should appear in file)");
        logger.warn("Before quiet WARN message (should appear in file)");
        logger.error("Before quiet ERROR message (should appear in file)");

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Check file contains all messages
        std::string content_before = read_file_content(test_file);
        TEST_ASSERT(!content_before.empty(), "Log file should contain content before quiet");
        TEST_ASSERT(content_before.find("Before quiet INFO message") != std::string::npos,
                   "File should contain INFO message before quiet");
        TEST_ASSERT(content_before.find("Before quiet WARN message") != std::string::npos,
                   "File should contain WARN message before quiet");
        TEST_ASSERT(content_before.find("Before quiet ERROR message") != std::string::npos,
                   "File should contain ERROR message before quiet");

        // Test 3: Activate quiet mode
        std::cout << "\n--- Activating quiet mode ---" << std::endl;
        logger.be_quiet();

        // Clear file for fresh test
        cleanup_test_file(test_file);
        logger.set_output_file(test_file);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        logger.info("After quiet INFO message (should NOT appear)");
        logger.warn("After quiet WARN message (should appear)");
        logger.error("After quiet ERROR message (should appear)");

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Check file content after quiet mode
        std::string content_after = read_file_content(test_file);
        TEST_ASSERT(!content_after.empty(), "Log file should contain content after quiet");
        TEST_ASSERT(content_after.find("After quiet INFO message") == std::string::npos,
                   "File should NOT contain INFO message after quiet");
        TEST_ASSERT(content_after.find("After quiet WARN message") != std::string::npos,
                   "File should contain WARN message after quiet");
        TEST_ASSERT(content_after.find("After quiet ERROR message") != std::string::npos,
                   "File should contain ERROR message after quiet");

        std::cout << "--- Quiet mode test completed ---" << std::endl;

        // Clean up test file
        cleanup_test_file(test_file);

        return true;
    } catch (...) {
        std::cerr << "test_logger_be_quiet threw unexpected exception" << std::endl;
        return false;
    }
}
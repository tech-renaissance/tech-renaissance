/**
 * @file test_tr_exception.cpp
 * @brief TRException unit test (enhanced with sub-exception classes)
 * @details Test TR_THROW macro, TRException::what() method, and new exception sub-classes
 * @version 1.10.00
 * @date 2025-11-09
 * @author Tech Renaissance Team
 * @note Dependencies: tech_renaissance/utils/tr_exception.h, iostream, string
 * @note Series: tests
 */

#include "tech_renaissance/utils/tr_exception.h"
#include <iostream>
#include <string>
#include <cassert>

// Simple test framework macros
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAIL: " << message << " (Line: " << __LINE__ << ")" << std::endl; \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_THROWS(expression, exception_type, message) \
    do { \
        bool threw = false; \
        try { \
            expression; \
        } catch (const exception_type&) { \
            threw = true; \
        } catch (...) { \
            std::cerr << "FAIL: " << message << " - threw wrong exception type (Line: " << __LINE__ << ")" << std::endl; \
            return false; \
        } \
        if (!threw) { \
            std::cerr << "FAIL: " << message << " - did not throw (Line: " << __LINE__ << ")" << std::endl; \
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

// Test function declarations
bool test_exception_basic_functionality();
bool test_tr_throw_macro();
bool test_tr_throw_if_macro();
bool test_exception_inheritance();
bool test_exception_subclasses();
bool test_new_macros();

int main() {
    std::cout << "=== TRException Unit Tests Start ===" << std::endl;

    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;

    // Run all tests
    RUN_TEST(test_exception_basic_functionality);
    RUN_TEST(test_tr_throw_macro);
    RUN_TEST(test_tr_throw_if_macro);
    RUN_TEST(test_exception_inheritance);
    RUN_TEST(test_exception_subclasses);
    RUN_TEST(test_new_macros);

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

bool test_exception_basic_functionality() {
    try {
        std::string message = "Test error message";
        tr::TRException ex(message);

        std::string what_str = ex.what();
        TEST_ASSERT(what_str.find(message) != std::string::npos,
                   "Exception message should contain specified error info");
        TEST_ASSERT(ex.file().empty(), "File name should be empty by default");
        TEST_ASSERT(ex.line() == 0, "Line number should be 0 by default");

        // Test exception with file name
        std::string file = "test_file.cpp";
        tr::TRException ex2(message, file);
        what_str = ex2.what();
        TEST_ASSERT(what_str.find(message) != std::string::npos,
                   "Exception message with file should contain error info");
        TEST_ASSERT(ex2.file() == file, "File name should be set correctly");
        TEST_ASSERT(ex2.line() == 0, "Line number should be 0");

        // Test exception with file name and line number
        int line = 42;
        tr::TRException ex3(message, file, line);
        what_str = ex3.what();
        TEST_ASSERT(what_str.find(message) != std::string::npos,
                   "Exception message with file and line should contain error info");
        TEST_ASSERT(what_str.find(file) != std::string::npos,
                   "Exception message should contain file name");
        TEST_ASSERT(what_str.find("42") != std::string::npos,
                   "Exception message should contain line number");
        TEST_ASSERT(ex3.file() == file, "File name should be set correctly");
        TEST_ASSERT(ex3.line() == line, "Line number should be set correctly");

        // Test message format
        TEST_ASSERT(what_str.find("TRException:") == 0,
                   "Exception message should start with 'TRException:'");
        TEST_ASSERT(what_str.find("File:") != std::string::npos,
                   "Exception message should contain 'File:'");
        TEST_ASSERT(what_str.find("Line:") != std::string::npos,
                   "Exception message should contain 'Line:'");

        // Test type() method
        TEST_ASSERT(std::string(ex.type()) == "TRException",
                   "Base exception type should be 'TRException'");

        return true;
    } catch (...) {
        std::cerr << "test_exception_basic_functionality threw unexpected exception" << std::endl;
        return false;
    }
}

bool test_tr_throw_macro() {
    try {
        // Test TR_THROW macro
        TEST_ASSERT_THROWS(TR_THROW("Test macro throw"), tr::TRException,
                          "TR_THROW should throw tr::TRException");

        // Test TR_THROW macro includes file and line info
        try {
            TR_THROW("Macro test message");
            TEST_ASSERT(false, "Should not reach here");
        } catch (const tr::TRException& ex) {
            std::string what_str = ex.what();
            TEST_ASSERT(what_str.find("Macro test message") != std::string::npos,
                       "Exception message should contain test message");
            TEST_ASSERT(!ex.file().empty(), "Should contain file info");
            TEST_ASSERT(ex.line() > 0, "Line number should be greater than 0");
        }

        // Test different message types
        TEST_ASSERT_THROWS(TR_THROW("String literal"), tr::TRException,
                          "TR_THROW should support string literal");
        TEST_ASSERT_THROWS(TR_THROW(std::string("String object")), tr::TRException,
                          "TR_THROW should support string object");

        return true;
    } catch (...) {
        std::cerr << "test_tr_throw_macro threw unexpected exception" << std::endl;
        return false;
    }
}

bool test_tr_throw_if_macro() {
    try {
        // Test TR_THROW_IF macro when condition is true
        TEST_ASSERT_THROWS(TR_THROW_IF(true, "Conditional throw"), tr::TRException,
                          "TR_THROW_IF should throw when condition is true");

        // Test TR_THROW_IF macro when condition is false
        try {
            TR_THROW_IF(false, "Should not throw");
            // Should reach here normally
        } catch (...) {
            TEST_ASSERT(false, "TR_THROW_IF should not throw when condition is false");
        }

        // Test complex condition
        int value = 0;
        try {
            TR_THROW_IF(value == 0, "Value is zero");
            TEST_ASSERT(false, "Should not reach here");
        } catch (const tr::TRException&) {
            // Correctly threw exception
        }

        value = 1;
        try {
            TR_THROW_IF(value != 0, "Value is not zero");
            TEST_ASSERT(false, "Should not reach here");
        } catch (const tr::TRException&) {
            // Correctly threw exception
        }

        value = 0;
        try {
            TR_THROW_IF(value != 0, "Should not throw");
            // Should reach here normally
        } catch (...) {
            TEST_ASSERT(false, "Should not throw when condition is false");
        }

        return true;
    } catch (...) {
        std::cerr << "test_tr_throw_if_macro threw unexpected exception" << std::endl;
        return false;
    }
}

bool test_exception_inheritance() {
    try {
        // Test inheritance relationship
        std::string message = "Inheritance test";
        tr::TRException ex(message);

        std::exception* base_ptr = &ex;
        std::string base_what = base_ptr->what();
        TEST_ASSERT(base_what.find(message) != std::string::npos,
                   "As base class pointer, what() should return correct message");

        // Test can be caught as std::exception
        try {
            TR_THROW("Caught as base class");
            TEST_ASSERT(false, "Should not reach here");
        } catch (const std::exception& e) {
            std::string caught_message = e.what();
            TEST_ASSERT(caught_message.find("Caught as base class") != std::string::npos,
                       "Caught as std::exception, message should be correct");
        }

        return true;
    } catch (...) {
        std::cerr << "test_exception_inheritance threw unexpected exception" << std::endl;
        return false;
    }
}

bool test_exception_subclasses() {
    try {
        // Test FileNotFoundError
        try {
            TR_THROW_FILE_NOT_FOUND("File not found: test.txt");
            TEST_ASSERT(false, "Should not reach here");
        } catch (const tr::FileNotFoundError& ex) {
            std::string what_str = ex.what();
            TEST_ASSERT(what_str.find("FileNotFoundError:") == 0,
                       "FileNotFoundError message should start with 'FileNotFoundError:'");
            TEST_ASSERT(what_str.find("File not found: test.txt") != std::string::npos,
                       "Should contain original message");
            TEST_ASSERT(std::string(ex.type()) == "FileNotFoundError",
                       "type() should return 'FileNotFoundError'");
        }

        // Test NotImplementedError
        try {
            TR_THROW_NOT_IMPLEMENTED("Feature not implemented yet");
            TEST_ASSERT(false, "Should not reach here");
        } catch (const tr::NotImplementedError& ex) {
            std::string what_str = ex.what();
            TEST_ASSERT(what_str.find("NotImplementedError:") == 0,
                       "NotImplementedError message should start with 'NotImplementedError:'");
            TEST_ASSERT(what_str.find("Feature not implemented yet") != std::string::npos,
                       "Should contain original message");
            TEST_ASSERT(std::string(ex.type()) == "NotImplementedError",
                       "type() should return 'NotImplementedError'");
        }

        // Test ValueError
        try {
            TR_THROW_VALUE_ERROR("Invalid parameter value");
            TEST_ASSERT(false, "Should not reach here");
        } catch (const tr::ValueError& ex) {
            std::string what_str = ex.what();
            TEST_ASSERT(what_str.find("ValueError:") == 0,
                       "ValueError message should start with 'ValueError:'");
            TEST_ASSERT(what_str.find("Invalid parameter value") != std::string::npos,
                       "Should contain original message");
            TEST_ASSERT(std::string(ex.type()) == "ValueError",
                       "type() should return 'ValueError'");
        }

        // Test IndexError
        try {
            TR_THROW_INDEX_ERROR("Array index out of bounds");
            TEST_ASSERT(false, "Should not reach here");
        } catch (const tr::IndexError& ex) {
            std::string what_str = ex.what();
            TEST_ASSERT(what_str.find("IndexError:") == 0,
                       "IndexError message should start with 'IndexError:'");
            TEST_ASSERT(what_str.find("Array index out of bounds") != std::string::npos,
                       "Should contain original message");
            TEST_ASSERT(std::string(ex.type()) == "IndexError",
                       "type() should return 'IndexError'");
        }

        // Test TypeError
        try {
            TR_THROW_TYPE_ERROR("Wrong data type provided");
            TEST_ASSERT(false, "Should not reach here");
        } catch (const tr::TypeError& ex) {
            std::string what_str = ex.what();
            TEST_ASSERT(what_str.find("TypeError:") == 0,
                       "TypeError message should start with 'TypeError:'");
            TEST_ASSERT(what_str.find("Wrong data type provided") != std::string::npos,
                       "Should contain original message");
            TEST_ASSERT(std::string(ex.type()) == "TypeError",
                       "type() should return 'TypeError'");
        }

        // Test ZeroDivisionError
        try {
            TR_THROW_ZERO_DIVISION("Division by zero");
            TEST_ASSERT(false, "Should not reach here");
        } catch (const tr::ZeroDivisionError& ex) {
            std::string what_str = ex.what();
            TEST_ASSERT(what_str.find("ZeroDivisionError:") == 0,
                       "ZeroDivisionError message should start with 'ZeroDivisionError:'");
            TEST_ASSERT(what_str.find("Division by zero") != std::string::npos,
                       "Should contain original message");
            TEST_ASSERT(std::string(ex.type()) == "ZeroDivisionError",
                       "type() should return 'ZeroDivisionError'");
        }

        // Test inheritance chain - all subclasses should be catchable as TRException
        TEST_ASSERT_THROWS(TR_THROW_NOT_IMPLEMENTED("Test"), tr::TRException,
                          "Subclass exceptions should be catchable as base TRException");

        // Test inheritance chain - all subclasses should be catchable as std::exception
        TEST_ASSERT_THROWS(TR_THROW_VALUE_ERROR("Test"), std::exception,
                          "Subclass exceptions should be catchable as std::exception");

        return true;
    } catch (...) {
        std::cerr << "test_exception_subclasses threw unexpected exception" << std::endl;
        return false;
    }
}

bool test_new_macros() {
    try {
        // Test backward compatibility - old TR_THROW macro should still work
        try {
            TR_THROW("Old style macro test");
            TEST_ASSERT(false, "Should not reach here");
        } catch (const tr::TRException& ex) {
            std::string what_str = ex.what();
            TEST_ASSERT(what_str.find("TRException:") == 0,
                       "Old TR_THROW should still create TRException");
            TEST_ASSERT(what_str.find("Old style macro test") != std::string::npos,
                       "Should contain original message");
        }

        // Test new TYPE macro works
        try {
            TR_THROW_TYPE(ValueError, "Custom type test");
            TEST_ASSERT(false, "Should not reach here");
        } catch (const tr::ValueError& ex) {
            std::string what_str = ex.what();
            TEST_ASSERT(what_str.find("ValueError:") == 0,
                       "TR_THROW_TYPE should create specified exception type");
            TEST_ASSERT(what_str.find("Custom type test") != std::string::npos,
                       "Should contain original message");
        }

        // Test all specific macros work and include file/line info
        try {
            TR_THROW_NOT_IMPLEMENTED("Macro with location test");
            TEST_ASSERT(false, "Should not reach here");
        } catch (const tr::NotImplementedError& ex) {
            std::string what_str = ex.what();
            TEST_ASSERT(what_str.find("Macro with location test") != std::string::npos,
                       "Should contain original message");
            TEST_ASSERT(!ex.file().empty(), "Should contain file info");
            TEST_ASSERT(ex.line() > 0, "Should contain line info");
            TEST_ASSERT(what_str.find("File:") != std::string::npos,
                       "Should contain 'File:' in formatted output");
            TEST_ASSERT(what_str.find("Line:") != std::string::npos,
                       "Should contain 'Line:' in formatted output");
        }

        // Test TR_THROW_IF still works with simple message
        try {
            TR_THROW_IF(true, "Conditional test with simple message");
            TEST_ASSERT(false, "Should not reach here");
        } catch (const tr::TRException& ex) {
            std::string what_str = ex.what();
            TEST_ASSERT(what_str.find("TRException:") == 0,
                       "Should be TRException");
            TEST_ASSERT(what_str.find("Conditional test with simple message") != std::string::npos,
                       "Should contain the message");
        }

        return true;
    } catch (...) {
        std::cerr << "test_new_macros threw unexpected exception" << std::endl;
        return false;
    }
}
/**
 * @file test_workspace.cpp
 * @brief 测试workspace目录宏是否正确设置
 * @details 验证WORKSPACE_DIR宏在编译时是否能正确指向workspace目录
 * @version 1.00.00
 * @date 2025-10-28
 * @author 技术觉醒团队
 * @note 依赖项: 无
 * @note 所属系列: unit_tests
 */

#include <iostream>
#include <fstream>
#include <filesystem>

int main() {
    std::cout << "=== Workspace Directory Test ===" << std::endl;

    // 输出编译时定义的WORKSPACE_DIR宏
    std::cout << "WORKSPACE_DIR macro: " << WORKSPACE_DIR << std::endl;

    // 检查目录是否存在
    if (std::filesystem::exists(WORKSPACE_DIR)) {
        std::cout << "✓ Workspace directory exists: " << WORKSPACE_DIR << std::endl;

        // 检查是否为目录
        if (std::filesystem::is_directory(WORKSPACE_DIR)) {
            std::cout << "✓ Path is indeed a directory" << std::endl;

            // 列出目录内容（如果有的话）
            auto it = std::filesystem::directory_iterator(WORKSPACE_DIR);
            if (it == std::filesystem::directory_iterator{}) {
                std::cout << "✓ Directory is empty (ready for use)" << std::endl;
            } else {
                std::cout << "✓ Directory contains files:" << std::endl;
                for (const auto& entry : std::filesystem::directory_iterator(WORKSPACE_DIR)) {
                    std::cout << "  - " << entry.path().filename().string() << std::endl;
                }
            }
        } else {
            std::cout << "✗ Path exists but is not a directory!" << std::endl;
            return 1;
        }
    } else {
        std::cout << "✗ Workspace directory does not exist: " << WORKSPACE_DIR << std::endl;
        return 1;
    }

    // 测试在workspace中创建文件
    try {
        std::string test_file = std::string(WORKSPACE_DIR) + "/test_created_by_cpp.txt";
        std::ofstream ofs(test_file);
        if (ofs.is_open()) {
            ofs << "This file was created by test_workspace.exe to verify write access.\n";
            ofs.close();
            std::cout << "✓ Successfully created test file: " << test_file << std::endl;

            // 清理测试文件
            std::filesystem::remove(test_file);
            std::cout << "✓ Test file cleaned up successfully" << std::endl;
        } else {
            std::cout << "✗ Failed to create test file in workspace directory" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cout << "✗ Exception while testing file creation: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n=== All Tests Passed! ===" << std::endl;
    std::cout << "Workspace directory is properly configured and accessible." << std::endl;

    return 0;
}
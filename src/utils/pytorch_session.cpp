/**
 * @file pytorch_session.cpp
 * @brief PyTorch会话管理类实现
 * @details 实现Python进程生命周期管理和临时文件通信机制
 * @version 1.00.00
 * @date 2025-10-27
 * @author 技术觉醒团队
 * @note 依赖项: pytorch_session.h, filesystem, process
 * @note 所属系列: utils
 */

#include "tech_renaissance/utils/pytorch_session.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/backend/backend.h"
#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include <filesystem>
#include <fstream>
#include <thread>
#include <iostream>

namespace tr {

PyTorchSession::PyTorchSession(const std::string& script_path, const std::string& session_id, bool quiet_mode)
    : script_path_(script_path), session_id_(session_id), running_(false), quiet_mode_(quiet_mode) {

    if (script_path == "default") script_path_ = std::string(WORKSPACE_PATH) +
            "/../python/tests/python_server.py";
    create_session_dir();
    if (!quiet_mode_) std::cout << "[PyTorchSession] Created session: " << session_id_ << " at " << session_dir_ << std::endl;
}

PyTorchSession::~PyTorchSession() {
    try {
        if (running_) {
            terminate();
        }
        cleanup_temp_files();
        if (!quiet_mode_) std::cout << "[PyTorchSession] Session destroyed: " << session_id_ << std::endl;
    } catch (const std::exception& e) {
        if (!quiet_mode_) std::cout << "[PyTorchSession] Error in destructor: " << e.what() << std::endl;
    }
}

void PyTorchSession::create_session_dir() {
    namespace fs = std::filesystem;

    // 使用workspace目录作为临时文件存储位置
    std::string base_dir = std::string(WORKSPACE_PATH) + "/pytorch_session";
    if (!fs::exists(base_dir)) {
        fs::create_directories(base_dir);
    }

    session_dir_ = base_dir + "/tr_session_" + session_id_;
    fs::create_directories(session_dir_);

    if (!quiet_mode_) std::cout << "[PyTorchSession] Created session directory: " << session_dir_ << std::endl;
}

void PyTorchSession::start() {
    if (running_) {
        if (!quiet_mode_) std::cout << "[PyTorchSession] Session already started" << std::endl;
        return;
    }

    // 检查Python脚本是否存在
    if (!std::filesystem::exists(script_path_)) {
        throw TRException("[PyTorchSession::start] Python script not found: " + script_path_);
    }

    std::string cmd;
#ifdef _WIN32
    // 在Windows上使用start命令在后台启动Python进程
    cmd = "start /B python \"" + script_path_ + "\" " + session_id_;
#else
    cmd = "python3 \"" + script_path_ + "\" " + session_id_ + " &";
#endif

    if (!quiet_mode_) std::cout << "[PyTorchSession] Launching Python: " << cmd << std::endl;
    int ret = std::system(cmd.c_str());
    // Windows的start命令总是返回成功，所以我们不检查返回值
#ifndef _WIN32
    if (ret != 0) {
        if (!quiet_mode_) std::cout << "[PyTorchSession] std::system returned: " << ret << std::endl;
        throw TRException("[PyTorchSession::start] Failed to start Python script");
    }
#endif
    running_ = true;

    // 等待一小段时间确保进程启动
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // 写入状态文件通知Python可以开始工作
    write_status_file("ready");
    if (!quiet_mode_) std::cout << "[PyTorchSession] Python process started successfully" << std::endl;
}

bool PyTorchSession::is_alive() {
    if (!running_) {
        return false;
    }

    // 首先检查状态文件（快速检测）
    std::string status_file = get_temp_file_path("status", "txt");
    std::ifstream file(status_file);
    if (file) {
        std::string status;
        std::getline(file, status);
        file.close();

        // 如果状态是terminated或error开头，说明进程已结束
        if (status == "terminated" || status.find("error:") == 0) {
            running_ = false;
            return false;
        }

        // 如果状态是done，说明进程完成了任务但仍在运行
        // 这是正常情况，返回true
    }

    return true;
}

void PyTorchSession::terminate() {
    if (!running_) {
        return;
    }

    // 写入退出信号文件
    write_status_file("exit");

    // 给Python进程一些时间自然退出
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    running_ = false;
    if (!quiet_mode_) std::cout << "[PyTorchSession] Python process terminated" << std::endl;
}

void PyTorchSession::join() {
    // 等待Python进程自然结束
    while (is_alive()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    running_ = false;
}

void PyTorchSession::send_request(const std::string& msg) const {
    // 原子操作，避免读写冲突
    std::string request_file_temp = session_dir_ + "/request.tmp";
    std::string request_file = session_dir_ + "/request.json";
    std::ofstream(request_file_temp) << msg;
    std::error_code ec;          // 用 error_code 避免抛异常
    std::filesystem::path old_name = request_file_temp;
    std::filesystem::path new_name = request_file;
    std::filesystem::rename(old_name, new_name, ec);
    if (ec) {
        throw TRException("[PyTorchSession::send_request] rename failed");
    }
}

void PyTorchSession::please_exit(uint32_t timeout_ms, bool ensure) {
    if (!wait_until_ready(timeout_ms)) {
        std::cout << "[PyTorchSession::wait_for_response] WARNING: Write timeout" << std::endl;
        if (ensure) {
            std::cout << "[PyTorchSession::wait_for_response] WARNING: Manually terminated" << std::endl;
            terminate();
        }
    }
    send_request(R"({"cmd": "exit"})");
}

bool PyTorchSession::is_ready() const {
    std::filesystem::path request_file = session_dir_ + "/request.json";
    if (std::filesystem::exists(request_file))
        return false;
    return true;
}

bool PyTorchSession::is_busy() const {
    std::filesystem::path request_file = session_dir_ + "/request.json";
    if (std::filesystem::exists(request_file))
        return true;
    return false;
}

bool PyTorchSession::new_response() const {
    std::filesystem::path request_file = session_dir_ + "/response.json";
    if (std::filesystem::exists(request_file))
        return true;
    return false;
}

bool PyTorchSession::wait_until_ready(uint32_t timeout_ms) const {
    uint32_t wait_count = 0;
    uint32_t shortest_quest_interval = 32;
    uint32_t quest_interval = shortest_quest_interval;
    namespace fs = std::filesystem;
    fs::path request_file = session_dir_ + "/request.json";
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    bool wait_forever = timeout_ms == 0;  // 特殊值 0 表示无限等待
    while (wait_forever || std::chrono::steady_clock::now() < deadline) {
        if (!fs::exists(request_file)) {
            return true;
        }
        if (AUTO_QUEST_FREQUENCY) {
            std::this_thread::sleep_for(std::chrono::milliseconds(quest_interval));
            auto level = wait_count / 8;
            if (level >= 5) {
                quest_interval = shortest_quest_interval * 32;
            }
            else {
                quest_interval = shortest_quest_interval * (1 << level);
                wait_count++;
            }
        }
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    return false;
}

bool PyTorchSession::wait_until_ok(uint32_t timeout_ms) const {
    uint32_t wait_count = 0;
    uint32_t shortest_quest_interval = 32;
    uint32_t quest_interval = shortest_quest_interval;
    namespace fs = std::filesystem;
    fs::path response_file = session_dir_ + "/response.json";
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    bool wait_forever = timeout_ms == 0;  // 特殊值 0 表示无限等待
    while (wait_forever || std::chrono::steady_clock::now() < deadline) {
        if (fs::exists(response_file)) {
            return true;
        }
        if (AUTO_QUEST_FREQUENCY) {
            std::this_thread::sleep_for(std::chrono::milliseconds(quest_interval));
            auto level = wait_count / 8;
            if (level >= 5) {
                quest_interval = shortest_quest_interval * 32;
            }
            else {
                quest_interval = shortest_quest_interval * (1 << level);
                wait_count++;
            }
        }
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    return false;
}

std::string PyTorchSession::read_response() const {
    namespace fs = std::filesystem;
    std::string file_name = session_dir_ + "/response.json";
    std::ifstream in(file_name);
    if (!in) return "";

    // 读取整个JSON文件
    std::string json_str((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
    in.close();

    // 找到params字段的值
    const std::string key = "\"params\":";
    size_t pos = json_str.find(key);
    if (pos == std::string::npos) {
        std::remove(file_name.c_str());    // 清理文件
        return "";        // 文件不符合约定
    }

    pos += key.length();                             // 跳过键名

    // 定位到第一个引号（字符串值开始）
    while (pos < json_str.size() && json_str[pos] != '"') ++pos;
    if (pos >= json_str.size()) {
        std::remove(file_name.c_str());
        return "";
    }
    ++pos; // 跳过开始的引号

    // 找到匹配的结束引号
    size_t end = pos;
    while (end < json_str.size()) {
        if (json_str[end] == '"') {
            // 检查是否转义引号
            if (end == 0 || json_str[end - 1] != '\\') {
                break;
            }
        }
        ++end;
    }

    std::string params = json_str.substr(pos, end - pos);
    std::remove(file_name.c_str());    // 阅后即焚，以便接受下一个响应
    return params;
}

std::string PyTorchSession::wait_for_response(uint32_t timeout_ms) const {
    if (!wait_until_ok(timeout_ms)) {
        std::cout << "[PyTorchSession::wait_for_response] WARNING: Read timeout" << std::endl;
        return "";
    }
    return read_response();
}

std::string PyTorchSession::fetch_response(const std::string& msg, uint32_t timeout_ms) const {
    if (!wait_until_ready(timeout_ms)) {
        std::cout << "[PyTorchSession::fetch_response] WARNING: Write timeout" << std::endl;
        return "";
    }
    send_request(msg);
    if (!wait_until_ok(timeout_ms)) {
        std::cout << "[PyTorchSession::fetch_response] WARNING: Read timeout" << std::endl;
        return "";
    }
    return read_response();
}

Tensor PyTorchSession::fetch_tensor(const std::string& msg, uint32_t timeout_ms) const {
    if (!running_) {
        throw TRException("[PyTorchSession::fetch_tensor] Session not running");
    }

    // 发送请求并等待响应
    std::string response = fetch_response(msg, timeout_ms);
    if (response.empty()) {
        throw TRException("[PyTorchSession::fetch_tensor] Failed to get tensor response");
    }

    // 构建结果文件路径
    std::string result_path = session_dir_ + "/" + response + ".tsr";

    if (!wait_for_file(result_path, timeout_ms)) {
        throw TRException("[PyTorchSession::fetch_tensor] Timeout waiting for tensor file: " + result_path);
    }

    // 使用后端导入张量
    auto backend = BackendManager::instance().get_backend(CPU);
    return dynamic_cast<CpuBackend*>(backend.get())->import_tensor(result_path);
}

Tensor PyTorchSession::wait_for_tensor(uint32_t timeout_ms) const {
    if (!running_) {
        throw TRException("[PyTorchSession::wait_for_tensor] Session not running");
    }

    // 等待响应
    std::string response = wait_for_response(timeout_ms);
    if (response.empty()) {
        throw TRException("[PyTorchSession::wait_for_tensor] Failed to get tensor response");
    }

    // 构建结果文件路径
    std::string result_path = session_dir_ + "/" + response + ".tsr";

    if (!wait_for_file(result_path, timeout_ms)) {
        throw TRException("[PyTorchSession::wait_for_tensor] Timeout waiting for tensor file: " + result_path);
    }

    // 使用后端导入张量
    auto backend = BackendManager::instance().get_backend(CPU);
    return dynamic_cast<CpuBackend*>(backend.get())->import_tensor(result_path);
}

void PyTorchSession::send_tensor(const Tensor& tensor, const std::string& tag) const {
    if (!running_) {
        throw TRException("[PyTorchSession::send_tensor] Session not running");
    }

    // 构建文件路径
    std::string tensor_path = session_dir_ + "/" + tag + ".tsr";

    // 使用后端导出张量
    auto backend = BackendManager::instance().get_backend(CPU);
    dynamic_cast<CpuBackend*>(backend.get())->export_tensor(tensor, tensor_path);
}

bool PyTorchSession::wait_for_file(const std::string& file_path, uint32_t timeout_ms) const {
    namespace fs = std::filesystem;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (std::chrono::steady_clock::now() < deadline) {
        if (fs::exists(file_path)) {
            // 检查文件是否完整（非零大小）
            std::ifstream file(file_path, std::ios::binary | std::ios::ate);
            if (file.tellg() > 0) {
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return false;
}

std::string PyTorchSession::get_temp_file_path(const std::string& tag, const std::string& extension) {
    return session_dir_ + "/" + tag + "." + extension;
}

void PyTorchSession::write_status_file(const std::string& status) {
    std::string status_file = get_temp_file_path("status", "txt");

    try {
        std::ofstream file(status_file);
        if (file) {
            file << status;
            file.close();
            if (!quiet_mode_) std::cout << "[PyTorchSession] Wrote status: " << status << std::endl;
        }
    } catch (const std::exception& e) {
        if (!quiet_mode_) std::cout << "[PyTorchSession] Failed to write status file: " << e.what() << std::endl;
    }
}

void PyTorchSession::cleanup_temp_files() {
    try {
        namespace fs = std::filesystem;
        if (fs::exists(session_dir_)) {
            fs::remove_all(session_dir_);
            if (!quiet_mode_) std::cout << "[PyTorchSession] Cleaned up temp directory: " << session_dir_ << std::endl;
        }
    } catch (const std::exception& e) {
        if (!quiet_mode_) std::cout << "[PyTorchSession] Failed to cleanup temp files: " << e.what() << std::endl;
    }
}

} // namespace tr
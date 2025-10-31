/**
 * @file python_session.cpp
 * @brief Python会话管理类实现
 * @details 实现Python进程生命周期管理和临时文件通信机制
 * @version 1.00.00
 * @date 2025-10-27
 * @author 技术觉醒团队
 * @note 依赖项: python_session.h, filesystem, process
 * @note 所属系列: utils
 */

#include "tech_renaissance/utils/python_session.h"
#include "tech_renaissance/backend/backend_manager.h"
#include "tech_renaissance/backend/backend.h"
#include "tech_renaissance/backend/cpu/cpu_backend.h"
#include <filesystem>
#include <fstream>
#include <thread>
#include <iostream>

namespace tr {

PythonSession::PythonSession(const std::string& script_path, const std::string& session_id, bool quiet_mode)
    : script_path_(script_path), session_id_(session_id), running_(false), quiet_mode_(quiet_mode) {

    if (script_path == "default") script_path_ = std::string(WORKSPACE_PATH) +
            "/../python/module/python_server.py";
    create_session_dir();
    if (!quiet_mode_) std::cout << "[PythonSession] Created session: " << session_id_ << " at " << session_dir_ << std::endl;

    namespace fs = std::filesystem;
    std::string request_file_name = session_dir_ + "/request.json";
    std::string response_file_name = session_dir_ + "/response.json";
    fs::path request_file = request_file_name;
    fs::path response_file = response_file_name;
    if (fs::exists(request_file)) {
        std::remove(request_file_name.c_str());
    }
    if (fs::exists(response_file)) {
        std::remove(response_file_name.c_str());
    }
}

PythonSession::~PythonSession() {
    try {
        if (running_) {
            terminate();
        }
        cleanup_temp_files();
        if (!quiet_mode_) std::cout << "[PythonSession] Session destroyed: " << session_id_ << std::endl;
    } catch (const std::exception& e) {
        if (!quiet_mode_) std::cout << "[PythonSession] Error in destructor: " << e.what() << std::endl;
    }
}

void PythonSession::create_session_dir() {
    namespace fs = std::filesystem;

    // 使用workspace目录作为临时文件存储位置
    std::string base_dir = std::string(WORKSPACE_PATH) + "/python_session";
    if (!fs::exists(base_dir)) {
        fs::create_directories(base_dir);
    }

    session_dir_ = base_dir + "/tr_session_" + session_id_;
    fs::create_directories(session_dir_);

    if (!quiet_mode_) std::cout << "[PythonSession] Created session directory: " << session_dir_ << std::endl;
}

void PythonSession::start(int warmup_time) {
    if (running_) {
        if (!quiet_mode_) std::cout << "[PythonSession] Session already started" << std::endl;
        return;
    }

    // 检查Python脚本是否存在
    if (!std::filesystem::exists(script_path_)) {
        throw TRException("[PythonSession::start] Python script not found: " + script_path_);
    }

    std::string cmd;
#ifdef _WIN32
    // 在Windows上使用start命令在后台启动Python进程
    cmd = "start /B python \"" + script_path_ + "\" " + session_id_;
#else
    cmd = "python3 \"" + script_path_ + "\" " + session_id_ + " &";
#endif

    if (!quiet_mode_) std::cout << "[PythonSession] Launching Python: " << cmd << std::endl;
    int ret = std::system(cmd.c_str());
    // Windows的start命令总是返回成功，所以我们不检查返回值
#ifndef _WIN32
    if (ret != 0) {
        if (!quiet_mode_) std::cout << "[PythonSession] std::system returned: " << ret << std::endl;
        throw TRException("[PythonSession::start] Failed to start Python script");
    }
#endif
    running_ = true;

    // 等待一小段时间确保进程启动
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // 写入状态文件通知Python可以开始工作
    write_status_file("ready");
    if (!quiet_mode_) std::cout << "[PythonSession] Python process started successfully" << std::endl;

    // 等待进程启动
    std::this_thread::sleep_for(std::chrono::milliseconds(warmup_time));
}

bool PythonSession::is_alive() {
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

void PythonSession::terminate() {
    if (!running_) {
        return;
    }

    // 写入退出信号文件
    write_status_file("exit");

    // 给Python进程一些时间自然退出
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    running_ = false;
    if (!quiet_mode_) std::cout << "[PythonSession] Python process terminated" << std::endl;
}

void PythonSession::join() {
    // 等待Python进程自然结束
    while (is_alive()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    running_ = false;
}

void PythonSession::send_request(const std::string& msg) const {
    // 原子操作，避免读写冲突
    std::string request_file_temp = session_dir_ + "/request.tmp";
    std::string request_file = session_dir_ + "/request.json";
    std::ofstream(request_file_temp) << msg;
    std::error_code ec;          // 用 error_code 避免抛异常
    std::filesystem::path old_name = request_file_temp;
    std::filesystem::path new_name = request_file;
    std::filesystem::rename(old_name, new_name, ec);
    if (ec) {
        throw TRException("[PythonSession::send_request] rename failed");
    }
}

void PythonSession::please_exit(uint32_t timeout_ms, bool ensure) {
    if (!wait_until_ready(timeout_ms)) {
        std::cout << "[PythonSession::wait_for_response] WARNING: Write timeout" << std::endl;
        if (ensure) {
            std::cout << "[PythonSession::wait_for_response] WARNING: Manually terminated" << std::endl;
            terminate();
        }
    }
    send_request(R"({"cmd": "exit"})");
}

bool PythonSession::is_ready() const {
    std::filesystem::path request_file = session_dir_ + "/request.json";
    if (std::filesystem::exists(request_file))
        return false;
    return true;
}

bool PythonSession::is_busy() const {
    std::filesystem::path request_file = session_dir_ + "/request.json";
    if (std::filesystem::exists(request_file))
        return true;
    return false;
}

bool PythonSession::new_response() const {
    std::filesystem::path request_file = session_dir_ + "/response.json";
    if (std::filesystem::exists(request_file))
        return true;
    return false;
}

bool PythonSession::wait_until_ready(uint32_t timeout_ms) const {
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

bool PythonSession::wait_until_ok(uint32_t timeout_ms) const {
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

std::string PythonSession::read_response() const {
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

std::string PythonSession::wait_for_response(uint32_t timeout_ms) const {
    if (!wait_until_ok(timeout_ms)) {
        std::cout << "[PythonSession::wait_for_response] WARNING: Read timeout" << std::endl;
        return "";
    }
    return read_response();
}

std::string PythonSession::fetch_response(const std::string& msg, uint32_t timeout_ms) const {
    if (!wait_until_ready(timeout_ms)) {
        std::cout << "[PythonSession::fetch_response] WARNING: Write timeout" << std::endl;
        return "";
    }
    send_request(msg);
    if (!wait_until_ok(timeout_ms)) {
        std::cout << "[PythonSession::fetch_response] WARNING: Read timeout" << std::endl;
        return "";
    }
    return read_response();
}

Tensor PythonSession::fetch_tensor(const std::string& msg, uint32_t timeout_ms) const {
    if (!running_) {
        throw TRException("[PythonSession::fetch_tensor] Session not running");
    }

    // 发送请求并等待响应
    std::string response = fetch_response(msg, timeout_ms);
    if (response.empty()) {
        throw TRException("[PythonSession::fetch_tensor] Failed to get tensor response");
    }

    // 构建结果文件路径
    std::string result_path = session_dir_ + "/" + response + ".tsr";

    if (!wait_for_file(result_path, timeout_ms)) {
        throw TRException("[PythonSession::fetch_tensor] Timeout waiting for tensor file: " + result_path);
    }

    // 使用后端导入张量
    auto backend = BackendManager::instance().get_backend(CPU);
    return dynamic_cast<CpuBackend*>(backend.get())->import_tensor(result_path);
}

Tensor PythonSession::calculate(const std::string& msg, const Tensor& tensor_a, uint32_t timeout_ms) const {
    send_tensor(tensor_a, "a");
    return fetch_tensor(R"({"cmd": ")" + msg + R"(", "params": "a"})", timeout_ms);
}

Tensor PythonSession::calculate(const std::string& msg, const Tensor& tensor_a, const Tensor& tensor_b, uint32_t timeout_ms) const {
    send_tensor(tensor_a, "a");
    send_tensor(tensor_b, "b");
    return fetch_tensor(R"({"cmd": ")" + msg + R"(", "params": "a,b"})", timeout_ms);
}

Tensor PythonSession::calculate(const std::string& msg, const Tensor& tensor_a, const Tensor& tensor_b, const Tensor& tensor_c, uint32_t timeout_ms) const {
    send_tensor(tensor_a, "a");
    send_tensor(tensor_b, "b");
    send_tensor(tensor_c, "c");
    return fetch_tensor(R"({"cmd": ")" + msg + R"(", "params": "a,b,c"})", timeout_ms);
}

Tensor PythonSession::wait_for_tensor(uint32_t timeout_ms) const {
    if (!running_) {
        throw TRException("[PythonSession::wait_for_tensor] Session not running");
    }

    // 等待响应
    std::string response = wait_for_response(timeout_ms);
    if (response.empty()) {
        throw TRException("[PythonSession::wait_for_tensor] Failed to get tensor response");
    }

    // 构建结果文件路径
    std::string result_path = session_dir_ + "/" + response + ".tsr";

    if (!wait_for_file(result_path, timeout_ms)) {
        throw TRException("[PythonSession::wait_for_tensor] Timeout waiting for tensor file: " + result_path);
    }

    // 使用后端导入张量
    auto backend = BackendManager::instance().get_backend(CPU);
    return dynamic_cast<CpuBackend*>(backend.get())->import_tensor(result_path);
}

void PythonSession::send_tensor(const Tensor& tensor, const std::string& tag) const {
    if (!running_) {
        throw TRException("[PythonSession::send_tensor] Session not running");
    }

    // 构建文件路径
    std::string tensor_path = session_dir_ + "/" + tag + ".tsr";

    // 使用后端导出张量
    auto backend = BackendManager::instance().get_backend(CPU);
    dynamic_cast<CpuBackend*>(backend.get())->export_tensor(tensor, tensor_path);
}

bool PythonSession::wait_for_file(const std::string& file_path, uint32_t timeout_ms) const {
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

std::string PythonSession::get_temp_file_path(const std::string& tag, const std::string& extension) {
    return session_dir_ + "/" + tag + "." + extension;
}

void PythonSession::write_status_file(const std::string& status) {
    std::string status_file = get_temp_file_path("status", "txt");

    try {
        std::ofstream file(status_file);
        if (file) {
            file << status;
            file.close();
            if (!quiet_mode_) std::cout << "[PythonSession] Wrote status: " << status << std::endl;
        }
    } catch (const std::exception& e) {
        if (!quiet_mode_) std::cout << "[PythonSession] Failed to write status file: " << e.what() << std::endl;
    }
}

void PythonSession::cleanup_temp_files() {
    try {
        namespace fs = std::filesystem;
        if (fs::exists(session_dir_)) {
            fs::remove_all(session_dir_);
            if (!quiet_mode_) std::cout << "[PythonSession] Cleaned up temp directory: " << session_dir_ << std::endl;
        }
    } catch (const std::exception& e) {
        if (!quiet_mode_) std::cout << "[PythonSession] Failed to cleanup temp files: " << e.what() << std::endl;
    }
}

} // namespace tr
/**
 * @file python_session.h
 * @brief Python会话管理类声明
 * @details 管理Python进程生命周期，实现C++与Python的实时交互，支持临时文件通信
 * @version 1.00.00
 * @date 2025-10-27
 * @author 技术觉醒团队
 * @note 依赖项: Python可执行文件
 * @note 所属系列: utils
 */

#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <filesystem>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>
#include "tech_renaissance/utils/logger.h"
#include "tech_renaissance/utils/tr_exception.h"
#include "tech_renaissance/data/tensor.h"

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>
#endif

#define AUTO_QUEST_FREQUENCY true

namespace tr {

class PythonSession {
public:
    PythonSession(const std::string& script_path = "default", const std::string& session_id = "default", bool quiet_mode = false);
    ~PythonSession();

    void start(int warmup_time = 300);           // 启动Python脚本，启动后默认等待300ms
    bool is_alive();        // 检查Python是否仍在运行
    void terminate();       // 强制终止
    void join();            // 等待
    void send_request(const std::string& msg) const;
    void please_exit(uint32_t timeout_ms = 10000, bool ensure = true);
    bool is_ready() const;         // 检查Python是否已就绪
    bool is_busy() const;         // 检查Python是否正在处理请求
    bool new_response() const;     // 检查是否有新的响应
    bool wait_until_ready(uint32_t timeout_ms = 10000) const;
    bool wait_until_ok(uint32_t timeout_ms = 10000) const;
    std::string read_response() const;  // 直接读取响应，不检查状态
    std::string wait_for_response(uint32_t timeout_ms = 10000) const;  // 等待并读取响应
    std::string fetch_response(const std::string& msg, uint32_t timeout_ms = 10000) const;  // 发送请求并等待响应
    void send_tensor(const Tensor& tensor, const std::string& tag) const;
    Tensor fetch_tensor(const std::string& msg, uint32_t timeout_ms = 10000) const;
    Tensor calculate(const std::string& msg, const Tensor& tensor_a, uint32_t timeout_ms = 10000) const;
    Tensor calculate(const std::string& msg, const Tensor& tensor_a, const Tensor& tensor_b, uint32_t timeout_ms = 10000) const;
    Tensor calculate(const std::string& msg, const Tensor& tensor_a, const Tensor& tensor_b, const Tensor& tensor_c, uint32_t timeout_ms = 10000) const;
    Tensor wait_for_tensor(uint32_t timeout_ms = 10000) const;
    std::string session_dir() const {return session_dir_;}

    const std::string& session_id() const { return session_id_; }
    void set_quiet_mode(bool quiet_mode) {quiet_mode_ = quiet_mode;}

private:
    std::string script_path_;
    std::string session_id_;
    std::string session_dir_;
    std::atomic<bool> running_;
    bool quiet_mode_;            // 静默模式：禁用INFO日志

    // 创建会话目录
    void create_session_dir();

    // 等待文件出现
    bool wait_for_file(const std::string& file_path, uint32_t timeout_ms) const;

    // 生成临时文件路径
    std::string get_temp_file_path(const std::string& tag, const std::string& extension);

    // 写入状态文件
    void write_status_file(const std::string& status);

    // 清理临时文件
    void cleanup_temp_files();

#ifdef _WIN32
    PROCESS_INFORMATION proc_info_{};
#else
    pid_t pid_{-1};
#endif
};

} // namespace tr
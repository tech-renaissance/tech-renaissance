#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python任务处理器示例（简化版）
用于与C++的PyTorchSession类进行交互
不依赖第三方库，使用纯Python实现
"""

import sys
import time
import json
import os
import struct
import signal

# 设置信号处理，用于优雅退出
running = True

def signal_handler(signum, frame):
    global running
    print(f"[Python] Received signal {signum}, preparing to exit...")
    running = False

# 注册信号处理函数
if hasattr(signal, 'SIGINT'):
    signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, signal_handler)

def load_tensor_from_file(file_path):
    """从TXT文件加载张量信息"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            print(f"[Python] Read tensor info from {file_path}:")
            for line in lines:
                print(f"[Python] {line.strip()}")
        return True  # 简化：只要成功读取就返回True
    except Exception as e:
        print(f"[Python] Error loading tensor from {file_path}: {e}")
        return False

def save_tensor_to_file(tensor_info, file_path):
    """将张量信息保存为TXT文件"""
    try:
        with open(file_path, 'w') as f:
            f.write(f"shape: {tensor_info['shape']}\n")
            f.write(f"result: {tensor_info['result']}\n")
            f.write(f"operation: {tensor_info['operation']}\n")

        print(f"[Python] Saved tensor result to {file_path}")
        return True
    except Exception as e:
        print(f"[Python] Error saving tensor to {file_path}: {e}")
        return False

def process_square_command(session_dir):
    """处理张量平方命令"""
    input_file = os.path.join(session_dir, "input.txt")
    output_file = os.path.join(session_dir, "output.txt")

    if not os.path.exists(input_file):
        print(f"[Python] Input file not found: {input_file}")
        return False

    # 加载输入张量信息
    if not load_tensor_from_file(input_file):
        return False

    print(f"[Python] Processing tensor square command")

    # 创建结果信息
    result_info = {
        'shape': '1x1x4x4',
        'result': '4.0',  # 2.0的平方
        'operation': 'square'
    }

    # 保存结果
    if save_tensor_to_file(result_info, output_file):
        # 更新状态文件
        status_file = os.path.join(session_dir, "status.txt")
        with open(status_file, 'w') as f:
            f.write("done")
        print(f"[Python] Square operation completed")
        return True

    return False

def process_add_command(session_dir, params):
    """处理张量加法命令（简化为TXT通信）"""
    try:
        input_file = os.path.join(session_dir, "input.txt")
        output_file = os.path.join(session_dir, "output.txt")

        if not load_tensor_from_file(input_file):
            return False

        print(f"[Python] Processing tensor addition command")

        # 创建结果信息
        result_info = {
            'shape': '1x1x4x4',
            'result': '4.0',  # 简化结果
            'operation': 'addition'
        }

        # 保存结果
        if save_tensor_to_file(result_info, output_file):
            # 更新状态文件
            status_file = os.path.join(session_dir, "status.txt")
            with open(status_file, 'w') as f:
                f.write("done")
            print(f"[Python] Addition operation completed")
            return True

    except Exception as e:
        print(f"[Python] Error in add command: {e}")

    return False

def process_multiply_command(session_dir, params):
    """处理张量乘法命令（简化为TXT通信）"""
    try:
        input_file = os.path.join(session_dir, "input.txt")
        output_file = os.path.join(session_dir, "output.txt")

        if not load_tensor_from_file(input_file):
            return False

        print(f"[Python] Processing tensor multiplication command")

        # 创建结果信息
        result_info = {
            'shape': '1x1x4x4',
            'result': '4.0',  # 简化结果
            'operation': 'multiplication'
        }

        # 保存结果
        if save_tensor_to_file(result_info, output_file):
            # 更新状态文件
            status_file = os.path.join(session_dir, "status.txt")
            with open(status_file, 'w') as f:
                f.write("done")
            print(f"[Python] Multiplication operation completed")
            return True

    except Exception as e:
        print(f"[Python] Error in multiply command: {e}")

    return False

def process_request(session_dir):
    """处理请求文件"""
    request_file = os.path.join(session_dir, "request.json")

    if not os.path.exists(request_file):
        return True  # 没有请求是正常的

    try:
        with open(request_file, 'r') as f:
            request = json.load(f)

        cmd = request.get('cmd', '')
        params = request.get('params', '')

        print(f"[Python] Processing command: {cmd}, params: {params}")

        success = False
        if cmd == "tensor_square":
            success = process_square_command(session_dir)
        elif cmd == "tensor_add":
            success = process_add_command(session_dir, params)
        elif cmd == "tensor_multiply":
            success = process_multiply_command(session_dir, params)
        elif cmd == "exit":
            print("[Python] Received exit command")
            return False
        else:
            print(f"[Python] Unknown command: {cmd}")

        if not success:
            # 更新状态文件为错误
            status_file = os.path.join(session_dir, "status.txt")
            with open(status_file, 'w') as f:
                f.write(f"error:Failed to process command {cmd}")

        # 删除请求文件，避免重复处理
        os.remove(request_file)

    except Exception as e:
        print(f"[Python] Error processing request: {e}")
        # 更新状态文件为错误
        status_file = os.path.join(session_dir, "status.txt")
        with open(status_file, 'w') as f:
            f.write(f"error:{str(e)}")
        return False

    return True

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("Usage: python_task_simple.py <session_id>")
        sys.exit(1)

    session_id = sys.argv[1]
    # 使用绝对路径，与C++保持一致
    import os
    # 从 python/tests/python_task_simple.py 找到项目根目录
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    session_dir = f"{project_root}/workspace/pytorch_session/tr_session_{session_id}"

    # 创建会话目录
    os.makedirs(session_dir, exist_ok=True)

    # 写入初始状态文件
    status_file = os.path.join(session_dir, "status.txt")
    with open(status_file, 'w') as f:
        f.write("running")

    print(f"[Python] Session {session_id} started, directory: {session_dir}")
    print(f"[Python] Waiting for requests...")

    try:
        # 主循环：持续运行20秒，每秒检查一次
        for i in range(20):  # 20秒
            # 打印心跳信息
            if i % 5 == 0:  # 每5秒打印一次
                print(f"[Python] Heartbeat: {i}s elapsed")

            # 检查退出标志文件
            exit_flag = os.path.join(session_dir, "exit.flag")
            if os.path.exists(exit_flag):
                print("[Python] Exit flag detected, exiting...")
                break

            # 检查status.txt文件，如果内容是"exit"则退出
            status_file = os.path.join(session_dir, "status.txt")
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        status = f.read().strip()
                    if status == "exit":
                        print("[Python] Exit status detected, exiting...")
                        break
                except:
                    pass

            # 处理请求
            if not process_request(session_dir):
                break

            # 短暂休眠，避免CPU占用过高
            time.sleep(1)  # 每秒检查一次

        print(f"[Python] Session {session_id} completed 20-second run")

    except KeyboardInterrupt:
        print("[Python] KeyboardInterrupt received, exiting...")
    except Exception as e:
        print(f"[Python] Unexpected error: {e}")
        # 更新状态文件为错误
        with open(status_file, 'w') as f:
            f.write(f"error:{str(e)}")

    # 更新最终状态
    try:
        with open(status_file, 'w') as f:
            f.write("terminated")
    except:
        pass

    print(f"[Python] Session {session_id} terminated")

if __name__ == "__main__":
    main()
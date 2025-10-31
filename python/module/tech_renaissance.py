"""
技术觉醒框架Python模块 - TSR文件读写与服务器支持
实现TSR文件格式与PyTorch张量的相互转换，以及C++通信支持
"""

import torch
import struct
import os
import sys
import time
import json
import signal
from typing import Union, Tuple, Optional


class TSRError(Exception):
    """TSR文件处理异常"""
    pass


def _validate_tensor_for_export(tensor: torch.Tensor) -> None:
    """
    验证张量是否符合TSR导出要求

    Args:
        tensor: 待验证的PyTorch张量

    Raises:
        TSRError: 如果张量不符合要求
    """
    # 检查维度限制
    if tensor.dim() > 4:
        raise TSRError(f"张量维度{tensor.dim()}超过4维限制")

    # 检查数据类型
    if tensor.dtype not in [torch.float32, torch.int8]:
        raise TSRError(f"不支持的数据类型{tensor.dtype}，仅支持float32和int8")

    # 检查设备类型
    if tensor.device.type != 'cpu':
        raise TSRError(f"张量必须在CPU设备上，当前在{tensor.device.type}")


def _tensor_to_tsr_dtype(tensor: torch.Tensor) -> int:
    """
    将PyTorch张量类型转换为TSR数据类型枚举

    Args:
        tensor: PyTorch张量

    Returns:
        TSR数据类型枚举值
    """
    if tensor.dtype == torch.float32:
        return 1  # FP32
    elif tensor.dtype == torch.int8:
        return 2  # INT8
    else:
        raise TSRError(f"不支持的数据类型: {tensor.dtype}")


def _tsr_dtype_to_tensor(dtype_enum: int) -> torch.dtype:
    """
    将TSR数据类型枚举转换为PyTorch张量类型

    Args:
        dtype_enum: TSR数据类型枚举值

    Returns:
        PyTorch数据类型
    """
    if dtype_enum == 1:
        return torch.float32
    elif dtype_enum == 2:
        return torch.int8
    else:
        raise TSRError(f"未知的TSR数据类型枚举值: {dtype_enum}")


def _shape_to_nchw(tensor: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    将任意形状的张量转换为NCHW格式的4元组

    Args:
        tensor: PyTorch张量

    Returns:
        (n, c, h, w)格式的4元组，缺失维度用1填充
    """
    dims = [1] * 4  # 初始化为[1, 1, 1, 1]

    # 按照右对齐规则填充维度
    if tensor.dim() == 0:  # 标量
        dims = [1, 1, 1, 1]
    elif tensor.dim() == 1:  # 1D张量 [W]
        dims = [1, 1, 1, tensor.shape[0]]
    elif tensor.dim() == 2:  # 2D张量 [H, W]
        dims = [1, 1, tensor.shape[0], tensor.shape[1]]
    elif tensor.dim() == 3:  # 3D张量 [C, H, W]
        dims = [1, tensor.shape[0], tensor.shape[1], tensor.shape[2]]
    elif tensor.dim() == 4:  # 4D张量 [N, C, H, W]
        dims = [tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]]

    return tuple(dims)


def _nchw_to_shape(nchw: Tuple[int, int, int, int], ndim: int) -> Tuple[int, ...]:
    """
    将NCHW格式的4元组转换回原始张量形状

    Args:
        nchw: (n, c, h, w)格式的4元组
        ndim: 原始张量维度数

    Returns:
        原始张量形状
    """
    if ndim == 0:  # 标量
        return ()
    elif ndim == 1:  # 1D张量 [W]
        return (nchw[3],)
    elif ndim == 2:  # 2D张量 [H, W]
        return (nchw[2], nchw[3])
    elif ndim == 3:  # 3D张量 [C, H, W]
        return (nchw[1], nchw[2], nchw[3])
    elif ndim == 4:  # 4D张量 [N, C, H, W]
        return nchw
    else:
        raise TSRError(f"不支持的维度数: {ndim}")


def export_tsr(tensor: torch.Tensor, filename: str) -> None:
    """
    将PyTorch张量导出为TSR文件

    Args:
        tensor: 要导出的PyTorch张量（必须是float32或int8，最多4维）
        filename: 输出TSR文件路径

    Raises:
        TSRError: 如果张量不符合要求或写入失败
    """
    # 验证张量
    _validate_tensor_for_export(tensor)

    # 转换形状和类型
    dtype_enum = _tensor_to_tsr_dtype(tensor)
    ndim = tensor.dim()
    nchw = _shape_to_nchw(tensor)
    total_elements = tensor.numel()

    # 创建文件头 (64字节)
    # 格式: <4s i i i i i 4i q q q (小端序，共64字节)
    header = struct.pack(
        '<4s i i i i i 4i q q q',
        b'TSR!',      # 魔数 (4字节)
        1,            # 版本 (4字节)
        64,           # 头部大小 (4字节)
        0,            # reserved_1 (4字节)
        dtype_enum,   # 数据类型 (4字节)
        ndim,         # 维度数量 (4字节)
        nchw[0], nchw[1], nchw[2], nchw[3],  # 维度数组 (16字节)
        total_elements,  # 元素总数 (8字节)
        0,            # reserved_2 (8字节)
        0             # reserved_3 (8字节)
    )

    try:
        # 写入文件
        with open(filename, 'wb') as f:
            f.write(header)
            # 按照NCHW顺序写入数据（对于非4D张量，已经按右对齐处理）
            # PyTorch张量的存储已经是连续的，直接写入即可
            tensor_bytes = tensor.contiguous().numpy().tobytes()
            f.write(tensor_bytes)

    except Exception as e:
        raise TSRError(f"写入TSR文件失败: {e}")


def import_tsr(filename: str) -> torch.Tensor:
    """
    从TSR文件读取数据为PyTorch张量

    Args:
        filename: TSR文件路径

    Returns:
        PyTorch张量

    Raises:
        TSRError: 如果文件格式错误或读取失败
    """
    if not os.path.exists(filename):
        raise TSRError(f"TSR文件不存在: {filename}")

    try:
        with open(filename, 'rb') as f:
            # 读取文件头
            header_data = f.read(64)
            if len(header_data) != 64:
                raise TSRError("TSR文件头大小不正确")

            # 解析文件头 (64字节)
            # 格式: <4s i i i i i 4i q q q (与导出时一致)
            magic, version, header_size, reserved_1, dtype_enum, ndim, \
            dim0, dim1, dim2, dim3, total_elements, reserved_2, reserved_3 = struct.unpack(
                '<4s i i i i i 4i q q q', header_data
            )

            # 验证魔数和版本
            if magic != b'TSR!':
                raise TSRError(f"无效的TSR魔数: {magic}")

            if version != 1:
                raise TSRError(f"不支持的TSR版本: {version}")

            if header_size != 64:
                raise TSRError(f"不支持的头部大小: {header_size}，期望64字节")

            # 转换数据类型
            dtype = _tsr_dtype_to_tensor(dtype_enum)

            # 重建形状
            nchw = (dim0, dim1, dim2, dim3)
            shape = _nchw_to_shape(nchw, ndim)

            # 计算期望的数据大小
            element_size = 4 if dtype == torch.float32 else 1  # FP32=4字节, INT8=1字节
            expected_size = total_elements * element_size

            # 读取数据
            data = f.read(expected_size)
            if len(data) != expected_size:
                raise TSRError(f"数据大小不匹配，期望{expected_size}字节，实际{len(data)}字节")

            # 创建张量
            if ndim == 0:  # 标量
                # 对于标量，从单个字节/浮点数创建
                if dtype == torch.float32:
                    value = struct.unpack('<f', data)[0]
                    tensor = torch.tensor(value, dtype=dtype)
                else:  # int8
                    value = struct.unpack('<b', data)[0]
                    tensor = torch.tensor(value, dtype=dtype)
            else:
                # 对于多维张量，从字节数组创建
                import numpy as np
                np_dtype = np.float32 if dtype == torch.float32 else np.int8
                np_array = np.frombuffer(data, dtype=np_dtype).reshape(shape)
                # 创建可写副本以避免PyTorch警告
                np_array = np.copy(np_array)
                tensor = torch.from_numpy(np_array)

            return tensor

    except struct.error as e:
        raise TSRError(f"解析TSR文件头失败: {e}")
    except Exception as e:
        if isinstance(e, TSRError):
            raise
        raise TSRError(f"读取TSR文件失败: {e}")


def get_tsr_info(filename: str) -> dict:
    """
    获取TSR文件的元数据信息

    Args:
        filename: TSR文件路径

    Returns:
        包含TSR文件信息的字典
    """
    tensor = import_tsr(filename)
    dtype_str = "FP32" if tensor.dtype == torch.float32 else "INT8"

    return {
        "filename": filename,
        "shape": list(tensor.shape),
        "dtype": dtype_str,
        "ndim": tensor.dim(),
        "numel": tensor.numel(),
        "size_bytes": tensor.numel() * (4 if tensor.dtype == torch.float32 else 1)
    }


# =============================================================================
# C++通信服务器支持模块 (V1.19.02新增)
# =============================================================================

class ServerError(Exception):
    """服务器运行时异常"""
    pass


class TechRenaissanceServer:
    """
    技术觉醒框架Python服务器基类
    提供与C++通信的完整基础设施
    """

    def __init__(self, debug: bool = False):
        """
        初始化服务器

        Args:
            debug: 是否启用调试模式
        """
        self.debug = debug
        self.running = True
        self.session_id = -1
        self.session_dir = ""
        self.status_file = ""

        # 智能轮询频率配置
        self.auto_check_frequency = True
        self.wait_counter = 0
        self.shortest_sleep_time = 32  # 毫秒
        self.sleep_time = self.shortest_sleep_time
        self.default_sleep_time = 100  # 毫秒

        # 注册信号处理
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """设置信号处理器，支持优雅退出"""
        def signal_handler(signum, frame):
            self.debug_message(f"[Python] Received signal {signum}, preparing to exit...")
            self.running = False

        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)

    def debug_message(self, *args):
        """调试信息输出"""
        if self.debug:
            print(*args, flush=True)

    def counter_update(self):
        """智能轮询频率调节器"""
        level = self.wait_counter // 8
        if level >= 5:
            self.sleep_time = self.shortest_sleep_time * 32  # 最慢1.024秒
        else:
            self.sleep_time = self.shortest_sleep_time * (1 << level)
            self.wait_counter += 1

    def counter_reset(self):
        """重置轮询频率为最快级别"""
        self.wait_counter = 0
        self.sleep_time = self.shortest_sleep_time

    def init_session(self, session_id: str):
        """
        初始化会话

        Args:
            session_id: 会话ID
        """
        self.session_id = session_id
        # 计算项目根目录和会话目录
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        self.session_dir = f"{project_root}/workspace/python_session/tr_session_{session_id}"

        # 创建会话目录
        os.makedirs(self.session_dir, exist_ok=True)
        self.status_file = os.path.join(self.session_dir, "status.txt")

        # 智能状态管理
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    status = f.read().strip()
                if status != "ready":
                    self.write_status("running")
            except:
                self.write_status("running")
        else:
            self.write_status("running")

        self.debug_message(f"[Python] Session {session_id} started, directory: {self.session_dir}")
        self.debug_message(f"[Python] Waiting for requests...")

    def write_status(self, status_str: str):
        """
        写入状态文件

        Args:
            status_str: 状态字符串
        """
        os.makedirs(self.session_dir, exist_ok=True)
        with open(self.status_file, 'w') as f:
            f.write(status_str)

    def write_response(self, command: str, parameters: str):
        """
        原子写入响应文件

        Args:
            command: 命令名称
            parameters: 参数
        """
        response_file = os.path.join(self.session_dir, "response.json")
        response_file_temp = os.path.join(self.session_dir, "response.tmp")
        output_str = f'{{"cmd": "{command}", "params": "{parameters}"}}'

        self.debug_message(response_file)
        self.debug_message(output_str)

        # 确保会话目录存在
        os.makedirs(self.session_dir, exist_ok=True)

        # 原子操作：等待文件被读取后再写入
        while True:
            if not os.path.exists(response_file):
                break
            # 智能轮询等待
            if self.auto_check_frequency:
                time.sleep(self.sleep_time * 0.001)
                self.counter_update()
            else:
                time.sleep(self.default_sleep_time * 0.001)

        # 原子写入：临时文件 -> 重命名
        with open(response_file_temp, 'w') as f:
            f.write(output_str)
        os.rename(response_file_temp, response_file)

        if self.auto_check_frequency:
            self.counter_reset()

    def process_request(self) -> bool:
        """
        处理请求文件（基类方法，子类应重写main_logic）

        Returns:
            bool: 是否继续运行
        """
        go_on = True
        request_file = os.path.join(self.session_dir, "request.json")

        if not os.path.exists(request_file):
            return go_on  # 没有请求是正常的

        try:
            with open(request_file, 'r') as f:
                request = json.load(f)

            # 重置轮询频率为最快
            if self.auto_check_frequency:
                self.counter_reset()

            cmd = request.get('cmd', '')
            params = request.get('params', '')

            self.debug_message(f"[Python] Processing command: {cmd}, params: {params}")

            if cmd == "exit":
                self.debug_message("[Python] Received exit command")
                go_on = False
                return go_on
            else:
                go_on = self.main_logic(cmd, params)

            if not go_on:
                self.write_status(f"error:Failed to process command {cmd}")

            # 删除请求文件，避免重复处理
            os.remove(request_file)

        except Exception as e:
            self.debug_message(f"[Python] Error processing request: {e}")
            self.write_status(f"error:{str(e)}")
            go_on = False
            return go_on

        return go_on

    def main_logic(self, command: str, parameters: str) -> bool:
        """
        主业务逻辑（子类应重写此方法）

        Args:
            command: 命令名称
            parameters: 参数

        Returns:
            bool: 处理是否成功
        """
        self.debug_message(f"[Python] Unknown command: {command}")
        return False

    def detected_exit_flag(self) -> bool:
        """检查退出标志文件"""
        exit_flag = os.path.join(self.session_dir, "exit.flag")
        if os.path.exists(exit_flag):
            self.debug_message("[Python] Exit flag detected, exiting...")
            return True
        return False

    def detected_exit_status(self) -> bool:
        """检查status.txt文件中的exit状态"""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    status = f.read().strip()
                if status == "exit":
                    self.debug_message("[Python] Exit status detected, exiting...")
                    return True
            except:
                return True
        return False

    def get_tensors(self, p, num_tensors):
        tensors = p.split(',')
        if len(tensors) != num_tensors:
            self.write_response('', 'invalid')
            return None
        else:
            tensor_a_path = os.path.join(self.session_dir, f"{tensors[0]}.tsr")
            tensor_b_path = os.path.join(self.session_dir, f"{tensors[1]}.tsr")
            try:
                tensor_a = import_tsr(tensor_a_path)
                tensor_b = import_tsr(tensor_b_path)
            except:
                return None
            finally:
                os.remove(tensor_a_path)
                os.remove(tensor_b_path)
            return tensor_a, tensor_b

    def send_tensors(self, *tensors):
        num_tensors = len(tensors)
        if num_tensors == 1:
            result_path = os.path.join(self.session_dir, "r.tsr")    # 默认r.tsr是结果
            export_tsr(tensors[0], result_path)
            self.write_response('', 'r')    # 把文件名（无后缀）告诉C++
        else:
            i = 0
            out_params = ''
            for each_tensor in tensors:
                result_path = os.path.join(self.session_dir, f"r{i}.tsr")
                export_tsr(each_tensor, result_path)
                out_params += f'r{i}'
                i += 1
            self.write_response('', out_params[:-1])

    def run(self, session_id: str):
        """
        运行服务器主循环

        Args:
            session_id: 会话ID
        """
        self.init_session(session_id)

        try:
            while self.running:
                if self.detected_exit_flag() or self.detected_exit_status():
                    break

                # 处理请求
                if not self.process_request():
                    break

                # 智能轮询休眠
                if self.auto_check_frequency:
                    time.sleep(self.sleep_time * 0.001)
                    self.counter_update()
                else:
                    time.sleep(self.default_sleep_time * 0.001)

            self.debug_message(f"[Python] Session {session_id} completed run")

        except KeyboardInterrupt:
            self.debug_message("[Python] KeyboardInterrupt received, exiting...")
        except Exception as e:
            self.debug_message(f"[Python] Unexpected error: {e}")
            self.write_status(f"error:{str(e)}")

        # 更新最终状态
        try:
            self.write_status("terminated")
        except:
            pass

        self.debug_message(f"[Python] Session {session_id} terminated")

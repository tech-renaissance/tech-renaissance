#!/usr/bin/env python3
"""
技术觉醒框架Python服务器脚本（V1.19.02重构版）
使用tech_renaissance模块提供的基类，专注于业务逻辑实现
"""

import sys
import os
import torch
from torch.nn import functional as F

# 调试模式开关，设置为False可关闭所有调试输出
DEBUG_MODE = False

# 添加python模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
python_module_path = os.path.join(project_root, 'python', 'module')
sys.path.insert(0, python_module_path)

# 导入tech_renaissance模块
from tech_renaissance import TechRenaissanceServer
from tech_renaissance import import_tsr
from tech_renaissance import export_tsr


class SimpleHelloServer(TechRenaissanceServer):
    """
    简单的Hello World服务器示例
    子类化TechRenaissanceServer，专注于业务逻辑实现
    """

    def main_logic(self, command: str, parameters: str) -> bool:
        """重写父类方法，实现具体的业务逻辑"""
        if DEBUG_MODE: print(f"[PYTHON_DEBUG] Received command: '{command}', parameters: '{parameters}'")

        if command.lower() == 'hello':
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing hello command")
            self.write_response('', f'Hello {parameters.title()}')
        elif command.lower() == 'hi':
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing hi command")
            self.write_response('', f'Hi {parameters.title()}')
        elif command.lower() == 'matmul':    # 执行矩阵乘法！
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing matmul command")
            tensors = self.get_tensors(parameters, 2)
            if tensors is None:
                print(f"[ERROR] Failed to get tensors for matmul")
                self.write_response('', 'invalid')
            else:
                try:
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] Successfully got {len(tensors)} tensors for matmul")
                    result = torch.mm(tensors[0], tensors[1])
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] Matmul computation successful, result shape: {result.shape}")
                    self.send_tensors(result)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] Matmul result sent successfully")
                except Exception as e:
                    print(f"[ERROR] Matmul computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'conv_k3_s1_p0':
            tensors = self.get_tensors(parameters, 2)
            if tensors is None:
                print(f"[ERROR] Failed to get tensors")
                self.write_response('', 'invalid')
            else:
                try:
                    result = F.conv2d(tensors[0], tensors[1], bias=None, stride=1, padding=0)
                    self.send_tensors(result)
                except Exception as e:
                    print(f"[ERROR] Computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'conv_k3_s1_p1':
            tensors = self.get_tensors(parameters, 2)
            if tensors is None:
                print(f"[ERROR] Failed to get tensors")
                self.write_response('', 'invalid')
            else:
                try:
                    result = F.conv2d(tensors[0], tensors[1], bias=None, stride=1, padding=1)
                    self.send_tensors(result)
                except Exception as e:
                    print(f"[ERROR] Computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'conv_k3_s2_p1':
            tensors = self.get_tensors(parameters, 2)
            if tensors is None:
                print(f"[ERROR] Failed to get tensors")
                self.write_response('', 'invalid')
            else:
                try:
                    result = F.conv2d(tensors[0], tensors[1], bias=None, stride=2, padding=1)
                    self.send_tensors(result)
                except Exception as e:
                    print(f"[ERROR] Computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'conv_k1_s1_p0':
            tensors = self.get_tensors(parameters, 2)
            if tensors is None:
                print(f"[ERROR] Failed to get tensors")
                self.write_response('', 'invalid')
            else:
                try:
                    result = F.conv2d(tensors[0], tensors[1], bias=None, stride=1, padding=0)
                    self.send_tensors(result)
                except Exception as e:
                    print(f"[ERROR] Computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'conv_k1_s2_p0':
            tensors = self.get_tensors(parameters, 2)
            if tensors is None:
                print(f"[ERROR] Failed to get tensors")
                self.write_response('', 'invalid')
            else:
                try:
                    result = F.conv2d(tensors[0], tensors[1], bias=None, stride=2, padding=0)
                    self.send_tensors(result)
                except Exception as e:
                    print(f"[ERROR] Computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'conv_k7_s2_p3':
            tensors = self.get_tensors(parameters, 2)
            if tensors is None:
                print(f"[ERROR] Failed to get tensors")
                self.write_response('', 'invalid')
            else:
                try:
                    result = F.conv2d(tensors[0], tensors[1], bias=None, stride=2, padding=3)
                    self.send_tensors(result)
                except Exception as e:
                    print(f"[ERROR] Computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'add':    # 执行矩阵加法！
            tensors = self.get_tensors(parameters, 2)
            if tensors is None:
                self.write_response('', 'invalid')
            else:
                try:
                    result = tensors[0] + tensors[1]
                    self.send_tensors(result)
                except:
                    self.write_response('', 'invalid')
        # ===== 单目运算命令 =====
        elif command.lower() == 'zeros_like':  # 清零
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing zeros_like command")
            tensor = self.get_tensors(parameters, 1)
            if tensor is None:
                print(f"[ERROR] Failed to get tensor for zeros_like")
                self.write_response('', 'invalid')
            else:
                try:
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] Successfully got tensor for zeros_like, shape: {tensor.shape}")
                    result = torch.zeros_like(tensor)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] zeros_like computation successful, result shape: {result.shape}")
                    self.send_tensors(result)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] zeros_like result sent successfully")
                except Exception as e:
                    print(f"[ERROR] zeros_like computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'ones_like':   # 置一
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing ones_like command")
            tensor = self.get_tensors(parameters, 1)
            if tensor is None:
                print(f"[ERROR] Failed to get tensor for ones_like")
                self.write_response('', 'invalid')
            else:
                try:
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] Successfully got tensor for ones_like, shape: {tensor.shape}")
                    result = torch.ones_like(tensor)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] ones_like computation successful, result shape: {result.shape}")
                    self.send_tensors(result)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] ones_like result sent successfully")
                except Exception as e:
                    print(f"[ERROR] ones_like computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'relu':       # ReLU激活
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing relu command")
            tensor = self.get_tensors(parameters, 1)
            if tensor is None:
                print(f"[ERROR] Failed to get tensor for relu")
                self.write_response('', 'invalid')
            else:
                try:
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] Successfully got tensor for relu, shape: {tensor.shape}")
                    result = torch.relu(tensor)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] relu computation successful, result shape: {result.shape}")
                    self.send_tensors(result)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] relu result sent successfully")
                except Exception as e:
                    print(f"[ERROR] relu computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'sign':       # 符号函数
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing sign command")
            tensor = self.get_tensors(parameters, 1)
            if tensor is None:
                print(f"[ERROR] Failed to get tensor for sign")
                self.write_response('', 'invalid')
            else:
                try:
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] Successfully got tensor for sign, shape: {tensor.shape}")
                    result = torch.sign(tensor)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] sign computation successful, result shape: {result.shape}")
                    self.send_tensors(result)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] sign result sent successfully")
                except Exception as e:
                    print(f"[ERROR] sign computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'square':     # 平方
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing square command")
            tensor = self.get_tensors(parameters, 1)
            if tensor is None:
                print(f"[ERROR] Failed to get tensor for square")
                self.write_response('', 'invalid')
            else:
                try:
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] Successfully got tensor for square, shape: {tensor.shape}")
                    result = torch.square(tensor)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] square computation successful, result shape: {result.shape}")
                    self.send_tensors(result)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] square result sent successfully")
                except Exception as e:
                    print(f"[ERROR] square computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'sqrt':       # 平方根
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing sqrt command")
            tensor = self.get_tensors(parameters, 1)
            if tensor is None:
                print(f"[ERROR] Failed to get tensor for sqrt")
                self.write_response('', 'invalid')
            else:
                try:
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] Successfully got tensor for sqrt, shape: {tensor.shape}")
                    result = torch.sqrt(tensor)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] sqrt computation successful, result shape: {result.shape}")
                    self.send_tensors(result)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] sqrt result sent successfully")
                except Exception as e:
                    print(f"[ERROR] sqrt computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'abs':        # 绝对值
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing abs command")
            tensor = self.get_tensors(parameters, 1)
            if tensor is None:
                print(f"[ERROR] Failed to get tensor for abs")
                self.write_response('', 'invalid')
            else:
                try:
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] Successfully got tensor for abs, shape: {tensor.shape}")
                    result = torch.abs(tensor)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] abs computation successful, result shape: {result.shape}")
                    self.send_tensors(result)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] abs result sent successfully")
                except Exception as e:
                    print(f"[ERROR] abs computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'negative':    # 相反数
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing negative command")
            tensor = self.get_tensors(parameters, 1)
            if tensor is None:
                print(f"[ERROR] Failed to get tensor for negative")
                self.write_response('', 'invalid')
            else:
                try:
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] Successfully got tensor for negative, shape: {tensor.shape}")
                    result = torch.negative(tensor)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] negative computation successful, result shape: {result.shape}")
                    self.send_tensors(result)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] negative result sent successfully")
                except Exception as e:
                    print(f"[ERROR] negative computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'reciprocal': # 倒数
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing reciprocal command")
            tensor = self.get_tensors(parameters, 1)
            if tensor is None:
                print(f"[ERROR] Failed to get tensor for reciprocal")
                self.write_response('', 'invalid')
            else:
                try:
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] Successfully got tensor for reciprocal, shape: {tensor.shape}")
                    result = torch.reciprocal(tensor)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] reciprocal computation successful, result shape: {result.shape}")
                    self.send_tensors(result)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] reciprocal result sent successfully")
                except Exception as e:
                    print(f"[ERROR] reciprocal computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'round':      # 四舍五入
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing round command")
            tensor = self.get_tensors(parameters, 1)
            if tensor is None:
                print(f"[ERROR] Failed to get tensor for round")
                self.write_response('', 'invalid')
            else:
                try:
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] Successfully got tensor for round, shape: {tensor.shape}")
                    result = torch.round(tensor)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] round computation successful, result shape: {result.shape}")
                    self.send_tensors(result)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] round result sent successfully")
                except Exception as e:
                    print(f"[ERROR] round computation failed: {e}")
                    self.write_response('', 'invalid')
        elif command.lower() == 'transpose':   # 矩阵转置
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing transpose command")
            tensor = self.get_tensors(parameters, 1)
            if tensor is None:
                print(f"[ERROR] Failed to get tensor for transpose")
                self.write_response('', 'invalid')
            else:
                try:
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] Successfully got tensor for transpose, shape: {tensor.shape}")
                    result = torch.transpose(tensor, 0, 1)  # 转置前两个维度
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] transpose computation successful, result shape: {result.shape}")
                    self.send_tensors(result)
                    if DEBUG_MODE: print(f"[PYTHON_DEBUG] transpose result sent successfully")
                except Exception as e:
                    print(f"[ERROR] transpose computation failed: {e}")
                    self.write_response('', 'invalid')
        else:
            print(f"[ERROR] Unknown command: {command}")
            self.debug_message(f'[Python] Invalid command: {command}')
            return False
        return True


def main():
    """主函数 - 简化为仅需几行代码"""
    if len(sys.argv) != 2:
        print("Usage: python_server.py <session_id>")
        sys.exit(1)

    session_id = sys.argv[1]

    # 创建服务器实例（可启用调试模式）
    server = SimpleHelloServer(debug=False)

    # 运行服务器
    server.run(session_id)


if __name__ == "__main__":
    main()
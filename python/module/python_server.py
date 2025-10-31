#!/usr/bin/env python3
"""
技术觉醒框架Python服务器脚本（V1.19.02重构版）
使用tech_renaissance模块提供的基类，专注于业务逻辑实现
"""

import sys
import os
import torch

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
        if command.lower() == 'hello':
            self.write_response('', f'Hello {parameters.title()}')
        elif command.lower() == 'hi':
            self.write_response('', f'Hi {parameters.title()}')
        elif command.lower() == 'matmul':    # 执行矩阵乘法！
            tensors = self.get_tensors(parameters, 2)
            if tensors is None:
                self.write_response('', 'invalid')
            else:
                try:
                    result = torch.mm(tensors[0], tensors[1])
                    self.send_tensors(result)
                except:
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
        else:
            self.debug_message('[Python] Invalid command.')
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
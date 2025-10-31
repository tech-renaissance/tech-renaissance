#!/usr/bin/env python3
"""
技术觉醒框架张量打印对比服务器脚本（基于TechRenaissanceServer）
使用tech_renaissance模块提供的基类，专注于张量打印对比逻辑
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


class TensorPrintCompareServer(TechRenaissanceServer):
    """
    张量打印对比服务器
    子类化TechRenaissanceServer，专注于张量打印和对比逻辑
    """

    def main_logic(self, command: str, parameters: str) -> bool:
        """重写父类方法，实现具体的张量打印对比逻辑"""
        if command.lower() == 'print_tensor':
            return self.handle_print_tensor(parameters)
        elif command.lower() == 'hello':
            self.write_response('', f'Hello {parameters.title()}')
        elif command.lower() == 'hi':
            self.write_response('', f'Hi {parameters.title()}')
        elif command.lower() == 'exit':
            print("[Python] Received exit command")
            return False
        else:
            print(f"[Python] Unknown command: {command}")
            return True

    def handle_print_tensor(self, tensor_name: str) -> bool:
        """处理张量打印命令"""
        try:
            print(f"[Python] Processing print_tensor command for: {tensor_name}")

            # 从C++接收的张量文件
            input_file = os.path.join(self.session_dir, "input.tsr")

            if os.path.exists(input_file):
                print(f"[Python] Loading tensor from: {input_file}")

                # 使用tech_renaissance模块的import_tsr函数
                tensor = import_tsr(input_file)
                print(f"[Python] Successfully loaded tensor with shape: {tensor.shape}")

                # 打印真正的PyTorch张量
                print("\n--- PyTorch Tensor Output ---")
                print(tensor)

                # 如果指定了精度，也打印精度版本
                if hasattr(tensor, 'dtype') and tensor.dtype == torch.float32:
                    print("\n--- PyTorch Tensor with 2 decimal precision ---")
                    precise_tensor = torch.round(tensor * 100) / 100
                    print(precise_tensor)

                self.write_response('', f'Tensor {tensor_name} printed successfully')

            else:
                print(f"[Python] Input tensor file not found: {input_file}")
                # 列出目录中所有文件用于调试
                print(f"[Python] Files in directory {self.session_dir}:")
                for file in os.listdir(self.session_dir):
                    print(f"[Python]   - {file}")

                # 创建示例张量进行对比
                print("\n--- Creating Example PyTorch Tensor for Comparison ---")
                example_tensor = torch.full((2, 2, 2, 2), 3.14159, dtype=torch.float32)
                print(example_tensor)

                self.write_response('', f'Created example tensor for {tensor_name}')

            return True

        except Exception as e:
            print(f"[Python] Error handling print_tensor: {e}")
            self.write_response('error', f'Failed to print tensor {tensor_name}: {str(e)}')
            return True

def main():
    """主函数 - 使用TechRenaissanceServer基类"""
    if len(sys.argv) != 2:
        print("Usage: tensor_print_compare.py <session_id>")
        sys.exit(1)

    session_id = sys.argv[1]

    # 创建服务器实例
    server = TensorPrintCompareServer(debug=False)

    print(f"\n{'='*60}")
    print(f" PyTorch Tensor Print Comparison Service")
    print(f"{'='*60}")

    # 启动服务器（基类会处理所有通信逻辑）
    server.run(session_id)

if __name__ == "__main__":
    main()
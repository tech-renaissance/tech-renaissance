"""
简单的TSR文件读取测试 - 只读取现有文件并展示
"""

import sys
import os
import torch

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'module'))

import tech_renaissance as tr


def main():
    print("=== TSR文件读取演示 ===")

    workspace_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'workspace')
    tsr_files = [f for f in os.listdir(workspace_dir) if f.endswith('.tsr')]
    tsr_files.sort()

    for filename in tsr_files:
        print(f"\n文件: {filename}")
        print("-" * 30)

        filepath = os.path.join(workspace_dir, filename)
        tensor = tr.import_tsr(filepath)

        print(f"形状: {tensor.shape}")
        print(f"类型: {tensor.dtype}")
        print(f"数据: {tensor}")

    print("\n=== 读取完成 ===")


if __name__ == "__main__":
    main()
"""
技术觉醒框架 - TSR文件读写测试
测试TSR文件的导入导出功能，并与workspace中的现有TSR文件进行验证
"""

import sys
import os
import torch

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'module'))

import tech_renaissance as tr


def test_import_existing_tsr_files():
    """测试导入workspace中现有的TSR文件"""
    print("=" * 60)
    print("测试导入workspace中现有的TSR文件")
    print("=" * 60)

    # 获取workspace目录
    workspace_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'workspace')

    # 列出所有TSR文件
    tsr_files = [f for f in os.listdir(workspace_dir) if f.endswith('.tsr')]
    tsr_files.sort()  # 按名称排序

    print(f"找到 {len(tsr_files)} 个TSR文件:")
    for i, filename in enumerate(tsr_files, 1):
        print(f"\n{i}. {filename}")
        print("-" * 40)

        filepath = os.path.join(workspace_dir, filename)

        try:
            # 获取文件信息
            info = tr.get_tsr_info(filepath)
            print(f"  形状: {info['shape']}")
            print(f"  类型: {info['dtype']}")
            print(f"  维度: {info['ndim']}")
            print(f"  元素数: {info['numel']}")
            print(f"  文件大小: {info['size_bytes']} 字节")

            # 读取张量
            tensor = tr.import_tsr(filepath)
            print(f"  张量类型: {tensor.dtype}")
            print(f"  张量形状: {tensor.shape}")
            print(f"  张量设备: {tensor.device}")

            # 打印部分数据（对于大张量只打印前几个元素）
            if tensor.numel() <= 100:
                print(f"  张量数据: {tensor}")
            else:
                print(f"  张量数据 (前5个): {tensor.flatten()[:5]}")
                print(f"  张量数据 (后5个): {tensor.flatten()[-5:]}")

                # 打印统计信息
                if tensor.dtype == torch.float32:
                    print(f"  数据范围: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
                    print(f"  平均值: {tensor.float().mean().item():.6f}")
                else:
                    print(f"  数据范围: [{tensor.min().item()}, {tensor.max().item()}]")

            print("  [OK] 导入成功")

        except Exception as e:
            print(f"  [ERROR] 导入失败: {e}")

    print(f"\n总计: {len(tsr_files)} 个TSR文件")


def test_export_import_roundtrip():
    """测试导出-导入往返转换"""
    print("\n" + "=" * 60)
    print("测试导出-导入往返转换")
    print("=" * 60)

    # 创建测试张量
    test_cases = [
        ("标量FP32", torch.tensor(3.14159, dtype=torch.float32)),
        ("1D张量FP32", torch.arange(5, dtype=torch.float32) * 0.1),
        ("2D张量FP32", torch.randn(3, 4, dtype=torch.float32)),
        ("3D张量FP32", torch.randn(2, 3, 4, dtype=torch.float32)),
        ("4D张量FP32", torch.randn(2, 3, 4, 5, dtype=torch.float32)),
        ("2D张量INT8", torch.randint(-128, 127, (3, 4), dtype=torch.int8)),
        ("4D张量INT8", torch.randint(-128, 127, (2, 3, 4, 5), dtype=torch.int8)),
    ]

    for i, (name, original_tensor) in enumerate(test_cases, 1):
        print(f"\n{i}. 测试 {name}")
        print("-" * 40)
        print(f"  原始张量形状: {original_tensor.shape}")
        print(f"  原始张量类型: {original_tensor.dtype}")

        # 临时文件路径
        temp_file = f"temp_test_{i}.tsr"

        try:
            # 导出
            tr.export_tsr(original_tensor, temp_file)
            print(f"  [OK] 导出成功: {temp_file}")

            # 导入
            loaded_tensor = tr.import_tsr(temp_file)
            print(f"  [OK] 导入成功")
            print(f"  加载张量形状: {loaded_tensor.shape}")
            print(f"  加载张量类型: {loaded_tensor.dtype}")

            # 验证数据一致性
            if torch.allclose(original_tensor, loaded_tensor, atol=1e-6):
                print(f"  [OK] 数据一致性验证通过")
            else:
                print(f"  [ERROR] 数据一致性验证失败")
                print(f"    原始数据: {original_tensor.flatten()[:5]}")
                print(f"    加载数据: {loaded_tensor.flatten()[:5]}")

            # 验证形状一致性
            if original_tensor.shape == loaded_tensor.shape:
                print(f"  [OK] 形状一致性验证通过")
            else:
                print(f"  [ERROR] 形状一致性验证失败")

            # 验证类型一致性
            if original_tensor.dtype == loaded_tensor.dtype:
                print(f"  [OK] 类型一致性验证通过")
            else:
                print(f"  [ERROR] 类型一致性验证失败")

        except Exception as e:
            print(f"  [ERROR] 测试失败: {e}")

        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"  [CLEAN] 清理临时文件: {temp_file}")


def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 60)
    print("测试错误处理")
    print("=" * 60)

    # 测试不存在的文件
    print("\n1. 测试读取不存在的文件")
    try:
        tr.import_tsr("nonexistent_file.tsr")
        print("  [ERROR] 应该抛出异常但没有")
    except tr.TSRError as e:
        print(f"  [OK] 正确抛出异常: {e}")

    # 测试不支持的数据类型
    print("\n2. 测试不支持的数据类型 (int32)")
    try:
        tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
        tr.export_tsr(tensor, "temp_int32.tsr")
        print("  [ERROR] 应该抛出异常但没有")
    except tr.TSRError as e:
        print(f"  [OK] 正确抛出异常: {e}")

    # 测试超过4维的张量
    print("\n3. 测试超过4维的张量")
    try:
        tensor = torch.randn(2, 3, 4, 5, 6)  # 5维张量
        tr.export_tsr(tensor, "temp_5d.tsr")
        print("  [ERROR] 应该抛出异常但没有")
    except tr.TSRError as e:
        print(f"  [OK] 正确抛出异常: {e}")

    # 测试GPU张量
    if torch.cuda.is_available():
        print("\n4. 测试GPU张量")
        try:
            tensor = torch.randn(2, 3, device='cuda')
            tr.export_tsr(tensor, "temp_gpu.tsr")
            print("  [ERROR] 应该抛出异常但没有")
        except tr.TSRError as e:
            print(f"  [OK] 正确抛出异常: {e}")
    else:
        print("\n4. 跳过GPU张量测试 (CUDA不可用)")


def main():
    """主测试函数"""
    print("技术觉醒框架 - TSR文件读写测试")
    print("Python版本:", sys.version)
    print("PyTorch版本:", torch.__version__)

    try:
        # 测试现有TSR文件
        test_import_existing_tsr_files()

        # 测试往返转换
        test_export_import_roundtrip()

        # 测试错误处理
        test_error_handling()

        print("\n" + "=" * 60)
        print("[SUCCESS] 所有测试完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] 测试过程中发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
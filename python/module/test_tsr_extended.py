"""
TSR文件格式扩展功能Python测试
验证FP32、INT32、INT8三种数据类型的TSR文件导入导出功能
"""

import torch
import os
import sys
import tempfile
import shutil

# 添加当前模块到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tech_renaissance import export_tsr, import_tsr, TSRError


def test_dtype_io(dtype, shape, test_name, value_range=(-100, 100)):
    """
    测试单个数据类型的TSR导入导出

    Args:
        dtype: torch数据类型
        shape: 张量形状
        test_name: 测试名称
        value_range: 随机数值范围 (low, high)
    """
    print(f"\n=== Testing {test_name} ===")

    try:
        # 创建随机测试张量
        if dtype == torch.float32:
            original = torch.randn(shape, dtype=torch.float32)
            print(f"Created FP32 tensor: {list(original.shape)}")
        elif dtype == torch.int32:
            original = torch.randint(value_range[0], value_range[1], shape, dtype=torch.int32)
            print(f"Created INT32 tensor: {list(original.shape)}")
        elif dtype == torch.int8:
            original = torch.randint(value_range[0], value_range[1], shape, dtype=torch.int8)
            print(f"Created INT8 tensor: {list(original.shape)}")
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        temp_filename = os.path.join(temp_dir, f"test_{test_name}.tsr")

        try:
            # 导出到TSR文件
            export_tsr(original, temp_filename)
            print(f"Exported to: {temp_filename}")

            # 检查文件是否存在
            if not os.path.exists(temp_filename):
                print("ERROR: Exported file does not exist!")
                return False

            # 检查文件大小
            file_size = os.path.getsize(temp_filename)
            expected_size = 64 + original.numel() * original.element_size()
            if file_size != expected_size:
                print(f"ERROR: File size mismatch! Expected: {expected_size}, Got: {file_size}")
                return False
            print(f"File size correct: {file_size} bytes")

            # 从TSR文件导入
            imported = import_tsr(temp_filename)
            print(f"Imported tensor: {list(imported.shape)}")

            # 验证张量相等
            if dtype == torch.float32:
                # 对于浮点数，使用torch.allclose
                is_equal = torch.allclose(original, imported, atol=1e-6)
            else:
                # 对于整数，使用torch.equal
                is_equal = torch.equal(original, imported)

            if is_equal:
                print("SUCCESS: Original and imported tensors are identical")
            else:
                print("ERROR: Original and imported tensors differ!")
                return False

            return True

        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("Cleaned up temporary files")

    except Exception as e:
        print(f"ERROR: Exception during test: {e}")
        return False


def test_2d_and_4d_tensors():
    """
    测试2D和4D张量的TSR导入导出
    """
    print("\n=== Testing 2D and 4D Tensors ===")

    # 定义测试配置
    test_configs = [
        # 2D张量测试
        (torch.float32, (3, 4), "fp32_2d", None),
        (torch.int32, (3, 4), "int32_2d", (-100, 100)),
        (torch.int8, (3, 4), "int8_2d", (-50, 50)),

        # 4D张量测试
        (torch.float32, (1, 2, 3, 4), "fp32_4d", None),
        (torch.int32, (1, 2, 3, 4), "int32_4d", (-100, 100)),
        (torch.int8, (1, 2, 3, 4), "int8_4d", (-50, 50)),
    ]

    all_passed = True

    for dtype, shape, test_name, value_range in test_configs:
        print(f"\n--- Testing {test_name} ---")
        print(f"Shape: {shape}, Dtype: {dtype}")

        if not test_dtype_io(dtype, shape, test_name, value_range):
            all_passed = False
            print(f"FAILED: {test_name}")
        else:
            print(f"PASSED: {test_name}")

    return all_passed


def test_backward_compatibility():
    """
    测试向后兼容性 - 确保新的INT32支持不会破坏现有的FP32/INT8功能
    """
    print("\n=== Testing Backward Compatibility ===")

    try:
        # 创建FP32张量并测试
        fp32_tensor = torch.randn(2, 3, dtype=torch.float32)
        temp_dir = tempfile.mkdtemp()
        temp_filename = os.path.join(temp_dir, "compatibility_fp32.tsr")

        try:
            export_tsr(fp32_tensor, temp_filename)
            imported_fp32 = import_tsr(temp_filename)

            fp32_compatible = torch.allclose(fp32_tensor, imported_fp32, atol=1e-6)

            # 创建INT8张量并测试
            int8_tensor = torch.randint(-50, 50, (2, 3), dtype=torch.int8)
            int8_filename = os.path.join(temp_dir, "compatibility_int8.tsr")

            export_tsr(int8_tensor, int8_filename)
            imported_int8 = import_tsr(int8_filename)

            int8_compatible = torch.equal(int8_tensor, imported_int8)

            if fp32_compatible and int8_compatible:
                print("SUCCESS: Backward compatibility maintained for FP32 and INT8")
                return True
            else:
                print("ERROR: Backward compatibility test failed!")
                return False

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"ERROR: Exception during compatibility test: {e}")
        return False


def test_error_handling():
    """
    测试错误处理
    """
    print("\n=== Testing Error Handling ===")

    temp_dir = tempfile.mkdtemp()

    try:
        # 测试不支持的dtype
        print("Testing unsupported dtype...")
        try:
            bool_tensor = torch.tensor([True, False], dtype=torch.bool)
            bool_filename = os.path.join(temp_dir, "bool_test.tsr")
            export_tsr(bool_tensor, bool_filename)
            print("ERROR: Should have failed for bool tensor!")
            return False
        except TSRError as e:
            print(f"Correctly caught error for bool tensor: {e}")

        # 测试不存在的文件
        print("Testing non-existent file...")
        try:
            import_tsr("non_existent_file.tsr")
            print("ERROR: Should have failed for non-existent file!")
            return False
        except TSRError as e:
            print(f"Correctly caught error for non-existent file: {e}")

        print("SUCCESS: Error handling works correctly")
        return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """
    主测试函数
    """
    print("=== TSR Extended I/O Python Test Suite ===")
    print("Testing FP32, INT32, and INT8 data types")

    all_tests_passed = True

    # 测试2D和4D张量
    if not test_2d_and_4d_tensors():
        all_tests_passed = False

    # 测试向后兼容性
    if not test_backward_compatibility():
        all_tests_passed = False

    # 测试错误处理
    if not test_error_handling():
        all_tests_passed = False

    # 输出最终结果
    print("\n=== Test Summary ===")
    if all_tests_passed:
        print("SUCCESS: All Python TSR I/O tests passed!")
        print("FP32, INT32, and INT8 data types are fully supported.")
        return 0
    else:
        print("FAILURE: Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
"""
TSR格式兼容性验证脚本
验证扩展后的TSR格式能够正确读写原有的FP32和INT8文件
"""

import torch
import os
import sys
import tempfile
import shutil

# 添加当前模块到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tech_renaissance import export_tsr, import_tsr, TSRError


def verify_file_format_compatibility():
    """
    验证文件格式兼容性
    确保新的INT32支持不会破坏原有的FP32和INT8文件格式
    """
    print("=== TSR Format Compatibility Verification ===")

    compatibility_passed = True

    # 测试原有格式的兼容性
    test_cases = [
        {
            'name': 'FP32 2D Matrix',
            'tensor': torch.randn(3, 4, dtype=torch.float32),
            'dtype_name': 'FP32'
        },
        {
            'name': 'FP32 4D Tensor',
            'tensor': torch.randn(1, 2, 3, 4, dtype=torch.float32),
            'dtype_name': 'FP32'
        },
        {
            'name': 'INT8 2D Matrix',
            'tensor': torch.randint(-50, 50, (3, 4), dtype=torch.int8),
            'dtype_name': 'INT8'
        },
        {
            'name': 'INT8 4D Tensor',
            'tensor': torch.randint(-20, 20, (1, 2, 3, 4), dtype=torch.int8),
            'dtype_name': 'INT8'
        }
    ]

    for test_case in test_cases:
        print(f"\n--- Testing {test_case['name']} ---")

        try:
            # 创建临时文件
            temp_dir = tempfile.mkdtemp()
            temp_filename = os.path.join(temp_dir, f"compatibility_{test_case['name'].lower().replace(' ', '_')}.tsr")

            # 导出张量
            export_tsr(test_case['tensor'], temp_filename)
            print(f"Exported {test_case['dtype_name']} tensor to TSR file")

            # 验证文件大小
            file_size = os.path.getsize(temp_filename)
            expected_size = 64 + test_case['tensor'].numel() * test_case['tensor'].element_size()

            if file_size != expected_size:
                print(f"ERROR: File size mismatch for {test_case['name']}")
                print(f"Expected: {expected_size} bytes, Got: {file_size} bytes")
                compatibility_passed = False
                continue
            else:
                print(f"File size correct: {file_size} bytes")

            # 导入张量
            imported_tensor = import_tsr(temp_filename)
            print(f"Imported {test_case['dtype_name']} tensor")

            # 验证张量相等
            if test_case['dtype_name'] == 'FP32':
                is_equal = torch.allclose(test_case['tensor'], imported_tensor, atol=1e-6)
            else:  # INT8
                is_equal = torch.equal(test_case['tensor'], imported_tensor)

            if is_equal:
                print(f"SUCCESS: {test_case['name']} compatibility verified")
            else:
                print(f"ERROR: {test_case['name']} tensor data mismatch")
                compatibility_passed = False

            # 清理
            shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            print(f"ERROR: Exception during {test_case['name']} test: {e}")
            compatibility_passed = False

    return compatibility_passed


def verify_dtype_enum_compatibility():
    """
    验证数据类型枚举的兼容性
    确保原有类型值不变，新类型值不冲突
    """
    print("\n=== Data Type Enum Compatibility ===")

    # 创建一个包含所有数据类型的测试张量
    test_tensors = {
        'FP32': torch.randn(2, 3, dtype=torch.float32),
        'INT8': torch.randint(-10, 10, (2, 3), dtype=torch.int8),
        'INT32': torch.randint(-100, 100, (2, 3), dtype=torch.int32)  # 新增类型
    }

    temp_dir = tempfile.mkdtemp()
    enum_compatibility_passed = True

    try:
        for dtype_name, tensor in test_tensors.items():
            filename = os.path.join(temp_dir, f"enum_test_{dtype_name.lower()}.tsr")

            # 导出和导入
            export_tsr(tensor, filename)
            imported = import_tsr(filename)

            # 验证数据类型保持一致
            if imported.dtype != tensor.dtype:
                print(f"ERROR: Dtype mismatch for {dtype_name}")
                enum_compatibility_passed = False
            else:
                print(f"SUCCESS: {dtype_name} enum value consistent")

        # 清理
        shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"ERROR: Exception during enum compatibility test: {e}")
        enum_compatibility_passed = False

    return enum_compatibility_passed


def main():
    """
    主函数：执行所有兼容性验证
    """
    print("TSR Format Compatibility Verification")
    print("Verifying that extended TSR format maintains backward compatibility")

    all_tests_passed = True

    # 验证文件格式兼容性
    if not verify_file_format_compatibility():
        all_tests_passed = False

    # 验证数据类型枚举兼容性
    if not verify_dtype_enum_compatibility():
        all_tests_passed = False

    # 输出结果
    print("\n=== Compatibility Verification Summary ===")
    if all_tests_passed:
        print("SUCCESS: All compatibility tests passed!")
        print("Extended TSR format is fully backward compatible")
        print("Existing FP32 and INT8 files work correctly")
        print("New INT32 support is properly integrated")
        return 0
    else:
        print("FAILURE: Some compatibility tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
import torch
import time

# 设置矩阵大小
M, N, K = 1024, 2048, 512

# 设置随机种子，保证可复现
torch.manual_seed(0)

# 创建CPU上的随机矩阵（正态分布 N(0,1)）
A_cpu = torch.randn((M, N), dtype=torch.float32)
B_cpu = torch.randn((N, K), dtype=torch.float32)

# 在CPU上计算矩阵乘法（用于结果比较）
print("Running on CPU...")
C_cpu = torch.matmul(A_cpu, B_cpu)

# 检查GPU可用性
if torch.cuda.is_available():
    print("Running on GPU...")
    device = torch.device("cpu")
    A_gpu = A_cpu.to(device)
    B_gpu = B_cpu.to(device)

    # 预热 GPU，避免首次调用引起的初始化延迟
    for _ in range(3):
        _ = torch.matmul(A_gpu, B_gpu)
    torch.cuda.synchronize()

    # 计时：仅测量矩阵乘法本身
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(20):
        C_gpu = torch.matmul(A_gpu, B_gpu)
    end_event.record()

    # 等待GPU计算完成
    torch.cuda.synchronize()

    # 计算耗时（毫秒）
    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_s = elapsed_ms / 1000.0

    # 将结果移回CPU
    C_gpu_cpu = C_gpu.to("cpu")

    # 计算误差
    abs_diff = torch.abs(C_cpu - C_gpu_cpu)
    abs_error = abs_diff.mean().item()
    rel_error = abs_error / torch.abs(C_cpu).mean().item()

    # 计算FLOPS性能（矩阵乘法的乘加操作数：2*M*N*K）
    flops = 2 * M * N * K
    gflops = flops / (elapsed_s * 1e9)

    # print(f"\n=== Result Comparison ===")
    print(f"\n\n\nAbsolute error: {abs_error:e}")
    print(f"Relative error: {rel_error:e}")

    # print(f"\n=== GPU Performance ===")
    # print(f"Time for matmul: {elapsed_ms:.3f} ms")
    print(f"Performance: {20*gflops:.2f} GFLOPS")
else:
    print("CUDA not available, skipping GPU test.")

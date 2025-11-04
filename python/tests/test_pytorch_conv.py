import torch
import torch.nn as nn
import time

# 设置设备
device = torch.device('cuda')

# 输入与卷积层参数
batch_size = 32
in_channels = 512
out_channels = 512
height = 7
width = 7
kernel_size = 1  # kernel_size = 3
padding = 0  # padding = 1

# 构造输入与卷积层
x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=torch.float32)
conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False).to(device)
# conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False).to(device)

# 预热
for _ in range(10):
    y = conv(x)
torch.cuda.synchronize()

# 正式测试
n_iters = 100
start = time.time()
for _ in range(n_iters):
    y = conv(x)
torch.cuda.synchronize()
end = time.time()

elapsed = end - start
avg_time = elapsed / n_iters

# 计算 FLOPs
# 对于卷积：每个输出元素执行 Cin * Kh * Kw 次乘加
# 乘加算作 2 FLOPs
out_h = height  # 因为 padding=1, stride=1
out_w = width
flops_per_instance = 2 * in_channels * kernel_size * kernel_size * out_h * out_w
total_flops = flops_per_instance * out_channels * batch_size
gflops = total_flops / avg_time / 1e9

print(f"平均耗时: {avg_time*1e3:.3f} ms")
print(f"平均性能: {gflops:.2f} GFLOPS")

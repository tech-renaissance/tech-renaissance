"""
PyTorch训练脚本 - 与Tech Renaissance框架对齐测试
创建简单的两层神经网络(4-5-2)进行训练，导出所有中间结果为TSR文件
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# 添加当前目录到路径，确保可以导入tech_renaissance
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tech_renaissance import export_tsr

def main():
    print("=== PyTorch Training Script for Tech Renaissance Alignment ===")

    # 设置随机种子以确保可重现性
    torch.manual_seed(42)

    # 1. 创建简单数据集 (2个样本，4维特征)
    # 样本1: 特征[0.5, -0.3, 0.8, -0.1], 标签[1.0, 0.0] (类别0)
    # 样本2: 特征[-0.2, 0.8, 0.1, -0.5], 标签[0.0, 1.0] (类别1)
    data = torch.tensor([
        [0.5, -0.3, 0.8, -0.1],   # 样本1
        [-0.2, 0.8, 0.1, -0.5]    # 样本2
    ], dtype=torch.float32)

    labels = torch.tensor([
        [1.0, 0.0],   # 样本1的one-hot标签
        [0.0, 1.0]    # 样本2的one-hot标签
    ], dtype=torch.float32)

    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")

    # 2. 导出数据集和标签
    export_tsr(data, "data.tsr")
    export_tsr(labels, "labels.tsr")
    print("Exported dataset and labels to TSR files")

    # 3. 创建神经网络 (4-5-2)
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.layer1 = nn.Linear(4, 5)  # 输入4维 -> 隐藏层5维
            self.activation = nn.Tanh()
            self.layer2 = nn.Linear(5, 2)  # 隐藏层5维 -> 输出2维

        def forward(self, x):
            x = self.layer1(x)
            x = self.activation(x)
            x = self.layer2(x)
            return x

    model = SimpleNN()
    print("Created neural network: 4-5-2 architecture")

    # 4. 导出初始化后的权重
    export_tsr(model.layer1.weight, "init_l1_weight.tsr")
    export_tsr(model.layer1.bias, "init_l1_bias.tsr")
    export_tsr(model.layer2.weight, "init_l2_weight.tsr")
    export_tsr(model.layer2.bias, "init_l2_bias.tsr")
    print("Exported initial weights to TSR files")

    # 5. 设置优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    criterion = nn.CrossEntropyLoss()

    print("Set up SGD optimizer (lr=0.1, momentum=0.9, weight_decay=1e-4)")
    print("Set up CrossEntropyLoss")

    # 6. 训练循环 (1个epoch, 2个batch, batch_size=1)
    model.train()

    for batch_idx in range(2):  # 2个batch
        print(f"\n=== Batch {batch_idx + 1} ===")

        # 获取当前batch的数据
        batch_data = data[batch_idx:batch_idx+1]  # [1, 4]
        batch_labels = labels[batch_idx:batch_idx+1]  # [1, 2]

        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        logits = model(batch_data)  # [1, 2]
        print(f"Logits shape: {logits.shape}")
        print(f"Logits values: {logits}")

        # 导出logits
        export_tsr(logits, f"batch{batch_idx + 1}_logits.tsr")

        # 计算loss (需要转换为LongTensor标签)
        batch_labels_long = torch.argmax(batch_labels, dim=1)  # [1]
        loss = criterion(logits, batch_labels_long)
        print(f"Loss value: {loss.item()}")

        # 导出loss (创建标量tensor)
        loss_tensor = torch.tensor([loss.item()], dtype=torch.float32)
        export_tsr(loss_tensor, f"batch{batch_idx + 1}_loss.tsr")

        # 反向传播
        loss.backward()

        # 导出梯度
        export_tsr(model.layer1.weight.grad, f"batch{batch_idx + 1}_l1_weight_grad.tsr")
        export_tsr(model.layer1.bias.grad, f"batch{batch_idx + 1}_l1_bias_grad.tsr")
        export_tsr(model.layer2.weight.grad, f"batch{batch_idx + 1}_l2_weight_grad.tsr")
        export_tsr(model.layer2.bias.grad, f"batch{batch_idx + 1}_l2_bias_grad.tsr")
        print("Exported gradients to TSR files")

        # 更新权重
        optimizer.step()

        # 导出更新后的权重
        export_tsr(model.layer1.weight, f"batch{batch_idx + 1}_l1_weight.tsr")
        export_tsr(model.layer1.bias, f"batch{batch_idx + 1}_l1_bias.tsr")
        export_tsr(model.layer2.weight, f"batch{batch_idx + 1}_l2_weight.tsr")
        export_tsr(model.layer2.bias, f"batch{batch_idx + 1}_l2_bias.tsr")
        print("Exported updated weights to TSR files")

        print(f"Batch {batch_idx + 1} completed")

    print("\n=== Training Complete ===")
    print("All TSR files have been exported to the current directory")

    # 7. 打印所有生成的TSR文件
    print("\nGenerated TSR files:")
    tsr_files = [
        "data.tsr", "labels.tsr",
        "init_l1_weight.tsr", "init_l1_bias.tsr",
        "init_l2_weight.tsr", "init_l2_bias.tsr",
        "batch1_logits.tsr", "batch1_loss.tsr",
        "batch1_l1_weight_grad.tsr", "batch1_l1_bias_grad.tsr",
        "batch1_l2_weight_grad.tsr", "batch1_l2_bias_grad.tsr",
        "batch1_l1_weight.tsr", "batch1_l1_bias.tsr",
        "batch1_l2_weight.tsr", "batch1_l2_bias.tsr",
        "batch2_logits.tsr", "batch2_loss.tsr",
        "batch2_l1_weight_grad.tsr", "batch2_l1_bias_grad.tsr",
        "batch2_l2_weight_grad.tsr", "batch2_l2_bias_grad.tsr",
        "batch2_l1_weight.tsr", "batch2_l1_bias.tsr",
        "batch2_l2_weight.tsr", "batch2_l2_bias.tsr"
    ]

    for tsr_file in tsr_files:
        if os.path.exists(tsr_file):
            print(f"  [OK] {tsr_file}")
        else:
            print(f"  [MISSING] {tsr_file}")

if __name__ == "__main__":
    main()
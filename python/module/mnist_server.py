#!/usr/bin/env python3
"""
技术觉醒框架Python服务器脚本（V1.19.02重构版）
使用tech_renaissance模块提供的基类，专注于业务逻辑实现
"""

import sys
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from six.moves import urllib


BATCH_SIZE = 10000
NUM_EPOCHS = 20
NUM_WORKERS = 0
data_list = list()
label_list = list()
output_list = list()
loss_list = list()
data_batch_id = 0
label_batch_id = 0
output_batch_id = 0
loss_batch_id = 0
weights = list()


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

samples = None
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.name = 'MLP'
        self.classifier = None
        self.make_layers()

    def forward(self, x):
        global samples
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        x = x.view(-1, num_features)
        if samples is None:
            samples = x
        x = self.classifier(x)
        return x

    def make_layers(self):
        classifier_layers = list()
        classifier_layers += [nn.Linear(784, 512, bias=False)]
        classifier_layers += [nn.Tanh()]
        classifier_layers += [nn.Linear(512, 256, bias=False)]
        classifier_layers += [nn.Tanh()]
        classifier_layers += [nn.Linear(256, 10, bias=False)]
        self.classifier = nn.Sequential(*classifier_layers)


class SimpleHelloServer(TechRenaissanceServer):
    """
    简单的Hello World服务器示例
    子类化TechRenaissanceServer，专注于业务逻辑实现
    """

    def main_logic(self, command: str, parameters: str) -> bool:
        global data_batch_id, label_batch_id, output_batch_id, loss_batch_id
        """重写父类方法，实现具体的业务逻辑"""
        if DEBUG_MODE: print(f"[PYTHON_DEBUG] Received command: '{command}', parameters: '{parameters}'")

        if command.lower() == 'hello':
            if DEBUG_MODE: print(f"[PYTHON_DEBUG] Processing hello command")
            self.write_response('', f'Hello {parameters.title()}')
        elif command.lower() == 'data':
            self.send_tensors(data_list[data_batch_id])
            data_batch_id += 1
        elif command.lower() == 'label':
            result = label_list[label_batch_id]
            result = result.to(torch.float)
            self.send_tensors(result)
            label_batch_id += 1
        elif command.lower() == 'output':
            self.send_tensors(output_list[output_batch_id])
            output_batch_id += 1
        elif command.lower() == 'loss':
            self.send_tensors(loss_list[loss_batch_id])
            loss_batch_id += 1
        elif command.lower() == 'samples':
            self.send_tensors(samples)
        elif command.lower() == 'weights0':
            self.send_tensors(weights[0])
        elif command.lower() == 'weights1':
            self.send_tensors(weights[1])
        elif command.lower() == 'weights2':
            self.send_tensors(weights[2])
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
        else:
            print(f"[ERROR] Unknown command: {command}")
            self.debug_message(f'[Python] Invalid command: {command}')
            return False
        return True


def main():
    global data_list, label_list, output_list, loss_list, weights
    """主函数 - 简化为仅需几行代码"""
    if len(sys.argv) != 2:
        print("Usage: python_server.py <session_id>")
        sys.exit(1)

    session_id = sys.argv[1]




    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='R:\\', train=False, transform=test_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load('R:\\tech-renaissance\\python\\module\\models\\best_model.pth', map_location='cpu'))
    model.eval()
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data)

    with torch.no_grad():
        for data, target in test_loader:
            data_list.append(data)
            label_list.append(target)
            output = model(data)
            output_list.append(output)
            loss_list.append(criterion(output, target))

    print('Hello World3!')
    # 创建服务器实例（可启用调试模式）
    server = SimpleHelloServer(debug=False)

    # 运行服务器
    server.run(session_id)


if __name__ == "__main__":
    main()

import torch
import torchvision.models as models


model = torch.nn.Sequential(
    torch.nn.Linear(2, 2, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(2, 2, bias=False)
)
# model = models.resnet18(pretrained=True)
model.eval()          # 必须！BN、Dropout 等行为与训练期不同

print(model)
with torch.no_grad():                       # 必须停住自动求导
    new_w = torch.zeros(2, 2)
    model[0].weight.copy_(new_w)

input_tensor = torch.rand((1, 2), dtype=torch.float32)

torch.onnx.export(
    model,                  # model to export
    (input_tensor,),        # inputs of the model,
    "my_model.onnx",        # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    dynamo=False             # True or False to select the exporter to use
)
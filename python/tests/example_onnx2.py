import onnx

# 1. 加载模型
model = onnx.load("my_model.onnx")   # 或 .pb / .prototxt

# 2. 打印最简摘要
print(onnx.printer.to_text(model.graph))

# 3. 想单独看节点、输入、输出、维度
print("Inputs :", [i.name  + " : " + str(i.type) for i in model.graph.input])
print("Outputs:", [o.name + " : " + str(o.type) for o in model.graph.output])
for n in model.graph.node:
    print(n.op_type, n.name, n.input, "->", n.output)

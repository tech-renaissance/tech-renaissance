import numpy as np
import torch

n = np.array([1,2,3,4,5,6,7,8])
t = torch.from_numpy(n)
t = t.reshape([2,2,2])
print(t)
t.transpose_(1,0)
print(t)

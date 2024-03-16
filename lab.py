import numpy as np
import torch

a = np.random.rand(3, 3)
b = torch.FloatTensor(a)

a = a**2
print(a, b)

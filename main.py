import torch
from git.repo.fun import touch
from torch.onnx.symbolic_opset9 import tensor
from torch.signal.windows import cosine
import math

from mathlib.math_object.function import funtion
from Gradient_descent import *
import numpy as np

def cal(x):
    y = x[0]**2 / 3 + x[1]**2 / 4 - 2 * math.sqrt(x[1])
    return y

f = funtion(cal)
alpha = 0.2
ten = torch.tensor([500,23051],dtype=torch.float, requires_grad=True)
min = gradient_descent(alpha, f, ten)

print(min)
print(cal(min))


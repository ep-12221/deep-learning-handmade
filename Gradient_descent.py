from xmlrpc.client import Error

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mathlib.math_object.function import funtion
import torch



def norm(x):
    return torch.norm(x)


def gradient_descent(alpha, function, start_point):
    x = torch.tensor(start_point, dtype=torch.float, requires_grad=True)
    x_history = [x]
    cnt = 0
    while True:
        cnt += 1
        grad = function.gradient(x)
        if grad is None:
            raise ValueError("Gradient is None. Check the definition of function.gradient.")
        if norm(grad) < 1e-6:
            break
        x = x - alpha * function.gradient(x)
        x_history.append(x)
        if cnt > 1e7:
            raise Error("Reach max iteration.")
        if cnt % 1000 == 0:
            print(cnt)
    return x



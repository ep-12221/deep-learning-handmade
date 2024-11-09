import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mathlib.math_object.function import funtion



def norm(x):
    return np.sqrt(
        np.sum(x**2)
    )


def gradient_descent(alpha, function, start_point):
    x = start_point
    x_history = [x]
    while norm(function.gradient(x)) > 1e-6:
        x = x - alpha * function.gradient(x)
        x_history.append(x)
    return x



import numpy as np


def linear_combination(x, y, a):
    return a * x + (1. - a) * y


def rms(x, eps):
    return np.sqrt(np.mean(x**2) + eps)

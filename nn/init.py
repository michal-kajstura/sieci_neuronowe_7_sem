import numpy as np


def init_weights(weight: np.ndarray, std=0.3):
    return np.random.randn(*weight.shape) * std

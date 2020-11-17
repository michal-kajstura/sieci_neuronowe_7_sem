import numpy as np
from numpy.random import randn


def random_normal_init(weight: np.ndarray, std=0.3):
    return randn(*weight.shape) * std


def xavier_init(weight: np.ndarray):
    n_in, n_out = weight.shape
    variance = 2. / (n_in + n_out)
    return randn(n_in, n_out) * np.sqrt(variance)


def he_init(weight: np.ndarray):
    n_in, n_out = weight.shape
    variance = 2. / n_in
    return randn(n_in, n_out) * np.sqrt(variance)

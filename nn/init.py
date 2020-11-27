import numpy as np
from numpy.random import randn


def random_normal_init(weight: np.ndarray, std=0.3):
    return randn(*weight.shape) * std


def xavier_init(weight: np.ndarray):
    n_in, n_out = _get_shapes(weight.shape)
    variance = 2. / (n_in + n_out)
    return randn(*weight.shape) * np.sqrt(variance)


def he_init(weight: np.ndarray):
    n_in, n_out = _get_shapes(weight.shape)
    variance = 2. / n_in
    return randn(*weight.shape) * np.sqrt(variance)


def _get_shapes(shape):
    n_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    n_out = shape[1] if len(shape) == 2 else shape[0]
    return n_in, n_out
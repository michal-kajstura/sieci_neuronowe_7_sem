from nn.init import random_normal_init
from nn.layers.base import Layer
import numpy as np


class Linear(Layer):
    def __init__(self, in_size, out_size, init_func=random_normal_init):
        self._weight = init_func(np.zeros((in_size, out_size), dtype=np.float32))
        self._bias = np.zeros((1, out_size))

    def forward(self, x):
        z = x.dot(self._weight)
        return z + self._bias, {'x': x}

    def backward(self, grads, cache):
        d_downsream = grads.dot(self._weight.T)
        d_weight = cache['x'].T.dot(grads)
        d_bias = grads.sum(axis=0, keepdims=True)
        return d_downsream, {'_weight': d_weight, '_bias': d_bias}

    @property
    def weights(self):
        return {'_weight': self._weight, '_bias': self._bias}
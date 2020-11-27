from typing import Iterable

import numpy as np

from nn.init import random_normal_init
from nn.layers.base import Layer


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 init_func=random_normal_init):
        kernel_size = kernel_size
        self._weight = init_func(
            np.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32))
        self._stride = stride
        self._padding = padding
        self._bias = np.zeros((1, out_channels))

    def forward(self, x):
        batch, _, height, width = x.shape
        channels_out, _, filter_height, filter_width = self._weight.shape
        height_out = _compute_size(height, filter_height, self._padding, self._stride)
        width_out = _compute_size(width, filter_width, self._padding, self._stride)

        padding = (self._padding, self._padding)
        x_padded = np.pad(x, ((0, 0), (0, 0), padding, padding), mode='constant', constant_values=0)
        filters_reshaped = self._weight.reshape(channels_out, -1).T

        out = np.zeros((batch, channels_out, height_out, width_out), dtype=self._weight.dtype)
        for h in range(height_out):
            for w in range(width_out):
                h_start, w_start = h * self._stride, w * self._stride
                h_end, w_end = h_start + filter_height, w_start + filter_width
                window = x_padded[..., h_start: h_end, w_start: w_end]
                out[..., h, w] = np.dot(
                    window.reshape(batch, -1),
                    filters_reshaped,
                ) + self._bias

        return out, {'x': x_padded}

    def backward(self, grads, cache):
        channels_out, channels_in, filter_height, filter_width = self._weight.shape
        filters_reshaped = self._weight.reshape(channels_out, -1)

        x = cache['x']
        d_downstream = np.zeros_like(x)
        d_weight = np.zeros_like(self._weight)
        batch, _, height, width = grads.shape
        for h in range(height):
            for w in range(width):
                h_start, w_start = h * self._stride, w * self._stride
                h_end, w_end = h_start + filter_height, w_start + filter_width
                d_downstream[..., h_start: h_end, w_start: w_end] = np.dot(
                    grads[..., h, w].reshape(batch, -1),
                    filters_reshaped,
                ).reshape((batch, channels_in, filter_height, filter_width))
                d_weight += grads[..., h, w].T.dot(
                    x[..., h_start: h_end, w_start: w_end].reshape(batch, -1)
                ).reshape(self._weight.shape)

        d_bias = grads.sum(axis=(0, 2, 3))
        return d_downstream, {'_weight': d_weight, '_bias': d_bias}

    @property
    def weights(self):
        return {'_weight': self._weight, '_bias': self._bias}


def _maybe_cast_to_tuple(x):
    return x if isinstance(x, Iterable) else (x, x)

def _compute_size(image_size, filter_size, padding, stride):
    return 1 + (image_size + 2 * padding - filter_size) // stride
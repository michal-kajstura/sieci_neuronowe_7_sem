from nn.layers.base import Layer
from nn.layers.conv import _compute_size
import numpy as np


class MaxPool(Layer):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

    def forward(self, x):
        batch, channels, height, width = x.shape
        height_out = _compute_size(height, self._kernel_size, self._padding, self._stride)
        width_out = _compute_size(width, self._kernel_size, self._padding, self._stride)

        padding = (self._padding, self._padding)
        x_padded = np.pad(x, ((0, 0), (0, 0), padding, padding), mode='constant', constant_values=0)

        out = np.zeros((batch, channels, height_out, width_out), dtype=x.dtype)
        mask = np.zeros_like(x_padded, dtype=np.uint8)
        for h in range(height_out):
            for w in range(width_out):
                h_start, w_start = h * self._stride, w * self._stride
                h_end, w_end = h_start + self._kernel_size, w_start + self._kernel_size
                window = x_padded[:, :, h_start: h_end, w_start: w_end]
                flat_window = window.reshape(*window.shape[:2], -1)
                window_mask = flat_window.argmax(-1)[..., None] == range(flat_window.shape[-1])
                out[..., h, w] = flat_window[window_mask].reshape(window.shape[:2])
                mask[..., h_start: h_end, w_start: w_end] = window_mask.reshape((*window.shape[:2], 2, 2))

        return out, {'x': x_padded, 'mask': mask.astype(bool)}

    def backward(self, grads, cache):
        x = cache['x']
        mask = cache['mask']

        d_downstream = np.zeros_like(x)
        batch, _, height, width = grads.shape
        for h in range(height):
            for w in range(width):
                h_start, w_start = h * self._stride, w * self._stride
                h_end, w_end = h_start + self._kernel_size, w_start + self._kernel_size
                mask_window = mask[..., h_start: h_end, w_start: w_end]
                d_downstream[..., h_start: h_end, w_start: w_end][mask_window] = grads[..., h, w].flatten()

        return d_downstream, {}

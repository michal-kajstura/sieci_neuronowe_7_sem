from nn.layers.base import Layer
import numpy as np


class ReLU(Layer):
    def forward(self, x):
        return np.maximum(0, x), {'x': x}

    def backward(self, grads, cache):
        x = cache['x']
        grads[x < 0] = 0
        return grads, {}


class Sigmoid(Layer):
    def forward(self, x):
        sigmoid = 1. / (1. + np.exp(-x))
        return sigmoid, {'sigmoid': sigmoid}

    def backward(self, grads, cache):
        sigmoid = cache['sigmoid']
        d_sigmoid = sigmoid * (1. - sigmoid)
        return d_sigmoid * grads, {}


class Tanh(Layer):
    def forward(self, x):
        tanh = np.tanh(x)
        return tanh, {'tanh': tanh}

    def backward(self, grads, cache):
        return (1 - cache['tanh']**2) * grads, {}

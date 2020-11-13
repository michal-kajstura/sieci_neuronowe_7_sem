from collections import OrderedDict

from nn.layers.base import Layer


class Model(Layer):
    def __init__(self, *layers):
        self._layers = OrderedDict(
            {f'{i}_{type(layer).__name__}': layer for i, layer in enumerate(layers)})

    def forward(self, x):
        caches = {}
        for name, layer in self._layers.items():
            x, cache = layer.forward(x)
            caches[name] = cache
        return x, {'caches': caches}

    def backward(self, upstream_grads, cache):
        grads = {}
        caches = cache['caches']
        for name in reversed(self._layers):
            layer = self._layers[name]
            cache = caches[name]
            upstream_grads, grad = layer.backward(upstream_grads, cache)
            grads[name] = grad

        return grads

    @property
    def weights(self):
        return {name: layer.weights for name, layer in self._layers.items()}
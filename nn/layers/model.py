from nn.layers.base import Layer


class Model(Layer):
    def __init__(self, *layers):
        self._layers = layers

    def forward(self, x):
        caches = []
        for layer in self._layers:
            x, cache = layer.forward(x)
            caches.append(cache)
        return x, {'caches': caches}

    def backward(self, upstream_grads, cache):
        grads = []
        caches = cache['caches']
        for idx in reversed(range(len(self._layers))):
            layer = self._layers[idx]
            cache = caches[idx]
            upstream_grads, grad = layer.backward(upstream_grads, cache)
            grads.append(grad)

        return grads[::-1]

    @property
    def weights(self):
        return [layer.weights for layer in self._layers]
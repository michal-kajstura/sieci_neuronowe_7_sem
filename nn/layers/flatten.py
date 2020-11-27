from nn.layers.base import Layer


class Flatten(Layer):
    def forward(self, x):
        return x.reshape(len(x), -1), {'shape': x.shape}

    def backward(self, grads, cache):
        return grads.reshape(cache['shape']), {}

import abc


class Layer(abc.ABC):
    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def backward(self, grads, cache):
        pass

    @property
    def weights(self):
        return {}

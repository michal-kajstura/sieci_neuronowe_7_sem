import abc

import numpy as np


class BaseModel(abc.ABC):
    def __init__(self, size, interval):
        self.weights = self._init_weights(size, (-interval, interval))

    @abc.abstractmethod
    def forward(self, x):
        pass

    @staticmethod
    def _init_weights(size, interval):
        return np.random.uniform(*interval, size)

    def update_weights(self, learning_rate, delta):
        self.weights += learning_rate * delta


class Perceptron(BaseModel):
    def __init__(self, size, theta=0, interval=1, bipolar=False):
        super().__init__(size, interval)
        self._theta = theta
        self._bipolar = bipolar

    def forward(self, x):
        result = x.dot(self.weights)
        activated = (result > self._theta).astype(float)
        if self._bipolar:
            activated = activated * 2 - 1
        return activated


class Adaline(BaseModel):
    def __init__(self, size, interval=1):
        super().__init__(size, interval)

    def forward(self, x):
        return x.dot(self.weights)

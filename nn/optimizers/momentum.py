from typing import Dict, Any

import numpy as np

from nn.optimizers.base import BaseOptimizer


class Momentum(BaseOptimizer):
    def __init__(self, layers: Dict[str, Any], learning_rate: float,
                 momentum: float=0.7):
        super().__init__(layers)
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._cache = {}

    def _update(self, layer: np.ndarray, grad: np.ndarray, name: str):
        v_t_minus_1 = self._cache.get(name, 0.)
        v = self._momentum * v_t_minus_1 + self._learning_rate * grad
        layer -= v
        self._cache[name] = v


class NesterovMomentum(Momentum):
    def __init__(self, layers: Dict[str, Any], learning_rate: float,
                 momentum: float = 0.7):
        super().__init__(layers, learning_rate, momentum)

    def _update(self, layer: np.ndarray, grad: np.ndarray, name: str):
        v_t_minus_1 = self._cache.get(name, 0.)
        v = self._momentum * v_t_minus_1 + self._learning_rate * (grad - self._momentum * v_t_minus_1)
        layer -= v
        self._cache[name] = v
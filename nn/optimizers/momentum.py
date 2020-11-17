from typing import Dict, Any

import numpy as np

from nn.optimizers.base import BaseOptimizer


class Momentum(BaseOptimizer):
    def __init__(self, weights: Dict[str, Any], learning_rate: float,
                 momentum: float=0.7):
        super().__init__(weights)
        self._learning_rate = learning_rate
        self._momentum = momentum

    def _update(self, weight: np.ndarray, grad: np.ndarray, cache: Dict[str, Any]) -> Dict[str, Any]:
        v_t_minus_1 = cache.get('v', 0.)

        v = self._momentum * v_t_minus_1 + self._learning_rate * grad
        weight -= v

        return {'v': v}


class NesterovMomentum(Momentum):
    def __init__(self, weights: Dict[str, Any], learning_rate: float,
                 momentum: float = 0.7):
        super().__init__(weights, learning_rate, momentum)

    def _update(self, weight: np.ndarray, grad: np.ndarray, cache: Dict[str, Any]) -> Dict[str, Any]:
        v_t_minus_1 = cache.get('v', 0.)

        v = self._momentum * v_t_minus_1 + self._learning_rate * (grad - self._momentum * v_t_minus_1)
        weight -= v

        return {'v': v}
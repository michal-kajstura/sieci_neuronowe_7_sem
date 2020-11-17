from typing import Dict, Any

import numpy as np

from nn.optimizers.base import BaseOptimizer


class Adagrad(BaseOptimizer):
    def __init__(self, weights: Dict[str, Any], learning_rate: float, eps: float = 1e-8):
        super().__init__(weights)
        self._learning_rate = learning_rate
        self._cache = {}
        self._eps = eps

    def _update(self, weight: np.ndarray, grad: np.ndarray, cache: Dict[str, Any]) -> Dict[str, Any]:
        g_t_i = self._cache.get('g', np.zeros_like(grad))

        g_t_i += grad**2
        lr = self._learning_rate / np.sqrt(g_t_i + self._eps)
        weight -= lr * grad

        return {'g': g_t_i}
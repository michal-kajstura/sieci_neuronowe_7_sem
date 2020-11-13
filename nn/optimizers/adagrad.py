from typing import Dict, Any

from pandas import np

from nn.optimizers.base import BaseOptimizer


class Adagrad(BaseOptimizer):
    def __init__(self, layers: Dict[str, Any], learning_rate: float, eps: float = 1e-8):
        super().__init__(layers)
        self._learning_rate = learning_rate
        self._cache = {}
        self._eps = eps

    def _update(self, layer: np.ndarray, grad: np.ndarray, name: str):
        g_t_i = self._cache.get(name, np.zeros_like(grad))
        g_t_i += grad**2
        g = grad / np.sqrt(g_t_i + self._eps)
        layer -= self._learning_rate * g
        self._cache[name] = g_t_i
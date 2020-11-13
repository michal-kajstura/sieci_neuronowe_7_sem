from typing import Dict, Any

from pandas import np

from nn.optimizers.base import BaseOptimizer
from nn.optimizers.utils import linear_combination


class Adadelta(BaseOptimizer):
    def __init__(self, layers: Dict[str, Any], learning_rate: float, delta: float = 0.9,
                 eps: float = 1e-8):
        super().__init__(layers)
        self._learning_rate = learning_rate
        self._cache = {}
        self._delta = delta
        self._eps = eps

    def _update(self, layer: np.ndarray, grad: np.ndarray, cache: Dict[str, Any]) -> Dict[str, Any]:
        g_avg_t_minus_1 = cache.get('g_avg', np.zeros_like(grad))
        update_avg_t_minus_1 = cache.get('u_avg', np.zeros_like(grad))

        g_avg_square = linear_combination(np.mean(g_avg_t_minus_1**2), grad**2, self._delta)

        update = - (grad * self._learning_rate) / np.sqrt(g_avg_square + self._eps)
        update_avg_square = linear_combination(np.mean(update_avg_t_minus_1**2), update**2, self._delta)

        g_rms = np.sqrt(g_avg_square + self._eps)
        update_rms = np.sqrt(update_avg_square + self._eps)

        layer -= (update_rms / g_rms) * grad

        return {'g_avg': g_avg_square, 'u_avg': update_avg_square}

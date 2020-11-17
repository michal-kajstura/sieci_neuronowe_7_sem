from typing import Dict, Any

from pandas import np

from nn.optimizers.base import BaseOptimizer
from nn.optimizers.utils import linear_combination


class Adadelta(BaseOptimizer):
    def __init__(self, weights: Dict[str, Any], delta: float = 0.9,
                 eps: float = 1e-8):
        super().__init__(weights)
        self._cache = {}
        self._delta = delta
        self._eps = eps

    def _update(self, weight: np.ndarray, grad: np.ndarray, cache: Dict[str, Any]) -> Dict[str, Any]:
        g_avg_t_minus_1 = cache.get('g_avg', np.zeros_like(grad))
        update_avg_t_minus_1 = cache.get('u_avg', np.zeros_like(grad))

        g_avg = linear_combination(g_avg_t_minus_1, grad**2, self._delta)

        update = grad * ((np.sqrt(update_avg_t_minus_1 + self._eps)) / np.sqrt(g_avg + self._eps))

        update_avg = linear_combination(update_avg_t_minus_1, update**2, self._delta)
        weight -= update

        return {'g_avg': g_avg, 'u_avg': update_avg}

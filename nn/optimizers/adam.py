from typing import Dict, Any

import numpy as np

from nn.optimizers.base import BaseOptimizer
from nn.optimizers.utils import linear_combination


class Adam(BaseOptimizer):
    def __init__(self, weights: Dict[str, Any], learning_rate: float,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(weights)
        self._learning_rate = learning_rate
        self._cache = {}
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps

    def _update(self, weight: np.ndarray, grad: np.ndarray, cache: Dict[str, Any]) -> Dict[str, Any]:
        m_t_minus_1 = cache.get('m', np.zeros_like(grad))
        v_t_minus_1 = cache.get('v', np.zeros_like(grad))
        timestep = cache.get('timestep', 1)

        m = linear_combination(m_t_minus_1, grad, self._beta1)
        v = linear_combination(v_t_minus_1, grad**2, self._beta2)

        adjusted_m = m / (1. - self._beta1**timestep)
        adjusted_v = v / (1. - self._beta2**timestep)

        weight -= (self._learning_rate * adjusted_m) / (np.sqrt(adjusted_v) + self._eps)

        return {'m': m, 'v': v, 'timestep': timestep}

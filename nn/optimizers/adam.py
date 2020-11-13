from typing import Dict, Any

import numpy as np

from nn.optimizers.base import BaseOptimizer


class Adam(BaseOptimizer):
    def __init__(self, layers: Dict[str, Any], learning_rate: float,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(layers)
        self._learning_rate = learning_rate
        self._cache = {}
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps

    def _update(self, layer: np.ndarray, grad: np.ndarray, name: str):
        cache = self._cache.get(name, {'v': np.zeros_like(grad), 'm': np.zeros_like(grad)})
        m_t_minus_1 = cache['m']
        v_t_minus_1 = cache['v']

        m = self._beta1 * m_t_minus_1 + (1. - self._beta1) * grad
        v = self._beta2 * v_t_minus_1 + (1. - self._beta2) * grad**2

        timestep = self._cache.get('__timestep', 1)
        adjusted_m = m / (1. - self._beta1**timestep)
        adjusted_v = v / (1. - self._beta2**timestep)

        layer -= (self._learning_rate * adjusted_m) / (np.sqrt(adjusted_v) + self._eps)

        cache['__timestep'] = timestep + 1
        cache['m'] = adjusted_m
        cache['v'] = adjusted_v

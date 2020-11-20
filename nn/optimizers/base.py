import abc
from typing import Dict, Any

import numpy as np


class BaseOptimizer(abc.ABC):
    def __init__(self, weights: Dict[str, Any]):
        self.weights = weights
        self._cache = {}

    def step(self, grads: Dict[str, Any]):
        self._step(self.weights, grads)

    def _step(self, weights: Dict[str, Any], grads: Dict[str, Any], unique_name: str = ''):
        for name, weight in weights.items():
            unique_name += f'_{name}'

            grad = grads[name]
            if isinstance(weight, Dict):
                self._step(weight, grad, unique_name)
            else:
                cache = self._cache.get(unique_name, {})
                self._cache[unique_name] = self._update(weight, grad, cache)

    @abc.abstractmethod
    def _update(self, weight: np.ndarray, grad: np.ndarray, cache: Dict[str, Any]) -> Dict[str, Any]:
        pass

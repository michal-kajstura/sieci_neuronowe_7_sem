import abc
from typing import Dict, Any

import numpy as np


class BaseOptimizer(abc.ABC):
    def __init__(self, layers: Dict[str, Any]):
        self._layers = layers
        self._cache = {}

    def step(self, grads: Dict[str, Any]):
        self._step(self._layers, grads)

    def _step(self, layers: Dict[str, Any], grads: Dict[str, Any], unique_name: str = ''):
        for name, layer in layers.items():
            unique_name += f'_{name}'

            grad = grads[name]
            if isinstance(layer, Dict):
                self._step(layer, grad, unique_name)
            else:
                cache = self._cache.get(unique_name, {})
                cache.update(self._update(layer, grad, cache))

    @abc.abstractmethod
    def _update(self, layer: np.ndarray, grad: np.ndarray, cache: Dict[str, Any]) -> Dict[str, Any]:
        pass

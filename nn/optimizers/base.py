import abc
from typing import Dict, Any

import numpy as np


class BaseOptimizer(abc.ABC):
    def __init__(self, layers: Dict[str, Any]):
        self._layers = layers

    def step(self, grads: Dict[str, Any]):
        self._step(self._layers, grads)

    def _step(self, layers: Dict[str, Any], grads: Dict[str, Any], unique_name: str = ''):
        for name, layer in layers.items():
            unique_name += f'_{name}'

            grad = grads[name]
            if isinstance(layer, Dict):
                self._step(layer, grad, unique_name)
            else:
                self._update(layer, grad, unique_name)

    @abc.abstractmethod
    def _update(self, layer: np.ndarray, grad: np.ndarray, name: str):
        pass

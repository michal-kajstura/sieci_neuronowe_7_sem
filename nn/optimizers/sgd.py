from typing import Dict, Any

import numpy as np

from nn.optimizers.base import BaseOptimizer


class SGD(BaseOptimizer):
    def __init__(self, layers: Dict[str, Any], learning_rate: float):
        super().__init__(layers)
        self._learning_rate = learning_rate

    def _update(self, layer: np.ndarray, grad: np.ndarray, name: str):
        layer -= self._learning_rate * grad



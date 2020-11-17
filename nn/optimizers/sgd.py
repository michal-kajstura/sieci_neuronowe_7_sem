from typing import Dict, Any

import numpy as np

from nn.optimizers.base import BaseOptimizer


class SGD(BaseOptimizer):
    def __init__(self, weights: Dict[str, Any], learning_rate: float):
        super().__init__(weights)
        self._learning_rate = learning_rate

    def _update(self, weight: np.ndarray, grad: np.ndarray, _: Dict[str, Any]) -> Dict[str, Any]:
        weight -= self._learning_rate * grad
        return {}



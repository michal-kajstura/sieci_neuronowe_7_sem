import numpy as np


def accuracy(outputs, y_true):
    y_pred = outputs.argmax(-1)
    return np.mean(y_true == y_pred)

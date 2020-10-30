import numpy as np


def softmax(x):
    # Max trick to ensure numerical stability
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(logits, y_true):
    probs = softmax(logits)
    n = len(y_true)
    log_likelihood = -np.log(probs[range(n), y_true])
    loss = log_likelihood.mean()

    d_scores = probs
    d_scores[range(n), y_true] -= 1
    d_scores = d_scores / n 
    return loss, d_scores

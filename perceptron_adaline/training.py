import numpy as np


def sgd_step(model, x, y, learning_rate):
    delta_sum = 0.
    for xx, yy in zip(x, y):
        delta_sum += step(model, x, y, learning_rate)
    return delta_sum / len(x)


def step(model, x, y, learning_rate):
    y_pred = model.forward(x)
    error = y - y_pred
    delta = x.T.dot(error)
    model.update_weights(learning_rate, delta)
    return error


def perceptron_break(delta):
    return all(delta == 0)


def adaline_break(delta, tolerance):
    if any(np.isnan(delta)):
        raise OverflowError()

    loss = np.mean(delta ** 2)
    return loss < tolerance


def train(model, x, y,
          learning_rate=0.1,
          bias=True,
          bipolar=False,
          step_func=step,
          break_func=perceptron_break):
    if bipolar:
        y = y * 2. - 1.

    if bias:
        x = np.concatenate((np.ones((len(x), 1)), x), axis=1)

    epochs = 0

    while (True):
        epochs += 1
        delta = step_func(model, x, y, learning_rate)
        if break_func(delta):
            break

    return epochs

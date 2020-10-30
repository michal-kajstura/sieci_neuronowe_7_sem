import operator
from itertools import islice

import numpy as np


class Trainer:
    def __init__(self, optimizer_func, loss_func, batch_size=32, epochs=10, metrics=None,
                 monitor=None, mode='max'):
        self._optimizer_func = optimizer_func
        self._loss_func = loss_func
        self._batch_size = batch_size
        self._epochs = epochs
        self._metrics = metrics or {}
        self._monitor = monitor
        self._mode = mode
        self._best_metric_val = np.inf if mode == 'min' else -np.inf

    def train(self, model, train_generator, validation_generator):
        for epoch in range(self._epochs):
            for x_batch, y_batch in train_generator:
                logits, caches = model.forward(x_batch)
                loss_value, loss_grad = self._loss_func(logits, y_batch)
                grads = model.backward(loss_grad, caches)
                self._optimizer_func(grads, model.weights)

            train_metrics = self._validate(model, train_generator)
            val_metrics = self._validate(model, validation_generator)

            if self._stop_condition(val_metrics):
                break

            yield train_metrics, val_metrics


    def _validate(self, model, generator):
        outputs, y_true = zip(*((model.forward(x), y) for x, y in generator))
        return {name: metric(outputs, y_true) for name, metric in self._metrics.items()}

    def _stop_condition(self, metrics):
        if self._monitor:
            try:
                monitored_metric_val = metrics[self._monitor]
                func = operator.gt if self._mode == 'min' else operator.lt
                if func(monitored_metric_val, self._best_metric_val):
                    self._best_metric_val = monitored_metric_val
                    return True
            except KeyError:
                raise KeyError("Monitored metric not in validation metrics")
        return False



def data_generator(data, batch_size, shuffle=True):
    batches = iter(_batchify(data, batch_size))
    while True:
        try:
            yield next(batches)
        except StopIteration:
            if shuffle:
                np.random.shuffle(data)
            batches = iter(_batchify(data, batch_size))


def _batchify(items, batch_size):
    items_iter = iter(items)
    while True:
        batch = list(islice(items_iter, batch_size))
        if batch:
            yield (np.stack(item) for item in zip(*batch))
        else:
            break
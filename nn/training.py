import operator
from itertools import islice

import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self, optimizer, loss_func, epochs=10, metrics=None,
                 monitor=None, mode='max', patience=3, delta=0.005):
        self._optimizer = optimizer
        self._loss_func = loss_func
        self._epochs = epochs
        self._metrics = metrics or {}
        self._monitor = monitor
        self._mode = mode
        self._best_metric_val = np.inf if mode == 'min' else -np.inf
        self._patience = patience
        self._delta = delta

    def train(self, model, train_generator, validation_generator):
        patience = self._patience

        training_results = []
        for epoch in range(self._epochs):
            pbar = tqdm(train_generator, total=len(train_generator))
            loss_value = 0
            for x_batch, y_batch in pbar:
                logits, caches = model.forward(x_batch)
                loss_value, loss_grad = self._loss_func(logits, y_batch)
                grads = model.backward(loss_grad, caches)
                self._optimizer.step(grads)

                pbar.set_description(f'Loss: {loss_value:.4}, Patience: {patience}')

            train_metrics = self._validate(model, train_generator)
            val_metrics = self._validate(model, validation_generator)
            training_results.append({'train': train_metrics, 'val': val_metrics})

            pbar.set_description(f'Loss: {loss_value:.4},'
                                 f' Patience: {patience},'
                                 f' Accuracy: {val_metrics["accuracy"]}')

            if self._stop_condition(val_metrics):
                patience -= 1
            else:
                patience = self._patience

            if patience == 0:
                break

    def _validate(self, model, generator):
        outputs_y = zip(*((model.forward(x)[0], y) for x, y in generator))
        outputs, y_true = map(np.concatenate, outputs_y)
        loss, _ = self._loss_func(outputs, y_true)
        metrics = {name: metric(outputs, y_true) for name, metric in self._metrics.items()}
        return {'loss': loss, **metrics}

    def _stop_condition(self, metrics):
        if self._monitor:
            try:
                monitored_metric_val = metrics[self._monitor]
            except KeyError:
                raise KeyError("Monitored metric not in validation metrics")
            else:
                func = operator.lt if self._mode == 'min' else operator.gt
                value_func = operator.add if self._mode == 'min' else operator.sub
                if func(value_func(monitored_metric_val, self._delta), self._best_metric_val):
                    self._best_metric_val = monitored_metric_val
                    return False
        return True



class DataGenerator:
    def __init__(self, data, batch_size, shuffle=True):
        self._data = data
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __iter__(self):
        if self._shuffle:
            np.random.shuffle(self._data)

        for batch in _batchify(self._data, self._batch_size):
            yield batch

    def __len__(self):
        return len(self._data) // self._batch_size


def _batchify(items, batch_size):
    items_iter = iter(items)
    while True:
        batch = list(islice(items_iter, batch_size))
        if batch:
            yield (np.stack(item) for item in zip(*batch))
        else:
            break
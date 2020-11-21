import json
from pathlib import Path
from time import time

from torchvision.datasets import MNIST

from nn.init import xavier_init, random_normal_init, he_init
from nn.layers.activations import ReLU
from nn.layers.linear import Linear
from nn.layers.model import Model
from nn.loss import cross_entropy
from nn.metrics import accuracy
from nn.optimizers.adadelta import Adadelta
from nn.optimizers.adagrad import Adagrad
from nn.optimizers.adam import Adam
from nn.optimizers.momentum import Momentum, NesterovMomentum
from nn.optimizers.sgd import SGD
from nn.training import Trainer, DataGenerator

#%%

dataset = MNIST('./data', train=True, download=True)
x_train = dataset.data.numpy()
x_train = x_train
y_train = dataset.targets.numpy()

dataset = MNIST('./data', train=False, download=True)
x_test = dataset.data.numpy()
y_test = dataset.targets.numpy()

def preprocess_x(x):
    return x.reshape(-1, 784) / 255.

train_set = list(zip(preprocess_x(x_train), y_train))
test_set = list(zip(preprocess_x(x_test), y_test))


def time_it(func):
    def inner(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        stop = time()
        return result, stop - start
    return inner

@time_it
def experiment(
        optimizer_class=Adam,
        optimizer_kwargs=None,
        weight_init_func=he_init,
        hidden_size=512,
        batch_size=128,
        activation_func=ReLU):
    model = Model(
        Linear(784, hidden_size, weight_init_func),
        activation_func(),
        Linear(hidden_size, 10),
    )

    optimizer_kwargs = optimizer_kwargs or {'learning_rate': 0.01}
    optimizer = optimizer_class(weights=model.weights, **optimizer_kwargs)

    trainer = Trainer(
        optimizer=optimizer,
        loss_func=cross_entropy,
        epochs=30,
        metrics={'accuracy': accuracy},
        monitor='accuracy',
        mode='max',
        delta=0.00,
        patience=30,
    )

    train_generator = DataGenerator(train_set, batch_size=batch_size)
    test_generator = DataGenerator(test_set, batch_size=128)
    return trainer.train(model, train_generator, test_generator)


experiment_specs = {
    'optimizer_class': [
        (SGD, {'learning_rate': 1e-1}),
        (SGD, {'learning_rate': 1e-2}),
        (SGD, {'learning_rate': 1e-3}),
        (SGD, {'learning_rate': 1e-4}),
        (Momentum, {'learning_rate': 1e-1}),
        (Momentum, {'learning_rate': 1e-2}),
        (Momentum, {'learning_rate': 1e-3}),
        (Momentum, {'learning_rate': 1e-4}),
        (NesterovMomentum, {'learning_rate': 1e-1}),
        (NesterovMomentum, {'learning_rate': 1e-2}),
        (NesterovMomentum, {'learning_rate': 1e-3}),
        (NesterovMomentum, {'learning_rate': 1e-4}),
        (Adagrad, {'learning_rate': 1e-1}),
        (Adagrad, {'learning_rate': 1e-2}),
        (Adagrad, {'learning_rate': 1e-3}),
        (Adagrad, {'learning_rate': 1e-4}),
        (Adadelta, {'delta': 0.5}),
        (Adadelta, {'delta': 0.8}),
        (Adadelta, {'delta': 0.9}),
        (Adadelta, {'delta': 0.99}),
        (Adam, {'learning_rate': 1e-1}),
        (Adam, {'learning_rate': 1e-2}),
        (Adam, {'learning_rate': 1e-3}),
        (Adam, {'learning_rate': 1e-4}),
    ],
    'weight_init_func': [random_normal_init, xavier_init, he_init],
}

n_iters = 10
result_path = Path('./result.json')
result = {}
for arg, values in experiment_specs.items():
    result[arg] = {}
    for value in values:
        result[arg][str(value)] = []
        for it in range(n_iters):
            print(f'{arg}: {str(value)}\n')

            if isinstance(value, tuple):
                o, k = value
                args = {'optimizer_class': o, 'optimizer_kwargs': k}
            else:
                args = {arg: value}

            # result[arg][str(value)] =
            metrics, _ = experiment(**args)
            result[arg][str(value)].append(metrics)

            print(f'Best validation accuracy: {max(a["val"]["accuracy"] for a in metrics):.4}\n')

        with result_path.open('w') as file:
            json.dump(result, file, indent=4)
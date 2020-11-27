import json
from pathlib import Path
from time import time

from torchvision.datasets import MNIST

from nn.init import he_init
from nn.layers.activations import ReLU
from nn.layers.conv import Conv2d
from nn.layers.flatten import Flatten
from nn.layers.linear import Linear
from nn.layers.model import Model
from nn.layers.pool import MaxPool
from nn.loss import cross_entropy
from nn.metrics import accuracy
from nn.optimizers.adam import Adam
from nn.training import DataGenerator
from nn.training import Trainer

dataset = MNIST('./data', train=True, download=True)
x_train = dataset.data.numpy()
x_train = x_train
y_train = dataset.targets.numpy()

dataset = MNIST('./data', train=False, download=True)
x_test = dataset.data.numpy()
y_test = dataset.targets.numpy()

def preprocess_x(x):
    return x.reshape(-1, 1, 28, 28) / 255.

train_set = list(zip(preprocess_x(x_train), y_train))
test_set = list(zip(preprocess_x(x_test), y_test))


train_generator = DataGenerator(train_set, batch_size=24)
test_generator = DataGenerator(test_set, batch_size=128)

def time_it(func):
    def inner(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        stop = time()
        return result, stop - start
    return inner

@time_it
def experiment(kernel_size, padding):
    model = Model(
        Conv2d(1, 8, kernel_size=kernel_size, padding=padding, init_func=he_init),
        ReLU(),
        MaxPool(),
        Flatten(),
        Linear(1568, 128),
        Linear(128, 10),
    )

    trainer = Trainer(
        optimizer=Adam(weights=model.weights, learning_rate=0.0006),
        loss_func=cross_entropy,
        epochs=20,
        metrics={'accuracy': accuracy},
        monitor='accuracy',
        mode='max',
        delta=0.00,
        patience=30,
    )

    train_generator = DataGenerator(train_set, batch_size=64)
    test_generator = DataGenerator(test_set, batch_size=128)
    return trainer.train(model, train_generator, test_generator)


experiment_specs = [
    (3, 1),
    (5, 2),
    (7, 3),
    (11, 5),
    (17, 8),
]

result_path = Path('./result-conv.json')
result = {}
n_iter = 5
for kernel_size, padding in experiment_specs:
    result[kernel_size] = []
    for _ in range(n_iter):
        metrics, exp_time = experiment(kernel_size, padding)
        result[kernel_size].append(metrics)
        print(f'Best validation accuracy: {max(a["val"]["accuracy"] for a in metrics):.4}\n')
    with result_path.open('w') as file:
        json.dump(result, file, indent=4)


# import sys
# sys.path.append('/media/data/studia/sieci_neuronowe_7_sem')
import json
from functools import partial
from pathlib import Path

from torchvision.datasets import MNIST

from nn.init import init_weights
from nn.layers.activations import ReLU, Tanh, Sigmoid
from nn.layers.linear import Linear
from nn.layers.model import Model
from nn.loss import cross_entropy
from nn.metrics import accuracy
from nn.optimization import sgd
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


#%%

from time import time

def time_it(func):
    def inner(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        stop = time()
        return result, stop - start
    return inner

@time_it
def experiment(hidden_size=256, learning_rate=0.001, batch_size=128, std=0.05, activation_func=ReLU):
    model = Model(
        Linear(784, hidden_size, init_func=partial(init_weights, std=std)),
        activation_func(),
        Linear(hidden_size, 10),
    )

    trainer = Trainer(
        optimizer_func=partial(sgd, learning_rate=1e-3),
        loss_func=cross_entropy,
        epochs=90,
        metrics={'accuracy': accuracy},
        monitor='accuracy',
        mode='max',
        delta=0.00,
        patience=90,

    )

    train_generator = DataGenerator(train_set, batch_size=batch_size)
    test_generator = DataGenerator(test_set, batch_size=128)
    return list(trainer.train(model, train_generator, test_generator))

#%%

experiment_specs = {
    'hidden_size': [16, 64, 128, 256, 1024],
    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-1],
    'batch_size': [1, 64, 128, 256, 512],
    'std': [0.0001, 0.1, 0.3, 0.8],
    'activation_func': [ReLU, Tanh, Sigmoid]
}

result_path = Path('./result.json')
result = {}
for arg, values in experiment_specs.items():
    result[arg] = {}
    for value in values:
        print(f'{arg}: {value}\n')
        result[arg][str(value)] = {}
        metrics, experiment_time = experiment(**{arg: value})
        result[arg][str(value)]['value'] = metrics
        result[arg][str(value)]['time'] = experiment_time

        print(f'Best validation accuracy: {max(a["accuracy"] for _, a in metrics):.4}\n')

        with result_path.open('w') as file:
            json.dump(result, file, indent=4)
